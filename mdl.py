from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from src.benchmark import benchmark_split, benchmark_training
from src.config import load_app_config
from src.dataloader import (
    discover_parquet_inputs,
    iter_flat_tables,
    required_columns_for_split,
    scan_flat_table_stats,
    validate_matching_schemas,
)
from src.features import fit_vocabs, plan_vocab_fit, vocab_artifacts, vocab_strategy_fingerprint
from src.train import is_main_process, predict_mdl, train_mdl


MDL_PAPER = Path("paper/MDL/main.tex")
ONETRANS_PAPER = Path("paper/OneTrans/main.tex")
ALIGNMENT_DOC = Path("PAPER_ALIGNMENT.md")


def _load_config(args: argparse.Namespace):
    return load_app_config(args.config)


def _cmd_validate_config(args: argparse.Namespace) -> int:
    config = _load_config(args)
    print(f"config: OK ({args.config})")
    print(f"model: {config.model.name}")
    print(f"features: {len(config.features)}")
    print(f"vocab_strategy_hash: {vocab_strategy_fingerprint(config)}")
    return 0


def _require_text(path: Path, patterns: list[str]) -> list[str]:
    if not path.exists():
        return [f"missing file: {path}"]
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [f"{path}: missing pattern {pattern!r}" for pattern in patterns if pattern not in text]


def _cmd_check_paper_alignment(args: argparse.Namespace) -> int:
    errors: list[str] = []
    errors.extend(
        _require_text(
            MDL_PAPER,
            [
                "Multi",
                "Distribution",
                "Unified Information Tokenization",
                "Domain-aware Attention",
                "Domain-fused Module",
                "TokenMixing",
                "PertokenFFN",
            ],
        )
    )
    errors.extend(
        _require_text(
            ONETRANS_PAPER,
            [
                "OneTrans",
                "Non-Sequential Tokenization",
                "Sequential Tokenization",
                "Mixed (shared/token-specific) Causal Attention",
                "Pyramid Stack",
                "Cross Request KV Caching",
            ],
        )
    )
    errors.extend(
        _require_text(
            ALIGNMENT_DOC,
            [
                "`rankmixer`",
                "mdl_rankmixer",
                "onetrans",
                "mdl_onetrans",
                "Hybrid MDL + OneTrans",
                "Open Deviations",
            ],
        )
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("paper alignment sources: OK")
    print(f"MDL paper: {MDL_PAPER}")
    print(f"OneTrans paper: {ONETRANS_PAPER}")
    print(f"alignment doc: {ALIGNMENT_DOC}")
    return 0


def _cmd_profile(args: argparse.Namespace) -> int:
    config = _load_config(args)
    split = config.data.train if args.split == "train" else config.data.test
    if split is None:
        raise ValueError(f"split {args.split!r} is not configured")
    paths = discover_parquet_inputs(split.inputs)
    fingerprint = validate_matching_schemas(paths)
    columns = required_columns_for_split(config, split)
    stats = scan_flat_table_stats(config, args.split, max_batches=args.max_batches)
    print(f"split: {args.split}")
    print(f"format: {split.format}")
    print(f"adapter: {split.adapter.callable if split.adapter else 'identity'}")
    print(f"files: {len(paths)}")
    print(f"schema_fingerprint: {fingerprint}")
    print(f"required_columns: {len(columns)}")
    print(f"sample_record_batches: {stats.raw_record_batches}")
    print(f"sample_raw_rows: {stats.raw_rows}")
    print(f"sample_rows: {stats.flat_rows}")
    print(f"sample_flat_tables: {stats.flat_tables}")
    print(f"vocab_strategy_hash: {vocab_strategy_fingerprint(config)}")
    for ref in vocab_artifacts(config):
        print(
            "vocab_feature "
            f"name={ref.feature_name} encoding={ref.encoding} artifact={ref.artifact_path} size_hint={ref.size_hint}"
        )
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    config = _load_config(args)
    result = benchmark_split(config, args.split, max_batches=args.max_batches)
    print(f"split: {result.split}")
    print(f"files: {result.files}")
    print(f"record_batches: {result.record_batches}")
    print(f"input_rows: {result.input_rows}")
    print(f"flat_rows: {result.flat_rows}")
    print(f"elapsed_seconds: {result.elapsed_seconds:.6f}")
    print(f"rows_per_second: {result.rows_per_second:.2f}")
    return 0


def _cmd_benchmark_train(args: argparse.Namespace) -> int:
    config = _load_config(args)
    result = benchmark_training(config, max_steps=args.max_steps)
    print(f"steps: {result.steps}")
    print(f"rows: {result.rows}")
    print(f"last_loss: {result.last_loss:.6f}")
    print(f"elapsed_seconds: {result.elapsed_seconds:.6f}")
    print(f"steps_per_second: {result.steps_per_second:.2f}")
    print(f"rows_per_second: {result.rows_per_second:.2f}")
    return 0


def _cmd_fit_vocab(args: argparse.Namespace) -> int:
    config = _load_config(args)
    plan = plan_vocab_fit(config)
    tables = iter_flat_tables(config, "train", extra_columns=plan.columns)
    fitted = fit_vocabs(config, tables, plan)
    if not fitted:
        print("no vocab features configured")
        return 0
    for vocab in fitted:
        print(
            "fitted_vocab "
            f"feature={vocab.feature_name} path={vocab.path} size={vocab.size} "
            f"min_count={vocab.min_count} max_size={vocab.max_size}"
        )
    return 0


def _in_distributed_launcher() -> bool:
    return "LOCAL_RANK" in os.environ or os.environ.get("MDL_DDP_LAUNCHED") == "1"


def _effective_distributed_mode(args: argparse.Namespace, config) -> str:
    return args.distributed or config.runtime.distributed


def _effective_nproc_per_node(args: argparse.Namespace, config) -> int:
    if args.nproc_per_node is not None:
        return args.nproc_per_node
    if config.runtime.nproc_per_node is not None:
        return config.runtime.nproc_per_node
    try:
        import torch
    except ImportError:
        return 1
    return max(torch.cuda.device_count(), 1)


def _launch_ddp_train(args: argparse.Namespace, config) -> int:
    nproc_per_node = _effective_nproc_per_node(args, config)
    if nproc_per_node <= 0:
        raise ValueError("nproc_per_node must be positive")
    master_addr = args.master_addr or config.runtime.master_addr
    master_port = args.master_port or config.runtime.master_port
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(nproc_per_node),
        "--master-addr",
        master_addr,
        "--master-port",
        str(master_port),
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    env = os.environ.copy()
    env["MDL_DDP_LAUNCHED"] = "1"
    return subprocess.run(command, env=env, check=False).returncode


def _cmd_train(args: argparse.Namespace) -> int:
    config = _load_config(args)
    if _effective_distributed_mode(args, config) == "ddp" and not _in_distributed_launcher():
        return _launch_ddp_train(args, config)
    result = train_mdl(config, max_steps=args.max_steps)
    if is_main_process():
        print(
            "train_result "
            f"steps={result.steps} rows={result.rows} last_loss={result.last_loss:.6f} "
            f"elapsed_seconds={result.elapsed_seconds:.6f} "
            f"steps_per_second={result.steps_per_second:.2f} "
            f"rows_per_second={result.rows_per_second:.2f}"
        )
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    config = _load_config(args)
    result = predict_mdl(
        config,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        max_batches=args.max_batches,
        allow_random_init=args.allow_random_init,
    )
    print(f"predict_result rows={result.rows} output_path={result.output_path}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parquet-native MDL CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", required=True)
    validate.set_defaults(func=_cmd_validate_config)

    align = subparsers.add_parser("check-paper-alignment")
    align.set_defaults(func=_cmd_check_paper_alignment)

    profile = subparsers.add_parser("profile")
    profile.add_argument("--config", required=True)
    profile.add_argument("--split", choices=["train", "test"], default="train")
    profile.add_argument("--max-batches", type=int, default=10)
    profile.set_defaults(func=_cmd_profile)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--split", choices=["train", "test"], default="train")
    benchmark.add_argument("--max-batches", type=int, default=10)
    benchmark.set_defaults(func=_cmd_benchmark)

    benchmark_train = subparsers.add_parser("benchmark-train")
    benchmark_train.add_argument("--config", required=True)
    benchmark_train.add_argument("--max-steps", type=int, default=10)
    benchmark_train.set_defaults(func=_cmd_benchmark_train)

    fit_vocab = subparsers.add_parser("fit-vocab")
    fit_vocab.add_argument("--config", required=True)
    fit_vocab.set_defaults(func=_cmd_fit_vocab)

    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--distributed", choices=["none", "ddp"], default=None)
    train.add_argument("--nproc-per-node", type=int, default=None)
    train.add_argument("--master-addr", default=None)
    train.add_argument("--master-port", type=int, default=None)
    train.set_defaults(func=_cmd_train)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--config", required=True)
    predict.add_argument("--checkpoint-path", default=None)
    predict.add_argument("--output-path", default=None)
    predict.add_argument("--max-batches", type=int, default=None)
    predict.add_argument("--allow-random-init", action="store_true")
    predict.set_defaults(func=_cmd_predict)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
