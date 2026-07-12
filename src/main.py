from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

MAIN_SCRIPT = Path(__file__).resolve()


def _bootstrap_import_path() -> None:
    if __package__ not in {None, ""}:
        return
    repo_root = MAIN_SCRIPT.parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_import_path()

from src.config import load_app_config
from src.dataloader import (
    discover_parquet_inputs,
    iter_flat_tables,
    required_columns_for_split,
    scan_flat_table_stats,
    validate_matching_schemas,
)
from src.features import fit_vocabs, plan_vocab_fit, vocab_artifacts, vocab_strategy_fingerprint
from src.train import evaluate_mdl, is_main_process, predict_mdl, train_mdl


def _load_config(args: argparse.Namespace):
    return load_app_config(args.config)


def _cmd_validate_config(args: argparse.Namespace) -> int:
    config = _load_config(args)
    print(f"config: OK ({args.config})")
    print(f"model: {config.model.name}")
    print(f"features: {len(config.features)}")
    print(f"vocab_strategy_hash: {vocab_strategy_fingerprint(config)}")
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
        str(MAIN_SCRIPT),
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


def _cmd_evaluate(args: argparse.Namespace) -> int:
    config = _load_config(args)
    result = evaluate_mdl(
        config,
        split_name=args.split,
        checkpoint_path=args.checkpoint_path,
        max_batches=args.max_batches,
        allow_random_init=args.allow_random_init,
        group_metric_name=args.group_metric_name,
    )
    print(
        f"evaluate_result rows={result.rows} "
        f"group_metric={result.group_metric_name}"
    )
    for task_name, metrics in result.metrics.items():
        formatted = " ".join(
            f"{name}={'NA' if value is None else f'{value:.8f}'}"
            for name, value in metrics.items()
        )
        print(f"evaluate_task task={task_name} {formatted}")
    return 0



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parquet-native MDL CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", required=True)
    validate.set_defaults(func=_cmd_validate_config)

    profile = subparsers.add_parser("profile")
    profile.add_argument("--config", required=True)
    profile.add_argument("--split", choices=["train", "test"], default="train")
    profile.add_argument("--max-batches", type=int, default=10)
    profile.set_defaults(func=_cmd_profile)

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

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--split", choices=["train", "test"], default="test")
    evaluate.add_argument("--checkpoint-path", default=None)
    evaluate.add_argument("--max-batches", type=int, default=None)
    evaluate.add_argument("--allow-random-init", action="store_true")
    evaluate.add_argument(
        "--group-metric-name", choices=["qauc", "uauc"], default="qauc"
    )
    evaluate.set_defaults(func=_cmd_evaluate)

    return parser

def main() -> None:
    args = build_arg_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
