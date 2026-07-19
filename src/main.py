from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timedelta
import os
from pathlib import Path
import subprocess
import sys

MAIN_SCRIPT = Path(__file__).resolve()
DEFAULT_DATA_BASE_DIR = (
    "hdfs://temu-data-ns/apps/nothive/warehouse/searchrec/"
    "searchrec_cvr_allscene_agg_fgoutput_hour_dracarys_exp"
)


def _bootstrap_import_path() -> None:
    if __package__ not in {None, ""}:
        return
    repo_root = MAIN_SCRIPT.parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_import_path()

from src.benchmark import BenchmarkOptions, run_benchmark, write_benchmark_report
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


def _expand_hour_partition(
    base_dir: str,
    start_hour: str,
    end_hour: str,
) -> tuple[str, ...]:
    base = base_dir.rstrip("/")
    if not base:
        raise ValueError("--data-base-dir must be non-empty when expanding hour windows")
    try:
        start = datetime.strptime(start_hour, "%Y-%m-%d-%H")
        end = datetime.strptime(end_hour, "%Y-%m-%d-%H")
    except ValueError as error:
        raise ValueError("hour windows must use YYYY-MM-DD-HH") from error
    if end <= start:
        raise ValueError("end hour must be later than start hour")
    inputs: list[str] = []
    current = start
    while current < end:
        inputs.append(f"{base}/pt={current:%Y-%m-%d}/hr={current:%H}")
        current += timedelta(hours=1)
    return tuple(inputs)


def _resolve_split_inputs(
    *,
    explicit: list[str] | None,
    start_hour: str | None,
    end_hour: str | None,
    base_dir: str,
    configured: tuple[str, ...],
) -> tuple[str, ...]:
    if explicit:
        return tuple(explicit)
    if start_hour or end_hour:
        if not start_hour or not end_hour:
            raise ValueError("start hour and end hour must be provided together")
        return _expand_hour_partition(base_dir, start_hour, end_hour)
    return configured


def _apply_data_input_overrides(config, args: argparse.Namespace):
    """Optionally override empty/fixed split inputs from CLI without editing YAML."""
    base_dir = getattr(args, "data_base_dir", None) or DEFAULT_DATA_BASE_DIR
    train_inputs = _resolve_split_inputs(
        explicit=getattr(args, "train_input", None),
        start_hour=getattr(args, "train_start_hour", None),
        end_hour=getattr(args, "train_end_hour", None),
        base_dir=base_dir,
        configured=config.data.train.inputs,
    )
    test_inputs = config.data.test.inputs if config.data.test is not None else ()
    if config.data.test is not None or getattr(args, "test_input", None) or getattr(
        args, "test_start_hour", None
    ):
        test_inputs = _resolve_split_inputs(
            explicit=getattr(args, "test_input", None),
            start_hour=getattr(args, "test_start_hour", None),
            end_hour=getattr(args, "test_end_hour", None),
            base_dir=base_dir,
            configured=test_inputs,
        )
    train = (
        config.data.train
        if train_inputs == config.data.train.inputs
        else replace(config.data.train, inputs=train_inputs)
    )
    test = config.data.test
    if config.data.test is not None and test_inputs != config.data.test.inputs:
        test = replace(config.data.test, inputs=test_inputs)
    if train is config.data.train and test is config.data.test:
        return config
    return replace(config, data=replace(config.data, train=train, test=test))


_TRAINING_OVERRIDE_FIELDS = (
    "batch_size",
    "lr_dense",
    "lr_sparse",
    "lr_warmup_steps",
    "lr_decay_steps",
    "log_every_steps",
    "dense_clip_norm",
    "sparse_clip_norm",
    "checkpoint_path",
)


def _scale_length_buckets(buckets: tuple, old_batch_size: int, new_batch_size: int) -> tuple:
    """Keep length-bucket ratios when CLI overrides training.batch_size."""
    if not buckets or old_batch_size <= 0 or old_batch_size == new_batch_size:
        return buckets
    scale = new_batch_size / float(old_batch_size)
    return tuple(
        replace(bucket, batch_size=max(1, int(round(bucket.batch_size * scale))))
        for bucket in buckets
    )


def _replace_split_length_buckets(split, buckets: tuple):
    if split is None or split.reader.length_buckets == buckets:
        return split
    return replace(split, reader=replace(split.reader, length_buckets=buckets))


def _apply_training_overrides(config, args: argparse.Namespace):
    """Apply optional CLI training hyperparameter overrides."""
    updates: dict[str, object] = {}
    for field_name in _TRAINING_OVERRIDE_FIELDS:
        value = getattr(args, field_name, None)
        if value is not None:
            updates[field_name] = value
    if not updates:
        return config

    old_batch_size = config.training.batch_size
    training = replace(config.training, **updates)
    training.validate()

    data = config.data
    if "batch_size" in updates:
        new_batch_size = training.batch_size
        train = _replace_split_length_buckets(
            config.data.train,
            _scale_length_buckets(
                config.data.train.reader.length_buckets,
                old_batch_size,
                new_batch_size,
            ),
        )
        test = _replace_split_length_buckets(
            config.data.test,
            _scale_length_buckets(
                config.data.test.reader.length_buckets if config.data.test else (),
                old_batch_size,
                new_batch_size,
            ),
        )
        if train is not config.data.train or test is not config.data.test:
            data = replace(config.data, train=train, test=test)

    if training is config.training and data is config.data:
        return config
    return replace(config, training=training, data=data)


def _load_config(args: argparse.Namespace):
    config = load_app_config(args.config)
    if any(
        getattr(args, name, None)
        for name in (
            "train_input",
            "test_input",
            "train_start_hour",
            "train_end_hour",
            "test_start_hour",
            "test_end_hour",
            "data_base_dir",
        )
    ):
        config = _apply_data_input_overrides(config, args)
    if any(getattr(args, name, None) is not None for name in _TRAINING_OVERRIDE_FIELDS):
        config = _apply_training_overrides(config, args)
    return config


def _parse_named_positive_ints(raw: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, separator, value_text = item.partition("=")
        if not separator or not name.strip():
            raise argparse.ArgumentTypeError(
                "expected comma-separated name=positive_integer entries"
            )
        try:
            value = int(value_text)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid integer for {name.strip()!r}: {value_text!r}"
            ) from error
        if value <= 0:
            raise argparse.ArgumentTypeError(
                f"length for {name.strip()!r} must be positive"
            )
        result[name.strip()] = value
    if not result:
        raise argparse.ArgumentTypeError("at least one named length is required")
    return result


def _add_data_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--train-input",
        action="append",
        default=None,
        help="override data.train.inputs (repeatable); default leaves YAML empty/unset",
    )
    parser.add_argument(
        "--test-input",
        action="append",
        default=None,
        help="override data.test.inputs (repeatable); default leaves YAML empty/unset",
    )
    parser.add_argument(
        "--data-base-dir",
        default=None,
        help="HDFS/local base dir used with --*-start-hour/--*-end-hour",
    )
    parser.add_argument(
        "--train-start-hour",
        default=None,
        help="inclusive train hour window start as YYYY-MM-DD-HH",
    )
    parser.add_argument(
        "--train-end-hour",
        default=None,
        help="exclusive train hour window end as YYYY-MM-DD-HH",
    )
    parser.add_argument(
        "--test-start-hour",
        default=None,
        help="inclusive test hour window start as YYYY-MM-DD-HH",
    )
    parser.add_argument(
        "--test-end-hour",
        default=None,
        help="exclusive test hour window end as YYYY-MM-DD-HH",
    )


def _add_training_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "override training.batch_size (per rank); also scales "
            "data.*.reader.length_buckets batch sizes proportionally"
        ),
    )
    parser.add_argument(
        "--lr-dense",
        type=float,
        default=None,
        help="override training.lr_dense",
    )
    parser.add_argument(
        "--lr-sparse",
        type=float,
        default=None,
        help="override training.lr_sparse",
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=None,
        help="override training.lr_warmup_steps",
    )
    parser.add_argument(
        "--lr-decay-steps",
        type=int,
        default=None,
        help="override training.lr_decay_steps",
    )
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=None,
        help="override training.log_every_steps",
    )
    parser.add_argument(
        "--dense-clip-norm",
        type=float,
        default=None,
        help="override training.dense_clip_norm",
    )
    parser.add_argument(
        "--sparse-clip-norm",
        type=float,
        default=None,
        help="override training.sparse_clip_norm",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="override training.checkpoint_path",
    )


def _cmd_validate_config(args: argparse.Namespace) -> int:
    config = _load_config(args)
    print(f"config: OK ({args.config})")
    print(f"model: {config.model.name}")
    print(f"features: {len(config.features)}")
    print(f"train_inputs: {len(config.data.train.inputs)}")
    if config.data.test is not None:
        print(f"test_inputs: {len(config.data.test.inputs)}")
    print(f"batch_size: {config.training.batch_size}")
    print(f"lr_dense: {config.training.lr_dense}")
    print(f"lr_sparse: {config.training.lr_sparse}")
    buckets = config.data.train.reader.length_buckets
    if buckets:
        rendered = ",".join(
            f"{bucket.max_length}:{bucket.batch_size}" for bucket in buckets
        )
        print(f"train_length_buckets: {rendered}")
    print(f"vocab_strategy_hash: {vocab_strategy_fingerprint(config)}")
    return 0


def _cmd_profile(args: argparse.Namespace) -> int:
    config = _load_config(args)
    split = config.data.train if args.split == "train" else config.data.test
    if split is None:
        raise ValueError(f"split {args.split!r} is not configured")
    split.require_inputs(args.split)
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
    config.data.train.require_inputs("train")
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


def _launch_ddp_command(args: argparse.Namespace, config) -> int:
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
        return _launch_ddp_command(args, config)
    config.data.train.require_inputs("train")
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


def _cmd_benchmark(args: argparse.Namespace) -> int:
    config = _load_config(args)
    if _effective_distributed_mode(args, config) == "ddp" and not _in_distributed_launcher():
        return _launch_ddp_command(args, config)
    if args.mode in {"data", "end-to-end"}:
        config.data.train.require_inputs("train")
    report = run_benchmark(
        config,
        BenchmarkOptions(
            mode=args.mode,
            warmup_steps=args.warmup_steps,
            measured_steps=args.steps,
            profile_steps=args.profile_steps,
            seed=args.seed,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            sequence_lengths=args.sequence_lengths,
            embedding_lookups_per_table=args.embedding_lookups_per_table,
            id_distribution=args.id_distribution,
            zipf_exponent=args.zipf_exponent,
            peak_tflops=args.peak_tflops,
            reserve_hbm_gib=args.reserve_hbm_gib,
            candidates_per_request=args.candidates_per_request,
            synthetic_scenario_count=args.synthetic_scenario_count,
        ),
    )
    if is_main_process():
        print(report.to_json())
        if args.output:
            output_path = write_benchmark_report(report, args.output)
            print(f"benchmark_report path={output_path}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    config = _load_config(args)
    if config.data.test is None:
        raise ValueError("data.test is required for predict")
    config.data.test.require_inputs("test")
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
    if _effective_distributed_mode(args, config) == "ddp" and not _in_distributed_launcher():
        return _launch_ddp_command(args, config)
    split = config.data.train if args.split == "train" else config.data.test
    if split is None:
        raise ValueError(f"split {args.split!r} is not configured")
    split.require_inputs(args.split)
    result = evaluate_mdl(
        config,
        split_name=args.split,
        checkpoint_path=args.checkpoint_path,
        max_batches=args.max_batches,
        allow_random_init=args.allow_random_init,
        group_metric_name=(
            None if args.group_metric_name == "none" else args.group_metric_name
        ),
        auc_bins=args.auc_bins,
    )
    if not is_main_process():
        return 0
    print(
        f"evaluate_result rows={result.rows} "
        f"group_metric={result.group_metric_name or 'none'} "
        f"auc_histogram_bins={result.auc_histogram_bins}"
    )
    for task_name, metrics in result.metrics.items():
        formatted = " ".join(
            f"{name}={('NA' if value is None else str(value) if isinstance(value, int) else f'{value:.8f}')}"
            for name, value in metrics.items()
        )
        print(f"evaluate_task task={task_name} {formatted}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parquet-native MDL CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", required=True)
    _add_data_input_args(validate)
    _add_training_override_args(validate)
    validate.set_defaults(func=_cmd_validate_config)

    profile = subparsers.add_parser("profile")
    profile.add_argument("--config", required=True)
    profile.add_argument("--split", choices=["train", "test"], default="train")
    profile.add_argument("--max-batches", type=int, default=10)
    _add_data_input_args(profile)
    _add_training_override_args(profile)
    profile.set_defaults(func=_cmd_profile)

    fit_vocab = subparsers.add_parser("fit-vocab")
    fit_vocab.add_argument("--config", required=True)
    _add_data_input_args(fit_vocab)
    fit_vocab.set_defaults(func=_cmd_fit_vocab)

    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--distributed", choices=["none", "ddp"], default=None)
    train.add_argument("--nproc-per-node", type=int, default=None)
    train.add_argument("--master-addr", default=None)
    train.add_argument("--master-port", type=int, default=None)
    _add_data_input_args(train)
    _add_training_override_args(train)
    train.set_defaults(func=_cmd_train)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument(
        "--mode",
        choices=["data", "embedding", "compute", "end-to-end"],
        required=True,
    )
    benchmark.add_argument("--warmup-steps", type=int, default=10)
    benchmark.add_argument("--steps", type=int, default=50)
    benchmark.add_argument("--profile-steps", type=int, default=1)
    benchmark.add_argument("--seed", type=int, default=2025)
    benchmark.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="per-rank synthetic batch size for compute-only benchmarking",
    )
    benchmark.add_argument("--sequence-length", type=int, default=None)
    benchmark.add_argument(
        "--sequence-lengths",
        type=_parse_named_positive_ints,
        default={},
        help="compute-only per-sequence lengths, e.g. impr=256,clk_long=512",
    )
    benchmark.add_argument("--embedding-lookups-per-table", type=int, default=65536)
    benchmark.add_argument(
        "--id-distribution", choices=["uniform", "zipf"], default="uniform"
    )
    benchmark.add_argument("--zipf-exponent", type=float, default=1.2)
    benchmark.add_argument("--peak-tflops", type=float, default=None)
    benchmark.add_argument(
        "--reserve-hbm-gib",
        type=float,
        default=0.0,
        help="compute-only HBM reservation for embedding weights and optimizer state",
    )
    benchmark.add_argument(
        "--candidates-per-request",
        type=int,
        default=1,
        help="compute-only agg simulation: candidates sharing one request",
    )
    benchmark.add_argument(
        "--synthetic-scenario-count",
        type=int,
        default=32,
        help=(
            "number of synthetic scenarios used by compute and embedding "
            "benchmark modes when scenarios.auto_discover is true"
        ),
    )
    benchmark.add_argument("--output", default=None)
    benchmark.add_argument("--distributed", choices=["none", "ddp"], default=None)
    benchmark.add_argument("--nproc-per-node", type=int, default=None)
    benchmark.add_argument("--master-addr", default=None)
    benchmark.add_argument("--master-port", type=int, default=None)
    _add_data_input_args(benchmark)
    benchmark.set_defaults(func=_cmd_benchmark)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--config", required=True)
    predict.add_argument("--output-path", default=None)
    predict.add_argument("--max-batches", type=int, default=None)
    predict.add_argument("--allow-random-init", action="store_true")
    _add_data_input_args(predict)
    _add_training_override_args(predict)
    predict.set_defaults(func=_cmd_predict)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--split", choices=["train", "test"], default="test")
    evaluate.add_argument("--max-batches", type=int, default=None)
    evaluate.add_argument("--allow-random-init", action="store_true")
    evaluate.add_argument(
        "--group-metric-name", choices=["none", "qauc", "uauc"], default="none"
    )
    evaluate.add_argument("--auc-bins", type=int, default=65536)
    evaluate.add_argument("--distributed", choices=["none", "ddp"], default=None)
    evaluate.add_argument("--nproc-per-node", type=int, default=None)
    evaluate.add_argument("--master-addr", default=None)
    evaluate.add_argument("--master-port", type=int, default=None)
    _add_data_input_args(evaluate)
    _add_training_override_args(evaluate)
    evaluate.set_defaults(func=_cmd_evaluate)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
