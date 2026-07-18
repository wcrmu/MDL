#!/usr/bin/env python3
"""Tune A100 batch size with a local production-shaped Parquet pipeline.

By default each trial runs the full path: gzip Parquet projection/decode, agg
request filtering, adapter conversion, feature tensorization, pinned-memory
prefetch, H2D, real sharded embeddings, forward/backward, and optimizers. Local
files deliberately avoid HDFS access; the report also provides remote-bandwidth
sensitivity from projected compressed bytes per candidate.

``--compute-only`` retains the lighter graph/HBM estimator for quick iteration.
Every candidate runs in a fresh DDP process group so a CUDA OOM cannot poison
later trials.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "src" / "main.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_app_config
from scripts.generate_synthetic_agg_parquet import (
    OBSERVED_MEDIAN_SEQUENCE_LENGTHS,
    SyntheticAggManifest,
    generate_synthetic_agg_dataset,
)


_MEDIAN_SEQUENCE_LENGTHS = {
    **OBSERVED_MEDIAN_SEQUENCE_LENGTHS,
    "task_fst_cart_prior": 210,
    "task_upid_pay_prior": 22,
    "task_cateid_filter_prior": 22,
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _candidate_batches(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("candidate batches must be positive integers")
    return values


def _named_lengths(raw: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in raw.split(","):
        name, separator, value_text = item.strip().partition("=")
        if not separator or not name:
            raise argparse.ArgumentTypeError(
                "sequence lengths must be comma-separated name=integer entries"
            )
        try:
            value = int(value_text)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid sequence length {value_text!r} for {name!r}"
            ) from error
        if value <= 0:
            raise argparse.ArgumentTypeError("sequence lengths must be positive")
        result[name] = value
    return result


def _default_sequence_lengths(config_path: Path) -> dict[str, int]:
    config = load_app_config(config_path)
    result: dict[str, int] = {}
    for sequence in config.sequences:
        preferred = _MEDIAN_SEQUENCE_LENGTHS.get(
            sequence.name,
            sequence.max_length or 128,
        )
        if sequence.max_length is not None:
            preferred = min(preferred, sequence.max_length)
        result[sequence.name] = preferred
    return result


def _render_named_lengths(values: dict[str, int]) -> str:
    return ",".join(f"{name}={value}" for name, value in values.items())


def _tail(text: str, lines: int = 60) -> str:
    return "\n".join(text.splitlines()[-lines:])


def _write_trial_overlay(
    *,
    base_config: Path,
    output_path: Path,
    parquet_dir: Path,
    scenario_cache: Path,
    batch_size: int,
    scanner_batch_rows: int,
) -> None:
    payload = {
        "extends": str(base_config),
        "scenarios": {"discovery_cache_path": str(scenario_cache)},
        "data": {
            "train": {
                "inputs": [str(parquet_dir)],
                "reader": {
                    # The generated workload has one controlled length class;
                    # use the trial batch directly instead of inherited buckets.
                    "length_buckets": [],
                    "scanner_batch_rows": scanner_batch_rows,
                    "eager_schema_validation": "all",
                },
            }
        },
        "training": {"batch_size": batch_size},
    }
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _hdfs_sensitivity(
    manifest: SyntheticAggManifest,
    *,
    nproc_per_node: int,
    open_latency_ms: float,
    bandwidths_mib_s: Sequence[float] = (512.0, 1024.0, 2048.0, 4096.0),
) -> list[dict[str, float]]:
    candidates_per_file = manifest.candidates / manifest.files
    open_rounds = math.ceil(manifest.files / nproc_per_node)
    open_seconds = open_rounds * open_latency_ms / 1000.0
    result: list[dict[str, float]] = []
    for bandwidth in bandwidths_mib_s:
        transfer_seconds = manifest.projected_compressed_bytes / (
            bandwidth * 1024.0 * 1024.0
        )
        wall_seconds = transfer_seconds + open_seconds
        result.append(
            {
                "aggregate_bandwidth_mib_s": bandwidth,
                "open_latency_ms": open_latency_ms,
                "estimated_samples_per_second_ceiling": (
                    manifest.candidates / wall_seconds
                    if wall_seconds > 0.0
                    else math.inf
                ),
                "candidates_per_file": candidates_per_file,
            }
        )
    return result


def _run_candidate(
    args: argparse.Namespace,
    batch_size: int,
    *,
    workspace: Path,
    parquet_dir: Path | None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="mdl-a100-tune-") as temporary:
        report_path = Path(temporary) / "report.json"
        benchmark_mode = "compute" if args.compute_only else "end-to-end"
        if args.compute_only:
            trial_config = args.config
        else:
            if parquet_dir is None:
                raise RuntimeError("end-to-end tuning requires synthetic Parquet")
            trial_config = workspace / f"trial_batch_{batch_size}.yaml"
            _write_trial_overlay(
                base_config=args.config,
                output_path=trial_config,
                parquet_dir=parquet_dir,
                scenario_cache=workspace / "scenario_cache.json",
                batch_size=batch_size,
                scanner_batch_rows=args.scanner_batch_rows,
            )
        command = [
            sys.executable,
            str(MAIN),
            "benchmark",
            "--config",
            str(trial_config),
            "--mode",
            benchmark_mode,
            "--warmup-steps",
            str(args.warmup_steps),
            "--steps",
            str(args.steps),
            "--profile-steps",
            str(args.profile_steps),
            "--peak-tflops",
            str(args.peak_tflops),
            "--distributed",
            "ddp",
            "--nproc-per-node",
            str(args.nproc_per_node),
            "--master-port",
            str(_free_port()),
            "--output",
            str(report_path),
        ]
        if args.compute_only:
            command.extend(
                [
                    "--batch-size",
                    str(batch_size),
                    "--reserve-hbm-gib",
                    str(args.reserve_hbm_gib),
                    "--candidates-per-request",
                    str(args.candidates_per_request),
                    "--sequence-lengths",
                    _render_named_lengths(args.sequence_lengths),
                ]
            )
        environment = os.environ.copy()
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        environment.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        environment.setdefault("OMP_NUM_THREADS", str(args.omp_num_threads))
        print(f"trial batch_size={batch_size}", flush=True)
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        combined = "\n".join((result.stdout, result.stderr))
        if result.returncode != 0 or not report_path.exists():
            lowered = combined.lower()
            status = "oom" if "out of memory" in lowered else "failed"
            print(
                f"trial_result batch_size={batch_size} status={status} "
                f"returncode={result.returncode}",
                flush=True,
            )
            if status != "oom":
                print(_tail(combined), file=sys.stderr, flush=True)
            return {
                "batch_size": batch_size,
                "status": status,
                "returncode": result.returncode,
                "error_tail": _tail(combined),
            }

        report = json.loads(report_path.read_text(encoding="utf-8"))
        peak_hbm = max(report["peak_hbm_reserved_bytes_per_rank"], default=0)
        trial = {
            "batch_size": batch_size,
            "status": "ok",
            "benchmark_mode": benchmark_mode,
            "samples_per_second": float(report["samples_per_second"]),
            "tokens_per_second": float(report["tokens_per_second"]),
            "p95_step_seconds": float(report["p95_step_seconds"]),
            "dataloader_wait_ratio": float(report["dataloader_wait_ratio"]),
            "mean_dataloader_wait_seconds": float(
                report["mean_dataloader_wait_seconds"]
            ),
            "mean_h2d_seconds": float(report["mean_h2d_seconds"]),
            "mean_forward_seconds": float(report["mean_forward_seconds"]),
            "mean_backward_seconds": float(report["mean_backward_seconds"]),
            "mean_optimizer_seconds": float(report["mean_optimizer_seconds"]),
            "peak_hbm_reserved_gib": peak_hbm / (1024 ** 3),
            "host_batch_bytes_peak_per_rank": report[
                "host_batch_bytes_peak_per_rank"
            ],
            "gpu_utilization_percent_per_rank": report[
                "gpu_utilization_percent_per_rank"
            ],
            "mfu": report.get("mfu"),
        }
        print(
            "trial_result "
            f"batch_size={batch_size} status=ok "
            f"samples_per_second={trial['samples_per_second']:.2f} "
            f"peak_hbm_reserved_gib={trial['peak_hbm_reserved_gib']:.2f}",
            flush=True,
        )
        return trial


def _run_reader_trial(
    args: argparse.Namespace,
    scanner_batch_rows: int,
    *,
    workspace: Path,
    parquet_dir: Path,
) -> dict[str, Any]:
    trial_config = workspace / f"reader_rows_{scanner_batch_rows}.yaml"
    _write_trial_overlay(
        base_config=args.config,
        output_path=trial_config,
        parquet_dir=parquet_dir,
        scenario_cache=workspace / "scenario_cache.json",
        batch_size=args.reader_benchmark_batch_size,
        scanner_batch_rows=scanner_batch_rows,
    )
    with tempfile.TemporaryDirectory(prefix="mdl-a100-reader-") as temporary:
        report_path = Path(temporary) / "report.json"
        command = [
            sys.executable,
            str(MAIN),
            "benchmark",
            "--config",
            str(trial_config),
            "--mode",
            "data",
            "--warmup-steps",
            str(args.reader_warmup_steps),
            "--steps",
            str(args.reader_steps),
            "--profile-steps",
            "0",
            "--distributed",
            "ddp",
            "--nproc-per-node",
            str(args.nproc_per_node),
            "--master-port",
            str(_free_port()),
            "--output",
            str(report_path),
        ]
        environment = os.environ.copy()
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        environment.setdefault("OMP_NUM_THREADS", str(args.omp_num_threads))
        print(f"reader_trial scanner_batch_rows={scanner_batch_rows}", flush=True)
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        combined = "\n".join((result.stdout, result.stderr))
        if result.returncode != 0 or not report_path.exists():
            print(_tail(combined), file=sys.stderr, flush=True)
            return {
                "scanner_batch_rows": scanner_batch_rows,
                "status": "failed",
                "returncode": result.returncode,
                "error_tail": _tail(combined),
            }
        report = json.loads(report_path.read_text(encoding="utf-8"))
        trial = {
            "scanner_batch_rows": scanner_batch_rows,
            "status": "ok",
            "samples_per_second": float(report["samples_per_second"]),
            "p95_dataloader_wait_seconds": float(
                report["p95_dataloader_wait_seconds"]
            ),
            "cpu_utilization_percent_per_rank": report[
                "cpu_utilization_percent_per_rank"
            ],
            "process_peak_rss_bytes_per_rank": report[
                "process_peak_rss_bytes_per_rank"
            ],
            "host_batch_bytes_peak_per_rank": report[
                "host_batch_bytes_peak_per_rank"
            ],
        }
        print(
            "reader_trial_result "
            f"scanner_batch_rows={scanner_batch_rows} "
            f"samples_per_second={trial['samples_per_second']:.2f}",
            flush=True,
        )
        return trial


def _recommended_yaml_override(
    config_path: Path,
    batch_size: int,
    sequence_lengths: Mapping[str, int],
    scanner_batch_rows: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    config = load_app_config(config_path)
    reader = config.data.train.reader
    override: dict[str, Any] = {"training": {"batch_size": batch_size}}
    if scanner_batch_rows is not None:
        override["data"] = {
            "train": {"reader": {"scanner_batch_rows": scanner_batch_rows}}
        }
        if config.data.test is not None:
            override["data"]["test"] = {
                "reader": {"scanner_batch_rows": scanner_batch_rows}
            }
    if not reader.length_buckets:
        return override, None

    configured_lengths = []
    for sequence in config.sequences:
        value = sequence_lengths.get(sequence.name, sequence.max_length or 1)
        if sequence.max_length is not None:
            value = min(value, sequence.max_length)
        configured_lengths.append(value)
    workload = (
        sum(configured_lengths)
        if reader.length_bucket_metric == "sum"
        else max(configured_lengths, default=0)
    )
    target_index = len(reader.length_buckets) - 1
    for index, bucket in enumerate(reader.length_buckets):
        if bucket.max_length is None or workload <= bucket.max_length:
            target_index = index
            break
    rendered_buckets = [
        {
            "max_length": bucket.max_length,
            "batch_size": (
                batch_size if index == target_index else bucket.batch_size
            ),
        }
        for index, bucket in enumerate(reader.length_buckets)
    ]
    data_override = override.setdefault("data", {})
    train_reader = data_override.setdefault("train", {}).setdefault("reader", {})
    train_reader["length_buckets"] = rendered_buckets
    if config.data.test is not None:
        test_reader = data_override.setdefault("test", {}).setdefault("reader", {})
        test_reader["length_buckets"] = rendered_buckets
    return override, {
        "metric": reader.length_bucket_metric,
        "workload_length": workload,
        "bucket_index": target_index,
        "max_length": reader.length_buckets[target_index].max_length,
    }


def _execute_tuning(
    args: argparse.Namespace,
    *,
    workspace: Path,
    parquet_dir: Path | None,
    manifest: SyntheticAggManifest | None,
    reader_trials: Sequence[Mapping[str, Any]] = (),
    reader_samples_per_second: float | None = None,
) -> int:
    trials: list[dict[str, Any]] = []
    saw_success = False
    assumed_hdfs_ceiling: float | None = None
    sensitivity: list[dict[str, float]] = []
    if manifest is not None:
        sensitivity = _hdfs_sensitivity(
            manifest,
            nproc_per_node=args.nproc_per_node,
            open_latency_ms=args.hdfs_open_latency_ms,
        )
        if args.assumed_hdfs_bandwidth_mib_s > 0.0:
            assumed_hdfs_ceiling = _hdfs_sensitivity(
                manifest,
                nproc_per_node=args.nproc_per_node,
                open_latency_ms=args.hdfs_open_latency_ms,
                bandwidths_mib_s=(args.assumed_hdfs_bandwidth_mib_s,),
            )[0]["estimated_samples_per_second_ceiling"]

    for batch_size in args.candidate_batches:
        trial = _run_candidate(
            args,
            batch_size,
            workspace=workspace,
            parquet_dir=parquet_dir,
        )
        if trial["status"] == "ok":
            local_throughput = float(trial["samples_per_second"])
            effective = (
                min(local_throughput, assumed_hdfs_ceiling)
                if assumed_hdfs_ceiling is not None
                else local_throughput
            )
            if reader_samples_per_second is not None:
                effective = min(effective, reader_samples_per_second)
            trial["estimated_effective_samples_per_second"] = effective
            trial["assumed_hdfs_ceiling_samples_per_second"] = assumed_hdfs_ceiling
            trial["reader_ceiling_samples_per_second"] = reader_samples_per_second
        trials.append(trial)
        if trial["status"] == "ok":
            saw_success = True
            continue
        if trial["status"] == "oom" and saw_success:
            # Activation memory is monotonic for this fixed synthetic workload.
            break

    eligible = [
        trial
        for trial in trials
        if trial["status"] == "ok"
        and trial["peak_hbm_reserved_gib"] <= args.hbm_limit_gib
    ]
    if not eligible:
        print("No successful candidate stayed below the HBM limit.", file=sys.stderr)
        return 2
    best = max(
        eligible,
        key=lambda item: (
            item["estimated_effective_samples_per_second"],
            -item["peak_hbm_reserved_gib"],
        ),
    )
    yaml_override, tuned_bucket = _recommended_yaml_override(
        args.config,
        best["batch_size"],
        args.sequence_lengths,
        None if args.compute_only else args.scanner_batch_rows,
    )
    result = {
        "config": str(args.config),
        "benchmark_mode": "compute" if args.compute_only else "end-to-end",
        "nproc_per_node": args.nproc_per_node,
        "reserve_hbm_gib": args.reserve_hbm_gib if args.compute_only else None,
        "requests_per_agg": args.requests_per_agg if manifest is not None else None,
        "candidates_per_request": args.candidates_per_request,
        "sequence_lengths": args.sequence_lengths,
        "raw_parquet_sequence_lengths": args.raw_sequence_lengths,
        "hbm_limit_gib": args.hbm_limit_gib,
        "assumed_hdfs_bandwidth_mib_s": (
            args.assumed_hdfs_bandwidth_mib_s
            if args.assumed_hdfs_bandwidth_mib_s > 0.0
            else None
        ),
        "hdfs_open_latency_ms": args.hdfs_open_latency_ms,
        "hdfs_sensitivity": sensitivity,
        "reader_trials": list(reader_trials),
        "selected_scanner_batch_rows": (
            None if args.compute_only else args.scanner_batch_rows
        ),
        "reader_ceiling_samples_per_second": reader_samples_per_second,
        "synthetic_parquet": None if manifest is None else manifest.as_dict(),
        "recommended_batch_size_per_rank": best["batch_size"],
        "recommended_global_batch_size": best["batch_size"] * args.nproc_per_node,
        "tuned_length_bucket": tuned_bucket,
        "recommended_trial": best,
        "trials": trials,
        "yaml_override": yaml_override,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"tuning_report path={output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--candidate-batches",
        type=_candidate_batches,
        default=_candidate_batches("8,12,16,20,24,32,40,48,64,80,96,128"),
    )
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=0)
    parser.add_argument(
        "--compute-only",
        action="store_true",
        help="skip Parquet and use the lighter synthetic-compute estimator",
    )
    parser.add_argument("--reserve-hbm-gib", type=float, default=32.0)
    parser.add_argument("--requests-per-agg", type=int, default=4)
    parser.add_argument("--candidates-per-request", type=int, default=8)
    parser.add_argument(
        "--sequence-lengths",
        type=_named_lengths,
        default=None,
        help=(
            "per-sequence synthetic lengths; default uses observed medians "
            "clamped to each model's configured cap"
        ),
    )
    parser.add_argument("--peak-tflops", type=float, default=312.0)
    parser.add_argument("--hbm-limit-gib", type=float, default=76.0)
    parser.add_argument("--omp-num-threads", type=int, default=4)
    parser.add_argument("--synthetic-parquet-dir", type=Path, default=None)
    parser.add_argument("--synthetic-files", type=int, default=None)
    parser.add_argument("--synthetic-raw-rows-per-file", type=int, default=None)
    parser.add_argument("--sequence-overlap", type=float, default=0.85)
    parser.add_argument("--bag-length-scale", type=float, default=1.0)
    parser.add_argument("--scenario-count", type=int, default=32)
    parser.add_argument("--physical-column-count", type=int, default=630)
    parser.add_argument(
        "--scanner-batch-rows",
        type=int,
        default=None,
        help="fixed scanner batch rows; omitted means run the reader sweep",
    )
    parser.add_argument(
        "--scanner-batch-row-candidates",
        type=_candidate_batches,
        default=_candidate_batches("16,32,64"),
    )
    parser.add_argument("--reader-warmup-steps", type=int, default=8)
    parser.add_argument("--reader-steps", type=int, default=128)
    parser.add_argument("--reader-benchmark-batch-size", type=int, default=None)
    parser.add_argument("--hdfs-open-latency-ms", type=float, default=10.0)
    parser.add_argument(
        "--assumed-hdfs-bandwidth-mib-s",
        type=float,
        default=0.0,
        help=(
            "optional aggregate node bandwidth used for recommendation; zero "
            "selects by measured local full-pipeline throughput and only reports sensitivity"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    args.config = args.config.resolve()
    if not args.config.is_file():
        parser.error(f"config does not exist: {args.config}")
    base_config = load_app_config(args.config)
    if args.reader_benchmark_batch_size is None:
        args.reader_benchmark_batch_size = base_config.training.batch_size
    explicit_lengths = args.sequence_lengths
    default_lengths = _default_sequence_lengths(args.config)
    if explicit_lengths is None:
        args.sequence_lengths = default_lengths
    else:
        unknown = set(explicit_lengths) - set(default_lengths)
        if unknown:
            parser.error(
                "--sequence-lengths contains unknown sequences: "
                + ", ".join(sorted(unknown))
            )
        args.sequence_lengths = {**default_lengths, **explicit_lengths}
    args.raw_sequence_lengths = dict(OBSERVED_MEDIAN_SEQUENCE_LENGTHS)
    if explicit_lengths is not None:
        args.raw_sequence_lengths.update(
            {
                name: value
                for name, value in explicit_lengths.items()
                if name in OBSERVED_MEDIAN_SEQUENCE_LENGTHS
            }
        )
    for name in (
        "nproc_per_node",
        "steps",
        "requests_per_agg",
        "candidates_per_request",
        "omp_num_threads",
        "scenario_count",
        "physical_column_count",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")
    if args.profile_steps < 0:
        parser.error("--profile-steps must be non-negative")
    for name in ("synthetic_files", "synthetic_raw_rows_per_file"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.scanner_batch_rows is not None and args.scanner_batch_rows <= 0:
        parser.error("--scanner-batch-rows must be positive")
    if args.reader_warmup_steps < 0 or args.reader_steps <= 0:
        parser.error("reader warmup/steps must be non-negative/positive")
    if (
        args.reader_benchmark_batch_size is not None
        and args.reader_benchmark_batch_size <= 0
    ):
        parser.error("--reader-benchmark-batch-size must be positive")
    if args.reserve_hbm_gib < 0.0 or args.hbm_limit_gib <= 0.0:
        parser.error("HBM values must be non-negative/positive")
    if not 0.0 <= args.sequence_overlap <= 1.0:
        parser.error("--sequence-overlap must be in [0, 1]")
    if args.bag_length_scale <= 0.0:
        parser.error("--bag-length-scale must be positive")
    if args.hdfs_open_latency_ms < 0.0 or args.assumed_hdfs_bandwidth_mib_s < 0.0:
        parser.error("HDFS latency/bandwidth assumptions must be non-negative")

    with tempfile.TemporaryDirectory(prefix="mdl-a100-full-tune-") as temporary:
        workspace = Path(temporary)
        if args.compute_only:
            return _execute_tuning(
                args,
                workspace=workspace,
                parquet_dir=None,
                manifest=None,
            )

        parquet_dir = (
            args.synthetic_parquet_dir.resolve()
            if args.synthetic_parquet_dir is not None
            else workspace / "synthetic_parquet"
        )
        files = args.synthetic_files or (args.nproc_per_node * 2)
        if files < args.nproc_per_node:
            parser.error("--synthetic-files must be at least --nproc-per-node")
        max_batch = max(args.candidate_batches)
        full_required_candidates = (
            (args.warmup_steps + args.steps)
            * max_batch
            * args.nproc_per_node
        )
        reader_required_candidates = (
            (args.reader_warmup_steps + args.reader_steps)
            * args.reader_benchmark_batch_size
            * args.nproc_per_node
        )
        required_candidates = max(
            full_required_candidates,
            reader_required_candidates,
        )
        candidates_per_raw_row = (
            args.requests_per_agg * args.candidates_per_request
        )
        automatic_rows = math.ceil(
            required_candidates / (files * candidates_per_raw_row)
        )
        if args.scanner_batch_rows is None:
            automatic_rows = max(
                automatic_rows,
                max(args.scanner_batch_row_candidates),
            )
        raw_rows_per_file = args.synthetic_raw_rows_per_file or max(1, automatic_rows)
        available_candidates = files * raw_rows_per_file * candidates_per_raw_row
        if available_candidates < required_candidates:
            parser.error(
                "synthetic dataset is too small for the largest candidate; increase "
                "--synthetic-files or --synthetic-raw-rows-per-file"
            )
        print(
            "generating synthetic agg Parquet "
            f"files={files} raw_rows_per_file={raw_rows_per_file} "
            f"candidates={available_candidates}",
            flush=True,
        )
        manifest = generate_synthetic_agg_dataset(
            base_config,
            parquet_dir,
            files=files,
            raw_rows_per_file=raw_rows_per_file,
            requests_per_agg=args.requests_per_agg,
            candidates_per_request=args.candidates_per_request,
            sequence_lengths=args.raw_sequence_lengths,
            sequence_overlap=args.sequence_overlap,
            bag_length_scale=args.bag_length_scale,
            scenario_count=args.scenario_count,
            physical_column_count=args.physical_column_count,
            compression="gzip",
        )
        print(
            "synthetic_parquet_ready "
            f"projected_bytes_per_candidate="
            f"{manifest.projected_compressed_bytes_per_candidate:.1f}",
            flush=True,
        )
        reader_trials: list[dict[str, Any]] = []
        if args.scanner_batch_rows is None:
            for scanner_rows in args.scanner_batch_row_candidates:
                reader_trials.append(
                    _run_reader_trial(
                        args,
                        scanner_rows,
                        workspace=workspace,
                        parquet_dir=parquet_dir,
                    )
                )
            successful_reader_trials = [
                trial for trial in reader_trials if trial["status"] == "ok"
            ]
            if not successful_reader_trials:
                print("All synthetic Parquet reader trials failed.", file=sys.stderr)
                return 2
            selected_reader = max(
                successful_reader_trials,
                key=lambda item: item["samples_per_second"],
            )
            args.scanner_batch_rows = int(selected_reader["scanner_batch_rows"])
        else:
            selected_reader = _run_reader_trial(
                args,
                args.scanner_batch_rows,
                workspace=workspace,
                parquet_dir=parquet_dir,
            )
            reader_trials.append(selected_reader)
            if selected_reader["status"] != "ok":
                return 2
        reader_ceiling = float(selected_reader["samples_per_second"])
        print(
            "reader_profile_selected "
            f"scanner_batch_rows={args.scanner_batch_rows} "
            f"samples_per_second={reader_ceiling:.2f}",
            flush=True,
        )
        return _execute_tuning(
            args,
            workspace=workspace,
            parquet_dir=parquet_dir,
            manifest=manifest,
            reader_trials=reader_trials,
            reader_samples_per_second=reader_ceiling,
        )


if __name__ == "__main__":
    raise SystemExit(main())
