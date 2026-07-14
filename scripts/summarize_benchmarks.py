#!/usr/bin/env python3
"""Summarize benchmark JSON files and compute 1-GPU scaling efficiency."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


def _mean_present(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _fmt(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _load_reports(directory: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read benchmark report {path}: {error}") from error
        payload["_path"] = str(path)
        reports.append(payload)
    if not reports:
        raise ValueError(f"no benchmark JSON reports found under {directory}")
    return reports


def _group_key(report: dict[str, Any]) -> tuple[Any, ...]:
    options = report.get("benchmark_options", {})
    per_rank_batch = options.get("batch_size")
    fixed_global_batch = (
        int(per_rank_batch) * int(report["world_size"])
        if per_rank_batch is not None
        else None
    )
    return (
        report["mode"],
        options.get("id_distribution", "uniform"),
        options.get("sequence_length"),
        fixed_global_batch,
    )


def summarize(directory: Path) -> None:
    reports = _load_reports(directory)
    one_gpu_throughput = {
        _group_key(report): float(report["samples_per_second"])
        for report in reports
        if int(report["world_size"]) == 1
    }
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    writer.writerow(
        [
            "mode",
            "id_distribution",
            "sequence_length",
            "fixed_global_batch",
            "gpus",
            "samples/s",
            "tokens/s",
            "mean_step_ms",
            "p95_step_ms",
            "scaling_efficiency",
            "peak_hbm_gib",
            "gpu_util_percent",
            "mfu",
            "dataloader_wait_ratio",
            "sparse_payload_mib",
            "profiled_comm_ms",
            "padding_ratio",
            "attention_kernels",
            "report",
        ]
    )
    for report in sorted(reports, key=lambda item: (_group_key(item), item["world_size"])):
        options = report.get("benchmark_options", {})
        world_size = int(report["world_size"])
        baseline = one_gpu_throughput.get(_group_key(report))
        efficiency = None
        if baseline is not None and baseline > 0.0:
            efficiency = float(report["samples_per_second"]) / (baseline * world_size)
        hbm = report.get("peak_hbm_allocated_bytes_per_rank", [])
        peak_hbm_gib = max(hbm, default=0) / (1024**3)
        per_rank_batch = options.get("batch_size")
        fixed_global_batch = (
            int(per_rank_batch) * world_size if per_rank_batch is not None else None
        )
        writer.writerow(
            [
                report["mode"],
                options.get("id_distribution", "uniform"),
                options.get("sequence_length"),
                fixed_global_batch or "",
                world_size,
                _fmt(float(report["samples_per_second"]), 2),
                _fmt(float(report["tokens_per_second"]), 2),
                _fmt(float(report["mean_step_seconds"]) * 1000.0, 3),
                _fmt(float(report["p95_step_seconds"]) * 1000.0, 3),
                _fmt(efficiency, 4),
                _fmt(peak_hbm_gib, 3),
                _fmt(_mean_present(report.get("gpu_utilization_percent_per_rank", [])), 2),
                _fmt(report.get("mfu"), 4),
                _fmt(float(report["dataloader_wait_ratio"]), 4),
                _fmt(float(report["sparse_payload_bytes_per_step_rank_max"]) / (1024**2), 3),
                _fmt(
                    None
                    if report.get("profiled_communication_operator_seconds_rank_max") is None
                    else float(report["profiled_communication_operator_seconds_rank_max"])
                    * 1000.0,
                    3,
                ),
                _fmt(float(report["padding_ratio"]), 4),
                ",".join(report.get("attention_kernels", [])),
                report["_path"],
            ]
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        summarize(args.directory)
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
