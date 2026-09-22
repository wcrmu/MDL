#!/usr/bin/env python3
"""Profile the production direct host pipeline one prepared batch at a time.

Unlike the older flat-table profiler, this follows the path used by training:
scan/adapt/pack -> axis tensorization -> dtype coalescing -> shared-memory IPC.
It intentionally runs outside the CUDA training process so stage timings are
not hidden inside the host-prepare child.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_app_config
from src.features import load_vocab_maps
from src.train import (
    _coalesce_feature_batch,
    _iter_batch_tables,
    _prepare_feature_batch,
    _share_feature_batch_payload_for_ipc,
)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction) - 1))
    return ordered[index]


def _summary(name: str, values: list[float]) -> str:
    return (
        f"{name}_ms mean={statistics.fmean(values) * 1000:.2f} "
        f"p50={statistics.median(values) * 1000:.2f} "
        f"p95={_percentile(values, 0.95) * 1000:.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument(
        "--include-output-metadata",
        action="store_true",
        help="Materialize group IDs and prediction keys as evaluation does.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Also time compact shared-memory publication.",
    )
    args = parser.parse_args()

    config = load_app_config(args.config)
    split = config.data.train
    if split is None:
        raise ValueError("training split is required")
    # We are profiling the body of host prepare, not recursively spawning it.
    reader = replace(
        split.reader,
        host_prepare_prefetch=0,
        device_prefetch_batches=0,
        pin_memory=False,
    )
    split = replace(split, reader=reader)
    config = replace(config, data=replace(config.data, train=split))
    vocab_maps = load_vocab_maps(config)
    iterator = _iter_batch_tables(
        config,
        "train",
        shard_rank=0,
        shard_world_size=1,
        require_labels=True,
    )

    timings: dict[str, list[float]] = {
        "fetch_pack": [],
        "tensorize": [],
        "coalesce": [],
        "share_publish": [],
        "total": [],
    }
    rows: list[int] = []
    total_steps = max(0, args.warmup) + max(1, args.steps)
    try:
        for index in range(total_steps):
            started = perf_counter()
            table = next(iterator)
            fetched = perf_counter()
            batch = _prepare_feature_batch(
                config,
                split,
                table,
                vocab_maps,
                True,
                False,
                False,
                args.include_output_metadata,
            )
            tensorized = perf_counter()
            batch = _coalesce_feature_batch(
                batch,
                pin_memory=False,
                shared_memory=args.share,
            )
            coalesced = perf_counter()
            payload: dict[str, Any] | None = None
            if args.share:
                payload = _share_feature_batch_payload_for_ipc(batch)
            published = perf_counter()
            if index >= args.warmup:
                timings["fetch_pack"].append(fetched - started)
                timings["tensorize"].append(tensorized - fetched)
                timings["coalesce"].append(coalesced - tensorized)
                timings["share_publish"].append(published - coalesced)
                timings["total"].append(published - started)
                rows.append(int(batch.scenario_id.numel()))
            del payload, batch, table
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()

    print(
        f"model={config.model.name} steps={len(rows)} "
        f"rows_mean={statistics.fmean(rows):.1f} "
        f"adapter_workers={reader.adapter_workers} num_workers={reader.num_workers}"
    )
    for name in ("fetch_pack", "tensorize", "coalesce", "share_publish", "total"):
        print(_summary(name, timings[name]))
    total_rows = sum(rows)
    total_seconds = sum(timings["total"])
    print(f"host_pipeline_samples_per_second={total_rows / total_seconds:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
