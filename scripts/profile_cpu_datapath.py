"""CPU data-path profiler for the aggregated-format training pipeline.

Measures the real per-stage cost of the training data path on local Parquet:

    parquet_decode  -> ParquetScanner record batches / raw tables
    row_adaptation  -> adapt_mdl_rankmixer_parquet (agg -> per-candidate rows)
    tensorization   -> table_to_feature_batch

Reports rows/candidates/requests per second so optimization work has a
reproducible, hardware-consistent baseline that does not require a GPU.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from src.config import load_app_config
from src.dataloader import (
    _mdl_rankmixer_adapter_plan,
    _adapter_context,
    iter_flat_tables,
    required_columns_for_split,
    resolve_auto_scenarios,
    table_to_feature_batch,
)


def _override_inputs(
    config,
    data_dir: str,
    *,
    scanner_batch_rows: int | None = None,
    num_workers: int | None = None,
    prefetch_batches: int | None = None,
):
    reader_updates = {}
    if scanner_batch_rows is not None:
        reader_updates["scanner_batch_rows"] = scanner_batch_rows
    if num_workers is not None:
        reader_updates["num_workers"] = num_workers
    if prefetch_batches is not None:
        reader_updates["prefetch_batches"] = prefetch_batches
    reader = (
        replace(config.data.train.reader, **reader_updates)
        if reader_updates
        else config.data.train.reader
    )
    train = replace(config.data.train, inputs=(data_dir,), reader=reader)
    return replace(config, data=replace(config.data, train=train))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rankmixer.yaml")
    parser.add_argument(
        "--data-dir", default="artifacts/4090_bench/synthetic_parquet"
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--scanner-batch-rows", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-batches", type=int, default=None)
    parser.add_argument("--tensorize", action="store_true")
    parser.add_argument("--pylist-breakdown", action="store_true")
    args = parser.parse_args()

    config = _override_inputs(
        load_app_config(args.config),
        args.data_dir,
        scanner_batch_rows=args.scanner_batch_rows,
        num_workers=args.num_workers,
        prefetch_batches=args.prefetch_batches,
    )
    if args.tensorize and config.scenarios.auto_discover:
        # Tensorization needs concrete scenario tokens; production resolves
        # these once before training. Do the same here for a faithful path.
        config = resolve_auto_scenarios(config)
    split = config.data.train
    required = required_columns_for_split(config, split)
    print(f"config={args.config} model={config.model.name}")
    print(f"features={len(config.features)} sequences={len(config.sequences)}")
    print(f"required_columns={len(required)}")
    print(
        "reader="
        f"scanner_batch_rows={split.reader.scanner_batch_rows} "
        f"num_workers={split.reader.num_workers} "
        f"prefetch_batches={split.reader.prefetch_batches}"
    )

    # Optional: measure to_pylist cost per column category on the raw table.
    if args.pylist_breakdown:
        _pylist_breakdown(config, split)

    total_tables = 0
    total_rows = 0
    adapt_seconds = 0.0
    tensorize_seconds = 0.0

    for _ in range(max(1, args.repeat)):
        start = time.perf_counter()
        tables = list(iter_flat_tables(config, "train"))
        adapt_seconds += time.perf_counter() - start
        total_tables += len(tables)
        total_rows += sum(t.num_rows for t in tables)

        if args.tensorize:
            vocab_maps: dict[str, dict[str, int]] = {}
            start = time.perf_counter()
            for table in tables:
                table_to_feature_batch(
                    config,
                    table,
                    vocab_maps,
                    require_labels=True,
                    include_group_id=True,
                    split=split,
                )
            tensorize_seconds += time.perf_counter() - start

    rows = total_rows
    print("\n=== data path (decode + adapt) ===")
    print(f"flat_tables={total_tables} candidate_rows={rows}")
    print(f"adapt_seconds={adapt_seconds:.4f}")
    print(f"candidates_per_second={rows / adapt_seconds:.1f}")
    if args.tensorize and tensorize_seconds > 0:
        print("\n=== tensorization ===")
        print(f"tensorize_seconds={tensorize_seconds:.4f}")
        print(f"tensorized_rows_per_second={rows / tensorize_seconds:.1f}")
    return 0


def _pylist_breakdown(config, split) -> None:
    """Time to_pylist per column category on the first raw record batch."""
    import src.dataloader as D

    required = required_columns_for_split(config, split)
    context = _adapter_context("train", split, required)
    plan = _mdl_rankmixer_adapter_plan(context)

    scan_columns = D._scan_columns_for_split(split, required)
    scanner = D.ParquetScanner(split, scan_columns)
    raw_table = next(iter(scanner.iter_tables()))
    print(f"\nraw_table columns={raw_table.num_columns} rows={raw_table.num_rows}")

    seq_cols = set(plan.raw_sequence_columns)
    timings: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name in raw_table.column_names:
        array = raw_table[name].combine_chunks()
        cat = "sequence" if name in seq_cols else "other"
        start = time.perf_counter()
        _ = array.to_pylist()
        elapsed = time.perf_counter() - start
        timings[cat] = timings.get(cat, 0.0) + elapsed
        counts[cat] = counts.get(cat, 0) + 1
    print("to_pylist cost by category (first raw table):")
    for cat, seconds in sorted(timings.items(), key=lambda kv: -kv[1]):
        print(f"  {cat}: {seconds*1000:.1f} ms over {counts[cat]} columns")


if __name__ == "__main__":
    raise SystemExit(main())
