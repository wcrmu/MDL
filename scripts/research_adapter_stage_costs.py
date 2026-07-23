#!/usr/bin/env python3
"""Read-only research: stage-level timing of the data producer.

Does not modify src/*. Wraps existing callables at runtime to split:

  scan/decode
  adapter: Arrow→Python (pylist)
  adapter: expand/normalize/rebuild (everything else inside adapt)
  flat Arrow → FeatureBatch tensorize
  coalesce/pin

Usage:
  python scripts/research_adapter_stage_costs.py \\
    --config artifacts/mock_full_rankmixer_capped_b512_adapter4.yaml \\
    --data-dir artifacts/mock_parquet_full_2x2500_zstd \\
    --max-raw-tables 8 --warmup 1
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from dataclasses import replace
from typing import Any

import src.dataloader as D
from src.config import load_app_config
from src.dataloader import (
    _adapter_context,
    _mdl_rankmixer_adapter_plan,
    pin_feature_batch,
    required_columns_for_split,
    resolve_auto_scenarios,
    table_to_feature_batch,
)


class StageTimer:
    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, name: str, seconds: float) -> None:
        self.totals[name] += seconds
        self.counts[name] += 1

    def report(self, candidate_rows: int) -> None:
        total = sum(self.totals.values()) or 1e-9
        print("\n=== stage breakdown (wall, sequential wraps) ===")
        for name, seconds in sorted(self.totals.items(), key=lambda kv: -kv[1]):
            share = 100.0 * seconds / total
            print(
                f"{name:32s}  {seconds:8.3f}s  {share:5.1f}%  "
                f"n={self.counts[name]}"
            )
        print(f"{'TOTAL_STAGED':32s}  {total:8.3f}s")
        if candidate_rows:
            print(f"candidate_rows={candidate_rows}")
            print(f"staged_candidates_per_second={candidate_rows / total:.1f}")
            for name, seconds in sorted(self.totals.items(), key=lambda kv: -kv[1]):
                if seconds > 0:
                    print(
                        f"  if-only {name}: {candidate_rows / seconds:.1f} cand/s"
                    )


def _override_inputs(config, data_dir: str, **reader_updates):
    reader = (
        replace(config.data.train.reader, **reader_updates)
        if reader_updates
        else config.data.train.reader
    )
    # Force single-process adapter so stage wraps stay in this process.
    reader = replace(reader, adapter_workers=0, prefetch_batches=0, num_workers=1)
    train = replace(config.data.train, inputs=(data_dir,), reader=reader)
    return replace(config, data=replace(config.data, train=train))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="artifacts/mock_full_rankmixer_capped_b512_adapter4.yaml",
    )
    parser.add_argument(
        "--data-dir", default="artifacts/mock_parquet_full_2x2500_zstd"
    )
    parser.add_argument("--max-raw-tables", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--scanner-batch-rows", type=int, default=64)
    args = parser.parse_args()

    config = _override_inputs(
        load_app_config(args.config),
        args.data_dir,
        scanner_batch_rows=args.scanner_batch_rows,
    )
    if config.scenarios.auto_discover:
        config = resolve_auto_scenarios(config)
    split = config.data.train
    required = required_columns_for_split(config, split)
    plan = _mdl_rankmixer_adapter_plan(_adapter_context("train", split, required))

    print(f"config={args.config}")
    print(f"model={config.model.name} features={len(config.features)} sequences={len(config.sequences)}")
    print(f"required_columns={len(required)} raw_sequence_columns={len(plan.raw_sequence_columns)}")
    print(f"context_features={len(plan.context_features)} item_features={len(plan.item_features)}")
    print(f"bag_features={len(plan.bag_features)} scanner_batch_rows={split.reader.scanner_batch_rows}")
    print("note: adapter_workers forced to 0 for in-process stage attribution")

    timer = StageTimer()
    original_to_python = D._adapter_table_to_python
    original_adapt = D.adapt_mdl_rankmixer_parquet
    original_output_array = D._output_array

    pylist_seconds = 0.0
    rebuild_array_seconds = 0.0
    adapt_depth = 0

    def timed_to_python(*a, **kw):
        nonlocal pylist_seconds
        start = time.perf_counter()
        result = original_to_python(*a, **kw)
        # Only attribute the outermost adapt's pylist materialization.
        if adapt_depth == 1:
            pylist_seconds += time.perf_counter() - start
        return result

    def timed_output_array(*a, **kw):
        nonlocal rebuild_array_seconds
        start = time.perf_counter()
        result = original_output_array(*a, **kw)
        if adapt_depth == 1:
            rebuild_array_seconds += time.perf_counter() - start
        return result

    def timed_adapt(table, *, context):
        nonlocal pylist_seconds, rebuild_array_seconds, adapt_depth
        adapt_depth += 1
        try:
            if adapt_depth > 1:
                # Nested one-row contract warm-up: do not double-count stages.
                return original_adapt(table, context=context)
            pylist_before = pylist_seconds
            rebuild_before = rebuild_array_seconds
            start = time.perf_counter()
            result = original_adapt(table, context=context)
            total = time.perf_counter() - start
            pylist = pylist_seconds - pylist_before
            rebuild = rebuild_array_seconds - rebuild_before
            expand = max(0.0, total - pylist - rebuild)
            timer.add("adapt.pylist_arrow_to_python", pylist)
            timer.add("adapt.expand_normalize_python", expand)
            timer.add("adapt.rebuild_python_to_arrow", rebuild)
            return result
        finally:
            adapt_depth -= 1

    D._adapter_table_to_python = timed_to_python  # type: ignore[assignment]
    D.adapt_mdl_rankmixer_parquet = timed_adapt  # type: ignore[assignment]
    D._output_array = timed_output_array  # type: ignore[assignment]

    scan_columns = D._scan_columns_for_split(split, required)
    scanner = D.ParquetScanner(split, scan_columns)
    context = _adapter_context("train", split, required)
    vocab_maps: dict[str, dict[str, int]] = {}

    candidate_rows = 0
    raw_seen = 0
    measured = 0
    table_iter = iter(scanner.iter_tables())

    # Drain warmup without mixing nested sample-adapt into measured totals.
    for _ in range(max(0, args.warmup)):
        try:
            raw_table = next(table_iter)
        except StopIteration:
            break
        _ = original_adapt(raw_table, context=context)
        raw_seen += 1

    # Reset stage timers after warmup / recursive sample validation.
    timer = StageTimer()
    pylist_seconds = 0.0
    rebuild_array_seconds = 0.0

    while measured < args.max_raw_tables:
        start = time.perf_counter()
        try:
            raw_table = next(table_iter)
        except StopIteration:
            break
        timer.add("scan.decode_and_yield_raw_table", time.perf_counter() - start)

        flat = D.adapt_mdl_rankmixer_parquet(raw_table, context=context)
        tables = [flat] if hasattr(flat, "num_rows") else list(flat)

        for table in tables:
            candidate_rows += int(table.num_rows)
            start = time.perf_counter()
            batch = table_to_feature_batch(
                config,
                table,
                vocab_maps,
                require_labels=True,
                include_group_id=False,
                split=split,
            )
            timer.add("tensorize.table_to_feature_batch", time.perf_counter() - start)

            start = time.perf_counter()
            try:
                _ = pin_feature_batch(batch, coalesce_tensors=False)
                timer.add("host.pin_only", time.perf_counter() - start)
            except RuntimeError as error:
                timer.add("host.pin_skipped", time.perf_counter() - start)
                if measured == 0:
                    print(f"pin skipped: {error}")

        measured += 1
        raw_seen += 1

    print(f"measured_raw_tables={measured} warmup_raw_tables={args.warmup}")

    # Also time raw to_pylist on one table for column-category cost.
    scanner2 = D.ParquetScanner(split, scan_columns)
    raw = next(iter(scanner2.iter_tables()))
    seq_cols = set(plan.raw_sequence_columns)
    cat_timings: dict[str, float] = defaultdict(float)
    cat_counts: dict[str, int] = defaultdict(int)
    cat_bytes: dict[str, int] = defaultdict(int)
    for name in raw.column_names:
        array = raw[name].combine_chunks()
        if name in seq_cols:
            cat = "sequence"
        elif name in plan.bag_features or name.endswith("_hn"):
            # Rough: many bags are item/context list columns.
            cat = "list_or_bag_like"
        else:
            cat = "other"
        start = time.perf_counter()
        values = D._arrow_array_to_pylist(D._require_pyarrow()[0], array)
        elapsed = time.perf_counter() - start
        cat_timings[cat] += elapsed
        cat_counts[cat] += 1
        # crude size proxy
        cat_bytes[cat] += len(str(type(values))) + (len(values) if isinstance(values, list) else 0)

    print(f"\nraw_table_sample columns={raw.num_columns} rows={raw.num_rows}")
    print("optimized _arrow_array_to_pylist by rough category (one raw table):")
    for cat, seconds in sorted(cat_timings.items(), key=lambda kv: -kv[1]):
        print(f"  {cat}: {seconds*1000:.1f} ms over {cat_counts[cat]} columns")

    # Generic to_pylist vs optimized for list columns only.
    list_opt = 0.0
    list_generic = 0.0
    list_n = 0
    pa, *_ = D._require_pyarrow()
    for name in raw.column_names:
        array = raw[name].combine_chunks()
        if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type)):
            continue
        start = time.perf_counter()
        _ = D._arrow_array_to_pylist(pa, array)
        list_opt += time.perf_counter() - start
        start = time.perf_counter()
        _ = array.to_pylist()
        list_generic += time.perf_counter() - start
        list_n += 1
        if list_n >= 40:
            break
    if list_n:
        print(
            f"\nlist-column sample n={list_n}: "
            f"optimized={list_opt*1000:.1f}ms generic_to_pylist={list_generic*1000:.1f}ms "
            f"speedup={list_generic / max(list_opt, 1e-9):.2f}x"
        )

    timer.report(candidate_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
