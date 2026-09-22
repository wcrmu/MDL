#!/usr/bin/env python3
"""Profile the production axis adapter without host/train IPC noise."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_app_config
from src.dataloader import (
    ParquetScanner,
    _adapter_context,
    _load_parquet_adapter,
    _optional_scan_columns_for_split,
    _scan_columns_for_split,
    required_columns_for_split,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    config = load_app_config(args.config)
    split = config.data.train
    assert split is not None and split.request_id is not None
    required = required_columns_for_split(config, split, require_labels=True)
    scan_columns = _scan_columns_for_split(split, required)
    scanner = ParquetScanner(
        split,
        scan_columns,
        shard_rank=0,
        shard_world_size=1,
        optional_columns=set(_optional_scan_columns_for_split(split))
        & set(scan_columns),
    )
    adapter_name, adapter = _load_parquet_adapter(split)
    context = _adapter_context("train", split, required)
    context._runtime_cache["axis_separated"] = True
    context._runtime_cache["axis_request_id_column"] = split.request_id
    iterator = scanner.iter_tables()
    rows = candidates = 0
    started = perf_counter()
    try:
        for _ in range(args.steps):
            table = next(iterator)
            rows += int(table.num_rows)
            bundle = adapter(table, context=context)
            candidates += int(bundle.n_candidates)
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    elapsed = perf_counter() - started
    print(
        f"adapter={adapter_name} raw_rows={rows} candidates={candidates} "
        f"seconds={elapsed:.3f} raw_rows_per_second={rows / elapsed:.1f} "
        f"candidates_per_second={candidates / elapsed:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
