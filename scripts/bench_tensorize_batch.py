#!/usr/bin/env python3
"""Repeat tensorization of one fixed packed axis batch for stable A/B timings."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_app_config
from src.dataloader import axis_batch_to_feature_batch
from src.features import load_vocab_maps
from src.train import _iter_batch_tables


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    config = load_app_config(args.config)
    split = config.data.train
    assert split is not None
    vocab = load_vocab_maps(config)
    iterator = _iter_batch_tables(
        config, "train", shard_rank=0, shard_world_size=1, require_labels=True
    )
    try:
        table = None
        for _ in range(4):
            table = next(iterator)
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()

    def tensorize():
        return axis_batch_to_feature_batch(
            config,
            table,
            vocab,
            require_labels=True,
            include_group_id=False,
            split=split,
        )

    for _ in range(args.warmup):
        tensorize()
    started = perf_counter()
    for _ in range(args.repeats):
        batch = tensorize()
    elapsed = perf_counter() - started
    print(
        f"tensorize_ms={elapsed / args.repeats * 1000:.3f} "
        f"rows={batch.scenario_id.numel()} repeats={args.repeats}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
