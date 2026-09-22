#!/usr/bin/env python
"""Break down table_iter next() with/without producer runway."""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from src.config import load_app_config
from src.features import load_vocab_maps
from src.train import _iter_batch_tables, _prepare_feature_batch


def _run(label: str, sleep_s: float) -> None:
    cfg = load_app_config("artifacts/gpu_util_e2e_mock/rankmixer_e2e.yaml")
    vocab = load_vocab_maps(cfg)
    split = cfg.data.train
    assert split is not None
    reader = split.reader
    pin = bool(reader.pin_memory)
    coalesce = bool(reader.coalesce_pinned_tensors and pin)
    it = _iter_batch_tables(
        cfg, "train", shard_rank=0, shard_world_size=1, require_labels=True
    )
    for _ in range(3):
        table = next(it)
        _prepare_feature_batch(
            cfg, split, table, vocab, True, pin, coalesce, True
        )
    fetches: list[float] = []
    preps: list[float] = []
    for _ in range(6):
        if sleep_s > 0:
            time.sleep(sleep_s)
        t0 = time.perf_counter()
        table = next(it)
        t1 = time.perf_counter()
        _prepare_feature_batch(
            cfg, split, table, vocab, True, pin, coalesce, True
        )
        t2 = time.perf_counter()
        fetches.append((t1 - t0) * 1000)
        preps.append((t2 - t1) * 1000)
    print(
        label,
        "fetch",
        round(sum(fetches) / len(fetches), 1),
        [round(x, 1) for x in fetches],
        "prep",
        round(sum(preps) / len(preps), 1),
    )


def main() -> None:
    _run("back-to-back", 0.0)
    _run("sleep_200ms", 0.20)
    _run("sleep_400ms", 0.40)


if __name__ == "__main__":
    main()
