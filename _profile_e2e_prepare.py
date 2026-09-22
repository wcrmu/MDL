#!/usr/bin/env python
"""Profile fetch vs tensorize vs pin on the e2e critical path."""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from src.dataloader import PreparedAxisBatch
from src.config import load_app_config
from src.dataloader import axis_batch_to_feature_batch, pin_feature_batch
from src.features import load_vocab_maps
from src.train import _iter_batch_tables, _prepare_feature_batch


def main() -> None:
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
    for _ in range(2):
        table = next(it)
        _prepare_feature_batch(
            cfg, split, table, vocab, True, pin, coalesce, True
        )

    fetch_times: list[float] = []
    prep_times: list[float] = []
    for _ in range(8):
        t0 = time.perf_counter()
        table = next(it)
        t1 = time.perf_counter()
        _prepare_feature_batch(
            cfg, split, table, vocab, True, pin, coalesce, True
        )
        t2 = time.perf_counter()
        fetch_times.append(t1 - t0)
        prep_times.append(t2 - t1)
    print(
        "fetch_ms mean",
        round(sum(fetch_times) / len(fetch_times) * 1000, 1),
        [round(x * 1000, 1) for x in fetch_times],
    )
    print(
        "prep_ms mean",
        round(sum(prep_times) / len(prep_times) * 1000, 1),
        [round(x * 1000, 1) for x in prep_times],
    )

    it = _iter_batch_tables(
        cfg, "train", shard_rank=0, shard_world_size=1, require_labels=True
    )
    for _ in range(2):
        next(it)
    tens: list[float] = []
    pins: list[float] = []
    for _ in range(6):
        table = next(it)
        assert isinstance(table, PreparedAxisBatch)
        t0 = time.perf_counter()
        batch = axis_batch_to_feature_batch(
            cfg,
            table,
            vocab,
            require_labels=True,
            include_group_id=True,
            split=split,
        )
        t1 = time.perf_counter()
        pin_feature_batch(batch, coalesce_tensors=coalesce)
        t2 = time.perf_counter()
        tens.append(t1 - t0)
        pins.append(t2 - t1)
    print(
        "tensorize_ms mean",
        round(sum(tens) / len(tens) * 1000, 1),
        [round(x * 1000, 1) for x in tens],
    )
    print(
        "pin_coalesce_ms mean",
        round(sum(pins) / len(pins) * 1000, 1),
        [round(x * 1000, 1) for x in pins],
    )


if __name__ == "__main__":
    main()
