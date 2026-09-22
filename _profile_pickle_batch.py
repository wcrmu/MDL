#!/usr/bin/env python
from __future__ import annotations

import os
import pickle
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from src.config import load_app_config
from src.dataloader import axis_batch_to_feature_batch, pin_feature_batch
from src.features import load_vocab_maps
from src.train import _iter_batch_tables


def main() -> None:
    cfg = load_app_config("artifacts/gpu_util_e2e_mock/rankmixer_e2e.yaml")
    vocab = load_vocab_maps(cfg)
    split = cfg.data.train
    assert split is not None
    it = _iter_batch_tables(
        cfg, "train", shard_rank=0, shard_world_size=1, require_labels=True
    )
    for _ in range(2):
        next(it)
    table = next(it)
    t0 = time.perf_counter()
    blob = pickle.dumps(table, protocol=pickle.HIGHEST_PROTOCOL)
    t1 = time.perf_counter()
    pickle.loads(blob)
    t2 = time.perf_counter()
    print(
        "PreparedAxisBatch pickle bytes",
        len(blob),
        "dump_ms",
        round((t1 - t0) * 1000, 1),
        "load_ms",
        round((t2 - t1) * 1000, 1),
    )
    batch = axis_batch_to_feature_batch(
        cfg, table, vocab, require_labels=True, include_group_id=True, split=split
    )
    batch = pin_feature_batch(batch, coalesce_tensors=True)
    t0 = time.perf_counter()
    blob = pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)
    t1 = time.perf_counter()
    print(
        "FeatureBatch pickle bytes",
        len(blob),
        "dump_ms",
        round((t1 - t0) * 1000, 1),
    )


if __name__ == "__main__":
    main()
