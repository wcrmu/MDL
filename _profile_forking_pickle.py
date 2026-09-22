#!/usr/bin/env python
"""Test ForkingPickler IPC size for shared FeatureBatch."""
from __future__ import annotations

import io
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from multiprocessing.reduction import ForkingPickler

from src.config import load_app_config
from src.dataloader import axis_batch_to_feature_batch, pin_feature_batch
from src.features import load_vocab_maps
from src.train import _iter_batch_tables


def share(obj):
    if isinstance(obj, torch.Tensor):
        return obj.share_memory_()
    if isinstance(obj, dict):
        return {k: share(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [share(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(share(v) for v in obj)
    return obj


def share_batch(batch):
    batch.features = share(batch.features)
    if batch.labels is not None:
        batch.labels = batch.labels.share_memory_()
    if batch.label_mask is not None:
        batch.label_mask = batch.label_mask.share_memory_()
    batch.scenario_id = batch.scenario_id.share_memory_()
    batch._packed_buffers = tuple(t.share_memory_() for t in batch._packed_buffers)
    return batch


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
    batch = axis_batch_to_feature_batch(
        cfg, table, vocab, require_labels=True, include_group_id=True, split=split
    )
    batch = pin_feature_batch(batch, coalesce_tensors=True)
    share_batch(batch)
    buf = io.BytesIO()
    t0 = time.perf_counter()
    ForkingPickler(buf).dump(batch)
    t1 = time.perf_counter()
    blob = buf.getvalue()
    print(
        "coalesce_forking_bytes",
        len(blob),
        "dump_ms",
        round((t1 - t0) * 1000, 1),
    )
    t0 = time.perf_counter()
    batch2 = ForkingPickler.loads(blob)
    t1 = time.perf_counter()
    print(
        "load_ms",
        round((t1 - t0) * 1000, 1),
        "scenario",
        tuple(batch2.scenario_id.shape),
    )

    it = _iter_batch_tables(
        cfg, "train", shard_rank=0, shard_world_size=1, require_labels=True
    )
    for _ in range(2):
        next(it)
    table = next(it)
    batch = axis_batch_to_feature_batch(
        cfg, table, vocab, require_labels=True, include_group_id=True, split=split
    )
    batch = pin_feature_batch(batch, coalesce_tensors=False)
    share_batch(batch)
    buf = io.BytesIO()
    t0 = time.perf_counter()
    ForkingPickler(buf).dump(batch)
    t1 = time.perf_counter()
    print(
        "nocoalesce_forking_bytes",
        len(buf.getvalue()),
        "dump_ms",
        round((t1 - t0) * 1000, 1),
    )


if __name__ == "__main__":
    main()
