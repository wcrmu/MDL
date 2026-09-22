#!/usr/bin/env python
"""Measure FeatureBatch IPC after share_memory_."""
from __future__ import annotations

import os
import pickle
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

from src.config import load_app_config
from src.dataloader import FeatureBatch, axis_batch_to_feature_batch, pin_feature_batch
from src.features import load_vocab_maps
from src.train import _iter_batch_tables


def _share_feature_batch(batch: FeatureBatch) -> FeatureBatch:
    def share(obj):
        if isinstance(obj, torch.Tensor):
            return obj.share_memory_()
        if isinstance(obj, dict):
            return {k: share(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            typed = type(obj)
            return typed(share(v) for v in obj)
        return obj

    batch.features = share(batch.features)
    if batch.labels is not None:
        batch.labels = batch.labels.share_memory_()
    if batch.label_mask is not None:
        batch.label_mask = batch.label_mask.share_memory_()
    batch.scenario_id = batch.scenario_id.share_memory_()
    batch._packed_buffers = tuple(t.share_memory_() for t in batch._packed_buffers)
    return batch


def _tensor_nbytes(batch: FeatureBatch) -> int:
    total = 0
    seen = set()

    def walk(obj):
        nonlocal total
        if isinstance(obj, torch.Tensor):
            key = obj.data_ptr()
            if key in seen:
                return
            seen.add(key)
            total += obj.numel() * obj.element_size()
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)

    walk(batch.features)
    walk(batch.labels)
    walk(batch.label_mask)
    walk(batch.scenario_id)
    walk(batch._packed_buffers)
    return total


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
    print("tensor_nbytes_mb", round(_tensor_nbytes(batch) / 1e6, 1))
    print("n_packed", len(batch._packed_buffers))
    t0 = time.perf_counter()
    _share_feature_batch(batch)
    t1 = time.perf_counter()
    print("share_ms", round((t1 - t0) * 1000, 1))
    t0 = time.perf_counter()
    blob = pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)
    t1 = time.perf_counter()
    batch2 = pickle.loads(blob)
    t2 = time.perf_counter()
    print(
        "shared_pickle_bytes",
        len(blob),
        "dump_ms",
        round((t1 - t0) * 1000, 1),
        "load_ms",
        round((t2 - t1) * 1000, 1),
    )
    # sanity: still usable
    assert batch2.scenario_id.shape == batch.scenario_id.shape


if __name__ == "__main__":
    main()
