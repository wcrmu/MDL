#!/usr/bin/env python
"""Check pinned status survives share_memory + ForkingPickler."""
from __future__ import annotations

import io
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from multiprocessing.reduction import ForkingPickler

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
    batch = pin_feature_batch(
        axis_batch_to_feature_batch(
            cfg,
            next(it),
            vocab,
            require_labels=True,
            include_group_id=True,
            split=split,
        ),
        coalesce_tensors=True,
    )
    print("before_share pinned", all(t.is_pinned() for t in batch._packed_buffers))
    batch._packed_buffers = tuple(t.share_memory_() for t in batch._packed_buffers)
    batch.scenario_id = batch.scenario_id.share_memory_()
    print("after_share pinned", all(t.is_pinned() for t in batch._packed_buffers))
    print("scenario pinned", batch.scenario_id.is_pinned())
    buf = io.BytesIO()
    ForkingPickler(buf).dump(batch)
    batch2 = ForkingPickler.loads(buf.getvalue())
    print(
        "after_ipc packed_pinned",
        all(t.is_pinned() for t in batch2._packed_buffers),
        "scenario_pinned",
        batch2.scenario_id.is_pinned(),
    )
    # H2D smoke
    if torch.cuda.is_available():
        t0 = batch2.scenario_id
        d = t0.to(device="cuda", non_blocking=True)
        torch.cuda.synchronize()
        print("h2d_ok", tuple(d.shape), "non_blocking_src_pinned", t0.is_pinned())


if __name__ == "__main__":
    main()
