#!/usr/bin/env python
"""Smoke-test process host prepare iterator."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from src.config import load_app_config
from src.features import load_vocab_maps
from src.train import iter_feature_batches


def main() -> None:
    cfg = load_app_config("artifacts/gpu_util_e2e_mock/rankmixer_e2e.yaml")
    print("host_prepare_prefetch", cfg.data.train.reader.host_prepare_prefetch)
    vocab = load_vocab_maps(cfg)
    it = iter_feature_batches(
        cfg,
        "train",
        vocab,
        require_labels=True,
        pin_memory=True,
        include_group_id=True,
    )
    batch = next(it)
    print(
        "batch scenario",
        tuple(batch.scenario_id.shape),
        "pinned",
        batch.scenario_id.is_pinned(),
        "n_packed",
        len(batch._packed_buffers),
    )
    batch2 = next(it)
    print("batch2", tuple(batch2.scenario_id.shape))
    close = getattr(it, "close", None)
    if callable(close):
        close()
    print("OK")


if __name__ == "__main__":
    main()
