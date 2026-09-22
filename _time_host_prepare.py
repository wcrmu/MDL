#!/usr/bin/env python
"""Time parent-side next() for process host-prepare."""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from src.config import load_app_config
from src.features import load_vocab_maps
from src.train import iter_feature_batches


def main() -> None:
    cfg = load_app_config("artifacts/gpu_util_e2e_mock/rankmixer_e2e.yaml")
    vocab = load_vocab_maps(cfg)
    it = iter_feature_batches(
        cfg,
        "train",
        vocab,
        require_labels=True,
        pin_memory=True,
        include_group_id=True,
    )
    for _ in range(2):
        next(it)
    waits: list[float] = []
    batch = None
    for _ in range(8):
        started = time.perf_counter()
        batch = next(it)
        waits.append((time.perf_counter() - started) * 1000.0)
    assert batch is not None
    print(
        "next_ms",
        [round(value, 1) for value in waits],
        "mean",
        round(sum(waits) / len(waits), 1),
        "pinned",
        batch.scenario_id.is_pinned(),
        "n_packed",
        len(batch._packed_buffers),
    )
    close = getattr(it, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
