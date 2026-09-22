#!/usr/bin/env python
"""Measure process-prepare next() latency with a warm queue."""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

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
    for _ in range(3):
        next(it)
    time.sleep(1.0)  # let queue refill
    waits: list[float] = []
    for _ in range(6):
        t0 = time.perf_counter()
        batch = next(it)
        waits.append((time.perf_counter() - t0) * 1000)
    print(
        "next_ms",
        [round(x, 1) for x in waits],
        "mean",
        round(sum(waits) / len(waits), 1),
        "pinned",
        batch.scenario_id.is_pinned(),
    )
    close = getattr(it, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
