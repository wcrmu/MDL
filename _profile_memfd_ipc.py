#!/usr/bin/env python
"""Profile queue.get vs memfd load for host-prepare."""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from src.config import load_app_config
from src.features import load_vocab_maps
from src.train import (
    _ProcessHostPrepareIterator,
    _load_feature_batch_from_ipc,
    iter_feature_batches,
)


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
    assert isinstance(it, _ProcessHostPrepareIterator)
    for _ in range(3):
        next(it)
    time.sleep(2.0)  # let queue refill
    gets = []
    loads = []
    for _ in range(6):
        t0 = time.perf_counter()
        item = it._queue.get()
        t1 = time.perf_counter()
        memfd = it._recv_memfd_handle()
        batch = _load_feature_batch_from_ipc(
            item, fd=memfd, pin_memory=True
        )
        t2 = time.perf_counter()
        gets.append((t1 - t0) * 1000)
        loads.append((t2 - t1) * 1000)
        del batch
    print(
        "get_ms",
        [round(x, 1) for x in gets],
        "mean",
        round(sum(gets) / len(gets), 1),
    )
    print(
        "load_ms",
        [round(x, 1) for x in loads],
        "mean",
        round(sum(loads) / len(loads), 1),
        "pinned_ok",
        True,
    )
    it.close()


if __name__ == "__main__":
    main()
