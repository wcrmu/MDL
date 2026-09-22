"""Compare pin_memory coalesce vs per-tensor pin vs no pin."""

from __future__ import annotations

import time
from dataclasses import replace

from src.config import load_app_config
import src.dataloader as dl
import src.train as T


def run(*, pin: bool, coalesce: bool) -> float:
    cfg = load_app_config("artifacts/bench_b512_direct.yaml")
    reader = replace(
        cfg.data.train.reader,
        adapter_workers=2,
        pin_memory=pin,
        coalesce_pinned_tensors=coalesce,
    )
    c = replace(
        cfg, data=replace(cfg.data, train=replace(cfg.data.train, reader=reader))
    )
    it = T.iter_feature_batches(
        c, "train", {}, require_labels=True, pin_memory=pin
    )
    for _ in range(3):
        next(it)
    n = 0
    t0 = time.perf_counter()
    for _ in range(10):
        n += int(next(it).labels.shape[0])
    return n / (time.perf_counter() - t0)


if __name__ == "__main__":
    print("no pin", round(run(pin=False, coalesce=False), 1))
    print("pin+coalesce", round(run(pin=True, coalesce=True), 1))
    print("pin no coalesce", round(run(pin=True, coalesce=False), 1))
