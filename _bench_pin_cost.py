"""Measure pin cost via train.pin_feature_batch patch."""

from __future__ import annotations

import time
from dataclasses import replace

from src.config import load_app_config
import src.dataloader as dl
import src.train as T


def main() -> None:
    cfg = load_app_config("artifacts/bench_b512_direct.yaml")
    reader = replace(cfg.data.train.reader, adapter_workers=3, pin_memory=True)
    c = replace(
        cfg, data=replace(cfg.data, train=replace(cfg.data.train, reader=reader))
    )
    pin_times: list[float] = []
    orig = dl.pin_feature_batch

    def wrap(batch, coalesce_tensors=False):
        t0 = time.perf_counter()
        out = orig(batch, coalesce_tensors=coalesce_tensors)
        pin_times.append(time.perf_counter() - t0)
        return out

    T.pin_feature_batch = wrap
    dl.pin_feature_batch = wrap
    it = T.iter_feature_batches(
        c, "train", {}, require_labels=True, pin_memory=True
    )
    for _ in range(3):
        next(it)
    pin_times.clear()
    n = 0
    t0 = time.perf_counter()
    for _ in range(10):
        n += int(next(it).labels.shape[0])
    print("rate", round(n / (time.perf_counter() - t0), 1))
    if pin_times:
        print(
            f"pin n={len(pin_times)} per_batch≈{sum(pin_times)/len(pin_times)*1000:.1f}ms"
        )
    else:
        print("pin not observed")


if __name__ == "__main__":
    main()
