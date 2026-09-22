from dataclasses import replace
import time

from src.config import load_app_config
import src.train as T


def run(pin: bool) -> float:
    cfg = load_app_config("artifacts/bench_b512_direct.yaml")
    reader = replace(cfg.data.train.reader, adapter_workers=2, pin_memory=pin)
    c = replace(
        cfg, data=replace(cfg.data, train=replace(cfg.data.train, reader=reader))
    )
    it = T.iter_feature_batches(
        c, "train", {}, require_labels=True, pin_memory=pin
    )
    for _ in range(4):
        next(it)
    n = 0
    t0 = time.perf_counter()
    for _ in range(12):
        n += int(next(it).labels.shape[0])
    return n / (time.perf_counter() - t0)


if __name__ == "__main__":
    print("pin=False", round(run(False), 1))
    print("pin=True", round(run(True), 1))
