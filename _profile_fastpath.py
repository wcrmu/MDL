"""Check padded sequence fast-path hit rate and window sizes."""

from __future__ import annotations

import statistics as st
import time
from dataclasses import replace

from src.config import load_app_config
import src.dataloader as dl
import src.train as T


def main() -> None:
    cfg = load_app_config("artifacts/bench_b512_direct.yaml")
    reader = replace(cfg.data.train.reader, adapter_workers=2, pin_memory=False)
    c = replace(
        cfg, data=replace(cfg.data, train=replace(cfg.data.train, reader=reader))
    )

    hits = {"fast": 0, "fallback": 0}
    sizes: list[tuple[int, int]] = []
    orig = dl._gather_abs_windows_prehashed_padded

    def wrap(*a, **k):
        hits["fast"] += 1
        abs_lo = a[2]
        sizes.append((int(abs_lo.shape[0]), int(k["max_length"])))
        return orig(*a, **k)

    dl._gather_abs_windows_prehashed_padded = wrap
    orig_pad = dl._gather_padded_sequence

    def wrap_pad(*a, **k):
        hits["fallback"] += 1
        return orig_pad(*a, **k)

    dl._gather_padded_sequence = wrap_pad

    it = T.iter_feature_batches(
        c, "train", {}, require_labels=True, pin_memory=False
    )
    for _ in range(3):
        next(it)
    hits["fast"] = 0
    hits["fallback"] = 0
    sizes.clear()
    t0 = time.perf_counter()
    n = 0
    for _ in range(5):
        n += int(next(it).labels.shape[0])
    print("rate", round(n / (time.perf_counter() - t0), 1))
    print("hits", hits)
    if sizes:
        rows = [s[0] for s in sizes]
        ml = [s[1] for s in sizes]
        print("n_rows mean/max", round(st.mean(rows), 1), max(rows))
        print("max_len mean/max", round(st.mean(ml), 1), max(ml))
        print("sample", sizes[:6], "...", sizes[-2:])


if __name__ == "__main__":
    main()
