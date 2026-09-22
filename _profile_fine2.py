"""Fine profile of tensorize sub-stages."""

from __future__ import annotations

import time
from dataclasses import replace

from src.config import load_app_config
import src.dataloader as ad
import src.dataloader as dl
import src.train as T


def main() -> None:
    cfg = load_app_config("artifacts/bench_b512_direct.yaml")
    reader = replace(cfg.data.train.reader, adapter_workers=3, pin_memory=False)
    c = replace(cfg, data=replace(cfg.data, train=replace(cfg.data.train, reader=reader)))

    buckets = {
        "prepare": [],
        "tensorize": [],
        "seq": [],
        "bag": [],
        "cat": [],
        "pad": [],
    }
    orig_prep = ad.prepare_packed_axis_batch
    orig_tens = dl.axis_batch_to_feature_batch
    orig_seq = dl._tensorize_axis_sequence
    orig_bag = dl._tensorize_python_categorical_bag
    orig_cat = dl._tensorize_python_categorical_values
    orig_pad = dl._gather_padded_sequence
    orig_bag_gather = dl._gather_bag_from_sequence_column_batch

    def wrap(name, fn):
        def inner(*a, **k):
            t0 = time.perf_counter()
            out = fn(*a, **k)
            buckets[name].append(time.perf_counter() - t0)
            return out

        return inner

    def timed_prep(*a, **k):
        t0 = time.perf_counter()
        out = orig_prep(*a, **k)
        buckets["prepare"].append(time.perf_counter() - t0)
        return out

    def timed_tens(*a, **k):
        t0 = time.perf_counter()
        out = orig_tens(*a, **k)
        buckets["tensorize"].append(time.perf_counter() - t0)
        return out

    T.prepare_packed_axis_batch = timed_prep
    T.axis_batch_to_feature_batch = timed_tens
    dl._tensorize_axis_sequence = wrap("seq", orig_seq)
    dl._tensorize_python_categorical_bag = wrap("bag", orig_bag)
    dl._tensorize_python_categorical_values = wrap("cat", orig_cat)
    dl._gather_padded_sequence = wrap("pad", orig_pad)

    it = T.iter_feature_batches(c, "train", {}, require_labels=True, pin_memory=False)
    for _ in range(3):
        next(it)
    for k in buckets:
        buckets[k].clear()
    n = 0
    t0 = time.perf_counter()
    for _ in range(10):
        n += int(next(it).labels.shape[0])
    wall = time.perf_counter() - t0
    print(f"rate={n/wall:.1f}")
    for k, v in buckets.items():
        if not v:
            continue
        per_batch = sum(v) / max(len(buckets["tensorize"]), 1)
        print(
            f"{k}: n={len(v)} sum={sum(v)*1000:.0f}ms "
            f"per_batch≈{per_batch*1000:.1f}ms mean_call={sum(v)/len(v)*1000:.2f}ms"
        )


if __name__ == "__main__":
    main()
