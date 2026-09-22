"""Profile prepare_packed_axis_batch sub-stages."""

from __future__ import annotations

import time
from dataclasses import replace

from src.config import load_app_config
import src.dataloader as ad
import src.train as T


def main() -> None:
    cfg = load_app_config("artifacts/bench_b512_direct.yaml")
    reader = replace(cfg.data.train.reader, adapter_workers=2, pin_memory=False)
    c = replace(
        cfg, data=replace(cfg.data, train=replace(cfg.data.train, reader=reader))
    )

    buckets: dict[str, list[float]] = {
        "prepare_total": [],
        "seq_plans": [],
        "request_gather": [],
        "candidate_gather": [],
    }

    orig_plan = ad.build_axis_sequence_selection_plan
    orig_prep = ad.prepare_packed_axis_batch

    def timed_plan(*a, **k):
        t0 = time.perf_counter()
        out = orig_plan(*a, **k)
        buckets["seq_plans"].append(time.perf_counter() - t0)
        return out

    ad.build_axis_sequence_selection_plan = timed_plan

    # Monkeypatch prepare internals via wrapping whole prepare and estimating
    # by temporarily timing inside a copy — just time plan vs rest.
    def timed_prep(*a, **k):
        t0 = time.perf_counter()
        out = orig_prep(*a, **k)
        buckets["prepare_total"].append(time.perf_counter() - t0)
        return out

    T.prepare_packed_axis_batch = timed_prep
    ad.prepare_packed_axis_batch = timed_prep

    it = T.iter_feature_batches(
        c, "train", {}, require_labels=True, pin_memory=False
    )
    for _ in range(3):
        next(it)
    for v in buckets.values():
        v.clear()
    n = 0
    t0 = time.perf_counter()
    for _ in range(10):
        n += int(next(it).labels.shape[0])
    wall = time.perf_counter() - t0
    print(f"rate={n/wall:.1f}")
    n_batch = max(len(buckets["prepare_total"]), 1)
    for k, v in buckets.items():
        if not v:
            continue
        print(
            f"{k}: n={len(v)} sum={sum(v)*1000:.0f}ms "
            f"per_batch≈{sum(v)/n_batch*1000:.1f}ms"
        )
    if buckets["prepare_total"] and buckets["seq_plans"]:
        rest = sum(buckets["prepare_total"]) - sum(buckets["seq_plans"])
        print(f"prepare_minus_plans per_batch≈{rest/n_batch*1000:.1f}ms")


if __name__ == "__main__":
    main()
