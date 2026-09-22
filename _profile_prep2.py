"""Count prepare gather work and time dense/object paths."""

from __future__ import annotations

import time
from dataclasses import replace

from src.config import load_app_config
import src.dataloader as ad
import src.train as T


def main() -> None:
    cfg = load_app_config("artifacts/bench_b512_direct.yaml")
    reader = replace(cfg.data.train.reader, adapter_workers=0, pin_memory=False)
    c = replace(
        cfg, data=replace(cfg.data, train=replace(cfg.data.train, reader=reader))
    )

    orig = ad.prepare_packed_axis_batch
    stats = {"n": 0}

    def wrapped(bundles, packed, **kwargs):
        t0 = time.perf_counter()
        out = orig(bundles, packed, **kwargs)
        dt = time.perf_counter() - t0
        stats["n"] += 1
        if stats["n"] <= 3:
            return out
        # Inspect one batch composition
        if stats.get("printed"):
            stats.setdefault("prep", []).append(dt)
            return out
        req = out.request_values
        cand = out.candidate_values
        from collections import Counter

        def kind(v):
            name = type(v).__name__
            if name == "SequenceColumnBatch":
                return "list_batch"
            if isinstance(v, np.ndarray):
                return f"ndarray:{v.dtype}"
            return type(v).__name__

        import numpy as np

        print("n_requests", out.n_requests, "n_candidates", out.n_candidates)
        print("request kinds", Counter(kind(v) for v in req.values()))
        print("cand kinds", Counter(kind(v) for v in cand.values()))
        print("n_req_names", len(req), "n_cand_names", len(cand))
        stats["printed"] = True
        stats.setdefault("prep", []).append(dt)
        return out

    import numpy as np

    ad.prepare_packed_axis_batch = wrapped
    T.prepare_packed_axis_batch = wrapped

    it = T.iter_feature_batches(
        c, "train", {}, require_labels=True, pin_memory=False
    )
    for _ in range(2):
        next(it)
    stats["prep"] = []
    n = 0
    t0 = time.perf_counter()
    for _ in range(8):
        n += int(next(it).labels.shape[0])
    print(f"aw=0 rate={n/(time.perf_counter()-t0):.1f}")
    if stats.get("prep"):
        print(
            f"prepare per_batch≈{sum(stats['prep'])/len(stats['prep'])*1000:.1f}ms"
        )


if __name__ == "__main__":
    main()
