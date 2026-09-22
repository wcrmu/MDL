"""Inspect sequence unique-column cardinality and token totals."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np

from src.config import load_app_config
import src.dataloader as ad
import src.train as T


def main() -> None:
    cfg = load_app_config("artifacts/bench_b512_direct.yaml")
    reader = replace(cfg.data.train.reader, adapter_workers=3, pin_memory=False)
    c = replace(
        cfg, data=replace(cfg.data, train=replace(cfg.data.train, reader=reader))
    )

    stats = Counter()
    totals = []
    nrows = []
    nuniques = []

    orig = ad.prepare_packed_axis_batch

    def wrap(*a, **k):
        out = orig(*a, **k)
        # inspect first sequence batch layout
        for name, plan in out.sequence_plans.items():
            n = int(plan.compacted_lengths.shape[0])
            total = int(plan.compacted_lengths.sum())
            nrows.append(n)
            totals.append(total)
            # get a sequence field batch
            seq = next(s for s in c.sequences if s.name == name)
            batch = out.request_values[seq.fields[0].source]
            if type(batch).__name__ == "SequenceColumnBatch":
                if batch.column_index is None:
                    u = n
                else:
                    u = int(np.unique(batch.column_index).size)
                nuniques.append(u)
                stats[u] += 1
        return out

    ad.prepare_packed_axis_batch = wrap
    T.prepare_packed_axis_batch = wrap
    it = T.iter_feature_batches(
        c, "train", {}, require_labels=True, pin_memory=False
    )
    for _ in range(2):
        next(it)
    stats.clear()
    nrows.clear()
    totals.clear()
    nuniques.clear()
    for _ in range(5):
        next(it)
    print("unique_col counts (per seq plan)", dict(stats))
    print(
        "n_rows mean/max",
        round(float(np.mean(nrows)), 1),
        max(nrows),
        "tokens mean/max",
        round(float(np.mean(totals)), 1),
        max(totals),
    )


if __name__ == "__main__":
    main()
