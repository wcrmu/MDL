"""Inspect bag row-length distribution on one packed batch."""

from __future__ import annotations

from dataclasses import replace

from src.config import load_app_config
import src.dataloader as ad
import src.dataloader as dl
import src.train as T
import numpy as np


def main() -> None:
    cfg = load_app_config("artifacts/bench_b512_direct.yaml")
    reader = replace(cfg.data.train.reader, adapter_workers=2, pin_memory=False)
    c = replace(
        cfg, data=replace(cfg.data, train=replace(cfg.data.train, reader=reader))
    )

    lengths_all = []
    n_unique_cols = []

    orig = dl._gather_bag_from_sequence_column_batch

    def wrap(values, **k):
        cols = values.columns
        n_unique_cols.append(len(cols))
        flat, lengths = orig(values, **k)
        lengths_all.append(lengths)
        return flat, lengths

    dl._gather_bag_from_sequence_column_batch = wrap

    it = T.iter_feature_batches(
        c, "train", {}, require_labels=True, pin_memory=False
    )
    for _ in range(3):
        next(it)
    lengths_all.clear()
    n_unique_cols.clear()
    next(it)
    cat = np.concatenate(lengths_all) if lengths_all else np.array([])
    print("n_bag_calls", len(lengths_all))
    print("unique_cols mean/max", float(np.mean(n_unique_cols)), max(n_unique_cols))
    print(
        "len mean/p50/p90/max",
        float(cat.mean()),
        int(np.median(cat)),
        int(np.percentile(cat, 90)),
        int(cat.max()),
    )
    print("frac_len<=1", float((cat <= 1).mean()))
    print("frac_len==0", float((cat == 0).mean()))


if __name__ == "__main__":
    main()
