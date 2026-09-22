#!/usr/bin/env python
from __future__ import annotations

import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from src.dataloader import PreparedAxisBatch
from src.config import load_app_config
from src.dataloader import _tensorize_axis_sequence
from src.features import load_vocab_maps
from src.train import _iter_batch_tables


def main() -> None:
    cfg = load_app_config("artifacts/gpu_util_e2e_mock/rankmixer_e2e.yaml")
    vocab = load_vocab_maps(cfg)
    it = _iter_batch_tables(
        cfg, "train", shard_rank=0, shard_world_size=1, require_labels=True
    )
    for _ in range(2):
        next(it)
    table = next(it)
    assert isinstance(table, PreparedAxisBatch)
    for name in ["view_long", "impr", "clk_long"]:
        seq = next(s for s in cfg.sequences if s.name == name)
        plan = table.sequence_plans[name]
        print(
            name,
            "fields",
            len(seq.fields),
            "max_len",
            seq.max_length,
            "selections_are_ranges",
            getattr(plan, "selections_are_ranges", None),
            "compacted_sum",
            int(plan.compacted_lengths.sum()),
            "n_rows",
            len(plan.compacted_lengths),
        )
        for _ in range(2):
            _tensorize_axis_sequence(
                cfg,
                seq,
                table.request_values,
                plan,
                vocab,
                validate_prehashed_nonzero=True,
            )
        t0 = time.perf_counter()
        for _ in range(5):
            _tensorize_axis_sequence(
                cfg,
                seq,
                table.request_values,
                plan,
                vocab,
                validate_prehashed_nonzero=True,
            )
        print("  mean_ms", round((time.perf_counter() - t0) / 5 * 1000, 1))


if __name__ == "__main__":
    main()
