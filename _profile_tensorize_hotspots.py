#!/usr/bin/env python
from __future__ import annotations

import os
import time
import argparse
from collections import defaultdict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from src.dataloader import PreparedAxisBatch
from src.config import load_app_config
from src.dataloader import (
    _indexed_request_value,
    _tensorize_axis_sequence,
    _tensorize_dense,
    _tensorize_python_categorical_bag,
    _tensorize_python_categorical_values,
)
from src.features import load_vocab_maps
from src.train import _iter_batch_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="artifacts/gpu_util_e2e_mock/rankmixer_e2e.yaml",
    )
    args = parser.parse_args()
    cfg = load_app_config(args.config)
    vocab = load_vocab_maps(cfg)
    split = cfg.data.train
    assert split is not None
    adapter_options = {} if split.adapter is None else split.adapter.options
    context_sources = {
        str(source) for source in adapter_options.get("context_features", ())
    }
    validate = split.reader.validate_prehashed_nonzero

    it = _iter_batch_tables(
        cfg, "train", shard_rank=0, shard_world_size=1, require_labels=True
    )
    for _ in range(2):
        next(it)
    table = next(it)
    assert isinstance(table, PreparedAxisBatch)

    bag_window_cache: dict = {}
    bag_column_groups: dict = {}
    times: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    row_indices = table.request_row_indices
    for feature in cfg.features:
        request_level = feature.source in context_sources
        source_values = (
            table.request_values if request_level else table.candidate_values
        )
        values = source_values[feature.source]
        t0 = time.perf_counter()
        if feature.kind == "categorical":
            if feature.is_bag:
                column_groups = None
                if type(values).__name__ == "SequenceColumnBatch":
                    column_index = values.column_index
                    if column_index is not None:
                        key = id(column_index)
                        column_groups = bag_column_groups.get(key)
                        if column_groups is None:
                            n_unique = len(values.columns)
                            column_groups = [
                                __import__("numpy").flatnonzero(
                                    column_index == unique_idx
                                )
                                for unique_idx in range(n_unique)
                            ]
                            bag_column_groups[key] = column_groups
                value = _tensorize_python_categorical_bag(
                    cfg,
                    feature,
                    values,
                    vocab,
                    validate_prehashed_nonzero=validate,
                    column_groups=column_groups,
                    window_cache=bag_window_cache,
                )
                kind = "bag"
            else:
                value = _tensorize_python_categorical_values(
                    cfg,
                    cfg.resolved.categorical_input_by_name[feature.name],
                    values,
                    vocab,
                    validate_prehashed_nonzero=validate,
                )
                kind = "cat"
        else:
            value = _tensorize_dense(feature, list(values))
            kind = "dense"
        if request_level:
            value = _indexed_request_value(value, row_indices)
        dt = time.perf_counter() - t0
        times[kind] += dt
        counts[kind] += 1
        times[f"feat:{feature.name}"] = dt

    seq_total = 0.0
    for sequence in cfg.sequences:
        plan = table.sequence_plans[sequence.name]
        t0 = time.perf_counter()
        _tensorize_axis_sequence(
            cfg,
            sequence,
            table.request_values,
            plan,
            vocab,
            validate_prehashed_nonzero=validate,
        )
        dt = time.perf_counter() - t0
        seq_total += dt
        times[f"seq:{sequence.name}"] = dt
    times["sequence"] = seq_total
    counts["sequence"] = len(cfg.sequences)

    print("by_kind_ms")
    for kind in ("bag", "cat", "dense", "sequence"):
        print(
            f"  {kind}: {times[kind]*1000:.1f}ms over {counts[kind]} items"
        )
    top = sorted(
        (
            (k, v)
            for k, v in times.items()
            if k.startswith("feat:") or k.startswith("seq:")
        ),
        key=lambda kv: kv[1],
        reverse=True,
    )[:15]
    print("top_items_ms")
    for k, v in top:
        print(f"  {k}: {v*1000:.1f}")


if __name__ == "__main__":
    main()
