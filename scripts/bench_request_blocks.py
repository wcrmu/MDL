#!/usr/bin/env python3
"""Microbenchmark axis-bundle request descriptor construction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_app_config
from src.dataloader import (
    RequestGroupBlock,
    _sequence_tensor_max_length,
    effective_bucket_length_from_pre_compaction,
    iter_adapted_axis_bundles,
    request_group_blocks_from_axis_bundle,
)


def _legacy(bundle, *, source_id, sequences, length_bucket_metric):
    positions_by_slot = [[] for _ in range(bundle.n_requests)]
    for candidate_index, slot in enumerate(bundle.candidate_to_request):
        positions_by_slot[int(slot)].append(int(candidate_index))
    blocks = []
    for stable_group_order, positions in enumerate(positions_by_slot):
        pre_compaction = {}
        for sequence in sequences:
            if not sequence.fields:
                continue
            column = bundle.sequence_features[sequence.fields[0].source]
            row_length = getattr(column, "row_length", None)
            length = (
                int(row_length(stable_group_order))
                if callable(row_length)
                else len(column[stable_group_order])
            )
            maximum = _sequence_tensor_max_length(sequence)
            if maximum is not None:
                length = min(length, int(maximum))
            pre_compaction[sequence.name] = int(length)
        blocks.append(
            RequestGroupBlock(
                source_id=source_id,
                raw_row_index=int(bundle.request_raw_rows[stable_group_order]),
                request_id=bundle.request_ids[stable_group_order],
                representative_request_position=stable_group_order,
                candidate_positions=np.asarray(positions, dtype=np.int64),
                candidate_offset=0,
                candidate_count=len(positions),
                pre_compaction_sequence_lengths=pre_compaction,
                effective_bucket_length=effective_bucket_length_from_pre_compaction(
                    pre_compaction, metric=length_bucket_metric
                ),
                stable_group_order=stable_group_order,
            )
        )
    return tuple(blocks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    config = load_app_config(args.config)
    split = config.data.train
    assert split is not None
    iterator = iter_adapted_axis_bundles(
        config,
        "train",
        shard_rank=0,
        shard_world_size=1,
        require_labels=True,
        producer_queue_size=2,
        arrow_axis=False,
    )
    try:
        bundle = next(iterator)
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    kwargs = {
        "source_id": 0,
        "sequences": config.sequences,
        "length_bucket_metric": split.reader.length_bucket_metric,
    }
    expected = _legacy(bundle, **kwargs)
    actual = request_group_blocks_from_axis_bundle(bundle, **kwargs)
    assert len(expected) == len(actual)
    for left, right in zip(expected, actual):
        assert left.request_id == right.request_id
        assert left.pre_compaction_sequence_lengths == right.pre_compaction_sequence_lengths
        assert left.effective_bucket_length == right.effective_bucket_length
        np.testing.assert_array_equal(left.candidate_positions, right.candidate_positions)
    for name, function in (
        ("legacy", _legacy),
        ("vectorized", request_group_blocks_from_axis_bundle),
    ):
        started = perf_counter()
        for _ in range(args.repeats):
            function(bundle, **kwargs)
        elapsed = perf_counter() - started
        print(
            f"{name}_ms={elapsed / args.repeats * 1000:.3f} "
            f"requests={bundle.n_requests} candidates={bundle.n_candidates}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
