#!/usr/bin/env python
"""Profile pack stages matching train._iter_batch_tables_direct."""
from __future__ import annotations

import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from src.dataloader import (
    PreparedAxisBatch,
    SourceRegistry,
    build_packed_request_plan,
    iter_length_bucketed_packs,
    prepare_packed_axis_batch,
    request_group_blocks_from_axis_bundle,
    reset_direct_pipeline_stats,
)
from src.config import load_app_config
from src.dataloader import iter_adapted_axis_bundles


def main() -> None:
    cfg = load_app_config("artifacts/gpu_util_e2e_mock/rankmixer_e2e.yaml")
    split = cfg.data.train
    assert split is not None
    reader = split.reader
    registry = SourceRegistry()
    reset_direct_pipeline_stats()
    adapter_options = {} if split.adapter is None else split.adapter.options
    context_sources = {
        str(source) for source in adapter_options.get("context_features", ())
    }
    sequence_sources = {
        field.source for sequence in cfg.sequences for field in sequence.fields
    }

    bundle_iter = iter_adapted_axis_bundles(
        cfg,
        "train",
        shard_rank=0,
        shard_world_size=1,
        require_labels=True,
        producer_queue_size=2 if reader.prefetch_batches > 0 else 1,
        arrow_axis=False,
    )

    def blocks():
        for bundle in bundle_iter:
            source_id = registry.put(bundle)
            group_blocks = request_group_blocks_from_axis_bundle(
                bundle,
                source_id=source_id,
                sequences=cfg.sequences,
                length_bucket_metric=reader.length_bucket_metric,
            )
            if not group_blocks:
                registry.release(source_id, 0)
                continue
            registry.acquire(source_id, len(group_blocks))
            yield from group_blocks

    pack_iter = iter_length_bucketed_packs(
        blocks(),
        buckets=reader.length_buckets,
        default_batch_size=cfg.training.batch_size,
        shuffle_buffer_rows=reader.shuffle_buffer_rows,
        shuffle_seed=reader.shuffle_seed,
        shard_rank=0,
    )

    def one() -> tuple[float, float, float]:
        t0 = time.perf_counter()
        pack = next(pack_iter)
        t1 = time.perf_counter()
        packed = build_packed_request_plan(pack)
        t2 = time.perf_counter()
        retained = {
            source_id: registry.get(source_id)
            for source_id in {b.source_id for b in packed.blocks}
        }
        candidate_request_columns = sorted(
            {
                *([split.request_id] if split.request_id is not None else []),
                *([split.group_id] if split.group_id is not None else []),
                *(
                    [cfg.scenarios.source]
                    if cfg.scenarios.source is not None
                    else []
                ),
                *split.prediction_keys.values(),
            }
        )
        prepared = prepare_packed_axis_batch(
            retained,
            packed,
            sequences=cfg.sequences,
            request_id_column=split.request_id,
            candidate_request_columns=candidate_request_columns,
        )
        t3 = time.perf_counter()
        assert isinstance(prepared, PreparedAxisBatch)
        for source_id, count in {
            b.source_id: 1
            for b in packed.blocks
            if b.releases_source_reference
        }.items():
            registry.release(source_id, count)
        return (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000

    for _ in range(2):
        one()
    a, b, c = [], [], []
    for _ in range(6):
        x, y, z = one()
        a.append(x)
        b.append(y)
        c.append(z)
    print("bucket_next_ms", round(sum(a) / len(a), 1), [round(x, 1) for x in a])
    print("build_plan_ms", round(sum(b) / len(b), 1), [round(x, 1) for x in b])
    print("prepare_axis_ms", round(sum(c) / len(c), 1), [round(x, 1) for x in c])


if __name__ == "__main__":
    main()
