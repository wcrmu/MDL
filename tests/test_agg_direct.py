"""Unit tests for agg_direct descriptor builders and batcher."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import pyarrow as pa
import torch

from src.agg_direct import (
    AdaptedAxisBundle,
    RequestGroupBlock,
    build_packed_request_plan,
    build_sequence_selection_plan,
    effective_bucket_length_from_pre_compaction,
    iter_packed_request_groups,
    iter_shuffled_request_groups,
    length_bucket_index,
    materialize_packed_axis_bundles,
    prepare_packed_axis_batch,
    request_group_blocks_from_adapted_table,
    request_group_blocks_from_axis_bundle,
    row_sequence_selection_after_truncate_then_compact,
    table_pre_compaction_sequence_lengths,
)
from src.train import _table_effective_sequence_lengths, _table_sequence_lengths


def _sequence(name: str, source: str, max_length: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        max_length=max_length,
        fields=[SimpleNamespace(source=source)],
    )


class RequestGroupBlockTest(unittest.TestCase):
    def test_slice_candidates_is_logical_view(self) -> None:
        positions = np.asarray([0, 2, 5, 7], dtype=np.int64)
        block = RequestGroupBlock(
            source_id=0,
            raw_row_index=0,
            request_id="A",
            representative_request_position=10,
            candidate_positions=positions,
            candidate_offset=0,
            candidate_count=4,
            pre_compaction_sequence_lengths={"clk_long": 3},
            effective_bucket_length=3,
            stable_group_order=0,
        )

        sliced = block.slice_candidates(1, 2)

        self.assertEqual(sliced.candidate_offset, 1)
        self.assertEqual(sliced.candidate_count, 2)
        self.assertEqual(sliced.slice_ordinal, 1)
        np.testing.assert_array_equal(
            sliced.active_candidate_positions(),
            np.asarray([2, 5], dtype=np.int64),
        )
        self.assertIs(sliced.candidate_positions, positions)

    def test_slice_candidates_rejects_overflow(self) -> None:
        block = RequestGroupBlock(
            source_id=0,
            raw_row_index=0,
            request_id="A",
            representative_request_position=0,
            candidate_positions=np.asarray([0, 1], dtype=np.int64),
            candidate_offset=0,
            candidate_count=2,
            pre_compaction_sequence_lengths={},
            effective_bucket_length=0,
            stable_group_order=0,
        )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            block.slice_candidates(1, 2)


class RequestGroupBuilderTest(unittest.TestCase):
    def test_groups_by_request_id_not_row_contiguity(self) -> None:
        # Same search_id with interleaved candidates → one group; positions non-contiguous.
        table = pa.table(
            {
                "search_id": ["A", "B", "A", "A"],
                "item": [10, 20, 11, 12],
                "clk_long": [[1, 2, 3], [9], [1, 2, 3], [1, 2, 3]],
            }
        )
        sequences = [_sequence("clk_long", "clk_long", max_length=10)]
        blocks = request_group_blocks_from_adapted_table(
            table,
            source_id=7,
            request_id_column="search_id",
            sequences=sequences,
            length_bucket_metric="max",
        )

        self.assertEqual([block.request_id for block in blocks], ["A", "B"])
        self.assertEqual(blocks[0].candidate_count, 3)
        self.assertEqual(blocks[0].representative_request_position, 0)
        self.assertEqual(blocks[0].raw_row_index, 0)
        self.assertEqual(blocks[0].source_id, 7)
        np.testing.assert_array_equal(
            blocks[0].candidate_positions,
            np.asarray([0, 2, 3], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            blocks[1].candidate_positions,
            np.asarray([1], dtype=np.int64),
        )
        self.assertEqual(blocks[0].pre_compaction_sequence_lengths["clk_long"], 3)
        self.assertEqual(blocks[0].effective_bucket_length, 3)
        self.assertEqual(blocks[1].effective_bucket_length, 1)

    def test_bucket_length_matches_legacy_flat_metric(self) -> None:
        table = pa.table(
            {
                "request_id": ["r0", "r0", "r1"],
                "seq_a": [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [7, 8]],
                "seq_b": [[1], [1], [1, 2, 3, 4]],
            }
        )
        sequences = [
            _sequence("seq_a", "seq_a", max_length=3),
            _sequence("seq_b", "seq_b", max_length=10),
        ]
        config = SimpleNamespace(sequences=sequences)

        legacy = _table_effective_sequence_lengths(config, table, metric="max")
        blocks = request_group_blocks_from_adapted_table(
            table,
            source_id=0,
            request_id_column="request_id",
            sequences=sequences,
            length_bucket_metric="max",
        )

        # Clamp: r0 seq_a len 5 → 3; seq_b len 1 → max=3.
        # r1 seq_a len 2; seq_b len 4 → max=4.
        self.assertEqual(blocks[0].effective_bucket_length, 3)
        self.assertEqual(blocks[1].effective_bucket_length, 4)
        self.assertEqual(int(legacy[0].item()), blocks[0].effective_bucket_length)
        self.assertEqual(int(legacy[2].item()), blocks[1].effective_bucket_length)

        per_seq = table_pre_compaction_sequence_lengths(sequences, table)
        for sequence in sequences:
            torch.testing.assert_close(
                torch.from_numpy(per_seq[sequence.name]),
                _table_sequence_lengths(config, sequence, table),
            )

    def test_max_length_clamp_without_null_anchor(self) -> None:
        # Truncated length 3 is the bucket key even if a later compaction would drop nulls.
        table = pa.table(
            {
                "request_id": ["A"],
                "ups": [[1, None, 3, 4]],
            }
        )
        sequences = [_sequence("ups", "ups", max_length=3)]
        blocks = request_group_blocks_from_adapted_table(
            table,
            source_id=0,
            request_id_column="request_id",
            sequences=sequences,
        )
        self.assertEqual(blocks[0].pre_compaction_sequence_lengths["ups"], 3)
        self.assertEqual(blocks[0].effective_bucket_length, 3)

    def test_sum_metric(self) -> None:
        lengths = {"a": 3, "b": 4}
        self.assertEqual(
            effective_bucket_length_from_pre_compaction(lengths, metric="sum"),
            7,
        )


class PackedPlanAndBatcherTest(unittest.TestCase):
    def test_packed_plan_is_identity(self) -> None:
        blocks = request_group_blocks_from_adapted_table(
            pa.table(
                {
                    "request_id": ["A", "A", "B"],
                    "x": [1, 2, 3],
                }
            ),
            source_id=0,
            request_id_column="request_id",
        )
        plan = build_packed_request_plan(blocks)
        self.assertEqual(len(plan.blocks), 2)
        np.testing.assert_array_equal(plan.unique_block_indices, [0, 1])
        np.testing.assert_array_equal(plan.block_to_request, [0, 1])
        np.testing.assert_array_equal(plan.candidate_to_request, [0, 0, 1])

    def test_shuffle_buffer_zero_preserves_order_without_rng(self) -> None:
        blocks = request_group_blocks_from_adapted_table(
            pa.table({"request_id": ["A", "B", "C"], "x": [1, 2, 3]}),
            source_id=0,
            request_id_column="request_id",
        )
        out = list(
            iter_shuffled_request_groups(
                iter(blocks),
                shuffle_buffer_rows=0,
                shuffle_seed=99,
            )
        )
        self.assertEqual([block.request_id for block in out], ["A", "B", "C"])

    def test_shuffle_is_deterministic(self) -> None:
        blocks = request_group_blocks_from_adapted_table(
            pa.table(
                {
                    "request_id": ["A", "A", "B", "C", "C", "D"],
                    "x": list(range(6)),
                }
            ),
            source_id=0,
            request_id_column="request_id",
        )

        def run() -> list[str]:
            return [
                block.request_id
                for block in iter_shuffled_request_groups(
                    iter(blocks),
                    shuffle_buffer_rows=3,
                    shuffle_seed=11,
                    shard_rank=0,
                )
            ]

        self.assertEqual(run(), run())

    def test_pack_splits_oversized_then_buffers_remainder(self) -> None:
        blocks = request_group_blocks_from_adapted_table(
            pa.table(
                {
                    "request_id": ["A", "A", "A", "A", "A", "B", "B"],
                    "x": list(range(7)),
                }
            ),
            source_id=0,
            request_id_column="request_id",
        )
        packs = list(iter_packed_request_groups(iter(blocks), batch_size=2))
        sizes = [
            [block.candidate_count for block in pack] for pack in packs
        ]
        # Matches _iter_group_preserving_batches: flush A remainder alone when
        # B cannot fit, then emit B as its own full pack.
        self.assertEqual(sizes, [[2], [2], [1], [2]])
        self.assertEqual(packs[0][0].request_id, "A")
        self.assertEqual(packs[1][0].request_id, "A")
        self.assertEqual(packs[2][0].request_id, "A")
        self.assertEqual(packs[3][0].request_id, "B")
        np.testing.assert_array_equal(
            packs[0][0].active_candidate_positions(),
            [0, 1],
        )
        np.testing.assert_array_equal(
            packs[1][0].active_candidate_positions(),
            [2, 3],
        )
        np.testing.assert_array_equal(
            packs[2][0].active_candidate_positions(),
            [4],
        )
        self.assertFalse(packs[0][0].releases_source_reference)
        self.assertFalse(packs[1][0].releases_source_reference)
        self.assertTrue(packs[2][0].releases_source_reference)

    def test_oversized_group_keeps_source_until_final_slice(self) -> None:
        from dataclasses import replace
        from pathlib import Path
        from unittest.mock import patch

        from src.config import load_app_config
        from src.train import _iter_batch_tables

        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "reference" / "default.yaml")
        sequence_source = config.sequences[0].fields[0].source
        train_split = replace(
            config.data.train,
            request_id="request_id",
            reader=replace(
                config.data.train.reader,
                deduplicate_request_features=True,
                shuffle_buffer_rows=0,
                length_buckets=(),
                agg_direct_mode="direct",
            ),
        )
        config = replace(
            config,
            data=replace(config.data, train=train_split),
            training=replace(config.training, batch_size=2),
        )
        table = pa.table(
            {
                "request_id": ["A"] * 5,
                "row_id": list(range(5)),
                sequence_source: [[1, 2]] * 5,
            }
        )

        with patch("src.train.iter_candidate_tables", return_value=iter([table])):
            batches = list(_iter_batch_tables(config, "train", 0, 1, True))

        self.assertEqual(
            [batch.table["row_id"].to_pylist() for batch in batches],
            [[0, 1], [2, 3], [4]],
        )

    def test_length_bucket_index_matches_bisect(self) -> None:
        boundaries = [128, 256, 512]
        self.assertEqual(length_bucket_index(0, boundaries), 0)
        self.assertEqual(length_bucket_index(128, boundaries), 0)
        self.assertEqual(length_bucket_index(129, boundaries), 1)
        self.assertEqual(length_bucket_index(1000, boundaries), 3)

    def test_direct_mode_falls_back_to_legacy_without_request_dedup(self) -> None:
        from dataclasses import replace
        from pathlib import Path
        from unittest.mock import patch

        from src.config import load_app_config
        from src.train import _iter_batch_tables

        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "reference" / "default.yaml")
        train_split = replace(
            config.data.train,
            reader=replace(
                config.data.train.reader,
                deduplicate_request_features=False,
                agg_direct_mode="direct",
                length_buckets=(),
                shuffle_buffer_rows=0,
            ),
        )
        config = replace(
            config,
            data=replace(config.data, train=train_split),
            sequences=(),
            training=replace(config.training, batch_size=2),
        )
        table = pa.table({"row_id": [0, 1, 2]})
        with patch("src.train.iter_candidate_tables", return_value=iter([table])):
            batches = list(_iter_batch_tables(config, "train", 0, 1, True))
        self.assertEqual(
            [batch["row_id"].to_pylist() for batch in batches],
            [[0, 1], [2]],
        )

    def test_length_bucketed_packs_match_legacy_row_coverage(self) -> None:
        from dataclasses import replace
        from pathlib import Path
        from unittest.mock import patch

        from src.agg_direct import iter_length_bucketed_packs, materialize_packed_blocks
        from src.config import LengthBucketConfig, load_app_config
        from src.train import _iter_batch_tables, _iter_length_bucketed_tables

        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "reference" / "default.yaml")
        buckets = (
            LengthBucketConfig(max_length=2, batch_size=3),
            LengthBucketConfig(max_length=4, batch_size=2),
            LengthBucketConfig(max_length=None, batch_size=2),
        )
        seq_source = config.sequences[0].fields[0].source
        train_split = replace(
            config.data.train,
            request_id="request_id",
            reader=replace(
                config.data.train.reader,
                deduplicate_request_features=True,
                shuffle_buffer_rows=0,
                length_buckets=buckets,
                length_bucket_metric="max",
                agg_direct_mode="legacy",
            ),
        )
        config = replace(
            config,
            data=replace(config.data, train=train_split),
            sequences=[replace(config.sequences[0], max_length=100)],
            training=replace(config.training, batch_size=4),
        )
        table = pa.table(
            {
                "request_id": ["a", "a", "b", "c", "c", "c", "d"],
                "row_id": list(range(7)),
                seq_source: [
                    [1],
                    [1],
                    [1, 2, 3],
                    [1, 2],
                    [1, 2],
                    [1, 2],
                    [1, 2, 3, 4, 5],
                ],
            }
        )

        with patch("src.train.iter_candidate_tables", return_value=iter([table])):
            legacy = [
                out["row_id"].to_pylist()
                for out in _iter_length_bucketed_tables(config, "train", 0, 1, True)
            ]

        blocks = request_group_blocks_from_adapted_table(
            table,
            source_id=0,
            request_id_column="request_id",
            sequences=config.sequences,
            length_bucket_metric="max",
        )
        packs = list(
            iter_length_bucketed_packs(
                iter(blocks),
                buckets=buckets,
                default_batch_size=config.training.batch_size,
                shuffle_buffer_rows=0,
            )
        )
        direct = [
            materialize_packed_blocks({0: table}, pack)["row_id"].to_pylist()
            for pack in packs
        ]
        self.assertEqual(direct, legacy)
        self.assertEqual(
            sorted(value for pack in direct for value in pack),
            list(range(7)),
        )

        direct_config = replace(
            config,
            data=replace(
                config.data,
                train=replace(
                    train_split,
                    reader=replace(train_split.reader, agg_direct_mode="direct"),
                ),
            ),
        )
        with patch("src.train.iter_candidate_tables", return_value=iter([table])):
            wired = [
                (
                    out.table if hasattr(out, "table") else out
                )["row_id"].to_pylist()
                for out in _iter_batch_tables(direct_config, "train", 0, 1, True)
            ]
        self.assertEqual(wired, legacy)


class SequenceSelectionPlanTest(unittest.TestCase):
    def test_truncate_then_compact_separates_bucket_and_final_lengths(self) -> None:
        # membership length 4 → truncate head to 3 → one null anchor → final 2.
        kept, pre, compact = row_sequence_selection_after_truncate_then_compact(
            list_length=4,
            anchor_is_null=np.asarray([False, True, False, False]),
            max_length=3,
            truncation="head",
        )
        self.assertEqual(pre, 3)
        self.assertEqual(compact, 2)
        np.testing.assert_array_equal(kept, [0, 2])

    def test_tail_truncate_then_compact(self) -> None:
        # [a, null, b, c], tail max_length=3 → window [null, b, c] → [b, c]
        kept, pre, compact = row_sequence_selection_after_truncate_then_compact(
            list_length=4,
            anchor_is_null=np.asarray([False, True, False, False]),
            max_length=3,
            truncation="tail",
        )
        self.assertEqual(pre, 3)
        self.assertEqual(compact, 2)
        np.testing.assert_array_equal(kept, [2, 3])

    def test_build_plan_from_packed_blocks(self) -> None:
        table = pa.table(
            {
                "request_id": ["A", "A", "B"],
                "goods": [[1, None, 3, 4], [1, None, 3, 4], [10, 11]],
                "age": [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [1.0, 2.0]],
            }
        )
        sequence = SimpleNamespace(
            name="hist",
            max_length=3,
            truncation="head",
            null_anchor_field="goods_id_hn",
            fields=[
                SimpleNamespace(name="goods_id_hn", source="goods"),
                SimpleNamespace(name="age", source="age"),
            ],
        )
        blocks = request_group_blocks_from_adapted_table(
            table,
            source_id=0,
            request_id_column="request_id",
            sequences=[
                _sequence("hist", "goods", max_length=3),
            ],
        )
        # Block bucket length ignores null_anchor: clamp(4,3)=3 for A, 2 for B.
        self.assertEqual(blocks[0].effective_bucket_length, 3)
        self.assertEqual(blocks[1].effective_bucket_length, 2)

        packed = build_packed_request_plan(blocks)
        plan = build_sequence_selection_plan(
            sequence,
            packed=packed,
            source_tables={0: table},
        )
        np.testing.assert_array_equal(plan.pre_compaction_lengths, [3, 2])
        np.testing.assert_array_equal(plan.compacted_lengths, [2, 2])
        np.testing.assert_array_equal(plan.selections[0], [0, 2])
        np.testing.assert_array_equal(plan.selections[1], [0, 1])
        np.testing.assert_array_equal(plan.token_to_request, [0, 0, 1, 1])

    def test_direct_shuffle_matches_legacy_with_buffer(self) -> None:
        from dataclasses import replace
        from pathlib import Path
        from unittest.mock import patch

        from src.config import LengthBucketConfig, load_app_config
        from src.train import _iter_batch_tables, _iter_length_bucketed_tables

        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "reference" / "default.yaml")
        buckets = (
            LengthBucketConfig(max_length=8, batch_size=4),
            LengthBucketConfig(max_length=None, batch_size=3),
        )
        seq_source = config.sequences[0].fields[0].source
        train_split = replace(
            config.data.train,
            request_id="request_id",
            reader=replace(
                config.data.train.reader,
                deduplicate_request_features=True,
                shuffle_buffer_rows=5,
                shuffle_seed=42,
                length_buckets=buckets,
                length_bucket_metric="max",
                agg_direct_mode="legacy",
            ),
        )
        config = replace(
            config,
            data=replace(config.data, train=train_split),
            sequences=[replace(config.sequences[0], max_length=100)],
            training=replace(config.training, batch_size=4),
        )
        table = pa.table(
            {
                "request_id": ["a", "a", "b", "c", "c", "d", "e", "e", "e"],
                "row_id": list(range(9)),
                seq_source: [
                    [1],
                    [1],
                    [1, 2],
                    [1, 2, 3],
                    [1, 2, 3],
                    [1],
                    [1, 2, 3, 4],
                    [1, 2, 3, 4],
                    [1, 2, 3, 4],
                ],
            }
        )
        with patch("src.train.iter_candidate_tables", return_value=iter([table])):
            legacy = [
                out["row_id"].to_pylist()
                for out in _iter_length_bucketed_tables(config, "train", 0, 1, True)
            ]
        direct_config = replace(
            config,
            data=replace(
                config.data,
                train=replace(
                    train_split,
                    reader=replace(train_split.reader, agg_direct_mode="direct"),
                ),
            ),
        )
        with patch("src.train.iter_candidate_tables", return_value=iter([table])):
            direct = [
                out["row_id"].to_pylist()
                for out in _iter_batch_tables(direct_config, "train", 0, 1, True)
            ]
        self.assertEqual(direct, legacy)

    def test_precomputed_request_dedup_matches_auto_plan(self) -> None:
        from src.agg_direct import (
            build_packed_request_plan,
            build_request_deduplication_from_pack,
            materialize_packed_blocks,
        )

        table = pa.table(
            {
                "request_id": ["A", "A", "B"],
                "hist_item_id": [[1, 2], [1, 2], [3]],
                "click": [0, 1, 1],
            }
        )
        blocks = request_group_blocks_from_adapted_table(
            table,
            source_id=0,
            request_id_column="request_id",
        )
        packed = build_packed_request_plan(blocks)
        candidate = materialize_packed_blocks({0: table}, packed.blocks)
        request_table, row_indices = build_request_deduplication_from_pack(
            packed,
            {0: table},
            columns=["request_id", "hist_item_id"],
        )
        self.assertEqual(candidate.num_rows, 3)
        self.assertEqual(request_table.num_rows, 2)
        self.assertEqual(request_table["request_id"].to_pylist(), ["A", "B"])
        self.assertEqual(row_indices.tolist(), [0, 0, 1])

    def test_multi_source_materialize_preserves_pack_order(self) -> None:
        from src.agg_direct import materialize_packed_blocks

        table0 = pa.table(
            {
                "request_id": ["A", "A", "B"],
                "row_id": [0, 1, 2],
            }
        )
        table1 = pa.table(
            {
                "request_id": ["C", "D", "D"],
                "row_id": [10, 11, 12],
            }
        )
        blocks = request_group_blocks_from_adapted_table(
            table0,
            source_id=0,
            request_id_column="request_id",
        ) + request_group_blocks_from_adapted_table(
            table1,
            source_id=1,
            request_id_column="request_id",
        )
        # Pack A(2)+B(1) then C(1)+D(2) with capacity 3.
        packs = list(iter_packed_request_groups(iter(blocks), batch_size=3))
        self.assertEqual(
            [
                materialize_packed_blocks({0: table0, 1: table1}, pack)[
                    "row_id"
                ].to_pylist()
                for pack in packs
            ],
            [[0, 1, 2], [10, 11, 12]],
        )


# --- Axis-separated adapt / source registry / pack materialize ---

class SourceRegistryTest(unittest.TestCase):
    def test_acquire_release_drops_payload(self) -> None:
        from src.agg_direct import SourceRegistry

        registry = SourceRegistry()
        source_id = registry.put({"payload": 1})
        registry.acquire(source_id, 2)
        self.assertEqual(registry.retained_count, 1)
        registry.release(source_id, 1)
        self.assertEqual(registry.retained_count, 1)
        registry.release(source_id, 1)
        self.assertEqual(registry.retained_count, 0)
        self.assertEqual(registry.release_events, 1)
        with self.assertRaises(KeyError):
            registry.get(source_id)

    def test_release_zero_drops_unreferenced_put(self) -> None:
        from src.agg_direct import SourceRegistry

        registry = SourceRegistry()
        source_id = registry.put("empty")
        registry.release(source_id, 0)
        self.assertEqual(registry.retained_count, 0)


class AxisSeparatedAdaptTest(unittest.TestCase):
    def test_direct_feature_batch_matches_legacy_narrow_arrow(self) -> None:
        from dataclasses import replace
        from pathlib import Path

        from src.config import ParquetAdapterConfig, load_app_config
        from src.dataloader import (
            axis_batch_to_feature_batch,
            table_to_feature_batch,
        )

        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "reference" / "default.yaml")
        sequence = replace(
            config.sequences[0],
            max_length=3,
            truncation="head",
            null_anchor_field="item_id",
        )
        adapter = ParquetAdapterConfig(
            callable="unused:test",
            options={
                "context_features": [
                    "user_id",
                    "rankmixer_context_dense",
                ],
            },
        )
        train_split = replace(
            config.data.train,
            request_id="request_id",
            group_id="request_id",
            adapter=adapter,
            reader=replace(
                config.data.train.reader,
                deduplicate_request_features=True,
            ),
        )
        config = replace(
            config,
            data=replace(config.data, train=train_split),
            sequences=(sequence,),
        )

        bundle = AdaptedAxisBundle(
            n_candidates=3,
            n_requests=2,
            request_ids=("r0", "r1"),
            candidate_to_request=np.asarray([0, 1, 0], dtype=np.int64),
            request_features={
                "user_id": ("u0", "u1"),
                "rankmixer_context_dense": (
                    tuple(float(index) for index in range(16)),
                    tuple(float(index + 20) for index in range(16)),
                ),
            },
            sequence_features={
                "hist_item_id": (
                    ("i0", None, "i2"),
                    ("i3", "i4"),
                ),
                "hist_shop_id": (
                    ("s0", "s1", "s2"),
                    ("s3", "s4"),
                ),
                "hist_action": (
                    ("a0", "a1", "a2"),
                    ("a3", "a4"),
                ),
                "hist_age": (
                    (0.1, 0.2, 0.3),
                    (0.4, 0.5),
                ),
                "hist_time_delta": (
                    (1.0, 2.0, 3.0),
                    (4.0, 5.0),
                ),
            },
            item_features={
                "item_id": ("c0", "c1", "c2"),
                "shop_id": ("cs0", "cs1", "cs2"),
            },
            label_features={"click": (0, 1, 0)},
            label_mask_features={},
            candidate_metadata={},
            request_raw_rows=np.asarray([0, 0], dtype=np.int64),
            candidate_raw_rows=np.asarray([0, 0, 0], dtype=np.int64),
        )
        blocks = request_group_blocks_from_axis_bundle(
            bundle,
            source_id=0,
            sequences=config.sequences,
        )
        packed = build_packed_request_plan(blocks)
        direct_input = prepare_packed_axis_batch(
            {0: bundle},
            packed,
            sequences=config.sequences,
            request_id_column="request_id",
            candidate_request_columns=("request_id",),
        )
        selection = direct_input.sequence_plans["hist"]
        np.testing.assert_array_equal(
            selection.pre_compaction_lengths,
            [3, 2],
        )
        np.testing.assert_array_equal(selection.compacted_lengths, [2, 2])

        candidate_table, request_table, row_indices = (
            materialize_packed_axis_bundles(
                {0: bundle},
                packed,
                request_columns=tuple(bundle.request_features),
                sequence_columns=tuple(bundle.sequence_features),
                candidate_columns=("item_id", "shop_id", "click", "request_id"),
                request_id_column="request_id",
            )
        )
        all_item_values = {
            value
            for values in (
                ("c0", "c1", "c2"),
                ("i0", "i2", "i3", "i4"),
            )
            for value in values
        }
        user_vocab = {"u0": 1, "u1": 2}
        item_vocab = {
            value: index + 1
            for index, value in enumerate(sorted(all_item_values))
        }
        vocab_maps = {
            "user_id": user_vocab,
            "scenario_user_id": user_vocab,
            "task_user_id": user_vocab,
            "item_id": item_vocab,
            "scenario_item_id": item_vocab,
            "task_item_id": item_vocab,
            "hist.item_id": item_vocab,
        }
        legacy = table_to_feature_batch(
            config,
            candidate_table,
            vocab_maps,
            split=train_split,
            request_deduplication=(request_table, row_indices),
        )
        direct = axis_batch_to_feature_batch(
            config,
            direct_input,
            vocab_maps,
            split=train_split,
        )

        def assert_equal(left: object, right: object, path: str) -> None:
            if isinstance(left, torch.Tensor):
                self.assertIsInstance(right, torch.Tensor, path)
                assert isinstance(right, torch.Tensor)
                self.assertEqual(left.dtype, right.dtype, path)
                self.assertEqual(left.shape, right.shape, path)
                torch.testing.assert_close(
                    left,
                    right,
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                    msg=path,
                )
                return
            if isinstance(left, dict):
                self.assertIsInstance(right, dict, path)
                assert isinstance(right, dict)
                self.assertEqual(set(left), set(right), path)
                for key in left:
                    assert_equal(left[key], right[key], f"{path}.{key}")
                return
            self.assertEqual(left, right, path)

        for attribute in (
            "features",
            "labels",
            "label_mask",
            "scenario_id",
            "group_id",
            "prediction_keys",
        ):
            assert_equal(
                getattr(legacy, attribute),
                getattr(direct, attribute),
                attribute,
            )

    def test_axis_oversized_group_survives_all_packed_slices(self) -> None:
        from dataclasses import replace
        from pathlib import Path
        from unittest.mock import patch

        from src.config import ParquetAdapterConfig, load_app_config
        from src.train import _iter_batch_tables

        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "reference" / "default.yaml")
        split = replace(
            config.data.train,
            format="adapter_parquet",
            request_id="request_id",
            group_id="request_id",
            adapter=ParquetAdapterConfig(callable="unused:test"),
            reader=replace(
                config.data.train.reader,
                deduplicate_request_features=True,
                agg_direct_mode="direct",
                prefetch_batches=0,
                shuffle_buffer_rows=0,
                length_buckets=(),
            ),
        )
        config = replace(
            config,
            data=replace(config.data, train=split),
            features=(),
            sequences=(),
            training=replace(config.training, batch_size=2),
        )
        bundle = AdaptedAxisBundle(
            n_candidates=5,
            n_requests=1,
            request_ids=("A",),
            candidate_to_request=np.zeros(5, dtype=np.int64),
            request_features={},
            sequence_features={},
            item_features={"row_id": tuple(range(5))},
            label_features={},
            label_mask_features={},
            candidate_metadata={},
            request_raw_rows=np.asarray([0], dtype=np.int64),
            candidate_raw_rows=np.zeros(5, dtype=np.int64),
        )
        with patch(
            "src.train.iter_adapted_axis_bundles",
            return_value=iter([bundle]),
        ):
            batches = list(_iter_batch_tables(config, "train", 0, 1, False))
        self.assertEqual(
            [list(batch.candidate_values["row_id"]) for batch in batches],
            [[0, 1], [2, 3], [4]],
        )

    def test_runtime_compare_runs_legacy_and_direct_oracles(self) -> None:
        from dataclasses import replace
        from pathlib import Path
        from unittest.mock import patch

        from src.config import load_app_config
        from src.train import iter_feature_batches

        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "reference" / "default.yaml")
        train_split = replace(
            config.data.train,
            reader=replace(
                config.data.train.reader,
                deduplicate_request_features=True,
                agg_direct_mode="compare",
                prefetch_batches=0,
                length_buckets=(),
            ),
        )
        config = replace(config, data=replace(config.data, train=train_split))
        table = pa.table(
            {
                "request_id": ["r0", "r1"],
                "user_id": ["u0", "u1"],
                "item_id": ["i0", "i1"],
                "shop_id": ["s0", "s1"],
                "rankmixer_context_dense": [
                    [float(index) for index in range(16)],
                    [float(index + 20) for index in range(16)],
                ],
                "hist_item_id": [["i0"], ["i1"]],
                "hist_shop_id": [["s0"], ["s1"]],
                "hist_action": [["a0"], ["a1"]],
                "hist_age": [[0.1], [0.2]],
                "hist_time_delta": [[1.0], [2.0]],
                "click": [0, 1],
            }
        )
        user_vocab = {"u0": 1, "u1": 2}
        item_vocab = {"i0": 1, "i1": 2}
        vocab_maps = {
            "user_id": user_vocab,
            "scenario_user_id": user_vocab,
            "task_user_id": user_vocab,
            "item_id": item_vocab,
            "scenario_item_id": item_vocab,
            "task_item_id": item_vocab,
            "hist.item_id": item_vocab,
        }
        modes: list[str] = []

        def batch_tables(active_config: object, *_args: object, **_kwargs: object):
            modes.append(active_config.data.train.reader.agg_direct_mode)
            return iter([table])

        with patch("src.train._iter_batch_tables", side_effect=batch_tables):
            batches = list(
                iter_feature_batches(
                    config,
                    "train",
                    vocab_maps,
                    require_labels=True,
                    pin_memory=False,
                )
            )
        self.assertEqual(modes, ["legacy", "direct"])
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].labels.tolist(), [[0.0], [1.0]])

    def test_axis_bundle_skips_candidate_flat_and_matches_legacy_axes(self) -> None:
        from types import SimpleNamespace

        from src.agg_direct import (
            AdaptedAxisBundle,
            build_packed_request_plan,
            materialize_packed_axis_bundles,
            request_group_blocks_from_axis_bundle,
        )
        from src.dataloader import adapt_mdl_rankmixer_parquet

        table = pa.table(
            {
                "context_indices": pa.array([[0, 1]], type=pa.list_(pa.int64())),
                "target_indices": pa.array([[0, 1, 1]], type=pa.list_(pa.int64())),
                "ctx_scalar_hn": pa.array(
                    [[[101], [102]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "ctx_bag_hn": pa.array(
                    [[[1, 2], None]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "item_scalar_hn": pa.array(
                    [[[201], [202], [203]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "sku_a_hn": pa.array(
                    [[[11, 12], [13], [14, 15]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "sku_b_hn": pa.array(
                    [[[21, None], [22], [23, 24]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "impr_x_goods_id_hn": pa.array(
                    [[-1, -2, -3]], type=pa.list_(pa.int64())
                ),
                "impr_x_time": pa.array(
                    [[4900, 4500, 3000]], type=pa.list_(pa.int64())
                ),
                "impr_x_indices": pa.array(
                    [[[0, 1], [1], [0]]], type=pa.list_(pa.list_(pa.int64()))
                ),
                "scene_id": pa.array([[7, 8]], type=pa.list_(pa.int64())),
                "search_id": pa.array([["r0", "r1"]], type=pa.list_(pa.string())),
                "impr_time": pa.array([[5000, 6000]], type=pa.list_(pa.int64())),
                "label_a": pa.array([[0, 1, 0]], type=pa.list_(pa.int64())),
                "label_b": pa.array([[1, 0, 1]], type=pa.list_(pa.int64())),
                "label_c": pa.array([[0, 0, 1]], type=pa.list_(pa.int64())),
            }
        )
        required = [
            "ctx_scalar_hn",
            "ctx_bag_hn",
            "item_scalar_hn",
            "sku_a_hn",
            "sku_b_hn",
            "impr_x_goods_id_hn",
            "impr_x_time_delta_ms",
            "scene_id",
            "search_id",
            "label_a",
            "label_b",
            "label_c",
        ]
        options = {
            "context_features": ["ctx_scalar_hn", "ctx_bag_hn"],
            "item_features": ["item_scalar_hn", "sku_a_hn", "sku_b_hn"],
            "multivalue_features": ["ctx_bag_hn", "sku_a_hn", "sku_b_hn"],
            "aligned_multivalue_groups": [["sku_a_hn", "sku_b_hn"]],
            "ups_types": ["impr"],
            "request_columns": ["scene_id", "search_id", "impr_time"],
            "integer_request_columns": ["scene_id", "impr_time"],
            "labels": {"a": "label_a", "b": "label_b", "c": "label_c"},
            "request_time_column": "impr_time",
            "time_delta_outputs": {"impr": "impr_x_time_delta_ms"},
        }
        legacy_context = SimpleNamespace(
            required_columns=tuple(required),
            options=options,
            trusted_input=False,
            _runtime_cache={},
        )
        legacy = adapt_mdl_rankmixer_parquet(table, context=legacy_context).to_pydict()

        axis_context = SimpleNamespace(
            required_columns=tuple(required),
            options=options,
            trusted_input=False,
            _runtime_cache={
                "axis_separated": True,
                "axis_request_id_column": "search_id",
            },
        )
        bundle = adapt_mdl_rankmixer_parquet(table, context=axis_context)
        self.assertIsInstance(bundle, AdaptedAxisBundle)
        self.assertEqual(bundle.n_candidates, 3)
        self.assertEqual(bundle.n_requests, 2)
        self.assertEqual(bundle.request_ids, ("r0", "r1"))
        # Request/sequence stored once per request, not broadcast onto candidates.
        self.assertEqual(len(bundle.request_features["search_id"]), 2)
        self.assertEqual(len(bundle.sequence_features["impr_x_goods_id_hn"]), 2)
        self.assertEqual(len(bundle.item_features["item_scalar_hn"]), 3)
        np.testing.assert_array_equal(bundle.candidate_to_request, [0, 1, 1])

        sequences = [
            SimpleNamespace(
                name="impr",
                max_length=None,
                fields=[SimpleNamespace(source="impr_x_goods_id_hn")],
            )
        ]
        blocks = request_group_blocks_from_axis_bundle(
            bundle, source_id=0, sequences=sequences
        )
        self.assertEqual([block.request_id for block in blocks], ["r0", "r1"])
        self.assertEqual(blocks[0].candidate_count, 1)
        self.assertEqual(blocks[1].candidate_count, 2)
        packed = build_packed_request_plan(blocks)
        candidate_table, request_table, row_indices = materialize_packed_axis_bundles(
            {0: bundle},
            packed,
            request_columns=sorted(bundle.request_features.keys()),
            sequence_columns=sorted(bundle.sequence_features.keys()),
            candidate_columns=sorted(
                {
                    *bundle.item_features.keys(),
                    *bundle.label_features.keys(),
                    "search_id",
                }
            ),
            request_id_column="search_id",
        )
        self.assertEqual(candidate_table.num_rows, 3)
        self.assertEqual(request_table.num_rows, 2)
        self.assertEqual(row_indices.tolist(), [0, 1, 1])
        self.assertEqual(
            candidate_table["item_scalar_hn"].to_pylist(),
            legacy["item_scalar_hn"],
        )
        self.assertEqual(
            request_table["impr_x_goods_id_hn"].to_pylist(),
            [[-1, -3], [-1, -2]],
        )
        self.assertEqual(
            candidate_table["search_id"].to_pylist(),
            legacy["search_id"],
        )

    def test_same_search_id_across_contexts_is_one_request_slot(self) -> None:
        from types import SimpleNamespace

        from src.agg_direct import AdaptedAxisBundle
        from src.dataloader import adapt_mdl_rankmixer_parquet

        # Two context positions, same search_id A; three candidates.
        table = pa.table(
            {
                "context_indices": [[10, 20]],
                "target_indices": [[10, 20, 10]],
                "ctx_scalar_hn": [[[101], [999]]],
                "ctx_bag_hn": [[[1], [2]]],
                "item_scalar_hn": [[[201], [202], [203]]],
                "sku_a_hn": [[[11], [12], [13]]],
                "sku_b_hn": [[[21], [22], [23]]],
                "impr_x_goods_id_hn": [[-1, -2]],
                "impr_x_time": [[4900, 4800]],
                "impr_x_indices": [[[10, 20], [10]]],
                "scene_id": [[7, 8]],
                "search_id": [["A", "A"]],
                "impr_time": [[5000, 6000]],
                "label_a": [[0, 1, 0]],
                "label_b": [[1, 0, 1]],
                "label_c": [[0, 0, 1]],
            }
        )
        required = [
            "ctx_scalar_hn",
            "ctx_bag_hn",
            "item_scalar_hn",
            "sku_a_hn",
            "sku_b_hn",
            "impr_x_goods_id_hn",
            "impr_x_time_delta_ms",
            "scene_id",
            "search_id",
            "label_a",
            "label_b",
            "label_c",
        ]
        options = {
            "context_features": ["ctx_scalar_hn", "ctx_bag_hn"],
            "item_features": ["item_scalar_hn", "sku_a_hn", "sku_b_hn"],
            "multivalue_features": ["ctx_bag_hn", "sku_a_hn", "sku_b_hn"],
            "aligned_multivalue_groups": [["sku_a_hn", "sku_b_hn"]],
            "ups_types": ["impr"],
            "request_columns": ["scene_id", "search_id", "impr_time"],
            "integer_request_columns": ["scene_id", "impr_time"],
            "labels": {"a": "label_a", "b": "label_b", "c": "label_c"},
            "request_time_column": "impr_time",
            "time_delta_outputs": {"impr": "impr_x_time_delta_ms"},
        }
        bundle = adapt_mdl_rankmixer_parquet(
            table,
            context=SimpleNamespace(
                required_columns=tuple(required),
                options=options,
                trusted_input=False,
                _runtime_cache={
                    "axis_separated": True,
                    "axis_request_id_column": "search_id",
                },
            ),
        )
        self.assertIsInstance(bundle, AdaptedAxisBundle)
        self.assertEqual(bundle.n_requests, 1)
        self.assertEqual(bundle.request_ids, ("A",))
        # First-wins representative payload is context 10 (ctx_scalar 101), not 999.
        self.assertEqual(bundle.request_features["ctx_scalar_hn"], (101,))
        np.testing.assert_array_equal(bundle.candidate_to_request, [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
