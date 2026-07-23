from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.config import (
    LengthBucketConfig,
    ParquetAdapterConfig,
    ParquetSplitConfig,
    ReaderConfig,
    load_app_config,
)
from src.dataloader import (
    FeatureBatch,
    ParquetInputRef,
    ParquetScanner,
    _ByteBudget,
    _coalesce_feature_batch,
    _eager_schema_validation_refs,
    _iter_adapted_flat_tables,
    _request_deduplication_plan,
    _safe_table_take,
    _scan_columns_for_split,
    _validate_flat_table_contract,
    move_feature_batch,
    table_to_feature_batch,
)
from src.features import _flatten_array_values
from src.train import (
    _DevicePrefetchIterator,
    _concat_batch_tables,
    _estimate_prepared_batch_bytes,
    _iter_batch_tables,
    _iter_shuffled_candidate_tables,
    _shuffle_table,
    _table_effective_sequence_lengths,
)


class LengthBucketTest(unittest.TestCase):
    def test_streaming_shuffle_is_deterministic_and_preserves_exact_coverage(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "reference" / "default.yaml")
        train_split = replace(
            config.data.train,
            reader=replace(
                config.data.train.reader,
                shuffle_buffer_rows=3,
                shuffle_seed=17,
            ),
        )
        config = replace(config, data=replace(config.data, train=train_split))
        tables = [
            pa.table({"row_id": [0, 1, 2, 3]}),
            pa.table({"row_id": [4, 5]}),
            pa.table({"row_id": [6, 7, 8, 9]}),
        ]

        def shuffled(rank: int = 0) -> list[int]:
            with patch(
                "src.train.iter_candidate_tables",
                return_value=iter(tables),
            ):
                output = list(
                    _iter_shuffled_candidate_tables(
                        config,
                        "train",
                        rank,
                        2,
                        True,
                    )
                )
            return [
                value
                for table in output
                for value in table["row_id"].to_pylist()
            ]

        first = shuffled()
        self.assertEqual(first, shuffled())
        self.assertEqual(sorted(first), list(range(10)))
        self.assertNotEqual(first, list(range(10)))
        self.assertNotEqual(first, shuffled(rank=1))

    def test_request_shuffle_keeps_candidates_grouped_and_ordered(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "reference" / "default.yaml")
        train_split = replace(
            config.data.train,
            reader=replace(
                config.data.train.reader,
                deduplicate_request_features=True,
                shuffle_buffer_rows=4,
                shuffle_seed=23,
            ),
        )
        config = replace(config, data=replace(config.data, train=train_split))
        tables = [
            pa.table(
                {
                    "request_id": ["a", "b", "a", "b"],
                    "candidate": [0, 0, 1, 1],
                    "row_id": [0, 1, 2, 3],
                }
            ),
            pa.table(
                {
                    "request_id": ["c", "c", "d"],
                    "candidate": [0, 1, 0],
                    "row_id": [4, 5, 6],
                }
            ),
        ]

        def shuffled() -> list[pa.Table]:
            with patch("src.train.iter_candidate_tables", return_value=iter(tables)):
                return list(
                    _iter_shuffled_candidate_tables(config, "train", 0, 1, True)
                )

        first = shuffled()
        second = shuffled()
        self.assertEqual(
            [table["row_id"].to_pylist() for table in first],
            [table["row_id"].to_pylist() for table in second],
        )
        self.assertEqual(
            sorted(value for table in first for value in table["row_id"].to_pylist()),
            list(range(7)),
        )
        for table in first:
            self.assertEqual(len(set(table["request_id"].to_pylist())), 1)
            self.assertEqual(
                table["candidate"].to_pylist(),
                sorted(table["candidate"].to_pylist()),
            )

    def test_rows_are_vectorized_into_configured_length_buckets(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "reference" / "default.yaml")
        buckets = (
            LengthBucketConfig(max_length=128, batch_size=2),
            LengthBucketConfig(max_length=256, batch_size=2),
            LengthBucketConfig(max_length=None, batch_size=1),
        )
        train_split = replace(
            config.data.train,
            reader=replace(config.data.train.reader, length_buckets=buckets),
        )
        config = replace(
            config,
            data=replace(config.data, train=train_split),
            sequences=[replace(config.sequences[0], max_length=1000)],
        )
        source = config.sequences[0].fields[0].source
        table = pa.table(
            {
                "row_id": list(range(7)),
                source: [
                    list(range(10)),
                    list(range(130)),
                    list(range(20)),
                    list(range(300)),
                    list(range(129)),
                    list(range(128)),
                    list(range(257)),
                ],
            }
        )

        with patch("src.train.iter_candidate_tables", return_value=iter([table])):
            batches = list(_iter_batch_tables(config, "train", 0, 1))

        observed = [batch["row_id"].to_pylist() for batch in batches]
        self.assertEqual(observed, [[0, 2], [1, 4], [3], [6], [5]])
        for batch in batches:
            lengths = [len(items) for items in batch[source].to_pylist()]
            self.assertLessEqual(max(lengths) - min(lengths), 128)

    def test_sum_metric_counts_work_across_sequences(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "reference" / "default.yaml")
        first = config.sequences[0]
        second = replace(first, name="second", fields=first.fields)
        config = replace(config, sequences=(first, second))
        source = first.fields[0].source
        table = pa.table({source: [[1, 2], [1]]})

        maximum = _table_effective_sequence_lengths(config, table, metric="max")
        summed = _table_effective_sequence_lengths(config, table, metric="sum")

        self.assertEqual(maximum.tolist(), [2, 1])
        self.assertEqual(summed.tolist(), [4, 2])

    def test_request_bucket_computes_shared_length_once_and_does_not_split(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "reference" / "default.yaml")
        buckets = (
            LengthBucketConfig(max_length=128, batch_size=3),
            LengthBucketConfig(max_length=None, batch_size=3),
        )
        train_split = replace(
            config.data.train,
            reader=replace(
                config.data.train.reader,
                deduplicate_request_features=True,
                shuffle_buffer_rows=0,
                length_buckets=buckets,
            ),
        )
        config = replace(
            config,
            data=replace(config.data, train=train_split),
            sequences=(replace(config.sequences[0], max_length=1000),),
        )
        source = config.sequences[0].fields[0].source
        table = pa.table(
            {
                "request_id": ["r0", "r1", "r0", "r1", "r2"],
                "row_id": [0, 1, 2, 3, 4],
                source: [
                    list(range(10)),
                    list(range(130)),
                    list(range(10)),
                    list(range(130)),
                    list(range(20)),
                ],
            }
        )

        with patch("src.train.iter_candidate_tables", return_value=iter([table])), patch(
            "src.train._table_effective_sequence_lengths",
            wraps=_table_effective_sequence_lengths,
        ) as lengths:
            batches = list(_iter_batch_tables(config, "train", 0, 1))

        self.assertEqual(
            [batch["row_id"].to_pylist() for batch in batches],
            [[0, 2, 4], [1, 3]],
        )
        self.assertEqual(len(lengths.call_args_list), 1)
        self.assertEqual(lengths.call_args_list[0].args[1].num_rows, 5)


class EagerSchemaValidationTest(unittest.TestCase):
    def test_full_flat_contract_is_validated_only_on_the_first_nonempty_batch(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "reference" / "default.yaml")
        tables = [pa.table({"value": [1]}), pa.table({"value": [2]})]
        scanner = SimpleNamespace(
            split=config.data.train,
            iter_tables=lambda: iter(tables),
        )
        context = SimpleNamespace()

        with (
            patch("src.dataloader._validate_flat_table_static_contract") as validate_static,
            patch("src.dataloader._validate_complete_label_contract") as validate_labels,
        ):
            actual = list(
                _iter_adapted_flat_tables(
                    config,
                    "train",
                    scanner,
                    "identity",
                    lambda table, *, context: table,
                    context,
                    ["value"],
                )
            )

        self.assertEqual([table["value"].to_pylist() for table in actual], [[1], [2]])
        validate_static.assert_called_once()
        self.assertEqual(validate_labels.call_count, 2)

    def test_complete_label_contract_runs_on_every_flat_batch(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "reference" / "default.yaml")
        split = replace(
            config.data.train,
            reader=replace(config.data.train.reader, trusted_input=True),
        )
        tables = [pa.table({"value": [1, 2, 3]}), pa.table({"value": [4, 5]})]
        scanner = SimpleNamespace(split=split, iter_tables=lambda: iter(tables))

        with (
            patch("src.dataloader._validate_flat_table_static_contract") as validate_static,
            patch("src.dataloader._validate_complete_label_contract") as validate_labels,
        ):
            actual = list(
                _iter_adapted_flat_tables(
                    config,
                    "train",
                    scanner,
                    "identity",
                    lambda table, *, context: table,
                    SimpleNamespace(),
                    ["value"],
                )
            )

        self.assertEqual([table.num_rows for table in actual], [3, 2])
        validate_static.assert_called_once()
        self.assertEqual(validate_static.call_args.args[3].num_rows, 3)
        self.assertEqual(validate_labels.call_count, 2)
        self.assertEqual(
            [call.args[1].num_rows for call in validate_labels.call_args_list],
            [3, 2],
        )

    def test_row_group_sharding_covers_rows_exactly_once_across_eight_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            expected = list(range(24))
            for file_index in range(4):
                first = file_index * 6
                pq.write_table(
                    pa.table({"row_id": list(range(first, first + 6))}),
                    Path(directory) / f"part-{file_index}.parquet",
                    row_group_size=3,
                )
            split = ParquetSplitConfig(
                format="flat_parquet",
                inputs=(directory,),
                reader=ReaderConfig(
                    num_workers=0,
                    prefetch_batches=0,
                    shard_unit="row_group",
                    scanner_batch_rows=2,
                ),
            )

            rows_by_rank: list[list[int]] = []
            fingerprints: set[str | None] = set()
            for rank in range(8):
                scanner = ParquetScanner(
                    split,
                    ["row_id"],
                    shard_rank=rank,
                    shard_world_size=8,
                )
                rows_by_rank.append(
                    [
                        value
                        for batch in scanner.iter_record_batches()
                        for value in batch.column("row_id").to_pylist()
                    ]
                )
                fingerprints.add(scanner.shard_plan_fingerprint)

        flattened = [value for rows in rows_by_rank for value in rows]
        self.assertEqual(sorted(flattened), expected)
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertTrue(all(rows for rows in rows_by_rank))
        self.assertEqual(len(fingerprints), 1)

    def test_unlabeled_inference_omits_adapter_label_inputs(self) -> None:
        split = ParquetSplitConfig(
            format="adapter_parquet",
            inputs=("/tmp/unused.parquet",),
            adapter=ParquetAdapterConfig(
                callable="examples.parquet_identity_adapter:adapt",
                input_columns=("value", "label", "label_valid"),
                optional_input_columns=("optional",),
            ),
            labels={"task": "label"},
            label_masks={"task": "label_valid"},
        )

        self.assertEqual(
            _scan_columns_for_split(split, ["value", "optional"]),
            ["value", "optional"],
        )

    @staticmethod
    def _refs(count: int) -> list[ParquetInputRef]:
        filesystem = object()
        return [
            ParquetInputRef(
                canonical_uri=f"hdfs://cluster/data/part-{index:05d}.parquet",
                filesystem_key="hdfs://cluster",
                fs_path=f"/data/part-{index:05d}.parquet",
                filesystem=filesystem,
            )
            for index in range(count)
        ]

    def test_sample_is_evenly_spaced_and_includes_endpoints(self) -> None:
        refs = self._refs(1000)
        selected = _eager_schema_validation_refs(refs, "sample", 64)

        self.assertEqual(len(selected), 64)
        self.assertIs(selected[0], refs[0])
        self.assertIs(selected[-1], refs[-1])
        self.assertEqual(selected, sorted(selected, key=lambda item: item.canonical_uri))

    def test_all_mode_keeps_every_file(self) -> None:
        refs = self._refs(5)
        self.assertIs(_eager_schema_validation_refs(refs, "all", 2), refs)

    def test_file_shard_uses_parquet_file_not_dataset_scanner(self) -> None:
        """HDFS file sharding must not use Dataset fragment_readahead opens."""

        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(3):
                path = Path(directory) / f"part-{index}.parquet"
                pq.write_table(pa.table({"row_id": [index, index + 10]}), path)
                paths.append(path)

            split = ParquetSplitConfig(
                format="flat_parquet",
                inputs=tuple(str(path) for path in paths),
                reader=ReaderConfig(
                    num_workers=8,
                    prefetch_batches=8,
                    shard_unit="file",
                    scanner_batch_rows=2,
                ),
            )
            scanner = ParquetScanner(
                split,
                ["row_id"],
                shard_rank=0,
                shard_world_size=1,
            )
            opened: list[str] = []
            real_parquet_file = pq.ParquetFile

            def tracking_parquet_file(fs_path, **kwargs):
                opened.append(str(fs_path))
                return real_parquet_file(fs_path, **kwargs)

            with patch("pyarrow.parquet.ParquetFile", side_effect=tracking_parquet_file):
                with patch("pyarrow.dataset.dataset") as dataset_factory:
                    rows = [
                        value
                        for batch in scanner.iter_record_batches()
                        for value in batch.column("row_id").to_pylist()
                    ]

        self.assertEqual(sorted(rows), [0, 1, 2, 10, 11, 12])
        self.assertEqual(len(opened), 3)
        dataset_factory.assert_not_called()
        self.assertFalse(scanner._filesystem_is_remote())

    def test_remote_filesystem_scales_prefetch_workers(self) -> None:
        from src.dataloader import RemoteIoPolicy

        scanner = ParquetScanner.__new__(ParquetScanner)
        scanner.paths = self._refs(2)
        scanner.shard_world_size = 4
        scanner.split = type(
            "Split",
            (),
            {"reader": ReaderConfig(num_workers=8, prefetch_batches=8)},
        )()
        scanner._io_policy = RemoteIoPolicy.from_reader(scanner.split.reader, remote=True)
        self.assertTrue(scanner._filesystem_is_remote())
        self.assertEqual(scanner._prefetch_active_workers(8), 4)

    def test_optional_adapter_projection_is_selected_only_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "req.parquet"
            pq.write_table(pa.table({"value": [1, 2]}), path)
            split = ParquetSplitConfig(
                format="adapter_parquet",
                inputs=(str(path),),
                adapter=ParquetAdapterConfig(
                    callable="examples.parquet_identity_adapter:adapt",
                    input_columns=("value",),
                    optional_input_columns=("context_indices",),
                ),
            )
            scanner = ParquetScanner(
                split,
                ["value", "context_indices"],
                optional_columns=("context_indices",),
            )

            batches = list(scanner.iter_record_batches())

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].schema.names, ["value"])


class RequestDeduplicationTest(unittest.TestCase):
    def test_stably_selects_one_row_per_request(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(
            root / "configs" / "rankmixer.yaml"
        )
        split = replace(
            config.data.train,
            reader=replace(
                config.data.train.reader,
                deduplicate_request_features=True,
            ),
        )
        table = pa.table(
            {
                "search_id": ["r0", "r0", "r1", "r0", "r2"],
                "value": [10, 11, 20, 12, 30],
            }
        )

        selected, row_indices = _request_deduplication_plan(split, table)

        self.assertEqual(selected["value"].to_pylist(), [10, 20, 30])
        self.assertEqual(row_indices.tolist(), [0, 0, 1, 0, 2])

    def test_dedup_survives_multi_chunk_dictionary_list_columns(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "rankmixer.yaml")
        split = replace(
            config.data.train,
            reader=replace(
                config.data.train.reader,
                deduplicate_request_features=True,
            ),
        )
        table = _multi_chunk_dictionary_list_table(
            request_ids=["r0", "r0", "r1", "r1"],
            bags=[[[1, 2], [1, 2]], [[3], [4, 5, 6]]],
            request_column=split.request_id or "search_id",
            bag_column="bag",
        )
        with self.assertRaises(pa.lib.ArrowNotImplementedError):
            table.take(pa.array([0, 2], type=pa.int64()))

        selected, row_indices = _request_deduplication_plan(split, table)

        self.assertEqual(selected[split.request_id or "search_id"].to_pylist(), ["r0", "r1"])
        self.assertEqual(row_indices.tolist(), [0, 0, 1, 1])
        self.assertEqual(
            _flatten_array_values(selected["bag"]),
            [1, 2, 3],
        )


def _dictionary_list_chunk(rows: list[list[int]]) -> pa.DictionaryArray:
    unique: list[list[int]] = []
    indices: list[int] = []
    seen: dict[tuple[int, ...], int] = {}
    for row in rows:
        key = tuple(row)
        if key not in seen:
            seen[key] = len(unique)
            unique.append(row)
        indices.append(seen[key])
    return pa.DictionaryArray.from_arrays(
        pa.array(indices, type=pa.int32()),
        pa.array(unique, type=pa.list_(pa.int64())),
    )


def _multi_chunk_dictionary_list_table(
    *,
    request_ids: list[str],
    bags: list[list[list[int]]],
    request_column: str,
    bag_column: str,
) -> pa.Table:
    mid = len(request_ids) // 2
    chunk_a = pa.table(
        {
            request_column: request_ids[:mid],
            bag_column: _dictionary_list_chunk(bags[0]),
            "label": list(range(mid)),
        }
    )
    chunk_b = pa.table(
        {
            request_column: request_ids[mid:],
            bag_column: _dictionary_list_chunk(bags[1]),
            "label": list(range(mid, len(request_ids))),
        }
    )
    return pa.concat_tables([chunk_a, chunk_b])


class NestedDictionarySafetyTest(unittest.TestCase):
    def test_batch_concat_decodes_only_selected_dictionary_rows(self) -> None:
        table = _multi_chunk_dictionary_list_table(
            request_ids=["r0", "r0", "r1", "r1"],
            bags=[[[1, 2], [1, 2]], [[3], [4, 5, 6]]],
            request_column="search_id",
            bag_column="bag",
        )

        combined = _concat_batch_tables(
            pa,
            [table.slice(0, 1), table.slice(3, 1)],
        )

        self.assertEqual(combined["bag"].to_pylist(), [[1, 2], [4, 5, 6]])
        self.assertTrue(pa.types.is_list(combined.schema.field("bag").type))
        self.assertEqual(combined["bag"].num_chunks, 1)

    def test_safe_table_take_preserves_alignment_under_permutation(self) -> None:
        table = _multi_chunk_dictionary_list_table(
            request_ids=["r0", "r0", "r1", "r1"],
            bags=[[[1, 2], [1, 2]], [[3], [4, 5, 6]]],
            request_column="search_id",
            bag_column="bag",
        )
        taken = _safe_table_take(table, [3, 0, 2, 1])
        self.assertEqual(taken["search_id"].to_pylist(), ["r1", "r0", "r1", "r0"])
        self.assertEqual(taken["label"].to_pylist(), [3, 0, 2, 1])
        self.assertEqual(taken["bag"].to_pylist(), [[4, 5, 6], [1, 2], [3], [1, 2]])

    def test_safe_table_take_supports_empty_and_repeated_indices(self) -> None:
        table = _multi_chunk_dictionary_list_table(
            request_ids=["r0", "r0", "r1", "r1"],
            bags=[[[1], [1]], [[2, 2], [3]]],
            request_column="search_id",
            bag_column="bag",
        )
        empty = _safe_table_take(table, [])
        self.assertEqual(empty.num_rows, 0)
        self.assertEqual(empty.column_names, table.column_names)
        repeated = _safe_table_take(table, [1, 1, 0])
        self.assertEqual(repeated["label"].to_pylist(), [1, 1, 0])
        self.assertEqual(repeated["bag"].to_pylist(), [[1], [1], [1]])

    def test_shuffle_table_survives_multi_chunk_dictionary_lists(self) -> None:
        table = _multi_chunk_dictionary_list_table(
            request_ids=["r0", "r0", "r1", "r1"],
            bags=[[[1, 2], [1, 2]], [[3], [4, 5, 6]]],
            request_column="search_id",
            bag_column="bag",
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(7)
        shuffled = _shuffle_table(table, generator)
        self.assertEqual(shuffled.num_rows, table.num_rows)
        self.assertCountEqual(shuffled["label"].to_pylist(), [0, 1, 2, 3])
        self.assertCountEqual(
            shuffled["bag"].to_pylist(),
            [[1, 2], [1, 2], [3], [4, 5, 6]],
        )

    def test_flatten_array_values_survives_multi_chunk_dictionary_lists(self) -> None:
        table = _multi_chunk_dictionary_list_table(
            request_ids=["r0", "r0", "r1", "r1"],
            bags=[[[1, 2], [1, 2]], [[3], [4, 5, 6]]],
            request_column="search_id",
            bag_column="bag",
        )
        with self.assertRaises(pa.lib.ArrowNotImplementedError):
            table["bag"].combine_chunks()
        self.assertEqual(
            _flatten_array_values(table["bag"]),
            [1, 2, 1, 2, 3, 4, 5, 6],
        )


class CompleteLabelTest(unittest.TestCase):
    def test_first_batch_contract_rejects_non_binary_complete_labels(self) -> None:
        root = Path(__file__).resolve().parents[1]
        base = load_app_config(root / "configs" / "reference" / "default.yaml")
        split = replace(
            base.data.train,
            labels={"click": "click"},
            label_masks={},
        )
        config = replace(base, features=(), sequences=())

        with self.assertRaisesRegex(ValueError, "must contain only 0/1"):
            _validate_flat_table_contract(
                config,
                split,
                "train",
                pa.table({"click": [0, 2]}),
                ["click"],
            )

    def test_complete_labels_do_not_allocate_an_all_ones_mask(self) -> None:
        root = Path(__file__).resolve().parents[1]
        base = load_app_config(root / "configs" / "reference" / "default.yaml")
        split = replace(
            base.data.train,
            labels={"click": "click"},
            label_masks={},
            request_id=None,
            group_id=None,
        )
        config = replace(base, features=(), sequences=())

        batch = table_to_feature_batch(
            config,
            pa.table({"click": [0, 1]}),
            {},
            split=split,
        )

        self.assertEqual(batch.labels.tolist(), [[0.0], [1.0]])
        self.assertIsNone(batch.label_mask)


class CoalescedBatchTest(unittest.TestCase):
    def test_packs_shared_tensor_leaves_once_and_preserves_views(self) -> None:
        row_indices = torch.tensor([0, 0, 1], dtype=torch.long)
        batch = FeatureBatch(
            features={
                "a": {"values": torch.tensor([3, 4]), "row_indices": row_indices},
                "b": {"values": torch.tensor([[5], [6]]), "row_indices": row_indices},
                "dense": torch.tensor([[1.0], [2.0], [3.0]]),
            },
            labels=torch.tensor([[1.0], [0.0], [1.0]]),
            label_mask=torch.ones(3, 1),
            scenario_id=torch.tensor([0, 1, 0]),
            group_id=[],
        )

        packed = _coalesce_feature_batch(batch, pin_memory=False)

        self.assertEqual(len(packed._packed_buffers), 2)
        self.assertEqual(packed.features["a"]["values"].tolist(), [3, 4])
        self.assertEqual(packed.features["dense"].tolist(), [[1.0], [2.0], [3.0]])
        self.assertEqual(
            packed.features["a"]["row_indices"].data_ptr(),
            packed.features["b"]["row_indices"].data_ptr(),
        )

        moved = move_feature_batch(packed, torch.device("cpu"))
        self.assertEqual(moved.labels.tolist(), [[1.0], [0.0], [1.0]])
        self.assertEqual(moved.scenario_id.tolist(), [0, 1, 0])


class DevicePrefetchTest(unittest.TestCase):
    def test_resolves_implicit_cuda_device_before_starting_worker(self) -> None:
        with patch("src.train.torch.cuda.current_device", return_value=3), patch(
            "src.train.threading.Thread"
        ) as thread_type:
            iterator = _DevicePrefetchIterator(
                iter(()),
                torch.device("cuda"),
                depth=1,
            )

        self.assertEqual(iterator.device, torch.device("cuda", 3))
        thread_type.return_value.start.assert_called_once_with()
        iterator.close()
        thread_type.return_value.join.assert_called_once_with()


class ByteBudgetTest(unittest.TestCase):
    def test_prepared_batch_estimate_supports_dictionary_encoded_list_bags(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(
            root / "configs" / "rankmixer.yaml"
        )
        feature = next(
            item
            for item in config.features
            if item.kind == "categorical" and item.pooling == "mean"
        )
        dictionary = pa.array([[1, 2, 3], [4]], type=pa.list_(pa.int64()))
        encoded = pa.DictionaryArray.from_arrays(
            pa.array([0, 1, 0], type=pa.int32()),
            dictionary,
        )
        table = pa.table({feature.source: encoded})
        config = replace(config, features=(feature,), sequences=())

        reservation = _estimate_prepared_batch_bytes(config, table)

        self.assertGreater(reservation, table.nbytes)

    def test_prepared_batch_estimate_supports_multi_chunk_dictionary_list_bags(
        self,
    ) -> None:
        """concat of compact_request_lists bags must not require dict unify."""

        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "rankmixer.yaml")
        feature = next(
            item
            for item in config.features
            if item.kind == "categorical" and item.pooling == "mean"
        )
        dictionary_a = pa.array([[1, 2], [3]], type=pa.list_(pa.int64()))
        dictionary_b = pa.array([[4], [5, 6, 7]], type=pa.list_(pa.int64()))
        chunk_a = pa.DictionaryArray.from_arrays(
            pa.array([0, 1], type=pa.int32()),
            dictionary_a,
        )
        chunk_b = pa.DictionaryArray.from_arrays(
            pa.array([1, 0], type=pa.int32()),
            dictionary_b,
        )
        table = pa.Table.from_arrays(
            [pa.chunked_array([chunk_a, chunk_b])],
            names=[feature.source],
        )
        config = replace(config, features=(feature,), sequences=())

        with self.assertRaises(pa.lib.ArrowNotImplementedError):
            table[feature.source].combine_chunks()

        reservation = _estimate_prepared_batch_bytes(config, table)

        self.assertGreater(reservation, table.nbytes)
        # Conservative path may use full dictionary max (3), not only referenced.
        self.assertGreaterEqual(
            reservation,
            table.nbytes + table.num_rows * (8 + 3 * 8),
        )

    def test_budget_blocks_until_bytes_are_released(self) -> None:
        budget = _ByteBudget(100)
        stop = threading.Event()
        self.assertTrue(budget.acquire(80, stop))
        acquired = threading.Event()

        def wait_for_budget() -> None:
            if budget.acquire(40, stop):
                acquired.set()

        worker = threading.Thread(target=wait_for_budget)
        worker.start()
        time.sleep(0.05)
        self.assertFalse(acquired.is_set())
        budget.release(80)
        worker.join(timeout=1.0)
        self.assertTrue(acquired.is_set())
        budget.release(40)

    def test_one_oversized_batch_is_admitted_for_progress(self) -> None:
        budget = _ByteBudget(100)
        stop = threading.Event()
        self.assertTrue(budget.acquire(150, stop))
        self.assertEqual(budget.used, 150)
        budget.release(150)


if __name__ == "__main__":
    unittest.main()
