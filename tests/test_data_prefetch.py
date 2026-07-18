from __future__ import annotations

from dataclasses import replace
from pathlib import Path
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
    _request_deduplication_plan,
    _scan_columns_for_split,
    move_feature_batch,
)
from src.train import (
    _estimate_prepared_batch_bytes,
    _iter_batch_tables,
    _iter_shuffled_candidate_tables,
    _table_effective_sequence_lengths,
)


class LengthBucketTest(unittest.TestCase):
    def test_streaming_shuffle_is_deterministic_and_preserves_exact_coverage(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "default.yaml")
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

    def test_rows_are_vectorized_into_configured_length_buckets(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "default.yaml")
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
        config = load_app_config(root / "configs" / "default.yaml")
        first = config.sequences[0]
        second = replace(first, name="second", fields=first.fields)
        config = replace(config, sequences=(first, second))
        source = first.fields[0].source
        table = pa.table({source: [[1, 2], [1]]})

        maximum = _table_effective_sequence_lengths(config, table, metric="max")
        summed = _table_effective_sequence_lengths(config, table, metric="sum")

        self.assertEqual(maximum.tolist(), [2, 1])
        self.assertEqual(summed.tolist(), [4, 2])


class EagerSchemaValidationTest(unittest.TestCase):
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
            root / "configs" / "rankmixer_a100_8x80g.yaml"
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


class ByteBudgetTest(unittest.TestCase):
    def test_prepared_batch_estimate_supports_dictionary_encoded_list_bags(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(
            root / "configs" / "rankmixer_a100_8x80g.yaml"
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
