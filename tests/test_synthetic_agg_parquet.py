from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import pyarrow.parquet as pq

from scripts.generate_synthetic_agg_parquet import (
    OBSERVED_MEDIAN_SEQUENCE_LENGTHS,
    SyntheticAggManifest,
    generate_synthetic_agg_dataset,
)
from scripts.tune_batch_size import (
    _default_sequence_lengths,
    _hdfs_sensitivity,
    _recommended_yaml_override,
)
from src.config import load_app_config
from src.dataloader import iter_flat_tables


ROOT = Path(__file__).resolve().parents[1]


class SyntheticAggParquetTest(unittest.TestCase):
    def test_generated_wide_agg_file_runs_through_production_adapter(self) -> None:
        config = load_app_config(ROOT / "configs" / "rankmixer.yaml")
        short_lengths = {name: 2 for name in OBSERVED_MEDIAN_SEQUENCE_LENGTHS}
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "parquet"
            manifest = generate_synthetic_agg_dataset(
                config,
                output_dir,
                files=1,
                raw_rows_per_file=1,
                requests_per_agg=2,
                candidates_per_request=2,
                sequence_lengths=short_lengths,
                physical_column_count=300,
            )
            parquet_path = next(output_dir.glob("*.parquet"))
            self.assertEqual(len(pq.read_schema(parquet_path)), 300)
            self.assertEqual(manifest.projected_columns, 293)
            self.assertEqual(manifest.candidates, 4)
            self.assertGreater(manifest.projected_compressed_bytes, 0)

            train = replace(
                config.data.train,
                inputs=(str(output_dir),),
                reader=replace(
                    config.data.train.reader,
                    eager_schema_validation="all",
                ),
            )
            local_config = replace(
                config,
                data=replace(config.data, train=train),
            )
            table = next(iter(iter_flat_tables(local_config, "train")))

        self.assertEqual(table.num_rows, 4)
        self.assertEqual(
            table["search_id"].to_pylist(),
            [
                "synthetic-0-0-0",
                "synthetic-0-0-0",
                "synthetic-0-0-1",
                "synthetic-0-0-1",
            ],
        )
        self.assertEqual(
            [len(value) for value in table["impr_x_goods_id_hn"].to_pylist()],
            [2, 2, 2, 2],
        )

    def test_tuner_updates_the_bucket_matching_the_synthetic_workload(self) -> None:
        path = ROOT / "configs" / "mdl_rankmixer.yaml"
        lengths = _default_sequence_lengths(path)

        override, bucket = _recommended_yaml_override(path, 40, lengths)

        self.assertIsNotNone(bucket)
        self.assertEqual(bucket["workload_length"], 4308)
        self.assertEqual(bucket["max_length"], 6144)
        rendered = override["data"]["train"]["reader"]["length_buckets"]
        self.assertEqual(rendered[bucket["bucket_index"]]["batch_size"], 40)

    def test_hdfs_sensitivity_uses_projected_bytes_not_full_file_size(self) -> None:
        manifest = SyntheticAggManifest(
            output_dir="/tmp/synthetic",
            files=8,
            raw_rows_per_file=2,
            raw_rows=16,
            requests_per_agg=2,
            candidates_per_request=4,
            candidates=128,
            sequence_overlap=0.85,
            sequence_lengths_after_request_filter={"impr": 10},
            raw_sequence_lengths={"impr": 12},
            bag_lengths={"query_hash_hn": 2},
            physical_columns=630,
            projected_columns=293,
            arrow_bytes_per_file=100_000,
            parquet_file_bytes=800_000,
            projected_compressed_bytes=400_000,
            projected_compressed_bytes_per_candidate=3125.0,
            compression="gzip",
        )

        sensitivity = _hdfs_sensitivity(
            manifest,
            nproc_per_node=1,
            open_latency_ms=10.0,
            bandwidths_mib_s=(1024.0,),
        )
        self.assertEqual(len(sensitivity), 1)
        self.assertGreater(
            sensitivity[0]["estimated_samples_per_second_ceiling"],
            0.0,
        )
        self.assertLess(
            manifest.projected_compressed_bytes,
            manifest.parquet_file_bytes,
        )


if __name__ == "__main__":
    unittest.main()
