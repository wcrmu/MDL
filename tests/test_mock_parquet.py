from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pyarrow.parquet as pq

from scripts.generate_mock_parquet import generate_mock_parquet_dataset
from src.config import load_app_config
from src.dataloader import iter_flat_tables


ROOT = Path(__file__).resolve().parents[1]


class MockParquetGenerationTest(unittest.TestCase):
    def test_generated_files_preserve_all_anonymized_physical_slots(
        self,
    ) -> None:
        config_path = ROOT / "configs" / "rankmixer.yaml"
        config = load_app_config(config_path)
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "mock"
            manifest = generate_mock_parquet_dataset(
                config,
                ROOT / "sample_row_mock_json",
                output,
                files=2,
                rows_per_file=3,
                row_group_size=2,
                compression="zstd",
                config_path=config_path,
            )

            self.assertEqual(manifest.raw_rows, 6)
            self.assertEqual(manifest.requests_per_raw_row, 5)
            self.assertEqual(manifest.candidates_per_raw_row, 7)
            self.assertEqual(manifest.candidate_rows, 42)
            self.assertEqual(manifest.sequence_tokens_per_raw_row["impr"], 161)
            self.assertEqual(manifest.sequence_tokens_per_raw_row["clk_long"], 78)
            self.assertEqual(manifest.sequence_tokens_per_raw_row["view_long"], 75)
            self.assertEqual(manifest.physical_columns, 293)
            self.assertEqual(manifest.row_groups, 4)
            self.assertEqual(
                json.loads((output / "manifest.json").read_text())["candidate_rows"],
                42,
            )
            for path in sorted(output.glob("*.parquet")):
                parquet_file = pq.ParquetFile(path)
                metadata = parquet_file.metadata
                self.assertEqual(metadata.num_rows, 3)
                self.assertEqual(metadata.num_row_groups, 2)
                raw = parquet_file.read_row_group(
                    0,
                    columns=[
                        "context_indices",
                        "target_indices",
                        "goods_id_hn",
                        "impr_x_goods_id_hn",
                        "impr_x_time",
                    ],
                ).slice(0, 1).to_pylist()[0]
                self.assertEqual(len(raw["context_indices"]), 5)
                self.assertEqual(len(raw["target_indices"]), 7)
                self.assertEqual(len(raw["goods_id_hn"]), 7)
                self.assertEqual(len(raw["impr_x_goods_id_hn"]), 161)
                self.assertEqual(len(raw["impr_x_time"]), 161)
                self.assertTrue(
                    all(cell[0] != 0 for cell in raw["goods_id_hn"])
                )
                self.assertTrue(
                    all(value != 0 for value in raw["impr_x_goods_id_hn"])
                )

            reader = replace(
                config.data.train.reader,
                prefetch_batches=0,
                scanner_batch_rows=2,
            )
            train = replace(
                config.data.train,
                inputs=(str(output),),
                reader=reader,
            )
            config = replace(config, data=replace(config.data, train=train))
            tables = list(iter_flat_tables(config, "train"))
            self.assertEqual(sum(table.num_rows for table in tables), 42)
            search_ids = [
                value
                for table in tables
                for value in table["search_id"].to_pylist()
            ]
            self.assertEqual(len(set(search_ids)), 30)
            self.assertNotIn("0", search_ids)
            self.assertTrue(
                all(
                    len(sequence) > 0
                    for table in tables
                    for sequence in table["impr_x_goods_id_hn"].to_pylist()
                )
            )

            parallel_reader = replace(reader, adapter_workers=2)
            parallel_train = replace(train, reader=parallel_reader)
            parallel_config = replace(
                config,
                data=replace(config.data, train=parallel_train),
            )
            parallel_tables = list(iter_flat_tables(parallel_config, "train"))
            self.assertEqual(
                sum(table.num_rows for table in parallel_tables),
                42,
            )


if __name__ == "__main__":
    unittest.main()
