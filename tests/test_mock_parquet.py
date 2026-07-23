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
    def test_generated_files_preserve_physical_shape_and_load_as_live_candidates(
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
            self.assertEqual(manifest.candidate_rows, 12)
            self.assertEqual(manifest.physical_columns, 293)
            self.assertEqual(manifest.row_groups, 4)
            self.assertEqual(
                json.loads((output / "manifest.json").read_text())["candidate_rows"],
                12,
            )
            for path in sorted(output.glob("*.parquet")):
                metadata = pq.ParquetFile(path).metadata
                self.assertEqual(metadata.num_rows, 3)
                self.assertEqual(metadata.num_row_groups, 2)

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
            self.assertEqual(sum(table.num_rows for table in tables), 12)
            search_ids = [
                value
                for table in tables
                for value in table["search_id"].to_pylist()
            ]
            self.assertEqual(len(search_ids), len(set(search_ids)))
            self.assertNotIn("0", search_ids)
            self.assertTrue(
                all(
                    len(sequence) <= 2
                    for table in tables
                    for sequence in table["impr_x_goods_id_hn"].to_pylist()
                )
            )


if __name__ == "__main__":
    unittest.main()
