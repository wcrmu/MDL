from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

import pyarrow as pa

from src.config import LengthBucketConfig, load_app_config
from src.dataloader import _ByteBudget
from src.train import _iter_batch_tables


class LengthBucketTest(unittest.TestCase):
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


class ByteBudgetTest(unittest.TestCase):
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
