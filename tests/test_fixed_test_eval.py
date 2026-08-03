from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.config import load_app_config
from src.main import _cmd_train, _load_config, build_arg_parser
from src.train import (
    DistributedContext,
    _evenly_spaced_file_uris,
    _prepare_fixed_test_eval,
)


ROOT = Path(__file__).resolve().parents[1]


class FixedTestEvalCliTest(unittest.TestCase):
    def test_train_requires_test_hours_or_a_derivable_train_window(self) -> None:
        args = build_arg_parser().parse_args(
            ["train", "--config", "configs/rankmixer.yaml"]
        )

        with patch("src.main._load_config") as load_config:
            with self.assertRaisesRegex(
                ValueError,
                "requires an explicit test hour window",
            ):
                _cmd_train(args)

        load_config.assert_not_called()

    def test_train_defaults_test_to_calendar_day_after_train_end(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "train",
                "--config",
                str(ROOT / "configs" / "rankmixer.yaml"),
                "--data-base-dir",
                "/datasets/hourly",
                "--train-start-hour",
                "2026-07-22-22",
                "--train-end-hour",
                "2026-07-29-22",
            ]
        )

        config = _load_config(args)

        assert config.data.test is not None
        self.assertEqual(len(config.data.test.inputs), 24)
        self.assertEqual(
            config.data.test.inputs[0],
            "/datasets/hourly/pt=2026-07-30/hr=00",
        )
        self.assertEqual(
            config.data.test.inputs[-1],
            "/datasets/hourly/pt=2026-07-30/hr=23",
        )

    def test_test_window_and_eval_size_overrides_are_applied(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "train",
                "--config",
                str(ROOT / "configs" / "rankmixer.yaml"),
                "--data-base-dir",
                "/datasets/hourly",
                "--test-start-hour",
                "2026-07-31-22",
                "--test-end-hour",
                "2026-08-01-00",
                "--eval-every-steps",
                "2000",
                "--test-files-per-rank",
                "6",
            ]
        )

        config = _load_config(args)

        self.assertEqual(
            config.data.test.inputs,
            (
                "/datasets/hourly/pt=2026-07-31/hr=22",
                "/datasets/hourly/pt=2026-07-31/hr=23",
            ),
        )
        self.assertEqual(config.training.fixed_test_eval.every_steps, 2000)
        self.assertEqual(config.training.fixed_test_eval.files_per_rank, 6)

    def test_train_rejects_overlapping_train_and_test_partitions(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "train",
                "--config",
                str(ROOT / "configs" / "rankmixer.yaml"),
                "--data-base-dir",
                "/datasets/hourly",
                "--train-start-hour",
                "2026-07-31-22",
                "--train-end-hour",
                "2026-07-31-23",
                "--test-start-hour",
                "2026-07-31-22",
                "--test-end-hour",
                "2026-07-31-23",
            ]
        )

        with patch("src.main.train_mdl") as train:
            with self.assertRaisesRegex(ValueError, "must be disjoint"):
                _cmd_train(args)

        train.assert_not_called()


class FixedTestManifestTest(unittest.TestCase):
    def test_even_subset_is_deterministic_and_spans_the_window(self) -> None:
        uris = [f"part-{index:03d}.parquet" for index in range(100)]

        selected = _evenly_spaced_file_uris(uris, 8)

        self.assertEqual(len(selected), 8)
        self.assertEqual(selected[0], uris[0])
        self.assertEqual(selected[-1], uris[-1])
        self.assertEqual(selected, _evenly_spaced_file_uris(uris, 8))
        self.assertEqual(len(set(selected)), len(selected))

    def test_prepare_freezes_four_files_per_rank_and_uses_train_batches(
        self,
    ) -> None:
        config = load_app_config(ROOT / "configs" / "rankmixer.yaml")
        assert config.data.test is not None
        config = replace(
            config,
            data=replace(
                config.data,
                test=replace(config.data.test, inputs=("/test/hour",)),
            ),
        )
        refs = [
            SimpleNamespace(canonical_uri=f"/test/hour/part-{index:03d}.parquet")
            for index in range(100)
        ]
        context = DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=2,
            device=torch.device("cpu"),
        )

        with patch("src.train.discover_parquet_inputs", return_value=refs):
            prepared = _prepare_fixed_test_eval(config, context)

        assert prepared.data.test is not None
        self.assertEqual(config.training.fixed_test_eval.files_per_rank, 4)
        self.assertEqual(len(prepared.data.test.inputs), 8)
        self.assertEqual(prepared.data.test.inputs[0], refs[0].canonical_uri)
        self.assertEqual(prepared.data.test.inputs[-1], refs[-1].canonical_uri)
        self.assertEqual(
            prepared.data.test.reader.length_buckets,
            prepared.data.train.reader.length_buckets,
        )
        self.assertEqual(prepared.data.test.reader.device_prefetch_batches, 1)
        self.assertEqual(prepared.data.test.reader.shuffle_buffer_rows, 0)
        self.assertFalse(prepared.data.test.prediction_keys)


if __name__ == "__main__":
    unittest.main()
