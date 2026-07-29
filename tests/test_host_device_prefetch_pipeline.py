"""Infra: host_prepare_prefetch + device_prefetch_batches two-stage pipeline."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.config import LengthBucketConfig, ReaderConfig, load_app_config
from src.train import iter_feature_batches


ROOT = Path(__file__).resolve().parents[1]


class HostDevicePrefetchPipelineTest(unittest.TestCase):
    def test_reader_allows_both_prefetch_stages(self) -> None:
        reader = ReaderConfig(
            length_buckets=(LengthBucketConfig(max_length=None, batch_size=8),),
            host_prepare_prefetch=3,
            device_prefetch_batches=2,
            pin_memory=True,
            coalesce_pinned_tensors=True,
        )
        reader.validate()

    def test_iter_feature_batches_uses_process_host_when_device_prefetch_set(
        self,
    ) -> None:
        config = load_app_config(ROOT / "configs" / "reference" / "default.yaml")
        train = replace(
            config.data.train,
            reader=replace(
                config.data.train.reader,
                host_prepare_prefetch=2,
                device_prefetch_batches=1,
                pin_memory=True,
                coalesce_pinned_tensors=True,
                adapter_workers=0,
            ),
        )
        config = replace(config, data=replace(config.data, train=train))
        with patch("src.train._ProcessHostPrepareIterator") as process_cls:
            process_cls.return_value = iter(())
            list(
                iter_feature_batches(
                    config,
                    "train",
                    vocab_maps={},
                    require_labels=False,
                    pin_memory=True,
                )
            )
        process_cls.assert_called_once()
        kwargs = process_cls.call_args.kwargs
        self.assertEqual(kwargs["queue_size"], 2)


if __name__ == "__main__":
    unittest.main()
