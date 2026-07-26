"""Host-prepare IPC mode selection (memfd vs share_memory)."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from src.dataloader import FeatureBatch
from src.train import (
    _HOST_PREPARE_SHARE_SHM_BYTES,
    _host_prepare_ipc_mode,
    _share_feature_batch_for_ipc,
)


class HostPrepareIpcModeTest(unittest.TestCase):
    def test_explicit_override(self) -> None:
        self.assertEqual(_host_prepare_ipc_mode({"MDL_HOST_PREPARE_IPC": "memfd"}), "memfd")
        self.assertEqual(_host_prepare_ipc_mode({"MDL_HOST_PREPARE_IPC": "share"}), "share")

    def test_auto_picks_share_when_shm_large(self) -> None:
        with mock.patch(
            "src.train._dev_shm_free_bytes",
            return_value=_HOST_PREPARE_SHARE_SHM_BYTES,
        ):
            self.assertEqual(_host_prepare_ipc_mode({}), "share")

    def test_auto_picks_memfd_when_shm_tiny(self) -> None:
        with mock.patch("src.train._dev_shm_free_bytes", return_value=64 * 1024 * 1024):
            self.assertEqual(_host_prepare_ipc_mode({}), "memfd")

    def test_share_feature_batch_marks_storage_shared(self) -> None:
        base = torch.arange(8, dtype=torch.int64)
        batch = FeatureBatch(
            features={"x": base.view(2, 4)},
            labels=None,
            label_mask=None,
            scenario_id=torch.zeros(2, dtype=torch.int64),
            group_id=["a", "b"],
            _packed_buffers=(base,),
        )
        shared = _share_feature_batch_for_ipc(batch)
        self.assertTrue(shared._packed_buffers[0].is_shared())
        self.assertTrue(shared.scenario_id.is_shared())


if __name__ == "__main__":
    unittest.main()
