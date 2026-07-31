"""Host-prepare IPC mode selection (memfd vs share_memory)."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from src.dataloader import FeatureBatch, pin_feature_batch, privatize_shared_feature_batch
from src.train import (
    _HOST_PREPARE_SHARE_SHM_BYTES,
    _host_prepare_ipc_mode,
    _share_feature_batch_for_ipc,
)


class HostPrepareIpcModeTest(unittest.TestCase):
    def test_explicit_override(self) -> None:
        self.assertEqual(_host_prepare_ipc_mode({"MDL_HOST_PREPARE_IPC": "memfd"}), "memfd")
        self.assertEqual(_host_prepare_ipc_mode({"MDL_HOST_PREPARE_IPC": "share"}), "share")

    def test_auto_defaults_to_memfd_even_when_shm_large(self) -> None:
        # share is opt-in: automatic share selection previously ratcheted RSS.
        with mock.patch(
            "src.train._dev_shm_free_bytes",
            return_value=_HOST_PREPARE_SHARE_SHM_BYTES,
        ):
            self.assertEqual(_host_prepare_ipc_mode({}), "memfd")
            self.assertEqual(_host_prepare_ipc_mode({"MDL_HOST_PREPARE_IPC": "auto"}), "memfd")

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

    def test_pin_feature_batch_clones_shared_buffers(self) -> None:
        # Production order: pin then share_memory_. After share, torch reports
        # the storage as shared (and typically no longer pinned). Parent must
        # clone into private pinned pages so /dev/shm IPC files can unlink.
        base = torch.arange(8, dtype=torch.int64).pin_memory().share_memory_()
        batch = FeatureBatch(
            features={"x": base.view(2, 4)},
            labels=None,
            label_mask=None,
            scenario_id=base[:2],
            group_id=["a", "b"],
            _packed_buffers=(base,),
        )
        self.assertTrue(batch._packed_buffers[0].is_shared())
        privatized = pin_feature_batch(batch, coalesce_tensors=False)
        self.assertTrue(privatized._packed_buffers[0].is_pinned())
        self.assertFalse(privatized._packed_buffers[0].is_shared())
        torch.testing.assert_close(
            privatized.features["x"],
            batch.features["x"],
        )

    def test_privatize_shared_feature_batch_drops_share_memory(self) -> None:
        base = torch.arange(8, dtype=torch.int64).share_memory_()
        batch = FeatureBatch(
            features={"x": base.view(2, 4)},
            labels=None,
            label_mask=None,
            scenario_id=base[:2],
            group_id=["a", "b"],
            _packed_buffers=(base,),
        )
        private = privatize_shared_feature_batch(batch)
        self.assertFalse(private._packed_buffers[0].is_shared())
        torch.testing.assert_close(private.features["x"], batch.features["x"])


if __name__ == "__main__":
    unittest.main()
