"""Recycled pinned-host buffer pool for host-prepare RSS control."""

from __future__ import annotations

import gc
import unittest

import torch

from src.dataloader import FeatureBatch
from src.train import (
    _PinnedHostBufferPool,
    _load_feature_batch_from_ipc,
    _spill_feature_batch_for_ipc,
)


def _pinned_memory_available() -> bool:
    try:
        torch.empty(1, pin_memory=True)
    except RuntimeError:
        return False
    return True


@unittest.skipUnless(
    _pinned_memory_available(),
    "pinned host allocation requires a usable CUDA driver",
)
class PinnedHostPoolTest(unittest.TestCase):
    def test_checkout_reuses_storage_after_lease_release(self) -> None:
        pool = _PinnedHostBufferPool(max_free_slots=2)
        views1, lease1 = pool.checkout([(torch.int64, 128), (torch.float32, 64)])
        ptr1 = views1[0].untyped_storage().data_ptr()
        lease1.release()
        del views1
        gc.collect()
        views2, lease2 = pool.checkout([(torch.int64, 96), (torch.float32, 32)])
        self.assertEqual(views2[0].untyped_storage().data_ptr(), ptr1)
        self.assertEqual(int(views2[0].numel()), 96)
        self.assertTrue(views2[0].is_pinned())
        lease2.release()

    def test_memfd_roundtrip_with_pool_keeps_lease(self) -> None:
        pool = _PinnedHostBufferPool(max_free_slots=2)
        # One packed buffer: features + scenario_id are views into it.
        packed = torch.arange(20, dtype=torch.int64)
        batch = FeatureBatch(
            features={"x": packed[:16].view(4, 4)},
            labels=None,
            label_mask=None,
            scenario_id=packed[16:20],
            group_id=["a", "b", "c", "d"],
            _packed_buffers=(packed,),
        )
        payload, memfd = _spill_feature_batch_for_ipc(batch)
        loaded = _load_feature_batch_from_ipc(
            payload, fd=memfd, pin_memory=True, pinned_pool=pool
        )
        self.assertTrue(loaded._packed_buffers[0].is_pinned())
        self.assertIsNotNone(loaded._keepalive)
        torch.testing.assert_close(loaded.features["x"], batch.features["x"])
        torch.testing.assert_close(loaded.scenario_id, batch.scenario_id)
        del loaded
        gc.collect()
        views, lease = pool.checkout([(torch.int64, 20)])
        self.assertTrue(views[0].is_pinned())
        lease.release()


if __name__ == "__main__":
    unittest.main()
