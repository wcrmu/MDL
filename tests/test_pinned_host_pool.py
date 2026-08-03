"""Recycled pinned-host buffer pool for host-prepare RSS control."""

from __future__ import annotations

import gc
import unittest
from unittest import mock

import torch

from src.dataloader import FeatureBatch
from src.train import (
    _PinnedHostBufferPool,
    _load_feature_batch_from_ipc,
    _pinned_pool_max_slot_bytes_from_env,
    _spill_feature_batch_for_ipc,
)


def _pinned_memory_available() -> bool:
    try:
        torch.empty(1, pin_memory=True)
    except RuntimeError:
        return False
    return True


PINNED_MEMORY_AVAILABLE = _pinned_memory_available()

_REAL_EMPTY = torch.empty


def _test_empty(*args, pin_memory: bool = False, **kwargs):
    # Prefer real pinned alloc when the driver allows it; otherwise fall back to
    # ordinary CPU empty so sizing / shrink logic remains testable offline.
    if pin_memory and not PINNED_MEMORY_AVAILABLE:
        return _REAL_EMPTY(*args, **kwargs)
    return _REAL_EMPTY(*args, pin_memory=pin_memory, **kwargs)


class PinnedHostPoolEnvTest(unittest.TestCase):
    def test_env_max_slot_bytes_parser(self) -> None:
        self.assertIsNone(_pinned_pool_max_slot_bytes_from_env({}))
        self.assertEqual(
            _pinned_pool_max_slot_bytes_from_env(
                {"MDL_PINNED_POOL_MAX_SLOT_BYTES": "1048576"}
            ),
            1048576,
        )
        with self.assertRaises(ValueError):
            _pinned_pool_max_slot_bytes_from_env(
                {"MDL_PINNED_POOL_MAX_SLOT_BYTES": "0"}
            )


@mock.patch("src.train.torch.empty", side_effect=_test_empty)
class PinnedHostPoolTest(unittest.TestCase):
    def test_checkout_reuses_storage_after_lease_release(
        self, _empty: mock.MagicMock
    ) -> None:
        del _empty
        pool = _PinnedHostBufferPool(max_free_slots=2)
        views1, lease1 = pool.checkout([(torch.int64, 128), (torch.float32, 64)])
        ptr1 = views1[0].untyped_storage().data_ptr()
        lease1.release()
        del views1
        gc.collect()
        views2, lease2 = pool.checkout([(torch.int64, 96), (torch.float32, 32)])
        self.assertEqual(views2[0].untyped_storage().data_ptr(), ptr1)
        self.assertEqual(int(views2[0].numel()), 96)
        lease2.release()

    def test_release_shrinks_spike_after_sliding_window(
        self, _empty: mock.MagicMock
    ) -> None:
        del _empty
        pool = _PinnedHostBufferPool(
            max_free_slots=2,
            recent_window=8,
            shrink_factor=2.0,
        )
        _, spike = pool.checkout([(torch.int64, 10_000)])
        spike.release()
        self.assertEqual(len(pool._free), 1)
        self.assertEqual(int(pool._free[0][torch.int64].numel()), 10_000 * 9 // 8)

        for _ in range(8):
            views, lease = pool.checkout([(torch.int64, 100)])
            lease.release()
            del views

        self.assertGreaterEqual(pool._shrinks, 1)
        if pool._free:
            slot = pool._free[0]
            self.assertTrue(
                torch.int64 not in slot or int(slot[torch.int64].numel()) <= 200
            )

    def test_max_slot_bytes_skips_headroom_and_trims_idle(
        self, _empty: mock.MagicMock
    ) -> None:
        del _empty
        pool = _PinnedHostBufferPool(
            max_free_slots=2,
            recent_window=64,
            shrink_factor=100.0,  # disable sliding shrink; rely on byte cap
            max_slot_bytes=64,  # 8 int64 elements
        )
        views, lease = pool.checkout([(torch.int64, 16)])
        storage_numel = int(views[0].untyped_storage().size()) // 8
        self.assertEqual(storage_numel, 16)
        lease.release()
        del views
        self.assertGreaterEqual(pool._shrinks, 1)
        self.assertTrue(not pool._free or torch.int64 not in pool._free[0])


@unittest.skipUnless(
    PINNED_MEMORY_AVAILABLE,
    "pinned host allocation requires a usable CUDA driver",
)
class PinnedHostPoolMemfdTest(unittest.TestCase):
    def test_memfd_roundtrip_with_pool_keeps_lease(self) -> None:
        pool = _PinnedHostBufferPool(max_free_slots=2)
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
