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


class HostHeapTuningTest(unittest.TestCase):
    """glibc arena capping and heap trimming keep freed memory from staying charged."""

    def test_helpers_report_success_on_glibc(self) -> None:
        import platform

        from src.dataloader import _glibc, limit_malloc_arenas, trim_process_heap

        capped = limit_malloc_arenas(2)
        trimmed = trim_process_heap()
        if _glibc() is None:
            self.assertFalse(capped)
            self.assertFalse(trimmed)
            return
        self.assertTrue(capped, f"mallopt failed on {platform.libc_ver()}")
        self.assertTrue(trimmed)

    def test_helpers_are_inert_without_glibc(self) -> None:
        from src import dataloader

        with mock.patch.object(dataloader, "_glibc", return_value=None):
            self.assertFalse(dataloader.limit_malloc_arenas())
            self.assertFalse(dataloader.trim_process_heap())


class ArangeCacheTest(unittest.TestCase):
    """The shared aranges must cost the max length, not the sum of lengths."""

    def setUp(self) -> None:
        from src import dataloader

        self.dataloader = dataloader
        self._saved = (dataloader._ARANGE_BUFFER, dataloader._NP_ARANGE_BUFFER)
        dataloader._ARANGE_BUFFER = None
        dataloader._NP_ARANGE_BUFFER = None

    def tearDown(self) -> None:
        self.dataloader._ARANGE_BUFFER, self.dataloader._NP_ARANGE_BUFFER = self._saved

    def test_values_match_a_fresh_arange(self) -> None:
        import numpy as np

        for length in (0, 1, 7, 4096, 33, 8191):
            with self.subTest(length=length):
                torch.testing.assert_close(
                    self.dataloader._cached_arange(length),
                    torch.arange(length, dtype=torch.long),
                )
                np.testing.assert_array_equal(
                    self.dataloader._cached_np_arange(length),
                    np.arange(length, dtype=np.int64),
                )

    def test_many_distinct_lengths_keep_one_buffer(self) -> None:
        # Mirrors _build_abs_window_gather_plan asking for one arange per
        # unique request: thousands of distinct window totals per run.
        largest = 0
        for length in range(1, 4000, 7):
            self.dataloader._cached_arange(length)
            self.dataloader._cached_np_arange(length)
            largest = length

        torch_numel = self.dataloader._ARANGE_BUFFER.numel()
        numpy_numel = int(self.dataloader._NP_ARANGE_BUFFER.shape[0])
        # Doubling growth, so at most 2x the largest length ever requested.
        self.assertLessEqual(torch_numel, 2 * max(largest, 1024))
        self.assertLessEqual(numpy_numel, 2 * max(largest, 1024))
        self.assertGreaterEqual(torch_numel, largest)
        self.assertGreaterEqual(numpy_numel, largest)

    def test_growth_does_not_invalidate_earlier_views(self) -> None:
        small = self.dataloader._cached_np_arange(16).copy()
        held = self.dataloader._cached_np_arange(16)
        self.dataloader._cached_np_arange(1 << 16)
        import numpy as np

        np.testing.assert_array_equal(held, small)


class PerFileLockRegistryTest(unittest.TestCase):
    def test_registry_drops_keys_once_no_lock_is_held(self) -> None:
        from src.dataloader import PerFileLock

        keys = [f"hdfs://ns/part-{index:05d}.parquet" for index in range(200)]
        for key in keys:
            with PerFileLock(key, enabled=True):
                pass
        gc.collect()
        self.assertEqual(
            [key for key in keys if key in PerFileLock._thread_locks],
            [],
            "one RLock plus its HDFS path per file touched would grow all job",
        )

    def test_live_holders_share_one_lock(self) -> None:
        from src.dataloader import PerFileLock

        key = "hdfs://ns/shared.parquet"
        first = PerFileLock(key, enabled=True)
        second = PerFileLock(key, enabled=True)
        self.assertIs(first._thread_lock, second._thread_lock)
        self.assertIn(key, PerFileLock._thread_locks)

        del first, second
        gc.collect()
        self.assertNotIn(key, PerFileLock._thread_locks)


if __name__ == "__main__":
    unittest.main()
