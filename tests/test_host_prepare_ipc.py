"""Host-prepare IPC mode selection (memfd vs share_memory)."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as thread_queue
import unittest
from unittest import mock

import torch

from src.dataloader import FeatureBatch, pin_feature_batch, privatize_shared_feature_batch
from src.train import (
    _HOST_PREPARE_SHARE_SHM_BYTES,
    _host_prepare_ipc_mode,
    _load_feature_batch_from_ipc,
    _memfd_handle_channel,
    _publish_memfd_payload,
    _share_feature_batch_for_ipc,
    _spill_feature_batch_for_ipc,
    _wait_for_host_prepare_terminal_ack,
)


def _pinned_memory_available() -> bool:
    try:
        torch.empty(1, pin_memory=True)
    except RuntimeError:
        return False
    return True


PINNED_MEMORY_AVAILABLE = _pinned_memory_available()


def _packed_test_batch(index: int = 0) -> FeatureBatch:
    packed = torch.arange(20, dtype=torch.int64) + index * 20
    return FeatureBatch(
        features={"x": packed[:16].view(4, 4)},
        labels=None,
        label_mask=None,
        scenario_id=packed[16:20],
        group_id=["a", "b", "c", "d"],
        _packed_buffers=(packed,),
    )


def _memfd_send_handle_child(
    queue: object,
    conn: object,
    parent_pid: int,
    batch_count: int,
) -> None:
    """Child helper for ``test_memfd_send_handle_roundtrip_across_spawn``."""

    try:
        for index in range(batch_count):
            payload, memfd = _spill_feature_batch_for_ipc(
                _packed_test_batch(index)
            )
            _publish_memfd_payload(queue, conn, parent_pid, payload, memfd)
        queue.put(None)
    finally:
        conn.close()


def _share_memory_child(queue: object, terminal_ack: object) -> None:
    """Publish one file-system shared batch and wait for consumer ownership."""

    import torch.multiprocessing as torch_mp

    torch_mp.set_sharing_strategy("file_system")
    try:
        queue.put(_share_feature_batch_for_ipc(_packed_test_batch()))
        queue.put(None)
        _wait_for_host_prepare_terminal_ack(terminal_ack)
    finally:
        terminal_ack.close()


class HostPrepareIpcModeTest(unittest.TestCase):
    def test_explicit_override(self) -> None:
        self.assertEqual(_host_prepare_ipc_mode({"MDL_HOST_PREPARE_IPC": "memfd"}), "memfd")
        self.assertEqual(_host_prepare_ipc_mode({"MDL_HOST_PREPARE_IPC": "share"}), "share")

    def test_auto_picks_share_when_shm_large(self) -> None:
        # Prod nodes have large /dev/shm: prefer zero-copy share for util.
        # Parent privatize + pinned pool prevents the old RSS ratchet.
        with mock.patch(
            "src.train._dev_shm_free_bytes",
            return_value=_HOST_PREPARE_SHARE_SHM_BYTES,
        ):
            self.assertEqual(_host_prepare_ipc_mode({}), "share")
            self.assertEqual(_host_prepare_ipc_mode({"MDL_HOST_PREPARE_IPC": "auto"}), "share")

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

    @unittest.skipUnless(
        PINNED_MEMORY_AVAILABLE,
        "pinned host allocation requires a usable CUDA driver",
    )
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

    def test_file_system_share_waits_for_parent_before_producer_exit(self) -> None:
        """The large-/dev/shm producer owns files until parent privatization."""

        import torch.multiprocessing as torch_mp

        previous_strategy = torch_mp.get_sharing_strategy()
        torch_mp.set_sharing_strategy("file_system")
        ctx = mp.get_context("spawn")
        metadata_queue = ctx.Queue(maxsize=2)
        child_ack, parent_ack = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_share_memory_child,
            args=(metadata_queue, child_ack),
        )
        proc.start()
        child_ack.close()
        try:
            shared = metadata_queue.get(timeout=2)
            self.assertIsInstance(shared, FeatureBatch)
            private = privatize_shared_feature_batch(shared)
            self.assertFalse(private._packed_buffers[0].is_shared())
            torch.testing.assert_close(
                private.features["x"],
                torch.arange(16, dtype=torch.int64).view(4, 4),
            )
            self.assertIsNone(metadata_queue.get(timeout=2))
            self.assertTrue(proc.is_alive())
            parent_ack.send_bytes(b"done")
            parent_ack.close()
            proc.join(timeout=10)
            self.assertEqual(proc.exitcode, 0)
        finally:
            try:
                parent_ack.close()
            except Exception:
                pass
            if proc.is_alive():
                proc.kill()
            proc.join(timeout=10)
            metadata_queue.close()
            metadata_queue.join_thread()
            torch_mp.set_sharing_strategy(previous_strategy)

    def test_memfd_send_handle_roundtrip_across_spawn(self) -> None:
        """FDs remain valid after the spawned producer has exited."""

        from multiprocessing.reduction import recv_handle

        ctx = mp.get_context("spawn")
        metadata_queue = ctx.Queue(maxsize=16)
        parent_recv, child_send = _memfd_handle_channel(ctx)
        parent_pid = os.getpid()
        batch_count = 8
        proc = ctx.Process(
            target=_memfd_send_handle_child,
            args=(metadata_queue, child_send, parent_pid, batch_count),
        )
        proc.start()
        child_send.close()
        try:
            # Reproduce the old failure condition: the resource_sharer owner
            # was gone before queued DupFd tokens were detached by the parent.
            proc.join(timeout=10)
            self.assertEqual(proc.exitcode, 0)
            for index in range(batch_count):
                payload = metadata_queue.get(timeout=2)
                memfd = int(recv_handle(parent_recv))
                loaded = _load_feature_batch_from_ipc(
                    payload, fd=memfd, pin_memory=False
                )
                torch.testing.assert_close(
                    loaded.features["x"],
                    (torch.arange(16, dtype=torch.int64) + index * 20).view(4, 4),
                )
                torch.testing.assert_close(
                    loaded.scenario_id,
                    torch.arange(16, 20, dtype=torch.int64) + index * 20,
                )
            self.assertIsNone(metadata_queue.get(timeout=2))
        finally:
            parent_recv.close()
            if proc.is_alive():
                proc.kill()
            proc.join(timeout=10)
            metadata_queue.close()
            metadata_queue.join_thread()

    def test_memfd_payload_never_uses_resource_sharer_token(self) -> None:
        """Regression: queued metadata must not require a later fd.detach()."""

        with mock.patch(
            "multiprocessing.reduction.DupFd",
            side_effect=AssertionError("resource_sharer must not be used"),
        ):
            payload, memfd = _spill_feature_batch_for_ipc(_packed_test_batch())
        try:
            self.assertNotIn("fd", payload)
            self.assertIn("buffers", payload)
        finally:
            os.close(memfd)

    def test_memfd_metadata_is_not_published_when_handle_send_fails(self) -> None:
        metadata_queue: thread_queue.Queue[object] = thread_queue.Queue()
        payload, memfd = _spill_feature_batch_for_ipc(_packed_test_batch())
        with mock.patch(
            "multiprocessing.reduction.send_handle",
            side_effect=OSError("send failed"),
        ), self.assertRaisesRegex(OSError, "send failed"):
            _publish_memfd_payload(
                metadata_queue, object(), os.getpid(), payload, memfd
            )
        self.assertTrue(metadata_queue.empty())
        with self.assertRaises(OSError):
            os.fstat(memfd)


if __name__ == "__main__":
    unittest.main()
