"""Host-prepare child startup/idle watchdog and process-group kill."""

from __future__ import annotations

import errno
import queue
import time
import unittest
from unittest.mock import MagicMock, patch

from src.config import LengthBucketConfig, ReaderConfig
from src.dataloader import (
    FeatureBatch,
    RemoteIoStallError,
    note_io_progress,
    set_io_progress_hook,
)
from src.train import (
    _ProcessHostPrepareIterator,
    _is_host_prepare_ipc_teardown_error,
    _terminate_process_group,
)


class HostPrepareWatchdogTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_io_progress_hook(None)

    def test_reader_accepts_host_prepare_timeouts(self) -> None:
        reader = ReaderConfig(
            length_buckets=(LengthBucketConfig(max_length=None, batch_size=8),),
            host_prepare_prefetch=1,
            host_prepare_startup_timeout_sec=12.0,
            host_prepare_idle_timeout_sec=34.0,
        )
        reader.validate()
        self.assertEqual(reader.host_prepare_startup_timeout_sec, 12.0)
        self.assertEqual(reader.host_prepare_idle_timeout_sec, 34.0)

    def test_terminate_process_group_escalates_to_kill(self) -> None:
        process = MagicMock()
        # terminate check → SIGTERM wait → SIGKILL wait sees exit
        process.is_alive.side_effect = [True, True, False]
        process.pid = 4242
        with patch("src.train.os.killpg") as killpg, patch(
            "src.train.perf_counter", side_effect=[0.0, 10.0, 10.0]
        ):
            _terminate_process_group(
                process, grace_sec=0.01, kill_grace_sec=0.01, label="test"
            )
        self.assertEqual(killpg.call_count, 2)

    def test_terminate_process_group_abandons_after_sigkill_timeout(self) -> None:
        process = MagicMock()
        process.is_alive.return_value = True
        process.pid = 4242
        with patch("src.train.os.killpg"), patch(
            "src.train.perf_counter", side_effect=[0.0, 10.0, 10.0, 20.0]
        ), patch("builtins.print") as printed:
            _terminate_process_group(
                process, grace_sec=0.01, kill_grace_sec=0.01, label="test"
            )
        self.assertTrue(
            any("abandoning" in str(call) for call in printed.call_args_list)
        )

    def test_startup_timeout_aborts_rank_without_respawn(self) -> None:
        iterator = _ProcessHostPrepareIterator.__new__(_ProcessHostPrepareIterator)
        iterator._pin_memory = False
        iterator._ipc_mode = "memfd"
        iterator._closed = False
        iterator._startup_timeout_sec = 0.05
        iterator._idle_timeout_sec = None
        iterator._started_at = 0.0
        iterator._last_progress_at = 0.0
        iterator._received_item = False
        iterator._progress_mtime = None
        iterator._queue = MagicMock()
        iterator._queue.get.side_effect = queue.Empty
        iterator._process = MagicMock()
        iterator._process.is_alive.return_value = True
        iterator._process.pid = 4242

        aborted: list[BaseException] = []

        def _abort(error: BaseException) -> None:
            aborted.append(error)
            raise SystemExit(70)

        with patch("src.train.perf_counter", side_effect=[0.0, 1.0]), patch(
            "src.train.abort_rank_for_remote_io_stall",
            side_effect=_abort,
        ), patch("builtins.print") as print_message, patch(
            "src.train._close_process_queue"
        ) as close_queue, patch("src.train.os.killpg") as killpg:
            with self.assertRaises(SystemExit) as raised:
                next(iterator)
        self.assertEqual(raised.exception.code, 70)
        self.assertEqual(len(aborted), 1)
        self.assertIsInstance(aborted[0], RemoteIoStallError)
        self.assertIn("startup exceeded", str(aborted[0]))
        print_message.assert_any_call(str(aborted[0]), flush=True)
        close_queue.assert_called_once()
        killpg.assert_called()
        self.assertTrue(iterator._closed)

    def test_startup_heartbeat_keeps_watchdog_alive(self) -> None:
        iterator = _ProcessHostPrepareIterator.__new__(_ProcessHostPrepareIterator)
        iterator._pin_memory = False
        iterator._ipc_mode = "memfd"
        iterator._closed = False
        iterator._startup_timeout_sec = 0.2
        iterator._idle_timeout_sec = None
        iterator._started_at = 0.0
        iterator._last_progress_at = 0.0
        iterator._received_item = False
        progress = MagicMock()
        progress.value = 1_000.0
        iterator._progress_mtime = progress
        iterator._queue = MagicMock()
        iterator._process = MagicMock()
        iterator._process.is_alive.return_value = True
        batch = MagicMock(spec=FeatureBatch)
        iterator._queue.get.side_effect = [queue.Empty, batch]

        # Wall clock stays near the fresh heartbeat while perf time is long past
        # the startup timeout measured from process start.
        with patch("src.train.perf_counter", side_effect=[10.0, 10.0]), patch(
            "src.train.time", side_effect=[1_000.05, 1_000.05]
        ), patch(
            "src.train.privatize_shared_feature_batch",
            side_effect=lambda item: item,
        ):
            got = next(iterator)
        self.assertIs(got, batch)
        self.assertTrue(iterator._received_item)

    def test_note_io_progress_invokes_hook(self) -> None:
        seen: list[int] = []
        set_io_progress_hook(lambda: seen.append(1))
        note_io_progress()
        self.assertEqual(seen, [1])

    def test_io_progress_pulses_during_long_work(self) -> None:
        from src.dataloader import io_progress_pulses

        seen: list[int] = []
        set_io_progress_hook(lambda: seen.append(1))
        with io_progress_pulses(0.05):
            time.sleep(0.18)
        # start pulse + at least one interval pulse + final pulse
        self.assertGreaterEqual(len(seen), 3)

    def test_close_discards_queue_without_unbounded_join(self) -> None:
        iterator = _ProcessHostPrepareIterator.__new__(_ProcessHostPrepareIterator)
        iterator._closed = False
        iterator._process = MagicMock()
        iterator._process.is_alive.return_value = False
        iterator._queue = MagicMock()

        iterator.close()

        iterator._queue.cancel_join_thread.assert_called_once_with()
        iterator._queue.close.assert_called_once_with()
        iterator._queue._reader.close.assert_called_once_with()
        iterator._queue._writer.close.assert_called_once_with()
        iterator._queue.join_thread.assert_not_called()

    def test_teardown_error_helper_narrows_oserror(self) -> None:
        self.assertTrue(_is_host_prepare_ipc_teardown_error(BrokenPipeError()))
        self.assertTrue(_is_host_prepare_ipc_teardown_error(EOFError()))
        self.assertTrue(
            _is_host_prepare_ipc_teardown_error(OSError(errno.EPIPE, "Broken pipe"))
        )
        # Mid-run HDFS / shm failures must surface to the parent queue.
        self.assertFalse(
            _is_host_prepare_ipc_teardown_error(OSError("Filesystem closed"))
        )
        self.assertFalse(
            _is_host_prepare_ipc_teardown_error(OSError(errno.ENOMEM, "Cannot allocate"))
        )

    def test_dead_child_without_terminal_is_remote_io_stall(self) -> None:
        iterator = _ProcessHostPrepareIterator.__new__(_ProcessHostPrepareIterator)
        iterator._pin_memory = False
        iterator._ipc_mode = "memfd"
        iterator._closed = False
        iterator._startup_timeout_sec = None
        iterator._idle_timeout_sec = None
        iterator._started_at = 0.0
        iterator._last_progress_at = 0.0
        iterator._received_item = True
        iterator._progress_mtime = None
        iterator._queue = MagicMock()
        iterator._queue.get.side_effect = queue.Empty
        iterator._queue.empty.return_value = True
        iterator._process = MagicMock()
        iterator._process.is_alive.return_value = False
        iterator._process.exitcode = -9

        with patch.object(iterator, "close") as close, self.assertRaises(
            RemoteIoStallError
        ) as raised:
            next(iterator)
        self.assertIn("exitcode=-9", str(raised.exception))
        self.assertIn("without a terminal queue item", str(raised.exception))
        close.assert_called_once()

    def test_close_unblocks_queue_before_terminate(self) -> None:
        iterator = _ProcessHostPrepareIterator.__new__(_ProcessHostPrepareIterator)
        iterator._closed = False
        iterator._process = MagicMock()
        iterator._process.is_alive.return_value = True
        iterator._process.pid = 4242
        iterator._queue = MagicMock()
        order: list[str] = []

        def _close_queue(_queue: object) -> None:
            order.append("queue")

        def _terminate(_process: object, **_kwargs: object) -> None:
            order.append("terminate")

        with patch("src.train._close_process_queue", side_effect=_close_queue), patch(
            "src.train._terminate_process_group", side_effect=_terminate
        ):
            iterator.close()

        self.assertEqual(order, ["queue", "terminate"])
        self.assertTrue(iterator._closed)


if __name__ == "__main__":
    unittest.main()
