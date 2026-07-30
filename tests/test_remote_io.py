"""Tests for HDFS/viewfs remote IO resilience helpers."""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from src.config import ReaderConfig
from src.dataloader import (
    PerFileLock,
    RemoteIoPolicy,
    RemoteIoTimeoutError,
    apply_worker_stagger,
    call_with_timeout,
    close_hdfs_native_file,
    invalidate_thread_local_hdfs_filesystem,
    is_poisoned_iterator_error,
    is_retryable_remote_error,
    iter_parquet_record_batches,
    open_parquet_via_native,
    retry_with_backoff,
    scaled_hdfs_prefetch_workers,
    thread_local_hdfs_filesystem,
)


def _remote_policy(**overrides) -> RemoteIoPolicy:
    values = dict(
        enabled=True,
        op_timeout=1.0,
        open_timeout=1.0,
        retry_count=0,
        retry_base_sec=0.01,
        file_lock=False,
        on_failure="fail",
        worker_stagger_sec=0.0,
        pre_buffer=True,
        close_timeout=0.5,
    )
    values.update(overrides)
    return RemoteIoPolicy(**values)


class RemoteIoHelperTest(unittest.TestCase):
    def test_timeout_raises_when_call_hangs(self) -> None:
        def hang() -> None:
            time.sleep(1.0)

        with self.assertRaises(RemoteIoTimeoutError):
            call_with_timeout(hang, 0.05, description="hang")

    def test_retry_with_backoff_eventually_succeeds(self) -> None:
        attempts = {"n": 0}

        def flaky() -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise OSError("Filesystem closed")
            return "ok"

        self.assertEqual(
            retry_with_backoff(
                flaky,
                retries=5,
                base_sec=0.01,
                description="flaky",
            ),
            "ok",
        )
        self.assertEqual(attempts["n"], 3)

    def test_filesystem_closed_is_retryable(self) -> None:
        self.assertTrue(is_retryable_remote_error(RuntimeError("Filesystem closed")))
        self.assertFalse(is_retryable_remote_error(ValueError("bad schema")))
        # Poisoned generators must not be retried in place via run_remote_op.
        self.assertTrue(
            is_poisoned_iterator_error(ValueError("generator already executing"))
        )
        self.assertFalse(
            is_retryable_remote_error(ValueError("generator already executing"))
        )

    def test_per_file_lock_serializes_threads(self) -> None:
        hold = threading.Event()
        order: list[str] = []

        def first() -> None:
            with PerFileLock("hdfs://ns/path.parquet", enabled=True):
                order.append("a-enter")
                hold.wait(timeout=2)
                order.append("a-exit")

        def second() -> None:
            with PerFileLock("hdfs://ns/path.parquet", enabled=True):
                order.append("b-enter")
                order.append("b-exit")

        first_thread = threading.Thread(target=first)
        second_thread = threading.Thread(target=second)
        first_thread.start()
        time.sleep(0.05)
        second_thread.start()
        time.sleep(0.05)
        self.assertEqual(order, ["a-enter"])
        hold.set()
        first_thread.join(timeout=5)
        second_thread.join(timeout=5)
        self.assertEqual(order, ["a-enter", "a-exit", "b-enter", "b-exit"])

    def test_worker_stagger_sleeps_for_nonzero_rank(self) -> None:
        with patch("src.dataloader.time.sleep") as sleep:
            apply_worker_stagger(2, 1.0)
        sleep.assert_called_once_with(2.0)

    def test_policy_from_reader_disabled_for_local(self) -> None:
        reader = ReaderConfig(on_hdfs_failure="skip", worker_stagger_sec=1.0)
        policy = RemoteIoPolicy.from_reader(reader, remote=False)
        self.assertFalse(policy.enabled)
        self.assertFalse(policy.skip_on_failure)

    def test_policy_from_reader_enables_pre_buffer(self) -> None:
        reader = ReaderConfig(hdfs_pre_buffer=True, hdfs_close_timeout=5)
        policy = RemoteIoPolicy.from_reader(reader, remote=True)
        self.assertTrue(policy.pre_buffer)
        self.assertEqual(policy.close_timeout, 5)

    def test_close_native_file_abandons_on_timeout(self) -> None:
        native = MagicMock()
        native.close.side_effect = lambda: time.sleep(1.0)
        close_hdfs_native_file(native, timeout_sec=0.05, description="close-test")
        native.close.assert_called()

    def test_thread_local_filesystem_differs_across_threads(self) -> None:
        keys: dict[str, int] = {}
        barrier = threading.Barrier(2)

        def worker(name: str) -> None:
            with patch(
                "src.dataloader._filesystem_from_uri",
                side_effect=lambda uri: object(),
            ):
                fs = thread_local_hdfs_filesystem("hdfs://ns")
                keys[name] = id(fs)
                barrier.wait(timeout=2)
                again = thread_local_hdfs_filesystem("hdfs://ns")
                self.assertIs(fs, again)

        threads = [
            threading.Thread(target=worker, args=("a",)),
            threading.Thread(target=worker, args=("b",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(len(keys), 2)
        self.assertNotEqual(keys["a"], keys["b"])

    def test_open_parquet_via_native_passes_pre_buffer(self) -> None:
        native = object()
        filesystem = MagicMock()
        filesystem.open_input_file.return_value = native
        captured: dict[str, object] = {}

        class FakePq:
            @staticmethod
            def ParquetFile(handle, **kwargs):
                captured["handle"] = handle
                captured["kwargs"] = kwargs
                return object()

        policy = _remote_policy(pre_buffer=True)
        parquet_file, native_file = open_parquet_via_native(
            filesystem=filesystem,
            fs_path="/data/part.parquet",
            lock_key="hdfs://ns/data/part.parquet",
            policy=policy,
            pq_module=FakePq,
        )
        self.assertIs(native_file, native)
        self.assertIs(captured["handle"], native)
        self.assertEqual(captured["kwargs"], {"pre_buffer": True})
        self.assertIsNotNone(parquet_file)

    def test_iter_batches_skips_on_open_failure(self) -> None:
        filesystem = MagicMock()
        filesystem.open_input_file.side_effect = OSError("Filesystem closed")
        policy = _remote_policy(on_failure="skip")
        with patch(
            "src.dataloader.thread_local_hdfs_filesystem",
            return_value=filesystem,
        ):
            batches = list(
                iter_parquet_record_batches(
                    fs_path="/missing.parquet",
                    filesystem=filesystem,
                    filesystem_key="hdfs://ns",
                    lock_key="hdfs://ns/missing.parquet",
                    policy=policy,
                    pq_module=MagicMock(),
                    description="skip-open",
                )
            )
        self.assertEqual(batches, [])

    def test_iter_batches_raises_when_fail_policy(self) -> None:
        filesystem = MagicMock()
        filesystem.open_input_file.side_effect = OSError("Filesystem closed")
        policy = _remote_policy(on_failure="fail", retry_count=0)
        with patch(
            "src.dataloader.thread_local_hdfs_filesystem",
            return_value=filesystem,
        ):
            with self.assertRaises(OSError):
                list(
                    iter_parquet_record_batches(
                        fs_path="/missing.parquet",
                        filesystem=filesystem,
                        filesystem_key="hdfs://ns",
                        lock_key="hdfs://ns/missing.parquet",
                        policy=policy,
                        pq_module=MagicMock(),
                    )
                )

    def test_batch_timeout_reopens_fresh_session_before_yield(self) -> None:
        """Timed-out next() retains old client and retries on a fresh one."""

        policy = _remote_policy(
            on_failure="skip",
            retry_count=1,
            retry_base_sec=0.01,
            op_timeout=0.05,
            open_timeout=1.0,
            close_timeout=0.05,
        )
        first_read_started = threading.Event()
        release_first_read = threading.Event()
        first_iterator_closed = threading.Event()
        first_native_closed = threading.Event()

        class FakeBatchIter:
            def __init__(self, session: int) -> None:
                self.session = session
                self._n = 0

            def __iter__(self):
                return self

            def __next__(self):
                self._n += 1
                if self.session == 0:
                    first_read_started.set()
                    release_first_read.wait(timeout=2.0)
                if self._n > 1:
                    raise StopIteration
                return f"batch-{self.session}-{self._n}"

            def close(self) -> None:
                if self.session == 0:
                    first_iterator_closed.set()

        class FakeParquet:
            def __init__(self, session: int) -> None:
                self.session = session

            def iter_batches(self, **_kwargs):
                return FakeBatchIter(self.session)

        class FakePq:
            @staticmethod
            def ParquetFile(handle, **_kwargs):
                return FakeParquet(handle.session)

        class FakeNative:
            def __init__(self, session: int) -> None:
                self.session = session

            def close(self) -> None:
                if self.session == 0:
                    first_native_closed.set()

        filesystems = [MagicMock(), MagicMock()]
        for session, filesystem in enumerate(filesystems):
            filesystem.open_input_file.return_value = FakeNative(session)

        filesystem_key = "hdfs://timeout-reopen-fresh-client"
        with patch(
            "src.dataloader._filesystem_from_uri",
            side_effect=filesystems,
        ) as filesystem_factory:
            batches = list(
                iter_parquet_record_batches(
                    fs_path="/data/part.parquet",
                    filesystem=filesystems[0],
                    filesystem_key=filesystem_key,
                    lock_key="hdfs://ns/data/part.parquet",
                    policy=policy,
                    pq_module=FakePq,
                    description="reopen-on-timeout",
                )
            )

        self.assertTrue(first_read_started.is_set())
        self.assertEqual(filesystem_factory.call_count, 2)
        self.assertEqual(batches[0], "batch-1-1")
        # Poisoned sessions are retained forever without native close.
        self.assertFalse(first_iterator_closed.is_set())
        self.assertFalse(first_native_closed.is_set())

        release_first_read.set()
        time.sleep(0.2)
        self.assertFalse(first_iterator_closed.is_set())
        self.assertFalse(first_native_closed.is_set())
        invalidate_thread_local_hdfs_filesystem(filesystem_key, filesystems[1])

    def test_batch_timeout_after_yield_defers_close_until_read_finishes(self) -> None:
        filesystem = MagicMock()
        policy = _remote_policy(
            on_failure="skip",
            retry_count=2,
            op_timeout=0.05,
            open_timeout=1.0,
            close_timeout=0.05,
        )
        blocked_read_started = threading.Event()
        release_blocked_read = threading.Event()
        iterator_closed = threading.Event()
        native_closed = threading.Event()

        class FakeBatchIter:
            def __init__(self) -> None:
                self._n = 0

            def __iter__(self):
                return self

            def __next__(self):
                self._n += 1
                if self._n == 1:
                    return "ok"
                if self._n == 2:
                    blocked_read_started.set()
                    release_blocked_read.wait(timeout=2.0)
                raise StopIteration

            def close(self) -> None:
                iterator_closed.set()

        class FakeNative:
            def close(self) -> None:
                native_closed.set()

        class FakeParquet:
            def iter_batches(self, **_kwargs):
                return FakeBatchIter()

        class FakePq:
            @staticmethod
            def ParquetFile(_handle, **_kwargs):
                return FakeParquet()

        filesystem.open_input_file.return_value = FakeNative()
        with patch(
            "src.dataloader.thread_local_hdfs_filesystem",
            return_value=filesystem,
        ):
            batches = list(
                iter_parquet_record_batches(
                    fs_path="/data/part.parquet",
                    filesystem=filesystem,
                    filesystem_key="hdfs://ns",
                    lock_key="hdfs://ns/data/part.parquet",
                    policy=policy,
                    pq_module=FakePq,
                    description="partial-timeout",
                )
            )
        self.assertEqual(batches, ["ok"])
        self.assertTrue(blocked_read_started.is_set())
        self.assertFalse(iterator_closed.is_set())
        self.assertFalse(native_closed.is_set())

        release_blocked_read.set()
        time.sleep(0.2)
        # Mid-stream timeout quarantines without ever calling native close.
        self.assertFalse(iterator_closed.is_set())
        self.assertFalse(native_closed.is_set())

    def test_filesystem_closed_reopens_with_a_new_client(self) -> None:
        """DFSClient.checkOpen failures must evict, not reuse, the dead client."""

        policy = _remote_policy(
            on_failure="fail",
            retry_count=1,
            retry_base_sec=0.01,
        )
        next_calls = [0, 0]
        native_closes = [0, 0]

        class FakeBatchIter:
            def __init__(self, session: int) -> None:
                self.session = session

            def __iter__(self):
                return self

            def __next__(self):
                next_calls[self.session] += 1
                if self.session == 0:
                    raise OSError(
                        "[Errno 255] HDFS read failed: "
                        "java.io.IOException: Filesystem closed at "
                        "org.apache.hadoop.hdfs.DFSClient.checkOpen"
                    )
                if next_calls[self.session] == 1:
                    return "recovered"
                raise StopIteration

        class FakeParquet:
            def __init__(self, session: int) -> None:
                self.session = session

            def iter_batches(self, **_kwargs):
                return FakeBatchIter(self.session)

        class FakePq:
            @staticmethod
            def ParquetFile(handle, **_kwargs):
                return FakeParquet(handle.session)

        class FakeNative:
            def __init__(self, session: int) -> None:
                self.session = session

            def close(self) -> None:
                native_closes[self.session] += 1

        filesystems = [MagicMock(), MagicMock()]
        for session, filesystem in enumerate(filesystems):
            filesystem.open_input_file.return_value = FakeNative(session)

        filesystem_key = "hdfs://filesystem-closed-fresh-client"
        with patch(
            "src.dataloader._filesystem_from_uri",
            side_effect=filesystems,
        ) as filesystem_factory:
            batches = list(
                iter_parquet_record_batches(
                    fs_path="/data/part.parquet",
                    filesystem=filesystems[0],
                    filesystem_key=filesystem_key,
                    lock_key="hdfs://ns/data/part.parquet",
                    policy=policy,
                    pq_module=FakePq,
                    description="filesystem-closed",
                )
            )

        self.assertEqual(batches, ["recovered"])
        self.assertEqual(filesystem_factory.call_count, 2)
        self.assertEqual(next_calls, [1, 2])
        self.assertEqual(native_closes, [1, 1])
        invalidate_thread_local_hdfs_filesystem(filesystem_key, filesystems[1])

    def test_native_open_timeout_quarantines_late_handle_and_reopens(self) -> None:
        policy = _remote_policy(
            on_failure="fail",
            retry_count=1,
            retry_base_sec=0.01,
            open_timeout=0.05,
        )
        first_open_started = threading.Event()
        release_first_open = threading.Event()
        first_native_closed = threading.Event()

        class FakeNative:
            def __init__(self, session: int) -> None:
                self.session = session

            def close(self) -> None:
                if self.session == 0:
                    first_native_closed.set()

        class FakeBatchIter:
            def __init__(self, session: int) -> None:
                self.session = session
                self.done = False

            def __iter__(self):
                return self

            def __next__(self):
                if self.done:
                    raise StopIteration
                self.done = True
                return f"batch-{self.session}"

            def close(self) -> None:
                return None

        class FakeParquet:
            def __init__(self, session: int) -> None:
                self.session = session

            def iter_batches(self, **_kwargs):
                return FakeBatchIter(self.session)

        class FakePq:
            @staticmethod
            def ParquetFile(handle, **_kwargs):
                return FakeParquet(handle.session)

        filesystems = [MagicMock(), MagicMock()]

        def first_open(_path):
            first_open_started.set()
            release_first_open.wait(timeout=2.0)
            return FakeNative(0)

        filesystems[0].open_input_file.side_effect = first_open
        filesystems[1].open_input_file.return_value = FakeNative(1)

        filesystem_key = "hdfs://native-open-timeout"
        with patch(
            "src.dataloader._filesystem_from_uri",
            side_effect=filesystems,
        ) as filesystem_factory:
            batches = list(
                iter_parquet_record_batches(
                    fs_path="/data/part.parquet",
                    filesystem=filesystems[0],
                    filesystem_key=filesystem_key,
                    lock_key="hdfs://ns/data/part.parquet",
                    policy=policy,
                    pq_module=FakePq,
                    description="native-open-timeout",
                )
            )

        self.assertTrue(first_open_started.is_set())
        self.assertEqual(batches, ["batch-1"])
        self.assertEqual(filesystem_factory.call_count, 2)
        self.assertFalse(first_native_closed.is_set())

        release_first_open.set()
        time.sleep(0.2)
        # Timed-out open handle is quarantined forever without native close.
        self.assertFalse(first_native_closed.is_set())
        invalidate_thread_local_hdfs_filesystem(filesystem_key, filesystems[1])

    def test_native_close_timeout_evicts_client_before_next_open(self) -> None:
        close_started = threading.Event()
        release_close = threading.Event()
        close_finished = threading.Event()

        class BlockingNative:
            def close(self) -> None:
                close_started.set()
                release_close.wait(timeout=2.0)
                close_finished.set()

        filesystems = [MagicMock(), MagicMock()]
        filesystem_key = "hdfs://native-close-timeout"
        with patch(
            "src.dataloader._filesystem_from_uri",
            side_effect=filesystems,
        ) as filesystem_factory:
            first = thread_local_hdfs_filesystem(filesystem_key)
            completed = close_hdfs_native_file(
                BlockingNative(),
                timeout_sec=0.05,
                description="blocking-native-close",
                filesystem=first,
                filesystem_key=filesystem_key,
            )
            second = thread_local_hdfs_filesystem(filesystem_key)

        self.assertTrue(close_started.is_set())
        self.assertFalse(completed)
        self.assertIs(first, filesystems[0])
        self.assertIs(second, filesystems[1])
        self.assertEqual(filesystem_factory.call_count, 2)

        release_close.set()
        self.assertTrue(close_finished.wait(timeout=1.0))
        invalidate_thread_local_hdfs_filesystem(filesystem_key, filesystems[1])

    def test_quarantine_limit_fail_fast(self) -> None:
        from src.dataloader import (
            RemoteIoQuarantineExhaustedError,
            _retain_abandoned_remote_session,
            abandoned_remote_session_count,
        )

        before = abandoned_remote_session_count()
        _retain_abandoned_remote_session(object(), label="q1", quarantine_limit=before + 2)
        _retain_abandoned_remote_session(object(), label="q2", quarantine_limit=before + 2)
        with self.assertRaises(RemoteIoQuarantineExhaustedError):
            _retain_abandoned_remote_session(
                object(),
                label="q3",
                quarantine_limit=before + 2,
            )

    def test_skip_tracker_budget_uses_rows(self) -> None:
        from src.dataloader import RemoteIoSkipBudgetExceededError, RemoteSkipTracker

        tracker = RemoteSkipTracker(max_skipped_row_groups=10, max_skipped_rows=100)
        tracker.record(row_groups=1, rows=40, label="a")
        tracker.record(row_groups=1, rows=40, label="b")
        with self.assertRaises(RemoteIoSkipBudgetExceededError):
            tracker.record(row_groups=1, rows=40, label="c")

    def test_interruptible_sleep_stops_early(self) -> None:
        from src.dataloader import _interruptible_sleep

        stop = threading.Event()

        def arm() -> None:
            time.sleep(0.05)
            stop.set()

        threading.Thread(target=arm, daemon=True).start()
        started = time.perf_counter()
        stopped = _interruptible_sleep(2.0, stop)
        elapsed = time.perf_counter() - started
        self.assertTrue(stopped)
        self.assertLess(elapsed, 1.0)

    def test_join_prefetch_thread_abandons_hung_worker(self) -> None:
        from src.dataloader import _join_prefetch_thread

        release = threading.Event()

        def hang() -> None:
            release.wait(timeout=5.0)

        worker = threading.Thread(target=hang, daemon=True)
        worker.start()
        started = time.perf_counter()
        _join_prefetch_thread(worker, timeout_sec=0.05, label="hung-prefetch")
        elapsed = time.perf_counter() - started
        self.assertTrue(worker.is_alive())
        self.assertLess(elapsed, 1.0)
        release.set()
        worker.join(timeout=1.0)

    def test_late_result_holder_survives_forced_gc(self) -> None:
        import gc
        import weakref

        from src.dataloader import (
            RemoteIoTimeoutError,
            _ABANDONED_REMOTE_SESSION_LOCK,
            _ABANDONED_REMOTE_SESSIONS,
            _TimedRemoteOperation,
            _defer_remote_session_cleanup,
            abandoned_remote_session_count,
        )

        class TrackedHandle:
            pass

        operation = _TimedRemoteOperation()
        timeout = RemoteIoTimeoutError("probe timeout", operation=operation)
        before = abandoned_remote_session_count()
        _defer_remote_session_cleanup(
            timeout,
            filesystem=object(),
            late_result_kind="native_file",
            label="late-gc-probe",
            quarantine_limit=before + 8,
        )
        handle = TrackedHandle()
        handle_ref = weakref.ref(handle)
        operation.result_box.append(handle)
        operation.done.set()

        late_holder = None
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            with _ABANDONED_REMOTE_SESSION_LOCK:
                for retained in reversed(_ABANDONED_REMOTE_SESSIONS):
                    for item in retained:
                        if isinstance(item, list) and item and item[0] is handle:
                            late_holder = item
                            break
                    if late_holder is not None:
                        break
            if late_holder is not None:
                break
            time.sleep(0.01)
        self.assertIsNotNone(late_holder)

        operation.result_box.clear()
        del handle
        for _ in range(5):
            gc.collect()
        self.assertIs(late_holder[0], handle_ref())
        self.assertIsNotNone(handle_ref())
        self.assertGreaterEqual(abandoned_remote_session_count(), before + 1)


class PrefetchScalingTest(unittest.TestCase):
    def test_prefetch_workers_reuse_thread_local_hdfs_clients(self) -> None:
        from src.dataloader import (
            ParquetScanner,
            _QueuedRecordBatch,
            _SENTINEL,
        )

        scanner = ParquetScanner.__new__(ParquetScanner)
        scanner.shard_world_size = 4
        scanner.split = type(
            "Split",
            (),
            {
                "reader": ReaderConfig(
                    num_workers=2,
                    prefetch_batches=2,
                    max_prefetch_bytes=1024,
                )
            },
        )()
        scanner._io_policy = RemoteIoPolicy.from_reader(
            scanner.split.reader,
            remote=True,
        )
        first_workers_ready = threading.Barrier(2)
        clients_by_item: dict[int, int] = {}

        def fake_worker(item, slot, stop_event) -> None:
            filesystem = thread_local_hdfs_filesystem(
                "hdfs://persistent-prefetch-test"
            )
            clients_by_item[item] = id(filesystem)
            if item < 2:
                first_workers_ready.wait(timeout=2.0)
            self.assertTrue(slot.byte_budget.acquire(1, stop_event))
            slot.queue.put(_QueuedRecordBatch(item, 1), timeout=2.0)
            slot.queue.put(_SENTINEL, timeout=2.0)

        with patch(
            "src.dataloader._filesystem_from_uri",
            side_effect=lambda _key: object(),
        ) as filesystem_factory, patch.object(
            scanner,
            "_row_group_worker",
            side_effect=fake_worker,
        ):
            observed = list(
                scanner._iter_row_group_record_batches_prefetch(
                    list(range(6)),
                    threading.Event(),
                )
            )

        self.assertEqual(observed, list(range(6)))
        self.assertEqual(filesystem_factory.call_count, 2)
        self.assertEqual(len(set(clients_by_item.values())), 2)

    def test_scaled_workers_for_gpu_counts(self) -> None:
        for world_size in (2, 4, 8):
            workers = scaled_hdfs_prefetch_workers(
                world_size=world_size,
                num_workers=8,
                prefetch_batches=8,
                work_item_count=100,
                remote=True,
                cpu_count=64,
            )
            self.assertEqual(workers, 4)

    def test_scaled_workers_respects_low_num_workers(self) -> None:
        workers = scaled_hdfs_prefetch_workers(
            world_size=4,
            num_workers=2,
            prefetch_batches=8,
            work_item_count=100,
            remote=True,
            cpu_count=64,
        )
        self.assertEqual(workers, 2)

    def test_local_workers_honor_num_workers_without_hardcap(self) -> None:
        # Local IO is multi-threaded: num_workers must not be silently capped at 4.
        self.assertEqual(
            scaled_hdfs_prefetch_workers(
                world_size=1,
                num_workers=8,
                prefetch_batches=8,
                work_item_count=100,
                remote=False,
            ),
            8,
        )
        self.assertEqual(
            scaled_hdfs_prefetch_workers(
                world_size=1,
                num_workers=4,
                prefetch_batches=8,
                work_item_count=100,
                remote=False,
            ),
            4,
        )
        # num_workers=0 → PyArrow defaults; single consumer prefetch thread only.
        self.assertEqual(
            scaled_hdfs_prefetch_workers(
                world_size=1,
                num_workers=0,
                prefetch_batches=8,
                work_item_count=100,
                remote=False,
            ),
            1,
        )

    def test_scanner_remote_prefetch_uses_gpu_scale(self) -> None:
        from src.dataloader import ParquetScanner

        scanner = ParquetScanner.__new__(ParquetScanner)
        scanner.shard_world_size = 4
        scanner.split = type(
            "Split",
            (),
            {"reader": ReaderConfig(num_workers=8, prefetch_batches=8)},
        )()
        scanner._io_policy = RemoteIoPolicy.from_reader(
            scanner.split.reader,
            remote=True,
        )
        self.assertEqual(scanner._prefetch_active_workers(100), 4)
        self.assertTrue(scanner._filesystem_is_remote())

    def test_production_configs_enable_hdfs_resilience(self) -> None:
        from pathlib import Path

        from src.config import load_app_config

        root = Path(__file__).resolve().parents[1]
        for name in (
            "rankmixer.yaml",
            "onetrans.yaml",
            "mdl_rankmixer.yaml",
            "mdl_onetrans.yaml",
        ):
            config = load_app_config(root / "configs" / name)
            reader = config.data.train.reader
            self.assertEqual(reader.shard_unit, "file")
            self.assertEqual(reader.on_hdfs_failure, "skip")
            self.assertEqual(reader.worker_stagger_sec, 1.0)
            self.assertEqual(reader.hdfs_retry_count, 2)
            self.assertTrue(reader.hdfs_file_lock)
            self.assertTrue(reader.hdfs_pre_buffer)
            self.assertEqual(reader.hdfs_close_timeout, 5)
            self.assertEqual(reader.hdfs_op_timeout, 30)
            self.assertEqual(reader.hdfs_open_timeout, 60)
            self.assertEqual(reader.hdfs_quarantine_limit, 32)
            self.assertEqual(reader.hdfs_max_skipped_row_groups, 64)
            self.assertEqual(reader.hdfs_max_skipped_rows, 2_000_000)
            self.assertEqual(reader.hdfs_prefetch_join_timeout, 5.0)
            self.assertEqual(reader.num_workers, 2)
            self.assertEqual(reader.prefetch_batches, 2)
            self.assertEqual(reader.max_prefetch_bytes, 2147483648)
            self.assertEqual(reader.cardinality_audit_raw_rows, 0)
            self.assertEqual(reader.host_prepare_startup_timeout_sec, 300.0)
            self.assertEqual(reader.host_prepare_idle_timeout_sec, 300.0)
            self.assertEqual(config.training.step_watchdog_sec, 600.0)
            if config.data.test is not None:
                self.assertEqual(config.data.test.reader.shard_unit, "file")
                self.assertEqual(config.data.test.reader.on_hdfs_failure, "skip")
                self.assertTrue(config.data.test.reader.hdfs_pre_buffer)


if __name__ == "__main__":
    unittest.main()
