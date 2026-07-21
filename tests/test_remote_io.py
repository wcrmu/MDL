"""Tests for HDFS/viewfs remote IO resilience helpers."""

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from src.config import ReaderConfig
from src.remote_io import (
    PerFileLock,
    RemoteIoPolicy,
    RemoteIoTimeoutError,
    apply_worker_stagger,
    call_with_timeout,
    close_hdfs_native_file,
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
        with patch("src.remote_io.time.sleep") as sleep:
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
                "src.remote_io._filesystem_from_uri",
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
            "src.remote_io.thread_local_hdfs_filesystem",
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
        policy = _remote_policy(on_failure="fail")
        with patch(
            "src.remote_io.thread_local_hdfs_filesystem",
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


class PrefetchScalingTest(unittest.TestCase):
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
            self.assertEqual(reader.shard_unit, "row_group")
            self.assertEqual(reader.on_hdfs_failure, "skip")
            self.assertEqual(reader.worker_stagger_sec, 1.0)
            self.assertEqual(reader.hdfs_retry_count, 5)
            self.assertTrue(reader.hdfs_file_lock)
            self.assertTrue(reader.hdfs_pre_buffer)
            self.assertEqual(reader.hdfs_close_timeout, 5)
            self.assertEqual(reader.hdfs_op_timeout, 30)
            if config.data.test is not None:
                self.assertEqual(config.data.test.reader.shard_unit, "row_group")
                self.assertEqual(config.data.test.reader.on_hdfs_failure, "skip")
                self.assertTrue(config.data.test.reader.hdfs_pre_buffer)


if __name__ == "__main__":
    unittest.main()
