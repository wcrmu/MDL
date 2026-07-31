from __future__ import (
    annotations,
)  # Defer annotation evaluation for forward references.

"""Parquet-to-PyTorch data pipeline.

This module owns the complete input path: it discovers and shards Parquet
files, streams Arrow batches, encodes configured features, and builds the
``FeatureBatch`` objects consumed by training and inference.

It also owns the direct agg Arrow path (``reader.agg_direct_mode``): request
group blocks, length-bucket packing, and axis ``PreparedAxisBatch`` materialization.

Remote HDFS/viewfs helpers live here as well (thread-local HadoopFileSystem,
timed open/close, retry, flock, and pre_buffer Parquet reads).
"""

from collections import defaultdict, deque
from collections.abc import Collection, Iterable as RuntimeIterable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import atexit
import fcntl
import fnmatch
from hashlib import sha256
from itertools import chain, islice
import glob
import heapq
import importlib
import json
import logging
import math
import multiprocessing
from numbers import Integral, Real
import os
import queue
import tempfile
import threading
import time
from pathlib import Path
from types import GeneratorType
from typing import Any, Callable, Iterable, Iterator, Literal, Mapping, TypeVar
from urllib.parse import unquote, urlsplit


class _LazyTorchModule:
    """Import torch on first use so adapter ProcessPool workers stay CUDA-free.

    ``adapter_workers`` children import this module only for
    ``adapt_mdl_rankmixer_parquet``. Pulling torch there costs ~1.5s+ per
    worker and stalls the parent on ``ProcessPoolExecutor.submit`` while
    forkserver bootstraps — looks like a hang / deadlock under load.
    """

    __slots__ = ("_mod",)

    def __init__(self) -> None:
        object.__setattr__(self, "_mod", None)

    def _load(self) -> Any:
        mod = object.__getattribute__(self, "_mod")
        if mod is None:
            import torch as torch_mod

            object.__setattr__(self, "_mod", torch_mod)
            return torch_mod
        return mod

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_mod":
            object.__setattr__(self, name, value)
            return
        setattr(self._load(), name, value)

    def __delattr__(self, name: str) -> None:
        if name == "_mod":
            object.__delattr__(self, name)
            return
        delattr(self._load(), name)


torch = _LazyTorchModule()  # type: ignore[assignment]
# Annotation / isinstance stand-in resolved via torch.Tensor at runtime.
Tensor = Any  # noqa: N816 - public alias used throughout this module

import numpy as np

from .config import (
    AppConfig,
    FeatureConfig,
    ParquetSplitConfig,
    ResolvedCategoricalInput,
    ResolvedIdentityEncoding,
    ResolvedPreHashedEncoding,
    SequenceConfig,
    resolve_categorical_base_input,
)
from .features import (
    encode_categorical_sequence_field,
    encode_categorical_value,
    encode_categorical_values,
)

logger = logging.getLogger(__name__)

_PROCESS_ADAPTER: Callable[..., Any] | None = None
_PROCESS_ADAPTER_CONTEXT: "ParquetAdapterContext | None" = None
_PROCESS_ADAPTER_NAME = ""
_PROCESS_ADAPTER_SPLIT_NAME = ""

# ---------------------------------------------------------------------------
# Remote filesystem IO (HDFS/viewfs)
#
# Eason-equivalent model inside one DDP rank:
# - each long-lived prefetch worker uses a thread-local HadoopFileSystem
# - open via open_input_file with timeout / retry / defensive flock
# - ParquetFile(native_file, pre_buffer=True) + iter_batches(use_threads=False)
# - timed native_file.close() so a corrupted stream cannot hang forever
# ---------------------------------------------------------------------------

T = TypeVar("T")

_RETRYABLE_NEEDLES = (
    "filesystem closed",
    "connection reset",
    "connection refused",
    "broken pipe",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "resource temporarily",
    "namenode",
    "datanode",
    "errno 255",
    "errno 110",
    "econnreset",
    "eagain",
)

_DEFAULT_NODE_CPU_COUNT = 64
_THREAD_LOCAL = threading.local()
_ABANDONED_REMOTE_SESSION_LOCK = threading.Lock()
_ABANDONED_REMOTE_SESSIONS: list[tuple[Any, ...]] = []
_DEFAULT_HDFS_QUARANTINE_LIMIT = 32
# Reused by wide-sequence prehashed gathers; creating a pool per batch was measurable.
_PREHASHED_GATHER_POOL: ThreadPoolExecutor | None = None
_PREHASHED_GATHER_POOL_WORKERS = 0
# Optional host-prepare progress hook. The spawn child installs a shared
# ``mp.Value`` writer so the parent watchdog can see HDFS list/footer/adapt
# progress before the first FeatureBatch is delivered.
_IO_PROGRESS_HOOK: Callable[[], None] | None = None
_IO_PROGRESS_HOOK_LOCK = threading.Lock()


def set_io_progress_hook(hook: Callable[[], None] | None) -> None:
    """Install or clear the process-wide remote-IO progress heartbeat callback."""

    global _IO_PROGRESS_HOOK
    with _IO_PROGRESS_HOOK_LOCK:
        _IO_PROGRESS_HOOK = hook


def note_io_progress() -> None:
    """Pulse the host-prepare watchdog when remote IO makes forward progress."""

    with _IO_PROGRESS_HOOK_LOCK:
        hook = _IO_PROGRESS_HOOK
    if hook is None:
        return
    try:
        hook()
    except Exception:
        pass


@contextmanager
def io_progress_pulses(interval_sec: float = 15.0) -> Iterator[None]:
    """Keep the host-prepare idle watchdog alive during long CPU prepare work.

    Used around adapt/tensorize that can exceed ``host_prepare_idle_timeout_sec``
    without touching HDFS. A hung JNI call that holds the GIL will also stall
    this helper (no false liveness); the step watchdog remains the backstop.
    """

    interval = float(interval_sec)
    with _IO_PROGRESS_HOOK_LOCK:
        hook_present = _IO_PROGRESS_HOOK is not None
    if interval <= 0 or not hook_present:
        yield
        return
    stop = threading.Event()

    def _pulse() -> None:
        while not stop.wait(interval):
            note_io_progress()

    thread = threading.Thread(
        target=_pulse,
        name="mdl-io-progress-pulse",
        daemon=True,
    )
    note_io_progress()
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=max(0.1, min(interval, 1.0)))
        note_io_progress()


class RemoteIoStallError(RuntimeError):
    """Fatal remote-IO stall; the rank should exit so the job can restart.

    Platform launchers should treat ``exit_code`` (default 70) as retryable.
    Do not respawn the reader inside the same training process: there is no
    precise row-group resume cursor, so in-process restart would duplicate rows.
    """

    exit_code: int = 70
    marker: str = "REMOTE_IO_STALL"

    def __init__(self, message: str, *, exit_code: int | None = None) -> None:
        prefix = self.marker if self.marker not in message else ""
        super().__init__(f"{prefix}: {message}" if prefix else message)
        if exit_code is not None:
            self.exit_code = int(exit_code)


class RemoteIoQuarantineExhaustedError(RemoteIoStallError):
    """Raised when too many poisoned HDFS sessions are retained without close."""


class RemoteIoSkipBudgetExceededError(RemoteIoStallError):
    """Raised when skipped row-group/row budgets are exhausted under skip policy."""


def abort_rank_for_remote_io_stall(error: BaseException) -> None:
    """Hard-exit the process for a fatal remote-IO stall (never returns)."""

    exit_code = int(getattr(error, "exit_code", RemoteIoStallError.exit_code))
    message = f"{RemoteIoStallError.marker}: aborting rank after fatal remote IO ({error})"
    logger.error(message)
    print(message, flush=True)
    os._exit(exit_code)


def is_remote_io_stall_error(error: BaseException) -> bool:
    """True when ``error`` (or its cause chain) is a fatal remote-IO stall."""

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, RemoteIoStallError):
            return True
        text = str(current)
        if RemoteIoStallError.marker in text:
            return True
        current = current.__cause__ or current.__context__
    return False


@dataclass
class RemoteSkipTracker:
    """Process-local counters for on_hdfs_failure=skip budgets."""

    max_skipped_row_groups: int | None = 64
    max_skipped_rows: int | None = 2_000_000
    skipped_row_groups: int = 0
    skipped_rows: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        *,
        row_groups: int = 0,
        rows: int = 0,
        label: str,
    ) -> None:
        row_groups = max(0, int(row_groups))
        rows = max(0, int(rows))
        if row_groups == 0 and rows == 0:
            return
        with self._lock:
            self.skipped_row_groups += row_groups
            self.skipped_rows += rows
            logger.warning(
                "remote skip budget | %s | +rg=%d +rows=%d | "
                "totals rg=%d/%s rows=%d/%s",
                label,
                row_groups,
                rows,
                self.skipped_row_groups,
                (
                    "inf"
                    if self.max_skipped_row_groups is None
                    else self.max_skipped_row_groups
                ),
                self.skipped_rows,
                "inf" if self.max_skipped_rows is None else self.max_skipped_rows,
            )
            if (
                self.max_skipped_row_groups is not None
                and self.skipped_row_groups > self.max_skipped_row_groups
            ):
                raise RemoteIoSkipBudgetExceededError(
                    f"skipped_row_groups={self.skipped_row_groups} exceeds "
                    f"reader.hdfs_max_skipped_row_groups="
                    f"{self.max_skipped_row_groups} while handling {label}"
                )
            if (
                self.max_skipped_rows is not None
                and self.skipped_rows > self.max_skipped_rows
            ):
                raise RemoteIoSkipBudgetExceededError(
                    f"skipped_rows={self.skipped_rows} exceeds "
                    f"reader.hdfs_max_skipped_rows={self.max_skipped_rows} "
                    f"while handling {label}"
                )

    def snapshot(self) -> dict[str, int | None]:
        with self._lock:
            return {
                "skipped_row_groups": self.skipped_row_groups,
                "skipped_rows": self.skipped_rows,
                "max_skipped_row_groups": self.max_skipped_row_groups,
                "max_skipped_rows": self.max_skipped_rows,
            }


def _prehashed_gather_pool(workers: int) -> ThreadPoolExecutor:
    global _PREHASHED_GATHER_POOL, _PREHASHED_GATHER_POOL_WORKERS
    workers = max(1, int(workers))
    if _PREHASHED_GATHER_POOL is None or _PREHASHED_GATHER_POOL_WORKERS != workers:
        if _PREHASHED_GATHER_POOL is not None:
            _PREHASHED_GATHER_POOL.shutdown(wait=False, cancel_futures=True)
        _PREHASHED_GATHER_POOL = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="prehashed-gather",
        )
        _PREHASHED_GATHER_POOL_WORKERS = workers
    return _PREHASHED_GATHER_POOL


def _shutdown_prehashed_gather_pool() -> None:
    global _PREHASHED_GATHER_POOL, _PREHASHED_GATHER_POOL_WORKERS
    pool = _PREHASHED_GATHER_POOL
    _PREHASHED_GATHER_POOL = None
    _PREHASHED_GATHER_POOL_WORKERS = 0
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


atexit.register(_shutdown_prehashed_gather_pool)


@dataclass
class _TimedRemoteOperation:
    """State retained after a timed call outlives its caller."""

    done: threading.Event = field(default_factory=threading.Event)
    result_box: list[Any] = field(default_factory=list)
    error_box: list[BaseException] = field(default_factory=list)


class RemoteIoTimeoutError(TimeoutError):
    """Raised when a remote IO call exceeds its configured timeout.

    Python cannot cancel the worker thread. ``operation`` therefore remains
    available so owners of native resources can keep them alive until the
    abandoned call actually exits.
    """

    def __init__(
        self,
        message: str,
        *,
        operation: _TimedRemoteOperation | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.cleanup_scheduled = False


class _ParquetSessionReopen(Exception):
    """Internal: abandon a poisoned parquet session and reopen from scratch.

    After ``call_with_timeout`` abandons a hung ``next()``, the daemon thread
    still owns the Arrow generator. Retrying ``next()`` / ``close()`` on that
    same iterator causes ``generator already executing`` and can wedge the
    process-wide HDFS client for tens of minutes. Callers must open a fresh
    ``ParquetFile`` instead of retrying in place.
    """

    def __init__(self, error: BaseException) -> None:
        super().__init__(str(error))
        self.error = error


@dataclass(frozen=True)
class RemoteIoPolicy:
    """Resolved remote-IO policy for one Parquet scanner."""

    enabled: bool
    op_timeout: float
    open_timeout: float
    retry_count: int
    retry_base_sec: float
    file_lock: bool
    on_failure: Literal["fail", "skip"]
    worker_stagger_sec: float
    pre_buffer: bool = True
    close_timeout: float = 5.0
    quarantine_limit: int = _DEFAULT_HDFS_QUARANTINE_LIMIT
    prefetch_join_timeout: float = 5.0

    @classmethod
    def disabled(cls) -> "RemoteIoPolicy":
        return cls(
            enabled=False,
            op_timeout=30.0,
            open_timeout=120.0,
            retry_count=0,
            retry_base_sec=0.5,
            file_lock=False,
            on_failure="fail",
            worker_stagger_sec=0.0,
            pre_buffer=False,
            close_timeout=5.0,
            quarantine_limit=_DEFAULT_HDFS_QUARANTINE_LIMIT,
            prefetch_join_timeout=5.0,
        )

    @classmethod
    def from_reader(cls, reader: Any, *, remote: bool) -> "RemoteIoPolicy":
        if not remote:
            return cls.disabled()
        return cls(
            enabled=True,
            op_timeout=float(reader.hdfs_op_timeout),
            open_timeout=float(reader.hdfs_open_timeout),
            retry_count=int(reader.hdfs_retry_count),
            retry_base_sec=float(reader.hdfs_retry_base_sec),
            file_lock=bool(reader.hdfs_file_lock),
            on_failure=reader.on_hdfs_failure,
            worker_stagger_sec=float(reader.worker_stagger_sec),
            pre_buffer=bool(getattr(reader, "hdfs_pre_buffer", True)),
            close_timeout=float(getattr(reader, "hdfs_close_timeout", 5.0)),
            quarantine_limit=int(
                getattr(reader, "hdfs_quarantine_limit", _DEFAULT_HDFS_QUARANTINE_LIMIT)
            ),
            prefetch_join_timeout=float(
                getattr(reader, "hdfs_prefetch_join_timeout", 5.0)
            ),
        )

    @property
    def skip_on_failure(self) -> bool:
        return self.enabled and self.on_failure == "skip"


class PerFileLock:
    """Serialize access to one URI across threads and local processes.

    Defensive only: root cause of DFSClient corruption is shared filesystem
    objects across threads, fixed by thread-local clients. Flock still helps
    when multiple ranks on one node touch the same URI (e.g. row_group LPT).
    """

    _thread_locks: dict[str, threading.RLock] = {}
    _registry_guard = threading.Lock()

    def __init__(self, key: str, *, enabled: bool) -> None:
        self.key = key
        self.enabled = enabled
        self._thread_lock: threading.RLock | None = None
        self._file_handle: Any | None = None
        if enabled:
            self._thread_lock = self._lock_for(key)

    @classmethod
    def _lock_for(cls, key: str) -> threading.RLock:
        with cls._registry_guard:
            lock = cls._thread_locks.get(key)
            if lock is None:
                lock = threading.RLock()
                cls._thread_locks[key] = lock
            return lock

    def __enter__(self) -> "PerFileLock":
        if not self.enabled:
            return self
        assert self._thread_lock is not None
        self._thread_lock.acquire()
        lock_dir = Path(tempfile.gettempdir()) / "mdl-hdfs-file-locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        digest = sha256(self.key.encode("utf-8")).hexdigest()
        lock_path = lock_dir / digest
        handle = open(lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except Exception:
            handle.close()
            self._thread_lock.release()
            raise
        self._file_handle = handle
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self.enabled:
            return
        handle = self._file_handle
        self._file_handle = None
        if handle is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        if self._thread_lock is not None:
            self._thread_lock.release()


def is_poisoned_iterator_error(error: BaseException) -> bool:
    """Return True when an Arrow/Python generator was left mid-execution.

    After ``call_with_timeout`` abandons a hung ``next()``, the daemon thread
    still owns the generator. Further ``next()`` / ``close()`` on that object
    raises this error or deadlocks — callers must reopen a fresh session.
    """

    text = f"{type(error).__name__}: {error}".lower()
    return "generator already executing" in text


def is_retryable_remote_error(error: BaseException) -> bool:
    """Return True for transient remote IO failures worth retrying."""

    # Never treat a poisoned generator as "retry the same callable" — the
    # retry must reopen a new Parquet iterator (see iter_parquet_record_batches).
    if is_poisoned_iterator_error(error):
        return False
    if isinstance(error, (RemoteIoTimeoutError, TimeoutError, InterruptedError)):
        return True
    if isinstance(error, (BlockingIOError, ConnectionError, BrokenPipeError, OSError)):
        return True
    text = f"{type(error).__name__}: {error}".lower()
    return any(needle in text for needle in _RETRYABLE_NEEDLES)


def _close_batch_iterator(
    batch_iterator: Any,
    *,
    poisoned: bool,
    label: str,
) -> None:
    """Close an Arrow batch iterator unless a timed-out thread still owns it."""

    if batch_iterator is None:
        return
    if poisoned:
        logger.warning(
            "abandoning batch iterator for %s without close "
            "(iterator timed out or poisoned)",
            label,
        )
        return
    close = getattr(batch_iterator, "close", None)
    if not callable(close):
        return
    try:
        close()
    except BaseException as error:
        logger.warning("failed to close batch iterator for %s: %s", label, error)


def abandoned_remote_session_count() -> int:
    """Return how many poisoned HDFS sessions are currently quarantined."""

    with _ABANDONED_REMOTE_SESSION_LOCK:
        return len(_ABANDONED_REMOTE_SESSIONS)


def _retain_abandoned_remote_session(
    *resources: Any,
    label: str,
    quarantine_limit: int = _DEFAULT_HDFS_QUARANTINE_LIMIT,
) -> None:
    """Keep a poisoned session alive until process exit; never native-close it.

    Entering JNI ``close`` on a handle that still has an abandoned ``pread``
    (or on a damaged DistributedRaid client) can wedge process-wide. Timeout
    paths therefore retain references and rely on process restart to reclaim.
    """

    retained = tuple(resource for resource in resources if resource is not None)
    if not retained:
        return
    limit = max(1, int(quarantine_limit))
    with _ABANDONED_REMOTE_SESSION_LOCK:
        if len(_ABANDONED_REMOTE_SESSIONS) >= limit:
            raise RemoteIoQuarantineExhaustedError(
                f"HDFS quarantine limit {limit} exceeded while handling {label}; "
                "restart the process to reclaim abandoned DFSClients"
            )
        _ABANDONED_REMOTE_SESSIONS.append(retained)
        quarantined = len(_ABANDONED_REMOTE_SESSIONS)
    logger.warning(
        "quarantining %s until process exit without native close "
        "(%d/%d abandoned sessions)",
        label,
        quarantined,
        limit,
    )


def _defer_remote_session_cleanup(
    timeout_error: RemoteIoTimeoutError,
    *,
    filesystem: Any,
    batch_iterator: Any = None,
    native_file: Any = None,
    late_result_kind: Literal["ignore", "batch_iterator", "native_file"] = "ignore",
    retained_resources: tuple[Any, ...] = (),
    label: str,
    quarantine_limit: int = _DEFAULT_HDFS_QUARANTINE_LIMIT,
) -> None:
    """Quarantine a timed-out HDFS session and never call native close on it.

    Arrow HDFS streams keep a raw ``hdfsFS`` pointer. Destroying the owning
    HadoopFileSystem (or closing the native handle) while ``pread`` is still
    running closes the DFSClient under that read and can produce
    ``DFSClient.checkOpen: Filesystem closed``. Python also cannot cancel the
    abandoned JNI close. The durable policy is: retain forever, open retries on
    a fresh client, and fail-fast when the quarantine cap is hit.
    """

    operation = timeout_error.operation
    # Retain the list object itself in quarantine. A late-returning handle is
    # written into late_holder[0] after the caller has already moved on; if only
    # the pre-timeout (often None) resources were retained, GC would destroy the
    # late native/iterator while JNI may still be using it.
    late_holder: list[Any] = [None]

    def _collect_resources() -> tuple[Any, ...]:
        iterator = batch_iterator
        native = native_file
        late_result = late_holder[0]
        if late_result_kind == "batch_iterator" and iterator is None:
            iterator = late_result
        elif late_result_kind == "native_file" and native is None:
            native = late_result
        return (
            filesystem,
            iterator,
            native,
            late_holder,
            *retained_resources,
        )

    if operation is None:
        _retain_abandoned_remote_session(
            *_collect_resources(),
            label=label,
            quarantine_limit=quarantine_limit,
        )
        return

    # Retain immediately so a late-returning handle cannot outlive the
    # caller's stack without a strong reference, and so the quarantine cap
    # applies before spawning the waiter thread.
    _retain_abandoned_remote_session(
        *_collect_resources(),
        label=label,
        quarantine_limit=quarantine_limit,
    )

    def observe_completion() -> None:
        operation.done.wait()
        if operation.result_box:
            late_holder[0] = operation.result_box[0]
        if operation.error_box:
            logger.warning(
                "%s timed operation eventually failed while quarantined "
                "(resources remain unclosed): %s",
                label,
                operation.error_box[0],
            )
        else:
            logger.warning(
                "%s timed operation finished while quarantined; "
                "resources remain unclosed until process exit",
                label,
            )
        # late_holder is already in _ABANDONED_REMOTE_SESSIONS; keep a local
        # strong reference until this observer returns as belt-and-suspenders.
        _ = _collect_resources()

    logger.warning(
        "tracking %s until its timed operation exits; retry uses a fresh "
        "HDFS client and will not native-close the poisoned session",
        label,
    )
    observer = threading.Thread(
        target=observe_completion,
        name=f"remote-io-quarantine:{label[:40]}",
        daemon=True,
    )
    observer.start()


def _interruptible_sleep(
    delay_sec: float,
    stop_event: threading.Event | None,
) -> bool:
    """Sleep up to ``delay_sec``; return True if ``stop_event`` was set."""

    delay = max(0.0, float(delay_sec))
    if delay <= 0.0:
        return bool(stop_event is not None and stop_event.is_set())
    if stop_event is None:
        time.sleep(delay)
        return False
    return bool(stop_event.wait(delay))


def _join_prefetch_thread(
    thread: threading.Thread | None,
    *,
    timeout_sec: float,
    label: str,
) -> None:
    """Join a prefetch worker with a hard timeout; abandon if still alive."""

    if thread is None:
        return
    timeout = max(0.01, float(timeout_sec))
    thread.join(timeout=timeout)
    if thread.is_alive():
        logger.warning(
            "abandoning %s after %.1fs join timeout (worker may still be "
            "blocked in HDFS JNI)",
            label,
            timeout,
        )


def maybe_skip_or_raise(
    error: BaseException,
    policy: RemoteIoPolicy,
    *,
    description: str,
    skip_tracker: RemoteSkipTracker | None = None,
    skipped_row_groups: int = 0,
    skipped_rows: int = 0,
) -> bool:
    """Log and return True when the policy says to skip; otherwise re-raise."""

    if policy.skip_on_failure:
        logger.warning("skipping %s after failure: %s", description, error)
        if skip_tracker is not None:
            skip_tracker.record(
                row_groups=skipped_row_groups,
                rows=skipped_rows,
                label=description,
            )
        return True
    raise error


def call_with_timeout(
    fn: Callable[[], T],
    timeout_sec: float,
    *,
    description: str = "remote IO",
) -> T:
    """Run ``fn`` in a daemon thread and raise if it exceeds ``timeout_sec``."""

    if timeout_sec <= 0:
        return fn()

    operation = _TimedRemoteOperation()

    def runner() -> None:
        try:
            operation.result_box.append(fn())
        except BaseException as error:  # noqa: BLE001 - surface to caller
            operation.error_box.append(error)
        finally:
            operation.done.set()

    thread = threading.Thread(
        target=runner,
        name=f"remote-io-timeout:{description[:48]}",
        daemon=True,
    )
    thread.start()
    if not operation.done.wait(timeout_sec):
        raise RemoteIoTimeoutError(
            f"{description} timed out after {timeout_sec:.1f}s",
            operation=operation,
        )
    if operation.error_box:
        raise operation.error_box[0]
    return operation.result_box[0]


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    retries: int,
    base_sec: float,
    description: str,
    is_retryable: Callable[[BaseException], bool] = is_retryable_remote_error,
) -> T:
    """Invoke ``fn`` with exponential backoff on transient failures."""

    attempt = 0
    while True:
        try:
            return fn()
        except BaseException as error:
            if attempt >= retries or not is_retryable(error):
                raise
            delay = base_sec * (2**attempt)
            logger.warning(
                "%s failed (%s); retry %d/%d in %.2fs",
                description,
                error,
                attempt + 1,
                retries,
                delay,
            )
            time.sleep(delay)
            attempt += 1


def apply_worker_stagger(rank: int, stagger_sec: float) -> None:
    """Sleep so DDP ranks open remote files at staggered times."""

    if stagger_sec <= 0 or rank <= 0:
        return
    delay = float(stagger_sec) * int(rank)
    logger.info(
        "staggering remote parquet scanner start by %.2fs for shard_rank=%d",
        delay,
        rank,
    )
    # Chunk the sleep and heartbeat so a large rank stagger does not look like
    # a host-prepare JNI hang to the parent startup watchdog.
    deadline = time.monotonic() + delay
    while True:
        note_io_progress()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(remaining, 5.0))


def run_remote_op(
    fn: Callable[[], T],
    policy: RemoteIoPolicy,
    *,
    description: str,
    timeout_sec: float | None = None,
) -> T:
    """Run a remote op with timeout + retry when the policy is enabled."""

    if not policy.enabled:
        return fn()
    effective_timeout = policy.op_timeout if timeout_sec is None else timeout_sec

    def once() -> T:
        return call_with_timeout(
            fn,
            effective_timeout,
            description=description,
        )

    return retry_with_backoff(
        once,
        retries=policy.retry_count,
        base_sec=policy.retry_base_sec,
        description=description,
    )


def run_under_file_lock(
    fn: Callable[[], T],
    *,
    lock_key: str,
    policy: RemoteIoPolicy,
    description: str,
    timeout_sec: float | None = None,
) -> T:
    """Hold the per-URI lock while running a timeout/retry protected call."""

    with PerFileLock(lock_key, enabled=policy.file_lock):
        return run_remote_op(
            fn,
            policy,
            description=description,
            timeout_sec=timeout_sec,
        )


def scaled_hdfs_prefetch_workers(
    *,
    world_size: int,
    num_workers: int,
    prefetch_batches: int,
    work_item_count: int,
    remote: bool,
    cpu_count: int = _DEFAULT_NODE_CPU_COUNT,
) -> int:
    """Bound concurrent Parquet prefetch *threads* (not adapter processes).

    Local files: ``num_workers`` is the reader-owned IO thread budget.
    ``num_workers=0`` leaves concurrency to PyArrow defaults (single consumer
    thread pulling batches; ``iter_batches(use_threads=True)``).

    HDFS: keep a GPU-scaled cap of 4/rank so NameNode / DFSClient pressure
    stays bounded under multi-GPU launches.
    """

    if prefetch_batches <= 0 or work_item_count <= 0:
        return 0
    if not remote:
        if num_workers <= 0:
            return 1
        return min(work_item_count, prefetch_batches, num_workers)

    auto = min(4, max(1, int(cpu_count) // (2 * max(1, int(world_size)))))
    if num_workers <= 0:
        configured = auto
    elif num_workers >= 4:
        configured = min(auto, num_workers)
    else:
        configured = num_workers
    return min(work_item_count, prefetch_batches, max(1, configured))


def is_hdfs_filesystem(filesystem: Any) -> bool:
    """Best-effort type/name check for pyarrow HadoopFileSystem."""

    if filesystem is None:
        return False
    type_name = type(filesystem).__name__.lower()
    module_name = type(filesystem).__module__.lower()
    return "hadoop" in type_name or "hadoop" in module_name or "hdfs" in type_name


def _filesystem_from_uri(filesystem_key: str) -> Any:
    """Create a filesystem from a URI; isolated for tests to patch."""

    import pyarrow.fs as pafs

    filesystem, _parsed = pafs.FileSystem.from_uri(filesystem_key)
    return filesystem


def thread_local_hdfs_filesystem(
    filesystem_key: str,
    *,
    prototype: Any | None = None,
) -> Any:
    """Return a per-thread filesystem for ``filesystem_key``.

    Reuses ``prototype`` only on the first call for that key in this thread when
    cloning via ``from_uri`` is unnecessary (local FS). For remote keys, always
    builds a fresh ``FileSystem.from_uri(filesystem_key)`` so each worker owns
    an independent DFSClient.
    """

    cache: dict[str, Any] | None = getattr(_THREAD_LOCAL, "filesystems", None)
    if cache is None:
        cache = {}
        _THREAD_LOCAL.filesystems = cache
    cached = cache.get(filesystem_key)
    if cached is not None:
        return cached

    if filesystem_key.startswith("file://") or filesystem_key == "file://":
        if prototype is not None:
            cache[filesystem_key] = prototype
            return prototype
        import pyarrow.fs as pafs

        filesystem = pafs.LocalFileSystem()
        cache[filesystem_key] = filesystem
        return filesystem

    filesystem = _filesystem_from_uri(filesystem_key)
    cache[filesystem_key] = filesystem
    return filesystem


def invalidate_thread_local_hdfs_filesystem(
    filesystem_key: str,
    filesystem: Any,
) -> bool:
    """Evict exactly ``filesystem`` from the current worker's HDFS cache.

    Identity checking prevents cleanup for an old session from evicting a newer
    replacement client created under the same URI key.
    """

    cache: dict[str, Any] | None = getattr(_THREAD_LOCAL, "filesystems", None)
    if cache is None or cache.get(filesystem_key) is not filesystem:
        return False
    del cache[filesystem_key]
    return True


def close_hdfs_native_file(
    native_file: Any,
    *,
    timeout_sec: float = 5.0,
    description: str = "close hdfs native file",
    filesystem: Any = None,
    filesystem_key: str | None = None,
) -> bool:
    """Close a native stream without ever reusing a client during hung close.

    Returns ``True`` when close completed (successfully or with a synchronous
    error) and ``False`` when close timed out. A timed close continues on its
    daemon thread while a cleanup waiter retains the owning filesystem.
    """

    if native_file is None:
        return True
    close = getattr(native_file, "close", None)
    if not callable(close):
        return True
    try:
        call_with_timeout(close, timeout_sec, description=description)
    except RemoteIoTimeoutError as error:
        if filesystem_key is not None and filesystem is not None:
            invalidate_thread_local_hdfs_filesystem(filesystem_key, filesystem)
            _defer_remote_session_cleanup(
                error,
                filesystem=filesystem,
                label=description,
                quarantine_limit=_DEFAULT_HDFS_QUARANTINE_LIMIT,
            )
        logger.warning(
            "%s timed out after %.1fs; quarantining client without further "
            "native close",
            description,
            timeout_sec,
        )
        return False
    except BaseException as error:
        if filesystem_key is not None and filesystem is not None:
            invalidate_thread_local_hdfs_filesystem(filesystem_key, filesystem)
        logger.warning("%s failed: %s", description, error)
    return True


def _close_remote_parquet_session(
    *,
    batch_iterator: Any,
    parquet_file: Any,
    native_file: Any,
    filesystem: Any,
    filesystem_key: str,
    policy: RemoteIoPolicy,
    label: str,
) -> None:
    """Close iterator then native handle without racing either operation."""

    iterator_close = getattr(batch_iterator, "close", None)
    if callable(iterator_close):
        try:
            call_with_timeout(
                iterator_close,
                policy.close_timeout,
                description=f"{label} (iterator close)",
            )
        except RemoteIoTimeoutError as error:
            invalidate_thread_local_hdfs_filesystem(filesystem_key, filesystem)
            _defer_remote_session_cleanup(
                error,
                filesystem=filesystem,
                native_file=native_file,
                retained_resources=(parquet_file, batch_iterator),
                label=f"{label} (iterator close)",
                quarantine_limit=policy.quarantine_limit,
            )
            error.cleanup_scheduled = True
            return
        except BaseException as error:
            logger.warning("failed to close batch iterator for %s: %s", label, error)
            invalidate_thread_local_hdfs_filesystem(
                filesystem_key,
                filesystem,
            )
            if is_poisoned_iterator_error(error):
                _retain_abandoned_remote_session(
                    filesystem,
                    parquet_file,
                    batch_iterator,
                    native_file,
                    label=f"{label} (iterator close)",
                    quarantine_limit=policy.quarantine_limit,
                )
                return

    close_hdfs_native_file(
        native_file,
        timeout_sec=policy.close_timeout,
        description=f"{label} (native close)",
        filesystem=filesystem,
        filesystem_key=filesystem_key,
    )


def open_hdfs_input_with_protection(
    filesystem: Any,
    fs_path: str,
    *,
    lock_key: str,
    policy: RemoteIoPolicy,
    description: str | None = None,
) -> Any:
    """Open one native HDFS handle under flock + timeout.

    Resource-producing calls are deliberately single-attempt. If they time
    out, their daemon thread can still return a live handle later; retrying on
    the same DFSClient would race that operation. Session owners perform retry
    with a fresh filesystem instead.
    """

    label = description or f"open_input_file {lock_key}"

    def open_fn() -> Any:
        return filesystem.open_input_file(fs_path)

    with PerFileLock(lock_key, enabled=policy.file_lock):
        return call_with_timeout(
            open_fn,
            policy.open_timeout if policy.enabled else policy.op_timeout,
            description=label,
        )


def open_parquet_via_native(
    *,
    filesystem: Any,
    fs_path: str,
    lock_key: str,
    policy: RemoteIoPolicy,
    pq_module: Any,
    filesystem_key: str | None = None,
    description: str | None = None,
) -> tuple[Any, Any | None]:
    """Open a ``ParquetFile``, using native_file + pre_buffer on remote FS.

    Returns ``(parquet_file, native_file_or_none)``. Caller must close the
    native file with ``close_hdfs_native_file`` when not None.
    """

    label = description or f"open parquet {lock_key}"
    if not policy.enabled:
        return pq_module.ParquetFile(fs_path, filesystem=filesystem), None

    try:
        native_file = open_hdfs_input_with_protection(
            filesystem,
            fs_path,
            lock_key=lock_key,
            policy=policy,
            description=f"{label} (native open)",
        )
    except RemoteIoTimeoutError as error:
        if filesystem_key is not None:
            invalidate_thread_local_hdfs_filesystem(filesystem_key, filesystem)
        _defer_remote_session_cleanup(
            error,
            filesystem=filesystem,
            late_result_kind="native_file",
            label=f"{label} (native open)",
            quarantine_limit=policy.quarantine_limit,
        )
        error.cleanup_scheduled = True
        raise
    except BaseException as error:
        if filesystem_key is not None and is_retryable_remote_error(error):
            invalidate_thread_local_hdfs_filesystem(filesystem_key, filesystem)
        raise

    def build() -> Any:
        return pq_module.ParquetFile(
            native_file,
            pre_buffer=policy.pre_buffer,
        )

    try:
        parquet_file = call_with_timeout(
            build,
            policy.open_timeout,
            description=f"{label} (ParquetFile)",
        )
    except RemoteIoTimeoutError as error:
        if filesystem_key is not None:
            invalidate_thread_local_hdfs_filesystem(filesystem_key, filesystem)
        _defer_remote_session_cleanup(
            error,
            filesystem=filesystem,
            native_file=native_file,
            retained_resources=(build,),
            label=f"{label} (ParquetFile)",
            quarantine_limit=policy.quarantine_limit,
        )
        error.cleanup_scheduled = True
        raise
    except BaseException as error:
        if filesystem_key is not None and is_retryable_remote_error(error):
            invalidate_thread_local_hdfs_filesystem(filesystem_key, filesystem)
        close_hdfs_native_file(
            native_file,
            timeout_sec=policy.close_timeout,
            description=f"{label} (close after open failure)",
            filesystem=filesystem,
            filesystem_key=filesystem_key,
        )
        raise
    return parquet_file, native_file


@contextmanager
def parquet_native_session(
    *,
    filesystem_key: str,
    fs_path: str,
    lock_key: str,
    policy: RemoteIoPolicy,
    pq_module: Any,
    prototype: Any | None = None,
    description: str | None = None,
) -> Iterator[tuple[Any, Any | None]]:
    """Thread-local FS + protected open; always timed-close the native handle."""

    filesystem = thread_local_hdfs_filesystem(
        filesystem_key,
        prototype=prototype,
    )
    parquet_file, native_file = open_parquet_via_native(
        filesystem=filesystem,
        fs_path=fs_path,
        lock_key=lock_key,
        policy=policy,
        pq_module=pq_module,
        filesystem_key=filesystem_key,
        description=description,
    )
    try:
        yield parquet_file, native_file
    finally:
        close_hdfs_native_file(
            native_file,
            timeout_sec=policy.close_timeout,
            description=f"{description or lock_key} (native close)",
            filesystem=filesystem,
            filesystem_key=filesystem_key,
        )


def _run_remote_parquet_operation(
    *,
    filesystem_key: str,
    prototype: Any,
    fs_path: str,
    lock_key: str,
    policy: RemoteIoPolicy,
    pq_module: Any,
    description: str,
    operation: Callable[[Any], T],
) -> T:
    """Run a metadata operation with whole-session retries on fresh clients."""

    max_sessions = 1 + max(0, int(policy.retry_count))
    last_error: BaseException | None = None
    for session_idx in range(max_sessions):
        if session_idx > 0 and last_error is not None:
            delay = float(policy.retry_base_sec) * (2 ** (session_idx - 1))
            logger.warning(
                "%s failed (%s); reopen session %d/%d in %.2fs",
                description,
                last_error,
                session_idx + 1,
                max_sessions,
                delay,
            )
            time.sleep(delay)

        filesystem = thread_local_hdfs_filesystem(
            filesystem_key,
            prototype=prototype,
        )
        native_file: Any = None
        try:
            parquet_file, native_file = open_parquet_via_native(
                filesystem=filesystem,
                fs_path=fs_path,
                lock_key=lock_key,
                policy=policy,
                pq_module=pq_module,
                filesystem_key=filesystem_key,
                description=description,
            )
            return operation(parquet_file)
        except BaseException as error:
            if not is_retryable_remote_error(error):
                raise
            invalidate_thread_local_hdfs_filesystem(
                filesystem_key,
                filesystem,
            )
            last_error = error
        finally:
            close_hdfs_native_file(
                native_file,
                timeout_sec=policy.close_timeout,
                description=f"{description} (native close)",
                filesystem=filesystem,
                filesystem_key=filesystem_key,
            )

    assert last_error is not None
    raise last_error


def _iter_parquet_record_batches_local(
    *,
    fs_path: str,
    filesystem: Any,
    lock_key: str,
    pq_module: Any,
    stop_event: threading.Event | None,
    kwargs: dict[str, Any],
) -> Iterator[Any]:
    """Plain local ``ParquetFile.iter_batches`` path (no remote timeouts)."""

    with PerFileLock(lock_key, enabled=False):
        parquet_file = pq_module.ParquetFile(fs_path, filesystem=filesystem)
    batch_iterator = iter(parquet_file.iter_batches(**kwargs))
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                return
            try:
                batch = next(batch_iterator)
            except StopIteration:
                return
            note_io_progress()
            yield batch
    finally:
        _close_batch_iterator(batch_iterator, poisoned=False, label=lock_key)


def _iter_parquet_record_batches_remote_session(
    *,
    fs_path: str,
    filesystem: Any,
    filesystem_key: str,
    lock_key: str,
    policy: RemoteIoPolicy,
    pq_module: Any,
    stop_event: threading.Event | None,
    label: str,
    kwargs: dict[str, Any],
    skip_tracker: RemoteSkipTracker | None = None,
    estimated_rows: int | None = None,
) -> Iterator[Any]:
    """One remote open→stream attempt. Raises ``_ParquetSessionReopen`` to retry."""

    fs: Any = None
    parquet_file: Any = None
    native_file: Any | None = None
    batch_iterator: Any = None
    yielded_any = False
    yielded_rows = 0
    timeout_error: RemoteIoTimeoutError | None = None
    timeout_late_result_kind: Literal["ignore", "batch_iterator"] = "ignore"
    untrackable_poison = False
    invalidate_filesystem = False

    def _skip_rows_for_abandon() -> int:
        if estimated_rows is None:
            return 0
        return max(0, int(estimated_rows) - int(yielded_rows))

    try:
        fs = thread_local_hdfs_filesystem(
            filesystem_key,
            prototype=filesystem,
        )
        parquet_file, native_file = open_parquet_via_native(
            filesystem=fs,
            fs_path=fs_path,
            lock_key=lock_key,
            policy=policy,
            pq_module=pq_module,
            filesystem_key=filesystem_key,
            description=label,
        )
        try:
            batch_iterator = call_with_timeout(
                lambda: iter(parquet_file.iter_batches(**kwargs)),
                policy.open_timeout,
                description=f"{label} (start)",
            )
        except RemoteIoTimeoutError as error:
            timeout_error = error
            timeout_late_result_kind = "batch_iterator"
            invalidate_filesystem = True
            raise _ParquetSessionReopen(error) from error

        while True:
            if stop_event is not None and stop_event.is_set():
                return

            def next_batch(iterator: Any = batch_iterator) -> Any:
                return next(iterator)

            try:
                batch = call_with_timeout(
                    next_batch,
                    policy.op_timeout,
                    description=f"{label} (batch)",
                )
            except StopIteration:
                return
            except RemoteIoTimeoutError as error:
                timeout_error = error
                invalidate_filesystem = True
                if yielded_any:
                    # Already emitted batches — reopening would duplicate rows.
                    logger.warning(
                        "%s timed out after yielding batches; "
                        "abandoning remainder of this row group",
                        label,
                    )
                    if maybe_skip_or_raise(
                        error,
                        policy,
                        description=f"{label} (batch)",
                        skip_tracker=skip_tracker,
                        skipped_row_groups=1,
                        skipped_rows=_skip_rows_for_abandon(),
                    ):
                        return
                    raise
                raise _ParquetSessionReopen(error) from error
            except BaseException as error:
                if is_poisoned_iterator_error(error):
                    untrackable_poison = True
                    invalidate_filesystem = True
                    if yielded_any:
                        if maybe_skip_or_raise(
                            error,
                            policy,
                            description=f"{label} (batch)",
                            skip_tracker=skip_tracker,
                            skipped_row_groups=1,
                            skipped_rows=_skip_rows_for_abandon(),
                        ):
                            return
                        raise
                    raise _ParquetSessionReopen(error) from error
                if is_retryable_remote_error(error):
                    invalidate_filesystem = True
                    if not yielded_any:
                        raise _ParquetSessionReopen(error) from error
                if maybe_skip_or_raise(
                    error,
                    policy,
                    description=f"{label} (batch)",
                    skip_tracker=skip_tracker,
                    skipped_row_groups=1,
                    skipped_rows=_skip_rows_for_abandon() if yielded_any else (
                        int(estimated_rows) if estimated_rows is not None else 0
                    ),
                ):
                    return
                raise

            yielded_any = True
            try:
                yielded_rows += int(getattr(batch, "num_rows", 0) or 0)
            except Exception:
                pass
            note_io_progress()
            yield batch
    except _ParquetSessionReopen:
        invalidate_filesystem = True
        raise
    except BaseException as error:
        # Failures before the batch loop are open/start errors. Prefer session
        # reopen for transient faults; honor skip only after retries exhaust
        # in the outer iter_parquet_record_batches loop (or immediately when
        # the error is not retryable).
        if batch_iterator is None and not yielded_any:
            if isinstance(error, RemoteIoTimeoutError):
                invalidate_filesystem = True
                if not error.cleanup_scheduled:
                    timeout_error = error
            if is_retryable_remote_error(error):
                invalidate_filesystem = True
                raise _ParquetSessionReopen(error) from error
            if maybe_skip_or_raise(
                error,
                policy,
                description=f"{label} (open)",
                skip_tracker=skip_tracker,
                skipped_row_groups=1,
                skipped_rows=int(estimated_rows) if estimated_rows is not None else 0,
            ):
                return
        raise
    finally:
        if invalidate_filesystem and fs is not None:
            invalidate_thread_local_hdfs_filesystem(filesystem_key, fs)
        if timeout_error is not None and not timeout_error.cleanup_scheduled:
            _defer_remote_session_cleanup(
                timeout_error,
                filesystem=fs,
                batch_iterator=batch_iterator,
                native_file=native_file,
                late_result_kind=timeout_late_result_kind,
                retained_resources=(parquet_file,),
                label=label,
                quarantine_limit=policy.quarantine_limit,
            )
            timeout_error.cleanup_scheduled = True
        elif untrackable_poison:
            _retain_abandoned_remote_session(
                fs,
                parquet_file,
                batch_iterator,
                native_file,
                label=label,
                quarantine_limit=policy.quarantine_limit,
            )
        else:
            _close_remote_parquet_session(
                batch_iterator=batch_iterator,
                parquet_file=parquet_file,
                native_file=native_file,
                filesystem=fs,
                filesystem_key=filesystem_key,
                policy=policy,
                label=label,
            )


def iter_parquet_record_batches(
    *,
    fs_path: str,
    filesystem: Any,
    lock_key: str,
    policy: RemoteIoPolicy,
    pq_module: Any,
    filesystem_key: str | None = None,
    stop_event: threading.Event | None = None,
    description: str | None = None,
    skip_tracker: RemoteSkipTracker | None = None,
    estimated_rows: int | None = None,
    **iter_kwargs: Any,
) -> Iterator[Any]:
    """Open and stream ``iter_batches`` under one remote IO session.

    Remote path uses thread-local FS + native_file + pre_buffer. Local path
    keeps a plain ``ParquetFile`` open. ``on_hdfs_failure: skip`` applies to
    body reads only.

    Critical resilience rule: never retry ``next()`` / ``close()`` on an Arrow
    batch generator after a timeout. Abandoned timeout threads still own the
    generator; in-place retries cause ``generator already executing`` and can
    wedge HDFS for a long time. Instead reopen a fresh session (before any
    batches were yielded) or abandon the remainder of the row group.
    """

    label = description or f"read parquet {lock_key}"
    kwargs = dict(iter_kwargs)
    if policy.enabled:
        kwargs["use_threads"] = False

    if not policy.enabled:
        yield from _iter_parquet_record_batches_local(
            fs_path=fs_path,
            filesystem=filesystem,
            lock_key=lock_key,
            pq_module=pq_module,
            stop_event=stop_event,
            kwargs=kwargs,
        )
        return

    resolved_key = filesystem_key
    if resolved_key is None and filesystem is not None:
        resolved_key = "hdfs://"
    resolved_key = resolved_key or "hdfs://"

    max_sessions = 1 + max(0, int(policy.retry_count))
    last_error: BaseException | None = None
    for session_idx in range(max_sessions):
        if stop_event is not None and stop_event.is_set():
            return
        if session_idx > 0 and last_error is not None:
            delay = float(policy.retry_base_sec) * (2 ** (session_idx - 1))
            logger.warning(
                "%s failed (%s); reopen session %d/%d in %.2fs",
                label,
                last_error,
                session_idx + 1,
                max_sessions,
                delay,
            )
            if _interruptible_sleep(delay, stop_event):
                return
        try:
            yield from _iter_parquet_record_batches_remote_session(
                fs_path=fs_path,
                filesystem=filesystem,
                filesystem_key=resolved_key,
                lock_key=lock_key,
                policy=policy,
                pq_module=pq_module,
                stop_event=stop_event,
                label=label,
                kwargs=kwargs,
                skip_tracker=skip_tracker,
                estimated_rows=estimated_rows,
            )
            return
        except _ParquetSessionReopen as reopen:
            last_error = reopen.error
            continue

    assert last_error is not None
    if maybe_skip_or_raise(
        last_error,
        policy,
        description=f"{label} (batch)",
        skip_tracker=skip_tracker,
        skipped_row_groups=1,
        skipped_rows=int(estimated_rows) if estimated_rows is not None else 0,
    ):
        return
    raise last_error



# Changing the planner algorithm changes which distributed rank sees each row
# group, so the version participates in the persisted diagnostic fingerprint.
_SHARD_PLANNER_VERSION = "lpt-v1"
# Prefetch queue end-of-stream marker; distinct from any real Arrow batch object.
_SENTINEL = object()
_LOCAL_FILESYSTEM_KEY = "file://"
_REMOTE_URI_SCHEMES = {"hdfs", "viewfs"}
_SUPPORTED_URI_SCHEMES = _REMOTE_URI_SCHEMES | {"file"}
_GLOB_META_CHARS = "*?["
_AUTO_SCENARIO_NAME = "__auto__"
_AUTO_SCENARIO_PRIOR_NAME = "scenario_prior_scene_id_hn"


@dataclass(frozen=True)
class ParquetInputRef:
    """One discovered Parquet file on a concrete PyArrow filesystem."""

    canonical_uri: str
    filesystem_key: str = field(compare=False)
    fs_path: str = field(compare=False)
    filesystem: Any = field(compare=False, hash=False, repr=False)

    def __str__(self) -> str:
        return self.canonical_uri


@dataclass(frozen=True)
class ParquetAdapterContext:
    """Context passed to external Parquet preprocessing adapters.

    Adapters receive raw Arrow tables and must return flat Arrow tables that
    satisfy the same one-row-per-sample contract as ``flat_parquet``.
    """

    split_name: str
    required_columns: tuple[str, ...]
    options: Mapping[str, Any]
    # Trusted production inputs validate one raw/flat sample, then avoid
    # diagnostic per-row/per-token checks on complete batches.
    trusted_input: bool = False
    # Built-in adapters may cache an immutable execution plan here. Keeping
    # this private cache on the context avoids reparsing hundreds of configured
    # column names for every Arrow record batch.
    _runtime_cache: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
        repr=False,
    )


# ---------------------------------------------------------------------------
# Parquet I/O: discovery, filesystem, schema validation, and column planning
# ---------------------------------------------------------------------------


def _require_pyarrow() -> tuple[Any, Any, Any, Any]:
    """Import optional Arrow dependencies only when the data pipeline is used.

    PyArrow is not imported at module load time so config-only workflows work
    without it installed. Returns ``(pa, pc, ds, pq)`` for callers to unpack.
    """
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "parquet-native data loading requires pyarrow; install it in the runtime environment"
        ) from error
    return pa, pc, ds, pq


def _require_pyarrow_fs() -> Any:
    """Import PyArrow filesystem support only when input discovery is used."""
    try:
        import pyarrow.fs as pafs
    except ImportError as error:
        raise RuntimeError(
            "parquet-native data loading requires pyarrow filesystem support; "
            "install pyarrow in the runtime environment"
        ) from error
    return pafs


def _looks_like_uri(item: str) -> bool:
    return "://" in item or item.startswith("file:")


def _input_uri_scheme(item: str) -> str:
    if not _looks_like_uri(item):
        return ""
    return urlsplit(item).scheme.lower()


def _split_uri_without_query_or_fragment(item: str) -> Any:
    parsed = urlsplit(item)
    if parsed.query or parsed.fragment:
        raise ValueError(
            f"parquet input URI must not include query or fragment: {item!r}"
        )
    return parsed


def _normalize_remote_path(path: str) -> str:
    normalized = path or "/"
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    while len(normalized) > 1 and normalized.endswith("/"):
        normalized = normalized[:-1]
    return normalized


def _remote_authority(parsed: Any, item: str) -> str:
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"invalid port in parquet input URI {item!r}") from error
    if parsed.netloc and parsed.hostname is None:
        raise ValueError(f"invalid parquet input URI authority: {item!r}")

    username = parsed.username
    password = parsed.password
    userinfo = ""
    if username is not None:
        userinfo = username
        if password is not None:
            userinfo += f":{password}"
        userinfo += "@"

    host = (parsed.hostname or "").lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port_text = f":{port}" if port is not None else ""
    return f"{userinfo}{host}{port_text}"


def _canonical_remote_uri(scheme: str, authority: str, fs_path: str) -> str:
    path = _normalize_remote_path(fs_path)
    if authority:
        return f"{scheme}://{authority}{path}"
    return f"{scheme}://{path}"


@dataclass(frozen=True)
class _RemoteInput:
    scheme: str
    authority: str
    filesystem_key: str
    canonical_uri: str
    fs_path: str
    filesystem: Any


def _remote_input_from_uri(item: str, filesystems: dict[str, Any]) -> _RemoteInput:
    pafs = _require_pyarrow_fs()
    parsed = _split_uri_without_query_or_fragment(item)
    scheme = parsed.scheme.lower()
    if scheme not in _REMOTE_URI_SCHEMES:
        raise ValueError(
            f"unsupported parquet input URI scheme {parsed.scheme!r}; "
            "supported URI schemes are file, hdfs, and viewfs"
        )

    authority = _remote_authority(parsed, item)
    filesystem_key = f"{scheme}://{authority}" if authority else f"{scheme}://"
    fs_path = _normalize_remote_path(parsed.path)
    canonical_uri = _canonical_remote_uri(scheme, authority, fs_path)
    filesystem = filesystems.get(filesystem_key)
    if filesystem is None:
        filesystem, parsed_fs_path = pafs.FileSystem.from_uri(canonical_uri)
        filesystems[filesystem_key] = filesystem
        fs_path = _normalize_remote_path(parsed_fs_path or fs_path)
    return _RemoteInput(
        scheme=scheme,
        authority=authority,
        filesystem_key=filesystem_key,
        canonical_uri=canonical_uri,
        fs_path=fs_path,
        filesystem=filesystem,
    )


def _local_input_pattern(item: str) -> str:
    if _input_uri_scheme(item) != "file":
        return item
    parsed = _split_uri_without_query_or_fragment(item)
    if parsed.netloc not in {"", "localhost"}:
        raise ValueError(
            f"only local file:// parquet input URIs are supported: {item!r}"
        )
    return unquote(parsed.path)


def _local_ref(path: Path, filesystem: Any) -> ParquetInputRef:
    resolved = path.resolve()
    canonical_uri = str(resolved)
    return ParquetInputRef(
        canonical_uri=canonical_uri,
        filesystem_key=_LOCAL_FILESYSTEM_KEY,
        fs_path=canonical_uri,
        filesystem=filesystem,
    )


def _discover_local_input(item: str, filesystem: Any) -> list[ParquetInputRef]:
    local_item = _local_input_pattern(item)
    path = Path(local_item)
    if path.is_dir():
        return [
            _local_ref(match, filesystem) for match in sorted(path.rglob("*.parquet"))
        ]
    matches = sorted(Path(match) for match in glob.glob(local_item, recursive=True))
    if matches:
        return [_local_ref(match, filesystem) for match in matches if match.is_file()]
    if path.is_file():
        return [_local_ref(path, filesystem)]
    raise FileNotFoundError(f"no parquet files matched input {item!r}")


def _has_glob_meta(value: str) -> bool:
    return any(char in value for char in _GLOB_META_CHARS)


def _remote_glob_base_dir(pattern_path: str) -> str:
    parts = pattern_path.split("/")
    base_parts: list[str] = []
    for index, segment in enumerate(parts):
        if index == 0 and segment == "":
            base_parts.append(segment)
            continue
        if segment == "**" or _has_glob_meta(segment):
            break
        base_parts.append(segment)
    if not base_parts or base_parts == [""]:
        return "/"
    return _normalize_remote_path("/".join(base_parts))


def _posix_segments(path: str) -> list[str]:
    stripped = path.strip("/")
    if not stripped:
        return []
    return [segment for segment in stripped.split("/") if segment]


def _match_remote_glob(path: str, pattern: str) -> bool:
    path_segments = _posix_segments(path)
    pattern_segments = _posix_segments(pattern)

    def match(path_index: int, pattern_index: int) -> bool:
        if pattern_index == len(pattern_segments):
            return path_index == len(path_segments)
        pattern_segment = pattern_segments[pattern_index]
        if pattern_segment == "**":
            return match(path_index, pattern_index + 1) or (
                path_index < len(path_segments) and match(path_index + 1, pattern_index)
            )
        if path_index >= len(path_segments):
            return False
        if not fnmatch.fnmatchcase(path_segments[path_index], pattern_segment):
            return False
        return match(path_index + 1, pattern_index + 1)

    return match(0, 0)


def _remote_ref(remote: _RemoteInput, fs_path: str) -> ParquetInputRef:
    normalized_path = _normalize_remote_path(fs_path)
    return ParquetInputRef(
        canonical_uri=_canonical_remote_uri(
            remote.scheme, remote.authority, normalized_path
        ),
        filesystem_key=remote.filesystem_key,
        fs_path=normalized_path,
        filesystem=remote.filesystem,
    )


def _discover_remote_directory(
    remote: _RemoteInput,
    *,
    list_timeout_sec: float | None = None,
) -> list[ParquetInputRef]:
    pafs = _require_pyarrow_fs()
    selector = pafs.FileSelector(remote.fs_path, recursive=True)

    def list_infos() -> list[Any]:
        return list(remote.filesystem.get_file_info(selector))

    timeout = None if list_timeout_sec is None else float(list_timeout_sec)
    infos = (
        list_infos()
        if timeout is None or timeout <= 0
        else call_with_timeout(
            list_infos,
            timeout,
            description=f"list hdfs directory {remote.canonical_uri}",
        )
    )
    return [
        _remote_ref(remote, info.path)
        for info in infos
        if info.type == pafs.FileType.File and info.path.endswith(".parquet")
    ]


def _discover_remote_glob(
    remote: _RemoteInput,
    item: str,
    *,
    list_timeout_sec: float | None = None,
) -> list[ParquetInputRef]:
    pafs = _require_pyarrow_fs()
    base_dir = _remote_glob_base_dir(remote.fs_path)
    timeout = None if list_timeout_sec is None else float(list_timeout_sec)

    def list_base_and_matches() -> tuple[Any, list[Any]]:
        base_info = remote.filesystem.get_file_info(base_dir)
        if base_info.type != pafs.FileType.Directory:
            raise FileNotFoundError(f"no parquet files matched input {item!r}")
        selector = pafs.FileSelector(base_dir, recursive=True)
        return base_info, list(remote.filesystem.get_file_info(selector))

    if timeout is None or timeout <= 0:
        _base_info, infos = list_base_and_matches()
    else:
        _base_info, infos = call_with_timeout(
            list_base_and_matches,
            timeout,
            description=f"list hdfs glob {item}",
        )

    refs: list[ParquetInputRef] = []
    matched_any = False
    for info in infos:
        if not _match_remote_glob(_normalize_remote_path(info.path), remote.fs_path):
            continue
        matched_any = True
        if info.type == pafs.FileType.File:
            refs.append(_remote_ref(remote, info.path))
    if not refs and not matched_any:
        raise FileNotFoundError(f"no parquet files matched input {item!r}")
    return refs


def _discover_remote_input(
    item: str,
    filesystems: dict[str, Any],
    *,
    list_timeout_sec: float | None = None,
) -> list[ParquetInputRef]:
    pafs = _require_pyarrow_fs()
    remote = _remote_input_from_uri(item, filesystems)
    if _has_glob_meta(remote.fs_path):
        return _discover_remote_glob(
            remote,
            item,
            list_timeout_sec=list_timeout_sec,
        )

    def stat_path() -> Any:
        return remote.filesystem.get_file_info(remote.fs_path)

    timeout = None if list_timeout_sec is None else float(list_timeout_sec)
    info = (
        stat_path()
        if timeout is None or timeout <= 0
        else call_with_timeout(
            stat_path,
            timeout,
            description=f"stat hdfs path {item}",
        )
    )
    if info.type == pafs.FileType.File:
        return [_remote_ref(remote, info.path)]
    if info.type == pafs.FileType.Directory:
        return _discover_remote_directory(
            remote,
            list_timeout_sec=list_timeout_sec,
        )
    raise FileNotFoundError(f"no parquet files matched input {item!r}")


def _unique_sorted_refs(refs: Iterable[ParquetInputRef]) -> list[ParquetInputRef]:
    unique = {ref.canonical_uri: ref for ref in refs}
    return sorted(unique.values(), key=lambda ref: ref.canonical_uri)


def discover_parquet_inputs(
    inputs: Iterable[str | Path],
    *,
    remote_list_timeout_sec: float | None = None,
) -> list[ParquetInputRef]:
    """Resolve parquet files from local paths or HDFS/viewfs URLs.

    Local inputs keep the existing file, directory, and Python glob behavior.
    HDFS/viewfs inputs use PyArrow filesystem discovery and support common
    POSIX-style glob segments, including ``**`` as a full path segment.
    ``remote_list_timeout_sec`` bounds recursive listing / stat calls so a hung
    NameNode RPC cannot stall host-prepare startup indefinitely.
    """
    refs: list[ParquetInputRef] = []
    remote_filesystems: dict[str, Any] = {}
    local_filesystem: Any | None = None
    for raw_item in inputs:
        item = os.fspath(raw_item)
        scheme = _input_uri_scheme(item)
        if scheme and scheme not in _SUPPORTED_URI_SCHEMES:
            raise ValueError(
                f"unsupported parquet input URI scheme {scheme!r}; "
                "supported URI schemes are file, hdfs, and viewfs"
            )
        if scheme in _REMOTE_URI_SCHEMES:
            refs.extend(
                _discover_remote_input(
                    item,
                    remote_filesystems,
                    list_timeout_sec=remote_list_timeout_sec,
                )
            )
            note_io_progress()
            continue
        if local_filesystem is None:
            local_filesystem = _require_pyarrow_fs().LocalFileSystem()
        refs.extend(_discover_local_input(item, local_filesystem))
        note_io_progress()

    unique_refs = _unique_sorted_refs(refs)
    if not unique_refs:
        raise FileNotFoundError("no parquet files discovered")
    filesystem_keys = {ref.filesystem_key for ref in unique_refs}
    if len(filesystem_keys) > 1:
        raise ValueError(
            "parquet inputs for one split must use a single filesystem; got "
            + ", ".join(sorted(filesystem_keys))
        )
    return unique_refs


def schema_fingerprint(schema: Any) -> str:
    """Hash logical field names/types/nullability; ignores physical layout."""
    payload = "\n".join(
        f"{field.name}:{field.type}:{field.nullable}" for field in schema
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _coerce_parquet_input_ref(path: str | Path | ParquetInputRef) -> ParquetInputRef:
    if isinstance(path, ParquetInputRef):
        return path
    refs = discover_parquet_inputs([os.fspath(path)])
    if len(refs) != 1:
        raise ValueError(
            f"expected exactly one parquet file, discovered {len(refs)} from {path!r}"
        )
    return refs[0]


def parquet_schema(
    path: str | Path | ParquetInputRef,
    policy: RemoteIoPolicy | None = None,
) -> Any:
    """Read Parquet schema metadata only; does not scan row data."""
    _pa, _pc, _ds, pq = _require_pyarrow()
    ref = _coerce_parquet_input_ref(path)
    remote = any(
        str(ref.filesystem_key).startswith(f"{scheme}://")
        for scheme in _REMOTE_URI_SCHEMES
    )
    if policy is None:
        io_policy = RemoteIoPolicy.from_reader(
            type(
                "Reader",
                (),
                {
                    "hdfs_op_timeout": 30.0,
                    "hdfs_open_timeout": 120.0,
                    "hdfs_retry_count": 5,
                    "hdfs_retry_base_sec": 0.5,
                    "hdfs_file_lock": True,
                    "on_hdfs_failure": "fail",
                    "worker_stagger_sec": 0.0,
                    "hdfs_pre_buffer": False,
                    "hdfs_close_timeout": 5.0,
                },
            )(),
            remote=remote,
        )
    else:
        io_policy = policy
    if not io_policy.enabled:
        return pq.read_schema(ref.fs_path, filesystem=ref.filesystem)

    schema_policy = replace(io_policy, pre_buffer=False)
    return _run_remote_parquet_operation(
        filesystem_key=ref.filesystem_key,
        prototype=ref.filesystem,
        fs_path=ref.fs_path,
        lock_key=ref.canonical_uri,
        policy=schema_policy,
        pq_module=pq,
        description=f"read schema {ref.canonical_uri}",
        operation=lambda parquet_file: parquet_file.schema_arrow,
    )


def validate_matching_schemas(
    paths: Iterable[str | Path | ParquetInputRef],
    policy: RemoteIoPolicy | None = None,
) -> str:
    """Require identical schemas across files; return the shared fingerprint."""
    refs = [_coerce_parquet_input_ref(path) for path in paths]
    if not refs:
        raise ValueError("paths must not be empty")
    fingerprints: dict[ParquetInputRef, str] = {}
    for ref in refs:
        fingerprints[ref] = schema_fingerprint(parquet_schema(ref, policy=policy))
        note_io_progress()
    expected = next(iter(fingerprints.values()))
    mismatched = [
        ref.canonical_uri
        for ref, fingerprint in fingerprints.items()
        if fingerprint != expected
    ]
    if mismatched:
        raise ValueError("parquet schema mismatch: " + ", ".join(mismatched))
    return expected


def _eager_schema_validation_refs(
    refs: list[ParquetInputRef],
    mode: str,
    sample_count: int,
) -> list[ParquetInputRef]:
    """Choose deterministic, evenly spaced files for startup validation."""

    if mode == "all" or len(refs) <= sample_count:
        return refs
    if mode != "sample":
        raise ValueError(f"unsupported eager schema validation mode {mode!r}")
    if sample_count == 1:
        return [refs[0]]
    last = len(refs) - 1
    indices = {
        round(index * last / (sample_count - 1)) for index in range(sample_count)
    }
    return [refs[index] for index in sorted(indices)]


def _configure_pyarrow_threads(
    pa: Any,
    num_workers: int,
    *,
    io_thread_count: int | None = None,
) -> None:
    """Align PyArrow CPU/IO threads with reader settings when set."""
    if num_workers <= 0 and io_thread_count is None:
        return
    if num_workers > 0 and hasattr(pa, "set_cpu_count"):
        pa.set_cpu_count(num_workers)
    resolved_io = io_thread_count if io_thread_count is not None else num_workers
    if resolved_io > 0 and hasattr(pa, "set_io_thread_count"):
        pa.set_io_thread_count(resolved_io)


def _put_queue_item(
    target_queue: queue.Queue[Any],
    item: Any,
    stop_event: threading.Event,
) -> bool:
    """Put into a bounded prefetch queue; back off when full or stopped."""
    while not stop_event.is_set():
        try:
            target_queue.put(item, timeout=0.05)
            return True
        except queue.Full:
            continue
    return False


class _ByteBudget:
    """A stoppable byte semaphore that admits one oversized item for progress."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, capacity)
        self.used = 0
        self.condition = threading.Condition()

    def acquire(self, amount: int, stop_event: threading.Event) -> bool:
        amount = max(1, amount)
        with self.condition:
            while not stop_event.is_set():
                if self.used + amount <= self.capacity or self.used == 0:
                    self.used += amount
                    return True
                self.condition.wait(timeout=0.05)
        return False

    def release(self, amount: int) -> None:
        with self.condition:
            self.used -= max(1, amount)
            if self.used < 0:
                raise RuntimeError(
                    "prefetch byte budget was released more than reserved"
                )
            self.condition.notify_all()

    def wake_all(self) -> None:
        with self.condition:
            self.condition.notify_all()


def _sequence_source_columns(config: AppConfig) -> set[str]:
    """Collect Parquet source columns referenced by configured sequence fields."""
    columns: set[str] = set()
    for sequence in config.sequences:
        columns.update(field.source for field in sequence.fields)
    return columns


def required_columns_for_split(
    config: AppConfig,
    split: ParquetSplitConfig,
    extra_columns: Iterable[str] = (),
    *,
    require_labels: bool = True,
) -> list[str]:
    """Return the minimal physical columns needed to build one model batch.

    These are columns required after any adapter has converted raw Parquet to
    the flat contract. For ``flat_parquet`` they are also the scan columns.
    """
    columns: set[str] = set()
    sequence_columns = _sequence_source_columns(config)
    for feature in config.features:
        columns.add(feature.source)
    columns.update(sequence_columns)
    if require_labels:
        columns.update(split.labels.values())
        columns.update(split.label_masks.values())
    columns.update(split.prediction_keys.values())
    if split.request_id:
        columns.add(split.request_id)
    if split.group_id:
        columns.add(split.group_id)
    if config.scenarios.source:
        columns.add(config.scenarios.source)
    columns.update(extra_columns)
    return sorted(columns)


def _scan_columns_for_split(
    split: ParquetSplitConfig, flat_columns: list[str]
) -> list[str]:
    """Return raw Parquet scan columns for a split.

    ``ParquetScanner`` interprets an empty list as "read all columns" when
    pruning is enabled, which is the right fallback for adapters that do not
    declare ``input_columns``.
    """
    if split.format == "adapter_parquet":
        if split.adapter is None:
            raise ValueError("adapter_parquet split requires adapter config")
        if split.adapter.input_columns is None:
            return []
        columns = [
            *split.adapter.input_columns,
            *split.adapter.optional_input_columns,
        ]
        # Inference over req files commonly has no labels. Adapter input lists
        # describe the superset used by train/evaluate, so omit raw label columns
        # whenever the flat contract does not request them.
        omitted_labels = (
            set(split.labels.values()) | set(split.label_masks.values())
        ) - set(flat_columns)
        return [column for column in columns if column not in omitted_labels]
    return flat_columns


def _optional_scan_columns_for_split(split: ParquetSplitConfig) -> tuple[str, ...]:
    if split.format != "adapter_parquet" or split.adapter is None:
        return ()
    return split.adapter.optional_input_columns


# ---------------------------------------------------------------------------
# Shard planning: metadata cache and LPT assignment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanStats:
    """Summary counters returned by ``ParquetScanner.scan_stats``."""

    files: int
    record_batches: int
    rows: int


@dataclass(frozen=True)
class _RowGroupMetadata:
    """Per-row-group stats read from the Parquet footer (no row data scanned)."""

    input_ref: ParquetInputRef
    local_row_group_index: int
    num_rows: int
    # Sum of compressed sizes for scan columns; None if any column lacks metadata.
    compressed_bytes: int | None


@dataclass(frozen=True)
class _FileMetadataCache:
    """Cached footer metadata for one Parquet file."""

    schema: Any
    row_groups: tuple[_RowGroupMetadata, ...]


@dataclass(frozen=True)
class _RowGroupWorkItem:
    """One row group after LPT assignment to a distributed rank."""

    input_ref: ParquetInputRef
    local_row_group_index: int
    weight: int  # compressed_bytes or num_rows, depending on the plan
    rank: int
    scan_order: int  # global order before LPT; restores deterministic yield order
    num_rows: int = 0


@dataclass(frozen=True)
class _ShardPlan:
    """Immutable LPT shard plan plus a diagnostic fingerprint."""

    requested_shard_unit: str
    effective_shard_unit: str
    world_size: int
    scan_columns: tuple[str, ...] | None
    weight_source: str
    work_items: tuple[_RowGroupWorkItem, ...]
    fingerprint: str


def _metadata_worker_count(
    num_workers: int,
    file_count: int,
    *,
    remote: bool = False,
) -> int:
    """Cap parallel metadata readers by file count and a hard limit of 16."""
    if remote:
        # Footer opens on HDFS are NameNode-heavy; keep them serial per rank.
        return 1 if file_count else 0
    configured = num_workers if num_workers > 0 else min(8, os.cpu_count() or 1)
    return min(file_count, configured, 16)


def _refs_are_remote(refs: Sequence[ParquetInputRef]) -> bool:
    if not refs:
        return False
    key = str(refs[0].filesystem_key)
    return any(key.startswith(f"{scheme}://") for scheme in _REMOTE_URI_SCHEMES)


def _load_file_metadata_cache(
    ref: ParquetInputRef,
    scan_columns: list[str] | None,
    policy: RemoteIoPolicy | None = None,
) -> _FileMetadataCache:
    """Read row-group row counts and compressed-byte weights from the footer only."""
    io_policy = policy or RemoteIoPolicy.disabled()
    _pa, _pc, _ds, pq = _require_pyarrow()

    def build_cache(parquet_file: Any) -> _FileMetadataCache:
        schema = parquet_file.schema_arrow
        column_names = scan_columns if scan_columns is not None else list(schema.names)
        column_indices = {
            column_name: schema.get_field_index(column_name)
            for column_name in column_names
        }
        row_groups: list[_RowGroupMetadata] = []
        for local_row_group_index in range(parquet_file.metadata.num_row_groups):
            row_group = parquet_file.metadata.row_group(local_row_group_index)
            compressed_bytes = 0
            missing_bytes = False
            for column_name in column_names:
                column_index = column_indices[column_name]
                if column_index < 0:
                    missing_bytes = True
                    break
                column_meta = row_group.column(column_index)
                if (
                    column_meta.total_compressed_size is None
                    or column_meta.total_compressed_size < 0
                ):
                    missing_bytes = True
                    break
                compressed_bytes += int(column_meta.total_compressed_size)
            row_groups.append(
                _RowGroupMetadata(
                    input_ref=ref,
                    local_row_group_index=local_row_group_index,
                    num_rows=row_group.num_rows,
                    compressed_bytes=None if missing_bytes else compressed_bytes,
                )
            )
        return _FileMetadataCache(schema=schema, row_groups=tuple(row_groups))

    # Footer planning must stay strict: skipping here would desync LPT shards.
    if io_policy.enabled:
        return _run_remote_parquet_operation(
            filesystem_key=ref.filesystem_key,
            prototype=ref.filesystem,
            fs_path=ref.fs_path,
            lock_key=ref.canonical_uri,
            policy=replace(io_policy, pre_buffer=False),
            pq_module=pq,
            description=f"load parquet metadata {ref.canonical_uri}",
            operation=build_cache,
        )

    def load_local() -> _FileMetadataCache:
        parquet_file = pq.ParquetFile(ref.fs_path, filesystem=ref.filesystem)
        return build_cache(parquet_file)

    return run_under_file_lock(
        load_local,
        lock_key=ref.canonical_uri,
        policy=io_policy,
        description=f"load parquet metadata {ref.canonical_uri}",
        timeout_sec=None,
    )


def _load_metadata_cache(
    paths: list[ParquetInputRef],
    scan_columns: list[str] | None,
    num_workers: int,
    policy: RemoteIoPolicy | None = None,
) -> dict[ParquetInputRef, _FileMetadataCache]:
    """Load per-file footer metadata, in parallel when beneficial."""
    io_policy = policy or RemoteIoPolicy.disabled()
    worker_count = _metadata_worker_count(
        num_workers,
        len(paths),
        remote=io_policy.enabled,
    )
    metadata_by_path: dict[ParquetInputRef, _FileMetadataCache] = {}
    if worker_count <= 1:
        for ref in paths:
            metadata_by_path[ref] = _load_file_metadata_cache(
                ref,
                scan_columns,
                io_policy,
            )
            note_io_progress()
        return metadata_by_path

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _load_file_metadata_cache, ref, scan_columns, io_policy
            ): ref
            for ref in paths
        }
        for future in as_completed(futures):
            metadata_by_path[futures[future]] = future.result()
            note_io_progress()
    return metadata_by_path


def _build_lpt_shard_plan(
    paths: list[ParquetInputRef],
    metadata_by_path: dict[ParquetInputRef, _FileMetadataCache],
    scan_columns: list[str] | None,
    world_size: int,
    requested_shard_unit: str,
    effective_shard_unit: str,
) -> _ShardPlan:
    """Assign row groups with deterministic longest-processing-time scheduling.

    Compressed byte size is the closest cheap proxy for scan work. If any row
    group lacks that metadata, the whole plan falls back to row counts so all
    assignments use comparable weights.
    """
    # Flatten all row groups in file order and assign a global scan_order.
    ordered_items: list[tuple[int, _RowGroupMetadata]] = []
    scan_order = 0
    for ref in paths:
        for row_group in metadata_by_path[ref].row_groups:
            ordered_items.append((scan_order, row_group))
            scan_order += 1

    # Prefer compressed-byte weights; fall back to row counts if any RG lacks bytes.
    use_row_weights = all(
        item.compressed_bytes is not None for _, item in ordered_items
    )
    weight_source = "compressed_bytes" if use_row_weights else "num_rows"
    weighted_items: list[tuple[int, _RowGroupMetadata, int]] = []
    for order, item in ordered_items:
        if use_row_weights:
            weight = int(item.compressed_bytes)  # type: ignore[arg-type]
        else:
            weight = item.num_rows
        weighted_items.append((order, item, weight))

    # LPT: heaviest row groups first, each to the currently lightest rank.
    weighted_items.sort(
        key=lambda entry: (
            -entry[2],
            entry[1].input_ref.canonical_uri,
            entry[1].local_row_group_index,
        )
    )

    rank_totals = [0] * world_size
    rank_counts = [0] * world_size
    assignments: list[tuple[int, _RowGroupMetadata, int, int]] = []
    for order, item, weight in weighted_items:
        rank = min(
            range(world_size),
            key=lambda candidate: (
                rank_totals[candidate],
                rank_counts[candidate],
                candidate,
            ),
        )
        rank_totals[rank] += weight
        rank_counts[rank] += 1
        assignments.append((order, item, weight, rank))

    # Restore global scan order for deterministic iteration within each rank.
    assignments.sort(key=lambda entry: entry[0])
    work_items = tuple(
        _RowGroupWorkItem(
            input_ref=item.input_ref,
            local_row_group_index=item.local_row_group_index,
            weight=weight,
            rank=rank,
            scan_order=order,
            num_rows=int(item.num_rows),
        )
        for order, item, weight, rank in assignments
    )

    column_label = ",".join(scan_columns) if scan_columns is not None else "*"
    assignment_lines = [
        f"{item.rank}:{item.input_ref.canonical_uri}:{item.local_row_group_index}:{item.weight}"
        for item in work_items
    ]
    # Persist assignment details for reproducibility and debugging.
    fingerprint_payload = "\n".join(
        [
            f"planner={_SHARD_PLANNER_VERSION}",
            f"requested={requested_shard_unit}",
            f"effective={effective_shard_unit}",
            f"world_size={world_size}",
            f"columns={column_label}",
            f"weight_source={weight_source}",
            *assignment_lines,
        ]
    )
    fingerprint = sha256(fingerprint_payload.encode("utf-8")).hexdigest()
    return _ShardPlan(
        requested_shard_unit=requested_shard_unit,
        effective_shard_unit=effective_shard_unit,
        world_size=world_size,
        scan_columns=tuple(scan_columns) if scan_columns is not None else None,
        weight_source=weight_source,
        work_items=work_items,
        fingerprint=fingerprint,
    )


@dataclass
class _PrefetchSlot:
    """One bounded output queue plus its persistent reader thread."""

    index: int
    queue: queue.Queue[Any]
    byte_budget: _ByteBudget
    thread: threading.Thread | None = None
    error: BaseException | None = None


@dataclass(frozen=True)
class _QueuedRecordBatch:
    value: Any
    nbytes: int


def _drain_prefetch_slot(slot: _PrefetchSlot) -> None:
    while not slot.queue.empty():
        item = slot.queue.get_nowait()
        if isinstance(item, _QueuedRecordBatch):
            slot.byte_budget.release(item.nbytes)


class _ClosableIterator:
    """Iterator wrapper that signals prefetch workers to stop on ``close()``."""

    def __init__(self, generator: Iterator[Any], stop_event: threading.Event) -> None:
        self._generator = generator
        self._stop_event = stop_event

    def __iter__(self) -> _ClosableIterator:
        return self

    def __next__(self) -> Any:
        return next(self._generator)

    def close(self) -> None:
        """Stop prefetch threads and close the underlying generator if possible."""
        self._stop_event.set()
        if isinstance(self._generator, GeneratorType):
            self._generator.close()


# ---------------------------------------------------------------------------
# Scanning: sharding, prefetch, and Arrow batch streaming
# ---------------------------------------------------------------------------


class ParquetScanner:
    """Stream a configured Parquet split for one distributed worker.

    File sharding uses deterministic slicing. Row-group sharding uses an LPT
    plan so differently sized row groups are distributed more evenly while
    retaining deterministic scan order inside each rank.
    """

    def __init__(
        self,
        split: ParquetSplitConfig,
        columns: list[str],
        shard_rank: int = 0,
        shard_world_size: int = 1,
        optional_columns: Iterable[str] = (),
    ) -> None:
        self.split = split
        self.columns = list(columns)
        self.optional_columns = frozenset(optional_columns)
        unknown_optional = self.optional_columns - set(self.columns)
        if unknown_optional:
            raise ValueError(
                "optional parquet scan columns must also be present in columns: "
                + ", ".join(sorted(unknown_optional))
            )
        self.shard_rank = shard_rank
        self.shard_world_size = shard_world_size
        if not 0 <= shard_rank < shard_world_size:
            raise ValueError("shard_rank must be in [0, shard_world_size)")
        requested_shard_unit = self._requested_shard_unit()
        if requested_shard_unit not in {"file", "row_group", "record_batch"}:
            raise ValueError(
                f"unsupported reader.shard_unit {requested_shard_unit!r}; "
                "expected file, row_group, or record_batch"
            )
        if shard_world_size > 1 and self._effective_shard_unit() not in {
            "file",
            "row_group",
        }:
            raise ValueError(
                f"unsupported reader.shard_unit {requested_shard_unit!r} "
                "for distributed scanning"
            )
        self.all_paths = discover_parquet_inputs(
            split.inputs,
            remote_list_timeout_sec=(
                float(split.reader.hdfs_open_timeout)
                if any(
                    str(item).startswith(("hdfs://", "viewfs://"))
                    for item in split.inputs
                )
                else None
            ),
        )
        self._io_policy = RemoteIoPolicy.from_reader(
            split.reader,
            remote=_refs_are_remote(self.all_paths),
        )
        self._skip_tracker = RemoteSkipTracker(
            max_skipped_row_groups=getattr(
                split.reader, "hdfs_max_skipped_row_groups", 64
            ),
            max_skipped_rows=getattr(split.reader, "hdfs_max_skipped_rows", 2_000_000),
        )
        if self._io_policy.enabled:
            apply_worker_stagger(shard_rank, split.reader.worker_stagger_sec)
        global_schema_refs = _eager_schema_validation_refs(
            self.all_paths,
            split.reader.eager_schema_validation,
            split.reader.schema_validation_samples,
        )
        if shard_world_size > 1 and len(global_schema_refs) > 1:
            # Validate the chosen global set collectively instead of making
            # every DDP rank reopen the same remote footers. Every rank also
            # checks the common anchor, so fingerprints remain transitively
            # comparable across rank-local subsets.
            anchor = global_schema_refs[0]
            local_refs = global_schema_refs[shard_rank::shard_world_size]
            schema_refs = list(dict.fromkeys([anchor, *local_refs]))
        else:
            schema_refs = global_schema_refs
        validate_matching_schemas(schema_refs, policy=self._io_policy)
        if self.columns:
            # Auto-detecting adapters may support layout-specific raw columns
            # (the agg indices are absent from req files). Project optional
            # columns only when the split schema contains them, while still
            # failing early for a missing mandatory input.
            schema_names = set(
                parquet_schema(schema_refs[0], policy=self._io_policy).names
            )
            missing = [
                column
                for column in self.columns
                if column not in self.optional_columns and column not in schema_names
            ]
            if missing:
                raise ValueError(
                    "parquet schema is missing required scan column(s): "
                    + ", ".join(missing)
                )
            self.columns = [
                column
                for column in self.columns
                if column not in self.optional_columns or column in schema_names
            ]
        pa, _pc, _ds, _pq = _require_pyarrow()
        if self._io_policy.enabled:
            # IO pool sized to eason-style prefetch workers (pre_buffer only).
            # CPU pool can stay larger for decode after copy-off-HDFS.
            io_workers = scaled_hdfs_prefetch_workers(
                world_size=shard_world_size,
                num_workers=split.reader.num_workers,
                prefetch_batches=max(1, split.reader.prefetch_batches),
                work_item_count=10**9,
                remote=True,
            )
            _configure_pyarrow_threads(
                pa,
                max(split.reader.num_workers, 1),
                io_thread_count=max(1, io_workers),
            )
        else:
            _configure_pyarrow_threads(pa, split.reader.num_workers)
        # File sharding: each rank scans a disjoint subset of paths.
        if shard_world_size > 1 and split.reader.shard_unit == "file":
            self.paths = self.all_paths[shard_rank::shard_world_size]
        else:
            # Row-group sharding keeps all paths visible; LPT picks work items per rank.
            self.paths = self.all_paths
        self._metadata_cache: dict[ParquetInputRef, _FileMetadataCache] | None = None
        self._shard_plan: _ShardPlan | None = None
        self._empty_rank_warning_emitted = False

    @property
    def shard_plan_fingerprint(self) -> str | None:
        """Return the LPT plan fingerprint, or None when file/dataset sharding is used."""
        if self._uses_lpt_row_group_sharding():
            return self._get_shard_plan().fingerprint
        return None

    def _requested_shard_unit(self) -> str:
        return self.split.reader.shard_unit

    def _effective_shard_unit(self) -> str:
        """Map ``record_batch`` to ``row_group`` under multi-rank for deterministic sharding."""
        requested = self._requested_shard_unit()
        if requested == "record_batch" and self.shard_world_size > 1:
            return "row_group"
        return requested

    def _uses_lpt_row_group_sharding(self) -> bool:
        return self._effective_shard_unit() == "row_group"

    def _scan_columns(self) -> list[str] | None:
        """Return pruned columns, or None to read every column in the file."""
        if not self.split.reader.columns_pruning:
            return None
        return self.columns or None

    def _reader_batch_size(self, default: int) -> int:
        return self.split.reader.scanner_batch_rows or default

    def _get_metadata_cache(self) -> dict[ParquetInputRef, _FileMetadataCache]:
        if self._metadata_cache is None:
            self._metadata_cache = _load_metadata_cache(
                self.all_paths,
                self._scan_columns(),
                self.split.reader.num_workers,
                self._io_policy,
            )
        return self._metadata_cache

    def _get_shard_plan(self) -> _ShardPlan:
        if self._shard_plan is not None:
            return self._shard_plan
        plan = _build_lpt_shard_plan(
            paths=self.all_paths,
            metadata_by_path=self._get_metadata_cache(),
            scan_columns=self._scan_columns(),
            world_size=self.shard_world_size,
            requested_shard_unit=self._requested_shard_unit(),
            effective_shard_unit=self._effective_shard_unit(),
        )
        self._shard_plan = plan
        self._maybe_warn_empty_ranks(plan)
        return plan

    def _maybe_warn_empty_ranks(self, plan: _ShardPlan) -> None:
        """Log once from rank 0 when LPT leaves some ranks with no row groups."""
        if self._empty_rank_warning_emitted:
            return
        if self.shard_world_size <= 1 or self.shard_rank != 0:
            return
        counts = defaultdict(int)
        for item in plan.work_items:
            counts[item.rank] += 1
        empty_rank_count = sum(
            1 for rank in range(self.shard_world_size) if counts[rank] == 0
        )
        if empty_rank_count == 0:
            return
        effective_rank_count = self.shard_world_size - empty_rank_count
        logger.warning(
            "parquet shard plan leaves %d empty rank(s) out of %d for %d work units "
            "(effective ranks=%d, requested=%s, effective=%s)",
            empty_rank_count,
            self.shard_world_size,
            len(plan.work_items),
            effective_rank_count,
            plan.requested_shard_unit,
            plan.effective_shard_unit,
        )
        self._empty_rank_warning_emitted = True

    def _assigned_row_group_work_items(self) -> list[_RowGroupWorkItem]:
        """Row groups owned by this rank, sorted for deterministic in-rank scan order."""
        plan = self._get_shard_plan()
        assigned = [item for item in plan.work_items if item.rank == self.shard_rank]
        assigned.sort(
            key=lambda item: (item.input_ref.canonical_uri, item.local_row_group_index)
        )
        return assigned

    def _prefetch_active_workers(self, row_group_count: int) -> int:
        """Bound concurrent row-group readers; on HDFS scale with GPU count."""
        return scaled_hdfs_prefetch_workers(
            world_size=self.shard_world_size,
            num_workers=self.split.reader.num_workers,
            prefetch_batches=self.split.reader.prefetch_batches,
            work_item_count=row_group_count,
            remote=self._filesystem_is_remote(),
        )

    def _prefetch_queue_capacities(self, active_workers: int) -> list[int]:
        """Split ``prefetch_batches`` across workers as evenly as possible."""
        prefetch_batches = self.split.reader.prefetch_batches
        base, remainder = divmod(prefetch_batches, active_workers)
        return [
            base + (1 if index < remainder else 0) for index in range(active_workers)
        ]

    def _iter_row_group_record_batches_sync(
        self,
        work_items: list[_RowGroupWorkItem],
        stop_event: threading.Event,
    ) -> Iterator[Any]:
        """Sequentially read each assigned row group into Arrow record batches."""
        _pa, _pc, _ds, pq = _require_pyarrow()
        batch_size = self._reader_batch_size(default=65536)
        scan_columns = self._scan_columns()
        for work_item in work_items:
            if stop_event.is_set():
                return
            ref = work_item.input_ref
            yield from iter_parquet_record_batches(
                fs_path=ref.fs_path,
                filesystem=ref.filesystem,
                filesystem_key=ref.filesystem_key,
                lock_key=ref.canonical_uri,
                policy=self._io_policy,
                pq_module=pq,
                stop_event=stop_event,
                description=(
                    f"row group {work_item.local_row_group_index} of "
                    f"{ref.canonical_uri}"
                ),
                skip_tracker=self._skip_tracker,
                estimated_rows=int(work_item.num_rows),
                batch_size=batch_size,
                row_groups=[work_item.local_row_group_index],
                columns=scan_columns,
                # Sync path: local may use Arrow inner threads; HDFS keeps them off.
                use_threads=not self._io_policy.enabled,
            )

    def _row_group_worker(
        self,
        work_item: _RowGroupWorkItem,
        slot: _PrefetchSlot,
        stop_event: threading.Event,
    ) -> None:
        """Background worker: stream one row group into a bounded prefetch queue."""
        try:
            _pa, _pc, _ds, pq = _require_pyarrow()
            batch_size = self._reader_batch_size(default=65536)
            scan_columns = self._scan_columns()
            ref = work_item.input_ref
            for batch in iter_parquet_record_batches(
                fs_path=ref.fs_path,
                filesystem=ref.filesystem,
                filesystem_key=ref.filesystem_key,
                lock_key=ref.canonical_uri,
                policy=self._io_policy,
                pq_module=pq,
                stop_event=stop_event,
                description=(
                    f"prefetch row group {work_item.local_row_group_index} of "
                    f"{ref.canonical_uri}"
                ),
                skip_tracker=self._skip_tracker,
                estimated_rows=int(work_item.num_rows),
                batch_size=batch_size,
                row_groups=[work_item.local_row_group_index],
                columns=scan_columns,
                # Prefetch workers: HDFS keeps Arrow inner threads off (JNI).
                # Local keeps them on for decode; outer threads cover cross-RG IO.
                use_threads=not self._io_policy.enabled,
            ):
                if stop_event.is_set():
                    return
                batch_bytes = max(1, int(getattr(batch, "nbytes", 0)))
                if not slot.byte_budget.acquire(batch_bytes, stop_event):
                    return
                queued = _QueuedRecordBatch(batch, batch_bytes)
                if not _put_queue_item(slot.queue, queued, stop_event):
                    slot.byte_budget.release(batch_bytes)
                    return
        except BaseException as error:
            slot.error = error
        finally:
            _put_queue_item(slot.queue, _SENTINEL, stop_event)

    def _iter_record_batches_with_persistent_prefetch(
        self,
        work_items: Sequence[Any],
        stop_event: threading.Event,
        *,
        active_workers: int,
        worker: Callable[[Any, _PrefetchSlot, threading.Event], None],
        thread_name_prefix: str,
    ) -> Iterator[Any]:
        """Stream ordered work through a fixed set of long-lived daemon workers.

        HDFS clients are cached in thread-local storage.  Creating a fresh
        thread for every row group/file destroys that cache when the thread
        exits; with Hadoop implementations that share an underlying client,
        one wrapper's destruction can close a DFSClient still used by another
        prefetch worker.  Persistent workers bound client lifetime to the whole
        scan while preserving the existing bounded, deterministic output order.
        """

        if not work_items or active_workers <= 0:
            return

        capacities = self._prefetch_queue_capacities(active_workers)
        byte_capacity = max(1, self.split.reader.max_prefetch_bytes // active_workers)
        slots = [
            _PrefetchSlot(
                index=index,
                queue=queue.Queue(maxsize=capacity),
                byte_budget=_ByteBudget(byte_capacity),
            )
            for index, capacity in enumerate(capacities)
        ]
        task_queues = [queue.Queue(maxsize=1) for _ in slots]
        slot_for_item: dict[int, _PrefetchSlot] = {}

        def worker_loop(slot: _PrefetchSlot) -> None:
            task_queue = task_queues[slot.index]
            while not stop_event.is_set():
                try:
                    work_item = task_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    worker(work_item, slot, stop_event)
                except BaseException as error:  # defensive: workers normally capture
                    slot.error = error
                    _put_queue_item(slot.queue, _SENTINEL, stop_event)

        for slot in slots:
            slot.thread = threading.Thread(
                target=worker_loop,
                args=(slot,),
                name=f"{thread_name_prefix}-slot-{slot.index}",
                daemon=True,
            )
            slot.thread.start()

        next_assign_index = 0

        def assign(slot: _PrefetchSlot) -> None:
            nonlocal next_assign_index
            if next_assign_index >= len(work_items) or stop_event.is_set():
                return
            work_index = next_assign_index
            next_assign_index += 1
            slot.error = None
            slot_for_item[work_index] = slot
            task_queues[slot.index].put(work_items[work_index])

        for slot in slots:
            assign(slot)

        try:
            for work_index in range(len(work_items)):
                if stop_event.is_set():
                    return
                slot = slot_for_item.pop(work_index)
                while True:
                    if stop_event.is_set() and slot.queue.empty():
                        return
                    try:
                        item = slot.queue.get(timeout=0.1)
                    except queue.Empty:
                        if (
                            slot.thread is not None
                            and not slot.thread.is_alive()
                            and slot.queue.empty()
                        ):
                            if slot.error is not None:
                                raise slot.error
                            raise RuntimeError(
                                f"{thread_name_prefix} worker exited without sentinel"
                            )
                        continue
                    if item is _SENTINEL:
                        if slot.error is not None:
                            raise slot.error
                        break
                    if not isinstance(item, _QueuedRecordBatch):
                        raise RuntimeError("invalid parquet prefetch queue item")
                    try:
                        yield item.value
                    finally:
                        slot.byte_budget.release(item.nbytes)

                _drain_prefetch_slot(slot)
                assign(slot)
        finally:
            stop_event.set()
            for slot in slots:
                slot.byte_budget.wake_all()
            for slot in slots:
                _join_prefetch_thread(
                    slot.thread,
                    timeout_sec=self._io_policy.prefetch_join_timeout,
                    label=f"{thread_name_prefix} slot-{slot.index}",
                )
                _drain_prefetch_slot(slot)
                slot.thread = None
                slot.error = None
            slot_for_item.clear()

    def _iter_row_group_record_batches_prefetch(
        self,
        work_items: list[_RowGroupWorkItem],
        stop_event: threading.Event,
    ) -> Iterator[Any]:
        """Read row groups concurrently while yielding them in deterministic order.

        Each active worker owns a bounded queue. Completed slots are recycled
        for later row groups, which caps both thread count and prefetched Arrow
        memory independently of the total number of input files.
        """
        if not work_items:
            return

        active_workers = self._prefetch_active_workers(len(work_items))
        if active_workers <= 0:
            yield from self._iter_row_group_record_batches_sync(work_items, stop_event)
            return
        yield from self._iter_record_batches_with_persistent_prefetch(
            work_items,
            stop_event,
            active_workers=active_workers,
            worker=self._row_group_worker,
            thread_name_prefix="parquet-prefetch",
        )

    def _iter_row_group_record_batches(
        self, stop_event: threading.Event
    ) -> Iterator[Any]:
        """Dispatch to sync or prefetch row-group readers based on configuration."""
        work_items = self._assigned_row_group_work_items()
        if self.split.reader.prefetch_batches <= 0:
            yield from self._iter_row_group_record_batches_sync(work_items, stop_event)
            return
        yield from self._iter_row_group_record_batches_prefetch(work_items, stop_event)

    def _filesystem_is_remote(self) -> bool:
        return self._io_policy.enabled

    def _iter_file_record_batches_sync(
        self,
        paths: list[ParquetInputRef],
        stop_event: threading.Event,
    ) -> Iterator[Any]:
        """Sequentially read whole files via eason-style ParquetFile opens."""
        _pa, _pc, _ds, pq = _require_pyarrow()
        batch_size = self._reader_batch_size(default=65536)
        scan_columns = self._scan_columns()
        for ref in paths:
            if stop_event.is_set():
                return
            yield from iter_parquet_record_batches(
                fs_path=ref.fs_path,
                filesystem=ref.filesystem,
                filesystem_key=ref.filesystem_key,
                lock_key=ref.canonical_uri,
                policy=self._io_policy,
                pq_module=pq,
                stop_event=stop_event,
                description=f"file scan {ref.canonical_uri}",
                skip_tracker=self._skip_tracker,
                batch_size=batch_size,
                columns=scan_columns,
                use_threads=not self._io_policy.enabled,
            )

    def _file_worker(
        self,
        ref: ParquetInputRef,
        slot: _PrefetchSlot,
        stop_event: threading.Event,
    ) -> None:
        """Background worker: stream one whole file into a bounded prefetch queue."""
        try:
            _pa, _pc, _ds, pq = _require_pyarrow()
            batch_size = self._reader_batch_size(default=65536)
            scan_columns = self._scan_columns()
            for batch in iter_parquet_record_batches(
                fs_path=ref.fs_path,
                filesystem=ref.filesystem,
                filesystem_key=ref.filesystem_key,
                lock_key=ref.canonical_uri,
                policy=self._io_policy,
                pq_module=pq,
                stop_event=stop_event,
                description=f"prefetch file {ref.canonical_uri}",
                skip_tracker=self._skip_tracker,
                batch_size=batch_size,
                columns=scan_columns,
                # Prefetch workers: HDFS keeps Arrow inner threads off (JNI).
                # Local keeps them on for decode; outer threads cover cross-file IO.
                use_threads=not self._io_policy.enabled,
            ):
                if stop_event.is_set():
                    return
                batch_bytes = max(1, int(getattr(batch, "nbytes", 0)))
                if not slot.byte_budget.acquire(batch_bytes, stop_event):
                    return
                queued = _QueuedRecordBatch(batch, batch_bytes)
                if not _put_queue_item(slot.queue, queued, stop_event):
                    slot.byte_budget.release(batch_bytes)
                    return
        except BaseException as error:
            slot.error = error
        finally:
            _put_queue_item(slot.queue, _SENTINEL, stop_event)

    def _iter_file_record_batches_prefetch(
        self,
        paths: list[ParquetInputRef],
        stop_event: threading.Event,
    ) -> Iterator[Any]:
        """Read rank-local files concurrently while yielding in deterministic order."""
        if not paths:
            return

        active_workers = scaled_hdfs_prefetch_workers(
            world_size=self.shard_world_size,
            num_workers=self.split.reader.num_workers,
            prefetch_batches=self.split.reader.prefetch_batches,
            work_item_count=len(paths),
            remote=self._filesystem_is_remote(),
        )
        if active_workers <= 0:
            yield from self._iter_file_record_batches_sync(paths, stop_event)
            return
        yield from self._iter_record_batches_with_persistent_prefetch(
            paths,
            stop_event,
            active_workers=active_workers,
            worker=self._file_worker,
            thread_name_prefix="parquet-file-prefetch",
        )

    def _iter_file_record_batches(self, stop_event: threading.Event) -> Iterator[Any]:
        """Scan rank-local files via ``ParquetFile`` (never Dataset scanner).

        File sharding previously used Arrow Dataset ``scanner()`` with
        ``fragment_readahead`` / ``use_threads`` against a shared
        ``HadoopFileSystem``. That path observed ``Filesystem closed`` during
        concurrent fragment opens. Whole-file ``ParquetFile`` reads with
        thread-local HDFS clients match the eason model; optional prefetch
        parallelizes across disjoint files only.
        """

        if not self.paths:
            return
        if self.split.reader.prefetch_batches <= 0:
            yield from self._iter_file_record_batches_sync(self.paths, stop_event)
            return
        yield from self._iter_file_record_batches_prefetch(self.paths, stop_event)

    def _iter_dataset_record_batches(
        self, stop_event: threading.Event
    ) -> Iterator[Any]:
        """Backward-compatible alias for file-sharded scans."""

        yield from self._iter_file_record_batches(stop_event)

    def iter_record_batches(self) -> Iterator[Any]:
        """Return a closeable iterator so early exits also stop prefetch workers."""
        stop_event = threading.Event()
        if not self.paths and not self._uses_lpt_row_group_sharding():
            return _ClosableIterator(iter(()), stop_event)

        def generator() -> Iterator[Any]:
            if self._uses_lpt_row_group_sharding():
                yield from self._iter_row_group_record_batches(stop_event)
            else:
                yield from self._iter_file_record_batches(stop_event)

        return _ClosableIterator(generator(), stop_event)

    def iter_tables(self) -> Iterator[Any]:
        """Yield Arrow tables, the boundary consumed by feature-batch building."""
        _pa, _pc, _ds, _pq = _require_pyarrow()
        for batch in self.iter_record_batches():
            yield _pa.Table.from_batches([batch])

    def scan_stats(self, max_batches: int | None = None) -> ScanStats:
        """Count record batches and rows; always closes the underlying iterator."""
        iterator = self.iter_record_batches()
        record_batches = 0
        rows = 0
        try:
            batches: Iterable[Any]
            if max_batches is None:
                batches = iterator
            else:
                batches = islice(iterator, max_batches)
            for batch in batches:
                record_batches += 1
                rows += batch.num_rows
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        return ScanStats(
            files=len(self.paths), record_batches=record_batches, rows=rows
        )


# ---------------------------------------------------------------------------
# Adapters: raw Parquet -> flat table boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlatScanStats:
    """Counters for the unified raw-scan plus adapter path."""

    files: int
    raw_record_batches: int
    raw_rows: int
    flat_tables: int
    flat_rows: int


@dataclass
class _FlatScanCounters:
    files: int
    raw_record_batches: int = 0
    raw_rows: int = 0
    flat_tables: int = 0
    flat_rows: int = 0

    def snapshot(self) -> FlatScanStats:
        return FlatScanStats(
            files=self.files,
            raw_record_batches=self.raw_record_batches,
            raw_rows=self.raw_rows,
            flat_tables=self.flat_tables,
            flat_rows=self.flat_rows,
        )


# ---------------------------------------------------------------------------
# Built-in MDL-RankMixer agg/req adapter
# ---------------------------------------------------------------------------

# This production layout is intentionally built into the dataloader. Field
# membership remains config-driven through ParquetAdapterContext.options.


def _string_list(options: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = options.get(key, ())
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"adapter option {key!r} must be a list of column names")
    result = tuple(str(item) for item in value)
    if any(not item for item in result) or len(set(result)) != len(result):
        raise ValueError(f"adapter option {key!r} must contain unique non-empty names")
    return result


def _mapping(options: Mapping[str, Any], key: str) -> dict[str, str]:
    value = options.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"adapter option {key!r} must be an object")
    result = {str(name): str(source) for name, source in value.items()}
    if any(not name or not source for name, source in result.items()):
        raise ValueError(f"adapter option {key!r} must contain non-empty names")
    return result


def _positive_int_mapping(options: Mapping[str, Any], key: str) -> dict[str, int]:
    value = options.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"adapter option {key!r} must be an object")
    result: dict[str, int] = {}
    for raw_name, raw_limit in value.items():
        name = str(raw_name)
        if (
            not name
            or isinstance(raw_limit, bool)
            or not isinstance(raw_limit, int)
            or raw_limit <= 0
        ):
            raise ValueError(
                f"adapter option {key!r} must map non-empty names to positive integers"
            )
        result[name] = raw_limit
    return result


def _column_aliases(options: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    raw = options.get("column_aliases", {})
    if not isinstance(raw, Mapping):
        raise ValueError("adapter option 'column_aliases' must be an object")
    result: dict[str, tuple[str, ...]] = {}
    claimed: dict[str, str] = {}
    for canonical_raw, aliases_raw in raw.items():
        canonical = str(canonical_raw)
        if not canonical:
            raise ValueError("column_aliases canonical names must be non-empty")
        if isinstance(aliases_raw, (str, bytes)) or not isinstance(
            aliases_raw, Sequence
        ):
            raise ValueError(f"column_aliases.{canonical} must be a list")
        aliases = tuple(str(alias) for alias in aliases_raw)
        if (
            not aliases
            or any(not alias for alias in aliases)
            or len(set(aliases)) != len(aliases)
            or canonical in aliases
        ):
            raise ValueError(
                f"column_aliases.{canonical} must contain unique non-empty alternate names"
            )
        for name in (canonical, *aliases):
            owner = claimed.get(name)
            if owner is not None and owner != canonical:
                raise ValueError(
                    f"column alias {name!r} belongs to both {owner!r} and "
                    f"{canonical!r}"
                )
            claimed[name] = canonical
        result[canonical] = aliases
    return result


def _label_missing_values(
    options: Mapping[str, Any],
    labels: Mapping[str, str],
) -> dict[str, tuple[Any, ...]]:
    """Resolve explicitly declared missing-label sentinels per task.

    A list applies to every task; an object can declare different sentinels for
    different tasks. Binary 0/1 can never be configured as missing.
    """

    raw = options.get("label_missing_values", ())
    by_task: dict[str, Any]
    if isinstance(raw, Mapping):
        unknown = sorted(set(str(task) for task in raw) - set(labels))
        if unknown:
            raise ValueError(
                "label_missing_values contains unknown tasks: " + ", ".join(unknown)
            )
        by_task = {task: raw.get(task, ()) for task in labels}
    else:
        by_task = {task: raw for task in labels}

    result: dict[str, tuple[Any, ...]] = {}
    for task, values in by_task.items():
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError(
                f"label_missing_values.{task} must be a list of explicit sentinels"
            )
        sentinels = tuple(values)
        for sentinel in sentinels:
            if sentinel is not None and not isinstance(sentinel, (Real, str)):
                raise ValueError(
                    f"label_missing_values.{task} must contain only null, numeric, "
                    "or string scalar sentinels"
                )
            if isinstance(sentinel, bool) or (
                isinstance(sentinel, Real) and float(sentinel) in {0.0, 1.0}
            ):
                raise ValueError(
                    f"label_missing_values.{task} cannot mark binary value {sentinel!r} as missing"
                )
            if isinstance(sentinel, Real) and not math.isfinite(float(sentinel)):
                raise ValueError(
                    f"label_missing_values.{task} cannot contain non-finite values"
                )
        result[task] = sentinels
    return result


def _is_missing_label(value: Any, sentinels: Sequence[Any]) -> bool:
    for sentinel in sentinels:
        if value is sentinel:
            return True
        try:
            if value == sentinel:
                return True
        except (TypeError, ValueError):
            continue
    return False


# Coarse search/recommendation routing: index space is 0/1 for scenarios.source
# (source_encoding=index). Prior embeddings use a separate identity space 1/2 so
# padding_id=0 never collides with a real scenario.
COARSE_SCENE_INDEX_COLUMN = "coarse_scene_index"
COARSE_SCENE_PRIOR_ID_COLUMN = "coarse_scene_prior_id"
SEARCH_SCENARIO_INDEX = 0
RECOMMENDATION_SCENARIO_INDEX = 1
SCENARIO_NAMES = ("search", "recommendation")
COARSE_SCENE_PRIOR_NUM_BUCKETS = 3
COARSE_SCENE_PRIOR_EMBEDDING_DIM = 16
SEARCH_PRIOR_FEATURE = "scenario_search_prior_coarse_scene"
RECOMMENDATION_PRIOR_FEATURE = "scenario_recommendation_prior_coarse_scene"
INDEPENDENT_COARSE_SCENARIO_PRIORS = frozenset(
    {
        SEARCH_PRIOR_FEATURE,
        RECOMMENDATION_PRIOR_FEATURE,
    }
)
# Production search scene_ids. Unlisted integer scene_ids default to recommendation.
SEARCH_SCENE_IDS: frozenset[int] = frozenset(
    {
        2,
        21,
        23,
        27,
        28,
        31,
        35,
        38,
        39,
        40,
        42,
        45,
        50,
        62,
        68,
        70,
        76,
        77,
        78,
        81,
        85,
        90,
        93,
        94,
        95,
        98,
        100,
        110,
        111,
        130,
        133,
        135,
        137,
        141,
        145,
        146,
        150,
        152,
        159,
        160,
        167,
        168,
        175,
        186,
        187,
        191,
        197,
        198,
        204,
        211,
        219,
        231,
        233,
        240,
        245,
        246,
        252,
        255,
        282,
        283,
        294,
        298,
        299,
        301,
        310,
        317,
        319,
        325,
        335,
        338,
        340,
        341,
        351,
        356,
        357,
        366,
        377,
        383,
        384,
        385,
        391,
        392,
        393,
        394,
        396,
        398,
        401,
        403,
        404,
        415,
        416,
        417,
        419,
        420,
        421,
        423,
        424,
        428,
        430,
        435,
        436,
        437,
        438,
        446,
        448,
        452,
        459,
        464,
        471,
        475,
        482,
        485,
        492,
        504,
        519,
        1105,
        1106,
        1116,
        1121,
        1136,
        1137,
    }
)
EXPECTED_SEARCH_SCENE_ID_COUNT = 121


def validate_production_search_scene_ids(
    search_scene_ids: Collection[Any],
    *,
    expected_count: int = EXPECTED_SEARCH_SCENE_ID_COUNT,
) -> frozenset[int]:
    """Validate the production search scene id set."""

    normalized: list[int] = []
    for value in search_scene_ids:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(
                f"SEARCH_SCENE_IDS values must be non-negative integers, got {value!r}"
            )
        scene_id = int(value)
        if scene_id < 0:
            raise ValueError(
                f"SEARCH_SCENE_IDS values must be non-negative integers, got {value!r}"
            )
        normalized.append(scene_id)
    values = frozenset(normalized)
    if len(values) != expected_count:
        raise ValueError(
            "SEARCH_SCENE_IDS must contain exactly "
            f"{expected_count} unique integers, got {len(values)}"
        )
    return values


def coarse_scene_ids(
    raw_scene_id: Any,
    search_scene_ids: Collection[int],
    *,
    unlisted_policy: str = "recommendation",
) -> tuple[int, int]:
    """Map one raw scene_id to ``(coarse_scene_index, coarse_scene_prior_id)``.

    ``unlisted_policy``:
    - ``recommendation``: any non-negative integer outside ``search_scene_ids``
      maps to recommendation (production default).
    - ``error``: unlisted non-negative integers raise (closed allowlist mode).
    Negative IDs are always rejected.
    """

    if isinstance(raw_scene_id, bool) or not isinstance(raw_scene_id, Integral):
        raise ValueError(f"scene_id must be an integer, got {raw_scene_id!r}")
    scene_id = int(raw_scene_id)
    if scene_id < 0:
        raise ValueError(f"scene_id must be non-negative, got {scene_id}")
    if unlisted_policy not in {"recommendation", "error"}:
        raise ValueError(
            "unlisted_scene_policy must be 'recommendation' or 'error', "
            f"got {unlisted_policy!r}"
        )
    if scene_id in search_scene_ids:
        coarse_index = SEARCH_SCENARIO_INDEX
    elif unlisted_policy == "recommendation":
        coarse_index = RECOMMENDATION_SCENARIO_INDEX
    else:
        raise ValueError(
            f"scene_id {scene_id} is not in the configured search allowlist "
            "and unlisted_scene_policy='error'"
        )
    return coarse_index, coarse_index + 1


def _request_value_maps(
    options: Mapping[str, Any],
    request_columns: set[str],
) -> dict[str, dict[Any, int]]:
    raw = options.get("request_value_maps", {})
    if not isinstance(raw, Mapping):
        raise ValueError("adapter option 'request_value_maps' must be an object")
    result: dict[str, dict[Any, int]] = {}
    for column, raw_mapping in raw.items():
        column = str(column)
        if column not in request_columns:
            raise ValueError(
                f"request_value_maps contains non-request column {column!r}"
            )
        if not isinstance(raw_mapping, Mapping) or not raw_mapping:
            raise ValueError(f"request_value_maps.{column} must be a non-empty object")
        mapping: dict[Any, int] = {}
        for source, target in raw_mapping.items():
            if isinstance(target, bool) or not isinstance(target, int) or target < 0:
                raise ValueError(
                    f"request_value_maps.{column} targets must be non-negative integers"
                )
            mapping[source] = target
        expected = set(range(len(mapping)))
        if set(mapping.values()) != expected:
            raise ValueError(
                f"request_value_maps.{column} targets must be unique contiguous ids "
                f"0..{len(mapping) - 1}"
            )
        result[column] = mapping
    return result


def _map_request_value(
    value: Any,
    *,
    column: str,
    mapping: Mapping[Any, int],
    validate_contract: bool = True,
) -> int:
    if not validate_contract:
        try:
            return mapping[value]
        except (KeyError, TypeError):
            return mapping[str(value)]
    try:
        if value in mapping:
            return mapping[value]
    except TypeError:
        pass
    rendered = str(value)
    if rendered in mapping:
        return mapping[rendered]
    raise ValueError(
        f"request-level column {column!r} contains unmapped value {value!r}"
    )


@dataclass(frozen=True)
class _CoarseScenePlan:
    search_scene_ids: frozenset[int]
    raw_scene_column: str
    index_column: str
    prior_id_column: str
    unlisted_policy: str

    @property
    def derived_columns(self) -> frozenset[str]:
        return frozenset({self.index_column, self.prior_id_column})


def _adapter_derived_request_sources(options: Mapping[str, Any]) -> frozenset[str]:
    """Adapter-derived request columns that are not listed in context_features.

    Coarse-scene index/prior are computed from ``scene_id`` and written onto the
    request axis, but the adapter plan conflict check forbids putting them in
    ``context_features``. Direct-path readers must still treat them as
    request-level (third bucket alongside context and candidate).
    """

    if options.get("search_scene_ids") is None:
        return frozenset()
    index_column = str(
        options.get("coarse_scene_index_column", COARSE_SCENE_INDEX_COLUMN)
    )
    prior_id_column = str(
        options.get("coarse_scene_prior_id_column", COARSE_SCENE_PRIOR_ID_COLUMN)
    )
    return frozenset({index_column, prior_id_column})


def _adapter_request_level_sources(options: Mapping[str, Any]) -> set[str]:
    """Sources tensorized from the request axis (context + derived request cols)."""

    return {str(source) for source in options.get("context_features", ())} | set(
        _adapter_derived_request_sources(options)
    )


def _coarse_scene_plan(
    options: Mapping[str, Any],
    request_columns: set[str],
) -> _CoarseScenePlan | None:
    raw_ids = options.get("search_scene_ids")
    if raw_ids is None:
        return None
    if (
        isinstance(raw_ids, (str, bytes))
        or not isinstance(raw_ids, Sequence)
        or not raw_ids
    ):
        raise ValueError("adapter option 'search_scene_ids' must be a non-empty list")
    search_scene_ids: set[int] = set()
    for value in raw_ids:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(
                f"adapter option 'search_scene_ids' values must be integers, got {value!r}"
            )
        scene_id = int(value)
        if scene_id < 0:
            raise ValueError(
                f"adapter option 'search_scene_ids' values must be non-negative, got {scene_id}"
            )
        search_scene_ids.add(scene_id)
    raw_scene_column = str(options.get("coarse_scene_raw_column", "scene_id"))
    if raw_scene_column not in request_columns:
        raise ValueError(
            "coarse scene mapping requires request column "
            f"{raw_scene_column!r} in adapter request_columns"
        )
    index_column = str(
        options.get("coarse_scene_index_column", COARSE_SCENE_INDEX_COLUMN)
    )
    prior_id_column = str(
        options.get("coarse_scene_prior_id_column", COARSE_SCENE_PRIOR_ID_COLUMN)
    )
    unlisted_policy = str(options.get("unlisted_scene_policy", "recommendation"))
    if unlisted_policy not in {"recommendation", "error"}:
        raise ValueError(
            "adapter option 'unlisted_scene_policy' must be 'recommendation' or 'error'"
        )
    if not index_column or not prior_id_column:
        raise ValueError("coarse scene derived column names must be non-empty")
    if index_column == prior_id_column:
        raise ValueError("coarse scene index and prior id columns must be distinct")
    derived = {index_column, prior_id_column}
    if derived & request_columns:
        raise ValueError(
            "coarse scene derived columns conflict with request_columns: "
            + ", ".join(sorted(derived & request_columns))
        )
    return _CoarseScenePlan(
        search_scene_ids=frozenset(search_scene_ids),
        raw_scene_column=raw_scene_column,
        index_column=index_column,
        prior_id_column=prior_id_column,
        unlisted_policy=unlisted_policy,
    )


def _normalize_optional_outer_list(value: Any) -> Any:
    """Map top-level null/[] to an empty payload for optional list features.

    Only the outermost list/tuple/ndarray is normalized. Nested memberships such
    as ``[[], [0]]`` are preserved so orphan UPS tokens can still be rejected.
    NumPy 1-d rows from the Arrow fast path are kept as ndarray (no ``tolist``).
    """

    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    type_name = type(value).__name__
    raise TypeError(
        f"optional list-valued field must be null or a list/tuple, got {type_name}"
    )


@dataclass
class _FieldCardinalityStats:
    null_count: int = 0
    empty_count: int = 0
    singleton_count: int = 0
    multi_count: int = 0
    max_length: int = 0
    length_histogram: dict[int, int] = field(default_factory=dict)
    sample_multi_values: list[Any] = field(default_factory=list)

    def observe_length(self, length: int, *, sample: Any | None = None) -> None:
        self.max_length = max(self.max_length, length)
        self.length_histogram[length] = self.length_histogram.get(length, 0) + 1
        if length == 0:
            self.empty_count += 1
        elif length == 1:
            self.singleton_count += 1
        else:
            self.multi_count += 1
            if sample is not None and len(self.sample_multi_values) < 3:
                self.sample_multi_values.append(sample)

    def merge(self, other: "_FieldCardinalityStats") -> None:
        self.null_count += other.null_count
        self.empty_count += other.empty_count
        self.singleton_count += other.singleton_count
        self.multi_count += other.multi_count
        self.max_length = max(self.max_length, other.max_length)
        for length, count in other.length_histogram.items():
            self.length_histogram[length] = self.length_histogram.get(length, 0) + count
        for sample in other.sample_multi_values:
            if len(self.sample_multi_values) >= 3:
                break
            self.sample_multi_values.append(sample)

    def to_payload(self) -> dict[str, Any]:
        return {
            "null_count": self.null_count,
            "empty_count": self.empty_count,
            "singleton_count": self.singleton_count,
            "multi_count": self.multi_count,
            "max_length": self.max_length,
            "length_histogram": dict(sorted(self.length_histogram.items())),
            "sample_multi_values": list(self.sample_multi_values),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "_FieldCardinalityStats":
        stats = cls(
            null_count=int(payload.get("null_count", 0)),
            empty_count=int(payload.get("empty_count", 0)),
            singleton_count=int(payload.get("singleton_count", 0)),
            multi_count=int(payload.get("multi_count", 0)),
            max_length=int(payload.get("max_length", 0)),
        )
        histogram = payload.get("length_histogram", {})
        if isinstance(histogram, Mapping):
            stats.length_histogram = {
                int(length): int(count) for length, count in histogram.items()
            }
        samples = payload.get("sample_multi_values", ())
        if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes)):
            stats.sample_multi_values = list(samples)[:3]
        return stats


@dataclass
class FeatureCardinalityAuditor:
    """Collect scalar/bag list-length stats for a soft sample window."""

    bag_features: frozenset[str] = field(default_factory=frozenset)
    soft: bool = False
    raw_rows_seen: int = 0
    scalar_stats: dict[str, _FieldCardinalityStats] = field(default_factory=dict)
    bag_stats: dict[str, _FieldCardinalityStats] = field(default_factory=dict)

    def _stats_for(self, column: str, *, bag: bool) -> _FieldCardinalityStats:
        store = self.bag_stats if bag else self.scalar_stats
        stats = store.get(column)
        if stats is None:
            stats = _FieldCardinalityStats()
            store[column] = stats
        return stats

    def observe_scalar(self, column: str, value: Any) -> None:
        stats = self._stats_for(column, bag=False)
        if value is None:
            stats.null_count += 1
            return
        if isinstance(value, (list, tuple, np.ndarray)):
            length = len(value)
            stats.observe_length(
                length,
                sample=list(value[:8]) if length > 1 else None,
            )
            return
        stats.observe_length(1)

    def observe_bag(self, column: str, value: Any) -> None:
        stats = self._stats_for(column, bag=True)
        if value is None:
            stats.null_count += 1
            stats.observe_length(0)
            return
        if isinstance(value, (list, tuple, np.ndarray)):
            stats.observe_length(len(value))
            return
        stats.observe_length(1)

    def note_raw_rows(self, count: int) -> None:
        self.raw_rows_seen += max(0, int(count))

    def has_scalar_multis(self) -> bool:
        return any(stats.multi_count > 0 for stats in self.scalar_stats.values())

    def to_payload(self) -> dict[str, Any]:
        return {
            "raw_rows_seen": self.raw_rows_seen,
            "scalar_stats": {
                name: stats.to_payload()
                for name, stats in sorted(self.scalar_stats.items())
            },
            "bag_stats": {
                name: stats.to_payload()
                for name, stats in sorted(self.bag_stats.items())
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FeatureCardinalityAuditor":
        auditor = cls(raw_rows_seen=int(payload.get("raw_rows_seen", 0)))
        for name, stats_payload in dict(payload.get("scalar_stats", {})).items():
            auditor.scalar_stats[str(name)] = _FieldCardinalityStats.from_payload(
                stats_payload
            )
        for name, stats_payload in dict(payload.get("bag_stats", {})).items():
            auditor.bag_stats[str(name)] = _FieldCardinalityStats.from_payload(
                stats_payload
            )
        return auditor

    def merge_payload(self, payload: Mapping[str, Any]) -> None:
        other = self.from_payload(payload)
        self.raw_rows_seen += other.raw_rows_seen
        for name, stats in other.scalar_stats.items():
            self._stats_for(name, bag=False).merge(stats)
        for name, stats in other.bag_stats.items():
            self._stats_for(name, bag=True).merge(stats)

    def format_report(self) -> str:
        lines = [
            "Feature cardinality audit",
            f"raw_rows_seen={self.raw_rows_seen}",
            "",
            "Scalar cardinality violations:"
            if self.has_scalar_multis()
            else "Scalar fields (no multi-value cells observed):",
        ]
        scalar_items = sorted(
            self.scalar_stats.items(),
            key=lambda item: (-item[1].multi_count, item[0]),
        )
        if not scalar_items:
            lines.append("  (no scalar observations)")
        for name, stats in scalar_items:
            if stats.multi_count == 0 and self.has_scalar_multis():
                continue
            lines.append(
                f"{name}\n"
                f"  null={stats.null_count} empty={stats.empty_count} "
                f"singleton={stats.singleton_count} multi={stats.multi_count} "
                f"max_length={stats.max_length}\n"
                f"  length_histogram={dict(sorted(stats.length_histogram.items()))}"
            )
            if stats.sample_multi_values:
                lines.append(f"  sample_multi_values={stats.sample_multi_values!r}")
        suspicious_bags = [
            (name, stats)
            for name, stats in sorted(self.bag_stats.items())
            if stats.multi_count == 0 and stats.singleton_count > 0
        ]
        if suspicious_bags:
            lines.extend(
                [
                    "",
                    "Bags that only observed length 0/1 in this sample "
                    "(may be scalars or fixed singleton encodings):",
                ]
            )
            for name, stats in suspicious_bags[:20]:
                lines.append(
                    f"{name}: empty={stats.empty_count} singleton={stats.singleton_count} "
                    f"null={stats.null_count}"
                )
        return "\n".join(lines)


def _as_list(
    value: Any,
    *,
    column: str,
    row_index: int,
    validate_contract: bool = True,
) -> list[Any]:
    if value is None:
        return []
    # NumPy rows from flattened UPS columns: keep as ndarray on the trusted
    # hot path so membership gather can fancy-index without a pylist copy.
    if isinstance(value, np.ndarray):
        if validate_contract:
            return value.tolist()
        return value  # type: ignore[return-value]
    if not validate_contract:
        if isinstance(value, (list, tuple)):
            return list(value)
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ValueError(
        f"column {column!r} must be list-valued at raw row {row_index}, "
        f"got {type(value).__name__}"
    )


def _request_index(
    value: Any,
    *,
    column: str,
    row_index: int,
    validate_contract: bool = True,
) -> int:
    if isinstance(value, np.integer):
        value = int(value)
    if validate_contract and (
        isinstance(value, bool) or not isinstance(value, int) or value < 0
    ):
        raise ValueError(
            f"column {column!r} contains invalid request index {value!r} "
            f"at raw row {row_index}"
        )
    return int(value)


def _scalarize(
    value: Any,
    *,
    column: str,
    raw_row: int,
    logical_row: int,
    validate_contract: bool = True,
    auditor: "FeatureCardinalityAuditor | None" = None,
) -> Any:
    """Collapse optional singleton list wrappers for scalar features.

    Contract (independent of trusted_input / validate_contract):

    - ``None`` / ``[]`` → ``None`` (missing → padding ID 0 downstream)
    - ``[v]`` → ``v``
    - length > 1 → always raise (never silently take the first element)

    When an auditor is in soft mode, length > 1 is recorded and the cell is
    treated as missing so the rest of the row can still be audited.
    """

    del validate_contract  # Scalar cardinality is never relaxed under trusted_input.
    if value is None:
        if auditor is not None:
            auditor.observe_scalar(column, None)
        return None
    if isinstance(value, np.ndarray):
        length = int(value.shape[0])
        if length == 0:
            if auditor is not None:
                auditor.observe_scalar(column, [])
            return None
        if length != 1:
            if auditor is not None and auditor.soft:
                auditor.observe_scalar(column, value)
                return None
            raise ValueError(
                f"single-valued feature {column!r} has inner length {length} "
                f"at raw row {raw_row}, logical row {logical_row}"
            )
        if auditor is not None:
            auditor.observe_scalar(column, value)
        return value[0]
    if isinstance(value, np.generic):
        # Fancy-index / Arrow→NumPy can yield 0-d numpy scalars.
        if auditor is not None:
            auditor.observe_scalar(column, value)
        return value
    if not isinstance(value, (list, tuple)):
        if auditor is not None:
            auditor.observe_scalar(column, value)
        return value
    length = len(value)
    if length == 0:
        if auditor is not None:
            auditor.observe_scalar(column, [])
        return None
    if length != 1:
        if auditor is not None and auditor.soft:
            auditor.observe_scalar(column, value)
            return None
        raise ValueError(
            f"single-valued feature {column!r} has inner length {length} "
            f"at raw row {raw_row}, logical row {logical_row}"
        )
    if auditor is not None:
        auditor.observe_scalar(column, value)
    return value[0]


def _scalarize_column_values(
    values: Sequence[Any],
    *,
    column: str,
    raw_row: int,
    validate_contract: bool = True,
    auditor: "FeatureCardinalityAuditor | None" = None,
) -> list[Any]:
    """Scalarize one candidate/request column; vectorize trusted singleton ndarrays."""

    if (
        auditor is None
        and not validate_contract
        and len(values) > 0
        and all(
            isinstance(value, np.ndarray) and value.shape == (1,) for value in values
        )
    ):
        dtype = values[0].dtype
        out = np.empty(len(values), dtype=dtype)
        for index, value in enumerate(values):
            out[index] = value[0]
        return list(out)
    return [
        _scalarize(
            value,
            column=column,
            raw_row=raw_row,
            logical_row=candidate_index,
            validate_contract=validate_contract,
            auditor=auditor,
        )
        for candidate_index, value in enumerate(values)
    ]


def _bag_value(
    value: Any,
    *,
    column: str,
    raw_row: int,
    logical_row: int,
    validate_contract: bool = True,
    auditor: "FeatureCardinalityAuditor | None" = None,
) -> Any:
    """Normalize optional categorical bags; top-level null/[] mean length 0."""

    del validate_contract  # Outer null/[] are always accepted as zero-length bags.
    try:
        normalized = _normalize_optional_outer_list(value)
    except TypeError as error:
        raise ValueError(
            f"multivalue feature {column!r} must be list-valued at raw row "
            f"{raw_row}, logical row {logical_row}"
        ) from error
    if auditor is not None:
        auditor.observe_bag(column, normalized)
    return normalized


def _candidate_count_req(
    row: Mapping[str, Any],
    item_features: Sequence[str],
    label_columns: Sequence[str],
    raw_row: int,
    *,
    validate_contract: bool = True,
) -> int:
    if not validate_contract:
        for column in [*item_features, *label_columns]:
            if column in row and row[column] is not None:
                return len(
                    _as_list(
                        row[column],
                        column=column,
                        row_index=raw_row,
                        validate_contract=False,
                    )
                )
        raise ValueError(
            f"cannot infer candidate count for req raw row {raw_row}; "
            "no item or label arrays are present"
        )
    observed: dict[str, int] = {}
    for column in [*item_features, *label_columns]:
        if column not in row or row[column] is None:
            continue
        observed[column] = len(_as_list(row[column], column=column, row_index=raw_row))
    if not observed:
        raise ValueError(
            f"cannot infer candidate count for req raw row {raw_row}; "
            "no item or label arrays are present"
        )
    counts = set(observed.values())
    if len(counts) != 1:
        raise ValueError(
            f"req raw row {raw_row} has inconsistent candidate counts: {observed}"
        )
    return next(iter(counts))


def _request_positions(
    context_indices: Sequence[Any],
    *,
    raw_row: int,
    validate_contract: bool = True,
) -> dict[int, int]:
    positions: dict[int, int] = {}
    for position, raw_request in enumerate(context_indices):
        request = _request_index(
            raw_request,
            column="context_indices",
            row_index=raw_row,
            validate_contract=validate_contract,
        )
        if validate_contract and request in positions:
            raise ValueError(
                f"context_indices contains duplicate request {request} at raw row {raw_row}"
            )
        positions[request] = position
    return positions


def _request_level_value(
    value: Any,
    *,
    request_position: int,
    request_count: int,
    column: str,
    raw_row: int,
    agg: bool,
    validate_contract: bool = True,
) -> Any:
    """Select one request-axis cell, then collapse inner singleton wrappers.

    Agg request-level lists are indexed by request, not treated as scalar
    singletons: length-1 or bare scalars must not silently broadcast across
    ``request_count > 1``. Soft cardinality auditors are never applied here.
    """

    if agg:
        if isinstance(value, (list, tuple)):
            # Request-axis length is structural correctness, not a soft contract.
            if len(value) != request_count:
                raise ValueError(
                    f"agg request-level column {column!r} has length "
                    f"{len(value)}, expected {request_count} at raw row {raw_row}"
                )
            selected = value[request_position]
        elif isinstance(value, np.ndarray) and value.ndim == 1:
            if int(value.shape[0]) != request_count:
                raise ValueError(
                    f"agg request-level column {column!r} has length "
                    f"{int(value.shape[0])}, expected {request_count} at raw row {raw_row}"
                )
            selected = value[request_position]
        else:
            if request_count != 1:
                raise ValueError(
                    f"agg request-level column {column!r} is scalar "
                    f"but request_count={request_count} at raw row {raw_row}"
                )
            selected = value
    elif isinstance(value, (list, tuple)):
        if validate_contract and len(value) != 1:
            raise ValueError(
                f"req request-level column {column!r} must be scalar or length one "
                f"at raw row {raw_row}, got length {len(value)}"
            )
        selected = value[0]
    elif isinstance(value, np.ndarray) and value.ndim == 1:
        if validate_contract and int(value.shape[0]) != 1:
            raise ValueError(
                f"req request-level column {column!r} must be scalar or length one "
                f"at raw row {raw_row}, got length {int(value.shape[0])}"
            )
        selected = value[0] if value.size else None
    else:
        selected = value

    return _scalarize(
        selected,
        column=column,
        raw_row=raw_row,
        logical_row=request_position,
        validate_contract=validate_contract,
    )


def _req_context_value(
    value: Any,
    *,
    has_request_axis: bool,
    multivalue: bool,
    column: str,
    raw_row: int,
    validate_contract: bool = True,
) -> Any:
    """Normalize the two observed req encodings of request-level features.

    Most req fields remove the train request axis and arrive as ``list<int64>``.
    A small set of multivalue User fields remains ``list<list<int64>>``; it can
    be either one request containing a bag or a bag of singleton encoded values.
    Top-level null/[] mean missing: multivalue → ``[]``, scalar → ``None``.
    """

    if not has_request_axis:
        return value
    try:
        outer = _normalize_optional_outer_list(value)
    except TypeError as error:
        raise ValueError(
            f"req context column {column!r} must be list-valued at raw row {raw_row}"
        ) from error
    if not outer:
        return [] if multivalue else None
    if not multivalue:
        if validate_contract and len(outer) != 1:
            raise ValueError(
                f"req scalar context column {column!r} has nested outer length "
                f"{len(outer)} at raw row {raw_row}; expected 1"
            )
        return outer[0]
    if len(outer) == 1:
        return outer[0]
    if not validate_contract or all(
        item is None or (isinstance(item, (list, tuple, np.ndarray)) and len(item) == 1)
        for item in outer
    ):
        return [None if item is None else item[0] for item in outer]
    raise ValueError(
        f"req multivalue context column {column!r} has unsupported nested layout "
        f"at raw row {raw_row}; expected one request bag or singleton token lists"
    )


def _sequence_membership_positions(
    memberships: Sequence[Any],
    *,
    known_requests: set[int],
    index_column: str,
    raw_row: int,
    validate_contract: bool = True,
    validate_structure: bool | None = None,
) -> dict[int, list[int]]:
    """Validate one UPS membership vector and index it once per raw row.

    When ``validate_structure`` is false (trusted hot path), membership is
    indexed without empty/unknown/duplicate checks.
    """

    if validate_structure is None:
        validate_structure = validate_contract
    selected: dict[int, list[int]] = {request: [] for request in known_requests}
    for token_position, raw_membership in enumerate(memberships):
        if isinstance(raw_membership, np.ndarray):
            members: Any = raw_membership
        elif isinstance(raw_membership, list):
            members = raw_membership
        elif isinstance(raw_membership, tuple):
            members = list(raw_membership)
        else:
            members = [raw_membership]
        if validate_structure and len(members) == 0:
            raise ValueError(
                f"UPS indices column {index_column!r} has an empty membership "
                f"at raw row {raw_row}, token {token_position}"
            )
        if validate_structure:
            # Inline ``_request_index`` to avoid hundreds of thousands of
            # Python function calls on long UPS membership vectors.
            normalized: list[int] = []
            seen: set[int] = set()
            for value in members:
                if isinstance(value, np.integer):
                    value = int(value)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"column {index_column!r} contains invalid request index "
                        f"{value!r} at raw row {raw_row}"
                    )
                if value in seen:
                    raise ValueError(
                        f"UPS indices column {index_column!r} repeats a request at "
                        f"raw row {raw_row}, token {token_position}"
                    )
                seen.add(value)
                normalized.append(value)
            unknown = seen - known_requests
            if unknown:
                raise ValueError(
                    f"UPS indices column {index_column!r} references requests without "
                    f"context at raw row {raw_row}, token {token_position}: "
                    f"{sorted(unknown)}"
                )
        else:
            if isinstance(members, np.ndarray) and members.dtype.kind in "iu":
                normalized = [int(value) for value in members.tolist()]
            else:
                normalized = [int(value) for value in members]
        for request in normalized:
            selected[request].append(token_position)
    return selected


def _select_sequence(
    values: Any,
    selected_positions: Sequence[int] | None,
    *,
    expected_length: int | None,
    column: str,
    raw_row: int,
    validated_flat: bool = False,
    max_length: int | None = None,
    validate_contract: bool = True,
    validate_structure: bool | None = None,
    validate_payload: bool | None = None,
) -> list[Any]:
    """Select and optionally validate one UPS attribute sequence.

    Top-level null/[] are zero-length. Length alignment against indices runs
    under ``validate_structure`` (before truncation). Token singleton/null
    diagnostics run under ``validate_payload``.
    """

    if validate_structure is None:
        validate_structure = validate_contract
    if validate_payload is None:
        validate_payload = validate_contract
    items = _as_list(
        values,
        column=column,
        row_index=raw_row,
        validate_contract=validate_structure,
    )
    if selected_positions is not None:
        if expected_length is None:
            raise RuntimeError("selected UPS positions require an expected raw length")
        item_length = len(items)
        if validate_structure and item_length != expected_length:
            raise ValueError(
                f"UPS column {column!r} length {item_length} does not match its indices "
                f"length {expected_length} at raw row {raw_row}"
            )
        if max_length is not None:
            selected_positions = selected_positions[:max_length]
        if not selected_positions:
            return []
        # Trusted / flattened histories arrive as NumPy row views. Fancy-index
        # once instead of a Python listcomp per (request × sequence column).
        if isinstance(items, np.ndarray):
            gathered = items[np.asarray(selected_positions, dtype=np.int64)]
            if validated_flat or not validate_payload:
                return gathered
            items = gathered.tolist()
        else:
            items = [items[position] for position in selected_positions]
    elif max_length is not None:
        if isinstance(items, np.ndarray):
            items = items[:max_length]
            if validated_flat or not validate_payload:
                return items
            items = items.tolist()
        else:
            items = items[:max_length]

    if isinstance(items, np.ndarray):
        if not validate_payload:
            return items
        items = items.tolist()

    if validated_flat:
        return items

    if not validate_payload:
        return [
            (
                item[0]
                if isinstance(item, (list, tuple))
                or (isinstance(item, np.ndarray) and item.shape == (1,))
                else item
            )
            for item in items
        ]

    # Token-level nulls are allowed: null_anchor_field compresses whole steps
    # downstream, and non-anchor nulls encode as padding ID 0 / 0.0.
    normalized: list[Any] = []
    for token_position, item in enumerate(items):
        if isinstance(item, (list, tuple)):
            if len(item) != 1:
                raise ValueError(
                    f"UPS column {column!r} token {token_position} has inner length "
                    f"{len(item)} at raw row {raw_row}; expected exactly 1"
                )
            item = item[0]
        normalized.append(item)
    return normalized


def _select_global_recent_sequence_positions(
    positions_by_type: Mapping[str, Sequence[int]],
    event_times_by_type: Mapping[str, Sequence[Any]],
    *,
    ups_types: Sequence[str],
    max_length: int,
    raw_row: int,
    validate_contract: bool,
) -> dict[str, list[int]]:
    """Select one newest-event window across heterogeneous UPS streams.

    Each input stream remains in its original newest-to-oldest physical order
    after selection. Ties are deterministic: configured UPS order, then the
    original position within that stream.
    """

    normalized_times_by_type: dict[str, list[int | None]] = {}
    for ups in ups_types:
        positions = positions_by_type.get(ups, ())
        event_times = event_times_by_type.get(ups, ())
        if len(positions) != len(event_times):
            raise RuntimeError(
                f"global sequence selection received {len(positions)} positions "
                f"but {len(event_times)} timestamps for {ups!r} at raw row {raw_row}"
            )
        normalized_times: list[int | None] = []
        previous_timestamp: int | None = None
        saw_null = False
        for local_order, raw_time in enumerate(event_times):
            if raw_time is None:
                normalized_times.append(None)
                saw_null = True
                continue
            if isinstance(raw_time, np.integer):
                raw_time = int(raw_time)
            if validate_contract and (
                isinstance(raw_time, bool) or not isinstance(raw_time, int)
            ):
                raise ValueError(
                    f"sequence {ups!r} has invalid event time {raw_time!r} "
                    f"at raw row {raw_row}, position {local_order}"
                )
            try:
                timestamp = int(raw_time)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"sequence {ups!r} has invalid event time {raw_time!r} "
                    f"at raw row {raw_row}, position {local_order}"
                ) from error
            if validate_contract and (
                saw_null
                or (
                    previous_timestamp is not None
                    and timestamp > previous_timestamp
                )
            ):
                raise ValueError(
                    f"sequence {ups!r} event times must be newest-to-oldest "
                    f"before global selection at raw row {raw_row}, position "
                    f"{local_order}"
                )
            normalized_times.append(timestamp)
            previous_timestamp = timestamp
        normalized_times_by_type[ups] = normalized_times

    # Every UPS is already newest-to-oldest, so this is a k-way merge rather
    # than a sort of up to len(ups_types) * max_length Python tuples.
    heap: list[tuple[int, int, int, str, int]] = []

    def _push(ups: str, stream_order: int, local_order: int) -> None:
        times = normalized_times_by_type[ups]
        if local_order >= len(times):
            return
        timestamp = times[local_order]
        if timestamp is None:
            # An event without a timestamp cannot participate in a global
            # chronological window. Nulls are required to be a suffix under
            # validation, so there can be no later comparable event to push.
            return
        heapq.heappush(
            heap,
            (
                -timestamp,
                stream_order,
                local_order,
                ups,
                int(positions_by_type[ups][local_order]),
            ),
        )

    for stream_order, ups in enumerate(ups_types):
        _push(ups, stream_order, 0)

    selected_by_type: dict[str, list[tuple[int, int]]] = {
        ups: [] for ups in ups_types
    }
    selected_count = 0
    while heap and selected_count < max_length:
        (
            _newest_rank,
            stream_order,
            local_order,
            ups,
            position,
        ) = heapq.heappop(heap)
        selected_by_type[ups].append((local_order, position))
        selected_count += 1
        _push(ups, stream_order, local_order + 1)
    return {
        ups: [position for _local_order, position in sorted(selected_by_type[ups])]
        for ups in ups_types
    }


def _flatten_singleton_ups_array(
    pa: Any,
    pc: Any,
    array: Any,
    *,
    validate_contract: bool = True,
) -> tuple[Any, bool]:
    """Collapse list<list<int64>> singleton tokens before Python conversion.

    fgout stores every S-token property as an inner singleton list. Flattening
    that level with Arrow avoids allocating millions of one-element Python
    lists in ``to_pydict``. Invalid/null token payloads deliberately fall back
    to the validated Python path so error semantics are unchanged.
    """

    if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type)):
        return array, False
    child = array.values
    if pa.types.is_list(child.type) or pa.types.is_large_list(child.type):
        if validate_contract:
            lengths = pc.list_value_length(child)
            if lengths.null_count:
                return array, False
            invalid = pc.any(pc.not_equal(lengths, 1)).as_py()
            if invalid:
                return array, False
        flattened = pc.list_flatten(child)
        if validate_contract and flattened.null_count:
            return array, False
        offsets = array.offsets
        base = int(offsets[0].as_py())
        stop = int(offsets[-1].as_py())
        normalized_offsets = pc.subtract(offsets, base)
        flattened = flattened.slice(base, stop - base)
        mask = array.is_null() if array.null_count else None
        if pa.types.is_large_list(array.type):
            rebuilt = pa.LargeListArray.from_arrays(
                normalized_offsets,
                flattened,
                mask=mask,
            )
        else:
            rebuilt = pa.ListArray.from_arrays(
                normalized_offsets,
                flattened,
                mask=mask,
            )
        return rebuilt, True
    if validate_contract and child.null_count:
        return array, False
    return array, True


def _arrow_array_to_pylist(pa: Any, array: Any) -> list[Any]:
    """Materialize an Arrow array as Python objects / NumPy row views.

    For the common ``list<primitive>`` and ``list<list<primitive>>`` columns in
    the aggregated format, Arrow's generic ``to_pylist`` allocates one
    intermediate object per element and dominates adapter CPU. When the array
    is a flat/nested list of a numeric primitive with no inner nulls, we return
    NumPy views per row (and per inner list) instead of ``.tolist()``. Callers
    on the trusted path keep those views through bag/scalarize/compact.
    """

    if array.offset != 0:
        return array.to_pylist()
    if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type)):
        return array.to_pylist()
    child = array.values
    child_type = child.type
    if pa.types.is_list(child_type) or pa.types.is_large_list(child_type):
        grandchild = child.values
        if (
            pa.types.is_integer(grandchild.type)
            or pa.types.is_floating(grandchild.type)
        ) and not grandchild.null_count:
            try:
                outer_offsets = array.offsets.to_numpy()
                inner_offsets = child.offsets.to_numpy()
                values = grandchild.to_numpy(zero_copy_only=False)
                outer_null = (
                    array.is_null().to_numpy(zero_copy_only=False)
                    if array.null_count
                    else None
                )
                inner_null = (
                    child.is_null().to_numpy(zero_copy_only=False)
                    if child.null_count
                    else None
                )
            except (TypeError, ValueError, NotImplementedError):
                return array.to_pylist()
            result: list[Any] = []
            for row_index in range(len(array)):
                if outer_null is not None and outer_null[row_index]:
                    result.append(None)
                    continue
                start = int(outer_offsets[row_index])
                stop = int(outer_offsets[row_index + 1])
                if inner_null is None:
                    nested = [
                        values[
                            int(inner_offsets[inner_index]) : int(
                                inner_offsets[inner_index + 1]
                            )
                        ]
                        for inner_index in range(start, stop)
                    ]
                else:
                    nested = []
                    for inner_index in range(start, stop):
                        if inner_null[inner_index]:
                            nested.append(None)
                        else:
                            nested.append(
                                values[
                                    int(inner_offsets[inner_index]) : int(
                                        inner_offsets[inner_index + 1]
                                    )
                                ]
                            )
                result.append(nested)
            return result
    if not (pa.types.is_integer(child_type) or pa.types.is_floating(child_type)):
        return array.to_pylist()
    if child.null_count:
        return array.to_pylist()
    try:
        offsets = array.offsets.to_numpy()
        values = child.to_numpy(zero_copy_only=False)
    except (TypeError, ValueError, NotImplementedError):
        return array.to_pylist()
    if array.null_count:
        is_null = array.is_null().to_numpy(zero_copy_only=False)
        return [
            None
            if is_null[index]
            else values[int(offsets[index]) : int(offsets[index + 1])]
            for index in range(len(array))
        ]
    return [
        values[int(offsets[index]) : int(offsets[index + 1])]
        for index in range(len(array))
    ]


def _arrow_list_array_to_numpy_rows(pa: Any, array: Any) -> list[Any] | None:
    """Return per-row NumPy views for ``list<primitive>`` columns.

    Used for flattened UPS histories so membership gather can fancy-index
    without first allocating a Python ``int``/``float`` per token. Returns
    ``None`` when the array is not a safe flat numeric list (caller falls
    back to ``_arrow_array_to_pylist``).
    """

    import numpy as np

    if array.offset != 0:
        return None
    if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type)):
        return None
    child = array.values
    if not (pa.types.is_integer(child.type) or pa.types.is_floating(child.type)):
        return None
    if child.null_count:
        return None
    try:
        offsets = array.offsets.to_numpy()
        values = child.to_numpy(zero_copy_only=False)
    except (TypeError, ValueError, NotImplementedError):
        return None
    if array.null_count:
        is_null = array.is_null().to_numpy(zero_copy_only=False)
        return [
            None if is_null[index] else values[offsets[index] : offsets[index + 1]]
            for index in range(len(array))
        ]
    return [values[offsets[index] : offsets[index + 1]] for index in range(len(array))]


def _adapter_table_to_python(
    table: Any,
    raw_sequence_columns: frozenset[str],
    *,
    validate_contract: bool = True,
) -> tuple[dict[str, list[Any]], frozenset[str]]:
    """Convert a raw table while flattening valid singleton S-token columns.

    Flattened UPS columns are kept as per-row NumPy views when possible so
    request membership gather avoids tens of thousands of Python listcomps.
    """

    pa, pc, _ds, _pq = _require_pyarrow()
    raw: dict[str, list[Any]] = {}
    flattened: set[str] = set()
    column_names = table.column_names
    for column_index, name in enumerate(column_names):
        chunked = table.column(column_index)
        if chunked.num_chunks == 1:
            array = chunked.chunk(0)
        else:
            array = _column_array(table, name)
        if name in raw_sequence_columns:
            array, validated_flat = _flatten_singleton_ups_array(
                pa,
                pc,
                array,
                validate_contract=validate_contract,
            )
            if validated_flat:
                flattened.add(name)
                numpy_rows = _arrow_list_array_to_numpy_rows(pa, array)
                if numpy_rows is not None:
                    raw[name] = numpy_rows
                    continue
        else:
            # Non-UPS list<primitive> columns (indices, bags already flat, …)
            # also prefer NumPy row views on the trusted path.
            numpy_rows = _arrow_list_array_to_numpy_rows(pa, array)
            if numpy_rows is not None:
                raw[name] = numpy_rows
                continue
        raw[name] = _arrow_array_to_pylist(pa, array)
    return raw, frozenset(flattened)


def _vectorized_time_delta_transform(
    deltas: Any,
    *,
    transform: str,
) -> Any:
    """Apply the configured delta transform with a single NumPy pass."""

    values = np.asarray(deltas, dtype=np.float64)
    if transform == "raw_ms":
        return values
    if transform == "seconds":
        return values / 1000.0
    if transform == "log1p_seconds":
        return np.log1p(values / 1000.0)
    raise RuntimeError(f"unsupported time delta transform {transform!r}")


def _time_deltas(
    event_times: Sequence[Any],
    request_time: Any,
    *,
    sequence: str,
    raw_row: int,
    transform: str,
    validate_contract: bool = True,
) -> Any:
    def _transform_delta(delta: int) -> float:
        if transform == "raw_ms":
            return float(delta)
        if transform == "seconds":
            return float(delta) / 1000.0
        if transform == "log1p_seconds":
            return math.log1p(float(delta) / 1000.0)
        raise RuntimeError(f"unsupported time delta transform {transform!r}")

    if not validate_contract:
        if isinstance(event_times, np.ndarray):
            if event_times.size == 0:
                return event_times[:0].astype(np.float64, copy=False)
            if event_times.dtype == object and any(
                event_time is None for event_time in event_times
            ):
                return [
                    0.0
                    if event_time is None
                    else _transform_delta(int(request_time) - int(event_time))
                    for event_time in event_times
                ]
            return _vectorized_time_delta_transform(
                np.asarray(request_time, dtype=np.int64)
                - event_times.astype(np.int64, copy=False),
                transform=transform,
            )
        if not event_times:
            return []
        # Trusted hot path: prefer one NumPy pass when the selected times have
        # no null placeholders. Null event times still pad to 0.0.
        if any(event_time is None for event_time in event_times):
            result: list[float] = []
            for event_time in event_times:
                if event_time is None:
                    result.append(0.0)
                    continue
                result.append(_transform_delta(int(request_time) - int(event_time)))
            return result
        values = np.asarray(event_times, dtype=np.float64)
        return _vectorized_time_delta_transform(
            float(request_time) - values,
            transform=transform,
        )

    if event_times and (
        isinstance(request_time, bool)
        or not isinstance(request_time, (int, np.integer))
    ):
        raise ValueError(
            f"request time is required to derive {sequence!r} time deltas at raw row {raw_row}"
        )
    if event_times and all(event_time is not None for event_time in event_times):
        # Histories dominate adapter CPU. NumPy performs validation,
        # subtraction, and log1p in native loops instead of one Python/math
        # call per event. Skip when any null times are present.
        try:
            values = np.asarray(event_times)
        except Exception:
            values = None
        if values is not None:
            if values.dtype.kind not in {"i", "u"}:
                raise ValueError(
                    f"sequence {sequence!r} has non-integer event time at raw row {raw_row}"
                )
            increasing = values[1:] > values[:-1]
            if bool(np.any(increasing)):
                position = int(np.flatnonzero(increasing)[0]) + 1
                raise ValueError(
                    f"sequence {sequence!r} event times must be newest-to-oldest at "
                    f"raw row {raw_row}; position {position - 1} is "
                    f"{int(values[position - 1])}, position {position} is "
                    f"{int(values[position])}"
                )
            deltas = int(request_time) - values
            if bool(np.any(deltas < 0)):
                position = int(np.flatnonzero(deltas < 0)[0])
                raise ValueError(
                    f"sequence {sequence!r} event time is later than request time "
                    f"at raw row {raw_row}, position {position}: "
                    f"delta_ms={int(deltas[position])}"
                )
            return _vectorized_time_delta_transform(deltas, transform=transform)
    result = []
    previous_time: int | None = None
    for position, event_time in enumerate(event_times):
        if event_time is None:
            result.append(0.0)
            continue
        if isinstance(event_time, bool) or not isinstance(event_time, int):
            raise ValueError(
                f"sequence {sequence!r} has invalid event time {event_time!r} "
                f"at raw row {raw_row}, position {position}"
            )
        if previous_time is not None and event_time > previous_time:
            raise ValueError(
                f"sequence {sequence!r} event times must be newest-to-oldest at "
                f"raw row {raw_row}; position {position - 1} is {previous_time}, "
                f"position {position} is {event_time}"
            )
        previous_time = event_time
        delta = int(request_time) - event_time
        if delta < 0:
            raise ValueError(
                f"sequence {sequence!r} event time is later than request time "
                f"at raw row {raw_row}, position {position}: delta_ms={delta}"
            )
        result.append(_transform_delta(delta))
    return result


def _output_array(
    pa: Any,
    column: str,
    values: list[Any],
    *,
    scalar_features: set[str],
    bag_features: set[str],
    sequence_columns: set[str],
    time_delta_columns: set[str],
    label_columns: set[str],
    integer_request_columns: set[str],
    dictionary_encode: bool = False,
) -> Any:
    if dictionary_encode:
        if column in time_delta_columns:
            value_type = pa.list_(pa.float32())
        else:
            value_type = pa.list_(pa.int64())
        dictionary_values: list[Any] = []
        dictionary_index_by_identity: dict[int, int] = {}
        indices: list[int | None] = []
        for value in values:
            if value is None:
                indices.append(None)
                continue
            identity = id(value)
            dictionary_index = dictionary_index_by_identity.get(identity)
            if dictionary_index is None:
                dictionary_index = len(dictionary_values)
                dictionary_index_by_identity[identity] = dictionary_index
                dictionary_values.append(value)
            indices.append(dictionary_index)
        dictionary = pa.array(dictionary_values, type=value_type)
        return pa.DictionaryArray.from_arrays(
            pa.array(indices, type=pa.int32()),
            dictionary,
        )
    if column in time_delta_columns:
        return pa.array(values, type=pa.list_(pa.float32()))
    if column in bag_features or column in sequence_columns:
        return pa.array(values, type=pa.list_(pa.int64()))
    if (
        column in scalar_features
        or column in label_columns
        or column in integer_request_columns
    ):
        return pa.array(values, type=pa.int64())
    return pa.array(values)


def _candidate_metadata_arrow_type(pa: Any, raw_type: Any) -> Any:
    """Derive the scalar output type from a candidate-list Arrow field."""

    while pa.types.is_dictionary(raw_type):
        raw_type = raw_type.value_type
    if not _is_arrow_list_type(pa, raw_type):
        raise ValueError(
            f"candidate metadata must be list-valued, got Arrow type {raw_type}"
        )
    output_type = raw_type.value_type
    while pa.types.is_dictionary(output_type):
        output_type = output_type.value_type
    # Some producers retain the singleton feature axis (list<list<T>>), while
    # others write candidate metadata directly as list<T>.
    if _is_arrow_list_type(pa, output_type):
        output_type = output_type.value_type
        while pa.types.is_dictionary(output_type):
            output_type = output_type.value_type
    return output_type


@dataclass(frozen=True)
class _MdlRankMixerAdapterPlan:
    context_features: tuple[str, ...]
    item_features: tuple[str, ...]
    bag_features: frozenset[str]
    ups_types: tuple[str, ...]
    request_columns: tuple[str, ...]
    request_maps: Mapping[str, Mapping[Any, int]]
    coarse_scene: _CoarseScenePlan | None
    integer_request_columns: frozenset[str]
    labels: Mapping[str, str]
    label_masks: Mapping[str, str]
    label_missing_values: Mapping[str, tuple[Any, ...]]
    candidate_position_column: str | None
    candidate_metadata_columns: tuple[str, ...]
    column_aliases: Mapping[str, tuple[str, ...]]
    time_delta_outputs: Mapping[str, str]
    time_delta_transform: str
    sequence_max_lengths: Mapping[str, int]
    global_sequence_max_length: int | None
    compact_request_lists: bool
    request_time_column: str
    aligned_groups: tuple[tuple[str, ...], ...]
    required: tuple[str, ...]
    required_set: frozenset[str]
    context_set: frozenset[str]
    item_set: frozenset[str]
    scalar_features: frozenset[str]
    label_columns: frozenset[str]
    label_mask_columns: frozenset[str]
    sequence_columns_by_type: Mapping[str, tuple[str, ...]]
    sequence_columns: frozenset[str]
    time_delta_columns: frozenset[str]
    label_output_columns: tuple[str, ...]
    label_mask_output_columns: tuple[str, ...]
    candidate_metadata_output_columns: tuple[str, ...]
    sequence_output_columns: tuple[str, ...]
    item_output_columns: tuple[str, ...]
    request_output_columns: tuple[str, ...]
    compact_list_columns: frozenset[str]
    raw_sequence_columns: frozenset[str]
    integer_output_columns: frozenset[str]


def _build_mdl_rankmixer_adapter_plan(context: Any) -> _MdlRankMixerAdapterPlan:
    options = context.options
    context_features = _string_list(options, "context_features")
    item_features = _string_list(options, "item_features")
    bag_features = frozenset(_string_list(options, "multivalue_features"))
    for obsolete in (
        "request_shared_features",
        "request_axis_item_features",
        "candidate_axis_context_features",
    ):
        if obsolete in options:
            raise ValueError(
                f"adapter option {obsolete!r} is removed; context_features are "
                "request-axis and item_features are candidate-axis"
            )
    ups_types = _string_list(options, "ups_types")
    request_columns = _string_list(options, "request_columns")
    request_maps = _request_value_maps(options, set(request_columns))
    coarse_scene = _coarse_scene_plan(options, set(request_columns))
    integer_request_columns = frozenset(
        _string_list(options, "integer_request_columns")
    )
    labels = _mapping(options, "labels")
    label_masks = _mapping(options, "label_masks")
    if label_masks and set(label_masks) != set(labels):
        missing = sorted(set(labels) - set(label_masks))
        unknown = sorted(set(label_masks) - set(labels))
        details = []
        if missing:
            details.append("missing tasks: " + ", ".join(missing))
        if unknown:
            details.append("unknown tasks: " + ", ".join(unknown))
        raise ValueError(
            "adapter label_masks must match labels exactly; " + "; ".join(details)
        )
    label_missing_values = _label_missing_values(options, labels)
    if any(label_missing_values.values()) and not label_masks:
        raise ValueError(
            "adapter label_missing_values requires label_masks so missing labels cannot become negatives"
        )
    raw_candidate_position_column = options.get("candidate_position_column")
    candidate_position_column = (
        None
        if raw_candidate_position_column is None
        else str(raw_candidate_position_column)
    )
    if candidate_position_column == "":
        raise ValueError("adapter candidate_position_column must be a non-empty name")
    candidate_metadata_columns = _string_list(options, "candidate_metadata_columns")
    column_aliases = _column_aliases(options)
    time_delta_outputs = _mapping(options, "time_delta_outputs")
    time_delta_transform = str(options.get("time_delta_transform", "raw_ms"))
    sequence_max_lengths = _positive_int_mapping(options, "sequence_max_lengths")
    global_sequence_max_length = options.get("global_sequence_max_length")
    if global_sequence_max_length is not None and (
        type(global_sequence_max_length) is not int
        or global_sequence_max_length <= 0
    ):
        raise ValueError(
            "adapter option 'global_sequence_max_length' must be a positive integer or null"
        )
    compact_request_lists = options.get("compact_request_lists", False)
    if type(compact_request_lists) is not bool:
        raise ValueError("adapter option 'compact_request_lists' must be a boolean")
    if time_delta_transform not in {"raw_ms", "seconds", "log1p_seconds"}:
        raise ValueError(
            "adapter option 'time_delta_transform' must be raw_ms, seconds, "
            "or log1p_seconds"
        )
    request_time_column = str(options.get("request_time_column", "impr_time"))
    aligned_groups_raw = options.get("aligned_multivalue_groups", ())
    if isinstance(aligned_groups_raw, (str, bytes)) or not isinstance(
        aligned_groups_raw, Sequence
    ):
        raise ValueError(
            "adapter option 'aligned_multivalue_groups' must be a list of lists"
        )
    aligned_groups = tuple(
        tuple(str(item) for item in group) for group in aligned_groups_raw
    )

    context_set = frozenset(context_features)
    item_set = frozenset(item_features)
    if context_set & item_set:
        raise ValueError("context_features and item_features must be disjoint")
    known_alias_targets = (
        context_set
        | item_set
        | set(request_columns)
        | set(labels.values())
        | set(candidate_metadata_columns)
    )
    unknown_alias_targets = sorted(set(column_aliases) - known_alias_targets)
    if unknown_alias_targets:
        raise ValueError(
            "column_aliases contains unknown canonical fields: "
            + ", ".join(unknown_alias_targets)
        )
    if not bag_features <= context_set | item_set:
        unknown = sorted(bag_features - context_set - item_set)
        raise ValueError(
            "multivalue_features contains unknown fields: " + ", ".join(unknown)
        )
    for group in aligned_groups:
        if not group or not set(group) <= bag_features:
            raise ValueError(
                "every aligned_multivalue_groups entry must contain configured multivalue fields"
            )
    if set(time_delta_outputs) - set(ups_types):
        raise ValueError("time_delta_outputs contains an unknown UPS type")
    if global_sequence_max_length is not None:
        missing_global_times = sorted(set(ups_types) - set(time_delta_outputs))
        if missing_global_times:
            raise ValueError(
                "adapter global_sequence_max_length requires a time_delta_outputs "
                "entry for every UPS type; missing: "
                + ", ".join(missing_global_times)
            )
    unknown_sequence_limits = sorted(set(sequence_max_lengths) - set(ups_types))
    if unknown_sequence_limits:
        raise ValueError(
            "sequence_max_lengths contains an unknown UPS type: "
            + ", ".join(unknown_sequence_limits)
        )

    required = tuple(context.required_columns)
    required_set = frozenset(required)
    scalar_features = frozenset((context_set | item_set) - bag_features)
    label_columns = frozenset(labels.values())
    label_mask_columns = frozenset(label_masks.values())
    if label_columns & label_mask_columns:
        raise ValueError("adapter label and label-mask output columns must be disjoint")
    generated_candidate_columns = frozenset(
        [
            *candidate_metadata_columns,
            *(
                ()
                if candidate_position_column is None
                else (candidate_position_column,)
            ),
        ]
    )
    expected_generated_count = len(candidate_metadata_columns) + int(
        candidate_position_column is not None
    )
    if len(generated_candidate_columns) != expected_generated_count:
        raise ValueError(
            "candidate_position_column and candidate_metadata_columns must use "
            "distinct output names"
        )
    sequence_columns_by_type = {
        ups: tuple(
            column
            for column in required
            if column.startswith(f"{ups}_x_") and column != time_delta_outputs.get(ups)
        )
        for ups in ups_types
    }
    sequence_columns = frozenset(
        column for columns in sequence_columns_by_type.values() for column in columns
    )
    time_delta_columns = frozenset(time_delta_outputs.values())
    derived_request_columns = (
        frozenset() if coarse_scene is None else coarse_scene.derived_columns
    )
    if derived_request_columns & (
        context_set
        | item_set
        | set(request_columns)
        | label_columns
        | label_mask_columns
        | sequence_columns
        | time_delta_columns
        | generated_candidate_columns
    ):
        raise ValueError(
            "coarse scene derived columns conflict with other adapter outputs: "
            + ", ".join(
                sorted(
                    derived_request_columns
                    & (
                        context_set
                        | item_set
                        | set(request_columns)
                        | label_columns
                        | label_mask_columns
                        | sequence_columns
                        | time_delta_columns
                        | generated_candidate_columns
                    )
                )
            )
        )
    generated_overlap = generated_candidate_columns & (
        context_set
        | item_set
        | set(request_columns)
        | label_columns
        | label_mask_columns
        | sequence_columns
        | time_delta_columns
        | derived_request_columns
    )
    if generated_overlap:
        raise ValueError(
            "generated candidate identity columns conflict with feature/label outputs: "
            + ", ".join(sorted(generated_overlap))
        )
    label_output_columns = tuple(
        column for column in required if column in label_columns
    )
    label_mask_output_columns = tuple(
        column for column in required if column in label_mask_columns
    )
    candidate_metadata_output_columns = tuple(
        column for column in required if column in generated_candidate_columns
    )
    sequence_output_columns = tuple(
        column
        for column in required
        if column not in label_columns
        and column not in label_mask_columns
        and column not in generated_candidate_columns
        and (column in sequence_columns or column in time_delta_columns)
    )
    item_output_columns = tuple(
        column
        for column in required
        if column not in label_columns
        and column not in label_mask_columns
        and column not in generated_candidate_columns
        and column not in sequence_columns
        and column not in time_delta_columns
        and column in item_set
    )
    request_output_columns = tuple(
        column
        for column in required
        if column not in label_columns
        and column not in label_mask_columns
        and column not in generated_candidate_columns
        and column not in sequence_columns
        and column not in time_delta_columns
        and column not in item_set
        and (
            column in context_set
            or column in request_columns
            or column in derived_request_columns
        )
    )
    classified_output_columns = {
        *label_output_columns,
        *label_mask_output_columns,
        *candidate_metadata_output_columns,
        *sequence_output_columns,
        *item_output_columns,
        *request_output_columns,
    }
    unknown_required = sorted(required_set - classified_output_columns)
    if unknown_required:
        raise ValueError(
            "adapter options do not define required output columns: "
            + ", ".join(unknown_required)
        )
    compact_list_columns = frozenset(
        (bag_features & context_set) | sequence_columns | time_delta_columns
    )
    raw_sequence_columns = frozenset(
        {
            *sequence_columns,
            *(f"{ups}_x_time" for ups in ups_types if ups in time_delta_outputs),
        }
    )
    integer_output_columns = frozenset(
        {
            *integer_request_columns,
            *derived_request_columns,
            *label_mask_columns,
            *(
                ()
                if candidate_position_column is None
                else (candidate_position_column,)
            ),
        }
    )
    return _MdlRankMixerAdapterPlan(
        context_features=context_features,
        item_features=item_features,
        bag_features=bag_features,
        ups_types=ups_types,
        request_columns=request_columns,
        request_maps=request_maps,
        coarse_scene=coarse_scene,
        integer_request_columns=integer_request_columns,
        labels=labels,
        label_masks=label_masks,
        label_missing_values=label_missing_values,
        candidate_position_column=candidate_position_column,
        candidate_metadata_columns=candidate_metadata_columns,
        column_aliases=column_aliases,
        time_delta_outputs=time_delta_outputs,
        time_delta_transform=time_delta_transform,
        sequence_max_lengths=sequence_max_lengths,
        global_sequence_max_length=global_sequence_max_length,
        compact_request_lists=compact_request_lists,
        request_time_column=request_time_column,
        aligned_groups=aligned_groups,
        required=required,
        required_set=required_set,
        context_set=context_set,
        item_set=item_set,
        scalar_features=scalar_features,
        label_columns=label_columns,
        label_mask_columns=label_mask_columns,
        sequence_columns_by_type=sequence_columns_by_type,
        sequence_columns=sequence_columns,
        time_delta_columns=time_delta_columns,
        label_output_columns=label_output_columns,
        label_mask_output_columns=label_mask_output_columns,
        candidate_metadata_output_columns=candidate_metadata_output_columns,
        sequence_output_columns=sequence_output_columns,
        item_output_columns=item_output_columns,
        request_output_columns=request_output_columns,
        compact_list_columns=compact_list_columns,
        raw_sequence_columns=raw_sequence_columns,
        integer_output_columns=integer_output_columns,
    )


def _mdl_rankmixer_adapter_plan(context: Any) -> _MdlRankMixerAdapterPlan:
    # Only the repository-owned immutable context has stable options. External
    # tests/adapters may pass mutable SimpleNamespace objects, which are rebuilt
    # so mutations remain observable.
    if isinstance(context, ParquetAdapterContext):
        cached = context._runtime_cache.get("mdl_rankmixer_plan")
        if isinstance(cached, _MdlRankMixerAdapterPlan):
            return cached
        plan = _build_mdl_rankmixer_adapter_plan(context)
        context._runtime_cache["mdl_rankmixer_plan"] = plan
        return plan
    return _build_mdl_rankmixer_adapter_plan(context)


def _resolve_physical_column(
    schema_names: set[str],
    canonical: str,
    column_aliases: Mapping[str, tuple[str, ...]],
) -> str | None:
    if canonical in schema_names:
        return canonical
    for alias in column_aliases.get(canonical, ()):
        if alias in schema_names:
            return alias
    return None


def build_arrow_axis_source(
    table: Any,
    *,
    context: Any,
    adapter_plan: _MdlRankMixerAdapterPlan | None = None,
    request_id_column: str,
) -> Any:
    """Build ``ArrowAxisSource`` from control columns only (no full pylist).

    Payload columns remain in ``table``. Only indices / request-id / UPS
    membership are converted to Python/NumPy for shuffle/bucket/pack.
    """

    pa, _pc, _ds, _pq = _require_pyarrow()
    plan = adapter_plan or _mdl_rankmixer_adapter_plan(context)
    trusted_input = bool(getattr(context, "trusted_input", False))
    validate_structure = not trusted_input
    schema_names = set(table.schema.names)

    physical_columns: dict[str, str] = {}
    for canonical in {
        *plan.context_features,
        *plan.item_features,
        *plan.request_columns,
        *plan.sequence_columns,
        *plan.label_columns,
        *plan.candidate_metadata_columns,
        plan.request_time_column,
        request_id_column,
        *(f"{ups}_x_time" for ups in plan.ups_types),
        *(f"{ups}_x_indices" for ups in plan.ups_types),
        "context_indices",
        "target_indices",
    }:
        resolved = _resolve_physical_column(
            schema_names, canonical, plan.column_aliases
        )
        if resolved is not None:
            physical_columns[canonical] = resolved

    def _control_pylist(canonical: str) -> list[Any]:
        physical = physical_columns.get(canonical)
        if physical is None:
            raise ValueError(f"missing control column {canonical!r}")
        return table[physical].to_pylist()

    if (
        "context_indices" not in physical_columns
        or "target_indices" not in physical_columns
    ):
        raise ValueError(
            "arrow_axis path requires agg layout with context_indices and target_indices"
        )
    if request_id_column not in physical_columns:
        raise ValueError(f"missing request_id column {request_id_column!r}")

    context_indices_rows = _control_pylist("context_indices")
    target_indices_rows = _control_pylist("target_indices")
    request_id_rows = _control_pylist(request_id_column)
    ups_index_rows: dict[str, list[Any]] = {}
    for ups in plan.ups_types:
        index_name = f"{ups}_x_indices"
        if index_name not in physical_columns:
            # Null / absent UPS histories → empty membership for every request.
            ups_index_rows[ups] = [None] * table.num_rows
        else:
            ups_index_rows[ups] = _control_pylist(index_name)
    ups_time_rows: dict[str, list[Any]] = {}
    if plan.global_sequence_max_length is not None:
        for ups in plan.ups_types:
            time_name = f"{ups}_x_time"
            if time_name not in physical_columns:
                raise ValueError(
                    f"adapter global sequence selection requires column {time_name!r}"
                )
            time_column = table[physical_columns[time_name]]
            if hasattr(time_column, "combine_chunks"):
                time_column = time_column.combine_chunks()
            ups_time_rows[ups] = _arrow_array_to_pylist(pa, time_column)

    request_id_to_slot: dict[Any, int] = {}
    request_ids: list[Any] = []
    request_raw_rows: list[int] = []
    request_local_positions: list[int] = []
    # Per-request UPS token positions (first-wins with request_id).
    ups_positions_by_slot: dict[str, list[list[int]]] = {
        ups: [] for ups in plan.ups_types
    }
    candidate_to_request: list[int] = []
    candidate_raw_rows: list[int] = []
    candidate_locals: list[int] = []
    candidate_positions: list[int] = []

    for raw_row in range(table.num_rows):
        context_indices = _as_list(
            context_indices_rows[raw_row],
            column="context_indices",
            row_index=raw_row,
            validate_contract=validate_structure,
        )
        target_indices = _as_list(
            target_indices_rows[raw_row],
            column="target_indices",
            row_index=raw_row,
            validate_contract=validate_structure,
        )
        positions = _request_positions(
            context_indices,
            raw_row=raw_row,
            validate_contract=validate_structure,
        )
        request_count = len(positions)
        candidate_requests = [
            _request_index(
                value,
                column="target_indices",
                row_index=raw_row,
                validate_contract=validate_structure,
            )
            for value in target_indices
        ]
        request_id_values = _as_list(
            request_id_rows[raw_row],
            column=request_id_column,
            row_index=raw_row,
            validate_contract=validate_structure,
        )
        if validate_structure and len(request_id_values) != request_count:
            raise ValueError(
                f"agg request-level column {request_id_column!r} has length "
                f"{len(request_id_values)}, expected {request_count} at raw row {raw_row}"
            )

        known_requests = set(positions)
        membership_positions: dict[str, dict[int, list[int]]] = {}
        membership_lengths: dict[str, int] = {}
        for ups in plan.ups_types:
            index_column = f"{ups}_x_indices"
            memberships_raw = ups_index_rows[ups][raw_row]
            if memberships_raw is None:
                memberships: list[Any] = []
            else:
                memberships = _as_list(
                    memberships_raw,
                    column=index_column,
                    row_index=raw_row,
                    validate_contract=validate_structure,
                )
            membership_lengths[ups] = len(memberships)
            membership_positions[ups] = _sequence_membership_positions(
                memberships,
                known_requests=known_requests,
                index_column=index_column,
                raw_row=raw_row,
                validate_structure=validate_structure,
            )

        # First-wins request slots keyed by request_id (search_id).
        row_request_slots: dict[int, int] = {}
        for request_index in dict.fromkeys(candidate_requests):
            if request_index not in positions:
                if validate_structure:
                    raise ValueError(
                        f"target request {request_index} has no context at raw row {raw_row}"
                    )
                continue
            request_position = positions[request_index]
            request_id = _scalarize(
                request_id_values[request_position],
                column=request_id_column,
                raw_row=raw_row,
                logical_row=request_index,
                validate_contract=validate_structure,
            )
            if request_id is None:
                raise ValueError(
                    f"request_id column {request_id_column!r} contains null "
                    f"at raw row {raw_row}, request {request_index}"
                )
            try:
                slot = request_id_to_slot.get(request_id)
            except TypeError as error:
                raise ValueError(
                    f"request_id column {request_id_column!r} must contain hashable scalars"
                ) from error
            if slot is None:
                slot = len(request_ids)
                request_id_to_slot[request_id] = slot
                request_ids.append(request_id)
                request_raw_rows.append(raw_row)
                request_local_positions.append(int(request_position))
                selected_by_type: dict[str, list[int]]
                if plan.global_sequence_max_length is not None:
                    candidates_by_type = {
                        ups: list(
                            membership_positions[ups].get(request_index, ())[
                                : plan.global_sequence_max_length
                            ]
                        )
                        for ups in plan.ups_types
                    }
                    candidate_times = {
                        ups: _select_sequence(
                            ups_time_rows[ups][raw_row],
                            candidates_by_type[ups],
                            expected_length=membership_lengths[ups],
                            column=f"{ups}_x_time",
                            raw_row=raw_row,
                            validate_structure=validate_structure,
                            validate_payload=validate_structure,
                        )
                        for ups in plan.ups_types
                    }
                    selected_by_type = _select_global_recent_sequence_positions(
                        candidates_by_type,
                        candidate_times,
                        ups_types=plan.ups_types,
                        max_length=plan.global_sequence_max_length,
                        raw_row=raw_row,
                        validate_contract=validate_structure,
                    )
                else:
                    selected_by_type = {}
                    for ups in plan.ups_types:
                        selected = list(
                            membership_positions[ups].get(request_index, ())
                        )
                        max_length = plan.sequence_max_lengths.get(ups)
                        if max_length is not None:
                            selected = selected[: int(max_length)]
                        selected_by_type[ups] = selected
                for ups in plan.ups_types:
                    ups_positions_by_slot[ups].append(selected_by_type[ups])
            row_request_slots[request_index] = slot

        request_ordinals: defaultdict[int, int] = defaultdict(int)
        for local, request_index in enumerate(candidate_requests):
            slot = row_request_slots[request_index]
            candidate_to_request.append(slot)
            candidate_raw_rows.append(raw_row)
            candidate_locals.append(local)
            candidate_positions.append(request_ordinals[request_index])
            request_ordinals[request_index] += 1

    axis_plan = AggAxisPlan(
        n_candidates=len(candidate_to_request),
        n_requests=len(request_ids),
        request_ids=tuple(request_ids),
        candidate_to_request=np.asarray(candidate_to_request, dtype=np.int64),
        request_raw_rows=np.asarray(request_raw_rows, dtype=np.int64),
        request_local_positions=np.asarray(request_local_positions, dtype=np.int64),
        candidate_raw_rows=np.asarray(candidate_raw_rows, dtype=np.int64),
        candidate_locals=np.asarray(candidate_locals, dtype=np.int64),
        ups_token_positions={
            ups: tuple(
                np.asarray(positions, dtype=np.int64)
                for positions in ups_positions_by_slot[ups]
            )
            for ups in plan.ups_types
        },
        candidate_positions=(
            np.asarray(candidate_positions, dtype=np.int64)
            if plan.candidate_position_column is not None
            else None
        ),
    )

    # Include derived coarse-scene columns in the request feature set.
    request_feature_columns = list(plan.request_output_columns)
    if plan.coarse_scene is not None:
        for derived in (
            plan.coarse_scene.index_column,
            plan.coarse_scene.prior_id_column,
        ):
            if derived not in request_feature_columns:
                request_feature_columns.append(derived)

    candidate_metadata = list(plan.candidate_metadata_output_columns)
    if (
        plan.candidate_position_column is not None
        and plan.candidate_position_column not in candidate_metadata
        and plan.candidate_position_column in plan.required_set
    ):
        candidate_metadata.append(plan.candidate_position_column)

    runtime_cache = getattr(context, "_runtime_cache", None)
    if isinstance(runtime_cache, dict) and table.num_rows > 0:
        runtime_cache["mdl_rankmixer_first_batch_adapted"] = True

    source = ArrowAxisSource(
        table=table,
        plan=axis_plan,
        request_feature_columns=tuple(request_feature_columns),
        sequence_feature_columns=tuple(plan.sequence_output_columns),
        item_feature_columns=tuple(plan.item_output_columns),
        label_feature_columns=tuple(plan.label_output_columns),
        label_mask_feature_columns=tuple(plan.label_mask_output_columns),
        candidate_metadata_columns=tuple(candidate_metadata),
        bag_features=plan.bag_features,
        physical_columns=physical_columns,
        ups_types=plan.ups_types,
        sequence_columns_by_type=plan.sequence_columns_by_type,
        time_delta_outputs=plan.time_delta_outputs,
        time_delta_transform=plan.time_delta_transform,
        request_time_column=plan.request_time_column,
        request_id_column=request_id_column,
        request_maps=plan.request_maps,
        coarse_scene=plan.coarse_scene,
        label_masks=plan.label_masks,
        label_missing_values=plan.label_missing_values,
        labels=plan.labels,
        trusted_input=trusted_input,
    )
    # Materialize once in the adapter worker (ProcessPool owns the CPU work).
    # Drop the Arrow table before return so the parent only unpickles the
    # Python bundle — not table + bundle. Pack then copies references.
    materialize_arrow_axis_source(source)
    return source


def adapt_mdl_rankmixer_parquet(table: Any, *, context: Any) -> Any:
    """Convert one raw Arrow table from agg or req layout to per-item rows."""

    pa, _pc, _ds, _pq = _require_pyarrow()
    plan = _mdl_rankmixer_adapter_plan(context)
    context_features = plan.context_features
    item_features = plan.item_features
    bag_features = plan.bag_features
    ups_types = plan.ups_types
    request_columns = plan.request_columns
    request_maps = plan.request_maps
    coarse_scene = plan.coarse_scene
    integer_request_columns = plan.integer_request_columns
    labels = plan.labels
    label_masks = plan.label_masks
    label_missing_values = plan.label_missing_values
    candidate_position_column = plan.candidate_position_column
    candidate_metadata_columns = plan.candidate_metadata_columns
    column_aliases = plan.column_aliases
    time_delta_outputs = plan.time_delta_outputs
    time_delta_transform = plan.time_delta_transform
    sequence_max_lengths = plan.sequence_max_lengths
    global_sequence_max_length = plan.global_sequence_max_length
    compact_request_lists = plan.compact_request_lists
    request_time_column = plan.request_time_column
    aligned_groups = plan.aligned_groups
    context_set = plan.context_set
    item_set = plan.item_set
    required = plan.required
    required_set = plan.required_set
    scalar_features = plan.scalar_features
    label_columns = plan.label_columns
    sequence_columns_by_type = plan.sequence_columns_by_type
    sequence_columns = plan.sequence_columns
    time_delta_columns = plan.time_delta_columns
    label_output_columns = plan.label_output_columns
    label_mask_output_columns = plan.label_mask_output_columns
    candidate_metadata_output_columns = plan.candidate_metadata_output_columns
    sequence_output_columns = plan.sequence_output_columns
    item_output_columns = plan.item_output_columns
    request_output_columns = plan.request_output_columns
    compact_list_columns = plan.compact_list_columns
    runtime_cache = getattr(context, "_runtime_cache", None)
    trusted_input = bool(getattr(context, "trusted_input", False))
    axis_separated = bool(
        isinstance(runtime_cache, dict) and runtime_cache.get("axis_separated")
    )
    arrow_axis = bool(
        isinstance(runtime_cache, dict) and runtime_cache.get("arrow_axis")
    )
    axis_request_id_column = (
        runtime_cache.get("axis_request_id_column")
        if isinstance(runtime_cache, dict)
        else None
    )
    if (axis_separated or arrow_axis) and not axis_request_id_column:
        raise ValueError(
            "axis_separated/arrow_axis adapt requires "
            "runtime_cache['axis_request_id_column']"
        )

    cardinality_auditor: FeatureCardinalityAuditor | None = None
    if isinstance(runtime_cache, dict):
        cached_auditor = runtime_cache.get("cardinality_auditor")
        if isinstance(cached_auditor, FeatureCardinalityAuditor):
            cardinality_auditor = cached_auditor
    soft_cardinality_audit = bool(
        cardinality_auditor is not None and cardinality_auditor.soft
    )
    raw_sample_already_validated = isinstance(
        runtime_cache, dict
    ) and runtime_cache.get("mdl_rankmixer_raw_sample_validated", False)
    # Soft cardinality audit already walks every configured field; skip the
    # one-row hard warm-up so the first multi-valued scalar does not abort the
    # aggregate report.
    if (
        trusted_input
        and table.num_rows > 0
        and not raw_sample_already_validated
        and not soft_cardinality_audit
    ):
        sample_runtime_cache: dict[str, Any] = {"mdl_rankmixer_plan": plan}
        if axis_separated or arrow_axis:
            # Warm-up always uses the Python axis path so structure checks do
            # not depend on the Arrow-native gather implementation.
            sample_runtime_cache["axis_separated"] = True
            sample_runtime_cache["axis_request_id_column"] = axis_request_id_column
        sample_context = ParquetAdapterContext(
            split_name=str(getattr(context, "split_name", "unknown")),
            required_columns=tuple(context.required_columns),
            options=context.options,
            trusted_input=False,
            _runtime_cache=sample_runtime_cache,
        )
        # Validate only one physical Parquet row. The full table is converted
        # below with diagnostics disabled.
        adapt_mdl_rankmixer_parquet(table.slice(0, 1), context=sample_context)
        if isinstance(runtime_cache, dict):
            runtime_cache["mdl_rankmixer_raw_sample_validated"] = True

    if arrow_axis:
        return build_arrow_axis_source(
            table,
            context=context,
            adapter_plan=plan,
            request_id_column=str(axis_request_id_column),
        )

    first_batch_already_adapted = isinstance(runtime_cache, dict) and runtime_cache.get(
        "mdl_rankmixer_first_batch_adapted", False
    )
    complete_label_contract = not label_masks and not any(label_missing_values.values())
    # Under trusted_input the producer owns schema/shape correctness, so skip
    # per-row structure and payload diagnostics on the hot path. Non-trusted
    # runs keep both checks (and the one-row sample warm-up above).
    validate_payload = not trusted_input and (
        not first_batch_already_adapted or not complete_label_contract
    )
    validate_structure = not trusted_input
    validate_row_contract = validate_structure
    output: dict[str, list[Any]] = {column: [] for column in required}
    # Axis-separated accumulators (direct path). First-wins on request_id.
    axis_request_id_to_slot: dict[Any, int] = {}
    axis_request_ids: list[Any] = []
    axis_request_raw_rows: list[int] = []
    axis_candidate_to_request: list[int] = []
    axis_candidate_raw_rows: list[int] = []
    axis_request_features: dict[str, list[Any]] = {
        column: [] for column in request_output_columns
    }
    axis_sequence_features: dict[str, list[Any]] = {
        column: [] for column in sequence_output_columns
    }
    axis_item_features: dict[str, list[Any]] = {
        column: [] for column in item_output_columns
    }
    axis_label_features: dict[str, list[Any]] = {
        column: [] for column in label_output_columns
    }
    axis_label_mask_features: dict[str, list[Any]] = {
        column: [] for column in label_mask_output_columns
    }
    axis_candidate_metadata: dict[str, list[Any]] = {
        column: [] for column in candidate_metadata_output_columns
    }
    raw, validated_flat_sequence_columns = _adapter_table_to_python(
        table,
        plan.raw_sequence_columns,
        validate_contract=validate_payload,
    )
    candidate_metadata_types: dict[str, Any] = {}
    schema_names = set(table.schema.names)
    for column in candidate_metadata_columns:
        present = [
            name
            for name in (column, *column_aliases.get(column, ()))
            if name in schema_names
        ]
        if len(present) == 1:
            candidate_metadata_types[column] = _candidate_metadata_arrow_type(
                pa,
                table.schema.field(present[0]).type,
            )
    validated_flat = set(validated_flat_sequence_columns)
    alias_to_canonical: dict[str, str] = {}
    for canonical, aliases in column_aliases.items():
        alias_to_canonical.update({alias: canonical for alias in aliases})
        present = [name for name in (canonical, *aliases) if name in raw]
        if validate_row_contract and len(present) > 1:
            raise ValueError(
                f"raw schema contains multiple aliases for {canonical!r}: {present}"
            )
        if present and present[0] != canonical:
            raw[canonical] = raw[present[0]]
            if present[0] in validated_flat:
                validated_flat.add(canonical)
    validated_flat_sequence_columns = frozenset(validated_flat)

    has_context_indices = "context_indices" in raw
    has_target_indices = "target_indices" in raw
    if validate_row_contract and has_context_indices != has_target_indices:
        raise ValueError(
            "agg/req detection requires both context_indices and target_indices or neither"
        )
    is_agg = has_context_indices
    nested_req_context = {
        alias_to_canonical.get(field.name, field.name)
        for field in table.schema
        if alias_to_canonical.get(field.name, field.name) in context_set
        and (pa.types.is_list(field.type) or pa.types.is_large_list(field.type))
        and (
            pa.types.is_list(field.type.value_type)
            or pa.types.is_large_list(field.type.value_type)
        )
    }

    for raw_row in range(table.num_rows):
        row = {column: values[raw_row] for column, values in raw.items()}
        if is_agg:
            context_indices = _as_list(
                row["context_indices"],
                column="context_indices",
                row_index=raw_row,
                validate_contract=validate_row_contract,
            )
            target_indices = _as_list(
                row["target_indices"],
                column="target_indices",
                row_index=raw_row,
                validate_contract=validate_row_contract,
            )
            positions = _request_positions(
                context_indices,
                raw_row=raw_row,
                validate_contract=validate_row_contract,
            )
            candidate_requests = [
                _request_index(
                    value,
                    column="target_indices",
                    row_index=raw_row,
                    validate_contract=validate_row_contract,
                )
                for value in target_indices
            ]
        else:
            positions = {0: 0}
            candidate_count = _candidate_count_req(
                row,
                item_features,
                tuple(labels.values()),
                raw_row,
                validate_contract=validate_row_contract,
            )
            candidate_requests = [0] * candidate_count

        candidate_count = len(candidate_requests)
        request_count = len(positions)
        item_arrays: dict[str, list[Any]] = {}
        if is_agg:
            for column in item_features:
                if validate_row_contract and column not in row:
                    raise ValueError(f"missing item column {column!r}")
                outer = _as_list(
                    row[column],
                    column=column,
                    row_index=raw_row,
                    validate_contract=validate_row_contract,
                )
                if validate_structure and len(outer) != candidate_count:
                    raise ValueError(
                        f"candidate-axis feature {column!r} length {len(outer)} != "
                        f"candidate count {candidate_count} at raw row {raw_row}"
                    )
                item_arrays[column] = outer
        else:
            for column in item_features:
                if validate_row_contract and column not in row:
                    raise ValueError(f"missing item column {column!r}")
                outer = _as_list(
                    row[column],
                    column=column,
                    row_index=raw_row,
                    validate_contract=validate_row_contract,
                )
                if validate_structure and len(outer) != candidate_count:
                    raise ValueError(
                        f"candidate-axis feature {column!r} length {len(outer)} != "
                        f"candidate count {candidate_count} at raw row {raw_row}"
                    )
                item_arrays[column] = outer

        label_arrays: dict[str, list[Any]] = {}
        for task, column in labels.items():
            mask_column = label_masks.get(task)
            if column not in required_set and mask_column not in required_set:
                continue
            if validate_row_contract and column not in row:
                raise ValueError(f"missing label column {column!r} for task {task!r}")
            if row[column] is None and _is_missing_label(
                None,
                label_missing_values[task],
            ):
                values = [None] * candidate_count
            else:
                values = _as_list(
                    row[column],
                    column=column,
                    row_index=raw_row,
                    validate_contract=validate_row_contract,
                )
            if validate_structure and len(values) != candidate_count:
                raise ValueError(
                    f"label {column!r} length {len(values)} != candidate count "
                    f"{candidate_count} at raw row {raw_row}"
                )
            # Scalarize at entry for both complete and masked paths. Never pass
            # a soft cardinality auditor: length > 1 must raise, not become None.
            label_arrays[column] = _scalarize_column_values(
                values,
                column=column,
                raw_row=raw_row,
                validate_contract=validate_row_contract,
            )

        candidate_metadata: dict[str, list[Any]] = {}
        request_ordinals: defaultdict[int, int] = defaultdict(int)
        if (
            candidate_position_column is not None
            and candidate_position_column in required_set
        ):
            positions_by_candidate: list[int] = []
            for request_index in candidate_requests:
                positions_by_candidate.append(request_ordinals[request_index])
                request_ordinals[request_index] += 1
            candidate_metadata[candidate_position_column] = positions_by_candidate
        for column in candidate_metadata_columns:
            if column not in required_set:
                continue
            if column not in row:
                candidate_metadata[column] = [None] * candidate_count
                continue
            values = _as_list(
                row[column],
                column=column,
                row_index=raw_row,
                validate_contract=validate_row_contract,
            )
            if validate_structure and len(values) != candidate_count:
                raise ValueError(
                    f"candidate metadata {column!r} length {len(values)} != candidate "
                    f"count {candidate_count} at raw row {raw_row}"
                )
            candidate_metadata[column] = [
                _scalarize(
                    value,
                    column=column,
                    raw_row=raw_row,
                    logical_row=candidate_index,
                    validate_contract=validate_row_contract,
                    auditor=cardinality_auditor,
                )
                for candidate_index, value in enumerate(values)
            ]

        context_arrays: dict[str, Any] = {}
        for column in context_features:
            if validate_row_contract and column not in row:
                raise ValueError(f"missing context column {column!r}")
            if is_agg:
                outer = _as_list(
                    row[column],
                    column=column,
                    row_index=raw_row,
                    validate_contract=validate_row_contract,
                )
                if validate_structure and len(outer) != request_count:
                    raise ValueError(
                        f"request-axis feature {column!r} length {len(outer)} != "
                        f"request count {request_count} at raw row {raw_row}"
                    )
                context_arrays[column] = outer
            else:
                context_arrays[column] = row[column]

        membership_positions: dict[str, dict[int, list[int]]] = {}
        membership_lengths: dict[str, int] = {}
        if is_agg:
            known_requests = set(positions)
            for ups in ups_types:
                index_column = f"{ups}_x_indices"
                if validate_row_contract and index_column not in row:
                    raise ValueError(f"missing UPS indices column {index_column!r}")
                memberships = _as_list(
                    row[index_column],
                    column=index_column,
                    row_index=raw_row,
                    validate_contract=validate_structure,
                )
                # Top-level null/[] mean zero UPS tokens (token-major empty list).
                membership_lengths[ups] = len(memberships)
                membership_positions[ups] = _sequence_membership_positions(
                    memberships,
                    known_requests=known_requests,
                    index_column=index_column,
                    raw_row=raw_row,
                    validate_structure=validate_structure,
                )

        unique_candidate_requests = tuple(dict.fromkeys(candidate_requests))
        if validate_structure:
            for request_index in unique_candidate_requests:
                if request_index not in positions:
                    raise ValueError(
                        f"target request {request_index} has no context at raw row {raw_row}"
                    )

        # Normalize request payloads once, then append whole output columns.
        # The previous candidate-major loop revisited ~169 dictionaries for
        # every item even though Context/UPS are shared by request.
        request_cache: dict[int, dict[str, Any]] = {}
        sequence_cache: dict[int, dict[str, list[Any]]] = {}
        for request_index in unique_candidate_requests:
            request_position = positions[request_index]
            cached: dict[str, Any] = {}
            for column in context_features:
                if is_agg:
                    value = context_arrays[column][request_position]
                else:
                    value = _req_context_value(
                        context_arrays[column],
                        has_request_axis=column in nested_req_context,
                        multivalue=column in bag_features,
                        column=column,
                        raw_row=raw_row,
                        validate_contract=validate_row_contract,
                    )
                cached[column] = (
                    _bag_value(
                        value,
                        column=column,
                        raw_row=raw_row,
                        logical_row=request_index,
                        validate_contract=validate_row_contract,
                        auditor=cardinality_auditor,
                    )
                    if column in bag_features
                    else _scalarize(
                        value,
                        column=column,
                        raw_row=raw_row,
                        logical_row=request_index,
                        validate_contract=validate_row_contract,
                        auditor=cardinality_auditor,
                    )
                )
            for column in request_columns:
                if validate_row_contract and column not in row:
                    raise ValueError(f"missing request-level column {column!r}")
                value = _request_level_value(
                    row[column],
                    request_position=request_position,
                    request_count=request_count,
                    column=column,
                    raw_row=raw_row,
                    agg=is_agg,
                    validate_contract=validate_row_contract,
                )
                if coarse_scene is not None and column == coarse_scene.raw_scene_column:
                    coarse_index, coarse_prior_id = coarse_scene_ids(
                        value,
                        coarse_scene.search_scene_ids,
                        unlisted_policy=coarse_scene.unlisted_policy,
                    )
                    cached[coarse_scene.index_column] = coarse_index
                    cached[coarse_scene.prior_id_column] = coarse_prior_id
                if column in request_maps:
                    value = _map_request_value(
                        value,
                        column=column,
                        mapping=request_maps[column],
                        validate_contract=validate_row_contract,
                    )
                cached[column] = value
            request_cache[request_index] = cached

            cached_sequences: dict[str, list[Any]] = {}
            request_time = _request_level_value(
                row.get(request_time_column),
                request_position=request_position,
                request_count=request_count,
                column=request_time_column,
                raw_row=raw_row,
                agg=is_agg,
                validate_contract=validate_row_contract,
            )
            globally_selected_positions: dict[str, list[int]] | None = None
            global_expected_lengths: dict[str, int] = {}
            if global_sequence_max_length is not None:
                candidate_positions_by_type: dict[str, list[int]] = {}
                candidate_times_by_type: dict[str, Sequence[Any]] = {}
                for ups in ups_types:
                    raw_time_column = f"{ups}_x_time"
                    if validate_row_contract and raw_time_column not in row:
                        raise ValueError(
                            f"missing UPS time column {raw_time_column!r}"
                        )
                    time_values = row.get(raw_time_column)
                    if is_agg:
                        expected = membership_lengths[ups]
                        positions_for_global = list(
                            membership_positions[ups][request_index][
                                :global_sequence_max_length
                            ]
                        )
                    else:
                        raw_times = _as_list(
                            time_values,
                            column=raw_time_column,
                            row_index=raw_row,
                            validate_contract=validate_structure,
                        )
                        expected = len(raw_times)
                        positions_for_global = list(
                            range(min(expected, global_sequence_max_length))
                        )
                    global_expected_lengths[ups] = expected
                    candidate_positions_by_type[ups] = positions_for_global
                    candidate_times_by_type[ups] = _select_sequence(
                        time_values,
                        positions_for_global,
                        expected_length=expected,
                        column=raw_time_column,
                        raw_row=raw_row,
                        validated_flat=(
                            raw_time_column in validated_flat_sequence_columns
                        ),
                        validate_structure=validate_structure,
                        validate_payload=validate_payload,
                    )
                globally_selected_positions = (
                    _select_global_recent_sequence_positions(
                        candidate_positions_by_type,
                        candidate_times_by_type,
                        ups_types=ups_types,
                        max_length=global_sequence_max_length,
                        raw_row=raw_row,
                        validate_contract=validate_payload,
                    )
                )
            # Trusted + flattened UPS rows are NumPy views. Build one index
            # array per UPS and fancy-index every aligned column, instead of
            # ~100× ``_select_sequence`` call/listcomp overhead per request.
            use_numpy_seq = not validate_structure and not validate_payload and is_agg
            for ups in ups_types:
                if globally_selected_positions is not None:
                    selected_positions = globally_selected_positions[ups]
                    expected_length = global_expected_lengths[ups]
                    max_length = None
                else:
                    selected_positions = (
                        membership_positions[ups][request_index] if is_agg else None
                    )
                    expected_length = membership_lengths.get(ups)
                    max_length = sequence_max_lengths.get(ups)
                pos_index: Any = None
                if use_numpy_seq and selected_positions is not None:
                    if max_length is not None:
                        selected_positions = selected_positions[:max_length]
                    if selected_positions:
                        pos_index = np.asarray(selected_positions, dtype=np.int64)
                for column in sequence_columns_by_type[ups]:
                    if validate_row_contract and column not in row:
                        raise ValueError(f"missing UPS column {column!r}")
                    values = row[column]
                    if (
                        use_numpy_seq
                        and column in validated_flat_sequence_columns
                        and isinstance(values, np.ndarray)
                    ):
                        if pos_index is None:
                            cached_sequences[column] = values[:0]
                        else:
                            # Keep NumPy; CompactListColumn packs once at return.
                            cached_sequences[column] = values[pos_index]
                        continue
                    if (
                        use_numpy_seq
                        and values is None
                        and selected_positions is not None
                    ):
                        cached_sequences[column] = []
                        continue
                    cached_sequences[column] = _select_sequence(
                        values,
                        selected_positions,
                        expected_length=expected_length,
                        column=column,
                        raw_row=raw_row,
                        validated_flat=column in validated_flat_sequence_columns,
                        max_length=max_length,
                        validate_structure=validate_structure,
                        validate_payload=validate_payload,
                    )
                if ups in time_delta_outputs:
                    raw_time_column = f"{ups}_x_time"
                    if validate_row_contract and raw_time_column not in row:
                        raise ValueError(f"missing UPS time column {raw_time_column!r}")
                    time_values = row.get(raw_time_column)
                    if (
                        use_numpy_seq
                        and raw_time_column in validated_flat_sequence_columns
                        and isinstance(time_values, np.ndarray)
                    ):
                        event_times = (
                            time_values[:0]
                            if pos_index is None
                            else time_values[pos_index]
                        )
                    else:
                        event_times = _select_sequence(
                            time_values,
                            selected_positions,
                            expected_length=expected_length,
                            column=raw_time_column,
                            raw_row=raw_row,
                            validated_flat=(
                                raw_time_column in validated_flat_sequence_columns
                            ),
                            max_length=max_length,
                            validate_structure=validate_structure,
                            validate_payload=validate_payload,
                        )
                    cached_sequences[time_delta_outputs[ups]] = _time_deltas(
                        event_times,
                        request_time,
                        sequence=ups,
                        raw_row=raw_row,
                        transform=time_delta_transform,
                        validate_contract=validate_payload,
                    )
            sequence_cache[request_index] = cached_sequences

        normalized_items: dict[str, list[Any]] = {}
        for column in item_features:
            if column in bag_features:
                normalized_items[column] = [
                    _bag_value(
                        value,
                        column=column,
                        raw_row=raw_row,
                        logical_row=candidate_index,
                        validate_contract=validate_row_contract,
                        auditor=cardinality_auditor,
                    )
                    for candidate_index, value in enumerate(item_arrays[column])
                ]
            else:
                normalized_items[column] = _scalarize_column_values(
                    item_arrays[column],
                    column=column,
                    raw_row=raw_row,
                    validate_contract=validate_row_contract,
                    auditor=cardinality_auditor,
                )

        if validate_structure:
            for group in aligned_groups:
                for candidate_index in range(candidate_count):
                    lengths = {
                        column: len(normalized_items[column][candidate_index])
                        for column in group
                        if isinstance(
                            normalized_items[column][candidate_index],
                            (list, tuple, np.ndarray),
                        )
                    }
                    if len(lengths) != len(group) or len(set(lengths.values())) != 1:
                        raise ValueError(
                            f"aligned multivalue group mismatch at raw row {raw_row}, "
                            f"candidate {candidate_index}: {lengths}"
                        )

        normalized_labels: dict[str, list[int | None]] = {}
        normalized_label_masks: dict[str, list[int]] = {}
        for task, column in labels.items():
            mask_column = label_masks.get(task)
            if column not in required_set and mask_column not in required_set:
                continue
            values = label_arrays[column]
            complete_labels = mask_column is None and not label_missing_values[task]
            if complete_labels and not axis_separated:
                # Legacy/flat batches still run ``_validate_complete_label_contract``.
                # Keep that cheap path; axis-separated direct never builds flat
                # Arrow, so it must validate below.
                normalized_labels[column] = values
                continue
            # Axis-separated complete-label configs (no masks / missing
            # sentinels) reject null, bool, NaN, and non-{0,1} here — including
            # under trusted_input.
            task_labels: list[int | None] = []
            task_masks: list[int] = []
            for candidate_index, value in enumerate(values):
                if not complete_labels and _is_missing_label(
                    value, label_missing_values[task]
                ):
                    task_labels.append(None)
                    task_masks.append(0)
                    continue
                valid_binary = (
                    not isinstance(value, bool)
                    and isinstance(value, Real)
                    and math.isfinite(float(value))
                    and float(value) in {0.0, 1.0}
                )
                if not valid_binary:
                    if complete_labels:
                        raise ValueError(
                            f"label {column!r} must be numeric 0/1 at raw row "
                            f"{raw_row}, candidate {candidate_index}; got {value!r}"
                        )
                    raise ValueError(
                        f"label {column!r} must be numeric 0/1 or an explicitly configured "
                        f"missing sentinel at raw row {raw_row}, candidate {candidate_index}; "
                        f"got {value!r}"
                    )
                task_labels.append(int(value))
                task_masks.append(1)
            normalized_labels[column] = task_labels
            if mask_column is not None:
                normalized_label_masks[mask_column] = task_masks

        if axis_separated:
            for candidate_index, request_index in enumerate(candidate_requests):
                request_id = request_cache[request_index][axis_request_id_column]
                if request_id is None:
                    raise ValueError(
                        f"request_id column {axis_request_id_column!r} contains null "
                        f"at raw row {raw_row}, candidate {candidate_index}"
                    )
                try:
                    slot = axis_request_id_to_slot.get(request_id)
                except TypeError as error:
                    raise ValueError(
                        f"request_id column {axis_request_id_column!r} must contain "
                        "hashable scalars"
                    ) from error
                if slot is None:
                    slot = len(axis_request_ids)
                    axis_request_id_to_slot[request_id] = slot
                    axis_request_ids.append(request_id)
                    axis_request_raw_rows.append(raw_row)
                    for column in request_output_columns:
                        axis_request_features[column].append(
                            request_cache[request_index][column]
                        )
                    for column in sequence_output_columns:
                        axis_sequence_features[column].append(
                            sequence_cache[request_index][column]
                        )
                axis_candidate_to_request.append(slot)
                axis_candidate_raw_rows.append(raw_row)
            for column in item_output_columns:
                axis_item_features[column].extend(normalized_items[column])
            for column in candidate_metadata_output_columns:
                axis_candidate_metadata[column].extend(candidate_metadata[column])
            for column in label_output_columns:
                axis_label_features[column].extend(normalized_labels[column])
            for column in label_mask_output_columns:
                axis_label_mask_features[column].extend(normalized_label_masks[column])
        else:
            for column in request_output_columns:
                output[column].extend(
                    request_cache[request_index][column]
                    for request_index in candidate_requests
                )
            for column in item_output_columns:
                output[column].extend(normalized_items[column])
            for column in candidate_metadata_output_columns:
                output[column].extend(candidate_metadata[column])
            for column in sequence_output_columns:
                output[column].extend(
                    sequence_cache[request_index][column]
                    for request_index in candidate_requests
                )
            for column in label_output_columns:
                output[column].extend(normalized_labels[column])
            for column in label_mask_output_columns:
                output[column].extend(normalized_label_masks[column])

    if axis_separated:
        if isinstance(runtime_cache, dict) and table.num_rows > 0:
            runtime_cache["mdl_rankmixer_first_batch_adapted"] = True
        sequence_features = share_compact_list_offsets(
            {
                name: compact_list_column_from_rows(values)
                for name, values in axis_sequence_features.items()
            }
        )
        return AdaptedAxisBundle(
            n_candidates=len(axis_candidate_to_request),
            n_requests=len(axis_request_ids),
            request_ids=tuple(axis_request_ids),
            candidate_to_request=np.asarray(axis_candidate_to_request, dtype=np.int64),
            request_features={
                name: axis_feature_column_from_values(values)
                for name, values in axis_request_features.items()
            },
            sequence_features=sequence_features,
            item_features={
                name: axis_feature_column_from_values(values)
                for name, values in axis_item_features.items()
            },
            label_features={
                name: axis_feature_column_from_values(values)
                for name, values in axis_label_features.items()
            },
            label_mask_features={
                name: axis_feature_column_from_values(values)
                for name, values in axis_label_mask_features.items()
            },
            candidate_metadata={
                name: axis_feature_column_from_values(values)
                for name, values in axis_candidate_metadata.items()
            },
            request_raw_rows=np.asarray(axis_request_raw_rows, dtype=np.int64),
            candidate_raw_rows=np.asarray(axis_candidate_raw_rows, dtype=np.int64),
        )

    arrays: dict[str, Any] = {}
    for column, values in output.items():
        if column in candidate_metadata_types:
            arrays[column] = pa.array(values, type=candidate_metadata_types[column])
            continue
        arrays[column] = _output_array(
            pa,
            column,
            values,
            scalar_features=scalar_features,
            bag_features=bag_features,
            sequence_columns=sequence_columns,
            time_delta_columns=time_delta_columns,
            label_columns=label_columns,
            integer_request_columns=plan.integer_output_columns,
            dictionary_encode=(
                compact_request_lists and column in compact_list_columns
            ),
        )
    result = pa.table(arrays)
    if isinstance(runtime_cache, dict) and table.num_rows > 0:
        runtime_cache["mdl_rankmixer_first_batch_adapted"] = True
    return result


def _split_for_name(config: AppConfig, split_name: str) -> ParquetSplitConfig:
    split = config.data.train if split_name == "train" else config.data.test
    if split is None:
        raise ValueError(f"split {split_name!r} is not configured")
    return split


def _load_parquet_adapter(split: ParquetSplitConfig) -> tuple[str, Callable[..., Any]]:
    if split.format == "flat_parquet":
        return "identity", lambda table, *, context: table
    if split.format != "adapter_parquet":
        raise ValueError(f"unsupported parquet split format {split.format!r}")
    if split.adapter is None:
        raise ValueError("adapter_parquet split requires adapter config")
    dotted_path = split.adapter.callable
    module_name, attribute_name = dotted_path.split(":", 1)
    module = importlib.import_module(module_name)
    target: Any = module
    for part in attribute_name.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(f"parquet adapter {dotted_path!r} is not callable")
    return dotted_path, target


def _adapter_context(
    split_name: str,
    split: ParquetSplitConfig,
    required_columns: list[str],
) -> ParquetAdapterContext:
    options: Mapping[str, Any] = {}
    if split.adapter is not None:
        options = dict(split.adapter.options)
    return ParquetAdapterContext(
        split_name=split_name,
        required_columns=tuple(required_columns),
        options=options,
        trusted_input=split.reader.trusted_input,
    )


def run_feature_cardinality_audit(
    config: AppConfig,
    split_name: str,
    *,
    shard_rank: int = 0,
    shard_world_size: int = 1,
    process_group: Any | None = None,
) -> FeatureCardinalityAuditor | None:
    """Soft-sample raw rows on the normal scan path and report all scalar multis.

    Under ``trusted_input``, defaults to 256 raw rows per rank unless
    ``reader.cardinality_audit_raw_rows`` overrides it. Length > 1 on a declared
    scalar is recorded without aborting mid-sample; ranks merge via
    ``all_gather_object`` and then fail once with the full report. YAML is not
    auto-rewritten from list lengths.
    """

    split = _split_for_name(config, split_name)
    audit_rows = split.reader.effective_cardinality_audit_raw_rows()
    if audit_rows <= 0:
        return None
    if split.format != "adapter_parquet":
        return None

    world_size = 1
    if process_group is not None:
        world_size = int(torch.distributed.get_world_size(process_group))
    elif shard_world_size > 1:
        world_size = int(shard_world_size)

    bag_features = frozenset()
    local_payload: dict[str, Any] = {
        "raw_rows_seen": 0,
        "scalar_stats": {},
        "bag_stats": {},
    }
    local_error: str | None = None
    try:
        required_columns = required_columns_for_split(config, split)
        scan_columns = _scan_columns_for_split(split, required_columns)
        scanner = ParquetScanner(
            split,
            scan_columns,
            shard_rank=shard_rank,
            shard_world_size=shard_world_size,
            optional_columns=(
                set(_optional_scan_columns_for_split(split)) & set(scan_columns)
            ),
        )
        adapter_name, adapter = _load_parquet_adapter(split)
        context = _adapter_context(split_name, split, required_columns)
        bag_features = frozenset(
            str(name) for name in context.options.get("multivalue_features", ())
        )
        auditor = FeatureCardinalityAuditor(bag_features=bag_features, soft=True)
        context._runtime_cache["cardinality_auditor"] = auditor
        # Soft audit already covers every field; skip trusted one-row hard warm-up.
        context._runtime_cache["mdl_rankmixer_raw_sample_validated"] = True

        remaining = audit_rows
        try:
            for raw_table in scanner.iter_tables():
                if remaining <= 0:
                    break
                take = min(remaining, int(raw_table.num_rows))
                table = (
                    raw_table
                    if take == raw_table.num_rows
                    else raw_table.slice(0, take)
                )
                auditor.note_raw_rows(table.num_rows)
                result = adapter(table, context=context)
                for _flat in _normalize_adapter_result(
                    result, adapter_name, split_name
                ):
                    del _flat
                remaining -= take
        finally:
            context._runtime_cache.pop("cardinality_auditor", None)
        local_payload = auditor.to_payload()
    except Exception as error:
        local_error = (
            f"feature cardinality audit failed for split {split_name!r}: {error}"
        )

    if world_size > 1:
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            raise RuntimeError(
                "feature cardinality audit requires an initialized process group "
                f"when world_size={world_size}"
            )
        gathered: list[Any] = [None] * world_size
        torch.distributed.all_gather_object(
            gathered,
            {"payload": local_payload, "error": local_error},
            group=process_group,
        )
        peer_errors = [
            item.get("error")
            for item in gathered
            if isinstance(item, Mapping) and item.get("error")
        ]
        if peer_errors:
            raise RuntimeError(str(peer_errors[0]))
        auditor = FeatureCardinalityAuditor(bag_features=bag_features, soft=False)
        for item in gathered:
            if not isinstance(item, Mapping) or not isinstance(
                item.get("payload"), Mapping
            ):
                raise RuntimeError(
                    "feature cardinality audit gathered an invalid peer payload"
                )
            auditor.merge_payload(item["payload"])
    else:
        if local_error is not None:
            raise RuntimeError(local_error)
        auditor = FeatureCardinalityAuditor.from_payload(local_payload)
        auditor.bag_features = bag_features
        auditor.soft = False

    report = auditor.format_report()
    logger.info("Feature cardinality audit for split %s:\n%s", split_name, report)
    if auditor.has_scalar_multis():
        raise ValueError(
            "Feature cardinality audit found declared scalar fields with length > 1. "
            "Keep runtime scalar checks hard; fix field roles in YAML from this report "
            "(do not auto-switch to mean pooling).\n\n"
            f"{report}"
        )
    return auditor


def _normalize_adapter_result(
    result: Any, adapter_name: str, split_name: str
) -> Iterator[Any]:
    pa, _pc, _ds, _pq = _require_pyarrow()

    if isinstance(result, (AdaptedAxisBundle, ArrowAxisSource)):
        yield result
        return
    if isinstance(result, pa.Table):
        yield result
        return
    if isinstance(result, RuntimeIterable):
        for index, table in enumerate(result):
            if not isinstance(table, (pa.Table, AdaptedAxisBundle, ArrowAxisSource)):
                raise TypeError(
                    f"parquet adapter {adapter_name!r} for split {split_name!r} returned "
                    f"item {index} of type {type(table).__name__}; expected pyarrow.Table"
                )
            yield table
        return
    raise TypeError(
        f"parquet adapter {adapter_name!r} for split {split_name!r} returned "
        f"{type(result).__name__}; expected pyarrow.Table or iterable of pyarrow.Table"
    )


def _is_arrow_list_type(pa: Any, arrow_type: Any) -> bool:
    while pa.types.is_dictionary(arrow_type):
        arrow_type = arrow_type.value_type
    return (
        pa.types.is_list(arrow_type)
        or pa.types.is_large_list(arrow_type)
        or (
            hasattr(pa.types, "is_fixed_size_list")
            and pa.types.is_fixed_size_list(arrow_type)
        )
    )


def _table_list_columns(table: Any) -> set[str]:
    pa, _pc, _ds, _pq = _require_pyarrow()
    return {field.name for field in table.schema if _is_arrow_list_type(pa, field.type)}


def _validate_sequence_contract(config: AppConfig, table: Any, split_name: str) -> None:
    pa, pc, _ds, _pq = _require_pyarrow()
    for sequence in config.sequences:
        reference_lengths: Any | None = None
        reference_field: str | None = None
        for field in sequence.fields:
            arrow_type = table.schema.field(field.source).type
            if not _is_arrow_list_type(pa, arrow_type):
                raise ValueError(
                    f"adapter output for split {split_name!r} column {field.source!r} "
                    f"must be a list column because it backs sequence {sequence.name!r}."
                )
            array = _column_array(table, field.source)
            if pa.types.is_dictionary(array.type):
                dictionary_lengths = pc.list_value_length(array.dictionary)
                lengths = pc.take(dictionary_lengths, array.indices)
            else:
                lengths = pc.list_value_length(array)
            if lengths.null_count:
                lengths = pc.fill_null(lengths, 0)
            if reference_lengths is None:
                reference_lengths = lengths
                reference_field = field.name
                continue
            mismatch = pc.not_equal(reference_lengths, lengths)
            if pc.any(mismatch).as_py():
                row_index = int(pc.index(mismatch, True).as_py())
                raise ValueError(
                    f"adapter output for split {split_name!r} sequence {sequence.name!r} "
                    f"has misaligned row {row_index}: field {field.name!r} length "
                    f"{lengths[row_index].as_py()} != field {reference_field!r} length "
                    f"{reference_lengths[row_index].as_py()}."
                )


def _validate_flat_table_static_contract(
    config: AppConfig,
    split: ParquetSplitConfig,
    split_name: str,
    table: Any,
    required_columns: list[str],
) -> None:
    """Validate required columns, list typing, and sequence field alignment once."""

    del split  # Reserved for callers that already selected the active split.
    missing = sorted(set(required_columns) - set(table.column_names))
    if missing:
        raise ValueError(
            f"adapter output for split {split_name!r} is missing flat_parquet column(s): "
            + ", ".join(missing)
        )

    sequence_columns = _sequence_source_columns(config)
    dense_vector_columns = {
        feature.source
        for feature in config.features
        if feature.kind == "dense" and feature.dimension > 1
    }
    categorical_bag_columns = {
        feature.source
        for feature in config.features
        if feature.kind == "categorical" and feature.pooling == "mean"
    }
    scenario_columns = {config.scenarios.source} if config.scenarios.source else set()
    allowed_list_columns = (
        sequence_columns
        | dense_vector_columns
        | categorical_bag_columns
        | scenario_columns
    )
    unexpected_list_columns = sorted(
        column
        for column in _table_list_columns(table)
        if column in required_columns and column not in allowed_list_columns
    )
    if unexpected_list_columns:
        raise ValueError(
            f"adapter output for split {split_name!r} has list-valued non-sequence column(s): "
            + ", ".join(unexpected_list_columns)
            + ". Only configured sequence fields, categorical features with pooling=mean, "
            "dense features with dimension > 1, and scenario masks may use list-valued cells."
        )

    _validate_sequence_contract(config, table, split_name)


def _validate_complete_label_contract(
    split: ParquetSplitConfig,
    table: Any,
    required_columns: list[str],
) -> None:
    """Reject null / non-binary labels on every flat batch for complete-label paths."""

    if split.label_masks:
        return
    required_set = set(required_columns)
    pa, pc, _ds, _pq = _require_pyarrow()
    for task, column in split.labels.items():
        if column not in required_set:
            continue
        array = _column_array(table, column)
        if not (pa.types.is_integer(array.type) or pa.types.is_floating(array.type)):
            raise ValueError(
                f"adapter output label {column!r} for task {task!r} must be numeric 0/1"
            )
        if array.null_count:
            raise ValueError(
                f"adapter output label {column!r} for task {task!r} contains null"
            )
        if len(array):
            binary = pc.or_(pc.equal(array, 0), pc.equal(array, 1))
            if not bool(pc.all(binary).as_py()):
                raise ValueError(
                    f"adapter output label {column!r} for task {task!r} must contain only 0/1"
                )


def _validate_flat_table_contract(
    config: AppConfig,
    split: ParquetSplitConfig,
    split_name: str,
    table: Any,
    required_columns: list[str],
) -> None:
    """Backward-compatible wrapper used by tests and non-iterating callers."""

    _validate_flat_table_static_contract(
        config, split, split_name, table, required_columns
    )
    _validate_complete_label_contract(split, table, required_columns)


def _initialize_adapter_process(
    adapter_name: str,
    split_name: str,
    required_columns: tuple[str, ...],
    options: dict[str, Any],
    trusted_input: bool,
    runtime_cache_options: dict[str, Any] | None,
) -> None:
    """Initialize one isolated adapter worker without touching CUDA state."""

    global _PROCESS_ADAPTER
    global _PROCESS_ADAPTER_CONTEXT
    global _PROCESS_ADAPTER_NAME
    global _PROCESS_ADAPTER_SPLIT_NAME

    # Prefer parent prepare/tensorize when cores are contended: adapter workers
    # are throughput-oriented background producers.
    try:
        os.nice(10)
    except OSError:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    # Pin each adapter worker away from the parent's first cores so
    # prepare/tensorize keep warm cache/bandwidth for the critical path.
    try:
        import psutil

        n_cpu = psutil.cpu_count(logical=True) or 4
        core = 4 + (os.getpid() % max(1, n_cpu - 4))
        psutil.Process().cpu_affinity([core])
    except Exception:
        pass

    module_name, attribute_name = adapter_name.split(":", 1)
    module = importlib.import_module(module_name)
    target: Any = module
    for part in attribute_name.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(f"parquet adapter {adapter_name!r} is not callable")
    _PROCESS_ADAPTER = target
    _PROCESS_ADAPTER_CONTEXT = ParquetAdapterContext(
        split_name=split_name,
        required_columns=required_columns,
        options=options,
        trusted_input=trusted_input,
    )
    if runtime_cache_options:
        _PROCESS_ADAPTER_CONTEXT._runtime_cache.update(runtime_cache_options)
    _PROCESS_ADAPTER_NAME = adapter_name
    _PROCESS_ADAPTER_SPLIT_NAME = split_name


def _warmup_adapter_process() -> None:
    """No-op task so ProcessPool workers finish initializer before real tables."""

    if _PROCESS_ADAPTER is None:
        raise RuntimeError("Parquet adapter process was not initialized")


def _adapt_table_in_process(raw_table: Any) -> list[Any]:
    """Apply the process-local adapter and return materialized Arrow tables."""

    if _PROCESS_ADAPTER is None or _PROCESS_ADAPTER_CONTEXT is None:
        raise RuntimeError("Parquet adapter process was not initialized")
    result = _PROCESS_ADAPTER(raw_table, context=_PROCESS_ADAPTER_CONTEXT)
    return list(
        _normalize_adapter_result(
            result,
            _PROCESS_ADAPTER_NAME,
            _PROCESS_ADAPTER_SPLIT_NAME,
        )
    )


def _iter_process_adapter_results(
    raw_tables: Iterable[Any],
    *,
    adapter_name: str,
    context: ParquetAdapterContext,
    worker_count: int,
    max_pending: int,
    runtime_cache_options: Mapping[str, Any] | None = None,
) -> Iterator[tuple[int, list[Any]]]:
    """Adapt raw Arrow tables concurrently in deterministic input order."""

    pending: deque[tuple[int, Any]] = deque()
    source = iter(raw_tables)
    exhausted = False
    previous_mkl_force = os.environ.get("MKL_SERVICE_FORCE_INTEL")
    # Conda's mkl-service can otherwise abort a clean forkserver/spawn child
    # when the launcher already imported a libgomp-using extension.
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
    try:
        import psutil

        # Prefer parent prepare/tensorize when cores are contended: adapter
        # workers are throughput-oriented background producers. A dedicated
        # host-prepare process keeps a wider affinity so pack+tensorize is not
        # stuck on 4 cores while training occupies the rest of the machine.
        n_cpu = int(psutil.cpu_count(logical=True) or 4)
        if os.environ.get("MDL_HOST_PREPARE_PROCESS") == "1":
            prepare_cores = min(max(16, n_cpu // 3), n_cpu)
            psutil.Process().cpu_affinity(list(range(prepare_cores)))
        else:
            psutil.Process().cpu_affinity([0, 1, 2, 3])
    except Exception:
        pass
    executor: ProcessPoolExecutor | None = None
    try:
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            # Forkserver children inherit neither the CUDA-initialized training
            # process nor its threads. Fall back to spawn off Linux.
            mp_context=multiprocessing.get_context(
                "forkserver"
                if "forkserver" in multiprocessing.get_all_start_methods()
                else "spawn"
            ),
            initializer=_initialize_adapter_process,
            initargs=(
                adapter_name,
                context.split_name,
                context.required_columns,
                dict(context.options),
                context.trusted_input,
                (
                    None
                    if runtime_cache_options is None
                    else dict(runtime_cache_options)
                ),
            ),
        )
        # Force all workers through initializer before the first large Arrow
        # submit. Otherwise the parent blocks inside submit()/forkserver write
        # while each child imports the adapter — multi-second stalls that look
        # like a deadlock and leave the GPU idle.
        warmups = [
            executor.submit(_warmup_adapter_process) for _ in range(worker_count)
        ]
        for future in warmups:
            future.result()
        while pending or not exhausted:
            while len(pending) < max_pending and not exhausted:
                try:
                    raw_table = next(source)
                except StopIteration:
                    exhausted = True
                    break
                pending.append(
                    (
                        int(raw_table.num_rows),
                        executor.submit(_adapt_table_in_process, raw_table),
                    )
                )
            if not pending:
                break
            raw_rows, future = pending.popleft()
            yield raw_rows, future.result()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if previous_mkl_force is None:
            os.environ.pop("MKL_SERVICE_FORCE_INTEL", None)
        else:
            os.environ["MKL_SERVICE_FORCE_INTEL"] = previous_mkl_force
        close = getattr(source, "close", None)
        if callable(close):
            close()


def _iter_adapted_flat_tables(
    config: AppConfig,
    split_name: str,
    scanner: ParquetScanner,
    adapter_name: str,
    adapter: Callable[..., Any],
    context: ParquetAdapterContext,
    required_columns: list[str],
    counters: _FlatScanCounters | None = None,
    max_batches: int | None = None,
) -> Iterator[Any]:
    validated_static_contract = False
    # Complete-label paths omit masks; every flat batch must still prove 0/1/no-null.
    complete_label_contract = not bool(scanner.split.label_masks)

    def limited_raw_tables() -> Iterator[Any]:
        for raw_batch_index, raw_table in enumerate(scanner.iter_tables()):
            if max_batches is not None and raw_batch_index >= max_batches:
                break
            yield raw_table

    adapter_workers = scanner.split.reader.adapter_workers
    if adapter_workers > 0 and adapter_name != "identity":
        adapted_results: Iterable[
            tuple[int, list[Any]]
        ] = _iter_process_adapter_results(
            limited_raw_tables(),
            adapter_name=adapter_name,
            context=context,
            worker_count=adapter_workers,
            max_pending=max(
                adapter_workers,
                min(
                    max(adapter_workers, scanner.split.reader.prefetch_batches),
                    adapter_workers * 2,
                ),
            ),
        )
    else:

        def sequential_results() -> Iterator[tuple[int, list[Any]]]:
            for raw_table in limited_raw_tables():
                result = adapter(raw_table, context=context)
                yield int(raw_table.num_rows), list(
                    _normalize_adapter_result(result, adapter_name, split_name)
                )

        adapted_results = sequential_results()

    adapted_iterator = iter(adapted_results)
    try:
        while True:
            try:
                raw_rows, flat_tables = next(adapted_iterator)
            except StopIteration:
                break
            except Exception as error:
                if adapter_name == "identity":
                    raise
                raise RuntimeError(
                    f"parquet adapter {adapter_name!r} failed for split {split_name!r}: {error}"
                ) from error
            if counters is not None:
                counters.raw_record_batches += 1
                counters.raw_rows += raw_rows
            try:
                for flat_table in flat_tables:
                    if not validated_static_contract:
                        _validate_flat_table_static_contract(
                            config,
                            scanner.split,
                            split_name,
                            flat_table,
                            required_columns,
                        )
                        validated_static_contract = flat_table.num_rows > 0
                    if complete_label_contract:
                        _validate_complete_label_contract(
                            scanner.split,
                            flat_table,
                            required_columns,
                        )
                    if counters is not None:
                        counters.flat_tables += 1
                        counters.flat_rows += flat_table.num_rows
                    yield flat_table
            except Exception as error:
                if adapter_name == "identity":
                    raise
                raise RuntimeError(
                    f"parquet adapter {adapter_name!r} failed for split {split_name!r}: {error}"
                ) from error
    finally:
        close = getattr(adapted_iterator, "close", None)
        if callable(close):
            close()


def iter_flat_tables(
    config: AppConfig,
    split_name: str,
    *,
    shard_rank: int = 0,
    shard_world_size: int = 1,
    extra_columns: Iterable[str] = (),
    require_labels: bool = True,
) -> Iterator[Any]:
    """Yield flat Arrow tables for any configured Parquet split.

    This is the single model-facing table entry point. ``flat_parquet`` uses an
    identity adapter; ``adapter_parquet`` applies the configured external
    adapter before validating the flat contract.
    """
    split = _split_for_name(config, split_name)
    required_columns = required_columns_for_split(
        config,
        split,
        extra_columns=extra_columns,
        require_labels=require_labels,
    )
    scan_columns = _scan_columns_for_split(split, required_columns)
    scanner = ParquetScanner(
        split,
        scan_columns,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
        optional_columns=(
            set(_optional_scan_columns_for_split(split)) & set(scan_columns)
        ),
    )
    adapter_name, adapter = _load_parquet_adapter(split)
    context = _adapter_context(split_name, split, required_columns)
    yield from _iter_adapted_flat_tables(
        config,
        split_name,
        scanner,
        adapter_name,
        adapter,
        context,
        required_columns,
    )


def iter_adapted_axis_bundles(
    config: AppConfig,
    split_name: str,
    *,
    shard_rank: int = 0,
    shard_world_size: int = 1,
    require_labels: bool = True,
    producer_queue_size: int = 2,
    arrow_axis: bool = False,
) -> Iterator[Any]:
    """Yield axis-separated adapted payloads (no candidate-flat Arrow).

    When ``arrow_axis`` is true, yields ``ArrowAxisSource`` (control-plan +
    raw Arrow table) instead of Python ``AdaptedAxisBundle``.

    ``producer_queue_size`` bounds a shallow in-process queue so adaptation of
    the next raw batch overlaps descriptor/batch work on the consumer side.
    Size ``<= 1`` disables prefetch (purely synchronous).
    """

    split = _split_for_name(config, split_name)
    if split.request_id is None:
        raise ValueError("iter_adapted_axis_bundles requires split.request_id")
    if split.format != "adapter_parquet":
        raise ValueError(
            "iter_adapted_axis_bundles requires adapter_parquet "
            f"(got {split.format!r})"
        )
    required_columns = required_columns_for_split(
        config,
        split,
        require_labels=require_labels,
    )
    scan_columns = _scan_columns_for_split(split, required_columns)
    scanner = ParquetScanner(
        split,
        scan_columns,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
        optional_columns=(
            set(_optional_scan_columns_for_split(split)) & set(scan_columns)
        ),
    )
    adapter_name, adapter = _load_parquet_adapter(split)
    context = _adapter_context(split_name, split, required_columns)
    if arrow_axis:
        context._runtime_cache["arrow_axis"] = True
    else:
        context._runtime_cache["axis_separated"] = True
    context._runtime_cache["axis_request_id_column"] = split.request_id
    expected_type = ArrowAxisSource if arrow_axis else AdaptedAxisBundle
    runtime_cache_options = {
        "axis_request_id_column": split.request_id,
        **({"arrow_axis": True} if arrow_axis else {"axis_separated": True}),
    }

    def produce() -> Iterator[Any]:
        adapter_workers = split.reader.adapter_workers
        if adapter_workers > 0 and adapter_name != "identity":
            adapted_results: Iterable[
                tuple[int, list[Any]]
            ] = _iter_process_adapter_results(
                scanner.iter_tables(),
                adapter_name=adapter_name,
                context=context,
                worker_count=adapter_workers,
                max_pending=max(
                    adapter_workers * 2,
                    min(
                        max(adapter_workers * 2, split.reader.prefetch_batches),
                        adapter_workers * 4,
                    ),
                ),
                runtime_cache_options=runtime_cache_options,
            )
        else:

            def sequential_results() -> Iterator[tuple[int, list[Any]]]:
                for raw_table in scanner.iter_tables():
                    if not raw_table.num_rows:
                        continue
                    result = adapter(raw_table, context=context)
                    yield int(raw_table.num_rows), list(
                        _normalize_adapter_result(
                            result,
                            adapter_name,
                            split_name,
                        )
                    )

            adapted_results = sequential_results()

        adapted_iterator = iter(adapted_results)
        try:
            for _raw_rows, outputs in adapted_iterator:
                for bundle in outputs:
                    if not isinstance(bundle, expected_type):
                        raise TypeError(
                            "axis adapter must return "
                            f"{expected_type.__name__}, got {type(bundle).__name__}"
                        )
                    if bundle.n_candidates == 0:
                        continue
                    yield bundle
        except Exception as error:
            if adapter_name == "identity":
                raise
            raise RuntimeError(
                f"parquet adapter {adapter_name!r} failed for split "
                f"{split_name!r}: {error}"
            ) from error
        finally:
            close = getattr(adapted_iterator, "close", None)
            if callable(close):
                close()

    # ProcessPoolExecutor must be created/driven from this caller's thread.
    # Spawning a second producer thread around forkserver workers races with
    # scanner prefetch and contends for CPU/GIL during host prepare (measured
    # e2e regression: gap 194→294ms, sps 871→657). Overlap for
    # adapter_workers>0 comes from the process runway (max_pending) instead.
    if producer_queue_size <= 1 or split.reader.adapter_workers > 0:
        yield from produce()
        return

    import queue as queue_mod
    import threading

    out_queue: queue_mod.Queue[Any] = queue_mod.Queue(maxsize=int(producer_queue_size))
    error_holder: list[BaseException] = []

    def worker() -> None:
        try:
            for bundle in produce():
                out_queue.put(bundle)
            out_queue.put(None)
        except BaseException as error:  # noqa: BLE001 - propagate to consumer
            error_holder.append(error)
            out_queue.put(error)

    thread = threading.Thread(
        target=worker,
        name=f"mdl-axis-producer-{split_name}",
        daemon=True,
    )
    thread.start()
    while True:
        item = out_queue.get()
        if item is None:
            break
        if isinstance(item, BaseException):
            raise item
        yield item
    thread.join()
    if error_holder:
        raise error_holder[0]


def scan_flat_table_stats(
    config: AppConfig,
    split_name: str,
    *,
    max_batches: int | None = None,
) -> FlatScanStats:
    """Scan through the unified flat-table path and return raw/flat counters."""
    split = _split_for_name(config, split_name)
    required_columns = required_columns_for_split(config, split)
    scanner = ParquetScanner(
        split,
        _scan_columns_for_split(split, required_columns),
        optional_columns=_optional_scan_columns_for_split(split),
    )
    counters = _FlatScanCounters(files=len(scanner.paths))
    adapter_name, adapter = _load_parquet_adapter(split)
    context = _adapter_context(split_name, split, required_columns)
    for _table in _iter_adapted_flat_tables(
        config,
        split_name,
        scanner,
        adapter_name,
        adapter,
        context,
        required_columns,
        counters=counters,
        max_batches=max_batches,
    ):
        pass
    return counters.snapshot()


# ---------------------------------------------------------------------------
# Batch assembly: Arrow table -> FeatureBatch
# ---------------------------------------------------------------------------


@dataclass
class FeatureBatch:
    """One model-ready batch plus metadata needed by loss and evaluation.

    ``features`` may contain nested dictionaries for multi-field sequences.
    Tensor leaves stay on CPU until ``move_feature_batch`` is called. Group IDs
    remain Python strings because they are evaluation metadata, not model input.
    """

    features: dict[str, Any]
    labels: Tensor | None
    label_mask: Tensor | None
    scenario_id: Tensor
    group_id: list[str]
    prediction_keys: dict[str, list[Any]] = field(default_factory=dict)
    # Optional same-dtype base buffers. Tensor leaves are views into these
    # buffers so one H2D copy per dtype replaces hundreds of small copies.
    _packed_buffers: tuple[Tensor, ...] = field(default_factory=tuple, repr=False)
    # Opaque owner for recycled pinned-host leases (see train._PinnedHostBufferPool).
    # Kept alive for the FeatureBatch lifetime so pooled storages are not reused
    # while views still exist; ignored by equality / repr.
    _keepalive: Any = field(default=None, repr=False, compare=False)


# --- Column accessors ---


def _column_values(table: Any, column: str) -> list[Any]:
    """Read an Arrow column as Python values for encoding or nested handling."""
    if column not in table.column_names:
        raise ValueError(f"missing required batch column {column!r}")
    return table[column].to_pylist()


def _column_array(table: Any, column: str) -> Any:
    """Return a contiguous Arrow array while preserving a useful missing-column error."""
    if column not in table.column_names:
        raise ValueError(f"missing required batch column {column!r}")
    pa, pc, _ds, _pq = _require_pyarrow()
    chunked = table[column]
    if not chunked.num_chunks:
        return chunked.combine_chunks()
    if chunked.num_chunks == 1:
        return chunked.chunk(0)
    if not all(pa.types.is_dictionary(chunk.type) for chunk in chunked.chunks):
        return chunked.combine_chunks()
    dictionaries: list[Any] = []
    shifted_indices: list[Any] = []
    offset = 0
    for chunk in chunked.chunks:
        dictionaries.append(chunk.dictionary)
        indices = chunk.indices
        if offset:
            indices = pc.add(indices, pa.scalar(offset, type=indices.type))
        shifted_indices.append(indices)
        offset += len(chunk.dictionary)
    return pa.DictionaryArray.from_arrays(
        pa.concat_arrays(shifted_indices),
        pa.concat_arrays(dictionaries),
    )


def _safe_table_take(table: Any, indices: Any) -> Any:
    """Row-select without Arrow nested-dictionary unification.

    ``Table.take`` fails on multi-chunk ``dictionary<list<...>>`` columns
    because Arrow cannot unify those dictionaries. Rebuild each column from
    ``_column_array`` first, then ``pc.take`` on the contiguous array.
    """

    pa, pc, _ds, _pq = _require_pyarrow()
    if hasattr(indices, "numpy") and not isinstance(indices, pa.Array):
        index_array = pa.array(indices.numpy(), type=pa.int64())
    elif isinstance(indices, pa.Array):
        index_array = (
            indices.cast(pa.int64()) if indices.type != pa.int64() else indices
        )
    else:
        index_array = pa.array(indices, type=pa.int64())
    if len(index_array) == 0:
        return table.slice(0, 0)
    arrays = [
        pc.take(_column_array(table, column), index_array)
        for column in table.column_names
    ]
    return pa.Table.from_arrays(arrays, schema=table.schema)


def _numeric_column_tensor(table: Any, column: str, dtype: torch.dtype) -> Tensor:
    """Convert a scalar numeric Arrow column with a fast NumPy path.

    Nulls map to zero consistently with categorical OOV/padding semantics. Some
    Arrow types cannot expose a NumPy representation, so the explicit Python
    conversion remains as a correctness fallback.
    """
    array = _column_array(table, column)
    try:
        import pyarrow.compute as pc

        if array.null_count:
            fill_value = 0 if dtype in {torch.long, torch.int64, torch.int32} else 0.0
            array = pc.fill_null(array, fill_value)
        values = array.to_numpy(zero_copy_only=False)
        if hasattr(values, "flags") and not values.flags.writeable:
            values = values.copy()
        return torch.as_tensor(values, dtype=dtype)
    except (TypeError, ValueError, NotImplementedError):
        if dtype in {torch.long, torch.int64, torch.int32}:
            return torch.tensor(
                [0 if value is None else int(value) for value in array.to_pylist()],
                dtype=dtype,
            )
        return torch.tensor(
            [0.0 if value is None else float(value) for value in array.to_pylist()],
            dtype=dtype,
        )


def _numpy_backed_tensor(array: Any, dtype: torch.dtype) -> Tensor:
    values = array.to_numpy(zero_copy_only=False)
    if hasattr(values, "flags") and not values.flags.writeable:
        values = values.copy()
    return torch.as_tensor(values, dtype=dtype)


def _identity_array_tensor(
    array: Any,
    categorical_input: ResolvedCategoricalInput,
) -> Tensor:
    """Convert a numeric identity column without Python element processing."""

    encoding = categorical_input.encoding
    if not isinstance(encoding, ResolvedIdentityEncoding):
        raise TypeError("_identity_array_tensor requires identity encoding")
    pa, pc, _ds, _pq = _require_pyarrow()
    if not pa.types.is_integer(array.type):
        raise TypeError(
            f"identity input {categorical_input.name!r} must be an Arrow integer column, "
            f"got {array.type}"
        )
    if array.null_count:
        array = pc.fill_null(array, encoding.padding_id)

    min_max = pc.min_max(array).as_py()
    minimum = min_max.get("min") if min_max is not None else None
    maximum = min_max.get("max") if min_max is not None else None
    invalid_bounds = (minimum is not None and int(minimum) < 0) or (
        maximum is not None and int(maximum) >= encoding.num_buckets
    )
    if invalid_bounds and encoding.out_of_range == "error":
        raise ValueError(
            f"identity input {categorical_input.name!r} contains IDs outside "
            f"[0, {encoding.num_buckets}): min={minimum}, max={maximum}"
        )
    if invalid_bounds:
        valid = pc.and_(
            pc.greater_equal(array, 0),
            pc.less(array, encoding.num_buckets),
        )
        array = pc.if_else(valid, array, encoding.padding_id)
    array = pc.cast(array, target_type=pa.int64(), safe=True)
    return _numpy_backed_tensor(array, torch.long)


def _identity_column_tensor(
    table: Any,
    categorical_input: ResolvedCategoricalInput,
) -> Tensor:
    return _identity_array_tensor(
        _column_array(table, categorical_input.source),
        categorical_input,
    )


def _pre_hashed_array_tensor(
    array: Any,
    categorical_input: ResolvedCategoricalInput,
    *,
    validate_nonzero: bool = True,
) -> Tensor:
    """Vectorize unsigned-low-bit bucketing while preserving null as zero."""

    encoding = categorical_input.encoding
    if not isinstance(encoding, ResolvedPreHashedEncoding):
        raise TypeError("_pre_hashed_array_tensor requires pre_hashed encoding")
    pa, pc, _ds, _pq = _require_pyarrow()
    if not pa.types.is_int64(array.type):
        raise TypeError(
            f"pre_hashed input {categorical_input.name!r} must be an Arrow int64 column, "
            f"got {array.type}"
        )
    if validate_nonzero:
        zero_mask = pc.equal(array, 0)
        has_zero = pc.any(zero_mask).as_py()
        if has_zero:
            raise ValueError(
                f"pre_hashed input {categorical_input.name!r} contains non-null zero values"
            )
    encoded = pc.add(
        pc.bit_wise_and(array, encoding.num_buckets - 1),
        1,
    )
    if encoded.null_count:
        encoded = pc.fill_null(encoded, encoding.padding_id)
    return _numpy_backed_tensor(encoded, torch.long)


def _pre_hashed_column_tensor(
    table: Any,
    categorical_input: ResolvedCategoricalInput,
    *,
    validate_nonzero: bool = True,
) -> Tensor:
    return _pre_hashed_array_tensor(
        _column_array(table, categorical_input.source),
        categorical_input,
        validate_nonzero=validate_nonzero,
    )


# --- Categorical encoding ---


def _effective_categorical_input(
    config: AppConfig,
    categorical_input: ResolvedCategoricalInput,
) -> ResolvedCategoricalInput:
    """Apply a shared namespace's base encoding to the current source column."""

    base_input = resolve_categorical_base_input(
        config.resolved.categorical_input_by_name,
        categorical_input.name,
    )
    if base_input.name == categorical_input.name:
        return categorical_input
    return ResolvedCategoricalInput(
        name=categorical_input.name,
        source=categorical_input.source,
        location=categorical_input.location,
        sequence_name=categorical_input.sequence_name,
        field_name=categorical_input.field_name,
        encoding=base_input.encoding,
    )


def _tensorize_categorical(
    config: AppConfig,
    feature: FeatureConfig,
    table: Any,
    vocab_maps: dict[str, dict[str, int]],
    *,
    validate_prehashed_nonzero: bool = True,
) -> Tensor:
    """Build a rank-one integer tensor for a configured categorical feature."""
    categorical_input = _effective_categorical_input(
        config,
        config.resolved.categorical_input_by_name[feature.name],
    )
    if isinstance(categorical_input.encoding, ResolvedIdentityEncoding):
        return _identity_column_tensor(table, categorical_input)
    if isinstance(categorical_input.encoding, ResolvedPreHashedEncoding):
        return _pre_hashed_column_tensor(
            table,
            categorical_input,
            validate_nonzero=validate_prehashed_nonzero,
        )
    unseen_policy = config.vocab_strategy.defaults.unseen_policy
    encoded = encode_categorical_values(
        _column_values(table, categorical_input.source),
        categorical_input,
        vocab_maps,
        unseen_policy,
    )
    return torch.tensor(encoded, dtype=torch.long)


def _list_truncate_array(array: Any, max_length: int, *, truncation: str) -> Any:
    """Truncate list elements without building a padded ``[N, max_length]`` table.

    Head uses Arrow ``list_slice``. Tail rebuilds offsets (PyArrow 14
    ``list_slice`` rejects negative ``start``).
    """

    if max_length < 0:
        raise ValueError(f"max_length must be non-negative, got {max_length}")
    pa, pc, _ds, _pq = _require_pyarrow()
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if truncation == "head":
        return pc.list_slice(array, 0, int(max_length))
    if truncation != "tail":
        raise ValueError(f"unsupported list truncation {truncation!r}")

    import numpy as np

    offsets = array.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    lengths = offsets[1:] - offsets[:-1]
    take_n = np.minimum(lengths, int(max_length)).astype(np.int64, copy=False)
    starts = offsets[:-1] + (lengths - take_n)
    total = int(take_n.sum())
    if total == 0:
        flat = array.values.slice(0, 0)
    else:
        row_ids = np.repeat(np.arange(len(take_n), dtype=np.int64), take_n)
        within = np.arange(total, dtype=np.int64) - np.repeat(
            np.cumsum(take_n) - take_n, take_n
        )
        flat = pc.take(
            array.values, pa.array(starts[row_ids] + within, type=pa.int64())
        )
    new_offsets = np.empty(len(take_n) + 1, dtype=np.int64)
    new_offsets[0] = 0
    np.cumsum(take_n, out=new_offsets[1:])
    if pa.types.is_large_list(array.type):
        return pa.LargeListArray.from_arrays(new_offsets, flat)
    return pa.ListArray.from_arrays(new_offsets, flat)


def _tensorize_categorical_bag(
    config: AppConfig,
    feature: FeatureConfig,
    table: Any,
    vocab_maps: dict[str, dict[str, int]],
    *,
    validate_prehashed_nonzero: bool = True,
) -> dict[str, Tensor]:
    """Encode one list-valued categorical feature as flat values + lengths.

    Truncation follows ``feature.max_length`` / ``truncation`` via Arrow-native
    list ops + ``list_flatten`` (no temporary ``[N, max_length]`` pad).
    The returned ``values`` tensor is CSR-like ``[sum(lengths)]``; mean-pool
    reconstructs per-row segments from ``lengths``.
    """

    if feature.pooling != "mean":
        raise TypeError("_tensorize_categorical_bag requires pooling=mean")
    categorical_input = _effective_categorical_input(
        config,
        config.resolved.categorical_input_by_name[feature.name],
    )
    pa, pc, _ds, _pq = _require_pyarrow()
    array = _normalized_list_array(table, feature.source)
    if feature.max_length is not None:
        array = _list_truncate_array(
            array,
            int(feature.max_length),
            truncation=feature.truncation,
        )
    length_array = pc.list_value_length(array)
    if length_array.null_count:
        length_array = pc.fill_null(length_array, 0)
    lengths = _numpy_backed_tensor(length_array, torch.long)
    flat_array = pc.list_flatten(array)
    if isinstance(categorical_input.encoding, ResolvedIdentityEncoding):
        encoded = _identity_array_tensor(flat_array, categorical_input)
    elif isinstance(categorical_input.encoding, ResolvedPreHashedEncoding):
        encoded = _pre_hashed_array_tensor(
            flat_array,
            categorical_input,
            validate_nonzero=validate_prehashed_nonzero,
        )
    else:
        unseen_policy = config.vocab_strategy.defaults.unseen_policy
        vocab_map = vocab_maps.get(categorical_input.name)
        encoded = torch.tensor(
            [
                encode_categorical_value(
                    value,
                    categorical_input,
                    vocab_map,
                    unseen_policy,
                )
                for value in flat_array.to_pylist()
            ],
            dtype=torch.long,
        )
    return {"values": encoded, "lengths": lengths}


# --- Dense features ---


def _dense_feature_value(
    value: Any, dimension: int
) -> tuple[float | list[float], float]:
    """Normalize one dense value and its presence bit.

    Returns ``(filled_value, presence)`` where missing/null maps to zeros with
    presence 0, and a real zero keeps presence 1.
    """
    if value is None:
        filled: float | list[float] = 0.0 if dimension == 1 else [0.0] * dimension
        return filled, 0.0
    if dimension == 1:
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"dense feature expected 1 value, got {len(value)}")
            value = value[0]
        if value is None:
            return 0.0, 0.0
        return float(value), 1.0
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"dense feature expected {dimension} values, got scalar {value!r}"
        )
    if len(value) != dimension:
        raise ValueError(f"dense feature expected {dimension} values, got {len(value)}")
    return [0.0 if item is None else float(item) for item in value], 1.0


def _tensorize_dense(
    feature: FeatureConfig, values: list[Any]
) -> Tensor | dict[str, Tensor]:
    """Build a ``[batch, dim]`` float tensor; optionally attach presence."""
    normalized = [_dense_feature_value(value, feature.dimension) for value in values]
    filled = [item[0] for item in normalized]
    tensor = torch.tensor(filled, dtype=torch.float32)
    if feature.kind == "dense" and feature.presence:
        presence = torch.tensor(
            [[item[1]] for item in normalized],
            dtype=torch.float32,
        )
        return {"values": tensor, "presence": presence}
    return tensor


def _numeric_column_with_presence(table: Any, column: str) -> tuple[Tensor, Tensor]:
    """Convert a scalar numeric column to values + presence, null→0 / presence 0."""
    array = _column_array(table, column)
    try:
        import pyarrow.compute as pc

        presence = torch.ones(len(array), 1, dtype=torch.float32)
        if array.null_count:
            null_mask = array.is_null()
            presence = torch.tensor(
                [[0.0 if flag else 1.0] for flag in null_mask.to_pylist()],
                dtype=torch.float32,
            )
            array = pc.fill_null(array, 0.0)
        values = array.to_numpy(zero_copy_only=False)
        if hasattr(values, "flags") and not values.flags.writeable:
            values = values.copy()
        return torch.as_tensor(values, dtype=torch.float32), presence
    except (TypeError, ValueError, NotImplementedError):
        filled: list[float] = []
        presence_rows: list[list[float]] = []
        for value in array.to_pylist():
            if value is None:
                filled.append(0.0)
                presence_rows.append([0.0])
            else:
                filled.append(float(value))
                presence_rows.append([1.0])
        return (
            torch.tensor(filled, dtype=torch.float32),
            torch.tensor(presence_rows, dtype=torch.float32),
        )


def _tensorize_dense_column(
    feature: FeatureConfig, table: Any
) -> Tensor | dict[str, Tensor]:
    # Scalar columns use the Arrow/NumPy fast path; vector columns require
    # row-level shape validation before tensor construction.
    if feature.dimension == 1:
        if feature.presence:
            values, presence = _numeric_column_with_presence(table, feature.source)
            return {"values": values, "presence": presence}
        return _numeric_column_tensor(table, feature.source, torch.float32)
    return _tensorize_dense(feature, _column_values(table, feature.source))


# --- Multi-field sequences ---


def _coerce_sequence_items(row: Any) -> list[Any]:
    """Normalize null, scalar, and tuple sequence cells to Python lists."""
    if row is None:
        return []
    if isinstance(row, list):
        return row
    if isinstance(row, tuple):
        return list(row)
    return [row]


def _sequence_step_is_present(value: Any) -> bool:
    """Return False when an anchor step is missing (null or singleton [null])."""
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return value[0] is not None
        return len(value) > 0
    return True


def _sequence_tensor_max_length(sequence: Any) -> int | None:
    """Resolve the physical cap, including unified-event transport overrides."""

    return getattr(sequence, "tensor_max_length", sequence.max_length)


def _compress_row_fields_by_anchor(
    raw_items_by_field: dict[str, list[Any]],
    *,
    anchor_field: str | None,
) -> dict[str, list[Any]]:
    """Drop steps whose anchor value is null from every aligned field."""
    if anchor_field is None:
        return raw_items_by_field
    anchor_items = raw_items_by_field[anchor_field]
    keep = [
        index
        for index, value in enumerate(anchor_items)
        if _sequence_step_is_present(value)
    ]
    if len(keep) == len(anchor_items):
        return raw_items_by_field
    return {
        name: [items[index] for index in keep]
        for name, items in raw_items_by_field.items()
    }


def _sequence_bounds(length: int, sequence: SequenceConfig) -> tuple[int, int]:
    """Return the configured physical head/tail window before order canonicalization."""
    max_length = _sequence_tensor_max_length(sequence)
    if max_length is None or length <= max_length:
        return 0, length
    if sequence.truncation == "tail":
        return length - max_length, length
    return 0, max_length


def _direct_sequence_supported(config: AppConfig, sequence: SequenceConfig) -> bool:
    return all(
        field.kind != "categorical"
        or isinstance(
            _effective_categorical_input(
                config,
                config.resolved.categorical_input_by_name[
                    field.qualified_name(sequence.name)
                ],
            ).encoding,
            (ResolvedIdentityEncoding, ResolvedPreHashedEncoding),
        )
        for field in sequence.fields
    )


def _normalized_list_array(table: Any, column: str) -> Any:
    pa, pc, _ds, _pq = _require_pyarrow()
    array = _column_array(table, column)
    if pa.types.is_dictionary(array.type):
        # Arrow 14 cannot dictionary_decode list-valued dictionaries, while
        # take(dictionary, indices) has the required list kernel.
        array = pc.take(array.dictionary, array.indices)
    if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type)):
        raise TypeError(
            f"direct sequence input {column!r} must be an Arrow list column, got {array.type}"
        )
    if array.null_count:
        array = pc.fill_null(array, pa.scalar([], type=array.type))
    return array


def _list_offsets_tensor(array: Any) -> Tensor:
    return _numpy_backed_tensor(array.offsets, torch.long)


def _direct_dense_values(array: Any, dimension: int, field_name: str) -> Tensor:
    pa, pc, _ds, _pq = _require_pyarrow()
    if dimension == 1:
        if not (pa.types.is_integer(array.type) or pa.types.is_floating(array.type)):
            raise TypeError(
                f"dense sequence field {field_name!r} must contain numeric values, got {array.type}"
            )
        if array.null_count:
            array = pc.fill_null(array, 0.0)
        return _numpy_backed_tensor(
            pc.cast(array, target_type=pa.float32(), safe=False),
            torch.float32,
        )

    if not (
        pa.types.is_list(array.type)
        or pa.types.is_large_list(array.type)
        or pa.types.is_fixed_size_list(array.type)
    ):
        raise TypeError(
            f"dense sequence field {field_name!r} with dimension={dimension} must contain "
            f"list values, got {array.type}"
        )
    if array.null_count:
        raise ValueError(
            f"dense sequence field {field_name!r} contains null event vectors"
        )
    lengths = _numpy_backed_tensor(pc.list_value_length(array), torch.long)
    if lengths.numel() and bool((lengths != dimension).any().item()):
        observed = torch.unique(lengths)[:5].tolist()
        raise ValueError(
            f"dense sequence field {field_name!r} expected dimension={dimension}, "
            f"observed lengths={observed}"
        )
    flattened = pc.list_flatten(array)
    if flattened.null_count:
        flattened = pc.fill_null(flattened, 0.0)
    values = _numpy_backed_tensor(
        pc.cast(flattened, target_type=pa.float32(), safe=False),
        torch.float32,
    )
    return values.view(-1, dimension)


_ARANGE_CACHE: dict[int, Tensor] = {}
_NP_ARANGE_CACHE: dict[int, np.ndarray] = {}


def _cached_arange(length: int) -> Tensor:
    cached = _ARANGE_CACHE.get(length)
    if cached is None:
        cached = torch.arange(length, dtype=torch.long)
        _ARANGE_CACHE[length] = cached
    return cached


def _cached_np_arange(length: int) -> np.ndarray:
    cached = _NP_ARANGE_CACHE.get(length)
    if cached is None:
        cached = np.arange(length, dtype=np.int64)
        _NP_ARANGE_CACHE[length] = cached
    return cached


def _build_abs_window_gather_plan(
    unique_idx: np.ndarray,
    abs_lo: np.ndarray,
    abs_hi: np.ndarray,
    *,
    n_unique: int,
    window_lengths: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """Build per-unique src/dst index arrays shared by every aligned field.

    Pack batches often have many sequence fields over the same windows; paying
    for index construction once then doing ``flat[dst] = buf[src]`` per field
    beats concatenating hundreds of short views for each field.
    """

    lengths = abs_hi - abs_lo if window_lengths is None else window_lengths
    total = int(lengths.sum())
    empty = np.empty(0, dtype=np.int64)
    if total == 0 or n_unique <= 0:
        return [empty] * max(n_unique, 0), [empty] * max(n_unique, 0), 0
    n_rows = int(lengths.shape[0])
    row_flat_start = np.zeros(n_rows, dtype=np.int64)
    if n_rows > 1:
        row_flat_start[1:] = np.cumsum(lengths[:-1])
    src_by_unique: list[np.ndarray] = []
    dst_by_unique: list[np.ndarray] = []
    for unique in range(n_unique):
        rows = np.flatnonzero(unique_idx == unique)
        if rows.size == 0:
            src_by_unique.append(empty)
            dst_by_unique.append(empty)
            continue
        lens = lengths[rows]
        group_total = int(lens.sum())
        if group_total == 0:
            src_by_unique.append(empty)
            dst_by_unique.append(empty)
            continue
        cs = np.cumsum(lens)
        flat_r = _cached_np_arange(group_total)
        which = np.searchsorted(cs, flat_r, side="right")
        start_g = np.empty(lens.shape[0], dtype=np.int64)
        start_g[0] = 0
        if lens.shape[0] > 1:
            start_g[1:] = cs[:-1]
        within = flat_r - start_g[which]
        row_ids = rows[which]
        src_by_unique.append(abs_lo[row_ids] + within)
        dst_by_unique.append(row_flat_start[row_ids] + within)
    return src_by_unique, dst_by_unique, total


def _gather_flat_with_abs_plan(
    values_by_unique: Sequence[np.ndarray],
    src_by_unique: Sequence[np.ndarray],
    dst_by_unique: Sequence[np.ndarray],
    total: int,
    *,
    dtype: Any,
) -> np.ndarray:
    flat = np.empty(total, dtype=dtype)
    for unique, buffer in enumerate(values_by_unique):
        src = src_by_unique[unique]
        if src.size:
            flat[dst_by_unique[unique]] = buffer[src]
    return flat


def _gather_abs_windows_prehashed_padded(
    values_by_unique: Sequence[np.ndarray],
    unique_idx: np.ndarray,
    abs_lo: np.ndarray,
    abs_hi: np.ndarray,
    *,
    max_length: int,
    num_buckets: int,
    padding_id: int,
    validate_nonzero: bool,
    feature_name: str,
    window_lengths: np.ndarray | None = None,
    pad_mask: np.ndarray | None = None,
    unique_list: Sequence[int] | None = None,
    lo_list: Sequence[int] | None = None,
    hi_list: Sequence[int] | None = None,
    gather_plan: tuple[list[np.ndarray], list[np.ndarray], int] | None = None,
) -> Tensor:
    """Gather ragged abs windows into a padded pre_hashed ``(rows, max_len)`` tensor.

    Prefer a shared abs-window gather plan (one index build / many fields) when
    available; otherwise fall back to concatenate-of-views.
    """

    n_rows = int(abs_lo.shape[0])
    if n_rows == 0:
        return torch.zeros((0, max_length), dtype=torch.long)
    lengths = abs_hi - abs_lo if window_lengths is None else window_lengths
    if gather_plan is not None:
        src_by_unique, dst_by_unique, total = gather_plan
    else:
        total = int(lengths.sum())
        src_by_unique = dst_by_unique = None  # type: ignore[assignment]
    if max_length == 0 or total == 0:
        return torch.full(
            (n_rows, max_length),
            int(padding_id),
            dtype=torch.long,
        )
    sample_dtype = np.int64
    for buffer in values_by_unique:
        if buffer.size:
            sample_dtype = buffer.dtype
            break
    if src_by_unique is not None and dst_by_unique is not None:
        flat = _gather_flat_with_abs_plan(
            values_by_unique,
            src_by_unique,
            dst_by_unique,
            total,
            dtype=sample_dtype,
        )
    else:
        if unique_list is None:
            unique_list = unique_idx.tolist()
        if lo_list is None:
            lo_list = abs_lo.tolist()
        if hi_list is None:
            hi_list = abs_hi.tolist()
        views = [
            values_by_unique[unique_list[row_index]][
                lo_list[row_index] : hi_list[row_index]
            ]
            for row_index in range(n_rows)
            if lo_list[row_index] < hi_list[row_index]
        ]
        flat = np.concatenate(views) if views else np.empty(0, dtype=sample_dtype)
    normalized = flat.astype(np.int64, copy=False)
    if validate_nonzero and normalized.size and bool(np.any(normalized == 0)):
        raise ValueError(
            f"pre_hashed input {feature_name!r} contains non-null zero values"
        )
    encoded = (normalized & (int(num_buckets) - 1)) + 1
    if int(lengths.min(initial=0)) == max_length and total == n_rows * max_length:
        return torch.from_numpy(
            np.ascontiguousarray(encoded.reshape(n_rows, max_length))
        )
    out = (
        np.zeros((n_rows, max_length), dtype=np.int64)
        if padding_id == 0
        else np.full((n_rows, max_length), int(padding_id), dtype=np.int64)
    )
    mask = (
        pad_mask
        if pad_mask is not None
        else _cached_np_arange(max_length)[None, :] < lengths[:, None]
    )
    out[mask] = encoded
    return torch.from_numpy(out)


def _gather_abs_windows_dense_padded(
    values_by_unique: Sequence[np.ndarray],
    unique_idx: np.ndarray,
    abs_lo: np.ndarray,
    abs_hi: np.ndarray,
    *,
    max_length: int,
    dimension: int,
    window_lengths: np.ndarray | None = None,
    pad_mask: np.ndarray | None = None,
    gather_plan: tuple[list[np.ndarray], list[np.ndarray], int] | None = None,
) -> Tensor:
    """Gather ragged abs windows into a padded float32 sequence tensor."""

    n_rows = int(abs_lo.shape[0])
    if dimension <= 1:
        shape: tuple[int, ...] = (n_rows, max_length)
    else:
        shape = (n_rows, max_length, dimension)
    if n_rows == 0 or max_length == 0:
        return torch.zeros(shape, dtype=torch.float32)
    lengths = abs_hi - abs_lo if window_lengths is None else window_lengths
    if gather_plan is not None:
        src_by_unique, dst_by_unique, total = gather_plan
    else:
        total = int(lengths.sum())
        src_by_unique = dst_by_unique = None  # type: ignore[assignment]
    if total == 0:
        return torch.zeros(shape, dtype=torch.float32)
    sample_dtype = np.float32
    for buffer in values_by_unique:
        if buffer.size:
            sample_dtype = buffer.dtype
            break
    if src_by_unique is not None and dst_by_unique is not None:
        flat = _gather_flat_with_abs_plan(
            values_by_unique,
            src_by_unique,
            dst_by_unique,
            total,
            dtype=sample_dtype,
        )
    else:
        views = [
            values_by_unique[int(unique_idx[row_index])][
                int(abs_lo[row_index]) : int(abs_hi[row_index])
            ]
            for row_index in range(n_rows)
        ]
        flat = np.concatenate(views)
    values = flat.astype(np.float32, copy=False)
    if dimension <= 1:
        if int(lengths.min(initial=0)) == max_length and total == n_rows * max_length:
            return torch.from_numpy(
                np.ascontiguousarray(values.reshape(n_rows, max_length))
            )
        out = np.zeros((n_rows, max_length), dtype=np.float32)
        mask = (
            pad_mask
            if pad_mask is not None
            else _cached_np_arange(max_length)[None, :] < lengths[:, None]
        )
        out[mask] = values
        return torch.from_numpy(out)
    values = np.asarray(values, dtype=np.float32).reshape(total, dimension)
    out = np.zeros((n_rows, max_length, dimension), dtype=np.float32)
    cursor = 0
    for row_index in range(n_rows):
        length = int(lengths[row_index])
        if length:
            out[row_index, :length] = values[cursor : cursor + length]
            cursor += length
    return torch.from_numpy(out)


def _gather_padded_sequence(
    values: Tensor,
    starts: Tensor,
    lengths: Tensor,
    max_length: int,
    padding_value: int | float,
    *,
    flat_indices: Tensor | None = None,
    valid_mask: Tensor | None = None,
) -> Tensor:
    output_shape = (int(lengths.numel()), max_length, *values.shape[1:])
    if max_length == 0:
        return values.new_full(output_shape, padding_value)
    n_rows = int(lengths.numel())
    if n_rows == 0:
        return values.new_full(output_shape, padding_value)
    # Fast path: every row already has length == max_length.
    if (
        bool((lengths == max_length).all().item())
        and int(values.size(0)) == n_rows * max_length
    ):
        return values.view(output_shape)
    if flat_indices is None or valid_mask is None:
        positions = _cached_arange(max_length).unsqueeze(0)
        valid_mask = positions < lengths.unsqueeze(1)
        indices = starts.unsqueeze(1) + positions
        flat_indices = indices.clamp(
            min=0, max=max(int(values.size(0)) - 1, 0)
        ).reshape(-1)
    gathered = values.index_select(0, flat_indices).view(output_shape)
    mask = valid_mask
    for _ in values.shape[1:]:
        mask = mask.unsqueeze(-1)
    if padding_value == 0 or padding_value == 0.0:
        return gathered.masked_fill(~mask, 0)
    return torch.where(mask, gathered, gathered.new_full((), padding_value))


def _compact_direct_sequence_by_anchor(
    arrays: dict[str, Any],
    reference_offsets: Tensor,
    reference_base: int,
    reference_stop: int,
    *,
    anchor_field: str,
) -> tuple[dict[str, Any], Tensor, int, int]:
    """Remove flat tokens whose anchor value is null and rebuild list arrays."""

    pa, pc, _ds, _pq = _require_pyarrow()
    total_values = reference_stop - reference_base
    if total_values <= 0:
        return arrays, reference_offsets, reference_base, reference_stop

    anchor_array = arrays[anchor_field]
    anchor_flat = anchor_array.values.slice(reference_base, total_values)
    if anchor_flat.null_count == 0:
        return arrays, reference_offsets, reference_base, reference_stop

    keep_mask = pc.invert(anchor_flat.is_null())
    if not bool(pc.any(pc.invert(keep_mask)).as_py()):
        return arrays, reference_offsets, reference_base, reference_stop

    raw_lengths = (reference_offsets[1:] - reference_offsets[:-1]).tolist()
    keep_flags = keep_mask.to_pylist()
    new_lengths: list[int] = []
    offset = 0
    for length in raw_lengths:
        kept = sum(1 for flag in keep_flags[offset : offset + length] if flag)
        new_lengths.append(kept)
        offset += length

    new_offsets = [0]
    for length in new_lengths:
        new_offsets.append(new_offsets[-1] + length)
    keep_indices = [index for index, flag in enumerate(keep_flags) if flag]

    compacted: dict[str, Any] = {}
    for name, array in arrays.items():
        flat = array.values.slice(reference_base, total_values)
        if keep_indices:
            taken = pc.take(flat, pa.array(keep_indices, type=pa.int64()))
        else:
            taken = flat.slice(0, 0)
        offsets = pa.array(new_offsets, type=array.offsets.type)
        if pa.types.is_large_list(array.type):
            compacted[name] = pa.LargeListArray.from_arrays(offsets, taken)
        else:
            compacted[name] = pa.ListArray.from_arrays(offsets, taken)

    new_reference = torch.tensor(new_offsets, dtype=torch.long)
    return compacted, new_reference, 0, new_offsets[-1]


def _tensorize_direct_sequence(
    config: AppConfig,
    sequence: SequenceConfig,
    table: Any,
    *,
    validate_prehashed_nonzero: bool = True,
    validate_sequence_alignment: bool = True,
) -> dict[str, Any]:
    """Vectorize an identity-ID sequence from Arrow offsets and flat values."""

    arrays = {
        field.name: _normalized_list_array(table, field.source)
        for field in sequence.fields
    }
    reference_offsets: Tensor | None = None
    reference_base = 0
    reference_stop = 0
    for field in sequence.fields:
        if reference_offsets is not None and not validate_sequence_alignment:
            continue
        offsets = _list_offsets_tensor(arrays[field.name])
        base = int(offsets[0].item()) if offsets.numel() else 0
        normalized = offsets - base
        if reference_offsets is None:
            reference_offsets = normalized
            reference_base = base
            reference_stop = int(offsets[-1].item()) if offsets.numel() else base
        elif not torch.equal(normalized, reference_offsets):
            raise ValueError(
                f"sequence {sequence.name!r} field {field.name!r} has offsets that do not "
                "match the other aligned fields"
            )
    if reference_offsets is None:
        empty_lengths = torch.empty(0, dtype=torch.long)
        return {
            "fields": {},
            "lengths": empty_lengths,
            "has_sequence": empty_lengths > 0,
        }

    if sequence.null_anchor_field is not None:
        (
            arrays,
            reference_offsets,
            reference_base,
            reference_stop,
        ) = _compact_direct_sequence_by_anchor(
            arrays,
            reference_offsets,
            reference_base,
            reference_stop,
            anchor_field=sequence.null_anchor_field,
        )

    raw_lengths = reference_offsets[1:] - reference_offsets[:-1]
    lengths = raw_lengths
    tensor_max_length = _sequence_tensor_max_length(sequence)
    if tensor_max_length is not None:
        lengths = torch.clamp(raw_lengths, max=tensor_max_length)
    if sequence.truncation == "tail":
        starts = reference_offsets[1:] - lengths
    else:
        starts = reference_offsets[:-1]
    max_length = int(lengths.max().item()) if lengths.numel() else 0

    tensor_fields: dict[str, Tensor] = {}
    total_values = reference_stop - reference_base
    for field in sequence.fields:
        array = arrays[field.name]
        offsets = _list_offsets_tensor(array)
        base = int(offsets[0].item()) if offsets.numel() else 0
        flat = array.values.slice(base, total_values)
        if field.kind == "categorical":
            qualified = field.qualified_name(sequence.name)
            categorical_input = _effective_categorical_input(
                config,
                config.resolved.categorical_input_by_name[qualified],
            )
            if isinstance(categorical_input.encoding, ResolvedIdentityEncoding):
                values = _identity_array_tensor(flat, categorical_input)
            elif isinstance(categorical_input.encoding, ResolvedPreHashedEncoding):
                values = _pre_hashed_array_tensor(
                    flat,
                    categorical_input,
                    validate_nonzero=validate_prehashed_nonzero,
                )
            else:  # Guarded by _direct_sequence_supported.
                raise TypeError("unsupported direct categorical sequence encoding")
            padding_value: int | float = categorical_input.encoding.padding_id
        else:
            values = _direct_dense_values(flat, field.dimension, field.name)
            padding_value = 0.0
        tensor_fields[field.name] = _gather_padded_sequence(
            values,
            starts,
            lengths,
            max_length,
            padding_value,
        )
    return {
        "fields": tensor_fields,
        "lengths": lengths,
        "has_sequence": lengths > 0,
    }


def _dense_vector(value: Any, dimension: int) -> list[float]:
    """Normalize one dense element inside a sequence field."""
    if value is None:
        return [0.0] * dimension
    if dimension == 1 and not isinstance(value, (list, tuple)):
        return [float(value)]
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"dense sequence field expected {dimension} values, got scalar {value!r}"
        )
    if len(value) != dimension:
        raise ValueError(
            f"dense sequence field expected {dimension} values, got {len(value)}"
        )
    return [0.0 if item is None else float(item) for item in value]


def _sequence_rows(
    table: Any,
    sequence: SequenceConfig,
    *,
    validate_sequence_alignment: bool = True,
) -> tuple[dict[str, list[list[Any]]], list[int]]:
    """Align and truncate every field of a multi-field sequence.

    Fields within one sequence describe the same events and therefore must have
    equal lengths per row. Validating that invariant here prevents categorical
    and dense event attributes from becoming misaligned after padding.
    """
    if not sequence.fields:
        return {}, []
    values_by_field = {
        field.name: _column_values(table, field.source) for field in sequence.fields
    }
    batch_size = len(next(iter(values_by_field.values())))
    rows_by_field = {field.name: [] for field in sequence.fields}
    lengths: list[int] = []

    for row_index in range(batch_size):
        raw_items_by_field: dict[str, list[Any]] = {}
        row_length: int | None = None
        for field in sequence.fields:
            items = _coerce_sequence_items(values_by_field[field.name][row_index])
            if row_length is None:
                row_length = len(items)
            elif validate_sequence_alignment and len(items) != row_length:
                raise ValueError(
                    f"sequence {sequence.name!r} field {field.name!r} has length {len(items)} "
                    f"but expected {row_length} at row {row_index}"
                )
            raw_items_by_field[field.name] = items
        raw_items_by_field = _compress_row_fields_by_anchor(
            raw_items_by_field,
            anchor_field=sequence.null_anchor_field,
        )
        row_length = (
            len(next(iter(raw_items_by_field.values()))) if raw_items_by_field else 0
        )
        start, end = _sequence_bounds(row_length or 0, sequence)
        lengths.append(end - start)
        for field in sequence.fields:
            source_items = raw_items_by_field[field.name]
            rows_by_field[field.name].append(source_items[start:end])
    return rows_by_field, lengths


def _tensorize_multi_field_sequence(
    config: AppConfig,
    sequence: SequenceConfig,
    table: Any,
    vocab_maps: dict[str, dict[str, int]],
    *,
    validate_prehashed_nonzero: bool = True,
    validate_sequence_alignment: bool = True,
) -> dict[str, Any]:
    """Encode and right-pad one configured sequence to the batch maximum length."""
    if _direct_sequence_supported(config, sequence):
        return _tensorize_direct_sequence(
            config,
            sequence,
            table,
            validate_prehashed_nonzero=validate_prehashed_nonzero,
            validate_sequence_alignment=validate_sequence_alignment,
        )
    rows_by_field, row_lengths = _sequence_rows(
        table,
        sequence,
        validate_sequence_alignment=validate_sequence_alignment,
    )

    lengths = torch.tensor(row_lengths, dtype=torch.long)
    max_length = int(lengths.max().item()) if row_lengths else 0
    tensor_fields: dict[str, Tensor] = {}
    unseen_policy = config.vocab_strategy.defaults.unseen_policy
    for field in sequence.fields:
        rows = rows_by_field[field.name]
        if field.kind == "categorical":
            qualified = field.qualified_name(sequence.name)
            categorical_input = _effective_categorical_input(
                config,
                config.resolved.categorical_input_by_name[qualified],
            )
            encoded_rows = encode_categorical_sequence_field(
                rows,
                categorical_input,
                vocab_maps,
                unseen_policy,
            )
            padded = [row + [0] * (max_length - len(row)) for row in encoded_rows]
            tensor_fields[field.name] = (
                torch.tensor(padded, dtype=torch.long)
                if max_length > 0
                else torch.zeros(len(rows), 0, dtype=torch.long)
            )
        elif field.kind == "dense":
            encoded_dense = [
                [_dense_vector(item, field.dimension) for item in row] for row in rows
            ]
            zero = [0.0] * field.dimension
            padded_dense = [
                row + [zero] * (max_length - len(row)) for row in encoded_dense
            ]
            tensor_fields[field.name] = (
                torch.tensor(padded_dense, dtype=torch.float32)
                if max_length > 0
                else torch.zeros(len(rows), 0, field.dimension, dtype=torch.float32)
            )
        else:
            raise ValueError(f"unsupported sequence field kind {field.kind!r}")
    return {
        "fields": tensor_fields,
        "lengths": lengths,
        "has_sequence": lengths > 0,
    }


# --- Scenario and evaluation metadata ---


def _scenario_discovery_signature(
    config: AppConfig,
    split_name: str,
) -> str:
    split = _split_for_name(config, split_name)
    payload = {
        "version": 1,
        "source": config.scenarios.source,
        "split_format": split.format,
        "inputs": list(split.inputs),
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_scenario_discovery_cache(
    config: AppConfig,
    split_name: str,
) -> tuple[int, ...] | None:
    raw_path = config.scenarios.discovery_cache_path
    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("signature") != _scenario_discovery_signature(config, split_name):
        return None
    raw_values = payload.get("values")
    if not isinstance(raw_values, list) or not raw_values:
        raise ValueError(f"scenario discovery cache {path} has invalid values")
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in raw_values
    ):
        raise ValueError(f"scenario discovery cache {path} must contain integer values")
    values = tuple(sorted(set(raw_values)))
    if len(values) != len(raw_values):
        raise ValueError(f"scenario discovery cache {path} contains duplicate values")
    if len(values) > config.scenarios.max_discovered:
        raise ValueError(
            f"scenario discovery cache {path} exceeds max_discovered="
            f"{config.scenarios.max_discovered}"
        )
    return values


def _write_scenario_discovery_cache(
    config: AppConfig,
    split_name: str,
    values: tuple[int, ...],
) -> None:
    raw_path = config.scenarios.discovery_cache_path
    if raw_path is None:
        return
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = {
        "signature": _scenario_discovery_signature(config, split_name),
        "values": list(values),
    }
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _scenario_discovery_split(split: ParquetSplitConfig) -> ParquetSplitConfig:
    """Clone a train split with scanner settings suited to scene_id discovery.

    Training often uses tiny ``scanner_batch_rows`` and row-group LPT for the
    full Adapter path. Discovery only needs one integer column and should not
    inherit those knobs.
    """

    return replace(
        split,
        reader=replace(
            split.reader,
            scanner_batch_rows=262_144,
            shard_unit="file",
            eager_schema_validation="sample",
            schema_validation_samples=1,
            # Discovery is a one-shot metadata pass; keep I/O parallel but skip
            # Adapter-oriented length buckets / shuffle buffering.
            length_buckets=(),
            shuffle_buffer_rows=0,
            trusted_input=False,
        ),
    )


def _add_unique_scenario_values(
    values: set[int],
    array: Any,
    *,
    source: str,
    max_discovered: int,
) -> None:
    """Merge unique integer scenario ids from one Arrow array into ``values``."""

    pa, pc, _ds, _pq = _require_pyarrow()
    current = array
    if pa.types.is_dictionary(current.type):
        current = pc.take(current.dictionary, current.indices)

    # Agg layouts store request-level scene_id as list<int64>. Flatten once.
    if pa.types.is_list(current.type) or pa.types.is_large_list(current.type):
        if current.null_count:
            raise ValueError(f"scenario source {source!r} contains null")
        lengths = pc.list_value_length(current)
        if lengths.null_count:
            lengths = pc.fill_null(lengths, 0)
        if bool(pc.any(pc.equal(lengths, 0)).as_py()):
            raise ValueError(f"scenario source {source!r} contains an empty list")
        current = pc.list_flatten(current)
        if pa.types.is_list(current.type) or pa.types.is_large_list(current.type):
            raise ValueError(
                f"scenario source {source!r} must be a scalar or list of integers; "
                f"got nested list type {current.type}"
            )

    if current.null_count:
        raise ValueError(f"scenario source {source!r} contains null")
    if not (
        pa.types.is_integer(current.type)
        or pa.types.is_string(current.type)
        or pa.types.is_large_string(current.type)
    ):
        raise ValueError(
            f"scenario source {source!r} must contain integer ids; got {current.type}"
        )
    if pa.types.is_string(current.type) or pa.types.is_large_string(current.type):
        # Keep string scenes as a rejected path for raw integer discovery.
        raise ValueError(
            f"scenario source {source!r} must contain integer ids; got string values"
        )

    for value in pc.unique(current).to_pylist():
        if value is None:
            raise ValueError(f"scenario source {source!r} contains null")
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(
                f"scenario source {source!r} must contain integer ids; got {value!r}"
            )
        values.add(int(value))
        if len(values) > max_discovered:
            raise ValueError(
                f"scenario discovery found more than {max_discovered} values; "
                "increase scenarios.max_discovered only after checking the source column"
            )


def discover_scenario_values(
    config: AppConfig,
    *,
    split_name: str = "train",
) -> tuple[int, ...]:
    """Discover the complete finite raw integer scenario set from Parquet.

    This intentionally scans only the configured raw scenario column and runs
    before row sharding.  The caller is responsible for executing it on rank 0
    and broadcasting the result in distributed jobs.
    """

    scenario = config.scenarios
    if not scenario.auto_discover:
        raise ValueError("scenario discovery requires scenarios.auto_discover=true")
    if scenario.source is None:
        raise ValueError("scenario discovery requires scenarios.source")
    cached = _load_scenario_discovery_cache(config, split_name)
    if cached is not None:
        return cached
    split = _scenario_discovery_split(_split_for_name(config, split_name))
    scanner = ParquetScanner(split, [scenario.source])
    values: set[int] = set()

    for table in scanner.iter_tables():
        if scenario.source not in table.column_names:
            raise ValueError(
                f"scenario discovery is missing source column {scenario.source!r}"
            )
        _add_unique_scenario_values(
            values,
            _column_array(table, scenario.source),
            source=scenario.source,
            max_discovered=scenario.max_discovered,
        )
    if not values:
        raise ValueError(
            f"scenario discovery found no values in source column {scenario.source!r}"
        )
    result = tuple(sorted(values))
    _write_scenario_discovery_cache(config, split_name, result)
    return result


def resolve_auto_scenarios(
    config: AppConfig,
    discovered_values: Sequence[int] | None = None,
) -> AppConfig:
    """Return a validated immutable config with auto scenarios resolved."""

    scenario = config.scenarios
    if not scenario.auto_discover:
        return config
    values = (
        discover_scenario_values(config)
        if discovered_values is None
        else tuple(discovered_values)
    )
    if not values:
        raise ValueError("auto scenario resolution requires at least one value")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in values
    ):
        raise ValueError("auto scenario values must be integers")
    ordered = tuple(sorted({int(value) for value in values}))
    if len(ordered) != len(values):
        raise ValueError("auto scenario values must be unique")
    if len(ordered) > scenario.max_discovered:
        raise ValueError(
            f"auto scenario values exceed scenarios.max_discovered={scenario.max_discovered}"
        )
    # Pure non-MDL backbones use raw scenes only for batch routing and
    # per-scene evaluation; they do not instantiate MDL domain tokens or
    # scenario-scoped embedding tables. Resolving their scenario names is
    # therefore only a metadata operation.
    if config.model.name not in {
        "mdl_rankmixer",
        "mdl_onetrans",
        "mdl_mixformer",
    }:
        resolved = replace(
            config,
            scenarios=replace(
                scenario,
                names=tuple(str(value) for value in ordered),
                auto_discover=False,
                source_encoding="raw",
            ),
        )
        resolved.validate()
        return resolved

    template_features = [
        feature
        for feature in config.features
        if feature.name == _AUTO_SCENARIO_PRIOR_NAME
    ]
    if len(template_features) != 1:
        raise ValueError(
            "auto scenario resolution requires exactly one scenario prior template "
            f"feature named {_AUTO_SCENARIO_PRIOR_NAME!r}"
        )
    template_feature = template_features[0]
    template_tokens = [
        token
        for token in config.tokenization.scenario_tokens
        if token.name == _AUTO_SCENARIO_NAME
    ]
    if len(template_tokens) != 1:
        raise ValueError(
            "auto scenario resolution requires exactly one scenario token template "
            f"named {_AUTO_SCENARIO_NAME!r}"
        )
    template_token = template_tokens[0]
    if _AUTO_SCENARIO_PRIOR_NAME not in template_token.prior_inputs:
        raise ValueError(
            "auto scenario token template must reference the scenario prior template "
            "in prior_inputs"
        )

    def feature_name(value: int) -> str:
        slug = f"neg_{abs(value)}" if value < 0 else str(value)
        return f"scenario_{slug}_prior_scene_id_hn"

    expanded_features: list[Any] = []
    for feature in config.features:
        if feature.name != _AUTO_SCENARIO_PRIOR_NAME:
            expanded_features.append(feature)
            continue
        expanded_features.extend(
            replace(template_feature, name=feature_name(value)) for value in ordered
        )
    expanded_tokens = [
        replace(
            template_token,
            name=str(value),
            prior_inputs=tuple(
                feature_name(value)
                if input_name == _AUTO_SCENARIO_PRIOR_NAME
                else input_name
                for input_name in template_token.prior_inputs
            ),
        )
        for value in ordered
    ]
    expanded_tokens.extend(
        token
        for token in config.tokenization.scenario_tokens
        if token.name != _AUTO_SCENARIO_NAME
    )
    resolved = replace(
        config,
        features=tuple(expanded_features),
        scenarios=replace(
            scenario,
            names=tuple(str(value) for value in ordered),
            auto_discover=False,
            source_encoding="raw",
        ),
        tokenization=replace(
            config.tokenization,
            scenario_tokens=tuple(expanded_tokens),
        ),
    )
    resolved.validate()
    return resolved


def _encode_scenario_item(
    value: Any,
    scenario_to_id: dict[str, int],
    scenario_count: int,
    row_index: int,
    source_encoding: str,
) -> int:
    """Resolve a configured scenario name or ID, rejecting unknown routing values.

    Scenario IDs are model-routing semantics, not categorical vocab IDs: zero
    is a valid scenario rather than an OOV bucket, so categorical unseen_policy
    intentionally does not apply here.
    """
    if value is None:
        raise ValueError(f"scenario value is null at row {row_index}")
    if isinstance(value, bool):
        raise ValueError(
            f"scenario value must be a name or integer id at row {row_index}, got bool"
        )
    if isinstance(value, Integral):
        raw_name = str(int(value))
        if source_encoding != "index" and raw_name in scenario_to_id:
            return scenario_to_id[raw_name]
        if source_encoding == "raw":
            raise ValueError(f"unknown raw scenario id {int(value)} at row {row_index}")
        index = int(value)
        if 0 <= index < scenario_count:
            return index
        raise ValueError(
            f"scenario id {index} at row {row_index} is outside [0, {scenario_count - 1}]"
        )
    if isinstance(value, str):
        if source_encoding == "index":
            raise ValueError(
                f"scenario index must be an integer at row {row_index}, got {value!r}"
            )
        if value in scenario_to_id:
            return scenario_to_id[value]
        raise ValueError(f"unknown scenario name {value!r} at row {row_index}")
    raise ValueError(
        f"scenario value must be a configured name or integer id at row {row_index}, "
        f"got {type(value).__name__}"
    )


def _trusted_scalar_scenario_tensor(config: AppConfig, table: Any) -> Tensor | None:
    """Map a trusted scalar scenario column with Arrow kernels.

    The mapping itself is required model input work. Trusted profiles avoid a
    second Python type/null/range validation loop around that mapping.
    """

    pa, pc, _ds, _pq = _require_pyarrow()
    array = _column_array(table, config.scenarios.source)
    if pa.types.is_dictionary(array.type):
        array = pc.dictionary_decode(array)
    if _is_arrow_list_type(pa, array.type):
        return None

    source_encoding = config.scenarios.source_encoding
    if source_encoding == "index" and pa.types.is_integer(array.type):
        encoded = pc.cast(array, target_type=pa.int64(), safe=False)
        return _numpy_backed_tensor(encoded, torch.long)
    if source_encoding != "raw":
        return None

    if pa.types.is_integer(array.type):
        raw_values: list[Any] = [int(name) for name in config.scenarios.names]
    elif pa.types.is_string(array.type) or pa.types.is_large_string(array.type):
        raw_values = list(config.scenarios.names)
    else:
        return None
    encoded = pc.index_in(
        array,
        value_set=pa.array(raw_values, type=array.type),
    )
    encoded = pc.cast(encoded, target_type=pa.int64(), safe=False)
    return _numpy_backed_tensor(encoded, torch.long)


def _scenario_tensor(
    config: AppConfig,
    table: Any,
    batch_size: int,
    *,
    trusted_input: bool = False,
) -> Tensor:
    """Build scenario IDs or a multi-hot scenario mask for each row."""
    scenario_count = len(config.scenarios.names)
    if config.scenarios.source is None:
        if scenario_count != 1:
            raise ValueError(
                "scenarios.source is required when multiple scenarios are configured"
            )
        # Single-scenario models default every row to scenario index 0.
        return torch.zeros(batch_size, dtype=torch.long)

    if trusted_input:
        trusted = _trusted_scalar_scenario_tensor(config, table)
        if trusted is not None:
            return trusted

    scenario_to_id = {name: index for index, name in enumerate(config.scenarios.names)}
    row_indices: list[list[int]] = []
    saw_list_value = False
    for row_index, value in enumerate(_column_values(table, config.scenarios.source)):
        if isinstance(value, (list, tuple)):
            saw_list_value = True
            if not value:
                raise ValueError(f"scenario list is empty at row {row_index}")
            items = value
        else:
            items = [value]
        row_indices.append(
            [
                _encode_scenario_item(
                    item,
                    scenario_to_id,
                    scenario_count,
                    row_index,
                    config.scenarios.source_encoding,
                )
                for item in items
            ]
        )

    if saw_list_value:
        # List-valued cells produce a multi-hot mask over configured scenarios.
        mask = torch.zeros(batch_size, scenario_count, dtype=torch.float32)
        for row_index, indices in enumerate(row_indices):
            for index in indices:
                mask[row_index, index] = 1.0
        return mask
    return torch.tensor([indices[0] for indices in row_indices], dtype=torch.long)


def _group_ids(split: ParquetSplitConfig, table: Any, batch_size: int) -> list[str]:
    """Read grouping metadata from the active split, falling back to request ID."""
    source = split.group_id or split.request_id
    if source is None:
        return ["" for _ in range(batch_size)]
    if source not in table.column_names:
        raise ValueError(f"missing configured group-id column {source!r}")
    return [
        "" if value is None else str(value) for value in _column_values(table, source)
    ]


def _prediction_keys(split: ParquetSplitConfig, table: Any) -> dict[str, list[Any]]:
    """Preserve configured candidate identity without coercing scalar types."""

    result: dict[str, list[Any]] = {}
    for output_name, source in split.prediction_keys.items():
        values = _column_values(table, source)
        if len(values) != table.num_rows:
            raise RuntimeError(
                f"prediction key source {source!r} produced {len(values)} values for "
                f"{table.num_rows} rows"
            )
        result[output_name] = values
    return result


def _request_deduplication_plan(
    split: ParquetSplitConfig,
    table: Any,
    *,
    columns: Sequence[str] | None = None,
) -> tuple[Any, Tensor] | None:
    """Select one physical row per request and map candidates back to it.

    When ``columns`` is set, only those columns are taken (request/context/
    sequence sources). Candidate/item/label columns stay on the full table.
    """

    if not split.reader.deduplicate_request_features:
        return None
    if split.request_id is None:
        raise ValueError("request feature deduplication requires request_id")
    request_ids = _column_values(table, split.request_id)
    unique_positions: list[int] = []
    candidate_to_request: list[int] = []
    request_index: dict[Any, int] = {}
    trusted_input = split.reader.trusted_input
    if trusted_input:
        for row_index, request_id in enumerate(request_ids):
            existing = request_index.get(request_id)
            if existing is None:
                existing = len(unique_positions)
                request_index[request_id] = existing
                unique_positions.append(row_index)
            candidate_to_request.append(existing)
    else:
        for row_index, request_id in enumerate(request_ids):
            if request_id is None:
                raise ValueError(
                    f"request_id column {split.request_id!r} contains null at row {row_index}"
                )
            try:
                existing = request_index.get(request_id)
            except TypeError as error:
                raise ValueError(
                    f"request_id column {split.request_id!r} must contain hashable scalars"
                ) from error
            if existing is None:
                existing = len(unique_positions)
                request_index[request_id] = existing
                unique_positions.append(row_index)
            candidate_to_request.append(existing)
    take_table = table
    if columns is not None:
        names = set(table.column_names)
        selected_columns = [name for name in columns if name in names]
        if split.request_id not in selected_columns and split.request_id in names:
            selected_columns.insert(0, split.request_id)
        if not selected_columns:
            raise ValueError(
                "request feature deduplication projected an empty column set"
            )
        take_table = table.select(selected_columns)
    selected = (
        take_table
        if len(unique_positions) == table.num_rows
        else _safe_table_take(take_table, unique_positions)
    )
    return selected, torch.tensor(candidate_to_request, dtype=torch.long)


def _indexed_request_value(value: Any, row_indices: Tensor) -> dict[str, Any]:
    if isinstance(value, dict):
        return {**value, "row_indices": row_indices}
    return {"values": value, "row_indices": row_indices}


def _tensorize_python_categorical_values(
    config: AppConfig,
    categorical_input: ResolvedCategoricalInput,
    values: Sequence[Any],
    vocab_maps: dict[str, dict[str, int]],
    *,
    validate_prehashed_nonzero: bool,
) -> Tensor:
    """Encode normalized Python values without rebuilding an Arrow array."""

    categorical_input = _effective_categorical_input(config, categorical_input)
    encoding = categorical_input.encoding

    def int64_values() -> tuple[np.ndarray, np.ndarray]:
        if isinstance(values, np.ndarray):
            if values.size == 0:
                return (
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=bool),
                )
            array = values
        else:
            if not values:
                return (
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=bool),
                )
            array = np.asarray(values)
        if array.dtype.kind == "i" and array.dtype.itemsize <= 8:
            return array.astype(np.int64, copy=False), np.zeros(
                array.size,
                dtype=bool,
            )
        if array.dtype.kind == "u" and array.dtype.itemsize <= 8:
            if array.size and int(array.max()) >= (1 << 63):
                raise OverflowError(
                    f"categorical input {categorical_input.name!r} contains "
                    "a value outside signed int64"
                )
            return array.astype(np.int64, copy=False), np.zeros(
                array.size,
                dtype=bool,
            )

        normalized = np.empty(
            array.size if isinstance(values, np.ndarray) else len(values),
            dtype=np.int64,
        )
        nulls = np.zeros(normalized.size, dtype=bool)
        for index, value in enumerate(values):
            if value is None:
                nulls[index] = True
                normalized[index] = 0
                continue
            if isinstance(value, bool) or not isinstance(
                value,
                (int, np.integer),
            ):
                raise TypeError(
                    f"categorical input {categorical_input.name!r} must contain "
                    f"int64 values, got {type(value).__name__}"
                )
            integer = int(value)
            if integer < -(1 << 63) or integer >= (1 << 63):
                raise OverflowError(
                    f"categorical input {categorical_input.name!r} contains "
                    f"a value outside signed int64: {integer}"
                )
            normalized[index] = integer
        return normalized, nulls

    if isinstance(encoding, ResolvedIdentityEncoding):
        normalized, nulls = int64_values()
        present = normalized[~nulls]
        minimum = int(present.min()) if present.size else None
        maximum = int(present.max()) if present.size else None
        invalid_bounds = (minimum is not None and minimum < 0) or (
            maximum is not None and maximum >= encoding.num_buckets
        )
        if invalid_bounds and encoding.out_of_range == "error":
            raise ValueError(
                f"identity input {categorical_input.name!r} contains IDs outside "
                f"[0, {encoding.num_buckets}): min={minimum}, max={maximum}"
            )
        if invalid_bounds:
            valid = (normalized >= 0) & (normalized < encoding.num_buckets)
            normalized = np.where(
                valid,
                normalized,
                int(encoding.padding_id),
            )
        elif nulls.any():
            normalized = normalized.copy()
        if nulls.any():
            normalized[nulls] = int(encoding.padding_id)
        return torch.from_numpy(normalized)

    if isinstance(encoding, ResolvedPreHashedEncoding):
        normalized, nulls = int64_values()
        if (
            validate_prehashed_nonzero
            and normalized.size
            and bool(np.any((normalized == 0) & ~nulls))
        ):
            raise ValueError(
                f"pre_hashed input {categorical_input.name!r} contains "
                "non-null zero values"
            )
        encoded_values = np.bitwise_and(normalized, encoding.num_buckets - 1) + 1
        if nulls.any():
            encoded_values[nulls] = int(encoding.padding_id)
        return torch.from_numpy(encoded_values)

    unseen_policy = config.vocab_strategy.defaults.unseen_policy
    return torch.tensor(
        encode_categorical_values(
            values,
            categorical_input,
            vocab_maps,
            unseen_policy,
        ),
        dtype=torch.long,
    )


def _gather_bag_from_sequence_column_batch(
    values: Any,
    *,
    max_length: int | None,
    truncation: str,
    column_groups: Sequence[np.ndarray] | None = None,
    window_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray, np.ndarray]]
    | None = None,
) -> tuple[Any, np.ndarray]:
    """Flat-gather one CompactListColumn bag with optional head/tail truncate."""

    columns = values.columns
    slots = values.slots
    column_index = values.column_index
    n_rows = int(slots.shape[0]) if hasattr(slots, "shape") else len(slots)
    values_by_unique = [column.values for column in columns]
    offsets_by_unique = [column.offsets for column in columns]
    cache_key = None
    if window_cache is not None:
        cache_key = (
            id(slots),
            id(column_index) if column_index is not None else None,
            tuple(id(offsets) for offsets in offsets_by_unique),
            None if max_length is None else int(max_length),
            truncation,
        )
        cached = window_cache.get(cache_key)
        if cached is not None:
            starts, stops, lengths = cached
            sample_dtype = np.int64
            for buffer in values_by_unique:
                if buffer.size:
                    sample_dtype = buffer.dtype
                    break
            total = int(lengths.sum())
            if total == 0:
                return np.empty(0, dtype=sample_dtype), lengths
            start_list = starts.tolist()
            stop_list = stops.tolist()
            if column_index is None:
                views = [
                    values_by_unique[index][start_list[index] : stop_list[index]]
                    for index in range(n_rows)
                    if start_list[index] < stop_list[index]
                ]
            else:
                unique_list = column_index.tolist()
                views = [
                    values_by_unique[unique_list[index]][
                        start_list[index] : stop_list[index]
                    ]
                    for index in range(n_rows)
                    if start_list[index] < stop_list[index]
                ]
            return np.concatenate(views), lengths

    starts = np.empty(n_rows, dtype=np.int64)
    stops = np.empty(n_rows, dtype=np.int64)
    if column_index is None:
        for index in range(n_rows):
            offsets = offsets_by_unique[index]
            slot = int(slots[index])
            starts[index] = offsets[slot]
            stops[index] = offsets[slot + 1]
        unique_for_row = None
    else:
        groups = column_groups
        if groups is None:
            groups = [
                np.flatnonzero(column_index == unique_idx)
                for unique_idx in range(len(offsets_by_unique))
            ]
        for unique_idx, offsets in enumerate(offsets_by_unique):
            reqs = groups[unique_idx]
            if reqs.size == 0:
                continue
            row_slots = slots[reqs]
            starts[reqs] = offsets[row_slots]
            stops[reqs] = offsets[row_slots + 1]
        unique_for_row = column_index
    lengths = stops - starts
    if max_length is not None:
        max_length_i = int(max_length)
        if int(lengths.max(initial=0)) > max_length_i:
            capped = np.minimum(lengths, max_length_i)
            if truncation == "head":
                stops = starts + capped
            elif truncation == "tail":
                starts = stops - capped
            else:
                raise ValueError(f"unsupported list truncation {truncation!r}")
            lengths = capped
    if window_cache is not None and cache_key is not None:
        window_cache[cache_key] = (starts, stops, lengths)
    sample_dtype = np.int64
    for buffer in values_by_unique:
        if buffer.size:
            sample_dtype = buffer.dtype
            break
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=sample_dtype), lengths
    start_list = starts.tolist()
    stop_list = stops.tolist()
    if unique_for_row is None:
        views = [
            values_by_unique[index][start_list[index] : stop_list[index]]
            for index in range(n_rows)
            if start_list[index] < stop_list[index]
        ]
    else:
        unique_list = unique_for_row.tolist()
        views = [
            values_by_unique[unique_list[index]][start_list[index] : stop_list[index]]
            for index in range(n_rows)
            if start_list[index] < stop_list[index]
        ]
    return np.concatenate(views), lengths


def _tensorize_python_categorical_bag(
    config: AppConfig,
    feature: FeatureConfig,
    values: Sequence[Any],
    vocab_maps: dict[str, dict[str, int]],
    *,
    validate_prehashed_nonzero: bool,
    column_groups: Sequence[np.ndarray] | None = None,
    window_cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray, np.ndarray]]
    | None = None,
) -> dict[str, Tensor]:
    """Encode list-valued Python rows as flat values plus row lengths."""

    if feature.pooling != "mean":
        raise TypeError("_tensorize_python_categorical_bag requires pooling=mean")
    if type(values).__name__ == "SequenceColumnBatch":
        flat, lengths_arr = _gather_bag_from_sequence_column_batch(
            values,
            max_length=feature.max_length,
            truncation=feature.truncation,
            column_groups=column_groups,
            window_cache=window_cache,
        )
        categorical_input = _effective_categorical_input(
            config,
            config.resolved.categorical_input_by_name[feature.name],
        )
        encoding = categorical_input.encoding
        if (
            isinstance(encoding, ResolvedPreHashedEncoding)
            and isinstance(flat, np.ndarray)
            and flat.dtype.kind in "iu"
        ):
            normalized = flat.astype(np.int64, copy=False)
            if (
                validate_prehashed_nonzero
                and normalized.size
                and bool(np.any(normalized == 0))
            ):
                raise ValueError(
                    f"pre_hashed input {categorical_input.name!r} contains "
                    "non-null zero values"
                )
            encoded = np.bitwise_and(normalized, int(encoding.num_buckets) - 1) + 1
            if not encoded.flags.c_contiguous:
                encoded = np.ascontiguousarray(encoded)
            return {
                "values": torch.from_numpy(encoded),
                "lengths": torch.from_numpy(
                    lengths_arr
                    if lengths_arr.flags.c_contiguous
                    else np.ascontiguousarray(lengths_arr)
                ),
            }
        return {
            "values": _tensorize_python_categorical_values(
                config,
                categorical_input,
                flat,
                vocab_maps,
                validate_prehashed_nonzero=validate_prehashed_nonzero,
            ),
            "lengths": torch.from_numpy(
                lengths_arr
                if lengths_arr.flags.c_contiguous
                else np.ascontiguousarray(lengths_arr)
            ),
        }

    n_rows = len(values)
    if n_rows == 0:
        categorical_input = config.resolved.categorical_input_by_name[feature.name]
        return {
            "values": _tensorize_python_categorical_values(
                config,
                categorical_input,
                np.empty(0, dtype=np.int64),
                vocab_maps,
                validate_prehashed_nonzero=validate_prehashed_nonzero,
            ),
            "lengths": torch.zeros(0, dtype=torch.long),
        }

    max_length = feature.max_length
    truncation = feature.truncation
    # Prepare emits ndarray views for CompactListColumn bags; avoid scanning
    # isinstance/len twice via generator expressions.
    first = values[0]
    ndarray_rows = first is None or isinstance(first, np.ndarray)
    if ndarray_rows:
        lengths_arr = np.empty(n_rows, dtype=np.int64)
        pieces: list[np.ndarray] = []
        empty = np.empty(
            0,
            dtype=getattr(first, "dtype", np.int64)
            if isinstance(first, np.ndarray)
            else np.int64,
        )
        for index, row in enumerate(values):
            if row is None:
                pieces.append(empty)
                lengths_arr[index] = 0
                continue
            if not isinstance(row, np.ndarray):
                ndarray_rows = False
                break
            length = int(row.shape[0])
            if max_length is not None and length > max_length:
                if truncation == "tail":
                    row = row[-max_length:]
                elif truncation == "head":
                    row = row[:max_length]
                else:
                    raise ValueError(f"unsupported list truncation {truncation!r}")
                length = int(max_length)
            pieces.append(row)
            lengths_arr[index] = length
        if ndarray_rows:
            total = int(lengths_arr.sum())
            flat = np.concatenate(pieces) if total else empty
            categorical_input = config.resolved.categorical_input_by_name[feature.name]
            return {
                "values": _tensorize_python_categorical_values(
                    config,
                    categorical_input,
                    flat,
                    vocab_maps,
                    validate_prehashed_nonzero=validate_prehashed_nonzero,
                ),
                "lengths": torch.from_numpy(np.ascontiguousarray(lengths_arr)),
            }

    rows: list[Sequence[Any]] = []
    lengths: list[int] = []
    for row_index, value in enumerate(values):
        if value is None:
            row: Sequence[Any] = ()
        elif isinstance(value, np.ndarray):
            row = value
        elif isinstance(value, (list, tuple)):
            row = value
        else:
            raise TypeError(
                f"categorical bag {feature.source!r} row {row_index} must be "
                f"list-valued, got {type(value).__name__}"
            )
        if max_length is not None and len(row) > max_length:
            if truncation == "tail":
                row = row[-max_length:]
            elif truncation == "head":
                row = row[:max_length]
            else:
                raise ValueError(f"unsupported list truncation {truncation!r}")
        rows.append(row)
        lengths.append(len(row))
    if rows and all(isinstance(row, np.ndarray) for row in rows):
        flat = (
            np.concatenate(rows)  # type: ignore[arg-type]
            if any(len(row) for row in rows)
            else np.empty(0, dtype=getattr(rows[0], "dtype", np.int64))
        )
    else:
        flat = [item for row in rows for item in row]
    categorical_input = config.resolved.categorical_input_by_name[feature.name]
    return {
        "values": _tensorize_python_categorical_values(
            config,
            categorical_input,
            flat,
            vocab_maps,
            validate_prehashed_nonzero=validate_prehashed_nonzero,
        ),
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }


def _tensorize_axis_sequence(
    config: AppConfig,
    sequence: SequenceConfig,
    request_values: Mapping[str, Sequence[Any]],
    selection_plan: Any,
    vocab_maps: dict[str, dict[str, int]],
    *,
    validate_prehashed_nonzero: bool,
) -> dict[str, Any]:
    """Tensorize every aligned field using one shared pack-time selection."""

    lengths = torch.as_tensor(
        selection_plan.compacted_lengths,
        dtype=torch.long,
    )
    starts = torch.zeros_like(lengths)
    if lengths.numel() > 1:
        starts[1:] = torch.cumsum(lengths[:-1], dim=0)
    max_length = int(lengths.max().item()) if lengths.numel() else 0
    tensor_fields: dict[str, Tensor] = {}
    use_direct_shapes = _direct_sequence_supported(config, sequence)

    selections = selection_plan.selections
    compacted = selection_plan.compacted_lengths
    n_plan_rows = (
        int(compacted.shape[0]) if hasattr(compacted, "shape") else len(compacted)
    )
    total_tokens = int(compacted.sum()) if n_plan_rows else 0
    all_ranges = bool(getattr(selection_plan, "selections_are_ranges", False))
    if not all_ranges and selections:
        all_ranges = all(
            isinstance(indices, tuple)
            and len(indices) == 2
            and not isinstance(indices[0], (list, tuple, np.ndarray))
            for indices in selections
        )

    # When every field is a SequenceColumnBatch sharing slots/column_index,
    # compute absolute value-buffer windows once and reuse across fields.
    shared_abs_lo: np.ndarray | None = None
    shared_abs_hi: np.ndarray | None = None
    shared_unique_idx: np.ndarray | None = None
    primary_batch = None
    if sequence.fields and all_ranges:
        primary_rows = request_values.get(sequence.fields[0].source)
        if type(primary_rows).__name__ == "SequenceColumnBatch":
            primary_batch = primary_rows
            same_layout = True
            for field in sequence.fields[1:]:
                other = request_values.get(field.source)
                if (
                    type(other).__name__ != "SequenceColumnBatch"
                    or other.slots is not primary_rows.slots
                    or other.column_index is not primary_rows.column_index
                ):
                    same_layout = False
                    break
            if same_layout:
                n_rows = n_plan_rows
                shared_unique_idx = np.empty(n_rows, dtype=np.int16)
                shared_abs_lo = np.empty(n_rows, dtype=np.int64)
                shared_abs_hi = np.empty(n_rows, dtype=np.int64)
                columns = primary_rows.columns
                slots = primary_rows.slots
                column_index = primary_rows.column_index
                range_starts = getattr(selection_plan, "range_starts", None)
                range_ends = getattr(selection_plan, "range_ends", None)
                if range_starts is not None and range_ends is not None:
                    # Vectorize bases per unique CompactListColumn.
                    if column_index is None:
                        shared_unique_idx[:] = np.arange(n_rows, dtype=np.int16)
                        for row_index in range(n_rows):
                            base = int(
                                columns[row_index].offsets[int(slots[row_index])]
                            )
                            shared_abs_lo[row_index] = base + int(
                                range_starts[row_index]
                            )
                            shared_abs_hi[row_index] = base + int(range_ends[row_index])
                    else:
                        shared_unique_idx[:] = column_index
                        for unique_idx, column in enumerate(columns):
                            reqs = np.flatnonzero(column_index == unique_idx)
                            if reqs.size == 0:
                                continue
                            bases = column.offsets[
                                slots[reqs].astype(np.int64, copy=False)
                            ]
                            shared_abs_lo[reqs] = bases + range_starts[reqs]
                            shared_abs_hi[reqs] = bases + range_ends[reqs]
                else:
                    for row_index, indices in enumerate(selections):
                        unique_idx = (
                            row_index
                            if column_index is None
                            else int(column_index[row_index])
                        )
                        base = int(columns[unique_idx].offsets[int(slots[row_index])])
                        shared_unique_idx[row_index] = unique_idx
                        shared_abs_lo[row_index] = base + int(indices[0])
                        shared_abs_hi[row_index] = base + int(indices[1])

    # Shared pad indices across fields that still take the flat→pad path.
    shared_pad_indices: Tensor | None = None
    shared_pad_mask: Tensor | None = None
    shared_window_lengths: np.ndarray | None = None
    shared_np_pad_mask: np.ndarray | None = None
    shared_unique_list: list[int] | None = None
    shared_lo_list: list[int] | None = None
    shared_hi_list: list[int] | None = None
    shared_gather_plan: tuple[list[np.ndarray], list[np.ndarray], int] | None = None
    if shared_abs_lo is not None and shared_abs_hi is not None and max_length > 0:
        shared_window_lengths = shared_abs_hi - shared_abs_lo
        shared_np_pad_mask = (
            _cached_np_arange(max_length)[None, :] < shared_window_lengths[:, None]
        )
        if shared_unique_idx is not None:
            shared_unique_list = shared_unique_idx.tolist()
            shared_lo_list = shared_abs_lo.tolist()
            shared_hi_list = shared_abs_hi.tolist()
            n_unique = (
                int(len(primary_batch.columns))
                if primary_batch is not None
                else int(shared_unique_idx.max(initial=-1)) + 1
            )
            if n_unique > 0 and sequence.fields:
                shared_gather_plan = _build_abs_window_gather_plan(
                    shared_unique_idx,
                    shared_abs_lo,
                    shared_abs_hi,
                    n_unique=n_unique,
                    window_lengths=shared_window_lengths,
                )

    # Wide sequences (e.g. view_long with ~30 fields) pay the same abs-window
    # gather per field. Numpy releases the GIL, so a small thread pool overlaps
    # independent pre_hashed gathers without contending with training.
    parallel_prehashed: list[tuple[Any, Any, Sequence[np.ndarray]]] = []
    if (
        use_direct_shapes
        and shared_abs_lo is not None
        and shared_abs_hi is not None
        and shared_unique_idx is not None
        and primary_batch is not None
        and shared_gather_plan is not None
        and len(sequence.fields) >= 6
    ):
        for field in sequence.fields:
            if field.kind != "categorical" or field.source not in request_values:
                continue
            rows = request_values[field.source]
            if (
                type(rows).__name__ != "SequenceColumnBatch"
                or rows.slots is not primary_batch.slots
                or rows.column_index is not primary_batch.column_index
                or len(rows) != n_plan_rows
            ):
                continue
            qualified = field.qualified_name(sequence.name)
            categorical_input = _effective_categorical_input(
                config,
                config.resolved.categorical_input_by_name[qualified],
            )
            encoding = categorical_input.encoding
            if not isinstance(encoding, ResolvedPreHashedEncoding):
                continue
            parallel_prehashed.append(
                (
                    field,
                    categorical_input,
                    [column.values for column in rows.columns],
                )
            )

    if len(parallel_prehashed) >= 6:

        def _parallel_prehashed_one(
            item: tuple[Any, Any, Sequence[np.ndarray]],
        ) -> tuple[str, Tensor]:
            field, categorical_input, values_by_unique = item
            encoding = categorical_input.encoding
            assert isinstance(encoding, ResolvedPreHashedEncoding)
            return field.name, _gather_abs_windows_prehashed_padded(
                values_by_unique,
                shared_unique_idx,
                shared_abs_lo,
                shared_abs_hi,
                max_length=max_length,
                num_buckets=int(encoding.num_buckets),
                padding_id=int(encoding.padding_id),
                validate_nonzero=validate_prehashed_nonzero,
                feature_name=categorical_input.name,
                window_lengths=shared_window_lengths,
                pad_mask=shared_np_pad_mask,
                unique_list=shared_unique_list,
                lo_list=shared_lo_list,
                hi_list=shared_hi_list,
                gather_plan=shared_gather_plan,
            )

        workers = min(4, len(parallel_prehashed))
        pool = _prehashed_gather_pool(workers)
        for name, tensor in pool.map(_parallel_prehashed_one, parallel_prehashed):
            tensor_fields[name] = tensor
        parallel_done = {field.name for field, _ci, _vals in parallel_prehashed}
    else:
        parallel_done = set()

    for field in sequence.fields:
        if field.name in parallel_done:
            continue
        if field.source not in request_values:
            raise ValueError(
                f"sequence source {field.source!r} missing from direct request axis"
            )
        rows = request_values[field.source]
        if len(rows) != n_plan_rows:
            raise RuntimeError(
                f"sequence {sequence.name!r} source {field.source!r} has "
                f"{len(rows)} request rows, expected {n_plan_rows}"
            )
        pieces: list[np.ndarray] = []
        list_rows: list[Sequence[Any]] = []
        use_pieces = True
        selected: Any = None
        selected_ready = False
        batch_name = type(rows).__name__
        # Fast path: shared abs windows + pre_hashed/dense → padded 2D directly.
        if (
            use_direct_shapes
            and batch_name == "SequenceColumnBatch"
            and shared_abs_lo is not None
            and shared_abs_hi is not None
            and shared_unique_idx is not None
            and primary_batch is not None
            and rows.slots is primary_batch.slots
            and rows.column_index is primary_batch.column_index
        ):
            values_by_unique = [column.values for column in rows.columns]
            if field.kind == "categorical":
                qualified = field.qualified_name(sequence.name)
                categorical_input = _effective_categorical_input(
                    config,
                    config.resolved.categorical_input_by_name[qualified],
                )
                encoding = categorical_input.encoding
                if isinstance(encoding, ResolvedPreHashedEncoding):
                    tensor_fields[field.name] = _gather_abs_windows_prehashed_padded(
                        values_by_unique,
                        shared_unique_idx,
                        shared_abs_lo,
                        shared_abs_hi,
                        max_length=max_length,
                        num_buckets=int(encoding.num_buckets),
                        padding_id=int(encoding.padding_id),
                        validate_nonzero=validate_prehashed_nonzero,
                        feature_name=categorical_input.name,
                        window_lengths=shared_window_lengths,
                        pad_mask=shared_np_pad_mask,
                        unique_list=shared_unique_list,
                        lo_list=shared_lo_list,
                        hi_list=shared_hi_list,
                        gather_plan=shared_gather_plan,
                    )
                    continue
                if isinstance(encoding, ResolvedIdentityEncoding):
                    n_rows = int(shared_abs_lo.shape[0])
                    padding_id = int(encoding.padding_id)
                    num_buckets = int(encoding.num_buckets)
                    window_lengths = (
                        shared_window_lengths
                        if shared_window_lengths is not None
                        else shared_abs_hi - shared_abs_lo
                    )
                    total = int(window_lengths.sum())
                    if max_length == 0 or total == 0:
                        tensor_fields[field.name] = torch.full(
                            (n_rows, max_length),
                            padding_id,
                            dtype=torch.long,
                        )
                        continue
                    if (
                        shared_unique_list is not None
                        and shared_lo_list is not None
                        and shared_hi_list is not None
                    ):
                        views = [
                            values_by_unique[shared_unique_list[row_index]][
                                shared_lo_list[row_index] : shared_hi_list[row_index]
                            ]
                            for row_index in range(n_rows)
                            if shared_lo_list[row_index] < shared_hi_list[row_index]
                        ]
                    else:
                        views = [
                            values_by_unique[int(shared_unique_idx[row_index])][
                                int(shared_abs_lo[row_index]) : int(
                                    shared_abs_hi[row_index]
                                )
                            ]
                            for row_index in range(n_rows)
                        ]
                    flat = np.concatenate(views).astype(np.int64, copy=False)
                    if flat.size:
                        minimum = int(flat.min())
                        maximum = int(flat.max())
                        invalid = minimum < 0 or maximum >= num_buckets
                        if invalid and encoding.out_of_range == "error":
                            raise ValueError(
                                f"identity input {categorical_input.name!r} "
                                f"contains IDs outside [0, {num_buckets}): "
                                f"min={minimum}, max={maximum}"
                            )
                        if invalid:
                            valid = (flat >= 0) & (flat < num_buckets)
                            flat = np.where(valid, flat, padding_id)
                    if (
                        int(window_lengths.min(initial=0)) == max_length
                        and total == n_rows * max_length
                    ):
                        tensor_fields[field.name] = torch.from_numpy(
                            np.ascontiguousarray(flat.reshape(n_rows, max_length))
                        )
                        continue
                    out = (
                        np.zeros((n_rows, max_length), dtype=np.int64)
                        if padding_id == 0
                        else np.full(
                            (n_rows, max_length),
                            padding_id,
                            dtype=np.int64,
                        )
                    )
                    mask = (
                        shared_np_pad_mask
                        if shared_np_pad_mask is not None
                        else _cached_np_arange(max_length)[None, :]
                        < window_lengths[:, None]
                    )
                    out[mask] = flat
                    tensor_fields[field.name] = torch.from_numpy(out)
                    continue
            elif field.kind == "dense":
                flat = _gather_abs_windows_dense_padded(
                    values_by_unique,
                    shared_unique_idx,
                    shared_abs_lo,
                    shared_abs_hi,
                    max_length=max_length,
                    dimension=int(field.dimension),
                    window_lengths=shared_window_lengths,
                    pad_mask=shared_np_pad_mask,
                    gather_plan=shared_gather_plan,
                )
                if use_direct_shapes and int(field.dimension) == 1:
                    tensor_fields[field.name] = flat
                elif int(field.dimension) == 1:
                    tensor_fields[field.name] = flat.unsqueeze(-1)
                else:
                    tensor_fields[field.name] = flat
                continue

        if (
            batch_name == "SequenceColumnBatch"
            and shared_abs_lo is not None
            and shared_abs_hi is not None
            and shared_unique_idx is not None
            and primary_batch is not None
            and rows.slots is primary_batch.slots
            and rows.column_index is primary_batch.column_index
        ):
            columns = rows.columns
            sample_dtype = np.int64
            values_by_unique = [column.values for column in columns]
            for buffer in values_by_unique:
                if buffer.size:
                    sample_dtype = buffer.dtype
                    break
            if total_tokens == 0:
                selected = np.empty(0, dtype=sample_dtype)
            else:
                n_rows = n_plan_rows
                views = [
                    values_by_unique[int(shared_unique_idx[row_index])][
                        int(shared_abs_lo[row_index]) : int(shared_abs_hi[row_index])
                    ]
                    for row_index in range(n_rows)
                ]
                selected = np.concatenate(views)
            selected_ready = True
        elif batch_name == "SequenceColumnBatch":
            columns = rows.columns
            slots = rows.slots
            column_index = rows.column_index
            sample_dtype = None
            for probe in range(min(n_plan_rows, 8)):
                column = (
                    columns[probe]
                    if column_index is None
                    else columns[int(column_index[probe])]
                )
                if column.values.size:
                    sample_dtype = column.values.dtype
                    break
            if sample_dtype is None:
                sample_dtype = np.int64
            if all_ranges:
                out = np.empty(total_tokens, dtype=sample_dtype)
                pos = 0
                range_starts = getattr(selection_plan, "range_starts", None)
                range_ends = getattr(selection_plan, "range_ends", None)
                if range_starts is not None and range_ends is not None:
                    for row_index in range(n_plan_rows):
                        column = (
                            columns[row_index]
                            if column_index is None
                            else columns[int(column_index[row_index])]
                        )
                        base = int(column.offsets[int(slots[row_index])])
                        rel_start = int(range_starts[row_index])
                        rel_end = int(range_ends[row_index])
                        length = rel_end - rel_start
                        if length:
                            out[pos : pos + length] = column.values[
                                base + rel_start : base + rel_end
                            ]
                            pos += length
                else:
                    for row_index, indices in enumerate(selections):
                        column = (
                            columns[row_index]
                            if column_index is None
                            else columns[int(column_index[row_index])]
                        )
                        base = int(column.offsets[int(slots[row_index])])
                        rel_start = int(indices[0])
                        rel_end = int(indices[1])
                        length = rel_end - rel_start
                        if length:
                            out[pos : pos + length] = column.values[
                                base + rel_start : base + rel_end
                            ]
                            pos += length
                selected = out
            else:
                for row_index, indices in enumerate(selections):
                    column = (
                        columns[row_index]
                        if column_index is None
                        else columns[int(column_index[row_index])]
                    )
                    slot = int(slots[row_index])
                    base = int(column.offsets[slot])
                    if (
                        isinstance(indices, tuple)
                        and len(indices) == 2
                        and not isinstance(indices[0], (list, tuple, np.ndarray))
                    ):
                        rel_start = int(indices[0])
                        rel_end = int(indices[1])
                        pieces.append(column.values[base + rel_start : base + rel_end])
                        continue
                    if len(indices) == 0:
                        pieces.append(column.values[base:base])
                    else:
                        pieces.append(
                            column.values[base + np.asarray(indices, dtype=np.int64)]
                        )
                selected = (
                    np.concatenate(pieces)
                    if pieces and any(piece.size for piece in pieces)
                    else (pieces[0][:0] if pieces else np.empty(0, dtype=sample_dtype))
                )
            selected_ready = True
        else:
            iter_selections = selections
            if (
                not iter_selections
                and all_ranges
                and getattr(selection_plan, "range_starts", None) is not None
                and getattr(selection_plan, "range_ends", None) is not None
            ):
                iter_selections = tuple(
                    zip(
                        selection_plan.range_starts.tolist(),
                        selection_plan.range_ends.tolist(),
                    )
                )
            for row_index, (row, indices) in enumerate(zip(rows, iter_selections)):
                if type(row).__name__ == "SequenceColumnRef":
                    column = row.column
                    slot = int(row.slot)
                    base = int(column.offsets[slot])
                    if (
                        isinstance(indices, tuple)
                        and len(indices) == 2
                        and not isinstance(indices[0], (list, tuple, np.ndarray))
                    ):
                        rel_start = int(indices[0])
                        rel_end = int(indices[1])
                        pieces.append(column.values[base + rel_start : base + rel_end])
                        continue
                    if len(indices) == 0:
                        pieces.append(column.values[base:base])
                    else:
                        pieces.append(
                            column.values[base + np.asarray(indices, dtype=np.int64)]
                        )
                    continue

                use_pieces = False
                if row is None:
                    items: Sequence[Any] = ()
                elif isinstance(row, np.ndarray):
                    items = row
                elif isinstance(row, (list, tuple)):
                    items = row
                else:
                    raise TypeError(
                        f"sequence {sequence.name!r} field {field.name!r} row "
                        f"{row_index} must be list-valued"
                    )
                list_rows.append(items)

        if not selected_ready:
            if use_pieces:
                selected = (
                    np.concatenate(pieces)
                    if pieces and any(piece.size for piece in pieces)
                    else (pieces[0][:0] if pieces else np.empty(0, dtype=np.int64))
                )
            else:
                full_rows = True
                ndarray_rows = True
                normalized_rows: list[Sequence[Any]] = []
                resolved_indices: list[Any] = []
                for items, indices in zip(list_rows, iter_selections):
                    normalized_rows.append(items)
                    if not isinstance(items, np.ndarray):
                        ndarray_rows = False
                    if (
                        isinstance(indices, tuple)
                        and len(indices) == 2
                        and not isinstance(indices[0], (list, tuple, np.ndarray))
                    ):
                        rel_start = int(indices[0])
                        rel_end = int(indices[1])
                        resolved_indices.append(slice(rel_start, rel_end))
                        if rel_start != 0 or rel_end != len(items):
                            full_rows = False
                    else:
                        resolved_indices.append(indices)
                        item_length = len(items)
                        if len(indices) != item_length or (
                            item_length
                            and (
                                int(indices[0]) != 0
                                or int(indices[-1]) != item_length - 1
                            )
                        ):
                            full_rows = False
                if ndarray_rows:
                    built: list[np.ndarray] = []
                    for items, indices in zip(normalized_rows, resolved_indices):
                        row_arr = items  # type: ignore[assignment]
                        if full_rows:
                            built.append(row_arr)  # type: ignore[arg-type]
                        elif isinstance(indices, slice):
                            built.append(row_arr[indices])  # type: ignore[index]
                        elif len(indices) == 0:
                            built.append(row_arr[:0])  # type: ignore[index]
                        else:
                            built.append(row_arr[indices])  # type: ignore[index]
                    selected = (
                        np.concatenate(built)
                        if built and any(piece.size for piece in built)
                        else (built[0][:0] if built else np.empty(0, dtype=np.int64))
                    )
                elif full_rows:
                    selected = list(chain.from_iterable(normalized_rows))
                else:
                    selected = []
                    for items, indices in zip(normalized_rows, resolved_indices):
                        if isinstance(indices, slice):
                            selected.extend(items[indices])
                        else:
                            selected.extend(items[int(index)] for index in indices)

        if field.kind == "categorical":
            qualified = field.qualified_name(sequence.name)
            categorical_input = config.resolved.categorical_input_by_name[qualified]
            flat_values = _tensorize_python_categorical_values(
                config,
                categorical_input,
                selected,
                vocab_maps,
                validate_prehashed_nonzero=validate_prehashed_nonzero,
            )
            effective_input = _effective_categorical_input(
                config,
                categorical_input,
            )
            padding_value: int | float = int(
                getattr(effective_input.encoding, "padding_id", 0)
            )
        else:
            if field.dimension == 1:
                if isinstance(selected, np.ndarray) and selected.dtype.kind in "fiu":
                    scalar_np = selected.astype(np.float32, copy=False)
                    if use_direct_shapes:
                        flat_values = torch.from_numpy(np.ascontiguousarray(scalar_np))
                    else:
                        flat_values = torch.from_numpy(
                            np.ascontiguousarray(scalar_np).reshape(-1, 1)
                        )
                else:
                    scalar_values = [
                        0.0 if value is None else float(value) for value in selected
                    ]
                    flat_values = torch.tensor(
                        (
                            scalar_values
                            if use_direct_shapes
                            else [[value] for value in scalar_values]
                        ),
                        dtype=torch.float32,
                    )
                    if not scalar_values and not use_direct_shapes:
                        flat_values = torch.empty((0, 1), dtype=torch.float32)
                padding_value = 0.0
            else:
                dense_rows = [
                    _dense_vector(value, field.dimension) for value in selected
                ]
                flat_values = (
                    torch.tensor(dense_rows, dtype=torch.float32)
                    if dense_rows
                    else torch.empty((0, field.dimension), dtype=torch.float32)
                )
                padding_value = 0.0
        if shared_pad_indices is None and max_length > 0 and lengths.numel() > 0:
            all_full = (
                bool((lengths == max_length).all().item())
                and int(flat_values.size(0)) == int(lengths.numel()) * max_length
            )
            if not all_full:
                positions = _cached_arange(max_length).unsqueeze(0)
                shared_pad_mask = positions < lengths.unsqueeze(1)
                shared_pad_indices = (
                    (starts.unsqueeze(1) + positions)
                    .clamp(min=0, max=max(int(flat_values.size(0)) - 1, 0))
                    .reshape(-1)
                )
        tensor_fields[field.name] = _gather_padded_sequence(
            flat_values,
            starts,
            lengths,
            max_length,
            padding_value,
            flat_indices=shared_pad_indices,
            valid_mask=shared_pad_mask,
        )

    return {
        "fields": tensor_fields,
        "lengths": lengths,
        "has_sequence": lengths > 0,
    }


def _scenario_values_tensor(
    config: AppConfig,
    values: Sequence[Any] | None,
    batch_size: int,
) -> Tensor:
    """Python-axis equivalent of ``_scenario_tensor``."""

    scenario_count = len(config.scenarios.names)
    if config.scenarios.source is None:
        if scenario_count != 1:
            raise ValueError(
                "scenarios.source is required when multiple scenarios are configured"
            )
        return torch.zeros(batch_size, dtype=torch.long)
    if values is None:
        raise ValueError(
            f"missing configured scenario column {config.scenarios.source!r}"
        )

    scenario_to_id = {name: index for index, name in enumerate(config.scenarios.names)}
    row_indices: list[list[int]] = []
    saw_list_value = False
    for row_index, value in enumerate(values):
        if isinstance(value, (list, tuple)):
            saw_list_value = True
            if not value:
                raise ValueError(f"scenario list is empty at row {row_index}")
            items = value
        else:
            items = [value]
        row_indices.append(
            [
                _encode_scenario_item(
                    item,
                    scenario_to_id,
                    scenario_count,
                    row_index,
                    config.scenarios.source_encoding,
                )
                for item in items
            ]
        )
    if saw_list_value:
        mask = torch.zeros(batch_size, scenario_count, dtype=torch.float32)
        for row_index, indices in enumerate(row_indices):
            for index in indices:
                mask[row_index, index] = 1.0
        return mask
    return torch.tensor([indices[0] for indices in row_indices], dtype=torch.long)


def axis_batch_to_feature_batch(
    config: AppConfig,
    axis_batch: Any,
    vocab_maps: dict[str, dict[str, int]],
    require_labels: bool = True,
    include_group_id: bool = True,
    split: ParquetSplitConfig | None = None,
) -> FeatureBatch:
    """Construct a FeatureBatch directly from a packed three-axis payload."""

    if not isinstance(axis_batch, PreparedAxisBatch):
        raise TypeError(
            "axis_batch_to_feature_batch requires PreparedAxisBatch, got "
            f"{type(axis_batch).__name__}"
        )
    active_split = config.data.train if split is None else split
    validate_prehashed_nonzero = active_split.reader.validate_prehashed_nonzero
    row_indices = axis_batch.request_row_indices
    features: dict[str, Any] = {}
    bag_window_cache: dict[
        tuple[Any, ...], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    bag_column_groups: dict[int, list[np.ndarray]] = {}

    for feature in config.features:
        # Own the axis by payload presence, not context_features alone.
        # Derived request columns (coarse_scene_prior_id, …) live only in
        # request_values and are intentionally excluded from context_features.
        in_request = feature.source in axis_batch.request_values
        in_candidate = feature.source in axis_batch.candidate_values
        if not in_request and not in_candidate:
            raise ValueError(
                f"feature source {feature.source!r} missing from direct "
                "request and candidate axes"
            )
        # Broadcast metadata may appear on both axes; prefer candidate then.
        request_level = in_request and not in_candidate
        source_values = (
            axis_batch.request_values if request_level else axis_batch.candidate_values
        )
        values = source_values[feature.source]
        if feature.kind == "categorical":
            if feature.pooling == "mean":
                column_groups = None
                if type(values).__name__ == "SequenceColumnBatch":
                    column_index = values.column_index
                    if column_index is not None:
                        key = id(column_index)
                        column_groups = bag_column_groups.get(key)
                        if column_groups is None:
                            n_unique = len(values.columns)
                            column_groups = [
                                np.flatnonzero(column_index == unique_idx)
                                for unique_idx in range(n_unique)
                            ]
                            bag_column_groups[key] = column_groups
                value = _tensorize_python_categorical_bag(
                    config,
                    feature,
                    values,
                    vocab_maps,
                    validate_prehashed_nonzero=validate_prehashed_nonzero,
                    column_groups=column_groups,
                    window_cache=bag_window_cache,
                )
            else:
                value = _tensorize_python_categorical_values(
                    config,
                    config.resolved.categorical_input_by_name[feature.name],
                    values,
                    vocab_maps,
                    validate_prehashed_nonzero=validate_prehashed_nonzero,
                )
        elif feature.kind == "dense":
            value = _tensorize_dense(feature, list(values))
        else:
            raise ValueError(f"unsupported feature kind {feature.kind!r}")
        features[feature.name] = (
            _indexed_request_value(value, row_indices) if request_level else value
        )

    for sequence in config.sequences:
        try:
            selection_plan = axis_batch.sequence_plans[sequence.name]
        except KeyError as error:
            raise ValueError(
                f"sequence plan {sequence.name!r} missing from direct batch"
            ) from error
        value = _tensorize_axis_sequence(
            config,
            sequence,
            axis_batch.request_values,
            selection_plan,
            vocab_maps,
            validate_prehashed_nonzero=validate_prehashed_nonzero,
        )
        value["row_indices"] = row_indices
        features[sequence.name] = value

    labels = None
    label_mask = None
    label_columns = active_split.labels
    if label_columns and all(
        column in axis_batch.candidate_values for column in label_columns.values()
    ):
        label_names = list(label_columns)
        label_tensors: list[Tensor] = []
        for name in label_names:
            raw = axis_batch.candidate_values[label_columns[name]]
            if isinstance(raw, np.ndarray):
                arr = raw.astype(np.float32, copy=False)
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32, copy=False)
                label_tensors.append(torch.from_numpy(np.ascontiguousarray(arr)))
            else:
                label_tensors.append(
                    torch.tensor(
                        [0.0 if value is None else float(value) for value in raw],
                        dtype=torch.float32,
                    )
                )
        labels = torch.stack(label_tensors, dim=1)
        mask_columns = active_split.label_masks
        mask_column_names = [mask_columns.get(name) for name in label_names]
        if mask_columns and all(
            column is not None and column in axis_batch.candidate_values
            for column in mask_column_names
        ):
            mask_tensors: list[Tensor] = []
            for column in mask_column_names:
                if column is None:
                    continue
                raw = axis_batch.candidate_values[column]
                if isinstance(raw, np.ndarray):
                    arr = raw.astype(np.float32, copy=False)
                    mask_tensors.append(torch.from_numpy(np.ascontiguousarray(arr)))
                else:
                    mask_tensors.append(
                        torch.tensor(
                            [0.0 if value is None else float(value) for value in raw],
                            dtype=torch.float32,
                        )
                    )
            label_mask = torch.stack(mask_tensors, dim=1)
    elif require_labels:
        raise ValueError("required label columns are missing from direct batch")

    group_ids: list[str] = []
    if include_group_id:
        group_source = active_split.group_id or active_split.request_id
        if group_source is None:
            group_ids = ["" for _ in range(axis_batch.n_candidates)]
        elif group_source not in axis_batch.candidate_values:
            raise ValueError(f"missing configured group-id column {group_source!r}")
        else:
            group_ids = [
                "" if value is None else str(value)
                for value in axis_batch.candidate_values[group_source]
            ]

    prediction_keys: dict[str, list[Any]] = {}
    for output_name, source in active_split.prediction_keys.items():
        if source not in axis_batch.candidate_values:
            raise ValueError(
                f"prediction key source {source!r} missing from direct "
                "candidate axis"
            )
        values = list(axis_batch.candidate_values[source])
        if len(values) != axis_batch.n_candidates:
            raise RuntimeError(
                f"prediction key source {source!r} produced {len(values)} "
                f"values for {axis_batch.n_candidates} rows"
            )
        prediction_keys[output_name] = values

    scenario_values = (
        None
        if config.scenarios.source is None
        else axis_batch.candidate_values.get(config.scenarios.source)
    )
    return FeatureBatch(
        features=features,
        labels=labels,
        label_mask=label_mask,
        scenario_id=_scenario_values_tensor(
            config,
            scenario_values,
            axis_batch.n_candidates,
        ),
        group_id=group_ids,
        prediction_keys=prediction_keys,
    )


_REQUEST_DEDUP_AUTO = object()


def table_to_feature_batch(
    config: AppConfig,
    table: Any,
    vocab_maps: dict[str, dict[str, int]],
    require_labels: bool = True,
    include_group_id: bool = True,
    split: ParquetSplitConfig | None = None,
    request_deduplication: tuple[Any, Tensor] | None | object = _REQUEST_DEDUP_AUTO,
) -> FeatureBatch:
    """Convert one Arrow table into the exact structure consumed by the model.

    Labels are optional for inference. When labels exist but explicit masks do
    not, every label is treated as observed. Feature and label ordering follows
    configuration order so it remains stable across training and evaluation.
    Callers processing a non-training split must pass it explicitly.

    ``request_deduplication`` may supply a precomputed
    ``(request_table, row_indices)`` pair from the direct pack path. Pass
    ``None`` to disable dedup for this call.
    """
    active_split = config.data.train if split is None else split
    batch_size = table.num_rows
    adapter_options = (
        {} if active_split.adapter is None else active_split.adapter.options
    )
    request_level_sources = _adapter_request_level_sources(adapter_options)
    if request_deduplication is _REQUEST_DEDUP_AUTO:
        request_take_columns: list[str] | None = None
        if active_split.reader.deduplicate_request_features:
            sequence_sources = {
                field.source
                for sequence in config.sequences
                for field in sequence.fields
            }
            request_take_columns = sorted(
                {
                    *(
                        [active_split.request_id]
                        if active_split.request_id is not None
                        else []
                    ),
                    *request_level_sources,
                    *sequence_sources,
                }
            )
        deduplication = _request_deduplication_plan(
            active_split,
            table,
            columns=request_take_columns,
        )
        request_table, request_row_indices = (
            (table, None) if deduplication is None else deduplication
        )
    elif request_deduplication is None:
        request_table, request_row_indices = table, None
    else:
        request_table, request_row_indices = request_deduplication

    validate_prehashed_nonzero = active_split.reader.validate_prehashed_nonzero
    validate_sequence_alignment = not active_split.reader.trusted_input
    features: dict[str, Any] = {}
    for feature in config.features:
        request_level = (
            request_row_indices is not None and feature.source in request_level_sources
        )
        source_table = request_table if request_level else table
        if feature.kind == "categorical":
            value = (
                _tensorize_categorical_bag(
                    config,
                    feature,
                    source_table,
                    vocab_maps,
                    validate_prehashed_nonzero=validate_prehashed_nonzero,
                )
                if feature.pooling == "mean"
                else _tensorize_categorical(
                    config,
                    feature,
                    source_table,
                    vocab_maps,
                    validate_prehashed_nonzero=validate_prehashed_nonzero,
                )
            )
        elif feature.kind == "dense":
            value = _tensorize_dense_column(feature, source_table)
        else:
            raise ValueError(f"unsupported feature kind {feature.kind!r}")
        features[feature.name] = (
            _indexed_request_value(value, request_row_indices)
            if request_level and request_row_indices is not None
            else value
        )
    for sequence in config.sequences:
        value = _tensorize_multi_field_sequence(
            config,
            sequence,
            request_table if request_row_indices is not None else table,
            vocab_maps,
            validate_prehashed_nonzero=validate_prehashed_nonzero,
            validate_sequence_alignment=validate_sequence_alignment,
        )
        if request_row_indices is not None:
            value["row_indices"] = request_row_indices
        features[sequence.name] = value

    # Labels and masks follow config order. Complete-label paths keep
    # ``label_mask=None`` so no [batch, task] all-ones tensor is allocated,
    # pinned, copied to the device, or multiplied into BCE.
    labels = None
    label_mask = None
    label_columns = active_split.labels
    if label_columns and all(
        column in table.column_names for column in label_columns.values()
    ):
        label_names = list(label_columns)
        labels = torch.stack(
            [
                _numeric_column_tensor(table, label_columns[name], torch.float32)
                for name in label_names
            ],
            dim=1,
        )
        mask_columns = active_split.label_masks
        mask_column_names = [mask_columns.get(name) for name in label_names]
        if mask_columns and all(
            column is not None and column in table.column_names
            for column in mask_column_names
        ):
            label_mask = torch.stack(
                [
                    _numeric_column_tensor(table, mask_columns[name], torch.float32)
                    for name in label_names
                ],
                dim=1,
            )
    elif require_labels:
        raise ValueError("required label columns are missing from batch")

    return FeatureBatch(
        features=features,
        labels=labels,
        label_mask=label_mask,
        scenario_id=_scenario_tensor(
            config,
            table,
            batch_size,
            trusted_input=active_split.reader.trusted_input,
        ),
        group_id=_group_ids(active_split, table, batch_size)
        if include_group_id
        else [],
        prediction_keys=_prediction_keys(active_split, table),
    )


# ---------------------------------------------------------------------------
# Device transfer: pin memory and move tensors to GPU
# ---------------------------------------------------------------------------


def _map_feature_value(value: Any, tensor_fn: Callable[[Tensor], Tensor]) -> Any:
    """Apply a device or memory operation recursively to nested tensor leaves."""
    if isinstance(value, dict):
        return {
            key: _map_feature_value(child, tensor_fn) for key, child in value.items()
        }
    if isinstance(value, torch.Tensor):
        return tensor_fn(value)
    return value


def _coalesce_feature_batch(
    batch: FeatureBatch,
    *,
    pin_memory: bool,
) -> FeatureBatch:
    """Copy every tensor leaf into one contiguous base buffer per dtype."""

    leaves: list[Tensor] = []
    seen_leaves: set[int] = set()

    def collect(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError("feature-batch coalescing requires CPU tensors")
            tensor_id = id(value)
            if tensor_id not in seen_leaves:
                leaves.append(value)
                seen_leaves.add(tensor_id)
        return value

    for value in batch.features.values():
        _map_feature_value(value, collect)
    for value in (batch.labels, batch.label_mask, batch.scenario_id):
        if isinstance(value, torch.Tensor):
            collect(value)

    by_dtype: dict[torch.dtype, list[Tensor]] = defaultdict(list)
    for tensor in leaves:
        by_dtype[tensor.dtype].append(tensor)

    replacements: dict[int, Tensor] = {}
    buffers: list[Tensor] = []
    for dtype, tensors in by_dtype.items():
        total = sum(tensor.numel() for tensor in tensors)
        buffer = torch.empty(total, dtype=dtype, pin_memory=pin_memory)
        buffers.append(buffer)
        buf_np = buffer.numpy()
        offset = 0
        for tensor in tensors:
            count = tensor.numel()
            src = tensor.detach().numpy().reshape(-1)
            if src.dtype != buf_np.dtype:
                src = np.asarray(src, dtype=buf_np.dtype)
            buf_np[offset : offset + count] = src
            replacements[id(tensor)] = buffer.narrow(0, offset, count).view(
                tensor.shape
            )
            offset += count

    def replace_tensor(tensor: Tensor) -> Tensor:
        return replacements[id(tensor)]

    return FeatureBatch(
        features={
            key: _map_feature_value(value, replace_tensor)
            for key, value in batch.features.items()
        },
        labels=None if batch.labels is None else replace_tensor(batch.labels),
        label_mask=(
            None if batch.label_mask is None else replace_tensor(batch.label_mask)
        ),
        scenario_id=replace_tensor(batch.scenario_id),
        group_id=batch.group_id,
        prediction_keys=batch.prediction_keys,
        _packed_buffers=tuple(buffers),
    )


def pin_feature_batch(
    batch: FeatureBatch,
    *,
    coalesce_tensors: bool = False,
) -> FeatureBatch:
    """Pin CPU tensors so CUDA transfers can use the non-blocking path.

    Host-prepare ``share`` IPC may hand back storages that are both pinned and
    ``share_memory_()``-backed. Those must still be cloned into private pinned
    pages; otherwise ``/dev/shm`` IPC files stay alive for the FeatureBatch
    lifetime and parent RSS ratchets up across steps.
    """

    if batch._packed_buffers:
        # Already coalesced (host-prepare child): copy the few base buffers into
        # fresh private pinned storage and remap views.
        pinned_buffers = tuple(
            buffer.detach().clone().pin_memory()
            if (not buffer.is_pinned() or buffer.is_shared())
            else buffer
            for buffer in batch._packed_buffers
        )
        pinned_by_dtype = {buffer.dtype: buffer for buffer in pinned_buffers}

        def pin_view(tensor: Tensor) -> Tensor:
            base = pinned_by_dtype[tensor.dtype]
            return base.as_strided(
                tensor.size(),
                tensor.stride(),
                tensor.storage_offset(),
            )

        return FeatureBatch(
            features={
                key: _map_feature_value(value, pin_view)
                for key, value in batch.features.items()
            },
            labels=None if batch.labels is None else pin_view(batch.labels),
            label_mask=(
                None if batch.label_mask is None else pin_view(batch.label_mask)
            ),
            scenario_id=pin_view(batch.scenario_id),
            group_id=batch.group_id,
            prediction_keys=batch.prediction_keys,
            _packed_buffers=pinned_buffers,
        )
    if coalesce_tensors:
        return _coalesce_feature_batch(batch, pin_memory=True)

    def pin_leaf(tensor: Tensor) -> Tensor:
        if tensor.is_shared() or not tensor.is_pinned():
            return tensor.detach().clone().pin_memory()
        return tensor

    return FeatureBatch(
        features={
            key: _map_feature_value(value, pin_leaf)
            for key, value in batch.features.items()
        },
        labels=None if batch.labels is None else pin_leaf(batch.labels),
        label_mask=None if batch.label_mask is None else pin_leaf(batch.label_mask),
        scenario_id=pin_leaf(batch.scenario_id),
        group_id=batch.group_id,
        prediction_keys=batch.prediction_keys,
    )


def privatize_shared_feature_batch(batch: FeatureBatch) -> FeatureBatch:
    """Clone torch IPC shared storages into process-private CPU memory.

    Used when host-prepare delivers ``share_memory_`` batches without pinning.
    """

    def clone_if_shared(tensor: Tensor) -> Tensor:
        return tensor.detach().clone() if tensor.is_shared() else tensor

    if batch._packed_buffers:
        if not any(buffer.is_shared() for buffer in batch._packed_buffers):
            return batch
        private_buffers = tuple(
            clone_if_shared(buffer) for buffer in batch._packed_buffers
        )
        private_by_dtype = {buffer.dtype: buffer for buffer in private_buffers}

        def private_view(tensor: Tensor) -> Tensor:
            base = private_by_dtype[tensor.dtype]
            return base.as_strided(
                tensor.size(),
                tensor.stride(),
                tensor.storage_offset(),
            )

        return FeatureBatch(
            features={
                key: _map_feature_value(value, private_view)
                for key, value in batch.features.items()
            },
            labels=None if batch.labels is None else private_view(batch.labels),
            label_mask=(
                None if batch.label_mask is None else private_view(batch.label_mask)
            ),
            scenario_id=private_view(batch.scenario_id),
            group_id=batch.group_id,
            prediction_keys=batch.prediction_keys,
            _packed_buffers=private_buffers,
        )

    def tree_shared(value: Any) -> bool:
        if isinstance(value, Tensor):
            return bool(value.is_shared())
        if isinstance(value, dict):
            return any(tree_shared(child) for child in value.values())
        return False

    if not (
        any(tree_shared(value) for value in batch.features.values())
        or (batch.labels is not None and batch.labels.is_shared())
        or (batch.label_mask is not None and batch.label_mask.is_shared())
        or batch.scenario_id.is_shared()
    ):
        return batch
    return FeatureBatch(
        features={
            key: _map_feature_value(value, clone_if_shared)
            for key, value in batch.features.items()
        },
        labels=None if batch.labels is None else clone_if_shared(batch.labels),
        label_mask=(
            None if batch.label_mask is None else clone_if_shared(batch.label_mask)
        ),
        scenario_id=clone_if_shared(batch.scenario_id),
        group_id=batch.group_id,
        prediction_keys=batch.prediction_keys,
    )


def move_feature_batch(
    batch: FeatureBatch,
    device: torch.device,
    non_blocking: bool = False,
) -> FeatureBatch:
    """Move every tensor leaf while leaving string evaluation metadata on CPU."""
    if batch._packed_buffers:
        moved_buffers = tuple(
            buffer.to(device, non_blocking=non_blocking)
            for buffer in batch._packed_buffers
        )
        moved_by_dtype = {buffer.dtype: buffer for buffer in moved_buffers}

        def move_view(tensor: Tensor) -> Tensor:
            base = moved_by_dtype[tensor.dtype]
            return base.as_strided(
                tensor.size(),
                tensor.stride(),
                tensor.storage_offset(),
            )

        return FeatureBatch(
            features={
                key: _map_feature_value(value, move_view)
                for key, value in batch.features.items()
            },
            labels=None if batch.labels is None else move_view(batch.labels),
            label_mask=(
                None if batch.label_mask is None else move_view(batch.label_mask)
            ),
            scenario_id=move_view(batch.scenario_id),
            group_id=batch.group_id,
            prediction_keys=batch.prediction_keys,
            _packed_buffers=moved_buffers,
        )
    return FeatureBatch(
        features={
            key: _map_feature_value(
                value,
                lambda tensor: tensor.to(device, non_blocking=non_blocking),
            )
            for key, value in batch.features.items()
        },
        labels=None
        if batch.labels is None
        else batch.labels.to(device, non_blocking=non_blocking),
        label_mask=(
            None
            if batch.label_mask is None
            else batch.label_mask.to(device, non_blocking=non_blocking)
        ),
        scenario_id=batch.scenario_id.to(device, non_blocking=non_blocking),
        group_id=batch.group_id,
        prediction_keys=batch.prediction_keys,
    )


# ---------------------------------------------------------------------------
# Direct agg Arrow → FeatureBatch path (merged from former src/agg_direct.py)
#
# RequestGroupBlock holds axis descriptors before shuffle/bucket/pack.
# PreparedAxisBatch / SequenceSelectionPlan feed the Arrow-free tensorizer.
# Controlled by reader.agg_direct_mode (default legacy).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectPipelineStats:
    """Host-side direct-path counters for benchmark / compare reporting."""

    peak_retained_sources: int = 0
    release_events: int = 0
    packs_observed: int = 0
    cross_source_duplicate_request_id_events: int = 0
    packs_with_cross_source_duplicate_request_ids: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "peak_retained_sources": self.peak_retained_sources,
            "release_events": self.release_events,
            "packs_observed": self.packs_observed,
            "cross_source_duplicate_request_id_events": (
                self.cross_source_duplicate_request_id_events
            ),
            "packs_with_cross_source_duplicate_request_ids": (
                self.packs_with_cross_source_duplicate_request_ids
            ),
        }


_LAST_DIRECT_PIPELINE_STATS = DirectPipelineStats()


def get_direct_pipeline_stats() -> DirectPipelineStats:
    return _LAST_DIRECT_PIPELINE_STATS


def publish_direct_pipeline_stats(stats: DirectPipelineStats) -> None:
    global _LAST_DIRECT_PIPELINE_STATS
    _LAST_DIRECT_PIPELINE_STATS = stats


def reset_direct_pipeline_stats() -> None:
    publish_direct_pipeline_stats(DirectPipelineStats())


@dataclass(frozen=True)
class RequestGroupBlock:
    """One request group as a logical view over a shared raw Arrow table.

    Grouping key is ``split.request_id`` (e.g. ``search_id``), not a raw
    ``context_indices`` position. ``representative_request_position`` records
    the first-occurrence request payload source for legacy dedup parity.
    """

    source_id: int
    raw_row_index: int
    request_id: Any
    representative_request_position: int
    candidate_positions: Any
    candidate_offset: int
    candidate_count: int
    pre_compaction_sequence_lengths: Mapping[str, int]
    effective_bucket_length: int
    stable_group_order: int
    slice_ordinal: int = 0

    def slice_candidates(self, offset: int, length: int) -> "RequestGroupBlock":
        """Return a descriptor view; does not take Arrow payload or allocate tensors."""

        if offset < 0 or length < 0:
            raise ValueError("slice_candidates offset/length must be non-negative")
        if offset + length > self.candidate_count:
            raise ValueError(
                f"slice_candidates[{offset}:{offset + length}] exceeds "
                f"candidate_count={self.candidate_count}"
            )
        return replace(
            self,
            candidate_offset=self.candidate_offset + offset,
            candidate_count=length,
            slice_ordinal=self.slice_ordinal + 1,
        )

    def active_candidate_positions(self) -> Any:
        start = self.candidate_offset
        stop = self.candidate_offset + self.candidate_count
        return self.candidate_positions[start:stop]

    @property
    def releases_source_reference(self) -> bool:
        """Whether consuming this descriptor finishes its original group.

        The registry owns one reference per *original* request group.  An
        oversized group may be emitted as several descriptor-only slices, so
        only the slice covering the tail of ``candidate_positions`` is allowed
        to release that reference.
        """

        return self.candidate_offset + self.candidate_count == len(
            self.candidate_positions
        )


@dataclass(frozen=True)
class PackedRequestPlan:
    """Final-batch request-axis plan after pack.

    One packed block → one request (identity). Multi-candidate rows that share
    a request stay inside one :class:`RequestGroupBlock`.
    """

    blocks: tuple[RequestGroupBlock, ...]
    unique_block_indices: Any
    block_to_request: Any
    candidate_to_request: Any


@dataclass(frozen=True)
class SequenceSelectionPlan:
    """Pack-time sequence selection for one UPS.

    Order is fixed: truncation window first (pre_compaction), then null_anchor
    compaction (compacted). Bucket keys must use pre_compaction lengths only.

    Contiguous kept windows are stored as ``(start, end)`` ranges; only
    null-anchor holes materialize an index ndarray. ``selections_are_ranges``
    is True when every request used a contiguous range (fast tensorize path).
    When ranges are used, ``range_starts`` / ``range_ends`` parallel arrays
    avoid per-field tuple unpacking in the gather hot loop.
    """

    sequence_name: str
    # Per-request local indices into the representative row list after compaction.
    selections: tuple[Any, ...]
    pre_compaction_lengths: Any
    compacted_lengths: Any
    token_to_request: Any
    selections_are_ranges: bool = False
    range_starts: Any = None  # np.ndarray[int64] | None
    range_ends: Any = None  # np.ndarray[int64] | None


@dataclass(frozen=True)
class PreparedAxisBatch:
    """Pack-boundary, Arrow-free payload for direct FeatureBatch construction.

    Values remain separated on request and candidate axes. Sequence plans are
    built only after shuffle/bucket/pack and are shared by every aligned field
    of the same sequence.

    Column values may be ``tuple``/``list``, dense ``ndarray``, or
    :class:`SequenceColumnBatch` (lazy CompactListColumn gather).
    """

    request_values: Mapping[str, Any]
    candidate_values: Mapping[str, Any]
    request_row_indices: Any
    sequence_plans: Mapping[str, SequenceSelectionPlan]
    n_requests: int
    n_candidates: int

    @property
    def num_rows(self) -> int:
        return self.n_candidates


@dataclass(frozen=True)
class SequenceColumnRef:
    """Lazy prepare handle: resolve ``column[slot]`` during tensorize gather."""

    column: Any
    slot: int


@dataclass(frozen=True)
class SequenceColumnBatch:
    """Per-request CompactListColumn gather without per-cell ref objects.

    ``columns`` stores unique CompactListColumn objects; ``column_index[i]``
    selects which column request ``i`` reads, and ``slots[i]`` is the row
    inside that column. Tensorize applies the shared selection plan against
    these handles in one pass.
    """

    columns: tuple[Any, ...]
    slots: Any  # np.ndarray[int64], length == n_requests
    column_index: Any = None  # np.ndarray[int16] or None => identity columns[i]

    def __len__(self) -> int:
        return int(np.asarray(self.slots).shape[0])

    def _column_at(self, index: int) -> Any:
        if self.column_index is None:
            return self.columns[index]
        return self.columns[int(self.column_index[index])]

    def __getitem__(self, index: int) -> Any:
        return self._column_at(index)[int(self.slots[index])]

    def __iter__(self):
        slots = self.slots
        for index in range(len(self)):
            yield self._column_at(index)[int(slots[index])]


def _truncation_window(
    length: int,
    max_length: int | None,
    truncation: str,
) -> tuple[int, int]:
    if max_length is None or length <= max_length:
        return 0, length
    if truncation == "tail":
        return length - max_length, length
    if truncation == "head":
        return 0, max_length
    raise ValueError(f"unsupported sequence truncation {truncation!r}")


def row_sequence_selection_after_truncate_then_compact(
    *,
    list_length: int,
    anchor_is_null: np.ndarray | None,
    max_length: int | None,
    truncation: str,
) -> tuple[Any, int, int]:
    """Return (kept_local_indices, pre_compaction_length, compacted_length).

    Matches the pack-time contract: clamp/truncate first, then drop null-anchor
    steps. ``pre_compaction_length`` is the bucket input; ``compacted_length``
    is the final FeatureBatch sequence length.

    When the kept window is contiguous (no null-anchor holes), indices are a
    ``(start, end)`` range instead of a materialized ``arange`` — tensorize
    slices the CompactListColumn buffer directly.
    """

    if list_length < 0:
        raise ValueError("list_length must be non-negative")
    start, end = _truncation_window(list_length, max_length, truncation)
    pre_compaction = end - start
    if anchor_is_null is None:
        return (start, end), pre_compaction, pre_compaction
    if anchor_is_null.shape[0] != list_length:
        raise ValueError(
            f"anchor_is_null length {anchor_is_null.shape[0]} != list_length {list_length}"
        )
    if pre_compaction == 0:
        return (start, end), 0, 0
    window_nulls = anchor_is_null[start:end]
    if not bool(window_nulls.any()):
        return (start, end), pre_compaction, pre_compaction
    kept = np.flatnonzero(~window_nulls).astype(np.int64, copy=False) + start
    return kept, pre_compaction, int(kept.size)


def _list_value_is_null_flags(array: Any, row_index: int) -> np.ndarray:
    """Boolean null flags for one list row's values (length = list length)."""

    pa, pc, _ds, _pq = _require_pyarrow()
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    offsets = array.offsets
    start = int(offsets[row_index].as_py())
    stop = int(offsets[row_index + 1].as_py())
    if stop <= start:
        return np.asarray([], dtype=bool)
    flat = array.values.slice(start, stop - start)
    return np.asarray(pc.is_null(flat).to_numpy(zero_copy_only=False), dtype=bool)


def _list_row_length_and_array(column: Any, row_index: int) -> tuple[int, Any]:
    """Return (list_length, combined list array) for one row."""

    pa, pc, _ds, _pq = _require_pyarrow()
    array = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    if pa.types.is_dictionary(array.type):
        array = pc.take(array.dictionary, array.indices)
    length = int(pc.list_value_length(array[row_index : row_index + 1])[0].as_py() or 0)
    return length, array


def build_sequence_selection_plan(
    sequence: Any,
    *,
    packed: PackedRequestPlan,
    source_tables: Mapping[int, Any],
) -> SequenceSelectionPlan:
    """Build truncate-then-compact selections for unique requests in a pack.

    Reads sequence lists from each block's representative row on its source
    table (adapted candidate-flat transitional layout).
    """

    if not sequence.fields:
        empty = np.asarray([], dtype=np.int64)
        return SequenceSelectionPlan(
            sequence_name=sequence.name,
            selections=tuple(),
            pre_compaction_lengths=empty,
            compacted_lengths=empty,
            token_to_request=empty,
        )

    anchor_field = getattr(sequence, "null_anchor_field", None)
    anchor_source = None
    if anchor_field is not None:
        for field in sequence.fields:
            if field.name == anchor_field:
                anchor_source = field.source
                break
        if anchor_source is None:
            raise ValueError(
                f"sequence {sequence.name!r} null_anchor_field {anchor_field!r} "
                "is not one of its fields"
            )

    primary_source = sequence.fields[0].source
    selections: list[np.ndarray] = []
    pre_lengths: list[int] = []
    compacted: list[int] = []
    token_to_request: list[int] = []

    for request_index, block_index in enumerate(packed.unique_block_indices):
        block = packed.blocks[int(block_index)]
        table = source_tables[block.source_id]
        row = int(block.representative_request_position)
        list_length, _primary = _list_row_length_and_array(table[primary_source], row)

        anchor_flags = None
        if anchor_source is not None:
            _length, anchor_array = _list_row_length_and_array(
                table[anchor_source], row
            )
            anchor_flags = _list_value_is_null_flags(anchor_array, row)

        kept, pre_len, compact_len = row_sequence_selection_after_truncate_then_compact(
            list_length=list_length,
            anchor_is_null=anchor_flags,
            max_length=_sequence_tensor_max_length(sequence),
            truncation=sequence.truncation,
        )
        selections.append(kept)
        pre_lengths.append(pre_len)
        compacted.append(compact_len)
        token_to_request.extend([request_index] * compact_len)

    return SequenceSelectionPlan(
        sequence_name=sequence.name,
        selections=tuple(selections),
        pre_compaction_lengths=np.asarray(pre_lengths, dtype=np.int64),
        compacted_lengths=np.asarray(compacted, dtype=np.int64),
        token_to_request=np.asarray(token_to_request, dtype=np.int64),
    )


def build_axis_sequence_selection_plan(
    sequence: Any,
    *,
    packed: PackedRequestPlan,
    bundles: Mapping[int, "AdaptedAxisBundle"],
) -> SequenceSelectionPlan:
    """Build one truncate-then-compact plan over axis-separated payloads.

    The adapter may already apply its configured UPS limit. Reapplying the
    sequence window is idempotent and makes this boundary correct for adapters
    that return the full membership selection.
    """

    if not sequence.fields:
        empty = np.asarray([], dtype=np.int64)
        return SequenceSelectionPlan(
            sequence_name=sequence.name,
            selections=tuple(),
            pre_compaction_lengths=empty,
            compacted_lengths=empty,
            token_to_request=empty,
        )

    anchor_source = None
    if sequence.null_anchor_field is not None:
        for field in sequence.fields:
            if field.name == sequence.null_anchor_field:
                anchor_source = field.source
                break
        if anchor_source is None:
            raise ValueError(
                f"sequence {sequence.name!r} null_anchor_field "
                f"{sequence.null_anchor_field!r} is not one of its fields"
            )

    primary_source = sequence.fields[0].source
    # Field-length alignment is invariant for a whole AdaptedAxisBundle (same
    # membership gather). Validate once per source instead of once per
    # (request × field) — that check dominated prepare_packed_axis_batch.
    alignment_checked_sources: set[int] = set()

    def _sequence_row_length(column: Any, slot: int) -> int:
        row_length = getattr(column, "row_length", None)
        if callable(row_length):
            return int(row_length(slot))
        row = column[slot]
        return 0 if row is None else len(row)

    def _anchor_null_mask(
        column: Any, slot: int, list_length: int
    ) -> np.ndarray | None:
        if list_length == 0:
            return np.asarray([], dtype=bool)
        values = getattr(column, "values", None)
        if values is not None and getattr(values, "dtype", None) != object:
            # Dense numeric CompactListColumn buffers cannot hold Python None.
            return None
        anchor_row = column[slot]
        if anchor_row is None:
            return np.ones(list_length, dtype=bool)
        if isinstance(anchor_row, np.ndarray):
            if anchor_row.dtype == object:
                return anchor_row == None  # noqa: E711 — element-wise None check
            return None
        return np.fromiter(
            (value is None for value in anchor_row),
            dtype=bool,
            count=list_length,
        )

    def _validate_source_alignment(bundle: "AdaptedAxisBundle", source_id: int) -> None:
        if source_id in alignment_checked_sources:
            return
        try:
            primary_column = bundle.sequence_features[primary_source]
        except KeyError as error:
            raise ValueError(
                f"sequence source {primary_source!r} missing from axis bundle"
            ) from error
        primary_offsets = getattr(primary_column, "offsets", None)
        for field in sequence.fields[1:]:
            try:
                aligned_column = bundle.sequence_features[field.source]
            except KeyError as error:
                raise ValueError(
                    f"sequence source {field.source!r} missing from axis bundle"
                ) from error
            aligned_offsets = getattr(aligned_column, "offsets", None)
            if (
                primary_offsets is not None
                and aligned_offsets is not None
                and (
                    primary_offsets is aligned_offsets
                    or np.array_equal(primary_offsets, aligned_offsets)
                )
            ):
                continue
            for slot in range(bundle.n_requests):
                primary_length = _sequence_row_length(primary_column, slot)
                aligned_length = _sequence_row_length(aligned_column, slot)
                if aligned_length != primary_length:
                    raise ValueError(
                        f"sequence {sequence.name!r} field {field.name!r} has length "
                        f"{aligned_length}, expected {primary_length} for request "
                        f"slot {slot} on source {source_id}"
                    )
        alignment_checked_sources.add(source_id)

    n_plan_requests = len(packed.unique_block_indices)
    range_starts_arr = np.empty(n_plan_requests, dtype=np.int64)
    range_ends_arr = np.empty(n_plan_requests, dtype=np.int64)
    pre_lengths_arr = np.empty(n_plan_requests, dtype=np.int64)
    compacted_lengths_arr = np.empty(n_plan_requests, dtype=np.int64)
    max_length = _sequence_tensor_max_length(sequence)
    truncation = sequence.truncation

    source_ids_arr = np.empty(n_plan_requests, dtype=np.int64)
    request_slots_arr = np.empty(n_plan_requests, dtype=np.int64)
    for request_index, block_index in enumerate(packed.unique_block_indices):
        block = packed.blocks[int(block_index)]
        source_ids_arr[request_index] = int(block.source_id)
        request_slots_arr[request_index] = int(block.representative_request_position)
        _validate_source_alignment(bundles[block.source_id], block.source_id)

    # Dense numeric anchors cannot store None → truncate-only, fully vectorizable.
    anchor_dense_numeric = True
    if anchor_source is not None and n_plan_requests:
        for source_id in np.unique(source_ids_arr):
            try:
                anchor_column = bundles[int(source_id)].sequence_features[anchor_source]
            except KeyError as error:
                raise ValueError(
                    f"sequence source {anchor_source!r} missing from axis bundle"
                ) from error
            values = getattr(anchor_column, "values", None)
            if values is None or getattr(values, "dtype", None) == object:
                anchor_dense_numeric = False
                break
    elif anchor_source is None:
        anchor_dense_numeric = True
    else:
        anchor_dense_numeric = True

    if anchor_dense_numeric and n_plan_requests:
        list_lengths = np.empty(n_plan_requests, dtype=np.int64)
        for source_id in np.unique(source_ids_arr):
            reqs = np.flatnonzero(source_ids_arr == source_id)
            try:
                primary_column = bundles[int(source_id)].sequence_features[
                    primary_source
                ]
            except KeyError as error:
                raise ValueError(
                    f"sequence source {primary_source!r} missing from axis bundle"
                ) from error
            offsets = getattr(primary_column, "offsets", None)
            if offsets is None:
                for out_index, slot in zip(reqs, request_slots_arr[reqs]):
                    list_lengths[int(out_index)] = _sequence_row_length(
                        primary_column, int(slot)
                    )
                continue
            row_slots = request_slots_arr[reqs]
            list_lengths[reqs] = offsets[row_slots + 1] - offsets[row_slots]

        if max_length is None:
            range_starts_arr.fill(0)
            range_ends_arr[:] = list_lengths
        elif truncation == "head":
            range_starts_arr.fill(0)
            range_ends_arr[:] = np.minimum(list_lengths, int(max_length))
        elif truncation == "tail":
            range_ends_arr[:] = list_lengths
            range_starts_arr[:] = np.maximum(list_lengths - int(max_length), 0)
        else:
            raise ValueError(f"unsupported sequence truncation {truncation!r}")
        pre_lengths_arr[:] = range_ends_arr - range_starts_arr
        compacted_lengths_arr[:] = pre_lengths_arr

        has_expected = False
        for block_index in packed.unique_block_indices:
            if (
                sequence.name
                in packed.blocks[int(block_index)].pre_compaction_sequence_lengths
            ):
                has_expected = True
                break
        if has_expected:
            for request_index, block_index in enumerate(packed.unique_block_indices):
                block = packed.blocks[int(block_index)]
                expected_pre_length = block.pre_compaction_sequence_lengths.get(
                    sequence.name
                )
                if expected_pre_length is not None and int(expected_pre_length) != int(
                    pre_lengths_arr[request_index]
                ):
                    raise RuntimeError(
                        f"sequence {sequence.name!r} pre-compaction length changed "
                        f"between bucket and pack for request {block.request_id!r}: "
                        f"bucket={expected_pre_length}, pack={int(pre_lengths_arr[request_index])}"
                    )

        # Direct tensorize uses range_starts/ends; skip materializing per-request
        # selection tuples and token_to_request (only tests assert the latter).
        return SequenceSelectionPlan(
            sequence_name=sequence.name,
            selections=tuple(),
            pre_compaction_lengths=pre_lengths_arr,
            compacted_lengths=compacted_lengths_arr,
            token_to_request=np.empty(0, dtype=np.int64),
            selections_are_ranges=True,
            range_starts=range_starts_arr,
            range_ends=range_ends_arr,
        )

    selections: list[Any] = []
    saw_sparse_selection = False
    for request_index, block_index in enumerate(packed.unique_block_indices):
        block = packed.blocks[int(block_index)]
        bundle = bundles[block.source_id]
        request_slot = int(request_slots_arr[request_index])
        try:
            primary_column = bundle.sequence_features[primary_source]
        except KeyError as error:
            raise ValueError(
                f"sequence source {primary_source!r} missing from axis bundle"
            ) from error
        list_length = _sequence_row_length(primary_column, request_slot)

        anchor_is_null = None
        if anchor_source is not None:
            try:
                anchor_column = bundle.sequence_features[anchor_source]
            except KeyError as error:
                raise ValueError(
                    f"sequence source {anchor_source!r} missing from axis bundle"
                ) from error
            anchor_is_null = _anchor_null_mask(anchor_column, request_slot, list_length)

        (
            kept,
            pre_length,
            compacted_length,
        ) = row_sequence_selection_after_truncate_then_compact(
            list_length=list_length,
            anchor_is_null=anchor_is_null,
            max_length=max_length,
            truncation=truncation,
        )
        expected_pre_length = block.pre_compaction_sequence_lengths.get(sequence.name)
        if expected_pre_length is not None and int(expected_pre_length) != pre_length:
            raise RuntimeError(
                f"sequence {sequence.name!r} pre-compaction length changed "
                f"between bucket and pack for request {block.request_id!r}: "
                f"bucket={expected_pre_length}, pack={pre_length}"
            )
        if (
            isinstance(kept, tuple)
            and len(kept) == 2
            and not isinstance(kept[0], (list, tuple, np.ndarray))
        ):
            range_starts_arr[request_index] = int(kept[0])
            range_ends_arr[request_index] = int(kept[1])
        else:
            saw_sparse_selection = True
            range_starts_arr[request_index] = 0
            range_ends_arr[request_index] = 0
        selections.append(kept)
        pre_lengths_arr[request_index] = pre_length
        compacted_lengths_arr[request_index] = compacted_length

    selections_tuple = tuple(selections)
    selections_are_ranges = not saw_sparse_selection
    if n_plan_requests and int(compacted_lengths_arr.sum()):
        token_to_request_arr = np.repeat(
            np.arange(n_plan_requests, dtype=np.int64),
            compacted_lengths_arr,
        )
    else:
        token_to_request_arr = np.empty(0, dtype=np.int64)
    return SequenceSelectionPlan(
        sequence_name=sequence.name,
        selections=selections_tuple,
        pre_compaction_lengths=pre_lengths_arr,
        compacted_lengths=compacted_lengths_arr,
        token_to_request=token_to_request_arr,
        selections_are_ranges=selections_are_ranges,
        range_starts=range_starts_arr if selections_are_ranges else None,
        range_ends=range_ends_arr if selections_are_ranges else None,
    )


def table_pre_compaction_sequence_lengths(
    sequences: Sequence[Any],
    table: Any,
) -> dict[str, np.ndarray]:
    """Per-row list lengths after ``max_length`` clamp; no null_anchor filter.

    Matches ``train._table_sequence_lengths`` (bucket metric input).
    """

    pa, pc, _ds, _pq = _require_pyarrow()
    result: dict[str, np.ndarray] = {}
    for sequence in sequences:
        if not sequence.fields:
            continue
        source = sequence.fields[0].source
        array = table[source].combine_chunks()
        if pa.types.is_dictionary(array.type):
            dictionary_lengths = pc.list_value_length(array.dictionary)
            lengths = pc.take(dictionary_lengths, array.indices)
        else:
            lengths = pc.list_value_length(array)
        if lengths.null_count:
            lengths = pc.fill_null(lengths, 0)
        values = lengths.to_numpy(zero_copy_only=False).astype(np.int64, copy=True)
        tensor_max_length = _sequence_tensor_max_length(sequence)
        if tensor_max_length is not None:
            np.minimum(values, int(tensor_max_length), out=values)
        result[sequence.name] = values
    return result


def effective_bucket_length_from_pre_compaction(
    lengths: Mapping[str, int],
    *,
    metric: str = "max",
) -> int:
    """Collapse per-UPS pre-compaction lengths into the configured bucket key."""

    if not lengths:
        return 0
    values = list(lengths.values())
    if metric == "sum":
        return int(sum(values))
    if metric == "max":
        return int(max(values))
    raise ValueError(f"length_bucket_metric must be max or sum, got {metric!r}")


def request_group_blocks_from_adapted_table(
    table: Any,
    *,
    source_id: int,
    request_id_column: str,
    sequences: Sequence[Any] = (),
    length_bucket_metric: str = "max",
) -> tuple[RequestGroupBlock, ...]:
    """Build descriptor blocks grouped by ``request_id`` (first-occurrence order).

    Operates on an adapted candidate-flat table for transitional parity with
    ``_request_group_tables``. Same ``request_id`` with interleaved rows becomes
    one block; ``candidate_positions`` preserve adapter output order and may be
    non-contiguous. Does not take payload columns or allocate feature tensors.
    """

    if request_id_column not in table.column_names:
        raise ValueError(
            f"request_id column {request_id_column!r} missing from adapted table"
        )
    request_ids = table[request_id_column].to_pylist()
    positions_by_request: dict[Any, list[int]] = {}
    for row_index, request_id in enumerate(request_ids):
        if request_id is None:
            raise ValueError(
                f"request_id column {request_id_column!r} contains null at row {row_index}"
            )
        try:
            positions_by_request.setdefault(request_id, []).append(row_index)
        except TypeError as error:
            raise ValueError(
                f"request_id column {request_id_column!r} must contain hashable scalars"
            ) from error

    per_sequence = table_pre_compaction_sequence_lengths(sequences, table)
    blocks: list[RequestGroupBlock] = []
    for stable_group_order, (request_id, positions) in enumerate(
        positions_by_request.items()
    ):
        representative = positions[0]
        pre_compaction = {
            name: int(values[representative]) for name, values in per_sequence.items()
        }
        candidate_positions = np.asarray(positions, dtype=np.int64)
        blocks.append(
            RequestGroupBlock(
                source_id=source_id,
                raw_row_index=representative,
                request_id=request_id,
                representative_request_position=representative,
                candidate_positions=candidate_positions,
                candidate_offset=0,
                candidate_count=len(positions),
                pre_compaction_sequence_lengths=pre_compaction,
                effective_bucket_length=effective_bucket_length_from_pre_compaction(
                    pre_compaction,
                    metric=length_bucket_metric,
                ),
                stable_group_order=stable_group_order,
            )
        )
    return tuple(blocks)


def build_packed_request_plan(
    blocks: Sequence[RequestGroupBlock],
) -> PackedRequestPlan:
    """Identity request plan: one block → one request row in the packed batch."""

    block_tuple = tuple(blocks)
    n_blocks = len(block_tuple)
    unique_block_indices = np.arange(n_blocks, dtype=np.int64)
    block_to_request = np.arange(n_blocks, dtype=np.int64)
    if n_blocks == 0:
        candidate_to_request = np.asarray([], dtype=np.int64)
    else:
        candidate_to_request = np.repeat(
            block_to_request,
            [block.candidate_count for block in block_tuple],
        )
    return PackedRequestPlan(
        blocks=block_tuple,
        unique_block_indices=unique_block_indices,
        block_to_request=block_to_request,
        candidate_to_request=candidate_to_request,
    )


def _shuffle_blocks(
    blocks: list[RequestGroupBlock],
    generator: torch.Generator,
) -> list[RequestGroupBlock]:
    if len(blocks) <= 1:
        return blocks
    permutation = torch.randperm(len(blocks), generator=generator).tolist()
    return [blocks[index] for index in permutation]


def iter_shuffled_request_groups(
    blocks: Iterator[RequestGroupBlock],
    *,
    shuffle_buffer_rows: int,
    shuffle_seed: int,
    shard_rank: int = 0,
) -> Iterator[RequestGroupBlock]:
    """Bounded deterministic shuffle; groups stay intact (candidate-row buffer).

    ``shuffle_buffer_rows == 0`` consumes no RNG and yields source order.
    """

    if shuffle_buffer_rows < 0:
        raise ValueError("shuffle_buffer_rows must be non-negative")
    if shuffle_buffer_rows == 0:
        yield from blocks
        return

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(shuffle_seed) + int(shard_rank))
    buffered: list[RequestGroupBlock] = []
    buffered_rows = 0
    for block in blocks:
        if block.candidate_count > shuffle_buffer_rows:
            yield from _shuffle_blocks(buffered, generator)
            buffered = []
            buffered_rows = 0
            yield block
            continue
        while buffered and buffered_rows + block.candidate_count > shuffle_buffer_rows:
            selected_index = int(
                torch.randint(len(buffered), (), generator=generator).item()
            )
            selected = buffered[selected_index]
            buffered[selected_index] = buffered[-1]
            buffered.pop()
            buffered_rows -= selected.candidate_count
            yield selected
        buffered.append(block)
        buffered_rows += block.candidate_count
    yield from _shuffle_blocks(buffered, generator)


def iter_packed_request_groups(
    blocks: Iterator[RequestGroupBlock],
    *,
    batch_size: int,
) -> Iterator[tuple[RequestGroupBlock, ...]]:
    """Pack request groups without splitting unless one group exceeds capacity."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    buffered: list[RequestGroupBlock] = []
    buffered_rows = 0
    for original in blocks:
        block = original
        while block.candidate_count > batch_size:
            if buffered_rows:
                yield tuple(buffered)
                buffered = []
                buffered_rows = 0
            yield (block.slice_candidates(0, batch_size),)
            block = block.slice_candidates(
                batch_size, block.candidate_count - batch_size
            )
        if not block.candidate_count:
            continue
        if buffered_rows and buffered_rows + block.candidate_count > batch_size:
            yield tuple(buffered)
            buffered = []
            buffered_rows = 0
        buffered.append(block)
        buffered_rows += block.candidate_count
        if buffered_rows == batch_size:
            yield tuple(buffered)
            buffered = []
            buffered_rows = 0
    if buffered_rows:
        yield tuple(buffered)


def length_bucket_index(effective_length: int, finite_boundaries: Sequence[int]) -> int:
    """Bisect into ``length_buckets`` using the same rule as train.py."""

    # bisect_left over finite max_length boundaries; catch-all is last.
    lo = 0
    hi = len(finite_boundaries)
    while lo < hi:
        mid = (lo + hi) // 2
        if finite_boundaries[mid] < effective_length:
            lo = mid + 1
        else:
            hi = mid
    return lo


def iter_length_bucketed_packs(
    blocks: Iterator[RequestGroupBlock],
    *,
    buckets: Sequence[Any],
    default_batch_size: int,
    shuffle_buffer_rows: int = 0,
    shuffle_seed: int = 0,
    shard_rank: int = 0,
) -> Iterator[tuple[RequestGroupBlock, ...]]:
    """Shuffle then pack by sequence-length bucket (request groups preserved).

    ``buckets`` entries expose ``max_length`` / ``batch_size`` like
    ``LengthBucketConfig``. Empty ``buckets`` packs with ``default_batch_size``.
    """

    shuffled = iter_shuffled_request_groups(
        blocks,
        shuffle_buffer_rows=shuffle_buffer_rows,
        shuffle_seed=shuffle_seed,
        shard_rank=shard_rank,
    )
    if not buckets:
        yield from iter_packed_request_groups(shuffled, batch_size=default_batch_size)
        return

    finite_boundaries = [
        int(bucket.max_length) for bucket in buckets if bucket.max_length is not None
    ]
    buffered: list[list[RequestGroupBlock]] = [[] for _ in buckets]
    buffered_rows = [0] * len(buckets)

    for block in shuffled:
        bucket_index = length_bucket_index(
            int(block.effective_bucket_length),
            finite_boundaries,
        )
        bucket = buckets[bucket_index]
        capacity = int(bucket.batch_size)
        remaining = block
        while remaining.candidate_count > capacity:
            if buffered_rows[bucket_index]:
                yield tuple(buffered[bucket_index])
                buffered[bucket_index] = []
                buffered_rows[bucket_index] = 0
            yield (remaining.slice_candidates(0, capacity),)
            remaining = remaining.slice_candidates(
                capacity,
                remaining.candidate_count - capacity,
            )
        if not remaining.candidate_count:
            continue
        if (
            buffered_rows[bucket_index]
            and buffered_rows[bucket_index] + remaining.candidate_count > capacity
        ):
            yield tuple(buffered[bucket_index])
            buffered[bucket_index] = []
            buffered_rows[bucket_index] = 0
        buffered[bucket_index].append(remaining)
        buffered_rows[bucket_index] += remaining.candidate_count
        if buffered_rows[bucket_index] == capacity:
            yield tuple(buffered[bucket_index])
            buffered[bucket_index] = []
            buffered_rows[bucket_index] = 0

    for bucket_index in range(len(buckets)):
        if buffered_rows[bucket_index]:
            yield tuple(buffered[bucket_index])


def build_request_deduplication_from_pack(
    packed: PackedRequestPlan,
    source_tables: Mapping[int, Any],
    *,
    columns: Sequence[str],
) -> tuple[Any, Any]:
    """Take one representative request row per packed block (identity dedup)."""

    pa, _pc, _ds, _pq = _require_pyarrow()
    if not packed.blocks:
        raise ValueError("cannot build request deduplication from an empty pack")
    pieces: list[Any] = []
    for block_index in packed.unique_block_indices:
        block = packed.blocks[int(block_index)]
        table = source_tables[block.source_id]
        available = [name for name in columns if name in table.column_names]
        if not available:
            raise ValueError(
                "request deduplication projected an empty column set for packed blocks"
            )
        row = int(block.representative_request_position)
        pieces.append(table.select(available).slice(row, 1))
    request_table = pieces[0] if len(pieces) == 1 else pa.concat_tables(pieces)
    row_indices = torch.as_tensor(packed.candidate_to_request, dtype=torch.long)
    return request_table, row_indices


@dataclass(frozen=True)
class CompactListColumn:
    """Columnar ragged lists for one feature across requests/candidates.

    Stores a single flat NumPy values buffer plus offsets so ProcessPool IPC
    pickles ~one array per feature instead of tens of thousands of tiny Python
    lists (sequence features dominate AdaptedAxisBundle pickle cost).
    ``column[i]`` returns a NumPy view into ``values`` (empty when the row is
    empty). Pack/tensorize accept ndarray and avoid a Python list materialization
    on every gather.
    """

    values: Any
    offsets: Any

    def __len__(self) -> int:
        return int(self.offsets.shape[0]) - 1

    def __getitem__(self, index: int) -> Any:
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        return self.values[start:stop]

    def row_length(self, index: int) -> int:
        return int(self.offsets[index + 1] - self.offsets[index])


def compact_list_column_from_rows(rows: Sequence[Any]) -> CompactListColumn:
    """Pack a sequence of row lists/ndarrays into :class:`CompactListColumn`."""

    n_rows = len(rows)
    offsets = np.empty(n_rows + 1, dtype=np.int64)
    offsets[0] = 0
    total = 0
    sample = None
    saw_none = False
    numeric_nd = True
    sample_dtype: Any = None
    for index, row in enumerate(rows):
        if row is None:
            length = 0
        elif isinstance(row, np.ndarray):
            length = int(row.shape[0])
            if length:
                if row.dtype == object:
                    numeric_nd = False
                    # Must detect nulls on every object row (a later None after
                    # sample is set used to pick int64 and TypeError on assign).
                    # Vectorized equality beats a Python element loop here.
                    if not saw_none and bool(np.equal(row, None).any()):
                        saw_none = True
                    if sample is None:
                        for item in row:
                            if item is not None:
                                sample = item
                                break
                elif sample_dtype is None:
                    sample_dtype = row.dtype
                    sample = row.item(0) if length else sample
        else:
            numeric_nd = False
            length = len(row)
            # ``None in seq`` is a C-level scan for list/tuple.
            if not saw_none and None in row:
                saw_none = True
            if sample is None:
                for item in row:
                    if item is not None:
                        sample = item
                        break
        total += length
        offsets[index + 1] = total

    if total == 0:
        return CompactListColumn(
            values=np.empty(0, dtype=np.int64),
            offsets=offsets,
        )

    if numeric_nd and sample_dtype is not None and sample_dtype != object:
        values = np.empty(total, dtype=sample_dtype)
        cursor = 0
        for row in rows:
            if isinstance(row, np.ndarray):
                length = int(row.shape[0])
                if length:
                    values[cursor : cursor + length] = row
                    cursor += length
        return CompactListColumn(values=values, offsets=offsets)

    if saw_none or sample is None:
        dtype: Any = object
    elif isinstance(sample, (bool, np.bool_)):
        dtype = object
    elif isinstance(sample, (int, np.integer)):
        dtype = np.int64
    elif isinstance(sample, (float, np.floating)):
        dtype = np.float64
    else:
        dtype = object
    if dtype is object:
        values = np.empty(total, dtype=object)
    else:
        values = np.empty(total, dtype=dtype)
    for index, row in enumerate(rows):
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        if stop > start:
            values[start:stop] = row
    return CompactListColumn(values=values, offsets=offsets)


def axis_feature_column_from_values(values: Sequence[Any]) -> Any:
    """Pack one axis feature into a NumPy column or :class:`CompactListColumn`.

    Scalar request/item/label columns become dense ``ndarray``; ragged bag
    columns reuse CompactListColumn. Both pickle far cheaper than tuples of
    Python ints/lists across ProcessPool IPC.
    """

    n_rows = len(values)
    if n_rows == 0:
        return np.empty(0, dtype=np.int64)
    sample = None
    for value in values:
        if value is not None:
            sample = value
            break
    if sample is None:
        return np.asarray(values, dtype=object)
    if isinstance(sample, (list, tuple, np.ndarray)):
        return compact_list_column_from_rows(values)
    if isinstance(sample, (bool, np.bool_)):
        return np.asarray(values, dtype=object)
    if isinstance(sample, (int, np.integer)):
        if any(value is None for value in values):
            return np.asarray(values, dtype=object)
        return np.asarray(values, dtype=np.int64)
    if isinstance(sample, (float, np.floating)):
        return np.asarray(values, dtype=np.float64)
    return np.asarray(values, dtype=object)


def share_compact_list_offsets(
    columns: Mapping[str, CompactListColumn],
) -> dict[str, CompactListColumn]:
    """Reuse identical offsets buffers across columns (same UPS membership).

    Adapt builds one CompactListColumn per sequence field; fields from the same
    membership gather share row lengths. Sharing the offsets array makes pack-time
    alignment checks an identity compare instead of ``np.array_equal``.
    """

    unique_offsets: list[Any] = []
    shared: dict[str, CompactListColumn] = {}
    for name, column in columns.items():
        matched = None
        offsets = column.offsets
        for candidate in unique_offsets:
            if candidate.shape == offsets.shape and np.array_equal(candidate, offsets):
                matched = candidate
                break
        if matched is None:
            unique_offsets.append(offsets)
            matched = offsets
        if matched is offsets:
            shared[name] = column
        else:
            shared[name] = CompactListColumn(
                values=column.values,
                offsets=matched,
            )
    return shared


@dataclass(frozen=True)
class AdaptedAxisBundle:
    """Adapted scanner payload without candidate-flat Arrow.

    Request/sequence values are stored once per unique ``request_id`` (first
    occurrence wins, matching legacy dedup). Item/label/metadata values are
    stored once per candidate. ``candidate_to_request`` is the FeatureBatch
    ``row_indices`` vector for this source.

    Feature columns may be dense ``ndarray`` or :class:`CompactListColumn` for
    cheaper ProcessPool IPC; ``column[i]`` still returns the per-row value (a
    scalar or a NumPy view for ragged rows).
    """

    n_candidates: int
    n_requests: int
    request_ids: tuple[Any, ...]
    candidate_to_request: Any
    request_features: Mapping[str, Any]
    sequence_features: Mapping[str, Any]
    item_features: Mapping[str, Any]
    label_features: Mapping[str, Any]
    label_mask_features: Mapping[str, Any]
    candidate_metadata: Mapping[str, Any]
    request_raw_rows: Any
    candidate_raw_rows: Any


class SourceRegistry:
    """Retain axis bundles (or tables) while shuffle/bucket buffers reference them.

    ``acquire`` / ``release`` are counted per retained block. When the last
    reference drops, the payload is deleted so Arrow/Python buffers can GC.
    Peak retained source count is exposed for RSS monitoring (plan phase 8).
    """

    def __init__(self) -> None:
        self._sources: dict[int, Any] = {}
        self._refcount: dict[int, int] = {}
        self._next_id = 0
        self.peak_retained_sources = 0
        self.release_events = 0
        self.packs_observed = 0
        self.cross_source_duplicate_request_id_events = 0
        self.packs_with_cross_source_duplicate_request_ids = 0

    def put(self, payload: Any) -> int:
        source_id = self._next_id
        self._next_id += 1
        self._sources[source_id] = payload
        self._refcount[source_id] = 0
        self.peak_retained_sources = max(self.peak_retained_sources, len(self._sources))
        return source_id

    def get(self, source_id: int) -> Any:
        try:
            return self._sources[source_id]
        except KeyError as error:
            raise KeyError(f"source_id {source_id} is not retained") from error

    def acquire(self, source_id: int, count: int = 1) -> None:
        if count < 0:
            raise ValueError("acquire count must be non-negative")
        if source_id not in self._sources:
            raise KeyError(f"source_id {source_id} is not retained")
        self._refcount[source_id] = self._refcount.get(source_id, 0) + count

    def release(self, source_id: int, count: int = 1) -> None:
        if count < 0:
            raise ValueError("release count must be non-negative")
        if source_id not in self._refcount:
            raise KeyError(f"source_id {source_id} has no references")
        remaining = self._refcount[source_id] - count
        if remaining > 0:
            self._refcount[source_id] = remaining
            return
        if remaining < 0:
            raise ValueError(
                f"source_id {source_id} release underflow "
                f"(held={self._refcount[source_id]}, release={count})"
            )
        del self._refcount[source_id]
        del self._sources[source_id]
        self.release_events += 1

    def clear(self) -> None:
        """Drop every retained payload (interrupted shuffle/bucket teardown)."""

        self._sources.clear()
        self._refcount.clear()

    def retained_source_ids(self) -> tuple[int, ...]:
        return tuple(self._sources.keys())

    @property
    def retained_count(self) -> int:
        return len(self._sources)

    def observe_pack(self, blocks: Sequence[RequestGroupBlock]) -> int:
        """Count cross-source duplicate ``request_id`` values in one pack.

        Production ranking logs should not repeat a request across scanned
        tables. Direct path keeps one request row per block (no cross-table
        merge); this counter surfaces contract violations without changing
        pack semantics.
        """

        self.packs_observed += 1
        sources_by_request: dict[Any, set[int]] = {}
        for block in blocks:
            try:
                sources_by_request.setdefault(block.request_id, set()).add(
                    int(block.source_id)
                )
            except TypeError:
                # Unhashable request_id is rejected earlier in the adapter.
                continue
        duplicate_events = 0
        for request_id, source_ids in sources_by_request.items():
            if len(source_ids) < 2:
                continue
            duplicate_events += 1
            logger.warning(
                "cross-source duplicate request_id=%r across source_ids=%s "
                "(direct path keeps separate request rows; legacy may dedup)",
                request_id,
                sorted(source_ids),
            )
        if duplicate_events:
            self.packs_with_cross_source_duplicate_request_ids += 1
            self.cross_source_duplicate_request_id_events += duplicate_events
        return duplicate_events

    def snapshot_stats(self) -> DirectPipelineStats:
        return DirectPipelineStats(
            peak_retained_sources=int(self.peak_retained_sources),
            release_events=int(self.release_events),
            packs_observed=int(self.packs_observed),
            cross_source_duplicate_request_id_events=int(
                self.cross_source_duplicate_request_id_events
            ),
            packs_with_cross_source_duplicate_request_ids=int(
                self.packs_with_cross_source_duplicate_request_ids
            ),
        )


def request_group_blocks_from_axis_bundle(
    bundle: AdaptedAxisBundle,
    *,
    source_id: int,
    sequences: Sequence[Any] = (),
    length_bucket_metric: str = "max",
) -> tuple[RequestGroupBlock, ...]:
    """Build descriptors from an axis-separated adapted bundle."""

    if bundle.n_requests == 0:
        return ()
    positions_by_slot: list[list[int]] = [[] for _ in range(bundle.n_requests)]
    for candidate_index, slot in enumerate(bundle.candidate_to_request):
        positions_by_slot[int(slot)].append(int(candidate_index))

    blocks: list[RequestGroupBlock] = []
    for stable_group_order, positions in enumerate(positions_by_slot):
        if not positions:
            raise ValueError(
                f"request slot {stable_group_order} has no candidates in axis bundle"
            )
        pre_compaction: dict[str, int] = {}
        for sequence in sequences:
            if not sequence.fields:
                continue
            source = sequence.fields[0].source
            if source not in bundle.sequence_features:
                raise ValueError(f"sequence source {source!r} missing from axis bundle")
            seq_column = bundle.sequence_features[source]
            if isinstance(seq_column, CompactListColumn):
                length = seq_column.row_length(stable_group_order)
            else:
                length = len(seq_column[stable_group_order])
            tensor_max_length = _sequence_tensor_max_length(sequence)
            if tensor_max_length is not None:
                length = min(length, int(tensor_max_length))
            pre_compaction[sequence.name] = int(length)
        blocks.append(
            RequestGroupBlock(
                source_id=source_id,
                raw_row_index=int(bundle.request_raw_rows[stable_group_order]),
                request_id=bundle.request_ids[stable_group_order],
                representative_request_position=stable_group_order,
                candidate_positions=np.asarray(positions, dtype=np.int64),
                candidate_offset=0,
                candidate_count=len(positions),
                pre_compaction_sequence_lengths=pre_compaction,
                effective_bucket_length=effective_bucket_length_from_pre_compaction(
                    pre_compaction,
                    metric=length_bucket_metric,
                ),
                stable_group_order=stable_group_order,
            )
        )
    return tuple(blocks)


def _arrow_array_from_python_values(values: Sequence[Any]) -> Any:
    """Build an Arrow array without collapsing empty lists to null type."""

    pa, _pc, _ds, _pq = _require_pyarrow()
    array = pa.array(list(values))
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        value_type = array.type.value_type
        if pa.types.is_null(value_type):
            # All-empty lists infer list<null>; pin a concrete value type.
            sample = next((value for value in values if value), None)
            if sample and isinstance(sample[0], float):
                return pa.array(list(values), type=pa.list_(pa.float32()))
            return pa.array(list(values), type=pa.list_(pa.int64()))
        return array
    if not pa.types.is_null(array.type):
        return array
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            return pa.array(list(values), type=pa.bool_())
        if isinstance(value, int):
            return pa.array(list(values), type=pa.int64())
        if isinstance(value, float):
            return pa.array(list(values), type=pa.float32())
        if isinstance(value, str):
            return pa.array(list(values), type=pa.string())
        if isinstance(value, (list, tuple)):
            if value and isinstance(value[0], float):
                return pa.array(list(values), type=pa.list_(pa.float32()))
            return pa.array(list(values), type=pa.list_(pa.int64()))
        break
    return pa.array(list(values), type=pa.int64())


def _is_dense_numeric_column(column: Any) -> bool:
    """True when a column can be fancy-indexed into a non-object ndarray."""

    return (
        isinstance(column, np.ndarray)
        and column.ndim == 1
        # ``kind in 'iufb'`` excludes object ('O') without a slower dtype!=object.
        and column.dtype.kind in "iufb"
    )


def _columns_share_dense_dtype(columns: Sequence[Any]) -> bool:
    """All columns dense with an identical dtype (safe for one ``np.empty`` out)."""

    if not columns:
        return False
    first = columns[0]
    if not _is_dense_numeric_column(first):
        return False
    sample_dtype = first.dtype
    for column in columns[1:]:
        if (
            not isinstance(column, np.ndarray)
            or column.ndim != 1
            or column.dtype != sample_dtype
        ):
            return False
    return True


# Candidate-column kind codes for pack classify (faster than repeated isinstance).
_PACK_COL_OBJECT = 0
_PACK_COL_DENSE = 1
_PACK_COL_LIST = 2


def _pack_column_kind(column: Any) -> tuple[int, int]:
    """Return ``(kind, dtype_num)`` for dense-compatible pack classification."""

    if isinstance(column, CompactListColumn):
        return _PACK_COL_LIST, 0
    if isinstance(column, np.ndarray) and column.ndim == 1:
        dtype = column.dtype
        if dtype.kind in "iufb":
            return _PACK_COL_DENSE, int(dtype.num)
    return _PACK_COL_OBJECT, 0


def prepare_packed_axis_batch(
    bundles: Mapping[int, AdaptedAxisBundle],
    packed: PackedRequestPlan,
    *,
    sequences: Sequence[Any],
    request_id_column: str | None = None,
    candidate_request_columns: Sequence[str] = (),
) -> PreparedAxisBatch:
    """Gather one packed batch without constructing candidate/request Arrow.

    Candidate payload is copied only as references to the already-normalized
    Python scalars/lists owned by each axis bundle. Request and sequence values
    remain unique per request; ``request_row_indices`` performs the only
    candidate-to-request broadcast required by the model.
    """

    if not packed.blocks:
        raise ValueError("cannot prepare an empty packed axis batch")

    source_ids = {block.source_id for block in packed.blocks}
    missing_sources = sorted(source_ids - set(bundles))
    if missing_sources:
        raise KeyError(f"axis bundles missing source IDs {missing_sources}")

    request_names: set[str] = set()
    candidate_names: set[str] = set()
    for source_id in source_ids:
        bundle = bundles[source_id]
        request_names.update(bundle.request_features)
        request_names.update(bundle.sequence_features)
        candidate_names.update(bundle.item_features)
        candidate_names.update(bundle.label_features)
        candidate_names.update(bundle.label_mask_features)
        candidate_names.update(bundle.candidate_metadata)
    if request_id_column is not None:
        request_names.add(request_id_column)

    broadcast_names = tuple(dict.fromkeys(candidate_request_columns))
    broadcast_set = set(broadcast_names)
    candidate_name_list = sorted(set(candidate_names) | broadcast_set)
    request_name_list = sorted(request_names)

    # Unique-request gather: resolve column once per (source, name), then index.
    request_order: list[tuple[AdaptedAxisBundle, int]] = []
    for block_index in packed.unique_block_indices:
        block = packed.blocks[int(block_index)]
        request_order.append(
            (
                bundles[block.source_id],
                int(block.representative_request_position),
            )
        )
    n_requests = len(request_order)
    shared_slots = np.fromiter(
        (slot for _bundle, slot in request_order),
        dtype=np.int64,
        count=n_requests,
    )
    # Bundle identity → compact index (shared by every sequence field).
    unique_bundles: list[AdaptedAxisBundle] = []
    bundle_id_to_index: dict[int, int] = {}
    shared_column_index = np.empty(n_requests, dtype=np.int16)
    for out_index, (bundle, _slot) in enumerate(request_order):
        key = id(bundle)
        index = bundle_id_to_index.get(key)
        if index is None:
            index = len(unique_bundles)
            bundle_id_to_index[key] = index
            unique_bundles.append(bundle)
        shared_column_index[out_index] = index

    # Contiguous same-bundle runs for vectorized ndarray fancy-index.
    bundle_groups: list[tuple[AdaptedAxisBundle, np.ndarray, np.ndarray]] = []
    if request_order:
        group_bundle = request_order[0][0]
        group_out: list[int] = []
        group_slots: list[int] = []
        for out_index, (bundle, slot) in enumerate(request_order):
            if bundle is not group_bundle:
                bundle_groups.append(
                    (
                        group_bundle,
                        np.asarray(group_out, dtype=np.int64),
                        np.asarray(group_slots, dtype=np.int64),
                    )
                )
                group_bundle = bundle
                group_out = []
                group_slots = []
            group_out.append(out_index)
            group_slots.append(slot)
        bundle_groups.append(
            (
                group_bundle,
                np.asarray(group_out, dtype=np.int64),
                np.asarray(group_slots, dtype=np.int64),
            )
        )

    request_values: dict[str, Any] = {}
    # Classify once from unique bundles instead of re-scanning request_order
    # per feature name (160+ request features × hundreds of requests).
    sequence_name_set: set[str] = set()
    request_feature_name_set: set[str] = set()
    for bundle in unique_bundles:
        sequence_name_set.update(bundle.sequence_features)
        request_feature_name_set.update(bundle.request_features)
    for name in request_name_list:
        if name in sequence_name_set:
            sample_column = None
            for bundle in unique_bundles:
                if name in bundle.sequence_features:
                    sample_column = bundle.sequence_features[name]
                    break
            if isinstance(sample_column, CompactListColumn):
                request_values[name] = SequenceColumnBatch(
                    columns=tuple(
                        bundle.sequence_features[name] for bundle in unique_bundles
                    ),
                    slots=shared_slots,
                    column_index=shared_column_index,
                )
                continue
            rows = [
                bundle.sequence_features[name][slot] for bundle, slot in request_order
            ]
            request_values[name] = tuple(rows)
            continue
        if name in request_feature_name_set:
            sample_column = None
            for bundle in unique_bundles:
                if name in bundle.request_features:
                    sample_column = bundle.request_features[name]
                    break
            if isinstance(sample_column, CompactListColumn):
                # CompactListColumn is homogeneous by construction across sources.
                request_values[name] = SequenceColumnBatch(
                    columns=tuple(
                        bundle.request_features[name] for bundle in unique_bundles
                    ),
                    slots=shared_slots,
                    column_index=shared_column_index,
                )
                continue
            # Single-bundle: first-column dtype is enough. Multi-bundle: require
            # shared dense dtype so a null object column cannot poison int64 out.
            use_dense = _is_dense_numeric_column(sample_column)
            if use_dense and len(unique_bundles) > 1:
                use_dense = _columns_share_dense_dtype(
                    [
                        bundle.request_features[name]
                        for bundle in unique_bundles
                        if name in bundle.request_features
                    ]
                )
            if use_dense:
                out = np.empty(n_requests, dtype=sample_column.dtype)
                for bundle, out_idx, slots_arr in bundle_groups:
                    out[out_idx] = bundle.request_features[name][slots_arr]
                request_values[name] = out
                continue
            rows = [
                bundle.request_features[name][slot] for bundle, slot in request_order
            ]
            request_values[name] = tuple(rows)
            continue
        if request_id_column is not None and name == request_id_column:
            sample_ids = np.asarray(unique_bundles[0].request_ids)
            use_dense = _is_dense_numeric_column(sample_ids)
            if use_dense and len(unique_bundles) > 1:
                use_dense = _columns_share_dense_dtype(
                    [np.asarray(bundle.request_ids) for bundle in unique_bundles]
                )
            if use_dense:
                out = np.empty(n_requests, dtype=sample_ids.dtype)
                for bundle, out_idx, slots_arr in bundle_groups:
                    out[out_idx] = np.asarray(bundle.request_ids)[slots_arr]
                request_values[name] = out
            else:
                request_values[name] = tuple(
                    bundle.request_ids[slot] for bundle, slot in request_order
                )
            continue
        raise KeyError(f"request column {name!r} missing from axis bundle")

    # Candidate gather: shared source index/slots for CompactListColumn bags;
    # dense scalars fancy-index+concat; broadcast/object stay per-row.
    block_positions = [
        np.asarray(block.active_candidate_positions(), dtype=np.int64)
        for block in packed.blocks
    ]
    unique_source_ids: list[int] = []
    source_to_unique: dict[int, int] = {}
    for block in packed.blocks:
        source_id = block.source_id
        if source_id not in source_to_unique:
            source_to_unique[source_id] = len(unique_source_ids)
            unique_source_ids.append(source_id)
    n_candidates = int(sum(block.candidate_count for block in packed.blocks))
    shared_candidate_slots = np.empty(n_candidates, dtype=np.int64)
    shared_candidate_index = np.empty(n_candidates, dtype=np.int16)
    cursor = 0
    for block, positions in zip(packed.blocks, block_positions):
        count = int(positions.size)
        shared_candidate_index[cursor : cursor + count] = source_to_unique[
            block.source_id
        ]
        shared_candidate_slots[cursor : cursor + count] = positions
        cursor += count

    def _candidate_column(bundle: AdaptedAxisBundle, name: str) -> Any:
        if name in bundle.item_features:
            return bundle.item_features[name]
        if name in bundle.label_features:
            return bundle.label_features[name]
        if name in bundle.label_mask_features:
            return bundle.label_mask_features[name]
        if name in bundle.candidate_metadata:
            return bundle.candidate_metadata[name]
        raise KeyError(name)

    # Resolve each candidate name to its owning map once (item/label/...).
    # Fast paths:
    # - 1 source: inline 4-map probe (no helper, no cross-source scan)
    # - N sources: one index pass/source with precomputed kind+dtype_num, then
    #   O(names) equality checks (no per-name isinstance/all scans)
    candidate_maps: dict[str, str] = {}
    list_batch_names: list[str] = []
    dense_names: list[str] = []
    object_names: list[str] = []
    n_unique_sources = len(unique_source_ids)

    if n_unique_sources == 1:
        bundle = bundles[unique_source_ids[0]]
        item_feats = bundle.item_features
        label_feats = bundle.label_features
        mask_feats = bundle.label_mask_features
        meta_feats = bundle.candidate_metadata
        for name in candidate_name_list:
            if name in broadcast_set:
                object_names.append(name)
                continue
            if name in item_feats:
                map_name = "item_features"
                sample_column = item_feats[name]
            elif name in label_feats:
                map_name = "label_features"
                sample_column = label_feats[name]
            elif name in mask_feats:
                map_name = "label_mask_features"
                sample_column = mask_feats[name]
            elif name in meta_feats:
                map_name = "candidate_metadata"
                sample_column = meta_feats[name]
            else:
                object_names.append(name)
                continue
            candidate_maps[name] = map_name
            kind, _dtype_num = _pack_column_kind(sample_column)
            if kind == _PACK_COL_LIST:
                list_batch_names.append(name)
            elif kind == _PACK_COL_DENSE:
                dense_names.append(name)
            else:
                object_names.append(name)
    else:
        # name -> (map_name, kind, dtype_num) per source; column fetched later.
        source_indexes: list[dict[str, tuple[str, int, int]]] = []
        for source_id in unique_source_ids:
            bundle = bundles[source_id]
            index: dict[str, tuple[str, int, int]] = {}
            for map_name in (
                "item_features",
                "label_features",
                "label_mask_features",
                "candidate_metadata",
            ):
                for feature_name, column in getattr(bundle, map_name).items():
                    if feature_name in index:
                        continue
                    kind, dtype_num = _pack_column_kind(column)
                    index[feature_name] = (map_name, kind, dtype_num)
            source_indexes.append(index)
        first_index = source_indexes[0]
        rest_indexes = source_indexes[1:]
        for name in candidate_name_list:
            if name in broadcast_set:
                object_names.append(name)
                continue
            first = first_index.get(name)
            if first is None:
                object_names.append(name)
                continue
            map_name, kind0, dtype0 = first
            ok = True
            for index in rest_indexes:
                other = index.get(name)
                if (
                    other is None
                    or other[0] != map_name
                    or other[1] != kind0
                    or other[2] != dtype0
                ):
                    ok = False
                    break
            if not ok:
                object_names.append(name)
                continue
            candidate_maps[name] = map_name
            if kind0 == _PACK_COL_LIST:
                list_batch_names.append(name)
            elif kind0 == _PACK_COL_DENSE:
                dense_names.append(name)
            else:
                object_names.append(name)

    candidate_values: dict[str, Any] = {}
    for name in list_batch_names:
        map_name = candidate_maps[name]
        candidate_values[name] = SequenceColumnBatch(
            columns=tuple(
                getattr(bundles[source_id], map_name)[name]
                for source_id in unique_source_ids
            ),
            slots=shared_candidate_slots,
            column_index=shared_candidate_index,
        )

    # Pre-resolve dense columns per source to avoid per-block dict walks.
    # Gather via shared_candidate_slots/index (one fancy-index per source), not
    # per pack block — packs here are often ~1 candidate/request, so the old
    # per-block path did hundreds of size-1 takes + concatenate.
    dense_by_source: list[dict[str, np.ndarray]] = []
    for source_id in unique_source_ids:
        bundle = bundles[source_id]
        dense_by_source.append(
            {name: getattr(bundle, candidate_maps[name])[name] for name in dense_names}
        )
    reqs_by_unique = [
        np.flatnonzero(shared_candidate_index == unique_idx)
        for unique_idx in range(len(dense_by_source))
    ]
    # slots are already int64; hoist per-source index vectors out of the name loop.
    slots_by_unique = [shared_candidate_slots[reqs] for reqs in reqs_by_unique]
    for name in dense_names:
        sample_column = None
        for cols in dense_by_source:
            sample_column = cols.get(name)
            if sample_column is not None:
                break
        if sample_column is None:
            candidate_values[name] = np.empty(0, dtype=np.int64)
            continue
        out = np.empty(n_candidates, dtype=sample_column.dtype)
        for unique_idx, cols in enumerate(dense_by_source):
            reqs = reqs_by_unique[unique_idx]
            if reqs.size == 0:
                continue
            out[reqs] = cols[name][slots_by_unique[unique_idx]]
        candidate_values[name] = out

    object_rows: dict[str, list[Any]] = {name: [] for name in object_names}
    for block, positions in zip(packed.blocks, block_positions):
        bundle = bundles[block.source_id]
        request_slot = int(block.representative_request_position)
        for name in object_names:
            if name in broadcast_set:
                if name in bundle.request_features:
                    value = bundle.request_features[name][request_slot]
                elif name in bundle.sequence_features:
                    value = bundle.sequence_features[name][request_slot]
                elif request_id_column is not None and name == request_id_column:
                    value = bundle.request_ids[request_slot]
                else:
                    raise KeyError(
                        f"candidate column {name!r} missing from axis bundle "
                        f"source {block.source_id}"
                    )
                object_rows[name].extend([value] * int(positions.size))
                continue
            map_name = candidate_maps.get(name)
            column = (
                getattr(bundle, map_name)[name]
                if map_name is not None
                else _candidate_column(bundle, name)
            )
            if isinstance(column, np.ndarray) and column.ndim == 1:
                object_rows[name].extend(column[positions].tolist())
            else:
                for slot in positions:
                    object_rows[name].append(column[int(slot)])
    for name in object_names:
        candidate_values[name] = tuple(object_rows[name])

    sequence_plans = {
        sequence.name: build_axis_sequence_selection_plan(
            sequence,
            packed=packed,
            bundles=bundles,
        )
        for sequence in sequences
    }
    if any(len(values) != n_requests for values in request_values.values()):
        raise RuntimeError("packed request-axis column lengths are inconsistent")
    if any(len(values) != n_candidates for values in candidate_values.values()):
        raise RuntimeError("packed candidate-axis column lengths are inconsistent")

    return PreparedAxisBatch(
        request_values=request_values,
        candidate_values=candidate_values,
        request_row_indices=torch.as_tensor(
            packed.candidate_to_request,
            dtype=torch.long,
        ),
        sequence_plans=sequence_plans,
        n_requests=n_requests,
        n_candidates=n_candidates,
    )


def materialize_packed_axis_bundles(
    bundles: Mapping[int, AdaptedAxisBundle],
    packed: PackedRequestPlan,
    *,
    request_columns: Sequence[str],
    sequence_columns: Sequence[str],
    candidate_columns: Sequence[str],
    request_id_column: str | None = None,
) -> tuple[Any, Any, Any]:
    """Build narrow request + candidate Arrow tables for one packed batch.

    This is the only Arrow materialization on the direct path: once per packed
    batch boundary, never a candidate-flat rebuild of an entire scanner table.
    Request-id (and other request scalars listed in ``candidate_columns`` that
    live on the request axis) are broadcast onto candidates only here, for
    group_id / scenario / prediction-key parity with legacy flat tables.
    """

    pa, _pc, _ds, _pq = _require_pyarrow()
    if not packed.blocks:
        raise ValueError("cannot materialize an empty packed plan")

    request_rows: dict[str, list[Any]] = {
        name: [] for name in (*request_columns, *sequence_columns)
    }
    candidate_rows: dict[str, list[Any]] = {name: [] for name in candidate_columns}
    row_indices: list[int] = []

    for request_index, block_index in enumerate(packed.unique_block_indices):
        block = packed.blocks[int(block_index)]
        bundle = bundles[block.source_id]
        slot = int(block.representative_request_position)
        for name in request_columns:
            request_rows[name].append(bundle.request_features[name][slot])
        for name in sequence_columns:
            request_rows[name].append(bundle.sequence_features[name][slot])

    for block_index, block in enumerate(packed.blocks):
        bundle = bundles[block.source_id]
        request_slot = int(packed.block_to_request[block_index])
        bundle_slot = int(block.representative_request_position)
        for candidate_pos in block.active_candidate_positions():
            cand = int(candidate_pos)
            for name in candidate_columns:
                if name in bundle.item_features:
                    candidate_rows[name].append(bundle.item_features[name][cand])
                elif name in bundle.label_features:
                    candidate_rows[name].append(bundle.label_features[name][cand])
                elif name in bundle.label_mask_features:
                    candidate_rows[name].append(bundle.label_mask_features[name][cand])
                elif name in bundle.candidate_metadata:
                    candidate_rows[name].append(bundle.candidate_metadata[name][cand])
                elif name in bundle.request_features:
                    candidate_rows[name].append(
                        bundle.request_features[name][bundle_slot]
                    )
                elif request_id_column is not None and name == request_id_column:
                    candidate_rows[name].append(bundle.request_ids[bundle_slot])
                else:
                    raise KeyError(
                        f"candidate column {name!r} missing from axis bundle"
                    )
            row_indices.append(request_slot)

    request_table = pa.table(
        {
            name: _arrow_array_from_python_values(values)
            for name, values in request_rows.items()
        }
    )
    candidate_table = pa.table(
        {
            name: _arrow_array_from_python_values(values)
            for name, values in candidate_rows.items()
        }
    )
    return (
        candidate_table,
        request_table,
        torch.tensor(row_indices, dtype=torch.long),
    )


@dataclass(frozen=True)
class PreparedBatchTable:
    """Candidate-major Arrow table plus optional precomputed request dedup."""

    table: Any
    request_deduplication: tuple[Any, Any] | None = None

    @property
    def num_rows(self) -> int:
        return int(self.table.num_rows)

    @property
    def nbytes(self) -> int:
        return int(getattr(self.table, "nbytes", 0))

    @property
    def column_names(self) -> Any:
        return self.table.column_names

    def __getitem__(self, key: Any) -> Any:
        return self.table[key]


def materialize_packed_blocks(
    source_tables: Mapping[int, Any],
    blocks: Sequence[RequestGroupBlock],
) -> Any:
    """Take candidate rows for a packed batch (transitional oracle / compare path).

    Does not broadcast or re-encode features; only gathers rows already present
    on adapted source tables. Same ``request_id`` never spans sources. Takes are
    coalesced per ``source_id`` to avoid one ``take``/``concat`` per block.
    """

    pa, _pc, _ds, _pq = _require_pyarrow()
    if not blocks:
        raise ValueError("cannot materialize an empty packed block list")

    # Preserve pack order while coalescing runs of the same source.
    pieces: list[Any] = []
    run_source: int | None = None
    run_positions: list[np.ndarray] = []

    def flush_run() -> None:
        nonlocal run_source, run_positions
        if run_source is None:
            return
        table = source_tables[run_source]
        positions = (
            run_positions[0]
            if len(run_positions) == 1
            else np.concatenate(run_positions)
        )
        if (
            len(positions) > 0
            and int(positions[-1]) == int(positions[0]) + len(positions) - 1
            and bool(
                np.all(
                    positions
                    == np.arange(int(positions[0]), int(positions[0]) + len(positions))
                )
            )
        ):
            pieces.append(table.slice(int(positions[0]), len(positions)))
        else:
            pieces.append(table.take(pa.array(positions, type=pa.int64())))
        run_source = None
        run_positions = []

    for block in blocks:
        positions = np.asarray(block.active_candidate_positions(), dtype=np.int64)
        if run_source is not None and block.source_id != run_source:
            flush_run()
        run_source = block.source_id
        run_positions.append(positions)
    flush_run()

    if len(pieces) == 1:
        return pieces[0]
    return pa.concat_tables(pieces)


@dataclass(frozen=True)
class AggAxisPlan:
    """Control indices only: payload stays in the source Arrow table.

    Built from ``context_indices`` / ``target_indices`` / request-id /
    ``{ups}_x_indices``. Feature values are gathered at pack time.
    """

    n_candidates: int
    n_requests: int
    request_ids: tuple[Any, ...]
    candidate_to_request: Any
    request_raw_rows: Any
    request_local_positions: Any
    candidate_raw_rows: Any
    candidate_locals: Any
    # ups -> per-request token position arrays (already max_length truncated)
    ups_token_positions: Mapping[str, tuple[np.ndarray, ...]]
    # Derived candidate_position ordinals (within request), if configured
    candidate_positions: Any | None = None


@dataclass(frozen=True)
class ArrowAxisSource:
    """Raw Arrow table + compact axis plan (direct_arrow producer payload)."""

    table: Any
    plan: AggAxisPlan
    # Column role maps mirrored from the adapter plan (names are output names).
    request_feature_columns: tuple[str, ...]
    sequence_feature_columns: tuple[str, ...]
    item_feature_columns: tuple[str, ...]
    label_feature_columns: tuple[str, ...]
    label_mask_feature_columns: tuple[str, ...]
    candidate_metadata_columns: tuple[str, ...]
    bag_features: frozenset[str]
    # Physical name resolution: output/canonical -> present table column
    physical_columns: Mapping[str, str]
    # Sequence / time-delta gather hints
    ups_types: tuple[str, ...]
    sequence_columns_by_type: Mapping[str, tuple[str, ...]]
    time_delta_outputs: Mapping[str, str]
    time_delta_transform: str
    request_time_column: str
    request_id_column: str
    # Request-level transforms applied at pack gather
    request_maps: Mapping[str, Mapping[Any, int]]
    coarse_scene: Any | None
    label_masks: Mapping[str, str]
    label_missing_values: Mapping[str, tuple[Any, ...]]
    labels: Mapping[str, str]
    trusted_input: bool = True
    # Mutable once-per-source materialization cache (pack reuses like python direct).
    _cache: dict[str, Any] = field(
        default_factory=dict, repr=False, compare=False, hash=False
    )

    @property
    def n_candidates(self) -> int:
        return self.plan.n_candidates

    @property
    def n_requests(self) -> int:
        return self.plan.n_requests

    @property
    def request_ids(self) -> tuple[Any, ...]:
        return self.plan.request_ids

    @property
    def candidate_to_request(self) -> Any:
        return self.plan.candidate_to_request

    @property
    def request_raw_rows(self) -> Any:
        return self.plan.request_raw_rows


def _combine_array(column: Any) -> Any:
    if hasattr(column, "num_chunks"):
        if column.num_chunks == 1:
            return column.chunk(0)
        return column.combine_chunks()
    return column


def _list_row_values(array: Any, row_index: int) -> Any:
    """Return the values slice for one list cell (may be empty)."""

    pa, _pc, _ds, _pq = _require_pyarrow()
    if pa.types.is_null(array.type):
        return None
    if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type)):
        raise TypeError(f"expected list array, got {array.type}")
    if array[row_index].is_valid is False:
        return None
    start = int(array.offsets[row_index].as_py())
    stop = int(array.offsets[row_index + 1].as_py())
    return array.values.slice(start, stop - start)


def _gather_request_axis_cells(
    array: Any,
    raw_rows: Sequence[int],
    locals_: Sequence[int],
    *,
    bag: bool,
) -> list[Any]:
    """Gather request-axis list cells using numpy offsets (batched)."""

    pa, _pc, _ds, _pq = _require_pyarrow()
    array = _combine_array(array)
    n = len(raw_rows)
    if n == 0:
        return []
    out: list[Any] = [None] * n
    if pa.types.is_null(array.type):
        fill = [] if bag else None
        return [fill for _ in range(n)]

    offsets = array.offsets.to_numpy(zero_copy_only=False)
    values = array.values
    nested = pa.types.is_list(values.type) or pa.types.is_large_list(values.type)
    inner_offsets = values.offsets.to_numpy(zero_copy_only=False) if nested else None
    flat = values.values if nested else None

    for i, (raw_row, local) in enumerate(zip(raw_rows, locals_)):
        row = int(raw_row)
        loc = int(local)
        if not array[row].is_valid:
            out[i] = [] if bag else None
            continue
        start = int(offsets[row])
        stop = int(offsets[row + 1])
        if nested:
            cell = start + loc
            if cell >= stop:
                raise ValueError(
                    f"request local {loc} out of range for length {stop - start}"
                )
            s = int(inner_offsets[cell])
            e = int(inner_offsets[cell + 1])
            if bag:
                out[i] = flat.slice(s, e - s).to_pylist()
            elif e == s:
                out[i] = None
            elif e == s + 1:
                out[i] = flat[s].as_py()
            else:
                raise ValueError(f"single-valued feature has inner length {e - s}")
        else:
            if loc >= (stop - start):
                raise ValueError(
                    f"request local {loc} out of range for length {stop - start}"
                )
            scalar = values[start + loc].as_py()
            if bag:
                if scalar is None:
                    out[i] = []
                elif isinstance(scalar, list):
                    out[i] = scalar
                else:
                    out[i] = [scalar]
            elif isinstance(scalar, list):
                if not scalar:
                    out[i] = None
                elif len(scalar) == 1:
                    out[i] = scalar[0]
                else:
                    raise ValueError(
                        f"single-valued feature has inner length {len(scalar)}"
                    )
            else:
                out[i] = scalar
    return out


def _gather_candidate_axis_cells(
    array: Any,
    raw_rows: Sequence[int],
    locals_: Sequence[int],
    *,
    bag: bool,
) -> list[Any]:
    """Gather candidate-axis cells via Arrow offsets (batched by raw row)."""

    pa, _pc, _ds, _pq = _require_pyarrow()
    array = _combine_array(array)
    n = len(raw_rows)
    if n == 0:
        return []
    out: list[Any] = [None] * n
    if pa.types.is_null(array.type):
        fill = [] if bag else None
        return [fill for _ in range(n)]

    offsets = array.offsets.to_numpy(zero_copy_only=False)
    values = array.values
    nested = pa.types.is_list(values.type) or pa.types.is_large_list(values.type)

    by_row: dict[int, list[tuple[int, int]]] = {}
    for i, (raw_row, local) in enumerate(zip(raw_rows, locals_)):
        by_row.setdefault(int(raw_row), []).append((i, int(local)))

    if nested:
        inner_offsets = values.offsets.to_numpy(zero_copy_only=False)
        flat = values.values
        for row, items in by_row.items():
            if not array[row].is_valid:
                fill = [] if bag else None
                for out_i, _local in items:
                    out[out_i] = fill
                continue
            start = int(offsets[row])
            stop = int(offsets[row + 1])
            for out_i, loc in items:
                cell = start + loc
                if cell >= stop:
                    raise ValueError(
                        f"candidate local {loc} out of range for length {stop - start}"
                    )
                s = int(inner_offsets[cell])
                e = int(inner_offsets[cell + 1])
                if bag:
                    out[out_i] = flat.slice(s, e - s).to_pylist()
                elif e == s:
                    out[out_i] = None
                elif e == s + 1:
                    out[out_i] = flat[s].as_py()
                else:
                    raise ValueError(f"single-valued feature has inner length {e - s}")
        return out

    for row, items in by_row.items():
        if not array[row].is_valid:
            fill = [] if bag else None
            for out_i, _local in items:
                out[out_i] = fill
            continue
        start = int(offsets[row])
        stop = int(offsets[row + 1])
        for out_i, loc in items:
            if loc >= (stop - start):
                raise ValueError(
                    f"candidate local {loc} out of range for length {stop - start}"
                )
            scalar = values[start + loc].as_py()
            if bag:
                if scalar is None:
                    out[out_i] = []
                elif isinstance(scalar, list):
                    out[out_i] = scalar
                else:
                    out[out_i] = [scalar]
            else:
                out[out_i] = scalar
    return out


def _take_sequence_tokens(
    array: Any,
    raw_row: int,
    positions: np.ndarray,
) -> list[Any]:
    """Shared UPS selection: one ``pc.take`` per packed request field."""

    pa, pc, _ds, _pq = _require_pyarrow()
    array = _combine_array(array)
    if pa.types.is_null(array.type) or len(positions) == 0:
        return []
    row_values = _list_row_values(array, int(raw_row))
    if row_values is None:
        return []
    # Flatten validated singleton S-token wrappers: list<list<T>> -> flat T.
    if pa.types.is_list(row_values.type) or pa.types.is_large_list(row_values.type):
        # Prefer list_flatten when every cell is a singleton / null.
        flat = pc.list_flatten(row_values)
        # list_flatten drops nulls' contribution oddly for empty; fall back.
        if len(flat) == len(row_values):
            row_values = flat
        else:
            # Uneven inner lengths: materialize selected positions only.
            selected = [row_values[int(pos)].as_py() for pos in positions.tolist()]
            return [
                item[0]
                if isinstance(item, list) and len(item) == 1
                else (None if item == [] else item)
                for item in selected
            ]
    taken = pc.take(row_values, pa.array(positions, type=pa.int64()))
    return taken.to_pylist()


def _vectorized_time_deltas(
    event_times: Sequence[Any],
    request_time: Any,
    *,
    transform: str,
) -> list[float]:
    if not event_times:
        return []
    values = np.asarray(
        [0.0 if t is None else float(t) for t in event_times],
        dtype=np.float64,
    )
    # Null event times pad to 0.0 (match adapter trusted hot path).
    nulls = np.asarray([t is None for t in event_times], dtype=bool)
    deltas = float(request_time) - values
    if transform == "raw_ms":
        out = deltas
    elif transform == "seconds":
        out = deltas / 1000.0
    elif transform == "log1p_seconds":
        out = np.log1p(deltas / 1000.0)
    else:
        raise RuntimeError(f"unsupported time delta transform {transform!r}")
    out = out.astype(np.float64, copy=False)
    out[nulls] = 0.0
    return out.tolist()


def request_group_blocks_from_arrow_source(
    source: ArrowAxisSource,
    *,
    source_id: int,
    sequences: Sequence[Any] = (),
    length_bucket_metric: str = "max",
) -> tuple[RequestGroupBlock, ...]:
    """Build descriptors from an Arrow-backed axis source."""

    plan = source.plan
    if plan.n_requests == 0:
        return ()
    positions_by_slot: list[list[int]] = [[] for _ in range(plan.n_requests)]
    for candidate_index, slot in enumerate(plan.candidate_to_request):
        positions_by_slot[int(slot)].append(int(candidate_index))

    ups_by_sequence: dict[str, str] = {}
    for sequence in sequences:
        if not sequence.fields:
            continue
        if sequence.name in plan.ups_token_positions:
            ups_by_sequence[sequence.name] = sequence.name
            continue
        # Fall back: match ups prefix of the primary sequence source.
        primary = sequence.fields[0].source
        matched = None
        for ups in source.ups_types:
            if primary.startswith(f"{ups}_x_"):
                matched = ups
                break
        if matched is None:
            raise ValueError(
                f"sequence {sequence.name!r} source {primary!r} has no UPS mapping"
            )
        ups_by_sequence[sequence.name] = matched

    blocks: list[RequestGroupBlock] = []
    for stable_group_order, positions in enumerate(positions_by_slot):
        if not positions:
            raise ValueError(
                f"request slot {stable_group_order} has no candidates in arrow source"
            )
        pre_compaction: dict[str, int] = {}
        for sequence in sequences:
            if not sequence.fields:
                continue
            ups = ups_by_sequence[sequence.name]
            length = int(len(plan.ups_token_positions[ups][stable_group_order]))
            tensor_max_length = _sequence_tensor_max_length(sequence)
            if tensor_max_length is not None:
                length = min(length, int(tensor_max_length))
            pre_compaction[sequence.name] = length
        blocks.append(
            RequestGroupBlock(
                source_id=source_id,
                raw_row_index=int(plan.request_raw_rows[stable_group_order]),
                request_id=plan.request_ids[stable_group_order],
                representative_request_position=stable_group_order,
                candidate_positions=np.asarray(positions, dtype=np.int64),
                candidate_offset=0,
                candidate_count=len(positions),
                pre_compaction_sequence_lengths=pre_compaction,
                effective_bucket_length=effective_bucket_length_from_pre_compaction(
                    pre_compaction,
                    metric=length_bucket_metric,
                ),
                stable_group_order=stable_group_order,
            )
        )
    return tuple(blocks)


def _sequence_selection_plan_from_request_values(
    sequence: Any,
    *,
    packed: PackedRequestPlan,
    request_values: Mapping[str, Sequence[Any]],
) -> SequenceSelectionPlan:
    """Truncate-then-compact over already membership-selected request lists."""

    if not sequence.fields:
        empty = np.asarray([], dtype=np.int64)
        return SequenceSelectionPlan(
            sequence_name=sequence.name,
            selections=tuple(),
            pre_compaction_lengths=empty,
            compacted_lengths=empty,
            token_to_request=empty,
        )

    anchor_source = None
    if sequence.null_anchor_field is not None:
        for field in sequence.fields:
            if field.name == sequence.null_anchor_field:
                anchor_source = field.source
                break
        if anchor_source is None:
            raise ValueError(
                f"sequence {sequence.name!r} null_anchor_field "
                f"{sequence.null_anchor_field!r} is not one of its fields"
            )

    primary_source = sequence.fields[0].source
    selections: list[np.ndarray] = []
    pre_lengths: list[int] = []
    compacted_lengths: list[int] = []
    token_to_request: list[int] = []

    for request_index, block_index in enumerate(packed.unique_block_indices):
        block = packed.blocks[int(block_index)]
        primary_row = request_values[primary_source][request_index]
        list_length = 0 if primary_row is None else len(primary_row)

        for field in sequence.fields[1:]:
            aligned_row = request_values[field.source][request_index]
            aligned_length = 0 if aligned_row is None else len(aligned_row)
            if aligned_length != list_length:
                raise ValueError(
                    f"sequence {sequence.name!r} field {field.name!r} has length "
                    f"{aligned_length}, expected {list_length} for request "
                    f"{block.request_id!r}"
                )

        anchor_is_null = None
        if anchor_source is not None:
            anchor_row = request_values[anchor_source][request_index]
            anchor_values = () if anchor_row is None else anchor_row
            anchor_is_null = np.fromiter(
                (value is None for value in anchor_values),
                dtype=bool,
                count=list_length,
            )

        (
            kept,
            pre_length,
            compacted_length,
        ) = row_sequence_selection_after_truncate_then_compact(
            list_length=list_length,
            anchor_is_null=anchor_is_null,
            max_length=_sequence_tensor_max_length(sequence),
            truncation=sequence.truncation,
        )
        expected_pre_length = block.pre_compaction_sequence_lengths.get(sequence.name)
        if expected_pre_length is not None and int(expected_pre_length) != pre_length:
            raise RuntimeError(
                f"sequence {sequence.name!r} pre-compaction length changed "
                f"between bucket and pack for request {block.request_id!r}: "
                f"bucket={expected_pre_length}, pack={pre_length}"
            )
        selections.append(kept)
        pre_lengths.append(pre_length)
        compacted_lengths.append(compacted_length)
        token_to_request.extend([request_index] * compacted_length)

    return SequenceSelectionPlan(
        sequence_name=sequence.name,
        selections=tuple(selections),
        pre_compaction_lengths=np.asarray(pre_lengths, dtype=np.int64),
        compacted_lengths=np.asarray(compacted_lengths, dtype=np.int64),
        token_to_request=np.asarray(token_to_request, dtype=np.int64),
    )


def _column_array_cache(
    source: ArrowAxisSource,
    physical_names: set[str],
) -> dict[str, Any]:
    """Combine only the physical columns needed for this pack gather."""

    cache: dict[str, Any] = {}
    table = source.table
    for physical in physical_names:
        if physical in table.column_names:
            cache[physical] = _combine_array(table[physical])
    return cache


def _row_values_cached(
    array: Any,
    raw_row: int,
    row_cache: dict[int, Any],
) -> Any:
    """Cache one list-row values slice (shared by all UPS fields of that row)."""

    if raw_row in row_cache:
        return row_cache[raw_row]
    values = _list_row_values(array, raw_row)
    row_cache[raw_row] = values
    return values


def _numpy_row_cached(
    row_values: Any,
    raw_row: int,
    numpy_cache: dict[int, Any],
) -> Any | None:
    """Cache a numeric numpy view of one UPS row when possible."""

    if raw_row in numpy_cache:
        return numpy_cache[raw_row]
    if row_values is None:
        numpy_cache[raw_row] = None
        return None
    try:
        values = row_values.to_numpy(zero_copy_only=False)
    except (TypeError, ValueError):
        numpy_cache[raw_row] = None
        return None
    if getattr(values.dtype, "kind", "") not in {"i", "u", "f"}:
        numpy_cache[raw_row] = None
        return None
    numpy_cache[raw_row] = values
    return values


def _take_from_row_values(
    row_values: Any,
    positions: np.ndarray,
    *,
    raw_row: int | None = None,
    numpy_cache: dict[int, Any] | None = None,
) -> list[Any]:
    """Select UPS tokens from an already-sliced row values array."""

    pa, pc, _ds, _pq = _require_pyarrow()
    if row_values is None or len(positions) == 0:
        return []
    if pa.types.is_list(row_values.type) or pa.types.is_large_list(row_values.type):
        flat = pc.list_flatten(row_values)
        if len(flat) == len(row_values):
            row_values = flat
        else:
            selected = [row_values[int(pos)].as_py() for pos in positions.tolist()]
            return [
                item[0]
                if isinstance(item, list) and len(item) == 1
                else (None if item == [] else item)
                for item in selected
            ]
    if raw_row is not None and numpy_cache is not None:
        values = _numpy_row_cached(row_values, raw_row, numpy_cache)
        if values is not None:
            return values[positions].tolist()
    try:
        values = row_values.to_numpy(zero_copy_only=False)
        if getattr(values.dtype, "kind", "") in {"i", "u", "f"}:
            return values[positions].tolist()
    except (TypeError, ValueError):
        pass
    return pc.take(row_values, pa.array(positions, type=pa.int64())).to_pylist()


def _index_request_cell(row_cell: Any, local: int, *, bag: bool) -> Any:
    """Index one raw-row cell for a request-local position."""

    if row_cell is None:
        return [] if bag else None
    if isinstance(row_cell, (list, tuple, np.ndarray)):
        # Nested request-axis: row_cell is the per-request vector.
        if local >= len(row_cell):
            raise ValueError(
                f"request local {local} out of range for length {len(row_cell)}"
            )
        cell = row_cell[local]
    else:
        if local != 0:
            raise ValueError(f"scalar request-axis cell cannot index local={local}")
        cell = row_cell
    if bag:
        if cell is None:
            return []
        if isinstance(cell, np.ndarray):
            return cell.tolist()
        if isinstance(cell, list):
            return cell
        return [cell]
    if cell is None:
        return None
    if isinstance(cell, np.ndarray):
        if cell.size == 0:
            return None
        if cell.shape == (1,):
            item = cell[0]
            return item.item() if isinstance(item, np.generic) else item
        return cell.tolist()
    if isinstance(cell, list):
        if not cell:
            return None
        if len(cell) != 1:
            raise ValueError(f"single-valued feature has inner length {len(cell)}")
        return cell[0]
    if isinstance(cell, np.generic):
        return cell.item()
    return cell


def _index_candidate_cell(row_cell: Any, local: int, *, bag: bool) -> Any:
    """Index one raw-row cell for a candidate-local position."""

    if row_cell is None:
        return [] if bag else None
    if not isinstance(row_cell, (list, tuple, np.ndarray)):
        raise ValueError("candidate-axis feature must be list-valued")
    if local >= len(row_cell):
        raise ValueError(
            f"candidate local {local} out of range for length {len(row_cell)}"
        )
    cell = row_cell[local]
    if bag:
        if cell is None:
            return []
        if isinstance(cell, np.ndarray):
            return cell.tolist()
        if isinstance(cell, list):
            return cell
        return [cell]
    if cell is None:
        return None
    if isinstance(cell, np.ndarray):
        if cell.size == 0:
            return None
        if cell.shape == (1,):
            item = cell[0]
            return item.item() if isinstance(item, np.generic) else item
        return cell.tolist()
    if isinstance(cell, list):
        if not cell:
            return None
        if len(cell) != 1:
            raise ValueError(f"single-valued feature has inner length {len(cell)}")
        return cell[0]
    if isinstance(cell, np.generic):
        return cell.item()
    return cell


def materialize_arrow_axis_source(source: ArrowAxisSource) -> AdaptedAxisBundle:
    """Convert one ArrowAxisSource to AdaptedAxisBundle once (columnar).

    Matches python ``direct`` economics: pay feature materialization once per
    retained source, then pack only copies references. Sequence columns avoid
    full-history ``to_pylist`` and keep membership ``take`` only.
    """

    cached = source._cache.get("bundle")
    if isinstance(cached, AdaptedAxisBundle):
        return cached

    from .dataloader import (
        _arrow_array_to_pylist,
        _map_request_value,
        coarse_scene_ids,
    )

    plan = source.plan
    table = source.table
    n_requests = plan.n_requests
    n_candidates = plan.n_candidates
    pa, _pc, _ds, _pq = _require_pyarrow()

    def physical(name: str) -> str:
        return source.physical_columns.get(name, name)

    def column_pylist(name: str) -> list[Any] | None:
        phys = physical(name)
        if phys not in table.column_names:
            return None
        return _arrow_array_to_pylist(pa, _combine_array(table[phys]))

    request_raw_rows = [int(v) for v in plan.request_raw_rows.tolist()]
    request_locals = [int(v) for v in plan.request_local_positions.tolist()]
    candidate_raw_rows = [int(v) for v in plan.candidate_raw_rows.tolist()]
    candidate_locals = [int(v) for v in plan.candidate_locals.tolist()]

    # --- Request scalars / bags (whole-column pylist once) ---
    request_features: dict[str, tuple[Any, ...]] = {}
    coarse = source.coarse_scene
    coarse_names = set()
    if coarse is not None:
        coarse_names = {coarse.index_column, coarse.prior_id_column}

    for name in source.request_feature_columns:
        if name in coarse_names:
            continue
        raw = column_pylist(name)
        if raw is None:
            raise KeyError(f"request column {name!r} missing from table")
        bag = name in source.bag_features
        values = [
            _index_request_cell(raw[row], loc, bag=bag)
            for row, loc in zip(request_raw_rows, request_locals)
        ]
        if name in source.request_maps:
            values = [
                _map_request_value(
                    value,
                    column=name,
                    mapping=source.request_maps[name],
                    validate_contract=not source.trusted_input,
                )
                for value in values
            ]
        request_features[name] = tuple(values)

    if coarse is not None:
        raw_scene_col = coarse.raw_scene_column
        if raw_scene_col in request_features:
            scene_values = request_features[raw_scene_col]
        else:
            raw = column_pylist(raw_scene_col)
            if raw is None:
                raise KeyError(f"coarse scene column {raw_scene_col!r} missing")
            scene_values = tuple(
                _index_request_cell(raw[row], loc, bag=False)
                for row, loc in zip(request_raw_rows, request_locals)
            )
        indexes: list[int] = []
        priors: list[int] = []
        for scene in scene_values:
            index, prior = coarse_scene_ids(
                scene,
                coarse.search_scene_ids,
                unlisted_policy=coarse.unlisted_policy,
            )
            indexes.append(index)
            priors.append(prior)
        if coarse.index_column in source.request_feature_columns:
            request_features[coarse.index_column] = tuple(indexes)
        if coarse.prior_id_column in source.request_feature_columns:
            request_features[coarse.prior_id_column] = tuple(priors)

    # --- Sequences: whole-column pylist once, then vectorized membership index ---
    sequence_features: dict[str, tuple[Any, ...]] = {}
    seq_values: dict[str, list[Any]] = {
        name: [None] * n_requests for name in source.sequence_feature_columns
    }

    req_times: list[Any] | None = None
    if source.time_delta_outputs:
        raw_times = column_pylist(source.request_time_column)
        if raw_times is None:
            raise KeyError(
                f"request time column {source.request_time_column!r} missing"
            )
        req_times = [
            _index_request_cell(raw_times[row], loc, bag=False)
            for row, loc in zip(request_raw_rows, request_locals)
        ]

    unique_request_rows = sorted(set(request_raw_rows))

    def _row_as_indexable(row: Any) -> Any:
        if row is None:
            return None
        if isinstance(row, np.ndarray):
            return row
        if not isinstance(row, (list, tuple)):
            raise ValueError("UPS column row must be list-valued")
        if not row:
            return np.empty(0, dtype=np.int64)
        first = row[0]
        if isinstance(first, (list, np.ndarray)):
            # Singleton-wrapped tokens: flatten once.
            flat = []
            for item in row:
                if item is None:
                    flat.append(None)
                elif isinstance(item, np.ndarray):
                    if item.size == 0:
                        flat.append(None)
                    elif item.shape == (1,):
                        value = item[0]
                        flat.append(
                            value.item() if isinstance(value, np.generic) else value
                        )
                    else:
                        flat.append(item)
                elif isinstance(item, list):
                    if not item:
                        flat.append(None)
                    elif len(item) == 1:
                        flat.append(item[0])
                    else:
                        flat.append(item)
                else:
                    flat.append(item)
            row = flat
            if not row or row[0] is None or isinstance(row[0], bool):
                return row
            first = row[0]
        if isinstance(first, bool) or first is None:
            return row
        if isinstance(first, (int, float, np.integer, np.floating)):
            try:
                return np.asarray(row)
            except (TypeError, ValueError):
                return row
        return row

    def _index_positions(row_obj: Any, positions: np.ndarray) -> list[Any]:
        if row_obj is None or len(positions) == 0:
            return []
        if isinstance(row_obj, np.ndarray):
            return row_obj[positions].tolist()
        return [row_obj[int(pos)] for pos in positions.tolist()]

    for ups, columns in source.sequence_columns_by_type.items():
        positions_list = plan.ups_token_positions[ups]
        for column in columns:
            if column not in seq_values:
                continue
            raw = column_pylist(column)
            if raw is None:
                for slot in range(n_requests):
                    seq_values[column][slot] = []
                continue
            row_cache = {
                row: _row_as_indexable(raw[row]) for row in unique_request_rows
            }
            for slot in range(n_requests):
                seq_values[column][slot] = _index_positions(
                    row_cache[request_raw_rows[slot]],
                    positions_list[slot],
                )

        out_name = source.time_delta_outputs.get(ups)
        if out_name is None or out_name not in seq_values:
            continue
        time_raw = column_pylist(f"{ups}_x_time")
        assert req_times is not None
        if time_raw is None:
            for slot in range(n_requests):
                seq_values[out_name][slot] = []
            continue
        row_cache = {
            row: _row_as_indexable(time_raw[row]) for row in unique_request_rows
        }
        for slot in range(n_requests):
            event_times = _index_positions(
                row_cache[request_raw_rows[slot]],
                positions_list[slot],
            )
            seq_values[out_name][slot] = _vectorized_time_deltas(
                event_times,
                req_times[slot],
                transform=source.time_delta_transform,
            )

    for name, values in seq_values.items():
        sequence_features[name] = compact_list_column_from_rows(values)
    sequence_features = share_compact_list_offsets(sequence_features)

    # --- Candidate item / label / metadata ---
    item_features: dict[str, tuple[Any, ...]] = {}
    for name in source.item_feature_columns:
        raw = column_pylist(name)
        if raw is None:
            raise KeyError(f"item column {name!r} missing from table")
        bag = name in source.bag_features
        item_features[name] = tuple(
            _index_candidate_cell(raw[row], loc, bag=bag)
            for row, loc in zip(candidate_raw_rows, candidate_locals)
        )

    label_features: dict[str, tuple[Any, ...]] = {}
    task_by_column = {column: task for task, column in source.labels.items()}
    for name in source.label_feature_columns:
        raw = column_pylist(name)
        if raw is None:
            raise KeyError(f"label column {name!r} missing from table")
        values = [
            _index_candidate_cell(raw[row], loc, bag=False)
            for row, loc in zip(candidate_raw_rows, candidate_locals)
        ]
        task = task_by_column.get(name)
        if task is not None:
            mask_column = source.label_masks.get(task)
            missing_vals = source.label_missing_values.get(task, ())
            complete = mask_column is None and not missing_vals
            if complete:
                normalized: list[Any] = []
                for value in values:
                    valid_binary = (
                        not isinstance(value, bool)
                        and isinstance(value, (int, float))
                        and float(value) in {0.0, 1.0}
                        and (not isinstance(value, float) or value == value)
                    )
                    if not valid_binary:
                        raise ValueError(
                            f"label {name!r} must be numeric 0/1; got {value!r}"
                        )
                    normalized.append(int(value))
                values = normalized
        label_features[name] = tuple(values)

    if source.label_mask_feature_columns:
        raise KeyError(
            "label mask columns are not yet materialized on direct_arrow "
            "(configure complete labels)"
        )

    candidate_metadata: dict[str, tuple[Any, ...]] = {}
    for name in source.candidate_metadata_columns:
        if name == "candidate_position" and plan.candidate_positions is not None:
            candidate_metadata[name] = tuple(
                int(v) for v in plan.candidate_positions.tolist()
            )
            continue
        raw = column_pylist(name)
        if raw is None:
            if plan.candidate_positions is not None:
                candidate_metadata[name] = tuple(
                    int(v) for v in plan.candidate_positions.tolist()
                )
            else:
                candidate_metadata[name] = tuple(None for _ in range(n_candidates))
            continue
        candidate_metadata[name] = tuple(
            _index_candidate_cell(raw[row], loc, bag=False)
            for row, loc in zip(candidate_raw_rows, candidate_locals)
        )

    bundle = AdaptedAxisBundle(
        n_candidates=n_candidates,
        n_requests=n_requests,
        request_ids=plan.request_ids,
        candidate_to_request=plan.candidate_to_request,
        request_features=request_features,
        sequence_features=sequence_features,
        item_features=item_features,
        label_features=label_features,
        label_mask_features={},
        candidate_metadata=candidate_metadata,
        request_raw_rows=plan.request_raw_rows,
        candidate_raw_rows=plan.candidate_raw_rows,
    )
    source._cache["bundle"] = bundle
    # Drop the raw table after materialization so retained sources don't keep
    # both Arrow and Python payloads for the rest of the shuffle window.
    source._cache["table_released"] = True
    object.__setattr__(source, "table", None)
    return bundle


def prepare_packed_arrow_axis_batch(
    sources: Mapping[int, ArrowAxisSource],
    packed: PackedRequestPlan,
    *,
    sequences: Sequence[Any],
    request_id_column: str | None = None,
    candidate_request_columns: Sequence[str] = (),
) -> PreparedAxisBatch:
    """Materialize each Arrow source once, then reuse python-direct pack gather.

    The previous pack-time cell/column gather re-converted Arrow→Python on every
    pack and was slower than ``direct``. Caching an ``AdaptedAxisBundle`` per
    retained source restores the same "pay once, pack by reference" model while
    still building control indices without ``_adapter_table_to_python``.
    """

    if not packed.blocks:
        raise ValueError("cannot prepare an empty packed arrow axis batch")

    bundles = {
        source_id: materialize_arrow_axis_source(source)
        for source_id, source in sources.items()
    }
    return prepare_packed_axis_batch(
        bundles,
        packed,
        sequences=sequences,
        request_id_column=request_id_column,
        candidate_request_columns=candidate_request_columns,
    )
