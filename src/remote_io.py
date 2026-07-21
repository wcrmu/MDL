"""Remote filesystem IO helpers for HDFS/viewfs Parquet reads.

Eason-equivalent model inside one DDP rank:
- each prefetch worker uses a thread-local HadoopFileSystem (own DFSClient)
- open via ``open_input_file`` with timeout / retry / defensive flock
- ``ParquetFile(native_file, pre_buffer=True)`` + ``iter_batches(use_threads=False)``
- timed ``native_file.close()`` so a corrupted stream cannot hang forever
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import fcntl
import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Literal, TypeVar

logger = logging.getLogger(__name__)

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


class RemoteIoTimeoutError(TimeoutError):
    """Raised when a remote IO call exceeds its configured timeout."""


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


def is_retryable_remote_error(error: BaseException) -> bool:
    """Return True for transient remote IO failures worth retrying."""

    if isinstance(error, (RemoteIoTimeoutError, TimeoutError, InterruptedError)):
        return True
    if isinstance(error, (BlockingIOError, ConnectionError, BrokenPipeError, OSError)):
        return True
    text = f"{type(error).__name__}: {error}".lower()
    return any(needle in text for needle in _RETRYABLE_NEEDLES)


def call_with_timeout(
    fn: Callable[[], T],
    timeout_sec: float,
    *,
    description: str = "remote IO",
) -> T:
    """Run ``fn`` in a daemon thread and raise if it exceeds ``timeout_sec``."""

    if timeout_sec <= 0:
        return fn()

    result_box: list[T] = []
    error_box: list[BaseException] = []
    done = threading.Event()

    def runner() -> None:
        try:
            result_box.append(fn())
        except BaseException as error:  # noqa: BLE001 - surface to caller
            error_box.append(error)
        finally:
            done.set()

    thread = threading.Thread(
        target=runner,
        name=f"remote-io-timeout:{description[:48]}",
        daemon=True,
    )
    thread.start()
    if not done.wait(timeout_sec):
        raise RemoteIoTimeoutError(
            f"{description} timed out after {timeout_sec:.1f}s"
        )
    if error_box:
        raise error_box[0]
    return result_box[0]


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
    time.sleep(delay)


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


def maybe_skip_or_raise(
    error: BaseException,
    policy: RemoteIoPolicy,
    *,
    description: str,
) -> bool:
    """Log and return True when the policy says to skip; otherwise re-raise."""

    if policy.skip_on_failure:
        logger.warning("skipping %s after failure: %s", description, error)
        return True
    raise error


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
    """Bound concurrent readers; on HDFS scale with GPU count up to 4/rank."""

    if prefetch_batches <= 0 or work_item_count <= 0:
        return 0
    if not remote:
        worker_budget = num_workers if num_workers > 0 else 4
        return min(work_item_count, prefetch_batches, worker_budget, 4)

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


def close_hdfs_native_file(
    native_file: Any,
    *,
    timeout_sec: float = 5.0,
    description: str = "close hdfs native file",
) -> None:
    """Close a native input stream with a short timeout; abandon on hang."""

    if native_file is None:
        return
    close = getattr(native_file, "close", None)
    if not callable(close):
        return
    try:
        call_with_timeout(close, timeout_sec, description=description)
    except RemoteIoTimeoutError:
        logger.warning("%s timed out after %.1fs; abandoning handle", description, timeout_sec)
    except BaseException as error:
        logger.warning("%s failed: %s", description, error)


def open_hdfs_input_with_protection(
    filesystem: Any,
    fs_path: str,
    *,
    lock_key: str,
    policy: RemoteIoPolicy,
    description: str | None = None,
) -> Any:
    """Open ``fs_path`` under flock + timeout + retry; return native file."""

    label = description or f"open_input_file {lock_key}"

    def open_fn() -> Any:
        return filesystem.open_input_file(fs_path)

    with PerFileLock(lock_key, enabled=policy.file_lock):
        return run_remote_op(
            open_fn,
            policy,
            description=label,
            timeout_sec=policy.open_timeout if policy.enabled else policy.op_timeout,
        )


def open_parquet_via_native(
    *,
    filesystem: Any,
    fs_path: str,
    lock_key: str,
    policy: RemoteIoPolicy,
    pq_module: Any,
    description: str | None = None,
) -> tuple[Any, Any | None]:
    """Open a ``ParquetFile``, using native_file + pre_buffer on remote FS.

    Returns ``(parquet_file, native_file_or_none)``. Caller must close the
    native file with ``close_hdfs_native_file`` when not None.
    """

    label = description or f"open parquet {lock_key}"
    if not policy.enabled:
        return pq_module.ParquetFile(fs_path, filesystem=filesystem), None

    native_file = open_hdfs_input_with_protection(
        filesystem,
        fs_path,
        lock_key=lock_key,
        policy=policy,
        description=f"{label} (native open)",
    )

    def build() -> Any:
        return pq_module.ParquetFile(
            native_file,
            pre_buffer=policy.pre_buffer,
        )

    try:
        parquet_file = run_remote_op(
            build,
            policy,
            description=f"{label} (ParquetFile)",
            timeout_sec=policy.open_timeout,
        )
    except BaseException:
        close_hdfs_native_file(
            native_file,
            timeout_sec=policy.close_timeout,
            description=f"{label} (close after open failure)",
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
        description=description,
    )
    try:
        yield parquet_file, native_file
    finally:
        close_hdfs_native_file(
            native_file,
            timeout_sec=policy.close_timeout,
            description=f"{description or lock_key} (native close)",
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
    **iter_kwargs: Any,
) -> Iterator[Any]:
    """Open and stream ``iter_batches`` under one remote IO session.

    Remote path uses thread-local FS + native_file + pre_buffer. Local path
    keeps a plain ``ParquetFile`` open. ``on_hdfs_failure: skip`` applies to
    body reads only.
    """

    label = description or f"read parquet {lock_key}"
    kwargs = dict(iter_kwargs)
    if policy.enabled:
        kwargs["use_threads"] = False

    resolved_key = filesystem_key
    if resolved_key is None and filesystem is not None:
        # Best-effort: callers should pass filesystem_key for TLS cloning.
        resolved_key = "hdfs://" if policy.enabled else "file://"

    try:
        if policy.enabled:
            fs = thread_local_hdfs_filesystem(
                resolved_key or "hdfs://",
                prototype=filesystem,
            )
            parquet_file, native_file = open_parquet_via_native(
                filesystem=fs,
                fs_path=fs_path,
                lock_key=lock_key,
                policy=policy,
                pq_module=pq_module,
                description=label,
            )
        else:
            native_file = None
            with PerFileLock(lock_key, enabled=False):
                parquet_file = pq_module.ParquetFile(fs_path, filesystem=filesystem)
    except BaseException as error:
        if maybe_skip_or_raise(error, policy, description=f"{label} (open)"):
            return
        raise

    batch_iterator: Any = None
    try:
        try:
            batch_iterator = run_remote_op(
                lambda: iter(parquet_file.iter_batches(**kwargs)),
                policy,
                description=f"{label} (start)",
                timeout_sec=policy.open_timeout if policy.enabled else policy.op_timeout,
            )
        except BaseException as error:
            if maybe_skip_or_raise(error, policy, description=f"{label} (start)"):
                return
            raise

        while True:
            if stop_event is not None and stop_event.is_set():
                return

            def next_batch(iterator: Any = batch_iterator) -> Any:
                return next(iterator)

            try:
                yield run_remote_op(
                    next_batch,
                    policy,
                    description=f"{label} (batch)",
                    timeout_sec=policy.op_timeout,
                )
            except StopIteration:
                return
            except BaseException as error:
                if maybe_skip_or_raise(
                    error,
                    policy,
                    description=f"{label} (batch)",
                ):
                    return
                raise
    finally:
        if batch_iterator is not None:
            close = getattr(batch_iterator, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as error:
                    logger.warning(
                        "failed to close batch iterator for %s: %s",
                        label,
                        error,
                    )
        close_hdfs_native_file(
            native_file,
            timeout_sec=policy.close_timeout,
            description=f"{label} (native close)",
        )
