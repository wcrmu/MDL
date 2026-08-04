"""Checkpoint storage on local disks and Hadoop-compatible filesystems.

Long HDFS-backed runs need three properties a plain ``Path`` cannot provide:
writes must target ``hdfs://`` / ``viewfs://`` URIs, a partially written step
must never be mistaken for a resumable one, and a transient NameNode error must
be retried instead of killing a multi-hour job. This module supplies the small
directory abstraction those guarantees are built on; the commit protocol that
uses it lives in :mod:`src.checkpoint`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import posixpath
import shutil
import time
from typing import Any, Callable, TypeVar
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

REMOTE_URI_SCHEMES = frozenset({"hdfs", "viewfs"})

# Streaming copies keep peak RSS flat while moving multi-GiB shard files.
_COPY_CHUNK_BYTES = 8 * 1024 * 1024
_DEFAULT_RETRIES = 3
_DEFAULT_RETRY_BASE_SEC = 1.0

_T = TypeVar("_T")


class CheckpointStoreError(RuntimeError):
    """Raised when a checkpoint filesystem operation fails after all retries."""


@dataclass(frozen=True)
class StoredEntry:
    """One immediate child of a checkpoint directory."""

    name: str
    is_dir: bool


def uri_scheme(uri: str | os.PathLike[str]) -> str:
    """Return the lowercase URI scheme, or an empty string for local paths."""

    text = os.fspath(uri)
    scheme = urlsplit(text).scheme.lower()
    # Windows-style drive letters are not schemes; single-character schemes
    # never appear in the Hadoop URIs this project consumes.
    return scheme if len(scheme) > 1 else ""


def is_remote_uri(uri: str | os.PathLike[str]) -> bool:
    """True when ``uri`` names a Hadoop-compatible filesystem location."""

    return uri_scheme(uri) in REMOTE_URI_SCHEMES


def _retry(
    operation: Callable[[], _T],
    *,
    description: str,
    retries: int = _DEFAULT_RETRIES,
    base_sec: float = _DEFAULT_RETRY_BASE_SEC,
) -> _T:
    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            return operation()
        except Exception as error:  # noqa: BLE001 - every remote failure is retried
            last_error = error
            if attempt == retries:
                break
            delay = base_sec * (2**attempt)
            logger.warning(
                "checkpoint store: %s failed (attempt %d/%d): %s; retrying in %.1fs",
                description,
                attempt + 1,
                retries + 1,
                error,
                delay,
            )
            time.sleep(delay)
    raise CheckpointStoreError(
        f"{description} failed after {retries + 1} attempt(s)"
    ) from last_error


class CheckpointStore:
    """A directory on some filesystem, addressed by relative path components."""

    is_remote = False

    @property
    def root_uri(self) -> str:
        raise NotImplementedError

    def uri(self, *parts: str) -> str:
        raise NotImplementedError

    def child(self, *parts: str) -> "CheckpointStore":
        raise NotImplementedError

    def makedirs(self, *parts: str) -> None:
        raise NotImplementedError

    def exists(self, *parts: str) -> bool:
        raise NotImplementedError

    def list_entries(self, *parts: str) -> list[StoredEntry]:
        """Return immediate children, or an empty list when the path is absent."""

        raise NotImplementedError

    def write_bytes(self, payload: bytes, *parts: str) -> None:
        raise NotImplementedError

    def read_bytes(self, *parts: str) -> bytes:
        raise NotImplementedError

    def upload_file(self, source: Path, *parts: str) -> None:
        raise NotImplementedError

    def download_file(self, destination: Path, *parts: str) -> None:
        raise NotImplementedError

    def remove_tree(self, *parts: str) -> None:
        raise NotImplementedError

    # --- JSON convenience ---

    def write_json(self, payload: dict[str, Any], *parts: str) -> None:
        self.write_bytes(
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            *parts,
        )

    def read_json(self, *parts: str) -> dict[str, Any]:
        return json.loads(self.read_bytes(*parts).decode("utf-8"))

    def list_dir_names(self, *parts: str) -> list[str]:
        return sorted(entry.name for entry in self.list_entries(*parts) if entry.is_dir)


class LocalCheckpointStore(CheckpointStore):
    """Checkpoint directory backed by the local filesystem."""

    is_remote = False

    def __init__(self, root: str | os.PathLike[str]) -> None:
        text = os.fspath(root)
        if uri_scheme(text) == "file":
            text = urlsplit(text).path
        self._root = Path(text).expanduser()

    @property
    def root_path(self) -> Path:
        return self._root

    @property
    def root_uri(self) -> str:
        return str(self._root)

    def _resolve(self, parts: tuple[str, ...]) -> Path:
        return self._root.joinpath(*parts) if parts else self._root

    def uri(self, *parts: str) -> str:
        return str(self._resolve(parts))

    def child(self, *parts: str) -> "LocalCheckpointStore":
        return LocalCheckpointStore(self._resolve(parts))

    def makedirs(self, *parts: str) -> None:
        self._resolve(parts).mkdir(parents=True, exist_ok=True)

    def exists(self, *parts: str) -> bool:
        return self._resolve(parts).exists()

    def list_entries(self, *parts: str) -> list[StoredEntry]:
        directory = self._resolve(parts)
        if not directory.is_dir():
            return []
        return [StoredEntry(item.name, item.is_dir()) for item in directory.iterdir()]

    def write_bytes(self, payload: bytes, *parts: str) -> None:
        target = self._resolve(parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        temporary.write_bytes(payload)
        os.replace(temporary, target)

    def read_bytes(self, *parts: str) -> bytes:
        return self._resolve(parts).read_bytes()

    def upload_file(self, source: Path, *parts: str) -> None:
        target = self._resolve(parts)
        if Path(source).resolve() == target.resolve():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)

    def download_file(self, destination: Path, *parts: str) -> None:
        source = self._resolve(parts)
        destination = Path(destination)
        if source.resolve() == destination.resolve():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def remove_tree(self, *parts: str) -> None:
        target = self._resolve(parts)
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()


class HadoopCheckpointStore(CheckpointStore):
    """Checkpoint directory backed by an HDFS/viewfs PyArrow filesystem.

    A fresh client is created per process rather than shared with the reader's
    thread-local pool: checkpoint writes run on a background uploader thread and
    must not be able to poison (or be poisoned by) a DFSClient that the data
    path is streaming Parquet through.
    """

    is_remote = True

    def __init__(
        self,
        root_uri: str,
        *,
        filesystem: Any | None = None,
        root_path: str | None = None,
    ) -> None:
        self._root_uri = root_uri.rstrip("/")
        if filesystem is None or root_path is None:
            import pyarrow.fs as pafs

            filesystem, resolved = pafs.FileSystem.from_uri(self._root_uri)
            root_path = resolved or urlsplit(self._root_uri).path
        self._filesystem = filesystem
        self._root_path = "/" + str(root_path).strip("/")

    @property
    def filesystem(self) -> Any:
        return self._filesystem

    @property
    def root_uri(self) -> str:
        return self._root_uri

    def _resolve(self, parts: tuple[str, ...]) -> str:
        return posixpath.join(self._root_path, *parts) if parts else self._root_path

    def uri(self, *parts: str) -> str:
        return posixpath.join(self._root_uri, *parts) if parts else self._root_uri

    def child(self, *parts: str) -> "HadoopCheckpointStore":
        return HadoopCheckpointStore(
            self.uri(*parts),
            filesystem=self._filesystem,
            root_path=self._resolve(parts),
        )

    def makedirs(self, *parts: str) -> None:
        path = self._resolve(parts)
        _retry(
            lambda: self._filesystem.create_dir(path, recursive=True),
            description=f"create_dir {path}",
        )

    def exists(self, *parts: str) -> bool:
        import pyarrow.fs as pafs

        path = self._resolve(parts)
        info = _retry(
            lambda: self._filesystem.get_file_info(path),
            description=f"stat {path}",
        )
        return info.type != pafs.FileType.NotFound

    def list_entries(self, *parts: str) -> list[StoredEntry]:
        import pyarrow.fs as pafs

        path = self._resolve(parts)
        selector = pafs.FileSelector(path, recursive=False, allow_not_found=True)
        infos = _retry(
            lambda: self._filesystem.get_file_info(selector),
            description=f"list {path}",
        )
        return [
            StoredEntry(posixpath.basename(info.path), info.type == pafs.FileType.Directory)
            for info in infos
        ]

    def write_bytes(self, payload: bytes, *parts: str) -> None:
        path = self._resolve(parts)
        parent = posixpath.dirname(path)

        def write() -> None:
            self._filesystem.create_dir(parent, recursive=True)
            with self._filesystem.open_output_stream(path) as stream:
                stream.write(payload)

        _retry(write, description=f"write {path}")

    def read_bytes(self, *parts: str) -> bytes:
        path = self._resolve(parts)

        def read() -> bytes:
            with self._filesystem.open_input_stream(path) as stream:
                return stream.readall()

        return _retry(read, description=f"read {path}")

    def upload_file(self, source: Path, *parts: str) -> None:
        path = self._resolve(parts)
        parent = posixpath.dirname(path)

        def upload() -> None:
            self._filesystem.create_dir(parent, recursive=True)
            with open(source, "rb") as local, self._filesystem.open_output_stream(
                path
            ) as remote:
                while True:
                    chunk = local.read(_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    remote.write(chunk)

        _retry(upload, description=f"upload {source} -> {path}")

    def download_file(self, destination: Path, *parts: str) -> None:
        path = self._resolve(parts)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        def download() -> None:
            temporary = destination.with_name(f".{destination.name}.part")
            with self._filesystem.open_input_stream(path) as remote, open(
                temporary, "wb"
            ) as local:
                while True:
                    chunk = remote.read(_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    local.write(chunk)
            os.replace(temporary, destination)

        _retry(download, description=f"download {path} -> {destination}")

    def remove_tree(self, *parts: str) -> None:
        import pyarrow.fs as pafs

        path = self._resolve(parts)
        info = _retry(
            lambda: self._filesystem.get_file_info(path),
            description=f"stat {path}",
        )
        if info.type == pafs.FileType.NotFound:
            return
        if info.type == pafs.FileType.Directory:
            _retry(
                lambda: self._filesystem.delete_dir(path),
                description=f"delete_dir {path}",
            )
        else:
            _retry(
                lambda: self._filesystem.delete_file(path),
                description=f"delete_file {path}",
            )


def open_checkpoint_store(uri: str | os.PathLike[str]) -> CheckpointStore:
    """Return the store for ``uri``; HDFS/viewfs go remote, everything else local."""

    text = os.fspath(uri)
    scheme = uri_scheme(text)
    if scheme in REMOTE_URI_SCHEMES:
        return HadoopCheckpointStore(text)
    if scheme in {"", "file"}:
        return LocalCheckpointStore(text)
    raise ValueError(
        f"unsupported checkpoint URI scheme {scheme!r}; "
        "supported schemes are file, hdfs, and viewfs"
    )


def download_tree(store: CheckpointStore, destination: Path, *parts: str) -> Path:
    """Copy a checkpoint directory to ``destination`` and return the local path."""

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for entry in store.list_entries(*parts):
        if entry.is_dir:
            download_tree(store, destination / entry.name, *parts, entry.name)
        else:
            store.download_file(destination / entry.name, *parts, entry.name)
    return destination
