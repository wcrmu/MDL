from __future__ import annotations  # Defer annotation evaluation for forward references.

"""Parquet-to-PyTorch data pipeline.

This module owns the complete input path: it discovers and shards Parquet
files, streams Arrow batches, encodes configured features, and builds the
``FeatureBatch`` objects consumed by training and inference.
"""

from collections import defaultdict
from collections.abc import Collection, Iterable as RuntimeIterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
import fnmatch
from hashlib import sha256
from itertools import islice
import glob
import importlib
import json
import logging
import math
from numbers import Integral, Real
import os
import queue
import threading
import time
from pathlib import Path
from types import GeneratorType
from typing import Any, Callable, Iterable, Iterator, Mapping
from urllib.parse import unquote, urlsplit

import torch
from torch import Tensor

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
from .remote_io import (
    RemoteIoPolicy,
    apply_worker_stagger,
    close_hdfs_native_file,
    iter_parquet_record_batches,
    open_parquet_via_native,
    run_under_file_lock,
    scaled_hdfs_prefetch_workers,
    thread_local_hdfs_filesystem,
)

logger = logging.getLogger(__name__)

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
        raise ValueError(f"parquet input URI must not include query or fragment: {item!r}")
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
        raise ValueError(f"only local file:// parquet input URIs are supported: {item!r}")
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
        return [_local_ref(match, filesystem) for match in sorted(path.rglob("*.parquet"))]
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
        canonical_uri=_canonical_remote_uri(remote.scheme, remote.authority, normalized_path),
        filesystem_key=remote.filesystem_key,
        fs_path=normalized_path,
        filesystem=remote.filesystem,
    )


def _discover_remote_directory(remote: _RemoteInput) -> list[ParquetInputRef]:
    pafs = _require_pyarrow_fs()
    selector = pafs.FileSelector(remote.fs_path, recursive=True)
    infos = remote.filesystem.get_file_info(selector)
    return [
        _remote_ref(remote, info.path)
        for info in infos
        if info.type == pafs.FileType.File and info.path.endswith(".parquet")
    ]


def _discover_remote_glob(remote: _RemoteInput, item: str) -> list[ParquetInputRef]:
    pafs = _require_pyarrow_fs()
    base_dir = _remote_glob_base_dir(remote.fs_path)
    base_info = remote.filesystem.get_file_info(base_dir)
    if base_info.type != pafs.FileType.Directory:
        raise FileNotFoundError(f"no parquet files matched input {item!r}")

    selector = pafs.FileSelector(base_dir, recursive=True)
    refs: list[ParquetInputRef] = []
    matched_any = False
    for info in remote.filesystem.get_file_info(selector):
        if not _match_remote_glob(_normalize_remote_path(info.path), remote.fs_path):
            continue
        matched_any = True
        if info.type == pafs.FileType.File:
            refs.append(_remote_ref(remote, info.path))
    if not refs and not matched_any:
        raise FileNotFoundError(f"no parquet files matched input {item!r}")
    return refs


def _discover_remote_input(item: str, filesystems: dict[str, Any]) -> list[ParquetInputRef]:
    pafs = _require_pyarrow_fs()
    remote = _remote_input_from_uri(item, filesystems)
    if _has_glob_meta(remote.fs_path):
        return _discover_remote_glob(remote, item)

    info = remote.filesystem.get_file_info(remote.fs_path)
    if info.type == pafs.FileType.File:
        return [_remote_ref(remote, info.path)]
    if info.type == pafs.FileType.Directory:
        return _discover_remote_directory(remote)
    raise FileNotFoundError(f"no parquet files matched input {item!r}")


def _unique_sorted_refs(refs: Iterable[ParquetInputRef]) -> list[ParquetInputRef]:
    unique = {ref.canonical_uri: ref for ref in refs}
    return sorted(unique.values(), key=lambda ref: ref.canonical_uri)


def discover_parquet_inputs(inputs: Iterable[str | Path]) -> list[ParquetInputRef]:
    """Resolve parquet files from local paths or HDFS/viewfs URLs.

    Local inputs keep the existing file, directory, and Python glob behavior.
    HDFS/viewfs inputs use PyArrow filesystem discovery and support common
    POSIX-style glob segments, including ``**`` as a full path segment.
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
            refs.extend(_discover_remote_input(item, remote_filesystems))
            continue
        if local_filesystem is None:
            local_filesystem = _require_pyarrow_fs().LocalFileSystem()
        refs.extend(_discover_local_input(item, local_filesystem))

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
    payload = "\n".join(f"{field.name}:{field.type}:{field.nullable}" for field in schema)
    return sha256(payload.encode("utf-8")).hexdigest()


def _coerce_parquet_input_ref(path: str | Path | ParquetInputRef) -> ParquetInputRef:
    if isinstance(path, ParquetInputRef):
        return path
    refs = discover_parquet_inputs([os.fspath(path)])
    if len(refs) != 1:
        raise ValueError(f"expected exactly one parquet file, discovered {len(refs)} from {path!r}")
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

    filesystem = thread_local_hdfs_filesystem(
        ref.filesystem_key,
        prototype=ref.filesystem,
    )
    schema_policy = replace(io_policy, pre_buffer=False)
    parquet_file, native_file = open_parquet_via_native(
        filesystem=filesystem,
        fs_path=ref.fs_path,
        lock_key=ref.canonical_uri,
        policy=schema_policy,
        pq_module=pq,
        description=f"read schema {ref.canonical_uri}",
    )
    try:
        return parquet_file.schema_arrow
    finally:
        close_hdfs_native_file(
            native_file,
            timeout_sec=io_policy.close_timeout,
            description=f"close schema handle {ref.canonical_uri}",
        )


def validate_matching_schemas(
    paths: Iterable[str | Path | ParquetInputRef],
    policy: RemoteIoPolicy | None = None,
) -> str:
    """Require identical schemas across files; return the shared fingerprint."""
    refs = [_coerce_parquet_input_ref(path) for path in paths]
    if not refs:
        raise ValueError("paths must not be empty")
    fingerprints = {
        ref: schema_fingerprint(parquet_schema(ref, policy=policy))
        for ref in refs
    }
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
        round(index * last / (sample_count - 1))
        for index in range(sample_count)
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
                raise RuntimeError("prefetch byte budget was released more than reserved")
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


def _scan_columns_for_split(split: ParquetSplitConfig, flat_columns: list[str]) -> list[str]:
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

    def load() -> _FileMetadataCache:
        filesystem = thread_local_hdfs_filesystem(
            ref.filesystem_key,
            prototype=ref.filesystem,
        )
        meta_policy = replace(io_policy, pre_buffer=False) if io_policy.enabled else io_policy
        if io_policy.enabled:
            parquet_file, native_file = open_parquet_via_native(
                filesystem=filesystem,
                fs_path=ref.fs_path,
                lock_key=ref.canonical_uri,
                policy=meta_policy,
                pq_module=pq,
                description=f"load parquet metadata {ref.canonical_uri}",
            )
        else:
            parquet_file = pq.ParquetFile(ref.fs_path, filesystem=ref.filesystem)
            native_file = None
        try:
            schema = parquet_file.schema_arrow
            column_names = (
                scan_columns if scan_columns is not None else list(schema.names)
            )
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
        finally:
            close_hdfs_native_file(
                native_file,
                timeout_sec=io_policy.close_timeout,
                description=f"close metadata handle {ref.canonical_uri}",
            )

    # Footer planning must stay strict: skipping here would desync LPT shards.
    # open_parquet_via_native already locks; for local keep a simple call.
    if io_policy.enabled:
        return load()
    return run_under_file_lock(
        load,
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
        return metadata_by_path

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_load_file_metadata_cache, ref, scan_columns, io_policy): ref
            for ref in paths
        }
        for future in as_completed(futures):
            metadata_by_path[futures[future]] = future.result()
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
    use_row_weights = all(item.compressed_bytes is not None for _, item in ordered_items)
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
            key=lambda candidate: (rank_totals[candidate], rank_counts[candidate], candidate),
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
    """One bounded queue plus its row-group reader thread."""

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
        if shard_world_size > 1 and self._effective_shard_unit() not in {"file", "row_group"}:
            raise ValueError(
                f"unsupported reader.shard_unit {requested_shard_unit!r} "
                "for distributed scanning"
            )
        self.all_paths = discover_parquet_inputs(split.inputs)
        self._io_policy = RemoteIoPolicy.from_reader(
            split.reader,
            remote=_refs_are_remote(self.all_paths),
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
            schema_refs = list(
                dict.fromkeys([anchor, *local_refs])
            )
        else:
            schema_refs = global_schema_refs
        validate_matching_schemas(schema_refs, policy=self._io_policy)
        if self.columns:
            # Auto-detecting adapters may support layout-specific raw columns
            # (the agg indices are absent from req files). Project optional
            # columns only when the split schema contains them, while still
            # failing early for a missing mandatory input.
            schema_names = set(parquet_schema(schema_refs[0], policy=self._io_policy).names)
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
        empty_rank_count = sum(1 for rank in range(self.shard_world_size) if counts[rank] == 0)
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
        assigned.sort(key=lambda item: (item.input_ref.canonical_uri, item.local_row_group_index))
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
        return [base + (1 if index < remainder else 0) for index in range(active_workers)]

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
                batch_size=batch_size,
                row_groups=[work_item.local_row_group_index],
                columns=scan_columns,
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
                batch_size=batch_size,
                row_groups=[work_item.local_row_group_index],
                columns=scan_columns,
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

        capacities = self._prefetch_queue_capacities(active_workers)
        byte_capacity = max(
            1, self.split.reader.max_prefetch_bytes // active_workers
        )
        slots = [
            _PrefetchSlot(
                index=index,
                queue=queue.Queue(maxsize=capacity),
                byte_budget=_ByteBudget(byte_capacity),
            )
            for index, capacity in enumerate(capacities)
        ]
        slot_for_item: dict[int, _PrefetchSlot] = {}
        free_slots: queue.Queue[int] = queue.Queue()
        for index in range(active_workers):
            free_slots.put(index)
        assignment_condition = threading.Condition()
        next_assign_index = 0
        assignment_error: BaseException | None = None

        def assign_available_slots() -> None:
            nonlocal next_assign_index
            while next_assign_index < len(work_items) and not free_slots.empty():
                if stop_event.is_set():
                    return
                try:
                    slot_index = free_slots.get_nowait()
                except queue.Empty:
                    return
                work_item = work_items[next_assign_index]
                slot = slots[slot_index]

                def run_worker(item: _RowGroupWorkItem = work_item, target_slot: _PrefetchSlot = slot) -> None:
                    self._row_group_worker(item, target_slot, stop_event)

                slot.thread = threading.Thread(
                    target=run_worker,
                    name=f"parquet-prefetch-{next_assign_index}",
                    daemon=True,
                )
                slot.thread.start()
                slot_for_item[next_assign_index] = slot
                next_assign_index += 1

        def assignment_loop() -> None:
            nonlocal assignment_error
            try:
                while next_assign_index < len(work_items) and not stop_event.is_set():
                    with assignment_condition:
                        assign_available_slots()
                        if next_assign_index >= len(work_items) or stop_event.is_set():
                            break
                        assignment_condition.wait(timeout=0.01)
            except BaseException as error:
                assignment_error = error
                stop_event.set()

        assignment_thread = threading.Thread(target=assignment_loop, name="parquet-prefetch-assign", daemon=True)
        assignment_thread.start()

        try:
            for work_index, _work_item in enumerate(work_items):
                if stop_event.is_set():
                    return
                if assignment_error is not None:
                    raise assignment_error
                while work_index not in slot_for_item:
                    if stop_event.is_set():
                        return
                    if assignment_error is not None:
                        raise assignment_error
                    with assignment_condition:
                        assign_available_slots()
                        assignment_condition.notify_all()
                    if work_index not in slot_for_item:
                        time.sleep(0.001)

                slot = slot_for_item[work_index]
                while True:
                    if stop_event.is_set() and slot.queue.empty():
                        return
                    try:
                        item = slot.queue.get(timeout=0.1)
                    except queue.Empty:
                        if slot.thread is not None and not slot.thread.is_alive() and slot.queue.empty():
                            if slot.error is not None:
                                raise slot.error
                            break
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

                if slot.thread is not None:
                    slot.thread.join()
                slot.thread = None
                slot.error = None
                _drain_prefetch_slot(slot)
                free_slots.put(slot.index)
                with assignment_condition:
                    assign_available_slots()
                    assignment_condition.notify_all()
        finally:
            stop_event.set()
            for slot in slots:
                slot.byte_budget.wake_all()
            with assignment_condition:
                assignment_condition.notify_all()
            assignment_thread.join()
            for slot in slots:
                if slot.thread is not None:
                    slot.thread.join()
                _drain_prefetch_slot(slot)
                slot.thread = None
                slot.error = None
            slot_for_item.clear()

    def _iter_row_group_record_batches(self, stop_event: threading.Event) -> Iterator[Any]:
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
                batch_size=batch_size,
                columns=scan_columns,
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

        capacities = self._prefetch_queue_capacities(active_workers)
        byte_capacity = max(
            1, self.split.reader.max_prefetch_bytes // active_workers
        )
        slots = [
            _PrefetchSlot(
                index=index,
                queue=queue.Queue(maxsize=capacity),
                byte_budget=_ByteBudget(byte_capacity),
            )
            for index, capacity in enumerate(capacities)
        ]
        slot_for_item: dict[int, _PrefetchSlot] = {}
        free_slots: queue.Queue[int] = queue.Queue()
        for index in range(active_workers):
            free_slots.put(index)
        assignment_condition = threading.Condition()
        next_assign_index = 0
        assignment_error: BaseException | None = None

        def assign_available_slots() -> None:
            nonlocal next_assign_index
            while next_assign_index < len(paths) and not free_slots.empty():
                if stop_event.is_set():
                    return
                try:
                    slot_index = free_slots.get_nowait()
                except queue.Empty:
                    return
                ref = paths[next_assign_index]
                slot = slots[slot_index]

                def run_worker(
                    item: ParquetInputRef = ref,
                    target_slot: _PrefetchSlot = slot,
                ) -> None:
                    self._file_worker(item, target_slot, stop_event)

                slot.thread = threading.Thread(
                    target=run_worker,
                    name=f"parquet-file-prefetch-{next_assign_index}",
                    daemon=True,
                )
                slot.thread.start()
                slot_for_item[next_assign_index] = slot
                next_assign_index += 1

        def assignment_loop() -> None:
            nonlocal assignment_error
            try:
                while next_assign_index < len(paths) and not stop_event.is_set():
                    with assignment_condition:
                        assign_available_slots()
                        if next_assign_index >= len(paths) or stop_event.is_set():
                            break
                        assignment_condition.wait(timeout=0.01)
            except BaseException as error:
                assignment_error = error
                stop_event.set()

        assignment_thread = threading.Thread(
            target=assignment_loop,
            name="parquet-file-prefetch-assign",
            daemon=True,
        )
        assignment_thread.start()

        try:
            for work_index, _ref in enumerate(paths):
                if stop_event.is_set():
                    return
                if assignment_error is not None:
                    raise assignment_error
                while work_index not in slot_for_item:
                    if stop_event.is_set():
                        return
                    if assignment_error is not None:
                        raise assignment_error
                    with assignment_condition:
                        assign_available_slots()
                        assignment_condition.notify_all()
                    if work_index not in slot_for_item:
                        time.sleep(0.001)

                slot = slot_for_item[work_index]
                while True:
                    if stop_event.is_set():
                        return
                    if slot.error is not None:
                        raise slot.error
                    try:
                        item = slot.queue.get(timeout=0.1)
                    except queue.Empty:
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
                if slot.thread is not None:
                    slot.thread.join(timeout=0.01)
                slot.thread = None
                slot.error = None
                free_slots.put(slot.index)
                with assignment_condition:
                    assignment_condition.notify_all()
                del slot_for_item[work_index]
        finally:
            stop_event.set()
            assignment_thread.join(timeout=1.0)
            for slot in slots:
                if slot.thread is not None and slot.thread.is_alive():
                    slot.thread.join(timeout=0.5)
                _drain_prefetch_slot(slot)
            slot_for_item.clear()

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

    def _iter_dataset_record_batches(self, stop_event: threading.Event) -> Iterator[Any]:
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
        return ScanStats(files=len(self.paths), record_batches=record_batches, rows=rows)


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
            raise ValueError(
                f"request_value_maps.{column} must be a non-empty object"
            )
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


def _coarse_scene_plan(
    options: Mapping[str, Any],
    request_columns: set[str],
) -> _CoarseScenePlan | None:
    raw_ids = options.get("search_scene_ids")
    if raw_ids is None:
        return None
    if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Sequence) or not raw_ids:
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
    unlisted_policy = str(
        options.get("unlisted_scene_policy", "recommendation")
    )
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


def _normalize_optional_outer_list(value: Any) -> list[Any]:
    """Map top-level null/[] to an empty Python list for optional list payloads.

    Only the outermost list/tuple is normalized. Nested memberships such as
    ``[[], [0]]`` are preserved so orphan UPS tokens can still be rejected.
    """

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    raise TypeError(
        f"optional list-valued field must be null or a list/tuple, got {type(value).__name__}"
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
        if isinstance(value, (list, tuple)):
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
        if isinstance(value, (list, tuple)):
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
                name: stats.to_payload() for name, stats in sorted(self.scalar_stats.items())
            },
            "bag_stats": {
                name: stats.to_payload() for name, stats in sorted(self.bag_stats.items())
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
    if validate_contract and (
        isinstance(value, bool) or not isinstance(value, int) or value < 0
    ):
        raise ValueError(
            f"column {column!r} contains invalid request index {value!r} "
            f"at raw row {row_index}"
        )
    return value


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
        observed[column] = len(
            _as_list(row[column], column=column, row_index=raw_row)
        )
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
        item is None or (isinstance(item, (list, tuple)) and len(item) == 1)
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

    Structure checks (empty membership / unknown / duplicate request) stay on
    even when payload diagnostics are skipped under trusted_input.
    """

    if validate_structure is None:
        validate_structure = validate_contract
    selected: dict[int, list[int]] = {request: [] for request in known_requests}
    for token_position, raw_membership in enumerate(memberships):
        members = (
            list(raw_membership)
            if isinstance(raw_membership, (list, tuple))
            else [raw_membership]
        )
        if validate_structure and not members:
            raise ValueError(
                f"UPS indices column {index_column!r} has an empty membership "
                f"at raw row {raw_row}, token {token_position}"
            )
        normalized = [
            _request_index(
                value,
                column=index_column,
                row_index=raw_row,
                validate_contract=validate_structure,
            )
            for value in members
        ]
        if validate_structure and len(normalized) != len(set(normalized)):
            raise ValueError(
                f"UPS indices column {index_column!r} repeats a request at raw row "
                f"{raw_row}, token {token_position}"
            )
        if validate_structure:
            unknown = sorted(set(normalized) - known_requests)
            if unknown:
                raise ValueError(
                    f"UPS indices column {index_column!r} references requests without "
                    f"context at raw row {raw_row}, token {token_position}: {unknown}"
                )
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
        if validate_structure and len(items) != expected_length:
            raise ValueError(
                f"UPS column {column!r} length {len(items)} does not match its indices "
                f"length {expected_length} at raw row {raw_row}"
            )
        if max_length is not None:
            selected_positions = selected_positions[:max_length]
        items = [items[position] for position in selected_positions]
    elif max_length is not None:
        items = items[:max_length]

    if validated_flat:
        return items

    if not validate_payload:
        return [
            item[0] if isinstance(item, (list, tuple)) else item
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
    """Materialize an Arrow array as Python objects.

    For the common ``list<primitive>`` columns in the aggregated format
    (flattened UPS histories, bag features, candidate list metadata) Arrow's
    generic ``to_pylist`` allocates one intermediate object per element and is
    the dominant adapter cost. When the array is a flat list of a numeric
    primitive with no inner nulls, we slice a single NumPy view per row
    instead, which is ~15x faster and produces byte-identical Python objects
    (``numpy.int64.tolist()``/``float.tolist()`` yield the same ``int``/``float``
    that ``to_pylist`` would). Every case that does not meet these exact
    conditions falls back to ``to_pylist`` so semantics never change.
    """

    if array.offset != 0:
        return array.to_pylist()
    if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type)):
        return array.to_pylist()
    child = array.values
    child_type = child.type
    if not (
        pa.types.is_integer(child_type)
        or pa.types.is_floating(child_type)
    ):
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
            else values[offsets[index] : offsets[index + 1]].tolist()
            for index in range(len(array))
        ]
    return [
        values[offsets[index] : offsets[index + 1]].tolist()
        for index in range(len(array))
    ]


def _adapter_table_to_python(
    table: Any,
    raw_sequence_columns: frozenset[str],
    *,
    validate_contract: bool = True,
) -> tuple[dict[str, list[Any]], frozenset[str]]:
    """Convert a raw table while flattening valid singleton S-token columns."""

    pa, pc, _ds, _pq = _require_pyarrow()
    raw: dict[str, list[Any]] = {}
    flattened: set[str] = set()
    for name in table.column_names:
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
        raw[name] = _arrow_array_to_pylist(pa, array)
    return raw, frozenset(flattened)


def _time_deltas(
    event_times: Sequence[Any],
    request_time: Any,
    *,
    sequence: str,
    raw_row: int,
    transform: str,
    validate_contract: bool = True,
) -> list[float]:
    def _transform_delta(delta: int) -> float:
        if transform == "raw_ms":
            return float(delta)
        if transform == "seconds":
            return float(delta) / 1000.0
        if transform == "log1p_seconds":
            return math.log1p(float(delta) / 1000.0)
        raise RuntimeError(f"unsupported time delta transform {transform!r}")

    if not validate_contract:
        result: list[float] = []
        for event_time in event_times:
            if event_time is None:
                # Non-anchor null times pad to 0.0; anchor-null steps are dropped later.
                result.append(0.0)
                continue
            result.append(_transform_delta(int(request_time) - int(event_time)))
        return result

    if event_times and (
        isinstance(request_time, bool) or not isinstance(request_time, int)
    ):
        raise ValueError(
            f"request time is required to derive {sequence!r} time deltas at raw row {raw_row}"
        )
    if len(event_times) >= 64 and all(event_time is not None for event_time in event_times):
        # Long histories dominate adapter CPU. NumPy performs validation,
        # subtraction, and log1p in native loops instead of one Python/math
        # call per event. Skip when any null times are present.
        try:
            import numpy as np
        except ImportError:
            pass
        else:
            values = np.asarray(event_times)
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
            return [_transform_delta(int(delta)) for delta in deltas.tolist()]
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
    if column in scalar_features or column in label_columns or column in integer_request_columns:
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
        raise ValueError("adapter label_masks must match labels exactly; " + "; ".join(details))
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
        raise ValueError("adapter option 'aligned_multivalue_groups' must be a list of lists")
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
        raise ValueError("multivalue_features contains unknown fields: " + ", ".join(unknown))
    for group in aligned_groups:
        if not group or not set(group) <= bag_features:
            raise ValueError(
                "every aligned_multivalue_groups entry must contain configured multivalue fields"
            )
    if set(time_delta_outputs) - set(ups_types):
        raise ValueError("time_delta_outputs contains an unknown UPS type")
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
            *(() if candidate_position_column is None else (candidate_position_column,)),
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
            if column.startswith(f"{ups}_x_")
            and column != time_delta_outputs.get(ups)
        )
        for ups in ups_types
    }
    sequence_columns = frozenset(
        column
        for columns in sequence_columns_by_type.values()
        for column in columns
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
        (bag_features & context_set)
        | sequence_columns
        | time_delta_columns
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
            *(() if candidate_position_column is None else (candidate_position_column,)),
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
    cardinality_auditor: FeatureCardinalityAuditor | None = None
    if isinstance(runtime_cache, dict):
        cached_auditor = runtime_cache.get("cardinality_auditor")
        if isinstance(cached_auditor, FeatureCardinalityAuditor):
            cardinality_auditor = cached_auditor
    soft_cardinality_audit = bool(
        cardinality_auditor is not None and cardinality_auditor.soft
    )
    raw_sample_already_validated = (
        isinstance(runtime_cache, dict)
        and runtime_cache.get("mdl_rankmixer_raw_sample_validated", False)
    )
    # Soft cardinality audit already walks every configured field; skip the
    # one-row hard warm-up so the first multi-valued scalar does not abort the
    # aggregate report.
    if (
        trusted_input
        and table.num_rows > 0
        and not raw_sample_already_validated
        and not soft_cardinality_audit
    ):
        sample_context = ParquetAdapterContext(
            split_name=str(getattr(context, "split_name", "unknown")),
            required_columns=tuple(context.required_columns),
            options=context.options,
            trusted_input=False,
            _runtime_cache={"mdl_rankmixer_plan": plan},
        )
        # Validate only one physical Parquet row. The full table is converted
        # below with diagnostics disabled.
        adapt_mdl_rankmixer_parquet(table.slice(0, 1), context=sample_context)
        if isinstance(runtime_cache, dict):
            runtime_cache["mdl_rankmixer_raw_sample_validated"] = True
    first_batch_already_adapted = (
        isinstance(runtime_cache, dict)
        and runtime_cache.get("mdl_rankmixer_first_batch_adapted", False)
    )
    complete_label_contract = not label_masks and not any(
        label_missing_values.values()
    )
    # Payload diagnostics may stay off under trusted_input. Structure checks that
    # guarantee lossless UPS/candidate interpretation stay on for every row.
    validate_payload = not trusted_input and (
        not first_batch_already_adapted or not complete_label_contract
    )
    validate_structure = True
    validate_row_contract = validate_structure
    output: dict[str, list[Any]] = {column: [] for column in required}
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
            label_arrays[column] = [
                _scalarize(
                    value,
                    column=column,
                    raw_row=raw_row,
                    logical_row=candidate_index,
                    validate_contract=validate_row_contract,
                )
                for candidate_index, value in enumerate(values)
            ]

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
                if (
                    coarse_scene is not None
                    and column == coarse_scene.raw_scene_column
                ):
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
            for ups in ups_types:
                selected_positions = (
                    membership_positions[ups][request_index] if is_agg else None
                )
                expected_length = membership_lengths.get(ups)
                for column in sequence_columns_by_type[ups]:
                    if validate_row_contract and column not in row:
                        raise ValueError(f"missing UPS column {column!r}")
                    cached_sequences[column] = _select_sequence(
                        row[column],
                        selected_positions,
                        expected_length=expected_length,
                        column=column,
                        raw_row=raw_row,
                        validated_flat=column in validated_flat_sequence_columns,
                        max_length=sequence_max_lengths.get(ups),
                        validate_structure=validate_structure,
                        validate_payload=validate_payload,
                    )
                if ups in time_delta_outputs:
                    raw_time_column = f"{ups}_x_time"
                    if validate_row_contract and raw_time_column not in row:
                        raise ValueError(f"missing UPS time column {raw_time_column!r}")
                    event_times = _select_sequence(
                        row[raw_time_column],
                        selected_positions,
                        expected_length=expected_length,
                        column=raw_time_column,
                        raw_row=raw_row,
                        validated_flat=(
                            raw_time_column in validated_flat_sequence_columns
                        ),
                        max_length=sequence_max_lengths.get(ups),
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
            normalized = []
            for candidate_index, value in enumerate(item_arrays[column]):
                normalized.append(
                    _bag_value(
                        value,
                        column=column,
                        raw_row=raw_row,
                        logical_row=candidate_index,
                        validate_contract=validate_row_contract,
                        auditor=cardinality_auditor,
                    )
                    if column in bag_features
                    else _scalarize(
                        value,
                        column=column,
                        raw_row=raw_row,
                        logical_row=candidate_index,
                        validate_contract=validate_row_contract,
                        auditor=cardinality_auditor,
                    )
                )
            normalized_items[column] = normalized

        if validate_structure:
            for group in aligned_groups:
                for candidate_index in range(candidate_count):
                    lengths = {
                        column: len(normalized_items[column][candidate_index])
                        for column in group
                        if isinstance(
                            normalized_items[column][candidate_index],
                            (list, tuple),
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
            if mask_column is None and not label_missing_values[task]:
                # Complete-label rows are checked by
                # ``_validate_complete_label_contract`` on every flat batch.
                normalized_labels[column] = values
                continue
            task_labels: list[int | None] = []
            task_masks: list[int] = []
            for candidate_index, value in enumerate(values):
                if _is_missing_label(value, label_missing_values[task]):
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
                    raw_table if take == raw_table.num_rows else raw_table.slice(0, take)
                )
                auditor.note_raw_rows(table.num_rows)
                result = adapter(table, context=context)
                for _flat in _normalize_adapter_result(result, adapter_name, split_name):
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
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
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


def _normalize_adapter_result(result: Any, adapter_name: str, split_name: str) -> Iterator[Any]:
    pa, _pc, _ds, _pq = _require_pyarrow()
    if isinstance(result, pa.Table):
        yield result
        return
    if isinstance(result, RuntimeIterable):
        for index, table in enumerate(result):
            if not isinstance(table, pa.Table):
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
    return {
        field.name
        for field in table.schema
        if _is_arrow_list_type(pa, field.type)
    }


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
        if not (
            pa.types.is_integer(array.type)
            or pa.types.is_floating(array.type)
        ):
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
    raw_batch_index = 0
    validated_static_contract = False
    # Complete-label paths omit masks; every flat batch must still prove 0/1/no-null.
    complete_label_contract = not bool(scanner.split.label_masks)
    for raw_table in scanner.iter_tables():
        if max_batches is not None and raw_batch_index >= max_batches:
            break
        raw_batch_index += 1
        if counters is not None:
            counters.raw_record_batches += 1
            counters.raw_rows += raw_table.num_rows
        try:
            result = adapter(raw_table, context=context)
            flat_tables = _normalize_adapter_result(result, adapter_name, split_name)
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
    if not all(
        pa.types.is_dictionary(chunk.type)
        for chunk in chunked.chunks
    ):
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
        index_array = indices.cast(pa.int64()) if indices.type != pa.int64() else indices
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
    invalid_bounds = (
        (minimum is not None and int(minimum) < 0)
        or (maximum is not None and int(maximum) >= encoding.num_buckets)
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


def _tensorize_categorical_bag(
    config: AppConfig,
    feature: FeatureConfig,
    table: Any,
    vocab_maps: dict[str, dict[str, int]],
    *,
    validate_prehashed_nonzero: bool = True,
) -> dict[str, Tensor]:
    """Encode one list-valued categorical feature as flat values + lengths.

    Truncation follows ``feature.max_length`` / ``truncation``. The returned
    ``values`` tensor is CSR-like ``[sum(lengths)]`` (no pad slots); mean-pool
    reconstructs per-row segments from ``lengths``.
    """

    if feature.pooling != "mean":
        raise TypeError("_tensorize_categorical_bag requires pooling=mean")
    categorical_input = _effective_categorical_input(
        config,
        config.resolved.categorical_input_by_name[feature.name],
    )
    array = _normalized_list_array(table, feature.source)
    offsets = _list_offsets_tensor(array)
    base = int(offsets[0].item()) if offsets.numel() else 0
    normalized_offsets = offsets - base
    raw_lengths = normalized_offsets[1:] - normalized_offsets[:-1]
    lengths = raw_lengths
    if feature.max_length is not None:
        lengths = torch.clamp(raw_lengths, max=feature.max_length)
    if feature.truncation == "tail":
        starts = normalized_offsets[1:] - lengths
    else:
        starts = normalized_offsets[:-1]
    max_length = int(lengths.max().item()) if lengths.numel() else 0
    total_values = int(offsets[-1].item()) - base if offsets.numel() else 0
    flat = array.values.slice(base, total_values)
    if isinstance(categorical_input.encoding, ResolvedIdentityEncoding):
        encoded = _identity_array_tensor(flat, categorical_input)
    elif isinstance(categorical_input.encoding, ResolvedPreHashedEncoding):
        encoded = _pre_hashed_array_tensor(
            flat,
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
                for value in flat.to_pylist()
            ],
            dtype=torch.long,
        )
    # Truncate via the padded gather, then compact to CSR-like flat values so
    # embedding lookup skips pad slots and mean-pool uses offsets/lengths.
    padded = _gather_padded_sequence(
        encoded,
        starts,
        lengths,
        max_length,
        0,
    )
    if max_length == 0 or not lengths.numel():
        flat = encoded.new_empty((0,), dtype=encoded.dtype)
    else:
        positions = torch.arange(max_length, dtype=torch.long)
        flat = padded[positions.unsqueeze(0) < lengths.unsqueeze(1)]
    return {"values": flat, "lengths": lengths}


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
        raise ValueError(f"dense feature expected {dimension} values, got scalar {value!r}")
    if len(value) != dimension:
        raise ValueError(f"dense feature expected {dimension} values, got {len(value)}")
    return [0.0 if item is None else float(item) for item in value], 1.0


def _tensorize_dense(feature: FeatureConfig, values: list[Any]) -> Tensor | dict[str, Tensor]:
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


def _numeric_column_with_presence(
    table: Any, column: str
) -> tuple[Tensor, Tensor]:
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


def _tensorize_dense_column(feature: FeatureConfig, table: Any) -> Tensor | dict[str, Tensor]:
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
    if sequence.max_length is None or length <= sequence.max_length:
        return 0, length
    if sequence.truncation == "tail":
        return length - sequence.max_length, length
    return 0, sequence.max_length


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


def _gather_padded_sequence(
    values: Tensor,
    starts: Tensor,
    lengths: Tensor,
    max_length: int,
    padding_value: int | float,
) -> Tensor:
    output_shape = (int(lengths.numel()), max_length, *values.shape[1:])
    if max_length == 0:
        return values.new_full(output_shape, padding_value)
    positions = torch.arange(max_length, dtype=torch.long).unsqueeze(0)
    valid = positions < lengths.unsqueeze(1)
    indices = starts.unsqueeze(1) + positions
    safe_indices = indices.clamp(min=0, max=max(int(values.size(0)) - 1, 0))
    gathered = values.index_select(0, safe_indices.reshape(-1)).view(output_shape)
    mask = valid
    for _ in values.shape[1:]:
        mask = mask.unsqueeze(-1)
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
        arrays, reference_offsets, reference_base, reference_stop = (
            _compact_direct_sequence_by_anchor(
                arrays,
                reference_offsets,
                reference_base,
                reference_stop,
                anchor_field=sequence.null_anchor_field,
            )
        )

    raw_lengths = reference_offsets[1:] - reference_offsets[:-1]
    lengths = raw_lengths
    if sequence.max_length is not None:
        lengths = torch.clamp(raw_lengths, max=sequence.max_length)
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
        raise ValueError(f"dense sequence field expected {dimension} values, got scalar {value!r}")
    if len(value) != dimension:
        raise ValueError(f"dense sequence field expected {dimension} values, got {len(value)}")
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
    values_by_field = {field.name: _column_values(table, field.source) for field in sequence.fields}
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
        row_length = len(next(iter(raw_items_by_field.values()))) if raw_items_by_field else 0
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
                [_dense_vector(item, field.dimension) for item in row]
                for row in rows
            ]
            zero = [0.0] * field.dimension
            padded_dense = [row + [zero] * (max_length - len(row)) for row in encoded_dense]
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
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_values):
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
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
        raise ValueError("auto scenario values must be integers")
    ordered = tuple(sorted({int(value) for value in values}))
    if len(ordered) != len(values):
        raise ValueError("auto scenario values must be unique")
    if len(ordered) > scenario.max_discovered:
        raise ValueError(
            f"auto scenario values exceed scenarios.max_discovered={scenario.max_discovered}"
        )
    # Pure RankMixer/OneTrans use raw scenes only for batch routing and
    # per-scene evaluation; they do not instantiate MDL domain tokens or
    # scenario-scoped embedding tables.  Resolving their scenario names is
    # therefore only a metadata operation.
    if config.model.name not in {"mdl_rankmixer", "mdl_onetrans"}:
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
            replace(template_feature, name=feature_name(value))
            for value in ordered
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
        raise ValueError(f"scenario value must be a name or integer id at row {row_index}, got bool")
    if isinstance(value, Integral):
        raw_name = str(int(value))
        if source_encoding != "index" and raw_name in scenario_to_id:
            return scenario_to_id[raw_name]
        if source_encoding == "raw":
            raise ValueError(
                f"unknown raw scenario id {int(value)} at row {row_index}"
            )
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
            raise ValueError("scenarios.source is required when multiple scenarios are configured")
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
        row_indices.append([
            _encode_scenario_item(
                item,
                scenario_to_id,
                scenario_count,
                row_index,
                config.scenarios.source_encoding,
            )
            for item in items
        ])

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
    return ["" if value is None else str(value) for value in _column_values(table, source)]


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
) -> tuple[Any, Tensor] | None:
    """Select one physical row per request and map candidates back to it."""

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
    if len(unique_positions) == table.num_rows:
        return None
    selected = _safe_table_take(table, unique_positions)
    return selected, torch.tensor(candidate_to_request, dtype=torch.long)


def _indexed_request_value(value: Any, row_indices: Tensor) -> dict[str, Any]:
    if isinstance(value, dict):
        return {**value, "row_indices": row_indices}
    return {"values": value, "row_indices": row_indices}


def table_to_feature_batch(
    config: AppConfig,
    table: Any,
    vocab_maps: dict[str, dict[str, int]],
    require_labels: bool = True,
    include_group_id: bool = True,
    split: ParquetSplitConfig | None = None,
) -> FeatureBatch:
    """Convert one Arrow table into the exact structure consumed by the model.

    Labels are optional for inference. When labels exist but explicit masks do
    not, every label is treated as observed. Feature and label ordering follows
    configuration order so it remains stable across training and evaluation.
    Callers processing a non-training split must pass it explicitly.
    """
    active_split = config.data.train if split is None else split
    batch_size = table.num_rows
    deduplication = _request_deduplication_plan(active_split, table)
    request_table, request_row_indices = (
        (table, None) if deduplication is None else deduplication
    )
    adapter_options = (
        {} if active_split.adapter is None else active_split.adapter.options
    )
    context_sources = {
        str(source) for source in adapter_options.get("context_features", ())
    }
    validate_prehashed_nonzero = active_split.reader.validate_prehashed_nonzero
    validate_sequence_alignment = not active_split.reader.trusted_input
    features: dict[str, Any] = {}
    for feature in config.features:
        request_level = (
            request_row_indices is not None and feature.source in context_sources
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
    if label_columns and all(column in table.column_names for column in label_columns.values()):
        label_names = list(label_columns)
        labels = torch.stack(
            [_numeric_column_tensor(table, label_columns[name], torch.float32) for name in label_names],
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
        group_id=_group_ids(active_split, table, batch_size) if include_group_id else [],
        prediction_keys=_prediction_keys(active_split, table),
    )


# ---------------------------------------------------------------------------
# Device transfer: pin memory and move tensors to GPU
# ---------------------------------------------------------------------------


def _map_feature_value(value: Any, tensor_fn: Callable[[Tensor], Tensor]) -> Any:
    """Apply a device or memory operation recursively to nested tensor leaves."""
    if isinstance(value, dict):
        return {key: _map_feature_value(child, tensor_fn) for key, child in value.items()}
    if isinstance(value, Tensor):
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
        if isinstance(value, Tensor):
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
        if isinstance(value, Tensor):
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
        offset = 0
        for tensor in tensors:
            count = tensor.numel()
            target = buffer.narrow(0, offset, count)
            target.copy_(tensor.reshape(-1))
            replacements[id(tensor)] = target.view(tensor.shape)
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
    """Pin CPU tensors so CUDA transfers can use the non-blocking path."""
    if coalesce_tensors:
        return _coalesce_feature_batch(batch, pin_memory=True)
    return FeatureBatch(
        features={
            key: _map_feature_value(value, lambda tensor: tensor.pin_memory())
            for key, value in batch.features.items()
        },
        labels=None if batch.labels is None else batch.labels.pin_memory(),
        label_mask=None if batch.label_mask is None else batch.label_mask.pin_memory(),
        scenario_id=batch.scenario_id.pin_memory(),
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
        labels=None if batch.labels is None else batch.labels.to(device, non_blocking=non_blocking),
        label_mask=(
            None
            if batch.label_mask is None
            else batch.label_mask.to(device, non_blocking=non_blocking)
        ),
        scenario_id=batch.scenario_id.to(device, non_blocking=non_blocking),
        group_id=batch.group_id,
        prediction_keys=batch.prediction_keys,
    )
