from __future__ import annotations  # Defer annotation evaluation for forward references.

"""Parquet-to-PyTorch data pipeline.

This module owns the complete input path: it discovers and shards Parquet
files, streams Arrow batches, encodes configured features, and builds the
``FeatureBatch`` objects consumed by training and inference.
"""

from collections import defaultdict
from collections.abc import Iterable as RuntimeIterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import fnmatch
from hashlib import sha256
from itertools import islice
import glob
import importlib
import logging
from numbers import Integral
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
    SequenceConfig,
)
from .features import encode_categorical_sequence_field, encode_categorical_values

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
            "parquet-native data loading requires pyarrow; install requirements.txt"
        ) from error
    return pa, pc, ds, pq


def _require_pyarrow_fs() -> Any:
    """Import PyArrow filesystem support only when input discovery is used."""
    try:
        import pyarrow.fs as pafs
    except ImportError as error:
        raise RuntimeError(
            "parquet-native data loading requires pyarrow filesystem support; "
            "install requirements.txt"
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


def parquet_schema(path: str | Path | ParquetInputRef) -> Any:
    """Read Parquet schema metadata only; does not scan row data."""
    _pa, _pc, _ds, pq = _require_pyarrow()
    ref = _coerce_parquet_input_ref(path)
    return pq.read_schema(ref.fs_path, filesystem=ref.filesystem)


def validate_matching_schemas(paths: Iterable[str | Path | ParquetInputRef]) -> str:
    """Require identical schemas across files; return the shared fingerprint."""
    refs = [_coerce_parquet_input_ref(path) for path in paths]
    if not refs:
        raise ValueError("paths must not be empty")
    fingerprints = {ref: schema_fingerprint(parquet_schema(ref)) for ref in refs}
    expected = next(iter(fingerprints.values()))
    mismatched = [
        ref.canonical_uri
        for ref, fingerprint in fingerprints.items()
        if fingerprint != expected
    ]
    if mismatched:
        raise ValueError("parquet schema mismatch: " + ", ".join(mismatched))
    return expected


def _configure_pyarrow_threads(pa: Any, num_workers: int) -> None:
    """Align PyArrow CPU/IO threads with ``reader.num_workers`` when set."""
    if num_workers <= 0:
        return
    if hasattr(pa, "set_cpu_count"):
        pa.set_cpu_count(num_workers)
    if hasattr(pa, "set_io_thread_count"):
        pa.set_io_thread_count(num_workers)


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
    columns.update(split.labels.values())
    columns.update(split.label_masks.values())
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
        return list(split.adapter.input_columns or [])
    return flat_columns


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


def _metadata_worker_count(num_workers: int, file_count: int) -> int:
    """Cap parallel metadata readers by file count and a hard limit of 16."""
    configured = num_workers if num_workers > 0 else min(8, os.cpu_count() or 1)
    return min(file_count, configured, 16)


def _load_file_metadata_cache(ref: ParquetInputRef, scan_columns: list[str] | None) -> _FileMetadataCache:
    """Read row-group row counts and compressed-byte weights from the footer only."""
    _pa, _pc, _ds, pq = _require_pyarrow()
    parquet_file = pq.ParquetFile(ref.fs_path, filesystem=ref.filesystem)
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
            if column_meta.total_compressed_size is None or column_meta.total_compressed_size < 0:
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


def _load_metadata_cache(
    paths: list[ParquetInputRef],
    scan_columns: list[str] | None,
    num_workers: int,
) -> dict[ParquetInputRef, _FileMetadataCache]:
    """Load per-file footer metadata, in parallel when beneficial."""
    worker_count = _metadata_worker_count(num_workers, len(paths))
    metadata_by_path: dict[ParquetInputRef, _FileMetadataCache] = {}
    if worker_count <= 1:
        for ref in paths:
            metadata_by_path[ref] = _load_file_metadata_cache(ref, scan_columns)
        return metadata_by_path

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_load_file_metadata_cache, ref, scan_columns): ref for ref in paths}
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
    thread: threading.Thread | None = None
    error: BaseException | None = None


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
    ) -> None:
        self.split = split
        self.columns = columns
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
        validate_matching_schemas(self.all_paths)
        pa, _pc, _ds, _pq = _require_pyarrow()
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
        return self.split.reader.batch_size_rows or default

    def _get_metadata_cache(self) -> dict[ParquetInputRef, _FileMetadataCache]:
        if self._metadata_cache is None:
            self._metadata_cache = _load_metadata_cache(
                self.all_paths,
                self._scan_columns(),
                self.split.reader.num_workers,
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
        """Bound concurrent row-group readers by prefetch budget and a cap of 4."""
        prefetch_batches = self.split.reader.prefetch_batches
        if prefetch_batches <= 0:
            return 0
        num_workers = self.split.reader.num_workers
        worker_budget = num_workers if num_workers > 0 else 4
        return min(row_group_count, prefetch_batches, worker_budget, 4)

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
            parquet_file = pq.ParquetFile(work_item.input_ref.fs_path, filesystem=work_item.input_ref.filesystem)
            batch_iterator = iter(
                parquet_file.iter_batches(
                    batch_size=batch_size,
                    row_groups=[work_item.local_row_group_index],
                    columns=scan_columns,
                    use_threads=True,
                )
            )
            try:
                while not stop_event.is_set():
                    try:
                        batch = next(batch_iterator)
                    except StopIteration:
                        break
                    if stop_event.is_set():
                        return
                    yield batch
            finally:
                close = getattr(batch_iterator, "close", None)
                if callable(close):
                    close()

    def _row_group_worker(
        self,
        work_item: _RowGroupWorkItem,
        slot: _PrefetchSlot,
        stop_event: threading.Event,
    ) -> None:
        """Background worker: stream one row group into a bounded prefetch queue."""
        batch_iterator: Iterator[Any] | None = None
        try:
            _pa, _pc, _ds, pq = _require_pyarrow()
            batch_size = self._reader_batch_size(default=65536)
            scan_columns = self._scan_columns()
            parquet_file = pq.ParquetFile(work_item.input_ref.fs_path, filesystem=work_item.input_ref.filesystem)
            batch_iterator = iter(
                parquet_file.iter_batches(
                    batch_size=batch_size,
                    row_groups=[work_item.local_row_group_index],
                    columns=scan_columns,
                    use_threads=True,
                )
            )
            while not stop_event.is_set():
                try:
                    batch = next(batch_iterator)
                except StopIteration:
                    break
                if stop_event.is_set():
                    return
                if not _put_queue_item(slot.queue, batch, stop_event):
                    return
        except BaseException as error:
            slot.error = error
        finally:
            if batch_iterator is not None:
                close = getattr(batch_iterator, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException as error:
                        if slot.error is None:
                            slot.error = error
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
        slots = [
            _PrefetchSlot(index=index, queue=queue.Queue(maxsize=capacity))
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
                    yield item

                if slot.thread is not None:
                    slot.thread.join()
                slot.thread = None
                slot.error = None
                while not slot.queue.empty():
                    slot.queue.get_nowait()
                free_slots.put(slot.index)
                with assignment_condition:
                    assign_available_slots()
                    assignment_condition.notify_all()
        finally:
            stop_event.set()
            with assignment_condition:
                assignment_condition.notify_all()
            assignment_thread.join()
            for slot in slots:
                if slot.thread is not None:
                    slot.thread.join()
                while not slot.queue.empty():
                    slot.queue.get_nowait()
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

    def _iter_dataset_record_batches(self, stop_event: threading.Event) -> Iterator[Any]:
        """Scan via PyArrow Dataset API (file sharding or single-rank workloads)."""
        _pa, _pc, ds, _pq = _require_pyarrow()
        if not self.paths:
            return
        filesystem = self.paths[0].filesystem
        dataset = ds.dataset(
            [ref.fs_path for ref in self.paths],
            format="parquet",
            filesystem=filesystem,
        )
        scanner_kwargs: dict[str, Any] = {
            "columns": self._scan_columns(),
            "use_threads": True,
            "batch_readahead": max(1, self.split.reader.prefetch_batches),
            "fragment_readahead": max(1, self.split.reader.prefetch_batches),
        }
        if self.split.reader.batch_size_rows is not None:
            scanner_kwargs["batch_size"] = self.split.reader.batch_size_rows
        scanner = dataset.scanner(**scanner_kwargs)
        for batch in scanner.to_batches():
            if stop_event.is_set():
                return
            yield batch

    def iter_record_batches(self) -> Iterator[Any]:
        """Return a closeable iterator so early exits also stop prefetch workers."""
        stop_event = threading.Event()
        if not self.paths and not self._uses_lpt_row_group_sharding():
            return _ClosableIterator(iter(()), stop_event)

        def generator() -> Iterator[Any]:
            if self._uses_lpt_row_group_sharding():
                yield from self._iter_row_group_record_batches(stop_event)
            else:
                yield from self._iter_dataset_record_batches(stop_event)

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
    )


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
    pa, _pc, _ds, _pq = _require_pyarrow()
    for sequence in config.sequences:
        for field in sequence.fields:
            arrow_type = table.schema.field(field.source).type
            if not _is_arrow_list_type(pa, arrow_type):
                raise ValueError(
                    f"adapter output for split {split_name!r} column {field.source!r} "
                    f"must be a list column because it backs sequence {sequence.name!r}."
                )
        if len(sequence.fields) <= 1:
            continue
        values_by_field = {field.name: _column_values(table, field.source) for field in sequence.fields}
        row_count = table.num_rows
        for row_index in range(row_count):
            expected_length: int | None = None
            expected_field: str | None = None
            for field in sequence.fields:
                items = _coerce_sequence_items(values_by_field[field.name][row_index])
                if expected_length is None:
                    expected_length = len(items)
                    expected_field = field.name
                    continue
                if len(items) != expected_length:
                    raise ValueError(
                        f"adapter output for split {split_name!r} sequence {sequence.name!r} "
                        f"has misaligned row {row_index}: field {field.name!r} length "
                        f"{len(items)} != field {expected_field!r} length {expected_length}."
                    )


def _validate_flat_table_contract(
    config: AppConfig,
    split: ParquetSplitConfig,
    split_name: str,
    table: Any,
    required_columns: list[str],
) -> None:
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
    scenario_columns = {config.scenarios.source} if config.scenarios.source else set()
    allowed_list_columns = sequence_columns | dense_vector_columns | scenario_columns
    unexpected_list_columns = sorted(
        column
        for column in _table_list_columns(table)
        if column in required_columns and column not in allowed_list_columns
    )
    if unexpected_list_columns:
        raise ValueError(
            f"adapter output for split {split_name!r} has list-valued non-sequence column(s): "
            + ", ".join(unexpected_list_columns)
            + ". Only configured sequence fields, dense features with dimension > 1, "
            "and scenario masks may use list-valued cells."
        )

    _validate_sequence_contract(config, table, split_name)


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
                _validate_flat_table_contract(config, scanner.split, split_name, flat_table, required_columns)
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
) -> Iterator[Any]:
    """Yield flat Arrow tables for any configured Parquet split.

    This is the single model-facing table entry point. ``flat_parquet`` uses an
    identity adapter; ``adapter_parquet`` applies the configured external
    adapter before validating the flat contract.
    """
    split = _split_for_name(config, split_name)
    required_columns = required_columns_for_split(config, split, extra_columns=extra_columns)
    scanner = ParquetScanner(
        split,
        _scan_columns_for_split(split, required_columns),
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
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
    scanner = ParquetScanner(split, _scan_columns_for_split(split, required_columns))
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
    return table[column].combine_chunks()


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


# --- Categorical encoding ---


def _tensorize_categorical(
    config: AppConfig,
    feature: FeatureConfig,
    table: Any,
    vocab_maps: dict[str, dict[str, int]],
) -> Tensor:
    """Build a rank-one integer tensor for a configured categorical feature."""
    categorical_input = config.resolved.categorical_input_by_name[feature.name]
    unseen_policy = config.vocab_strategy.defaults.unseen_policy
    encoded = encode_categorical_values(
        _column_values(table, categorical_input.source),
        categorical_input,
        vocab_maps,
        unseen_policy,
    )
    return torch.tensor(encoded, dtype=torch.long)


# --- Dense features ---


def _dense_feature_value(value: Any, dimension: int) -> float | list[float]:
    """Normalize one dense value and reject shapes that would silently broadcast."""
    if value is None:
        return 0.0 if dimension == 1 else [0.0] * dimension
    if dimension == 1:
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"dense feature expected 1 value, got {len(value)}")
            value = value[0]
        return 0.0 if value is None else float(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"dense feature expected {dimension} values, got scalar {value!r}")
    if len(value) != dimension:
        raise ValueError(f"dense feature expected {dimension} values, got {len(value)}")
    return [0.0 if item is None else float(item) for item in value]


def _tensorize_dense(feature: FeatureConfig, values: list[Any]) -> Tensor:
    """Build a ``[batch, dim]`` float tensor from already-extracted Python values."""
    normalized = [_dense_feature_value(value, feature.dimension) for value in values]
    return torch.tensor(normalized, dtype=torch.float32)


def _tensorize_dense_column(feature: FeatureConfig, table: Any) -> Tensor:
    # Scalar columns use the Arrow/NumPy fast path; vector columns require
    # row-level shape validation before tensor construction.
    if feature.dimension == 1:
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


def _sequence_bounds(length: int, sequence: SequenceConfig) -> tuple[int, int]:
    """Return the configured head/tail truncation window for one sequence."""
    if sequence.max_length is None or length <= sequence.max_length:
        return 0, length
    if sequence.truncation == "tail":
        return length - sequence.max_length, length
    return 0, sequence.max_length


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
            elif len(items) != row_length:
                raise ValueError(
                    f"sequence {sequence.name!r} field {field.name!r} has length {len(items)} "
                    f"but expected {row_length} at row {row_index}"
                )
            raw_items_by_field[field.name] = items
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
) -> dict[str, Any]:
    """Encode and right-pad one configured sequence to the batch maximum length."""
    rows_by_field, row_lengths = _sequence_rows(table, sequence)

    lengths = torch.tensor(row_lengths, dtype=torch.long)
    max_length = int(lengths.max().item()) if row_lengths else 0
    tensor_fields: dict[str, Tensor] = {}
    unseen_policy = config.vocab_strategy.defaults.unseen_policy
    for field in sequence.fields:
        rows = rows_by_field[field.name]
        if field.kind == "categorical":
            qualified = field.qualified_name(sequence.name)
            categorical_input = config.resolved.categorical_input_by_name[qualified]
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
    return {"fields": tensor_fields, "lengths": lengths}


# --- Scenario and evaluation metadata ---


def _encode_scenario_item(
    value: Any,
    scenario_to_id: dict[str, int],
    scenario_count: int,
    row_index: int,
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
        index = int(value)
        if 0 <= index < scenario_count:
            return index
        raise ValueError(
            f"scenario id {index} at row {row_index} is outside [0, {scenario_count - 1}]"
        )
    if isinstance(value, str):
        if value in scenario_to_id:
            return scenario_to_id[value]
        raise ValueError(f"unknown scenario name {value!r} at row {row_index}")
    raise ValueError(
        f"scenario value must be a configured name or integer id at row {row_index}, "
        f"got {type(value).__name__}"
    )


def _scenario_tensor(config: AppConfig, table: Any, batch_size: int) -> Tensor:
    """Build scenario IDs or a multi-hot scenario mask for each row."""
    scenario_count = len(config.scenarios.names)
    if config.scenarios.source is None:
        if scenario_count != 1:
            raise ValueError("scenarios.source is required when multiple scenarios are configured")
        # Single-scenario models default every row to scenario index 0.
        return torch.zeros(batch_size, dtype=torch.long)

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
            _encode_scenario_item(item, scenario_to_id, scenario_count, row_index)
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
    features: dict[str, Any] = {}
    for feature in config.features:
        if feature.kind == "categorical":
            features[feature.name] = _tensorize_categorical(config, feature, table, vocab_maps)
        elif feature.kind == "dense":
            features[feature.name] = _tensorize_dense_column(feature, table)
        else:
            raise ValueError(f"unsupported feature kind {feature.kind!r}")
    for sequence in config.sequences:
        features[sequence.name] = _tensorize_multi_field_sequence(config, sequence, table, vocab_maps)

    # Labels and masks follow config order; missing masks default to all-observed.
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
        else:
            label_mask = torch.ones_like(labels)
    elif require_labels:
        raise ValueError("required label columns are missing from batch")

    return FeatureBatch(
        features=features,
        labels=labels,
        label_mask=label_mask,
        scenario_id=_scenario_tensor(config, table, batch_size),
        group_id=_group_ids(active_split, table, batch_size) if include_group_id else [],
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


def pin_feature_batch(batch: FeatureBatch) -> FeatureBatch:
    """Pin CPU tensors so CUDA transfers can use the non-blocking path."""
    return FeatureBatch(
        features={
            key: _map_feature_value(value, lambda tensor: tensor.pin_memory())
            for key, value in batch.features.items()
        },
        labels=None if batch.labels is None else batch.labels.pin_memory(),
        label_mask=None if batch.label_mask is None else batch.label_mask.pin_memory(),
        scenario_id=batch.scenario_id.pin_memory(),
        group_id=batch.group_id,
    )


def move_feature_batch(
    batch: FeatureBatch,
    device: torch.device,
    non_blocking: bool = False,
) -> FeatureBatch:
    """Move every tensor leaf while leaving string evaluation metadata on CPU."""
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
    )
