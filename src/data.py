from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import glob
from typing import Any, Iterable, Iterator, Protocol

from .config import AggLayout, AppConfig, ParquetSplitConfig


def _require_pyarrow() -> tuple[Any, Any, Any, Any]:
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


def discover_parquet_inputs(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.parquet")))
            continue
        matches = sorted(Path(match) for match in glob.glob(item, recursive=True))
        if matches:
            paths.extend(match for match in matches if match.is_file())
            continue
        if path.is_file():
            paths.append(path)
            continue
        raise FileNotFoundError(f"no parquet files matched input {item!r}")

    unique_paths = sorted({path.resolve() for path in paths})
    if not unique_paths:
        raise FileNotFoundError("no parquet files discovered")
    return unique_paths


def schema_fingerprint(schema: Any) -> str:
    payload = "\n".join(f"{field.name}:{field.type}:{field.nullable}" for field in schema)
    return sha256(payload.encode("utf-8")).hexdigest()


def parquet_schema(path: str | Path) -> Any:
    _pa, _pc, _ds, pq = _require_pyarrow()
    return pq.read_schema(path)


def validate_matching_schemas(paths: list[Path]) -> str:
    if not paths:
        raise ValueError("paths must not be empty")
    fingerprints = {path: schema_fingerprint(parquet_schema(path)) for path in paths}
    expected = next(iter(fingerprints.values()))
    mismatched = [str(path) for path, fingerprint in fingerprints.items() if fingerprint != expected]
    if mismatched:
        raise ValueError("parquet schema mismatch: " + ", ".join(mismatched))
    return expected


def _combine_column(table: Any, name: str) -> Any:
    if name not in table.column_names:
        raise ValueError(f"missing parquet column {name!r}")
    return table[name].combine_chunks()


def _list_parent_indices(array: Any) -> Any:
    _pa, pc, _ds, _pq = _require_pyarrow()
    if hasattr(pc, "list_parent_indices"):
        return pc.list_parent_indices(array)

    # Fallback for older pyarrow versions. This is not the preferred hot path,
    # but it keeps the decoder correct in development environments.
    offsets = array.offsets.to_pylist()
    parent_ids = []
    for parent_index, (start, end) in enumerate(zip(offsets, offsets[1:])):
        parent_ids.extend([parent_index] * (end - start))
    return _pa.array(parent_ids, type=_pa.int64())


def _list_lengths(array: Any) -> Any:
    _pa, pc, _ds, _pq = _require_pyarrow()
    if hasattr(pc, "list_value_length"):
        return pc.list_value_length(array)
    if hasattr(pc, "list_value_lengths"):
        return pc.list_value_lengths(array)
    offsets = array.offsets.to_pylist()
    return _pa.array([end - start for start, end in zip(offsets, offsets[1:])])


def _assert_equal_list_lengths(table: Any, columns: list[str]) -> None:
    if not columns:
        return
    first = _list_lengths(_combine_column(table, columns[0])).to_pylist()
    for column in columns[1:]:
        current = _list_lengths(_combine_column(table, column)).to_pylist()
        if current != first:
            raise ValueError(
                "candidate list columns must have equal per-request lengths; "
                f"{columns[0]!r} and {column!r} differ"
            )


class AggDecoder(Protocol):
    def decode(self, table: Any) -> Any:
        """Return a flat candidate-level pyarrow table."""


@dataclass(frozen=True)
class ParallelListsAggDecoder:
    layout: AggLayout

    def decode(self, table: Any) -> Any:
        pa, pc, _ds, _pq = _require_pyarrow()
        flatten_columns = list(dict.fromkeys([*self.layout.candidate_columns, *self.layout.labels.values(), *self.layout.label_masks.values()]))
        _assert_equal_list_lengths(table, flatten_columns)
        if not flatten_columns:
            raise ValueError("parallel_lists decoder requires at least one candidate column")

        first_array = _combine_column(table, flatten_columns[0])
        parent_indices = _list_parent_indices(first_array)
        output: dict[str, Any] = {"__request_index": parent_indices}

        if self.layout.request_id:
            output[self.layout.request_id] = pc.take(_combine_column(table, self.layout.request_id), parent_indices)
        for column in self.layout.shared_columns:
            output[column] = pc.take(_combine_column(table, column), parent_indices)
        for column in flatten_columns:
            output[column] = pc.list_flatten(_combine_column(table, column))

        return pa.table(output)


@dataclass(frozen=True)
class ListStructAggDecoder:
    layout: AggLayout

    def decode(self, table: Any) -> Any:
        pa, pc, _ds, _pq = _require_pyarrow()
        if self.layout.candidate_struct_column is None:
            raise ValueError("list_struct decoder requires candidate_struct_column")

        candidate_list = _combine_column(table, self.layout.candidate_struct_column)
        parent_indices = _list_parent_indices(candidate_list)
        flattened = pc.list_flatten(candidate_list)
        if flattened.type.num_fields == 0:
            raise ValueError("candidate struct contains no fields")

        field_names = self.layout.candidate_columns or [flattened.type[index].name for index in range(flattened.type.num_fields)]
        output: dict[str, Any] = {"__request_index": parent_indices}

        if self.layout.request_id:
            output[self.layout.request_id] = pc.take(_combine_column(table, self.layout.request_id), parent_indices)
        for column in self.layout.shared_columns:
            output[column] = pc.take(_combine_column(table, column), parent_indices)
        for field_name in field_names:
            output[field_name] = flattened.field(field_name)

        return pa.table(output)


class CustomAggDecoder:
    """Secure-environment extension point for request-to-candidate decoding.

    Implementations should accept the request-level pyarrow Table provided by the
    scanner and return a candidate-level pyarrow Table. The returned table should
    include `__request_index`, the configured request id, repeated shared columns,
    candidate feature columns, label columns, and label-mask columns when present.
    Keep row order stable so downstream grouping, metrics, and prediction export can
    map candidates back to the originating request.

    A secure environment can replace this class or make `build_agg_decoder` resolve
    `layout.custom_decoder` as a dotted import path that returns an AggDecoder.
    """

    def __init__(self, layout: AggLayout) -> None:
        self.layout = layout

    def decode(self, table: Any) -> Any:
        raise NotImplementedError(
            "custom agg decoding is a secure-environment hook, not a built-in layout. "
            f"Implement decoder {self.layout.custom_decoder!r} to convert a request-level "
            "pyarrow Table into a flat candidate-level table containing __request_index, "
            "request/shared/candidate columns, labels, and label masks."
        )


def build_agg_decoder(layout: AggLayout) -> AggDecoder:
    if layout.type == "parallel_lists":
        return ParallelListsAggDecoder(layout)
    if layout.type == "list_struct":
        return ListStructAggDecoder(layout)
    return CustomAggDecoder(layout)


def _sequence_source_columns(config: AppConfig) -> set[str]:
    columns: set[str] = set()
    for sequence in config.sequences:
        if sequence.layout == "list_struct":
            if sequence.source is not None:
                columns.add(sequence.source)
            continue
        columns.update(field.source for field in sequence.fields)
    return columns


def required_columns_for_split(config: AppConfig, split: ParquetSplitConfig) -> list[str]:
    columns: set[str] = set()
    sequence_columns = _sequence_source_columns(config)
    if split.format == "agg_parquet":
        if split.agg_layout is None:
            raise ValueError("agg split requires agg_layout")
        layout = split.agg_layout
        columns.add(layout.request_id)
        columns.update(layout.shared_columns)
        if layout.type == "parallel_lists":
            for feature in config.features:
                columns.add(feature.source)
            columns.update(sequence_columns)
            columns.update(layout.candidate_columns)
            columns.update(layout.labels.values())
            columns.update(layout.label_masks.values())
        elif layout.candidate_struct_column is not None:
            columns.add(layout.candidate_struct_column)
            columns.update(sequence_columns)
    else:
        for feature in config.features:
            columns.add(feature.source)
        columns.update(sequence_columns)
        if split.request_id:
            columns.add(split.request_id)
        if split.group_id:
            columns.add(split.group_id)
    return sorted(columns)


@dataclass(frozen=True)
class ScanStats:
    files: int
    record_batches: int
    rows: int


class ParquetScanner:
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
        self.all_paths = discover_parquet_inputs(split.inputs)
        validate_matching_schemas(self.all_paths)
        if shard_world_size > 1 and split.reader.shard_unit == "file":
            self.paths = self.all_paths[shard_rank::shard_world_size]
        else:
            self.paths = self.all_paths

    def iter_record_batches(self) -> Iterator[Any]:
        _pa, _pc, ds, _pq = _require_pyarrow()
        if not self.paths:
            return
        dataset = ds.dataset([str(path) for path in self.paths], format="parquet")
        scanner_kwargs: dict[str, Any] = {"columns": self.columns or None}
        if self.split.reader.batch_size_rows is not None:
            scanner_kwargs["batch_size"] = self.split.reader.batch_size_rows
        scanner = dataset.scanner(**scanner_kwargs)
        for batch_index, batch in enumerate(scanner.to_batches()):
            if (
                self.shard_world_size > 1
                and self.split.reader.shard_unit in {"record_batch", "row_group"}
                and batch_index % self.shard_world_size != self.shard_rank
            ):
                continue
            yield batch

    def iter_tables(self) -> Iterator[Any]:
        _pa, _pc, _ds, _pq = _require_pyarrow()
        for batch in self.iter_record_batches():
            yield _pa.Table.from_batches([batch])

    def scan_stats(self, max_batches: int | None = None) -> ScanStats:
        record_batches = 0
        rows = 0
        for batch in self.iter_record_batches():
            if max_batches is not None and record_batches >= max_batches:
                break
            record_batches += 1
            rows += batch.num_rows
        return ScanStats(files=len(self.paths), record_batches=record_batches, rows=rows)


class AggParquetScanner(ParquetScanner):
    def __init__(
        self,
        split: ParquetSplitConfig,
        columns: list[str],
        shard_rank: int = 0,
        shard_world_size: int = 1,
    ) -> None:
        if split.agg_layout is None:
            raise ValueError("agg parquet scanner requires agg_layout")
        super().__init__(split, columns, shard_rank=shard_rank, shard_world_size=shard_world_size)
        self.decoder = build_agg_decoder(split.agg_layout)

    def iter_candidate_tables(self) -> Iterator[Any]:
        for table in self.iter_tables():
            yield self.decoder.decode(table)
