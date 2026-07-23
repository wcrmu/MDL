#!/usr/bin/env python3
"""Expand one production-shaped mock agg row into benchmark Parquet files.

The source JSON remains physically faithful: fixed-width request, candidate,
and UPS arrays keep their zero-padded slots. Live IDs and labels are varied per
row so compression and embedding lookup behavior do not collapse to one
constant record. The configured adapter is expected to compact padding while
reading (``adapter.options.fixed_padding``).
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AppConfig, load_app_config
from src.dataloader import _is_fixed_padding_cell


MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class MockParquetManifest:
    source_json: str
    config: str
    output_dir: str
    files: int
    rows_per_file: int
    raw_rows: int
    live_requests_per_raw_row: int
    live_candidates_per_raw_row: int
    candidate_rows: int
    live_sequence_tokens_per_raw_row: Mapping[str, int]
    physical_columns: int
    row_group_size: int
    row_groups: int
    compression: str
    vary_feature_values: bool
    arrow_bytes_per_raw_row: int
    parquet_bytes: int
    projected_compressed_bytes: int
    schema_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("mock Parquet generation requires PyArrow") from error
    return pa, pq


def _load_one_row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(
                f"mock JSON must contain exactly one row, found {len(payload)}"
            )
        payload = payload[0]
    if not isinstance(payload, dict) or not payload:
        raise ValueError("mock JSON row must be a non-empty object")
    return payload


def _active_positions(values: Any, *, column: str) -> list[int]:
    if not isinstance(values, list):
        raise ValueError(f"mock anchor {column!r} must be list-valued")
    positions = [
        index
        for index, value in enumerate(values)
        if not _is_fixed_padding_cell(value)
    ]
    if not positions:
        raise ValueError(f"mock anchor {column!r} has no live slots")
    return positions


def _adapter_options(config: AppConfig) -> Mapping[str, Any]:
    adapter = config.data.train.adapter
    if adapter is None:
        raise ValueError("mock generation requires data.train.adapter")
    options = adapter.options
    padding = options.get("fixed_padding")
    if not isinstance(padding, Mapping):
        raise ValueError(
            "config must set data.train.adapter.options.fixed_padding"
        )
    return options


def _padding_options(options: Mapping[str, Any]) -> tuple[str, str, str]:
    padding = options["fixed_padding"]
    request_anchor = str(padding["request_anchor"])
    candidate_anchor = str(padding["candidate_anchor"])
    sequence_suffix = str(padding.get("sequence_anchor_suffix", "_x_time"))
    return request_anchor, candidate_anchor, sequence_suffix


def _repair_request_times(
    row: dict[str, Any],
    options: Mapping[str, Any],
) -> None:
    """Ensure sanitized mock event times are not later than request time."""

    request_anchor, _candidate_anchor, sequence_suffix = _padding_options(options)
    request_positions = _active_positions(row[request_anchor], column=request_anchor)
    context_indices = row.get("context_indices")
    request_times = row.get(str(options.get("request_time_column", "impr_time")))
    if not isinstance(context_indices, list) or not isinstance(request_times, list):
        raise ValueError("mock row is missing list-valued context indices/request times")
    request_position_by_id = {
        context_indices[position]: position for position in request_positions
    }
    latest_by_request: dict[Any, int] = {}
    for ups in options.get("ups_types", ()):
        time_column = f"{ups}{sequence_suffix}"
        index_column = f"{ups}_x_indices"
        times = row.get(time_column)
        memberships = row.get(index_column)
        if not isinstance(times, list) or not isinstance(memberships, list):
            continue
        for event_time, raw_members in zip(times, memberships):
            if _is_fixed_padding_cell(event_time):
                continue
            members = raw_members if isinstance(raw_members, list) else [raw_members]
            for request in set(members):
                if request not in request_position_by_id:
                    continue
                latest_by_request[request] = max(
                    latest_by_request.get(request, 0), int(event_time)
                )
    for request, position in request_position_by_id.items():
        latest = latest_by_request.get(request)
        if latest is not None and int(request_times[position]) <= latest:
            request_times[position] = latest + 1000


def _splitmix64(value: int) -> int:
    value &= MASK64
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & MASK64
    return (value ^ (value >> 31)) & MASK64


def _column_seed(column: str) -> int:
    return int.from_bytes(
        sha256(column.encode("utf-8")).digest()[:8], "little"
    )


def _vary_tree(value: Any, *, salt: int, position: list[int]) -> Any:
    if isinstance(value, list):
        return [
            _vary_tree(item, salt=salt, position=position)
            for item in value
        ]
    if isinstance(value, bool) or not isinstance(value, int) or value == 0:
        return value
    leaf = position[0]
    position[0] += 1
    mixed = (value & MASK64) ^ _splitmix64(salt + leaf + 1)
    if mixed == 0:
        mixed = 1
    return mixed if mixed < (1 << 63) else mixed - (1 << 64)


def _feature_value_columns(
    row: Mapping[str, Any],
    options: Mapping[str, Any],
) -> set[str]:
    columns = {
        str(column)
        for key in ("context_features", "item_features")
        for column in options.get(key, ())
    }
    for ups in options.get("ups_types", ()):
        prefix = f"{ups}_x_"
        columns.update(
            column
            for column in row
            if column.startswith(prefix)
            and column not in {f"{ups}_x_indices", f"{ups}_x_time"}
        )
    return columns


def _set_binary_label(cell: Any, value: int) -> Any:
    if isinstance(cell, list):
        return [value]
    return value


def _make_variant(
    base_row: Mapping[str, Any],
    options: Mapping[str, Any],
    *,
    global_row: int,
    vary_feature_values: bool,
) -> dict[str, Any]:
    row = deepcopy(base_row)
    request_anchor, candidate_anchor, _sequence_suffix = _padding_options(options)
    request_positions = _active_positions(row[request_anchor], column=request_anchor)
    candidate_positions = _active_positions(
        row[candidate_anchor], column=candidate_anchor
    )

    for position in request_positions:
        original = str(row[request_anchor][position])
        row[request_anchor][position] = f"mock-{global_row}-{position}-{original}"

    time_columns = {
        str(options.get("request_time_column", "impr_time")),
        *(f"{ups}_x_time" for ups in options.get("ups_types", ())),
    }
    time_offset = global_row * 10_000_000
    for column in time_columns:
        values = row.get(column)
        if not isinstance(values, list):
            continue
        row[column] = [
            value + time_offset
            if isinstance(value, int) and not isinstance(value, bool) and value != 0
            else value
            for value in values
        ]

    labels = options.get("labels", {})
    for task_index, column in enumerate(labels.values()):
        values = row.get(str(column))
        if not isinstance(values, list):
            continue
        for candidate_position in candidate_positions:
            target = (global_row + candidate_position + task_index) & 1
            values[candidate_position] = _set_binary_label(
                values[candidate_position], target
            )

    if vary_feature_values:
        for column in sorted(_feature_value_columns(row, options)):
            if column not in row:
                continue
            salt = _column_seed(column) ^ _splitmix64(global_row + 1)
            row[column] = _vary_tree(
                row[column],
                salt=salt,
                position=[0],
            )
    return row


def _projected_compressed_bytes(path: Path, projected: set[str], pq: Any) -> int:
    metadata = pq.ParquetFile(path).metadata
    total = 0
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            if column.path_in_schema.split(".", 1)[0] in projected:
                total += int(column.total_compressed_size)
    return total


def generate_mock_parquet_dataset(
    config: AppConfig,
    source_json: Path,
    output_dir: Path,
    *,
    files: int,
    rows_per_file: int,
    row_group_size: int = 256,
    compression: str = "zstd",
    vary_feature_values: bool = True,
    config_path: Path | None = None,
) -> MockParquetManifest:
    if min(files, rows_per_file, row_group_size) <= 0:
        raise ValueError("files, rows_per_file, and row_group_size must be positive")
    pa, pq = _require_pyarrow()
    options = _adapter_options(config)
    base_row = deepcopy(_load_one_row(source_json))
    _repair_request_times(base_row, options)
    request_anchor, candidate_anchor, sequence_suffix = _padding_options(options)
    live_requests = len(_active_positions(base_row[request_anchor], column=request_anchor))
    live_candidates = len(
        _active_positions(base_row[candidate_anchor], column=candidate_anchor)
    )
    live_sequence_tokens = {
        str(ups): len(
            _active_positions(
                base_row[f"{ups}{sequence_suffix}"],
                column=f"{ups}{sequence_suffix}",
            )
        )
        if any(
            not _is_fixed_padding_cell(value)
            for value in base_row[f"{ups}{sequence_suffix}"]
        )
        else 0
        for ups in options.get("ups_types", ())
    }

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")

    first_row = _make_variant(
        base_row,
        options,
        global_row=0,
        vary_feature_values=vary_feature_values,
    )
    first_table = pa.Table.from_pylist([first_row])
    schema = first_table.schema
    projected = set(config.data.train.adapter.input_columns or ())
    projected.update(config.data.train.adapter.optional_input_columns)
    total_parquet_bytes = 0
    total_projected_bytes = 0
    total_row_groups = 0

    for file_index in range(files):
        path = output_dir / f"mock_{file_index:05d}.parquet"
        writer = pq.ParquetWriter(
            path,
            schema,
            compression=compression,
            use_dictionary=True,
            write_statistics=True,
        )
        try:
            for offset in range(0, rows_per_file, row_group_size):
                count = min(row_group_size, rows_per_file - offset)
                rows = []
                for local_index in range(offset, offset + count):
                    global_row = file_index * rows_per_file + local_index
                    if global_row == 0:
                        rows.append(first_row)
                    else:
                        rows.append(
                            _make_variant(
                                base_row,
                                options,
                                global_row=global_row,
                                vary_feature_values=vary_feature_values,
                            )
                        )
                table = pa.Table.from_pylist(rows, schema=schema)
                writer.write_table(
                    table,
                    row_group_size=count,
                )
        finally:
            writer.close()
        parquet_file = pq.ParquetFile(path)
        total_row_groups += parquet_file.metadata.num_row_groups
        total_parquet_bytes += path.stat().st_size
        total_projected_bytes += _projected_compressed_bytes(path, projected, pq)

    raw_rows = files * rows_per_file
    manifest = MockParquetManifest(
        source_json=str(source_json.resolve()),
        config=(str(config_path.resolve()) if config_path is not None else "<in-memory>"),
        output_dir=str(output_dir),
        files=files,
        rows_per_file=rows_per_file,
        raw_rows=raw_rows,
        live_requests_per_raw_row=live_requests,
        live_candidates_per_raw_row=live_candidates,
        candidate_rows=raw_rows * live_candidates,
        live_sequence_tokens_per_raw_row=live_sequence_tokens,
        physical_columns=len(schema),
        row_group_size=row_group_size,
        row_groups=total_row_groups,
        compression=compression,
        vary_feature_values=vary_feature_values,
        arrow_bytes_per_raw_row=int(first_table.nbytes),
        parquet_bytes=total_parquet_bytes,
        projected_compressed_bytes=total_projected_bytes,
        schema_sha256=sha256(str(schema).encode("utf-8")).hexdigest(),
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rankmixer.yaml"))
    parser.add_argument(
        "--source-json", type=Path, default=Path("sample_row_mock_json")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--files", type=_positive_int, default=2)
    parser.add_argument("--rows-per-file", type=_positive_int, default=2500)
    parser.add_argument("--row-group-size", type=_positive_int, default=256)
    parser.add_argument(
        "--compression", choices=("zstd", "snappy", "gzip"), default="zstd"
    )
    parser.add_argument(
        "--vary-feature-values",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    config = load_app_config(args.config)
    manifest = generate_mock_parquet_dataset(
        config,
        args.source_json,
        args.output_dir,
        files=args.files,
        rows_per_file=args.rows_per_file,
        row_group_size=args.row_group_size,
        compression=args.compression,
        vary_feature_values=args.vary_feature_values,
        config_path=args.config,
    )
    print(json.dumps(manifest.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
