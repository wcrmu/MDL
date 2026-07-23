#!/usr/bin/env python3
"""Expand one anonymized production-shaped agg row into benchmark Parquet.

Every physical request, candidate, and UPS slot in the source is real. Numeric
zeros in feature payloads are anonymized values, not padding, so this generator
materializes them as deterministic non-zero int64 values. Structural indices,
timestamps, request IDs, and labels are generated separately to preserve their
contracts.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AppConfig, load_app_config


MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class MockParquetManifest:
    source_json: str
    config: str
    output_dir: str
    files: int
    rows_per_file: int
    raw_rows: int
    requests_per_raw_row: int
    candidates_per_raw_row: int
    candidate_rows: int
    sequence_tokens_per_raw_row: Mapping[str, int]
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


def _axis_length(values: Any, *, column: str) -> int:
    if not isinstance(values, list):
        raise ValueError(f"mock axis {column!r} must be list-valued")
    if not values:
        raise ValueError(f"mock axis {column!r} must not be empty")
    return len(values)


def _adapter_options(config: AppConfig) -> Mapping[str, Any]:
    adapter = config.data.train.adapter
    if adapter is None:
        raise ValueError("mock generation requires data.train.adapter")
    return adapter.options


def _splitmix64(value: int) -> int:
    value &= MASK64
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & MASK64
    return (value ^ (value >> 31)) & MASK64


def _column_seed(column: str) -> int:
    return int.from_bytes(
        sha256(column.encode("utf-8")).digest()[:8], "little"
    )


def _materialize_feature_tree(
    value: Any,
    *,
    salt: int,
    position: list[int],
    vary_existing_values: bool,
) -> Any:
    if isinstance(value, list):
        return [
            _materialize_feature_tree(
                item,
                salt=salt,
                position=position,
                vary_existing_values=vary_existing_values,
            )
            for item in value
        ]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        leaf = position[0]
        position[0] += 1
        if value != 0 and not vary_existing_values:
            return value
        mixed = (value & MASK64) ^ _splitmix64(salt + leaf + 1)
        if mixed == 0:
            mixed = 1
        return mixed if mixed < (1 << 63) else mixed - (1 << 64)
    if isinstance(value, float):
        leaf = position[0]
        position[0] += 1
        if value != 0.0 and not vary_existing_values:
            return value
        mixed = _splitmix64(salt + leaf + 1)
        return float((mixed % 1_000_000) + 1) / 1000.0
    if isinstance(value, str):
        leaf = position[0]
        position[0] += 1
        if value not in {"", "0"} and not vary_existing_values:
            return value
        mixed = _splitmix64(salt + leaf + 1)
        return f"mock-{mixed:016x}"
    return value


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
    request_count = _axis_length(row.get("context_indices"), column="context_indices")
    candidate_count = _axis_length(row.get("target_indices"), column="target_indices")

    # Index zero is a valid request key. Repeated sanitized zeros are replaced
    # with a complete, unique request axis and valid candidate references.
    row["context_indices"] = list(range(request_count))
    row["target_indices"] = [
        (global_row + candidate_position) % request_count
        for candidate_position in range(candidate_count)
    ]

    request_id_column = "search_id"
    if request_id_column in row:
        row[request_id_column] = [
            f"mock-request-{global_row:08d}-{position}"
            for position in range(request_count)
        ]

    if "scene_id" in row:
        scene_ids = (1, 11, 2, 21, 23, 27, 31)
        row["scene_id"] = [
            scene_ids[(global_row + position) % len(scene_ids)]
            for position in range(request_count)
        ]

    request_time_column = str(options.get("request_time_column", "impr_time"))
    base_time_ms = 1_800_000_000_000 + global_row * 10_000_000
    row[request_time_column] = [
        base_time_ms + position * 1000 for position in range(request_count)
    ]

    for ups_index, ups in enumerate(options.get("ups_types", ())):
        index_column = f"{ups}_x_indices"
        time_column = f"{ups}_x_time"
        memberships = row.get(index_column)
        times = row.get(time_column)
        if not isinstance(memberships, list) or not isinstance(times, list):
            raise ValueError(
                f"mock row must contain list-valued {index_column!r} and {time_column!r}"
            )
        if len(memberships) != len(times):
            raise ValueError(
                f"{index_column!r} length {len(memberships)} does not match "
                f"{time_column!r} length {len(times)}"
            )
        generated_memberships: list[Any] = []
        for token_position, raw_membership in enumerate(memberships):
            is_list = isinstance(raw_membership, list)
            membership_size = len(raw_membership) if is_list else 1
            if membership_size <= 0:
                raise ValueError(
                    f"{index_column!r} token {token_position} has empty membership"
                )
            if membership_size > request_count:
                raise ValueError(
                    f"{index_column!r} token {token_position} has "
                    f"{membership_size} memberships but only {request_count} requests"
                )
            first_request = (
                global_row + ups_index + token_position
            ) % request_count
            generated = [
                (first_request + offset) % request_count
                for offset in range(membership_size)
            ]
            generated_memberships.append(generated if is_list else generated[0])
        row[index_column] = generated_memberships
        row[time_column] = [
            base_time_ms
            - (ups_index + 1) * 1_000_000
            - (token_position + 1) * 1000
            for token_position in range(len(times))
        ]

    labels = options.get("labels", {})
    for task_index, column in enumerate(labels.values()):
        values = row.get(str(column))
        if not isinstance(values, list):
            continue
        if len(values) != candidate_count:
            raise ValueError(
                f"label {column!r} has {len(values)} values, expected {candidate_count}"
            )
        for candidate_position in range(candidate_count):
            target = (global_row + candidate_position + task_index) & 1
            values[candidate_position] = _set_binary_label(
                values[candidate_position], target
            )

    for column in sorted(_feature_value_columns(row, options)):
        if column not in row:
            continue
        row_salt = _splitmix64(global_row + 1) if vary_feature_values else 0
        salt = _column_seed(column) ^ row_salt
        row[column] = _materialize_feature_tree(
            row[column],
            salt=salt,
            position=[0],
            vary_existing_values=vary_feature_values,
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
    requests_per_row = _axis_length(
        base_row.get("context_indices"), column="context_indices"
    )
    candidates_per_row = _axis_length(
        base_row.get("target_indices"), column="target_indices"
    )
    sequence_tokens = {
        str(ups): len(base_row[f"{ups}_x_indices"])
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
        requests_per_raw_row=requests_per_row,
        candidates_per_raw_row=candidates_per_row,
        candidate_rows=raw_rows * candidates_per_row,
        sequence_tokens_per_raw_row=sequence_tokens,
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
