#!/usr/bin/env python3
"""Generate local CVR agg Parquet for reader and end-to-end benchmarks.

The generated files follow the production adapter contract: one raw row holds
multiple requests, candidates are routed by ``target_indices``, and every UPS
token carries a list of visible request IDs. The physical schema can be padded
to 630 columns so projection/footer behavior resembles production while model
input values remain limited to the YAML-declared 47 Context, 122 Item, and nine
UPS groups.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AppConfig, load_app_config


OBSERVED_MEDIAN_SEQUENCE_LENGTHS: dict[str, int] = {
    "impr": 946,
    "clk_long": 1340,
    "view_long": 1183,
    "cart_long": 210,
    "buy_long": 22,
    "semi_clk": 4,
    "srch_q2i": 71,
    "ups_clk_sku": 200,
    "flatten_query_hash": 78,
}


_BAG_LENGTH_ESTIMATES: dict[str, int] = {
    "origin_query_hash_hn": 4,
    "query_arr_hn": 4,
    "query_hash_hn": 2,
    "query_terms_hash_hn": 8,
    "query_tfidf_term_hash_list_hn": 6,
    "query_extend_translation_hash_hn": 4,
    "sess_q2q_hash_list_hn": 3,
    "recall_merge_cate_levels_hn": 32,
    "recall_merge_cate1_ids_hn": 8,
    "recall_merge_cate_ids_hn": 32,
    "scene_impr_cnt_15d_hn": 15,
    "u_fst_ordr_cnt_mix_d_hn": 3,
    "clk_cnt_1d_hn": 15,
    "clk_3d_cnt_hn": 15,
    "clk_1d_cat_cnt_hn": 15,
    "cart_cnt_1d_hn": 15,
    "cart_cnt_3d_hn": 15,
    "clk_7d_page_sns_hn": 64,
    "clk_7d_page_elsns_hn": 64,
    "cart_7d_cat1_ids_hn": 32,
    "flip_mall_ids_hn": 128,
    "list_clk_cat1_ids_hn": 16,
    "list_clk_cat_ids_hn": 16,
    "ups_in_cart_2h_sku_cur_prices_hn": 16,
    "ups_in_cart_goods_hn_share": 32,
    "ups_incart_cat1_id_nc_hn": 3,
    "ups_in_cart_tg_hn": 32,
    "ups_query_term_hash_v2_hn": 32,
    "ups_query_tg_hn": 32,
    "ups_search_method_hash_hn": 32,
    "view_30m_cat1_ids_hn": 16,
    "view_7d_page_sns_hn": 64,
    "view_7d_page_elsns_hn": 64,
    "goods_name_bigram_hn": 20,
    "goods_ner_infos_hn": 6,
    "goods_title_tfidf_term_hash_list_hn": 6,
    "rev_ratings_cnt_crs_pos_hn": 5,
    "g_sku_spec_hn": 6,
    "g_sku_spec_hash_hn": 6,
    "g_sku_spec_unit_list_hn": 4,
    "g_prpty_val_id_list_hn": 8,
    "sku_id_hn": 3,
    "sku_price_v2_hn": 3,
    "sku_sales_hn": 3,
    "sku_spec_hash_hn": 3,
    "sku_spec_hn": 3,
    "sku_cart_cnt_7d_hn": 3,
    "sku_ordr_cnt_1m_hn": 3,
    "sku_price_dis_hn": 3,
    "sku_sales_dis_hn": 3,
}


@dataclass(frozen=True)
class SyntheticAggManifest:
    output_dir: str
    files: int
    raw_rows_per_file: int
    raw_rows: int
    requests_per_agg: int
    candidates_per_request: int
    candidates: int
    sequence_overlap: float
    sequence_lengths_after_request_filter: Mapping[str, int]
    raw_sequence_lengths: Mapping[str, int]
    bag_lengths: Mapping[str, int]
    physical_columns: int
    projected_columns: int
    arrow_bytes_per_file: int
    parquet_file_bytes: int
    projected_compressed_bytes: int
    projected_compressed_bytes_per_candidate: float
    compression: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_arrow_numpy() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("synthetic Parquet generation requires NumPy and PyArrow") from error
    return np, pa, pq


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _named_lengths(raw: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in raw.split(","):
        name, separator, value_text = item.strip().partition("=")
        if not separator or not name:
            raise argparse.ArgumentTypeError(
                "lengths must be comma-separated name=positive_integer entries"
            )
        value = _positive_int(value_text)
        result[name] = value
    return result


def _hash_values(np: Any, count: int, seed: int) -> Any:
    """Return deterministic nonzero signed int64 bit patterns."""

    if count == 0:
        return np.empty(0, dtype=np.int64)
    with np.errstate(over="ignore"):
        values = np.arange(count, dtype=np.uint64) + np.uint64(seed + 1)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        values = (values ^ (values >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
        values ^= values >> np.uint64(31)
    signed = values.view(np.int64)
    signed[signed == 0] = 1
    return signed


def _list_int64(pa: Any, np: Any, rows: int, values_per_row: int, values: Any) -> Any:
    offsets = np.arange(rows + 1, dtype=np.int64) * values_per_row
    return pa.LargeListArray.from_arrays(pa.array(offsets), pa.array(values, type=pa.int64()))


def _nested_int64(
    pa: Any,
    np: Any,
    rows: int,
    inner_per_row: int,
    inner_lengths: Any,
    values: Any,
) -> Any:
    inner_lengths = np.asarray(inner_lengths, dtype=np.int64)
    if inner_lengths.size == 1:
        inner_lengths = np.full(rows * inner_per_row, int(inner_lengths[0]), dtype=np.int64)
    if inner_lengths.size != rows * inner_per_row:
        raise ValueError("inner length vector does not match nested list shape")
    inner_offsets = np.empty(inner_lengths.size + 1, dtype=np.int64)
    inner_offsets[0] = 0
    np.cumsum(inner_lengths, out=inner_offsets[1:])
    inner = pa.LargeListArray.from_arrays(
        pa.array(inner_offsets),
        pa.array(values, type=pa.int64()),
    )
    outer_offsets = np.arange(rows + 1, dtype=np.int64) * inner_per_row
    return pa.LargeListArray.from_arrays(pa.array(outer_offsets), inner)


def _list_strings(pa: Any, rows: Sequence[Sequence[str]]) -> Any:
    return pa.array(rows, type=pa.list_(pa.string()))


def _bag_lengths(config: AppConfig, scale: float) -> dict[str, int]:
    adapter = config.data.train.adapter
    if adapter is None:
        raise ValueError("synthetic agg generation requires an adapter_parquet config")
    options = adapter.options
    bag_sources = {str(item) for item in options.get("multivalue_features", ())}
    max_by_source: dict[str, int] = {}
    for feature in config.features:
        if feature.source not in bag_sources:
            continue
        if feature.max_length is not None:
            current = max_by_source.get(feature.source)
            max_by_source[feature.source] = (
                feature.max_length if current is None else max(current, feature.max_length)
            )
    result: dict[str, int] = {}
    for source in bag_sources:
        estimate = max(1, int(round(_BAG_LENGTH_ESTIMATES.get(source, 4) * scale)))
        configured = max_by_source.get(source)
        if configured is not None:
            estimate = min(estimate, configured)
        result[source] = estimate

    # SKU fields describe one aligned SKU list and must have identical lengths.
    for raw_group in options.get("aligned_multivalue_groups", ()):
        group = [str(item) for item in raw_group]
        group_length = min(result[item] for item in group)
        for item in group:
            result[item] = group_length
    return result


def _ups_raw_columns(config: AppConfig, ups_types: Sequence[str]) -> dict[str, list[str]]:
    adapter = config.data.train.adapter
    if adapter is None:
        raise ValueError("synthetic agg generation requires an adapter")
    declared = list(adapter.input_columns or ())
    if not declared:
        derived = {
            str(value)
            for value in adapter.options.get("time_delta_outputs", {}).values()
        }
        declared = [
            field.source
            for sequence in config.sequences
            for field in sequence.fields
            if field.source not in derived
        ]
        declared.extend(f"{ups}_x_time" for ups in ups_types)
    return {
        ups: list(
            dict.fromkeys(
                column
                for column in declared
                if column.startswith(f"{ups}_x_")
                and column != f"{ups}_x_indices"
            )
        )
        for ups in ups_types
    }


def _membership_pattern(
    np: Any,
    request_length: int,
    requests_per_agg: int,
    overlap: float,
) -> tuple[Any, Any, int]:
    shared = min(request_length, max(0, int(round(request_length * overlap))))
    private = request_length - shared
    inner_lengths = np.concatenate(
        (
            np.full(shared, requests_per_agg, dtype=np.int64),
            np.ones(private * requests_per_agg, dtype=np.int64),
        )
    )
    shared_values = np.tile(np.arange(requests_per_agg, dtype=np.int64), shared)
    private_values = np.repeat(np.arange(requests_per_agg, dtype=np.int64), private)
    values = np.concatenate((shared_values, private_values))
    return inner_lengths, values, int(inner_lengths.size)


def _projected_compressed_bytes(path: Path, projected: set[str]) -> int:
    _np, _pa, pq = _require_arrow_numpy()
    parquet_file = pq.ParquetFile(path)
    total = 0
    metadata = parquet_file.metadata
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            top_level = column.path_in_schema.split(".", 1)[0]
            if top_level in projected:
                total += int(column.total_compressed_size)
    return total


def generate_synthetic_agg_dataset(
    config: AppConfig,
    output_dir: Path,
    *,
    files: int,
    raw_rows_per_file: int,
    requests_per_agg: int,
    candidates_per_request: int,
    sequence_lengths: Mapping[str, int] | None = None,
    sequence_overlap: float = 0.85,
    bag_length_scale: float = 1.0,
    scenario_count: int = 32,
    physical_column_count: int = 630,
    compression: str = "gzip",
) -> SyntheticAggManifest:
    if min(files, raw_rows_per_file, requests_per_agg, candidates_per_request) <= 0:
        raise ValueError("file, row, request, and candidate counts must be positive")
    if not 0.0 <= sequence_overlap <= 1.0:
        raise ValueError("sequence_overlap must be in [0, 1]")
    if bag_length_scale <= 0.0 or scenario_count <= 0:
        raise ValueError("bag_length_scale and scenario_count must be positive")
    adapter = config.data.train.adapter
    if adapter is None:
        raise ValueError("config.data.train.adapter is required")
    options = adapter.options
    context_features = tuple(str(item) for item in options["context_features"])
    item_features = tuple(str(item) for item in options["item_features"])
    bag_features = {str(item) for item in options["multivalue_features"]}
    request_axis_features = context_features
    candidate_axis_features = item_features
    ups_types = tuple(str(item) for item in options["ups_types"])
    lengths = dict(OBSERVED_MEDIAN_SEQUENCE_LENGTHS)
    if sequence_lengths is not None:
        unknown = set(sequence_lengths) - set(ups_types)
        if unknown:
            raise ValueError("unknown UPS sequence lengths: " + ", ".join(sorted(unknown)))
        lengths.update({name: int(value) for name, value in sequence_lengths.items()})
    lengths = {name: lengths[name] for name in ups_types}
    if any(value <= 0 for value in lengths.values()):
        raise ValueError("sequence lengths must be positive")

    np, pa, pq = _require_arrow_numpy()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("*.parquet")):
        raise FileExistsError(f"output directory already contains Parquet files: {output_dir}")

    bag_lengths = _bag_lengths(config, bag_length_scale)
    candidates_per_agg = requests_per_agg * candidates_per_request
    columns: dict[str, Any] = {}
    seed = 1000

    for column in request_axis_features:
        inner_length = bag_lengths[column] if column in bag_features else 1
        count = raw_rows_per_file * requests_per_agg * inner_length
        columns[column] = _nested_int64(
            pa,
            np,
            raw_rows_per_file,
            requests_per_agg,
            np.array([inner_length]),
            _hash_values(np, count, seed),
        )
        seed += 1

    for column in candidate_axis_features:
        inner_length = bag_lengths[column] if column in bag_features else 1
        count = raw_rows_per_file * candidates_per_agg * inner_length
        columns[column] = _nested_int64(
            pa,
            np,
            raw_rows_per_file,
            candidates_per_agg,
            np.array([inner_length]),
            _hash_values(np, count, seed),
        )
        seed += 1

    raw_sequence_lengths: dict[str, int] = {}
    memberships_by_ups: dict[str, tuple[Any, Any, int]] = {}
    ups_columns = _ups_raw_columns(config, ups_types)
    for ups in ups_types:
        membership_lengths, membership_values, raw_length = _membership_pattern(
            np,
            lengths[ups],
            requests_per_agg,
            sequence_overlap,
        )
        raw_sequence_lengths[ups] = raw_length
        memberships_by_ups[ups] = (
            membership_lengths,
            membership_values,
            raw_length,
        )
        repeated_inner_lengths = np.tile(membership_lengths, raw_rows_per_file)
        repeated_memberships = np.tile(membership_values, raw_rows_per_file)
        columns[f"{ups}_x_indices"] = _nested_int64(
            pa,
            np,
            raw_rows_per_file,
            raw_length,
            repeated_inner_lengths,
            repeated_memberships,
        )
        for column in ups_columns[ups]:
            count = raw_rows_per_file * raw_length
            if column == f"{ups}_x_time":
                event_rows = []
                for row_index in range(raw_rows_per_file):
                    base = 1_780_000_000_000 + row_index * 10_000_000
                    event_rows.append(
                        base - 1000 - np.arange(raw_length, dtype=np.int64) * 1000
                    )
                values = np.concatenate(event_rows)
            else:
                values = _hash_values(np, count, seed)
                seed += 1
            columns[column] = _nested_int64(
                pa,
                np,
                raw_rows_per_file,
                raw_length,
                np.array([1]),
                values,
            )

    context_index_values = np.tile(
        np.arange(requests_per_agg, dtype=np.int64), raw_rows_per_file
    )
    target_pattern = np.repeat(
        np.arange(requests_per_agg, dtype=np.int64), candidates_per_request
    )
    target_index_values = np.tile(target_pattern, raw_rows_per_file)
    columns["context_indices"] = _list_int64(
        pa, np, raw_rows_per_file, requests_per_agg, context_index_values
    )
    columns["target_indices"] = _list_int64(
        pa, np, raw_rows_per_file, candidates_per_agg, target_index_values
    )

    label_columns = [str(value) for value in options["labels"].values()]
    for label_index, column in enumerate(label_columns):
        values = (
            np.arange(raw_rows_per_file * candidates_per_agg, dtype=np.int64)
            + label_index
        ) % 2
        columns[column] = _list_int64(
            pa, np, raw_rows_per_file, candidates_per_agg, values
        )

    # Extra scalar columns preserve the wide 630-column footer/projection
    # shape. Their bytes are intentionally excluded from projected-byte stats.
    reserved_names = set(columns) | {"scene_id", "search_id", "impr_time"}
    extra_count = max(0, physical_column_count - len(reserved_names))
    for index in range(extra_count):
        name = f"__unused_benchmark_{index:03d}"
        columns[name] = pa.array(
            _hash_values(np, raw_rows_per_file, seed), type=pa.int64()
        )
        seed += 1

    base_table = pa.table(columns)
    projected = set(adapter.input_columns or ()) | set(adapter.optional_input_columns)
    projected_present: set[str] | None = None
    total_file_bytes = 0
    total_projected_bytes = 0
    first_arrow_bytes = 0
    for file_index in range(files):
        scene_values = (
            np.arange(raw_rows_per_file * requests_per_agg, dtype=np.int64)
            + file_index * raw_rows_per_file * requests_per_agg
        ) % scenario_count
        request_times = []
        search_ids: list[list[str]] = []
        for row_index in range(raw_rows_per_file):
            base = 1_780_000_000_000 + row_index * 10_000_000
            request_times.extend(base + np.arange(requests_per_agg, dtype=np.int64) * 1000)
            search_ids.append(
                [
                    f"synthetic-{file_index}-{row_index}-{request_index}"
                    for request_index in range(requests_per_agg)
                ]
            )
        file_table = base_table.append_column(
            "scene_id",
            _list_int64(pa, np, raw_rows_per_file, requests_per_agg, scene_values),
        ).append_column(
            "search_id",
            _list_strings(pa, search_ids),
        ).append_column(
            "impr_time",
            _list_int64(
                pa,
                np,
                raw_rows_per_file,
                requests_per_agg,
                np.asarray(request_times, dtype=np.int64),
            ),
        )
        present = projected & set(file_table.column_names)
        if projected_present is None:
            projected_present = present
        elif present != projected_present:
            raise RuntimeError("synthetic Parquet files produced inconsistent projected schemas")
        if first_arrow_bytes == 0:
            first_arrow_bytes = int(file_table.nbytes)
        path = output_dir / f"{file_index:06d}_0.gz.parquet"
        pq.write_table(
            file_table,
            path,
            compression=compression,
            use_dictionary=True,
            row_group_size=raw_rows_per_file,
            write_batch_size=min(64, raw_rows_per_file),
        )
        total_file_bytes += path.stat().st_size
        total_projected_bytes += _projected_compressed_bytes(path, projected)

    raw_rows = files * raw_rows_per_file
    candidates = raw_rows * candidates_per_agg
    manifest = SyntheticAggManifest(
        output_dir=str(output_dir),
        files=files,
        raw_rows_per_file=raw_rows_per_file,
        raw_rows=raw_rows,
        requests_per_agg=requests_per_agg,
        candidates_per_request=candidates_per_request,
        candidates=candidates,
        sequence_overlap=sequence_overlap,
        sequence_lengths_after_request_filter=lengths,
        raw_sequence_lengths=raw_sequence_lengths,
        bag_lengths=bag_lengths,
        physical_columns=len(base_table.column_names) + 3,
        # Optional adapter columns absent from the physical schema are not read
        # and therefore must not inflate the projected-column count.
        projected_columns=len(projected_present or ()),
        arrow_bytes_per_file=first_arrow_bytes,
        parquet_file_bytes=total_file_bytes,
        projected_compressed_bytes=total_projected_bytes,
        projected_compressed_bytes_per_candidate=(
            total_projected_bytes / candidates
        ),
        compression=compression,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--files", type=_positive_int, default=8)
    parser.add_argument("--raw-rows-per-file", type=_positive_int, default=8)
    parser.add_argument("--requests-per-agg", type=_positive_int, default=4)
    parser.add_argument("--candidates-per-request", type=_positive_int, default=8)
    parser.add_argument("--sequence-lengths", type=_named_lengths, default=None)
    parser.add_argument("--sequence-overlap", type=float, default=0.85)
    parser.add_argument("--bag-length-scale", type=float, default=1.0)
    parser.add_argument("--scenario-count", type=_positive_int, default=32)
    parser.add_argument("--physical-column-count", type=_positive_int, default=630)
    parser.add_argument("--compression", choices=("gzip", "zstd", "snappy"), default="gzip")
    args = parser.parse_args()

    config = load_app_config(args.config)
    manifest = generate_synthetic_agg_dataset(
        config,
        args.output_dir,
        files=args.files,
        raw_rows_per_file=args.raw_rows_per_file,
        requests_per_agg=args.requests_per_agg,
        candidates_per_request=args.candidates_per_request,
        sequence_lengths=args.sequence_lengths,
        sequence_overlap=args.sequence_overlap,
        bag_length_scale=args.bag_length_scale,
        scenario_count=args.scenario_count,
        physical_column_count=args.physical_column_count,
        compression=args.compression,
    )
    print(json.dumps(manifest.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
