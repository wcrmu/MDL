#!/usr/bin/env python3
"""Build a production model YAML from sample fields and optional stats.

The reference ``sample.yaml`` is authoritative for ordered fields, the 47/122
context/item split, UPS schemas, and labels.  It is deliberately *not*
authoritative for hash buckets, embedding widths, sequence lengths, scenes, or
training capacity.  Those values come from ``profile_prehashed_parquet.py``.

This command is offline: it reads YAML/JSON files only and never opens a local
or HDFS Parquet path.  A real profile report remains the production-quality
source of sizes.  ``--estimate-from-names`` emits an explicitly marked,
memory-planned starting config when that report cannot be transferred.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.profile_prehashed_parquet import (  # noqa: E402
    DEFAULT_SKU_FIELDS,
    ProfileSpec,
    load_profile_spec,
    profile_spec_from_mapping,
)
from src.config import AppConfig  # noqa: E402
from src.embeddings import (  # noqa: E402
    EmbeddingTableSpec,
    embedding_local_bytes,
    plan_embedding_shards,
)


CONTEXT_FEATURE_COUNT = 47
EXPECTED_FEATURE_COUNT = 169
EXPECTED_UPS_TYPES = (
    "impr",
    "clk_long",
    "view_long",
    "cart_long",
    "buy_long",
    "semi_clk",
    "srch_q2i",
    "ups_clk_sku",
    "flatten_query_hash",
)
SUPPORTED_MODELS = (
    "rankmixer",
    "mdl_rankmixer",
    "onetrans",
    "mdl_onetrans",
)
CONTEXT_SCALAR_FIELDS = {
    "currency_hn",
    "hash_language_site_hn",
    "language_hn",
    "page_elsn_hn",
    "page_sn_hn",
    "plat_hn",
    "region_hn",
    "scene_id_hn",
    "site_id_hn",
    "timezone_hn",
    "search_method_hn",
    "scene_clk_cnt_15d_hit_hn",
    "scene_impr_cnt_15d_hit_hn",
    "uid_or_bg_hn",
}
ITEM_BAG_FIELDS = {
    "sku_id_hn",
    "sku_price_v2_hn",
    "sku_sales_hn",
    "sku_spec_hash_hn",
    "sku_spec_hn",
    "sku_cart_cnt_7d_hn",
    "sku_ordr_cnt_1m_hn",
    "sku_price_dis_hn",
    "sku_sales_dis_hn",
    "goods_name_bigram_hn",
    "goods_ner_infos_hn",
    "goods_title_tfidf_term_hash_list_hn",
    "g_prpty_val_id_list_hn",
    "g_sku_spec_unit_list_hn",
    "g_sku_spec_hn",
    "g_sku_spec_hash_hn",
    "rev_ratings_cnt_crs_pos_hn",
}
CORE_ITEM_FIELDS = ("goods_id_hn", "cat1_id_hn", "price_hn")
SCENARIO_IMPORTANT_FIELDS = (
    "currency_hn",
    "hash_language_site_hn",
    "language_hn",
    "page_elsn_hn",
    "page_sn_hn",
)
TASK_IMPORTANT_FIELDS = (
    "currency_hn",
    "hash_language_site_hn",
    "language_hn",
)
SCENARIO_SHARED_PRIOR_UPS = ("impr", "clk_long", "view_long")
TASK_PRIOR_UPS = {
    "fst_cart": "cart_long",
    "upid_pay": "buy_long",
    "cateid_filter": "buy_long",
}
# Alias kept for Phase 2 / docs; both MDL surfaces share this contract.
MDL_MODELS = frozenset({"mdl_rankmixer", "mdl_onetrans"})
MDL_TASK_PRIOR_SOURCES = TASK_PRIOR_UPS
MDL_SCENARIO_SHARED_PRIORS = SCENARIO_SHARED_PRIOR_UPS
EXPECTED_LABELS = {
    "fst_cart": "label_fst_cart",
    "upid_pay": "upid_fst_trgt_noc_clk_pay_24h",
    "cateid_filter": "cateid_is_fst_scene_sp_filter",
}
OPTIONAL_FEATURE_COLUMNS = ("f_goods_view_times_tg_l1_hn",)
TIME_DELTA_FIELD = "time_delta_log1p_seconds"
AUTO_SCENARIO_NAME = "__auto__"
ESTIMATED_SEQUENCE_LENGTHS = {
    "impr": 1024,
    "clk_long": 2048,
    "view_long": 2048,
    "cart_long": 512,
    "buy_long": 256,
    "semi_clk": 128,
    "srch_q2i": 100,
    "ups_clk_sku": 200,
    "flatten_query_hash": 512,
}
# OneTrans performs event-level mixed causal attention before pyramid reduction.
# Keep the first-layer S-token capacity bounded independently from the longer
# history windows consumed by RankMixer's per-sequence LONGER encoders.  These
# are architecture capacity choices, not claims about observed data lengths.
ONETRANS_SEQUENCE_LENGTH_CAPS = {
    "impr": 256,
    "clk_long": 512,
    "view_long": 512,
    "cart_long": 192,
    "buy_long": 128,
    "semi_clk": 64,
    "srch_q2i": 100,
    "ups_clk_sku": 128,
    "flatten_query_hash": 156,
}
ONETRANS_NS_TOKENS = 32

EMBEDDING_PROFILES = (
    "baseline",
    "shared",
    "shared_dim",
    "shared_dim_query_bucket",
    "shared_dim_aggressive_bucket",
)
PHASE2_TASK_PRIOR_SEQUENCES = (
    "task_fst_cart_prior",
    "task_upid_pay_prior",
    "task_cateid_filter_prior",
)
# Buy-prior dedup excludes timegap (Phase 2 keeps independent prior timegap tables)
# and excludes spec/sku_ids (those already flatten onto cart_long via the dedicated
# PHASE2_SPEC/SKU alias lists below).
PHASE2_BUY_PRIOR_SHARE_FIELDS = (
    "cat1_id_hn",
    "cat2_id_hn",
    "cat3_id_hn",
    "cat4_id_hn",
    "cat_id_hn",
    "goods_id_hn",
    "mall_id_hn",
    "sales_hn",
    "price_hn",
)
PHASE2_TASK_PRIOR_BASE_SHARES = {
    "goods_id_hn": "goods_id_hn",
    "cat1_id_hn": "cat1_id_hn",
    "cat2_id_hn": "cat2_id_hn",
    "cat3_id_hn": "cat3_id_hn",
    "cat4_id_hn": "cat4_id_hn",
    "cat_id_hn": "cat_id_hn",
    "mall_id_hn": "mall_id_hn",
    "price_hn": "price_hn",
    "sales_hn": "sales_hn",
}
PHASE2_SPEC_SHARE_ALIASES = (
    "buy_long.spec_hn",
    "ups_clk_sku.spec_hn",
    "task_fst_cart_prior.spec_hn",
    "task_upid_pay_prior.spec_hn",
    "task_cateid_filter_prior.spec_hn",
)
PHASE2_SKU_LIST_SHARE_ALIASES = (
    "buy_long.sku_ids_hn",
    "task_fst_cart_prior.sku_ids_hn",
    "task_upid_pay_prior.sku_ids_hn",
    "task_cateid_filter_prior.sku_ids_hn",
)
# scenario_prior_scene_id_hn (auto) and scenario_<id>_prior_scene_id_hn (fixed)
# both share onto scene_id_hn via _scenario_prior_scene_aliases().


def _semantic_source(source: str) -> tuple[str | None, str]:
    if "_x_" not in source:
        return None, source
    prefix, semantic = source.split("_x_", 1)
    return prefix, semantic


def _estimated_bucket(source: str) -> int:
    """Conservative power-of-two bucket tier inferred only from field semantics."""

    prefix, semantic = _semantic_source(source)
    name = semantic.lower()

    # Entity identifiers dominate memory. Per-task history copies deliberately
    # use smaller tables than the union shared by the nine main UPS sequences.
    if name == "uid_or_bg_hn":
        return 1 << 26
    if name == "goods_id_hn":
        if prefix == "buy_long":
            return 1 << 24
        if prefix == "cart_long":
            return 1 << 25
        if prefix in {"semi_clk", "srch_q2i", "ups_clk_sku"}:
            return 1 << 24
        return 1 << 27
    if name == "sku_id_hn":
        return 1 << 26
    if name == "sku_ids_hn":
        if prefix == "buy_long":
            return 1 << 23
        if prefix == "cart_long":
            return 1 << 24
        return 1 << 25
    if name in {"query_hash_hn", "origin_query_hash_hn"}:
        return 1 << 25
    if name == "query_arr_hn":
        return 1 << 24
    if name == "flat_q_hash_hn":
        return 1 << 25
    if name in {
        "ups_clkv2_i2i_goods_ids_hit_size",
        "ups_clkv2_i2i_goods_ids_hit_all_size",
        "impr_cat_clk_goods_ids_cnt_1d_hn",
    }:
        return 1 << 12
    if "goods_ids" in name or "goods_id_list" in name or name == "ups_in_cart_goods_hn_share":
        return 1 << 24
    if "goods_cluster_id" in name:
        return 1 << 22

    if "sku_spec" in name or name in {
        "spec_hn",
        "buy_long_spec_vids_hn",
        "cart_long_spec_vids_hn",
    }:
        return 1 << 23
    if name == "g_prpty_val_id_list_hn":
        return 1 << 22
    if name == "goods_name_bigram_hn":
        return 1 << 22
    if name == "goods_ner_infos_hn":
        return 1 << 20
    if "term_hash" in name or "terms_hash" in name or "q2q_hash" in name:
        return 1 << 23
    if name.endswith("_tg_hn") or "query_tg" in name:
        return 1 << 18

    if name in {"mall_id_hn", "flip_mall_ids_hn"}:
        return 1 << 22
    if name == "ad_id_bin_hn":
        return 1 << 24
    if name == "campaign_id_hn":
        return 1 << 22
    if "creative_id" in name:
        return 1 << 24
    if name == "opt_id_hn":
        return 1 << 18

    if "cat1_id" in name or "cate1_id" in name:
        return 1 << 8
    if "cat2_id" in name or "cate2" in name:
        return 1 << 12
    if "cat3_id" in name:
        return 1 << 15
    if "cat4_id" in name:
        return 1 << 17
    if "cat_id" in name or "cate_id" in name:
        return 1 << 18
    if "cate_levels" in name:
        return 1 << 8

    small_exact = {
        "currency_hn": 1 << 7,
        "language_hn": 1 << 9,
        "hash_language_site_hn": 1 << 12,
        "page_elsn_hn": 1 << 12,
        "page_sn_hn": 1 << 13,
        "plat_hn": 1 << 6,
        "region_hn": 1 << 13,
        "scene_id_hn": 1 << 10,
        "site_id_hn": 1 << 10,
        "timezone_hn": 1 << 8,
        "search_method_hn": 1 << 8,
        "ups_search_method_hash_hn": 1 << 8,
        "sellr_type_hn": 1 << 8,
        "site_x_asian_code_hn": 1 << 10,
        "is_promotion_hn": 1 << 4,
        "timegap_hn": 1 << 6,
    }
    if name in small_exact:
        return small_exact[name]
    if "page_sns" in name or "page_elsns" in name:
        return 1 << 13
    if name.startswith(("clk_", "slide_", "switch_", "fvid_")) and prefix == "view_long":
        return 1 << 6

    # These names denote already discretized measurements or compact derived
    # categories, not entity IDs. The original ordering is unavailable after
    # hashing, so they remain categorical tables.
    numeric_markers = (
        "_cnt", "cnt_", "_size", "size_", "price", "prc", "sales",
        "gmv", "cvr", "ctr", "score", "level", "ratio", "discount",
        "timegap", "_time_hn", "_dis", "_hit", "_tg", "stay_time",
    )
    if any(marker in name for marker in numeric_markers):
        return 1 << 12
    return 1 << 18


def _estimated_dimension(bucket: int) -> int:
    if bucket <= 1 << 8:
        return 8
    if bucket <= 1 << 12:
        return 16
    if bucket <= 1 << 16:
        return 24
    if bucket <= 1 << 22:
        return 32
    if bucket <= 1 << 25:
        return 48
    return 64


def _estimated_bag_length(source: str) -> int:
    name = source.lower()
    if name.startswith("sku_"):
        return 128
    if name == "goods_name_bigram_hn":
        return 64
    if name == "goods_ner_infos_hn":
        return 32
    if "tfidf" in name:
        return 32
    if name == "g_prpty_val_id_list_hn":
        return 48
    if name.startswith("g_sku_spec"):
        return 32
    if name == "rev_ratings_cnt_crs_pos_hn":
        return 8
    if "flip_mall" in name or "7d_page" in name:
        return 512
    if "recall_merge" in name:
        return 256
    if "cart_7d" in name:
        return 256
    if "ups_in_cart_goods" in name or "ups_in_cart_tg" in name:
        return 256
    if "ups_" in name:
        return 200
    if "view_30m" in name:
        return 128
    if "query" in name:
        return 32
    if "_cnt_" in name or name.endswith("_cnt_hn"):
        return 32
    return 64


def _load_mapping(path: Path, *, kind: str) -> dict[str, Any]:
    if kind == "yaml":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _require_empty_counter(value: Any, path: str, errors: list[str]) -> None:
    mapping = _require_mapping(value, path)
    nonzero = {str(key): count for key, count in mapping.items() if count}
    if nonzero:
        errors.append(f"{path} contains violations: {nonzero}")


def validate_profile_report(report: Mapping[str, Any], spec: ProfileSpec) -> tuple[int, ...]:
    errors: list[str] = []
    if int(report.get("format_version", 0)) < 4:
        errors.append(
            "report format_version must be >=4; rerun scripts/profile_prehashed_parquet.py"
        )
    if int(report.get("rows_scanned", 0)) <= 0:
        errors.append("report rows_scanned must be positive")

    missing_by_input = _require_mapping(
        report.get("missing_configured_columns_by_input", {}),
        "missing_configured_columns_by_input",
    )
    for input_path, missing in missing_by_input.items():
        if missing:
            errors.append(f"input {input_path!r} is missing configured columns: {missing}")

    contract = _require_mapping(report.get("contract"), "contract")
    if contract.get("partial_indices_rows", 0):
        errors.append("contract.partial_indices_rows must be zero")
    for key in (
        "invalid_context_indices",
        "invalid_target_indices",
        "duplicate_context_indices",
        "target_without_context",
    ):
        if contract.get(key, 0):
            errors.append(f"contract.{key} must be zero")
    for key in (
        "context_outer_mismatches",
        "item_outer_mismatches",
        "label_length_mismatches",
        "sequence_length_mismatches",
        "invalid_sequence_membership",
        "missing_sequence_membership",
        "empty_sequence_membership",
        "time_order_violations",
        "event_after_request_time",
    ):
        _require_empty_counter(contract.get(key, {}), f"contract.{key}", errors)
    label_distribution = contract.get("label_distribution")
    if isinstance(label_distribution, Mapping):
        for task in EXPECTED_LABELS:
            counts = _require_mapping(
                label_distribution.get(task),
                f"contract.label_distribution.{task}",
            )
            if not counts.get("total", 0):
                errors.append(
                    f"contract.label_distribution.{task}.total must be positive"
                )
            if counts.get("other", 0):
                errors.append(
                    f"contract.label_distribution.{task}.other must be zero"
                )
            if counts.get("null", 0):
                errors.append(
                    f"contract.label_distribution.{task}.null must be zero"
                )
            if counts.get("minus_one", 0):
                errors.append(
                    f"contract.label_distribution.{task}.minus_one must be zero"
                )
    else:
        # Backward compatibility for format-v4 reports written before detailed
        # null/-1 categories were added. Those reports can prove only that every
        # label was binary, so any aggregate invalid count remains fatal.
        _require_empty_counter(
            contract.get("invalid_labels", {}),
            "contract.invalid_labels",
            errors,
        )
    for key in ("invalid_request_time", "invalid_request_time_layout"):
        if contract.get(key, 0):
            errors.append(f"contract.{key} must be zero")
    if contract.get("sku_alignment_mismatches", 0):
        errors.append("contract.sku_alignment_mismatches must be zero")

    field_reports = _require_mapping(report.get("fields"), "fields")
    for source in spec.categorical_sources:
        field = _require_mapping(field_reports.get(source), f"fields.{source}")
        if field.get("invalid_leaf_count", 0):
            errors.append(f"fields.{source}.invalid_leaf_count must be zero")
        if field.get("zero_count", 0):
            errors.append(
                f"fields.{source} contains non-null zero values; 0 is reserved for padding"
            )
        if field.get("rows_with_empty_list", 0):
            errors.append(
                f"fields.{source} contains empty arrays, contrary to the production contract"
            )
    for source in spec.time_sources:
        field = _require_mapping(field_reports.get(source), f"fields.{source}")
        if field.get("invalid_leaf_count", 0):
            errors.append(f"fields.{source}.invalid_leaf_count must be zero")
        if field.get("rows_with_empty_list", 0):
            errors.append(
                f"fields.{source} contains empty arrays, contrary to the production contract"
            )

    sequence_sources = {
        source
        for sources in spec.sequence_sources.values()
        for source in sources
    }
    for source in sequence_sources:
        field = _require_mapping(field_reports.get(source), f"fields.{source}")
        if field.get("rows_with_nested_null", 0):
            errors.append(
                f"S-token field {source!r} contains inner nulls; only top-level null "
                "may represent an empty sequence"
            )
        lengths = _require_mapping(
            field.get("list_lengths_by_depth", {}),
            f"fields.{source}.list_lengths_by_depth",
        )
        token_inner = lengths.get("1")
        if isinstance(token_inner, Mapping) and int(token_inner.get("max") or 0) != 1:
            errors.append(
                f"S-token field {source!r} has inner max length "
                f"{token_inner.get('max')}; expected singleton token values"
            )

    for source in CORE_ITEM_FIELDS:
        field = _require_mapping(field_reports.get(source), f"fields.{source}")
        nulls = _require_mapping(field.get("nulls_by_depth", {}), f"fields.{source}.nulls_by_depth")
        if int(nulls.get("1", 0)) > 0:
            errors.append(f"core item field {source!r} contains candidate-level nulls")

    context_sources = set(spec.context_sources)
    item_sources = set(spec.item_sources)
    missing_context_scalars = CONTEXT_SCALAR_FIELDS - context_sources
    if missing_context_scalars:
        errors.append(
            "the first 47 fields are missing expected scalar context fields: "
            + ", ".join(sorted(missing_context_scalars))
        )
    if not ITEM_BAG_FIELDS <= item_sources:
        errors.append(
            "the final 122 fields are missing expected item bags: "
            + ", ".join(sorted(ITEM_BAG_FIELDS - item_sources))
        )
    declared_bags = (context_sources - CONTEXT_SCALAR_FIELDS) | ITEM_BAG_FIELDS
    for source in (*spec.context_sources, *spec.item_sources):
        if source in declared_bags:
            continue
        field = _require_mapping(field_reports.get(source), f"fields.{source}")
        lengths = _require_mapping(
            field.get("list_lengths_by_depth", {}),
            f"fields.{source}.list_lengths_by_depth",
        )
        inner = lengths.get("1")
        if isinstance(inner, Mapping) and int(inner.get("max") or 0) > 1:
            errors.append(
                f"field {source!r} is configured scalar but observed inner max length "
                f"{inner.get('max')}"
            )

    sequence_lengths = _require_mapping(
        contract.get("sequence_lengths_after_request_filter"),
        "contract.sequence_lengths_after_request_filter",
    )
    for sequence in spec.sequence_sources:
        summary = _require_mapping(
            sequence_lengths.get(sequence),
            f"contract.sequence_lengths_after_request_filter.{sequence}",
        )
        if int(summary.get("count", 0)) <= 0:
            errors.append(f"sequence {sequence!r} has no per-request length observations")

    scene_rows = contract.get("scene_values", [])
    if not isinstance(scene_rows, list):
        raise ValueError("contract.scene_values must be a list")
    scene_ids = sorted(
        {
            int(item["scene_id"])
            for item in scene_rows
            if isinstance(item, Mapping) and "scene_id" in item
        }
    )
    if not scene_ids:
        errors.append("contract.scene_values must enumerate at least one raw scene_id")

    shared_reports = _require_mapping(
        report.get("shared_embedding_groups"),
        "shared_embedding_groups",
    )
    source_to_group = {
        source: root for root, sources in spec.shared_groups.items() for source in sources
    }
    for source in spec.categorical_sources:
        if source in source_to_group:
            root = source_to_group[source]
            stats = _require_mapping(
                shared_reports.get(root),
                f"shared_embedding_groups.{root}",
            )
            label = f"shared_embedding_groups.{root}"
        else:
            stats = _require_mapping(field_reports.get(source), f"fields.{source}")
            label = f"fields.{source}"
        bucket = stats.get("recommended_bucket_size")
        if bucket is None:
            errors.append(
                f"{label}.recommended_bucket_size is null; rerun the scanner with larger "
                "--candidate-buckets or more representative data"
            )
        elif (
            isinstance(bucket, bool)
            or not isinstance(bucket, int)
            or bucket <= 0
            or bucket & (bucket - 1)
        ):
            errors.append(f"{label}.recommended_bucket_size must be a positive power of two")
        dimension = stats.get("suggested_embedding_dim")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            errors.append(f"{label}.suggested_embedding_dim must be positive")

    if errors:
        raise ValueError("profile report is not production-ready:\n- " + "\n- ".join(errors))
    return tuple(scene_ids)


class ReportValues:
    def __init__(self, report: Mapping[str, Any], spec: ProfileSpec) -> None:
        self.report = report
        self.spec = spec
        self.fields = _require_mapping(report["fields"], "fields")
        self.shared = _require_mapping(
            report["shared_embedding_groups"], "shared_embedding_groups"
        )
        self.source_to_group = {
            source: root for root, sources in spec.shared_groups.items() for source in sources
        }

    @staticmethod
    def _rounded_dimension(stats: Mapping[str, Any]) -> int:
        dimension = int(stats["suggested_embedding_dim"])
        return max(8, int(math.ceil(dimension / 8.0) * 8))

    def categorical(self, source: str, *, shared: bool) -> tuple[int, int]:
        root = self.source_to_group.get(source) if shared else None
        stats = self.shared[root] if root is not None else self.fields[source]
        return int(stats["recommended_bucket_size"]), self._rounded_dimension(stats)

    def group_root(self, source: str) -> str | None:
        return self.source_to_group.get(source)

    def bag_length(self, source: str, quantile: str) -> int:
        lengths = _require_mapping(
            self.fields[source].get("list_lengths_by_depth", {}),
            f"fields.{source}.list_lengths_by_depth",
        )
        # Production profiling is over agg rows, where depth 0 is requests or
        # candidates and depth 1 is the per-request/per-item value list.
        inner = lengths.get("1")
        if not isinstance(inner, Mapping):
            inner = lengths.get("0", {})
        value = inner.get(quantile) if isinstance(inner, Mapping) else None
        return max(1, int(value or 1))

    def sequence_length(self, sequence: str, quantile: str) -> int:
        contract = _require_mapping(self.report["contract"], "contract")
        lengths = _require_mapping(
            contract["sequence_lengths_after_request_filter"],
            "contract.sequence_lengths_after_request_filter",
        )
        summary = _require_mapping(
            lengths[sequence],
            f"contract.sequence_lengths_after_request_filter.{sequence}",
        )
        return max(1, int(summary.get(quantile) or 1))


def _partition_inputs(split: Mapping[str, Any]) -> list[str]:
    """Resolve split inputs; empty when no paths or hour window is configured."""
    configured = split.get("inputs")
    if isinstance(configured, str) and configured:
        return [configured]
    if isinstance(configured, list) and configured:
        return [str(value) for value in configured]
    reader = split.get("reader", {})
    if not isinstance(reader, Mapping):
        return []
    partition = reader.get("partition")
    if not isinstance(partition, Mapping):
        return []
    base = str(partition.get("base_dir", "") or "").rstrip("/")
    start_raw = str(partition.get("start_hour", "") or "").strip()
    end_raw = str(partition.get("end_hour", "") or "").strip()
    # Default leave empty: callers may pass --train-input/--test-input later.
    if not start_raw and not end_raw:
        return []
    if not base or not start_raw or not end_raw:
        raise ValueError(
            "sample split needs inputs or reader.partition.{base_dir,start_hour,end_hour}"
        )
    try:
        start = datetime.strptime(start_raw, "%Y-%m-%d-%H")
        end = datetime.strptime(end_raw, "%Y-%m-%d-%H")
    except ValueError as error:
        raise ValueError("partition hours must use YYYY-MM-DD-HH") from error
    if end <= start:
        raise ValueError("partition end_hour must be later than start_hour")
    inputs: list[str] = []
    current = start
    while current < end:
        inputs.append(f"{base}/pt={current:%Y-%m-%d}/hr={current:%H}")
        current += timedelta(hours=1)
    return inputs


def _scene_slug(scene_id: int) -> str:
    prefix = f"neg_{abs(scene_id)}" if scene_id < 0 else str(scene_id)
    return re.sub(r"[^a-zA-Z0-9_]", "_", prefix)


def _encoding(
    *,
    bucket: int,
    share_with: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "pre_hashed",
        "num_buckets": bucket,
        "padding_id": 0,
    }
    if share_with is not None:
        result["share_with"] = share_with
        result["share_embedding"] = True
    return result


def _independent_feature(
    logical_name: str,
    source: str,
    scope: str,
    values: ReportValues,
) -> dict[str, Any]:
    bucket, dimension = values.categorical(source, shared=False)
    return {
        "name": logical_name,
        "kind": "categorical",
        "source": source,
        "embedding_scope": scope,
        "embedding_dim": dimension,
        "encoding": _encoding(bucket=bucket),
    }


def _main_encoding(
    logical_name: str,
    source: str,
    values: ReportValues,
) -> tuple[dict[str, Any], int]:
    bucket, dimension = values.categorical(source, shared=True)
    root = values.group_root(source)
    share_with = None if root is None or logical_name == root else root
    return _encoding(bucket=bucket, share_with=share_with), dimension


def _feature_bag_fields(sample_features: Sequence[Mapping[str, Any]]) -> set[str]:
    context_sources = {str(item["source"]) for item in sample_features[:CONTEXT_FEATURE_COUNT]}
    context_bags = context_sources - CONTEXT_SCALAR_FIELDS
    item_sources = {str(item["source"]) for item in sample_features[CONTEXT_FEATURE_COUNT:]}
    missing_item_bags = ITEM_BAG_FIELDS - item_sources
    if missing_item_bags:
        raise ValueError(
            "sample.yaml is missing expected item bag fields: "
            + ", ".join(sorted(missing_item_bags))
        )
    return context_bags | ITEM_BAG_FIELDS


def _estimated_length_summary(value: int) -> dict[str, int]:
    return {
        "count": 1,
        "min": value,
        "p50": value,
        "p95": value,
        "p99": value,
        "max": value,
    }


def build_name_estimate_report(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Create a scanner-shaped report using names only, with no data access."""

    raw_features = sample.get("features")
    if not isinstance(raw_features, list) or len(raw_features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"sample.yaml must contain exactly {EXPECTED_FEATURE_COUNT} features")
    spec = profile_spec_from_mapping(
        sample,
        context_feature_count=CONTEXT_FEATURE_COUNT,
    )
    bag_fields = _feature_bag_fields(raw_features)
    source_sequence = {
        source: sequence
        for sequence, sources in spec.sequence_sources.items()
        for source in sources
    }
    fields: dict[str, Any] = {}
    for source in spec.all_sources:
        sequence = source_sequence.get(source)
        if sequence is not None:
            outer_length = ESTIMATED_SEQUENCE_LENGTHS[sequence]
            inner_length = 1
        else:
            outer_length = 4
            inner_length = _estimated_bag_length(source) if source in bag_fields else 1
        bucket = _estimated_bucket(source)
        fields[source] = {
            "leaf_count": 1,
            "invalid_leaf_count": 0,
            "zero_count": 0,
            "rows_with_empty_list": 0,
            "rows_with_nested_null": 0,
            "nulls_by_depth": {},
            "list_lengths_by_depth": {
                "0": _estimated_length_summary(outer_length),
                "1": _estimated_length_summary(inner_length),
            },
            "recommended_bucket_size": bucket,
            "suggested_embedding_dim": _estimated_dimension(bucket),
            "estimate_basis": "field_name",
        }

    shared = {}
    for root, sources in spec.shared_groups.items():
        bucket = _estimated_bucket(root)
        shared[root] = {
            "sources": list(sources),
            "recommended_bucket_size": bucket,
            "suggested_embedding_dim": _estimated_dimension(bucket),
            "estimate_basis": "shared_semantic_name",
        }
    sequence_lengths = {
        name: _estimated_length_summary(ESTIMATED_SEQUENCE_LENGTHS[name])
        for name in spec.sequence_sources
    }
    return {
        "format_version": 4,
        "rows_scanned": 1,
        "files_scanned": [],
        "settings": {
            "mode": "name_heuristic",
            "warning": "bucket sizes, dimensions, and lengths were not measured from data",
        },
        "missing_configured_columns_by_input": {"name-estimate": []},
        "fields": fields,
        "shared_embedding_groups": shared,
        "contract": {
            "agg_rows": 0,
            "req_rows": 0,
            "partial_indices_rows": 0,
            "context_outer_mismatches": {},
            "item_outer_mismatches": {},
            "label_length_mismatches": {},
            "invalid_labels": {},
            "sequence_length_mismatches": {},
            "invalid_sequence_membership": {},
            "missing_sequence_membership": {},
            "empty_sequence_membership": {},
            "time_order_violations": {},
            "event_after_request_time": {},
            "invalid_context_indices": 0,
            "invalid_target_indices": 0,
            "duplicate_context_indices": 0,
            "target_without_context": 0,
            "invalid_request_time": 0,
            "invalid_request_time_layout": 0,
            "sku_alignment_mismatches": 0,
            "sequence_lengths_after_request_filter": sequence_lengths,
            # Ignored by auto-scene generation, but retained so the common
            # report contract remains structurally valid.
            "scene_values": [{"scene_id": 0, "count": 1}],
        },
        "notes": [
            "offline name-only estimate; replace bucket sizes after representative profiling",
            "raw scenes are discovered from train Parquet before model construction",
        ],
    }


def _main_features(
    sample_features: Sequence[Mapping[str, Any]],
    values: ReportValues,
    *,
    length_quantile: str,
    max_bag_length: int | None,
) -> tuple[list[dict[str, Any]], set[str]]:
    bag_fields = _feature_bag_fields(sample_features)
    sku_length = max(
        values.bag_length(source, length_quantile)
        for source in DEFAULT_SKU_FIELDS
    )
    if max_bag_length is not None:
        sku_length = min(sku_length, max_bag_length)
    result: list[dict[str, Any]] = []
    for raw in sample_features:
        name = str(raw["name"])
        source = str(raw["source"])
        encoding, dimension = _main_encoding(name, source, values)
        feature: dict[str, Any] = {
            "name": name,
            "kind": "categorical",
            "source": source,
            "embedding_scope": "feature",
            "embedding_dim": dimension,
            "encoding": encoding,
        }
        if source in bag_fields:
            max_length = (
                sku_length
                if source in DEFAULT_SKU_FIELDS
                else values.bag_length(source, length_quantile)
            )
            if max_bag_length is not None:
                max_length = min(max_length, max_bag_length)
            feature.update(
                {
                    "pooling": "mean",
                    "pooling_null_policy": (
                        "include_as_padding"
                        if source in DEFAULT_SKU_FIELDS
                        else "exclude"
                    ),
                    "max_length": max_length,
                    "truncation": "head",
                }
            )
        result.append(feature)
    return result, bag_fields


def _sequence_max_length(
    name: str,
    values: ReportValues,
    *,
    length_quantile: str,
    max_sequence_length: int | None,
    sequence_length_caps: Mapping[str, int] | None = None,
) -> int:
    length = values.sequence_length(name, length_quantile)
    limits = [length]
    if max_sequence_length is not None:
        limits.append(max_sequence_length)
    if sequence_length_caps is not None and name in sequence_length_caps:
        limits.append(int(sequence_length_caps[name]))
    return min(limits)


def _main_sequences(
    sample_sequences: Sequence[Mapping[str, Any]],
    values: ReportValues,
    *,
    length_quantile: str,
    max_sequence_length: int | None,
    encoder: str = "longer",
    sequence_length_caps: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    if encoder not in {"longer", "raw"}:
        raise ValueError("production main sequence encoder must be longer or raw")
    result: list[dict[str, Any]] = []
    for raw_sequence in sample_sequences:
        name = str(raw_sequence["name"])
        max_length = _sequence_max_length(
            name,
            values,
            length_quantile=length_quantile,
            max_sequence_length=max_sequence_length,
            sequence_length_caps=sequence_length_caps,
        )
        fields: list[dict[str, Any]] = []
        for raw_field in raw_sequence.get("fields", []):
            field_name = str(raw_field["name"])
            source = str(raw_field["source"])
            if field_name == "time" or source.endswith("_x_time"):
                fields.append(
                    {
                        "name": TIME_DELTA_FIELD,
                        "kind": "dense",
                        "source": f"{name}_x_{TIME_DELTA_FIELD}",
                        "dimension": 1,
                    }
                )
                continue
            logical_name = f"{name}.{field_name}"
            encoding, dimension = _main_encoding(logical_name, source, values)
            fields.append(
                {
                    "name": field_name,
                    "kind": "categorical",
                    "source": source,
                    "embedding_dim": dimension,
                    "encoding": encoding,
                }
            )
        sequence: dict[str, Any] = {
            "name": name,
            "embedding_scope": "feature",
            "max_length": max_length,
            "truncation": "head",
            "sequence_order": "newest_to_oldest",
            "encoder": encoder,
            "time_delta_field": TIME_DELTA_FIELD,
            "fields": fields,
        }
        if encoder == "longer":
            sequence.update(
                {
                    "target_inputs": [],
                    "rankmixer_summary_tokens": 1,
                    "longer_query_tokens": min(32, max_length),
                    "longer_self_layers": 1,
                    "longer_token_merge": 1,
                    "longer_inner_layers": 0,
                    "longer_output": "summary",
                    "longer_user_global_inputs": [],
                    "longer_user_global_tokens": 0,
                    "longer_cls_tokens": 1,
                    "longer_candidate_global_tokens": 0,
                }
            )
        result.append(sequence)
    return result


def _task_prior_sequence(
    task: str,
    source_sequence: Mapping[str, Any],
    values: ReportValues,
    *,
    length_quantile: str,
    max_sequence_length: int | None,
) -> dict[str, Any]:
    source_name = str(source_sequence["name"])
    fields: list[dict[str, Any]] = []
    for raw_field in source_sequence.get("fields", []):
        field_name = str(raw_field["name"])
        source = str(raw_field["source"])
        if field_name == "time" or source.endswith("_x_time"):
            fields.append(
                {
                    "name": TIME_DELTA_FIELD,
                    "kind": "dense",
                    "source": f"{source_name}_x_{TIME_DELTA_FIELD}",
                    "dimension": 1,
                }
            )
            continue
        bucket, dimension = values.categorical(source, shared=False)
        fields.append(
            {
                "name": field_name,
                "kind": "categorical",
                "source": source,
                "embedding_dim": dimension,
                "encoding": _encoding(bucket=bucket),
            }
        )
    return {
        "name": f"task_{task}_prior",
        "embedding_scope": "task",
        "max_length": _sequence_max_length(
            source_name,
            values,
            length_quantile=length_quantile,
            max_sequence_length=max_sequence_length,
        ),
        "truncation": "head",
        "sequence_order": "newest_to_oldest",
        "encoder": "mean_pool",
        "time_delta_field": TIME_DELTA_FIELD,
        "fields": fields,
    }


def _align_rankmixer_input_width(
    features: list[dict[str, Any]],
    main_sequence_count: int,
    *,
    token_count: int,
    token_dim: int,
    shared_sources: set[str],
) -> dict[str, Any] | None:
    main_features = features[:EXPECTED_FEATURE_COUNT]
    input_width = sum(int(feature["embedding_dim"]) for feature in main_features)
    input_width += main_sequence_count * token_dim
    remainder = input_width % token_count
    if remainder == 0:
        return None
    increment = token_count - remainder
    candidates = [
        feature
        for feature in main_features
        if str(feature["source"]) not in shared_sources
    ]
    target = (
        min(
            candidates,
            key=lambda feature: (
                int(feature["encoding"]["num_buckets"]),
                str(feature["name"]),
            ),
        )
        if candidates
        else None
    )
    if target is None:
        raise ValueError("cannot align RankMixer input width without changing a shared table")
    before = int(target["embedding_dim"])
    target["embedding_dim"] = before + increment
    return {
        "feature": target["name"],
        "before": before,
        "after": before + increment,
        "increment": increment,
        "input_width_before": input_width,
        "input_width_after": input_width + increment,
    }


def _adapter_options(
    sample_features: Sequence[Mapping[str, Any]],
    bag_fields: set[str],
    scene_ids: Sequence[int] | None,
    sequence_max_lengths: Mapping[str, int],
) -> dict[str, Any]:
    context = [str(item["source"]) for item in sample_features[:CONTEXT_FEATURE_COUNT]]
    items = [str(item["source"]) for item in sample_features[CONTEXT_FEATURE_COUNT:]]
    options: dict[str, Any] = {
        "context_features": context,
        "item_features": items,
        "multivalue_features": [
            str(item["source"])
            for item in sample_features
            if str(item["source"]) in bag_fields
        ],
        "aligned_multivalue_groups": [list(DEFAULT_SKU_FIELDS)],
        "ups_types": list(EXPECTED_UPS_TYPES),
        "request_columns": ["scene_id", "search_id", "impr_time"],
        "integer_request_columns": ["scene_id", "impr_time"],
        "labels": dict(EXPECTED_LABELS),
        "sequence_max_lengths": dict(sequence_max_lengths),
        "candidate_position_column": "candidate_position",
        "candidate_metadata_columns": ["example_ids"],
        "request_time_column": "impr_time",
        "time_delta_outputs": {
            name: f"{name}_x_{TIME_DELTA_FIELD}" for name in EXPECTED_UPS_TYPES
        },
        "time_delta_transform": "log1p_seconds",
    }
    if scene_ids is not None:
        options["request_value_maps"] = {
            "scene_id": {scene_id: index for index, scene_id in enumerate(scene_ids)}
        }
    return options


def _reader_config(*, training: bool) -> dict[str, Any]:
    result = {
        "engine": "pyarrow_dataset",
        "columns_pruning": True,
        "num_workers": 8,
        "prefetch_batches": 4,
        "max_prefetch_bytes": 2 * 1024**3,
        "scanner_batch_rows": 64,
        "pin_memory": True,
        "shard_unit": "row_group",
        "validate_prehashed_nonzero": False,
        "trusted_input": True,
    }
    if training:
        result.update({"shuffle_buffer_rows": 8192, "shuffle_seed": 2025})
    return result


def _split_config(
    inputs: Sequence[str],
    adapter_options: Mapping[str, Any],
    sequence_input_columns: Sequence[str],
    *,
    training: bool,
) -> dict[str, Any]:
    optional_feature_columns = set(OPTIONAL_FEATURE_COLUMNS)
    mandatory_columns = list(
        dict.fromkeys(
            [
                *adapter_options["context_features"],
                *(
                    source
                    for source in adapter_options["item_features"]
                    if source not in optional_feature_columns
                ),
                *sequence_input_columns,
                *adapter_options["request_columns"],
                *adapter_options["labels"].values(),
            ]
        )
    )
    optional_columns = [
        "context_indices",
        "target_indices",
        *OPTIONAL_FEATURE_COLUMNS,
        *(
            f"{ups_type}_x_indices"
            for ups_type in adapter_options["ups_types"]
        ),
    ]
    if not training:
        optional_columns.append("example_ids")
    split_config = {
        "format": "adapter_parquet",
        "inputs": list(inputs),
        "request_id": "search_id",
        "group_id": "search_id",
        "labels": dict(EXPECTED_LABELS),
        "reader": _reader_config(training=training),
        "adapter": {
            "callable": "src.dataloader:adapt_mdl_rankmixer_parquet",
            # Mandatory raw fields are always projected. Agg-only indices are
            # selected when present, preserving agg/req auto detection without
            # falling back to all ~630 Parquet columns.
            "input_columns": mandatory_columns,
            "optional_input_columns": optional_columns,
            "options": deepcopy(dict(adapter_options)),
        },
    }
    if not training:
        split_config.update(
            {
                "prediction_keys": {
                    "search_id": "search_id",
                    "candidate_position": "candidate_position",
                    "example_id": "example_ids",
                    "goods_id_hn": "goods_id_hn",
                },
                "prediction_score_suffix": "_score",
            }
        )
    return split_config


def _power_of_two_floor(value: int) -> int:
    return 1 << (max(1, value).bit_length() - 1)


def _iter_categorical_entries(payload: Mapping[str, Any]):
    for feature in payload["features"]:
        if feature["kind"] != "categorical":
            continue
        yield str(feature["name"]), feature
    for sequence in payload["sequences"]:
        sequence_name = str(sequence["name"])
        for field in sequence["fields"]:
            if field["kind"] != "categorical":
                continue
            yield f"{sequence_name}.{field['name']}", field


def _categorical_entries_by_name(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for name, entry in _iter_categorical_entries(payload):
        if name in entries:
            raise ValueError(f"duplicate categorical input name {name!r}")
        entries[name] = entry
    return entries


def _find_feature(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    for feature in payload["features"]:
        if feature["name"] == name:
            return feature
    raise KeyError(f"feature {name!r} not found")


def _find_sequence_field(payload: Mapping[str, Any], qualified_name: str) -> dict[str, Any]:
    if "." not in qualified_name:
        raise KeyError(f"sequence field {qualified_name!r} must be qualified as sequence.field")
    sequence_name, field_name = qualified_name.split(".", 1)
    for sequence in payload["sequences"]:
        if sequence["name"] != sequence_name:
            continue
        for field in sequence["fields"]:
            if field["name"] == field_name:
                return field
        raise KeyError(f"sequence field {qualified_name!r} not found")
    raise KeyError(f"sequence {sequence_name!r} not found")


def _find_categorical_entry(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    if "." in name:
        return _find_sequence_field(payload, name)
    return _find_feature(payload, name)


def _resolve_share_root(
    entries: Mapping[str, Mapping[str, Any]],
    name: str,
) -> str:
    seen: set[str] = set()
    current = name
    while True:
        if current in seen:
            raise ValueError(f"shared embedding cycle detected at {name!r}")
        seen.add(current)
        try:
            entry = entries[current]
        except KeyError as error:
            raise ValueError(
                f"shared embedding target {current!r} does not exist (from {name!r})"
            ) from error
        encoding = entry["encoding"]
        if not encoding.get("share_embedding"):
            return current
        target = encoding.get("share_with")
        if not target:
            raise ValueError(
                f"share_embedding=true requires share_with for categorical input {current!r}"
            )
        current = str(target)


def _set_embedding_shape(
    payload: Mapping[str, Any],
    table_name: str,
    *,
    num_buckets: int | None = None,
    embedding_dim: int | None = None,
) -> None:
    entries = _categorical_entries_by_name(payload)
    root_name = _resolve_share_root(entries, table_name)
    entry = entries[root_name]
    if num_buckets is not None:
        if num_buckets <= 0:
            raise ValueError("num_buckets must be positive")
        entry["encoding"]["num_buckets"] = int(num_buckets)
    if embedding_dim is not None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        entry["embedding_dim"] = int(embedding_dim)


def _share_embedding(
    payload: Mapping[str, Any],
    alias_name: str,
    base_name: str,
) -> None:
    entries = _categorical_entries_by_name(payload)
    if alias_name not in entries:
        raise KeyError(f"categorical alias {alias_name!r} not found")
    if base_name not in entries:
        raise KeyError(f"categorical base {base_name!r} not found")
    if alias_name == base_name:
        raise ValueError(f"categorical input {alias_name!r} cannot share with itself")
    # Reject cycles before mutating the live payload.
    probe = {
        name: {
            **entry,
            "encoding": dict(entry["encoding"]),
        }
        for name, entry in entries.items()
    }
    probe[alias_name]["encoding"]["share_embedding"] = True
    probe[alias_name]["encoding"]["share_with"] = base_name
    root = _resolve_share_root(probe, alias_name)

    physical = entries[root]
    alias = entries[alias_name]
    alias["embedding_dim"] = int(physical["embedding_dim"])
    encoding = alias["encoding"]
    encoding["type"] = physical["encoding"]["type"]
    encoding["num_buckets"] = int(physical["encoding"]["num_buckets"])
    encoding["padding_id"] = int(physical["encoding"]["padding_id"])
    encoding["share_embedding"] = True
    # Always point at the physical root so configs never keep multi-hop chains
    # like cateid.spec → upid.spec → cart_long.spec.
    encoding["share_with"] = root


def _propagate_shared_shapes(payload: Mapping[str, Any]) -> None:
    entries = _categorical_entries_by_name(payload)
    for name, entry in entries.items():
        encoding = entry["encoding"]
        if not encoding.get("share_embedding"):
            continue
        root_name = _resolve_share_root(entries, name)
        physical = entries[root_name]
        entry["embedding_dim"] = int(physical["embedding_dim"])
        encoding["type"] = physical["encoding"]["type"]
        encoding["num_buckets"] = int(physical["encoding"]["num_buckets"])
        encoding["padding_id"] = int(physical["encoding"]["padding_id"])
        encoding["share_with"] = root_name


def _validate_share_graph(payload: Mapping[str, Any]) -> None:
    entries = _categorical_entries_by_name(payload)
    for name, entry in entries.items():
        encoding = entry["encoding"]
        if not encoding.get("share_embedding"):
            continue
        target = encoding.get("share_with")
        if not target:
            raise ValueError(
                f"share_embedding=true requires share_with for categorical input {name!r}"
            )
        root_name = _resolve_share_root(entries, name)
        if target != root_name:
            raise ValueError(
                f"shared alias {name!r} share_with={target!r} must point at the "
                f"physical root {root_name!r} (multi-hop alias chains are not allowed)"
            )
        physical = entries[root_name]
        if int(entry["embedding_dim"]) != int(physical["embedding_dim"]):
            raise ValueError(
                f"shared alias {name!r} embedding_dim={entry['embedding_dim']} "
                f"does not match base {root_name!r} embedding_dim={physical['embedding_dim']}"
            )
        if int(encoding["num_buckets"]) != int(physical["encoding"]["num_buckets"]):
            raise ValueError(
                f"shared alias {name!r} num_buckets={encoding['num_buckets']} "
                f"does not match base {root_name!r} "
                f"num_buckets={physical['encoding']['num_buckets']}"
            )
        if int(encoding["padding_id"]) != int(physical["encoding"]["padding_id"]):
            raise ValueError(
                f"shared alias {name!r} padding_id={encoding['padding_id']} "
                f"does not match base {root_name!r} "
                f"padding_id={physical['encoding']['padding_id']}"
            )
        if encoding["type"] != physical["encoding"]["type"]:
            raise ValueError(
                f"shared alias {name!r} encoding type {encoding['type']!r} "
                f"does not match base {root_name!r} type {physical['encoding']['type']!r}"
            )


def _physical_table_count(payload: Mapping[str, Any]) -> int:
    return sum(
        1
        for _name, entry in _iter_categorical_entries(payload)
        if not entry["encoding"].get("share_embedding")
    )


def _sequence_has_field(payload: Mapping[str, Any], sequence_name: str, field_name: str) -> bool:
    for sequence in payload["sequences"]:
        if sequence["name"] != sequence_name:
            continue
        return any(field["name"] == field_name for field in sequence["fields"])
    return False


def _payload_has_task_priors(payload: Mapping[str, Any]) -> bool:
    names = {str(sequence["name"]) for sequence in payload["sequences"]}
    return all(name in names for name in PHASE2_TASK_PRIOR_SEQUENCES)


def _apply_phase2_common(payload: dict[str, Any]) -> None:
    """Shared Phase 2 table merges that apply to all four production models."""

    _set_embedding_shape(
        payload,
        "cart_long.spec_hn",
        num_buckets=1 << 23,
        embedding_dim=48,
    )
    _set_embedding_shape(
        payload,
        "cart_long.sku_ids_hn",
        num_buckets=1 << 24,
        embedding_dim=48,
    )
    for alias in ("buy_long.spec_hn", "ups_clk_sku.spec_hn"):
        if _sequence_has_field(payload, alias.split(".", 1)[0], alias.split(".", 1)[1]):
            _share_embedding(payload, alias, "cart_long.spec_hn")
    if _sequence_has_field(payload, "buy_long", "sku_ids_hn"):
        _share_embedding(payload, "buy_long.sku_ids_hn", "cart_long.sku_ids_hn")
    _propagate_shared_shapes(payload)
    _validate_share_graph(payload)


def _scenario_prior_scene_aliases(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return scenario-prior scene_id features for auto or fixed scene layouts."""

    aliases: list[str] = []
    for feature in payload["features"]:
        name = str(feature["name"])
        if name == "scenario_prior_scene_id_hn":
            aliases.append(name)
            continue
        if name.startswith("scenario_") and name.endswith("_prior_scene_id_hn"):
            aliases.append(name)
    return tuple(aliases)


def _apply_phase2_mdl_priors(payload: dict[str, Any]) -> None:
    """MDL-only prior alias merges; requires the three task-prior sequences."""

    if not _payload_has_task_priors(payload):
        raise ValueError(
            "Phase 2 MDL prior profile requires task_fst_cart_prior, "
            "task_upid_pay_prior, and task_cateid_filter_prior sequences"
        )
    for alias in PHASE2_SPEC_SHARE_ALIASES:
        _share_embedding(payload, alias, "cart_long.spec_hn")
    for alias in PHASE2_SKU_LIST_SHARE_ALIASES:
        _share_embedding(payload, alias, "cart_long.sku_ids_hn")

    for field_name in PHASE2_BUY_PRIOR_SHARE_FIELDS:
        alias = f"task_cateid_filter_prior.{field_name}"
        base = f"task_upid_pay_prior.{field_name}"
        if not _sequence_has_field(payload, "task_cateid_filter_prior", field_name):
            continue
        if not _sequence_has_field(payload, "task_upid_pay_prior", field_name):
            continue
        _share_embedding(payload, alias, base)

    for sequence_name in PHASE2_TASK_PRIOR_SEQUENCES:
        for field_name, base_name in PHASE2_TASK_PRIOR_BASE_SHARES.items():
            if not _sequence_has_field(payload, sequence_name, field_name):
                continue
            _share_embedding(payload, f"{sequence_name}.{field_name}", base_name)

    scenario_aliases = _scenario_prior_scene_aliases(payload)
    if not scenario_aliases:
        raise ValueError(
            "Phase 2 MDL prior profile requires scenario_prior_scene_id_hn "
            "or scenario_<id>_prior_scene_id_hn features"
        )
    for alias_name in scenario_aliases:
        _share_embedding(payload, alias_name, "scene_id_hn")

    _propagate_shared_shapes(payload)
    _validate_share_graph(payload)


def _apply_phase2_shared(payload: dict[str, Any]) -> None:
    """Merge duplicate tables onto shared physical bases for the payload family."""

    _apply_phase2_common(payload)
    if _payload_has_task_priors(payload):
        _apply_phase2_mdl_priors(payload)


def _apply_phase2_dim_compression(payload: dict[str, Any]) -> None:
    _set_embedding_shape(payload, "goods_id_hn", embedding_dim=48)
    _set_embedding_shape(payload, "uid_or_bg_hn", embedding_dim=48)
    _set_embedding_shape(payload, "sku_id_hn", embedding_dim=48)
    _set_embedding_shape(payload, "origin_query_hash_hn", embedding_dim=32)
    _set_embedding_shape(payload, "query_hash_hn", embedding_dim=32)
    _set_embedding_shape(
        payload,
        "flatten_query_hash.flat_q_hash_hn",
        embedding_dim=32,
    )
    _propagate_shared_shapes(payload)
    _validate_share_graph(payload)


def _apply_phase2_shared_dim(payload: dict[str, Any]) -> None:
    _apply_phase2_shared(payload)
    _apply_phase2_dim_compression(payload)


def _apply_phase2_shared_dim_query_bucket(payload: dict[str, Any]) -> None:
    _apply_phase2_shared_dim(payload)
    query_buckets = 1 << 24
    for table_name in (
        "origin_query_hash_hn",
        "query_hash_hn",
        "flatten_query_hash.flat_q_hash_hn",
    ):
        _set_embedding_shape(payload, table_name, num_buckets=query_buckets)
    _propagate_shared_shapes(payload)
    _validate_share_graph(payload)


def _apply_phase2_shared_dim_aggressive_bucket(payload: dict[str, Any]) -> None:
    _apply_phase2_shared_dim_query_bucket(payload)
    _set_embedding_shape(payload, "goods_id_hn", num_buckets=1 << 26)
    _set_embedding_shape(payload, "uid_or_bg_hn", num_buckets=1 << 25)
    _set_embedding_shape(payload, "sku_id_hn", num_buckets=1 << 25)
    _propagate_shared_shapes(payload)
    _validate_share_graph(payload)


def apply_embedding_profile(payload: dict[str, Any], profile: str) -> dict[str, Any]:
    """Mutate payload embedding tables according to a Phase 2 memory profile."""

    if profile not in EMBEDDING_PROFILES:
        raise ValueError(
            "embedding_profile must be one of " + ", ".join(EMBEDDING_PROFILES)
        )
    if profile == "baseline":
        _validate_share_graph(payload)
        return payload
    if profile == "shared":
        _apply_phase2_shared(payload)
    elif profile == "shared_dim":
        _apply_phase2_shared_dim(payload)
    elif profile == "shared_dim_query_bucket":
        _apply_phase2_shared_dim_query_bucket(payload)
    else:
        _apply_phase2_shared_dim_aggressive_bucket(payload)
    return payload


def _embedding_profile_checkpoint_path(model_name: str, profile: str) -> str:
    if profile == "baseline":
        return f"artifacts/checkpoints/{model_name}.pt"
    return f"artifacts/checkpoints/{model_name}_2xh100_phase2_{profile}"


def _embedding_memory_summary(
    payload: Mapping[str, Any],
    *,
    gpu_count: int,
    budget_gib_per_gpu: float,
    embedding_weight_dtype: str = "fp32",
    sparse_optimizer: str = "adagrad",
) -> dict[str, Any]:
    tables: list[tuple[str, int, int]] = []
    for feature in payload["features"]:
        if feature["kind"] != "categorical":
            continue
        encoding = feature["encoding"]
        if encoding.get("share_embedding"):
            continue
        tables.append(
            (str(feature["name"]), int(encoding["num_buckets"]), int(feature["embedding_dim"]))
        )
    for sequence in payload["sequences"]:
        for field in sequence["fields"]:
            if field["kind"] != "categorical":
                continue
            encoding = field["encoding"]
            if encoding.get("share_embedding"):
                continue
            tables.append(
                (
                    f"{sequence['name']}.{field['name']}",
                    int(encoding["num_buckets"]),
                    int(field["embedding_dim"]),
                )
            )
    weight_element_size = {"fp32": 4, "bf16": 2}[embedding_weight_dtype]
    optimizer_state_layout = (
        "rowwise" if sparse_optimizer == "rowwise_adagrad" else "full"
    )
    weight_bytes = sum(
        (bucket + 1) * dimension * weight_element_size
        for _name, bucket, dimension in tables
    )
    if optimizer_state_layout == "full":
        optimizer_state_bytes = sum(
            (bucket + 1) * dimension * 4 for _name, bucket, dimension in tables
        )
    else:
        optimizer_state_bytes = sum((bucket + 1) * 4 for _name, bucket, _dimension in tables)
    ideal_per_gpu = (weight_bytes + optimizer_state_bytes) / gpu_count
    table_specs = [
        EmbeddingTableSpec(
            name=name,
            num_embeddings=bucket + 1,
            embedding_dim=dimension,
            element_size=weight_element_size,
        )
        for name, bucket, dimension in tables
    ]
    sharding = payload["training"]["embedding_sharding"]
    plan = plan_embedding_shards(
        table_specs,
        world_size=gpu_count,
        strategy=sharding["strategy"],
        table_wise_max_rows=int(sharding["table_wise_max_rows"]),
        optimizer_state_layout=optimizer_state_layout,
    )
    per_gpu_bytes = [0] * gpu_count
    for table in table_specs:
        shard = plan.tables[table.name]
        for rank in range(gpu_count):
            per_gpu_bytes[rank] += embedding_local_bytes(
                rows=shard.local_rows(table.num_embeddings, rank),
                embedding_dim=table.embedding_dim,
                weight_element_size=table.element_size,
                optimizer_state_layout=optimizer_state_layout,
            )
    planned_per_gpu = max(per_gpu_bytes, default=0)
    gib = 1024**3
    largest = sorted(
        (
            {
                "name": name,
                "num_buckets": bucket,
                "embedding_dim": dimension,
                "weight_gib": (bucket + 1) * dimension * weight_element_size / gib,
            }
            for name, bucket, dimension in tables
        ),
        key=lambda item: item["weight_gib"],
        reverse=True,
    )[:20]
    summary = {
        "unique_tables": len(tables),
        "weight_gib_total": weight_bytes / gib,
        "optimizer_state_gib_total": optimizer_state_bytes / gib,
        "optimizer_state_layout": optimizer_state_layout,
        "embedding_weight_dtype": embedding_weight_dtype,
        "gpu_count": gpu_count,
        "ideal_weight_plus_state_gib_per_gpu": ideal_per_gpu / gib,
        "planned_weight_plus_state_gib_per_gpu": planned_per_gpu / gib,
        "planned_weight_plus_state_gib_by_gpu": [
            value / gib for value in per_gpu_bytes
        ],
        "sharding_plan_fingerprint": plan.fingerprint,
        "budget_gib_per_gpu": budget_gib_per_gpu,
        "largest_tables": largest,
    }
    if planned_per_gpu / gib > budget_gib_per_gpu:
        raise ValueError(
            "recommended collision-safe tables exceed the embedding budget: "
            f"planned {planned_per_gpu / gib:.2f} GiB/GPU > "
            f"{budget_gib_per_gpu:.2f} GiB/GPU. Review the report and choose an explicit "
            "collision/dimension tradeoff instead of silently shrinking buckets."
        )
    return summary


def build_config(
    sample: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    model_name: str = "mdl_rankmixer",
    train_inputs: Sequence[str] | None = None,
    test_inputs: Sequence[str] | None = None,
    length_quantile: str = "p99",
    max_sequence_length: int | None = None,
    max_bag_length: int | None = None,
    embedding_budget_gib_per_gpu: float = 80.0,
    event_token_budget_per_gpu: int = 262_144,
    batch_size: int | None = None,
    auto_discover_scenes: bool = False,
    gpu_count: int = 2,
    embedding_weight_dtype: str = "bf16",
    sparse_optimizer: str = "rowwise_adagrad",
    embedding_profile: str = "shared_dim",
) -> tuple[dict[str, Any], dict[str, Any]]:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            "model_name must be one of " + ", ".join(SUPPORTED_MODELS)
        )
    if length_quantile not in {"p95", "p99", "max"}:
        raise ValueError("length_quantile must be p95, p99, or max")
    if max_sequence_length is not None and max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    if max_bag_length is not None and max_bag_length <= 0:
        raise ValueError("max_bag_length must be positive")
    if embedding_budget_gib_per_gpu <= 0:
        raise ValueError("embedding_budget_gib_per_gpu must be positive")
    if event_token_budget_per_gpu <= 0:
        raise ValueError("event_token_budget_per_gpu must be positive")
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")
    if embedding_weight_dtype not in {"fp32", "bf16"}:
        raise ValueError("embedding_weight_dtype must be fp32 or bf16")
    if sparse_optimizer not in {"adagrad", "rowwise_adagrad"}:
        raise ValueError("sparse_optimizer must be adagrad or rowwise_adagrad")
    if embedding_profile not in EMBEDDING_PROFILES:
        raise ValueError(
            "embedding_profile must be one of " + ", ".join(EMBEDDING_PROFILES)
        )

    raw_features = sample.get("features")
    raw_sequences = sample.get("sequences")
    if not isinstance(raw_features, list) or len(raw_features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"sample.yaml must contain exactly {EXPECTED_FEATURE_COUNT} features")
    if not isinstance(raw_sequences, list):
        raise ValueError("sample.yaml sequences must be a list")
    sequence_names = tuple(str(sequence.get("name")) for sequence in raw_sequences)
    if sequence_names != EXPECTED_UPS_TYPES:
        raise ValueError(
            "sample.yaml UPS order must be " + ", ".join(EXPECTED_UPS_TYPES)
        )
    sample_labels = (
        sample.get("data", {}).get("train", {}).get("agg_layout", {}).get("labels", {})
    )
    if sample_labels != EXPECTED_LABELS:
        raise ValueError(f"sample.yaml labels must equal {EXPECTED_LABELS}")

    spec = profile_spec_from_mapping(
        sample,
        context_feature_count=CONTEXT_FEATURE_COUNT,
    )
    scene_ids = validate_profile_report(report, spec)
    values = ReportValues(report, spec)
    rankmixer_family = model_name in {"rankmixer", "mdl_rankmixer"}
    onetrans_family = model_name in {"onetrans", "mdl_onetrans"}
    mdl_family = model_name in {"mdl_rankmixer", "mdl_onetrans"}
    features, bag_fields = _main_features(
        raw_features,
        values,
        length_quantile=length_quantile,
        max_bag_length=max_bag_length,
    )
    main_sequences = _main_sequences(
        raw_sequences,
        values,
        length_quantile=length_quantile,
        max_sequence_length=max_sequence_length,
        encoder="raw" if onetrans_family else "longer",
        sequence_length_caps=(
            ONETRANS_SEQUENCE_LENGTH_CAPS if onetrans_family else None
        ),
    )

    scenario_important_names: list[str] = []
    scenario_prior_names: dict[int, str] = {}
    auto_scenario_prior_name: str | None = None
    task_important_names: list[str] = []
    if mdl_family:
        for source in SCENARIO_IMPORTANT_FIELDS:
            name = f"scenario_important_{source}"
            features.append(_independent_feature(name, source, "scenario", values))
            scenario_important_names.append(name)
        if auto_discover_scenes:
            auto_scenario_prior_name = "scenario_prior_scene_id_hn"
            features.append(
                _independent_feature(
                    auto_scenario_prior_name,
                    "scene_id_hn",
                    "scenario",
                    values,
                )
            )
        else:
            for scene_id in scene_ids:
                name = f"scenario_{_scene_slug(scene_id)}_prior_scene_id_hn"
                features.append(
                    _independent_feature(name, "scene_id_hn", "scenario", values)
                )
                scenario_prior_names[scene_id] = name
        for source in TASK_IMPORTANT_FIELDS:
            name = f"task_important_{source}"
            features.append(_independent_feature(name, source, "task", values))
            task_important_names.append(name)

    sequence_by_name = {str(sequence["name"]): sequence for sequence in raw_sequences}
    task_prior_sequences = (
        [
            _task_prior_sequence(
                task,
                sequence_by_name[ups],
                values,
                length_quantile=length_quantile,
                max_sequence_length=max_sequence_length,
            )
            for task, ups in MDL_TASK_PRIOR_SOURCES.items()
        ]
        if mdl_family
        else []
    )
    sequences = [*main_sequences, *task_prior_sequences]

    adjustment = (
        _align_rankmixer_input_width(
            features,
            len(main_sequences),
            token_count=32,
            token_dim=768,
            shared_sources=set(values.source_to_group),
        )
        if rankmixer_family
        else None
    )

    if auto_discover_scenes:
        scenario_names = [AUTO_SCENARIO_NAME]
        scenario_config: dict[str, Any] = {
            "names": scenario_names,
            "source": "scene_id",
            "source_encoding": "raw",
            "auto_discover": True,
            "max_discovered": 64,
        }
        if mdl_family:
            if auto_scenario_prior_name is None:  # Defensive type narrowing.
                raise RuntimeError("auto scenario prior was not constructed")
            scenario_priors = list(MDL_SCENARIO_SHARED_PRIORS)
            scenario_tokens = [
                {
                    "name": AUTO_SCENARIO_NAME,
                    "important_inputs": scenario_important_names,
                    "prior_inputs": [auto_scenario_prior_name, *scenario_priors],
                },
                {
                    "name": "global",
                    "important_inputs": scenario_important_names,
                    "prior_inputs": scenario_priors,
                },
            ]
        else:
            scenario_tokens = []
    else:
        scenario_names = [str(scene_id) for scene_id in scene_ids]
        scenario_config = {
            "names": scenario_names,
            "source": "scene_id",
            "source_encoding": "index",
        }
        if mdl_family:
            scenario_priors = list(MDL_SCENARIO_SHARED_PRIORS)
            scenario_tokens = [
                {
                    "name": str(scene_id),
                    "important_inputs": scenario_important_names,
                    "prior_inputs": [
                        scenario_prior_names[scene_id],
                        *scenario_priors,
                    ],
                }
                for scene_id in scene_ids
            ]
            scenario_tokens.append(
                {
                    "name": "global",
                    "important_inputs": scenario_important_names,
                    "prior_inputs": scenario_priors,
                }
            )
        else:
            scenario_tokens = []
    task_tokens = (
        [
            {
                "name": task,
                "important_inputs": task_important_names,
                "prior_inputs": [f"task_{task}_prior"],
            }
            for task in EXPECTED_LABELS
        ]
        if mdl_family
        else []
    )

    data_payload = _require_mapping(sample.get("data"), "sample.data")
    sample_train = _require_mapping(data_payload.get("train"), "sample.data.train")
    resolved_train_inputs = (
        list(train_inputs) if train_inputs else _partition_inputs(sample_train)
    )
    sample_test = data_payload.get("test")
    if test_inputs:
        resolved_test_inputs = list(test_inputs)
    elif isinstance(sample_test, Mapping):
        resolved_test_inputs = _partition_inputs(sample_test)
    else:
        resolved_test_inputs = []

    adapter_sequence_limits = {
        str(sequence["name"]): int(sequence["max_length"])
        for sequence in main_sequences
    }
    for task, ups in MDL_TASK_PRIOR_SOURCES.items():
        prior_name = f"task_{task}_prior"
        prior = next(
            (
                sequence
                for sequence in task_prior_sequences
                if sequence["name"] == prior_name
            ),
            None,
        )
        if prior is None:
            continue
        adapter_sequence_limits[ups] = max(
            int(adapter_sequence_limits[ups]),
            int(prior["max_length"]),
        )
    adapter_options = _adapter_options(
        raw_features,
        bag_fields,
        None if auto_discover_scenes else scene_ids,
        adapter_sequence_limits,
    )
    derived_time_columns = set(adapter_options["time_delta_outputs"].values())
    sequence_input_columns = list(
        dict.fromkeys(
            [
                *(
                    str(field["source"])
                    for sequence in main_sequences
                    for field in sequence["fields"]
                    if str(field["source"]) not in derived_time_columns
                ),
                *(
                    f"{ups_type}_x_time"
                    for ups_type in adapter_options["ups_types"]
                ),
            ]
        )
    )
    data: dict[str, Any] = {
        "train": _split_config(
            resolved_train_inputs,
            adapter_options,
            sequence_input_columns,
            training=True,
        ),
        "schema_policy": {
            "require_same_schema": True,
            "allow_missing_nullable_columns": False,
            "validate_before_train": True,
        },
    }
    if resolved_test_inputs:
        data["test"] = _split_config(
            resolved_test_inputs,
            adapter_options,
            sequence_input_columns,
            training=False,
        )
    elif isinstance(sample_test, Mapping):
        # Keep a test split shell even when hour windows / inputs are left empty.
        data["test"] = _split_config(
            [],
            adapter_options,
            sequence_input_columns,
            training=False,
        )

    total_main_sequence_length = sum(
        int(sequence["max_length"]) for sequence in main_sequences
    )
    total_task_prior_length = sum(
        int(sequence["max_length"]) for sequence in task_prior_sequences
    )
    total_sequence_length = total_main_sequence_length + total_task_prior_length
    if batch_size is None:
        raw_batch = max(8, event_token_budget_per_gpu // max(1, total_sequence_length))
        resolved_batch_size = min(256, _power_of_two_floor(raw_batch))
        if model_name == "onetrans":
            resolved_batch_size = min(resolved_batch_size, 32)
        elif model_name == "mdl_onetrans":
            resolved_batch_size = min(resolved_batch_size, 16)
    else:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        resolved_batch_size = batch_size

    if rankmixer_family:
        tokenization: dict[str, Any] = {
            "feature_tokenizer": "rankmixer",
            "num_feature_tokens": 32,
            "feature_token_inputs": [
                *[str(feature["name"]) for feature in raw_features],
                *list(EXPECTED_UPS_TYPES),
            ],
        }
    else:
        tokenization = {
            "feature_tokenizer": "auto_split",
            "num_feature_tokens": ONETRANS_NS_TOKENS,
            "feature_token_inputs": [
                str(feature["name"]) for feature in raw_features
            ],
            "sequence_tokens": [
                {"name": name, "inputs": [name]}
                for name in EXPECTED_UPS_TYPES
            ],
        }
    if mdl_family:
        tokenization["task_tokens"] = task_tokens
        tokenization["scenario_tokens"] = scenario_tokens

    if rankmixer_family:
        model: dict[str, Any] = {
            "name": model_name,
            "embedding_dim": 32,
            "token_dim": 768,
            "num_layers": 2,
            "num_heads": 12,
            "hidden_dim": 1536,
            "init_std": 0.02,
            "ffn_activation": "gelu",
            "task_head_hidden_dim": 1536,
            "task_head_dropout": 0.0,
            "task_head_activation": "gelu",
            "rankmixer_ffn_type": "dense",
            "sequence_fusion": "intent_ordered",
            "use_task_tokens": mdl_family,
            "use_scenario_tokens": mdl_family,
            "use_global_scenario_token": mdl_family,
            "use_task_feature_interaction": mdl_family,
            "use_scenario_feature_interaction": mdl_family,
            "mdl_feature_interaction": "direct_ffn",
            "use_request_cache": True,
        }
        onetrans_s_token_capacity: int | None = None
    else:
        separator_tokens = len(main_sequences) - 1
        onetrans_s_token_capacity = total_main_sequence_length + separator_tokens
        model = {
            "name": model_name,
            "embedding_dim": 32,
            "token_dim": 256,
            "num_layers": 6,
            "num_heads": 4,
            "hidden_dim": 1024,
            "init_std": 0.02,
            "ffn_activation": "gelu",
            "task_head_hidden_dim": 1024,
            "task_head_dropout": 0.0,
            "task_head_activation": "gelu",
            "sequence_fusion": "intent_ordered",
            "ns_tokenizer": "auto_split",
            "num_ns_tokens": ONETRANS_NS_TOKENS,
            "max_position_embeddings": (
                onetrans_s_token_capacity + ONETRANS_NS_TOKENS
            ),
            "use_sep_tokens": True,
            "use_pyramid": True,
            "pyramid_round_to": 32,
            "final_s_tokens": 12,
            "use_task_tokens": mdl_family,
            "use_scenario_tokens": mdl_family,
            "use_global_scenario_token": mdl_family,
            "use_task_feature_interaction": mdl_family,
            "use_scenario_feature_interaction": mdl_family,
            "mdl_feature_interaction": "direct_ffn",
            "use_request_cache": True,
        }
        if model_name == "mdl_onetrans":
            model.update(
                {
                    "first_domain_sequence_layer": 4,
                    "experimental_model_acknowledged": True,
                }
            )

    payload: dict[str, Any] = {
        "runtime": {
            "device": "cuda",
            "precision": "bf16",
            "compile": False,
            "compile_mode": "default",
            "require_compact_sequence_batches": False,
            "allow_tf32": True,
            "activation_checkpoint": "none",
            "attention_backend": "sdpa",
            "distributed": "ddp",
            "nproc_per_node": gpu_count,
            "master_addr": "127.0.0.1",
            "master_port": 29500,
        },
        "data": data,
        "features": features,
        "sequences": sequences,
        "scenarios": scenario_config,
        "tokenization": tokenization,
        "vocab_strategy": {
            "defaults": {
                "fit_split": "train",
                "oov_id": 0,
                "padding_id": 0,
                "unseen_policy": "oov",
                "artifact_dir": "artifacts/vocab",
            },
            "features": {},
        },
        "model": model,
        "training": {
            "batch_size": resolved_batch_size,
            "embedding_distribution": "sharded",
            "dense_distribution": "ddp",
            "embedding_sharding": {
                "strategy": "auto",
                "local_dedup": True,
                "table_wise_max_rows": 65536,
            },
            "ddp": {
                "static_graph": False,
                "find_unused_parameters": True,
                "gradient_as_bucket_view": True,
                "bucket_cap_mb": 25.0,
                "audit_steps": 10,
                "validated_no_unused_parameters": False,
                "validated_static_graph": False,
            },
            "lr_dense": 0.001,
            "lr_sparse": 0.001,
            "lr_schedule": "constant",
            "lr_warmup_steps": 500,
            "lr_decay_steps": None,
            "lr_min_ratio": 0.0,
            "dense_optimizer": "rmsprop",
            "fused_dense_optimizer": True,
            "rmsprop_alpha": 0.99999,
            "rmsprop_momentum": 0.0,
            "sparse_optimizer": sparse_optimizer,
            "adagrad_initial_accumulator_value": 0.1,
            "adagrad_eps": 1.0e-10,
            "embedding_sparse_gradients": True,
            "embedding_weight_dtype": embedding_weight_dtype,
            "sparse_update_mode": "ddp_synced_adagrad",
            "sparse_parameter_server_adapter": None,
            "dense_clip_norm": 1.0,
            "sparse_clip_norm": 1.0,
            "loss_reduction": "mean_per_task",
            "quick_eval": {
                "enabled": True,
                "every_steps": 1000,
                "max_batches": 20,
                "split": "train",
                "auc_bins": 4096,
            },
            "checkpoint_path": _embedding_profile_checkpoint_path(
                model_name,
                embedding_profile,
            ),
            "save_checkpoint": False,
        },
    }

    apply_embedding_profile(payload, embedding_profile)
    if rankmixer_family:
        # Dim/bucket profiles can change feature widths after the initial align.
        adjustment = _align_rankmixer_input_width(
            payload["features"],
            len(main_sequences),
            token_count=32,
            token_dim=768,
            shared_sources=set(values.source_to_group),
        )

    # Validate the exact in-memory config before any output file is replaced.
    config = AppConfig.from_mapping(payload)
    config.validate()
    memory = _embedding_memory_summary(
        payload,
        gpu_count=gpu_count,
        budget_gib_per_gpu=embedding_budget_gib_per_gpu,
        embedding_weight_dtype=embedding_weight_dtype,
        sparse_optimizer=sparse_optimizer,
    )
    summary = {
        "model_name": model_name,
        "embedding_profile": embedding_profile,
        "physical_embedding_tables": memory["unique_tables"],
        "profile": {
            "format_version": report.get("format_version"),
            "rows_scanned": report.get("rows_scanned"),
            "files_scanned": len(report.get("files_scanned", [])),
            "settings": report.get("settings", {}),
        },
        "scene_id_to_index": (
            {"mode": "auto_discover_from_train"}
            if auto_discover_scenes
            else {str(scene_id): index for index, scene_id in enumerate(scene_ids)}
        ),
        "context_feature_count": CONTEXT_FEATURE_COUNT,
        "item_feature_count": EXPECTED_FEATURE_COUNT - CONTEXT_FEATURE_COUNT,
        "bag_feature_count": len(bag_fields),
        "main_ups_count": len(main_sequences),
        "task_prior_count": len(task_prior_sequences),
        "sequence_length_quantile": length_quantile,
        "main_sequence_max_lengths": {
            sequence["name"]: sequence["max_length"] for sequence in main_sequences
        },
        "task_prior_max_lengths": {
            sequence["name"]: sequence["max_length"]
            for sequence in task_prior_sequences
        },
        "total_main_sequence_length": total_main_sequence_length,
        "total_task_prior_length": total_task_prior_length,
        "total_encoded_sequence_length": total_sequence_length,
        "onetrans_s_token_capacity": onetrans_s_token_capacity,
        "event_token_budget_per_gpu": event_token_budget_per_gpu,
        "batch_size_per_gpu": resolved_batch_size,
        "rankmixer_alignment_adjustment": adjustment,
        "embedding_memory": memory,
    }
    return payload, summary


def render_config(payload: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    model_name = str(summary["model_name"])
    adjustment = summary.get("rankmixer_alignment_adjustment")
    adjustment_comment = (
        "none"
        if adjustment is None
        else f"{adjustment['feature']} {adjustment['before']}->{adjustment['after']}"
    )
    memory = summary["embedding_memory"]
    estimate_mode = summary["profile"].get("settings", {}).get("mode") == "name_heuristic"
    sizing_comment = (
        "# IMPORTANT: buckets/dimensions/lengths are name-only estimates; replace them after profiling."
        if estimate_mode
        else "# Buckets/dimensions/lengths come from the supplied Parquet profile JSON."
    )
    comments = [
        "# Generated by scripts/build_mdl_rankmixer_config.py; do not copy bucket sizes from sample.yaml.",
        f"# Production model surface: {model_name}.",
        "# Fields and ordering come from sample.yaml.",
        sizing_comment,
        f"# Profile rows scanned: {summary['profile']['rows_scanned']}; "
        f"files scanned: {summary['profile']['files_scanned']}.",
        f"# Raw scene_id -> model index: {summary['scene_id_to_index']}",
    ]
    if model_name in {"rankmixer", "mdl_rankmixer"}:
        comments.append(
            f"# RankMixer divisibility adjustment: {adjustment_comment}"
        )
    else:
        comments.append(
            "# OneTrans capacity includes intent separators but excludes NS tokens: "
            f"{summary['onetrans_s_token_capacity']}."
        )
    comments.extend(
        [
            f"# Embedding profile: {summary.get('embedding_profile', 'baseline')}; "
            f"physical tables: {summary.get('physical_embedding_tables', memory['unique_tables'])}.",
            "# Estimated sharded embedding weight + optimizer state: "
            f"{memory['planned_weight_plus_state_gib_per_gpu']:.2f} GiB/GPU "
            f"({memory['optimizer_state_layout']}, {memory['embedding_weight_dtype']}, "
            f"{memory['gpu_count']} GPU) within a {memory['budget_gib_per_gpu']:.2f} "
            "GiB/GPU planning budget.",
            "# Batch size is a conservative architecture-aware starting point; profile it before a long run.",
            "",
        ]
    )
    return "\n".join(comments) + yaml.safe_dump(
        dict(payload),
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=Path("sample.yaml"))
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="mdl_rankmixer",
        help="Model surface to generate (default: mdl_rankmixer).",
    )
    sizing = parser.add_mutually_exclusive_group(required=True)
    sizing.add_argument("--report", type=Path)
    sizing.add_argument(
        "--estimate-from-names",
        action="store_true",
        help="Estimate power-of-two buckets/dimensions/lengths without reading Parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output YAML (default: configs/<model>.yaml).",
    )
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--train-input", action="append")
    parser.add_argument("--test-input", action="append")
    parser.add_argument("--length-quantile", choices=("p95", "p99", "max"), default="p99")
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--max-bag-length", type=int)
    parser.add_argument("--embedding-budget-gib-per-gpu", type=float, default=80.0)
    parser.add_argument("--event-token-budget-per-gpu", type=int, default=262_144)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gpu-count", type=int, default=2)
    parser.add_argument(
        "--embedding-weight-dtype",
        choices=("fp32", "bf16"),
        default="bf16",
    )
    parser.add_argument(
        "--sparse-optimizer",
        choices=("adagrad", "rowwise_adagrad"),
        default="rowwise_adagrad",
    )
    parser.add_argument(
        "--embedding-profile",
        choices=EMBEDDING_PROFILES,
        default="shared_dim",
        help=(
            "Phase 2 embedding memory profile: baseline, shared tables, "
            "shared+dim, and optional bucket compression tiers. "
            "Default shared_dim targets 2×H100 + BF16 + Row-Wise."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        sample = _load_mapping(args.sample, kind="yaml")
        report = (
            build_name_estimate_report(sample)
            if args.estimate_from_names
            else _load_mapping(args.report, kind="json")
        )
        # Verify the file-backed scanner parser agrees with the generator's
        # in-memory parser before consuming its group reports.
        file_spec = load_profile_spec(args.sample, context_feature_count=CONTEXT_FEATURE_COUNT)
        memory_spec = profile_spec_from_mapping(
            sample,
            context_feature_count=CONTEXT_FEATURE_COUNT,
        )
        if file_spec != memory_spec:
            raise ValueError("internal sample parser disagrees with profile scanner grouping")
        payload, summary = build_config(
            sample,
            report,
            model_name=args.model,
            train_inputs=args.train_input,
            test_inputs=args.test_input,
            length_quantile=args.length_quantile,
            max_sequence_length=args.max_sequence_length,
            max_bag_length=args.max_bag_length,
            embedding_budget_gib_per_gpu=args.embedding_budget_gib_per_gpu,
            event_token_budget_per_gpu=args.event_token_budget_per_gpu,
            batch_size=args.batch_size,
            auto_discover_scenes=args.estimate_from_names,
            gpu_count=args.gpu_count,
            embedding_weight_dtype=args.embedding_weight_dtype,
            sparse_optimizer=args.sparse_optimizer,
            embedding_profile=args.embedding_profile,
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    rendered = render_config(payload, summary)
    output = args.output or Path("configs") / f"{args.model}.yaml"
    if args.dry_run:
        sys.stdout.write(rendered)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
