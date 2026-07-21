#!/usr/bin/env python3
"""Profile production Parquet fields before choosing hash buckets.

The production ``*_hn`` values are opaque, non-zero int64 bit patterns.  This
script deliberately does not run the training dataloader and does not encode
values.  It scans explicitly supplied local or HDFS Parquet inputs, summarizes
the physical contract, and estimates collisions for the power-of-two bucket
mapping used by ``pre_hashed`` inputs::

    index = (value & (bucket_size - 1)) + 1

``--input`` is mandatory so running this script on a development machine can
never accidentally probe the HDFS path embedded in a reference YAML file.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import heapq
from itertools import combinations
import json
import math
from numbers import Real
from pathlib import Path
import sys
from typing import Any, Collection, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import yaml


MASK64 = (1 << 64) - 1
# Include deliberately oversized candidates so a costly Parquet scan does not
# have to be repeated merely to diagnose that a strict collision target is
# incompatible with the available embedding-memory budget.
DEFAULT_BUCKETS = tuple(1 << exponent for exponent in range(10, 37))
DEFAULT_SKU_FIELDS = (
    "sku_id_hn",
    "sku_price_v2_hn",
    "sku_sales_hn",
    "sku_spec_hash_hn",
    "sku_spec_hn",
    "sku_cart_cnt_7d_hn",
    "sku_ordr_cnt_1m_hn",
    "sku_price_dis_hn",
    "sku_sales_dis_hn",
)
# Spec ID may be null/[] while other SKU bags stay length-aligned; keep it out of
# adapter aligned_multivalue_groups while still sharing SKU bag max_length/pooling.
ALIGNED_SKU_FIELDS = tuple(
    name for name in DEFAULT_SKU_FIELDS if name != "sku_spec_hn"
)


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow.dataset as ds
        import pyarrow.fs as fs
    except ImportError as error:  # pragma: no cover - depends on runtime setup.
        raise RuntimeError("profile_prehashed_parquet requires pyarrow>=14") from error
    return ds, fs


def _mix64(value: int) -> int:
    """SplitMix64 finalizer over the unchanged uint64 bit pattern."""

    value = (value & MASK64) ^ ((value & MASK64) >> 30)
    value = (value * 0xBF58476D1CE4E5B9) & MASK64
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


class HyperLogLog:
    """Small dependency-free cardinality sketch mergeable across fields."""

    def __init__(self, precision: int = 12) -> None:
        if not 4 <= precision <= 16:
            raise ValueError("HLL precision must be between 4 and 16")
        self.precision = precision
        self.registers = bytearray(1 << precision)

    def add(self, value: int) -> None:
        hashed = _mix64(value)
        index = hashed & ((1 << self.precision) - 1)
        remainder = hashed >> self.precision
        width = 64 - self.precision
        rank = width + 1 if remainder == 0 else width - remainder.bit_length() + 1
        if rank > self.registers[index]:
            self.registers[index] = rank

    def merge(self, other: "HyperLogLog") -> None:
        if other.precision != self.precision:
            raise ValueError("cannot merge HLL sketches with different precision")
        for index, value in enumerate(other.registers):
            if value > self.registers[index]:
                self.registers[index] = value

    def estimate(self) -> float:
        size = len(self.registers)
        if size == 16:
            alpha = 0.673
        elif size == 32:
            alpha = 0.697
        elif size == 64:
            alpha = 0.709
        else:
            alpha = 0.7213 / (1.0 + 1.079 / size)
        raw = alpha * size * size / sum(2.0 ** (-register) for register in self.registers)
        zero_registers = self.registers.count(0)
        if zero_registers and raw <= 2.5 * size:
            return size * math.log(size / zero_registers)
        return raw


class BottomKValues:
    """Deterministic bounded distinct-value sample selected by hash priority."""

    def __init__(self, capacity: int = 4096) -> None:
        if capacity <= 0:
            raise ValueError("sample capacity must be positive")
        self.capacity = capacity
        self._by_priority: dict[int, int] = {}
        self._max_heap: list[int] = []

    def add(self, value: int) -> None:
        priority = _mix64(value)
        existing = self._by_priority.get(priority)
        if existing is not None:
            return
        if len(self._by_priority) < self.capacity:
            self._by_priority[priority] = value
            heapq.heappush(self._max_heap, -priority)
            return
        largest = -self._max_heap[0]
        if priority >= largest:
            return
        heapq.heapreplace(self._max_heap, -priority)
        self._by_priority.pop(largest)
        self._by_priority[priority] = value

    def merge(self, other: "BottomKValues") -> None:
        for value in other.values():
            self.add(value)

    def values(self) -> tuple[int, ...]:
        return tuple(self._by_priority.values())

    def __len__(self) -> int:
        return len(self._by_priority)


def _counter_quantile(counts: Counter[int], quantile: float) -> int | None:
    total = sum(counts.values())
    if total == 0:
        return None
    threshold = max(1, math.ceil(total * quantile))
    seen = 0
    for value, count in sorted(counts.items()):
        seen += count
        if seen >= threshold:
            return value
    return max(counts)


def _length_summary(counts: Counter[int]) -> dict[str, int | None]:
    return {
        "count": sum(counts.values()),
        "min": min(counts) if counts else None,
        "p50": _counter_quantile(counts, 0.50),
        "p95": _counter_quantile(counts, 0.95),
        "p99": _counter_quantile(counts, 0.99),
        "max": max(counts) if counts else None,
    }


def detect_scalar_multi_conflicts(
    field_reports: Mapping[str, Mapping[str, Any]],
    *,
    bag_sources: Iterable[str] = (),
    sequence_sources: Mapping[str, Sequence[str]] | Iterable[str] = (),
    label_sources: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Flag configured scalars whose deepest list depth has length > 1.

    Bags and UPS sequence token columns are excluded. The deepest observed list
    depth is the inner value container after request/candidate axes; max > 1
    there means a scalar field is carrying a multi-valued payload.
    """

    bags = {str(source) for source in bag_sources}
    if isinstance(sequence_sources, Mapping):
        sequences = {
            str(source)
            for sources in sequence_sources.values()
            for source in sources
        }
    else:
        sequences = {str(source) for source in sequence_sources}
    labels = {str(source) for source in label_sources}
    skip = bags | sequences | labels
    conflicts: list[dict[str, Any]] = []
    for source, report in sorted(field_reports.items()):
        if source in skip:
            continue
        lengths = report.get("list_lengths_by_depth") or {}
        if not isinstance(lengths, Mapping) or not lengths:
            continue
        depth_keys = [int(depth) for depth in lengths]
        deepest = max(depth_keys)
        summary = lengths.get(str(deepest)) or lengths.get(deepest)
        if not isinstance(summary, Mapping):
            continue
        max_length = summary.get("max")
        if max_length is None or int(max_length) <= 1:
            continue
        conflicts.append(
            {
                "source": source,
                "depth": deepest,
                "max_inner_length": int(max_length),
                "p99_inner_length": summary.get("p99"),
                "observations": summary.get("count"),
            }
        )
    return conflicts


class FieldProfile:
    """Nested-list, null, sign, range, frequency, and distinct statistics."""

    def __init__(self, *, sample_size: int = 4096, hll_precision: int = 12) -> None:
        self.cells = 0
        self.nulls_by_depth: Counter[int] = Counter()
        self.empty_lists_by_depth: Counter[int] = Counter()
        self.list_lengths_by_depth: dict[int, Counter[int]] = {}
        self.rows_with_nested_null = 0
        self.rows_with_empty_list = 0
        self.leaf_count = 0
        self.invalid_leaf_count = 0
        self.negative_count = 0
        self.positive_count = 0
        self.zero_count = 0
        self.minimum: int | None = None
        self.maximum: int | None = None
        self.unsigned_minimum: int | None = None
        self.unsigned_maximum: int | None = None
        self.high16: Counter[int] = Counter()
        self.heavy_hitters: Counter[int] = Counter()
        self.distinct = HyperLogLog(hll_precision)
        self.sample = BottomKValues(sample_size)

    def observe(self, value: Any) -> None:
        self.cells += 1
        row_flags = {"nested_null": False, "empty": False}
        self._walk(value, depth=0, row_flags=row_flags)
        self.rows_with_nested_null += int(row_flags["nested_null"])
        self.rows_with_empty_list += int(row_flags["empty"])

    def _walk(self, value: Any, *, depth: int, row_flags: dict[str, bool]) -> None:
        if value is None:
            self.nulls_by_depth[depth] += 1
            if depth > 0:
                row_flags["nested_null"] = True
            return
        if isinstance(value, (list, tuple)):
            lengths = self.list_lengths_by_depth.setdefault(depth, Counter())
            lengths[len(value)] += 1
            if not value:
                self.empty_lists_by_depth[depth] += 1
                row_flags["empty"] = True
            for item in value:
                self._walk(item, depth=depth + 1, row_flags=row_flags)
            return
        if isinstance(value, bool) or not isinstance(value, int):
            self.invalid_leaf_count += 1
            return
        self.leaf_count += 1
        if value < 0:
            self.negative_count += 1
        elif value > 0:
            self.positive_count += 1
        else:
            self.zero_count += 1
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        unsigned = value & MASK64
        self.unsigned_minimum = (
            unsigned if self.unsigned_minimum is None else min(self.unsigned_minimum, unsigned)
        )
        self.unsigned_maximum = (
            unsigned if self.unsigned_maximum is None else max(self.unsigned_maximum, unsigned)
        )
        self.high16[unsigned >> 48] += 1
        self.heavy_hitters[value] += 1
        if len(self.heavy_hitters) > 256:
            self.heavy_hitters = Counter(dict(self.heavy_hitters.most_common(64)))
        self.distinct.add(value)
        self.sample.add(value)

    def merge_distinct_into(self, hll: HyperLogLog, sample: BottomKValues) -> None:
        hll.merge(self.distinct)
        sample.merge(self.sample)

    def as_dict(
        self,
        *,
        candidate_buckets: Sequence[int],
        collision_target: float,
        cardinality_headroom: float,
    ) -> dict[str, Any]:
        estimate = int(round(self.distinct.estimate())) if self.leaf_count else 0
        bucket_report, recommendation = _bucket_report(
            estimate,
            self.sample.values(),
            candidate_buckets,
            collision_target,
            cardinality_headroom,
        )
        lengths: dict[str, Any] = {}
        for depth, counts in sorted(self.list_lengths_by_depth.items()):
            lengths[str(depth)] = _length_summary(counts)
        signed_span = (
            None
            if self.minimum is None or self.maximum is None
            else self.maximum - self.minimum
        )
        return {
            "cells": self.cells,
            "leaf_count": self.leaf_count,
            "invalid_leaf_count": self.invalid_leaf_count,
            "nulls_by_depth": {str(k): v for k, v in sorted(self.nulls_by_depth.items())},
            "empty_lists_by_depth": {
                str(k): v for k, v in sorted(self.empty_lists_by_depth.items())
            },
            "rows_with_nested_null": self.rows_with_nested_null,
            "rows_with_empty_list": self.rows_with_empty_list,
            "list_lengths_by_depth": lengths,
            "negative_count": self.negative_count,
            "positive_count": self.positive_count,
            "zero_count": self.zero_count,
            "signed_min": self.minimum,
            "signed_max": self.maximum,
            "signed_span": signed_span,
            "unsigned_min": self.unsigned_minimum,
            "unsigned_max": self.unsigned_maximum,
            "distinct_estimate": estimate,
            "distinct_sample_size": len(self.sample),
            "top_values_approx": [
                {"value": value, "count": count}
                for value, count in self.heavy_hitters.most_common(16)
            ],
            "top_uint64_high16": [
                {"prefix": prefix, "count": count}
                for prefix, count in self.high16.most_common(8)
            ],
            "bucket_candidates": bucket_report,
            "recommended_bucket_size": recommendation,
            "suggested_embedding_dim": _suggest_embedding_dim(estimate),
        }


def _bucket_report(
    distinct_estimate: int,
    sample_values: Sequence[int],
    candidate_buckets: Sequence[int],
    collision_target: float,
    cardinality_headroom: float,
) -> tuple[list[dict[str, Any]], int | None]:
    sample_count = len(sample_values)
    report: list[dict[str, Any]] = []
    recommendation: int | None = None
    for bucket_size in candidate_buckets:
        occupied = len({value & (bucket_size - 1) for value in sample_values})
        collisions = sample_count - occupied
        collision_rate = collisions / sample_count if sample_count else 0.0
        if distinct_estimate <= 1:
            projected_collision_rate = 0.0
            projected_occupied = float(distinct_estimate)
        elif bucket_size == 1:
            projected_occupied = 1.0
            projected_collision_rate = 1.0 - 1.0 / distinct_estimate
        else:
            # Project the expected occupancy of the complete distinct set under
            # uniform low bits. Looking only at the bounded sample would make
            # large-cardinality fields appear artificially collision-free.
            projected_occupied = bucket_size * -math.expm1(
                distinct_estimate * math.log1p(-1.0 / bucket_size)
            )
            projected_collision_rate = max(
                0.0,
                1.0 - projected_occupied / distinct_estimate,
            )
        enough_capacity = bucket_size >= math.ceil(distinct_estimate * cardinality_headroom)
        accepted = (
            enough_capacity
            and collision_rate <= collision_target
            and projected_collision_rate <= collision_target
        )
        report.append(
            {
                "bucket_size": bucket_size,
                "sample_occupied": occupied,
                "sample_collisions": collisions,
                "sample_collision_rate": collision_rate,
                "projected_uniform_occupied": projected_occupied,
                "projected_uniform_collision_rate": projected_collision_rate,
                "estimated_cardinality_load": (
                    distinct_estimate / bucket_size if bucket_size else None
                ),
                "meets_target": accepted,
            }
        )
        if accepted and recommendation is None:
            recommendation = bucket_size
    return report, recommendation


def _suggest_embedding_dim(distinct_estimate: int) -> int:
    """Conservative starting width; the final YAML still needs a memory budget."""

    if distinct_estimate <= 16:
        return 8
    if distinct_estimate <= 256:
        return 16
    if distinct_estimate <= 4096:
        return 24
    if distinct_estimate <= 65_536:
        return 32
    if distinct_estimate <= 1_000_000:
        return 48
    if distinct_estimate <= 10_000_000:
        return 64
    if distinct_estimate <= 100_000_000:
        return 96
    return 128


@dataclass(frozen=True)
class ProfileSpec:
    all_sources: tuple[str, ...]
    categorical_sources: tuple[str, ...]
    time_sources: tuple[str, ...]
    context_sources: tuple[str, ...]
    item_sources: tuple[str, ...]
    sequence_sources: Mapping[str, tuple[str, ...]]
    sequence_time_sources: Mapping[str, str]
    label_sources: Mapping[str, str]
    shared_groups: Mapping[str, tuple[str, ...]]
    sku_fields: tuple[str, ...]
    scene_source: str
    request_time_source: str = "impr_time"
    # Physical Parquet outer axes (may differ from logical context/item).
    request_axis_sources: tuple[str, ...] = ()
    candidate_axis_sources: tuple[str, ...] = ()
    bag_sources: tuple[str, ...] = ()


def _physical_axis_sources(
    context_sources: Sequence[str],
    item_sources: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Request axis is context_features; candidate axis is item_features."""

    return tuple(context_sources), tuple(item_sources)


def profile_spec_from_mapping(
    payload: Mapping[str, Any],
    *,
    context_feature_count: int = 51,
    source_name: str = "sample config",
) -> ProfileSpec:
    raw_features = payload.get("features", [])
    if not isinstance(raw_features, list) or len(raw_features) < context_feature_count:
        raise ValueError(
            f"{source_name} must contain at least {context_feature_count} ordered features"
        )
    feature_logical_sources: dict[str, str] = {}
    for item in raw_features:
        if not isinstance(item, dict) or not item.get("name") or not item.get("source"):
            raise ValueError(f"{source_name} contains an invalid feature declaration")
        feature_logical_sources[str(item["name"])] = str(item["source"])

    sequence_sources: dict[str, tuple[str, ...]] = {}
    sequence_time_sources: dict[str, str] = {}
    logical_sources = dict(feature_logical_sources)
    time_sources: list[str] = []
    for sequence in payload.get("sequences", []):
        name = str(sequence["name"])
        sources: list[str] = []
        for field in sequence.get("fields", []):
            source = str(field["source"])
            logical = f"{name}.{field['name']}"
            logical_sources[logical] = source
            sources.append(source)
            if field.get("name") == "time" or source.endswith("_x_time"):
                sequence_time_sources[name] = source
                time_sources.append(source)
        sequence_sources[name] = tuple(sources)

    strategies = payload.get("vocab_strategy", {}).get("features", {})
    if not isinstance(strategies, dict):
        raise ValueError(f"{source_name} vocab_strategy.features must be an object")

    def shared_root(name: str) -> str:
        seen: set[str] = set()
        current = name
        while current in strategies:
            if current in seen:
                raise ValueError(f"shared embedding cycle at {current!r}")
            seen.add(current)
            strategy = strategies[current]
            if not isinstance(strategy, dict):
                break
            share_with = strategy.get("share_with")
            shares = bool(strategy.get("share_embedding")) or strategy.get("encoding") == "shared_vocab"
            if not share_with or not shares:
                break
            current = str(share_with)
        return current

    grouped_sources: dict[str, set[str]] = {}
    for logical, source in logical_sources.items():
        if source in time_sources:
            continue
        local_name = logical.split(".", 1)[1] if "." in logical else logical
        # Production contract: an exact same-name semantic field shares one
        # namespace/table across scalar Candidate/Context and every UPS use.
        root = (
            local_name
            if "." in logical and local_name in feature_logical_sources
            else shared_root(logical)
        )
        grouped_sources.setdefault(root, set()).add(source)
    shared_groups = {
        root: tuple(sorted(sources))
        for root, sources in grouped_sources.items()
        if len(sources) > 1
    }

    train = payload.get("data", {}).get("train", {})
    agg_layout = train.get("agg_layout", {}) if isinstance(train, dict) else {}
    label_sources = agg_layout.get("labels", {}) if isinstance(agg_layout, dict) else {}
    if not isinstance(label_sources, dict):
        label_sources = {}
    feature_sources = tuple(feature_logical_sources.values())
    categorical_sources = tuple(
        dict.fromkeys(
            source for source in logical_sources.values() if source not in set(time_sources)
        )
    )
    all_sources = tuple(dict.fromkeys([*categorical_sources, *time_sources]))
    adapter = train.get("adapter", {}) if isinstance(train, dict) else {}
    adapter_options = (
        adapter.get("options", {}) if isinstance(adapter, Mapping) else {}
    )
    if not isinstance(adapter_options, Mapping):
        adapter_options = {}
    adapter_context = adapter_options.get("context_features")
    adapter_items = adapter_options.get("item_features")
    if isinstance(adapter_context, list) and isinstance(adapter_items, list):
        context_sources = tuple(str(item) for item in adapter_context)
        item_sources = tuple(str(item) for item in adapter_items)
    else:
        context_sources = feature_sources[:context_feature_count]
        item_sources = feature_sources[context_feature_count:]
    request_axis_sources, candidate_axis_sources = _physical_axis_sources(
        context_sources,
        item_sources,
    )
    bag_sources = tuple(
        dict.fromkeys(
            str(item["source"])
            for item in raw_features
            if isinstance(item, dict)
            and item.get("pooling") == "mean"
            and item.get("source")
        )
    )
    return ProfileSpec(
        all_sources=all_sources,
        categorical_sources=categorical_sources,
        time_sources=tuple(dict.fromkeys(time_sources)),
        context_sources=context_sources,
        item_sources=item_sources,
        sequence_sources=sequence_sources,
        sequence_time_sources=sequence_time_sources,
        label_sources={str(k): str(v) for k, v in label_sources.items()},
        shared_groups=shared_groups,
        sku_fields=tuple(
            source for source in DEFAULT_SKU_FIELDS if source in feature_sources
        ),
        scene_source="scene_id",
        request_axis_sources=request_axis_sources,
        candidate_axis_sources=candidate_axis_sources,
        bag_sources=bag_sources,
    )


def load_profile_spec(
    config_path: str | Path,
    *,
    context_feature_count: int = 51,
) -> ProfileSpec:
    path = Path(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return profile_spec_from_mapping(
        payload,
        context_feature_count=context_feature_count,
        source_name=str(path),
    )


class ContractProfile:
    """Cross-column agg/req, alignment, label, and sequence-order checks."""

    def __init__(self, spec: ProfileSpec) -> None:
        self.spec = spec
        self.rows = 0
        self.agg_rows = 0
        self.req_rows = 0
        self.partial_indices_rows = 0
        self.invalid_context_indices = 0
        self.invalid_target_indices = 0
        self.duplicate_context_indices = 0
        self.target_without_context = 0
        self.context_outer_mismatches: Counter[str] = Counter()
        self.item_outer_mismatches: Counter[str] = Counter()
        self.request_outer_mismatches: Counter[str] = Counter()
        self.candidate_outer_mismatches: Counter[str] = Counter()
        self.label_length_mismatches: Counter[str] = Counter()
        self.invalid_labels: Counter[str] = Counter()
        self.sequence_length_mismatches: Counter[str] = Counter()
        self.invalid_sequence_membership: Counter[str] = Counter()
        self.missing_sequence_membership: Counter[str] = Counter()
        self.empty_sequence_membership: Counter[str] = Counter()
        self.time_order_violations: Counter[str] = Counter()
        self.event_after_request_time: Counter[str] = Counter()
        self.invalid_request_time = 0
        self.invalid_request_time_layout = 0
        self.sequence_request_lengths: dict[str, Counter[int]] = {
            sequence: Counter() for sequence in spec.sequence_sources
        }
        self.label_counts: dict[str, Counter[str]] = {
            task: Counter() for task in spec.label_sources
        }
        self.scene_label_counts: dict[str, dict[int, Counter[str]]] = {
            task: {} for task in spec.label_sources
        }
        self.sku_alignment_mismatches = 0
        self.scene_values: Counter[int] = Counter()
        self.candidate_scene_values: Counter[int] = Counter()

    @staticmethod
    def _list_length(value: Any) -> int | None:
        if value is None:
            return 0
        if isinstance(value, (list, tuple)):
            return len(value)
        return None

    @staticmethod
    def _leaves(value: Any) -> Iterable[Any]:
        if isinstance(value, (list, tuple)):
            for item in value:
                yield from ContractProfile._leaves(item)
        elif value is not None:
            yield value

    @staticmethod
    def _label_leaves(value: Any) -> Iterable[Any]:
        """Yield label leaves without hiding true nulls."""

        if isinstance(value, (list, tuple)):
            for item in value:
                yield from ContractProfile._label_leaves(item)
        else:
            yield value

    @staticmethod
    def _label_category(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool) or not isinstance(value, Real):
            return "other"
        numeric = float(value)
        if not math.isfinite(numeric):
            return "other"
        if numeric == -1.0:
            return "minus_one"
        if numeric == 0.0:
            return "zero"
        if numeric == 1.0:
            return "one"
        return "other"

    @staticmethod
    def _sequence_scalar(value: Any) -> tuple[int | None, bool]:
        """Normalize scalar or singleton-list S-token storage."""

        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                return None, False
            value = value[0]
        if isinstance(value, bool) or not isinstance(value, int):
            return None, False
        return value, True

    def observe(self, row: Mapping[str, Any]) -> None:
        self.rows += 1
        has_context = "context_indices" in row
        has_target = "target_indices" in row
        if has_context != has_target:
            self.partial_indices_rows += 1
            return
        is_agg = has_context and has_target
        if is_agg:
            self.agg_rows += 1
            context_count = self._list_length(row.get("context_indices"))
            target_count = self._list_length(row.get("target_indices"))
            request_axis_sources = (
                self.spec.request_axis_sources or self.spec.context_sources
            )
            candidate_axis_sources = (
                self.spec.candidate_axis_sources or self.spec.item_sources
            )
            if context_count is not None:
                for source in request_axis_sources:
                    if source not in row:
                        continue
                    if self._list_length(row[source]) != context_count:
                        self.request_outer_mismatches[source] += 1
                        # Legacy alias kept for older report consumers.
                        self.context_outer_mismatches[source] += 1
            if target_count is not None:
                for source in candidate_axis_sources:
                    if source not in row:
                        continue
                    if self._list_length(row[source]) != target_count:
                        self.candidate_outer_mismatches[source] += 1
                        self.item_outer_mismatches[source] += 1
                for task, source in self.spec.label_sources.items():
                    if source in row and self._list_length(row[source]) != target_count:
                        self.label_length_mismatches[task] += 1
        else:
            self.req_rows += 1

        for task, source in self.spec.label_sources.items():
            if source not in row:
                continue
            for value in self._label_leaves(row[source]):
                category = self._label_category(value)
                self.label_counts[task]["total"] += 1
                self.label_counts[task][category] += 1
                if category not in {"zero", "one"}:
                    self.invalid_labels[task] += 1
                    self.label_counts[task]["invalid"] += 1
                else:
                    self.label_counts[task]["examples"] += 1
                    self.label_counts[task][
                        "positives" if category == "one" else "negatives"
                    ] += 1

        known_requests: set[int] = set()
        if is_agg:
            raw_context = row.get("context_indices")
            raw_target = row.get("target_indices")
            context_requests: list[int] = []
            if isinstance(raw_context, (list, tuple)):
                for value in raw_context:
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        self.invalid_context_indices += 1
                    else:
                        context_requests.append(value)
            if len(context_requests) != len(set(context_requests)):
                self.duplicate_context_indices += 1
            known_requests = set(context_requests)
            if isinstance(raw_target, (list, tuple)):
                for value in raw_target:
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        self.invalid_target_indices += 1
                    elif value not in known_requests:
                        self.target_without_context += 1
        else:
            known_requests.add(0)

        request_times: dict[int, int] = {}
        raw_request_time = row.get(self.spec.request_time_source)
        assignments: list[tuple[int, Any]] = []
        if isinstance(raw_request_time, (list, tuple)):
            if len(raw_request_time) == 1:
                assignments = [
                    (request, raw_request_time[0]) for request in known_requests
                ]
            elif is_agg and isinstance(row.get("context_indices"), (list, tuple)) and len(
                raw_request_time
            ) == len(row["context_indices"]):
                assignments = list(zip(row["context_indices"], raw_request_time))
            else:
                self.invalid_request_time_layout += 1
        elif raw_request_time is not None:
            assignments = [(request, raw_request_time) for request in known_requests]
        else:
            self.invalid_request_time_layout += 1
        for request, request_time in assignments:
            if (
                isinstance(request, bool)
                or not isinstance(request, int)
                or request not in known_requests
                or isinstance(request_time, bool)
                or not isinstance(request_time, int)
            ):
                self.invalid_request_time += 1
            else:
                request_times[request] = request_time

        for sequence, sources in self.spec.sequence_sources.items():
            index_source = f"{sequence}_x_indices"
            if is_agg and index_source not in row:
                self.missing_sequence_membership[sequence] += 1
            expected = self._list_length(row.get(index_source)) if is_agg else None
            lengths: set[int] = set()
            for source in sources:
                if source not in row:
                    continue
                length = self._list_length(row[source])
                if length is not None:
                    lengths.add(length)
                if expected is not None and length != expected:
                    self.sequence_length_mismatches[sequence] += 1
            if len(lengths) > 1:
                self.sequence_length_mismatches[sequence] += 1
            if is_agg and index_source in row:
                if isinstance(row[index_source], (list, tuple)) and not row[index_source]:
                    self.empty_sequence_membership[sequence] += 1
                request_counts: Counter[int] = Counter()
                for membership in row[index_source] or []:
                    values = membership if isinstance(membership, (list, tuple)) else [membership]
                    if not values or any(
                        isinstance(value, bool) or not isinstance(value, int) or value < 0
                        for value in values
                    ):
                        self.invalid_sequence_membership[sequence] += 1
                        continue
                    if any(value not in known_requests for value in values):
                        self.invalid_sequence_membership[sequence] += 1
                        continue
                    for request in set(values):
                        request_counts[request] += 1
                for request in known_requests:
                    self.sequence_request_lengths[sequence][request_counts[request]] += 1
            elif not is_agg and lengths:
                # All parallel fields should agree. A mismatch is already
                # recorded above; using max keeps the diagnostic conservative.
                self.sequence_request_lengths[sequence][max(lengths)] += 1
            time_source = self.spec.sequence_time_sources.get(sequence)
            times = row.get(time_source) if time_source else None
            if isinstance(times, (list, tuple)):
                ordered_views: list[tuple[int | None, list[Any]]] = []
                memberships = row.get(index_source) if is_agg else None
                if isinstance(memberships, (list, tuple)) and len(memberships) == len(times):
                    for request in known_requests:
                        filtered: list[Any] = []
                        for value, membership in zip(times, memberships):
                            members = (
                                membership
                                if isinstance(membership, (list, tuple))
                                else [membership]
                            )
                            if request in members:
                                filtered.append(value)
                        ordered_views.append((request, filtered))
                else:
                    request = 0 if not is_agg else None
                    ordered_views.append((request, list(times)))
                for request, ordered in ordered_views:
                    normalized = [self._sequence_scalar(value) for value in ordered]
                    if any(not valid for _value, valid in normalized):
                        self.time_order_violations[sequence] += 1
                        break
                    present = [int(value) for value, _valid in normalized if value is not None]
                    if any(left < right for left, right in zip(present, present[1:])):
                        self.time_order_violations[sequence] += 1
                        break
                    request_time = request_times.get(request) if request is not None else None
                    if request_time is not None:
                        violations = sum(
                            int(value > request_time) for value in present
                        )
                        if violations:
                            self.event_after_request_time[sequence] += violations

        if all(source in row for source in self.spec.sku_fields):
            outer_values = [row[source] for source in self.spec.sku_fields]
            candidate_count = max(
                (len(value) for value in outer_values if isinstance(value, (list, tuple))),
                default=0,
            )
            for index in range(candidate_count):
                lengths: set[int] = set()
                for values in outer_values:
                    if not isinstance(values, (list, tuple)) or index >= len(values):
                        continue
                    candidate_value = values[index]
                    if isinstance(candidate_value, (list, tuple)):
                        lengths.add(len(candidate_value))
                if len(lengths) > 1:
                    self.sku_alignment_mismatches += 1

        if self.spec.scene_source in row:
            for value in self._leaves(row[self.spec.scene_source]):
                if isinstance(value, int) and not isinstance(value, bool):
                    self.scene_values[value] += 1

        self._observe_candidate_scene_labels(row, is_agg=is_agg)

    def _observe_candidate_scene_labels(
        self,
        row: Mapping[str, Any],
        *,
        is_agg: bool,
    ) -> None:
        raw_scene = row.get(self.spec.scene_source)
        if raw_scene is None:
            return
        if is_agg:
            context_indices = row.get("context_indices")
            target_indices = row.get("target_indices")
            if not isinstance(context_indices, (list, tuple)) or not isinstance(
                target_indices, (list, tuple)
            ):
                return
            scenes = list(raw_scene) if isinstance(raw_scene, (list, tuple)) else [raw_scene]
            if len(scenes) == len(context_indices):
                scene_by_request = dict(zip(context_indices, scenes))
            elif len(scenes) == 1 and len(context_indices) == 1:
                scene_by_request = {context_indices[0]: scenes[0]}
            else:
                # Do not silently broadcast length-1 scenes across multi-request rows.
                return
            candidate_scenes = [scene_by_request.get(request) for request in target_indices]
        else:
            scenes = list(raw_scene) if isinstance(raw_scene, (list, tuple)) else [raw_scene]
            if len(scenes) != 1:
                return
            candidate_count = max(
                (
                    len(row[source])
                    for source in self.spec.label_sources.values()
                    if isinstance(row.get(source), (list, tuple))
                ),
                default=0,
            )
            candidate_scenes = [scenes[0]] * candidate_count

        for scene in candidate_scenes:
            if isinstance(scene, int) and not isinstance(scene, bool):
                self.candidate_scene_values[scene] += 1
        for task, source in self.spec.label_sources.items():
            labels = row.get(source)
            if not isinstance(labels, (list, tuple)) or len(labels) != len(candidate_scenes):
                continue
            per_scene = self.scene_label_counts[task]
            for scene, label in zip(candidate_scenes, labels):
                if not isinstance(scene, int) or isinstance(scene, bool):
                    continue
                counts = per_scene.setdefault(scene, Counter())
                category = self._label_category(label)
                counts["total"] += 1
                counts[category] += 1
                if category not in {"zero", "one"}:
                    counts["invalid"] += 1
                    continue
                counts["examples"] += 1
                counts["positives" if category == "one" else "negatives"] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "agg_rows": self.agg_rows,
            "req_rows": self.req_rows,
            "partial_indices_rows": self.partial_indices_rows,
            "invalid_context_indices": self.invalid_context_indices,
            "invalid_target_indices": self.invalid_target_indices,
            "duplicate_context_indices": self.duplicate_context_indices,
            "target_without_context": self.target_without_context,
            "context_outer_mismatches": dict(self.context_outer_mismatches),
            "item_outer_mismatches": dict(self.item_outer_mismatches),
            "request_outer_mismatches": dict(self.request_outer_mismatches),
            "candidate_outer_mismatches": dict(self.candidate_outer_mismatches),
            "label_length_mismatches": dict(self.label_length_mismatches),
            "invalid_labels": dict(self.invalid_labels),
            "sequence_length_mismatches": dict(self.sequence_length_mismatches),
            "invalid_sequence_membership": dict(self.invalid_sequence_membership),
            "missing_sequence_membership": dict(self.missing_sequence_membership),
            "empty_sequence_membership": dict(self.empty_sequence_membership),
            "time_order_violations": dict(self.time_order_violations),
            "event_after_request_time": dict(self.event_after_request_time),
            "invalid_request_time": self.invalid_request_time,
            "invalid_request_time_layout": self.invalid_request_time_layout,
            "sequence_lengths_after_request_filter": {
                sequence: _length_summary(counts)
                for sequence, counts in self.sequence_request_lengths.items()
            },
            "sku_alignment_mismatches": self.sku_alignment_mismatches,
            "label_distribution": {
                task: {
                    "examples": counts["examples"],
                    "positives": counts["positives"],
                    "negatives": counts["negatives"],
                    "invalid": counts["invalid"],
                    "total": counts["total"],
                    "null": counts["null"],
                    "minus_one": counts["minus_one"],
                    "zero": counts["zero"],
                    "one": counts["one"],
                    "other": counts["other"],
                }
                for task, counts in self.label_counts.items()
            },
            "scene_values": [
                {"scene_id": value, "count": count}
                for value, count in sorted(self.scene_values.items())
            ],
            "candidate_scene_values": [
                {"scene_id": value, "count": count}
                for value, count in sorted(self.candidate_scene_values.items())
            ],
            "scene_label_distribution": {
                task: [
                    {
                        "scene_id": scene,
                        "examples": counts["examples"],
                        "positives": counts["positives"],
                        "negatives": counts["negatives"],
                        "invalid": counts["invalid"],
                        "total": counts["total"],
                        "null": counts["null"],
                        "minus_one": counts["minus_one"],
                        "zero": counts["zero"],
                        "one": counts["one"],
                        "other": counts["other"],
                    }
                    for scene, counts in sorted(per_scene.items())
                ]
                for task, per_scene in self.scene_label_counts.items()
            },
        }


def _dataset_from_uri(uri: str) -> Any:
    ds, fs = _require_pyarrow()
    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme != "file":
        filesystem, path = fs.FileSystem.from_uri(uri)
        return ds.dataset(path, filesystem=filesystem, format="parquet", partitioning="hive")
    local = parsed.path if parsed.scheme == "file" else uri
    return ds.dataset(local, format="parquet", partitioning="hive")


def _fragment_path(fragment: Any) -> str:
    return str(getattr(fragment, "path", fragment))


def profile_paths(
    inputs: Sequence[str],
    spec: ProfileSpec,
    *,
    candidate_buckets: Sequence[int] = DEFAULT_BUCKETS,
    collision_target: float = 0.01,
    cardinality_headroom: float = 1.5,
    sample_size: int = 4096,
    hll_precision: int = 12,
    batch_size: int = 2048,
    max_files: int | None = None,
    max_rows: int | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    if not inputs:
        raise ValueError("at least one explicit input is required")
    if not 0.0 <= collision_target <= 1.0:
        raise ValueError("collision_target must be inside [0, 1]")
    if cardinality_headroom < 1.0:
        raise ValueError("cardinality_headroom must be at least 1")
    if any(bucket <= 0 or bucket & (bucket - 1) for bucket in candidate_buckets):
        raise ValueError("all candidate buckets must be positive powers of two")

    profiles = {
        source: FieldProfile(sample_size=sample_size, hll_precision=hll_precision)
        for source in spec.all_sources
    }
    contract = ContractProfile(spec)
    observed_schema: dict[str, str] = {}
    missing_by_input: dict[str, list[str]] = {}
    resolved_aliases_by_input: dict[str, dict[str, str]] = {}
    files_scanned: list[str] = []
    rows_scanned = 0

    required_contract_columns = {
        "context_indices",
        "target_indices",
        spec.scene_source,
        spec.request_time_source,
        *spec.label_sources.values(),
        *(f"{name}_x_indices" for name in spec.sequence_sources),
    }
    required_in_both_layouts = {
        *spec.all_sources,
        spec.scene_source,
        spec.request_time_source,
    }
    for input_uri in inputs:
        dataset = _dataset_from_uri(input_uri)
        schema_names = set(dataset.schema.names)
        canonical_to_physical: dict[str, str] = {}
        for source in set(spec.all_sources) | required_contract_columns:
            if source in schema_names:
                canonical_to_physical[source] = source
        is_agg_schema = {
            "context_indices",
            "target_indices",
        } <= schema_names
        required_for_input = set(required_in_both_layouts)
        if is_agg_schema:
            required_for_input.update(spec.label_sources.values())
        missing_by_input[input_uri] = sorted(
            required_for_input - set(canonical_to_physical)
        )
        resolved_aliases_by_input[input_uri] = {
            canonical: physical
            for canonical, physical in canonical_to_physical.items()
            if canonical != physical
        }
        physical_to_canonical = {
            physical: canonical
            for canonical, physical in canonical_to_physical.items()
        }
        available = sorted(set(canonical_to_physical.values()))
        for field in dataset.schema:
            if field.name in available:
                observed_schema.setdefault(
                    physical_to_canonical[field.name], str(field.type)
                )
        fragments = sorted(dataset.get_fragments(), key=_fragment_path)
        for fragment in fragments:
            if max_files is not None and len(files_scanned) >= max_files:
                break
            if max_rows is not None and rows_scanned >= max_rows:
                break
            files_scanned.append(_fragment_path(fragment))
            fragment_rows = 0
            for batch in fragment.to_batches(columns=available, batch_size=batch_size):
                if max_rows is not None:
                    remaining = max_rows - rows_scanned
                    if remaining <= 0:
                        break
                    if batch.num_rows > remaining:
                        batch = batch.slice(0, remaining)
                physical_data = batch.to_pydict()
                data = {
                    physical_to_canonical.get(name, name): values
                    for name, values in physical_data.items()
                }
                for source, profile in profiles.items():
                    if source in data:
                        for value in data[source]:
                            profile.observe(value)
                for row_index in range(batch.num_rows):
                    contract.observe({name: values[row_index] for name, values in data.items()})
                rows_scanned += batch.num_rows
                fragment_rows += batch.num_rows
            if progress:
                print(
                    f"profiled {files_scanned[-1]}: {fragment_rows} rows "
                    f"({rows_scanned} total)",
                    file=sys.stderr,
                    flush=True,
                )
        if max_files is not None and len(files_scanned) >= max_files:
            break
        if max_rows is not None and rows_scanned >= max_rows:
            break

    field_reports = {
        source: profile.as_dict(
            candidate_buckets=candidate_buckets,
            collision_target=collision_target,
            cardinality_headroom=cardinality_headroom,
        )
        for source, profile in profiles.items()
    }
    shared_reports: dict[str, Any] = {}
    for root, sources in spec.shared_groups.items():
        hll = HyperLogLog(hll_precision)
        sample = BottomKValues(sample_size)
        total_occurrences = 0
        for source in sources:
            profile = profiles[source]
            profile.merge_distinct_into(hll, sample)
            total_occurrences += profile.leaf_count
        estimate = int(round(hll.estimate())) if total_occurrences else 0
        buckets, recommendation = _bucket_report(
            estimate,
            sample.values(),
            candidate_buckets,
            collision_target,
            cardinality_headroom,
        )
        dimension = _suggest_embedding_dim(estimate)
        pairwise_overlap: list[dict[str, Any]] = []
        for left, right in combinations(sources, 2):
            left_values = set(profiles[left].sample.values())
            right_values = set(profiles[right].sample.values())
            intersection = len(left_values & right_values)
            union = len(left_values | right_values)
            pairwise_overlap.append(
                {
                    "left": left,
                    "right": right,
                    "left_sample_size": len(left_values),
                    "right_sample_size": len(right_values),
                    "sample_intersection": intersection,
                    "sample_jaccard": intersection / union if union else None,
                }
            )
        shared_reports[root] = {
            "sources": list(sources),
            "source_distinct_estimates": {
                source: int(round(profiles[source].distinct.estimate()))
                if profiles[source].leaf_count
                else 0
                for source in sources
            },
            "leaf_occurrences": total_occurrences,
            "distinct_estimate": estimate,
            "distinct_sample_size": len(sample),
            "bottom_k_pairwise_overlap": pairwise_overlap,
            "bucket_candidates": buckets,
            "recommended_bucket_size": recommendation,
            "suggested_embedding_dim": dimension,
            "recommended_weight_bytes_fp32": (
                None
                if recommendation is None
                else (recommendation + 1) * dimension * 4
            ),
            "recommended_weight_bytes_fp32_per_8way_shard": (
                None
                if recommendation is None
                else math.ceil((recommendation + 1) / 8) * dimension * 4
            ),
        }

    return {
        "format_version": 4,
        "inputs": list(inputs),
        "files_scanned": files_scanned,
        "rows_scanned": rows_scanned,
        "settings": {
            "candidate_buckets": list(candidate_buckets),
            "collision_target": collision_target,
            "cardinality_headroom": cardinality_headroom,
            "sample_size": sample_size,
            "hll_precision": hll_precision,
            "max_files": max_files,
            "max_rows": max_rows,
        },
        "missing_configured_columns_by_input": missing_by_input,
        "resolved_column_aliases_by_input": resolved_aliases_by_input,
        "schema": observed_schema,
        "contract": contract.as_dict(),
        "fields": field_reports,
        "shared_embedding_groups": shared_reports,
        "scalar_multi_conflicts": detect_scalar_multi_conflicts(
            field_reports,
            bag_sources=spec.bag_sources,
            sequence_sources=spec.sequence_sources,
            label_sources=spec.label_sources.values(),
        ),
        "notes": [
            "distinct_estimate uses HyperLogLog and is approximate",
            "bucket collision rates use a deterministic bounded sample of distinct raw values",
            "bucket acceptance also projects uniform collisions at the full HLL cardinality",
            "shared-field overlap is diagnostic overlap among deterministic bottom-k samples",
            "signed min/max diagnose upstream encoding only and are not embedding indices",
            "embedding dimensions are starting points; finalize them with the 8-GPU memory budget",
            "scalar_multi_conflicts lists non-bag fields whose deepest list depth has max length > 1",
        ],
    }


def _parse_buckets(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(item.strip()) for item in raw.split(",") if item.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("candidate buckets must be comma-separated integers") from error
    if not values or any(value <= 0 or value & (value - 1) for value in values):
        raise argparse.ArgumentTypeError("candidate buckets must be positive powers of two")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mdl_rankmixer.yaml"))
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Local path or HDFS URI. Repeat to scan multiple partitions.",
    )
    parser.add_argument("--output", type=Path, help="JSON report path; stdout when omitted")
    parser.add_argument("--context-feature-count", type=int, default=51)
    parser.add_argument("--candidate-buckets", type=_parse_buckets, default=DEFAULT_BUCKETS)
    parser.add_argument("--collision-target", type=float, default=0.01)
    parser.add_argument("--cardinality-headroom", type=float, default=1.5)
    parser.add_argument("--sample-size", type=int, default=4096)
    parser.add_argument("--hll-precision", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        spec = load_profile_spec(
            args.config,
            context_feature_count=args.context_feature_count,
        )
        report = profile_paths(
            args.input,
            spec,
            candidate_buckets=args.candidate_buckets,
            collision_target=args.collision_target,
            cardinality_headroom=args.cardinality_headroom,
            sample_size=args.sample_size,
            hll_precision=args.hll_precision,
            batch_size=args.batch_size,
            max_files=args.max_files,
            max_rows=args.max_rows,
            progress=not args.quiet,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
