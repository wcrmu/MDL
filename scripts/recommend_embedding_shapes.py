#!/usr/bin/env python3
"""Recommend growth-aware embedding shapes from two nested Parquet profiles.

The profiler reports the cardinality seen in a bounded set of Parquet files.
This command compares a small prefix with a larger prefix from the same
partition, fits a per-source Heaps-law growth exponent, and projects the
cardinality to a production time horizon.

Bucket count and embedding width answer different questions:

* ``num_buckets`` reports a collision-safer ideal at load 0.5, then selects a
  deployable shape for the two-H100 budget.  The deployable shape retains the
  one-hour load-0.5 table as a floor and limits the 24-hour projection to load
  1.75.

  Load is a poor proxy for retained signal: a table at load 0.5 still loses
  about 20% of its distinct values to hash collisions.  That is ruinous for the
  medium-cardinality conversion features (scene-crossed CVR, price, cart and
  order counts) that carry most of the payment signal, while the low-cardinality
  relevance anchors sit near load 0.03 and are unaffected.  Tables projected
  below ``SMALL_TABLE_DISTINCT_CEILING`` therefore take an additional floor that
  bounds collision directly at ``SMALL_TABLE_MAX_COLLISION``.  These tables are
  the cheapest in the pack, so the floor costs a few MiB against a 68 GiB
  budget.
* ``embedding_dim`` reports the usual bounded cardinality tier as the ideal,
  but caps production widths at 32 once projected cardinality exceeds one
  million.  This trades width for hash capacity under the fixed GPU budget.

Shared embedding roots are projected from ``shared_embedding_groups`` (the
union of every source sharing the table), never from the root field alone.

``load_json_report`` deliberately accepts platform stdout with a final JSON
line.  This keeps ``docs/new_profile.json`` usable even though the captured
file contains launcher/progress logs before the profiler payload.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.profile_prehashed_parquet import load_profile_spec  # noqa: E402


FORMAT_VERSION = 3
MIN_BUCKETS = 1 << 8
MAX_PRODUCTION_BUCKETS = 1 << 30
MAX_GROWTH_ALPHA = 0.95
IDEAL_MAX_LOAD = 0.5
PRODUCTION_MAX_LOAD = 1.75
SMALL_TABLE_DISTINCT_CEILING = 1_000
SMALL_TABLE_MAX_COLLISION = 0.02
RETENTION_HOURS = 1.0
HIGH_CARDINALITY_THRESHOLD = 1_000_000
HIGH_CARDINALITY_PRODUCTION_DIM = 32
PLANNED_EMBEDDING_MEMORY_GIB_PER_GPU = 68.0

NON_EMBEDDING_SOURCES = frozenset({"coarse_scene_prior_id"})
NON_EMBEDDING_SUFFIXES = ("time_delta_log1p_seconds",)


@dataclass(frozen=True)
class GrowthEstimate:
    tier: str
    raw_alpha: float
    alpha: float
    projected_distinct: int
    stress_distinct: int


def load_json_report(path: str | Path) -> dict[str, Any]:
    """Load a JSON report, tolerating launcher logs before a final JSON line."""

    report_path = Path(path)
    text = report_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as whole_error:
        payload = None
        for line in reversed(text.splitlines()):
            candidate = line.strip()
            if not candidate.startswith("{"):
                continue
            try:
                payload = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        if payload is None:
            raise ValueError(
                f"{report_path} is neither JSON nor log-prefixed JSON"
            ) from whole_error
    if not isinstance(payload, dict):
        raise ValueError(f"{report_path} must contain a JSON object")
    return payload


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _file_count(report: Mapping[str, Any], label: str) -> int:
    files = report.get("files_scanned")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{label}.files_scanned must be a non-empty list")
    return len(files)


def _validate_nested_profiles(
    small: Mapping[str, Any],
    large: Mapping[str, Any],
) -> tuple[int, int]:
    small_files = small.get("files_scanned")
    large_files = large.get("files_scanned")
    if not isinstance(small_files, list) or not isinstance(large_files, list):
        raise ValueError("both profiles must contain files_scanned lists")
    if not small_files or len(large_files) <= len(small_files):
        raise ValueError("the large profile must scan more files than the small profile")
    if large_files[: len(small_files)] != small_files:
        raise ValueError(
            "profiles must be nested prefixes from the same partition; "
            "the small files are not the prefix of the large profile"
        )

    small_fields = set(_require_mapping(small.get("fields"), "small.fields"))
    large_fields = set(_require_mapping(large.get("fields"), "large.fields"))
    if small_fields != large_fields:
        raise ValueError(
            "profile field sets differ: "
            f"small_only={sorted(small_fields - large_fields)}, "
            f"large_only={sorted(large_fields - small_fields)}"
        )
    small_shared = set(
        _require_mapping(
            small.get("shared_embedding_groups"),
            "small.shared_embedding_groups",
        )
    )
    large_shared = set(
        _require_mapping(
            large.get("shared_embedding_groups"),
            "large.shared_embedding_groups",
        )
    )
    if small_shared != large_shared:
        raise ValueError(
            "profile shared embedding groups differ: "
            f"small_only={sorted(small_shared - large_shared)}, "
            f"large_only={sorted(large_shared - small_shared)}"
        )
    return len(small_files), len(large_files)


def _is_non_embedding_source(source: str) -> bool:
    return source in NON_EMBEDDING_SOURCES or source.endswith(
        NON_EMBEDDING_SUFFIXES
    )


def _next_power_of_two(value: int, *, minimum: int = MIN_BUCKETS) -> int:
    if value <= 1:
        return minimum
    return max(minimum, 1 << (value - 1).bit_length())


def _suggest_embedding_dim(distinct_estimate: int) -> int:
    """Return the unconstrained representation-width starting tier."""

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
    return 64


def _suggest_production_embedding_dim(distinct_estimate: int) -> int:
    """Trade high-cardinality width for bucket capacity on two 80-GiB GPUs."""

    ideal = _suggest_embedding_dim(distinct_estimate)
    if distinct_estimate > HIGH_CARDINALITY_THRESHOLD:
        return min(ideal, HIGH_CARDINALITY_PRODUCTION_DIM)
    return ideal


def _top1_share(stats: Mapping[str, Any]) -> float:
    leaf_count = int(stats.get("leaf_count") or 0)
    values = stats.get("top_values_approx")
    if leaf_count <= 0 or not isinstance(values, list) or not values:
        return 0.0
    first = values[0]
    if not isinstance(first, Mapping):
        return 0.0
    return min(1.0, max(0.0, int(first.get("count") or 0) / leaf_count))


def _power_project(
    distinct: int,
    *,
    from_files: int,
    to_files: int,
    alpha: float,
) -> int:
    if distinct <= 0:
        return 0
    scale = max(1.0, to_files / from_files)
    return int(math.ceil(distinct * scale**alpha))


def estimate_growth(
    *,
    small_distinct: int,
    large_distinct: int,
    small_files: int,
    large_files: int,
    target_files: int,
    stress_files: int,
    top1_share: float = 0.0,
) -> GrowthEstimate:
    """Fit a bounded Heaps exponent and project one source."""

    if small_distinct < 0 or large_distinct < 0:
        raise ValueError("distinct estimates must be non-negative")
    if small_files <= 0 or large_files <= small_files:
        raise ValueError("profile file counts must be positive and increasing")
    if target_files < large_files:
        raise ValueError("target_files must be at least the large profile size")
    if stress_files < target_files:
        raise ValueError("stress_files must be at least target_files")

    if small_distinct <= 0 or large_distinct <= small_distinct:
        raw_alpha = 0.0
    else:
        raw_alpha = math.log(large_distinct / small_distinct) / math.log(
            large_files / small_files
        )
    alpha = min(MAX_GROWTH_ALPHA, max(0.0, raw_alpha))

    if large_distinct <= 1:
        tier = "dead"
        primary = large_distinct
        stress = large_distinct
    elif large_distinct <= 4 and top1_share >= 0.5:
        tier = "near_const"
        primary = large_distinct
        stress = large_distinct
    elif large_distinct <= 512 and alpha < 0.35:
        tier = "saturated"
        primary = max(
            int(math.ceil(1.25 * large_distinct)),
            _power_project(
                large_distinct,
                from_files=large_files,
                to_files=target_files,
                alpha=alpha,
            ),
        )
        stress = max(
            int(math.ceil(1.25 * large_distinct)),
            _power_project(
                large_distinct,
                from_files=large_files,
                to_files=stress_files,
                alpha=alpha,
            ),
        )
    elif alpha < 0.15:
        tier = "slow"
        primary = max(
            int(math.ceil(1.5 * large_distinct)),
            _power_project(
                large_distinct,
                from_files=large_files,
                to_files=target_files,
                alpha=alpha,
            ),
        )
        stress = max(
            int(math.ceil(1.5 * large_distinct)),
            _power_project(
                large_distinct,
                from_files=large_files,
                to_files=stress_files,
                alpha=alpha,
            ),
        )
    else:
        if alpha < 0.35:
            tier = "mild"
        elif alpha < 0.7:
            tier = "sublinear"
        else:
            tier = "climbing"
        primary = _power_project(
            large_distinct,
            from_files=large_files,
            to_files=target_files,
            alpha=alpha,
        )
        stress = _power_project(
            large_distinct,
            from_files=large_files,
            to_files=stress_files,
            alpha=alpha,
        )

    return GrowthEstimate(
        tier=tier,
        raw_alpha=raw_alpha,
        alpha=alpha,
        projected_distinct=primary,
        stress_distinct=stress,
    )


def _projected_uniform_collision(distinct: int, buckets: int) -> float:
    if distinct <= 1:
        return 0.0
    if buckets <= 1:
        return 1.0 - 1.0 / distinct
    occupied = buckets * -math.expm1(
        distinct * math.log1p(-1.0 / buckets)
    )
    return max(0.0, 1.0 - occupied / distinct)


def _collision_bounded_buckets(distinct: int, max_collision: float) -> int:
    """Smallest power-of-two table whose uniform collision stays under a bound."""

    buckets = MIN_BUCKETS
    while (
        buckets < MAX_PRODUCTION_BUCKETS
        and _projected_uniform_collision(distinct, buckets) > max_collision
    ):
        buckets <<= 1
    return buckets


def _recommend_one(
    name: str,
    small_stats: Mapping[str, Any],
    large_stats: Mapping[str, Any],
    *,
    small_files: int,
    large_files: int,
    target_files: int,
    stress_files: int,
    retention_files: int,
    shared_group: bool,
) -> dict[str, Any]:
    d_small = int(small_stats.get("distinct_estimate") or 0)
    d_large = int(large_stats.get("distinct_estimate") or 0)
    top1 = 0.0 if shared_group else _top1_share(large_stats)
    growth = estimate_growth(
        small_distinct=d_small,
        large_distinct=d_large,
        small_files=small_files,
        large_files=large_files,
        target_files=target_files,
        stress_files=stress_files,
        top1_share=top1,
    )
    retention_growth = estimate_growth(
        small_distinct=d_small,
        large_distinct=d_large,
        small_files=small_files,
        large_files=large_files,
        target_files=retention_files,
        stress_files=retention_files,
        top1_share=top1,
    )

    result: dict[str, Any] = {
        "name": name,
        "tier": growth.tier,
        "d_small": d_small,
        "d_large": d_large,
        "raw_alpha": round(growth.raw_alpha, 6),
        "alpha": round(growth.alpha, 6),
        "top1_share": round(top1, 6) if not shared_group else None,
        "projected_distinct": growth.projected_distinct,
        "stress_distinct": growth.stress_distinct,
    }
    if growth.tier == "dead":
        result.update(
            {
                "action": "exclude_constant",
                "num_buckets": None,
                "embedding_dim": None,
                "ideal_num_buckets": None,
                "ideal_embedding_dim": None,
                "one_hour_floor_distinct": retention_growth.projected_distinct,
                "one_hour_floor_num_buckets": None,
                "production_load_num_buckets": None,
                "hard_bucket_cap": MAX_PRODUCTION_BUCKETS,
                "uncapped_num_buckets": None,
                "projected_load": None,
                "projected_uniform_collision": None,
                "bf16_weight_mib": 0.0,
            }
        )
        return result

    ideal_buckets = _next_power_of_two(
        int(math.ceil(growth.projected_distinct / IDEAL_MAX_LOAD))
    )
    one_hour_floor_buckets = _next_power_of_two(
        int(
            math.ceil(
                retention_growth.projected_distinct / IDEAL_MAX_LOAD
            )
        )
    )
    production_load_buckets = _next_power_of_two(
        int(math.ceil(growth.projected_distinct / PRODUCTION_MAX_LOAD))
    )
    collision_floor_buckets = MIN_BUCKETS
    if growth.projected_distinct <= SMALL_TABLE_DISTINCT_CEILING:
        collision_floor_buckets = _collision_bounded_buckets(
            growth.projected_distinct,
            SMALL_TABLE_MAX_COLLISION,
        )
        ideal_buckets = max(ideal_buckets, collision_floor_buckets)
    uncapped_production_buckets = max(
        one_hour_floor_buckets,
        production_load_buckets,
        collision_floor_buckets,
    )
    buckets = min(uncapped_production_buckets, MAX_PRODUCTION_BUCKETS)
    ideal_dimension = _suggest_embedding_dim(growth.projected_distinct)
    dimension = _suggest_production_embedding_dim(growth.projected_distinct)
    if growth.tier == "near_const":
        action = "weak_optional"
    elif buckets < uncapped_production_buckets:
        action = "use_hard_capped"
    elif buckets < ideal_buckets or dimension < ideal_dimension:
        action = "use_budgeted"
    else:
        action = "use"
    result.update(
        {
            "action": action,
            "num_buckets": buckets,
            "embedding_dim": dimension,
            "ideal_num_buckets": ideal_buckets,
            "ideal_embedding_dim": ideal_dimension,
            "one_hour_floor_distinct": retention_growth.projected_distinct,
            "one_hour_floor_num_buckets": one_hour_floor_buckets,
            "production_load_num_buckets": production_load_buckets,
            "hard_bucket_cap": MAX_PRODUCTION_BUCKETS,
            # Kept as an explicit alias for consumers of format_version <= 2.
            "uncapped_num_buckets": ideal_buckets,
            "projected_load": round(growth.projected_distinct / buckets, 6),
            "projected_uniform_collision": round(
                _projected_uniform_collision(
                    growth.projected_distinct,
                    buckets,
                ),
                6,
            ),
            "bf16_weight_mib": round(
                (buckets + 1) * dimension * 2 / (1 << 20),
                3,
            ),
        }
    )
    return result


def _configured_shapes(config_path: Path) -> dict[str, list[list[int]]]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{config_path} must contain a YAML object")
    source_shapes: dict[str, set[tuple[int, int]]] = {}

    def _capture(entry: Mapping[str, Any]) -> None:
        if entry.get("kind") != "categorical":
            return
        encoding = entry.get("encoding")
        if not isinstance(encoding, Mapping):
            return
        buckets = encoding.get("num_buckets")
        dimension = entry.get("embedding_dim")
        if not isinstance(buckets, int) or not isinstance(dimension, int):
            return
        source = str(entry.get("source") or entry.get("name") or "")
        if source:
            source_shapes.setdefault(source, set()).add((buckets, dimension))

    for feature in payload.get("features") or []:
        if isinstance(feature, Mapping):
            _capture(feature)
    for sequence in payload.get("sequences") or []:
        if not isinstance(sequence, Mapping):
            continue
        for field in sequence.get("fields") or []:
            if isinstance(field, Mapping):
                _capture(field)
    return {
        source: [list(shape) for shape in sorted(shapes)]
        for source, shapes in sorted(source_shapes.items())
    }


def build_recommendations(
    small: Mapping[str, Any],
    large: Mapping[str, Any],
    *,
    config_path: Path,
    small_path: str,
    large_path: str,
    files_per_hour: int,
    target_hours: float,
    stress_hours: float,
) -> dict[str, Any]:
    small_files, large_files = _validate_nested_profiles(small, large)
    if files_per_hour <= 0:
        raise ValueError("files_per_hour must be positive")
    if target_hours <= 0:
        raise ValueError("target_hours must be positive")
    if stress_hours < target_hours:
        raise ValueError("stress_hours must be at least target_hours")
    target_files = max(large_files, int(math.ceil(files_per_hour * target_hours)))
    stress_files = max(target_files, int(math.ceil(files_per_hour * stress_hours)))
    retention_files = min(
        target_files,
        max(large_files, int(math.ceil(files_per_hour * RETENTION_HOURS))),
    )

    spec = load_profile_spec(config_path)
    configured_sources = set(spec.categorical_sources)
    current_shapes = _configured_shapes(config_path)
    small_fields = _require_mapping(small["fields"], "small.fields")
    large_fields = _require_mapping(large["fields"], "large.fields")

    tier_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    fields: list[dict[str, Any]] = []
    recommended_shapes: dict[str, dict[str, Any]] = {}
    for name in sorted(large_fields):
        small_stats = _require_mapping(
            small_fields[name],
            f"small.fields.{name}",
        )
        large_stats = _require_mapping(
            large_fields[name],
            f"large.fields.{name}",
        )
        if _is_non_embedding_source(name):
            entry = {
                "name": name,
                "tier": "non_embedding",
                "action": "exclude_non_embedding",
                "reason": (
                    "derived dense time delta"
                    if name.endswith("time_delta_log1p_seconds")
                    else "coarse scene prior uses fixed identity encoding"
                ),
                "d_small": int(small_stats.get("distinct_estimate") or 0),
                "d_large": int(large_stats.get("distinct_estimate") or 0),
                "in_config": name in configured_sources,
                "configured_shapes": current_shapes.get(name, []),
            }
        else:
            entry = _recommend_one(
                name,
                small_stats,
                large_stats,
                small_files=small_files,
                large_files=large_files,
                target_files=target_files,
                stress_files=stress_files,
                retention_files=retention_files,
                shared_group=False,
            )
            entry["in_config"] = name in configured_sources
            entry["configured_shapes"] = current_shapes.get(name, [])
            if entry["num_buckets"] is not None:
                recommended_shapes[name] = {
                    "num_buckets": entry["num_buckets"],
                    "embedding_dim": entry["embedding_dim"],
                    "ideal_num_buckets": entry["ideal_num_buckets"],
                    "ideal_embedding_dim": entry["ideal_embedding_dim"],
                    "basis": "field",
                    "projected_distinct": entry["projected_distinct"],
                    "projected_load": entry["projected_load"],
                    "projected_uniform_collision": entry[
                        "projected_uniform_collision"
                    ],
                }
        tier_counts[str(entry["tier"])] += 1
        action_counts[str(entry["action"])] += 1
        fields.append(entry)

    small_shared = _require_mapping(
        small["shared_embedding_groups"],
        "small.shared_embedding_groups",
    )
    large_shared = _require_mapping(
        large["shared_embedding_groups"],
        "large.shared_embedding_groups",
    )
    shared_groups: list[dict[str, Any]] = []
    shared_tier_counts: Counter[str] = Counter()
    shared_action_counts: Counter[str] = Counter()
    for name in sorted(large_shared):
        entry = _recommend_one(
            name,
            _require_mapping(small_shared[name], f"small.shared.{name}"),
            _require_mapping(large_shared[name], f"large.shared.{name}"),
            small_files=small_files,
            large_files=large_files,
            target_files=target_files,
            stress_files=stress_files,
            retention_files=retention_files,
            shared_group=True,
        )
        sources = large_shared[name].get("sources")
        entry["sources"] = list(sources) if isinstance(sources, list) else []
        entry["configured_shapes"] = current_shapes.get(name, [])
        shared_groups.append(entry)
        shared_tier_counts[str(entry["tier"])] += 1
        shared_action_counts[str(entry["action"])] += 1
        if entry["num_buckets"] is not None:
            # Shared-root union evidence must override the root field estimate.
            recommended_shapes[name] = {
                "num_buckets": entry["num_buckets"],
                "embedding_dim": entry["embedding_dim"],
                "ideal_num_buckets": entry["ideal_num_buckets"],
                "ideal_embedding_dim": entry["ideal_embedding_dim"],
                "basis": "shared_group_union",
                "projected_distinct": entry["projected_distinct"],
                "projected_load": entry["projected_load"],
                "projected_uniform_collision": entry[
                    "projected_uniform_collision"
                ],
            }

    configured_with_recommendation = configured_sources & set(recommended_shapes)
    configured_excluded = configured_sources - set(recommended_shapes)
    return {
        "format_version": FORMAT_VERSION,
        "method": {
            "description": (
                "Per-source Heaps growth from nested profiles; project to the "
                f"{target_hours:g}-hour training window. Shared roots use "
                "union cardinality; production shapes are constrained by the "
                "two-H100 budget."
            ),
            "profiles": {
                "small": {
                    "path": small_path,
                    "files": small_files,
                    "rows": int(small.get("rows_scanned") or 0),
                },
                "large": {
                    "path": large_path,
                    "files": large_files,
                    "rows": int(large.get("rows_scanned") or 0),
                },
            },
            "production_files_per_hour": files_per_hour,
            "primary_horizon": {
                "hours": target_hours,
                "files": target_files,
                "used_for_config": True,
            },
            "retention_floor_horizon": {
                "hours": retention_files / files_per_hour,
                "files": retention_files,
                "target_max_load": IDEAL_MAX_LOAD,
                "used_for_config": True,
            },
            "stress_horizon": {
                "hours": stress_hours,
                "files": stress_files,
                "used_for_config": False,
            },
            "growth": {
                "formula": "d_target = d_large * (target_files / large_files)^alpha",
                "alpha": (
                    "log(d_large/d_small) / log(large_files/small_files), "
                    f"clamped to [0,{MAX_GROWTH_ALPHA}]"
                ),
                "saturated_floor": "max(power projection, 1.25*d_large)",
                "slow_floor": "max(power projection, 1.5*d_large)",
            },
            "bucket_policy": {
                "ideal_target_max_load": IDEAL_MAX_LOAD,
                "production_target_max_load": PRODUCTION_MAX_LOAD,
                "small_table_distinct_ceiling": SMALL_TABLE_DISTINCT_CEILING,
                "small_table_max_collision": SMALL_TABLE_MAX_COLLISION,
                "rounding": "next power of two, minimum 256",
                "production_formula": (
                    "max(one_hour_load_0.5_buckets, "
                    "primary_load_1.75_buckets, "
                    "small_table_collision_floor_buckets)"
                ),
                "hard_bucket_cap": MAX_PRODUCTION_BUCKETS,
                "planned_embedding_memory_gib_per_gpu": (
                    PLANNED_EMBEDDING_MEMORY_GIB_PER_GPU
                ),
                "note": (
                    "The collision estimate assumes uniformly distributed "
                    "pre-hashed low bits. The memory ceiling is validated on "
                    "the generated two-GPU bf16/row-wise-Adagrad configs. "
                    "Tables projected at or below the small-table ceiling are "
                    "sized by collision rather than load: load 0.5 discards "
                    "roughly a fifth of their distinct values, which strips the "
                    "resolution from the medium-cardinality conversion features."
                ),
            },
            "embedding_dim_policy": {
                "<=16": 8,
                "<=256": 16,
                "<=4096": 24,
                "<=65536": 32,
                "<=1000000": 48,
                ">1000000": 64,
                "production_cap_when_projected_distinct_gt_1000000": (
                    HIGH_CARDINALITY_PRODUCTION_DIM
                ),
                "note": (
                    "The uncapped tier is reported as ideal_embedding_dim. "
                    "Production caps high-cardinality widths to buy more hash "
                    "capacity; predictive optimum still requires an ablation."
                ),
            },
        },
        "summary": {
            "profiled_fields": len(fields),
            "configured_categorical_sources": len(configured_sources),
            "configured_sources_with_recommendation": len(
                configured_with_recommendation
            ),
            "configured_sources_excluded": sorted(configured_excluded),
            "recommended_shapes": len(recommended_shapes),
            "shared_groups": len(shared_groups),
            "tiers": dict(sorted(tier_counts.items())),
            "actions": dict(sorted(action_counts.items())),
            "shared_tiers": dict(sorted(shared_tier_counts.items())),
            "shared_actions": dict(sorted(shared_action_counts.items())),
        },
        "fields": fields,
        "shared_embedding_groups": shared_groups,
        "recommended_shapes": dict(sorted(recommended_shapes.items())),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--small-profile",
        type=Path,
        default=Path("docs/profile.json"),
    )
    parser.add_argument(
        "--large-profile",
        type=Path,
        default=Path("docs/new_profile.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mdl_rankmixer.yaml"),
    )
    parser.add_argument(
        "--files-per-hour",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--target-hours",
        type=float,
        default=24.0,
    )
    parser.add_argument(
        "--stress-hours",
        type=float,
        default=168.0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/emb_bucket_recommendation_growth.json"),
    )
    args = parser.parse_args(argv)

    small = load_json_report(args.small_profile)
    large = load_json_report(args.large_profile)
    payload = build_recommendations(
        small,
        large,
        config_path=args.config,
        small_path=str(args.small_profile),
        large_path=str(args.large_profile),
        files_per_hour=args.files_per_hour,
        target_hours=args.target_hours,
        stress_hours=args.stress_hours,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
