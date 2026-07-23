#!/usr/bin/env python3
"""Recommend per-field categorical encodings from multi-partition Parquet.

Existing ``profile_prehashed_parquet.py`` estimates power-of-two bucket sizes
under raw low-bit masking.  This script answers the next question: for each
configured categorical source, should production use::

    identity | vocab | remixed_pre_hashed | head_tail | keep_pre_hashed

It deliberately scans one or more ``--input`` partitions (local or HDFS),
compares raw vs SplitMix64-remixed low-bit collisions, and emits a JSON report
with evidence + a conservative strategy guess.  Multi-day union is expressed by
passing multiple partition URIs; a single file is accepted but marked low
confidence for dynamic entity fields.

Example::

    python scripts/recommend_categorical_encodings.py \\
      --config configs/mdl_rankmixer.yaml \\
      --input /data/dt=2026-07-01 \\
      --input /data/dt=2026-07-02 \\
      --input /data/dt=2026-07-03 \\
      --output /tmp/encoding_recommendations.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import yaml

from scripts.profile_prehashed_parquet import (  # noqa: E402
    DEFAULT_BUCKETS,
    MASK64,
    BottomKValues,
    FieldProfile,
    HyperLogLog,
    ProfileSpec,
    _dataset_from_uri,
    _fragment_path,
    _mix64,
    _parse_buckets,
    _suggest_embedding_dim,
    load_profile_spec,
)


IDENTITY_MAX_VALUE_DEFAULT = 8192
VOCAB_MAX_DISTINCT_DEFAULT = 50_000
HEAD_TAIL_DISTINCT_DEFAULT = 1_000_000
WITHIN_ROW_BUCKETS = (4096, 262_144, 16_777_216)


def _as_uint64(value: int) -> int:
    return value & MASK64


def _bucket_id(value: int, bucket_size: int, *, remix: bool, seed: int = 0) -> int:
    bits = _as_uint64(value)
    if remix:
        bits = _mix64(bits ^ (seed & MASK64))
    return bits & (bucket_size - 1)


def _collision_stats(
    values: Sequence[int],
    bucket_size: int,
    *,
    remix: bool,
    seed: int = 0,
) -> dict[str, float | int]:
    if not values or bucket_size <= 0:
        return {
            "sample_size": 0,
            "occupied": 0,
            "collision_rate": 0.0,
        }
    occupied = len({_bucket_id(value, bucket_size, remix=remix, seed=seed) for value in values})
    sample_size = len(values)
    return {
        "sample_size": sample_size,
        "occupied": occupied,
        "collision_rate": (sample_size - occupied) / sample_size,
    }


def _projected_uniform_collision(distinct_estimate: int, bucket_size: int) -> float:
    if distinct_estimate <= 1:
        return 0.0
    if bucket_size <= 1:
        return 1.0 - 1.0 / distinct_estimate
    occupied = bucket_size * -math.expm1(distinct_estimate * math.log1p(-1.0 / bucket_size))
    return max(0.0, 1.0 - occupied / distinct_estimate)


def _frequency_weighted_exposure(
    heavy_hitters: Mapping[int, int],
    bucket_size: int,
    *,
    remix: bool,
    seed: int = 0,
    top_k: int = 256,
) -> dict[str, float | int]:
    ranked = sorted(heavy_hitters.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    if not ranked or bucket_size <= 0:
        return {
            "top_k": 0,
            "distinct_in_top": 0,
            "occupied": 0,
            "distinct_collision_rate": 0.0,
            "event_weighted_exposure": 0.0,
            "colliding_bucket_count": 0,
        }
    buckets: dict[int, list[tuple[int, int]]] = {}
    total_events = 0
    for value, count in ranked:
        total_events += count
        buckets.setdefault(_bucket_id(value, bucket_size, remix=remix, seed=seed), []).append(
            (value, count)
        )
    colliding = {bucket: items for bucket, items in buckets.items() if len(items) > 1}
    exposed_events = sum(count for items in colliding.values() for _, count in items)
    distinct = len(ranked)
    occupied = len(buckets)
    return {
        "top_k": top_k,
        "distinct_in_top": distinct,
        "occupied": occupied,
        "distinct_collision_rate": (distinct - occupied) / distinct if distinct else 0.0,
        "event_weighted_exposure": exposed_events / total_events if total_events else 0.0,
        "colliding_bucket_count": len(colliding),
    }


class StrategyFieldProfile(FieldProfile):
    """FieldProfile plus within-row bag collision probes."""

    def __init__(
        self,
        *,
        sample_size: int = 4096,
        hll_precision: int = 12,
        within_row_buckets: Sequence[int] = WITHIN_ROW_BUCKETS,
        remix_seed: int = 0,
    ) -> None:
        super().__init__(sample_size=sample_size, hll_precision=hll_precision)
        self.remix_seed = remix_seed & MASK64
        self.within_row_buckets = tuple(within_row_buckets)
        self.within_row_groups = 0
        self.within_row_raw_distinct_sum = 0
        self.within_row_raw_mapped: Counter[int] = Counter()
        self.within_row_remix_mapped: Counter[int] = Counter()

    def _observe_bag(self, values: Sequence[int]) -> None:
        unique = tuple(dict.fromkeys(values))
        if len(unique) < 2:
            return
        self.within_row_groups += 1
        self.within_row_raw_distinct_sum += len(unique)
        for bucket_size in self.within_row_buckets:
            raw_mapped = len({_bucket_id(v, bucket_size, remix=False) for v in unique})
            remix_mapped = len(
                {
                    _bucket_id(v, bucket_size, remix=True, seed=self.remix_seed)
                    for v in unique
                }
            )
            self.within_row_raw_mapped[bucket_size] += raw_mapped
            self.within_row_remix_mapped[bucket_size] += remix_mapped

    def _walk(self, value: Any, *, depth: int, row_flags: dict[str, bool]) -> None:
        if isinstance(value, (list, tuple)) and value and all(
            not isinstance(item, (list, tuple)) for item in value
        ):
            leaves: list[int] = []
            for item in value:
                if item is None:
                    self.nulls_by_depth[depth + 1] += 1
                    row_flags["nested_null"] = True
                    continue
                if isinstance(item, bool) or not isinstance(item, int):
                    self.invalid_leaf_count += 1
                    continue
                leaves.append(item)
                # Reuse parent leaf accounting without recursing into a nested list.
                self.leaf_count += 1
                if item < 0:
                    self.negative_count += 1
                elif item > 0:
                    self.positive_count += 1
                else:
                    self.zero_count += 1
                self.minimum = item if self.minimum is None else min(self.minimum, item)
                self.maximum = item if self.maximum is None else max(self.maximum, item)
                unsigned = item & MASK64
                self.unsigned_minimum = (
                    unsigned
                    if self.unsigned_minimum is None
                    else min(self.unsigned_minimum, unsigned)
                )
                self.unsigned_maximum = (
                    unsigned
                    if self.unsigned_maximum is None
                    else max(self.unsigned_maximum, unsigned)
                )
                self.high16[unsigned >> 48] += 1
                self.heavy_hitters[item] += 1
                if len(self.heavy_hitters) > 256:
                    self.heavy_hitters = Counter(dict(self.heavy_hitters.most_common(64)))
                self.distinct.add(item)
                self.sample.add(item)
            lengths = self.list_lengths_by_depth.setdefault(depth, Counter())
            lengths[len(value)] += 1
            if not value:
                self.empty_lists_by_depth[depth] += 1
                row_flags["empty"] = True
            if leaves:
                self._observe_bag(leaves)
            return
        super()._walk(value, depth=depth, row_flags=row_flags)


def load_current_encodings(config_path: str | Path) -> dict[str, dict[str, Any]]:
    """Map physical Parquet source -> current encoding declared in YAML."""

    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a YAML object")
    encodings: dict[str, dict[str, Any]] = {}

    def _capture(entry: Mapping[str, Any]) -> None:
        encoding = entry.get("encoding")
        if not isinstance(encoding, Mapping):
            return
        source = str(entry.get("source") or entry.get("name") or "")
        if not source:
            return
        encodings[source] = {
            "name": entry.get("name"),
            "encoding_type": encoding.get("type"),
            "num_buckets": encoding.get("num_buckets"),
            "share_with": encoding.get("share_with"),
            "share_embedding": bool(encoding.get("share_embedding", False)),
            "embedding_dim": entry.get("embedding_dim"),
        }

    for feature in payload.get("features") or []:
        if isinstance(feature, Mapping):
            _capture(feature)
    for sequence in payload.get("sequences") or []:
        if not isinstance(sequence, Mapping):
            continue
        for field in sequence.get("fields") or []:
            if isinstance(field, Mapping):
                _capture(field)
    return encodings


def _span_fraction(unsigned_min: int | None, unsigned_max: int | None) -> float | None:
    if unsigned_min is None or unsigned_max is None:
        return None
    return (unsigned_max - unsigned_min) / float(MASK64)


def _dense_identity_shape(profile: FieldProfile) -> bool:
    if profile.minimum is None or profile.maximum is None:
        return False
    if profile.minimum < 0:
        return False
    width = profile.maximum - profile.minimum + 1
    if width <= 0 or width > IDENTITY_MAX_VALUE_DEFAULT:
        return False
    estimate = int(round(profile.distinct.estimate())) if profile.leaf_count else 0
    # Enough of the inclusive range is occupied to look like a bounded code space.
    return estimate > 0 and estimate <= width and estimate >= max(2, int(0.25 * width))


def recommend_strategy(
    *,
    source: str,
    profile: StrategyFieldProfile,
    current: Mapping[str, Any] | None,
    candidate_buckets: Sequence[int],
    input_count: int,
    identity_max_value: int = IDENTITY_MAX_VALUE_DEFAULT,
    vocab_max_distinct: int = VOCAB_MAX_DISTINCT_DEFAULT,
    head_tail_distinct: int = HEAD_TAIL_DISTINCT_DEFAULT,
) -> dict[str, Any]:
    estimate = int(round(profile.distinct.estimate())) if profile.leaf_count else 0
    sample_values = profile.sample.values()
    seed = _mix64(sum(ord(ch) for ch in source) & MASK64)
    current_buckets = None
    if current and isinstance(current.get("num_buckets"), int):
        current_buckets = int(current["num_buckets"])

    probe_buckets = list(candidate_buckets)
    if current_buckets and current_buckets not in probe_buckets:
        probe_buckets.append(current_buckets)
        probe_buckets.sort()

    bucket_comparisons: list[dict[str, Any]] = []
    best_remix_gain = 0.0
    for bucket_size in probe_buckets:
        raw = _collision_stats(sample_values, bucket_size, remix=False)
        remixed = _collision_stats(sample_values, bucket_size, remix=True, seed=seed)
        raw_rate = float(raw["collision_rate"])
        remix_rate = float(remixed["collision_rate"])
        gain = raw_rate - remix_rate
        best_remix_gain = max(best_remix_gain, gain)
        head_raw = _frequency_weighted_exposure(
            profile.heavy_hitters, bucket_size, remix=False, top_k=128
        )
        head_remix = _frequency_weighted_exposure(
            profile.heavy_hitters, bucket_size, remix=True, seed=seed, top_k=128
        )
        bucket_comparisons.append(
            {
                "bucket_size": bucket_size,
                "raw": raw,
                "remixed": remixed,
                "remix_collision_gain": gain,
                "projected_uniform_collision_rate": _projected_uniform_collision(
                    estimate, bucket_size
                ),
                "head_frequency_raw": head_raw,
                "head_frequency_remixed": head_remix,
                "is_current": bucket_size == current_buckets,
            }
        )

    within_row: dict[str, Any] = {"groups": profile.within_row_groups}
    if profile.within_row_groups:
        for bucket_size in profile.within_row_buckets:
            raw_mapped = profile.within_row_raw_mapped[bucket_size]
            remix_mapped = profile.within_row_remix_mapped[bucket_size]
            raw_distinct = profile.within_row_raw_distinct_sum
            within_row[str(bucket_size)] = {
                "avg_raw_collision_rate": 1.0 - (raw_mapped / raw_distinct),
                "avg_remixed_collision_rate": 1.0 - (remix_mapped / raw_distinct),
            }

    span_frac = _span_fraction(profile.unsigned_minimum, profile.unsigned_maximum)
    high16_diversity = len(profile.high16)
    reasons: list[str] = []
    warnings: list[str] = []
    strategy = "keep_pre_hashed"
    confidence = "low"
    suggested_buckets: int | None = current_buckets
    suggested_dim = _suggest_embedding_dim(estimate)

    pigeonhole = (
        current_buckets is not None
        and estimate > current_buckets
        and current_buckets > 0
    )
    if pigeonhole:
        warnings.append(
            f"distinct_estimate {estimate} exceeds current num_buckets "
            f"{current_buckets}; injective mapping is impossible"
        )

    if estimate == 0:
        strategy = "insufficient_data"
        reasons.append("no non-null integer leaves observed")
    elif (
        profile.minimum is not None
        and profile.maximum is not None
        and profile.minimum >= 0
        and profile.maximum < identity_max_value
        and _dense_identity_shape(profile)
        and high16_diversity <= 2
        and (span_frac is None or span_frac < 1e-12)
    ):
        strategy = "identity"
        confidence = "medium" if input_count >= 2 else "low"
        suggested_buckets = int(profile.maximum) + 1
        reasons.append(
            "non-negative values occupy a small dense code range; prefer identity "
            "over hash buckets"
        )
        if profile.zero_count:
            warnings.append(
                "zeros observed; confirm whether 0 is a real category or padding "
                "before switching to identity"
            )
    elif pigeonhole:
        strategy = "enlarge_pre_hashed"
        confidence = "high"
        suggested_buckets = None
        for bucket_size in probe_buckets:
            if bucket_size >= max(estimate * 2, current_buckets or 0):
                suggested_buckets = bucket_size
                break
        if suggested_buckets is None:
            suggested_buckets = max(probe_buckets) if probe_buckets else current_buckets
        reasons.append(
            "current bucket table is smaller than observed distinct values; "
            "injective mapping is impossible"
        )
        if best_remix_gain >= 0.02:
            reasons.append(
                "also consider remixed_pre_hashed while enlarging; raw low-bits "
                "show extra collisions versus SplitMix64"
            )
    elif (
        estimate <= vocab_max_distinct
        and high16_diversity <= 8
        and (span_frac is None or span_frac < 0.01)
        and not _dense_identity_shape(profile)
    ):
        strategy = "vocab_candidate"
        confidence = "low"
        suggested_buckets = None
        reasons.append(
            "moderate cardinality clustered in a narrow uint64 region; exact vocab "
            "may work if multi-day churn stays low"
        )
        warnings.append("confirm 7/30-day union and OOV rate before committing to vocab")
    elif estimate >= head_tail_distinct or (
        current_buckets is not None and current_buckets >= (1 << 24)
    ):
        strategy = "head_tail"
        confidence = "medium" if input_count >= 3 else "low"
        suggested_buckets = current_buckets
        reasons.append(
            "high cardinality / very large current table; protect hot IDs with an "
            "exact head and remixed-hash tail"
        )
        if input_count < 3:
            warnings.append("need more partitions before sizing head-K or shrinking tail B")
    elif best_remix_gain >= 0.02 or (
        span_frac is not None and span_frac > 0.1 and high16_diversity >= 16
    ):
        strategy = "remixed_pre_hashed"
        confidence = "medium"
        suggested_buckets = None
        for row in bucket_comparisons:
            if float(row["remixed"]["collision_rate"]) <= 0.05 and estimate <= int(
                row["bucket_size"]
            ) * 2:
                suggested_buckets = int(row["bucket_size"])
                break
        if suggested_buckets is None:
            suggested_buckets = current_buckets or (1 << 23)
        reasons.append(
            "field looks hash-like and/or remixed low-bits reduce collisions versus "
            "raw truncation"
        )
    else:
        strategy = "keep_pre_hashed"
        confidence = "low"
        reasons.append(
            "no strong identity/vocab signal; keep pre_hashed but re-evaluate bucket "
            "size with multi-day union"
        )

    if strategy in {"vocab_candidate", "head_tail", "keep_pre_hashed"} and input_count < 2:
        warnings.append("single input only; dynamic entity sizing remains inconclusive")

    return {
        "source": source,
        "strategy": strategy,
        "confidence": confidence,
        "reasons": reasons,
        "warnings": warnings,
        "remix_seed": seed,
        "suggested_num_buckets": suggested_buckets,
        "suggested_embedding_dim": suggested_dim,
        "current": current,
        "stats": {
            "leaf_count": profile.leaf_count,
            "distinct_estimate": estimate,
            "signed_min": profile.minimum,
            "signed_max": profile.maximum,
            "unsigned_min": profile.unsigned_minimum,
            "unsigned_max": profile.unsigned_maximum,
            "unsigned_span_fraction": span_frac,
            "zero_count": profile.zero_count,
            "high16_diversity": high16_diversity,
            "top_uint64_high16": [
                {"prefix": prefix, "count": count}
                for prefix, count in profile.high16.most_common(8)
            ],
            "top_values_approx": [
                {"value": value, "count": count}
                for value, count in profile.heavy_hitters.most_common(16)
            ],
        },
        "bucket_comparisons": bucket_comparisons,
        "within_row_collisions": within_row,
    }


def scan_for_recommendations(
    inputs: Sequence[str],
    spec: ProfileSpec,
    current_encodings: Mapping[str, Mapping[str, Any]],
    *,
    candidate_buckets: Sequence[int] = DEFAULT_BUCKETS,
    sample_size: int = 4096,
    hll_precision: int = 12,
    batch_size: int = 2048,
    max_files: int | None = None,
    max_rows: int | None = None,
    progress: bool = False,
    identity_max_value: int = IDENTITY_MAX_VALUE_DEFAULT,
    vocab_max_distinct: int = VOCAB_MAX_DISTINCT_DEFAULT,
    head_tail_distinct: int = HEAD_TAIL_DISTINCT_DEFAULT,
) -> dict[str, Any]:
    if not inputs:
        raise ValueError("at least one explicit --input is required")
    if any(bucket <= 0 or bucket & (bucket - 1) for bucket in candidate_buckets):
        raise ValueError("all candidate buckets must be positive powers of two")

    within_buckets = list(WITHIN_ROW_BUCKETS)
    for encoding in current_encodings.values():
        buckets = encoding.get("num_buckets")
        if isinstance(buckets, int) and buckets > 0 and buckets & (buckets - 1) == 0:
            if buckets not in within_buckets:
                within_buckets.append(buckets)
    within_buckets.sort()

    profiles = {
        source: StrategyFieldProfile(
            sample_size=sample_size,
            hll_precision=hll_precision,
            within_row_buckets=within_buckets,
            remix_seed=_mix64(sum(ord(ch) for ch in source) & MASK64),
        )
        for source in spec.categorical_sources
    }
    files_scanned: list[str] = []
    rows_scanned = 0
    missing_by_input: dict[str, list[str]] = {}

    for input_uri in inputs:
        dataset = _dataset_from_uri(input_uri)
        schema_names = set(dataset.schema.names)
        missing_by_input[input_uri] = sorted(
            source for source in spec.categorical_sources if source not in schema_names
        )
        available = [source for source in spec.categorical_sources if source in schema_names]
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
                data = batch.to_pydict()
                for source, profile in profiles.items():
                    if source not in data:
                        continue
                    for value in data[source]:
                        profile.observe(value)
                rows_scanned += batch.num_rows
                fragment_rows += batch.num_rows
            if progress:
                print(
                    f"scanned {files_scanned[-1]}: {fragment_rows} rows "
                    f"({rows_scanned} total)",
                    file=sys.stderr,
                    flush=True,
                )
        if max_files is not None and len(files_scanned) >= max_files:
            break
        if max_rows is not None and rows_scanned >= max_rows:
            break

    recommendations = [
        recommend_strategy(
            source=source,
            profile=profile,
            current=current_encodings.get(source),
            candidate_buckets=candidate_buckets,
            input_count=len(inputs),
            identity_max_value=identity_max_value,
            vocab_max_distinct=vocab_max_distinct,
            head_tail_distinct=head_tail_distinct,
        )
        for source, profile in sorted(profiles.items())
    ]

    by_strategy: Counter[str] = Counter(item["strategy"] for item in recommendations)
    high_priority = [
        {
            "source": item["source"],
            "strategy": item["strategy"],
            "confidence": item["confidence"],
            "warnings": item["warnings"],
            "current_num_buckets": (item.get("current") or {}).get("num_buckets"),
            "distinct_estimate": item["stats"]["distinct_estimate"],
        }
        for item in recommendations
        if item["strategy"] in {"identity", "enlarge_pre_hashed", "head_tail"}
        or item["warnings"]
    ]

    shared_overlap: list[dict[str, Any]] = []
    for root, sources in spec.shared_groups.items():
        hll = HyperLogLog(hll_precision)
        sample = BottomKValues(sample_size)
        for source in sources:
            profiles[source].merge_distinct_into(hll, sample)
        shared_overlap.append(
            {
                "root": root,
                "sources": list(sources),
                "union_distinct_estimate": int(round(hll.estimate()))
                if any(profiles[source].leaf_count for source in sources)
                else 0,
                "per_source_distinct_estimate": {
                    source: int(round(profiles[source].distinct.estimate()))
                    if profiles[source].leaf_count
                    else 0
                    for source in sources
                },
            }
        )

    return {
        "format_version": 1,
        "inputs": list(inputs),
        "input_count": len(inputs),
        "files_scanned": files_scanned,
        "rows_scanned": rows_scanned,
        "missing_configured_columns_by_input": missing_by_input,
        "settings": {
            "candidate_buckets": list(candidate_buckets),
            "sample_size": sample_size,
            "hll_precision": hll_precision,
            "max_files": max_files,
            "max_rows": max_rows,
            "identity_max_value": identity_max_value,
            "vocab_max_distinct": vocab_max_distinct,
            "head_tail_distinct": head_tail_distinct,
        },
        "strategy_counts": dict(sorted(by_strategy.items())),
        "high_priority": high_priority,
        "shared_embedding_groups": shared_overlap,
        "recommendations": recommendations,
        "notes": [
            "Strategies are heuristic proposals, not automatic config rewrites.",
            "Pass multiple --input partitions (e.g. 7/30 days) before shrinking "
            "sku/goods/user tables.",
            "remixed_pre_hashed means SplitMix64(uint64(raw) ^ field_seed) then low bits.",
            "identity/vocab still require an upstream value-domain contract check.",
            "Use profile_prehashed_parquet.py when you only need bucket-size estimates "
            "compatible with build_mdl_rankmixer_config.py.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mdl_rankmixer.yaml"))
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Local path or HDFS URI. Repeat for multi-day/partition union.",
    )
    parser.add_argument("--output", type=Path, help="JSON report path; stdout when omitted")
    parser.add_argument("--context-feature-count", type=int, default=51)
    parser.add_argument("--candidate-buckets", type=_parse_buckets, default=DEFAULT_BUCKETS)
    parser.add_argument("--sample-size", type=int, default=4096)
    parser.add_argument("--hll-precision", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--identity-max-value", type=int, default=IDENTITY_MAX_VALUE_DEFAULT)
    parser.add_argument("--vocab-max-distinct", type=int, default=VOCAB_MAX_DISTINCT_DEFAULT)
    parser.add_argument("--head-tail-distinct", type=int, default=HEAD_TAIL_DISTINCT_DEFAULT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        spec = load_profile_spec(
            args.config,
            context_feature_count=args.context_feature_count,
        )
        current = load_current_encodings(args.config)
        report = scan_for_recommendations(
            args.input,
            spec,
            current,
            candidate_buckets=args.candidate_buckets,
            sample_size=args.sample_size,
            hll_precision=args.hll_precision,
            batch_size=args.batch_size,
            max_files=args.max_files,
            max_rows=args.max_rows,
            progress=not args.quiet,
            identity_max_value=args.identity_max_value,
            vocab_max_distinct=args.vocab_max_distinct,
            head_tail_distinct=args.head_tail_distinct,
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
