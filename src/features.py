"""Categorical feature encoding for MDL.

Owns vocab fitting/loading, encoding-strategy fingerprints, and hash-bucket
helpers. Does not own logical feature definitions (config.FeatureConfig),
parquet I/O (dataloader.py), or model embeddings (model.py).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

from .config import (
    AppConfig,
    ResolvedCategoricalInput,
    ResolvedEncoding,
    ResolvedHashEncoding,
    ResolvedIdentityEncoding,
    ResolvedPreHashedEncoding,
    ResolvedSharedVocabEncoding,
    ResolvedVocabEncoding,
    VocabStrategy,
    resolve_categorical_base_input,
)


# --- Vocab fitting & artifacts ---


@dataclass(frozen=True)
class FittedVocab:
    feature_name: str
    path: Path
    size: int
    min_count: int
    max_size: int | None


@dataclass(frozen=True)
class VocabFitPlanEntry:
    feature_name: str
    source: str
    artifact: str
    min_count: int | None
    max_size: int | None


@dataclass(frozen=True)
class VocabFitPlan:
    entries: tuple[VocabFitPlanEntry, ...]
    columns: tuple[str, ...]


def _require_pyarrow() -> tuple[Any, Any, Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required for vocab fitting and loading; install it in the runtime environment"
        ) from error
    return pa, pc, ds, pq


def _artifact_path(config: AppConfig, feature_name: str, encoding: ResolvedVocabEncoding) -> Path:
    if not encoding.artifact:
        raise ValueError(f"vocab feature {feature_name!r} requires artifact")
    return Path(config.vocab_strategy.defaults.artifact_dir) / encoding.artifact


def _contiguous_array(array: Any) -> Any:
    """Combine chunks without Arrow nested-dictionary unification."""

    pa, pc, _ds, _pq = _require_pyarrow()
    if not hasattr(array, "num_chunks"):
        return array
    if not array.num_chunks:
        return array.combine_chunks()
    if array.num_chunks == 1:
        return array.chunk(0)
    if not all(pa.types.is_dictionary(chunk.type) for chunk in array.chunks):
        return array.combine_chunks()
    dictionaries: list[Any] = []
    shifted_indices: list[Any] = []
    offset = 0
    for chunk in array.chunks:
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


def _flatten_array_values(array: Any) -> list[Any]:
    pa, pc, _ds, _pq = _require_pyarrow()
    current = _contiguous_array(array)
    if pa.types.is_dictionary(current.type):
        # Arrow cannot dictionary_decode list-valued dictionaries; take works.
        current = pc.take(current.dictionary, current.indices)
    while pa.types.is_list(current.type) or pa.types.is_large_list(current.type):
        current = pc.list_flatten(current)
    return [value.as_py() for value in current if value.as_py() is not None]


def _update_counter(counter: Counter[str], table: Any, source: str) -> None:
    if source not in table.column_names:
        raise ValueError(f"vocab source column {source!r} is missing from parquet batch")
    for value in _flatten_array_values(table[source]):
        counter[str(value)] += 1


def _write_vocab(path: Path, values: list[tuple[str, int, int]]) -> None:
    pa, _pc, _ds, pq = _require_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "value": [value for value, _id, _count in values],
            "id": [_id for _value, _id, _count in values],
            "count": [_count for _value, _id, _count in values],
        }
    )
    pq.write_table(table, path)


def plan_vocab_fit(config: AppConfig) -> VocabFitPlan:
    entries: list[VocabFitPlanEntry] = []
    columns: list[str] = []
    for categorical_input in config.resolved.categorical_inputs:
        encoding = categorical_input.encoding
        if encoding.encoding != "vocab":
            continue
        entries.append(
            VocabFitPlanEntry(
                feature_name=categorical_input.name,
                source=categorical_input.source,
                artifact=encoding.artifact,
                min_count=encoding.min_count,
                max_size=encoding.max_size,
            )
        )
        columns.append(categorical_input.source)
    return VocabFitPlan(
        entries=tuple(entries),
        columns=tuple(dict.fromkeys(columns)),
    )


def fit_vocabs(config: AppConfig, tables: Iterable[Any], plan: VocabFitPlan) -> list[FittedVocab]:
    if not plan.entries:
        return []

    counters = {entry.feature_name: Counter() for entry in plan.entries}

    for table in tables:
        for entry in plan.entries:
            _update_counter(counters[entry.feature_name], table, entry.source)

    fitted: list[FittedVocab] = []
    for entry in plan.entries:
        min_count = entry.min_count or 1
        candidates = [
            (value, count)
            for value, count in counters[entry.feature_name].items()
            if count >= min_count
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        if entry.max_size is not None:
            candidates = candidates[: entry.max_size]
        rows = [(value, index + 1, count) for index, (value, count) in enumerate(candidates)]
        path = Path(config.vocab_strategy.defaults.artifact_dir) / entry.artifact
        _write_vocab(path, rows)
        fitted.append(
            FittedVocab(
                feature_name=entry.feature_name,
                path=path,
                size=len(rows) + 1,
                min_count=min_count,
                max_size=entry.max_size,
            )
        )
    return fitted


def load_vocab_map(path: str | Path) -> dict[str, int]:
    _pa, _pc, _ds, pq = _require_pyarrow()
    table = pq.read_table(path)
    values = table["value"].to_pylist()
    ids = table["id"].to_pylist()
    return {str(value): int(index) for value, index in zip(values, ids)}


def load_vocab_maps(config: AppConfig) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    resolving: set[str] = set()
    by_name = config.resolved.categorical_input_by_name

    def resolve(feature_name: str) -> dict[str, int]:
        if feature_name in maps:
            return maps[feature_name]
        if feature_name in resolving:
            raise ValueError(f"shared_vocab cycle detected at {feature_name!r}")
        categorical_input = by_name.get(feature_name)
        if categorical_input is None:
            raise ValueError(f"vocab feature {feature_name!r} is not configured")
        encoding = categorical_input.encoding
        if encoding.encoding == "vocab":
            maps[feature_name] = load_vocab_map(_artifact_path(config, feature_name, encoding))
            return maps[feature_name]
        if encoding.encoding == "shared_vocab":
            base_input = resolve_categorical_base_input(by_name, feature_name)
            if isinstance(base_input.encoding, ResolvedIdentityEncoding):
                # The physical column already stores bounded integer IDs. No
                # artifact or Python lookup is needed for this alias.
                return {}
            resolving.add(feature_name)
            try:
                maps[feature_name] = resolve(encoding.share_with)
            finally:
                resolving.remove(feature_name)
            return maps[feature_name]
        raise ValueError(
            f"feature {feature_name!r} with encoding {encoding.encoding!r} does not have a vocab map"
        )

    for categorical_input in config.resolved.categorical_inputs:
        if categorical_input.encoding.encoding not in {"vocab", "shared_vocab"}:
            continue
        base_input = resolve_categorical_base_input(
            by_name, categorical_input.name
        )
        if isinstance(base_input.encoding, ResolvedVocabEncoding):
            resolve(categorical_input.name)
    return maps


# --- Encoding strategy helpers ---


@dataclass(frozen=True)
class VocabArtifactRef:
    feature_name: str
    encoding: str
    artifact_path: str | None
    size_hint: int | None = None


def _encoding_payload(encoding: ResolvedEncoding, source: str) -> dict[str, Any]:
    return {
        "encoding": encoding.encoding,
        "source": source,
        "min_count": encoding.min_count if isinstance(encoding, ResolvedVocabEncoding) else None,
        "max_size": encoding.max_size if isinstance(encoding, ResolvedVocabEncoding) else None,
        "artifact": encoding.artifact if isinstance(encoding, ResolvedVocabEncoding) else None,
        "num_buckets": (
            encoding.num_buckets
            if isinstance(encoding, (ResolvedHashEncoding, ResolvedPreHashedEncoding))
            else None
        ),
        "salt": encoding.salt if isinstance(encoding, ResolvedHashEncoding) else None,
        "identity_num_buckets": (
            encoding.num_buckets if isinstance(encoding, ResolvedIdentityEncoding) else None
        ),
        "padding_id": (
            encoding.padding_id if isinstance(encoding, ResolvedIdentityEncoding) else None
        ),
        "out_of_range": (
            encoding.out_of_range if isinstance(encoding, ResolvedIdentityEncoding) else None
        ),
        "share_with": (
            encoding.share_with
            if isinstance(
                encoding,
                (
                    ResolvedIdentityEncoding,
                    ResolvedPreHashedEncoding,
                    ResolvedSharedVocabEncoding,
                ),
            )
            else None
        ),
        "share_embedding": bool(getattr(encoding, "share_embedding", False)),
    }


def _strategy_payload(strategy: VocabStrategy) -> dict[str, Any]:
    return {
        name: {
            "encoding": feature.encoding,
            "source": feature.source,
            "min_count": feature.min_count,
            "max_size": feature.max_size,
            "artifact": feature.artifact,
            "num_buckets": feature.num_buckets,
            "salt": feature.salt,
            "max_id": feature.max_id,
            "padding_id": feature.padding_id,
            "out_of_range": feature.out_of_range,
            "share_with": feature.share_with,
            "share_embedding": feature.share_embedding,
        }
        for name, feature in sorted(strategy.features.items())
    }


def _resolved_payload(config: AppConfig) -> dict[str, Any]:
    return {
        item.name: _encoding_payload(item.encoding, item.source)
        for item in sorted(config.resolved.categorical_inputs, key=lambda item: item.name)
    }


def vocab_strategy_fingerprint(strategy_or_config: VocabStrategy | AppConfig) -> str:
    strategy = (
        strategy_or_config.vocab_strategy
        if isinstance(strategy_or_config, AppConfig)
        else strategy_or_config
    )
    payload = {
        "defaults": {
            "fit_split": strategy.defaults.fit_split,
            "oov_id": strategy.defaults.oov_id,
            "padding_id": strategy.defaults.padding_id,
            "unseen_policy": strategy.defaults.unseen_policy,
            "artifact_dir": strategy.defaults.artifact_dir,
        },
        "features": (
            _resolved_payload(strategy_or_config)
            if isinstance(strategy_or_config, AppConfig)
            else _strategy_payload(strategy)
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def stable_hash_bucket(value: Any, num_buckets: int, salt: str | None = None) -> int:
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive")
    prefix = "" if salt is None else f"{salt}:"
    digest = sha256((prefix + str(value)).encode("utf-8")).digest()
    # Bucket ids start at 1 because 0 is reserved for OOV/padding.
    return int.from_bytes(digest[:8], "little") % num_buckets + 1


def stable_hash_buckets(values: Iterable[Any], num_buckets: int, salt: str | None = None) -> list[int]:
    return [stable_hash_bucket(value, num_buckets, salt) for value in values]


def pre_hashed_bucket(value: Any, num_buckets: int) -> int:
    """Map one signed int64 hash by its unchanged uint64 low bits.

    The power-of-two requirement makes signed and unsigned modulo identical
    without ``abs`` and avoids the ``INT64_MIN`` overflow corner case.  Zero is
    an upstream-contract violation; true null is handled separately as padding.
    """

    if num_buckets <= 0 or num_buckets & (num_buckets - 1):
        raise ValueError("pre_hashed num_buckets must be a positive power of two")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"pre_hashed value must be an int64, got {type(value).__name__}")
    if value < -(1 << 63) or value > (1 << 63) - 1:
        raise ValueError(f"pre_hashed value {value!r} is outside signed int64 range")
    if value == 0:
        raise ValueError("pre_hashed non-null value must not be zero")
    return (value & (num_buckets - 1)) + 1


def _unseen_value_error(feature_name: str, value: Any) -> ValueError:
    return ValueError(f"unseen categorical value {value!r} for {feature_name!r}")


def encode_categorical_value(
    value: Any,
    categorical_input: ResolvedCategoricalInput,
    vocab_map: dict[str, int] | None,
    unseen_policy: str = "oov",
) -> int:
    if value is None:
        return 0
    encoding = categorical_input.encoding
    if encoding.encoding in {"vocab", "shared_vocab"}:
        if vocab_map is None:
            raise ValueError(f"vocab map is required for categorical input {categorical_input.name!r}")
        key = str(value)
        if key in vocab_map:
            return vocab_map[key]
        if unseen_policy == "error":
            raise _unseen_value_error(categorical_input.name, value)
        return 0
    if encoding.encoding == "hash":
        return stable_hash_bucket(value, encoding.num_buckets, encoding.salt)
    if encoding.encoding == "pre_hashed":
        return pre_hashed_bucket(value, encoding.num_buckets)
    if encoding.encoding == "identity":
        encoded = int(value)
        if encoded < 0 or encoded >= encoding.num_buckets:
            if encoding.out_of_range == "error":
                raise _unseen_value_error(categorical_input.name, value)
            return encoding.padding_id
        return encoded
    raise ValueError(f"unsupported encoding {encoding.encoding!r}")


def encode_categorical_values(
    values: Iterable[Any],
    categorical_input: ResolvedCategoricalInput,
    vocab_maps: dict[str, dict[str, int]],
    unseen_policy: str = "oov",
) -> list[int]:
    vocab_map = vocab_maps.get(categorical_input.name)
    return [
        encode_categorical_value(value, categorical_input, vocab_map, unseen_policy)
        for value in values
    ]


def encode_categorical_sequence_field(
    rows: Iterable[Iterable[Any]],
    categorical_input: ResolvedCategoricalInput,
    vocab_maps: dict[str, dict[str, int]],
    unseen_policy: str = "oov",
) -> list[list[int]]:
    vocab_map = vocab_maps.get(categorical_input.name)
    return [
        [
            encode_categorical_value(item, categorical_input, vocab_map, unseen_policy)
            for item in row
        ]
        for row in rows
    ]


def vocab_artifacts(strategy_or_config: VocabStrategy | AppConfig) -> list[VocabArtifactRef]:
    refs: list[VocabArtifactRef] = []
    if isinstance(strategy_or_config, AppConfig):
        artifact_dir = strategy_or_config.vocab_strategy.defaults.artifact_dir.rstrip("/")
        for categorical_input in sorted(
            strategy_or_config.resolved.categorical_inputs,
            key=lambda item: item.name,
        ):
            encoding = categorical_input.encoding
            artifact_path = None
            if isinstance(encoding, ResolvedVocabEncoding) and encoding.artifact:
                artifact_path = f"{artifact_dir}/{encoding.artifact}"
            size_hint = None
            if isinstance(encoding, ResolvedVocabEncoding):
                size_hint = encoding.max_size
            elif isinstance(encoding, ResolvedHashEncoding):
                size_hint = encoding.num_buckets
            elif isinstance(encoding, ResolvedPreHashedEncoding):
                size_hint = encoding.num_buckets
            elif isinstance(encoding, ResolvedIdentityEncoding):
                size_hint = encoding.num_buckets
            refs.append(
                VocabArtifactRef(
                    feature_name=categorical_input.name,
                    encoding=encoding.encoding,
                    artifact_path=artifact_path,
                    size_hint=size_hint,
                )
            )
        return refs

    strategy = strategy_or_config
    artifact_dir = strategy.defaults.artifact_dir.rstrip("/")
    for feature_name, feature_strategy in sorted(strategy.features.items()):
        artifact_path = None
        if feature_strategy.artifact:
            artifact_path = f"{artifact_dir}/{feature_strategy.artifact}"
        size_hint = (
            feature_strategy.max_size
            or feature_strategy.num_buckets
            or (feature_strategy.max_id + 1 if feature_strategy.max_id is not None else None)
        )
        refs.append(
            VocabArtifactRef(
                feature_name=feature_name,
                encoding=feature_strategy.encoding,
                artifact_path=artifact_path,
                size_hint=size_hint,
            )
        )
    return refs
