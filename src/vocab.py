from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig, VocabFeatureStrategy
from .data import AggParquetScanner, ParquetScanner, _require_pyarrow, required_columns_for_split


@dataclass(frozen=True)
class FittedVocab:
    feature_name: str
    path: Path
    size: int
    min_count: int
    max_size: int | None


def _artifact_path(config: AppConfig, feature_name: str, strategy: VocabFeatureStrategy) -> Path:
    if strategy.artifact is None:
        raise ValueError(f"vocab feature {feature_name!r} requires artifact")
    return Path(config.vocab_strategy.defaults.artifact_dir) / strategy.artifact


def _flatten_array_values(array: Any) -> list[Any]:
    pa, pc, _ds, _pq = _require_pyarrow()
    current = array.combine_chunks() if hasattr(array, "combine_chunks") else array
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


def _vocab_columns(config: AppConfig) -> list[str]:
    return [
        strategy.source
        for strategy in config.vocab_strategy.features.values()
        if strategy.encoding == "vocab"
    ]


def fit_vocabs(config: AppConfig) -> list[FittedVocab]:
    vocab_strategies = {
        name: strategy
        for name, strategy in config.vocab_strategy.features.items()
        if strategy.encoding == "vocab"
    }
    if not vocab_strategies:
        return []

    split = config.data.train
    columns = sorted(set(required_columns_for_split(config, split)) | set(_vocab_columns(config)))
    scanner: ParquetScanner
    scanner = AggParquetScanner(split, columns) if split.format == "agg_parquet" else ParquetScanner(split, columns)
    counters = {name: Counter() for name in vocab_strategies}

    if isinstance(scanner, AggParquetScanner):
        table_iter = scanner.iter_candidate_tables()
    else:
        table_iter = scanner.iter_tables()

    for table in table_iter:
        for feature_name, strategy in vocab_strategies.items():
            _update_counter(counters[feature_name], table, strategy.source)

    fitted: list[FittedVocab] = []
    for feature_name, strategy in vocab_strategies.items():
        min_count = strategy.min_count or 1
        candidates = [
            (value, count)
            for value, count in counters[feature_name].items()
            if count >= min_count
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        if strategy.max_size is not None:
            candidates = candidates[: strategy.max_size]
        rows = [(value, index + 1, count) for index, (value, count) in enumerate(candidates)]
        path = _artifact_path(config, feature_name, strategy)
        _write_vocab(path, rows)
        fitted.append(
            FittedVocab(
                feature_name=feature_name,
                path=path,
                size=len(rows) + 1,
                min_count=min_count,
                max_size=strategy.max_size,
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
    for feature_name, strategy in config.vocab_strategy.features.items():
        if strategy.encoding == "vocab":
            maps[feature_name] = load_vocab_map(_artifact_path(config, feature_name, strategy))
    for feature_name, strategy in config.vocab_strategy.features.items():
        if strategy.encoding == "shared_vocab":
            if strategy.share_with not in maps:
                raise ValueError(
                    f"shared_vocab feature {feature_name!r} references vocab {strategy.share_with!r} "
                    "that has not been loaded"
                )
            maps[feature_name] = maps[strategy.share_with]
    return maps
