from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .config import AppConfig, FeatureConfig, SequenceConfig, SequenceFieldConfig, VocabFeatureStrategy
from .features import stable_hash_bucket


@dataclass
class FeatureBatch:
    features: dict[str, Any]
    labels: Tensor | None
    label_mask: Tensor | None
    scenario_id: Tensor
    group_id: list[str]


def _column_values(table: Any, column: str) -> list[Any]:
    if column not in table.column_names:
        raise ValueError(f"missing required batch column {column!r}")
    return table[column].to_pylist()


def _strategy_for(config: AppConfig, feature_name: str) -> VocabFeatureStrategy | None:
    return config.vocab_strategy.features.get(feature_name)


def _encode_scalar(value: Any, strategy: VocabFeatureStrategy | None, vocab_map: dict[str, int] | None) -> int:
    if value is None:
        return 0
    if strategy is None:
        return int(value)
    if strategy.encoding in {"vocab", "shared_vocab"}:
        if vocab_map is None:
            raise ValueError(f"vocab map is required for source {strategy.source!r}")
        return vocab_map.get(str(value), 0)
    if strategy.encoding == "hash":
        if strategy.num_buckets is None:
            raise ValueError("hash strategy requires num_buckets")
        return stable_hash_bucket(value, strategy.num_buckets, strategy.salt)
    if strategy.encoding == "identity":
        encoded = int(value)
        if strategy.max_id is not None and encoded > strategy.max_id:
            return 0
        return encoded
    raise ValueError(f"unsupported encoding {strategy.encoding!r}")


def _truncate(values: list[Any], feature: FeatureConfig) -> list[Any]:
    if feature.max_length is None or len(values) <= feature.max_length:
        return values
    if feature.truncation == "tail":
        return values[-feature.max_length :]
    return values[: feature.max_length]


def _tensorize_categorical(
    config: AppConfig,
    feature: FeatureConfig,
    values: list[Any],
    vocab_maps: dict[str, dict[str, int]],
) -> Tensor:
    strategy = _strategy_for(config, feature.name)
    vocab_map = vocab_maps.get(feature.name)
    encoded = [_encode_scalar(value, strategy, vocab_map) for value in values]
    return torch.tensor(encoded, dtype=torch.long)


def _dense_feature_value(value: Any, dimension: int) -> float | list[float]:
    if value is None:
        return 0.0 if dimension == 1 else [0.0] * dimension
    if dimension == 1:
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"dense feature expected 1 value, got {len(value)}")
            value = value[0]
        return 0.0 if value is None else float(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"dense feature expected {dimension} values, got scalar {value!r}")
    if len(value) != dimension:
        raise ValueError(f"dense feature expected {dimension} values, got {len(value)}")
    return [0.0 if item is None else float(item) for item in value]


def _tensorize_dense(feature: FeatureConfig, values: list[Any]) -> Tensor:
    normalized = [_dense_feature_value(value, feature.dimension) for value in values]
    return torch.tensor(normalized, dtype=torch.float32)


def _tensorize_sequence(
    config: AppConfig,
    feature: FeatureConfig,
    values: list[Any],
    vocab_maps: dict[str, dict[str, int]],
) -> dict[str, Tensor]:
    strategy = _strategy_for(config, feature.name)
    vocab_map = vocab_maps.get(feature.name)
    encoded_rows: list[list[int]] = []
    for row in values:
        if row is None:
            items: list[Any] = []
        elif isinstance(row, list):
            items = row
        else:
            items = [row]
        items = _truncate(items, feature)
        encoded_rows.append([_encode_scalar(item, strategy, vocab_map) for item in items])

    lengths = torch.tensor([len(row) for row in encoded_rows], dtype=torch.long)
    max_length = int(lengths.max().item()) if encoded_rows else 0
    padded = [row + [0] * (max_length - len(row)) for row in encoded_rows]
    return {
        "values": torch.tensor(padded, dtype=torch.long) if max_length > 0 else torch.zeros(len(encoded_rows), 0, dtype=torch.long),
        "lengths": lengths,
    }


def _coerce_sequence_items(row: Any) -> list[Any]:
    if row is None:
        return []
    if isinstance(row, list):
        return row
    if isinstance(row, tuple):
        return list(row)
    return [row]


def _truncate_sequence_items(values: list[Any], sequence: SequenceConfig) -> list[Any]:
    if sequence.max_length is None or len(values) <= sequence.max_length:
        return values
    if sequence.truncation == "tail":
        return values[-sequence.max_length :]
    return values[: sequence.max_length]


def _dense_vector(value: Any, dimension: int) -> list[float]:
    if value is None:
        return [0.0] * dimension
    if dimension == 1 and not isinstance(value, (list, tuple)):
        return [float(value)]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"dense sequence field expected {dimension} values, got scalar {value!r}")
    if len(value) != dimension:
        raise ValueError(f"dense sequence field expected {dimension} values, got {len(value)}")
    return [0.0 if item is None else float(item) for item in value]


def _parallel_sequence_rows(
    table: Any,
    sequence: SequenceConfig,
) -> tuple[dict[str, list[list[Any]]], list[int]]:
    if not sequence.fields:
        return {}, []
    values_by_field = {field.name: _column_values(table, field.source) for field in sequence.fields}
    batch_size = len(next(iter(values_by_field.values())))
    rows_by_field = {field.name: [] for field in sequence.fields}
    lengths: list[int] = []

    for row_index in range(batch_size):
        raw_items_by_field: dict[str, list[Any]] = {}
        row_length: int | None = None
        for field in sequence.fields:
            items = _coerce_sequence_items(values_by_field[field.name][row_index])
            if row_length is None:
                row_length = len(items)
            elif len(items) != row_length:
                raise ValueError(
                    f"sequence {sequence.name!r} field {field.name!r} has length {len(items)} "
                    f"but expected {row_length} at row {row_index}"
                )
            raw_items_by_field[field.name] = items
        indices = list(range(row_length or 0))
        indices = _truncate_sequence_items(indices, sequence)
        lengths.append(len(indices))
        for field in sequence.fields:
            source_items = raw_items_by_field[field.name]
            rows_by_field[field.name].append([source_items[index] for index in indices])
    return rows_by_field, lengths


def _struct_step_value(step: Any, field: SequenceFieldConfig) -> Any:
    if step is None:
        return None
    if isinstance(step, dict):
        return step.get(field.source)
    return getattr(step, field.source, None)


def _list_struct_sequence_rows(
    table: Any,
    sequence: SequenceConfig,
) -> tuple[dict[str, list[list[Any]]], list[int]]:
    if sequence.source is None:
        raise ValueError(f"sequence {sequence.name!r} source is required for list_struct layout")
    rows = _column_values(table, sequence.source)
    rows_by_field = {field.name: [] for field in sequence.fields}
    lengths: list[int] = []
    for row in rows:
        steps = _truncate_sequence_items(_coerce_sequence_items(row), sequence)
        lengths.append(len(steps))
        for field in sequence.fields:
            rows_by_field[field.name].append([_struct_step_value(step, field) for step in steps])
    return rows_by_field, lengths


def _tensorize_multi_field_sequence(
    config: AppConfig,
    sequence: SequenceConfig,
    table: Any,
    vocab_maps: dict[str, dict[str, int]],
) -> dict[str, Any]:
    if sequence.layout == "parallel_lists":
        rows_by_field, row_lengths = _parallel_sequence_rows(table, sequence)
    elif sequence.layout == "list_struct":
        rows_by_field, row_lengths = _list_struct_sequence_rows(table, sequence)
    else:
        raise ValueError(f"unsupported sequence layout {sequence.layout!r}")

    lengths = torch.tensor(row_lengths, dtype=torch.long)
    max_length = int(lengths.max().item()) if row_lengths else 0
    tensor_fields: dict[str, Tensor] = {}
    for field in sequence.fields:
        rows = rows_by_field[field.name]
        if field.kind == "categorical":
            qualified = field.qualified_name(sequence.name)
            strategy = _strategy_for(config, qualified)
            vocab_map = vocab_maps.get(qualified)
            encoded_rows = [
                [_encode_scalar(item, strategy, vocab_map) for item in row]
                for row in rows
            ]
            padded = [row + [0] * (max_length - len(row)) for row in encoded_rows]
            tensor_fields[field.name] = (
                torch.tensor(padded, dtype=torch.long)
                if max_length > 0
                else torch.zeros(len(rows), 0, dtype=torch.long)
            )
        elif field.kind == "dense":
            encoded_dense = [
                [_dense_vector(item, field.dimension) for item in row]
                for row in rows
            ]
            zero = [0.0] * field.dimension
            padded_dense = [row + [zero] * (max_length - len(row)) for row in encoded_dense]
            tensor_fields[field.name] = (
                torch.tensor(padded_dense, dtype=torch.float32)
                if max_length > 0
                else torch.zeros(len(rows), 0, field.dimension, dtype=torch.float32)
            )
        else:
            raise ValueError(f"unsupported sequence field kind {field.kind!r}")
    return {"fields": tensor_fields, "lengths": lengths}


def _scenario_tensor(config: AppConfig, table: Any, batch_size: int) -> Tensor:
    if config.scenarios.source is None:
        return torch.zeros(batch_size, dtype=torch.long)
    values = _column_values(table, config.scenarios.source)
    return torch.tensor([0 if value is None else int(value) for value in values], dtype=torch.long)


def _group_ids(config: AppConfig, table: Any, batch_size: int) -> list[str]:
    source = None
    if config.data.train.agg_layout is not None and config.data.train.agg_layout.request_id in table.column_names:
        source = config.data.train.agg_layout.request_id
    elif config.data.test is not None and config.data.test.group_id in table.column_names:
        source = config.data.test.group_id
    if source is None:
        return ["" for _ in range(batch_size)]
    return ["" if value is None else str(value) for value in _column_values(table, source)]


def table_to_feature_batch(
    config: AppConfig,
    table: Any,
    vocab_maps: dict[str, dict[str, int]],
    require_labels: bool = True,
) -> FeatureBatch:
    batch_size = table.num_rows
    features: dict[str, Any] = {}
    for feature in config.features:
        values = _column_values(table, feature.source)
        if feature.kind == "categorical":
            features[feature.name] = _tensorize_categorical(config, feature, values, vocab_maps)
        elif feature.kind == "dense":
            features[feature.name] = _tensorize_dense(feature, values)
        elif feature.kind == "sequence":
            features[feature.name] = _tensorize_sequence(config, feature, values, vocab_maps)
        else:
            raise ValueError(f"unsupported feature kind {feature.kind!r}")
    for sequence in config.sequences:
        features[sequence.name] = _tensorize_multi_field_sequence(config, sequence, table, vocab_maps)

    labels = None
    label_mask = None
    label_columns = config.data.train.agg_layout.labels if config.data.train.agg_layout is not None else {}
    if label_columns and all(column in table.column_names for column in label_columns.values()):
        labels = torch.tensor(
            [
                [0.0 if value is None else float(value) for value in row]
                for row in zip(*[_column_values(table, column) for column in label_columns.values()])
            ],
            dtype=torch.float32,
        )
        mask_columns = config.data.train.agg_layout.label_masks if config.data.train.agg_layout is not None else {}
        if mask_columns and all(column in table.column_names for column in mask_columns.values()):
            label_mask = torch.tensor(
                [
                    [0.0 if value is None else float(value) for value in row]
                    for row in zip(*[_column_values(table, column) for column in mask_columns.values()])
                ],
                dtype=torch.float32,
            )
        else:
            label_mask = torch.ones_like(labels)
    elif require_labels:
        raise ValueError("required label columns are missing from batch")

    return FeatureBatch(
        features=features,
        labels=labels,
        label_mask=label_mask,
        scenario_id=_scenario_tensor(config, table, batch_size),
        group_id=_group_ids(config, table, batch_size),
    )


def move_feature_batch(batch: FeatureBatch, device: torch.device) -> FeatureBatch:
    def move_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: move_value(child) for key, child in value.items()}
        if isinstance(value, Tensor):
            return value.to(device)
        return value

    return FeatureBatch(
        features={key: move_value(value) for key, value in batch.features.items()},
        labels=None if batch.labels is None else batch.labels.to(device),
        label_mask=None if batch.label_mask is None else batch.label_mask.to(device),
        scenario_id=batch.scenario_id.to(device),
        group_id=batch.group_id,
    )
