from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from .feature_schema import (
    feature_specs_from_manifest,
    scenario_feature_specs_from_manifest,
    task_feature_specs_from_manifest,
)


def load_manifest(data_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(data_dir) / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _data_columns(manifest: dict[str, Any]) -> dict[str, Any]:
    if "data_columns" not in manifest:
        raise ValueError("manifest must contain data_columns")
    return manifest["data_columns"]


def _csv_column_source(spec: dict[str, Any]) -> dict[str, Any]:
    source = spec.get("source")
    if not isinstance(source, dict) or source.get("type") != "csv_column":
        raise ValueError(f"feature {spec['name']!r} must use csv_column source")
    if "column" not in source:
        raise ValueError(f"feature {spec['name']!r} csv_column source must declare column")
    if "dtype" not in source:
        raise ValueError(f"feature {spec['name']!r} csv_column source must declare dtype")
    return source


def _add_source_spec(
    specs: list[dict[str, Any]],
    seen: set[str],
    spec: dict[str, Any],
) -> None:
    if "source" not in spec:
        return
    name = spec["name"]
    if name in seen:
        return
    specs.append(spec)
    seen.add(name)


def _add_encoder_source_specs(
    specs: list[dict[str, Any]],
    seen: set[str],
    feature_specs: list[dict[str, Any]],
) -> None:
    for spec in feature_specs:
        _add_source_spec(specs, seen, spec)
        if spec.get("encoder") not in {"din", "sequence_mean_pooling", "sim", "longer"}:
            continue
        for field_spec in spec.get("sequence_features", []):
            _add_source_spec(specs, seen, field_spec)
            target_feature = field_spec.get("target_feature")
            if isinstance(target_feature, dict):
                _add_source_spec(specs, seen, target_feature)
            elif isinstance(target_feature, str) and "target_source" in field_spec:
                _add_source_spec(
                    specs,
                    seen,
                    {"name": target_feature, "source": field_spec["target_source"]},
                )


def _feature_source_specs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    feature_groups = [feature_specs_from_manifest(manifest)]
    scenario_features = scenario_feature_specs_from_manifest(manifest)
    if scenario_features is not None:
        feature_groups.append(scenario_features)
    task_features = task_feature_specs_from_manifest(manifest)
    if task_features is not None:
        feature_groups.append(task_features)
    for feature_group in feature_groups:
        _add_encoder_source_specs(specs, seen, feature_group)
    return specs


def _default_missing_value(dtype: str) -> int | float | bool:
    if dtype in {"int", "int64", "long"}:
        return 0
    if dtype in {"float", "float32", "double"}:
        return 0.0
    if dtype in {"bool", "boolean"}:
        return False
    raise ValueError(f"unsupported csv source dtype {dtype!r}")


def _cast_value(value: Any, dtype: str) -> int | float | bool:
    if dtype in {"int", "int64", "long"}:
        return int(value)
    if dtype in {"float", "float32", "double"}:
        return float(value)
    if dtype in {"bool", "boolean"}:
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "t", "yes", "y"}:
            return True
        if normalized in {"0", "false", "f", "no", "n"}:
            return False
        raise ValueError(f"cannot parse {value!r} as bool")
    raise ValueError(f"unsupported csv source dtype {dtype!r}")


def _split_vector(value: str, source: dict[str, Any]) -> list[str]:
    delimiter = source.get("delimiter")
    if delimiter is None:
        return value.split()
    return [part.strip() for part in value.split(str(delimiter)) if part.strip() != ""]


def _parse_feature_value(value: str, spec: dict[str, Any]) -> Any:
    source = _csv_column_source(spec)
    dtype = str(source["dtype"]).lower()
    shape = source.get("shape", "scalar")
    missing_value = source.get("missing_value", _default_missing_value(dtype))

    if shape == "scalar":
        return _cast_value(missing_value if value == "" else value, dtype)
    if shape in {"vector", "list", "sequence"}:
        if value == "":
            if isinstance(missing_value, list):
                return [_cast_value(item, dtype) for item in missing_value]
            return []
        return [_cast_value(part, dtype) for part in _split_vector(value, source)]
    raise ValueError(f"feature {spec['name']!r} has unsupported csv source shape {shape!r}")


def _parse_scenario_value(row: dict[str, str], data_columns: dict[str, Any]) -> int | list[int]:
    if "scenario_ids" in data_columns:
        delimiter = str(data_columns.get("scenario_ids_delimiter", "|"))
        value = row[data_columns["scenario_ids"]]
        scenario_ids = [int(part.strip()) for part in value.split(delimiter) if part.strip() != ""]
        if not scenario_ids:
            raise ValueError("scenario_ids column must contain at least one scenario id")
        return scenario_ids
    return int(row[data_columns["scenario_id"]])


def _torch_dtype(dtype: str) -> torch.dtype:
    if dtype in {"int", "int64", "long"}:
        return torch.long
    if dtype in {"float", "float32"}:
        return torch.float32
    if dtype == "double":
        return torch.float64
    if dtype in {"bool", "boolean"}:
        return torch.bool
    raise ValueError(f"unsupported csv source dtype {dtype!r}")


def _collate_sequence(values: list[list[Any]], source: dict[str, Any]) -> dict[str, Tensor]:
    dtype = _torch_dtype(str(source["dtype"]).lower())
    padding_value = source.get("padding_value", _default_missing_value(str(source["dtype"]).lower()))
    lengths = torch.tensor([len(value) for value in values], dtype=torch.long)
    max_length = int(lengths.max().item()) if values else 0
    padded = [value + [padding_value] * (max_length - len(value)) for value in values]
    return {
        "values": torch.tensor(padded, dtype=dtype),
        "lengths": lengths,
    }


class ManifestDataset(IterableDataset[dict[str, Any]]):
    def __init__(self, data_dir: str | Path, split: str) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.manifest = load_manifest(self.data_dir)
        if split not in self.manifest["splits"]:
            raise ValueError(f"unknown split {split!r}")
        self.path = self.data_dir / f"{split}.csv"

    def __iter__(self) -> Iterator[dict[str, Any]]:
        task_names = self.manifest["task_names"]
        data_columns = _data_columns(self.manifest)
        feature_specs = _feature_source_specs(self.manifest)
        label_columns = data_columns["labels"]
        label_mask_columns = data_columns["label_masks"]
        sample_weight_column = data_columns.get("sample_weight")
        group_id_column = data_columns.get("group_id")

        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                features = {}
                feature_sources = {}
                for spec in feature_specs:
                    source = _csv_column_source(spec)
                    column = source["column"]
                    features[spec["name"]] = _parse_feature_value(row.get(column, ""), spec)
                    feature_sources[spec["name"]] = source

                yield {
                    "features": features,
                    "feature_sources": feature_sources,
                    "scenario_id": _parse_scenario_value(row, data_columns),
                    "num_scenarios": len(self.manifest["scenario_names"]),
                    "labels": [float(row[label_columns[name]]) for name in task_names],
                    "label_mask": [float(row[label_mask_columns[name]]) for name in task_names],
                    "sample_weight": float(row[sample_weight_column]) if sample_weight_column else 1.0,
                    "group_id": row[group_id_column] if group_id_column is not None else "",
                }


def build_dataset(data_dir: str | Path, split: str) -> ManifestDataset:
    return ManifestDataset(data_dir, split)


def collate_manifest_batch(rows: list[dict[str, Any]]) -> dict[str, Tensor | list[str] | dict[str, Any]]:
    feature_names = list(rows[0]["features"].keys()) if rows else []
    feature_sources = rows[0].get("feature_sources", {}) if rows else {}
    features: dict[str, Any] = {}
    for name in feature_names:
        source = feature_sources.get(name, {})
        values = [row["features"][name] for row in rows]
        if source.get("shape", "scalar") == "sequence":
            features[name] = _collate_sequence(values, source)
        else:
            dtype = _torch_dtype(str(source["dtype"]).lower()) if "dtype" in source else None
            features[name] = torch.tensor(values, dtype=dtype)
    scenario_values = [row["scenario_id"] for row in rows]
    if scenario_values and isinstance(scenario_values[0], list):
        num_scenarios = int(rows[0]["num_scenarios"])
        scenario_id = torch.zeros(len(rows), num_scenarios, dtype=torch.float32)
        for row_index, ids in enumerate(scenario_values):
            scenario_id[row_index, torch.tensor(ids, dtype=torch.long)] = 1.0
    else:
        scenario_id = torch.tensor(scenario_values, dtype=torch.long)
    return {
        "features": features,
        "scenario_id": scenario_id,
        "labels": torch.tensor([row["labels"] for row in rows], dtype=torch.float32),
        "label_mask": torch.tensor([row["label_mask"] for row in rows], dtype=torch.float32),
        "sample_weight": torch.tensor([row["sample_weight"] for row in rows], dtype=torch.float32),
        "group_id": [row["group_id"] for row in rows],
    }
