from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .build_dataset import _csv_column_source, _feature_source_specs, _parse_feature_value


def ensure_processed_layout(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_manifest(output_dir: str | Path, manifest: Mapping[str, Any]) -> Path:
    output_path = ensure_processed_layout(output_dir) / "manifest.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(manifest), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output_path


def write_csv_split(
    output_dir: str | Path,
    split: str,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: list[str],
) -> Path:
    output_path = ensure_processed_layout(output_dir) / f"{split}.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"manifest must contain {key} object")
    return value


def _require_non_empty_list(parent: Mapping[str, Any], key: str) -> list[Any]:
    value = parent.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"manifest {key} must be a non-empty list")
    return value


def _require_string_list(parent: Mapping[str, Any], key: str) -> list[str]:
    values = _require_non_empty_list(parent, key)
    if any(not isinstance(value, str) or value == "" for value in values):
        raise ValueError(f"manifest {key} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"manifest {key} must not contain duplicates")
    return values


def _feature_names(section: str, specs: list[Any]) -> set[str]:
    names = []
    for spec in specs:
        if not isinstance(spec, Mapping):
            raise ValueError(f"tokenization {section} entries must be objects")
        name = spec.get("name")
        if not isinstance(name, str) or name == "":
            raise ValueError(f"tokenization {section} entries must declare non-empty name")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError(f"tokenization {section} feature names must be unique")
    return set(names)


def _validate_token_specs(
    section: str,
    specs: list[Any],
    feature_names: set[str],
    expected_count: int | None = None,
) -> None:
    if not specs:
        raise ValueError(f"tokenization {section} must be a non-empty list")
    if expected_count is not None and len(specs) != expected_count:
        raise ValueError(f"tokenization {section} must contain {expected_count} entries")
    token_ids = []
    for spec in specs:
        if not isinstance(spec, Mapping):
            raise ValueError(f"tokenization {section} entries must be objects")
        try:
            token_id = int(spec.get("token_id", len(token_ids)))
        except (TypeError, ValueError) as error:
            raise ValueError(f"tokenization {section} token_id must be an integer") from error
        if token_id < 0:
            raise ValueError(f"tokenization {section} token_id must be non-negative")
        token_ids.append(token_id)
        inputs = spec.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise ValueError(f"tokenization {section} entries must contain non-empty inputs")
        for input_spec in inputs:
            input_name = input_spec if isinstance(input_spec, str) else input_spec.get("name")
            if input_name not in feature_names:
                raise ValueError(
                    f"tokenization {section} references unknown feature {input_name!r}"
                )
    if len(set(token_ids)) != len(token_ids):
        raise ValueError(f"tokenization {section} token_id values must be unique")


def validate_manifest(
    manifest: Mapping[str, Any],
    require_domain_tokenization: bool = True,
) -> None:
    splits = _require_string_list(manifest, "splits")
    _require_string_list(manifest, "scenario_names")
    task_names = _require_string_list(manifest, "task_names")
    data_columns = _require_mapping(manifest, "data_columns")

    if "scenario_id" not in data_columns and "scenario_ids" not in data_columns:
        raise ValueError("data_columns must declare scenario_id or scenario_ids")
    if "scenario_id" in data_columns and "scenario_ids" in data_columns:
        raise ValueError("data_columns must not declare both scenario_id and scenario_ids")
    if "group_id" not in data_columns:
        raise ValueError("data_columns must declare group_id")
    if "sample_weight" in data_columns and not isinstance(data_columns["sample_weight"], str):
        raise ValueError("data_columns.sample_weight must be a csv column name")
    labels = data_columns.get("labels")
    label_masks = data_columns.get("label_masks")
    if not isinstance(labels, Mapping) or not isinstance(label_masks, Mapping):
        raise ValueError("data_columns must declare labels and label_masks objects")
    if set(labels.keys()) != set(task_names):
        raise ValueError("data_columns.labels keys must exactly match task_names")
    if set(label_masks.keys()) != set(task_names):
        raise ValueError("data_columns.label_masks keys must exactly match task_names")

    tokenization = _require_mapping(manifest, "tokenization")
    if tokenization.get("version") != 2 or tokenization.get("kind") != "encoder_registry":
        raise ValueError("tokenization must use version=2 and kind='encoder_registry'")
    features = _require_non_empty_list(tokenization, "features")
    token_specs = _require_non_empty_list(tokenization, "token_specs")
    feature_names = _feature_names("features", features)
    _validate_token_specs("token_specs", token_specs, feature_names)

    if require_domain_tokenization:
        required_domain_keys = [
            "scenario_features",
            "scenario_token_specs",
            "task_features",
            "task_token_specs",
        ]
        missing = [key for key in required_domain_keys if key not in tokenization]
        if missing:
            raise ValueError(
                "tokenization must declare scenario_features, scenario_token_specs, "
                f"task_features, and task_token_specs; missing: {', '.join(missing)}"
            )
        scenario_features = _require_non_empty_list(tokenization, "scenario_features")
        scenario_token_specs = _require_non_empty_list(tokenization, "scenario_token_specs")
        task_features = _require_non_empty_list(tokenization, "task_features")
        task_token_specs = _require_non_empty_list(tokenization, "task_token_specs")

        scenario_feature_names = _feature_names("scenario_features", scenario_features)
        task_feature_names = _feature_names("task_features", task_features)
        _validate_token_specs(
            "scenario_token_specs",
            scenario_token_specs,
            scenario_feature_names,
            expected_count=len(manifest["scenario_names"]) + 1,
        )
        _validate_token_specs(
            "task_token_specs",
            task_token_specs,
            task_feature_names,
            expected_count=len(task_names),
        )
    if not splits:
        raise ValueError("manifest must declare at least one split")


def _required_csv_columns(manifest: Mapping[str, Any]) -> set[str]:
    data_columns = manifest["data_columns"]
    task_names = manifest["task_names"]
    columns = {str(data_columns.get("group_id"))}
    if "sample_weight" in data_columns:
        columns.add(str(data_columns["sample_weight"]))
    scenario_column = data_columns.get("scenario_ids", data_columns.get("scenario_id"))
    columns.add(str(scenario_column))
    labels = data_columns["labels"]
    label_masks = data_columns["label_masks"]
    for task_name in task_names:
        columns.add(str(labels[task_name]))
        columns.add(str(label_masks[task_name]))
    for spec in _feature_source_specs(dict(manifest)):
        source = _csv_column_source(spec)
        columns.add(str(source["column"]))
    return columns


def _validate_csv_header(split: str, fieldnames: list[str] | None, required_columns: set[str]) -> None:
    if fieldnames is None:
        raise ValueError(f"split {split!r} csv must contain a header")
    missing = sorted(required_columns - set(fieldnames))
    if missing:
        raise ValueError(f"split {split!r} csv is missing columns: {', '.join(missing)}")


def _validate_scenario_row(
    row: Mapping[str, str],
    data_columns: Mapping[str, Any],
    num_scenarios: int,
    split: str,
    row_number: int,
) -> None:
    try:
        if "scenario_ids" in data_columns:
            delimiter = str(data_columns.get("scenario_ids_delimiter", "|"))
            value = row[str(data_columns["scenario_ids"])]
            scenario_ids = [int(part.strip()) for part in value.split(delimiter) if part.strip() != ""]
            if not scenario_ids:
                raise ValueError("scenario_ids column must contain at least one id")
        else:
            scenario_ids = [int(row[str(data_columns["scenario_id"])])]
    except (KeyError, ValueError) as error:
        raise ValueError(f"split {split!r} row {row_number}: invalid scenario id") from error
    for scenario_id in scenario_ids:
        if scenario_id < 0 or scenario_id >= num_scenarios:
            raise ValueError(
                f"split {split!r} row {row_number}: scenario id {scenario_id} out of range"
            )


def _validate_csv_rows(
    split_path: Path,
    split: str,
    manifest: Mapping[str, Any],
    max_rows: int | None,
) -> None:
    data_columns = manifest["data_columns"]
    task_names = manifest["task_names"]
    feature_specs = _feature_source_specs(dict(manifest))
    row_count = 0
    with split_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_csv_header(split, reader.fieldnames, _required_csv_columns(manifest))
        for row_number, row in enumerate(reader, start=2):
            if max_rows is not None and row_count >= max_rows:
                break
            row_count += 1
            _validate_scenario_row(
                row,
                data_columns,
                len(manifest["scenario_names"]),
                split,
                row_number,
            )
            if "sample_weight" in data_columns:
                try:
                    float(row[str(data_columns["sample_weight"])])
                except (KeyError, ValueError) as error:
                    raise ValueError(
                        f"split {split!r} row {row_number}: invalid sample_weight"
                    ) from error
            for task_name in task_names:
                try:
                    float(row[str(data_columns["labels"][task_name])])
                    float(row[str(data_columns["label_masks"][task_name])])
                except (KeyError, ValueError) as error:
                    raise ValueError(
                        f"split {split!r} row {row_number}: invalid label or label_mask "
                        f"for task {task_name!r}"
                    ) from error
            for spec in feature_specs:
                source = _csv_column_source(spec)
                try:
                    _parse_feature_value(row.get(str(source["column"]), ""), spec)
                except (KeyError, ValueError) as error:
                    raise ValueError(
                        f"split {split!r} row {row_number}: invalid feature {spec['name']!r}"
                    ) from error
    if row_count == 0 and max_rows != 0:
        raise ValueError(f"split {split!r} csv must contain at least one row")


def validate_processed_dataset(
    data_dir: str | Path,
    max_rows: int | None = None,
    require_domain_tokenization: bool = True,
) -> None:
    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows must be non-negative")
    path = Path(data_dir)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_manifest(manifest, require_domain_tokenization=require_domain_tokenization)
    for split in manifest.get("splits", []):
        split_path = path / f"{split}.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"missing split csv: {split_path}")
        _validate_csv_rows(split_path, split, manifest, max_rows)
