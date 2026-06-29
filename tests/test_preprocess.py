from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.datasets.preprocess import validate_processed_dataset


def _valid_manifest() -> dict:
    return {
        "splits": ["train"],
        "scenario_names": ["default"],
        "task_names": ["click"],
        "data_columns": {
            "scenario_id": "scenario_id",
            "group_id": "group_id",
            "labels": {"click": "click"},
            "label_masks": {"click": "click_mask"},
        },
        "tokenization": {
            "version": 2,
            "kind": "encoder_registry",
            "features": [
                {
                    "name": "user_id",
                    "encoder": "embedding",
                    "vocab_size": 10,
                    "source": {"type": "csv_column", "column": "user_id", "dtype": "int64"},
                },
                {
                    "name": "score",
                    "encoder": "identity",
                    "dim": 1,
                    "source": {"type": "csv_column", "column": "score", "dtype": "float32"},
                },
            ],
            "token_specs": [
                {"token_id": 0, "projection": "linear", "inputs": ["user_id"]},
                {"token_id": 1, "projection": "linear", "inputs": ["score"]},
            ],
            "scenario_features": [
                {"name": "user_id", "encoder": "embedding", "vocab_size": 10},
            ],
            "scenario_token_specs": [
                {"token_id": 0, "inputs": ["user_id"]},
                {"token_id": 1, "inputs": ["user_id"]},
            ],
            "task_features": [
                {"name": "score", "encoder": "identity", "dim": 1},
            ],
            "task_token_specs": [
                {"token_id": 0, "inputs": ["score"]},
            ],
        },
    }


def _write_dataset(path: Path, manifest: dict, scenario_id: int = 0, click: str = "1") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (path / "train.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario_id", "group_id", "click", "click_mask", "user_id", "score"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scenario_id": scenario_id,
                "group_id": "g1",
                "click": click,
                "click_mask": 1,
                "user_id": 1,
                "score": 0.2,
            }
        )


def test_validate_processed_dataset_requires_domain_tokenization(tmp_path: Path) -> None:
    manifest = _valid_manifest()
    for key in ["scenario_features", "scenario_token_specs", "task_features", "task_token_specs"]:
        manifest["tokenization"].pop(key)
    _write_dataset(tmp_path, manifest)

    with pytest.raises(ValueError, match="missing: scenario_features"):
        validate_processed_dataset(tmp_path)


def test_validate_processed_dataset_allows_feature_only_without_domain_tokenization(
    tmp_path: Path,
) -> None:
    manifest = _valid_manifest()
    for key in ["scenario_features", "scenario_token_specs", "task_features", "task_token_specs"]:
        manifest["tokenization"].pop(key)
    _write_dataset(tmp_path, manifest)

    validate_processed_dataset(tmp_path, require_domain_tokenization=False)


def test_validate_processed_dataset_allows_missing_group_id(tmp_path: Path) -> None:
    manifest = _valid_manifest()
    manifest["data_columns"].pop("group_id")
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (tmp_path / "train.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario_id", "click", "click_mask", "user_id", "score"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scenario_id": 0,
                "click": 1,
                "click_mask": 1,
                "user_id": 1,
                "score": 0.2,
            }
        )

    validate_processed_dataset(tmp_path)


def test_validate_processed_dataset_rejects_scenario_id_out_of_range(tmp_path: Path) -> None:
    _write_dataset(tmp_path, _valid_manifest(), scenario_id=1)

    with pytest.raises(ValueError, match="scenario id 1 out of range"):
        validate_processed_dataset(tmp_path)


def test_validate_processed_dataset_rejects_invalid_label(tmp_path: Path) -> None:
    _write_dataset(tmp_path, _valid_manifest(), click="bad")

    with pytest.raises(ValueError, match="invalid label or label_mask"):
        validate_processed_dataset(tmp_path)



def test_validate_processed_dataset_rejects_invalid_sample_weight(tmp_path: Path) -> None:
    manifest = _valid_manifest()
    manifest["data_columns"]["sample_weight"] = "sample_weight"
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (tmp_path / "train.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario_id",
                "group_id",
                "sample_weight",
                "click",
                "click_mask",
                "user_id",
                "score",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scenario_id": 0,
                "group_id": "g1",
                "sample_weight": "bad",
                "click": 1,
                "click_mask": 1,
                "user_id": 1,
                "score": 0.2,
            }
        )

    with pytest.raises(ValueError, match="invalid sample_weight"):
        validate_processed_dataset(tmp_path)
