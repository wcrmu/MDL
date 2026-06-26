from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from src.datasets import ManifestDataset, collate_manifest_batch, load_manifest


def _write_manifest_dataset(path: Path) -> None:
    manifest = {
        "splits": ["train"],
        "scenario_names": ["home", "search"],
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
                    "name": "history",
                    "encoder": "sequence_mean_pooling",
                    "vocab_size": 20,
                    "source": {
                        "type": "csv_column",
                        "column": "history",
                        "dtype": "int64",
                        "shape": "sequence",
                        "delimiter": "|",
                    },
                },
            ],
            "token_specs": [
                {"token_id": 0, "projection": "linear", "inputs": ["user_id"]},
                {"token_id": 1, "projection": "linear", "inputs": ["history"]},
            ],
        },
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (path / "train.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario_id", "group_id", "click", "click_mask", "user_id", "history"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scenario_id": 0,
                "group_id": "q1",
                "click": 1,
                "click_mask": 1,
                "user_id": 3,
                "history": "1|2|3",
            }
        )
        writer.writerow(
            {
                "scenario_id": 1,
                "group_id": "q1",
                "click": 0,
                "click_mask": 1,
                "user_id": 4,
                "history": "2|5",
            }
        )


def test_manifest_dataset_collate(tmp_path: Path) -> None:
    _write_manifest_dataset(tmp_path)
    manifest = load_manifest(tmp_path)
    assert manifest["task_names"] == ["click"]

    rows = list(ManifestDataset(tmp_path, "train"))
    batch = collate_manifest_batch(rows)

    assert batch["scenario_id"].tolist() == [0, 1]
    assert torch.equal(batch["features"]["user_id"], torch.tensor([3, 4]))
    assert batch["features"]["history"]["values"].shape == (2, 3)
    assert batch["features"]["history"]["lengths"].tolist() == [3, 2]
    assert batch["labels"].shape == (2, 1)



def test_manifest_dataset_collate_multifield_din_sources(tmp_path: Path) -> None:
    manifest = {
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
                    "name": "item_id",
                    "encoder": "embedding",
                    "vocab_size": 20,
                    "source": {"type": "csv_column", "column": "item_id", "dtype": "int64"},
                },
                {
                    "name": "cate_id",
                    "encoder": "embedding",
                    "vocab_size": 8,
                    "source": {"type": "csv_column", "column": "cate_id", "dtype": "int64"},
                },
                {
                    "name": "price",
                    "encoder": "identity",
                    "dim": 1,
                    "source": {"type": "csv_column", "column": "price", "dtype": "float32"},
                },
                {
                    "name": "history_behavior",
                    "encoder": "din",
                    "fusion": "concat",
                    "sequence_features": [
                        {
                            "name": "hist_item_id",
                            "target_feature": "item_id",
                            "encoder": "embedding",
                            "vocab_size": 20,
                            "source": {
                                "type": "csv_column",
                                "column": "hist_item_id",
                                "dtype": "int64",
                                "shape": "sequence",
                                "delimiter": "|",
                            },
                        },
                        {
                            "name": "hist_cate_id",
                            "target_feature": "cate_id",
                            "encoder": "embedding",
                            "vocab_size": 8,
                            "source": {
                                "type": "csv_column",
                                "column": "hist_cate_id",
                                "dtype": "int64",
                                "shape": "sequence",
                                "delimiter": "|",
                            },
                        },
                        {
                            "name": "hist_price",
                            "target_feature": "price",
                            "encoder": "identity",
                            "dim": 1,
                            "projection_dim": 2,
                            "source": {
                                "type": "csv_column",
                                "column": "hist_price",
                                "dtype": "float32",
                                "shape": "sequence",
                                "delimiter": "|",
                            },
                        },
                    ],
                },
            ],
            "token_specs": [
                {"token_id": 0, "projection": "linear", "inputs": ["item_id"]},
                {"token_id": 1, "projection": "linear", "inputs": ["history_behavior"]},
            ],
        },
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (tmp_path / "train.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario_id",
                "group_id",
                "click",
                "click_mask",
                "item_id",
                "cate_id",
                "price",
                "hist_item_id",
                "hist_cate_id",
                "hist_price",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scenario_id": 0,
                "group_id": "q1",
                "click": 1,
                "click_mask": 1,
                "item_id": 3,
                "cate_id": 2,
                "price": 0.2,
                "hist_item_id": "1|2|3",
                "hist_cate_id": "2|2|1",
                "hist_price": "0.1|0.2|0.3",
            }
        )

    rows = list(ManifestDataset(tmp_path, "train"))
    batch = collate_manifest_batch(rows)

    assert batch["features"]["item_id"].tolist() == [3]
    assert batch["features"]["hist_item_id"]["values"].tolist() == [[1, 2, 3]]
    assert batch["features"]["hist_cate_id"]["values"].tolist() == [[2, 2, 1]]
    assert batch["features"]["hist_price"]["values"].shape == (1, 3)



def test_manifest_dataset_collate_multifield_sequence_mean_pooling_sources(tmp_path: Path) -> None:
    manifest = {
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
                    "name": "history_behavior",
                    "encoder": "sequence_mean_pooling",
                    "fusion": "concat",
                    "sequence_features": [
                        {
                            "name": "hist_item_id",
                            "encoder": "embedding",
                            "vocab_size": 20,
                            "source": {
                                "type": "csv_column",
                                "column": "hist_item_id",
                                "dtype": "int64",
                                "shape": "sequence",
                                "delimiter": "|",
                            },
                        },
                        {
                            "name": "hist_price",
                            "encoder": "identity",
                            "dim": 1,
                            "projection_dim": 2,
                            "source": {
                                "type": "csv_column",
                                "column": "hist_price",
                                "dtype": "float32",
                                "shape": "sequence",
                                "delimiter": "|",
                            },
                        },
                    ],
                },
            ],
            "token_specs": [
                {"token_id": 0, "projection": "linear", "inputs": ["history_behavior"]},
            ],
        },
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (tmp_path / "train.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario_id",
                "group_id",
                "click",
                "click_mask",
                "hist_item_id",
                "hist_price",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scenario_id": 0,
                "group_id": "q1",
                "click": 1,
                "click_mask": 1,
                "hist_item_id": "1|2|3",
                "hist_price": "0.1|0.2|0.3",
            }
        )

    rows = list(ManifestDataset(tmp_path, "train"))
    batch = collate_manifest_batch(rows)

    assert batch["features"]["hist_item_id"]["values"].tolist() == [[1, 2, 3]]
    assert batch["features"]["hist_price"]["values"].shape == (1, 3)
