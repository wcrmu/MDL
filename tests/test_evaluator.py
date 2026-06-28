from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch import Tensor, nn

from src.trainers.evaluator import evaluate_model, scenario_membership


class ScoreModel(nn.Module):
    def forward(self, features: dict[str, Tensor], scenario_id: Tensor) -> dict[str, Tensor]:
        score = features["score"].to(dtype=torch.float32).view(-1, 1)
        return {"logits": torch.cat([score, -score], dim=1)}


def test_evaluate_model_reports_scenario_task_metrics(tmp_path: Path) -> None:
    manifest = {
        "splits": ["val"],
        "scenario_names": ["home", "search"],
        "task_names": ["click", "like"],
        "data_columns": {
            "scenario_id": "scenario_id",
            "group_id": "group_id",
            "sample_weight": "sample_weight",
            "labels": {"click": "click", "like": "like"},
            "label_masks": {"click": "click_mask", "like": "like_mask"},
        },
        "tokenization": {
            "version": 2,
            "kind": "encoder_registry",
            "features": [
                {
                    "name": "score",
                    "encoder": "identity",
                    "dim": 1,
                    "source": {"type": "csv_column", "column": "score", "dtype": "float32"},
                }
            ],
            "token_specs": [{"token_id": 0, "projection": "linear", "inputs": ["score"]}],
            "scenario_features": [{"name": "score", "encoder": "identity", "dim": 1}],
            "scenario_token_specs": [
                {"token_id": 0, "inputs": ["score"]},
                {"token_id": 1, "inputs": ["score"]},
                {"token_id": 2, "inputs": ["score"]},
            ],
            "task_features": [{"name": "score", "encoder": "identity", "dim": 1}],
            "task_token_specs": [
                {"token_id": 0, "inputs": ["score"]},
                {"token_id": 1, "inputs": ["score"]},
            ],
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (tmp_path / "val.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario_id",
                "group_id",
                "sample_weight",
                "click",
                "click_mask",
                "like",
                "like_mask",
                "score",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scenario_id": 0,
                "group_id": "g1",
                "sample_weight": 1.0,
                "click": 0,
                "click_mask": 1,
                "like": 1,
                "like_mask": 1,
                "score": -1.0,
            }
        )
        writer.writerow(
            {
                "scenario_id": 0,
                "group_id": "g1",
                "sample_weight": 1.0,
                "click": 1,
                "click_mask": 1,
                "like": 0,
                "like_mask": 1,
                "score": 1.0,
            }
        )
        writer.writerow(
            {
                "scenario_id": 1,
                "group_id": "g2",
                "sample_weight": 2.0,
                "click": 0,
                "click_mask": 1,
                "like": 1,
                "like_mask": 1,
                "score": 0.25,
            }
        )

    result = evaluate_model(
        ScoreModel(),
        str(tmp_path),
        "val",
        manifest,
        batch_size=2,
        device=torch.device("cpu"),
        max_batches=None,
        task_weights=torch.tensor([1.0, 2.0]),
        scenario_weights=torch.tensor([1.0, 0.5]),
    )

    assert len(result.task_metrics) == 2
    assert {(metric.scenario_name, metric.task_name) for metric in result.scenario_task_metrics} == {
        ("home", "click"),
        ("home", "like"),
        ("search", "click"),
        ("search", "like"),
    }
    formatted = "\n".join(result.format_lines())
    assert "val_home_click_auc=" in formatted
    assert "val_search_like_qauc=" in formatted


def test_scenario_membership_accepts_index_and_multihot() -> None:
    index_membership = scenario_membership(torch.tensor([0, 2]), 3)
    multihot_membership = scenario_membership(torch.tensor([[1.0, 0.0, 1.0]]), 3)

    assert index_membership.tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    assert multihot_membership.tolist() == [[1.0, 0.0, 1.0]]
