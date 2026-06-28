from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from src.trainers import Trainer, TrainingConfig


def test_trainer_sparse_moe_single_step(tmp_path: Path) -> None:
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
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (tmp_path / "train.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario_id", "group_id", "click", "click_mask", "user_id", "score"],
        )
        writer.writeheader()
        writer.writerow(
            {"scenario_id": 0, "group_id": "g1", "click": 1, "click_mask": 1, "user_id": 1, "score": 0.2}
        )
        writer.writerow(
            {"scenario_id": 0, "group_id": "g2", "click": 0, "click_mask": 1, "user_id": 2, "score": 0.4}
        )

    trainer = Trainer(
        TrainingConfig(
            data_dir=str(tmp_path),
            batch_size=2,
            max_steps=1,
            embedding_dim=8,
            token_dim=16,
            num_layers=1,
            num_heads=4,
            ffn_hidden_dim=16,
            ffn_type="sparse_moe",
            sparse_moe_num_experts=2,
            sparse_moe_loss_weight=0.01,
            gradient_clip_norm=0.1,
            sparse_moe_target_active_ratio=0.99,
            sparse_moe_loss_weight_update_rate=0.5,
        )
    )
    assert trainer.dense_optimizer is not None
    assert trainer.dense_optimizer.__class__.__name__ == "RMSprop"
    assert trainer.sparse_optimizer is not None
    assert trainer.sparse_optimizer.__class__.__name__ == "Adagrad"

    for parameter in trainer.model.parameters():
        parameter.grad = torch.ones_like(parameter)
    pre_clip_norm = trainer._clip_gradients()
    assert pre_clip_norm is not None
    assert float(pre_clip_norm) > trainer.config.gradient_clip_norm
    post_clip_terms = [
        parameter.grad.detach().float().norm().square()
        for parameter in trainer.model.parameters()
        if parameter.grad is not None
    ]
    post_clip_norm = torch.stack(post_clip_terms).sum().sqrt()
    assert float(post_clip_norm) <= trainer.config.gradient_clip_norm + 1e-5
    trainer._zero_grad()

    trainer.train()

    assert trainer.sparse_moe_loss_weight != 0.01


def test_training_config_rejects_invalid_gradient_clip_norm() -> None:
    with pytest.raises(ValueError, match="gradient_clip_norm must be positive"):
        TrainingConfig(data_dir="unused", gradient_clip_norm=0.0)
