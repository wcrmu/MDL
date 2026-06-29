from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from src.models import RankMixerConfig, build_model_from_config
from src.trainers import Trainer, TrainingConfig
from src.utils import load_checkpoint


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


def test_training_config_rejects_invalid_learning_strategy_options() -> None:
    with pytest.raises(ValueError, match="warmup_steps requires lr_scheduler"):
        TrainingConfig(data_dir="unused", lr_scheduler="none", warmup_steps=1)
    with pytest.raises(ValueError, match="dense_weight_decay must be non-negative"):
        TrainingConfig(data_dir="unused", dense_weight_decay=-0.1)
    with pytest.raises(ValueError, match="must not both be set"):
        TrainingConfig(
            data_dir="unused",
            positive_class_weights=[1.0],
            auto_positive_class_weights=True,
        )


def test_trainer_rankmixer_feature_only_single_step_and_checkpoint(tmp_path: Path) -> None:
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

    checkpoint_path = tmp_path / "rankmixer.pt"
    trainer = Trainer(
        TrainingConfig(
            data_dir=str(tmp_path),
            model_name="rankmixer",
            batch_size=2,
            max_steps=1,
            embedding_dim=4,
            token_dim=8,
            num_layers=1,
            ffn_hidden_dim=8,
            checkpoint_path=str(checkpoint_path),
        )
    )
    assert isinstance(trainer.model_config, RankMixerConfig)
    assert trainer.sparse_optimizer is not None

    trainer.train()

    checkpoint = load_checkpoint(checkpoint_path)
    assert isinstance(checkpoint["model_config"], RankMixerConfig)
    model = build_model_from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])


def test_trainer_scheduler_weight_decay_positive_weights_and_overview(tmp_path: Path) -> None:
    manifest = {
        "splits": ["train"],
        "scenario_names": ["default"],
        "task_names": ["click"],
        "data_columns": {
            "scenario_id": "scenario_id",
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
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (tmp_path / "train.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario_id", "click", "click_mask", "user_id", "score"],
        )
        writer.writeheader()
        for index, click in enumerate([1, 0, 0, 0], start=1):
            writer.writerow(
                {
                    "scenario_id": 0,
                    "click": click,
                    "click_mask": 1,
                    "user_id": index,
                    "score": 0.1 * index,
                }
            )

    trainer = Trainer(
        TrainingConfig(
            data_dir=str(tmp_path),
            model_name="rankmixer",
            batch_size=1,
            max_steps=3,
            lr=1e-3,
            lr_scheduler="linear",
            warmup_steps=1,
            min_lr_ratio=0.1,
            dense_weight_decay=0.01,
            auto_positive_class_weights=True,
            embedding_dim=4,
            token_dim=8,
            num_layers=1,
            ffn_hidden_dim=8,
        )
    )

    assert trainer.train_overview.sample_count == 4
    assert trainer.train_overview.scenario_counts == [4]
    assert trainer.train_overview.task_positive_counts == [1]
    assert trainer.positive_class_weights is not None
    assert trainer.positive_class_weights.tolist() == [3.0]
    assert trainer.dense_optimizer is not None
    assert trainer.dense_optimizer.param_groups[0]["weight_decay"] == 0.01

    overview_text = "\n".join(trainer.train_overview.format_lines())
    assert "Dataset overview: train" in overview_text
    assert "positive_rate" in overview_text

    trainer._set_learning_rates(2)
    assert trainer.dense_optimizer.param_groups[0]["lr"] == pytest.approx(0.00055)
