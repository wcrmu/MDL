from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.datasets import ManifestDataset, collate_manifest_batch, load_manifest
from src.models import ModelFromManifest, config_from_manifest
from src.modules import multitask_bce_loss
from src.utils.checkpoint import save_checkpoint

from .evaluator import EvaluationResult, evaluate_model, move_batch


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: str
    epochs: int = 1
    batch_size: int = 2048
    max_steps: int | None = None
    eval_max_batches: int | None = 100
    device: str = "cpu"
    lr: float = 1e-3
    embedding_dim: int = 32
    token_dim: int = 36
    feature_backbone: str = "rankmixer"
    num_layers: int = 2
    num_heads: int = 4
    ffn_hidden_dim: int = 64
    dropout: float = 0.0
    checkpoint_path: str | None = None


class Trainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.manifest = load_manifest(config.data_dir)
        model_config = config_from_manifest(
            self.manifest,
            embedding_dim=config.embedding_dim,
            token_dim=config.token_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_hidden_dim=config.ffn_hidden_dim,
            dropout=config.dropout,
            feature_backbone=config.feature_backbone,
        )
        self.model_config = model_config
        self.model = ModelFromManifest(model_config).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=config.lr)

    def train(self) -> list[EvaluationResult]:
        global_step = 0
        eval_results: list[EvaluationResult] = []
        for epoch in range(1, self.config.epochs + 1):
            dataset = ManifestDataset(self.config.data_dir, "train")
            loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                collate_fn=collate_manifest_batch,
            )
            self.model.train()
            for batch in loader:
                batch = move_batch(batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                output = self.model(batch["features"], batch["scenario_id"])
                logits = output["logits"]
                if not isinstance(logits, Tensor):
                    raise TypeError("model output logits must be a tensor")
                loss = multitask_bce_loss(logits, batch["labels"], batch["label_mask"])
                loss.backward()
                self.optimizer.step()
                global_step += 1
                print(f"epoch={epoch} step={global_step} loss={loss.item():.6f}")
                if self.config.max_steps is not None and global_step >= self.config.max_steps:
                    break

            if "val" in self.manifest.get("splits", []):
                result = evaluate_model(
                    self.model,
                    self.config.data_dir,
                    "val",
                    self.manifest,
                    self.config.batch_size,
                    self.device,
                    self.config.eval_max_batches,
                )
                eval_results.append(result)
                for line in result.format_lines():
                    print(line)

            if self.config.max_steps is not None and global_step >= self.config.max_steps:
                break

        if self.config.checkpoint_path:
            save_checkpoint(
                Path(self.config.checkpoint_path),
                model=self.model,
                model_config=self.model_config,
                manifest=self.manifest,
            )
        return eval_results
