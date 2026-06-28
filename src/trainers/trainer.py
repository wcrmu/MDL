from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.datasets import ManifestDataset, collate_manifest_batch, load_manifest
from src.datasets.preprocess import validate_processed_dataset
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
    sparse_lr: float | None = None
    task_weights: list[float] | None = None
    scenario_weights: list[float] | None = None
    validate_data: bool = True
    validation_max_rows: int | None = 1000
    embedding_dim: int = 32
    token_dim: int = 36
    feature_backbone: str = "rankmixer"
    num_layers: int = 2
    num_heads: int = 4
    ffn_hidden_dim: int = 64
    dropout: float = 0.0
    ffn_type: str = "dense"
    sparse_moe_num_experts: int = 4
    sparse_moe_loss_weight: float = 0.0
    sparse_moe_target_active_ratio: float | None = None
    sparse_moe_loss_weight_update_rate: float = 0.05
    sparse_moe_loss_weight_min: float = 0.0
    sparse_moe_loss_weight_max: float | None = 1.0
    sparse_moe_use_dtsi: bool = True
    sparse_moe_dtsi_infer_weight: float = 0.5
    sparse_moe_inference_threshold: float = 0.0
    use_task_tokens: bool = True
    use_scenario_tokens: bool = True
    use_global_scenario_token: bool = True
    use_task_feature_interaction: bool = True
    use_scenario_feature_interaction: bool = True
    checkpoint_path: str | None = None

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.sparse_lr is not None and self.sparse_lr <= 0:
            raise ValueError("sparse_lr must be positive")
        if self.task_weights is not None and any(weight < 0 for weight in self.task_weights):
            raise ValueError("task_weights must be non-negative")
        if self.scenario_weights is not None and any(weight < 0 for weight in self.scenario_weights):
            raise ValueError("scenario_weights must be non-negative")
        if self.validation_max_rows is not None and self.validation_max_rows < 0:
            raise ValueError("validation_max_rows must be non-negative")
        if self.sparse_moe_loss_weight < 0:
            raise ValueError("sparse_moe_loss_weight must be non-negative")
        if self.sparse_moe_target_active_ratio is not None:
            if not 0.0 < self.sparse_moe_target_active_ratio <= 1.0:
                raise ValueError("sparse_moe_target_active_ratio must be in (0, 1]")
            if self.sparse_moe_loss_weight <= 0:
                raise ValueError(
                    "sparse_moe_loss_weight must be positive when "
                    "sparse_moe_target_active_ratio is set"
                )
        if self.sparse_moe_loss_weight_update_rate < 0:
            raise ValueError("sparse_moe_loss_weight_update_rate must be non-negative")
        if not 0.0 <= self.sparse_moe_dtsi_infer_weight <= 1.0:
            raise ValueError("sparse_moe_dtsi_infer_weight must be in [0, 1]")
        if self.sparse_moe_loss_weight_min < 0:
            raise ValueError("sparse_moe_loss_weight_min must be non-negative")
        if (
            self.sparse_moe_loss_weight_max is not None
            and self.sparse_moe_loss_weight_max < self.sparse_moe_loss_weight_min
        ):
            raise ValueError("sparse_moe_loss_weight_max must be >= sparse_moe_loss_weight_min")


def _optional_weight_tensor(
    name: str,
    values: list[float] | None,
    expected_count: int,
    device: torch.device,
) -> Tensor | None:
    if values is None:
        return None
    if len(values) != expected_count:
        raise ValueError(f"{name} must contain {expected_count} values")
    return torch.tensor(values, dtype=torch.float32, device=device)


def _scenario_weights_for_batch(
    scenario_id: Tensor,
    scenario_weights: Tensor,
    num_scenarios: int,
) -> Tensor:
    if scenario_id.ndim == 1:
        indices = scenario_id.to(device=scenario_weights.device, dtype=torch.long)
        if indices.numel() > 0:
            if int(indices.min().item()) < 0 or int(indices.max().item()) >= num_scenarios:
                raise ValueError("scenario_id out of range")
        return scenario_weights[indices]

    if scenario_id.ndim != 2:
        raise ValueError("scenario_id must have shape [batch], [batch, num_scenarios], or [batch, k]")

    if scenario_id.size(1) == num_scenarios:
        mask = scenario_id.to(device=scenario_weights.device, dtype=scenario_weights.dtype)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return (mask * scenario_weights.view(1, -1)).sum(dim=1) / denominator

    indices = scenario_id.to(device=scenario_weights.device, dtype=torch.long)
    if indices.numel() > 0:
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= num_scenarios:
            raise ValueError("scenario_id out of range")
    return scenario_weights[indices].mean(dim=1)


def _partition_embedding_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    embedding_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    dense_parameters = []
    embedding_parameters = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in embedding_parameter_ids:
            embedding_parameters.append(parameter)
        else:
            dense_parameters.append(parameter)
    return dense_parameters, embedding_parameters


class Trainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        if config.validate_data:
            validate_processed_dataset(config.data_dir, max_rows=config.validation_max_rows)
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
            ffn_type=config.ffn_type,
            sparse_moe_num_experts=config.sparse_moe_num_experts,
            sparse_moe_loss_weight=config.sparse_moe_loss_weight,
            sparse_moe_use_dtsi=config.sparse_moe_use_dtsi,
            sparse_moe_dtsi_infer_weight=config.sparse_moe_dtsi_infer_weight,
            sparse_moe_inference_threshold=config.sparse_moe_inference_threshold,
            use_task_tokens=config.use_task_tokens,
            use_scenario_tokens=config.use_scenario_tokens,
            use_global_scenario_token=config.use_global_scenario_token,
            use_task_feature_interaction=config.use_task_feature_interaction,
            use_scenario_feature_interaction=config.use_scenario_feature_interaction,
        )
        self.model_config = model_config
        self.task_weights = _optional_weight_tensor(
            "task_weights",
            config.task_weights,
            len(self.manifest["task_names"]),
            self.device,
        )
        self.scenario_weights = _optional_weight_tensor(
            "scenario_weights",
            config.scenario_weights,
            len(self.manifest["scenario_names"]),
            self.device,
        )
        self.model = ModelFromManifest(model_config).to(self.device)
        dense_parameters, embedding_parameters = _partition_embedding_parameters(self.model)
        self.dense_optimizer = (
            torch.optim.RMSprop(dense_parameters, lr=config.lr) if dense_parameters else None
        )
        sparse_lr = config.sparse_lr if config.sparse_lr is not None else config.lr
        self.sparse_optimizer = (
            torch.optim.Adagrad(embedding_parameters, lr=sparse_lr)
            if embedding_parameters
            else None
        )
        self.optimizers = [
            optimizer
            for optimizer in (self.dense_optimizer, self.sparse_optimizer)
            if optimizer is not None
        ]
        self.sparse_moe_loss_weight = config.sparse_moe_loss_weight

    def _zero_grad(self) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def _step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def _update_sparse_moe_loss_weight(self, output: dict[str, Any]) -> None:
        target = self.config.sparse_moe_target_active_ratio
        if target is None or self.config.ffn_type != "sparse_moe":
            return
        active_ratio_tensor = output.get("moe_active_ratio")
        if not isinstance(active_ratio_tensor, Tensor):
            return
        active_ratio = float(active_ratio_tensor.detach().cpu().item())
        error = (active_ratio - target) / target
        update_factor = math.exp(self.config.sparse_moe_loss_weight_update_rate * error)
        next_weight = self.sparse_moe_loss_weight * update_factor
        next_weight = max(self.config.sparse_moe_loss_weight_min, next_weight)
        if self.config.sparse_moe_loss_weight_max is not None:
            next_weight = min(self.config.sparse_moe_loss_weight_max, next_weight)
        self.sparse_moe_loss_weight = next_weight

    def _batch_sample_weights(self, batch: dict[str, Any]) -> Tensor | None:
        sample_weights = batch.get("sample_weight")
        if not isinstance(sample_weights, Tensor) and self.scenario_weights is None:
            return None
        if isinstance(sample_weights, Tensor):
            weights = sample_weights.to(device=self.device, dtype=torch.float32)
        else:
            weights = torch.ones(batch["labels"].size(0), device=self.device, dtype=torch.float32)
        if self.scenario_weights is not None:
            weights = weights * _scenario_weights_for_batch(
                batch["scenario_id"],
                self.scenario_weights,
                len(self.manifest["scenario_names"]),
            )
        return weights

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
                self._zero_grad()
                output = self.model(batch["features"], batch["scenario_id"])
                logits = output["logits"]
                if not isinstance(logits, Tensor):
                    raise TypeError("model output logits must be a tensor")
                sample_weights = self._batch_sample_weights(batch)
                loss = multitask_bce_loss(
                    logits,
                    batch["labels"],
                    batch["label_mask"],
                    task_weights=self.task_weights,
                    sample_weights=sample_weights,
                )
                moe_regularization_loss = output.get("moe_regularization_loss")
                if (
                    self.model_config.ffn_type == "sparse_moe"
                    and self.sparse_moe_loss_weight > 0
                    and isinstance(moe_regularization_loss, Tensor)
                ):
                    loss = loss + self.sparse_moe_loss_weight * moe_regularization_loss
                loss.backward()
                self._step()
                self._update_sparse_moe_loss_weight(output)
                global_step += 1
                log_parts = [f"epoch={epoch}", f"step={global_step}", f"loss={loss.item():.6f}"]
                moe_active_ratio = output.get("moe_active_ratio")
                if isinstance(moe_active_ratio, Tensor):
                    active_ratio = float(moe_active_ratio.detach().cpu().item())
                    log_parts.append(f"moe_active_ratio={active_ratio:.6f}")
                    log_parts.append(f"moe_loss_weight={self.sparse_moe_loss_weight:.6g}")
                print(" ".join(log_parts))
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
                    task_weights=self.task_weights,
                    scenario_weights=self.scenario_weights,
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
