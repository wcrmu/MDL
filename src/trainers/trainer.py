from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.datasets import ManifestDataset, collate_manifest_batch, load_manifest
from src.datasets.preprocess import validate_processed_dataset
from src.models import build_model_config_from_manifest, build_model_from_config
from src.modules import multitask_bce_loss
from src.utils.checkpoint import save_checkpoint
from src.utils.formatting import format_table

from .evaluator import EvaluationResult, evaluate_model, move_batch


@dataclass(frozen=True)
class DatasetOverview:
    split: str
    task_names: list[str]
    scenario_names: list[str]
    sample_count: int
    scenario_counts: list[int]
    task_valid_counts: list[int]
    task_positive_counts: list[int]
    scenario_task_valid_counts: list[list[int]]
    scenario_task_positive_counts: list[list[int]]

    @property
    def scenario_assignments(self) -> int:
        return sum(self.scenario_counts)

    def positive_class_weights(self) -> list[float]:
        weights = []
        for valid_count, positive_count in zip(self.task_valid_counts, self.task_positive_counts):
            negative_count = valid_count - positive_count
            if positive_count <= 0 or negative_count <= 0:
                weights.append(1.0)
            else:
                weights.append(negative_count / positive_count)
        return weights

    def format_lines(self) -> list[str]:
        lines = [f"Dataset overview: {self.split}", f"samples: {self.sample_count}"]
        if self.scenario_assignments != self.sample_count:
            lines.append(f"scenario_assignments: {self.scenario_assignments}")
        lines.extend(
            [
                "",
                "Task label distribution",
                *format_table(
                    ["task", "valid", "positive", "positive_rate"],
                    [
                        [
                            task_name,
                            str(valid_count),
                            str(positive_count),
                            _format_rate(positive_count, valid_count),
                        ]
                        for task_name, valid_count, positive_count in zip(
                            self.task_names,
                            self.task_valid_counts,
                            self.task_positive_counts,
                        )
                    ],
                ),
                "",
                "Scenario label distribution",
                *format_table(
                    ["scenario", "samples", "sample_rate", "task", "valid", "positive", "positive_rate"],
                    self._scenario_rows(),
                ),
            ]
        )
        return lines

    def _scenario_rows(self) -> list[list[str]]:
        rows = []
        for scenario_index, scenario_name in enumerate(self.scenario_names):
            scenario_count = self.scenario_counts[scenario_index]
            for task_index, task_name in enumerate(self.task_names):
                valid_count = self.scenario_task_valid_counts[scenario_index][task_index]
                positive_count = self.scenario_task_positive_counts[scenario_index][task_index]
                rows.append(
                    [
                        scenario_name,
                        str(scenario_count),
                        _format_rate(scenario_count, self.sample_count),
                        task_name,
                        str(valid_count),
                        str(positive_count),
                        _format_rate(positive_count, valid_count),
                    ]
                )
        return rows


def _format_rate(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "nan"
    return f"{numerator / denominator:.4f}"


def _format_lr(value: float | None) -> str:
    return f"{value:.6g}" if value is not None else "none"


def _parse_scenario_indices(
    row: dict[str, str],
    data_columns: dict[str, Any],
    num_scenarios: int,
) -> list[int]:
    if "scenario_ids" in data_columns:
        delimiter = str(data_columns.get("scenario_ids_delimiter", "|"))
        value = row[str(data_columns["scenario_ids"])]
        scenario_indices = [int(part.strip()) for part in value.split(delimiter) if part.strip() != ""]
        if not scenario_indices:
            raise ValueError("scenario_ids column must contain at least one scenario id")
    else:
        scenario_indices = [int(row[str(data_columns["scenario_id"])])]
    for scenario_index in scenario_indices:
        if scenario_index < 0 or scenario_index >= num_scenarios:
            raise ValueError(f"scenario_id {scenario_index} out of range")
    return scenario_indices


def build_dataset_overview(data_dir: str | Path, manifest: dict[str, Any], split: str) -> DatasetOverview:
    task_names = list(manifest["task_names"])
    scenario_names = list(manifest["scenario_names"])
    data_columns = manifest["data_columns"]
    label_columns = data_columns["labels"]
    label_mask_columns = data_columns["label_masks"]
    sample_count = 0
    scenario_counts = [0 for _ in scenario_names]
    task_valid_counts = [0 for _ in task_names]
    task_positive_counts = [0 for _ in task_names]
    scenario_task_valid_counts = [[0 for _ in task_names] for _ in scenario_names]
    scenario_task_positive_counts = [[0 for _ in task_names] for _ in scenario_names]

    path = Path(data_dir) / f"{split}.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_count += 1
            scenario_indices = _parse_scenario_indices(row, data_columns, len(scenario_names))
            for scenario_index in scenario_indices:
                scenario_counts[scenario_index] += 1

            for task_index, task_name in enumerate(task_names):
                label_mask = float(row[str(label_mask_columns[task_name])])
                if label_mask <= 0:
                    continue
                is_positive = float(row[str(label_columns[task_name])]) > 0.5
                task_valid_counts[task_index] += 1
                if is_positive:
                    task_positive_counts[task_index] += 1
                for scenario_index in scenario_indices:
                    scenario_task_valid_counts[scenario_index][task_index] += 1
                    if is_positive:
                        scenario_task_positive_counts[scenario_index][task_index] += 1

    return DatasetOverview(
        split=split,
        task_names=task_names,
        scenario_names=scenario_names,
        sample_count=sample_count,
        scenario_counts=scenario_counts,
        task_valid_counts=task_valid_counts,
        task_positive_counts=task_positive_counts,
        scenario_task_valid_counts=scenario_task_valid_counts,
        scenario_task_positive_counts=scenario_task_positive_counts,
    )


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: str
    epochs: int = 1
    batch_size: int = 2048
    max_steps: int | None = None
    eval_max_batches: int | None = 100
    device: str = "cpu"
    model_name: str = "mdl"
    lr: float = 1e-3
    sparse_lr: float | None = None
    lr_scheduler: str = "none"
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0
    dense_weight_decay: float = 0.0
    gradient_clip_norm: float | None = None
    task_weights: list[float] | None = None
    scenario_weights: list[float] | None = None
    positive_class_weights: list[float] | None = None
    auto_positive_class_weights: bool = False
    validate_data: bool = True
    validation_max_rows: int | None = 1000
    embedding_dim: int = 32
    token_dim: int = 36
    feature_backbone: str = "rankmixer"
    num_layers: int = 2
    num_heads: int = 4
    ffn_hidden_dim: int = 64
    dropout: float = 0.0
    task_head_type: str = "linear"
    task_head_hidden_dim: int = 64
    task_head_dropout: float = 0.0
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
        if self.model_name not in {"mdl", "rankmixer"}:
            raise ValueError("model_name must be 'mdl' or 'rankmixer'")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.eval_max_batches is not None and self.eval_max_batches < 0:
            raise ValueError("eval_max_batches must be non-negative")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.sparse_lr is not None and self.sparse_lr <= 0:
            raise ValueError("sparse_lr must be positive")
        if self.lr_scheduler not in {"none", "linear", "cosine"}:
            raise ValueError("lr_scheduler must be one of: none, linear, cosine")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.lr_scheduler == "none" and self.warmup_steps > 0:
            raise ValueError("warmup_steps requires lr_scheduler to be linear or cosine")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.dense_weight_decay < 0:
            raise ValueError("dense_weight_decay must be non-negative")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.task_weights is not None and any(weight < 0 for weight in self.task_weights):
            raise ValueError("task_weights must be non-negative")
        if self.scenario_weights is not None and any(weight < 0 for weight in self.scenario_weights):
            raise ValueError("scenario_weights must be non-negative")
        if self.positive_class_weights is not None and any(
            weight < 0 for weight in self.positive_class_weights
        ):
            raise ValueError("positive_class_weights must be non-negative")
        if self.auto_positive_class_weights and self.positive_class_weights is not None:
            raise ValueError(
                "positive_class_weights and auto_positive_class_weights must not both be set"
            )
        if self.validation_max_rows is not None and self.validation_max_rows < 0:
            raise ValueError("validation_max_rows must be non-negative")
        if self.task_head_type not in {"linear", "mlp"}:
            raise ValueError("task_head_type must be 'linear' or 'mlp'")
        if self.task_head_hidden_dim <= 0:
            raise ValueError("task_head_hidden_dim must be positive")
        if self.task_head_dropout < 0:
            raise ValueError("task_head_dropout must be non-negative")
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


def _estimate_total_steps(
    sample_count: int,
    batch_size: int,
    epochs: int,
    max_steps: int | None,
) -> int:
    steps_per_epoch = math.ceil(sample_count / batch_size) if sample_count > 0 else 0
    total_steps = steps_per_epoch * epochs
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    return max(total_steps, 1)


class Trainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        if config.validate_data:
            validate_processed_dataset(
                config.data_dir,
                max_rows=config.validation_max_rows,
                require_domain_tokenization=config.model_name == "mdl",
            )
        self.manifest = load_manifest(config.data_dir)
        self.train_overview = build_dataset_overview(config.data_dir, self.manifest, "train")
        model_config = build_model_config_from_manifest(
            self.manifest,
            model_name=config.model_name,
            embedding_dim=config.embedding_dim,
            token_dim=config.token_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_hidden_dim=config.ffn_hidden_dim,
            dropout=config.dropout,
            task_head_type=config.task_head_type,
            task_head_hidden_dim=config.task_head_hidden_dim,
            task_head_dropout=config.task_head_dropout,
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
        positive_class_weights = (
            self.train_overview.positive_class_weights()
            if config.auto_positive_class_weights
            else config.positive_class_weights
        )
        self.positive_class_weights = _optional_weight_tensor(
            "positive_class_weights",
            positive_class_weights,
            len(self.manifest["task_names"]),
            self.device,
        )
        self.model = build_model_from_config(model_config).to(self.device)
        dense_parameters, embedding_parameters = _partition_embedding_parameters(self.model)
        self.dense_optimizer = (
            torch.optim.RMSprop(
                dense_parameters,
                lr=config.lr,
                weight_decay=config.dense_weight_decay,
            )
            if dense_parameters
            else None
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
        self.optimizer_base_lrs = [
            (optimizer, [float(group["lr"]) for group in optimizer.param_groups])
            for optimizer in self.optimizers
        ]
        self.total_steps = _estimate_total_steps(
            self.train_overview.sample_count,
            config.batch_size,
            config.epochs,
            config.max_steps,
        )
        self.sparse_moe_loss_weight = config.sparse_moe_loss_weight

    def _zero_grad(self) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def _step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def _clip_gradients(self) -> Tensor | None:
        if self.config.gradient_clip_norm is None:
            return None
        return nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.gradient_clip_norm,
        )

    def _lr_multiplier(self, step: int) -> float:
        if self.config.lr_scheduler == "none":
            return 1.0
        if self.config.warmup_steps > 0 and step <= self.config.warmup_steps:
            return step / self.config.warmup_steps

        decay_steps = max(self.total_steps - self.config.warmup_steps, 1)
        progress = (step - self.config.warmup_steps) / decay_steps
        progress = min(max(progress, 0.0), 1.0)
        if self.config.lr_scheduler == "linear":
            return self.config.min_lr_ratio + (1.0 - self.config.min_lr_ratio) * (1.0 - progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_lr_ratio + (1.0 - self.config.min_lr_ratio) * cosine

    def _set_learning_rates(self, step: int) -> None:
        if self.config.lr_scheduler == "none":
            return
        multiplier = self._lr_multiplier(step)
        for optimizer, base_lrs in self.optimizer_base_lrs:
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * multiplier

    def _optimizer_lr(self, optimizer: torch.optim.Optimizer | None) -> float | None:
        if optimizer is None or not optimizer.param_groups:
            return None
        return float(optimizer.param_groups[0]["lr"])

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

    def _print_dataset_overview(self) -> None:
        for line in self.train_overview.format_lines():
            print(line)
        print("")

    def _print_train_step(
        self,
        epoch: int,
        global_step: int,
        loss: Tensor,
        gradient_norm: Tensor | None,
        output: dict[str, Any],
    ) -> None:
        parts = [
            f"epoch={epoch}/{self.config.epochs}",
            f"step={global_step}/{self.total_steps}",
            f"loss={loss.item():.6f}",
            f"dense_lr={_format_lr(self._optimizer_lr(self.dense_optimizer))}",
            f"sparse_lr={_format_lr(self._optimizer_lr(self.sparse_optimizer))}",
        ]
        if isinstance(gradient_norm, Tensor):
            grad_norm_value = float(gradient_norm.detach().cpu().item())
            parts.append(f"grad_norm={grad_norm_value:.6f}")
        moe_active_ratio = output.get("moe_active_ratio")
        if isinstance(moe_active_ratio, Tensor):
            active_ratio = float(moe_active_ratio.detach().cpu().item())
            parts.append(f"moe_active_ratio={active_ratio:.6f}")
            parts.append(f"moe_loss_weight={self.sparse_moe_loss_weight:.6g}")
        print("Train step | " + " | ".join(parts))

    def train(self) -> list[EvaluationResult]:
        global_step = 0
        eval_results: list[EvaluationResult] = []
        self._print_dataset_overview()
        print("Training")
        for epoch in range(1, self.config.epochs + 1):
            dataset = ManifestDataset(self.config.data_dir, "train")
            loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                collate_fn=collate_manifest_batch,
            )
            self.model.train()
            for batch in loader:
                next_step = global_step + 1
                self._set_learning_rates(next_step)
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
                    positive_class_weights=self.positive_class_weights,
                )
                moe_regularization_loss = output.get("moe_regularization_loss")
                if (
                    self.model_config.ffn_type == "sparse_moe"
                    and self.sparse_moe_loss_weight > 0
                    and isinstance(moe_regularization_loss, Tensor)
                ):
                    loss = loss + self.sparse_moe_loss_weight * moe_regularization_loss
                loss.backward()
                gradient_norm = self._clip_gradients()
                self._step()
                self._update_sparse_moe_loss_weight(output)
                global_step = next_step
                self._print_train_step(epoch, global_step, loss, gradient_norm, output)
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
                    positive_class_weights=self.positive_class_weights,
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
