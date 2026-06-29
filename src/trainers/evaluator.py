from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.datasets import ManifestDataset, collate_manifest_batch
from src.modules import binary_auc, multitask_bce_loss, qauc


@dataclass(frozen=True)
class TaskMetric:
    task_name: str
    auc: float | None
    qauc: float
    valid_groups: int
    skipped_groups: int


@dataclass(frozen=True)
class ScenarioTaskMetric:
    scenario_name: str
    task_name: str
    auc: float | None
    qauc: float
    valid_groups: int
    skipped_groups: int


@dataclass(frozen=True)
class EvaluationResult:
    split: str
    loss: float
    task_metrics: list[TaskMetric]
    scenario_task_metrics: list[ScenarioTaskMetric]

    def format_lines(self) -> list[str]:
        loss_text = f"{self.loss:.6f}" if not math.isnan(self.loss) else "nan"
        lines = [f"{self.split}_loss={loss_text}"]
        for metric in self.task_metrics:
            auc_text = f"{metric.auc:.6f}" if metric.auc is not None else "nan"
            qauc_text = f"{metric.qauc:.6f}" if not math.isnan(metric.qauc) else "nan"
            lines.append(
                f"{self.split}_{metric.task_name}_auc={auc_text} "
                f"{self.split}_{metric.task_name}_qauc={qauc_text} "
                f"valid_groups={metric.valid_groups}"
            )
        for metric in self.scenario_task_metrics:
            auc_text = f"{metric.auc:.6f}" if metric.auc is not None else "nan"
            qauc_text = f"{metric.qauc:.6f}" if not math.isnan(metric.qauc) else "nan"
            lines.append(
                f"{self.split}_{metric.scenario_name}_{metric.task_name}_auc={auc_text} "
                f"{self.split}_{metric.scenario_name}_{metric.task_name}_qauc={qauc_text} "
                f"valid_groups={metric.valid_groups}"
            )
        return lines


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    def move_value(value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move_value(child) for key, child in value.items()}
        return value

    return {key: move_value(value) for key, value in batch.items()}


def scenario_membership(scenario_id: Tensor, num_scenarios: int) -> Tensor:
    if scenario_id.ndim == 1:
        indices = scenario_id.to(dtype=torch.long)
        if indices.numel() > 0:
            if int(indices.min().item()) < 0 or int(indices.max().item()) >= num_scenarios:
                raise ValueError("scenario_id out of range")
        membership = torch.zeros(
            indices.size(0),
            num_scenarios,
            dtype=torch.float32,
            device=scenario_id.device,
        )
        membership.scatter_(1, indices.view(-1, 1), 1.0)
        return membership

    if scenario_id.ndim != 2:
        raise ValueError("scenario_id must have shape [batch], [batch, num_scenarios], or [batch, k]")

    if scenario_id.size(1) == num_scenarios:
        return scenario_id.to(dtype=torch.float32)

    indices = scenario_id.to(dtype=torch.long)
    if indices.numel() > 0:
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= num_scenarios:
            raise ValueError("scenario_id out of range")
    membership = torch.zeros(
        indices.size(0),
        num_scenarios,
        dtype=torch.float32,
        device=scenario_id.device,
    )
    membership.scatter_(1, indices, 1.0)
    return membership


def _loss_sample_weights(
    batch: dict[str, Any],
    scenario_weights: Tensor | None,
    num_scenarios: int,
) -> Tensor | None:
    sample_weights = batch.get("sample_weight")
    if not isinstance(sample_weights, Tensor) and scenario_weights is None:
        return None
    if isinstance(sample_weights, Tensor):
        weights = sample_weights.to(device=batch["labels"].device, dtype=torch.float32)
    else:
        weights = torch.ones(batch["labels"].size(0), device=batch["labels"].device)
    if scenario_weights is not None:
        scenario_weights = scenario_weights.to(device=batch["labels"].device, dtype=torch.float32)
        membership = scenario_membership(batch["scenario_id"], num_scenarios).to(
            device=batch["labels"].device,
            dtype=torch.float32,
        )
        denominator = membership.sum(dim=1).clamp_min(1.0)
        scenario_factor = (membership * scenario_weights.view(1, -1)).sum(dim=1) / denominator
        weights = weights * scenario_factor
    return weights


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_dir: str,
    split: str,
    manifest: dict[str, Any],
    batch_size: int,
    device: torch.device,
    max_batches: int | None = 100,
    task_weights: Tensor | None = None,
    scenario_weights: Tensor | None = None,
) -> EvaluationResult:
    dataset = ManifestDataset(data_dir, split)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_manifest_batch)
    task_names = manifest["task_names"]
    scenario_names = manifest["scenario_names"]
    labels_by_task = [[] for _ in task_names]
    scores_by_task = [[] for _ in task_names]
    groups_by_task = [[] for _ in task_names]
    labels_by_scenario_task = [[[] for _ in task_names] for _ in scenario_names]
    scores_by_scenario_task = [[[] for _ in task_names] for _ in scenario_names]
    groups_by_scenario_task = [[[] for _ in task_names] for _ in scenario_names]
    losses: list[float] = []

    model.eval()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        output = model(batch["features"], batch["scenario_id"])
        logits = output["logits"]
        if not isinstance(logits, Tensor):
            raise TypeError("model output logits must be a tensor")
        sample_weights = _loss_sample_weights(batch, scenario_weights, len(scenario_names))
        losses.append(
            multitask_bce_loss(
                logits,
                batch["labels"],
                batch["label_mask"],
                task_weights=task_weights,
                sample_weights=sample_weights,
            ).item()
        )
        probabilities = torch.sigmoid(logits).cpu()
        labels = batch["labels"].cpu()
        masks = batch["label_mask"].cpu()
        memberships = scenario_membership(batch["scenario_id"], len(scenario_names)).cpu()
        for task_index, _task_name in enumerate(task_names):
            valid = masks[:, task_index] > 0
            labels_by_task[task_index].extend(labels[valid, task_index].tolist())
            scores_by_task[task_index].extend(probabilities[valid, task_index].tolist())
            groups_by_task[task_index].extend(
                group for group, keep in zip(batch["group_id"], valid.tolist()) if keep
            )
            for scenario_index, _scenario_name in enumerate(scenario_names):
                scenario_valid = valid & (memberships[:, scenario_index] > 0)
                labels_by_scenario_task[scenario_index][task_index].extend(
                    labels[scenario_valid, task_index].tolist()
                )
                scores_by_scenario_task[scenario_index][task_index].extend(
                    probabilities[scenario_valid, task_index].tolist()
                )
                groups_by_scenario_task[scenario_index][task_index].extend(
                    group for group, keep in zip(batch["group_id"], scenario_valid.tolist()) if keep
                )

    task_metrics = []
    for task_index, task_name in enumerate(task_names):
        labels = labels_by_task[task_index]
        scores = scores_by_task[task_index]
        if not labels:
            continue
        auc = binary_auc(labels, scores)
        qauc_result = qauc(labels, scores, groups_by_task[task_index])
        task_metrics.append(
            TaskMetric(
                task_name=task_name,
                auc=auc,
                qauc=qauc_result.qauc,
                valid_groups=qauc_result.valid_groups,
                skipped_groups=qauc_result.skipped_groups,
            )
        )

    scenario_task_metrics = []
    for scenario_index, scenario_name in enumerate(scenario_names):
        for task_index, task_name in enumerate(task_names):
            labels = labels_by_scenario_task[scenario_index][task_index]
            scores = scores_by_scenario_task[scenario_index][task_index]
            if not labels:
                continue
            auc = binary_auc(labels, scores)
            qauc_result = qauc(labels, scores, groups_by_scenario_task[scenario_index][task_index])
            scenario_task_metrics.append(
                ScenarioTaskMetric(
                    scenario_name=scenario_name,
                    task_name=task_name,
                    auc=auc,
                    qauc=qauc_result.qauc,
                    valid_groups=qauc_result.valid_groups,
                    skipped_groups=qauc_result.skipped_groups,
                )
            )

    return EvaluationResult(
        split=split,
        loss=sum(losses) / len(losses) if losses else float("nan"),
        task_metrics=task_metrics,
        scenario_task_metrics=scenario_task_metrics,
    )
