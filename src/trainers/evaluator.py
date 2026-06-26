from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.datasets import ManifestDataset, collate_manifest_batch
from src.models import ModelFromManifest
from src.modules import binary_auc, multitask_bce_loss, qauc


@dataclass(frozen=True)
class TaskMetric:
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
        return lines


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    def move_value(value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move_value(child) for key, child in value.items()}
        return value

    return {key: move_value(value) for key, value in batch.items()}


@torch.no_grad()
def evaluate_model(
    model: ModelFromManifest,
    data_dir: str,
    split: str,
    manifest: dict[str, Any],
    batch_size: int,
    device: torch.device,
    max_batches: int | None = 100,
) -> EvaluationResult:
    dataset = ManifestDataset(data_dir, split)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_manifest_batch)
    task_names = manifest["task_names"]
    labels_by_task = [[] for _ in task_names]
    scores_by_task = [[] for _ in task_names]
    groups_by_task = [[] for _ in task_names]
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
        losses.append(multitask_bce_loss(logits, batch["labels"], batch["label_mask"]).item())
        probabilities = torch.sigmoid(logits).cpu()
        labels = batch["labels"].cpu()
        masks = batch["label_mask"].cpu()
        for task_index, _task_name in enumerate(task_names):
            valid = masks[:, task_index] > 0
            labels_by_task[task_index].extend(labels[valid, task_index].tolist())
            scores_by_task[task_index].extend(probabilities[valid, task_index].tolist())
            groups_by_task[task_index].extend(
                group for group, keep in zip(batch["group_id"], valid.tolist()) if keep
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

    return EvaluationResult(
        split=split,
        loss=sum(losses) / len(losses) if losses else float("nan"),
        task_metrics=task_metrics,
    )
