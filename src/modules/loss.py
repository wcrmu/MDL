from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def multitask_bce_loss(
    logits: Tensor,
    labels: Tensor,
    label_mask: Tensor | None = None,
    task_weights: Tensor | None = None,
    sample_weights: Tensor | None = None,
    positive_class_weights: Tensor | None = None,
) -> Tensor:
    if logits.shape != labels.shape:
        raise ValueError(f"logits and labels must have the same shape, got {logits.shape} and {labels.shape}")

    loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    weights = torch.ones_like(loss)

    if label_mask is not None:
        if label_mask.shape != logits.shape:
            raise ValueError("label_mask must have the same shape as logits")
        weights = weights * label_mask.to(device=logits.device, dtype=loss.dtype)

    if task_weights is not None:
        if task_weights.ndim != 1 or task_weights.numel() != logits.size(1):
            raise ValueError("task_weights must have shape [num_tasks]")
        weights = weights * task_weights.to(device=logits.device, dtype=loss.dtype).view(1, -1)

    if positive_class_weights is not None:
        if positive_class_weights.ndim != 1 or positive_class_weights.numel() != logits.size(1):
            raise ValueError("positive_class_weights must have shape [num_tasks]")
        positive_class_weights = positive_class_weights.to(
            device=logits.device,
            dtype=loss.dtype,
        ).view(1, -1)
        weights = weights * torch.where(labels.float() > 0.5, positive_class_weights, 1.0)

    if sample_weights is not None:
        sample_weights = sample_weights.to(device=logits.device, dtype=loss.dtype)
        if sample_weights.ndim == 1:
            sample_weights = sample_weights.view(-1, 1)
        if sample_weights.ndim != 2 or sample_weights.size(0) != logits.size(0):
            raise ValueError("sample_weights must have shape [batch] or [batch, 1]")
        if sample_weights.size(1) != 1:
            raise ValueError("sample_weights must have shape [batch] or [batch, 1]")
        weights = weights * sample_weights

    denominator = weights.sum().clamp_min(1.0)
    return (loss * weights).sum() / denominator
