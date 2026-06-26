from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def multitask_bce_loss(
    logits: Tensor,
    labels: Tensor,
    label_mask: Tensor | None = None,
    task_weights: Tensor | None = None,
) -> Tensor:
    if logits.shape != labels.shape:
        raise ValueError(f"logits and labels must have the same shape, got {logits.shape} and {labels.shape}")

    loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")

    if task_weights is not None:
        if task_weights.ndim != 1 or task_weights.numel() != logits.size(1):
            raise ValueError("task_weights must have shape [num_tasks]")
        loss = loss * task_weights.to(device=logits.device, dtype=loss.dtype).view(1, -1)

    if label_mask is None:
        return loss.mean()

    if label_mask.shape != logits.shape:
        raise ValueError("label_mask must have the same shape as logits")
    mask = label_mask.to(device=logits.device, dtype=loss.dtype)
    denominator = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denominator
