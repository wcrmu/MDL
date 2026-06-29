from __future__ import annotations

import torch
from torch.nn import functional as F

from src.modules import binary_auc, multitask_bce_loss


def test_binary_auc() -> None:
    assert binary_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75


def test_multitask_bce_loss_uses_weighted_denominator() -> None:
    logits = torch.tensor([[0.5, -1.0], [1.5, 0.25]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    task_weights = torch.tensor([1.0, 3.0])
    sample_weights = torch.tensor([2.0, 0.5])

    loss = multitask_bce_loss(
        logits,
        labels,
        mask,
        task_weights=task_weights,
        sample_weights=sample_weights,
    )

    elementwise = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    weights = mask * task_weights.view(1, -1) * sample_weights.view(-1, 1)
    expected = (elementwise * weights).sum() / weights.sum()
    assert torch.allclose(loss, expected)


def test_multitask_bce_loss_applies_positive_class_weights() -> None:
    logits = torch.tensor([[0.5, -1.0], [1.5, 0.25]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.ones_like(labels)
    positive_class_weights = torch.tensor([4.0, 2.0])

    loss = multitask_bce_loss(
        logits,
        labels,
        mask,
        positive_class_weights=positive_class_weights,
    )

    elementwise = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    weights = torch.where(labels > 0.5, positive_class_weights.view(1, -1), 1.0)
    expected = (elementwise * weights).sum() / weights.sum()
    assert torch.allclose(loss, expected)
