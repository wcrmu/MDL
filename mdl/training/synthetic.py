from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..models import MDLConfig


@dataclass
class SyntheticBatch:
    feature_tokens: Tensor
    scenario_context: Tensor
    task_context: Tensor
    scenario_mask: Tensor
    labels: Tensor
    label_mask: Tensor
    query_ids: list[str]


def make_synthetic_batch(
    config: MDLConfig,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> SyntheticBatch:
    feature_tokens = torch.randn(
        batch_size,
        config.num_feature_tokens,
        config.token_dim,
        device=device,
    )
    scenario_context = torch.randn(batch_size, config.scenario_context_dim, device=device)
    task_context = torch.randn(batch_size, config.task_context_dim, device=device)

    scenario_ids = torch.randint(config.num_scenarios, (batch_size,), device=device)
    scenario_mask = torch.zeros(batch_size, config.num_scenarios, device=device)
    scenario_mask.scatter_(1, scenario_ids.unsqueeze(1), 1.0)

    feature_signal = feature_tokens.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
    scenario_signal = scenario_ids.float().unsqueeze(1) / max(config.num_scenarios - 1, 1)
    task_offsets = torch.linspace(-0.5, 0.5, config.num_tasks, device=device).view(1, -1)
    logits = feature_signal + scenario_signal + task_offsets
    labels = torch.bernoulli(torch.sigmoid(logits))
    label_mask = torch.ones_like(labels)
    query_ids = [f"query-{index // 4}" for index in range(batch_size)]

    return SyntheticBatch(
        feature_tokens=feature_tokens,
        scenario_context=scenario_context,
        task_context=task_context,
        scenario_mask=scenario_mask,
        labels=labels,
        label_mask=label_mask,
        query_ids=query_ids,
    )

