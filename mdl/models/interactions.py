from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import MDLConfig


class RankMixerTokenMixing(nn.Module):
    def __init__(self, num_tokens: int, token_dim: int) -> None:
        super().__init__()
        if token_dim % num_tokens != 0:
            raise ValueError("token_dim must be divisible by num_tokens for RankMixer TokenMixing")
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.head_dim = token_dim // num_tokens

    def forward(self, tokens: Tensor) -> Tensor:
        batch_size, num_tokens, token_dim = tokens.shape
        if num_tokens != self.num_tokens or token_dim != self.token_dim:
            raise ValueError(
                f"expected tokens with shape [batch, {self.num_tokens}, {self.token_dim}], "
                f"got {tuple(tokens.shape)}"
            )
        split_tokens = tokens.view(batch_size, num_tokens, self.num_tokens, self.head_dim)
        mixed = split_tokens.permute(0, 2, 1, 3).contiguous()
        return mixed.view(batch_size, num_tokens, token_dim)


class FeatureInteraction(nn.Module):
    def __init__(self, config: MDLConfig, num_feature_tokens: int) -> None:
        super().__init__()
        self.backbone = config.feature_backbone
        if self.backbone == "rankmixer":
            self.rankmixer = RankMixerTokenMixing(num_feature_tokens, config.token_dim)
            self.attention = None
        else:
            self.rankmixer = None
            self.attention = nn.MultiheadAttention(
                embed_dim=config.token_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )

    def forward(
        self, tokens: Tensor, need_attention: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        if self.backbone == "rankmixer":
            if self.rankmixer is None:
                raise RuntimeError("rankmixer module is not initialized")
            return self.rankmixer(tokens), None
        if self.attention is None:
            raise RuntimeError("attention module is not initialized")
        mixed, weights = self.attention(
            tokens,
            tokens,
            tokens,
            need_weights=need_attention,
            average_attn_weights=False,
        )
        return mixed, weights if need_attention else None


class DomainAwareAttention(nn.Module):
    def __init__(self, token_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        domain_tokens: Tensor,
        feature_tokens: Tensor,
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        output, weights = self.attention(
            query=domain_tokens,
            key=feature_tokens,
            value=feature_tokens,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        return output, weights if need_weights else None


class DomainFusedModule(nn.Module):
    def forward(self, task_tokens: Tensor, scenario_tokens: Tensor, scenario_mask: Tensor) -> Tensor:
        if scenario_mask.ndim != 2:
            raise ValueError("scenario_mask must have shape [batch, num_scenarios]")
        if scenario_mask.size(0) != scenario_tokens.size(0):
            raise ValueError("scenario_mask batch size must match scenario_tokens")
        if scenario_mask.size(1) != scenario_tokens.size(1) - 1:
            raise ValueError("scenario_mask must exclude the global scenario token")

        mask = scenario_mask.to(dtype=scenario_tokens.dtype, device=scenario_tokens.device)
        global_mask = torch.ones(mask.size(0), 1, dtype=mask.dtype, device=mask.device)
        full_mask = torch.cat([mask, global_mask], dim=1)
        denominator = full_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        scenario_average = (scenario_tokens * full_mask.unsqueeze(-1)).sum(dim=1) / denominator
        return task_tokens + scenario_average.unsqueeze(1)
