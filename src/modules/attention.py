from __future__ import annotations

from contextlib import nullcontext
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - compatibility with older PyTorch builds.
    SDPBackend = None
    sdpa_kernel = None

from .mlp import PerTokenFFN


def _sdpa_context(attention_backend: str):
    if attention_backend in {"auto", "sdpa"}:
        return nullcontext()
    if attention_backend == "flash":
        if SDPBackend is None or sdpa_kernel is None:
            raise RuntimeError("runtime.attention_backend='flash' requires torch.nn.attention.sdpa_kernel")
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION])
    raise ValueError("attention_backend must be auto, sdpa, or flash")


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


class DomainAwareAttention(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        num_domain_tokens: int,
        num_feature_tokens: int,
        hidden_dim: int,
        dropout: float = 0.0,
        attention_backend: str = "auto",
    ) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.num_domain_tokens = num_domain_tokens
        self.num_feature_tokens = num_feature_tokens
        self.head_dim = token_dim // num_heads
        self.attention_backend = attention_backend
        self.query_projection = PerTokenFFN(
            num_domain_tokens,
            token_dim,
            hidden_dim,
            dropout=dropout,
            activation="relu",
        )
        self.key_projection = PerTokenFFN(
            num_feature_tokens,
            token_dim,
            hidden_dim,
            dropout=dropout,
            activation="relu",
        )
        self.value_projection = PerTokenFFN(
            num_feature_tokens,
            token_dim,
            hidden_dim,
            dropout=dropout,
            activation="relu",
        )
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, tokens: Tensor) -> Tensor:
        batch_size, num_tokens, _token_dim = tokens.shape
        return tokens.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, tokens: Tensor) -> Tensor:
        batch_size, _num_heads, num_tokens, _head_dim = tokens.shape
        return tokens.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.token_dim)

    def forward(
        self,
        domain_tokens: Tensor,
        feature_tokens: Tensor,
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if domain_tokens.size(1) != self.num_domain_tokens:
            raise ValueError(
                f"expected {self.num_domain_tokens} domain tokens, got {domain_tokens.size(1)}"
            )
        if feature_tokens.size(1) != self.num_feature_tokens:
            raise ValueError(
                f"expected {self.num_feature_tokens} feature tokens, got {feature_tokens.size(1)}"
            )

        query = self._split_heads(self.query_projection(domain_tokens))
        key = self._split_heads(self.key_projection(feature_tokens))
        value = self._split_heads(self.value_projection(feature_tokens))

        if not need_weights:
            dropout_p = self.dropout.p if self.training else 0.0
            with _sdpa_context(self.attention_backend):
                attended = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=dropout_p,
                )
            return self._merge_heads(attended), None

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(self.dropout(weights), value)
        return self._merge_heads(attended), weights if need_weights else None


class DomainFusedModule(nn.Module):
    def __init__(self, include_global: bool = True) -> None:
        super().__init__()
        self.include_global = include_global

    def forward(self, task_tokens: Tensor, scenario_tokens: Tensor, scenario_mask: Tensor) -> Tensor:
        if scenario_mask.ndim != 2:
            raise ValueError("scenario_mask must have shape [batch, num_scenarios]")
        if scenario_mask.size(0) != scenario_tokens.size(0):
            raise ValueError("scenario_mask batch size must match scenario_tokens")
        if scenario_mask.size(1) != scenario_tokens.size(1) - 1:
            raise ValueError("scenario_mask must exclude the global scenario token")

        mask = scenario_mask.to(dtype=scenario_tokens.dtype, device=scenario_tokens.device)
        if self.include_global:
            global_mask = torch.ones(mask.size(0), 1, dtype=mask.dtype, device=mask.device)
            full_mask = torch.cat([mask, global_mask], dim=1)
            fused_scenario_tokens = scenario_tokens
        else:
            full_mask = mask
            fused_scenario_tokens = scenario_tokens[:, :-1, :]
        denominator = full_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        scenario_average = (fused_scenario_tokens * full_mask.unsqueeze(-1)).sum(dim=1) / denominator
        return task_tokens + scenario_average.unsqueeze(1)
