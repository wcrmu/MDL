from __future__ import annotations

import torch
from torch import Tensor, nn


class PerTokenLinear(nn.Module):
    def __init__(self, num_tokens: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.num_tokens = num_tokens
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList(
            nn.Linear(input_dim, output_dim) for _ in range(num_tokens)
        )

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch, num_tokens, input_dim]")
        if tokens.size(1) != self.num_tokens or tokens.size(2) != self.input_dim:
            raise ValueError(
                f"expected tokens with shape [batch, {self.num_tokens}, {self.input_dim}], "
                f"got {tuple(tokens.shape)}"
            )
        outputs = [
            layer(tokens[:, token_index, :]).unsqueeze(1)
            for token_index, layer in enumerate(self.layers)
        ]
        return torch.cat(outputs, dim=1)


class PerTokenFFN(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        output_relu: bool = False,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if activation not in {"relu", "gelu"}:
            raise ValueError("activation must be 'relu' or 'gelu'")
        activation_layer = nn.GELU if activation == "gelu" else nn.ReLU
        self.networks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                activation_layer(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, token_dim),
                nn.ReLU() if output_relu else nn.Identity(),
            )
            for _ in range(num_tokens)
        )

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.size(1) != len(self.networks):
            raise ValueError(f"expected {len(self.networks)} tokens, got {tokens.size(1)}")
        outputs = [
            network(tokens[:, token_index, :]).unsqueeze(1)
            for token_index, network in enumerate(self.networks)
        ]
        return torch.cat(outputs, dim=1)


class ContextTokenizer(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        context_dim: int,
        token_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.networks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, token_dim),
                nn.ReLU(),
            )
            for _ in range(num_tokens)
        )

    def forward(self, context: Tensor) -> Tensor:
        return torch.stack([network(context) for network in self.networks], dim=1)
