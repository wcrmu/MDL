from __future__ import annotations

import torch
from torch import Tensor, nn


def _activation_layer(name: str) -> type[nn.Module]:
    if name == "gelu":
        return nn.GELU
    if name == "relu":
        return nn.ReLU
    raise ValueError("activation must be 'relu' or 'gelu'")


class PerTokenFFN(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        output_relu: bool = False,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        activation_layer = _activation_layer(activation)
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
