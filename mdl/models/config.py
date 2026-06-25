from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MDLConfig:
    num_feature_tokens: int
    scenario_context_dim: int
    task_context_dim: int
    num_scenarios: int
    num_tasks: int
    token_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4
    ffn_hidden_dim: int = 64
    dropout: float = 0.0
    feature_backbone: str = "rankmixer"

    def __post_init__(self) -> None:
        if self.num_feature_tokens <= 0:
            raise ValueError("num_feature_tokens must be positive")
        if self.scenario_context_dim <= 0:
            raise ValueError("scenario_context_dim must be positive")
        if self.task_context_dim <= 0:
            raise ValueError("task_context_dim must be positive")
        if self.num_scenarios <= 0:
            raise ValueError("num_scenarios must be positive")
        if self.num_tasks <= 0:
            raise ValueError("num_tasks must be positive")
        if self.token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.token_dim % self.num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if self.ffn_hidden_dim <= 0:
            raise ValueError("ffn_hidden_dim must be positive")
        if self.dropout < 0:
            raise ValueError("dropout must be non-negative")
        if self.feature_backbone not in {"rankmixer", "attention"}:
            raise ValueError("feature_backbone must be 'rankmixer' or 'attention'")
        if self.feature_backbone == "rankmixer" and self.token_dim % self.num_feature_tokens != 0:
            raise ValueError(
                "rankmixer backbone requires token_dim divisible by number of feature tokens"
            )
