from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .tokenization import (
    FeatureCompilerConfig,
    FeatureTokenCompiler,
    feature_specs_from_manifest,
    token_specs_from_manifest,
)
from .models import MDLConfig, MDLModel


@dataclass(frozen=True)
class TabularMDLConfig:
    token_specs: list[dict[str, Any]]
    feature_specs: list[dict[str, Any]]
    num_scenarios: int
    num_tasks: int
    embedding_dim: int = 32
    token_dim: int = 36
    num_layers: int = 2
    num_heads: int = 4
    ffn_hidden_dim: int = 64
    dropout: float = 0.0
    feature_backbone: str = "rankmixer"

    def feature_compiler_config(self) -> FeatureCompilerConfig:
        return FeatureCompilerConfig(
            token_specs=self.token_specs,
            feature_specs=self.feature_specs,
            embedding_dim=self.embedding_dim,
            token_dim=self.token_dim,
        )


def config_from_manifest(
    manifest: dict[str, Any],
    embedding_dim: int = 32,
    token_dim: int = 36,
    num_layers: int = 2,
    num_heads: int = 4,
    ffn_hidden_dim: int = 64,
    dropout: float = 0.0,
    feature_backbone: str = "rankmixer",
) -> TabularMDLConfig:
    return TabularMDLConfig(
        token_specs=token_specs_from_manifest(manifest),
        feature_specs=feature_specs_from_manifest(manifest),
        num_scenarios=len(manifest["scenario_names"]),
        num_tasks=len(manifest["task_names"]),
        embedding_dim=embedding_dim,
        token_dim=token_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_hidden_dim=ffn_hidden_dim,
        dropout=dropout,
        feature_backbone=feature_backbone,
    )


class TabularMDLModel(nn.Module):
    def __init__(self, config: TabularMDLConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_compiler = FeatureTokenCompiler(config.feature_compiler_config())
        self.scenario_embedding = nn.Embedding(config.num_scenarios, config.embedding_dim)
        self.task_context = nn.Parameter(torch.zeros(config.embedding_dim))

        mdl_config = MDLConfig(
            num_feature_tokens=len(config.token_specs),
            scenario_context_dim=config.embedding_dim,
            task_context_dim=config.embedding_dim,
            num_scenarios=config.num_scenarios,
            num_tasks=config.num_tasks,
            token_dim=config.token_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_hidden_dim=config.ffn_hidden_dim,
            dropout=config.dropout,
            feature_backbone=config.feature_backbone,
        )
        self.mdl = MDLModel(mdl_config)

    def compile_feature_tokens(
        self,
        features: dict[str, Tensor | dict[str, Tensor]],
    ) -> Tensor:
        return self.feature_compiler(features)

    def forward(
        self,
        features: dict[str, Tensor | dict[str, Tensor]],
        scenario_id: Tensor,
        return_attention: bool = False,
    ) -> dict[str, Tensor | list[dict[str, Tensor | None]]]:
        feature_tokens = self.compile_feature_tokens(features)
        return self.forward_tokens(feature_tokens, scenario_id, return_attention=return_attention)

    def forward_tokens(
        self,
        feature_tokens: Tensor,
        scenario_id: Tensor,
        return_attention: bool = False,
    ) -> dict[str, Tensor | list[dict[str, Tensor | None]]]:
        scenario_context = self.scenario_embedding(scenario_id)
        scenario_mask = torch.zeros(
            scenario_id.size(0),
            self.config.num_scenarios,
            dtype=torch.float32,
            device=scenario_id.device,
        )
        scenario_mask.scatter_(1, scenario_id.view(-1, 1), 1.0)
        task_context = self.task_context.view(1, -1).expand(scenario_id.size(0), -1)
        return self.mdl(
            feature_tokens,
            scenario_context,
            task_context,
            scenario_mask,
            return_attention=return_attention,
        )
