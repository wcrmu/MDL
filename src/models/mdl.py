from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from src.datasets.feature_schema import feature_specs_from_manifest, token_specs_from_manifest
from src.modules.attention import DomainAwareAttention, DomainFusedModule, FeatureInteraction
from src.modules.mlp import ContextTokenizer, PerTokenFFN
from src.modules.tokenizer import FeatureCompilerConfig, FeatureTokenCompiler

from .base import BaseRecommender


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


class MDLBlock(nn.Module):
    def __init__(self, config: MDLConfig) -> None:
        super().__init__()
        num_feature_tokens = config.num_feature_tokens
        num_scenario_tokens = config.num_scenarios + 1

        self.feature_interaction = FeatureInteraction(
            num_feature_tokens=num_feature_tokens,
            token_dim=config.token_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            backbone=config.feature_backbone,
        )
        self.feature_norm_1 = nn.LayerNorm(config.token_dim)
        self.feature_ffn = PerTokenFFN(
            num_feature_tokens,
            config.token_dim,
            config.ffn_hidden_dim,
            config.dropout,
            activation="gelu",
        )
        self.feature_norm_2 = nn.LayerNorm(config.token_dim)

        self.scenario_attention = DomainAwareAttention(
            config.token_dim,
            config.num_heads,
            config.dropout,
        )
        self.scenario_ffn = PerTokenFFN(
            num_scenario_tokens,
            config.token_dim,
            config.ffn_hidden_dim,
            config.dropout,
        )
        self.scenario_norm_1 = nn.LayerNorm(config.token_dim)
        self.scenario_norm_2 = nn.LayerNorm(config.token_dim)

        self.task_attention = DomainAwareAttention(
            config.token_dim,
            config.num_heads,
            config.dropout,
        )
        self.domain_fused = DomainFusedModule()
        self.task_ffn = PerTokenFFN(
            config.num_tasks,
            config.token_dim,
            config.ffn_hidden_dim,
            config.dropout,
        )
        self.task_norm_1 = nn.LayerNorm(config.token_dim)
        self.task_norm_2 = nn.LayerNorm(config.token_dim)

    def forward(
        self,
        feature_tokens: Tensor,
        scenario_tokens: Tensor,
        task_tokens: Tensor,
        scenario_mask: Tensor,
        need_attention: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor | None]]:
        mixed_features, feature_weights = self.feature_interaction(
            feature_tokens,
            need_attention=need_attention,
        )
        feature_tokens = self.feature_norm_1(feature_tokens + mixed_features)
        feature_tokens = self.feature_norm_2(feature_tokens + self.feature_ffn(feature_tokens))

        scenario_update, scenario_weights = self.scenario_attention(
            scenario_tokens,
            feature_tokens,
            need_weights=need_attention,
        )
        scenario_tokens = self.scenario_norm_1(scenario_tokens + scenario_update)
        scenario_tokens = self.scenario_norm_2(
            scenario_tokens + self.scenario_ffn(scenario_tokens)
        )

        task_update, task_weights = self.task_attention(
            task_tokens,
            feature_tokens,
            need_weights=need_attention,
        )
        task_tokens = self.task_norm_1(task_tokens + task_update)
        task_tokens = self.domain_fused(task_tokens, scenario_tokens, scenario_mask)
        task_tokens = self.task_norm_2(task_tokens + self.task_ffn(task_tokens))

        attention = {
            "feature": feature_weights,
            "scenario_feature": scenario_weights,
            "task_feature": task_weights,
        }
        return feature_tokens, scenario_tokens, task_tokens, attention


class MDLModel(BaseRecommender):
    def __init__(self, config: MDLConfig) -> None:
        super().__init__()
        self.config = config
        self.scenario_tokenizer = ContextTokenizer(
            config.num_scenarios + 1,
            config.scenario_context_dim,
            config.token_dim,
            config.ffn_hidden_dim,
            config.dropout,
        )
        self.task_tokenizer = ContextTokenizer(
            config.num_tasks,
            config.task_context_dim,
            config.token_dim,
            config.ffn_hidden_dim,
            config.dropout,
        )
        self.blocks = nn.ModuleList(MDLBlock(config) for _ in range(config.num_layers))
        self.logit_layers = nn.ModuleList(
            nn.Linear(config.token_dim, 1) for _ in range(config.num_tasks)
        )

    def forward(
        self,
        feature_tokens: Tensor,
        scenario_context: Tensor,
        task_context: Tensor,
        scenario_mask: Tensor,
        return_attention: bool = False,
    ) -> dict[str, Tensor | list[dict[str, Tensor | None]]]:
        if feature_tokens.ndim != 3:
            raise ValueError("feature_tokens must have shape [batch, num_feature_tokens, token_dim]")
        expected_shape = (self.config.num_feature_tokens, self.config.token_dim)
        actual_shape = (feature_tokens.size(1), feature_tokens.size(2))
        if actual_shape != expected_shape:
            raise ValueError(
                "feature_tokens must have shape "
                f"[batch, {expected_shape[0]}, {expected_shape[1]}], got {tuple(feature_tokens.shape)}"
            )
        scenario_tokens = self.scenario_tokenizer(scenario_context)
        task_tokens = self.task_tokenizer(task_context)

        attentions: list[dict[str, Tensor | None]] = []
        for block in self.blocks:
            feature_tokens, scenario_tokens, task_tokens, attention = block(
                feature_tokens,
                scenario_tokens,
                task_tokens,
                scenario_mask,
                need_attention=return_attention,
            )
            if return_attention:
                attentions.append(attention)

        logits = torch.cat(
            [
                logit_layer(task_tokens[:, task_index, :])
                for task_index, logit_layer in enumerate(self.logit_layers)
            ],
            dim=1,
        )
        output: dict[str, Tensor | list[dict[str, Tensor | None]]] = {"logits": logits}
        if return_attention:
            output["attentions"] = attentions
        return output

    @torch.no_grad()
    def predict_proba(
        self,
        feature_tokens: Tensor,
        scenario_context: Tensor,
        task_context: Tensor,
        scenario_mask: Tensor,
    ) -> Tensor:
        return torch.sigmoid(
            self.forward(feature_tokens, scenario_context, task_context, scenario_mask)[
                "logits"
            ]
        )


@dataclass(frozen=True)
class ModelConfig:
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
) -> ModelConfig:
    return ModelConfig(
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


class ModelFromManifest(BaseRecommender):
    def __init__(self, config: ModelConfig) -> None:
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


def count_parameters(modules: Iterable[nn.Module]) -> int:
    return sum(parameter.numel() for module in modules for parameter in module.parameters())
