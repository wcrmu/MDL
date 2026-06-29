from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from src.datasets.feature_schema import feature_specs_from_manifest, token_specs_from_manifest
from src.modules.attention import RankMixerTokenMixing
from src.modules.mlp import PerTokenFFN, SparseMoEPerTokenFFN
from src.modules.tokenizer import FeatureCompilerConfig, FeatureTokenCompiler

from .base import BaseRecommender


@dataclass(frozen=True)
class RankMixerConfig:
    token_specs: list[dict[str, Any]]
    feature_specs: list[dict[str, Any]]
    num_tasks: int
    embedding_dim: int = 32
    token_dim: int = 36
    num_layers: int = 2
    ffn_hidden_dim: int = 64
    dropout: float = 0.0
    ffn_type: str = "dense"
    sparse_moe_num_experts: int = 4
    sparse_moe_loss_weight: float = 0.0
    sparse_moe_use_dtsi: bool = True
    sparse_moe_dtsi_infer_weight: float = 0.5
    sparse_moe_inference_threshold: float = 0.0
    model_name: str = "rankmixer"

    def __post_init__(self) -> None:
        if self.model_name != "rankmixer":
            raise ValueError("RankMixerConfig model_name must be 'rankmixer'")
        if not self.token_specs:
            raise ValueError("token_specs must be non-empty")
        if not self.feature_specs:
            raise ValueError("feature_specs must be non-empty")
        if self.num_tasks <= 0:
            raise ValueError("num_tasks must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.ffn_hidden_dim <= 0:
            raise ValueError("ffn_hidden_dim must be positive")
        if self.dropout < 0:
            raise ValueError("dropout must be non-negative")
        if self.ffn_type not in {"dense", "sparse_moe"}:
            raise ValueError("ffn_type must be 'dense' or 'sparse_moe'")
        if self.sparse_moe_num_experts <= 0:
            raise ValueError("sparse_moe_num_experts must be positive")
        if self.sparse_moe_inference_threshold < 0:
            raise ValueError("sparse_moe_inference_threshold must be non-negative")
        if not 0.0 <= self.sparse_moe_dtsi_infer_weight <= 1.0:
            raise ValueError("sparse_moe_dtsi_infer_weight must be in [0, 1]")
        if self.token_dim % len(self.token_specs) != 0:
            raise ValueError("rankmixer requires token_dim divisible by number of feature tokens")

    def feature_compiler_config(self) -> FeatureCompilerConfig:
        return FeatureCompilerConfig(
            token_specs=self.token_specs,
            feature_specs=self.feature_specs,
            embedding_dim=self.embedding_dim,
            token_dim=self.token_dim,
            hidden_dim=self.ffn_hidden_dim,
            dropout=self.dropout,
            default_projection="linear",
        )


def _build_per_token_ffn(num_tokens: int, config: RankMixerConfig) -> nn.Module:
    if config.ffn_type == "dense":
        return PerTokenFFN(
            num_tokens,
            config.token_dim,
            config.ffn_hidden_dim,
            config.dropout,
            activation="gelu",
        )
    return SparseMoEPerTokenFFN(
        num_tokens=num_tokens,
        token_dim=config.token_dim,
        hidden_dim=config.ffn_hidden_dim,
        num_experts=config.sparse_moe_num_experts,
        dropout=config.dropout,
        activation="gelu",
        use_dtsi=config.sparse_moe_use_dtsi,
        dtsi_infer_weight=config.sparse_moe_dtsi_infer_weight,
        inference_threshold=config.sparse_moe_inference_threshold,
    )


class RankMixerBlock(nn.Module):
    def __init__(self, config: RankMixerConfig) -> None:
        super().__init__()
        num_tokens = len(config.token_specs)
        self.mixer = RankMixerTokenMixing(num_tokens, config.token_dim)
        self.norm_1 = nn.LayerNorm(config.token_dim)
        self.ffn = _build_per_token_ffn(num_tokens, config)
        self.norm_2 = nn.LayerNorm(config.token_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = self.norm_1(tokens + self.mixer(tokens))
        tokens = self.norm_2(tokens + self.ffn(tokens))
        return tokens


class RankMixerFromManifest(BaseRecommender):
    def __init__(self, config: RankMixerConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_compiler = FeatureTokenCompiler(config.feature_compiler_config())
        self.blocks = nn.ModuleList(RankMixerBlock(config) for _ in range(config.num_layers))
        self.output_layer = nn.Linear(len(config.token_specs) * config.token_dim, config.num_tasks)

    def compile_feature_tokens(self, features: dict[str, Tensor | dict[str, Tensor]]) -> Tensor:
        return self.feature_compiler(features)

    def sparse_moe_regularization_loss(self, reference: Tensor) -> Tensor:
        losses = [
            module.regularization_loss(reference).to(device=reference.device)
            for module in self.modules()
            if isinstance(module, SparseMoEPerTokenFFN)
        ]
        if not losses:
            return reference.new_zeros(())
        return torch.stack(losses).sum()

    def sparse_moe_active_ratio(self, reference: Tensor) -> Tensor:
        ratios = [
            module.active_ratio(reference).to(device=reference.device)
            for module in self.modules()
            if isinstance(module, SparseMoEPerTokenFFN)
        ]
        if not ratios:
            return reference.new_zeros(())
        return torch.stack(ratios).mean().detach()

    def forward(
        self,
        features: dict[str, Tensor | dict[str, Tensor]],
        scenario_id: Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, Tensor | list[dict[str, Tensor | None]]]:
        del scenario_id
        tokens = self.compile_feature_tokens(features)
        for block in self.blocks:
            tokens = block(tokens)
        logits = self.output_layer(tokens.flatten(start_dim=1))
        output: dict[str, Tensor | list[dict[str, Tensor | None]]] = {"logits": logits}
        if self.config.ffn_type == "sparse_moe":
            output["moe_regularization_loss"] = self.sparse_moe_regularization_loss(logits)
            output["moe_active_ratio"] = self.sparse_moe_active_ratio(logits)
        if return_attention:
            output["attentions"] = []
        return output


def rankmixer_config_from_manifest(
    manifest: dict[str, Any],
    embedding_dim: int = 32,
    token_dim: int = 36,
    num_layers: int = 2,
    ffn_hidden_dim: int = 64,
    dropout: float = 0.0,
    ffn_type: str = "dense",
    sparse_moe_num_experts: int = 4,
    sparse_moe_loss_weight: float = 0.0,
    sparse_moe_use_dtsi: bool = True,
    sparse_moe_dtsi_infer_weight: float = 0.5,
    sparse_moe_inference_threshold: float = 0.0,
) -> RankMixerConfig:
    return RankMixerConfig(
        token_specs=token_specs_from_manifest(manifest),
        feature_specs=feature_specs_from_manifest(manifest),
        num_tasks=len(manifest["task_names"]),
        embedding_dim=embedding_dim,
        token_dim=token_dim,
        num_layers=num_layers,
        ffn_hidden_dim=ffn_hidden_dim,
        dropout=dropout,
        ffn_type=ffn_type,
        sparse_moe_num_experts=sparse_moe_num_experts,
        sparse_moe_loss_weight=sparse_moe_loss_weight,
        sparse_moe_use_dtsi=sparse_moe_use_dtsi,
        sparse_moe_dtsi_infer_weight=sparse_moe_dtsi_infer_weight,
        sparse_moe_inference_threshold=sparse_moe_inference_threshold,
    )


__all__ = [
    "RankMixerBlock",
    "RankMixerConfig",
    "RankMixerFromManifest",
    "RankMixerTokenMixing",
    "rankmixer_config_from_manifest",
]
