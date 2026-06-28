from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import Tensor, nn

from src.datasets.feature_schema import (
    feature_specs_from_manifest,
    scenario_feature_specs_from_manifest,
    scenario_token_specs_from_manifest,
    task_feature_specs_from_manifest,
    task_token_specs_from_manifest,
    token_specs_from_manifest,
)
from src.modules.attention import DomainAwareAttention, DomainFusedModule, FeatureInteraction
from src.modules.mlp import ContextTokenizer, PerTokenFFN, SparseMoEPerTokenFFN
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
    ffn_type: str = "dense"
    sparse_moe_num_experts: int = 4
    sparse_moe_use_dtsi: bool = True
    sparse_moe_dtsi_infer_weight: float = 0.5
    sparse_moe_inference_threshold: float = 0.0
    use_task_tokens: bool = True
    use_scenario_tokens: bool = True
    use_global_scenario_token: bool = True
    use_task_feature_interaction: bool = True
    use_scenario_feature_interaction: bool = True

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
        if self.ffn_type not in {"dense", "sparse_moe"}:
            raise ValueError("ffn_type must be 'dense' or 'sparse_moe'")
        if self.sparse_moe_num_experts <= 0:
            raise ValueError("sparse_moe_num_experts must be positive")
        if self.sparse_moe_inference_threshold < 0:
            raise ValueError("sparse_moe_inference_threshold must be non-negative")
        if not 0.0 <= self.sparse_moe_dtsi_infer_weight <= 1.0:
            raise ValueError("sparse_moe_dtsi_infer_weight must be in [0, 1]")
        if self.feature_backbone == "rankmixer" and self.token_dim % self.num_feature_tokens != 0:
            raise ValueError(
                "rankmixer backbone requires token_dim divisible by number of feature tokens"
            )


def _build_per_token_ffn(num_tokens: int, config: MDLConfig) -> nn.Module:
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


class MDLBlock(nn.Module):
    def __init__(self, config: MDLConfig) -> None:
        super().__init__()
        num_feature_tokens = config.num_feature_tokens
        num_scenario_tokens = config.num_scenarios + 1
        self.use_task_tokens = config.use_task_tokens
        self.use_scenario_tokens = config.use_scenario_tokens
        self.use_global_scenario_token = config.use_global_scenario_token
        self.use_task_feature_interaction = config.use_task_feature_interaction
        self.use_scenario_feature_interaction = config.use_scenario_feature_interaction

        self.feature_interaction = FeatureInteraction(
            num_feature_tokens=num_feature_tokens,
            token_dim=config.token_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            backbone=config.feature_backbone,
        )
        self.feature_norm_1 = nn.LayerNorm(config.token_dim)
        self.feature_ffn = _build_per_token_ffn(num_feature_tokens, config)
        self.feature_norm_2 = nn.LayerNorm(config.token_dim)

        self.scenario_attention = DomainAwareAttention(
            config.token_dim,
            config.num_heads,
            num_scenario_tokens,
            num_feature_tokens,
            config.dropout,
        )
        self.scenario_ffn = _build_per_token_ffn(num_scenario_tokens, config)

        self.task_attention = DomainAwareAttention(
            config.token_dim,
            config.num_heads,
            config.num_tasks,
            num_feature_tokens,
            config.dropout,
        )
        self.domain_fused = DomainFusedModule(include_global=config.use_global_scenario_token)
        self.task_ffn = _build_per_token_ffn(config.num_tasks, config)

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

        if not self.use_scenario_tokens:
            scenario_tokens = torch.zeros_like(scenario_tokens)
        elif not self.use_global_scenario_token:
            scenario_tokens = scenario_tokens.clone()
            scenario_tokens[:, -1, :] = 0.0

        if not self.use_task_tokens:
            task_tokens = torch.zeros_like(task_tokens)

        if self.use_scenario_feature_interaction:
            scenario_update, scenario_weights = self.scenario_attention(
                scenario_tokens,
                feature_tokens,
                need_weights=need_attention,
            )
        else:
            scenario_update = torch.zeros_like(scenario_tokens)
            scenario_weights = None
        scenario_hat = scenario_tokens + scenario_update
        if not self.use_global_scenario_token:
            scenario_hat = scenario_hat.clone()
            scenario_hat[:, -1, :] = 0.0
        scenario_tokens = scenario_hat + self.scenario_ffn(scenario_hat)
        if not self.use_global_scenario_token:
            scenario_tokens = scenario_tokens.clone()
            scenario_tokens[:, -1, :] = 0.0

        if self.use_task_feature_interaction:
            task_update, task_weights = self.task_attention(
                task_tokens,
                feature_tokens,
                need_weights=need_attention,
            )
        else:
            task_update = torch.zeros_like(task_tokens)
            task_weights = None
        task_hat = task_tokens + task_update
        if self.use_scenario_tokens or self.use_scenario_feature_interaction:
            task_tokens = self.domain_fused(task_hat, scenario_hat, scenario_mask)
        else:
            task_tokens = task_hat
        task_tokens = task_tokens + self.task_ffn(task_tokens)

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

    def _validate_tokens(self, tokens: Tensor, name: str, expected_tokens: int) -> None:
        expected_shape = (expected_tokens, self.config.token_dim)
        if tokens.ndim != 3:
            raise ValueError(
                f"{name} must have shape [batch, {expected_shape[0]}, {expected_shape[1]}], "
                f"got {tuple(tokens.shape)}"
            )
        actual_shape = (tokens.size(1), tokens.size(2))
        if actual_shape != expected_shape:
            raise ValueError(
                f"{name} must have shape [batch, {expected_shape[0]}, {expected_shape[1]}], "
                f"got {tuple(tokens.shape)}"
            )

    def forward(
        self,
        feature_tokens: Tensor,
        scenario_context: Tensor | None = None,
        task_context: Tensor | None = None,
        scenario_mask: Tensor | None = None,
        return_attention: bool = False,
        *,
        scenario_tokens: Tensor | None = None,
        task_tokens: Tensor | None = None,
    ) -> dict[str, Tensor | list[dict[str, Tensor | None]]]:
        if feature_tokens.ndim != 3:
            raise ValueError("feature_tokens must have shape [batch, num_feature_tokens, token_dim]")
        self._validate_tokens(feature_tokens, "feature_tokens", self.config.num_feature_tokens)
        if scenario_mask is None:
            raise ValueError("scenario_mask is required")

        if scenario_tokens is None:
            if scenario_context is None:
                raise ValueError("scenario_context or scenario_tokens is required")
            scenario_tokens = self.scenario_tokenizer(scenario_context)
        else:
            self._validate_tokens(scenario_tokens, "scenario_tokens", self.config.num_scenarios + 1)

        if task_tokens is None:
            if task_context is None:
                raise ValueError("task_context or task_tokens is required")
            task_tokens = self.task_tokenizer(task_context)
        else:
            self._validate_tokens(task_tokens, "task_tokens", self.config.num_tasks)

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
        if self.config.ffn_type == "sparse_moe":
            output["moe_regularization_loss"] = self.sparse_moe_regularization_loss(logits)
            output["moe_active_ratio"] = self.sparse_moe_active_ratio(logits)
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
    ffn_type: str = "dense"
    sparse_moe_num_experts: int = 4
    sparse_moe_loss_weight: float = 0.0
    sparse_moe_use_dtsi: bool = True
    sparse_moe_dtsi_infer_weight: float = 0.5
    sparse_moe_inference_threshold: float = 0.0
    use_task_tokens: bool = True
    use_scenario_tokens: bool = True
    use_global_scenario_token: bool = True
    use_task_feature_interaction: bool = True
    use_scenario_feature_interaction: bool = True
    scenario_token_specs: list[dict[str, Any]] | None = None
    scenario_feature_specs: list[dict[str, Any]] | None = None
    task_token_specs: list[dict[str, Any]] | None = None
    task_feature_specs: list[dict[str, Any]] | None = None

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

    def scenario_compiler_config(self) -> FeatureCompilerConfig:
        if self.scenario_token_specs is None or self.scenario_feature_specs is None:
            raise ValueError(
                "manifest tokenization must declare scenario_features and scenario_token_specs"
            )
        return FeatureCompilerConfig(
            token_specs=self.scenario_token_specs,
            feature_specs=self.scenario_feature_specs,
            embedding_dim=self.embedding_dim,
            token_dim=self.token_dim,
            hidden_dim=self.ffn_hidden_dim,
            dropout=self.dropout,
            default_projection="ffn_relu",
        )

    def task_compiler_config(self) -> FeatureCompilerConfig:
        if self.task_token_specs is None or self.task_feature_specs is None:
            raise ValueError(
                "manifest tokenization must declare task_features and task_token_specs"
            )
        return FeatureCompilerConfig(
            token_specs=self.task_token_specs,
            feature_specs=self.task_feature_specs,
            embedding_dim=self.embedding_dim,
            token_dim=self.token_dim,
            hidden_dim=self.ffn_hidden_dim,
            dropout=self.dropout,
            default_projection="ffn_relu",
        )


def _required_domain_token_specs(
    manifest: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    scenario_feature_specs = scenario_feature_specs_from_manifest(manifest)
    scenario_token_specs = scenario_token_specs_from_manifest(manifest)
    task_feature_specs = task_feature_specs_from_manifest(manifest)
    task_token_specs = task_token_specs_from_manifest(manifest)
    missing = []
    if scenario_feature_specs is None:
        missing.append("scenario_features")
    if scenario_token_specs is None:
        missing.append("scenario_token_specs")
    if task_feature_specs is None:
        missing.append("task_features")
    if task_token_specs is None:
        missing.append("task_token_specs")
    if missing:
        raise ValueError(
            "manifest tokenization must declare "
            "scenario_features, scenario_token_specs, task_features, and task_token_specs; "
            f"missing: {', '.join(missing)}"
        )
    return scenario_feature_specs, scenario_token_specs, task_feature_specs, task_token_specs


def config_from_manifest(
    manifest: dict[str, Any],
    embedding_dim: int = 32,
    token_dim: int = 36,
    num_layers: int = 2,
    num_heads: int = 4,
    ffn_hidden_dim: int = 64,
    dropout: float = 0.0,
    feature_backbone: str = "rankmixer",
    ffn_type: str = "dense",
    sparse_moe_num_experts: int = 4,
    sparse_moe_loss_weight: float = 0.0,
    sparse_moe_use_dtsi: bool = True,
    sparse_moe_dtsi_infer_weight: float = 0.5,
    sparse_moe_inference_threshold: float = 0.0,
    use_task_tokens: bool = True,
    use_scenario_tokens: bool = True,
    use_global_scenario_token: bool = True,
    use_task_feature_interaction: bool = True,
    use_scenario_feature_interaction: bool = True,
) -> ModelConfig:
    (
        scenario_feature_specs,
        scenario_token_specs,
        task_feature_specs,
        task_token_specs,
    ) = _required_domain_token_specs(manifest)
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
        ffn_type=ffn_type,
        sparse_moe_num_experts=sparse_moe_num_experts,
        sparse_moe_loss_weight=sparse_moe_loss_weight,
        sparse_moe_use_dtsi=sparse_moe_use_dtsi,
        sparse_moe_dtsi_infer_weight=sparse_moe_dtsi_infer_weight,
        sparse_moe_inference_threshold=sparse_moe_inference_threshold,
        use_task_tokens=use_task_tokens,
        use_scenario_tokens=use_scenario_tokens,
        use_global_scenario_token=use_global_scenario_token,
        use_task_feature_interaction=use_task_feature_interaction,
        use_scenario_feature_interaction=use_scenario_feature_interaction,
        scenario_token_specs=scenario_token_specs,
        scenario_feature_specs=scenario_feature_specs,
        task_token_specs=task_token_specs,
        task_feature_specs=task_feature_specs,
    )


class ModelFromManifest(BaseRecommender):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_compiler = FeatureTokenCompiler(config.feature_compiler_config())

        scenario_compiler_config = config.scenario_compiler_config()
        if len(scenario_compiler_config.token_specs) != config.num_scenarios + 1:
            raise ValueError(
                "scenario_token_specs must contain one token per scenario plus one global token"
            )
        self.scenario_token_compiler = FeatureTokenCompiler(scenario_compiler_config)
        self.scenario_embedding = None

        task_compiler_config = config.task_compiler_config()
        if len(task_compiler_config.token_specs) != config.num_tasks:
            raise ValueError("task_token_specs must contain one token per task")
        self.task_token_compiler = FeatureTokenCompiler(task_compiler_config)
        self.register_parameter("task_context", None)

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
            ffn_type=config.ffn_type,
            sparse_moe_num_experts=config.sparse_moe_num_experts,
            sparse_moe_use_dtsi=config.sparse_moe_use_dtsi,
            sparse_moe_dtsi_infer_weight=config.sparse_moe_dtsi_infer_weight,
            sparse_moe_inference_threshold=config.sparse_moe_inference_threshold,
            use_task_tokens=config.use_task_tokens,
            use_scenario_tokens=config.use_scenario_tokens,
            use_global_scenario_token=config.use_global_scenario_token,
            use_task_feature_interaction=config.use_task_feature_interaction,
            use_scenario_feature_interaction=config.use_scenario_feature_interaction,
        )
        self.mdl = MDLModel(mdl_config)

    def compile_feature_tokens(
        self,
        features: dict[str, Tensor | dict[str, Tensor]],
    ) -> Tensor:
        return self.feature_compiler(features)

    def compile_scenario_tokens(
        self,
        features: dict[str, Tensor | dict[str, Tensor]],
    ) -> Tensor:
        return self.scenario_token_compiler(features)

    def compile_task_tokens(
        self,
        features: dict[str, Tensor | dict[str, Tensor]],
    ) -> Tensor:
        return self.task_token_compiler(features)

    def _scenario_mask(self, scenario_id: Tensor) -> Tensor:
        if scenario_id.ndim == 2 and scenario_id.size(1) == self.config.num_scenarios:
            return scenario_id.to(dtype=torch.float32)
        if scenario_id.ndim == 1:
            scenario_indices = scenario_id.view(-1, 1).to(dtype=torch.long)
        elif scenario_id.ndim == 2:
            scenario_indices = scenario_id.to(dtype=torch.long)
        else:
            raise ValueError("scenario_id must have shape [batch], [batch, num_scenarios], or [batch, k]")
        if scenario_indices.numel() > 0:
            if int(scenario_indices.min().item()) < 0:
                raise ValueError("scenario_id must be non-negative")
            if int(scenario_indices.max().item()) >= self.config.num_scenarios:
                raise ValueError("scenario_id out of range")
        mask = torch.zeros(
            scenario_indices.size(0),
            self.config.num_scenarios,
            dtype=torch.float32,
            device=scenario_id.device,
        )
        mask.scatter_(1, scenario_indices, 1.0)
        return mask

    def forward(
        self,
        features: dict[str, Tensor | dict[str, Tensor]],
        scenario_id: Tensor,
        return_attention: bool = False,
    ) -> dict[str, Tensor | list[dict[str, Tensor | None]]]:
        feature_tokens = self.compile_feature_tokens(features)
        scenario_mask = self._scenario_mask(scenario_id)
        scenario_tokens = self.compile_scenario_tokens(features)
        task_tokens = self.compile_task_tokens(features)

        return self.mdl(
            feature_tokens,
            scenario_mask=scenario_mask,
            return_attention=return_attention,
            scenario_tokens=scenario_tokens,
            task_tokens=task_tokens,
        )

    def forward_tokens(
        self,
        feature_tokens: Tensor,
        scenario_id: Tensor,
        return_attention: bool = False,
    ) -> dict[str, Tensor | list[dict[str, Tensor | None]]]:
        raise ValueError(
            "forward_tokens is unavailable for ModelFromManifest because manifest-driven "
            "scenario/task tokenization is required; call forward(features, scenario_id)"
        )


def count_parameters(modules: Iterable[nn.Module]) -> int:
    return sum(parameter.numel() for module in modules for parameter in module.parameters())
