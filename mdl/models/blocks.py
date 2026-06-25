from __future__ import annotations

from torch import Tensor, nn

from .config import MDLConfig
from .interactions import DomainAwareAttention, DomainFusedModule, FeatureInteraction
from .tokenizers import PerTokenFFN


class MDLBlock(nn.Module):
    def __init__(self, config: MDLConfig) -> None:
        super().__init__()
        num_feature_tokens = config.num_feature_tokens
        num_scenario_tokens = config.num_scenarios + 1

        self.feature_interaction = FeatureInteraction(config, num_feature_tokens)
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
