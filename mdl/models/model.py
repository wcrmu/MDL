from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn

from .blocks import MDLBlock
from .config import MDLConfig
from .tokenizers import ContextTokenizer


class MDLModel(nn.Module):
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


def count_parameters(modules: Iterable[nn.Module]) -> int:
    return sum(parameter.numel() for module in modules for parameter in module.parameters())
