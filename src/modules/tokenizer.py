from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .embedding import DEFAULT_ENCODER_REGISTRY, EncoderBuildContext, EncoderRegistry


@dataclass(frozen=True)
class FeatureCompilerConfig:
    token_specs: list[dict[str, Any]]
    feature_specs: list[dict[str, Any]]
    embedding_dim: int = 32
    token_dim: int = 36
    hidden_dim: int = 64
    dropout: float = 0.0
    default_projection: str = "linear"


class FeatureTokenCompiler(nn.Module):
    def __init__(
        self,
        config: FeatureCompilerConfig,
        registry: EncoderRegistry = DEFAULT_ENCODER_REGISTRY,
    ) -> None:
        super().__init__()
        self.config = config
        if config.default_projection not in {"linear", "ffn_relu"}:
            raise ValueError("default_projection must be 'linear' or 'ffn_relu'")
        self.token_specs = sorted(
            config.token_specs,
            key=lambda spec: int(spec.get("token_id", len(config.token_specs))),
        )

        context = EncoderBuildContext(default_embedding_dim=config.embedding_dim)
        self.feature_names: list[str] = []
        self.feature_index: dict[str, int] = {}
        self.encoders = nn.ModuleList()
        for feature_spec in config.feature_specs:
            name = feature_spec["name"]
            if name in self.feature_index:
                raise ValueError(f"duplicate feature spec {name!r}")
            self.feature_index[name] = len(self.feature_names)
            self.feature_names.append(name)
            self.encoders.append(registry.build(feature_spec, context))

        self.projections = nn.ModuleList(
            self._build_projection(spec) for spec in self.token_specs
        )

    def _token_input_names(self, spec: dict[str, Any]) -> list[str]:
        names = []
        for input_spec in spec.get("inputs", []):
            name = input_spec if isinstance(input_spec, str) else input_spec["name"]
            if name not in self.feature_index:
                raise ValueError(f"unknown token input feature {name!r}")
            names.append(name)
        if not names:
            raise ValueError("each token spec must contain at least one input")
        return names

    def _token_input_dim(self, spec: dict[str, Any]) -> int:
        return sum(
            int(self.encoders[self.feature_index[name]].output_dim)
            for name in self._token_input_names(spec)
        )

    def _projection_name(self, spec: dict[str, Any]) -> str:
        projection = str(spec.get("projection", self.config.default_projection))
        if projection not in {"linear", "ffn_relu"}:
            raise ValueError(f"unsupported token projection {projection!r}")
        return projection

    def _build_projection(self, spec: dict[str, Any]) -> nn.Module:
        input_dim = self._token_input_dim(spec)
        projection = self._projection_name(spec)
        if projection == "linear":
            return nn.Linear(input_dim, self.config.token_dim)
        return nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.token_dim),
            nn.ReLU(),
        )

    def _batch_size_and_device(self, features: dict[str, Any]) -> tuple[int, torch.device]:
        if not features:
            raise ValueError("cannot infer batch size without features")
        first_value = next(iter(features.values()))
        if isinstance(first_value, dict):
            values = first_value["values"]
        else:
            values = first_value
        return values.size(0), values.device

    def forward(self, features: dict[str, Any]) -> Tensor:
        batch_size, device = self._batch_size_and_device(features)
        batch = {
            "features": features,
            "batch_size": batch_size,
            "device": device,
        }
        encoded_features = {
            name: encoder(batch)
            for name, encoder in zip(self.feature_names, self.encoders)
        }

        token_inputs: list[Tensor] = []
        for spec, projection in zip(self.token_specs, self.projections):
            parts = [encoded_features[name] for name in self._token_input_names(spec)]
            token_inputs.append(projection(torch.cat(parts, dim=1)).unsqueeze(1))
        return torch.cat(token_inputs, dim=1)
