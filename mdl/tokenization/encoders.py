from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .registry import EncoderBuildContext, register_encoder


def sequence_values_and_mask(payload: Any) -> tuple[Tensor, Tensor]:
    if isinstance(payload, Tensor):
        values = payload
        return values, values.ne(0)

    values = payload["values"]
    mask = payload.get("mask")
    if mask is None and "lengths" in payload:
        positions = torch.arange(values.size(1), device=values.device).view(1, -1)
        mask = positions < payload["lengths"].view(-1, 1)
    if mask is None:
        mask = values.ne(0)
    return values, mask.to(dtype=torch.bool, device=values.device)


def masked_mean_sequence_embedding(
    embedding: nn.Embedding,
    payload: Any,
    device: torch.device,
) -> Tensor:
    values, mask = sequence_values_and_mask(payload)
    embedded = embedding(values.to(device=device))
    mask = mask.to(device=device)
    weighted = embedded * mask.unsqueeze(-1).to(dtype=embedded.dtype)
    denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=embedded.dtype)
    return weighted.sum(dim=1) / denominator


class FeatureEncoder(nn.Module):
    output_dim: int

    def forward(self, batch: dict[str, Any]) -> Tensor:
        raise NotImplementedError


@register_encoder("embedding")
class EmbeddingEncoder(FeatureEncoder):
    def __init__(self, spec: dict[str, Any], context: EncoderBuildContext) -> None:
        super().__init__()
        self.name = spec["name"]
        cardinality = int(spec.get("vocab_size", spec.get("cardinality")))
        self.output_dim = int(spec.get("embedding_dim", context.default_embedding_dim))
        self.embedding = nn.Embedding(cardinality, self.output_dim, padding_idx=0)

    def forward(self, batch: dict[str, Any]) -> Tensor:
        values = batch["features"][self.name].to(device=batch["device"], dtype=torch.long)
        return self.embedding(values)


@register_encoder("identity")
class IdentityEncoder(FeatureEncoder):
    def __init__(self, spec: dict[str, Any], context: EncoderBuildContext) -> None:
        super().__init__()
        self.name = spec["name"]
        self.output_dim = int(spec.get("dim", 1))

    def forward(self, batch: dict[str, Any]) -> Tensor:
        value = batch["features"][self.name].to(device=batch["device"], dtype=torch.float32)
        if value.ndim == 1:
            value = value.unsqueeze(1)
        if value.size(1) != self.output_dim:
            raise ValueError(
                f"identity feature {self.name!r} expected dim {self.output_dim}, got {value.size(1)}"
            )
        return value


@register_encoder("sequence_mean_pooling")
class SequenceMeanPoolingEncoder(FeatureEncoder):
    def __init__(self, spec: dict[str, Any], context: EncoderBuildContext) -> None:
        super().__init__()
        self.name = spec["name"]
        cardinality = int(spec.get("vocab_size", spec.get("cardinality")))
        self.output_dim = int(spec.get("embedding_dim", context.default_embedding_dim))
        self.embedding = nn.Embedding(cardinality, self.output_dim, padding_idx=0)

    def forward(self, batch: dict[str, Any]) -> Tensor:
        features = batch["features"]
        if self.name not in features:
            return torch.zeros(int(batch["batch_size"]), self.output_dim, device=batch["device"])
        return masked_mean_sequence_embedding(self.embedding, features[self.name], batch["device"])


