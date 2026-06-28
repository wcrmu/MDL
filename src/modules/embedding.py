from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class EncoderBuildContext:
    default_embedding_dim: int


class EncoderRegistry:
    def __init__(self) -> None:
        self._builders: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            if name in self._builders:
                raise ValueError(f"encoder {name!r} is already registered")
            self._builders[name] = cls
            return cls

        return decorator

    def build(self, spec: dict[str, Any], context: EncoderBuildContext) -> Any:
        encoder_name = spec.get("encoder")
        if not encoder_name:
            raise ValueError(f"feature {spec.get('name')!r} is missing an encoder")
        if encoder_name not in self._builders:
            raise ValueError(f"unknown feature encoder {encoder_name!r}")
        return self._builders[encoder_name](spec, context)


DEFAULT_ENCODER_REGISTRY = EncoderRegistry()


def register_encoder(name: str) -> Callable[[type], type]:
    return DEFAULT_ENCODER_REGISTRY.register(name)


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


class Dice(nn.Module):
    def __init__(self, input_dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(input_dim))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        if value.size(-1) != self.alpha.numel():
            raise ValueError(
                f"Dice expected last dimension {self.alpha.numel()}, got {value.size(-1)}"
            )
        reduce_dims = tuple(range(value.ndim - 1))
        mean = value.mean(dim=reduce_dims, keepdim=True)
        variance = (value - mean).square().mean(dim=reduce_dims, keepdim=True)
        probability = torch.sigmoid((value - mean) / torch.sqrt(variance + self.eps))
        alpha = self.alpha.view(*([1] * (value.ndim - 1)), -1)
        return probability * value + (1.0 - probability) * alpha * value


class _DINActivationUnit(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: list[int],
        activation: str = "dice",
    ) -> None:
        super().__init__()
        if activation not in {"dice", "prelu", "relu"}:
            raise ValueError("din activation must be 'dice', 'prelu', or 'relu'")
        layers: list[nn.Module] = []
        previous_dim = embedding_dim * 4
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            if activation == "dice":
                layers.append(Dice(hidden_dim))
            elif activation == "prelu":
                layers.append(nn.PReLU())
            else:
                layers.append(nn.ReLU())
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, sequence_embeddings: Tensor, target_embeddings: Tensor) -> Tensor:
        expanded_target = target_embeddings.unsqueeze(1).expand_as(sequence_embeddings)
        activation_input = torch.cat(
            [
                sequence_embeddings,
                expanded_target,
                sequence_embeddings - expanded_target,
                sequence_embeddings * expanded_target,
            ],
            dim=-1,
        )
        return self.network(activation_input).squeeze(-1)


class _DINFieldEncoder(nn.Module):
    def __init__(
        self,
        spec: dict[str, Any],
        context: EncoderBuildContext,
        require_target: bool = True,
    ) -> None:
        super().__init__()
        self.require_target = require_target
        self.name = spec["name"]
        target_feature = spec.get("target_feature")
        if isinstance(target_feature, dict):
            self.target_feature = target_feature["name"]
        else:
            self.target_feature = target_feature
        if self.require_target and not self.target_feature:
            raise ValueError(f"din sequence field {self.name!r} must declare target_feature")

        self.encoder = spec.get("encoder", "embedding")
        if self.encoder == "embedding":
            cardinality = int(spec.get("vocab_size", spec.get("cardinality")))
            self.output_dim = int(spec.get("embedding_dim", context.default_embedding_dim))
            self.embedding = nn.Embedding(cardinality, self.output_dim, padding_idx=0)
            self.projection = None
        elif self.encoder in {"identity", "numeric_projection"}:
            self.input_dim = int(spec.get("dim", 1))
            self.output_dim = int(spec.get("projection_dim", self.input_dim))
            self.embedding = None
            self.projection = (
                nn.Linear(self.input_dim, self.output_dim)
                if self.output_dim != self.input_dim
                else nn.Identity()
            )
        else:
            raise ValueError(
                f"din sequence field {self.name!r} has unsupported encoder {self.encoder!r}"
            )

    def _target_values(self, batch: dict[str, Any]) -> Tensor:
        if not self.target_feature:
            raise ValueError(f"sequence field {self.name!r} does not declare target_feature")
        features = batch["features"]
        if self.target_feature not in features:
            raise ValueError(
                f"din sequence field {self.name!r} target_feature {self.target_feature!r} "
                "is missing from batch features"
            )
        target = features[self.target_feature]
        if isinstance(target, dict):
            target = target["values"]
        return target.to(device=batch["device"])

    def _encode_embedding_sequence(self, payload: Any, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        if self.embedding is None:
            raise RuntimeError("embedding module is not initialized")
        values, mask = sequence_values_and_mask(payload)
        values = values.to(device=batch["device"], dtype=torch.long)
        mask = mask.to(device=batch["device"])
        return self.embedding(values), mask

    def _encode_embedding_target(self, batch: dict[str, Any]) -> Tensor:
        if self.embedding is None:
            raise RuntimeError("embedding module is not initialized")
        target = self._target_values(batch).to(dtype=torch.long)
        if target.ndim == 2:
            if target.size(1) != 1:
                raise ValueError(
                    f"din target_feature {self.target_feature!r} must be scalar, got {tuple(target.shape)}"
                )
            target = target.squeeze(1)
        if target.ndim != 1:
            raise ValueError(
                f"din target_feature {self.target_feature!r} must have shape [batch], got {tuple(target.shape)}"
            )
        return self.embedding(target)

    def _project_numeric(self, value: Tensor) -> Tensor:
        if self.projection is None:
            raise RuntimeError("projection module is not initialized")
        if value.ndim == 2 and self.input_dim == 1:
            value = value.unsqueeze(-1)
        if value.size(-1) != self.input_dim:
            raise ValueError(
                f"din numeric field {self.name!r} expected dim {self.input_dim}, "
                f"got {value.size(-1)}"
            )
        return self.projection(value.to(dtype=torch.float32))

    def _encode_numeric_sequence(self, payload: Any, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        values, mask = sequence_values_and_mask(payload)
        values = values.to(device=batch["device"])
        mask = mask.to(device=batch["device"])
        encoded = self._project_numeric(values)
        return encoded, mask

    def _encode_numeric_target(self, batch: dict[str, Any]) -> Tensor:
        if self.projection is None:
            raise RuntimeError("projection module is not initialized")
        target = self._target_values(batch).to(device=batch["device"], dtype=torch.float32)
        if target.ndim == 1:
            target = target.unsqueeze(1)
        if target.ndim != 2:
            raise ValueError(
                f"din numeric target_feature {self.target_feature!r} must have shape [batch, dim], "
                f"got {tuple(target.shape)}"
            )
        if target.size(-1) != self.input_dim:
            raise ValueError(
                f"din numeric target_feature {self.target_feature!r} expected dim {self.input_dim}, "
                f"got {target.size(-1)}"
            )
        return self.projection(target)

    def encode_sequence(self, payload: Any, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        if self.encoder == "embedding":
            return self._encode_embedding_sequence(payload, batch)
        return self._encode_numeric_sequence(payload, batch)

    def encode_target(self, batch: dict[str, Any]) -> Tensor:
        if self.encoder == "embedding":
            return self._encode_embedding_target(batch)
        return self._encode_numeric_target(batch)

    def forward(self, payload: Any, batch: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        sequence_embeddings, mask = self.encode_sequence(payload, batch)
        target_embeddings = self.encode_target(batch)
        return sequence_embeddings, target_embeddings, mask


def _sequence_field_specs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    if "sequence_features" in spec:
        field_specs = list(spec["sequence_features"])
    else:
        field_spec = {
            "name": spec["name"],
            "encoder": spec.get("sequence_encoder", "embedding"),
            "vocab_size": spec.get("vocab_size", spec.get("cardinality")),
        }
        if spec.get("embedding_dim") is not None:
            field_spec["embedding_dim"] = spec["embedding_dim"]
        if spec.get("dim") is not None:
            field_spec["dim"] = spec["dim"]
        if spec.get("projection_dim") is not None:
            field_spec["projection_dim"] = spec["projection_dim"]
        if spec.get("target_feature") is not None:
            field_spec["target_feature"] = spec["target_feature"]
        field_specs = [field_spec]
    if not field_specs:
        raise ValueError(f"sequence feature {spec['name']!r} must declare at least one sequence field")
    return field_specs


@register_encoder("sequence_mean_pooling")
class SequenceMeanPoolingEncoder(FeatureEncoder):
    def __init__(self, spec: dict[str, Any], context: EncoderBuildContext) -> None:
        super().__init__()
        self.name = spec["name"]
        self.fusion = spec.get("fusion", "concat")
        if self.fusion != "concat":
            raise ValueError("sequence_mean_pooling currently supports only fusion='concat'")
        field_specs = _sequence_field_specs(spec)
        self.field_encoders = nn.ModuleList(
            _DINFieldEncoder(field_spec, context, require_target=False)
            for field_spec in field_specs
        )
        self.output_dim = sum(int(field_encoder.output_dim) for field_encoder in self.field_encoders)

    def _combine_mask(self, masks: list[Tensor]) -> Tensor:
        mask = masks[0]
        for current_mask in masks[1:]:
            if current_mask.shape != mask.shape:
                raise ValueError(
                    "all sequence_mean_pooling fields must have the same padded shape, got "
                    f"{tuple(mask.shape)} and {tuple(current_mask.shape)}"
                )
            mask = mask & current_mask
        return mask

    def forward(self, batch: dict[str, Any]) -> Tensor:
        features = batch["features"]
        sequence_parts: list[Tensor] = []
        masks: list[Tensor] = []
        for field_encoder in self.field_encoders:
            if field_encoder.name not in features:
                return torch.zeros(int(batch["batch_size"]), self.output_dim, device=batch["device"])
            sequence_part, mask = field_encoder.encode_sequence(features[field_encoder.name], batch)
            sequence_parts.append(sequence_part)
            masks.append(mask)

        sequence_embeddings = torch.cat(sequence_parts, dim=-1)
        mask = self._combine_mask(masks)
        weighted = sequence_embeddings * mask.unsqueeze(-1).to(dtype=sequence_embeddings.dtype)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=sequence_embeddings.dtype)
        return weighted.sum(dim=1) / denominator


@register_encoder("din")
class DINSequenceEncoder(FeatureEncoder):
    def __init__(self, spec: dict[str, Any], context: EncoderBuildContext) -> None:
        super().__init__()
        self.name = spec["name"]
        self.fusion = spec.get("fusion", "concat")
        if self.fusion != "concat":
            raise ValueError("din currently supports only fusion='concat'")
        self.attention_normalization = spec.get("attention_normalization", "none")
        if self.attention_normalization not in {"none", "softmax"}:
            raise ValueError("din attention_normalization must be 'none' or 'softmax'")

        field_specs = self._field_specs(spec)
        self.field_encoders = nn.ModuleList(_DINFieldEncoder(field_spec, context) for field_spec in field_specs)
        self.output_dim = sum(int(field_encoder.output_dim) for field_encoder in self.field_encoders)

        hidden_dims = [int(dim) for dim in spec.get("attention_hidden_dims", [80, 40])]
        activation = str(spec.get("activation", "dice"))
        self.activation_unit = _DINActivationUnit(
            self.output_dim,
            hidden_dims,
            activation=activation,
        )

    def _field_specs(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        field_specs = _sequence_field_specs(spec)
        for field_spec in field_specs:
            if not field_spec.get("target_feature"):
                raise ValueError(
                    f"din sequence field {field_spec['name']!r} must declare target_feature"
                )
        return field_specs

    def _combine_mask(self, masks: list[Tensor]) -> Tensor:
        mask = masks[0]
        for current_mask in masks[1:]:
            if current_mask.shape != mask.shape:
                raise ValueError(
                    "all din sequence fields must have the same padded shape, got "
                    f"{tuple(mask.shape)} and {tuple(current_mask.shape)}"
                )
            mask = mask & current_mask
        return mask

    def forward(self, batch: dict[str, Any]) -> Tensor:
        features = batch["features"]
        sequence_parts: list[Tensor] = []
        target_parts: list[Tensor] = []
        masks: list[Tensor] = []
        for field_encoder in self.field_encoders:
            if field_encoder.name not in features:
                return torch.zeros(int(batch["batch_size"]), self.output_dim, device=batch["device"])
            sequence_part, target_part, mask = field_encoder(features[field_encoder.name], batch)
            sequence_parts.append(sequence_part)
            target_parts.append(target_part)
            masks.append(mask)

        sequence_embeddings = torch.cat(sequence_parts, dim=-1)
        target_embeddings = torch.cat(target_parts, dim=-1)
        mask = self._combine_mask(masks)
        scores = self.activation_unit(sequence_embeddings, target_embeddings)

        if self.attention_normalization == "softmax":
            masked_scores = scores.masked_fill(~mask, -1e9)
            weights = torch.softmax(masked_scores, dim=1) * mask.to(dtype=scores.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            weights = scores * mask.to(dtype=scores.dtype)
        return (sequence_embeddings * weights.unsqueeze(-1)).sum(dim=1)


@register_encoder("sim")
class SIMSequenceEncoder(DINSequenceEncoder):
    def __init__(self, spec: dict[str, Any], context: EncoderBuildContext) -> None:
        super().__init__(spec, context)
        self.search_top_k = int(spec.get("top_k", spec.get("search_top_k", 50)))
        if self.search_top_k <= 0:
            raise ValueError("sim top_k must be positive")

    def forward(self, batch: dict[str, Any]) -> Tensor:
        features = batch["features"]
        sequence_parts: list[Tensor] = []
        target_parts: list[Tensor] = []
        masks: list[Tensor] = []
        for field_encoder in self.field_encoders:
            if field_encoder.name not in features:
                return torch.zeros(int(batch["batch_size"]), self.output_dim, device=batch["device"])
            sequence_part, target_part, mask = field_encoder(features[field_encoder.name], batch)
            sequence_parts.append(sequence_part)
            target_parts.append(target_part)
            masks.append(mask)

        sequence_embeddings = torch.cat(sequence_parts, dim=-1)
        target_embeddings = torch.cat(target_parts, dim=-1)
        if sequence_embeddings.size(1) == 0:
            return torch.zeros(int(batch["batch_size"]), self.output_dim, device=batch["device"])

        mask = self._combine_mask(masks)
        search_scores = (sequence_embeddings * target_embeddings.unsqueeze(1)).sum(dim=-1)
        masked_search_scores = search_scores.masked_fill(~mask, -1e9)
        top_k = min(self.search_top_k, sequence_embeddings.size(1))
        _top_scores, top_indices = torch.topk(masked_search_scores, top_k, dim=1)
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, sequence_embeddings.size(-1))
        selected_embeddings = sequence_embeddings.gather(1, gather_index)
        selected_mask = mask.gather(1, top_indices)

        scores = self.activation_unit(selected_embeddings, target_embeddings)
        if self.attention_normalization == "softmax":
            masked_scores = scores.masked_fill(~selected_mask, -1e9)
            weights = torch.softmax(masked_scores, dim=1) * selected_mask.to(dtype=scores.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            weights = scores * selected_mask.to(dtype=scores.dtype)
        return (selected_embeddings * weights.unsqueeze(-1)).sum(dim=1)


@register_encoder("longer")
class LongerSequenceEncoder(SIMSequenceEncoder):
    pass
