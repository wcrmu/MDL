from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .config import AppConfig, DomainTokenConfig, FeatureConfig, ResolvedEncoding, SequenceConfig, TokenGroupConfig
from .modules.attention import DomainAwareAttention, DomainFusedModule, RankMixerTokenMixing, _sdpa_context
from .modules.mlp import PerTokenFFN


@dataclass(frozen=True)
class ModelMetadata:
    feature_token_count: int
    scenario_count: int
    task_count: int


@dataclass(frozen=True)
class OneTransOutput:
    feature_tokens: Tensor
    encoded_features: dict[str, Tensor]
    s_token_count: int
    ns_token_count: int
    s_valid_mask: Tensor


@dataclass(frozen=True)
class OneTransRequestCache:
    s_tokens: Tensor
    s_valid_mask: Tensor


def _encoding_for(config: AppConfig, feature_name: str) -> ResolvedEncoding:
    try:
        return config.resolved.categorical_input_by_name[feature_name].encoding
    except KeyError as error:
        raise ValueError(
            f"categorical feature or sequence field {feature_name!r} must declare encoding"
        ) from error


def _embedding_size(encoding: ResolvedEncoding, vocab_maps: dict[str, dict[str, int]], feature_name: str) -> int:
    if encoding.encoding == "hash":
        return encoding.num_buckets + 1
    if encoding.encoding == "identity":
        return encoding.max_id + 1
    if encoding.encoding in {"vocab", "shared_vocab"}:
        values = vocab_maps.get(feature_name, {})
        return max(values.values(), default=0) + 1
    raise ValueError(f"unsupported encoding {encoding.encoding!r}")


def _scenario_mask_from_ids(scenario_id: Tensor, scenario_count: int) -> Tensor:
    if scenario_count <= 0:
        raise ValueError("scenario_count must be positive")
    if scenario_id.ndim == 2:
        if scenario_id.size(1) != scenario_count:
            raise ValueError(
                f"scenario mask width must be {scenario_count}, got {scenario_id.size(1)}"
            )
        mask = scenario_id.float()
        invalid = (mask < 0.0) | (mask > 1.0) | ((mask != 0.0) & (mask != 1.0))
        if bool(invalid.any().item()):
            raise ValueError("scenario mask must be binary with shape [batch, num_scenarios]")
        return mask
    if scenario_id.ndim != 1:
        raise ValueError("scenario_id must have shape [batch] or [batch, num_scenarios]")
    indices = scenario_id.long().view(-1, 1)
    invalid = (indices < 0) | (indices >= scenario_count)
    if bool(invalid.any().item()):
        examples = indices[invalid][:5].detach().cpu().tolist()
        raise ValueError(f"scenario_id contains ids outside [0, {scenario_count - 1}]: {examples}")
    mask = torch.zeros(indices.size(0), scenario_count, device=indices.device)
    mask.scatter_(1, indices, 1.0)
    return mask


def _normal_parameter(shape: tuple[int, ...], std: float) -> nn.Parameter:
    parameter = nn.Parameter(torch.empty(*shape))
    nn.init.normal_(parameter, mean=0.0, std=std)
    return parameter


def _init_embedding(embedding: nn.Embedding, std: float) -> nn.Embedding:
    nn.init.normal_(embedding.weight, mean=0.0, std=std)
    if embedding.padding_idx is not None:
        with torch.no_grad():
            embedding.weight[embedding.padding_idx].zero_()
    return embedding


def _activation_module(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"unsupported activation {name!r}")


def _categorical_input_dims(config: AppConfig, embedding_dim: int) -> dict[str, int]:
    return dict(config.resolved.categorical_embedding_dims)


class TaskHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _activation_module(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


def _build_task_heads(config: AppConfig, input_dim: int, task_count: int) -> nn.ModuleList:
    hidden_dim = config.model.task_head_hidden_dim or config.model.hidden_dim
    return nn.ModuleList(
        TaskHead(
            input_dim,
            hidden_dim,
            config.model.task_head_dropout,
            config.model.task_head_activation,
        )
        for _ in range(task_count)
    )


@dataclass(frozen=True)
class LongerSequenceCache:
    merged_tokens: Tensor
    merged_mask: Tensor


class LongerSequenceAttentionBlock(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        hidden_dim: int,
        attention_backend: str = "auto",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.attention_backend = attention_backend
        self.query_norm = nn.LayerNorm(token_dim)
        self.key_norm = nn.LayerNorm(token_dim)
        self.query_projection = nn.Linear(token_dim, token_dim)
        self.key_projection = nn.Linear(token_dim, token_dim)
        self.value_projection = nn.Linear(token_dim, token_dim)
        self.output_projection = nn.Linear(token_dim, token_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
        )

    def _split_heads(self, tokens: Tensor) -> Tensor:
        batch_size, token_count, _dim = tokens.shape
        return tokens.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, tokens: Tensor) -> Tensor:
        batch_size, _heads, token_count, _dim = tokens.shape
        return tokens.transpose(1, 2).contiguous().view(batch_size, token_count, self.token_dim)

    def _nonempty_mask(self, allowed_mask: Tensor) -> Tensor:
        if allowed_mask.size(-1) == 0:
            return allowed_mask
        empty = ~allowed_mask.any(dim=-1, keepdim=True)
        fallback = torch.zeros_like(allowed_mask)
        fallback[..., 0:1] = True
        return allowed_mask | (empty & fallback)

    def forward(self, query_tokens: Tensor, key_tokens: Tensor, allowed_mask: Tensor) -> Tensor:
        expected_mask_shape = (query_tokens.size(0), query_tokens.size(1), key_tokens.size(1))
        if tuple(allowed_mask.shape) != expected_mask_shape:
            raise ValueError(f"attention mask shape must be {expected_mask_shape}, got {tuple(allowed_mask.shape)}")
        query_input = self.query_norm(query_tokens)
        key_input = self.key_norm(key_tokens)
        query = self._split_heads(self.query_projection(query_input))
        key = self._split_heads(self.key_projection(key_input))
        value = self._split_heads(self.value_projection(key_input))
        dropout_p = self.dropout.p if self.training else 0.0
        with _sdpa_context(self.attention_backend):
            attended = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=self._nonempty_mask(allowed_mask).unsqueeze(1),
                dropout_p=dropout_p,
            )
        hidden = query_tokens + self.output_projection(self._merge_heads(attended))
        return hidden + self.ffn(self.ffn_norm(hidden))


class LongerTokenMerger(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        hidden_dim: int,
        merge_size: int,
        inner_layers: int,
        attention_backend: str = "auto",
    ) -> None:
        super().__init__()
        if merge_size <= 0:
            raise ValueError("merge_size must be positive")
        if inner_layers < 0:
            raise ValueError("inner_layers must be non-negative")
        self.merge_size = merge_size
        self.concat_projection = nn.Linear(merge_size * token_dim, token_dim) if merge_size > 1 else nn.Identity()
        self.inner_blocks = nn.ModuleList(
            LongerSequenceAttentionBlock(
                token_dim,
                num_heads,
                hidden_dim,
                attention_backend=attention_backend,
            )
            for _ in range(inner_layers)
        )

    def _left_pad(self, tokens: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        remainder = tokens.size(1) % self.merge_size
        if remainder == 0:
            return tokens, mask
        pad = self.merge_size - remainder
        token_pad = tokens.new_zeros(tokens.size(0), pad, tokens.size(2))
        mask_pad = torch.zeros(tokens.size(0), pad, dtype=torch.bool, device=tokens.device)
        return torch.cat([token_pad, tokens], dim=1), torch.cat([mask_pad, mask], dim=1)

    def forward(self, tokens: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if self.merge_size == 1:
            return tokens, mask
        tokens, mask = self._left_pad(tokens, mask)
        batch_size, length, token_dim = tokens.shape
        group_count = length // self.merge_size
        grouped = tokens.view(batch_size, group_count, self.merge_size, token_dim)
        group_mask = mask.view(batch_size, group_count, self.merge_size)
        merged_mask = group_mask.any(dim=-1)

        if self.inner_blocks:
            hidden = grouped.reshape(batch_size * group_count, self.merge_size, token_dim)
            hidden_mask = group_mask.reshape(batch_size * group_count, self.merge_size)
            allowed = hidden_mask.unsqueeze(1) & hidden_mask.unsqueeze(2)
            for block in self.inner_blocks:
                hidden = block(hidden, hidden, allowed)
            hidden = hidden * hidden_mask.unsqueeze(-1).to(dtype=hidden.dtype)
            denominator = hidden_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=hidden.dtype)
            merged = (hidden.sum(dim=1) / denominator).view(batch_size, group_count, token_dim)
        else:
            grouped = grouped * group_mask.unsqueeze(-1).to(dtype=grouped.dtype)
            merged = self.concat_projection(grouped.reshape(batch_size, group_count, self.merge_size * token_dim))

        return merged * merged_mask.unsqueeze(-1).to(dtype=merged.dtype), merged_mask


class LongerSequenceEncoder(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        hidden_dim: int,
        query_token_count: int,
        self_layers: int,
        summary_tokens: int,
        token_merge: int,
        inner_layers: int,
        attention_backend: str = "auto",
    ) -> None:
        super().__init__()
        if query_token_count <= 0:
            raise ValueError("query_token_count must be positive")
        if self_layers < 0:
            raise ValueError("self_layers must be non-negative")
        if summary_tokens <= 0:
            raise ValueError("summary_tokens must be positive")
        self.query_token_count = query_token_count
        self.summary_tokens = summary_tokens
        self.token_merger = LongerTokenMerger(
            token_dim,
            num_heads,
            hidden_dim,
            token_merge,
            inner_layers,
            attention_backend=attention_backend,
        )
        self.cross_block = LongerSequenceAttentionBlock(
            token_dim,
            num_heads,
            hidden_dim,
            attention_backend=attention_backend,
        )
        self.self_blocks = nn.ModuleList(
            LongerSequenceAttentionBlock(
                token_dim,
                num_heads,
                hidden_dim,
                attention_backend=attention_backend,
            )
            for _ in range(self_layers)
        )

    def precompute_cache(self, tokens: Tensor, mask: Tensor) -> LongerSequenceCache:
        merged_tokens, merged_mask = self.token_merger(tokens, mask)
        return LongerSequenceCache(merged_tokens=merged_tokens, merged_mask=merged_mask)

    def _cross_allowed_mask(self, key_valid_mask: Tensor, sampled_mask: Tensor, full_length: int) -> Tensor:
        sample_count = sampled_mask.size(1)
        query_count = self.summary_tokens + sample_count
        allowed = key_valid_mask.unsqueeze(1).expand(-1, query_count, -1).clone()
        if sample_count == 0:
            return allowed

        device = key_valid_mask.device
        key_positions = torch.arange(full_length, device=device).view(1, 1, full_length)
        sampled_positions = torch.arange(full_length - sample_count, full_length, device=device).view(1, sample_count, 1)
        sequence_queries = allowed[:, self.summary_tokens :, :]
        sequence_queries[:, :, : self.summary_tokens] = False
        sequence_queries[:, :, self.summary_tokens :] &= key_positions <= sampled_positions
        sequence_queries &= sampled_mask.unsqueeze(-1)
        allowed[:, self.summary_tokens :, :] = sequence_queries
        return allowed

    def _self_allowed_mask(self, valid_mask: Tensor) -> Tensor:
        token_count = valid_mask.size(1)
        allowed = valid_mask.unsqueeze(1).expand(-1, token_count, -1).clone()
        sequence_count = token_count - self.summary_tokens
        if sequence_count <= 0:
            return allowed

        device = valid_mask.device
        key_positions = torch.arange(sequence_count, device=device).view(1, 1, sequence_count)
        query_positions = torch.arange(sequence_count, device=device).view(1, sequence_count, 1)
        sequence_queries = allowed[:, self.summary_tokens :, :]
        sequence_queries[:, :, : self.summary_tokens] = False
        sequence_queries[:, :, self.summary_tokens :] &= key_positions <= query_positions
        sequence_queries &= valid_mask[:, self.summary_tokens :].unsqueeze(-1)
        allowed[:, self.summary_tokens :, :] = sequence_queries
        return allowed

    def forward(
        self,
        tokens: Tensor,
        mask: Tensor,
        global_tokens: Tensor,
        cache: LongerSequenceCache | None = None,
    ) -> Tensor:
        if global_tokens.size(1) != self.summary_tokens or global_tokens.size(2) != tokens.size(2):
            raise ValueError(
                f"global_tokens must have shape [batch, {self.summary_tokens}, token_dim]"
            )
        if tokens.size(1) == 0:
            return global_tokens.flatten(start_dim=1)

        sequence_cache = self.precompute_cache(tokens, mask) if cache is None else cache
        merged_tokens = sequence_cache.merged_tokens
        merged_mask = sequence_cache.merged_mask
        sample_count = min(self.query_token_count, merged_tokens.size(1))
        sampled_tokens = merged_tokens[:, -sample_count:, :]
        sampled_mask = merged_mask[:, -sample_count:]
        key_tokens = torch.cat([global_tokens, merged_tokens], dim=1)
        key_valid_mask = torch.cat(
            [
                torch.ones(tokens.size(0), self.summary_tokens, dtype=torch.bool, device=tokens.device),
                merged_mask,
            ],
            dim=1,
        )
        query_tokens = torch.cat([global_tokens, sampled_tokens], dim=1)
        query_valid_mask = torch.cat(
            [
                torch.ones(tokens.size(0), self.summary_tokens, dtype=torch.bool, device=tokens.device),
                sampled_mask,
            ],
            dim=1,
        )

        hidden = self.cross_block(
            query_tokens,
            key_tokens,
            self._cross_allowed_mask(key_valid_mask, sampled_mask, merged_tokens.size(1)),
        )
        for block in self.self_blocks:
            hidden = block(hidden, hidden, self._self_allowed_mask(query_valid_mask))
        return hidden[:, : self.summary_tokens, :].flatten(start_dim=1)


class FeatureEncoderBank(nn.Module):
    def __init__(self, config: AppConfig, vocab_maps: dict[str, dict[str, int]], embedding_dim: int) -> None:
        super().__init__()
        self.config = config
        self.sequence_token_dim = config.model.token_dim
        self.output_dims: dict[str, int] = {}
        self.embeddings = nn.ModuleDict()
        self.sequence_field_embedding_keys: dict[str, str] = {}
        self.sequence_step_projectors = nn.ModuleDict()
        self.sequence_query_projectors = nn.ModuleDict()
        self.sequence_queries = nn.ParameterDict()
        self.sequence_position_embeddings = nn.ModuleDict()
        self.sequence_longer_encoders = nn.ModuleDict()
        self.sequences_by_name = {sequence.name: sequence for sequence in config.sequences}
        categorical_dims = _categorical_input_dims(config, embedding_dim)
        sparse_gradients = config.training.embedding_sparse_gradients

        for feature in config.features:
            if feature.kind == "dense":
                self.output_dims[feature.name] = feature.dimension
            elif feature.kind == "categorical":
                self.output_dims[feature.name] = categorical_dims[feature.name]

        for sequence in config.sequences:
            for field in sequence.fields:
                if field.kind == "categorical":
                    qualified = field.qualified_name(sequence.name)
                    self.sequence_field_embedding_keys[qualified] = self._module_key(qualified)

        for feature in config.features:
            if feature.kind == "dense":
                continue
            encoding = _encoding_for(config, feature.name)
            if encoding.encoding == "shared_vocab" and encoding.share_embedding:
                continue
            feature_embedding_dim = categorical_dims[feature.name]
            size = _embedding_size(encoding, vocab_maps, feature.name)
            self.embeddings[feature.name] = _init_embedding(
                nn.Embedding(
                    size,
                    feature_embedding_dim,
                    padding_idx=0,
                    sparse=sparse_gradients,
                ),
                config.model.init_std,
            )

        for sequence in config.sequences:
            step_input_dim = 0
            for field in sequence.fields:
                qualified = field.qualified_name(sequence.name)
                if field.kind == "categorical":
                    key = self.sequence_field_embedding_keys[qualified]
                    encoding = _encoding_for(config, qualified)
                    field_embedding_dim = categorical_dims[qualified]
                    if encoding.encoding == "shared_vocab" and encoding.share_embedding:
                        step_input_dim += field_embedding_dim
                        continue
                    size = _embedding_size(encoding, vocab_maps, qualified)
                    self.embeddings[key] = _init_embedding(
                        nn.Embedding(
                            size,
                            field_embedding_dim,
                            padding_idx=0,
                            sparse=sparse_gradients,
                        ),
                        config.model.init_std,
                    )
                    step_input_dim += field_embedding_dim
                else:
                    step_input_dim += field.dimension
            self.sequence_step_projectors[self._module_key(sequence.name)] = nn.Linear(
                step_input_dim,
                self.sequence_token_dim,
            )
            if sequence.max_length is not None:
                self.sequence_position_embeddings[self._module_key(sequence.name)] = _init_embedding(
                    nn.Embedding(
                        sequence.max_length,
                        self.sequence_token_dim,
                    ),
                    config.model.init_std,
                )
            sequence_key = self._module_key(sequence.name)
            summary_dim = sequence.rankmixer_summary_tokens * self.sequence_token_dim
            if sequence.target_inputs:
                target_dim = sum(self.output_dims[name] for name in sequence.target_inputs)
                self.sequence_query_projectors[sequence_key] = nn.Linear(
                    target_dim,
                    summary_dim,
                )
            else:
                self.sequence_queries[sequence_key] = _normal_parameter(
                    (1, sequence.rankmixer_summary_tokens, self.sequence_token_dim),
                    config.model.init_std,
                )
            if sequence.encoder == "longer":
                self.sequence_longer_encoders[sequence_key] = LongerSequenceEncoder(
                    self.sequence_token_dim,
                    config.model.num_heads,
                    config.model.hidden_dim,
                    sequence.longer_query_tokens,
                    sequence.longer_self_layers,
                    sequence.rankmixer_summary_tokens,
                    sequence.longer_token_merge,
                    sequence.longer_inner_layers,
                    attention_backend=config.runtime.attention_backend,
                )
            self.output_dims[sequence.name] = summary_dim

        for feature in config.features:
            if feature.kind != "categorical":
                continue
            encoding = _encoding_for(config, feature.name)
            if encoding.encoding == "shared_vocab" and encoding.share_embedding:
                if feature.name not in categorical_dims:
                    raise ValueError(f"shared_vocab feature {feature.name!r} references unknown feature")
                base_name = self._shared_base_name(feature.name)
                base_key = self._embedding_key(base_name)
                if base_key not in self.embeddings:
                    raise ValueError(f"shared_vocab base {base_name!r} has no embedding")
                self.embeddings[feature.name] = self.embeddings[base_key]
                self.output_dims[feature.name] = categorical_dims[feature.name]

        for sequence in config.sequences:
            for field in sequence.fields:
                if field.kind != "categorical":
                    continue
                qualified = field.qualified_name(sequence.name)
                encoding = _encoding_for(config, qualified)
                if encoding.encoding == "shared_vocab" and encoding.share_embedding:
                    if qualified not in categorical_dims:
                        raise ValueError(f"shared_vocab sequence field {qualified!r} references unknown feature")
                    base_name = self._shared_base_name(qualified)
                    base_key = self._embedding_key(base_name)
                    if base_key not in self.embeddings:
                        raise ValueError(f"shared_vocab base {base_name!r} has no embedding")
                    self.embeddings[self.sequence_field_embedding_keys[qualified]] = self.embeddings[base_key]

    @staticmethod
    def _module_key(name: str) -> str:
        return name.replace(".", "__")

    def _embedding_key(self, name: str) -> str:
        return self.sequence_field_embedding_keys.get(name, name)

    def _shared_base_name(self, name: str) -> str:
        seen: set[str] = set()
        current = name
        while True:
            if current in seen:
                raise ValueError(f"shared_vocab cycle detected at {name!r}")
            seen.add(current)
            encoding = _encoding_for(self.config, current)
            if encoding.encoding != "shared_vocab" or not encoding.share_embedding:
                return current
            current = encoding.share_with

    def _encode_scalar_feature(self, feature: FeatureConfig, value: Tensor) -> Tensor:
        if feature.kind == "dense":
            dense = value.float().view(value.size(0), -1)
            if dense.size(1) != feature.dimension:
                raise ValueError(
                    f"dense feature {feature.name!r} expected dimension {feature.dimension}, "
                    f"got {dense.size(1)}"
                )
            return dense
        if feature.kind == "categorical":
            return self.embeddings[feature.name](value.long())
        raise ValueError(f"feature {feature.name!r} is not scalar")

    def _right_aligned_sequence(self, embedded: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        if embedded.size(1) == 0:
            mask = torch.zeros(embedded.size(0), 0, dtype=torch.bool, device=embedded.device)
            return embedded, mask
        max_length = embedded.size(1)
        positions = torch.arange(max_length, device=embedded.device).view(1, -1)
        shifts = (max_length - lengths).clamp_min(0).view(-1, 1)
        source_positions = (positions - shifts).clamp(min=0, max=max_length - 1)
        gather_index = source_positions.unsqueeze(-1).expand(-1, -1, embedded.size(-1))
        mask = positions >= shifts
        aligned = embedded.gather(1, gather_index) * mask.unsqueeze(-1).to(dtype=embedded.dtype)
        return aligned, mask

    def _multi_field_sequence_tokens(
        self,
        sequence: SequenceConfig,
        value: dict[str, Any],
    ) -> tuple[Tensor, Tensor]:
        field_values = value["fields"]
        lengths = value["lengths"].long()
        parts: list[Tensor] = []
        for field in sequence.fields:
            tensor = field_values[field.name]
            if field.kind == "categorical":
                qualified = field.qualified_name(sequence.name)
                parts.append(self.embeddings[self.sequence_field_embedding_keys[qualified]](tensor.long()))
            else:
                dense = tensor.float()
                if dense.dim() == 2:
                    dense = dense.unsqueeze(-1)
                parts.append(dense)
        if not parts:
            raise ValueError(f"sequence {sequence.name!r} has no fields")
        step_inputs = torch.cat(parts, dim=-1)
        tokens = self.sequence_step_projectors[self._module_key(sequence.name)](step_inputs)
        tokens, mask = self._right_aligned_sequence(tokens, lengths)
        position_key = self._module_key(sequence.name)
        if position_key in self.sequence_position_embeddings and tokens.size(1) > 0:
            max_positions = self.sequence_position_embeddings[position_key].num_embeddings
            positions = torch.arange(tokens.size(1), device=tokens.device).clamp(max=max_positions - 1)
            tokens = tokens + self.sequence_position_embeddings[position_key](positions).unsqueeze(0)
            tokens = tokens * mask.unsqueeze(-1).to(dtype=tokens.dtype)
        return tokens, mask

    def encode_sequence_tokens(
        self,
        sequence_name: str,
        value: dict[str, Any],
    ) -> tuple[Tensor, Tensor]:
        sequence = self.sequences_by_name[sequence_name]
        return self._multi_field_sequence_tokens(sequence, value)

    def _pool_sequence(
        self,
        sequence: SequenceConfig,
        tokens: Tensor,
        mask: Tensor,
        encoded: dict[str, Tensor],
        sequence_cache: LongerSequenceCache | None = None,
    ) -> Tensor:
        output_dim = self.output_dims[sequence.name]
        if tokens.size(1) == 0:
            return tokens.new_zeros(tokens.size(0), output_dim)
        mask_float = mask.unsqueeze(-1).to(dtype=tokens.dtype)
        if sequence.encoder == "mean_pool":
            denominator = mask_float.sum(dim=1).clamp_min(1.0)
            return (tokens * mask_float).sum(dim=1) / denominator

        sequence_key = self._module_key(sequence.name)
        if sequence.target_inputs:
            query_input = torch.cat([encoded[name] for name in sequence.target_inputs], dim=1)
            query = self.sequence_query_projectors[sequence_key](query_input).view(
                tokens.size(0),
                sequence.rankmixer_summary_tokens,
                self.sequence_token_dim,
            )
        else:
            query = self.sequence_queries[sequence_key].expand(tokens.size(0), -1, -1)
        if sequence.encoder == "longer":
            return self.sequence_longer_encoders[sequence_key](tokens, mask, query, cache=sequence_cache)
        scores = (tokens * query[:, :1, :]).sum(dim=-1) / math.sqrt(tokens.size(-1))
        scores = scores.masked_fill(~mask, -1.0e9)
        weights = torch.softmax(scores, dim=1) * mask.to(dtype=tokens.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-9)
        return (tokens * weights.unsqueeze(-1)).sum(dim=1)

    def precompute_request_cache(self, features: dict[str, Any]) -> dict[str, LongerSequenceCache]:
        caches: dict[str, LongerSequenceCache] = {}
        for sequence in self.config.sequences:
            if sequence.encoder != "longer":
                continue
            value = features[sequence.name]
            if not isinstance(value, dict):
                raise ValueError(f"sequence {sequence.name!r} must be a payload dict")
            tokens, mask = self._multi_field_sequence_tokens(sequence, value)
            caches[sequence.name] = self.sequence_longer_encoders[self._module_key(sequence.name)].precompute_cache(tokens, mask)
        return caches

    def forward(
        self,
        features: dict[str, Any],
        request_cache: dict[str, LongerSequenceCache] | None = None,
    ) -> dict[str, Tensor]:
        if request_cache is None and self.config.model.use_request_cache:
            request_cache = self.precompute_request_cache(features)
        encoded: dict[str, Tensor] = {}
        for feature in self.config.features:
            value = features[feature.name]
            if feature.kind == "dense":
                if not isinstance(value, Tensor):
                    raise ValueError(f"dense feature {feature.name!r} must be a tensor")
                encoded[feature.name] = self._encode_scalar_feature(feature, value)
                continue
            if feature.kind == "categorical":
                if not isinstance(value, Tensor):
                    raise ValueError(f"categorical feature {feature.name!r} must be a tensor")
                encoded[feature.name] = self._encode_scalar_feature(feature, value)
                continue
            raise ValueError(f"unsupported feature kind {feature.kind!r}")

        for sequence in self.config.sequences:
            value = features[sequence.name]
            if not isinstance(value, dict):
                raise ValueError(f"sequence {sequence.name!r} must be a payload dict")
            tokens, mask = self._multi_field_sequence_tokens(sequence, value)
            sequence_cache = None if request_cache is None else request_cache.get(sequence.name)
            encoded[sequence.name] = self._pool_sequence(sequence, tokens, mask, encoded, sequence_cache)
        return encoded


class TokenProjector(nn.Module):
    def __init__(self, groups: list[TokenGroupConfig], input_dims: dict[str, int], token_dim: int) -> None:
        super().__init__()
        self.groups = groups
        self.projections = nn.ModuleList(
            nn.Linear(sum(input_dims[name] for name in group.inputs), token_dim)
            for group in groups
        )

    def forward(self, encoded: dict[str, Tensor]) -> Tensor:
        tokens = []
        for group, projection in zip(self.groups, self.projections):
            parts = [encoded[name] for name in group.inputs]
            tokens.append(projection(torch.cat(parts, dim=1)).unsqueeze(1))
        return torch.cat(tokens, dim=1)


class AutoSplitTokenProjector(nn.Module):
    def __init__(
        self,
        input_names: list[str],
        input_dims: dict[str, int],
        num_tokens: int,
        token_dim: int,
    ) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if not input_names:
            raise ValueError("auto_split tokenization requires at least one input feature")
        self.input_names = input_names
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        input_dim = sum(input_dims[name] for name in input_names)
        if input_dim <= 0:
            raise ValueError("auto_split tokenization input dimension must be positive")
        self.projection = nn.Linear(input_dim, num_tokens * token_dim)

    def forward(self, encoded: dict[str, Tensor]) -> Tensor:
        values = self.projection(torch.cat([encoded[name] for name in self.input_names], dim=1))
        return values.view(values.size(0), self.num_tokens, self.token_dim)


class RankMixerSliceTokenizer(nn.Module):
    def __init__(
        self,
        input_names: list[str],
        input_dims: dict[str, int],
        num_tokens: int,
        token_dim: int,
    ) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if not input_names:
            raise ValueError("rankmixer tokenization requires at least one input")
        self.input_names = input_names
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.input_dim = sum(input_dims[name] for name in input_names)
        self.target_dim = num_tokens * token_dim
        if self.input_dim != self.target_dim:
            raise ValueError(
                "rankmixer tokenization requires exact input dimension "
                "num_feature_tokens * token_dim; implicit zero padding is disabled: "
                f"{self.input_dim} != {self.target_dim}"
            )

    def forward(self, encoded: dict[str, Tensor]) -> Tensor:
        values = torch.cat([encoded[name] for name in self.input_names], dim=1)
        return values.view(values.size(0), self.num_tokens, self.token_dim)


def _build_rankmixer_feature_projector(
    config: AppConfig,
    encoder_bank: FeatureEncoderBank,
    feature_groups: list[TokenGroupConfig],
    feature_token_inputs: list[str],
    feature_token_count: int,
) -> nn.Module:
    if config.tokenization.feature_tokenizer == "auto_split":
        return AutoSplitTokenProjector(
            feature_token_inputs,
            encoder_bank.output_dims,
            feature_token_count,
            config.model.token_dim,
        )
    if config.tokenization.feature_tokenizer == "rankmixer":
        return RankMixerSliceTokenizer(
            feature_token_inputs,
            encoder_bank.output_dims,
            feature_token_count,
            config.model.token_dim,
        )
    return TokenProjector(feature_groups, encoder_bank.output_dims, config.model.token_dim)


class DomainTokenProjector(nn.Module):
    def __init__(
        self,
        tokens: list[DomainTokenConfig],
        input_dims: dict[str, int],
        token_dim: int,
        hidden_dim: int,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_names_by_token = [token.resolved_inputs() for token in tokens]
        self.networks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(sum(input_dims[name] for name in input_names), hidden_dim),
                _activation_module(activation),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, token_dim),
                # MDL tokenization applies ReLU(FFN(...)) to scenario/task tokens.
                nn.ReLU(),
            )
            for input_names in self.input_names_by_token
        )

    def forward(self, encoded: dict[str, Tensor]) -> Tensor:
        outputs = []
        for input_names, network in zip(self.input_names_by_token, self.networks):
            inputs = torch.cat([encoded[name] for name in input_names], dim=1)
            outputs.append(network(inputs))
        return torch.stack(outputs, dim=1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, values: Tensor) -> Tensor:
        scale = values.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return values * scale * self.weight


class OneTransTokenizer(nn.Module):
    def __init__(self, config: AppConfig, encoder_bank: FeatureEncoderBank) -> None:
        super().__init__()
        self.config = config
        self.encoder_bank = encoder_bank
        self.by_name = {feature.name: feature for feature in config.features}
        self.sequence_by_name = {sequence.name: sequence for sequence in config.sequences}
        self.sequence_groups = config.tokenization.resolved_sequence_tokens(config.features, config.sequences)
        self.ns_groups = config.tokenization.resolved_ns_tokens(config.features, config.sequences)
        self.token_dim = config.model.token_dim
        self.ns_tokenizer = config.model.ns_tokenizer
        self.use_sep_tokens = config.model.use_sep_tokens

        if not self.ns_groups and self.ns_tokenizer == "groupwise":
            raise ValueError("groupwise OneTrans tokenizer requires tokenization.ns_tokens or scalar features")
        if not self.sequence_groups:
            raise ValueError("OneTrans requires at least one sequence feature")

        self.sequence_projectors = nn.ModuleList(
            nn.Linear(self._group_input_dim(group), self.token_dim)
            for group in self.sequence_groups
        )
        self.sep_tokens = nn.ParameterList(
            _normal_parameter((1, 1, self.token_dim), config.model.init_std)
            for _ in range(max(len(self.sequence_groups) - 1, 0))
        )

        self.scalar_feature_names = [
            feature.name
            for feature in config.features
            if feature.embedding_scope in {"feature", "shared"}
        ]
        if self.ns_tokenizer == "auto_split":
            self.num_ns_tokens = config.model.num_ns_tokens or max(len(self.scalar_feature_names), 1)
            input_dim = sum(self.encoder_bank.output_dims[name] for name in self.scalar_feature_names)
            if input_dim <= 0:
                raise ValueError("auto_split OneTrans tokenizer requires at least one scalar feature")
            self.auto_ns_projection = nn.Linear(input_dim, self.num_ns_tokens * self.token_dim)
            self.ns_projectors = nn.ModuleList()
        else:
            self.num_ns_tokens = len(self.ns_groups)
            self.auto_ns_projection = None
            self.ns_projectors = nn.ModuleList(
                nn.Linear(self._group_input_dim(group), self.token_dim)
                for group in self.ns_groups
            )
        if self.num_ns_tokens <= 0:
            raise ValueError("OneTrans requires at least one NS token")

    def _group_input_dim(self, group: TokenGroupConfig) -> int:
        return sum(self.encoder_bank.output_dims[name] for name in group.inputs)

    def _payload_max_length(self, value: dict[str, Any]) -> int:
        fields = value.get("fields", {})
        first = next(iter(fields.values()), None)
        if first is None:
            return 0
        return int(first.size(1))

    def _group_sequence_length(self, features: dict[str, Any], group: TokenGroupConfig) -> tuple[int, Tensor]:
        sequence_lengths: list[Tensor] = []
        max_length = 0
        for name in group.inputs:
            if name in self.sequence_by_name:
                value = features[name]
                if not isinstance(value, dict):
                    raise ValueError(f"sequence {name!r} must be a payload dict")
                sequence_lengths.append(value["lengths"].long())
                max_length = max(max_length, self._payload_max_length(value))
                continue
        if not sequence_lengths:
            raise ValueError(f"sequence token group {group.name!r} must include a sequence input")
        first = sequence_lengths[0]
        for current in sequence_lengths[1:]:
            if not torch.equal(first, current):
                raise ValueError(f"sequence token group {group.name!r} has unaligned sequence lengths")
        return max_length, first

    def _sequence_group_tokens(
        self,
        group: TokenGroupConfig,
        projection: nn.Linear,
        features: dict[str, Any],
    ) -> tuple[Tensor, Tensor]:
        max_length, _lengths = self._group_sequence_length(features, group)
        parts: list[Tensor] = []
        mask: Tensor | None = None
        for name in group.inputs:
            value = features[name]
            if name in self.sequence_by_name:
                if not isinstance(value, dict):
                    raise ValueError(f"sequence {name!r} must be a payload dict")
                tokens, current_mask = self.encoder_bank.encode_sequence_tokens(name, value)
                mask = current_mask if mask is None else mask & current_mask
                parts.append(tokens)
                continue
            feature = self.by_name[name]
            if not isinstance(value, Tensor):
                raise ValueError(f"scalar feature {name!r} must be a tensor")
            scalar = self.encoder_bank._encode_scalar_feature(feature, value)
            parts.append(scalar.unsqueeze(1).expand(-1, max_length, -1))
        if mask is None:
            raise ValueError(f"sequence token group {group.name!r} produced no mask")
        return projection(torch.cat(parts, dim=-1)), mask

    def _ns_tokens_groupwise(self, encoded: dict[str, Tensor]) -> Tensor:
        tokens = []
        for group, projection in zip(self.ns_groups, self.ns_projectors):
            parts = [encoded[name] for name in group.inputs]
            tokens.append(projection(torch.cat(parts, dim=1)).unsqueeze(1))
        return torch.cat(tokens, dim=1)

    def _ns_tokens_auto_split(self, encoded: dict[str, Tensor]) -> Tensor:
        if self.auto_ns_projection is None:
            raise RuntimeError("auto NS projection is not initialized")
        parts = [encoded[name] for name in self.scalar_feature_names]
        values = self.auto_ns_projection(torch.cat(parts, dim=1))
        return values.view(values.size(0), self.num_ns_tokens, self.token_dim)

    def _sequence_token_part(self, features: dict[str, Any]) -> OneTransRequestCache:
        sequence_tokens: list[Tensor] = []
        sequence_masks: list[Tensor] = []
        for index, (group, projection) in enumerate(zip(self.sequence_groups, self.sequence_projectors)):
            tokens, mask = self._sequence_group_tokens(group, projection, features)
            sequence_tokens.append(tokens)
            sequence_masks.append(mask)
            if self.use_sep_tokens and index < len(self.sep_tokens):
                sep = self.sep_tokens[index].expand(tokens.size(0), -1, -1)
                sequence_tokens.append(sep)
                sequence_masks.append(torch.ones(tokens.size(0), 1, dtype=torch.bool, device=tokens.device))
        return OneTransRequestCache(
            s_tokens=torch.cat(sequence_tokens, dim=1),
            s_valid_mask=torch.cat(sequence_masks, dim=1),
        )

    def precompute_request_cache(self, features: dict[str, Any]) -> OneTransRequestCache:
        return self._sequence_token_part(features)

    def forward(
        self,
        features: dict[str, Any],
        request_cache: OneTransRequestCache | None = None,
    ) -> OneTransOutput:
        if request_cache is None and self.config.model.use_request_cache:
            request_cache = self.precompute_request_cache(features)
        encoded = self.encoder_bank(features)
        cache = self._sequence_token_part(features) if request_cache is None else request_cache
        s_tokens = cache.s_tokens
        s_mask = cache.s_valid_mask
        ns_tokens = (
            self._ns_tokens_auto_split(encoded)
            if self.ns_tokenizer == "auto_split"
            else self._ns_tokens_groupwise(encoded)
        )
        tokens = torch.cat([s_tokens, ns_tokens], dim=1)
        return OneTransOutput(
            feature_tokens=tokens,
            encoded_features=encoded,
            s_token_count=s_tokens.size(1),
            ns_token_count=ns_tokens.size(1),
            s_valid_mask=s_mask,
        )


class MixedCausalAttention(nn.Module):
    def __init__(self, token_dim: int, num_heads: int, ns_token_count: int, attention_backend: str = "auto") -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.attention_backend = attention_backend
        self.s_query = nn.Linear(token_dim, token_dim)
        self.s_key = nn.Linear(token_dim, token_dim)
        self.s_value = nn.Linear(token_dim, token_dim)
        self.ns_query = nn.ModuleList(nn.Linear(token_dim, token_dim) for _ in range(ns_token_count))
        self.ns_key = nn.ModuleList(nn.Linear(token_dim, token_dim) for _ in range(ns_token_count))
        self.ns_value = nn.ModuleList(nn.Linear(token_dim, token_dim) for _ in range(ns_token_count))
        self.output = nn.Linear(token_dim, token_dim)

    def _project_all(self, tokens: Tensor, s_count: int, s_layer: nn.Linear, ns_layers: nn.ModuleList) -> Tensor:
        parts: list[Tensor] = []
        if s_count > 0:
            parts.append(s_layer(tokens[:, :s_count, :]))
        for index, layer in enumerate(ns_layers):
            parts.append(layer(tokens[:, s_count + index, :]).unsqueeze(1))
        return torch.cat(parts, dim=1)

    def _project_query(self, tokens: Tensor, query_s_count: int) -> Tensor:
        parts: list[Tensor] = []
        if query_s_count > 0:
            parts.append(self.s_query(tokens[:, :query_s_count, :]))
        for index, layer in enumerate(self.ns_query):
            parts.append(layer(tokens[:, query_s_count + index, :]).unsqueeze(1))
        return torch.cat(parts, dim=1)

    def _split_heads(self, tokens: Tensor) -> Tensor:
        batch_size, token_count, _ = tokens.shape
        return tokens.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, tokens: Tensor) -> Tensor:
        batch_size, _heads, token_count, _dim = tokens.shape
        return tokens.transpose(1, 2).contiguous().view(batch_size, token_count, self.token_dim)

    def forward(
        self,
        tokens: Tensor,
        s_count: int,
        query_indices: Tensor,
        query_s_count: int,
        key_valid_mask: Tensor,
    ) -> Tensor:
        query_tokens = tokens.index_select(1, query_indices)
        query = self._split_heads(self._project_query(query_tokens, query_s_count))
        key = self._split_heads(self._project_all(tokens, s_count, self.s_key, self.ns_key))
        value = self._split_heads(self._project_all(tokens, s_count, self.s_value, self.ns_value))

        key_positions = torch.arange(tokens.size(1), device=tokens.device).view(1, -1)
        causal = key_positions <= query_indices.view(-1, 1)
        valid = causal.unsqueeze(0) & key_valid_mask.unsqueeze(1)

        with _sdpa_context(self.attention_backend):
            attended = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=valid.unsqueeze(1),
                dropout_p=0.0,
            )
        return self.output(self._merge_heads(attended))


class MixedFFN(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, ns_token_count: int) -> None:
        super().__init__()
        self.s_ffn = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.ns_ffn = nn.ModuleList(
            nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, token_dim),
            )
            for _ in range(ns_token_count)
        )

    def forward(self, tokens: Tensor, query_s_count: int) -> Tensor:
        parts: list[Tensor] = []
        if query_s_count > 0:
            parts.append(self.s_ffn(tokens[:, :query_s_count, :]))
        for index, network in enumerate(self.ns_ffn):
            parts.append(network(tokens[:, query_s_count + index, :]).unsqueeze(1))
        return torch.cat(parts, dim=1)


class OneTransBlock(nn.Module):
    def __init__(self, config: AppConfig, ns_token_count: int) -> None:
        super().__init__()
        self.norm_attention = RMSNorm(config.model.token_dim)
        self.attention = MixedCausalAttention(
            config.model.token_dim,
            config.model.num_heads,
            ns_token_count,
            config.runtime.attention_backend,
        )
        self.norm_ffn = RMSNorm(config.model.token_dim)
        self.ffn = MixedFFN(config.model.token_dim, config.model.hidden_dim, ns_token_count)

    def forward(
        self,
        tokens: Tensor,
        s_count: int,
        query_s_count: int,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        query_indices = torch.cat(
            [
                torch.arange(s_count - query_s_count, s_count, device=tokens.device),
                torch.arange(s_count, tokens.size(1), device=tokens.device),
            ]
        )
        normalized = self.norm_attention(tokens)
        residual = tokens.index_select(1, query_indices)
        attended = self.attention(normalized, s_count, query_indices, query_s_count, valid_mask)
        hidden = residual + attended
        output = hidden + self.ffn(self.norm_ffn(hidden), query_s_count)
        return output, valid_mask.index_select(1, query_indices)


class OneTransBackbone(nn.Module):
    def __init__(self, config: AppConfig, vocab_maps: dict[str, dict[str, int]], embedding_dim: int | None = None) -> None:
        super().__init__()
        self.config = config
        embedding_dim = config.model.embedding_dim if embedding_dim is None else embedding_dim
        self.encoder_bank = FeatureEncoderBank(config, vocab_maps, embedding_dim)
        self.tokenizer = OneTransTokenizer(config, self.encoder_bank)
        self.ns_token_count = self.tokenizer.num_ns_tokens
        self.blocks = nn.ModuleList(OneTransBlock(config, self.ns_token_count) for _ in range(config.model.num_layers))

    def _layer_s_count(self, initial_s_count: int, current_s_count: int, layer_index: int) -> int:
        if not self.config.model.use_pyramid or initial_s_count == 0:
            return current_s_count
        final = self.config.model.final_s_tokens
        if final is None:
            final = min(self.ns_token_count, initial_s_count)
        final = min(final, initial_s_count)
        if layer_index == self.config.model.num_layers - 1:
            target = final
        else:
            progress = float(layer_index + 1) / float(self.config.model.num_layers)
            target = round(initial_s_count - (initial_s_count - final) * progress)
        round_to = self.config.model.pyramid_round_to
        if target > round_to:
            target = max(final, int(round(target / round_to) * round_to))
        return max(0, min(current_s_count, target))

    def precompute_request_cache(self, features: dict[str, Any]) -> OneTransRequestCache:
        return self.tokenizer.precompute_request_cache(features)

    def forward(
        self,
        features: dict[str, Any],
        request_cache: OneTransRequestCache | None = None,
    ) -> OneTransOutput:
        tokenized = self.tokenizer(features, request_cache=request_cache)
        tokens = tokenized.feature_tokens
        s_count = tokenized.s_token_count
        ns_count = tokenized.ns_token_count
        valid_mask = torch.cat(
            [
                tokenized.s_valid_mask,
                torch.ones(tokens.size(0), ns_count, dtype=torch.bool, device=tokens.device),
            ],
            dim=1,
        )
        initial_s_count = s_count
        for layer_index, block in enumerate(self.blocks):
            query_s_count = self._layer_s_count(initial_s_count, s_count, layer_index)
            if self.config.runtime.activation_checkpoint and self.training:
                tokens, valid_mask = checkpoint(
                    lambda current_tokens, current_mask: block(
                        current_tokens,
                        s_count,
                        query_s_count,
                        current_mask,
                    ),
                    tokens,
                    valid_mask,
                    use_reentrant=False,
                )
            else:
                tokens, valid_mask = block(tokens, s_count, query_s_count, valid_mask)
            s_count = query_s_count
        return OneTransOutput(
            feature_tokens=tokens[:, s_count:, :],
            encoded_features=tokenized.encoded_features,
            s_token_count=s_count,
            ns_token_count=ns_count,
            s_valid_mask=valid_mask[:, :s_count],
        )


class RankMixerBlock(nn.Module):
    def __init__(self, config: AppConfig, feature_token_count: int) -> None:
        super().__init__()
        token_dim = config.model.token_dim
        self.token_mixing = RankMixerTokenMixing(feature_token_count, token_dim)
        self.feature_norm = nn.LayerNorm(token_dim)
        self.feature_ffn = PerTokenFFN(
            feature_token_count,
            token_dim,
            config.model.hidden_dim,
            activation=config.model.ffn_activation,
        )

    def forward(self, feature_tokens: Tensor) -> Tensor:
        mixed = self.token_mixing(feature_tokens) + feature_tokens
        return self.feature_ffn(self.feature_norm(mixed))


class RankMixerModel(nn.Module):
    def __init__(self, config: AppConfig, vocab_maps: dict[str, dict[str, int]], embedding_dim: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.feature_groups = config.tokenization.resolved_feature_tokens(config.features, config.sequences)
        self.feature_token_inputs = config.tokenization.resolved_feature_token_inputs(config.features, config.sequences)
        self.feature_token_count = config.tokenization.resolved_feature_token_count(config.features, config.sequences)
        if config.model.token_dim % self.feature_token_count != 0:
            raise ValueError("rankmixer requires token_dim divisible by feature token count")
        embedding_dim = config.model.embedding_dim if embedding_dim is None else embedding_dim
        self.encoder_bank = FeatureEncoderBank(config, vocab_maps, embedding_dim)
        self.feature_projector = _build_rankmixer_feature_projector(
            config,
            self.encoder_bank,
            self.feature_groups,
            self.feature_token_inputs,
            self.feature_token_count,
        )
        self.blocks = nn.ModuleList(RankMixerBlock(config, self.feature_token_count) for _ in range(config.model.num_layers))
        output_dim = self.feature_token_count * config.model.token_dim
        self.logit_layers = _build_task_heads(config, output_dim, len(config.task_names))

    def precompute_request_cache(self, features: dict[str, Any]) -> dict[str, LongerSequenceCache]:
        return self.encoder_bank.precompute_request_cache(features)

    def forward(
        self,
        features: dict[str, Any],
        scenario_id: Tensor,
        request_cache: dict[str, LongerSequenceCache] | None = None,
    ) -> dict[str, Tensor]:
        del scenario_id
        encoded = self.encoder_bank(features, request_cache=request_cache)
        feature_tokens = self.feature_projector(encoded)
        for block in self.blocks:
            if self.config.runtime.activation_checkpoint and self.training:
                feature_tokens = checkpoint(block, feature_tokens, use_reentrant=False)
            else:
                feature_tokens = block(feature_tokens)
        pooled = feature_tokens.flatten(start_dim=1)
        logits = torch.cat([layer(pooled) for layer in self.logit_layers], dim=1)
        return {"logits": logits}


def _init_domain_interaction_modules(block: nn.Module, config: AppConfig, metadata: ModelMetadata) -> None:
    token_dim = config.model.token_dim
    hidden_dim = config.model.hidden_dim
    block.use_task_tokens = config.model.use_task_tokens
    block.use_scenario_tokens = config.model.use_scenario_tokens
    block.use_global_scenario_token = config.model.use_global_scenario_token
    block.use_task_feature_interaction = config.model.use_task_feature_interaction
    block.use_scenario_feature_interaction = config.model.use_scenario_feature_interaction
    block.scenario_attention = DomainAwareAttention(
        token_dim,
        config.model.num_heads,
        metadata.scenario_count + 1,
        metadata.feature_token_count,
        hidden_dim,
        attention_backend=config.runtime.attention_backend,
        activation=config.model.ffn_activation,
    )
    block.scenario_ffn = PerTokenFFN(
        metadata.scenario_count + 1,
        token_dim,
        hidden_dim,
        activation=config.model.ffn_activation,
    )
    block.task_attention = DomainAwareAttention(
        token_dim,
        config.model.num_heads,
        metadata.task_count,
        metadata.feature_token_count,
        hidden_dim,
        attention_backend=config.runtime.attention_backend,
        activation=config.model.ffn_activation,
    )
    block.domain_fused = DomainFusedModule(include_global=config.model.use_global_scenario_token)
    block.task_ffn = PerTokenFFN(
        metadata.task_count,
        token_dim,
        hidden_dim,
        activation=config.model.ffn_activation,
    )


def _forward_domain_interaction(
    block: Any,
    feature_tokens: Tensor,
    scenario_tokens: Tensor,
    task_tokens: Tensor,
    scenario_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    if not block.use_scenario_tokens:
        scenario_tokens = torch.zeros_like(scenario_tokens)
    elif not block.use_global_scenario_token:
        scenario_tokens = scenario_tokens.clone()
        scenario_tokens[:, -1, :] = 0.0
    if not block.use_task_tokens:
        task_tokens = torch.zeros_like(task_tokens)

    if block.use_scenario_feature_interaction:
        scenario_update, _weights = block.scenario_attention(scenario_tokens, feature_tokens)
    else:
        scenario_update = torch.zeros_like(scenario_tokens)
    scenario_hat = scenario_tokens + scenario_update
    if not block.use_global_scenario_token:
        scenario_hat = scenario_hat.clone()
        scenario_hat[:, -1, :] = 0.0
    scenario_tokens = scenario_hat + block.scenario_ffn(scenario_hat)
    if not block.use_global_scenario_token:
        scenario_tokens = scenario_tokens.clone()
        scenario_tokens[:, -1, :] = 0.0

    if block.use_task_feature_interaction:
        task_update, _weights = block.task_attention(task_tokens, feature_tokens)
    else:
        task_update = torch.zeros_like(task_tokens)
    task_hat = task_tokens + task_update
    if block.use_scenario_tokens or block.use_scenario_feature_interaction:
        task_hat = block.domain_fused(task_hat, scenario_hat, scenario_mask)
    task_tokens = task_hat + block.task_ffn(task_hat)
    return scenario_tokens, task_tokens


class MDLRankMixerBlock(nn.Module):
    def __init__(self, config: AppConfig, metadata: ModelMetadata) -> None:
        super().__init__()
        token_dim = config.model.token_dim
        hidden_dim = config.model.hidden_dim

        self.token_mixing = RankMixerTokenMixing(metadata.feature_token_count, token_dim)
        self.feature_norm = nn.LayerNorm(token_dim)
        self.feature_ffn = PerTokenFFN(
            metadata.feature_token_count,
            token_dim,
            hidden_dim,
            activation=config.model.ffn_activation,
        )
        _init_domain_interaction_modules(self, config, metadata)

    def forward(self, feature_tokens: Tensor, scenario_tokens: Tensor, task_tokens: Tensor, scenario_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        feature_tokens = self.feature_ffn(self.feature_norm(self.token_mixing(feature_tokens) + feature_tokens))

        scenario_tokens, task_tokens = _forward_domain_interaction(
            self,
            feature_tokens,
            scenario_tokens,
            task_tokens,
            scenario_mask,
        )
        return feature_tokens, scenario_tokens, task_tokens


class MDLRankMixerModel(nn.Module):
    def __init__(self, config: AppConfig, vocab_maps: dict[str, dict[str, int]], embedding_dim: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.feature_groups = config.tokenization.resolved_feature_tokens(config.features, config.sequences)
        self.feature_token_inputs = config.tokenization.resolved_feature_token_inputs(config.features, config.sequences)
        feature_token_count = config.tokenization.resolved_feature_token_count(config.features, config.sequences)
        self.scenario_token_specs = config.tokenization.resolved_scenario_tokens(
            config.features,
            config.scenarios.names,
            config.sequences,
        )
        self.task_token_specs = config.tokenization.resolved_task_tokens(
            config.features,
            config.task_names,
            config.sequences,
        )
        self.metadata = ModelMetadata(
            feature_token_count=feature_token_count,
            scenario_count=len(config.scenarios.names),
            task_count=len(config.task_names),
        )
        if config.model.token_dim % self.metadata.feature_token_count != 0:
            raise ValueError("mdl_rankmixer requires token_dim divisible by feature token count")
        embedding_dim = config.model.embedding_dim if embedding_dim is None else embedding_dim
        self.encoder_bank = FeatureEncoderBank(config, vocab_maps, embedding_dim)
        self.feature_projector = _build_rankmixer_feature_projector(
            config,
            self.encoder_bank,
            self.feature_groups,
            self.feature_token_inputs,
            self.metadata.feature_token_count,
        )
        self.scenario_projector = DomainTokenProjector(
            self.scenario_token_specs,
            self.encoder_bank.output_dims,
            config.model.token_dim,
            config.model.hidden_dim,
            activation=config.model.ffn_activation,
        )
        self.task_projector = DomainTokenProjector(
            self.task_token_specs,
            self.encoder_bank.output_dims,
            config.model.token_dim,
            config.model.hidden_dim,
            activation=config.model.ffn_activation,
        )
        self.blocks = nn.ModuleList(MDLRankMixerBlock(config, self.metadata) for _ in range(config.model.num_layers))
        self.logit_layers = _build_task_heads(config, config.model.token_dim, self.metadata.task_count)

    def precompute_request_cache(self, features: dict[str, Any]) -> dict[str, LongerSequenceCache]:
        return self.encoder_bank.precompute_request_cache(features)

    def forward(
        self,
        features: dict[str, Any],
        scenario_id: Tensor,
        request_cache: dict[str, LongerSequenceCache] | None = None,
    ) -> dict[str, Tensor]:
        encoded = self.encoder_bank(features, request_cache=request_cache)
        feature_tokens = self.feature_projector(encoded)
        scenario_tokens = self.scenario_projector(encoded)
        task_tokens = self.task_projector(encoded)
        scenario_mask = _scenario_mask_from_ids(scenario_id, self.metadata.scenario_count)
        for block in self.blocks:
            if self.config.runtime.activation_checkpoint and self.training:
                feature_tokens, scenario_tokens, task_tokens = checkpoint(
                    block,
                    feature_tokens,
                    scenario_tokens,
                    task_tokens,
                    scenario_mask,
                    use_reentrant=False,
                )
            else:
                feature_tokens, scenario_tokens, task_tokens = block(
                    feature_tokens,
                    scenario_tokens,
                    task_tokens,
                    scenario_mask,
                )
        logits = torch.cat(
            [layer(task_tokens[:, index, :]) for index, layer in enumerate(self.logit_layers)],
            dim=1,
        )
        return {"logits": logits}


class OneTransModel(nn.Module):
    def __init__(self, config: AppConfig, vocab_maps: dict[str, dict[str, int]], embedding_dim: int | None = None) -> None:
        super().__init__()
        self.backbone = OneTransBackbone(config, vocab_maps, embedding_dim)
        output_dim = self.backbone.ns_token_count * config.model.token_dim
        self.logit_layers = _build_task_heads(config, output_dim, len(config.task_names))

    def precompute_request_cache(self, features: dict[str, Any]) -> OneTransRequestCache:
        return self.backbone.precompute_request_cache(features)

    def forward(
        self,
        features: dict[str, Any],
        scenario_id: Tensor,
        request_cache: OneTransRequestCache | None = None,
    ) -> dict[str, Tensor]:
        del scenario_id
        output = self.backbone(features, request_cache=request_cache)
        pooled = output.feature_tokens.flatten(start_dim=1)
        logits = torch.cat([layer(pooled) for layer in self.logit_layers], dim=1)
        return {"logits": logits}


class MDLDomainBlock(nn.Module):
    def __init__(self, config: AppConfig, metadata: ModelMetadata) -> None:
        super().__init__()
        _init_domain_interaction_modules(self, config, metadata)

    def forward(
        self,
        feature_tokens: Tensor,
        scenario_tokens: Tensor,
        task_tokens: Tensor,
        scenario_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return _forward_domain_interaction(self, feature_tokens, scenario_tokens, task_tokens, scenario_mask)


class MDLOneTransModel(nn.Module):
    def __init__(self, config: AppConfig, vocab_maps: dict[str, dict[str, int]], embedding_dim: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.backbone = OneTransBackbone(config, vocab_maps, embedding_dim)
        self.metadata = ModelMetadata(
            feature_token_count=self.backbone.ns_token_count,
            scenario_count=len(config.scenarios.names),
            task_count=len(config.task_names),
        )
        output_dims = self.backbone.encoder_bank.output_dims
        scenario_token_specs = config.tokenization.resolved_scenario_tokens(
            config.features,
            config.scenarios.names,
            config.sequences,
        )
        task_token_specs = config.tokenization.resolved_task_tokens(
            config.features,
            config.task_names,
            config.sequences,
        )
        self.scenario_projector = DomainTokenProjector(
            scenario_token_specs,
            output_dims,
            config.model.token_dim,
            config.model.hidden_dim,
            activation=config.model.ffn_activation,
        )
        self.task_projector = DomainTokenProjector(
            task_token_specs,
            output_dims,
            config.model.token_dim,
            config.model.hidden_dim,
            activation=config.model.ffn_activation,
        )
        self.blocks = nn.ModuleList(MDLDomainBlock(config, self.metadata) for _ in range(config.model.num_layers))
        self.logit_layers = _build_task_heads(config, config.model.token_dim, self.metadata.task_count)

    def precompute_request_cache(self, features: dict[str, Any]) -> OneTransRequestCache:
        return self.backbone.precompute_request_cache(features)

    def forward(
        self,
        features: dict[str, Any],
        scenario_id: Tensor,
        request_cache: OneTransRequestCache | None = None,
    ) -> dict[str, Tensor]:
        output = self.backbone(features, request_cache=request_cache)
        feature_tokens = output.feature_tokens
        scenario_tokens = self.scenario_projector(output.encoded_features)
        task_tokens = self.task_projector(output.encoded_features)
        scenario_mask = _scenario_mask_from_ids(scenario_id, self.metadata.scenario_count)
        for block in self.blocks:
            if self.config.runtime.activation_checkpoint and self.training:
                scenario_tokens, task_tokens = checkpoint(
                    block,
                    feature_tokens,
                    scenario_tokens,
                    task_tokens,
                    scenario_mask,
                    use_reentrant=False,
                )
            else:
                scenario_tokens, task_tokens = block(
                    feature_tokens,
                    scenario_tokens,
                    task_tokens,
                    scenario_mask,
                )
        logits = torch.cat(
            [layer(task_tokens[:, index, :]) for index, layer in enumerate(self.logit_layers)],
            dim=1,
        )
        return {"logits": logits}


def build_model(config: AppConfig, vocab_maps: dict[str, dict[str, int]]) -> nn.Module:
    if config.model.name == "rankmixer":
        return RankMixerModel(config, vocab_maps)
    if config.model.name == "mdl_rankmixer":
        return MDLRankMixerModel(config, vocab_maps)
    if config.model.name == "onetrans":
        return OneTransModel(config, vocab_maps)
    if config.model.name == "mdl_onetrans":
        return MDLOneTransModel(config, vocab_maps)
    raise NotImplementedError(f"model {config.model.name!r} is not implemented")
