from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .config import AppConfig, DomainTokenConfig, FeatureConfig, ResolvedEncoding, SequenceConfig, TokenGroupConfig
from .modules.attention import DomainAwareAttention, DomainFusedModule, RankMixerTokenMixing, _sdpa_context
from .modules.mlp import PerTokenFFN, PerTokenLinear, SparseMoEPerTokenFFN


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
class OneTransLayerCache:
    s_key: Tensor
    s_value: Tensor
    s_output: Tensor
    s_key_valid_mask: Tensor
    s_output_valid_mask: Tensor


@dataclass(frozen=True)
class OneTransRequestCache:
    s_tokens: Tensor
    s_valid_mask: Tensor
    layers: tuple[OneTransLayerCache, ...] = ()


@dataclass(frozen=True)
class OneTransBackboneState:
    tokens: Tensor
    valid_mask: Tensor
    s_count: int
    ns_count: int
    initial_s_count: int
    encoded_features: dict[str, Tensor]


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


def _projection_mlp(input_dim: int, token_dim: int, hidden_dim: int, activation: str) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        _activation_module(activation),
        nn.Linear(hidden_dim, token_dim),
    )


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
class LongerSelfLayerCache:
    recent_key: Tensor
    recent_value: Tensor
    recent_output: Tensor


@dataclass(frozen=True)
class LongerSequenceCache:
    merged_tokens: Tensor
    merged_mask: Tensor
    cross_sequence_key: Tensor
    cross_sequence_value: Tensor
    cross_recent_output: Tensor
    recent_mask: Tensor
    self_layers: tuple[LongerSelfLayerCache, ...]


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

    def project_kv(self, key_tokens: Tensor) -> tuple[Tensor, Tensor]:
        key_input = self.key_norm(key_tokens)
        return (
            self._split_heads(self.key_projection(key_input)),
            self._split_heads(self.value_projection(key_input)),
        )

    def forward_projected_kv(
        self,
        query_tokens: Tensor,
        key: Tensor,
        value: Tensor,
        allowed_mask: Tensor,
    ) -> Tensor:
        expected_mask_shape = (query_tokens.size(0), query_tokens.size(1), key.size(2))
        if tuple(allowed_mask.shape) != expected_mask_shape:
            raise ValueError(f"attention mask shape must be {expected_mask_shape}, got {tuple(allowed_mask.shape)}")
        query_input = self.query_norm(query_tokens)
        query = self._split_heads(self.query_projection(query_input))
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

    def forward(self, query_tokens: Tensor, key_tokens: Tensor, allowed_mask: Tensor) -> Tensor:
        key, value = self.project_kv(key_tokens)
        return self.forward_projected_kv(query_tokens, key, value, allowed_mask)


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
        self.input_dim = token_dim
        self.output_dim = merge_size * token_dim
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
        if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
            raise ValueError("tokens/mask must have shapes [batch, length, dim] and [batch, length]")
        if tokens.size(2) != self.input_dim:
            raise ValueError(f"expected token width {self.input_dim}, got {tokens.size(2)}")
        if tokens.size(1) == 0:
            return tokens.new_zeros(tokens.size(0), 0, self.output_dim), mask
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
            grouped = hidden.view(batch_size, group_count, self.merge_size, token_dim)
        grouped = grouped * group_mask.unsqueeze(-1).to(dtype=grouped.dtype)
        merged = grouped.reshape(batch_size, group_count, self.output_dim)
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
        self.input_dim = token_dim
        self.token_merger = LongerTokenMerger(
            token_dim,
            num_heads,
            hidden_dim,
            token_merge,
            inner_layers,
            attention_backend=attention_backend,
        )
        self.token_dim = self.token_merger.output_dim
        self.output_dim = (summary_tokens + query_token_count) * self.token_dim
        merged_hidden_dim = hidden_dim * token_merge
        self.cross_block = LongerSequenceAttentionBlock(
            self.token_dim,
            num_heads,
            merged_hidden_dim,
            attention_backend=attention_backend,
        )
        self.self_blocks = nn.ModuleList(
            LongerSequenceAttentionBlock(
                self.token_dim,
                num_heads,
                merged_hidden_dim,
                attention_backend=attention_backend,
            )
            for _ in range(self_layers)
        )

    def _recent_tokens(self, merged_tokens: Tensor, merged_mask: Tensor) -> tuple[Tensor, Tensor]:
        count = self.query_token_count
        if merged_tokens.size(1) >= count:
            return merged_tokens[:, -count:, :], merged_mask[:, -count:]
        pad = count - merged_tokens.size(1)
        token_pad = merged_tokens.new_zeros(merged_tokens.size(0), pad, self.token_dim)
        mask_pad = torch.zeros(merged_mask.size(0), pad, dtype=torch.bool, device=merged_mask.device)
        return (
            torch.cat([token_pad, merged_tokens], dim=1),
            torch.cat([mask_pad, merged_mask], dim=1),
        )

    def _recent_allowed_mask(self, key_mask: Tensor, query_mask: Tensor) -> Tensor:
        key_count = key_mask.size(1)
        query_count = query_mask.size(1)
        key_positions = torch.arange(key_count, device=key_mask.device).view(1, 1, key_count)
        query_positions = torch.arange(
            key_count - query_count, key_count, device=key_mask.device
        ).view(1, query_count, 1)
        return (
            key_mask.unsqueeze(1)
            & (key_positions <= query_positions)
            & query_mask.unsqueeze(-1)
        )

    def _recent_self_allowed_mask(self, recent_mask: Tensor) -> Tensor:
        count = recent_mask.size(1)
        causal = torch.arange(count, device=recent_mask.device).view(1, 1, count) <= torch.arange(
            count, device=recent_mask.device
        ).view(1, count, 1)
        return recent_mask.unsqueeze(1) & causal & recent_mask.unsqueeze(-1)

    def precompute_cache(self, tokens: Tensor, mask: Tensor) -> LongerSequenceCache:
        merged_tokens, merged_mask = self.token_merger(tokens, mask)
        recent_tokens, recent_mask = self._recent_tokens(merged_tokens, merged_mask)
        cross_key, cross_value = self.cross_block.project_kv(merged_tokens)
        cross_recent_output = self.cross_block.forward_projected_kv(
            recent_tokens,
            cross_key,
            cross_value,
            self._recent_allowed_mask(merged_mask, recent_mask),
        )
        cross_recent_output = cross_recent_output * recent_mask.unsqueeze(-1).to(cross_recent_output.dtype)

        current_recent = cross_recent_output
        layer_caches: list[LongerSelfLayerCache] = []
        recent_allowed = self._recent_self_allowed_mask(recent_mask)
        for block in self.self_blocks:
            recent_key, recent_value = block.project_kv(current_recent)
            recent_output = block.forward_projected_kv(
                current_recent, recent_key, recent_value, recent_allowed
            )
            recent_output = recent_output * recent_mask.unsqueeze(-1).to(recent_output.dtype)
            layer_caches.append(
                LongerSelfLayerCache(
                    recent_key=recent_key,
                    recent_value=recent_value,
                    recent_output=recent_output,
                )
            )
            current_recent = recent_output
        return LongerSequenceCache(
            merged_tokens=merged_tokens,
            merged_mask=merged_mask,
            cross_sequence_key=cross_key,
            cross_sequence_value=cross_value,
            cross_recent_output=cross_recent_output,
            recent_mask=recent_mask,
            self_layers=tuple(layer_caches),
        )

    def _expand_cache(self, cache: LongerSequenceCache, batch_size: int) -> LongerSequenceCache:
        if cache.merged_tokens.size(0) == batch_size:
            return cache
        if cache.merged_tokens.size(0) != 1:
            raise ValueError("LONGER cache batch must be 1 or match candidate batch")
        layers = tuple(
            LongerSelfLayerCache(
                recent_key=item.recent_key.expand(batch_size, -1, -1, -1),
                recent_value=item.recent_value.expand(batch_size, -1, -1, -1),
                recent_output=item.recent_output.expand(batch_size, -1, -1),
            )
            for item in cache.self_layers
        )
        return LongerSequenceCache(
            merged_tokens=cache.merged_tokens.expand(batch_size, -1, -1),
            merged_mask=cache.merged_mask.expand(batch_size, -1),
            cross_sequence_key=cache.cross_sequence_key.expand(batch_size, -1, -1, -1),
            cross_sequence_value=cache.cross_sequence_value.expand(batch_size, -1, -1, -1),
            cross_recent_output=cache.cross_recent_output.expand(batch_size, -1, -1),
            recent_mask=cache.recent_mask.expand(batch_size, -1),
            self_layers=layers,
        )


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
        if global_tokens.size(1) != self.summary_tokens or global_tokens.size(2) != self.token_dim:
            raise ValueError(
                f"global_tokens must have shape [batch, {self.summary_tokens}, {self.token_dim}]"
            )
        sequence_cache = self.precompute_cache(tokens, mask) if cache is None else cache
        sequence_cache = self._expand_cache(sequence_cache, global_tokens.size(0))
        if sequence_cache.merged_tokens.size(0) != global_tokens.size(0):
            raise ValueError("LONGER cache and global tokens must have the same batch size")
        if len(sequence_cache.self_layers) != len(self.self_blocks):
            raise ValueError("LONGER cache depth does not match encoder depth")

        global_valid = torch.ones(
            global_tokens.size(0),
            self.summary_tokens,
            dtype=torch.bool,
            device=global_tokens.device,
        )
        global_key, global_value = self.cross_block.project_kv(global_tokens)
        cross_key = torch.cat([global_key, sequence_cache.cross_sequence_key], dim=2)
        cross_value = torch.cat([global_value, sequence_cache.cross_sequence_value], dim=2)
        cross_key_mask = torch.cat([global_valid, sequence_cache.merged_mask], dim=1)
        global_hidden = self.cross_block.forward_projected_kv(
            global_tokens,
            cross_key,
            cross_value,
            cross_key_mask.unsqueeze(1).expand(-1, self.summary_tokens, -1),
        )
        hidden = torch.cat([global_hidden, sequence_cache.cross_recent_output], dim=1)

        for block, layer_cache in zip(self.self_blocks, sequence_cache.self_layers):
            global_key, global_value = block.project_kv(global_hidden)
            key = torch.cat([global_key, layer_cache.recent_key], dim=2)
            value = torch.cat([global_value, layer_cache.recent_value], dim=2)
            key_mask = torch.cat([global_valid, sequence_cache.recent_mask], dim=1)
            global_hidden = block.forward_projected_kv(
                global_hidden,
                key,
                value,
                key_mask.unsqueeze(1).expand(-1, self.summary_tokens, -1),
            )
            hidden = torch.cat([global_hidden, layer_cache.recent_output], dim=1)
        return hidden.flatten(start_dim=1)


class FeatureEncoderBank(nn.Module):
    def __init__(
        self,
        config: AppConfig,
        vocab_maps: dict[str, dict[str, int]],
        embedding_dim: int,
        build_sequence_summaries: bool = True,
        included_scalar_feature_names: set[str] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.build_sequence_summaries = build_sequence_summaries
        self.sequence_token_dim = config.model.token_dim
        default_scalar_feature_names = {
            feature.name
            for feature in config.features
            if config.model.name in {"mdl_rankmixer", "mdl_onetrans"}
            or feature.embedding_scope in {"feature", "shared"}
        }
        if included_scalar_feature_names is None:
            self.included_scalar_feature_names = default_scalar_feature_names
        else:
            unknown = set(included_scalar_feature_names) - default_scalar_feature_names
            if unknown:
                raise ValueError("excluded scalar features requested: " + ", ".join(sorted(unknown)))
            self.included_scalar_feature_names = set(included_scalar_feature_names)
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
            if feature.name not in self.included_scalar_feature_names:
                continue
            if feature.kind == "dense":
                self.output_dims[feature.name] = feature.dimension
            elif feature.kind == "categorical":
                self.output_dims[feature.name] = categorical_dims[feature.name]

        for sequence in config.sequences:
            for field in sequence.fields:
                if field.kind == "categorical":
                    qualified = field.qualified_name(sequence.name)
                    self.sequence_field_embedding_keys[qualified] = self._module_key(qualified)

        scalar_feature_names = {feature.name for feature in config.features}
        for qualified in self.sequence_field_embedding_keys:
            encoding = _encoding_for(config, qualified)
            if encoding.encoding == "shared_vocab" and encoding.share_embedding:
                base_name = self._shared_base_name(qualified)
                if base_name in scalar_feature_names:
                    self.included_scalar_feature_names.add(base_name)

        for feature in config.features:
            if feature.name not in self.included_scalar_feature_names:
                continue
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
            if not self.build_sequence_summaries:
                self.output_dims[sequence.name] = self.sequence_token_dim
                continue
            sequence_key = self._module_key(sequence.name)
            query_token_dim = self.sequence_token_dim
            if sequence.encoder == "longer":
                longer_encoder = LongerSequenceEncoder(
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
                self.sequence_longer_encoders[sequence_key] = longer_encoder
                query_token_dim = longer_encoder.token_dim
                output_dim = longer_encoder.output_dim
            else:
                output_dim = sequence.rankmixer_summary_tokens * self.sequence_token_dim
            query_dim = sequence.rankmixer_summary_tokens * query_token_dim
            if sequence.target_inputs:
                target_dim = sum(self.output_dims[name] for name in sequence.target_inputs)
                self.sequence_query_projectors[sequence_key] = nn.Linear(target_dim, query_dim)
            else:
                self.sequence_queries[sequence_key] = _normal_parameter(
                    (1, sequence.rankmixer_summary_tokens, query_token_dim),
                    config.model.init_std,
                )
            self.output_dims[sequence.name] = output_dim

        for feature in config.features:
            if feature.name not in self.included_scalar_feature_names:
                continue
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

    def _right_aligned_sequence(
        self,
        embedded: Tensor,
        lengths: Tensor,
        target_length: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        output_length = embedded.size(1) if target_length is None else target_length
        if output_length == 0:
            mask = torch.zeros(embedded.size(0), 0, dtype=torch.bool, device=embedded.device)
            return embedded[:, :0, :], mask
        if embedded.size(1) == 0:
            return (
                embedded.new_zeros(embedded.size(0), output_length, embedded.size(2)),
                torch.zeros(embedded.size(0), output_length, dtype=torch.bool, device=embedded.device),
            )
        lengths = lengths.clamp(min=0, max=output_length)
        positions = torch.arange(output_length, device=embedded.device).view(1, -1)
        shifts = (output_length - lengths).view(-1, 1)
        source_positions = (positions - shifts).clamp(min=0, max=embedded.size(1) - 1)
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
        tokens, mask = self._right_aligned_sequence(tokens, lengths, sequence.max_length)
        position_key = self._module_key(sequence.name)
        if position_key in self.sequence_position_embeddings and tokens.size(1) > 0:
            max_positions = self.sequence_position_embeddings[position_key].num_embeddings
            valid_lengths = lengths.clamp(min=0, max=tokens.size(1)).view(-1, 1)
            physical_positions = torch.arange(tokens.size(1), device=tokens.device).view(1, -1)
            relative_positions = (physical_positions - (tokens.size(1) - valid_lengths)).clamp(
                min=0, max=max_positions - 1
            )
            tokens = tokens + self.sequence_position_embeddings[position_key](relative_positions)
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
        query_token_dim = (
            self.sequence_longer_encoders[sequence_key].token_dim
            if sequence.encoder == "longer"
            else self.sequence_token_dim
        )
        if sequence.target_inputs:
            query_input = torch.cat([encoded[name] for name in sequence.target_inputs], dim=1)
            query = self.sequence_query_projectors[sequence_key](query_input).view(
                tokens.size(0),
                sequence.rankmixer_summary_tokens,
                query_token_dim,
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

    def encode_scalar_features(
        self,
        features: dict[str, Any],
        names: set[str] | None = None,
    ) -> dict[str, Tensor]:
        encoded: dict[str, Tensor] = {}
        for feature in self.config.features:
            if feature.name not in self.included_scalar_feature_names:
                continue
            if names is not None and feature.name not in names:
                continue
            value = features[feature.name]
            if not isinstance(value, Tensor):
                raise ValueError(f"scalar feature {feature.name!r} must be a tensor")
            encoded[feature.name] = self._encode_scalar_feature(feature, value)
        return encoded


    def precompute_request_cache(self, features: dict[str, Any]) -> dict[str, LongerSequenceCache]:
        caches: dict[str, LongerSequenceCache] = {}
        if not self.build_sequence_summaries:
            return caches
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
            if feature.name not in self.included_scalar_feature_names:
                continue
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

        if not self.build_sequence_summaries:
            return encoded
        for sequence in self.config.sequences:
            value = features[sequence.name]
            if not isinstance(value, dict):
                raise ValueError(f"sequence {sequence.name!r} must be a payload dict")
            sequence_cache = None if request_cache is None else request_cache.get(sequence.name)
            if sequence.encoder == "longer" and sequence_cache is not None:
                # The cache owns sequence embedding, merge, K/V, and sequence-side
                # attention work. Only candidate-derived global tokens are recomputed.
                batch_size = (
                    int(encoded[sequence.target_inputs[0]].size(0))
                    if sequence.target_inputs
                    else int(value["lengths"].size(0))
                )
                tokens = sequence_cache.merged_tokens.new_zeros(
                    batch_size, 1, self.sequence_token_dim
                )
                mask = torch.ones(
                    batch_size,
                    1,
                    dtype=torch.bool,
                    device=tokens.device,
                )
            else:
                tokens, mask = self._multi_field_sequence_tokens(sequence, value)
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
        self.projection = PerTokenLinear(num_tokens, token_dim, token_dim)
        if self.input_dim != self.target_dim:
            raise ValueError(
                "rankmixer tokenization requires exact input dimension "
                "num_feature_tokens * token_dim; implicit zero padding is disabled: "
                f"{self.input_dim} != {self.target_dim}"
            )

    def forward(self, encoded: dict[str, Tensor]) -> Tensor:
        values = torch.cat([encoded[name] for name in self.input_names], dim=1)
        sliced = values.view(values.size(0), self.num_tokens, self.token_dim)
        return self.projection(sliced)


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
        self.sequence_fusion = config.model.sequence_fusion

        if not self.ns_groups and self.ns_tokenizer == "groupwise":
            raise ValueError("groupwise OneTrans tokenizer requires tokenization.ns_tokens or scalar features")
        if not self.sequence_groups:
            raise ValueError("OneTrans requires at least one sequence feature")

        self.sequence_projectors = nn.ModuleList(
            _projection_mlp(
                self._group_input_dim(group),
                self.token_dim,
                config.model.hidden_dim,
                config.model.ffn_activation,
            )
            for group in self.sequence_groups
        )
        self.sequence_type_embeddings = (
            _init_embedding(
                nn.Embedding(len(self.sequence_groups), self.token_dim),
                config.model.init_std,
            )
            if self.sequence_fusion == "timestamp_aware"
            else None
        )
        separator_count = (
            max(len(self.sequence_groups) - 1, 0)
            if self.sequence_fusion == "intent_ordered" and self.use_sep_tokens
            else 0
        )
        self.sep_tokens = nn.ParameterList(
            _normal_parameter((1, 1, self.token_dim), config.model.init_std)
            for _ in range(separator_count)
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
                _projection_mlp(
                    self._group_input_dim(group),
                    self.token_dim,
                    config.model.hidden_dim,
                    config.model.ffn_activation,
                )
                for group in self.ns_groups
            )
        self.ns_input_names = set(self.scalar_feature_names) if self.ns_tokenizer == "auto_split" else {
            name for group in self.ns_groups for name in group.inputs
        }
        if self.num_ns_tokens <= 0:
            raise ValueError("OneTrans requires at least one NS token")

    def _group_input_dim(self, group: TokenGroupConfig) -> int:
        return sum(
            self.token_dim
            if name in self.sequence_by_name
            else self.encoder_bank.output_dims[name]
            for name in group.inputs
        )

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
                configured_length = self.sequence_by_name[name].max_length
                max_length = max(
                    max_length, configured_length or self._payload_max_length(value)
                )
                continue
        if not sequence_lengths:
            raise ValueError(f"sequence token group {group.name!r} must include a sequence input")
        first = sequence_lengths[0]
        for current in sequence_lengths[1:]:
            if not torch.equal(first, current):
                raise ValueError(f"sequence token group {group.name!r} has unaligned sequence lengths")
        return max_length, first

    def _sequence_group_timestamps(
        self,
        group: TokenGroupConfig,
        features: dict[str, Any],
        expected_mask: Tensor,
    ) -> Tensor:
        timestamps: Tensor | None = None
        for name in group.inputs:
            sequence = self.sequence_by_name.get(name)
            if sequence is None:
                continue
            if sequence.timestamp_field is None:
                raise ValueError(f"sequence {name!r} has no timestamp_field")
            value = features[name]
            raw = value["fields"][sequence.timestamp_field].float()
            if raw.dim() == 2:
                raw = raw.unsqueeze(-1)
            if raw.size(-1) != 1:
                raise ValueError(f"sequence {name!r} timestamp field must be scalar")
            aligned, timestamp_mask = self.encoder_bank._right_aligned_sequence(
                raw, value["lengths"].long(), sequence.max_length
            )
            current = aligned.squeeze(-1)
            if timestamp_mask.shape != expected_mask.shape:
                raise ValueError(f"sequence {name!r} timestamp shape does not match token shape")
            if timestamps is None:
                timestamps = current
            elif not torch.equal(timestamps.masked_select(expected_mask), current.masked_select(expected_mask)):
                raise ValueError(f"sequence token group {group.name!r} contains inconsistent timestamps")
        if timestamps is None:
            raise ValueError(f"sequence token group {group.name!r} has no timestamp source")
        return timestamps


    def _sequence_group_tokens(
        self,
        group: TokenGroupConfig,
        projection: nn.Module,
        features: dict[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor | None]:
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
        tokens = projection(torch.cat(parts, dim=-1))
        tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
        timestamps = (
            self._sequence_group_timestamps(group, features, mask)
            if self.sequence_fusion == "timestamp_aware"
            else None
        )
        return tokens, mask, timestamps

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

    def _compact_valid_tokens(self, tokens: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        order = torch.argsort(mask.to(torch.int64), dim=1, stable=True)
        token_order = order.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
        return tokens.gather(1, token_order), mask.gather(1, order)

    def _sequence_token_part(self, features: dict[str, Any]) -> OneTransRequestCache:
        sequence_tokens: list[Tensor] = []
        sequence_masks: list[Tensor] = []
        sequence_timestamps: list[Tensor] = []
        for index, (group, projection) in enumerate(zip(self.sequence_groups, self.sequence_projectors)):
            tokens, mask, timestamps = self._sequence_group_tokens(group, projection, features)
            if self.sequence_fusion == "timestamp_aware":
                if self.sequence_type_embeddings is None:
                    raise RuntimeError("timestamp-aware fusion has no type embeddings")
                type_indicator = self.sequence_type_embeddings.weight[index].view(1, 1, -1)
                tokens = tokens + type_indicator
            tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
            sequence_tokens.append(tokens)
            sequence_masks.append(mask)
            if self.sequence_fusion == "timestamp_aware":
                if timestamps is None:
                    raise RuntimeError("timestamp-aware fusion requires timestamps")
                sequence_timestamps.append(timestamps)
            elif self.use_sep_tokens and index < len(self.sep_tokens):
                sep = self.sep_tokens[index].expand(tokens.size(0), -1, -1)
                sequence_tokens.append(sep)
                sequence_masks.append(
                    torch.ones(tokens.size(0), 1, dtype=torch.bool, device=tokens.device)
                )

        tokens = torch.cat(sequence_tokens, dim=1)
        mask = torch.cat(sequence_masks, dim=1)
        if self.sequence_fusion == "timestamp_aware":
            timestamps = torch.cat(sequence_timestamps, dim=1)
            sort_values = timestamps.masked_fill(~mask, -torch.inf)
            order = torch.argsort(sort_values, dim=1, stable=True)
            tokens = tokens.gather(
                1, order.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
            )
            mask = mask.gather(1, order)
        else:
            tokens, mask = self._compact_valid_tokens(tokens, mask)
        return OneTransRequestCache(s_tokens=tokens, s_valid_mask=mask)

    def precompute_request_cache(self, features: dict[str, Any]) -> OneTransRequestCache:
        return self._sequence_token_part(features)

    def forward(
        self,
        features: dict[str, Any],
        request_cache: OneTransRequestCache | None = None,
        encoded_features: dict[str, Tensor] | None = None,
    ) -> OneTransOutput:
        if request_cache is None and self.config.model.use_request_cache:
            request_cache = self.precompute_request_cache(features)
        encoded = encoded_features or self.encoder_bank.encode_scalar_features(
            features, self.ns_input_names
        )
        cache = self._sequence_token_part(features) if request_cache is None else request_cache
        s_tokens = cache.s_tokens
        s_mask = cache.s_valid_mask
        ns_tokens = (
            self._ns_tokens_auto_split(encoded)
            if self.ns_tokenizer == "auto_split"
            else self._ns_tokens_groupwise(encoded)
        )
        if s_tokens.size(0) != ns_tokens.size(0):
            if s_tokens.size(0) != 1:
                raise ValueError("OneTrans request cache batch must be 1 or match candidate batch")
            s_tokens = s_tokens.expand(ns_tokens.size(0), -1, -1)
            s_mask = s_mask.expand(ns_tokens.size(0), -1)
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

    def project_s_kv(self, normalized_s_tokens: Tensor) -> tuple[Tensor, Tensor]:
        return (
            self._split_heads(self.s_key(normalized_s_tokens)),
            self._split_heads(self.s_value(normalized_s_tokens)),
        )

    def project_ns_kv(self, normalized_ns_tokens: Tensor) -> tuple[Tensor, Tensor]:
        return (
            self._split_heads(self._project_all(normalized_ns_tokens, 0, self.s_key, self.ns_key)),
            self._split_heads(self._project_all(normalized_ns_tokens, 0, self.s_value, self.ns_value)),
        )

    def project_s_query(self, normalized_s_tokens: Tensor) -> Tensor:
        return self._split_heads(self.s_query(normalized_s_tokens))

    def project_ns_query(self, normalized_ns_tokens: Tensor) -> Tensor:
        return self._split_heads(self._project_query(normalized_ns_tokens, 0))

    def attend_projected(
        self, query: Tensor, key: Tensor, value: Tensor, allowed_mask: Tensor
    ) -> Tensor:
        with _sdpa_context(self.attention_backend):
            attended = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=allowed_mask.unsqueeze(1),
                dropout_p=0.0,
            )
        return self.output(self._merge_heads(attended))


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

    def precompute_s(
        self,
        s_tokens: Tensor,
        query_s_count: int,
        valid_mask: Tensor,
    ) -> OneTransLayerCache:
        normalized = self.norm_attention(s_tokens)
        s_key, s_value = self.attention.project_s_kv(normalized)
        output_mask = valid_mask[:, s_tokens.size(1) - query_s_count :]
        if query_s_count == 0:
            output = s_tokens[:, :0, :]
        else:
            query_tokens = normalized[:, -query_s_count:, :]
            query = self.attention.project_s_query(query_tokens)
            key_positions = torch.arange(s_tokens.size(1), device=s_tokens.device).view(1, -1)
            query_positions = torch.arange(
                s_tokens.size(1) - query_s_count, s_tokens.size(1), device=s_tokens.device
            ).view(-1, 1)
            allowed = (key_positions <= query_positions).unsqueeze(0) & valid_mask.unsqueeze(1)
            attended = self.attention.attend_projected(query, s_key, s_value, allowed)
            residual = s_tokens[:, -query_s_count:, :]
            hidden = residual + attended
            output = hidden + self.ffn.s_ffn(self.norm_ffn(hidden))
        return OneTransLayerCache(
            s_key=s_key,
            s_value=s_value,
            s_output=output,
            s_key_valid_mask=valid_mask,
            s_output_valid_mask=output_mask,
        )

    def forward_cached_ns(self, ns_tokens: Tensor, cache: OneTransLayerCache) -> Tensor:
        normalized = self.norm_attention(ns_tokens)
        query = self.attention.project_ns_query(normalized)
        ns_key, ns_value = self.attention.project_ns_kv(normalized)
        s_key = cache.s_key
        s_value = cache.s_value
        s_mask = cache.s_key_valid_mask
        if s_key.size(0) != ns_tokens.size(0):
            if s_key.size(0) != 1:
                raise ValueError("OneTrans layer cache batch must be 1 or match candidate batch")
            s_key = s_key.expand(ns_tokens.size(0), -1, -1, -1)
            s_value = s_value.expand(ns_tokens.size(0), -1, -1, -1)
            s_mask = s_mask.expand(ns_tokens.size(0), -1)
        key = torch.cat([s_key, ns_key], dim=2)
        value = torch.cat([s_value, ns_value], dim=2)
        ns_count = ns_tokens.size(1)
        s_count = s_key.size(2)
        ns_causal = torch.arange(ns_count, device=ns_tokens.device).view(1, 1, ns_count) <= torch.arange(
            ns_count, device=ns_tokens.device
        ).view(1, ns_count, 1)
        allowed = torch.cat(
            [
                s_mask.unsqueeze(1).expand(-1, ns_count, -1),
                ns_causal.expand(ns_tokens.size(0), -1, -1),
            ],
            dim=2,
        )
        if key.size(2) != s_count + ns_count:
            raise RuntimeError("OneTrans cached key shape is inconsistent")
        attended = self.attention.attend_projected(query, key, value, allowed)
        hidden = ns_tokens + attended
        return hidden + self.ffn(self.norm_ffn(hidden), query_s_count=0)


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
        self.encoder_bank = FeatureEncoderBank(
            config,
            vocab_maps,
            embedding_dim,
            build_sequence_summaries=config.model.name == "mdl_onetrans",
        )
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
        token_cache = self.tokenizer.precompute_request_cache(features)
        current_tokens = token_cache.s_tokens
        current_mask = token_cache.s_valid_mask
        initial_s_count = current_tokens.size(1)
        layer_caches: list[OneTransLayerCache] = []
        for layer_index, block in enumerate(self.blocks):
            query_s_count = self._layer_s_count(
                initial_s_count, current_tokens.size(1), layer_index
            )
            layer_cache = block.precompute_s(current_tokens, query_s_count, current_mask)
            layer_caches.append(layer_cache)
            current_tokens = layer_cache.s_output
            current_mask = layer_cache.s_output_valid_mask
        return OneTransRequestCache(
            s_tokens=token_cache.s_tokens,
            s_valid_mask=token_cache.s_valid_mask,
            layers=tuple(layer_caches),
        )

    def prepare(
        self,
        features: dict[str, Any],
        request_cache: OneTransRequestCache | None = None,
        encoded_features: dict[str, Tensor] | None = None,
    ) -> OneTransBackboneState:
        tokenized = self.tokenizer(
            features,
            request_cache=request_cache,
            encoded_features=encoded_features,
        )
        ns_mask = torch.ones(
            tokenized.feature_tokens.size(0),
            tokenized.ns_token_count,
            dtype=torch.bool,
            device=tokenized.feature_tokens.device,
        )
        valid_mask = torch.cat([tokenized.s_valid_mask, ns_mask], dim=1)
        return OneTransBackboneState(
            tokens=tokenized.feature_tokens,
            valid_mask=valid_mask,
            s_count=tokenized.s_token_count,
            ns_count=tokenized.ns_token_count,
            initial_s_count=tokenized.s_token_count,
            encoded_features=tokenized.encoded_features,
        )

    def step(
        self,
        state: OneTransBackboneState,
        layer_index: int,
        layer_cache: OneTransLayerCache | None = None,
    ) -> OneTransBackboneState:
        block = self.blocks[layer_index]
        query_s_count = self._layer_s_count(
            state.initial_s_count, state.s_count, layer_index
        )
        if layer_cache is not None:
            if layer_cache.s_key.size(2) != state.s_count:
                raise ValueError("OneTrans layer cache S-token count does not match backbone state")
            if layer_cache.s_output.size(1) != query_s_count:
                raise ValueError("OneTrans layer cache pyramid output count is invalid")
            ns_tokens = state.tokens[:, state.s_count :, :]
            ns_output = block.forward_cached_ns(ns_tokens, layer_cache)
            s_output = layer_cache.s_output
            s_output_mask = layer_cache.s_output_valid_mask
            if s_output.size(0) != ns_output.size(0):
                if s_output.size(0) != 1:
                    raise ValueError("OneTrans S-output cache batch must be 1 or match candidates")
                s_output = s_output.expand(ns_output.size(0), -1, -1)
                s_output_mask = s_output_mask.expand(ns_output.size(0), -1)
            tokens = torch.cat([s_output, ns_output], dim=1)
            valid_mask = torch.cat(
                [s_output_mask, state.valid_mask[:, state.s_count :]],
                dim=1,
            )
        elif self.config.runtime.activation_checkpoint and self.training:
            tokens, valid_mask = checkpoint(
                lambda current_tokens, current_mask: block(
                    current_tokens, state.s_count, query_s_count, current_mask
                ),
                state.tokens,
                state.valid_mask,
                use_reentrant=False,
            )
        else:
            tokens, valid_mask = block(
                state.tokens, state.s_count, query_s_count, state.valid_mask
            )
        return OneTransBackboneState(
            tokens=tokens,
            valid_mask=valid_mask,
            s_count=query_s_count,
            ns_count=state.ns_count,
            initial_s_count=state.initial_s_count,
            encoded_features=state.encoded_features,
        )

    def forward(
        self,
        features: dict[str, Any],
        request_cache: OneTransRequestCache | None = None,
        encoded_features: dict[str, Tensor] | None = None,
    ) -> OneTransOutput:
        if request_cache is None and self.config.model.use_request_cache:
            request_cache = self.precompute_request_cache(features)
        state = self.prepare(features, request_cache, encoded_features)
        if request_cache is not None and request_cache.layers:
            if len(request_cache.layers) != len(self.blocks):
                raise ValueError("OneTrans request cache depth does not match backbone")
            layer_caches: tuple[OneTransLayerCache | None, ...] = request_cache.layers
        else:
            layer_caches = tuple(None for _ in self.blocks)
        for layer_index, layer_cache in enumerate(layer_caches):
            state = self.step(state, layer_index, layer_cache)
        return OneTransOutput(
            feature_tokens=state.tokens[:, state.s_count :, :],
            encoded_features=state.encoded_features,
            s_token_count=state.s_count,
            ns_token_count=state.ns_count,
            s_valid_mask=state.valid_mask[:, : state.s_count],
        )


class LongerModel(nn.Module):
    """End-to-end LONGER predictor over one long behavior sequence."""

    def __init__(
        self,
        config: AppConfig,
        vocab_maps: dict[str, dict[str, int]],
        embedding_dim: int | None = None,
    ) -> None:
        super().__init__()
        if len(config.sequences) != 1 or config.sequences[0].encoder != "longer":
            raise ValueError("LongerModel requires exactly one encoder=longer sequence")
        self.config = config
        self.sequence_name = config.sequences[0].name
        embedding_dim = config.model.embedding_dim if embedding_dim is None else embedding_dim
        target_inputs = set(config.sequences[0].target_inputs)
        self.encoder_bank = FeatureEncoderBank(
            config,
            vocab_maps,
            embedding_dim,
            included_scalar_feature_names=target_inputs,
        )
        output_dim = self.encoder_bank.output_dims[self.sequence_name]
        self.logit_layers = _build_task_heads(config, output_dim, len(config.task_names))

    def precompute_request_cache(
        self, features: dict[str, Any]
    ) -> dict[str, LongerSequenceCache]:
        return self.encoder_bank.precompute_request_cache(features)

    def forward(
        self,
        features: dict[str, Any],
        scenario_id: Tensor,
        request_cache: dict[str, LongerSequenceCache] | None = None,
    ) -> dict[str, Tensor]:
        del scenario_id
        encoded = self.encoder_bank(features, request_cache=request_cache)
        representation = encoded[self.sequence_name]
        logits = torch.cat([head(representation) for head in self.logit_layers], dim=1)
        return {"logits": logits}


def _build_rankmixer_ffn(config: AppConfig, num_tokens: int) -> nn.Module:
    if config.model.rankmixer_ffn_type == "dense":
        return PerTokenFFN(
            num_tokens,
            config.model.token_dim,
            config.model.hidden_dim,
            activation=config.model.ffn_activation,
        )
    return SparseMoEPerTokenFFN(
        num_tokens=num_tokens,
        token_dim=config.model.token_dim,
        hidden_dim=config.model.hidden_dim,
        num_experts=config.model.sparse_moe_num_experts,
        activation=config.model.ffn_activation,
        use_dtsi=config.model.sparse_moe_use_dtsi,
        inference_threshold=config.model.sparse_moe_inference_threshold,
        target_active_ratio=config.model.sparse_moe_target_active_ratio,
        regularization_initial=config.model.sparse_moe_regularization_initial,
        regularization_multiplier=config.model.sparse_moe_regularization_multiplier,
    )


def _sparse_moe_outputs(module: nn.Module, reference: Tensor) -> dict[str, Tensor]:
    moe_modules = [item for item in module.modules() if isinstance(item, SparseMoEPerTokenFFN)]
    if not moe_modules:
        return {}
    return {
        "moe_regularization_loss": torch.stack(
            [item.regularization_loss(reference) for item in moe_modules]
        ).sum(),
        "moe_active_ratio": torch.stack(
            [item.active_ratio(reference).to(reference.device) for item in moe_modules]
        ).mean(),
    }


class RankMixerBlock(nn.Module):
    def __init__(self, config: AppConfig, feature_token_count: int) -> None:
        super().__init__()
        token_dim = config.model.token_dim
        self.token_mixing = RankMixerTokenMixing(feature_token_count, token_dim)
        self.feature_norm = nn.LayerNorm(token_dim)
        self.feature_ffn = _build_rankmixer_ffn(config, feature_token_count)
        self.feature_ffn_norm = nn.LayerNorm(token_dim)

    def forward(self, feature_tokens: Tensor) -> Tensor:
        mixed = self.feature_norm(self.token_mixing(feature_tokens) + feature_tokens)
        return self.feature_ffn_norm(self.feature_ffn(mixed) + mixed)


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
        self.logit_layers = _build_task_heads(config, config.model.token_dim, len(config.task_names))

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
        pooled = feature_tokens.mean(dim=1)
        logits = torch.cat([layer(pooled) for layer in self.logit_layers], dim=1)
        output = {"logits": logits}
        output.update(_sparse_moe_outputs(self, logits))
        return output


def _init_domain_interaction_modules(
    block: nn.Module,
    config: AppConfig,
    metadata: ModelMetadata,
    propagate_scenario_state: bool = True,
) -> None:
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
    block.scenario_ffn = (
        PerTokenFFN(
            metadata.scenario_count + 1,
            token_dim,
            hidden_dim,
            activation=config.model.ffn_activation,
        )
        if propagate_scenario_state
        else None
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
    scenario_tokens = scenario_hat
    if block.scenario_ffn is not None:
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
    def __init__(
        self,
        config: AppConfig,
        metadata: ModelMetadata,
        propagate_scenario_state: bool = True,
    ) -> None:
        super().__init__()
        token_dim = config.model.token_dim

        self.token_mixing = RankMixerTokenMixing(metadata.feature_token_count, token_dim)
        self.feature_norm = nn.LayerNorm(token_dim)
        self.feature_ffn = _build_rankmixer_ffn(config, metadata.feature_token_count)
        self.feature_ffn_norm = nn.LayerNorm(token_dim)
        _init_domain_interaction_modules(self, config, metadata, propagate_scenario_state)

    def forward(self, feature_tokens: Tensor, scenario_tokens: Tensor, task_tokens: Tensor, scenario_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mixed = self.feature_norm(self.token_mixing(feature_tokens) + feature_tokens)
        feature_tokens = self.feature_ffn_norm(self.feature_ffn(mixed) + mixed)

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
        self.blocks = nn.ModuleList(
            MDLRankMixerBlock(
                config,
                self.metadata,
                propagate_scenario_state=layer_index < config.model.num_layers - 1,
            )
            for layer_index in range(config.model.num_layers)
        )
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
        output = {"logits": logits}
        output.update(_sparse_moe_outputs(self, logits))
        return output


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
    def __init__(
        self,
        config: AppConfig,
        metadata: ModelMetadata,
        propagate_scenario_state: bool = True,
    ) -> None:
        super().__init__()
        _init_domain_interaction_modules(
            self, config, metadata, propagate_scenario_state
        )

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
        self.blocks = nn.ModuleList(
            MDLDomainBlock(
                config,
                self.metadata,
                propagate_scenario_state=layer_index < config.model.num_layers - 1,
            )
            for layer_index in range(config.model.num_layers)
        )
        self.logit_layers = _build_task_heads(config, config.model.token_dim, self.metadata.task_count)

    def precompute_request_cache(self, features: dict[str, Any]) -> OneTransRequestCache:
        return self.backbone.precompute_request_cache(features)

    def forward(
        self,
        features: dict[str, Any],
        scenario_id: Tensor,
        request_cache: OneTransRequestCache | None = None,
    ) -> dict[str, Tensor]:
        encoded = self.backbone.encoder_bank(features)
        if request_cache is None and self.config.model.use_request_cache:
            request_cache = self.backbone.precompute_request_cache(features)
        state = self.backbone.prepare(
            features, request_cache=request_cache, encoded_features=encoded
        )
        scenario_tokens = self.scenario_projector(encoded)
        task_tokens = self.task_projector(encoded)
        scenario_mask = _scenario_mask_from_ids(scenario_id, self.metadata.scenario_count)
        if request_cache is not None and request_cache.layers:
            if len(request_cache.layers) != len(self.blocks):
                raise ValueError("OneTrans request cache depth does not match MDL domain depth")
            layer_caches: tuple[OneTransLayerCache | None, ...] = request_cache.layers
        else:
            layer_caches = tuple(None for _ in self.blocks)
        for layer_index, (block, layer_cache) in enumerate(zip(self.blocks, layer_caches)):
            state = self.backbone.step(state, layer_index, layer_cache)
            feature_tokens = state.tokens[:, state.s_count :, :]
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
                    feature_tokens, scenario_tokens, task_tokens, scenario_mask
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
    if config.model.name == "longer":
        return LongerModel(config, vocab_maps)
    raise NotImplementedError(f"model {config.model.name!r} is not implemented")
