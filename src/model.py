from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from typing import Any

import torch
import torch.distributed as torch_dist
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

try:
    from torch.nn.attention.varlen import varlen_attn
except ImportError:  # pragma: no cover - older PyTorch compatibility.
    varlen_attn = None

_VARLEN_ATTN_USES_WINDOW_SIZE = (
    varlen_attn is not None
    and "window_size" in inspect.signature(varlen_attn).parameters
)

from .config import (
    AppConfig,
    DomainTokenConfig,
    FeatureConfig,
    ResolvedEncoding,
    SequenceConfig,
    TokenGroupConfig,
    resolve_categorical_base_input,
    resolve_onetrans_max_position_embeddings,
)
from .embeddings import (
    EmbeddingShardingPlan,
    EmbeddingTableSpec,
    ShardedEmbedding,
    grouped_sharded_embedding_lookup,
    plan_embedding_shards,
)
from .modules.attention import (
    DomainAwareAttention,
    DomainFusedModule,
    RankMixerDomainInteraction,
    RankMixerTokenMixing,
    VariableLengthDomainAttention,
    _sdpa_context,
    masked_scenario_pool,
)
from .modules.mlp import (
    PerTokenFFN,
    PerTokenLinear,
    SparseMoEPerTokenFFN,
    StackedPerTokenFFN,
)


def _call_varlen_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_query: Tensor,
    cu_key: Tensor,
    max_query_length: int,
    max_key_length: int,
    *,
    causal: bool,
) -> Tensor:
    """Call the PyTorch 2.10 or 2.12 varlen Flash API.

    PyTorch 2.10 exposes a positional ``is_causal`` flag. PyTorch 2.12
    replaced it with the more general keyword-only ``window_size`` option.
    The two forms below describe the same full or bottom-right causal mask.
    """

    if varlen_attn is None:
        raise RuntimeError("torch.nn.attention.varlen is unavailable")
    if _VARLEN_ATTN_USES_WINDOW_SIZE:
        output = varlen_attn(
            query,
            key,
            value,
            cu_query,
            cu_key,
            max_query_length,
            max_key_length,
            window_size=(-1, 0) if causal else (-1, -1),
        )
    else:
        output = varlen_attn(
            query,
            key,
            value,
            cu_query,
            cu_key,
            max_query_length,
            max_key_length,
            is_causal=causal,
        )
    if not isinstance(output, Tensor):
        raise RuntimeError("varlen attention unexpectedly returned auxiliary outputs")
    return output


class _PermutationGather(torch.autograd.Function):
    """Gather by a full permutation and invert it without atomic scatter-add.

    ``index_select`` normally lowers its backward to ``index_add`` because
    arbitrary indices may repeat.  Varlen packing indices are a bijection, so
    the gradient is exactly another gather by the inverse permutation.  This
    avoids large atomic index-add kernels around every FlashAttention call.
    """

    @staticmethod
    def forward(
        ctx: Any,
        values: Tensor,
        indices: Tensor,
        inverse_indices: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(inverse_indices)
        return values.index_select(0, indices)

    @staticmethod
    def backward(
        ctx: Any,
        output_gradient: Tensor,
    ) -> tuple[Tensor, None, None]:
        (inverse_indices,) = ctx.saved_tensors
        return output_gradient.index_select(0, inverse_indices), None, None


@dataclass(frozen=True)
class _VarlenPacking:
    """Reusable row-major packing metadata for one validity mask.

    Boolean tensor indexing lowers to ``nonzero`` and synchronizes CUDA so the
    host can learn the dynamic output shape.  Keep a fixed-capacity packed
    tensor instead: valid rows form the prefix consumed by ``cu_seqlens`` and
    invalid rows occupy the ignored tail.  The permutation is built entirely
    on-device and reused for Q, K/V, and output restoration.
    """

    packed_source_indices: Tensor
    source_to_packed_indices: Tensor
    flat_mask: Tensor
    packed_mask: Tensor
    lengths: Tensor
    cumulative_lengths: Tensor
    batch_size: int
    padded_length: int

    @classmethod
    def from_mask(cls, mask: Tensor) -> "_VarlenPacking":
        if mask.ndim != 2:
            raise ValueError("varlen packing mask must have shape [batch, length]")
        mask = mask.to(dtype=torch.bool)
        flat_mask = mask.reshape(-1)
        lengths = mask.sum(dim=1, dtype=torch.int32)
        source_positions = torch.arange(
            flat_mask.numel(),
            dtype=torch.long,
            device=mask.device,
        )
        valid_prefix = flat_mask.long().cumsum(0)
        if flat_mask.numel():
            total_valid = valid_prefix[-1]
            source_to_packed = torch.where(
                flat_mask,
                valid_prefix - 1,
                total_valid + source_positions - valid_prefix,
            )
            packed_source = torch.empty_like(source_positions).scatter(
                0,
                source_to_packed,
                source_positions,
            )
        else:
            source_to_packed = source_positions
            packed_source = source_positions
        return cls(
            packed_source_indices=packed_source,
            source_to_packed_indices=source_to_packed,
            flat_mask=flat_mask,
            packed_mask=flat_mask.index_select(0, packed_source),
            lengths=lengths,
            cumulative_lengths=torch.nn.functional.pad(
                lengths.cumsum(0, dtype=torch.int32),
                (1, 0),
            ),
            batch_size=mask.size(0),
            padded_length=mask.size(1),
        )

    def pack(self, values: Tensor) -> Tensor:
        if values.shape[:2] != (self.batch_size, self.padded_length):
            raise ValueError(
                "packed values must match the packing mask batch and length"
            )
        packed = _PermutationGather.apply(
            values.flatten(0, 1),
            self.packed_source_indices,
            self.source_to_packed_indices,
        )
        mask_shape = (self.packed_mask.numel(),) + (1,) * (values.ndim - 2)
        # Flash ignores the fixed-capacity tail in forward, but some kernels do
        # not define tail gradients. Multiplying by a zero mask here prevents
        # those unused gradients from reaching padded source tokens.
        return packed * self.packed_mask.view(mask_shape).to(packed.dtype)

    def unpack(self, packed: Tensor, reference: Tensor) -> Tensor:
        if reference.shape[:2] != (self.batch_size, self.padded_length):
            raise ValueError(
                "unpack reference must match the packing mask batch and length"
            )
        expected_capacity = self.batch_size * self.padded_length
        if packed.size(0) != expected_capacity:
            raise ValueError(
                "fixed-capacity varlen output must match the padded token capacity"
            )
        output = _PermutationGather.apply(
            packed,
            self.source_to_packed_indices,
            self.packed_source_indices,
        )
        mask_shape = (expected_capacity,) + (1,) * (reference.ndim - 2)
        output = output * self.flat_mask.view(mask_shape).to(output.dtype)
        return output.view_as(reference)


def _activation_checkpoint_enabled(
    value: str | bool,
    *,
    full_only: bool = False,
) -> bool:
    """Normalize legacy booleans and the three explicit checkpoint modes."""

    if isinstance(value, bool):
        return value
    if value not in {"none", "selective", "full"}:
        raise ValueError("activation checkpoint mode must be none, selective, or full")
    return value == "full" if full_only else value in {"selective", "full"}


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
    s_input: Tensor
    s_input_start: int
    s_reused_kv_tokens: int
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


def _embedding_share_target(encoding: ResolvedEncoding) -> str | None:
    if not getattr(encoding, "share_embedding", False):
        return None
    target = getattr(encoding, "share_with", None)
    if not target:
        raise ValueError("share_embedding=true requires a non-empty share_with")
    return target


def _embedding_size(
    config: AppConfig,
    vocab_maps: dict[str, dict[str, int]],
    feature_name: str,
    override: int | None = None,
) -> int:
    if override is not None:
        if override < 2:
            raise ValueError("embedding size override must be at least 2")
        return override
    base_input = resolve_categorical_base_input(
        config.resolved.categorical_input_by_name,
        feature_name,
    )
    encoding = base_input.encoding
    if encoding.encoding == "hash":
        return encoding.num_buckets + 1
    if encoding.encoding == "identity":
        return encoding.num_buckets
    if encoding.encoding in {"vocab", "shared_vocab"}:
        values = vocab_maps.get(base_input.name, {})
        return max(values.values(), default=0) + 1
    raise ValueError(f"unsupported encoding {encoding.encoding!r}")


def _scenario_mask_from_ids(scenario_id: Tensor, scenario_count: int) -> Tensor:
    def invalid_any(invalid: Tensor, message: str) -> bool:
        invalid_result = invalid.any()
        if invalid.device.type == "cuda" and hasattr(torch, "_assert_async"):
            # Preserve input validation without forcing a device-to-host sync
            # in every MDL forward. CUDA reports a failing assertion at the
            # next synchronization point.
            torch._assert_async(~invalid_result, message)
            return False
        return bool(invalid_result.item())

    if scenario_count <= 0:
        raise ValueError("scenario_count must be positive")
    if scenario_id.ndim == 2:
        if scenario_id.size(1) != scenario_count:
            raise ValueError(
                f"scenario mask width must be {scenario_count}, got {scenario_id.size(1)}"
            )
        mask = scenario_id.float()
        invalid = (mask < 0.0) | (mask > 1.0) | ((mask != 0.0) & (mask != 1.0))
        if invalid_any(
            invalid,
            "scenario mask must be binary with shape [batch, num_scenarios]",
        ):
            raise ValueError("scenario mask must be binary with shape [batch, num_scenarios]")
        return mask
    if scenario_id.ndim != 1:
        raise ValueError("scenario_id must have shape [batch] or [batch, num_scenarios]")
    if scenario_id.is_complex():
        raise ValueError("scenario_id must contain real integer ids")
    if torch.is_floating_point(scenario_id):
        invalid_values = ~torch.isfinite(scenario_id) | (
            scenario_id != torch.trunc(scenario_id)
        )
        if invalid_any(
            invalid_values,
            "scenario_id must contain integer-valued ids",
        ):
            examples = scenario_id[invalid_values][:5].detach().cpu().tolist()
            raise ValueError(
                "scenario_id must contain integer-valued ids; "
                f"got non-integer values {examples}"
            )
    indices = scenario_id.long().view(-1, 1)
    invalid = (indices < 0) | (indices >= scenario_count)
    if invalid_any(
        invalid,
        f"scenario_id contains ids outside [0, {scenario_count - 1}]",
    ):
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
    cacheable_key: Tensor
    cacheable_value: Tensor
    user_output: Tensor
    recent_output: Tensor


@dataclass(frozen=True)
class LongerSequenceCache:
    merged_tokens: Tensor
    merged_mask: Tensor
    cross_cacheable_key: Tensor
    cross_cacheable_value: Tensor
    cross_user_output: Tensor
    cross_recent_output: Tensor
    user_mask: Tensor
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

    @staticmethod
    def mixed_allowed_mask(
        key_valid_mask: Tensor,
        query_valid_mask: Tensor,
        global_query_count: int,
    ) -> Tensor:
        """LONGER visibility contract for ``[global; recent]`` queries.

        Global queries see every valid key. Recent queries use bottom-right
        causal alignment, so a short recent suffix attends to the matching
        historical prefix while still seeing any prepended global keys.
        """

        if key_valid_mask.ndim != 2 or query_valid_mask.ndim != 2:
            raise ValueError("key/query validity masks must have shape [batch, tokens]")
        if key_valid_mask.size(0) != query_valid_mask.size(0):
            raise ValueError("key/query validity masks must have the same batch size")
        if not 0 <= global_query_count <= query_valid_mask.size(1):
            raise ValueError("global_query_count is outside the query range")
        recent_count = query_valid_mask.size(1) - global_query_count
        key_count = key_valid_mask.size(1)
        parts: list[Tensor] = []
        if global_query_count:
            parts.append(
                key_valid_mask.unsqueeze(1).expand(
                    -1, global_query_count, -1
                )
                & query_valid_mask[:, :global_query_count].unsqueeze(-1)
            )
        if recent_count:
            key_positions = torch.arange(
                key_count, device=key_valid_mask.device
            ).view(1, 1, key_count)
            query_positions = torch.arange(
                key_count - recent_count,
                key_count,
                device=key_valid_mask.device,
            ).view(1, recent_count, 1)
            parts.append(
                key_valid_mask.unsqueeze(1)
                & (key_positions <= query_positions)
                & query_valid_mask[:, global_query_count:].unsqueeze(-1)
            )
        if not parts:
            return torch.zeros(
                key_valid_mask.size(0),
                0,
                key_count,
                dtype=torch.bool,
                device=key_valid_mask.device,
            )
        return torch.cat(parts, dim=1)

    def _flash_varlen_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_valid_mask: Tensor,
        key_valid_mask: Tensor,
        *,
        causal: bool,
        key_packing: _VarlenPacking | None = None,
    ) -> Tensor:
        if varlen_attn is None:
            raise RuntimeError(
                "runtime.attention_backend='flash' requires torch.nn.attention.varlen"
            )
        if query.device.type != "cuda":
            raise RuntimeError("Flash varlen attention requires CUDA tensors")
        if query.dtype not in {torch.float16, torch.bfloat16}:
            raise RuntimeError("Flash varlen attention requires FP16 or BF16 tensors")
        if self.training and self.dropout.p:
            raise RuntimeError(
                "Flash varlen attention does not support this block's non-zero dropout"
            )
        query_tokens = query.transpose(1, 2)
        key_tokens = key.transpose(1, 2)
        value_tokens = value.transpose(1, 2)
        query_packing = _VarlenPacking.from_mask(query_valid_mask)
        if key_packing is None:
            key_packing = (
                query_packing
                if query_valid_mask is key_valid_mask
                else _VarlenPacking.from_mask(key_valid_mask)
            )
        packed_query = query_packing.pack(query_tokens)
        packed_key = key_packing.pack(key_tokens)
        packed_value = key_packing.pack(value_tokens)
        if packed_query.numel() == 0:
            return torch.zeros_like(query)
        label = "causal" if causal else "full"
        with torch.profiler.record_function(f"longer::flash_varlen_{label}"):
            packed_output = _call_varlen_attention(
                packed_query.contiguous(),
                packed_key.contiguous(),
                packed_value.contiguous(),
                query_packing.cumulative_lengths,
                key_packing.cumulative_lengths,
                query_valid_mask.size(1),
                key_valid_mask.size(1),
                causal=causal,
            )
        output = query_packing.unpack(packed_output, query_tokens)
        return output.transpose(1, 2)

    def _finish_attention(self, query_tokens: Tensor, attended: Tensor) -> Tensor:
        hidden = query_tokens + self.output_projection(self._merge_heads(attended))
        return hidden + self.ffn(self.ffn_norm(hidden))

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
        return self._finish_attention(query_tokens, attended)

    def forward_full_projected_kv(
        self,
        query_tokens: Tensor,
        key: Tensor,
        value: Tensor,
        query_valid_mask: Tensor,
        key_valid_mask: Tensor,
    ) -> Tensor:
        query = self._split_heads(self.query_projection(self.query_norm(query_tokens)))
        if self.attention_backend == "flash":
            attended = self._flash_varlen_attention(
                query,
                key,
                value,
                query_valid_mask,
                key_valid_mask,
                causal=False,
            )
        else:
            allowed = query_valid_mask.unsqueeze(-1) & key_valid_mask.unsqueeze(1)
            dropout_p = self.dropout.p if self.training else 0.0
            with _sdpa_context(self.attention_backend):
                attended = torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=self._nonempty_mask(allowed).unsqueeze(1),
                    dropout_p=dropout_p,
                )
        return self._finish_attention(query_tokens, attended)

    def forward_mixed_projected_kv(
        self,
        query_tokens: Tensor,
        key: Tensor,
        value: Tensor,
        query_valid_mask: Tensor,
        key_valid_mask: Tensor,
        global_query_count: int,
    ) -> Tensor:
        """Evaluate LONGER globals and recent queries without semantic collapse."""

        query = self._split_heads(self.query_projection(self.query_norm(query_tokens)))
        if self.attention_backend != "flash":
            allowed = self.mixed_allowed_mask(
                key_valid_mask, query_valid_mask, global_query_count
            )
            dropout_p = self.dropout.p if self.training else 0.0
            with _sdpa_context(self.attention_backend):
                attended = torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=self._nonempty_mask(allowed).unsqueeze(1),
                    dropout_p=dropout_p,
                )
            return self._finish_attention(query_tokens, attended)

        key_packing = _VarlenPacking.from_mask(key_valid_mask)
        parts: list[Tensor] = []
        if global_query_count:
            parts.append(
                self._flash_varlen_attention(
                    query[:, :, :global_query_count, :],
                    key,
                    value,
                    query_valid_mask[:, :global_query_count],
                    key_valid_mask,
                    causal=False,
                    key_packing=key_packing,
                )
            )
        if global_query_count < query_tokens.size(1):
            parts.append(
                self._flash_varlen_attention(
                    query[:, :, global_query_count:, :],
                    key,
                    value,
                    query_valid_mask[:, global_query_count:],
                    key_valid_mask,
                    causal=True,
                    key_packing=key_packing,
                )
            )
        attended = torch.cat(parts, dim=2) if parts else torch.zeros_like(query)
        return self._finish_attention(query_tokens, attended)

    def forward_full(
        self,
        tokens: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        key, value = self.project_kv(tokens)
        return self.forward_full_projected_kv(
            tokens, key, value, valid_mask, valid_mask
        )

    def forward(self, query_tokens: Tensor, key_tokens: Tensor, allowed_mask: Tensor) -> Tensor:
        key, value = self.project_kv(key_tokens)
        return self.forward_projected_kv(query_tokens, key, value, allowed_mask)


class LongerTokenMerger(nn.Module):
    # Varlen Flash Attention uses the flattened local-group count as a CUDA
    # grid dimension.  A single launch fails once that dimension exceeds the
    # 65,535 legacy grid bound, even though the workload and HBM still fit.
    _INNER_ATTENTION_BATCH_LIMIT = 65_535

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

    def _forward_inner_block(
        self,
        block: LongerSequenceAttentionBlock,
        hidden: Tensor,
        hidden_mask: Tensor,
    ) -> Tensor:
        if hidden.size(0) <= self._INNER_ATTENTION_BATCH_LIMIT:
            return block.forward_full(hidden, hidden_mask)
        # Use balanced chunks instead of leaving a one-row tail at boundaries
        # such as 65,536.  Groups are independent, so concatenating their
        # outputs preserves the exact unchunked attention semantics.
        chunk_count = math.ceil(
            hidden.size(0) / self._INNER_ATTENTION_BATCH_LIMIT
        )
        with torch.profiler.record_function("longer::chunked_inner_attention"):
            return torch.cat(
                [
                    block.forward_full(hidden_chunk, mask_chunk)
                    for hidden_chunk, mask_chunk in zip(
                        hidden.chunk(chunk_count, dim=0),
                        hidden_mask.chunk(chunk_count, dim=0),
                    )
                ],
                dim=0,
            )

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
            for block in self.inner_blocks:
                hidden = self._forward_inner_block(block, hidden, hidden_mask)
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
        user_global_tokens: int = 0,
        attention_backend: str = "auto",
        activation_checkpoint: bool = False,
        checkpoint_token_merger: bool | None = None,
    ) -> None:
        super().__init__()
        if query_token_count <= 0:
            raise ValueError("query_token_count must be positive")
        if self_layers < 0:
            raise ValueError("self_layers must be non-negative")
        if summary_tokens <= 0:
            raise ValueError("summary_tokens must be positive")
        if not 0 <= user_global_tokens <= summary_tokens:
            raise ValueError("user_global_tokens must be in [0, summary_tokens]")
        self.query_token_count = query_token_count
        self.summary_tokens = summary_tokens
        self.user_global_tokens = user_global_tokens
        self.candidate_global_tokens = summary_tokens - user_global_tokens
        self.activation_checkpoint = activation_checkpoint
        self.checkpoint_token_merger = (
            activation_checkpoint
            if checkpoint_token_merger is None
            else checkpoint_token_merger
        )
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

    def precompute_cache(
        self,
        tokens: Tensor,
        mask: Tensor,
        user_global_tokens: Tensor | None = None,
    ) -> LongerSequenceCache:
        if user_global_tokens is None:
            user_global_tokens = tokens.new_zeros(tokens.size(0), 0, self.token_dim)
        expected_user_shape = (tokens.size(0), self.user_global_tokens, self.token_dim)
        if tuple(user_global_tokens.shape) != expected_user_shape:
            raise ValueError(
                f"user_global_tokens must have shape {expected_user_shape}, "
                f"got {tuple(user_global_tokens.shape)}"
            )
        user_mask = torch.ones(
            tokens.size(0),
            self.user_global_tokens,
            dtype=torch.bool,
            device=tokens.device,
        )
        if self.checkpoint_token_merger and self.training:
            merged_tokens, merged_mask = checkpoint(
                self.token_merger,
                tokens,
                mask,
                use_reentrant=False,
            )
        else:
            merged_tokens, merged_mask = self.token_merger(tokens, mask)
        recent_tokens, recent_mask = self._recent_tokens(merged_tokens, merged_mask)
        cross_inputs = torch.cat([user_global_tokens, merged_tokens], dim=1)
        cross_queries = torch.cat([user_global_tokens, recent_tokens], dim=1)
        cross_query_mask = torch.cat([user_mask, recent_mask], dim=1)
        cross_key_mask = torch.cat([user_mask, merged_mask], dim=1)
        if self.activation_checkpoint and self.training:
            def cross_forward(
                current_inputs: Tensor,
                current_queries: Tensor,
                current_query_mask: Tensor,
                current_key_mask: Tensor,
            ) -> tuple[Tensor, Tensor, Tensor]:
                current_key, current_value = self.cross_block.project_kv(current_inputs)
                current_output = self.cross_block.forward_mixed_projected_kv(
                    current_queries,
                    current_key,
                    current_value,
                    current_query_mask,
                    current_key_mask,
                    self.user_global_tokens,
                )
                return current_key, current_value, current_output

            cross_key, cross_value, cross_output = checkpoint(
                cross_forward,
                cross_inputs,
                cross_queries,
                cross_query_mask,
                cross_key_mask,
                use_reentrant=False,
            )
        else:
            cross_key, cross_value = self.cross_block.project_kv(cross_inputs)
            cross_output = self.cross_block.forward_mixed_projected_kv(
                cross_queries,
                cross_key,
                cross_value,
                cross_query_mask,
                cross_key_mask,
                self.user_global_tokens,
            )
        cross_user_output = cross_output[:, : self.user_global_tokens, :]
        cross_recent_output = cross_output[:, self.user_global_tokens :, :]
        cross_recent_output = cross_recent_output * recent_mask.unsqueeze(-1).to(cross_recent_output.dtype)

        current_user = cross_user_output
        current_recent = cross_recent_output
        layer_caches: list[LongerSelfLayerCache] = []
        for block in self.self_blocks:
            cacheable_inputs = torch.cat([current_user, current_recent], dim=1)
            cacheable_mask = torch.cat([user_mask, recent_mask], dim=1)
            if self.activation_checkpoint and self.training:
                def self_forward(
                    current_inputs: Tensor,
                    current_mask: Tensor,
                    current_block: LongerSequenceAttentionBlock = block,
                ) -> tuple[Tensor, Tensor, Tensor]:
                    current_key, current_value = current_block.project_kv(current_inputs)
                    current_output = current_block.forward_mixed_projected_kv(
                        current_inputs,
                        current_key,
                        current_value,
                        current_mask,
                        current_mask,
                        self.user_global_tokens,
                    )
                    return current_key, current_value, current_output

                cacheable_key, cacheable_value, cacheable_output = checkpoint(
                    self_forward,
                    cacheable_inputs,
                    cacheable_mask,
                    use_reentrant=False,
                )
            else:
                cacheable_key, cacheable_value = block.project_kv(cacheable_inputs)
                cacheable_output = block.forward_mixed_projected_kv(
                    cacheable_inputs,
                    cacheable_key,
                    cacheable_value,
                    cacheable_mask,
                    cacheable_mask,
                    self.user_global_tokens,
                )
            user_output = cacheable_output[:, : self.user_global_tokens, :]
            recent_output = cacheable_output[:, self.user_global_tokens :, :]
            recent_output = recent_output * recent_mask.unsqueeze(-1).to(recent_output.dtype)
            layer_caches.append(
                LongerSelfLayerCache(
                    cacheable_key=cacheable_key,
                    cacheable_value=cacheable_value,
                    user_output=user_output,
                    recent_output=recent_output,
                )
            )
            current_user = user_output
            current_recent = recent_output
        return LongerSequenceCache(
            merged_tokens=merged_tokens,
            merged_mask=merged_mask,
            cross_cacheable_key=cross_key,
            cross_cacheable_value=cross_value,
            cross_user_output=cross_user_output,
            cross_recent_output=cross_recent_output,
            user_mask=user_mask,
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
                cacheable_key=item.cacheable_key.expand(batch_size, -1, -1, -1),
                cacheable_value=item.cacheable_value.expand(batch_size, -1, -1, -1),
                user_output=item.user_output.expand(batch_size, -1, -1),
                recent_output=item.recent_output.expand(batch_size, -1, -1),
            )
            for item in cache.self_layers
        )
        return LongerSequenceCache(
            merged_tokens=cache.merged_tokens.expand(batch_size, -1, -1),
            merged_mask=cache.merged_mask.expand(batch_size, -1),
            cross_cacheable_key=cache.cross_cacheable_key.expand(batch_size, -1, -1, -1),
            cross_cacheable_value=cache.cross_cacheable_value.expand(batch_size, -1, -1, -1),
            cross_user_output=cache.cross_user_output.expand(batch_size, -1, -1),
            cross_recent_output=cache.cross_recent_output.expand(batch_size, -1, -1),
            user_mask=cache.user_mask.expand(batch_size, -1),
            recent_mask=cache.recent_mask.expand(batch_size, -1),
            self_layers=layers,
        )
    def forward(
        self,
        tokens: Tensor,
        mask: Tensor,
        candidate_global_tokens: Tensor,
        cache: LongerSequenceCache | None = None,
        user_global_tokens: Tensor | None = None,
    ) -> Tensor:
        if (
            candidate_global_tokens.size(1) != self.candidate_global_tokens
            or candidate_global_tokens.size(2) != self.token_dim
        ):
            raise ValueError(
                "candidate_global_tokens must have shape "
                f"[batch, {self.candidate_global_tokens}, {self.token_dim}]"
            )
        sequence_cache = (
            self.precompute_cache(tokens, mask, user_global_tokens)
            if cache is None
            else cache
        )
        sequence_cache = self._expand_cache(
            sequence_cache,
            candidate_global_tokens.size(0),
        )
        if sequence_cache.merged_tokens.size(0) != candidate_global_tokens.size(0):
            raise ValueError(
                "LONGER cache and candidate global tokens must have the same batch size"
            )
        if len(sequence_cache.self_layers) != len(self.self_blocks):
            raise ValueError("LONGER cache depth does not match encoder depth")

        candidate_valid = torch.ones(
            candidate_global_tokens.size(0),
            self.candidate_global_tokens,
            dtype=torch.bool,
            device=candidate_global_tokens.device,
        )
        candidate_key, candidate_value = self.cross_block.project_kv(
            candidate_global_tokens
        )
        cross_key = torch.cat([candidate_key, sequence_cache.cross_cacheable_key], dim=2)
        cross_value = torch.cat([candidate_value, sequence_cache.cross_cacheable_value], dim=2)
        cross_key_mask = torch.cat(
            [candidate_valid, sequence_cache.user_mask, sequence_cache.merged_mask],
            dim=1,
        )
        if self.activation_checkpoint and self.training:
            candidate_hidden = checkpoint(
                self.cross_block.forward_full_projected_kv,
                candidate_global_tokens,
                cross_key,
                cross_value,
                candidate_valid,
                cross_key_mask,
                use_reentrant=False,
            )
        else:
            candidate_hidden = self.cross_block.forward_full_projected_kv(
                candidate_global_tokens,
                cross_key,
                cross_value,
                candidate_valid,
                cross_key_mask,
            )
        hidden = torch.cat(
            [
                sequence_cache.cross_user_output,
                candidate_hidden,
                sequence_cache.cross_recent_output,
            ],
            dim=1,
        )

        for block, layer_cache in zip(self.self_blocks, sequence_cache.self_layers):
            candidate_key, candidate_value = block.project_kv(candidate_hidden)
            key = torch.cat([candidate_key, layer_cache.cacheable_key], dim=2)
            value = torch.cat([candidate_value, layer_cache.cacheable_value], dim=2)
            key_mask = torch.cat(
                [candidate_valid, sequence_cache.user_mask, sequence_cache.recent_mask],
                dim=1,
            )
            if self.activation_checkpoint and self.training:
                candidate_hidden = checkpoint(
                    block.forward_full_projected_kv,
                    candidate_hidden,
                    key,
                    value,
                    candidate_valid,
                    key_mask,
                    use_reentrant=False,
                )
            else:
                candidate_hidden = block.forward_full_projected_kv(
                    candidate_hidden,
                    key,
                    value,
                    candidate_valid,
                    key_mask,
                )
            hidden = torch.cat(
                [layer_cache.user_output, candidate_hidden, layer_cache.recent_output],
                dim=1,
            )
        return hidden.flatten(start_dim=1)


class FeatureEncoderBank(nn.Module):
    def __init__(
        self,
        config: AppConfig,
        vocab_maps: dict[str, dict[str, int]],
        embedding_dim: int,
        build_sequence_summaries: bool = True,
        included_scalar_feature_names: set[str] | None = None,
        embedding_size_override: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.build_sequence_summaries = build_sequence_summaries
        self.sequence_token_dim = config.model.token_dim
        self.embedding_size_override = embedding_size_override
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
        self.sequence_field_input_dims: dict[str, dict[str, int]] = {}
        self.sequence_event_input_dims: dict[str, int] = {}
        self.sequence_step_projectors = nn.ModuleDict()
        self.sequence_query_projectors = nn.ModuleDict()
        self.sequence_user_global_projectors = nn.ModuleDict()
        self.sequence_queries = nn.ParameterDict()
        self.sequence_cls_tokens = nn.ParameterDict()
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
            if _embedding_share_target(encoding) is not None:
                base_name = self._shared_base_name(qualified)
                if base_name in scalar_feature_names:
                    self.included_scalar_feature_names.add(base_name)

        self.embedding_sharding_plan: EmbeddingShardingPlan | None = None
        if config.training.embedding_distribution == "sharded":
            table_specs: list[EmbeddingTableSpec] = []
            for feature in config.features:
                if (
                    feature.name not in self.included_scalar_feature_names
                    or feature.kind != "categorical"
                ):
                    continue
                encoding = _encoding_for(config, feature.name)
                if _embedding_share_target(encoding) is not None:
                    continue
                table_specs.append(
                    EmbeddingTableSpec(
                        name=feature.name,
                        num_embeddings=_embedding_size(
                            config,
                            vocab_maps,
                            feature.name,
                            embedding_size_override,
                        ),
                        embedding_dim=categorical_dims[feature.name],
                    )
                )
            for sequence in config.sequences:
                for field in sequence.fields:
                    if field.kind != "categorical":
                        continue
                    qualified = field.qualified_name(sequence.name)
                    encoding = _encoding_for(config, qualified)
                    if _embedding_share_target(encoding) is not None:
                        continue
                    table_specs.append(
                        EmbeddingTableSpec(
                            name=qualified,
                            num_embeddings=_embedding_size(
                                config,
                                vocab_maps,
                                qualified,
                                embedding_size_override,
                            ),
                            embedding_dim=categorical_dims[qualified],
                        )
                    )
            world_size = (
                torch_dist.get_world_size()
                if torch_dist.is_available() and torch_dist.is_initialized()
                else 1
            )
            self.embedding_sharding_plan = plan_embedding_shards(
                table_specs,
                world_size=world_size,
                strategy=config.training.embedding_sharding.strategy,
                table_wise_max_rows=(
                    config.training.embedding_sharding.table_wise_max_rows
                ),
            )

        for feature in config.features:
            if feature.name not in self.included_scalar_feature_names:
                continue
            if feature.kind == "dense":
                continue
            encoding = _encoding_for(config, feature.name)
            if _embedding_share_target(encoding) is not None:
                continue
            feature_embedding_dim = categorical_dims[feature.name]
            size = _embedding_size(
                config,
                vocab_maps,
                feature.name,
                embedding_size_override,
            )
            self.embeddings[feature.name] = self._build_id_embedding(
                feature.name,
                size,
                feature_embedding_dim,
                sparse_gradients,
            )

        for sequence in config.sequences:
            step_input_dim = 0
            field_input_dims: dict[str, int] = {}
            for field in sequence.fields:
                qualified = field.qualified_name(sequence.name)
                if field.kind == "categorical":
                    key = self.sequence_field_embedding_keys[qualified]
                    encoding = _encoding_for(config, qualified)
                    field_embedding_dim = categorical_dims[qualified]
                    field_input_dims[field.name] = field_embedding_dim
                    if _embedding_share_target(encoding) is not None:
                        step_input_dim += field_embedding_dim
                        continue
                    size = _embedding_size(
                        config,
                        vocab_maps,
                        qualified,
                        embedding_size_override,
                    )
                    self.embeddings[key] = self._build_id_embedding(
                        qualified,
                        size,
                        field_embedding_dim,
                        sparse_gradients,
                    )
                    step_input_dim += field_embedding_dim
                else:
                    field_input_dims[field.name] = field.dimension
                    step_input_dim += field.dimension
            sequence_key = self._module_key(sequence.name)
            self.sequence_field_input_dims[sequence.name] = field_input_dims
            self.sequence_event_input_dims[sequence.name] = step_input_dim
            if not self.build_sequence_summaries:
                self.output_dims[sequence.name] = self.sequence_token_dim
                continue
            if sequence.encoder == "longer":
                self.sequence_step_projectors[sequence_key] = _projection_mlp(
                    step_input_dim,
                    self.sequence_token_dim,
                    config.model.hidden_dim,
                    config.model.ffn_activation,
                )
            else:
                self.sequence_step_projectors[sequence_key] = nn.Linear(
                    step_input_dim,
                    self.sequence_token_dim,
                )
            if sequence.max_length is not None:
                position_dim = self.sequence_token_dim
                if sequence.encoder == "longer":
                    if sequence.time_delta_field is None:
                        raise ValueError("LONGER requires time_delta_field")
                    position_dim = step_input_dim - field_input_dims[sequence.time_delta_field]
                self.sequence_position_embeddings[self._module_key(sequence.name)] = _init_embedding(
                    nn.Embedding(
                        sequence.max_length,
                        position_dim,
                    ),
                    config.model.init_std,
                )
            query_token_dim = self.sequence_token_dim
            if sequence.encoder == "longer":
                user_global_tokens = (
                    sequence.longer_user_global_tokens + sequence.longer_cls_tokens
                )
                longer_encoder = LongerSequenceEncoder(
                    self.sequence_token_dim,
                    config.model.num_heads,
                    config.model.hidden_dim,
                    sequence.longer_query_tokens,
                    sequence.longer_self_layers,
                    sequence.rankmixer_summary_tokens,
                    sequence.longer_token_merge,
                    sequence.longer_inner_layers,
                    user_global_tokens=user_global_tokens,
                    attention_backend=config.runtime.attention_backend,
                    activation_checkpoint=_activation_checkpoint_enabled(
                        config.runtime.activation_checkpoint
                    ),
                    checkpoint_token_merger=_activation_checkpoint_enabled(
                        config.runtime.activation_checkpoint,
                        full_only=True,
                    ),
                )
                self.sequence_longer_encoders[sequence_key] = longer_encoder
                query_token_dim = longer_encoder.token_dim
                output_dim = longer_encoder.output_dim
            else:
                output_dim = sequence.rankmixer_summary_tokens * self.sequence_token_dim
            if sequence.encoder == "longer":
                candidate_tokens = sequence.resolved_longer_candidate_global_tokens()
                if candidate_tokens > 0:
                    target_dim = sum(self.output_dims[name] for name in sequence.target_inputs)
                    self.sequence_query_projectors[sequence_key] = _projection_mlp(
                        target_dim,
                        candidate_tokens * query_token_dim,
                        config.model.hidden_dim,
                        config.model.ffn_activation,
                    )
                if sequence.longer_user_global_tokens > 0:
                    user_dim = sum(
                        self.output_dims[name]
                        for name in sequence.longer_user_global_inputs
                    )
                    self.sequence_user_global_projectors[sequence_key] = _projection_mlp(
                        user_dim,
                        sequence.longer_user_global_tokens * query_token_dim,
                        config.model.hidden_dim,
                        config.model.ffn_activation,
                    )
                if sequence.longer_cls_tokens > 0:
                    self.sequence_cls_tokens[sequence_key] = _normal_parameter(
                        (1, sequence.longer_cls_tokens, query_token_dim),
                        config.model.init_std,
                    )
            else:
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
            if _embedding_share_target(encoding) is not None:
                if feature.name not in categorical_dims:
                    raise ValueError(
                        f"shared embedding feature {feature.name!r} references unknown feature"
                    )
                base_name = self._shared_base_name(feature.name)
                base_key = self._embedding_key(base_name)
                if base_key not in self.embeddings:
                    raise ValueError(
                        f"shared embedding base {base_name!r} has no embedding"
                    )
                self.embeddings[feature.name] = self.embeddings[base_key]
                self.output_dims[feature.name] = categorical_dims[feature.name]

        for sequence in config.sequences:
            for field in sequence.fields:
                if field.kind != "categorical":
                    continue
                qualified = field.qualified_name(sequence.name)
                encoding = _encoding_for(config, qualified)
                if _embedding_share_target(encoding) is not None:
                    if qualified not in categorical_dims:
                        raise ValueError(
                            f"shared embedding sequence field {qualified!r} references "
                            "unknown feature"
                        )
                    base_name = self._shared_base_name(qualified)
                    base_key = self._embedding_key(base_name)
                    if base_key not in self.embeddings:
                        raise ValueError(
                            f"shared embedding base {base_name!r} has no embedding"
                        )
                    self.embeddings[self.sequence_field_embedding_keys[qualified]] = self.embeddings[base_key]

    @staticmethod
    def _module_key(name: str) -> str:
        return name.replace(".", "__")

    def _build_id_embedding(
        self,
        table_name: str,
        num_embeddings: int,
        embedding_dim: int,
        sparse_gradients: bool,
    ) -> nn.Module:
        if self.config.training.embedding_distribution == "replicated":
            embedding = _init_embedding(
                nn.Embedding(
                    num_embeddings,
                    embedding_dim,
                    padding_idx=0,
                    sparse=sparse_gradients,
                ),
                self.config.model.init_std,
            )
            embedding._mdl_id_embedding = True  # type: ignore[attr-defined]
            return embedding
        if self.embedding_sharding_plan is None:
            raise RuntimeError("sharded embedding plan was not initialized")
        try:
            shard_spec = self.embedding_sharding_plan.tables[table_name]
        except KeyError as error:
            raise RuntimeError(
                f"missing sharding plan entry for embedding {table_name!r}"
            ) from error
        return ShardedEmbedding(
            num_embeddings,
            embedding_dim,
            table_name=table_name,
            shard_spec=shard_spec,
            padding_idx=0,
            local_dedup=self.config.training.embedding_sharding.local_dedup,
            init_std=self.config.model.init_std,
        )

    def _embedding_key(self, name: str) -> str:
        return self.sequence_field_embedding_keys.get(name, name)

    def _shared_base_name(self, name: str) -> str:
        seen: set[str] = set()
        current = name
        while True:
            if current in seen:
                raise ValueError(f"shared embedding cycle detected at {name!r}")
            seen.add(current)
            encoding = _encoding_for(self.config, current)
            target = _embedding_share_target(encoding)
            if target is None:
                return current
            current = target

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

    def _embedded_sequence_parts(
        self,
        sequence: SequenceConfig,
        value: dict[str, Any],
        preencoded_inputs: dict[str, Tensor] | None = None,
    ) -> tuple[dict[str, Tensor], Tensor]:
        field_values = value["fields"]
        lengths = value["lengths"].long()
        parts: dict[str, Tensor] = {}
        sharded_requests: list[tuple[str, ShardedEmbedding, Tensor]] = []
        for field in sequence.fields:
            tensor = field_values[field.name]
            if field.kind == "categorical":
                qualified = field.qualified_name(sequence.name)
                if preencoded_inputs is not None and qualified in preencoded_inputs:
                    parts[field.name] = preencoded_inputs[qualified]
                    continue
                embedding = self.embeddings[
                    self.sequence_field_embedding_keys[qualified]
                ]
                indices = tensor.long()
                if isinstance(embedding, ShardedEmbedding):
                    sharded_requests.append((field.name, embedding, indices))
                else:
                    parts[field.name] = embedding(indices)
            else:
                dense = tensor.float()
                if dense.dim() == 2:
                    dense = dense.unsqueeze(-1)
                if dense.size(-1) != field.dimension:
                    raise ValueError(
                        f"sequence {sequence.name!r} field {field.name!r} expected "
                        f"dimension {field.dimension}, got {dense.size(-1)}"
                    )
                parts[field.name] = dense
        if sharded_requests:
            sharded_outputs = grouped_sharded_embedding_lookup(
                (embedding, indices)
                for _name, embedding, indices in sharded_requests
            )
            for (name, _embedding, _indices), output in zip(
                sharded_requests, sharded_outputs
            ):
                parts[name] = output
        if not parts:
            raise ValueError(f"sequence {sequence.name!r} has no fields")
        return parts, lengths

    def _align_sequence_inputs(
        self,
        sequence: SequenceConfig,
        inputs: Tensor,
        lengths: Tensor,
        target_length: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        # Dataloader batches are already truncated to sequence.max_length and
        # padded only to the longest row in the current batch.  Retain that
        # compact physical width by default; expanding every batch back to the
        # configured capacity wastes work, especially for LONGER profiles with
        # 2k/5k maximum lengths.  Callers that combine several aligned sequence
        # inputs can still request one explicit shared width.
        output_length = inputs.size(1) if target_length is None else target_length
        aligned, mask = self._right_aligned_sequence(inputs, lengths, output_length)
        if sequence.sequence_order == "oldest_to_newest" or aligned.size(1) == 0:
            return aligned, mask
        positions = torch.arange(aligned.size(1), device=aligned.device).view(1, -1)
        valid_lengths = lengths.clamp(min=0, max=aligned.size(1)).view(-1, 1)
        valid_starts = aligned.size(1) - valid_lengths
        source_positions = (aligned.size(1) - 1 - (positions - valid_starts)).clamp(
            min=0,
            max=aligned.size(1) - 1,
        )
        gather_index = source_positions.unsqueeze(-1).expand(-1, -1, aligned.size(-1))
        reversed_inputs = aligned.gather(1, gather_index)
        return reversed_inputs * mask.unsqueeze(-1).to(reversed_inputs.dtype), mask

    def _position_inputs(
        self,
        sequence: SequenceConfig,
        lengths: Tensor,
        token_count: int,
    ) -> Tensor:
        position_key = self._module_key(sequence.name)
        position_embedding = self.sequence_position_embeddings[position_key]
        max_positions = position_embedding.num_embeddings
        valid_lengths = lengths.clamp(min=0, max=token_count).view(-1, 1)
        physical_positions = torch.arange(token_count, device=lengths.device).view(1, -1)
        relative_positions = (physical_positions - (token_count - valid_lengths)).clamp(
            min=0,
            max=max_positions - 1,
        )
        return position_embedding(relative_positions)

    def encode_sequence_event_inputs(
        self,
        sequence_name: str,
        value: dict[str, Any],
        target_length: int | None = None,
        preencoded_inputs: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        sequence = self.sequences_by_name[sequence_name]
        parts, lengths = self._embedded_sequence_parts(
            sequence,
            value,
            preencoded_inputs,
        )
        event_inputs = torch.cat([parts[field.name] for field in sequence.fields], dim=-1)
        return self._align_sequence_inputs(
            sequence,
            event_inputs,
            lengths,
            target_length=target_length,
        )

    def _multi_field_sequence_tokens(
        self,
        sequence: SequenceConfig,
        value: dict[str, Any],
        preencoded_inputs: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        parts, lengths = self._embedded_sequence_parts(
            sequence,
            value,
            preencoded_inputs,
        )
        position_key = self._module_key(sequence.name)
        if sequence.encoder == "longer":
            if sequence.time_delta_field is None:
                raise ValueError(f"sequence {sequence.name!r} requires time_delta_field")
            base_inputs = torch.cat(
                [
                    parts[field.name]
                    for field in sequence.fields
                    if field.name != sequence.time_delta_field
                ],
                dim=-1,
            )
            time_delta = parts[sequence.time_delta_field]
            combined = torch.cat([base_inputs, time_delta], dim=-1)
            combined, mask = self._align_sequence_inputs(sequence, combined, lengths)
            base_dim = base_inputs.size(-1)
            aligned_base = combined[:, :, :base_dim]
            aligned_time_delta = combined[:, :, base_dim:]
            if position_key in self.sequence_position_embeddings and combined.size(1) > 0:
                aligned_base = aligned_base + self._position_inputs(
                    sequence,
                    lengths,
                    combined.size(1),
                )
            projector_inputs = torch.cat([aligned_base, aligned_time_delta], dim=-1)
            tokens = self.sequence_step_projectors[position_key](projector_inputs)
        else:
            step_inputs = torch.cat([parts[field.name] for field in sequence.fields], dim=-1)
            step_inputs, mask = self._align_sequence_inputs(sequence, step_inputs, lengths)
            tokens = self.sequence_step_projectors[position_key](step_inputs)
            if position_key in self.sequence_position_embeddings and tokens.size(1) > 0:
                tokens = tokens + self._position_inputs(sequence, lengths, tokens.size(1))
        tokens = tokens * mask.unsqueeze(-1).to(dtype=tokens.dtype)
        return tokens, mask

    def encode_sequence_tokens(
        self,
        sequence_name: str,
        value: dict[str, Any],
        preencoded_inputs: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        sequence = self.sequences_by_name[sequence_name]
        return self._multi_field_sequence_tokens(
            sequence,
            value,
            preencoded_inputs,
        )

    def _pool_sequence(
        self,
        sequence: SequenceConfig,
        tokens: Tensor,
        mask: Tensor,
        encoded: dict[str, Tensor],
        sequence_cache: LongerSequenceCache | None = None,
    ) -> Tensor:
        output_dim = self.output_dims[sequence.name]
        if tokens.size(1) == 0 and sequence.encoder != "longer":
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
        if sequence.encoder == "longer":
            candidate_count = sequence.resolved_longer_candidate_global_tokens()
            if candidate_count > 0:
                query_input = torch.cat(
                    [encoded[name] for name in sequence.target_inputs],
                    dim=1,
                )
                candidate_globals = self.sequence_query_projectors[sequence_key](
                    query_input
                ).view(tokens.size(0), candidate_count, query_token_dim)
            else:
                candidate_globals = tokens.new_zeros(
                    tokens.size(0),
                    0,
                    query_token_dim,
                )
            user_globals = self._longer_user_global_tokens(
                sequence,
                encoded,
                tokens.size(0),
                tokens,
            )
            return self.sequence_longer_encoders[sequence_key](
                tokens,
                mask,
                candidate_globals,
                cache=sequence_cache,
                user_global_tokens=user_globals,
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
        scores = (tokens * query[:, :1, :]).sum(dim=-1) / math.sqrt(tokens.size(-1))
        scores = scores.masked_fill(~mask, -1.0e9)
        weights = torch.softmax(scores, dim=1) * mask.to(dtype=tokens.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-9)
        return (tokens * weights.unsqueeze(-1)).sum(dim=1)

    def _longer_user_global_tokens(
        self,
        sequence: SequenceConfig,
        encoded: dict[str, Tensor],
        batch_size: int,
        reference: Tensor,
    ) -> Tensor:
        sequence_key = self._module_key(sequence.name)
        query_token_dim = self.sequence_longer_encoders[sequence_key].token_dim
        parts: list[Tensor] = []
        if sequence.longer_user_global_tokens > 0:
            user_input = torch.cat(
                [encoded[name] for name in sequence.longer_user_global_inputs],
                dim=1,
            )
            parts.append(
                self.sequence_user_global_projectors[sequence_key](user_input).view(
                    batch_size,
                    sequence.longer_user_global_tokens,
                    query_token_dim,
                )
            )
        if sequence.longer_cls_tokens > 0:
            parts.append(self.sequence_cls_tokens[sequence_key].expand(batch_size, -1, -1))
        if not parts:
            return reference.new_zeros(batch_size, 0, query_token_dim)
        return torch.cat(parts, dim=1)

    def _preencode_sharded_inputs(
        self,
        features: dict[str, Any],
        *,
        scalar_names: set[str] | None = None,
        additional_scalar_names: set[str] | None = None,
        sequence_names: set[str] | None = None,
    ) -> dict[str, Tensor]:
        """Issue one batch-wide grouped lookup for selected sharded ID inputs.

        The grouped embedding implementation partitions incompatible embedding
        widths internally.  Collecting requests here still collapses every
        compatible scalar and sequence field into one collective group and also
        deduplicates IDs shared by aliases such as ``item_id`` and
        ``hist.item_id``.
        """

        requests: list[tuple[str, ShardedEmbedding, Tensor]] = []
        for feature in self.config.features:
            if feature.name not in self.included_scalar_feature_names:
                continue
            if (
                scalar_names is not None
                and feature.name not in scalar_names
                and (
                    additional_scalar_names is None
                    or feature.name not in additional_scalar_names
                )
            ):
                continue
            if feature.kind != "categorical":
                continue
            value = features[feature.name]
            if not isinstance(value, Tensor):
                raise ValueError(f"scalar feature {feature.name!r} must be a tensor")
            embedding = self.embeddings[feature.name]
            if isinstance(embedding, ShardedEmbedding):
                requests.append((feature.name, embedding, value.long()))

        for sequence in self.config.sequences:
            if sequence_names is not None and sequence.name not in sequence_names:
                continue
            value = features[sequence.name]
            if not isinstance(value, dict):
                raise ValueError(f"sequence {sequence.name!r} must be a payload dict")
            field_values = value["fields"]
            for field in sequence.fields:
                if field.kind != "categorical":
                    continue
                qualified = field.qualified_name(sequence.name)
                embedding = self.embeddings[
                    self.sequence_field_embedding_keys[qualified]
                ]
                if not isinstance(embedding, ShardedEmbedding):
                    continue
                indices = field_values[field.name]
                if not isinstance(indices, Tensor):
                    raise ValueError(
                        f"sequence {sequence.name!r} field {field.name!r} must be a tensor"
                    )
                requests.append((qualified, embedding, indices.long()))

        if not requests:
            return {}
        outputs = grouped_sharded_embedding_lookup(
            (embedding, indices)
            for _name, embedding, indices in requests
        )
        return {
            name: output
            for (name, _embedding, _indices), output in zip(requests, outputs)
        }

    def encode_scalar_features(
        self,
        features: dict[str, Any],
        names: set[str] | None = None,
        preencoded_inputs: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        encoded: dict[str, Tensor] = {}
        sharded_requests: list[tuple[str, ShardedEmbedding, Tensor]] = []
        for feature in self.config.features:
            if feature.name not in self.included_scalar_feature_names:
                continue
            if names is not None and feature.name not in names:
                continue
            value = features[feature.name]
            if not isinstance(value, Tensor):
                raise ValueError(f"scalar feature {feature.name!r} must be a tensor")
            if feature.kind == "categorical":
                if preencoded_inputs is not None and feature.name in preencoded_inputs:
                    encoded[feature.name] = preencoded_inputs[feature.name]
                    continue
                embedding = self.embeddings[feature.name]
                if isinstance(embedding, ShardedEmbedding):
                    sharded_requests.append((feature.name, embedding, value.long()))
                    continue
            encoded[feature.name] = self._encode_scalar_feature(feature, value)
        if sharded_requests:
            sharded_outputs = grouped_sharded_embedding_lookup(
                (embedding, indices)
                for _name, embedding, indices in sharded_requests
            )
            for (name, _embedding, _indices), output in zip(
                sharded_requests, sharded_outputs
            ):
                encoded[name] = output
        return encoded


    def precompute_request_cache(self, features: dict[str, Any]) -> dict[str, LongerSequenceCache]:
        caches: dict[str, LongerSequenceCache] = {}
        if not self.build_sequence_summaries:
            return caches
        longer_sequences = [
            sequence
            for sequence in self.config.sequences
            if sequence.encoder == "longer"
        ]
        user_input_names = {
            name
            for sequence in longer_sequences
            for name in sequence.longer_user_global_inputs
        }
        preencoded_inputs = self._preencode_sharded_inputs(
            features,
            scalar_names=user_input_names,
            sequence_names={sequence.name for sequence in longer_sequences},
        )
        encoded_user = self.encode_scalar_features(
            features,
            user_input_names,
            preencoded_inputs,
        )
        for sequence in longer_sequences:
            value = features[sequence.name]
            if not isinstance(value, dict):
                raise ValueError(f"sequence {sequence.name!r} must be a payload dict")
            tokens, mask = self._multi_field_sequence_tokens(
                sequence,
                value,
                preencoded_inputs,
            )
            user_globals = self._longer_user_global_tokens(
                sequence,
                encoded_user,
                tokens.size(0),
                tokens,
            )
            caches[sequence.name] = self.sequence_longer_encoders[
                self._module_key(sequence.name)
            ].precompute_cache(tokens, mask, user_globals)
        return caches

    def forward(
        self,
        features: dict[str, Any],
        request_cache: dict[str, LongerSequenceCache] | None = None,
    ) -> dict[str, Tensor]:
        if request_cache is None and self.config.model.use_request_cache:
            request_cache = self.precompute_request_cache(features)
        active_sequence_names = (
            {
                sequence.name
                for sequence in self.config.sequences
                if not (
                    sequence.encoder == "longer"
                    and request_cache is not None
                    and sequence.name in request_cache
                )
            }
            if self.build_sequence_summaries
            else set()
        )
        preencoded_inputs = self._preencode_sharded_inputs(
            features,
            sequence_names=active_sequence_names,
        )
        encoded = self.encode_scalar_features(
            features,
            preencoded_inputs=preencoded_inputs,
        )

        if not self.build_sequence_summaries:
            return encoded
        for sequence in self.config.sequences:
            value = features[sequence.name]
            if not isinstance(value, dict):
                raise ValueError(f"sequence {sequence.name!r} must be a payload dict")
            sequence_cache = None if request_cache is None else request_cache.get(sequence.name)
            if sequence.encoder == "longer" and sequence_cache is not None:
                # The cache owns sequence embedding, user/CLS globals, merge, K/V,
                # and sequence-side attention. Candidate globals are recomputed.
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
                tokens, mask = self._multi_field_sequence_tokens(
                    sequence,
                    value,
                    preencoded_inputs,
                )
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
        if self.input_dim % num_tokens != 0:
            raise ValueError(
                "rankmixer tokenization requires an input width divisible by num_feature_tokens: "
                f"{self.input_dim} % {num_tokens} != 0"
            )
        self.input_slice_dim = self.input_dim // num_tokens
        self.projection = PerTokenLinear(num_tokens, self.input_slice_dim, token_dim)

    def forward(self, encoded: dict[str, Tensor]) -> Tensor:
        values = torch.cat([encoded[name] for name in self.input_names], dim=1)
        sliced = values.view(values.size(0), self.num_tokens, self.input_slice_dim)
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
        self.require_compact_sequence_batches = (
            getattr(config.runtime, "require_compact_sequence_batches", False)
        )

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
            self.auto_ns_projection = _projection_mlp(
                input_dim,
                self.num_ns_tokens * self.token_dim,
                config.model.hidden_dim,
                config.model.ffn_activation,
            )
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
        self.sequence_input_names = {
            name
            for group in self.sequence_groups
            for name in group.inputs
            if name in self.sequence_by_name
        }
        self.sequence_scalar_input_names = {
            name
            for group in self.sequence_groups
            for name in group.inputs
            if name not in self.sequence_by_name
        }
        if self.num_ns_tokens <= 0:
            raise ValueError("OneTrans requires at least one NS token")

    def _preencode_inputs(
        self,
        features: dict[str, Any],
        scalar_names: set[str],
        *,
        include_sequences: bool,
    ) -> dict[str, Tensor]:
        preencode = getattr(self.encoder_bank, "_preencode_sharded_inputs", None)
        if preencode is None:
            # Lightweight alignment-test encoders intentionally implement only
            # the public tokenizer-facing surface.
            return {}
        return preencode(
            features,
            scalar_names=scalar_names,
            additional_scalar_names=(
                self.sequence_scalar_input_names if include_sequences else None
            ),
            sequence_names=(self.sequence_input_names if include_sequences else set()),
        )

    def _group_input_dim(self, group: TokenGroupConfig) -> int:
        return sum(
            self.encoder_bank.sequence_event_input_dims[name]
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
                payload_length = self._payload_max_length(value)
                if configured_length is not None:
                    payload_length = min(payload_length, configured_length)
                max_length = max(
                    max_length,
                    payload_length,
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
            aligned, timestamp_mask = self.encoder_bank._align_sequence_inputs(
                sequence,
                raw,
                value["lengths"].long(),
                target_length=expected_mask.size(1),
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
        preencoded_inputs: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        max_length, _lengths = self._group_sequence_length(features, group)
        parts: list[Tensor] = []
        mask: Tensor | None = None
        for name in group.inputs:
            value = features[name]
            if name in self.sequence_by_name:
                if not isinstance(value, dict):
                    raise ValueError(f"sequence {name!r} must be a payload dict")
                if preencoded_inputs:
                    tokens, current_mask = self.encoder_bank.encode_sequence_event_inputs(
                        name,
                        value,
                        target_length=max_length,
                        preencoded_inputs=preencoded_inputs,
                    )
                else:
                    tokens, current_mask = self.encoder_bank.encode_sequence_event_inputs(
                        name,
                        value,
                        target_length=max_length,
                    )
                mask = current_mask if mask is None else mask & current_mask
                parts.append(tokens)
                continue
            feature = self.by_name[name]
            if not isinstance(value, Tensor):
                raise ValueError(f"scalar feature {name!r} must be a tensor")
            scalar = (
                preencoded_inputs[name]
                if preencoded_inputs is not None and name in preencoded_inputs
                else self.encoder_bank._encode_scalar_feature(feature, value)
            )
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

    def _trim_all_invalid_prefix(self, tokens: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if mask.size(1) == 0:
            return tokens, mask
        valid_columns = mask.any(dim=0)
        if not bool(valid_columns.any().item()):
            return tokens[:, :0, :], mask[:, :0]
        first_valid = int(torch.nonzero(valid_columns, as_tuple=False)[0].item())
        return tokens[:, first_valid:, :], mask[:, first_valid:]

    def _sequence_token_part(
        self,
        features: dict[str, Any],
        preencoded_inputs: dict[str, Tensor] | None = None,
    ) -> OneTransRequestCache:
        sequence_tokens: list[Tensor] = []
        sequence_masks: list[Tensor] = []
        sequence_timestamps: list[Tensor] = []
        for index, (group, projection) in enumerate(zip(self.sequence_groups, self.sequence_projectors)):
            tokens, mask, timestamps = self._sequence_group_tokens(
                group,
                projection,
                features,
                preencoded_inputs,
            )
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
        if self.require_compact_sequence_batches:
            if mask.size(1):
                compact = mask[:, 0].any()
                message = (
                    "runtime.require_compact_sequence_batches requires sequence "
                    "payloads padded only to the longest row in the batch"
                )
                if mask.device.type == "cuda" and hasattr(torch, "_assert_async"):
                    torch._assert_async(compact, message)
                elif not bool(compact.item()):
                    raise ValueError(message)
        else:
            tokens, mask = self._trim_all_invalid_prefix(tokens, mask)
        return OneTransRequestCache(s_tokens=tokens, s_valid_mask=mask)

    def precompute_request_cache(
        self,
        features: dict[str, Any],
        preencoded_inputs: dict[str, Tensor] | None = None,
    ) -> OneTransRequestCache:
        if preencoded_inputs is None:
            preencoded_inputs = self._preencode_inputs(
                features,
                set(),
                include_sequences=True,
            )
        return self._sequence_token_part(features, preencoded_inputs)

    def forward(
        self,
        features: dict[str, Any],
        request_cache: OneTransRequestCache | None = None,
        encoded_features: dict[str, Tensor] | None = None,
        preencoded_inputs: dict[str, Tensor] | None = None,
    ) -> OneTransOutput:
        if preencoded_inputs is None:
            preencoded_inputs = self._preencode_inputs(
                features,
                self.ns_input_names if encoded_features is None else set(),
                include_sequences=request_cache is None,
            )
        if request_cache is None and self.config.model.use_request_cache:
            request_cache = self.precompute_request_cache(
                features,
                preencoded_inputs,
            )
        encoded = (
            encoded_features
            if encoded_features is not None
            else self.encoder_bank.encode_scalar_features(
                features,
                self.ns_input_names,
                preencoded_inputs,
            )
        )
        cache = (
            self._sequence_token_part(features, preencoded_inputs)
            if request_cache is None
            else request_cache
        )
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

    @staticmethod
    def _project_ns_batched(tokens: Tensor, layers: nn.ModuleList) -> Tensor:
        if tokens.size(1) != len(layers):
            raise ValueError(
                f"expected {len(layers)} NS tokens, got {tokens.size(1)}"
            )
        if not layers:
            return tokens
        weight = torch.stack([layer.weight for layer in layers], dim=0)
        bias = torch.stack([layer.bias for layer in layers], dim=0)
        projected = torch.bmm(
            tokens.transpose(0, 1),
            weight.transpose(1, 2),
        )
        projected = projected + bias.unsqueeze(1).to(dtype=projected.dtype)
        return projected.transpose(0, 1)

    @classmethod
    def _project_ns(cls, tokens: Tensor, layers: nn.ModuleList) -> Tensor:
        if tokens.size(1) != len(layers):
            raise ValueError(
                f"expected {len(layers)} NS tokens, got {tokens.size(1)}"
            )
        if not layers:
            return tokens
        if tokens.device.type == "cuda":
            return cls._project_ns_batched(tokens, layers)
        return torch.cat(
            [
                layer(tokens[:, index, :]).unsqueeze(1)
                for index, layer in enumerate(layers)
            ],
            dim=1,
        )

    def _project_all(self, tokens: Tensor, s_count: int, s_layer: nn.Linear, ns_layers: nn.ModuleList) -> Tensor:
        parts: list[Tensor] = []
        if s_count > 0:
            parts.append(s_layer(tokens[:, :s_count, :]))
        parts.append(self._project_ns(tokens[:, s_count:, :], ns_layers))
        return torch.cat(parts, dim=1)

    def _project_query(self, tokens: Tensor, query_s_count: int) -> Tensor:
        parts: list[Tensor] = []
        if query_s_count > 0:
            parts.append(self.s_query(tokens[:, :query_s_count, :]))
        parts.append(
            self._project_ns(tokens[:, query_s_count:, :], self.ns_query)
        )
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

    def attend_causal_suffix(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_valid_mask: Tensor,
        key_valid_mask: Tensor,
    ) -> Tensor:
        """Attend a packed query suffix with exact bottom-right causality."""

        if self.attention_backend != "flash":
            key_count = key.size(2)
            query_count = query.size(2)
            key_positions = torch.arange(
                key_count, device=query.device
            ).view(1, 1, key_count)
            query_positions = torch.arange(
                key_count - query_count,
                key_count,
                device=query.device,
            ).view(1, query_count, 1)
            allowed = (
                (key_positions <= query_positions)
                & key_valid_mask.unsqueeze(1)
                & query_valid_mask.unsqueeze(-1)
            )
            with _sdpa_context(self.attention_backend):
                attended = torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=allowed.unsqueeze(1),
                    dropout_p=0.0,
                )
            return self.output(self._merge_heads(attended))

        if varlen_attn is None:
            raise RuntimeError(
                "runtime.attention_backend='flash' requires torch.nn.attention.varlen"
            )
        if query.device.type != "cuda":
            raise RuntimeError("Flash varlen attention requires CUDA tensors")
        if query.dtype not in {torch.float16, torch.bfloat16}:
            raise RuntimeError("Flash varlen attention requires FP16 or BF16 tensors")
        query_tokens = query.transpose(1, 2)
        key_tokens = key.transpose(1, 2)
        value_tokens = value.transpose(1, 2)
        query_packing = _VarlenPacking.from_mask(query_valid_mask)
        key_packing = (
            query_packing
            if query_valid_mask is key_valid_mask
            else _VarlenPacking.from_mask(key_valid_mask)
        )
        packed_query = query_packing.pack(query_tokens)
        packed_key = key_packing.pack(key_tokens)
        packed_value = key_packing.pack(value_tokens)
        if packed_query.numel() == 0:
            return self.output(self._merge_heads(torch.zeros_like(query)))
        with torch.profiler.record_function("onetrans::flash_varlen_causal"):
            packed_output = _call_varlen_attention(
                packed_query.contiguous(),
                packed_key.contiguous(),
                packed_value.contiguous(),
                query_packing.cumulative_lengths,
                key_packing.cumulative_lengths,
                query_valid_mask.size(1),
                key_valid_mask.size(1),
                causal=True,
            )
        attended_tokens = query_packing.unpack(packed_output, query_tokens)
        attended = attended_tokens.transpose(1, 2)
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

        query_valid_mask = key_valid_mask.index_select(1, query_indices)
        return self.attend_causal_suffix(
            query,
            key,
            value,
            query_valid_mask,
            key_valid_mask,
        )


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

    def _forward_ns_batched(self, ns_tokens: Tensor) -> Tensor:
        if ns_tokens.size(1) != len(self.ns_ffn):
            raise ValueError(
                f"expected {len(self.ns_ffn)} NS tokens, got {ns_tokens.size(1)}"
            )
        if not self.ns_ffn:
            return ns_tokens
        input_weight = torch.stack(
            [network[0].weight for network in self.ns_ffn],
            dim=0,
        )
        input_bias = torch.stack(
            [network[0].bias for network in self.ns_ffn],
            dim=0,
        )
        output_weight = torch.stack(
            [network[2].weight for network in self.ns_ffn],
            dim=0,
        )
        output_bias = torch.stack(
            [network[2].bias for network in self.ns_ffn],
            dim=0,
        )
        token_major = ns_tokens.transpose(0, 1)
        hidden = torch.bmm(
            token_major,
            input_weight.transpose(1, 2),
        )
        hidden = hidden + input_bias.unsqueeze(1).to(dtype=hidden.dtype)
        hidden = torch.nn.functional.gelu(hidden)
        output = torch.bmm(
            hidden,
            output_weight.transpose(1, 2),
        )
        output = output + output_bias.unsqueeze(1).to(dtype=output.dtype)
        return output.transpose(0, 1)

    def _forward_ns_independent(self, ns_tokens: Tensor) -> Tensor:
        if ns_tokens.size(1) != len(self.ns_ffn):
            raise ValueError(
                f"expected {len(self.ns_ffn)} NS tokens, got {ns_tokens.size(1)}"
            )
        if not self.ns_ffn:
            return ns_tokens
        return torch.cat(
            [
                network(ns_tokens[:, index, :]).unsqueeze(1)
                for index, network in enumerate(self.ns_ffn)
            ],
            dim=1,
        )

    def forward(self, tokens: Tensor, query_s_count: int) -> Tensor:
        parts: list[Tensor] = []
        if query_s_count > 0:
            parts.append(self.s_ffn(tokens[:, :query_s_count, :]))
        ns_tokens = tokens[:, query_s_count:, :]
        if ns_tokens.device.type == "cuda":
            parts.append(self._forward_ns_batched(ns_tokens))
        else:
            parts.append(self._forward_ns_independent(ns_tokens))
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
        input_start: int = 0,
    ) -> OneTransLayerCache:
        normalized = self.norm_attention(s_tokens)
        s_key, s_value = self.attention.project_s_kv(normalized)
        output_mask = valid_mask[:, s_tokens.size(1) - query_s_count :]
        if query_s_count == 0:
            output = s_tokens[:, :0, :]
        else:
            query_tokens = normalized[:, -query_s_count:, :]
            query = self.attention.project_s_query(query_tokens)
            attended = self.attention.attend_causal_suffix(
                query,
                s_key,
                s_value,
                output_mask,
                valid_mask,
            )
            residual = s_tokens[:, -query_s_count:, :]
            hidden = residual + attended
            output = hidden + self.ffn.s_ffn(self.norm_ffn(hidden))
        return OneTransLayerCache(
            s_input=s_tokens,
            s_input_start=input_start,
            s_reused_kv_tokens=0,
            s_key=s_key,
            s_value=s_value,
            s_output=output,
            s_key_valid_mask=valid_mask,
            s_output_valid_mask=output_mask,
        )

    def extend_precomputed_s(
        self,
        s_tokens: Tensor,
        query_s_count: int,
        valid_mask: Tensor,
        input_start: int,
        previous: OneTransLayerCache,
    ) -> OneTransLayerCache:
        if s_tokens.size(0) != previous.s_input.size(0):
            raise ValueError("incremental OneTrans cache update requires the same request batch size")
        old_start = previous.s_input_start
        old_end = old_start + previous.s_input.size(1)
        new_end = input_start + s_tokens.size(1)
        if input_start < old_start or new_end < old_end:
            raise ValueError(
                "incremental OneTrans layer input must retain a suffix of the previous input "
                "and append new tokens"
            )
        overlap_count = max(0, old_end - input_start)
        if overlap_count > s_tokens.size(1):
            raise ValueError("incremental OneTrans overlap exceeds the new layer input")
        old_offset = input_start - old_start
        overlap_matches = True
        if overlap_count > 0:
            old_overlap = previous.s_input[:, old_offset : old_offset + overlap_count, :]
            old_mask = previous.s_key_valid_mask[
                :, old_offset : old_offset + overlap_count
            ]
            if not torch.equal(old_overlap, s_tokens[:, :overlap_count, :]):
                overlap_matches = False
            if not torch.equal(old_mask, valid_mask[:, :overlap_count]):
                overlap_matches = False
        if not overlap_matches:
            # A moving pyramid window can change deeper-layer states even though
            # the raw sequence is append-only. Rebuild that layer to preserve
            # exact full-recompute semantics rather than reusing stale K/V.
            overlap_count = 0
            old_offset = previous.s_input.size(1)

        normalized = self.norm_attention(s_tokens)
        reused_key = previous.s_key[:, :, old_offset : old_offset + overlap_count, :]
        reused_value = previous.s_value[:, :, old_offset : old_offset + overlap_count, :]
        appended = normalized[:, overlap_count:, :]
        if appended.size(1) > 0:
            appended_key, appended_value = self.attention.project_s_kv(appended)
        else:
            appended_key = reused_key[:, :, :0, :]
            appended_value = reused_value[:, :, :0, :]
        s_key = torch.cat([reused_key, appended_key], dim=2)
        s_value = torch.cat([reused_value, appended_value], dim=2)
        if s_key.size(2) != s_tokens.size(1):
            raise RuntimeError("incremental OneTrans K/V length is inconsistent")

        output_mask = valid_mask[:, s_tokens.size(1) - query_s_count :]
        if query_s_count == 0:
            output = s_tokens[:, :0, :]
        else:
            query_tokens = normalized[:, -query_s_count:, :]
            query = self.attention.project_s_query(query_tokens)
            attended = self.attention.attend_causal_suffix(
                query,
                s_key,
                s_value,
                output_mask,
                valid_mask,
            )
            residual = s_tokens[:, -query_s_count:, :]
            hidden = residual + attended
            output = hidden + self.ffn.s_ffn(self.norm_ffn(hidden))
        return OneTransLayerCache(
            s_input=s_tokens,
            s_input_start=input_start,
            s_reused_kv_tokens=overlap_count,
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
        if key.size(2) != s_count + ns_count:
            raise RuntimeError("OneTrans cached key shape is inconsistent")
        query_mask = torch.ones(
            ns_tokens.size(0), ns_count, dtype=torch.bool, device=ns_tokens.device
        )
        key_mask = torch.cat([s_mask, query_mask], dim=1)
        attended = self.attention.attend_causal_suffix(
            query, key, value, query_mask, key_mask
        )
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
    def __init__(
        self,
        config: AppConfig,
        vocab_maps: dict[str, dict[str, int]],
        embedding_dim: int | None = None,
        included_scalar_feature_names: set[str] | None = None,
        embedding_size_override: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        embedding_dim = config.model.embedding_dim if embedding_dim is None else embedding_dim
        self.encoder_bank = FeatureEncoderBank(
            config,
            vocab_maps,
            embedding_dim,
            # OneTrans consumes raw event embeddings as S-tokens. Building a
            # summary encoder here (especially LONGER for mdl_onetrans) would
            # model the same history twice and restore encode-then-interaction.
            build_sequence_summaries=False,
            included_scalar_feature_names=included_scalar_feature_names,
            embedding_size_override=embedding_size_override,
        )
        self.tokenizer = OneTransTokenizer(config, self.encoder_bank)
        self.ns_token_count = self.tokenizer.num_ns_tokens
        self.unified_position_embeddings = _init_embedding(
            nn.Embedding(
                resolve_onetrans_max_position_embeddings(config),
                config.model.token_dim,
            ),
            config.model.init_std,
        )
        self.blocks = nn.ModuleList(OneTransBlock(config, self.ns_token_count) for _ in range(config.model.num_layers))

    def _add_unified_position_embeddings(
        self,
        tokens: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        if tokens.dim() != 3:
            raise ValueError("OneTrans tokens must have shape [batch, tokens, dim]")
        if valid_mask.shape != tokens.shape[:2]:
            raise ValueError("OneTrans position mask must match the token batch and length")
        capacity = self.unified_position_embeddings.num_embeddings
        if tokens.size(1) > capacity:
            raise ValueError(
                "OneTrans unified token count exceeds model.max_position_embeddings: "
                f"{tokens.size(1)} > {capacity}"
            )

        # Padding is kept as a masked prefix so pyramid queries remain aligned.
        # Count only valid tokens to make logical positions independent of other
        # samples' padding; the first NS token follows the final valid S token.
        position_ids = valid_mask.to(torch.long).cumsum(dim=1).sub(1).clamp_min(0)
        position_inputs = self.unified_position_embeddings(position_ids).to(tokens.dtype)
        position_inputs = position_inputs * valid_mask.unsqueeze(-1).to(
            position_inputs.dtype
        )
        return tokens + position_inputs

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

    def precompute_request_cache(
        self,
        features: dict[str, Any],
        preencoded_inputs: dict[str, Tensor] | None = None,
    ) -> OneTransRequestCache:
        token_cache = (
            self.tokenizer.precompute_request_cache(
                features,
                preencoded_inputs,
            )
            if preencoded_inputs
            else self.tokenizer.precompute_request_cache(features)
        )
        current_mask = token_cache.s_valid_mask
        current_tokens = self._add_unified_position_embeddings(
            token_cache.s_tokens,
            current_mask,
        )
        initial_s_count = current_tokens.size(1)
        current_start = 0
        layer_caches: list[OneTransLayerCache] = []
        for layer_index, block in enumerate(self.blocks):
            query_s_count = self._layer_s_count(
                initial_s_count, current_tokens.size(1), layer_index
            )
            layer_cache = block.precompute_s(
                current_tokens,
                query_s_count,
                current_mask,
                input_start=current_start,
            )
            layer_caches.append(layer_cache)
            current_start += current_tokens.size(1) - query_s_count
            current_tokens = layer_cache.s_output
            current_mask = layer_cache.s_output_valid_mask
        return OneTransRequestCache(
            s_tokens=token_cache.s_tokens,
            s_valid_mask=token_cache.s_valid_mask,
            layers=tuple(layer_caches),
        )

    def update_request_cache(
        self,
        features: dict[str, Any],
        previous: OneTransRequestCache,
    ) -> OneTransRequestCache:
        if len(previous.layers) != len(self.blocks):
            raise ValueError("previous OneTrans cache depth does not match backbone")
        token_cache = self.tokenizer.precompute_request_cache(features)
        if token_cache.s_tokens.size(0) != previous.s_tokens.size(0):
            raise ValueError(
                "cross-request OneTrans cache update requires the same request batch size"
            )
        old_count = previous.s_tokens.size(1)
        if token_cache.s_tokens.size(1) < old_count:
            raise ValueError("append-only OneTrans cache cannot remove prior S tokens")
        if not torch.equal(token_cache.s_tokens[:, :old_count, :], previous.s_tokens):
            raise ValueError(
                "OneTrans cross-request cache requires the previous S tokens to be an exact prefix"
            )
        if not torch.equal(token_cache.s_valid_mask[:, :old_count], previous.s_valid_mask):
            raise ValueError(
                "OneTrans cross-request cache requires the previous S mask to be an exact prefix"
            )
        if token_cache.s_tokens.size(1) == old_count:
            return previous

        current_mask = token_cache.s_valid_mask
        current_tokens = self._add_unified_position_embeddings(
            token_cache.s_tokens,
            current_mask,
        )
        initial_s_count = current_tokens.size(1)
        current_start = 0
        layer_caches: list[OneTransLayerCache] = []
        for layer_index, (block, previous_layer) in enumerate(
            zip(self.blocks, previous.layers)
        ):
            query_s_count = self._layer_s_count(
                initial_s_count,
                current_tokens.size(1),
                layer_index,
            )
            layer_cache = block.extend_precomputed_s(
                current_tokens,
                query_s_count,
                current_mask,
                current_start,
                previous_layer,
            )
            layer_caches.append(layer_cache)
            current_start += current_tokens.size(1) - query_s_count
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
        preencoded_inputs: dict[str, Tensor] | None = None,
    ) -> OneTransBackboneState:
        tokenizer_kwargs: dict[str, Any] = {
            "request_cache": request_cache,
            "encoded_features": encoded_features,
        }
        if preencoded_inputs:
            tokenizer_kwargs["preencoded_inputs"] = preencoded_inputs
        tokenized = self.tokenizer(features, **tokenizer_kwargs)
        ns_mask = torch.ones(
            tokenized.feature_tokens.size(0),
            tokenized.ns_token_count,
            dtype=torch.bool,
            device=tokenized.feature_tokens.device,
        )
        valid_mask = torch.cat([tokenized.s_valid_mask, ns_mask], dim=1)
        tokens = self._add_unified_position_embeddings(
            tokenized.feature_tokens,
            valid_mask,
        )
        return OneTransBackboneState(
            tokens=tokens,
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
        elif _activation_checkpoint_enabled(
            self.config.runtime.activation_checkpoint
        ) and self.training:
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
        preencoded_inputs: dict[str, Tensor] | None = None
        if request_cache is None and self.config.model.use_request_cache:
            preencoded_inputs = self.tokenizer._preencode_inputs(
                features,
                self.tokenizer.ns_input_names if encoded_features is None else set(),
                include_sequences=True,
            )
            request_cache = self.precompute_request_cache(
                features,
                preencoded_inputs,
            )
        state = self.prepare(
            features,
            request_cache,
            encoded_features,
            preencoded_inputs,
        )
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
        embedding_size_override: int | None = None,
    ) -> None:
        super().__init__()
        if len(config.sequences) != 1 or config.sequences[0].encoder != "longer":
            raise ValueError("LongerModel requires exactly one encoder=longer sequence")
        self.config = config
        self.sequence_name = config.sequences[0].name
        embedding_dim = config.model.embedding_dim if embedding_dim is None else embedding_dim
        target_inputs = {
            *config.sequences[0].target_inputs,
            *config.sequences[0].longer_user_global_inputs,
        }
        self.encoder_bank = FeatureEncoderBank(
            config,
            vocab_maps,
            embedding_dim,
            included_scalar_feature_names=target_inputs,
            embedding_size_override=embedding_size_override,
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
        return StackedPerTokenFFN(
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
        dtsi_training_output=config.model.sparse_moe_dtsi_training_output,
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
    def __init__(
        self,
        config: AppConfig,
        vocab_maps: dict[str, dict[str, int]],
        embedding_dim: int | None = None,
        embedding_size_override: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.feature_groups = config.tokenization.resolved_feature_tokens(config.features, config.sequences)
        self.feature_token_inputs = config.tokenization.resolved_feature_token_inputs(config.features, config.sequences)
        self.feature_token_count = config.tokenization.resolved_feature_token_count(config.features, config.sequences)
        if config.model.token_dim % self.feature_token_count != 0:
            raise ValueError("rankmixer requires token_dim divisible by feature token count")
        embedding_dim = config.model.embedding_dim if embedding_dim is None else embedding_dim
        self.encoder_bank = FeatureEncoderBank(
            config,
            vocab_maps,
            embedding_dim,
            embedding_size_override=embedding_size_override,
        )
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
            if _activation_checkpoint_enabled(
                self.config.runtime.activation_checkpoint
            ) and self.training:
                feature_tokens = checkpoint(block, feature_tokens, use_reentrant=False)
            else:
                feature_tokens = block(feature_tokens)
        pooled = feature_tokens.mean(dim=1)
        logits = torch.cat([layer(pooled) for layer in self.logit_layers], dim=1)
        output = {"logits": logits}
        output.update(_sparse_moe_outputs(self, logits))
        return output


class ScenarioTower(nn.Module):
    """Top-level per-scenario MLP fallback used by the no-token ablation.

    The MDL paper names a scenario-tower replacement but does not publish its
    exact topology. The repository convention is one independent two-layer MLP
    per scenario over the final mean-pooled feature state. Only the towers
    selected by ``scenario_mask`` contribute to each example.
    """

    def __init__(
        self,
        scenario_count: int,
        token_dim: int,
        hidden_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        if scenario_count <= 0:
            raise ValueError("scenario tower requires at least one scenario")
        self.scenario_count = scenario_count
        self.networks = nn.ModuleList(
            _projection_mlp(token_dim, token_dim, hidden_dim, activation)
            for _ in range(scenario_count)
        )

    def forward(self, feature_tokens: Tensor, scenario_mask: Tensor) -> Tensor:
        pooled_features = feature_tokens.mean(dim=1)
        scenario_states = torch.stack(
            [network(pooled_features) for network in self.networks],
            dim=1,
        )
        return masked_scenario_pool(
            scenario_states,
            scenario_mask,
            include_global=False,
            has_global_state=False,
        )


def _init_domain_interaction_modules(
    block: nn.Module,
    config: AppConfig,
    metadata: ModelMetadata,
) -> None:
    token_dim = config.model.token_dim
    hidden_dim = config.model.hidden_dim
    block.use_task_tokens = config.model.use_task_tokens
    block.use_scenario_tokens = config.model.use_scenario_tokens
    block.use_global_scenario_token = config.model.use_global_scenario_token
    block.use_task_feature_interaction = config.model.use_task_feature_interaction
    block.use_scenario_feature_interaction = config.model.use_scenario_feature_interaction

    scenario_token_count = metadata.scenario_count + int(
        config.model.use_global_scenario_token
    )
    if block.use_scenario_tokens:
        block.scenario_attention = (
            DomainAwareAttention(
                token_dim,
                config.model.num_heads,
                scenario_token_count,
                metadata.feature_token_count,
                hidden_dim,
                attention_backend=config.runtime.attention_backend,
                activation=config.model.ffn_activation,
            )
            if block.use_scenario_feature_interaction
            else None
        )
        block.scenario_rankmixer = (
            None
            if block.use_scenario_feature_interaction
            else RankMixerDomainInteraction(
                token_dim,
                scenario_token_count,
                metadata.feature_token_count,
            )
        )
        # Keep this FFN in every block, including the final block, to match the
        # published MDL propagation equation exactly.
        block.scenario_ffn = PerTokenFFN(
            scenario_token_count,
            token_dim,
            hidden_dim,
            activation=config.model.ffn_activation,
        )
    else:
        block.scenario_attention = None
        block.scenario_rankmixer = None
        block.scenario_ffn = None

    if block.use_task_tokens:
        block.task_attention = (
            DomainAwareAttention(
                token_dim,
                config.model.num_heads,
                metadata.task_count,
                metadata.feature_token_count,
                hidden_dim,
                attention_backend=config.runtime.attention_backend,
                activation=config.model.ffn_activation,
            )
            if block.use_task_feature_interaction
            else None
        )
        block.task_rankmixer = (
            None
            if block.use_task_feature_interaction
            else RankMixerDomainInteraction(
                token_dim,
                metadata.task_count,
                metadata.feature_token_count,
            )
        )
        block.task_ffn = PerTokenFFN(
            metadata.task_count,
            token_dim,
            hidden_dim,
            activation=config.model.ffn_activation,
        )
        block.domain_fused = (
            DomainFusedModule(
                include_global=config.model.use_global_scenario_token,
                has_global_token=config.model.use_global_scenario_token,
            )
            if block.use_scenario_tokens
            else None
        )
    else:
        block.task_attention = None
        block.task_rankmixer = None
        block.task_ffn = None
        block.domain_fused = None


def _domain_interaction_hat(
    domain_tokens: Tensor,
    feature_tokens: Tensor,
    attention: DomainAwareAttention | None,
    rankmixer: RankMixerDomainInteraction | None,
) -> Tensor:
    if attention is not None:
        update, _weights = attention(domain_tokens, feature_tokens)
        return domain_tokens + update
    if rankmixer is not None:
        return rankmixer(domain_tokens, feature_tokens)
    raise RuntimeError("enabled domain tokens require one feature interaction module")


def _gated_sequence_interaction_hat(
    domain_tokens: Tensor,
    ns_hat: Tensor,
    s_tokens: Tensor,
    s_mask: Tensor,
    attention: VariableLengthDomainAttention | None,
    gate: nn.Module | None,
) -> Tensor:
    if attention is None:
        return ns_hat
    if gate is None:
        raise RuntimeError("domain sequence attention requires a residual gate")
    s_update = attention(domain_tokens, s_tokens, s_mask)
    ns_update = ns_hat - domain_tokens
    sequence_gate = gate(torch.cat([domain_tokens, ns_update, s_update], dim=-1))
    return ns_hat + sequence_gate * s_update


def _forward_domain_interaction(
    block: Any,
    feature_tokens: Tensor,
    scenario_tokens: Tensor,
    task_tokens: Tensor,
    scenario_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    scenario_hat: Tensor | None = None
    if block.use_scenario_tokens:
        scenario_hat = _domain_interaction_hat(
            scenario_tokens,
            feature_tokens,
            block.scenario_attention,
            block.scenario_rankmixer,
        )
        scenario_tokens = scenario_hat + block.scenario_ffn(scenario_hat)
    elif scenario_tokens.size(1) != 0:
        raise ValueError("disabled scenario-token path expects an empty tensor")

    if block.use_task_tokens:
        task_hat = _domain_interaction_hat(
            task_tokens,
            feature_tokens,
            block.task_attention,
            block.task_rankmixer,
        )
        if scenario_hat is not None:
            task_hat = block.domain_fused(task_hat, scenario_hat, scenario_mask)
        task_tokens = task_hat + block.task_ffn(task_hat)
    elif task_tokens.size(1) != 0:
        raise ValueError("disabled task-token path expects an empty tensor")
    return scenario_tokens, task_tokens


def _empty_domain_tokens(feature_tokens: Tensor) -> Tensor:
    return feature_tokens.new_empty(
        feature_tokens.size(0),
        0,
        feature_tokens.size(2),
    )


def _active_scenario_token_specs(
    config: AppConfig,
    specs: list[DomainTokenConfig],
) -> list[DomainTokenConfig]:
    if not config.model.use_scenario_tokens:
        return []
    if config.model.use_global_scenario_token:
        return specs
    return [spec for spec in specs if spec.name != "global"]


def _mdl_scalar_feature_names(config: AppConfig) -> set[str]:
    active_scopes = {"feature", "shared"}
    if config.model.use_scenario_tokens:
        active_scopes.add("scenario")
    if config.model.use_task_tokens:
        active_scopes.add("task")
    included = {
        feature.name
        for feature in config.features
        if feature.embedding_scope in active_scopes
    }
    for sequence in config.sequences:
        included.update(sequence.target_inputs)
        included.update(sequence.longer_user_global_inputs)
    return included


def _init_mdl_output_modules(
    model: nn.Module,
    config: AppConfig,
    metadata: ModelMetadata,
) -> None:
    model.scenario_tower = (
        None
        if config.model.use_scenario_tokens
        else ScenarioTower(
            metadata.scenario_count,
            config.model.token_dim,
            config.model.hidden_dim,
            config.model.ffn_activation,
        )
    )
    model.logit_layers = _build_task_heads(
        config,
        config.model.token_dim,
        metadata.task_count,
    )


def _mdl_logits(
    model: Any,
    feature_tokens: Tensor,
    scenario_tokens: Tensor,
    task_tokens: Tensor,
    scenario_mask: Tensor,
) -> Tensor:
    use_scenario_tokens = getattr(model.config.model, "use_scenario_tokens", True)
    use_task_tokens = getattr(model.config.model, "use_task_tokens", True)
    scenario_context: Tensor | None = None
    if not use_scenario_tokens:
        scenario_context = model.scenario_tower(feature_tokens, scenario_mask)
    elif not use_task_tokens:
        use_global_scenario_token = getattr(
            model.config.model,
            "use_global_scenario_token",
            True,
        )
        scenario_context = masked_scenario_pool(
            scenario_tokens,
            scenario_mask,
            include_global=use_global_scenario_token,
            has_global_state=use_global_scenario_token,
        )

    if use_task_tokens:
        prediction_tokens = task_tokens
        if scenario_context is not None:
            prediction_tokens = prediction_tokens + scenario_context.unsqueeze(1)
        return torch.cat(
            [
                layer(prediction_tokens[:, index, :])
                for index, layer in enumerate(model.logit_layers)
            ],
            dim=1,
        )

    tower_input = feature_tokens.mean(dim=1)
    if scenario_context is not None:
        tower_input = tower_input + scenario_context
    return torch.cat([layer(tower_input) for layer in model.logit_layers], dim=1)


class MDLRankMixerBlock(nn.Module):
    def __init__(
        self,
        config: AppConfig,
        metadata: ModelMetadata,
    ) -> None:
        super().__init__()
        token_dim = config.model.token_dim

        self.token_mixing = RankMixerTokenMixing(metadata.feature_token_count, token_dim)
        self.feature_norm = nn.LayerNorm(token_dim)
        self.feature_ffn = _build_rankmixer_ffn(config, metadata.feature_token_count)
        self.feature_ffn_norm = (
            nn.LayerNorm(token_dim)
            if config.model.mdl_feature_interaction == "residual_ffn"
            else None
        )
        _init_domain_interaction_modules(self, config, metadata)

    def forward(self, feature_tokens: Tensor, scenario_tokens: Tensor, task_tokens: Tensor, scenario_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mixed = self.feature_norm(self.token_mixing(feature_tokens) + feature_tokens)
        feature_update = self.feature_ffn(mixed)
        feature_tokens = (
            feature_update
            if self.feature_ffn_norm is None
            else self.feature_ffn_norm(feature_update + mixed)
        )

        scenario_tokens, task_tokens = _forward_domain_interaction(
            self,
            feature_tokens,
            scenario_tokens,
            task_tokens,
            scenario_mask,
        )
        return feature_tokens, scenario_tokens, task_tokens


class MDLRankMixerModel(nn.Module):
    def __init__(
        self,
        config: AppConfig,
        vocab_maps: dict[str, dict[str, int]],
        embedding_dim: int | None = None,
        embedding_size_override: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.feature_groups = config.tokenization.resolved_feature_tokens(config.features, config.sequences)
        self.feature_token_inputs = config.tokenization.resolved_feature_token_inputs(config.features, config.sequences)
        feature_token_count = config.tokenization.resolved_feature_token_count(config.features, config.sequences)
        resolved_scenario_specs = config.tokenization.resolved_scenario_tokens(
            config.features,
            config.scenarios.names,
            config.sequences,
        )
        self.scenario_token_specs = _active_scenario_token_specs(
            config,
            resolved_scenario_specs,
        )
        self.task_token_specs = (
            config.tokenization.resolved_task_tokens(
                config.features,
                config.task_names,
                config.sequences,
            )
            if config.model.use_task_tokens
            else []
        )
        self.metadata = ModelMetadata(
            feature_token_count=feature_token_count,
            scenario_count=len(config.scenarios.names),
            task_count=len(config.task_names),
        )
        if config.model.token_dim % self.metadata.feature_token_count != 0:
            raise ValueError("mdl_rankmixer requires token_dim divisible by feature token count")
        embedding_dim = config.model.embedding_dim if embedding_dim is None else embedding_dim
        self.encoder_bank = FeatureEncoderBank(
            config,
            vocab_maps,
            embedding_dim,
            included_scalar_feature_names=_mdl_scalar_feature_names(config),
            embedding_size_override=embedding_size_override,
        )
        self.feature_projector = _build_rankmixer_feature_projector(
            config,
            self.encoder_bank,
            self.feature_groups,
            self.feature_token_inputs,
            self.metadata.feature_token_count,
        )
        self.scenario_projector = (
            DomainTokenProjector(
                self.scenario_token_specs,
                self.encoder_bank.output_dims,
                config.model.token_dim,
                config.model.hidden_dim,
                activation=config.model.ffn_activation,
            )
            if config.model.use_scenario_tokens
            else None
        )
        self.task_projector = (
            DomainTokenProjector(
                self.task_token_specs,
                self.encoder_bank.output_dims,
                config.model.token_dim,
                config.model.hidden_dim,
                activation=config.model.ffn_activation,
            )
            if config.model.use_task_tokens
            else None
        )
        self.blocks = nn.ModuleList(
            MDLRankMixerBlock(config, self.metadata)
            for _layer_index in range(config.model.num_layers)
        )
        _init_mdl_output_modules(self, config, self.metadata)

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
        scenario_tokens = (
            self.scenario_projector(encoded)
            if self.scenario_projector is not None
            else _empty_domain_tokens(feature_tokens)
        )
        task_tokens = (
            self.task_projector(encoded)
            if self.task_projector is not None
            else _empty_domain_tokens(feature_tokens)
        )
        scenario_mask = _scenario_mask_from_ids(scenario_id, self.metadata.scenario_count)
        for block in self.blocks:
            if _activation_checkpoint_enabled(
                self.config.runtime.activation_checkpoint
            ) and self.training:
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
        logits = _mdl_logits(
            self,
            feature_tokens,
            scenario_tokens,
            task_tokens,
            scenario_mask,
        )
        output = {"logits": logits}
        output.update(_sparse_moe_outputs(self, logits))
        return output


class OneTransModel(nn.Module):
    def __init__(
        self,
        config: AppConfig,
        vocab_maps: dict[str, dict[str, int]],
        embedding_dim: int | None = None,
        embedding_size_override: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = OneTransBackbone(
            config,
            vocab_maps,
            embedding_dim,
            embedding_size_override=embedding_size_override,
        )
        output_dim = self.backbone.ns_token_count * config.model.token_dim
        self.logit_layers = _build_task_heads(config, output_dim, len(config.task_names))

    def precompute_request_cache(self, features: dict[str, Any]) -> OneTransRequestCache:
        return self.backbone.precompute_request_cache(features)

    def update_request_cache(
        self,
        features: dict[str, Any],
        previous: OneTransRequestCache,
    ) -> OneTransRequestCache:
        return self.backbone.update_request_cache(features, previous)

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
        use_sequence_attention: bool = False,
    ) -> None:
        super().__init__()
        _init_domain_interaction_modules(self, config, metadata)
        token_dim = config.model.token_dim
        self.use_sequence_attention = use_sequence_attention

        self.scenario_sequence_attention = (
            VariableLengthDomainAttention(
                token_dim,
                config.model.num_heads,
                attention_backend=config.runtime.attention_backend,
            )
            if use_sequence_attention and self.use_scenario_tokens
            else None
        )
        self.scenario_sequence_gate = (
            self._sequence_gate(token_dim)
            if self.scenario_sequence_attention is not None
            else None
        )
        self.task_sequence_attention = (
            VariableLengthDomainAttention(
                token_dim,
                config.model.num_heads,
                attention_backend=config.runtime.attention_backend,
            )
            if use_sequence_attention and self.use_task_tokens
            else None
        )
        self.task_sequence_gate = (
            self._sequence_gate(token_dim)
            if self.task_sequence_attention is not None
            else None
        )

    @staticmethod
    def _sequence_gate(token_dim: int) -> nn.Sequential:
        gate = nn.Sequential(
            nn.Linear(3 * token_dim, token_dim),
            nn.Sigmoid(),
        )
        nn.init.constant_(gate[0].bias, -2.0)
        return gate

    def forward(
        self,
        ns_tokens: Tensor,
        s_tokens: Tensor,
        s_mask: Tensor,
        scenario_tokens: Tensor,
        task_tokens: Tensor,
        scenario_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        scenario_hat: Tensor | None = None
        if self.use_scenario_tokens:
            scenario_hat = _domain_interaction_hat(
                scenario_tokens,
                ns_tokens,
                self.scenario_attention,
                self.scenario_rankmixer,
            )
            scenario_hat = _gated_sequence_interaction_hat(
                scenario_tokens,
                scenario_hat,
                s_tokens,
                s_mask,
                self.scenario_sequence_attention,
                self.scenario_sequence_gate,
            )
            scenario_tokens = scenario_hat + self.scenario_ffn(scenario_hat)
        elif scenario_tokens.size(1) != 0:
            raise ValueError("disabled scenario-token path expects an empty tensor")

        if self.use_task_tokens:
            task_hat = _domain_interaction_hat(
                task_tokens,
                ns_tokens,
                self.task_attention,
                self.task_rankmixer,
            )
            task_hat = _gated_sequence_interaction_hat(
                task_tokens,
                task_hat,
                s_tokens,
                s_mask,
                self.task_sequence_attention,
                self.task_sequence_gate,
            )
            if scenario_hat is not None:
                task_hat = self.domain_fused(task_hat, scenario_hat, scenario_mask)
            task_tokens = task_hat + self.task_ffn(task_hat)
        elif task_tokens.size(1) != 0:
            raise ValueError("disabled task-token path expects an empty tensor")
        return scenario_tokens, task_tokens


class MDLOneTransModel(nn.Module):
    def __init__(
        self,
        config: AppConfig,
        vocab_maps: dict[str, dict[str, int]],
        embedding_dim: int | None = None,
        embedding_size_override: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone = OneTransBackbone(
            config,
            vocab_maps,
            embedding_dim,
            included_scalar_feature_names=_mdl_scalar_feature_names(config),
            embedding_size_override=embedding_size_override,
        )
        self.metadata = ModelMetadata(
            feature_token_count=self.backbone.ns_token_count,
            scenario_count=len(config.scenarios.names),
            task_count=len(config.task_names),
        )
        output_dims = self.backbone.encoder_bank.output_dims
        scenario_token_specs = _active_scenario_token_specs(
            config,
            config.tokenization.resolved_scenario_tokens(
                config.features,
                config.scenarios.names,
                config.sequences,
            ),
        )
        task_token_specs = (
            config.tokenization.resolved_task_tokens(
                config.features,
                config.task_names,
                config.sequences,
            )
            if config.model.use_task_tokens
            else []
        )
        self.scenario_projector = (
            DomainTokenProjector(
                scenario_token_specs,
                output_dims,
                config.model.token_dim,
                config.model.hidden_dim,
                activation=config.model.ffn_activation,
            )
            if config.model.use_scenario_tokens
            else None
        )
        self.task_projector = (
            DomainTokenProjector(
                task_token_specs,
                output_dims,
                config.model.token_dim,
                config.model.hidden_dim,
                activation=config.model.ffn_activation,
            )
            if config.model.use_task_tokens
            else None
        )
        first_sequence_layer = config.model.first_domain_sequence_layer
        self.blocks = nn.ModuleList(
            MDLDomainBlock(
                config,
                self.metadata,
                use_sequence_attention=(
                    first_sequence_layer is not None
                    and layer_index >= first_sequence_layer
                ),
            )
            for layer_index in range(config.model.num_layers)
        )
        _init_mdl_output_modules(self, config, self.metadata)

    def precompute_request_cache(self, features: dict[str, Any]) -> OneTransRequestCache:
        return self.backbone.precompute_request_cache(features)

    def update_request_cache(
        self,
        features: dict[str, Any],
        previous: OneTransRequestCache,
    ) -> OneTransRequestCache:
        return self.backbone.update_request_cache(features, previous)

    def forward(
        self,
        features: dict[str, Any],
        scenario_id: Tensor,
        request_cache: OneTransRequestCache | None = None,
    ) -> dict[str, Tensor]:
        tokenizer = getattr(self.backbone, "tokenizer", None)
        preencode = getattr(tokenizer, "_preencode_inputs", None)
        if preencode is None:
            # Keep lightweight/custom backbone implementations compatible with
            # the pre-fusion public surface.
            preencoded_inputs: dict[str, Tensor] = {}
            encoded = self.backbone.encoder_bank(features)
        else:
            preencoded_inputs = preencode(
                features,
                self.backbone.encoder_bank.included_scalar_feature_names,
                include_sequences=request_cache is None,
            )
            encoded = self.backbone.encoder_bank.encode_scalar_features(
                features,
                preencoded_inputs=preencoded_inputs,
            )
        if request_cache is None and self.config.model.use_request_cache:
            request_cache = (
                self.backbone.precompute_request_cache(
                    features,
                    preencoded_inputs,
                )
                if preencoded_inputs
                else self.backbone.precompute_request_cache(features)
            )
        prepare_kwargs: dict[str, Any] = {
            "request_cache": request_cache,
            "encoded_features": encoded,
        }
        if preencoded_inputs:
            prepare_kwargs["preencoded_inputs"] = preencoded_inputs
        state = self.backbone.prepare(features, **prepare_kwargs)
        scenario_mask = _scenario_mask_from_ids(scenario_id, self.metadata.scenario_count)
        scenario_tokens: Tensor | None = (
            self.scenario_projector(encoded)
            if self.scenario_projector is not None
            else None
        )
        task_tokens: Tensor | None = (
            self.task_projector(encoded)
            if self.task_projector is not None
            else None
        )
        if request_cache is not None and request_cache.layers:
            if len(request_cache.layers) != len(self.blocks):
                raise ValueError("OneTrans request cache depth does not match MDL domain depth")
            layer_caches: tuple[OneTransLayerCache | None, ...] = request_cache.layers
        else:
            layer_caches = tuple(None for _ in self.blocks)
        for layer_index, (block, layer_cache) in enumerate(zip(self.blocks, layer_caches)):
            state = self.backbone.step(state, layer_index, layer_cache)
            s_tokens = state.tokens[:, : state.s_count, :]
            s_mask = state.valid_mask[:, : state.s_count]
            feature_tokens = state.tokens[:, state.s_count :, :]
            if scenario_tokens is None:
                scenario_tokens = _empty_domain_tokens(feature_tokens)
            if task_tokens is None:
                task_tokens = _empty_domain_tokens(feature_tokens)
            if _activation_checkpoint_enabled(
                self.config.runtime.activation_checkpoint
            ) and self.training:
                scenario_tokens, task_tokens = checkpoint(
                    block,
                    feature_tokens,
                    s_tokens,
                    s_mask,
                    scenario_tokens,
                    task_tokens,
                    scenario_mask,
                    use_reentrant=False,
                )
            else:
                scenario_tokens, task_tokens = block(
                    feature_tokens,
                    s_tokens,
                    s_mask,
                    scenario_tokens,
                    task_tokens,
                    scenario_mask,
                )
        logits = _mdl_logits(
            self,
            feature_tokens,
            scenario_tokens,
            task_tokens,
            scenario_mask,
        )
        return {"logits": logits}


def build_model(
    config: AppConfig,
    vocab_maps: dict[str, dict[str, int]],
    *,
    embedding_size_override: int | None = None,
) -> nn.Module:
    if config.model.name == "rankmixer":
        return RankMixerModel(
            config,
            vocab_maps,
            embedding_size_override=embedding_size_override,
        )
    if config.model.name == "mdl_rankmixer":
        return MDLRankMixerModel(
            config,
            vocab_maps,
            embedding_size_override=embedding_size_override,
        )
    if config.model.name == "onetrans":
        return OneTransModel(
            config,
            vocab_maps,
            embedding_size_override=embedding_size_override,
        )
    if config.model.name == "mdl_onetrans":
        return MDLOneTransModel(
            config,
            vocab_maps,
            embedding_size_override=embedding_size_override,
        )
    if config.model.name == "longer":
        return LongerModel(
            config,
            vocab_maps,
            embedding_size_override=embedding_size_override,
        )
    raise NotImplementedError(f"model {config.model.name!r} is not implemented")
