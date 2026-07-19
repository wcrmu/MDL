from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import inspect
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - compatibility with older PyTorch builds.
    SDPBackend = None
    sdpa_kernel = None

try:
    from torch.nn.attention.varlen import varlen_attn
except ImportError:  # pragma: no cover - older PyTorch compatibility.
    varlen_attn = None

_VARLEN_ATTN_USES_WINDOW_SIZE = (
    varlen_attn is not None
    and "window_size" in inspect.signature(varlen_attn).parameters
)

from .mlp import PerTokenFFN


def varlen_attention_available() -> bool:
    """True iff ``torch.nn.attention.varlen.varlen_attn`` can be imported.

    This only reports Python API availability (PyTorch >= 2.10), not that every
    GPU/dtype/shape combination will successfully execute the kernel.
    """

    return varlen_attn is not None


def _sdpa_context(attention_backend: str):
    if attention_backend in {"auto", "sdpa"}:
        return nullcontext()
    if attention_backend == "flash":
        if SDPBackend is None or sdpa_kernel is None:
            raise RuntimeError("runtime.attention_backend='flash' requires torch.nn.attention.sdpa_kernel")
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION])
    raise ValueError("attention_backend must be auto, sdpa, or flash")


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
    """Call the PyTorch 2.10 or 2.12 varlen Flash API."""

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
    """Gather by a permutation and invert it without atomic scatter-add."""

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
    """Fixed-capacity row-major packing metadata for one validity mask."""

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
            cumulative_lengths=F.pad(
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


class RankMixerTokenMixing(nn.Module):
    def __init__(self, num_tokens: int, token_dim: int) -> None:
        super().__init__()
        if token_dim % num_tokens != 0:
            raise ValueError("token_dim must be divisible by num_tokens for RankMixer TokenMixing")
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.head_dim = token_dim // num_tokens

    def forward(self, tokens: Tensor) -> Tensor:
        batch_size, num_tokens, token_dim = tokens.shape
        if num_tokens != self.num_tokens or token_dim != self.token_dim:
            raise ValueError(
                f"expected tokens with shape [batch, {self.num_tokens}, {self.token_dim}], "
                f"got {tuple(tokens.shape)}"
            )
        split_tokens = tokens.view(batch_size, num_tokens, self.num_tokens, self.head_dim)
        mixed = split_tokens.permute(0, 2, 1, 3).contiguous()
        return mixed.view(batch_size, num_tokens, token_dim)


class RankMixerDomainInteraction(nn.Module):
    """Route feature information into domain tokens with RankMixer mixing.

    MDL only states that its interaction ablation replaces domain-aware
    attention with the RankMixer interaction; it does not publish a separate
    cross-interaction equation. This module makes that replacement explicit:
    concatenate ``[feature; domain]`` tokens, apply RankMixer's parameter-free
    token mixing plus residual/LayerNorm, and keep the domain-token outputs.
    When the configured width is not divisible by the combined token count,
    the smallest valid RankMixer width is supplied with zero-padding and
    cropped back to ``token_dim`` afterwards.
    """

    def __init__(
        self,
        token_dim: int,
        num_domain_tokens: int,
        num_feature_tokens: int,
    ) -> None:
        super().__init__()
        if token_dim <= 0 or num_domain_tokens <= 0 or num_feature_tokens <= 0:
            raise ValueError("token and domain/feature counts must be positive")
        self.token_dim = token_dim
        self.num_domain_tokens = num_domain_tokens
        self.num_feature_tokens = num_feature_tokens
        self.num_tokens = num_feature_tokens + num_domain_tokens
        self.mixing_dim = math.ceil(token_dim / self.num_tokens) * self.num_tokens
        self.token_mixing = RankMixerTokenMixing(self.num_tokens, self.mixing_dim)
        self.norm = nn.LayerNorm(self.mixing_dim)

    def forward(self, domain_tokens: Tensor, feature_tokens: Tensor) -> Tensor:
        if domain_tokens.ndim != 3 or domain_tokens.shape[1:] != (
            self.num_domain_tokens,
            self.token_dim,
        ):
            raise ValueError(
                "expected domain tokens with shape "
                f"[batch, {self.num_domain_tokens}, {self.token_dim}], "
                f"got {tuple(domain_tokens.shape)}"
            )
        if feature_tokens.ndim != 3 or feature_tokens.shape[1:] != (
            self.num_feature_tokens,
            self.token_dim,
        ):
            raise ValueError(
                "expected feature tokens with shape "
                f"[batch, {self.num_feature_tokens}, {self.token_dim}], "
                f"got {tuple(feature_tokens.shape)}"
            )
        if domain_tokens.size(0) != feature_tokens.size(0):
            raise ValueError("domain and feature token batches must match")

        tokens = torch.cat([feature_tokens, domain_tokens], dim=1)
        if self.mixing_dim != self.token_dim:
            tokens = F.pad(tokens, (0, self.mixing_dim - self.token_dim))
        mixed = self.norm(self.token_mixing(tokens) + tokens)
        return mixed[:, self.num_feature_tokens :, : self.token_dim]


class DomainAwareAttention(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        num_domain_tokens: int,
        num_feature_tokens: int,
        hidden_dim: int,
        dropout: float = 0.0,
        attention_backend: str = "auto",
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.num_domain_tokens = num_domain_tokens
        self.num_feature_tokens = num_feature_tokens
        self.head_dim = token_dim // num_heads
        self.attention_backend = attention_backend
        self.query_projection = PerTokenFFN(
            num_domain_tokens,
            token_dim,
            hidden_dim,
            dropout=dropout,
            activation=activation,
        )
        self.key_projection = PerTokenFFN(
            num_feature_tokens,
            token_dim,
            hidden_dim,
            dropout=dropout,
            activation=activation,
        )
        self.value_projection = PerTokenFFN(
            num_feature_tokens,
            token_dim,
            hidden_dim,
            dropout=dropout,
            activation=activation,
        )
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, tokens: Tensor) -> Tensor:
        batch_size, num_tokens, _token_dim = tokens.shape
        return tokens.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, tokens: Tensor) -> Tensor:
        batch_size, _num_heads, num_tokens, _head_dim = tokens.shape
        return tokens.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.token_dim)

    def forward(
        self,
        domain_tokens: Tensor,
        feature_tokens: Tensor,
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        if domain_tokens.size(1) != self.num_domain_tokens:
            raise ValueError(
                f"expected {self.num_domain_tokens} domain tokens, got {domain_tokens.size(1)}"
            )
        if feature_tokens.size(1) != self.num_feature_tokens:
            raise ValueError(
                f"expected {self.num_feature_tokens} feature tokens, got {feature_tokens.size(1)}"
            )

        query = self._split_heads(self.query_projection(domain_tokens))
        key = self._split_heads(self.key_projection(feature_tokens))
        value = self._split_heads(self.value_projection(feature_tokens))

        if not need_weights:
            dropout_p = self.dropout.p if self.training else 0.0
            with _sdpa_context(self.attention_backend):
                attended = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=dropout_p,
                )
            return self._merge_heads(attended), None

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(self.dropout(weights), value)
        return self._merge_heads(attended), weights if need_weights else None


class VariableLengthDomainAttention(nn.Module):
    """Cross-attention from fixed domain tokens to masked sequence states.

    Unlike :class:`DomainAwareAttention`, the key/value projections are shared
    across sequence positions.  This keeps the module valid when OneTrans's
    pyramid changes the number of S tokens from one layer to the next.
    """

    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        attention_backend: str = "auto",
    ) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")

        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.attention_backend = attention_backend

        self.query_norm = nn.LayerNorm(token_dim)
        self.memory_norm = nn.LayerNorm(token_dim)
        self.query_projection = nn.Linear(token_dim, token_dim)
        self.key_projection = nn.Linear(token_dim, token_dim)
        self.value_projection = nn.Linear(token_dim, token_dim)
        self.output_projection = nn.Linear(token_dim, token_dim)

    def _split_heads(self, values: Tensor) -> Tensor:
        batch_size, token_count, _token_dim = values.shape
        return values.view(
            batch_size,
            token_count,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _flash_varlen_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_valid_mask: Tensor,
    ) -> Tensor:
        if varlen_attn is None:
            raise RuntimeError(
                "runtime.attention_backend='flash' requires torch.nn.attention.varlen"
            )
        if query.device.type != "cuda":
            raise RuntimeError("Flash varlen attention requires CUDA tensors")
        if query.dtype not in {torch.float16, torch.bfloat16}:
            raise RuntimeError("Flash varlen attention requires FP16 or BF16 tensors")

        query_tokens = query.transpose(1, 2).contiguous()
        key_tokens = key.transpose(1, 2)
        value_tokens = value.transpose(1, 2)
        if query_tokens.numel() == 0:
            return torch.zeros_like(query)

        key_packing = _VarlenPacking.from_mask(key_valid_mask)
        batch_size, query_length = query_tokens.shape[:2]
        cumulative_query_lengths = torch.arange(
            batch_size + 1,
            dtype=torch.int32,
            device=query.device,
        ) * query_length
        with torch.profiler.record_function("mdl::flash_varlen_domain_sequence"):
            packed_output = _call_varlen_attention(
                query_tokens.view(-1, self.num_heads, self.head_dim),
                key_packing.pack(key_tokens).contiguous(),
                key_packing.pack(value_tokens).contiguous(),
                cumulative_query_lengths,
                key_packing.cumulative_lengths,
                query_length,
                key_valid_mask.size(1),
                causal=False,
            )
        return packed_output.view_as(query_tokens).transpose(1, 2)

    def forward(
        self,
        domain_tokens: Tensor,
        sequence_tokens: Tensor,
        sequence_mask: Tensor,
    ) -> Tensor:
        if domain_tokens.ndim != 3 or domain_tokens.size(-1) != self.token_dim:
            raise ValueError(
                f"domain_tokens must have shape [batch, tokens, {self.token_dim}]"
            )
        if sequence_tokens.ndim != 3 or sequence_tokens.size(-1) != self.token_dim:
            raise ValueError(
                f"sequence_tokens must have shape [batch, length, {self.token_dim}]"
            )
        if sequence_mask.shape != sequence_tokens.shape[:2]:
            raise ValueError("sequence_mask must match the sequence token batch and length")
        if domain_tokens.size(0) != sequence_tokens.size(0):
            raise ValueError("domain and sequence token batches must match")
        if sequence_tokens.size(1) == 0:
            return torch.zeros_like(domain_tokens)

        sequence_mask = sequence_mask.to(device=sequence_tokens.device, dtype=torch.bool)
        has_valid_sequence = sequence_mask.any(dim=1)

        # SDPA backends need at least one allowed key per row.  Empty-history
        # rows temporarily expose one zeroed key and are explicitly zeroed again
        # after projection, so neither padding values nor projection biases leak.
        safe_mask = sequence_mask.clone()
        safe_mask[:, 0] |= ~has_valid_sequence
        memory = sequence_tokens.masked_fill(~sequence_mask.unsqueeze(-1), 0.0)

        query = self._split_heads(
            self.query_projection(self.query_norm(domain_tokens))
        )
        normalized_memory = self.memory_norm(memory)
        key = self._split_heads(self.key_projection(normalized_memory))
        value = self._split_heads(self.value_projection(normalized_memory))
        if self.attention_backend == "flash":
            attended = self._flash_varlen_attention(
                query,
                key,
                value,
                safe_mask,
            )
        else:
            allowed = safe_mask[:, None, None, :]
            with _sdpa_context(self.attention_backend):
                attended = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=allowed,
                    dropout_p=0.0,
                )

        attended = attended.transpose(1, 2).contiguous().view(
            domain_tokens.size(0),
            domain_tokens.size(1),
            self.token_dim,
        )
        output = self.output_projection(attended)
        return output * has_valid_sequence[:, None, None].to(output.dtype)


def masked_scenario_pool(
    scenario_states: Tensor,
    scenario_mask: Tensor,
    *,
    include_global: bool = True,
    has_global_state: bool = True,
) -> Tensor:
    """Select and mean-pool per-example scenario states.

    This operation is shared by the MDL DomainFused path and the explicit
    scenario-tower ablation. Keeping it separate prevents the ablation from
    registering or executing a DomainFusedModule when scenario tokens do not
    exist.
    """

    if scenario_states.ndim != 3:
        raise ValueError("scenario_tokens must have shape [batch, tokens, dim]")
    if scenario_mask.ndim != 2:
        raise ValueError("scenario_mask must have shape [batch, num_scenarios]")
    if scenario_mask.size(0) != scenario_states.size(0):
        raise ValueError("scenario_mask batch size must match scenario_tokens")
    expected_states = scenario_mask.size(1) + int(has_global_state)
    if scenario_states.size(1) != expected_states:
        suffix = " plus one global token" if has_global_state else ""
        raise ValueError(
            "scenario token count must match scenario_mask width" + suffix
        )

    mask = scenario_mask.to(
        dtype=scenario_states.dtype,
        device=scenario_states.device,
    )
    if has_global_state and include_global:
        global_mask = torch.ones(
            mask.size(0),
            1,
            dtype=mask.dtype,
            device=mask.device,
        )
        full_mask = torch.cat([mask, global_mask], dim=1)
        selected_states = scenario_states
    elif has_global_state:
        full_mask = mask
        selected_states = scenario_states[:, :-1, :]
    else:
        full_mask = mask
        selected_states = scenario_states
    denominator = full_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (selected_states * full_mask.unsqueeze(-1)).sum(dim=1) / denominator


class DomainFusedModule(nn.Module):
    def __init__(
        self,
        include_global: bool = True,
        has_global_token: bool = True,
    ) -> None:
        super().__init__()
        self.include_global = include_global
        self.has_global_token = has_global_token

    def pool(self, scenario_tokens: Tensor, scenario_mask: Tensor) -> Tensor:
        return masked_scenario_pool(
            scenario_tokens,
            scenario_mask,
            include_global=self.include_global,
            has_global_state=self.has_global_token,
        )

    def forward(self, task_tokens: Tensor, scenario_tokens: Tensor, scenario_mask: Tensor) -> Tensor:
        return task_tokens + self.pool(scenario_tokens, scenario_mask).unsqueeze(1)
