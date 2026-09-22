"""Paper-aligned MixFormer building blocks.

The implementation follows ``paper/mixformer/main.tex``:

* non-sequential heads exchange information through parameter-free
  ``HeadMixing``;
* Query Mixer and Output Fusion use independent, per-head SwiGLU FFNs with
  pre-RMSNorm residuals;
* every layer owns an independent sequence SwiGLU projection;
* each mixed feature head is one full-dimensional cross-attention query;
* request-level batching shares the long sequence tensor across candidates.

The paper's commented efficient single-query reordering is used for the
cross-attention computation.  It is algebraically identical to materializing
projected keys and values, but it avoids length-sized K/V projections.
Eager attention is dispatched through ``scaled_dot_product_attention`` with
``K = V = H`` after the ``q W_k`` reorder, which is the same softmax.
A dense bool ``attn_mask`` disables PyTorch FLASH even when every slot is
valid, so a full MixFormer window (no padding) runs unmasked SDPA FLASH.
Mixed-length leftover False is only a padding mask: when Dao
``flash_attn_varlen_func`` is importable the valid keys are packed and that
kernel replaces masked SDPA. The varlen call is ``torch.compiler.disable``'d
so inductor graph-breaks instead of baking unmasked FLASH for mixed L.
Grouped item attention also packs dummy ``max_targets`` query slots.
Arbitrary attention patterns are not expressed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import (
    SDPBackend,
    _call_varlen_attention,
    ensure_contiguous,
    packing_for_masks,
    sdpa_kernel,
    validate_varlen_inputs,
    varlen_attention_available,
)


def _gather_request_rows(values: Tensor, row_indices: Tensor | None) -> Tensor:
    if row_indices is None:
        return values
    return values.index_select(0, row_indices.to(device=values.device, dtype=torch.long))


def assemble_mixformer_heads(
    user_heads: Tensor,
    item_heads: Tensor,
    row_indices: Tensor | None,
) -> Tensor:
    """Expand request-major user heads onto the candidate batch and concat."""

    gathered_user = _gather_request_rows(user_heads, row_indices)
    if gathered_user.size(0) != item_heads.size(0):
        raise ValueError(
            "assembled MixFormer heads require user and item batches to match, "
            f"got {gathered_user.size(0)} vs {item_heads.size(0)}"
        )
    return torch.cat([gathered_user, item_heads], dim=1)


def _bmm_per_head(values: Tensor, weight: Tensor) -> Tensor:
    """Per-head GEMM: ``values[B, N, K] @ weight[N, K, M] -> [B, N, M]``."""

    return torch.bmm(values.transpose(0, 1), weight).transpose(0, 1)


class MixFormerRMSNorm(nn.Module):
    """RMSNorm that keeps the residual stream dtype unchanged."""

    def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, values: Tensor) -> Tensor:
        if values.size(-1) != self.dim:
            raise ValueError(
                f"RMSNorm expected trailing dimension {self.dim}, "
                f"got {values.size(-1)}"
            )
        weight = self.weight
        if weight.dtype != values.dtype:
            weight = weight.to(dtype=values.dtype)
        # MixFormer dense compile maps off CUDAGraphs (grad-accum overwrites
        # residuals). Keep the live parameter so inductor can fuse RMSNorm
        # into the sequence GEMM instead of cloning a D-vector every call.
        if hasattr(F, "rms_norm"):
            return F.rms_norm(values, (self.dim,), weight, self.eps)
        scale = (
            values.float()
            .pow(2)
            .mean(dim=-1, keepdim=True)
            .add(self.eps)
            .rsqrt()
            .to(dtype=values.dtype)
        )
        return values * scale * weight


class DenseSwiGLUFFN(nn.Module):
    """Dimension-preserving SwiGLU with an explicit intermediate width.

    Up and gate share one ``Linear(dim, 2H)`` so the sequence residual is a
    single GEMM instead of concatenating two weight rows on every forward.
    The MixFormer equations are unchanged: bias-free, ``up * silu(gate)``.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        if dim <= 0 or hidden_dim <= 0:
            raise ValueError("dim and hidden_dim must be positive")
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.up_gate_projection = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, dim, bias=False)
        # Match two-matrix kaiming so packed vs split parameterization agrees.
        with torch.no_grad():
            nn.init.kaiming_uniform_(
                self.up_gate_projection.weight[:hidden_dim],
                a=math.sqrt(5),
            )
            nn.init.kaiming_uniform_(
                self.up_gate_projection.weight[hidden_dim:],
                a=math.sqrt(5),
            )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        up_key = f"{prefix}up_projection.weight"
        gate_key = f"{prefix}gate_projection.weight"
        packed_key = f"{prefix}up_gate_projection.weight"
        if packed_key not in state_dict and up_key in state_dict and gate_key in state_dict:
            state_dict[packed_key] = torch.cat(
                [state_dict.pop(up_key), state_dict.pop(gate_key)],
                dim=0,
            )
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, values: Tensor) -> Tensor:
        if values.size(-1) != self.dim:
            raise ValueError(
                f"SwiGLU expected trailing dimension {self.dim}, "
                f"got {values.size(-1)}"
            )
        up, gate = self.up_gate_projection(values).chunk(2, dim=-1)
        return self.output_projection(up * F.silu(gate))


class StackedPerHeadSwiGLUFFN(nn.Module):
    """Independent per-head SwiGLU FFNs executed with batched GEMMs."""

    def __init__(self, num_heads: int, dim: int, hidden_dim: int) -> None:
        super().__init__()
        if num_heads <= 0 or dim <= 0 or hidden_dim <= 0:
            raise ValueError("head count and dimensions must be positive")
        self.num_heads = num_heads
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.up_weight = nn.Parameter(torch.empty(num_heads, hidden_dim, dim))
        self.gate_weight = nn.Parameter(torch.empty(num_heads, hidden_dim, dim))
        self.output_weight = nn.Parameter(torch.empty(num_heads, dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for head in range(self.num_heads):
            nn.init.kaiming_uniform_(self.up_weight[head], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.gate_weight[head], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.output_weight[head], a=math.sqrt(5))

    def forward(self, values: Tensor) -> Tensor:
        return self.forward_slice(values, slice(None))

    def forward_slice(self, values: Tensor, head_slice: slice) -> Tensor:
        up_weight = self.up_weight[head_slice]
        gate_weight = self.gate_weight[head_slice]
        output_weight = self.output_weight[head_slice]
        expected = (int(up_weight.size(0)), self.dim)
        if values.ndim != 3 or tuple(values.shape[1:]) != expected:
            raise ValueError(
                f"expected values with shape [batch, {expected[0]}, "
                f"{self.dim}], got {tuple(values.shape)}"
            )
        head_major = values.transpose(0, 1)
        packed_weight = torch.cat((up_weight, gate_weight), dim=1)
        packed = torch.bmm(head_major, packed_weight.transpose(1, 2))
        up, gate = packed.chunk(2, dim=-1)
        hidden = up * F.silu(gate)
        output = torch.bmm(hidden, output_weight.transpose(1, 2))
        return output.transpose(0, 1)


class MixFormerHeadMixing(nn.Module):
    """Parameter-free HeadMixing from the MixFormer Query Mixer."""

    def __init__(
        self,
        num_heads: int,
        dim: int,
        *,
        user_head_count: int | None = None,
    ) -> None:
        super().__init__()
        if num_heads <= 0 or dim <= 0:
            raise ValueError("num_heads and dim must be positive")
        if dim % num_heads != 0:
            raise ValueError("MixFormer head dimension must be divisible by head count")
        if user_head_count is not None and not 0 < user_head_count < num_heads:
            raise ValueError("user_head_count must be inside (0, num_heads)")
        self.num_heads = num_heads
        self.dim = dim
        self.split_dim = dim // num_heads
        self.user_head_count = user_head_count

        if user_head_count is None:
            decoupling_mask = None
        else:
            # UI-MixFormer Eq. (mask): user outputs cannot contain item-side
            # chunks, while item outputs may consume both user and item chunks.
            decoupling_mask = torch.ones(num_heads, dim)
            decoupling_mask[
                :user_head_count,
                user_head_count * self.split_dim :,
            ] = 0.0
        self.register_buffer(
            "decoupling_mask",
            decoupling_mask,
            persistent=False,
        )

    def forward(self, heads: Tensor) -> Tensor:
        expected = (self.num_heads, self.dim)
        if heads.ndim != 3 or tuple(heads.shape[1:]) != expected:
            raise ValueError(
                f"expected heads with shape [batch, {self.num_heads}, "
                f"{self.dim}], got {tuple(heads.shape)}"
            )
        batch_size = heads.size(0)
        split = heads.reshape(
            batch_size,
            self.num_heads,
            self.num_heads,
            self.split_dim,
        )
        mixed = split.transpose(1, 2).contiguous().reshape_as(heads)
        decoupling_mask = self.decoupling_mask
        if isinstance(decoupling_mask, Tensor):
            mixed = mixed * decoupling_mask.to(dtype=mixed.dtype)
        return mixed

    def mix_user_heads(self, user_heads: Tensor) -> Tensor:
        """HeadMixing on user heads only; item chunks are zero after the mask.

        After the UI mask, user outputs ignore item-side slices, so padding
        item heads with zeros is algebraically the same as mixing the full
        request-expanded tensor.
        """

        user_head_count = self.user_head_count
        if user_head_count is None:
            raise RuntimeError("mix_user_heads requires user_head_count")
        expected = (user_head_count, self.dim)
        if user_heads.ndim != 3 or tuple(user_heads.shape[1:]) != expected:
            raise ValueError(
                f"expected user heads with shape [batch, {user_head_count}, "
                f"{self.dim}], got {tuple(user_heads.shape)}"
            )
        batch_size = user_heads.size(0)
        split = user_heads.reshape(
            batch_size,
            user_head_count,
            self.num_heads,
            self.split_dim,
        )
        mixed_user = (
            split[:, :, :user_head_count]
            .transpose(1, 2)
            .contiguous()
            .reshape(batch_size, user_head_count, user_head_count * self.split_dim)
        )
        zeros = user_heads.new_zeros(
            batch_size,
            user_head_count,
            (self.num_heads - user_head_count) * self.split_dim,
        )
        return torch.cat([mixed_user, zeros], dim=-1)


class MixFormerQueryMixer(nn.Module):
    """HeadMixing followed by a head-specific SwiGLU residual."""

    def __init__(
        self,
        num_heads: int,
        dim: int,
        hidden_dim: int,
        *,
        user_head_count: int | None = None,
    ) -> None:
        super().__init__()
        self.input_norm = MixFormerRMSNorm(dim)
        self.head_mixing = MixFormerHeadMixing(
            num_heads,
            dim,
            user_head_count=user_head_count,
        )
        self.ffn_norm = MixFormerRMSNorm(dim)
        self.ffn = StackedPerHeadSwiGLUFFN(num_heads, dim, hidden_dim)

    @property
    def user_head_count(self) -> int | None:
        return self.head_mixing.user_head_count

    def forward(self, heads: Tensor) -> Tensor:
        mixed = heads + self.head_mixing(self.input_norm(heads))
        return mixed + self.ffn(self.ffn_norm(mixed))

    def forward_user(self, user_heads: Tensor) -> Tensor:
        mixed = user_heads + self.head_mixing.mix_user_heads(
            self.input_norm(user_heads)
        )
        user_head_count = self.user_head_count
        if user_head_count is None:
            raise RuntimeError("forward_user requires user_head_count")
        return mixed + self.ffn.forward_slice(
            self.ffn_norm(mixed),
            slice(0, user_head_count),
        )

    def forward_decoupled(
        self,
        user_heads: Tensor,
        item_heads: Tensor,
        row_indices: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        user_head_count = self.user_head_count
        if user_head_count is None:
            raise RuntimeError("forward_decoupled requires user_head_count")
        user_query = self.forward_user(user_heads)
        user_for_item = _gather_request_rows(user_heads, row_indices)
        if user_for_item.size(0) != item_heads.size(0):
            raise ValueError(
                "UI-MixFormer item Query Mixer requires gathered user heads to "
                f"match the candidate batch, got {user_for_item.size(0)} vs "
                f"{item_heads.size(0)}"
            )
        # HeadMixing still sees gathered user chunks (item heads may consume
        # them). Per-head SwiGLU is independent, so user FFN stays request-major
        # and is not repeated on the candidate axis.
        heads = torch.cat([user_for_item, item_heads], dim=1)
        mixed = heads + self.head_mixing(self.input_norm(heads))
        item_mixed = mixed[:, user_head_count:]
        item_query = item_mixed + self.ffn.forward_slice(
            self.ffn_norm(item_mixed),
            slice(user_head_count, self.head_mixing.num_heads),
        )
        return user_query, item_query


class MixFormerOutputFusion(nn.Module):
    """Per-head SwiGLU output fusion from the MixFormer block."""

    def __init__(self, num_heads: int, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = MixFormerRMSNorm(dim)
        self.ffn = StackedPerHeadSwiGLUFFN(num_heads, dim, hidden_dim)

    def forward(self, heads: Tensor) -> Tensor:
        return self.forward_heads(heads, 0)

    def forward_heads(self, heads: Tensor, head_start: int) -> Tensor:
        update = self.ffn.forward_slice(
            self.norm(heads),
            slice(head_start, head_start + heads.size(1)),
        )
        return heads + update


@dataclass(frozen=True)
class MixFormerRequestLayout:
    """Small request/candidate packing shared by RLB cross attention."""

    row_indices: Tensor
    order: Tensor
    linear_slots: Tensor
    request_count: int
    max_targets: int


class MixFormerCrossAttention(nn.Module):
    """N full-dimensional single-query heads over one behavior sequence.

    If ``q`` and one sequence head are both ``D`` dimensional, the paper writes

    ``softmax(q (H W_k)^T / sqrt(D)) H W_v``.

    We compute the equivalent reordered form

    ``softmax((q W_k) H^T / sqrt(D)) H W_v``

    so projected K/V tensors are never materialized along sequence length.
    """

    def __init__(
        self,
        num_heads: int,
        dim: int,
        hidden_dim: int,
        *,
        sequence_chunk_tokens: int = 0,
        user_head_count: int | None = None,
    ) -> None:
        super().__init__()
        if num_heads <= 0 or dim <= 0 or hidden_dim <= 0:
            raise ValueError("head count and dimensions must be positive")
        if sequence_chunk_tokens < 0:
            raise ValueError("sequence_chunk_tokens must be non-negative")
        if user_head_count is not None and not 0 < user_head_count < num_heads:
            raise ValueError("user_head_count must be inside (0, num_heads)")
        self.num_heads = num_heads
        self.dim = dim
        self.user_head_count = user_head_count
        self.sequence_dim = num_heads * dim
        self.scale = dim**-0.5
        self.sequence_chunk_tokens = sequence_chunk_tokens
        self.sequence_norm = MixFormerRMSNorm(self.sequence_dim)
        self.sequence_ffn = DenseSwiGLUFFN(self.sequence_dim, hidden_dim)
        # One independent D x D key/value matrix for every query head.
        self.key_weight = nn.Parameter(torch.empty(num_heads, dim, dim))
        self.value_weight = nn.Parameter(torch.empty(num_heads, dim, dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for head in range(self.num_heads):
            nn.init.xavier_uniform_(self.key_weight[head])
            nn.init.xavier_uniform_(self.value_weight[head])

    @staticmethod
    def request_layout(
        row_indices: Tensor,
        request_count: int,
    ) -> MixFormerRequestLayout:
        if row_indices.ndim != 1:
            raise ValueError("sequence_row_indices must be rank one")
        candidate_count = int(row_indices.numel())
        if candidate_count == 0:
            empty = row_indices.new_empty(0)
            return MixFormerRequestLayout(
                row_indices=row_indices,
                order=empty,
                linear_slots=empty,
                request_count=request_count,
                max_targets=0,
            )
        order = torch.argsort(row_indices, stable=True)
        sorted_rows = row_indices.index_select(0, order)
        counts = torch.bincount(sorted_rows, minlength=request_count)
        max_targets = int(counts.max().item())
        request_offsets = counts.cumsum(dim=0) - counts
        within_request = torch.arange(
            candidate_count,
            device=row_indices.device,
        ) - torch.repeat_interleave(request_offsets, counts)
        linear_slots = sorted_rows * max_targets + within_request
        return MixFormerRequestLayout(
            row_indices=row_indices,
            order=order,
            linear_slots=linear_slots,
            request_count=request_count,
            max_targets=max_targets,
        )

    @staticmethod
    def _masked_softmax(scores: Tensor, valid_mask: Tensor) -> Tensor:
        if scores.size(-1) == 0:
            return scores
        if valid_mask.ndim != 2 or (
            valid_mask.size(0) != scores.size(0)
            or valid_mask.size(1) != scores.size(-1)
        ):
            raise ValueError(
                "valid_mask must match the request and sequence score dimensions"
            )
        mask = valid_mask.reshape(
            valid_mask.size(0),
            *([1] * (scores.ndim - 2)),
            valid_mask.size(1),
        )
        softmax_dtype = (
            torch.float32
            if scores.dtype in {torch.float16, torch.bfloat16}
            else scores.dtype
        )
        masked_scores = scores.masked_fill(
            ~mask,
            torch.finfo(scores.dtype).min,
        )
        weights = F.softmax(masked_scores, dim=-1, dtype=softmax_dtype).to(
            dtype=scores.dtype
        )
        weights = weights * mask.to(dtype=weights.dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny
        )

    @staticmethod
    def _sdpa_key_mask(valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        """SDPA bool mask (True = may attend) plus a per-row validity flag.

        All-False rows are undefined in fused attention, so those rows
        temporarily attend position 0 and are zeroed by the caller. That
        matches ``_masked_softmax`` renormalizing a fully masked row to 0.
        """

        valid_any = valid_mask.any(dim=-1)
        if valid_mask.size(-1) == 0:
            return valid_mask, valid_any
        dummy = valid_mask.new_zeros(valid_mask.shape)
        dummy[:, 0] = True
        return valid_mask | (dummy & ~valid_any.unsqueeze(-1)), valid_any

    @staticmethod
    def _history_is_dense(valid_mask: Tensor) -> bool:
        """True when every history slot may attend, so Flash needs no mask.

        MixFormer history is selected real actions, capped at 8000. After the
        tokenizer crops to the longest real row there is no pad-to-8000.
        A leftover bool mask that is all True still disables FLASH, so the
        dense window runs unmasked. Eager keeps a masked fallback only for
        mixed-length rows that still share one batch tensor.
        """

        if valid_mask.numel() == 0:
            return True
        if torch.compiler.is_compiling():
            return True
        cached = getattr(valid_mask, "_mixformer_dense", None)
        if isinstance(cached, bool):
            return cached
        return bool(valid_mask.all())

    def _can_varlen_history(self, query: Tensor) -> bool:
        return (
            varlen_attention_available()
            and query.is_cuda
            and query.dtype in {torch.float16, torch.bfloat16}
        )

    @torch.compiler.disable
    def _attend_history_varlen(
        self,
        query: Tensor,
        history: Tensor,
        valid_mask: Tensor,
        query_mask: Tensor | None = None,
    ) -> Tensor:
        """Pack valid MixFormer keys and run Dao varlen Flash.

        ``query`` is ``[B, Q, H, D]``, ``history`` is ``[B, L, H, D]``. Leftover
        False on the history axis is padding only. ``query_mask`` of ``None``
        means every query slot is live (flatten, no gather); a bool ``[B, Q]``
        packs occupancy so grouped dummy candidates never enter the kernel.
        Disabled under ``torch.compile`` so inductor does not bake unmasked
        FLASH into the dense MixFormer graph.
        """

        validate_varlen_inputs(
            query,
            history,
            history,
            dropout_p=0.0,
            training=self.training,
        )
        key_packing, _ = packing_for_masks(valid_mask, valid_mask)
        packed_key = ensure_contiguous(key_packing.pack(history, compact=True))
        query_packing = None
        if query_mask is None:
            packed_query = ensure_contiguous(
                query.reshape(-1, query.size(2), query.size(3))
            )
            batch = int(query.size(0))
            query_len = int(query.size(1))
            cu_query = (
                torch.arange(
                    batch + 1,
                    device=query.device,
                    dtype=torch.int32,
                )
                * query_len
            )
            max_query_length = query_len
        else:
            query_packing, _ = packing_for_masks(query_mask, query_mask)
            packed_query = ensure_contiguous(
                query_packing.pack(query, compact=True)
            )
            cu_query = query_packing.cumulative_lengths
            max_query_length = int(query.size(1))
        if packed_query.numel() == 0 or packed_key.numel() == 0:
            return query.new_zeros(query.shape)
        packed_output = _call_varlen_attention(
            packed_query,
            packed_key,
            packed_key,
            cu_query,
            key_packing.cumulative_lengths,
            max_query_length,
            int(valid_mask.size(1)),
            causal=False,
            fixed_capacity=False,
            softmax_scale=self.scale,
        )
        if query_packing is None:
            context = packed_output.view_as(query)
        else:
            context = query_packing.unpack(packed_output, query, compact=True)
        valid_any = valid_mask.any(dim=-1)
        return context * valid_any.to(dtype=context.dtype).reshape(
            valid_any.size(0),
            *([1] * (context.ndim - 1)),
        )

    def _attend_history(
        self,
        query: Tensor,
        history: Tensor,
        valid_mask: Tensor,
        query_mask: Tensor | None = None,
    ) -> Tensor:
        """``softmax((q W_k) H^T / sqrt(D)) H`` with fused SDPA, ``K = V = H``.

        ``query`` is ``[B, H, D]`` or candidate-packed ``[B, C, H, D]``. History
        stays event-major ``[B, L, H, D]``; the permute is a view so production
        ``L=8000`` histories are not copied into a contiguous BHLD layout.
        """

        squeeze = query.ndim == 3
        if squeeze:
            query = query.unsqueeze(1)
        if self._can_varlen_history(query) and not torch.compiler.is_compiling():
            # Compiled MixFormer captures unmasked SDPA FLASH (tokenizer
            # compact windows). Forcing Dao varlen into that graph graph-breaks
            # Query Mixer / sequence SwiGLU around packing `.item()` and lost
            # ~45% forward at 11% leftover pad. Eager mixed-length still packs.
            dense_keys = self._history_is_dense(valid_mask)
            if not dense_keys or query_mask is not None:
                context = self._attend_history_varlen(
                    query,
                    history,
                    valid_mask.bool(),
                    query_mask=query_mask,
                )
                if squeeze:
                    context = context.squeeze(1)
                return context
        query_heads = query.permute(0, 2, 1, 3)
        history_heads = history.permute(0, 2, 1, 3)
        dense = self._history_is_dense(valid_mask)
        attn_mask = None
        valid_any = None
        if not dense:
            safe_mask, valid_any = self._sdpa_key_mask(valid_mask)
            attn_mask = safe_mask[:, None, None, :]
        can_flash = (
            attn_mask is None
            and query_heads.is_cuda
            and query_heads.dtype in {torch.float16, torch.bfloat16}
            and SDPBackend is not None
            and sdpa_kernel is not None
        )
        if can_flash:
            with sdpa_kernel(
                [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ]
            ):
                context = F.scaled_dot_product_attention(
                    query_heads,
                    history_heads,
                    history_heads,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=self.scale,
                )
        else:
            context = F.scaled_dot_product_attention(
                query_heads,
                history_heads,
                history_heads,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scale,
            )
        context = context.permute(0, 2, 1, 3)
        if valid_any is not None:
            context = context * valid_any.to(dtype=context.dtype).reshape(
                valid_any.size(0),
                *([1] * (context.ndim - 1)),
            )
        if squeeze:
            context = context.squeeze(1)
        return context

    def _transform_sequence(self, sequence: Tensor) -> Tensor:
        return self._transform_sequence_slice(sequence, 0, sequence.size(1))

    def _transform_sequence_slice(
        self,
        sequence: Tensor,
        start: int,
        end: int,
    ) -> Tensor:
        """SwiGLU residual on ``sequence[:, start:end]`` → ``[R, L', H, D]``.

        Called from length-chunked attention so the full ``[R, L, N·D]`` history
        activation never coexists with score workspaces.
        """

        chunk = sequence[:, start:end]
        if chunk.size(1) == 0:
            return chunk.new_empty(
                chunk.size(0),
                0,
                self.num_heads,
                self.dim,
            )
        normalized = self.sequence_norm(chunk)
        flat = normalized.reshape(-1, self.sequence_dim)
        if (
            self.sequence_chunk_tokens > 0
            and flat.size(0) > self.sequence_chunk_tokens
        ):
            update = torch.cat(
                [
                    self.sequence_ffn(part)
                    for part in flat.split(self.sequence_chunk_tokens, dim=0)
                ],
                dim=0,
            )
        else:
            update = self.sequence_ffn(flat)
        return (chunk + update.view_as(chunk)).view(
            chunk.size(0),
            chunk.size(1),
            self.num_heads,
            self.dim,
        )

    def _validate(
        self,
        query: Tensor,
        sequence: Tensor,
        valid_mask: Tensor,
        row_indices: Tensor | None,
    ) -> Tensor | None:
        if query.ndim != 3 or tuple(query.shape[1:]) != (
            self.num_heads,
            self.dim,
        ):
            raise ValueError(
                f"query must have shape [batch, {self.num_heads}, {self.dim}]"
            )
        if sequence.ndim != 3 or sequence.size(-1) != self.sequence_dim:
            raise ValueError(
                f"sequence must have shape [request, length, {self.sequence_dim}]"
            )
        if valid_mask.ndim != 2 or valid_mask.shape != sequence.shape[:2]:
            raise ValueError("valid_mask must match sequence batch and length")
        if query.device != sequence.device or valid_mask.device != sequence.device:
            raise ValueError("query, sequence, and mask must be on one device")
        if row_indices is not None:
            if row_indices.ndim != 1 or row_indices.numel() != query.size(0):
                raise ValueError(
                    "sequence_row_indices must contain one request index per query"
                )
            row_indices = row_indices.to(device=sequence.device, dtype=torch.long)
            if row_indices.numel() and sequence.size(0) == 0:
                raise ValueError("sequence request batch cannot be empty")
            if row_indices.numel():
                invalid = (row_indices < 0) | (row_indices >= sequence.size(0))
                if invalid.device.type == "cuda" and hasattr(torch, "_assert_async"):
                    torch._assert_async(
                        ~invalid.any(),
                        "sequence_row_indices contains an out-of-range index",
                    )
                elif bool(invalid.any().item()):
                    raise ValueError(
                        "sequence_row_indices contains an out-of-range index"
                    )
        elif sequence.size(0) not in {1, query.size(0)}:
            raise ValueError(
                "sequence batch must equal query batch (or be one); provide "
                "sequence_row_indices for request-level batching"
            )
        return row_indices

    def _project_query(self, query: Tensor, head_start: int = 0) -> Tensor:
        # Paper Eq. (8) scores raw q_i against k = W_k h. Linear convention
        # k = h W_k^T makes q W_k the vector scored against unprojected H.
        head_count = query.size(1)
        weight = self.key_weight[head_start : head_start + head_count]
        return _bmm_per_head(query, weight)

    def _project_context(self, context: Tensor, head_start: int = 0) -> Tensor:
        head_count = context.size(1)
        weight = self.value_weight[head_start : head_start + head_count]
        return _bmm_per_head(context, weight.transpose(1, 2))

    def _attention_length_chunk(self, length: int) -> int:
        """Bound score storage when sequence FFN chunking is enabled.

        ``sequence_chunk_tokens`` is primarily a flattened FFN budget. Tiny
        values (<256) are used by tests to force length-chunked online softmax.
        Production budgets already split the FFN by flattened ``R·L'`` rows, so
        also slicing the length axis would only add launch overhead.
        """

        if self.sequence_chunk_tokens <= 0 or length <= 0:
            return length
        if self.sequence_chunk_tokens >= 256:
            return length
        budget = max(1, self.sequence_chunk_tokens // max(self.num_heads, 1))
        return min(length, budget)

    def _grouped_request_chunk(
        self,
        *,
        request_count: int,
        max_targets: int,
        sequence_length: int,
    ) -> int:
        """How many requests to attend in one grouped kernel.

        The previous heuristic compared the FFN token budget against ``L·C``.
        Production MixFormer (``L=8000``, ``C=16``, budget ``8192``) always
        collapsed to one request per kernel and wrecked GPU occupancy.
        Request splitting is kept only for tiny test budgets; the FFN already
        bounds HBM by flattening ``R·L'`` independently.
        """

        if request_count <= 0:
            return 1
        if (
            self.sequence_chunk_tokens <= 0
            or sequence_length <= 0
            or self.sequence_chunk_tokens >= 256
        ):
            return request_count
        tokens_per_request = sequence_length * max(max_targets, 1)
        return max(
            1,
            min(
                request_count,
                max(1, self.sequence_chunk_tokens // max(tokens_per_request, 1)),
            ),
        )

    def _context_from_aligned_scores(
        self,
        score_query: Tensor,
        history: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        return self._attend_history(score_query, history, valid_mask)

    def _context_from_grouped_scores(
        self,
        grouped_query: Tensor,
        history: Tensor,
        valid_mask: Tensor,
        query_mask: Tensor | None = None,
    ) -> Tensor:
        return self._attend_history(
            grouped_query,
            history,
            valid_mask,
            query_mask=query_mask,
        )

    @staticmethod
    def _occupancy_query_mask(
        layout: MixFormerRequestLayout,
        reference: Tensor,
    ) -> Tensor | None:
        """Bool ``[R, C]`` of live candidate slots, or ``None`` when full.

        ``None`` lets varlen flatten queries without a gather. A sparse
        rectangle packs only occupied slots so dummy ``max_targets`` pads
        do not enter Flash.
        """

        if layout.request_count <= 0 or layout.max_targets <= 0:
            return None
        capacity = layout.request_count * layout.max_targets
        occupied = int(layout.linear_slots.numel())
        if occupied == capacity:
            return None
        mask = torch.zeros(
            layout.request_count,
            layout.max_targets,
            dtype=torch.bool,
            device=reference.device,
        )
        if occupied:
            mask.view(-1).index_fill_(
                0,
                layout.linear_slots.to(device=reference.device, dtype=torch.long),
                True,
            )
        return mask

    def _context_aligned_chunked(
        self,
        score_query: Tensor,
        sequence: Tensor,
        valid_mask: Tensor,
        length_chunk: int,
    ) -> Tensor:
        """Online-softmax attention; FFN only the active length slice each step."""

        batch, num_heads, dim = score_query.shape
        length = sequence.size(1)
        neg = torch.finfo(torch.float32).min
        running_max = score_query.new_full(
            (batch, num_heads),
            neg,
            dtype=torch.float32,
        )
        running_sum = score_query.new_zeros(
            (batch, num_heads),
            dtype=torch.float32,
        )
        context = score_query.new_zeros(
            (batch, num_heads, dim),
            dtype=torch.float32,
        )
        for start in range(0, length, length_chunk):
            end = min(length, start + length_chunk)
            hist = self._transform_sequence_slice(sequence, start, end)
            mask = valid_mask[:, start:end]
            scores = (
                torch.einsum("bnd,blnd->bnl", score_query, hist) * self.scale
            ).float()
            scores = scores.masked_fill(~mask.unsqueeze(1), neg)
            block_max = scores.amax(dim=-1)
            block_max = torch.where(
                torch.isfinite(block_max),
                block_max,
                running_max,
            )
            new_max = torch.maximum(running_max, block_max)
            prior_scale = torch.exp(running_max - new_max)
            weights = torch.exp(scores - new_max.unsqueeze(-1))
            weights = weights * mask.unsqueeze(1).to(dtype=weights.dtype)
            context = context * prior_scale.unsqueeze(-1) + torch.einsum(
                "bnl,blnd->bnd",
                weights,
                hist.float(),
            )
            running_sum = running_sum * prior_scale + weights.sum(dim=-1)
            running_max = new_max
        context = context / running_sum.unsqueeze(-1).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        return context.to(dtype=score_query.dtype)

    def _context_grouped_chunked(
        self,
        grouped_query: Tensor,
        sequence: Tensor,
        valid_mask: Tensor,
        length_chunk: int,
    ) -> Tensor:
        request_count, max_targets, num_heads, dim = grouped_query.shape
        length = sequence.size(1)
        neg = torch.finfo(torch.float32).min
        running_max = grouped_query.new_full(
            (request_count, max_targets, num_heads),
            neg,
            dtype=torch.float32,
        )
        running_sum = grouped_query.new_zeros(
            (request_count, max_targets, num_heads),
            dtype=torch.float32,
        )
        context = grouped_query.new_zeros(
            (request_count, max_targets, num_heads, dim),
            dtype=torch.float32,
        )
        for start in range(0, length, length_chunk):
            end = min(length, start + length_chunk)
            hist = self._transform_sequence_slice(sequence, start, end)
            mask = valid_mask[:, start:end]
            scores = (
                torch.einsum("rmnd,rlnd->rmnl", grouped_query, hist) * self.scale
            ).float()
            scores = scores.masked_fill(~mask[:, None, None, :], neg)
            block_max = scores.amax(dim=-1)
            block_max = torch.where(
                torch.isfinite(block_max),
                block_max,
                running_max,
            )
            new_max = torch.maximum(running_max, block_max)
            prior_scale = torch.exp(running_max - new_max)
            weights = torch.exp(scores - new_max.unsqueeze(-1))
            weights = weights * mask[:, None, None, :].to(dtype=weights.dtype)
            context = context * prior_scale.unsqueeze(-1) + torch.einsum(
                "rmnl,rlnd->rmnd",
                weights,
                hist.float(),
            )
            running_sum = running_sum * prior_scale + weights.sum(dim=-1)
            running_max = new_max
        context = context / running_sum.unsqueeze(-1).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        return context.to(dtype=grouped_query.dtype)

    def _forward_aligned(
        self,
        score_query: Tensor,
        sequence: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        if sequence.size(0) == 1 and score_query.size(0) != 1:
            sequence = sequence.expand(score_query.size(0), -1, -1)
            valid_mask = valid_mask.expand(score_query.size(0), -1)
        length_chunk = self._attention_length_chunk(sequence.size(1))
        if length_chunk < sequence.size(1):
            context = self._context_aligned_chunked(
                score_query,
                sequence,
                valid_mask,
                length_chunk,
            )
        else:
            history = self._transform_sequence(sequence)
            context = self._context_from_aligned_scores(
                score_query,
                history,
                valid_mask,
            )
        return self._project_context(context)

    def _forward_grouped(
        self,
        score_query: Tensor,
        sequence: Tensor,
        valid_mask: Tensor,
        layout: MixFormerRequestLayout,
    ) -> Tensor:
        candidate_count = score_query.size(0)
        if candidate_count == 0:
            return score_query
        length_chunk = self._attention_length_chunk(sequence.size(1))
        request_chunk = self._grouped_request_chunk(
            request_count=layout.request_count,
            max_targets=layout.max_targets,
            sequence_length=sequence.size(1),
        )
        if request_chunk >= layout.request_count and length_chunk >= sequence.size(1):
            history = self._transform_sequence(sequence)
            return self._project_context(
                self._grouped_context_from_history(
                    score_query,
                    history,
                    valid_mask,
                    layout,
                )
            )
        grouped_query = self._pack_grouped_query(score_query, layout)
        occupancy = self._occupancy_query_mask(layout, grouped_query)
        head_count = score_query.size(1)
        if request_chunk < layout.request_count:
            grouped_context = grouped_query.new_empty(
                layout.request_count,
                layout.max_targets,
                head_count,
                self.dim,
            )
            for start in range(0, layout.request_count, request_chunk):
                end = min(layout.request_count, start + request_chunk)
                query_slice = grouped_query[start:end]
                seq_slice = sequence[start:end]
                mask_slice = valid_mask[start:end]
                query_mask = None if occupancy is None else occupancy[start:end]
                if length_chunk < sequence.size(1):
                    grouped_context[start:end] = self._context_grouped_chunked(
                        query_slice,
                        seq_slice,
                        mask_slice,
                        length_chunk,
                    )
                else:
                    hist_slice = self._transform_sequence(seq_slice)
                    grouped_context[start:end] = self._context_from_grouped_scores(
                        query_slice,
                        hist_slice,
                        mask_slice,
                        query_mask=query_mask,
                    )
        elif length_chunk < sequence.size(1):
            grouped_context = self._context_grouped_chunked(
                grouped_query,
                sequence,
                valid_mask,
                length_chunk,
            )
        else:
            history = self._transform_sequence(sequence)
            grouped_context = self._context_from_grouped_scores(
                grouped_query,
                history,
                valid_mask,
                query_mask=occupancy,
            )
        return self._project_context(
            self._unpack_grouped_context(grouped_context, score_query, layout)
        )

    def _pack_grouped_query(
        self,
        score_query: Tensor,
        layout: MixFormerRequestLayout,
    ) -> Tensor:
        head_count = score_query.size(1)
        sorted_query = score_query.index_select(0, layout.order)
        return (
            score_query.new_zeros(
                layout.request_count * layout.max_targets,
                head_count,
                self.dim,
            )
            .index_copy(0, layout.linear_slots, sorted_query)
            .view(
                layout.request_count,
                layout.max_targets,
                head_count,
                self.dim,
            )
        )

    def _unpack_grouped_context(
        self,
        grouped_context: Tensor,
        score_query: Tensor,
        layout: MixFormerRequestLayout,
    ) -> Tensor:
        head_count = score_query.size(1)
        sorted_context = grouped_context.reshape(
            layout.request_count * layout.max_targets,
            head_count,
            self.dim,
        ).index_select(0, layout.linear_slots)
        context = score_query.new_empty(
            score_query.size(0),
            head_count,
            self.dim,
        ).index_copy(0, layout.order, sorted_context)
        return context

    @staticmethod
    def _prefer_candidate_gather(
        *,
        candidate_count: int,
        request_count: int,
        max_targets: int,
        history_length: int,
        history_heads: int,
        history_dim: int,
        element_size: int,
    ) -> bool:
        """Gather per-candidate histories when rectangular packing is sparse.

        Grouped SDPA shares one K/V per request and pads queries to
        ``max_targets``. That wins when every request is nearly full (typical
        16-candidate lists). Agg-row packs often have a few candidates spread
        across many requests, so most query slots are dummy and still attend.
        Gathering is algebraically the same as the expanded path; skip it when
        duplicating ``L=8000`` keys would blow HBM. Dao varlen grouped attention
        packs occupancy instead, so callers skip gather when that kernel is live.
        """

        if candidate_count <= 0 or request_count <= 0 or max_targets <= 0:
            return False
        grouped_slots = request_count * max_targets
        if candidate_count * 5 >= grouped_slots * 4:
            return False
        gather_bytes = (
            candidate_count
            * max(history_length, 0)
            * max(history_heads, 0)
            * max(history_dim, 0)
            * max(element_size, 0)
        )
        return gather_bytes <= 256 * 1024 * 1024

    def _context_from_candidate_histories(
        self,
        score_query: Tensor,
        history: Tensor,
        valid_mask: Tensor,
        row_indices: Tensor,
    ) -> Tensor:
        indices = row_indices.to(device=history.device, dtype=torch.long)
        return self._attend_history(
            score_query,
            history.index_select(0, indices),
            valid_mask.index_select(0, indices),
        )

    def _grouped_context_from_history(
        self,
        score_query: Tensor,
        history: Tensor,
        valid_mask: Tensor,
        layout: MixFormerRequestLayout,
    ) -> Tensor:
        if score_query.size(0) == 0:
            return score_query
        if not self._can_varlen_history(score_query) and self._prefer_candidate_gather(
            candidate_count=int(score_query.size(0)),
            request_count=layout.request_count,
            max_targets=layout.max_targets,
            history_length=int(history.size(1)),
            history_heads=int(history.size(2)),
            history_dim=int(history.size(3)),
            element_size=int(history.element_size()),
        ):
            return self._context_from_candidate_histories(
                score_query,
                history,
                valid_mask,
                layout.row_indices,
            )
        grouped_query = self._pack_grouped_query(score_query, layout)
        grouped_context = self._context_from_grouped_scores(
            grouped_query,
            history,
            valid_mask,
            query_mask=self._occupancy_query_mask(layout, grouped_query),
        )
        return self._unpack_grouped_context(grouped_context, score_query, layout)

    def forward(
        self,
        query: Tensor,
        sequence: Tensor,
        valid_mask: Tensor,
        sequence_row_indices: Tensor | None = None,
        *,
        request_layout: MixFormerRequestLayout | None = None,
    ) -> Tensor:
        sequence_row_indices = self._validate(
            query,
            sequence,
            valid_mask,
            sequence_row_indices,
        )
        if sequence.size(1) == 0:
            return query
        score_query = self._project_query(query)
        if sequence_row_indices is None:
            if request_layout is not None:
                raise ValueError(
                    "request_layout requires sequence_row_indices"
                )
            update = self._forward_aligned(score_query, sequence, valid_mask.bool())
        else:
            layout = request_layout
            if layout is None:
                layout = self.request_layout(
                    sequence_row_indices,
                    sequence.size(0),
                )
            elif (
                layout.request_count != sequence.size(0)
                or layout.row_indices.shape != sequence_row_indices.shape
                or layout.order.numel() != sequence_row_indices.numel()
                or layout.linear_slots.numel() != sequence_row_indices.numel()
            ):
                raise ValueError("request_layout does not match attention inputs")
            update = self._forward_grouped(
                score_query,
                sequence,
                valid_mask.bool(),
                layout,
            )
        return query + update

    def forward_decoupled(
        self,
        user_query: Tensor,
        item_query: Tensor,
        sequence: Tensor,
        valid_mask: Tensor,
        sequence_row_indices: Tensor | None = None,
        *,
        request_layout: MixFormerRequestLayout | None = None,
    ) -> tuple[Tensor, Tensor]:
        user_head_count = self.user_head_count
        if user_head_count is None:
            raise RuntimeError("forward_decoupled requires user_head_count")
        item_head_count = self.num_heads - user_head_count
        if user_query.ndim != 3 or tuple(user_query.shape[1:]) != (
            user_head_count,
            self.dim,
        ):
            raise ValueError(
                f"user query must have shape [request, {user_head_count}, {self.dim}]"
            )
        if item_query.ndim != 3 or tuple(item_query.shape[1:]) != (
            item_head_count,
            self.dim,
        ):
            raise ValueError(
                f"item query must have shape [candidate, {item_head_count}, {self.dim}]"
            )
        if sequence.ndim != 3 or sequence.size(-1) != self.sequence_dim:
            raise ValueError(
                f"sequence must have shape [request, length, {self.sequence_dim}]"
            )
        if valid_mask.shape != sequence.shape[:2]:
            raise ValueError("valid_mask must match sequence batch and length")
        if user_query.size(0) != sequence.size(0):
            raise ValueError("UI user queries must be request-major")
        if sequence.size(1) == 0:
            return user_query, item_query

        user_score = self._project_query(user_query, 0)
        item_score = self._project_query(item_query, user_head_count)
        mask = valid_mask.bool()
        n_u = user_head_count
        if sequence_row_indices is None:
            if request_layout is not None:
                raise ValueError("request_layout requires sequence_row_indices")
            user_context, item_context = self._decoupled_aligned(
                user_score,
                item_score,
                sequence,
                mask,
                n_u,
            )
        else:
            layout = request_layout
            if layout is None:
                layout = self.request_layout(
                    sequence_row_indices,
                    sequence.size(0),
                )
            elif layout.request_count != sequence.size(0):
                raise ValueError("request_layout does not match attention inputs")
            user_context, item_context = self._decoupled_grouped(
                user_score,
                item_score,
                sequence,
                mask,
                layout,
                n_u,
            )
        return (
            user_query + self._project_context(user_context, 0),
            item_query + self._project_context(item_context, n_u),
        )

    def _broadcast_item_history(
        self,
        item_score: Tensor,
        history: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if item_score.size(0) == history.size(0):
            return history, valid_mask
        if history.size(0) != 1:
            raise ValueError(
                "aligned UI item queries must match the sequence request batch"
            )
        return (
            history.expand(item_score.size(0), *history.shape[1:]),
            valid_mask.expand(item_score.size(0), -1),
        )

    def _decoupled_aligned(
        self,
        user_score: Tensor,
        item_score: Tensor,
        sequence: Tensor,
        valid_mask: Tensor,
        user_head_count: int,
    ) -> tuple[Tensor, Tensor]:
        if user_score.size(0) != sequence.size(0):
            raise ValueError("UI user queries must be request-major")
        length_chunk = self._attention_length_chunk(sequence.size(1))
        if length_chunk < sequence.size(1):
            return self._decoupled_aligned_chunked(
                user_score,
                item_score,
                sequence,
                valid_mask,
                length_chunk,
                user_head_count,
            )
        history = self._transform_sequence(sequence)
        user_context = self._context_from_aligned_scores(
            user_score,
            history[:, :, :user_head_count],
            valid_mask,
        )
        item_history, item_mask = self._broadcast_item_history(
            item_score,
            history[:, :, user_head_count:],
            valid_mask,
        )
        item_context = self._context_from_aligned_scores(
            item_score,
            item_history,
            item_mask,
        )
        return user_context, item_context

    def _decoupled_aligned_chunked(
        self,
        user_score: Tensor,
        item_score: Tensor,
        sequence: Tensor,
        valid_mask: Tensor,
        length_chunk: int,
        user_head_count: int,
    ) -> tuple[Tensor, Tensor]:
        user_state = self._start_aligned_online(user_score)
        item_state = self._start_aligned_online(item_score)
        length = sequence.size(1)
        for start in range(0, length, length_chunk):
            end = min(length, start + length_chunk)
            hist = self._transform_sequence_slice(sequence, start, end)
            mask = valid_mask[:, start:end]
            user_state = self._step_aligned_online(
                user_state,
                user_score,
                hist[:, :, :user_head_count],
                mask,
            )
            item_hist, item_mask = self._broadcast_item_history(
                item_score,
                hist[:, :, user_head_count:],
                mask,
            )
            item_state = self._step_aligned_online(
                item_state,
                item_score,
                item_hist,
                item_mask,
            )
        return (
            self._finish_aligned_online(user_state, user_score),
            self._finish_aligned_online(item_state, item_score),
        )

    def _start_aligned_online(
        self, score_query: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, num_heads, dim = score_query.shape
        neg = torch.finfo(torch.float32).min
        running_max = score_query.new_full(
            (batch, num_heads),
            neg,
            dtype=torch.float32,
        )
        running_sum = score_query.new_zeros((batch, num_heads), dtype=torch.float32)
        context = score_query.new_zeros((batch, num_heads, dim), dtype=torch.float32)
        return running_max, running_sum, context

    def _step_aligned_online(
        self,
        state: tuple[Tensor, Tensor, Tensor],
        score_query: Tensor,
        hist: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        running_max, running_sum, context = state
        neg = torch.finfo(torch.float32).min
        scores = (
            torch.einsum("bnd,blnd->bnl", score_query, hist) * self.scale
        ).float()
        scores = scores.masked_fill(~mask.unsqueeze(1), neg)
        block_max = scores.amax(dim=-1)
        block_max = torch.where(
            torch.isfinite(block_max),
            block_max,
            running_max,
        )
        new_max = torch.maximum(running_max, block_max)
        prior_scale = torch.exp(running_max - new_max)
        weights = torch.exp(scores - new_max.unsqueeze(-1))
        weights = weights * mask.unsqueeze(1).to(dtype=weights.dtype)
        context = context * prior_scale.unsqueeze(-1) + torch.einsum(
            "bnl,blnd->bnd",
            weights,
            hist.float(),
        )
        running_sum = running_sum * prior_scale + weights.sum(dim=-1)
        return new_max, running_sum, context

    def _finish_aligned_online(
        self,
        state: tuple[Tensor, Tensor, Tensor],
        score_query: Tensor,
    ) -> Tensor:
        _running_max, running_sum, context = state
        context = context / running_sum.unsqueeze(-1).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        return context.to(dtype=score_query.dtype)

    def _decoupled_grouped(
        self,
        user_score: Tensor,
        item_score: Tensor,
        sequence: Tensor,
        valid_mask: Tensor,
        layout: MixFormerRequestLayout,
        user_head_count: int,
    ) -> tuple[Tensor, Tensor]:
        length_chunk = self._attention_length_chunk(sequence.size(1))
        request_chunk = self._grouped_request_chunk(
            request_count=layout.request_count,
            max_targets=layout.max_targets,
            sequence_length=sequence.size(1),
        )
        if request_chunk < layout.request_count or length_chunk < sequence.size(1):
            return self._decoupled_grouped_chunked(
                user_score,
                item_score,
                sequence,
                valid_mask,
                layout,
                user_head_count,
                length_chunk,
                request_chunk,
            )
        history = self._transform_sequence(sequence)
        user_context = self._context_from_aligned_scores(
            user_score,
            history[:, :, :user_head_count],
            valid_mask,
        )
        item_context = self._grouped_context_from_history(
            item_score,
            history[:, :, user_head_count:],
            valid_mask,
            layout,
        )
        return user_context, item_context

    def _decoupled_grouped_chunked(
        self,
        user_score: Tensor,
        item_score: Tensor,
        sequence: Tensor,
        valid_mask: Tensor,
        layout: MixFormerRequestLayout,
        user_head_count: int,
        length_chunk: int,
        request_chunk: int,
    ) -> tuple[Tensor, Tensor]:
        grouped_item = self._pack_grouped_query(item_score, layout)
        occupancy = self._occupancy_query_mask(layout, grouped_item)
        item_head_count = item_score.size(1)
        grouped_context = grouped_item.new_empty(
            layout.request_count,
            layout.max_targets,
            item_head_count,
            self.dim,
        )
        user_context = user_score.new_empty(user_score.shape)
        for request_start in range(0, layout.request_count, request_chunk):
            request_end = min(layout.request_count, request_start + request_chunk)
            seq_slice = sequence[request_start:request_end]
            mask_slice = valid_mask[request_start:request_end]
            user_slice = user_score[request_start:request_end]
            item_slice = grouped_item[request_start:request_end]
            query_mask = None if occupancy is None else occupancy[request_start:request_end]
            if length_chunk < sequence.size(1):
                user_part, item_part = self._decoupled_request_length_chunked(
                    user_slice,
                    item_slice,
                    seq_slice,
                    mask_slice,
                    length_chunk,
                    user_head_count,
                )
            else:
                history = self._transform_sequence(seq_slice)
                user_part = self._context_from_aligned_scores(
                    user_slice,
                    history[:, :, :user_head_count],
                    mask_slice,
                )
                item_part = self._context_from_grouped_scores(
                    item_slice,
                    history[:, :, user_head_count:],
                    mask_slice,
                    query_mask=query_mask,
                )
            user_context[request_start:request_end] = user_part
            grouped_context[request_start:request_end] = item_part
        return user_context, self._unpack_grouped_context(
            grouped_context,
            item_score,
            layout,
        )

    def _decoupled_request_length_chunked(
        self,
        user_score: Tensor,
        grouped_item_query: Tensor,
        sequence: Tensor,
        valid_mask: Tensor,
        length_chunk: int,
        user_head_count: int,
    ) -> tuple[Tensor, Tensor]:
        user_state = self._start_aligned_online(user_score)
        request_count, max_targets, item_heads, dim = grouped_item_query.shape
        neg = torch.finfo(torch.float32).min
        item_max = grouped_item_query.new_full(
            (request_count, max_targets, item_heads),
            neg,
            dtype=torch.float32,
        )
        item_sum = grouped_item_query.new_zeros(
            (request_count, max_targets, item_heads),
            dtype=torch.float32,
        )
        item_context = grouped_item_query.new_zeros(
            (request_count, max_targets, item_heads, dim),
            dtype=torch.float32,
        )
        length = sequence.size(1)
        for start in range(0, length, length_chunk):
            end = min(length, start + length_chunk)
            hist = self._transform_sequence_slice(sequence, start, end)
            mask = valid_mask[:, start:end]
            user_state = self._step_aligned_online(
                user_state,
                user_score,
                hist[:, :, :user_head_count],
                mask,
            )
            item_hist = hist[:, :, user_head_count:]
            scores = (
                torch.einsum("rmnd,rlnd->rmnl", grouped_item_query, item_hist)
                * self.scale
            ).float()
            scores = scores.masked_fill(~mask[:, None, None, :], neg)
            block_max = scores.amax(dim=-1)
            block_max = torch.where(
                torch.isfinite(block_max),
                block_max,
                item_max,
            )
            new_max = torch.maximum(item_max, block_max)
            prior_scale = torch.exp(item_max - new_max)
            weights = torch.exp(scores - new_max.unsqueeze(-1))
            weights = weights * mask[:, None, None, :].to(dtype=weights.dtype)
            item_context = item_context * prior_scale.unsqueeze(-1) + torch.einsum(
                "rmnl,rlnd->rmnd",
                weights,
                item_hist.float(),
            )
            item_sum = item_sum * prior_scale + weights.sum(dim=-1)
            item_max = new_max
        item_context = item_context / item_sum.unsqueeze(-1).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        return (
            self._finish_aligned_online(user_state, user_score),
            item_context.to(dtype=grouped_item_query.dtype),
        )

    def forward_reference(
        self,
        query: Tensor,
        sequence: Tensor,
        valid_mask: Tensor,
        sequence_row_indices: Tensor | None = None,
    ) -> Tensor:
        """Materialized K/V reference used by alignment tests only."""

        sequence_row_indices = self._validate(
            query,
            sequence,
            valid_mask,
            sequence_row_indices,
        )
        if sequence.size(1) == 0:
            return query
        history = self._transform_sequence(sequence)
        if sequence_row_indices is not None:
            history = history.index_select(0, sequence_row_indices)
            valid_mask = valid_mask.index_select(0, sequence_row_indices)
        elif history.size(0) == 1 and query.size(0) != 1:
            history = history.expand(query.size(0), -1, -1, -1)
            valid_mask = valid_mask.expand(query.size(0), -1)
        keys = torch.einsum("blni,noi->blno", history, self.key_weight)
        values = torch.einsum("blni,noi->blno", history, self.value_weight)
        scores = (
            torch.einsum("bno,blno->bnl", query, keys)
            * self.scale
        )
        weights = self._masked_softmax(scores, valid_mask.bool())
        return query + torch.einsum("bnl,blno->bno", weights, values)


class MixFormerBlock(nn.Module):
    """Query Mixer -> Cross Attention -> Output Fusion."""

    def __init__(
        self,
        num_heads: int,
        dim: int,
        hidden_dim: int,
        *,
        sequence_hidden_dim: int | None = None,
        sequence_chunk_tokens: int = 0,
        user_head_count: int | None = None,
    ) -> None:
        super().__init__()
        self.query_mixer = MixFormerQueryMixer(
            num_heads,
            dim,
            hidden_dim,
            user_head_count=user_head_count,
        )
        self.cross_attention = MixFormerCrossAttention(
            num_heads,
            dim,
            sequence_hidden_dim or hidden_dim,
            sequence_chunk_tokens=sequence_chunk_tokens,
            user_head_count=user_head_count,
        )
        self.output_fusion = MixFormerOutputFusion(
            num_heads,
            dim,
            hidden_dim,
        )

    def forward(
        self,
        feature_heads: Tensor,
        sequence: Tensor,
        valid_mask: Tensor,
        sequence_row_indices: Tensor | None = None,
        *,
        request_layout: MixFormerRequestLayout | None = None,
    ) -> Tensor:
        query = self.query_mixer(feature_heads)
        attended = self.cross_attention(
            query,
            sequence,
            valid_mask,
            sequence_row_indices,
            request_layout=request_layout,
        )
        return self.output_fusion(attended)

    def forward_decoupled(
        self,
        user_heads: Tensor,
        item_heads: Tensor,
        sequence: Tensor,
        valid_mask: Tensor,
        sequence_row_indices: Tensor | None = None,
        *,
        request_layout: MixFormerRequestLayout | None = None,
    ) -> tuple[Tensor, Tensor]:
        user_query, item_query = self.query_mixer.forward_decoupled(
            user_heads,
            item_heads,
            sequence_row_indices,
        )
        user_attended, item_attended = self.cross_attention.forward_decoupled(
            user_query,
            item_query,
            sequence,
            valid_mask,
            sequence_row_indices,
            request_layout=request_layout,
        )
        user_head_count = self.query_mixer.user_head_count
        if user_head_count is None:
            raise RuntimeError("forward_decoupled requires user_head_count")
        return (
            self.output_fusion.forward_heads(user_attended, 0),
            self.output_fusion.forward_heads(item_attended, user_head_count),
        )


__all__ = [
    "DenseSwiGLUFFN",
    "MixFormerBlock",
    "MixFormerCrossAttention",
    "MixFormerHeadMixing",
    "MixFormerOutputFusion",
    "MixFormerQueryMixer",
    "MixFormerRequestLayout",
    "MixFormerRMSNorm",
    "StackedPerHeadSwiGLUFFN",
    "assemble_mixformer_heads",
]
