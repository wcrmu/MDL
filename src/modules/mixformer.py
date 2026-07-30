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
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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
        scale = (
            values.float()
            .pow(2)
            .mean(dim=-1, keepdim=True)
            .add(self.eps)
            .rsqrt()
            .to(dtype=values.dtype)
        )
        return values * scale * self.weight.to(dtype=values.dtype)


class DenseSwiGLUFFN(nn.Module):
    """Dimension-preserving SwiGLU with an explicit intermediate width."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        if dim <= 0 or hidden_dim <= 0:
            raise ValueError("dim and hidden_dim must be positive")
        self.dim = dim
        self.hidden_dim = hidden_dim
        # The MixFormer equations contain matrices only; keep all projections
        # bias-free so the implementation has the same parameterization.
        self.up_projection = nn.Linear(dim, hidden_dim, bias=False)
        self.gate_projection = nn.Linear(dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, values: Tensor) -> Tensor:
        if values.size(-1) != self.dim:
            raise ValueError(
                f"SwiGLU expected trailing dimension {self.dim}, "
                f"got {values.size(-1)}"
            )
        return self.output_projection(
            self.up_projection(values) * F.silu(self.gate_projection(values))
        )


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
        expected = (self.num_heads, self.dim)
        if values.ndim != 3 or tuple(values.shape[1:]) != expected:
            raise ValueError(
                f"expected values with shape [batch, {self.num_heads}, "
                f"{self.dim}], got {tuple(values.shape)}"
            )
        head_major = values.transpose(0, 1)
        up = torch.bmm(head_major, self.up_weight.transpose(1, 2))
        gate = torch.bmm(head_major, self.gate_weight.transpose(1, 2))
        hidden = up * F.silu(gate)
        output = torch.bmm(hidden, self.output_weight.transpose(1, 2))
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

    def forward(self, heads: Tensor) -> Tensor:
        mixed = heads + self.head_mixing(self.input_norm(heads))
        return mixed + self.ffn(self.ffn_norm(mixed))


class MixFormerOutputFusion(nn.Module):
    """Per-head SwiGLU output fusion from the MixFormer block."""

    def __init__(self, num_heads: int, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = MixFormerRMSNorm(dim)
        self.ffn = StackedPerHeadSwiGLUFFN(num_heads, dim, hidden_dim)

    def forward(self, heads: Tensor) -> Tensor:
        return heads + self.ffn(self.norm(heads))


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
    ) -> None:
        super().__init__()
        if num_heads <= 0 or dim <= 0 or hidden_dim <= 0:
            raise ValueError("head count and dimensions must be positive")
        if sequence_chunk_tokens < 0:
            raise ValueError("sequence_chunk_tokens must be non-negative")
        self.num_heads = num_heads
        self.dim = dim
        self.sequence_dim = num_heads * dim
        self.scale = dim**-0.5
        self.sequence_chunk_tokens = sequence_chunk_tokens
        self.sequence_norm = MixFormerRMSNorm(self.sequence_dim)
        self.sequence_ffn = DenseSwiGLUFFN(self.sequence_dim, hidden_dim)
        # One independent D x D key/value matrix for every query head.
        self.key_weight = nn.Parameter(torch.empty(num_heads, dim, dim))
        self.value_weight = nn.Parameter(torch.empty(num_heads, dim, dim))
        self.query_norm = MixFormerRMSNorm(dim)
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

    def _transform_sequence(self, sequence: Tensor) -> Tensor:
        normalized = self.sequence_norm(sequence)
        flat = normalized.reshape(-1, self.sequence_dim)
        if (
            self.sequence_chunk_tokens > 0
            and flat.size(0) > self.sequence_chunk_tokens
        ):
            update = torch.cat(
                [
                    self.sequence_ffn(chunk)
                    for chunk in flat.split(self.sequence_chunk_tokens, dim=0)
                ],
                dim=0,
            ).view_as(sequence)
        else:
            update = self.sequence_ffn(normalized)
        return (sequence + update).view(
            sequence.size(0),
            sequence.size(1),
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

    def _project_query(self, query: Tensor) -> Tensor:
        normalized = self.query_norm(query)
        # Linear convention: k = h W_k^T, hence q W_k is the vector scored
        # directly against the unprojected history.
        return torch.einsum("bno,noi->bni", normalized, self.key_weight)

    def _project_context(self, context: Tensor) -> Tensor:
        return torch.einsum("bni,noi->bno", context, self.value_weight)

    def _attention_length_chunk(self, length: int) -> int:
        """Bound score storage when sequence FFN chunking is enabled.

        ``sequence_chunk_tokens`` is primarily a flattened FFN budget. When it
        is set we also split the attention length axis so peak score tensors
        stay proportional to the chunk instead of full ``L``. Production-sized
        budgets keep a 256-token floor to avoid tiny GEMMs; smaller budgets
        (tests / extreme HBM caps) honor the raw limit.
        """

        if self.sequence_chunk_tokens <= 0 or length <= 0:
            return length
        budget = max(1, self.sequence_chunk_tokens // max(self.num_heads, 1))
        if self.sequence_chunk_tokens >= 256:
            budget = max(budget, 256)
        return min(length, budget)

    def _context_from_aligned_scores(
        self,
        score_query: Tensor,
        history: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        scores = (
            torch.einsum("bnd,blnd->bnl", score_query, history) * self.scale
        )
        weights = self._masked_softmax(scores, valid_mask)
        return torch.einsum("bnl,blnd->bnd", weights, history)

    def _context_from_grouped_scores(
        self,
        grouped_query: Tensor,
        history: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        scores = (
            torch.einsum("rmnd,rlnd->rmnl", grouped_query, history) * self.scale
        )
        weights = self._masked_softmax(scores, valid_mask)
        return torch.einsum("rmnl,rlnd->rmnd", weights, history)

    def _context_aligned_chunked(
        self,
        score_query: Tensor,
        history: Tensor,
        valid_mask: Tensor,
        length_chunk: int,
    ) -> Tensor:
        """Online-softmax attention along the sequence length axis."""

        batch, num_heads, dim = score_query.shape
        length = history.size(1)
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
            hist = history[:, start:end]
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
        history: Tensor,
        valid_mask: Tensor,
        length_chunk: int,
    ) -> Tensor:
        request_count, max_targets, num_heads, dim = grouped_query.shape
        length = history.size(1)
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
            hist = history[:, start:end]
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
        history: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        if history.size(0) == 1 and score_query.size(0) != 1:
            history = history.expand(score_query.size(0), -1, -1, -1)
            valid_mask = valid_mask.expand(score_query.size(0), -1)
        length_chunk = self._attention_length_chunk(history.size(1))
        if length_chunk < history.size(1):
            context = self._context_aligned_chunked(
                score_query,
                history,
                valid_mask,
                length_chunk,
            )
        else:
            context = self._context_from_aligned_scores(
                score_query,
                history,
                valid_mask,
            )
        return self._project_context(context)

    def _forward_grouped(
        self,
        score_query: Tensor,
        history: Tensor,
        valid_mask: Tensor,
        layout: MixFormerRequestLayout,
    ) -> Tensor:
        candidate_count = score_query.size(0)
        if candidate_count == 0:
            return score_query
        sorted_query = score_query.index_select(0, layout.order)
        grouped_query = (
            score_query.new_zeros(
                layout.request_count * layout.max_targets,
                self.num_heads,
                self.dim,
            )
            .index_copy(0, layout.linear_slots, sorted_query)
            .view(
                layout.request_count,
                layout.max_targets,
                self.num_heads,
                self.dim,
            )
        )
        length_chunk = self._attention_length_chunk(history.size(1))
        # Bound padded request×target score storage by walking request rows when
        # the FFN token budget implies the full score tensor would dominate HBM.
        request_chunk = layout.request_count
        if self.sequence_chunk_tokens > 0 and history.size(1) > 0:
            tokens_per_request = history.size(1) * max(layout.max_targets, 1)
            request_chunk = max(
                1,
                min(
                    layout.request_count,
                    max(1, self.sequence_chunk_tokens // max(tokens_per_request, 1)),
                ),
            )
        if request_chunk < layout.request_count:
            grouped_context = grouped_query.new_empty(
                layout.request_count,
                layout.max_targets,
                self.num_heads,
                self.dim,
            )
            for start in range(0, layout.request_count, request_chunk):
                end = min(layout.request_count, start + request_chunk)
                query_slice = grouped_query[start:end]
                hist_slice = history[start:end]
                mask_slice = valid_mask[start:end]
                if length_chunk < history.size(1):
                    grouped_context[start:end] = self._context_grouped_chunked(
                        query_slice,
                        hist_slice,
                        mask_slice,
                        length_chunk,
                    )
                else:
                    grouped_context[start:end] = self._context_from_grouped_scores(
                        query_slice,
                        hist_slice,
                        mask_slice,
                    )
        elif length_chunk < history.size(1):
            grouped_context = self._context_grouped_chunked(
                grouped_query,
                history,
                valid_mask,
                length_chunk,
            )
        else:
            grouped_context = self._context_from_grouped_scores(
                grouped_query,
                history,
                valid_mask,
            )
        sorted_context = grouped_context.reshape(
            layout.request_count * layout.max_targets,
            self.num_heads,
            self.dim,
        ).index_select(0, layout.linear_slots)
        context = score_query.new_empty(
            candidate_count,
            self.num_heads,
            self.dim,
        ).index_copy(0, layout.order, sorted_context)
        return self._project_context(context)

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
        history = self._transform_sequence(sequence)
        score_query = self._project_query(query)
        if sequence_row_indices is None:
            if request_layout is not None:
                raise ValueError(
                    "request_layout requires sequence_row_indices"
                )
            update = self._forward_aligned(score_query, history, valid_mask.bool())
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
                history,
                valid_mask.bool(),
                layout,
            )
        return query + update

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
        normalized_query = self.query_norm(query)
        scores = (
            torch.einsum("bno,blno->bnl", normalized_query, keys)
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
]
