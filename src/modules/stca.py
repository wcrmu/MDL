"""Paper-aligned Stacked Target-to-History Cross Attention (STCA).

The implementation follows ``paper/STCA/main.tex``:

* history tokens are transformed independently at every layer by a
  dimension-preserving SwiGLU FFN followed by LayerNorm;
* the candidate is the only query in target-to-history cross attention;
* every new query fuses all preceding layer outputs with the original target;
* the final target-aware token fuses all layer outputs with the target.

The attention path uses the exact single-query reordering from Eq. (10) of the
paper.  It never materializes projected ``[batch, heads, length, head_dim]``
key/value tensors; its only length-dependent representations are the input
history and the attention weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class STCASequenceCache:
    """Request-side STCA state reusable across candidate targets.

    Every transformed history is candidate-independent:

    ``X_tilde[i] = LN(SwiGLUFFN[i](X))``.

    Keeping those tensors request-major mirrors the paper's RLB boundary.  The
    target-aware attention remains candidate-major and consumes
    ``history_row_indices`` without expanding a full ``[candidate, L, d]``
    history activation.
    """

    transformed_histories: tuple[Tensor, ...]
    valid_mask: Tensor


@dataclass(frozen=True)
class _RequestTargetLayout:
    """Candidate packing metadata shared by every STCA attention layer."""

    row_indices: Tensor
    order: Tensor
    linear_slots: Tensor
    request_count: int
    max_targets: int


class SwiGLUFFN(nn.Module):
    """Dimension-preserving SwiGLU FFN from the STCA paper.

    For an input ``x`` this computes

    ``(x W_u) * silu(x W_v) W_o``.

    ``expansion_ratio`` is the paper's ``r`` and therefore determines the
    intermediate width ``r * dim``.
    """

    def __init__(
        self,
        dim: int,
        expansion_ratio: int = 4,
        *,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if expansion_ratio <= 0:
            raise ValueError("expansion_ratio must be positive")
        hidden_dim = dim * expansion_ratio
        self.dim = dim
        self.expansion_ratio = expansion_ratio
        self.up_projection = nn.Linear(dim, hidden_dim, bias=bias)
        self.gate_projection = nn.Linear(dim, hidden_dim, bias=bias)
        self.output_projection = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, values: Tensor) -> Tensor:
        if values.size(-1) != self.dim:
            raise ValueError(
                f"SwiGLUFFN expected trailing dimension {self.dim}, "
                f"got {values.size(-1)}"
            )
        return self.output_projection(
            self.up_projection(values) * F.silu(self.gate_projection(values))
        )


class SingleQueryTargetAttention(nn.Module):
    """Exact multi-head target-to-history attention with reordered K/V work.

    The public ``forward`` implements

    ``softmax((((q W_Q) W_K^T) X^T) / sqrt(d_h)) X W_V``

    independently for each head.  This is algebraically identical to ordinary
    projected-key/value attention for a single query, while avoiding
    length-by-head-dimension K/V intermediates.

    ``history_row_indices`` supports request-level batching.  History-side
    tensors can have one row per request while queries have one row per
    candidate; each candidate selects its request with the supplied mapping.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.query_projection = nn.Linear(dim, dim, bias=bias)
        self.key_projection = nn.Linear(dim, dim, bias=bias)
        self.value_projection = nn.Linear(dim, dim, bias=bias)
        self.output_projection = nn.Linear(dim, dim, bias=bias)

    def _validate_inputs(
        self,
        query: Tensor,
        history: Tensor,
        valid_mask: Tensor,
        history_row_indices: Tensor | None,
        *,
        validate_row_bounds: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        if query.ndim == 3:
            if query.size(1) != 1:
                raise ValueError("query must contain exactly one target token")
            query = query[:, 0, :]
        if query.ndim != 2 or query.size(-1) != self.dim:
            raise ValueError(
                f"query must have shape [batch, {self.dim}] or "
                f"[batch, 1, {self.dim}]"
            )
        if history.ndim != 3 or history.size(-1) != self.dim:
            raise ValueError(f"history must have shape [batch, length, {self.dim}]")
        if valid_mask.ndim != 2 or valid_mask.shape != history.shape[:2]:
            raise ValueError(
                "valid_mask must have shape [history_batch, history_length]"
            )
        if history.device != query.device or valid_mask.device != history.device:
            raise ValueError("query, history, and valid_mask must be on one device")

        query_batch = query.size(0)
        history_batch = history.size(0)
        if history_row_indices is not None:
            if history_row_indices.ndim != 1:
                raise ValueError("history_row_indices must be rank one")
            if history_row_indices.numel() != query_batch:
                raise ValueError(
                    "history_row_indices must contain one request index per query"
                )
            row_indices = history_row_indices.to(
                device=history.device,
                dtype=torch.long,
            )
            if validate_row_bounds and row_indices.numel() > 0:
                if history_batch == 0:
                    raise ValueError(
                        "history cannot be empty on the batch axis when queries exist"
                    )
                if bool(((row_indices < 0) | (row_indices >= history_batch)).any()):
                    raise ValueError(
                        "history_row_indices contains an out-of-range request index"
                    )
        elif history_batch == query_batch:
            pass
        elif history_batch == 1:
            pass
        else:
            raise ValueError(
                "history batch must equal query batch (or be one); provide "
                "history_row_indices for request-level batches"
            )
        return (
            query,
            history,
            valid_mask.to(dtype=torch.bool),
            (row_indices if history_row_indices is not None else None),
        )

    @staticmethod
    def _align_history_rows(
        query_batch: int,
        history: Tensor,
        valid_mask: Tensor,
        history_row_indices: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Materialize aligned histories for the test-only reference path."""

        if history_row_indices is not None:
            return (
                history.index_select(0, history_row_indices),
                valid_mask.index_select(0, history_row_indices),
            )
        if history.size(0) == 1 and query_batch != 1:
            return (
                history.expand(query_batch, -1, -1),
                valid_mask.expand(query_batch, -1),
            )
        return history, valid_mask

    @staticmethod
    def _masked_softmax(scores: Tensor, valid_mask: Tensor) -> Tensor:
        """Stable masked softmax with a defined zero result for empty rows."""

        if scores.size(-1) == 0:
            return scores
        if (
            valid_mask.ndim != 2
            or valid_mask.size(0) != scores.size(0)
            or valid_mask.size(1) != scores.size(-1)
        ):
            raise ValueError(
                "valid_mask must match the first and last score dimensions"
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
        # Renormalization removes any finite-precision mass assigned to masked
        # positions and maps an all-masked history to exactly zero.
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny
        )

    def _project_query(self, query: Tensor) -> Tensor:
        return self.query_projection(query).view(
            query.size(0),
            self.num_heads,
            self.head_dim,
        )

    def _score_bias(self, projected_query: Tensor) -> Tensor | None:
        if self.key_projection.bias is None:
            return None
        # A projected-key bias is constant over history positions and therefore
        # cancels in softmax. Retaining it keeps the optional biased extension
        # exactly equivalent to the materialized reference, including autograd.
        key_bias = self.key_projection.bias.view(
            self.num_heads,
            self.head_dim,
        )
        return (
            torch.einsum(
                "bhd,hd->bh",
                projected_query,
                key_bias,
            )
            * self.scale
        )

    def _project_weighted_history(
        self,
        weighted_history: Tensor,
        has_history: Tensor,
    ) -> Tensor:
        value_weight = self.value_projection.weight.view(
            self.num_heads,
            self.head_dim,
            self.dim,
        )
        head_outputs = torch.einsum(
            "bhe,hde->bhd",
            weighted_history,
            value_weight,
        )
        if self.value_projection.bias is not None:
            value_bias = self.value_projection.bias.view(
                self.num_heads,
                self.head_dim,
            )
            head_outputs = head_outputs + (
                has_history.to(dtype=head_outputs.dtype).unsqueeze(1)
                * value_bias.unsqueeze(0)
            )
        output = self.output_projection(head_outputs.flatten(start_dim=1))
        return output * has_history.to(dtype=output.dtype)

    def _forward_aligned(
        self,
        score_vector: Tensor,
        score_bias: Tensor | None,
        history: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        query_batch = score_vector.size(0)
        if history.size(0) == 1 and query_batch != 1:
            # ``expand`` is a zero-stride view. Unlike request-indexed
            # ``index_select``, it does not allocate a candidate-sized history.
            history = history.expand(query_batch, -1, -1)
            valid_mask = valid_mask.expand(query_batch, -1)
        scores = (
            torch.einsum(
                "bhe,ble->bhl",
                score_vector,
                history,
            )
            * self.scale
        )
        if score_bias is not None:
            scores = scores + score_bias.unsqueeze(-1)
        weights = self._masked_softmax(scores, valid_mask)
        weighted_history = torch.einsum("bhl,ble->bhe", weights, history)
        return self._project_weighted_history(
            weighted_history,
            valid_mask.any(dim=-1, keepdim=True),
        )

    def _forward_request_grouped(
        self,
        score_vector: Tensor,
        score_bias: Tensor | None,
        history: Tensor,
        valid_mask: Tensor,
        layout: _RequestTargetLayout,
    ) -> Tensor:
        """RLB attention without candidate-expanding the long history tensor.

        Candidate queries are stably grouped into a padded
        ``[request, max_targets, heads, d]`` tensor. One request-major history is
        then shared by every target slot in the einsums. Padding is only on the
        small target axis (the paper uses ``m=8``), never on ``[L, d]``.
        """

        candidate_count = score_vector.size(0)
        if candidate_count == 0:
            return score_vector.new_zeros(0, self.dim)
        request_count = layout.request_count
        order = layout.order
        linear_slots = layout.linear_slots
        max_targets = layout.max_targets

        sorted_score_vector = score_vector.index_select(0, order)
        grouped_score_vector = (
            score_vector.new_zeros(
                request_count * max_targets,
                self.num_heads,
                self.dim,
            )
            .index_copy(
                0,
                linear_slots,
                sorted_score_vector,
            )
            .view(
                request_count,
                max_targets,
                self.num_heads,
                self.dim,
            )
        )
        scores = (
            torch.einsum(
                "rmhe,rle->rmhl",
                grouped_score_vector,
                history,
            )
            * self.scale
        )
        if score_bias is not None:
            sorted_score_bias = score_bias.index_select(0, order)
            grouped_score_bias = (
                score_bias.new_zeros(
                    request_count * max_targets,
                    self.num_heads,
                )
                .index_copy(
                    0,
                    linear_slots,
                    sorted_score_bias,
                )
                .view(
                    request_count,
                    max_targets,
                    self.num_heads,
                )
            )
            scores = scores + grouped_score_bias.unsqueeze(-1)
        weights = self._masked_softmax(scores, valid_mask)
        grouped_weighted_history = torch.einsum(
            "rmhl,rle->rmhe",
            weights,
            history,
        )
        sorted_weighted_history = grouped_weighted_history.view(
            request_count * max_targets,
            self.num_heads,
            self.dim,
        ).index_select(
            0,
            linear_slots,
        )
        weighted_history = score_vector.new_empty(
            candidate_count,
            self.num_heads,
            self.dim,
        ).index_copy(
            0,
            order,
            sorted_weighted_history,
        )
        request_has_history = valid_mask.any(dim=-1, keepdim=True)
        has_history = request_has_history.index_select(
            0,
            layout.row_indices,
        )
        return self._project_weighted_history(weighted_history, has_history)

    @staticmethod
    def _request_target_layout(
        history_row_indices: Tensor,
        request_count: int,
    ) -> _RequestTargetLayout:
        """Build the small RLB target packing once for all stacked layers."""

        candidate_count = int(history_row_indices.numel())
        if candidate_count == 0:
            empty = history_row_indices.new_empty(0)
            return _RequestTargetLayout(
                row_indices=history_row_indices,
                order=empty,
                linear_slots=empty,
                request_count=request_count,
                max_targets=0,
            )
        order = torch.argsort(history_row_indices, stable=True)
        sorted_rows = history_row_indices.index_select(0, order)
        counts = torch.bincount(sorted_rows, minlength=request_count)
        max_targets = int(counts.max().item())
        request_offsets = counts.cumsum(dim=0) - counts
        within_request = torch.arange(
            candidate_count,
            device=history_row_indices.device,
        ) - torch.repeat_interleave(request_offsets, counts)
        linear_slots = sorted_rows * max_targets + within_request
        return _RequestTargetLayout(
            row_indices=history_row_indices,
            order=order,
            linear_slots=linear_slots,
            request_count=request_count,
            max_targets=max_targets,
        )

    def forward(
        self,
        query: Tensor,
        history: Tensor,
        valid_mask: Tensor,
        history_row_indices: Tensor | None = None,
        *,
        request_target_layout: _RequestTargetLayout | None = None,
    ) -> Tensor:
        query, history, valid_mask, history_row_indices = self._validate_inputs(
            query,
            history,
            valid_mask,
            history_row_indices,
            validate_row_bounds=request_target_layout is None,
        )
        if history.size(1) == 0:
            return query.new_zeros(query.size(0), self.dim)

        projected_query = self._project_query(query)
        # nn.Linear stores [out_dim, in_dim].  Grouping its output rows by head
        # gives W_K^T in the paper's notation: [head, head_dim, dim].
        key_weight = self.key_projection.weight.view(
            self.num_heads,
            self.head_dim,
            self.dim,
        )
        score_vector = torch.einsum(
            "bhd,hde->bhe",
            projected_query,
            key_weight,
        )
        score_bias = self._score_bias(projected_query)
        if history_row_indices is not None:
            layout = request_target_layout
            if layout is None:
                layout = self._request_target_layout(
                    history_row_indices,
                    history.size(0),
                )
            elif (
                layout.request_count != history.size(0)
                or layout.row_indices.shape != history_row_indices.shape
                or layout.row_indices.device != history_row_indices.device
                or (
                    layout.row_indices.data_ptr() != history_row_indices.data_ptr()
                    and not torch.equal(
                        layout.row_indices,
                        history_row_indices,
                    )
                )
                or layout.order.numel() != history_row_indices.numel()
                or layout.linear_slots.numel() != history_row_indices.numel()
                or (history_row_indices.numel() > 0 and layout.max_targets <= 0)
            ):
                raise ValueError(
                    "request_target_layout does not match attention inputs"
                )
            return self._forward_request_grouped(
                score_vector,
                score_bias,
                history,
                valid_mask,
                layout,
            )
        if request_target_layout is not None:
            raise ValueError("request_target_layout requires history_row_indices")
        return self._forward_aligned(
            score_vector,
            score_bias,
            history,
            valid_mask,
        )

    def forward_reference(
        self,
        query: Tensor,
        history: Tensor,
        valid_mask: Tensor,
        history_row_indices: Tensor | None = None,
    ) -> Tensor:
        """Materialized K/V reference used to verify the optimized path.

        This method is intentionally not used by model inference.
        """

        query, history, valid_mask, history_row_indices = self._validate_inputs(
            query,
            history,
            valid_mask,
            history_row_indices,
        )
        history, valid_mask = self._align_history_rows(
            query.size(0),
            history,
            valid_mask,
            history_row_indices,
        )
        if history.size(1) == 0:
            return query.new_zeros(query.size(0), self.dim)
        projected_query = self._project_query(query)
        keys = (
            self.key_projection(history)
            .view(
                history.size(0),
                history.size(1),
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        values = (
            self.value_projection(history)
            .view(
                history.size(0),
                history.size(1),
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )
        scores = (
            torch.einsum(
                "bhd,bhld->bhl",
                projected_query,
                keys,
            )
            * self.scale
        )
        weights = self._masked_softmax(scores, valid_mask)
        head_outputs = torch.einsum("bhl,bhld->bhd", weights, values)
        output = self.output_projection(head_outputs.flatten(start_dim=1))
        return output * valid_mask.any(dim=-1, keepdim=True).to(dtype=output.dtype)


class STCAInputLayer(nn.Module):
    """One STCA history transform and target-to-history attention layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        expansion_ratio: int,
        *,
        bias: bool = False,
        history_chunk_tokens: int = 0,
    ) -> None:
        super().__init__()
        if history_chunk_tokens < 0:
            raise ValueError("history_chunk_tokens must be non-negative")
        self.history_chunk_tokens = history_chunk_tokens
        # The paper uses SwiGLUFFN^(i) for both X and q/fusion at layer i.
        # Sharing this module and LayerNorm follows that notation exactly.
        self.input_ffn = SwiGLUFFN(
            dim,
            expansion_ratio,
            bias=bias,
        )
        self.input_norm = nn.LayerNorm(dim)
        self.attention = SingleQueryTargetAttention(
            dim,
            num_heads,
            bias=bias,
        )

    def transform(self, values: Tensor) -> Tensor:
        # The transform is token-wise, so splitting the flattened history axis
        # is mathematically exact and bounds the r*d SwiGLU activation at 10k
        # lengths. Queries remain unchunked because they are rank two and small.
        if (
            values.ndim < 3
            or self.history_chunk_tokens <= 0
            or values.numel() // values.size(-1) <= self.history_chunk_tokens
        ):
            return self.input_norm(self.input_ffn(values))
        flat_values = values.reshape(-1, values.size(-1))
        transformed = torch.cat(
            [
                self.input_norm(self.input_ffn(chunk))
                for chunk in flat_values.split(self.history_chunk_tokens, dim=0)
            ],
            dim=0,
        )
        return transformed.view_as(values)

    def forward(
        self,
        history: Tensor,
        valid_mask: Tensor,
        query: Tensor,
        history_row_indices: Tensor | None = None,
        request_target_layout: _RequestTargetLayout | None = None,
    ) -> Tensor:
        return self.attention(
            query,
            self.transform(history),
            valid_mask,
            history_row_indices,
            request_target_layout=request_target_layout,
        )


class STCASequenceEncoder(nn.Module):
    """Stacked Target-to-History Cross Attention sequence encoder.

    Args:
        dim: History/target/output token width ``d``.
        num_heads: Number of attention heads ``h``.
        num_layers: Number of stacked cross-attention layers ``M``.
        expansion_ratio: SwiGLU expansion ratio ``r``.
        bias: Whether linear projections use biases.  The paper omits biases,
            so the default is ``False``.
        activation_checkpoint: Recompute each history-transform/attention layer
            during backward to bound long-sequence activation memory.
        history_chunk_tokens: Maximum flattened history tokens in one SwiGLU
            invocation. Token-wise chunking is equation-preserving; zero disables it.

    The output is the paper's final target-aware token ``z`` with shape
    ``[candidate_batch, dim]``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int = 4,
        expansion_ratio: int = 4,
        *,
        bias: bool = False,
        activation_checkpoint: bool = False,
        history_chunk_tokens: int = 0,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if expansion_ratio <= 0:
            raise ValueError("expansion_ratio must be positive")
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError("num_heads must be positive and divide dim")
        if history_chunk_tokens < 0:
            raise ValueError("history_chunk_tokens must be non-negative")
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.expansion_ratio = expansion_ratio
        self.activation_checkpoint = activation_checkpoint
        self.history_chunk_tokens = history_chunk_tokens
        self.layers = nn.ModuleList(
            STCAInputLayer(
                dim,
                num_heads,
                expansion_ratio,
                bias=bias,
                history_chunk_tokens=history_chunk_tokens,
            )
            for _ in range(num_layers)
        )
        # Transition after layer i consumes i+1 summaries plus x_t.  There is
        # no transition after the final layer.
        self.query_fusion_projections = nn.ModuleList(
            nn.Linear((layer_index + 2) * dim, dim, bias=bias)
            for layer_index in range(num_layers - 1)
        )
        self.final_projection = nn.Linear(
            (num_layers + 1) * dim,
            dim,
            bias=bias,
        )
        self.final_ffn = SwiGLUFFN(
            dim,
            expansion_ratio,
            bias=bias,
        )

    def precompute_cache(
        self,
        history: Tensor,
        valid_mask: Tensor,
    ) -> STCASequenceCache:
        """Compute the complete candidate-independent RLB history path once."""

        if history.ndim != 3 or history.size(-1) != self.dim:
            raise ValueError(f"history must have shape [batch, length, {self.dim}]")
        if valid_mask.ndim != 2 or valid_mask.shape != history.shape[:2]:
            raise ValueError("valid_mask shape must match history[:2]")
        if valid_mask.device != history.device:
            raise ValueError("history and valid_mask must be on one device")
        return STCASequenceCache(
            transformed_histories=tuple(
                layer.transform(history) for layer in self.layers
            ),
            valid_mask=valid_mask.to(dtype=torch.bool),
        )

    def _validate_cache(self, cache: STCASequenceCache) -> None:
        if len(cache.transformed_histories) != self.num_layers:
            raise ValueError("STCA cache depth does not match encoder depth")
        if not cache.transformed_histories:
            raise ValueError("STCA cache must contain transformed histories")
        reference = cache.transformed_histories[0]
        if reference.ndim != 3 or reference.size(-1) != self.dim:
            raise ValueError(
                f"cached histories must have shape [batch, length, {self.dim}]"
            )
        if cache.valid_mask.ndim != 2 or cache.valid_mask.shape != reference.shape[:2]:
            raise ValueError("STCA cache mask shape must match cached history[:2]")
        for transformed in cache.transformed_histories:
            if transformed.shape != reference.shape:
                raise ValueError("all STCA cached history layers must share one shape")
            if transformed.device != reference.device:
                raise ValueError("all STCA cached history layers must share one device")
        if cache.valid_mask.device != reference.device:
            raise ValueError("STCA cached history and mask must share one device")

    def _validate_inputs(
        self,
        history: Tensor,
        valid_mask: Tensor,
        target: Tensor,
        history_row_indices: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        if target.ndim == 3:
            if target.size(1) != 1:
                raise ValueError("target must contain exactly one token")
            target = target[:, 0, :]
        if target.ndim != 2 or target.size(-1) != self.dim:
            raise ValueError(f"target must have shape [batch, {self.dim}]")
        if history.ndim != 3 or history.size(-1) != self.dim:
            raise ValueError(f"history must have shape [batch, length, {self.dim}]")
        if valid_mask.shape != history.shape[:2] or valid_mask.ndim != 2:
            raise ValueError("valid_mask shape must match history[:2]")
        if history.device != target.device or valid_mask.device != history.device:
            raise ValueError("history, valid_mask, and target must be on one device")
        if history_row_indices is not None:
            if history_row_indices.ndim != 1:
                raise ValueError("history_row_indices must be rank one")
            if history_row_indices.numel() != target.size(0):
                raise ValueError(
                    "history_row_indices must contain one request index per target"
                )
            history_row_indices = history_row_indices.to(
                device=history.device,
                dtype=torch.long,
            )
            if history_row_indices.numel() > 0:
                history_batch = history.size(0)
                if history_batch == 0:
                    raise ValueError(
                        "history cannot be empty on the batch axis when targets exist"
                    )
                if bool(
                    (
                        (history_row_indices < 0)
                        | (history_row_indices >= history_batch)
                    ).any()
                ):
                    raise ValueError(
                        "history_row_indices contains an out-of-range request index"
                    )
        elif history.size(0) not in {1, target.size(0)}:
            raise ValueError(
                "history batch must match target batch (or be one); provide "
                "history_row_indices for request-level batching"
            )
        return history, valid_mask.to(dtype=torch.bool), target, history_row_indices

    def forward(
        self,
        history: Tensor | None,
        valid_mask: Tensor | None,
        target: Tensor,
        history_row_indices: Tensor | None = None,
        cache: STCASequenceCache | None = None,
    ) -> Tensor:
        if cache is None:
            if history is None or valid_mask is None:
                raise ValueError(
                    "history and valid_mask are required without an STCA cache"
                )
            history, valid_mask, target, history_row_indices = self._validate_inputs(
                history,
                valid_mask,
                target,
                history_row_indices,
            )
        else:
            self._validate_cache(cache)
            cache_history = cache.transformed_histories[0]
            (
                _cache_history,
                _cache_mask,
                target,
                history_row_indices,
            ) = self._validate_inputs(
                cache_history,
                cache.valid_mask,
                target,
                history_row_indices,
            )
        request_target_layout = (
            None
            if history_row_indices is None
            else SingleQueryTargetAttention._request_target_layout(
                history_row_indices,
                (
                    history.size(0)
                    if cache is None
                    else cache.transformed_histories[0].size(0)
                ),
            )
        )
        outputs: list[Tensor] = []
        query = self.layers[0].transform(target)
        for layer_index, layer in enumerate(self.layers):
            transformed_history = (
                None if cache is None else cache.transformed_histories[layer_index]
            )
            if self.activation_checkpoint and self.training:
                if transformed_history is not None:
                    assert cache is not None

                    def cached_layer_forward(
                        current_transformed_history: Tensor,
                        current_valid_mask: Tensor,
                        current_query: Tensor,
                        *,
                        current_layer: STCAInputLayer = layer,
                    ) -> Tensor:
                        return current_layer.attention(
                            current_query,
                            current_transformed_history,
                            current_valid_mask,
                            history_row_indices,
                            request_target_layout=request_target_layout,
                        )

                    layer_output = checkpoint(
                        cached_layer_forward,
                        transformed_history,
                        cache.valid_mask,
                        query,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    assert history is not None and valid_mask is not None

                    def layer_forward(
                        current_history: Tensor,
                        current_valid_mask: Tensor,
                        current_query: Tensor,
                        *,
                        current_layer: STCAInputLayer = layer,
                    ) -> Tensor:
                        return current_layer(
                            current_history,
                            current_valid_mask,
                            current_query,
                            history_row_indices,
                            request_target_layout,
                        )

                    layer_output = checkpoint(
                        layer_forward,
                        history,
                        valid_mask,
                        query,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
            else:
                if transformed_history is not None:
                    layer_output = layer.attention(
                        query,
                        transformed_history,
                        cache.valid_mask,
                        history_row_indices,
                        request_target_layout=request_target_layout,
                    )
                else:
                    assert history is not None and valid_mask is not None
                    layer_output = layer(
                        history,
                        valid_mask,
                        query,
                        history_row_indices,
                        request_target_layout,
                    )
            outputs.append(layer_output)
            if layer_index + 1 < self.num_layers:
                fused = self.query_fusion_projections[layer_index](
                    torch.cat([*outputs, target], dim=-1)
                )
                query = self.layers[layer_index + 1].transform(fused)

        final_input = self.final_projection(torch.cat([*outputs, target], dim=-1))
        return self.final_ffn(final_input)


__all__ = [
    "STCAInputLayer",
    "STCASequenceCache",
    "STCASequenceEncoder",
    "SingleQueryTargetAttention",
    "SwiGLUFFN",
]
