"""Paper-aligned MORE backbone.

MORE (Hello Group / Momo, arXiv:2609.07273) keeps Shared and Private Anchor
Tokens inside every block. Each block reads the behavior sequence, mixes
anchors with non-sequential tokens under a task-boundary mask, then enhances
each Private Anchor with a FiLM residual.

Ambiguities retained from the text:

* The prediction tower is not given as an equation. Each task head receives
  the mean of ``F``, the mean of the Shared Anchors, and its Private Anchor.
* The gate is a per-head token-wise FFN of width ``2 * head_dim`` followed by
  ``2 * sigmoid``, as drawn in Figure 2. Hidden width is an implementation choice.
  Each token's gate sees only that token; masked private tokens cannot affect
  another token through the gate.
* Cartesian action encoding, Eq. (3), is a separate input helper. The ranker
  consumes an already built sequence tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ragged import apply_masked_tokenwise
from .stca import SwiGLUFFN
from .ranking_utils import ProjectedCrossAttention, RequestLayout, SequenceMemory, token_swiglu


def task_boundary_mask(
    num_ns_tokens: int,
    num_shared_anchors: int,
    num_private_anchors: int,
) -> Tensor:
    """True where token ``j`` may contribute to output token ``i``.

    ``F`` and Shared Anchors are visible to every token. A Private Anchor is
    visible only to itself.
    """

    if min(num_ns_tokens, num_shared_anchors, num_private_anchors) <= 0:
        raise ValueError("NS, shared, and private token counts must be positive")
    width = num_ns_tokens + num_shared_anchors + num_private_anchors
    mask = torch.zeros(width, width, dtype=torch.bool)
    visible_prefix = num_ns_tokens + num_shared_anchors
    mask[:, :visible_prefix] = True
    mask.fill_diagonal_(True)
    return mask


def gated_rankmixer(tokens: Tensor, gates: Tensor, mask: Tensor) -> Tensor:
    """RankMixer token mixing with a per-head gate and a task-boundary mask.

    ``gates`` has shape ``[batch, head, input_token]`` or ``[head, input_token]``.
    ``mask`` has shape ``[output_token, input_token]`` and uses True = keep.
    With a unit gate and an all-true mask this is RankMixer TokenMixing.
    """

    if tokens.ndim != 3:
        raise ValueError(f"tokens must be [batch, tokens, dim], got {tuple(tokens.shape)}")
    batch, num_tokens, dim = tokens.shape
    if num_tokens == 0 or dim % num_tokens != 0:
        raise ValueError("token dim must be divisible by the token count")
    if mask.shape != (num_tokens, num_tokens):
        raise ValueError(
            f"mask must have shape {(num_tokens, num_tokens)}, got {tuple(mask.shape)}"
        )
    if gates.ndim == 2:
        gates = gates.unsqueeze(0)
    if gates.shape != (batch, num_tokens, num_tokens):
        raise ValueError(
            f"gates must have shape {(batch, num_tokens, num_tokens)}, got {tuple(gates.shape)}"
        )
    head_dim = dim // num_tokens
    head_major = tokens.view(batch, num_tokens, num_tokens, head_dim).permute(0, 2, 1, 3)
    weights = gates * mask.to(device=gates.device, dtype=gates.dtype)
    mixed = head_major * weights[:, :, :, None]
    return mixed.contiguous().view(batch, num_tokens, dim)


def cartesian_sequence_token(
    item_embedding: Tensor,
    action_embedding: Tensor,
    time_embedding: Tensor,
) -> Tensor:
    """Eq. (3): item + action-combination + time embeddings."""

    if item_embedding.shape != action_embedding.shape or item_embedding.shape != time_embedding.shape:
        raise ValueError("cartesian embeddings must share a shape")
    return item_embedding + action_embedding + time_embedding


def apply_per_token_ffn(ffns: nn.ModuleList, tokens: Tensor) -> Tensor:
    if tokens.ndim != 3 or tokens.size(1) != len(ffns):
        raise ValueError(
            f"expected [batch, {len(ffns)}, dim] tokens, got {tuple(tokens.shape)}"
        )
    return token_swiglu(ffns, tokens)


def _masked_mean(tokens: Tensor, mask: Tensor | None) -> Tensor:
    if tokens.size(1) == 0:
        return tokens.sum(dim=1)
    if mask is None:
        return tokens.mean(dim=1)
    if mask.shape != tokens.shape[:2]:
        raise ValueError(
            f"sequence mask must have shape {tuple(tokens.shape[:2])}, got {tuple(mask.shape)}"
        )
    weight = mask.to(dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


class QueryCrossAttention(ProjectedCrossAttention):
    """Request-shared attention with strict backend dispatch."""


class TokenGate(nn.Module):
    """Per-head token scores in (0, 2), matching Figure 2's 2*sigmoid."""

    def __init__(self, num_tokens: int, head_dim: int) -> None:
        super().__init__()
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(head_dim, 2 * head_dim),
                    nn.SiLU(),
                    nn.Linear(2 * head_dim, 1),
                )
                for _ in range(num_tokens)
            ]
        )

    def forward(self, tokens: Tensor) -> Tensor:
        batch, num_tokens, dim = tokens.shape
        head_dim = dim // num_tokens
        head_major = tokens.view(batch, num_tokens, num_tokens, head_dim).permute(0, 2, 1, 3)
        values = head_major.permute(1, 0, 2, 3).reshape(num_tokens, batch * num_tokens, head_dim)
        first = [mlp[0] for mlp in self.mlps]
        last = [mlp[2] for mlp in self.mlps]
        hidden = torch.bmm(values, torch.stack([m.weight for m in first]).transpose(1, 2))
        hidden = F.silu(hidden + torch.stack([m.bias for m in first])[:, None].to(hidden.dtype))
        scores = torch.bmm(hidden, torch.stack([m.weight for m in last]).transpose(1, 2))
        scores = scores + torch.stack([m.bias for m in last])[:, None].to(scores.dtype)
        return (2 * torch.sigmoid(scores)).reshape(num_tokens, batch, num_tokens).permute(1, 0, 2)


class MOREBlock(nn.Module):
    """Sequence reading, selective semantic mixing, and task enhancing."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_ns_tokens: int,
        num_shared_anchors: int,
        num_private_anchors: int,
        *,
        ffn_expansion: int,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_ns_tokens = num_ns_tokens
        self.num_shared_anchors = num_shared_anchors
        self.num_private_anchors = num_private_anchors
        width = num_ns_tokens + num_shared_anchors + num_private_anchors
        if d_model % width != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by token width {width} "
                "for selective semantic mixing"
            )
        self.width = width
        self.seq_ffn = SwiGLUFFN(d_model, expansion_ratio=ffn_expansion)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.register_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.register_norm = nn.LayerNorm(d_model)
        self.shared_cross = QueryCrossAttention(d_model, num_heads)
        self.private_cross = nn.ModuleList(
            [QueryCrossAttention(d_model, num_heads) for _ in range(num_private_anchors)]
        )
        self.shared_register_ffn = nn.ModuleList(
            [SwiGLUFFN(d_model, expansion_ratio=ffn_expansion) for _ in range(num_shared_anchors)]
        )
        self.private_register_ffn = nn.ModuleList(
            [SwiGLUFFN(d_model, expansion_ratio=ffn_expansion) for _ in range(num_private_anchors)]
        )
        self.token_gate = TokenGate(width, d_model // width)
        mask = task_boundary_mask(num_ns_tokens, num_shared_anchors, num_private_anchors)
        self.register_buffer("boundary_mask", mask, persistent=False)
        self.pffn1 = nn.ModuleList(
            [SwiGLUFFN(d_model, expansion_ratio=ffn_expansion) for _ in range(width)]
        )
        self.pffn2 = nn.ModuleList(
            [SwiGLUFFN(d_model, expansion_ratio=ffn_expansion) for _ in range(width)]
        )
        self.mix_norm1 = nn.LayerNorm(d_model)
        self.mix_norm2 = nn.LayerNorm(d_model)
        flat_dim = num_ns_tokens * d_model
        self.enhance = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(flat_dim, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, 2 * d_model),
                )
                for _ in range(num_private_anchors)
            ]
        )
        self.enhance_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        nonseq: Tensor,
        shared: Tensor,
        private: Tensor,
        sequence: Tensor,
        register: Tensor,
        sequence_mask: Tensor | None,
        request_index: RequestLayout | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        sequence_out, key, value, register = self._encode_sequence(
            sequence, register, sequence_mask
        )
        memory = SequenceMemory(key, value, sequence_mask)
        layout = request_index
        register_b = register if layout is None else register.index_select(0, layout.index)
        shared = self._read_shared(shared, memory, None, register_b, None, layout)
        private = self._read_private(private, memory, None, register_b, None, layout)
        mixed = self._mix(torch.cat([nonseq, shared, private], dim=1))
        nonseq, shared, private_bar = self._split(mixed)
        private = self._enhance(nonseq, private_bar)
        return nonseq, shared, private, sequence_out, register

    def _encode_sequence(
        self,
        sequence: Tensor,
        register: Tensor,
        sequence_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        updated = apply_masked_tokenwise(self.seq_ffn, sequence, sequence_mask)
        key = apply_masked_tokenwise(self.w_k, updated, sequence_mask)
        value = apply_masked_tokenwise(self.w_v, updated, sequence_mask)
        pooled = _masked_mean(updated, sequence_mask)
        register = self.register_norm(register + self.register_mlp(pooled))
        return updated, key, value, register

    def _read_shared(
        self,
        shared: Tensor,
        key: Tensor,
        value: Tensor,
        register: Tensor,
        key_mask: Tensor | None,
        layout: RequestLayout | None = None,
    ) -> Tensor:
        attended = self.shared_cross(shared, key, value, key_mask, layout)
        delta = token_swiglu(self.shared_register_ffn,
                            register[:, None].expand(-1, self.num_shared_anchors, -1))
        return attended + delta

    def _read_private(
        self,
        private: Tensor,
        key: Tensor,
        value: Tensor,
        register: Tensor,
        key_mask: Tensor | None,
        layout: RequestLayout | None = None,
    ) -> Tensor:
        outputs = []
        deltas = token_swiglu(self.private_register_ffn,
                             register[:, None].expand(-1, self.num_private_anchors, -1))
        for index, cross in enumerate(self.private_cross):
            attended = cross(private[:, index : index + 1], key, value, key_mask, layout)
            delta = deltas[:, index]
            outputs.append(attended.squeeze(1) + delta)
        return torch.stack(outputs, dim=1)

    def _mix(self, tokens: Tensor) -> Tensor:
        gates = self.token_gate(tokens)
        mixed = gated_rankmixer(tokens, gates, self.boundary_mask)
        mid = self.mix_norm1(apply_per_token_ffn(self.pffn1, mixed) + tokens)
        return self.mix_norm2(apply_per_token_ffn(self.pffn2, mid) + mid)

    def _split(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        ns_end = self.num_ns_tokens
        shared_end = ns_end + self.num_shared_anchors
        return tokens[:, :ns_end], tokens[:, ns_end:shared_end], tokens[:, shared_end:]

    def _enhance(self, nonseq: Tensor, private: Tensor) -> Tensor:
        flat = nonseq.flatten(1)
        updated = []
        for index, mlp in enumerate(self.enhance):
            scale, shift = mlp(flat).chunk(2, dim=-1)
            residual = private[:, index] * scale + shift
            updated.append(self.enhance_norm(residual + private[:, index]))
        return torch.stack(updated, dim=1)


def _gather_request(
    key: Tensor,
    value: Tensor,
    register: Tensor,
    sequence_mask: Tensor | None,
    request_index: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    if request_index is None:
        return key, value, register, sequence_mask
    index = request_index.to(device=key.device, dtype=torch.long)
    mask = None if sequence_mask is None else sequence_mask.index_select(0, index)
    return (
        key.index_select(0, index),
        value.index_select(0, index),
        register.index_select(0, index),
        mask,
    )


@dataclass(frozen=True)
class MOREOutput:
    """Per-candidate probabilities and the states after the last block."""

    probabilities: Tensor
    logits: Tensor
    nonseq_tokens: Tensor
    shared_anchors: Tensor
    private_anchors: Tensor
    sequence: Tensor


class MORERanker(nn.Module):
    """Anchor construction, stacked MORE blocks, and per-task towers.

    ``sequence`` is ``[requests, length, d_model]``. ``nonseq`` is
    ``[batch, M, d_model]``. Pass ``request_index`` of shape ``[batch]`` when
    several candidates share one request-level sequence.
    """

    def __init__(
        self,
        d_model: int,
        num_ns_tokens: int,
        num_shared_anchors: int,
        num_private_anchors: int,
        *,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_expansion: int = 2,
        attention_backend: str = "auto",
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        width = num_ns_tokens + num_shared_anchors + num_private_anchors
        if d_model % width != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by H={width}"
            )
        self.d_model = d_model
        self.num_ns_tokens = num_ns_tokens
        self.num_shared_anchors = num_shared_anchors
        self.num_private_anchors = num_private_anchors
        anchor_count = num_shared_anchors + num_private_anchors
        anchor_in = d_model + num_ns_tokens * d_model
        anchor_out = anchor_count * d_model
        self.anchor_mlp = nn.Sequential(
            nn.Linear(anchor_in, anchor_out),
            nn.GELU(),
            nn.Linear(anchor_out, anchor_out),
        )
        self.task_prior = nn.Parameter(torch.zeros(num_private_anchors, d_model))
        self.blocks = nn.ModuleList(
            [
                MOREBlock(
                    d_model,
                    num_heads,
                    num_ns_tokens,
                    num_shared_anchors,
                    num_private_anchors,
                    ffn_expansion=ffn_expansion,
                )
                for _ in range(num_blocks)
            ]
        )
        head_in = 3 * d_model
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(head_in, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, 1),
                )
                for _ in range(num_private_anchors)
            ]
        )
        for module in self.modules():
            if isinstance(module, ProjectedCrossAttention):
                module.attention_backend = attention_backend

    def forward(
        self,
        sequence: Tensor,
        nonseq: Tensor,
        *,
        sequence_mask: Tensor | None = None,
        request_index: Tensor | None = None,
    ) -> MOREOutput:
        self._check(sequence, nonseq, sequence_mask, request_index)
        pooled = _masked_mean(sequence, sequence_mask)
        if request_index is not None:
            pooled = pooled.index_select(0, request_index.to(device=pooled.device, dtype=torch.long))
        hidden = torch.cat([pooled, nonseq.flatten(1)], dim=-1)
        anchors = self.anchor_mlp(hidden).view(
            nonseq.size(0),
            self.num_shared_anchors + self.num_private_anchors,
            self.d_model,
        )
        shared = anchors[:, : self.num_shared_anchors]
        private = anchors[:, self.num_shared_anchors :] + self.task_prior.to(dtype=anchors.dtype)
        register = sequence.new_zeros(sequence.size(0), self.d_model)
        layout = None if request_index is None else RequestLayout.build(
            request_index.to(device=sequence.device, dtype=torch.long), sequence.size(0))
        for block in self.blocks:
            nonseq, shared, private, sequence, register = block(
                nonseq,
                shared,
                private,
                sequence,
                register,
                sequence_mask,
                layout,
            )
        shared_summary = torch.cat(
            [nonseq.mean(dim=1), shared.mean(dim=1)],
            dim=-1,
        )
        logits = torch.cat(
            [
                head(torch.cat([shared_summary, private[:, index]], dim=-1))
                for index, head in enumerate(self.heads)
            ],
            dim=1,
        )
        return MOREOutput(
            probabilities=torch.sigmoid(logits),
            logits=logits,
            nonseq_tokens=nonseq,
            shared_anchors=shared,
            private_anchors=private,
            sequence=sequence,
        )

    def _check(
        self,
        sequence: Tensor,
        nonseq: Tensor,
        sequence_mask: Tensor | None,
        request_index: Tensor | None,
    ) -> None:
        if sequence.ndim != 3 or sequence.size(-1) != self.d_model:
            raise ValueError(
                f"sequence must be [requests, length, {self.d_model}], got {tuple(sequence.shape)}"
            )
        if nonseq.ndim != 3 or nonseq.size(1) != self.num_ns_tokens or nonseq.size(-1) != self.d_model:
            raise ValueError(
                "nonseq must be "
                f"[batch, {self.num_ns_tokens}, {self.d_model}], got {tuple(nonseq.shape)}"
            )
        if request_index is None:
            if sequence.size(0) != nonseq.size(0):
                raise ValueError("sequence and nonseq batches must match without request_index")
        elif request_index.ndim != 1 or request_index.numel() != nonseq.size(0):
            raise ValueError("request_index must have one entry per candidate")
        if sequence_mask is not None and sequence_mask.shape != sequence.shape[:2]:
            raise ValueError("sequence_mask must match the sequence batch and length")


def weighted_bce_with_logits(
    logits: Tensor,
    labels: Tensor,
    weights: Tensor | None = None,
) -> Tensor:
    """Eq. (2): weighted sum of per-task binary cross-entropy."""

    if logits.shape != labels.shape:
        raise ValueError("logits and labels must match")
    per_task = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=0)
    if weights is None:
        return per_task.mean()
    weight = weights.to(device=per_task.device, dtype=per_task.dtype)
    return (per_task * weight).sum() / weight.sum().clamp_min(1e-6)
