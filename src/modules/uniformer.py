"""Paper-aligned UniFormer interaction stack.

UniFormer (Kuaishou, arXiv:2606.27058) splits ranking into a feature-space
interaction module (FIM) and a task-space interaction module (TIM).

This module follows the published equations, with the ambiguities below:

* The paper's default "3-layer" stack does not say how many layers are FIM
  versus TIM. ``fim_layers=2`` and ``tim_layers=1`` is the default here.
* ``d_head = 1280`` is reported without ``G_kv``. Callers pass ``d_model``
  and ``num_heads`` explicitly. Lazy KV uses ``S_kv = 1`` and ``L_kv = 1``.
* The split ratio stays at the paper default ``beta = 0.5``, so every FIM
  layer has the same ``2q`` token width. Pyramid schedules are not applied.
* Personalized fusion is a sigmoid MLP on the mean of the user-side queries.
  The paper names the signal ("activity and interaction features") but does
  not publish the network.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .stca import SwiGLUFFN
from .attention import _sdpa_context
from .ragged import RaggedTokens, pack_masked_sequences, read_ragged_role
from .ranking_utils import (ProjectedCrossAttention, RequestLayout,
                            RankingRMSNorm, token_swiglu)


def user_item_allow_mask(num_ns_tokens: int, num_user_tokens: int) -> Tensor:
    """Bool SDPA mask for the ``2q`` FIM self-attention, True = may attend.

    User-side queries cannot read item-side keys. Item-side queries stay
    dense. Token layout is ``[cross queries || self queries]``, and each half
    is ``[user groups || item groups]`` because ``beta = 0.5``.
    """

    if num_ns_tokens <= 0:
        raise ValueError("num_ns_tokens must be positive")
    if not 0 <= num_user_tokens <= num_ns_tokens:
        raise ValueError("num_user_tokens must lie in [0, num_ns_tokens]")
    side = torch.zeros(num_ns_tokens, dtype=torch.bool)
    side[:num_user_tokens] = True
    side = torch.cat([side, side], dim=0)
    user_query = side[:, None]
    item_key = ~side[None, :]
    return ~(user_query & item_key)


def apply_per_token_ffn(ffns: nn.ModuleList, tokens: Tensor) -> Tensor:
    """Run an independent FFN on every token slice."""

    if tokens.ndim != 3 or tokens.size(1) != len(ffns):
        raise ValueError(
            f"expected [batch, {len(ffns)}, dim] tokens, got {tuple(tokens.shape)}"
        )
    return token_swiglu(ffns, tokens)


class MultiHeadSelfAttention(nn.Module):
    """Bias-free multi-head self-attention. ``allow_mask`` uses True = keep."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attention_backend = "auto"
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, tokens: Tensor, allow_mask: Tensor | None = None,
                user_positions: Tensor | None = None, item_positions: Tensor | None = None) -> Tensor:
        batch, length, dim = tokens.shape
        if dim != self.d_model:
            raise ValueError(f"expected trailing dim {self.d_model}, got {dim}")
        query = self._split(self.q_proj(tokens))
        key = self._split(self.k_proj(tokens))
        value = self._split(self.v_proj(tokens))
        attn_mask = _broadcast_allow_mask(allow_mask)
        with _sdpa_context(self.attention_backend):
            if user_positions is None:
                context = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
            else:
                # The UI mask is exactly two dense attention problems. Avoid
                # arbitrary-mask fallback while retaining identical visibility.
                user = F.scaled_dot_product_attention(
                    query.index_select(2, user_positions), key.index_select(2, user_positions),
                    value.index_select(2, user_positions))
                item = F.scaled_dot_product_attention(query.index_select(2, item_positions), key, value)
                context = torch.zeros_like(query).index_copy(2, user_positions, user)
                context = context.index_copy(2, item_positions, item)
        return self.out_proj(self._merge(context, batch, length))

    def _split(self, projected: Tensor) -> Tensor:
        batch, length, _ = projected.shape
        return projected.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, context: Tensor, batch: int, length: int) -> Tensor:
        return (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, length, self.d_model)
        )


class QueryCrossAttention(ProjectedCrossAttention):
    """Request-shared attention with strict backend dispatch."""

    def read_role(
        self,
        query: Tensor,
        memory: RaggedTokens,
        role_id: int,
        layout: RequestLayout | None = None,
    ) -> Tensor:
        context = read_ragged_role(
            self.q_proj(query),
            memory,
            role_id,
            self.num_heads,
            self.attention_backend,
            layout,
        )
        return self.out_proj(context)


class LazySequenceMemory(nn.Module):
    """Tokenize each behavior sequence once and share KV across FIM layers.

    Eq. (3)–(4) with ``S_kv = 1`` and ``L_kv = 1``: SwiGLU into ``d_model``,
    then ``k = RMSNorm(z)`` and ``v = k``.
    """

    def __init__(
        self,
        d_model: int,
        num_sequences: int,
        *,
        ffn_expansion: int,
    ) -> None:
        super().__init__()
        if num_sequences <= 0:
            raise ValueError("num_sequences must be positive")
        self.d_model = d_model
        self.num_sequences = num_sequences
        self.ffn = nn.ModuleList(
            [SwiGLUFFN(d_model, expansion_ratio=ffn_expansion) for _ in range(num_sequences)]
        )
        self.k_norm = nn.ModuleList([RankingRMSNorm(d_model) for _ in range(num_sequences)])

    def forward(self, tokens: RaggedTokens) -> RaggedTokens:
        """SwiGLU each role on its real tokens. K and V stay this same tensor."""

        if tokens.role_lengths.shape[1] != self.num_sequences:
            raise ValueError(
                f"expected {self.num_sequences} roles, got {tokens.role_lengths.shape[1]}"
            )
        if tokens.values.numel() and tokens.values.size(-1) != self.d_model:
            raise ValueError(
                f"ragged tokens must end in dim {self.d_model}, got {tokens.values.size(-1)}"
            )
        keys = tokens.values.new_zeros(tokens.values.shape)
        anchor = tokens.values.sum() * 0
        for index, feedforward in enumerate(self.ffn):
            # An empty role still has to touch its parameters. DDP runs with
            # find_unused_parameters=False.
            unused = self.k_norm[index](feedforward(tokens.values.new_zeros(1, self.d_model)))
            anchor = anchor + unused.sum() * 0
            selected = tokens.role == index
            if tokens.role.numel() and bool(selected.any()):
                hidden = self.k_norm[index](feedforward(tokens.values[selected]))
                keys[selected] = hidden
        return RaggedTokens(
            keys + anchor,
            tokens.batch_lengths,
            tokens.batch_lengths_added,
            tokens.role,
            tokens.role_lengths,
        )


class TargetAttentionTokenizer(nn.Module):
    """Aggregate an item-dependent sequence into one NS token, Eq. (5)."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.attention = QueryCrossAttention(d_model, num_heads)

    def forward(
        self,
        target: Tensor,
        key: Tensor,
        value: Tensor,
        key_mask: Tensor | None = None,
    ) -> Tensor:
        if target.ndim != 2:
            raise ValueError(f"target must be [batch, dim], got {tuple(target.shape)}")
        pooled = self.attention(target.unsqueeze(1), key, value, key_mask)
        return pooled.squeeze(1)


class GroupedSwiGLUTokenizer(nn.Module):
    """Project each semantic group to one ``d_model`` token, Section 4.2."""

    def __init__(
        self,
        group_input_dims: list[int] | tuple[int, ...],
        d_model: int,
        *,
        ffn_expansion: int = 2,
    ) -> None:
        super().__init__()
        if not group_input_dims:
            raise ValueError("at least one group is required")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.d_model = d_model
        self.projections = nn.ModuleList(
            [nn.Linear(input_dim, d_model, bias=False) for input_dim in group_input_dims]
        )
        self.ffn = nn.ModuleList(
            [SwiGLUFFN(d_model, expansion_ratio=ffn_expansion) for _ in group_input_dims]
        )

    def forward(self, groups: list[Tensor]) -> Tensor:
        if len(groups) != len(self.projections):
            raise ValueError(
                f"expected {len(self.projections)} groups, got {len(groups)}"
            )
        tokens = []
        for index, group in enumerate(groups):
            if group.ndim != 2 or group.size(-1) != self.projections[index].in_features:
                raise ValueError(
                    f"group {index} must have shape "
                    f"[batch, {self.projections[index].in_features}], "
                    f"got {tuple(group.shape)}"
                )
            tokens.append(self.ffn[index](self.projections[index](group)))
        return torch.stack(tokens, dim=1)


class FeatureInteractionLayer(nn.Module):
    """One FIM layer: multi-sequence cross-attention, fusion, then self-attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_ns_tokens: int,
        num_user_tokens: int,
        num_sequences: int,
        *,
        ffn_expansion: int,
        fusion: str,
        user_item_decouple: bool,
    ) -> None:
        super().__init__()
        if fusion not in {"global", "personalized"}:
            raise ValueError("fusion must be 'global' or 'personalized'")
        self.num_ns_tokens = num_ns_tokens
        self.num_user_tokens = num_user_tokens
        self.num_sequences = num_sequences
        self.fusion = fusion
        self.user_item_decouple = user_item_decouple
        self.cross_norm = nn.ModuleList(
            [RankingRMSNorm(d_model) for _ in range(num_sequences)]
        )
        self.cross_attn = nn.ModuleList(
            [QueryCrossAttention(d_model, num_heads) for _ in range(num_sequences)]
        )
        self.s_ffn = nn.ModuleList(
            [SwiGLUFFN(d_model, expansion_ratio=ffn_expansion) for _ in range(num_sequences)]
        )
        if num_sequences == 2 and fusion == "global":
            self.alpha_logit = nn.Parameter(torch.zeros(()))
        elif fusion == "global":
            self.alpha_logit = nn.Parameter(torch.zeros(num_sequences))
        else:
            personal_out = 1 if num_sequences == 2 else num_sequences
            self.personal = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, personal_out),
            )
        self.self_norm = RankingRMSNorm(d_model)
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        width = 2 * num_ns_tokens
        self.ns_ffn = nn.ModuleList(
            [SwiGLUFFN(d_model, expansion_ratio=ffn_expansion) for _ in range(width)]
        )
        if user_item_decouple:
            mask = user_item_allow_mask(num_ns_tokens, num_user_tokens)
            self.register_buffer("allow_mask", mask, persistent=False)
        else:
            self.allow_mask = None
        positions = torch.arange(width)
        user = positions.remainder(num_ns_tokens) < num_user_tokens
        self.register_buffer("user_positions", positions[user], persistent=False)
        self.register_buffer("item_positions", positions[~user], persistent=False)

    def forward(
        self,
        query_cross: Tensor,
        query_self: Tensor,
        memory: RaggedTokens,
        layout: RequestLayout | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if query_cross.shape != query_self.shape:
            raise ValueError("cross and self queries must share a shape inside FIM")
        if query_cross.size(1) != self.num_ns_tokens:
            raise ValueError(
                f"expected {self.num_ns_tokens} query tokens, got {query_cross.size(1)}"
            )
        sequence_states = []
        for index in range(self.num_sequences):
            attended = self.cross_attn[index].read_role(
                self.cross_norm[index](query_cross),
                memory,
                index,
                layout,
            )
            attended = attended + query_cross
            sequence_states.append(self.s_ffn[index](attended) + attended)
        fused = self._fuse(sequence_states, query_cross)
        combined = torch.cat([fused, query_self], dim=1)
        split_ui = self.user_item_decouple and 0 < self.num_user_tokens < self.num_ns_tokens
        mixed = self.self_attn(self.self_norm(combined), self.allow_mask,
                              self.user_positions if split_ui else None,
                              self.item_positions if split_ui else None) + combined
        feature = self.feature_ffn(mixed)
        half = feature.size(1) // 2
        return feature, feature[:, :half], feature[:, half:]

    def feature_ffn(self, tokens: Tensor) -> Tensor:
        return apply_per_token_ffn(self.ns_ffn, tokens) + tokens

    def _fuse(self, states: list[Tensor], query_cross: Tensor) -> Tensor:
        if self.fusion == "global" and self.num_sequences == 2:
            alpha = torch.sigmoid(self.alpha_logit).to(dtype=states[0].dtype)
            return alpha * states[0] + (1 - alpha) * states[1]
        if self.fusion == "global":
            weights = torch.softmax(self.alpha_logit, dim=0).to(dtype=states[0].dtype)
            fused = states[0] * weights[0]
            for index in range(1, self.num_sequences):
                fused = fused + states[index] * weights[index]
            return fused
        if self.num_user_tokens > 0:
            summary = query_cross[:, : self.num_user_tokens].mean(dim=1)
        else:
            summary = query_cross.mean(dim=1)
        if self.num_sequences == 2:
            alpha = torch.sigmoid(self.personal(summary)).to(dtype=states[0].dtype)
            return alpha[:, None] * states[0] + (1 - alpha)[:, None] * states[1]
        weights = torch.softmax(self.personal(summary), dim=-1).to(dtype=states[0].dtype)
        fused = states[0] * weights[:, 0, None, None]
        for index in range(1, self.num_sequences):
            fused = fused + states[index] * weights[:, index, None, None]
        return fused


class TaskInteractionLayer(nn.Module):
    """One TIM layer: task queries read the final FIM state, then mix tasks."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_tasks: int,
        *,
        ffn_expansion: int,
    ) -> None:
        super().__init__()
        self.cross_norm = RankingRMSNorm(d_model)
        self.cross_attn = QueryCrossAttention(d_model, num_heads)
        self.self_norm = RankingRMSNorm(d_model)
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.t_ffn = nn.ModuleList(
            [SwiGLUFFN(d_model, expansion_ratio=ffn_expansion) for _ in range(num_tasks)]
        )

    def forward(self, task_tokens: Tensor, feature_tokens: Tensor) -> Tensor:
        cross = self.cross_attn(
            self.cross_norm(task_tokens),
            feature_tokens,
            feature_tokens,
        )
        cross = cross + task_tokens
        mixed = self.self_attn(self.self_norm(cross)) + cross
        return self.task_ffn(mixed)

    def task_ffn(self, tokens: Tensor) -> Tensor:
        return apply_per_token_ffn(self.t_ffn, tokens) + tokens


@dataclass(frozen=True)
class UniFormerOutput:
    """Task probabilities and the states that produced them."""

    probabilities: Tensor
    logits: Tensor
    feature_tokens: Tensor
    task_tokens: Tensor


class UniFormerRanker(nn.Module):
    """FIM stack followed by a TIM stack and per-task sigmoid heads.

    Inputs are already projected to ``d_model``:

    * ``ns_tokens``: ``[batch, q, d_model]``, user groups first;
    * ``task_tokens``: ``[batch, tasks, d_model]``;
    * ``sequences``: one ``[batch, length, d_model]`` tensor per behavior stream.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_ns_tokens: int,
        num_tasks: int,
        num_sequences: int,
        *,
        num_user_tokens: int | None = None,
        fim_layers: int = 2,
        tim_layers: int = 1,
        ffn_expansion: int = 2,
        fusion: str = "global",
        user_item_decouple: bool = True,
        attention_backend: str = "auto",
    ) -> None:
        super().__init__()
        if fim_layers <= 0 or tim_layers <= 0:
            raise ValueError("fim_layers and tim_layers must be positive")
        if num_tasks <= 0 or num_ns_tokens <= 0:
            raise ValueError("token and task counts must be positive")
        if num_user_tokens is None:
            num_user_tokens = num_ns_tokens // 2
        self.d_model = d_model
        self.num_ns_tokens = num_ns_tokens
        self.num_user_tokens = num_user_tokens
        self.num_tasks = num_tasks
        self.num_sequences = num_sequences
        self.memory = LazySequenceMemory(
            d_model,
            num_sequences,
            ffn_expansion=ffn_expansion,
        )
        self.fim = nn.ModuleList(
            [
                FeatureInteractionLayer(
                    d_model,
                    num_heads,
                    num_ns_tokens,
                    num_user_tokens,
                    num_sequences,
                    ffn_expansion=ffn_expansion,
                    fusion=fusion,
                    user_item_decouple=user_item_decouple,
                )
                for _ in range(fim_layers)
            ]
        )
        self.tim = nn.ModuleList(
            [
                TaskInteractionLayer(
                    d_model,
                    num_heads,
                    num_tasks,
                    ffn_expansion=ffn_expansion,
                )
                for _ in range(tim_layers)
            ]
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, 1),
                )
                for _ in range(num_tasks)
            ]
        )
        for module in self.modules():
            if isinstance(module, (ProjectedCrossAttention, MultiHeadSelfAttention)):
                module.attention_backend = attention_backend

    def forward(
        self,
        ns_tokens: Tensor,
        task_tokens: Tensor,
        sequences: list[Tensor] | RaggedTokens,
        sequence_masks: list[Tensor | None] | None = None,
        request_index: Tensor | None = None,
    ) -> UniFormerOutput:
        ragged = self._as_ragged(sequences, sequence_masks)
        self._check_inputs(ns_tokens, task_tokens, ragged, request_index)
        memory = self.memory(ragged)
        requests = int(ragged.batch_lengths.shape[0])
        layout = None if request_index is None else RequestLayout.build(
            request_index.to(device=ns_tokens.device, dtype=torch.long), requests)
        query_cross = ns_tokens
        query_self = ns_tokens
        feature = ns_tokens
        for layer in self.fim:
            feature, query_cross, query_self = layer(
                query_cross,
                query_self,
                memory,
                layout,
            )
        tasks = task_tokens
        for layer in self.tim:
            tasks = layer(tasks, feature)
        logits = torch.cat(
            [head(tasks[:, index]) for index, head in enumerate(self.heads)],
            dim=1,
        )
        return UniFormerOutput(
            probabilities=torch.sigmoid(logits),
            logits=logits,
            feature_tokens=feature,
            task_tokens=tasks,
        )

    def _as_ragged(
        self,
        sequences: list[Tensor] | RaggedTokens,
        sequence_masks: list[Tensor | None] | None,
    ) -> RaggedTokens:
        if isinstance(sequences, RaggedTokens):
            if sequence_masks is not None:
                raise ValueError("ragged tokens already dropped padding; do not pass masks")
            return sequences
        masks = [None] * len(sequences) if sequence_masks is None else list(sequence_masks)
        if len(masks) != len(sequences):
            raise ValueError("sequence_masks must align with sequences")
        return pack_masked_sequences(sequences, masks)

    def _check_inputs(
        self,
        ns_tokens: Tensor,
        task_tokens: Tensor,
        sequences: RaggedTokens,
        request_index: Tensor | None,
    ) -> None:
        batch = ns_tokens.size(0)
        expected_ns = (batch, self.num_ns_tokens, self.d_model)
        expected_tasks = (batch, self.num_tasks, self.d_model)
        if tuple(ns_tokens.shape) != expected_ns:
            raise ValueError(f"ns_tokens must have shape {expected_ns}, got {tuple(ns_tokens.shape)}")
        if tuple(task_tokens.shape) != expected_tasks:
            raise ValueError(
                f"task_tokens must have shape {expected_tasks}, got {tuple(task_tokens.shape)}"
            )
        if sequences.role_lengths.shape[1] != self.num_sequences:
            raise ValueError(
                f"expected {self.num_sequences} roles, got {sequences.role_lengths.shape[1]}"
            )
        sequence_batch = int(sequences.batch_lengths.shape[0])
        if request_index is None:
            if sequence_batch != batch:
                raise ValueError(
                    "sequence batch must match ns_tokens unless request_index is set, "
                    f"got {sequence_batch} vs {batch}"
                )
        elif request_index.ndim != 1 or request_index.numel() != batch:
            raise ValueError("request_index must have one entry per candidate")
        if sequences.values.numel() and sequences.values.size(-1) != self.d_model:
            raise ValueError(
                f"ragged tokens must end in dim {self.d_model}, got {sequences.values.size(-1)}"
            )


def _safe_key_mask(key_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Keep a fully padded row defined, then let the caller zero that row."""

    valid_any = key_mask.any(dim=-1)
    if key_mask.size(-1) == 0:
        return key_mask[:, None, None, :], valid_any
    safe = key_mask.clone()
    safe[~valid_any, 0] = True
    return safe[:, None, None, :], valid_any


def _broadcast_allow_mask(allow_mask: Tensor | None) -> Tensor | None:
    if allow_mask is None:
        return None
    if allow_mask.dtype != torch.bool:
        raise ValueError("allow_mask must be bool with True = may attend")
    if allow_mask.ndim == 2:
        return allow_mask[None, None]
    if allow_mask.ndim == 4:
        return allow_mask
    raise ValueError(
        f"allow_mask must be [query, key] or [batch, heads, query, key], got {tuple(allow_mask.shape)}"
    )


def weighted_bce_with_logits(
    logits: Tensor,
    labels: Tensor,
    weights: Tensor | None = None,
) -> Tensor:
    """Eq. (13): weighted sum of per-task binary cross-entropy."""

    if logits.shape != labels.shape:
        raise ValueError(f"logits and labels must match, got {tuple(logits.shape)} vs {tuple(labels.shape)}")
    per_task = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=0)
    if weights is None:
        return per_task.mean()
    if weights.ndim != 1 or weights.numel() != per_task.numel():
        raise ValueError("weights must be one scalar per task")
    weight = weights.to(device=per_task.device, dtype=per_task.dtype)
    return (per_task * weight).sum() / weight.sum().clamp_min(1e-6)
