"""UniFormer on the current feature pack.

This file is only the UniFormer model. MORE lives in ``more_model.py`` and does
not share this tokenizer or these projectors.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .model import FeatureEncoderBank, OneTransTokenizer, _consumed_scalar_feature_names
from .modules.ragged import RaggedTokens
from .modules.stca import SwiGLUFFN
from .modules.uniformer import UniFormerRanker
from .modules.uniformer import GroupedSwiGLUTokenizer
from .modules.ranking_utils import token_swiglu
from .config import uniformer_semantic_groups


class _SemanticProjector(nn.Module):
    """Preserve each declared semantic group as one independent SwiGLU token."""

    def __init__(
        self,
        groups,
        input_dims: dict[str, int],
        head_dim: int,
        expansion: int,
    ) -> None:
        super().__init__()
        self.groups = groups
        self.projection = GroupedSwiGLUTokenizer(
            [sum(input_dims[n] for n in g.inputs) for g in groups], head_dim,
            ffn_expansion=expansion)

    def forward(self, encoded: dict[str, Tensor]) -> Tensor:
        return self.projection([torch.cat([encoded[n] for n in g.inputs], dim=-1)
                                for g in self.groups])


class _UniFormerTaskTokens(nn.Module):
    """Per-task id plus the mean of the request-axis tokens."""

    def __init__(self, num_tasks: int, dim: int, expansion: int, init_std: float) -> None:
        super().__init__()
        self.prior = nn.Parameter(torch.empty(num_tasks, dim))
        nn.init.normal_(self.prior, std=init_std)
        self.ffn = nn.ModuleList(
            [SwiGLUFFN(dim, expansion_ratio=expansion) for _ in range(num_tasks)]
        )

    def forward(self, context: Tensor) -> Tensor:
        tokens = context.unsqueeze(1) + self.prior.to(dtype=context.dtype)
        return token_swiglu(self.ffn, tokens)


class UniFormerTokenizer(OneTransTokenizer):
    """Request-axis and candidate-axis queries, plus the shared behavior budget.

    The budget is ``global_sequence_max_length`` events per request. Each
    behavior keeps the newest events inside its own share of that budget.
    Stream identity stays on each event as its role.
    """

    def __init__(self, config: Any, encoder_bank: FeatureEncoderBank) -> None:
        super().__init__(config, encoder_bank)
        resolved = config.resolved
        user_names = tuple(resolved.mixformer_user_feature_inputs)
        item_names = tuple(resolved.mixformer_item_feature_inputs)
        user_heads = resolved.mixformer_user_head_count
        if not user_names or not item_names or user_heads is None:
            raise ValueError("UniFormer requires the resolved user/item feature split")
        feature_heads = int(resolved.tokenization.feature_token_count)
        self.user_head_count = int(user_heads)
        self.item_head_count = feature_heads - self.user_head_count
        self.feature_head_count = feature_heads
        if self.auto_ns_projection is not None:
            del self.auto_ns_projection
        self.auto_ns_projection = None
        dim = config.model.token_dim
        user_groups, item_groups = uniformer_semantic_groups(config, user_names, item_names)
        self.user_projector = _SemanticProjector(user_groups, encoder_bank.output_dims,
                                                dim, _expansion(config))
        self.item_projector = _SemanticProjector(item_groups, encoder_bank.output_dims,
                                                dim, _expansion(config))
        self.ns_input_names = set(user_names) | set(item_names)

    def encode(
        self,
        features: dict[str, Any],
    ) -> tuple[Tensor, Tensor, RaggedTokens, Tensor | None]:
        self.encoder_bank.prepare_gset_batch(features)
        preencoded = self._preencode_inputs(
            features,
            set(self.ns_input_names),
            include_sequences=True,
        )
        encoded = self.encoder_bank.encode_scalar_features(
            features,
            set(self.ns_input_names),
            preencoded,
            expand_request_rows=False,
        )
        user_heads = self.user_projector(encoded)
        item_heads = self.item_projector(encoded)
        request_index = self.request_row_indices(features)
        if request_index is not None:
            request_index = request_index.to(device=item_heads.device, dtype=torch.long)
        if request_index is None and user_heads.size(0) == 1 and item_heads.size(0) != 1:
            request_index = torch.zeros(
                item_heads.size(0),
                dtype=torch.long,
                device=item_heads.device,
            )
        if not self.sequence_groups:
            raise ValueError("UniFormer requires at least one behavior stream")
        window = self._sequence_token_part(features, preencoded)
        if window.sequences is None:
            raise RuntimeError("UniFormer event window was not packed")
        return user_heads, item_heads, window.sequences, request_index


def _expansion(config: Any) -> int:
    width = int(config.model.token_dim)
    hidden = int(config.model.hidden_dim)
    if hidden % width != 0:
        raise ValueError(
            "UniFormer requires model.hidden_dim to be a multiple of model.token_dim, "
            f"got {hidden} and {width}"
        )
    return hidden // width


def _align_candidates(
    tokens: Tensor,
    candidate_batch: int,
    request_index: Tensor | None,
) -> Tensor:
    if request_index is None and tokens.size(0) == candidate_batch:
        return tokens
    if request_index is None:
        raise ValueError(
            "UniFormer request-axis tokens need candidate-to-request row_indices, "
            f"got {tokens.size(0)} rows for {candidate_batch} candidates"
        )
    return tokens.index_select(
        0,
        request_index.to(device=tokens.device, dtype=torch.long),
    )


class UniFormerModel(nn.Module):
    """FIM over per-stream memories, then TIM over the three task labels."""

    def __init__(
        self,
        config: Any,
        vocab_maps: dict[str, dict[str, int]],
        embedding_dim: int | None = None,
        embedding_size_override: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        embedding_dim = (
            config.model.embedding_dim if embedding_dim is None else embedding_dim
        )
        self.encoder_bank = FeatureEncoderBank(
            config,
            vocab_maps,
            embedding_dim,
            build_sequence_summaries=False,
            included_scalar_feature_names=_consumed_scalar_feature_names(config),
            embedding_size_override=embedding_size_override,
        )
        self.tokenizer = UniFormerTokenizer(config, self.encoder_bank)
        expansion = _expansion(config)
        self.task_tokens = _UniFormerTaskTokens(
            len(config.task_names),
            config.model.token_dim,
            expansion,
            config.model.init_std,
        )
        self.ranker = UniFormerRanker(
            d_model=config.model.token_dim,
            num_heads=config.model.num_heads,
            num_ns_tokens=self.tokenizer.feature_head_count,
            num_user_tokens=self.tokenizer.user_head_count,
            num_tasks=len(config.task_names),
            num_sequences=len(self.tokenizer.sequence_groups),
            fim_layers=config.model.num_layers,
            tim_layers=config.model.uniformer_tim_layers,
            ffn_expansion=expansion,
            fusion="global",
            user_item_decouple=True,
            attention_backend=config.runtime.attention_backend,
        )

    def compile_dense_backbone(self) -> None:
        if getattr(self, "_dense_compiled", False):
            return
        mode = self.config.runtime.compile_mode
        for layer in self.ranker.fim:
            layer.feature_ffn = torch.compile(layer.feature_ffn, mode=mode, fullgraph=True)
        for layer in self.ranker.tim:
            layer.task_ffn = torch.compile(layer.task_ffn, mode=mode, fullgraph=True)
        self._dense_compiled = True

    def forward(
        self,
        features: dict[str, Any],
        scenario_id: Tensor,
        request_cache: Any | None = None,
    ) -> dict[str, Tensor]:
        del scenario_id
        if request_cache is not None:
            raise ValueError("UniFormer does not support cross-forward request caches")
        user_heads, item_heads, sequences, request_index = self.tokenizer.encode(features)
        candidate_batch = item_heads.size(0)
        user_heads = _align_candidates(user_heads, candidate_batch, request_index)
        ns_tokens = torch.cat([user_heads, item_heads], dim=1)
        task_tokens = self.task_tokens(user_heads.mean(dim=1))
        sequence_batch = int(sequences.batch_lengths.shape[0])
        if request_index is None and sequence_batch == candidate_batch:
            sequence_index = None
        elif request_index is None or request_index.numel() != candidate_batch:
            raise ValueError(
                "UniFormer streams are request-major but the candidate batch has no "
                f"row_indices ({sequence_batch} requests, {candidate_batch} candidates)"
            )
        else:
            sequence_index = request_index
        output = self.ranker(
            ns_tokens,
            task_tokens,
            sequences,
            request_index=sequence_index,
        )
        return {"logits": output.logits}
