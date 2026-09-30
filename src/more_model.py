"""MORE on the current feature pack.

This file is only the MORE model. UniFormer lives in ``uniformer_model.py`` and
does not share this tokenizer or these projectors.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .model import FeatureEncoderBank, OneTransTokenizer, _consumed_scalar_feature_names
from .modules.more import MORERanker


class _EvenHeadProjector(nn.Module):
    """Concatenate one axis and project each equal slice to ``token_dim``."""

    def __init__(
        self,
        input_names: tuple[str, ...],
        input_dims: dict[str, int],
        num_heads: int,
        head_dim: int,
        init_std: float,
    ) -> None:
        super().__init__()
        if not input_names or num_heads <= 0 or head_dim <= 0:
            raise ValueError("MORE head projector needs inputs, heads, and dim")
        self.input_names = input_names
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.input_dim = sum(input_dims[name] for name in input_names)
        if self.input_dim % num_heads != 0:
            raise ValueError(
                "MORE axis width must divide evenly across heads: "
                f"{self.input_dim} % {num_heads} != 0"
            )
        self.slice_dim = self.input_dim // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, head_dim, self.slice_dim))
        nn.init.normal_(self.weight, std=init_std)

    def forward(self, encoded: dict[str, Tensor]) -> Tensor:
        packed = torch.cat([encoded[name] for name in self.input_names], dim=1)
        sliced = packed.view(packed.size(0), self.num_heads, self.slice_dim)
        return torch.bmm(sliced.transpose(0, 1), self.weight.transpose(1, 2)).transpose(0, 1)


class MORETokenizer(OneTransTokenizer):
    """Non-sequential tokens plus one timestamp-fused behavior sequence."""

    def __init__(self, config: Any, encoder_bank: FeatureEncoderBank) -> None:
        super().__init__(config, encoder_bank)
        resolved = config.resolved
        user_names = tuple(resolved.mixformer_user_feature_inputs)
        item_names = tuple(resolved.mixformer_item_feature_inputs)
        user_heads = resolved.mixformer_user_head_count
        if not user_names or not item_names or user_heads is None:
            raise ValueError("MORE requires the resolved user/item feature split")
        feature_heads = int(resolved.tokenization.feature_token_count)
        self.user_head_count = int(user_heads)
        self.item_head_count = feature_heads - self.user_head_count
        self.feature_head_count = feature_heads
        if self.auto_ns_projection is not None:
            del self.auto_ns_projection
        self.auto_ns_projection = None
        dim = config.model.token_dim
        init_std = config.model.init_std
        # Positions are assigned after timestamp fusion, never to padding slots.
        self.position_capacity = (config.model.global_sequence_max_length
                                  or config.model.max_position_embeddings)
        if self.position_capacity is None:
            raise ValueError("MORE requires a finite position embedding capacity")
        self.position_embedding = nn.Embedding(self.position_capacity + 1, dim, padding_idx=0)
        nn.init.normal_(self.position_embedding.weight, std=init_std)
        with torch.no_grad():
            self.position_embedding.weight[0].zero_()
        self.user_projector = _EvenHeadProjector(
            user_names,
            encoder_bank.output_dims,
            self.user_head_count,
            dim,
            init_std,
        )
        self.item_projector = _EvenHeadProjector(
            item_names,
            encoder_bank.output_dims,
            self.item_head_count,
            dim,
            init_std,
        )
        self.ns_input_names = set(user_names) | set(item_names)

    def encode(
        self,
        features: dict[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
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
        cache = self._sequence_token_part(features, preencoded)
        sequence = cache.s_tokens
        mask = cache.s_valid_mask
        if sequence.size(1) == 0:
            sequence = sequence.new_zeros(sequence.size(0), 1, self.token_dim) + sequence.sum() * 0
            mask = torch.zeros(
                sequence.size(0),
                1,
                dtype=torch.bool,
                device=sequence.device,
            )
        sequence = self.add_positions(sequence, mask)
        return user_heads, item_heads, sequence, mask, request_index

    def add_positions(self, sequence: Tensor, mask: Tensor) -> Tensor:
        # The fused sequence is chronological; rank only valid events so left/right
        # padding and another request's history length cannot change the positions.
        positions = mask.long().cumsum(dim=1) * mask.long()
        valid = (positions <= self.position_capacity).all()
        if positions.is_cuda:
            torch._assert_async(valid, "MORE history exceeds position capacity")
        elif not bool(valid):
            raise ValueError("MORE history exceeds position capacity")
        return (sequence + self.position_embedding(positions).to(sequence.dtype)) * mask.unsqueeze(-1)


def _expansion(config: Any) -> int:
    width = int(config.model.token_dim)
    hidden = int(config.model.hidden_dim)
    if hidden % width != 0:
        raise ValueError(
            "MORE requires model.hidden_dim to be a multiple of model.token_dim, "
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
            "MORE request-axis tokens need candidate-to-request row_indices, "
            f"got {tokens.size(0)} rows for {candidate_batch} candidates"
        )
    return tokens.index_select(
        0,
        request_index.to(device=tokens.device, dtype=torch.long),
    )


class MOREModel(nn.Module):
    """Shared and private anchors over one fused sequence and the non-sequential tokens."""

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
        self.tokenizer = MORETokenizer(config, self.encoder_bank)
        self.ranker = MORERanker(
            d_model=config.model.token_dim,
            num_ns_tokens=self.tokenizer.feature_head_count,
            num_shared_anchors=config.model.more_num_shared_anchors,
            num_private_anchors=len(config.task_names),
            num_heads=config.model.num_heads,
            num_blocks=config.model.num_layers,
            ffn_expansion=_expansion(config),
            attention_backend=config.runtime.attention_backend,
        )

    def compile_dense_backbone(self) -> None:
        if getattr(self, "_dense_compiled", False):
            return
        mode = self.config.runtime.compile_mode
        for block in self.ranker.blocks:
            block._mix = torch.compile(block._mix, mode=mode, fullgraph=True)
            block._enhance = torch.compile(block._enhance, mode=mode, fullgraph=True)
        self._dense_compiled = True

    def forward(
        self,
        features: dict[str, Any],
        scenario_id: Tensor,
        request_cache: Any | None = None,
    ) -> dict[str, Tensor]:
        del scenario_id
        if request_cache is not None:
            raise ValueError("MORE does not support cross-forward request caches")
        user_heads, item_heads, sequence, sequence_mask, request_index = (
            self.tokenizer.encode(features)
        )
        candidate_batch = item_heads.size(0)
        user_heads = _align_candidates(user_heads, candidate_batch, request_index)
        nonseq = torch.cat([user_heads, item_heads], dim=1)
        sequence_batch = sequence.size(0)
        if request_index is None and sequence_batch == candidate_batch:
            sequence_index = None
        elif request_index is None or request_index.numel() != candidate_batch:
            raise ValueError(
                "MORE sequence is request-major but the candidate batch has no "
                f"row_indices ({sequence_batch} requests, {candidate_batch} candidates)"
            )
        else:
            sequence_index = request_index
        output = self.ranker(
            sequence,
            nonseq,
            sequence_mask=sequence_mask,
            request_index=sequence_index,
        )
        return {"logits": output.logits}
