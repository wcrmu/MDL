"""M-FALCON candidate isolation, request caching, and microbatch inference.

The HSTU paper's M-FALCON algorithm has two independent pieces:

* candidates appended to one request share an effective position/time, while
  each candidate can attend only to the request history and itself; and
* request-only work is computed once and reused while the candidate axis is
  evaluated in bounded microbatches.

The models in this repository already represent candidates on the batch axis
and expose request-level sequence caches.  Batch isolation is exactly the
paper's target-isolating attention mask, so :class:`MFalconInferenceEngine`
adds the missing serving orchestration without changing trained numerics.
The explicit layout helpers remain useful to attention implementations that
physically append candidates to a causal sequence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MFalconAttentionLayout:
    """Paper-aligned visibility and relative-position layout.

    ``allowed_mask[b, q, k]`` has the following contract:

    * valid history queries retain ordinary causal history attention;
    * a valid candidate query sees every valid history key and its own key;
    * history never sees candidates, and candidates never see one another.

    All candidates receive the same ``effective_position_ids`` value.  A
    relative position/time bias built from those IDs is therefore identical
    across candidates for their history keys, as required by M-FALCON.
    """

    allowed_mask: Tensor
    valid_mask: Tensor
    effective_position_ids: Tensor
    relative_position_offsets: Tensor
    effective_timestamps: Tensor | None
    relative_time_offsets: Tensor | None
    history_token_count: int
    candidate_token_count: int

    @property
    def candidate_slice(self) -> slice:
        return slice(
            self.history_token_count,
            self.history_token_count + self.candidate_token_count,
        )


def build_mfalcon_attention_layout(
    history_valid_mask: Tensor,
    candidate_valid_mask: Tensor | int,
    *,
    history_timestamps: Tensor | None = None,
    query_timestamp: Tensor | None = None,
) -> MFalconAttentionLayout:
    """Build the exact M-FALCON multi-target mask and position layout.

    Args:
        history_valid_mask: Boolean ``[requests, history_length]`` mask.  Both
            left- and right-padded histories are accepted.
        candidate_valid_mask: Boolean ``[requests, candidates]`` mask or an
            integer candidate capacity (all candidates are then valid).
        history_timestamps: Optional ``[requests, history_length]`` event times.
        query_timestamp: Optional ``[requests]`` or ``[requests, 1]`` query
            time. M-FALCON repeats this same time for every candidate so their
            target-to-history relative time biases are identical.
    """

    if history_valid_mask.ndim != 2:
        raise ValueError("history_valid_mask must have shape [requests, history]")
    if history_valid_mask.dtype != torch.bool:
        raise TypeError("history_valid_mask must be boolean")
    requests, history_count = history_valid_mask.shape
    if isinstance(candidate_valid_mask, int):
        if candidate_valid_mask < 0:
            raise ValueError("candidate count must be non-negative")
        candidate_valid = torch.ones(
            requests,
            candidate_valid_mask,
            dtype=torch.bool,
            device=history_valid_mask.device,
        )
    else:
        candidate_valid = candidate_valid_mask
        if candidate_valid.ndim != 2:
            raise ValueError(
                "candidate_valid_mask must have shape [requests, candidates]"
            )
        if candidate_valid.dtype != torch.bool:
            raise TypeError("candidate_valid_mask must be boolean")
        if candidate_valid.size(0) != requests:
            raise ValueError(
                "history and candidate validity masks must have the same request axis"
            )
        if candidate_valid.device != history_valid_mask.device:
            candidate_valid = candidate_valid.to(history_valid_mask.device)

    candidate_count = int(candidate_valid.size(1))
    total_count = history_count + candidate_count
    allowed = torch.zeros(
        requests,
        total_count,
        total_count,
        dtype=torch.bool,
        device=history_valid_mask.device,
    )

    if history_count:
        physical = torch.arange(history_count, device=history_valid_mask.device)
        causal = physical.view(1, -1, 1) >= physical.view(1, 1, -1)
        allowed[:, :history_count, :history_count] = (
            causal & history_valid_mask.unsqueeze(1) & history_valid_mask.unsqueeze(2)
        )

    if candidate_count:
        # Every candidate reads the same history.  Candidate validity gates the
        # query; history validity gates keys.
        allowed[:, history_count:, :history_count] = candidate_valid.unsqueeze(
            -1
        ) & history_valid_mask.unsqueeze(1)
        candidate_identity = torch.eye(
            candidate_count,
            dtype=torch.bool,
            device=history_valid_mask.device,
        ).unsqueeze(0)
        allowed[:, history_count:, history_count:] = (
            candidate_identity
            & candidate_valid.unsqueeze(-1)
            & candidate_valid.unsqueeze(1)
        )

    # Logical history positions ignore padding.  All candidate positions are
    # clamped to the history length, which is the construction used by the HSTU
    # reference mask for multiple targets.
    history_positions = history_valid_mask.long().cumsum(dim=1) - 1
    history_positions = torch.where(
        history_valid_mask,
        history_positions,
        torch.full_like(history_positions, -1),
    )
    target_position = history_valid_mask.sum(dim=1, keepdim=True)
    candidate_positions = target_position.expand(-1, candidate_count)
    candidate_positions = torch.where(
        candidate_valid,
        candidate_positions,
        torch.full_like(candidate_positions, -1),
    )
    positions = torch.cat([history_positions, candidate_positions], dim=1)
    relative = positions.unsqueeze(2) - positions.unsqueeze(1)
    valid = torch.cat([history_valid_mask, candidate_valid], dim=1)

    if (history_timestamps is None) != (query_timestamp is None):
        raise ValueError(
            "history_timestamps and query_timestamp must be provided together"
        )
    effective_timestamps: Tensor | None = None
    relative_time_offsets: Tensor | None = None
    if history_timestamps is not None and query_timestamp is not None:
        if history_timestamps.shape != history_valid_mask.shape:
            raise ValueError(
                "history_timestamps must have the same shape as history_valid_mask"
            )
        if history_timestamps.device != history_valid_mask.device:
            raise ValueError(
                "history_timestamps and history_valid_mask must share a device"
            )
        if query_timestamp.device != history_valid_mask.device:
            query_timestamp = query_timestamp.to(history_valid_mask.device)
        candidate_timestamps = expand_mfalcon_candidate_timestamps(
            query_timestamp,
            candidate_count,
        ).to(dtype=history_timestamps.dtype)
        effective_timestamps = torch.cat(
            [history_timestamps, candidate_timestamps],
            dim=1,
        )
        # [query, key] convention, matching the query_time-history_time delta
        # used for HSTU's timestamp buckets.
        relative_time_offsets = effective_timestamps.unsqueeze(
            2
        ) - effective_timestamps.unsqueeze(1)

    return MFalconAttentionLayout(
        allowed_mask=allowed,
        valid_mask=valid,
        effective_position_ids=positions,
        relative_position_offsets=relative,
        effective_timestamps=effective_timestamps,
        relative_time_offsets=relative_time_offsets,
        history_token_count=history_count,
        candidate_token_count=candidate_count,
    )


def expand_mfalcon_candidate_timestamps(
    query_timestamp: Tensor,
    candidate_count: int,
) -> Tensor:
    """Repeat one request timestamp for every M-FALCON candidate target."""

    if candidate_count < 0:
        raise ValueError("candidate_count must be non-negative")
    if query_timestamp.ndim == 1:
        query_timestamp = query_timestamp.unsqueeze(1)
    if query_timestamp.ndim != 2 or query_timestamp.size(1) != 1:
        raise ValueError("query_timestamp must have shape [requests] or [requests, 1]")
    return query_timestamp.expand(-1, candidate_count)


def _slice_flat_ragged_values(
    values: Tensor,
    lengths: Tensor,
    start: int,
    end: int,
) -> Tensor:
    """Slice a flat CSR-style values tensor by its candidate-row lengths."""

    offsets = torch.nn.functional.pad(lengths.long().cumsum(dim=0), (1, 0))
    value_start = int(offsets[start].item())
    value_end = int(offsets[end].item())
    return values.narrow(0, value_start, value_end - value_start)


def _slice_candidate_mapping(
    value: Mapping[str, Any],
    start: int,
    end: int,
    candidate_count: int,
) -> dict[str, Any]:
    # Request-deduplicated payloads retain their compact request tensors.  Only
    # the candidate->request gather map belongs to the candidate axis.
    row_indices = value.get("row_indices")
    if isinstance(row_indices, Tensor):
        if row_indices.ndim != 1 or row_indices.numel() != candidate_count:
            raise ValueError(
                "indexed feature row_indices must be rank one and match candidates"
            )
        return {
            key: (child[start:end] if key == "row_indices" else child)
            for key, child in value.items()
        }

    lengths = value.get("lengths")
    if isinstance(lengths, Tensor) and lengths.ndim == 1:
        if lengths.numel() != candidate_count:
            # This is request-level data without an explicit row map.  It must
            # not be guessed to be candidate-aligned.
            return dict(value)
        sliced: dict[str, Any] = {}
        total_values = int(lengths.long().sum().item())
        for key, child in value.items():
            if key == "lengths":
                sliced[key] = child[start:end]
            elif (
                key == "values"
                and isinstance(child, Tensor)
                and child.ndim >= 1
                and child.size(0) == total_values
                and (child.ndim == 1 or child.size(0) != candidate_count)
            ):
                sliced[key] = _slice_flat_ragged_values(child, lengths, start, end)
            else:
                sliced[key] = _slice_candidate_tree(
                    child,
                    start,
                    end,
                    candidate_count,
                )
        return sliced

    return {
        key: _slice_candidate_tree(child, start, end, candidate_count)
        for key, child in value.items()
    }


def _slice_candidate_tree(
    value: Any,
    start: int,
    end: int,
    candidate_count: int,
) -> Any:
    """Slice candidate-aligned leaves while retaining request-level leaves."""

    if isinstance(value, Tensor):
        if value.ndim > 0 and value.size(0) == candidate_count:
            return value[start:end]
        return value
    if isinstance(value, Mapping):
        return _slice_candidate_mapping(
            value,
            start,
            end,
            candidate_count,
        )
    if isinstance(value, list) and len(value) == candidate_count:
        return value[start:end]
    if isinstance(value, tuple) and len(value) == candidate_count:
        return value[start:end]
    return value


def slice_mfalcon_candidate_features(
    features: Mapping[str, Any],
    start: int,
    end: int,
    candidate_count: int,
) -> dict[str, Any]:
    """Create one candidate microbatch without duplicating request tensors."""

    if not 0 <= start <= end <= candidate_count:
        raise ValueError("invalid candidate microbatch bounds")
    return {
        name: _slice_candidate_tree(value, start, end, candidate_count)
        for name, value in features.items()
    }


def _unwrap_cache_model(model: nn.Module) -> nn.Module:
    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        wrapped = getattr(current, "module", None)
        if isinstance(wrapped, nn.Module):
            current = wrapped
            continue
        original = getattr(current, "_orig_mod", None)
        if isinstance(original, nn.Module):
            current = original
            continue
        break
    return current


@dataclass(frozen=True)
class MFalconRunStats:
    candidate_count: int
    microbatch_size: int
    microbatch_count: int
    request_cache_built: bool


@dataclass(frozen=True)
class MFalconResult:
    logits: Tensor
    request_cache: Any
    stats: MFalconRunStats


class MFalconInferenceEngine:
    """Score candidate rows in parallel microbatches with one request cache.

    Candidate rows remain in their original order.  Indexed request payloads
    retain their request-sized storage and their sliced ``row_indices`` map,
    avoiding both sequence recomputation and candidate-times-request copies.
    """

    def __init__(
        self,
        model: nn.Module,
        microbatch_size: int,
        *,
        cache_model: nn.Module | None = None,
    ) -> None:
        if microbatch_size <= 0:
            raise ValueError("M-FALCON microbatch_size must be positive")
        self.model = model
        self.cache_model = _unwrap_cache_model(
            model if cache_model is None else cache_model
        )
        self.microbatch_size = int(microbatch_size)

    @torch.no_grad()
    def precompute_request_cache(self, features: Mapping[str, Any]) -> Any:
        builder = getattr(self.cache_model, "precompute_request_cache", None)
        if not callable(builder):
            return None
        return builder(dict(features))

    @torch.no_grad()
    def update_request_cache(
        self,
        features: Mapping[str, Any],
        previous: Any,
    ) -> Any:
        """Increment or rebuild a cache for reuse by a later request."""

        if previous is None:
            return self.precompute_request_cache(features)
        updater = getattr(self.cache_model, "update_request_cache", None)
        if callable(updater):
            return updater(dict(features), previous)
        return self.precompute_request_cache(features)

    @torch.no_grad()
    def run(
        self,
        features: Mapping[str, Any],
        scenario_id: Tensor,
        *,
        request_cache: Any = None,
    ) -> MFalconResult:
        if self.model.training or self.cache_model.training:
            raise RuntimeError("M-FALCON is an inference-only serving path")
        if scenario_id.ndim < 1:
            raise ValueError("scenario_id must have a candidate batch axis")
        candidate_count = int(scenario_id.size(0))
        if candidate_count <= 0:
            raise ValueError("M-FALCON requires at least one candidate")

        built_cache = False
        if request_cache is None:
            request_cache = self.precompute_request_cache(features)
            built_cache = request_cache is not None

        chunks: list[Tensor] = []
        microbatch_count = 0
        for start in range(0, candidate_count, self.microbatch_size):
            end = min(candidate_count, start + self.microbatch_size)
            micro_features = slice_mfalcon_candidate_features(
                features,
                start,
                end,
                candidate_count,
            )
            micro_scenario = scenario_id[start:end]
            if request_cache is None:
                output = self.model(micro_features, micro_scenario)
            else:
                output = self.model(
                    micro_features,
                    micro_scenario,
                    request_cache=request_cache,
                )
            if not isinstance(output, Mapping):
                raise TypeError("model output must be a mapping containing logits")
            logits = output.get("logits")
            if not isinstance(logits, Tensor):
                raise TypeError("model output must contain tensor logits")
            if logits.ndim == 0 or logits.size(0) != end - start:
                raise ValueError(
                    "model logits must preserve the candidate microbatch axis"
                )
            chunks.append(logits)
            microbatch_count += 1

        combined = torch.cat(chunks, dim=0)
        return MFalconResult(
            logits=combined,
            request_cache=request_cache,
            stats=MFalconRunStats(
                candidate_count=candidate_count,
                microbatch_size=self.microbatch_size,
                microbatch_count=microbatch_count,
                request_cache_built=built_cache,
            ),
        )

    def score(
        self,
        features: Mapping[str, Any],
        scenario_id: Tensor,
        *,
        request_cache: Any = None,
    ) -> Tensor:
        return self.run(
            features,
            scenario_id,
            request_cache=request_cache,
        ).logits


def mfalcon_score_candidates(
    model: nn.Module,
    features: Mapping[str, Any],
    scenario_id: Tensor,
    *,
    microbatch_size: int,
    cache_model: nn.Module | None = None,
    request_cache: Any = None,
) -> Tensor:
    """Functional M-FALCON scoring entry point."""

    return MFalconInferenceEngine(
        model,
        microbatch_size,
        cache_model=cache_model,
    ).score(
        features,
        scenario_id,
        request_cache=request_cache,
    )
