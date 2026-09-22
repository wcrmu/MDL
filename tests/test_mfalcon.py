from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.mfalcon import (
    MFalconInferenceEngine,
    build_mfalcon_attention_layout,
    expand_mfalcon_candidate_timestamps,
    slice_mfalcon_candidate_features,
)


def test_attention_layout_isolates_targets_and_shares_their_position() -> None:
    history_valid = torch.tensor(
        [
            [False, True, True],
            [True, False, True],
        ]
    )
    candidate_valid = torch.tensor(
        [
            [True, True],
            [True, False],
        ]
    )

    layout = build_mfalcon_attention_layout(
        history_valid,
        candidate_valid,
        history_timestamps=torch.tensor([[0, 80, 90], [70, 0, 95]]),
        query_timestamp=torch.tensor([100, 110]),
    )

    expected_first = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 0, 1],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(layout.allowed_mask[0], expected_first)
    assert not bool(layout.allowed_mask[:, :3, 3:].any())
    assert not bool(layout.allowed_mask[:, 3, 4].any())
    assert not bool(layout.allowed_mask[:, 4, 3].any())
    torch.testing.assert_close(
        layout.effective_position_ids[0],
        torch.tensor([-1, 0, 1, 2, 2]),
    )
    torch.testing.assert_close(
        layout.effective_position_ids[1],
        torch.tensor([0, -1, 1, 2, -1]),
    )
    # Candidate-to-history relative offsets are identical for every valid
    # target because M-FALCON clamps all targets to one logical position.
    torch.testing.assert_close(
        layout.relative_position_offsets[0, 3, :3],
        layout.relative_position_offsets[0, 4, :3],
    )
    assert layout.effective_timestamps is not None
    assert layout.relative_time_offsets is not None
    torch.testing.assert_close(
        layout.effective_timestamps[0],
        torch.tensor([0, 80, 90, 100, 100]),
    )
    torch.testing.assert_close(
        layout.relative_time_offsets[0, 3, :3],
        torch.tensor([100, 20, 10]),
    )
    torch.testing.assert_close(
        layout.relative_time_offsets[0, 3, :3],
        layout.relative_time_offsets[0, 4, :3],
    )


def test_timestamp_expansion_assigns_one_request_time_to_all_targets() -> None:
    timestamps = torch.tensor([101, 205])
    expanded = expand_mfalcon_candidate_timestamps(timestamps, 3)
    torch.testing.assert_close(
        expanded,
        torch.tensor([[101, 101, 101], [205, 205, 205]]),
    )
    assert expanded.stride(1) == 0


def test_candidate_slicing_preserves_request_storage_and_slices_csr_bags() -> None:
    request_values = torch.tensor([100, 200])
    features = {
        "candidate": torch.arange(5),
        "request": {
            "values": request_values,
            "row_indices": torch.tensor([0, 0, 1, 1, 1]),
        },
        "bag": {
            "values": torch.tensor([10, 11, 12, 13, 14, 15]),
            "lengths": torch.tensor([2, 0, 1, 3, 0]),
        },
        "constant": torch.tensor([9, 8]),
    }

    sliced = slice_mfalcon_candidate_features(features, 2, 4, 5)

    torch.testing.assert_close(sliced["candidate"], torch.tensor([2, 3]))
    assert sliced["request"]["values"].data_ptr() == request_values.data_ptr()
    torch.testing.assert_close(sliced["request"]["row_indices"], torch.tensor([1, 1]))
    torch.testing.assert_close(sliced["bag"]["lengths"], torch.tensor([1, 3]))
    torch.testing.assert_close(sliced["bag"]["values"], torch.tensor([12, 13, 14, 15]))
    # Its leading size does not match the candidate axis, so it remains a
    # request/constant leaf rather than being guessed to be candidate data.
    assert sliced["constant"].data_ptr() == features["constant"].data_ptr()


class _CacheAwareScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cache_builds = 0
        self.cache_updates = 0
        self.forward_calls = 0
        self.cache_ids: list[int] = []

    def precompute_request_cache(self, features: dict[str, object]) -> Tensor:
        self.cache_builds += 1
        request = features["request"]
        assert isinstance(request, dict)
        values = request["values"]
        assert isinstance(values, Tensor)
        return values.float().mul(10)

    def update_request_cache(
        self,
        features: dict[str, object],
        previous: Tensor,
    ) -> Tensor:
        del features
        self.cache_updates += 1
        return previous + 1

    def forward(
        self,
        features: dict[str, object],
        scenario_id: Tensor,
        request_cache: Tensor | None = None,
    ) -> dict[str, Tensor]:
        self.forward_calls += 1
        request = features["request"]
        candidate = features["candidate"]
        assert isinstance(request, dict)
        assert isinstance(candidate, Tensor)
        rows = request["row_indices"]
        values = request["values"]
        assert isinstance(rows, Tensor)
        assert isinstance(values, Tensor)
        cache = values.float().mul(10) if request_cache is None else request_cache
        if request_cache is not None:
            self.cache_ids.append(id(request_cache))
        logits = candidate.float() + cache.index_select(0, rows) + scenario_id.float()
        return {"logits": logits.unsqueeze(1)}


def test_engine_builds_one_cache_and_preserves_full_batch_numerics() -> None:
    model = _CacheAwareScorer().eval()
    features = {
        "candidate": torch.tensor([1, 2, 3, 4, 5, 6, 7]),
        "request": {
            "values": torch.tensor([10, 20, 30]),
            "row_indices": torch.tensor([0, 0, 1, 1, 1, 2, 2]),
        },
    }
    scenarios = torch.tensor([0, 1, 0, 1, 2, 0, 2])
    expected = model(features, scenarios)["logits"]
    model.forward_calls = 0

    result = MFalconInferenceEngine(model, microbatch_size=3).run(
        features,
        scenarios,
    )

    torch.testing.assert_close(result.logits, expected)
    assert model.cache_builds == 1
    assert model.forward_calls == math.ceil(7 / 3)
    assert len(set(model.cache_ids)) == 1
    assert result.stats.request_cache_built
    assert result.stats.candidate_count == 7
    assert result.stats.microbatch_count == 3


def test_engine_rejects_training_mode() -> None:
    model = _CacheAwareScorer().train()
    engine = MFalconInferenceEngine(model, microbatch_size=2)
    try:
        engine.run(
            {
                "candidate": torch.tensor([1]),
                "request": {
                    "values": torch.tensor([2]),
                    "row_indices": torch.tensor([0]),
                },
            },
            torch.tensor([0]),
        )
    except RuntimeError as error:
        assert "inference-only" in str(error)
    else:
        raise AssertionError("training-mode M-FALCON should fail")


def test_engine_exposes_cross_request_incremental_cache_update() -> None:
    model = _CacheAwareScorer().eval()
    engine = MFalconInferenceEngine(model, microbatch_size=2)
    features = {
        "candidate": torch.tensor([1]),
        "request": {
            "values": torch.tensor([2]),
            "row_indices": torch.tensor([0]),
        },
    }
    initial = engine.precompute_request_cache(features)
    updated = engine.update_request_cache(features, initial)

    assert isinstance(initial, Tensor) and isinstance(updated, Tensor)
    torch.testing.assert_close(updated, initial + 1)
    assert model.cache_builds == 1
    assert model.cache_updates == 1
