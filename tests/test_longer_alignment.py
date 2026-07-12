from __future__ import annotations

from dataclasses import replace
import unittest
from typing import Any

import torch
from torch import Tensor, nn

from src.config import SequenceConfig, SequenceFieldConfig
from src.dataloader import _sequence_bounds
from src.model import FeatureEncoderBank, LongerSequenceEncoder, LongerTokenMerger


def _zero_parameters(module: nn.Module) -> None:
    """Turn residual attention blocks into identity mappings."""

    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()


def _compressed_tensor(output: Any) -> Tensor:
    """Normalize supported encoder outputs to one flattened compressed tensor.

    The paper-aligned encoder may expose its compressed token sequence through a
    small output dataclass.  Keeping this normalization in the test lets the
    public contract evolve from the legacy flat Tensor without weakening the
    required [global; recent] content assertion.
    """

    if isinstance(output, Tensor):
        tokens = output
    elif isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        tokens = output[0]
    elif isinstance(output, dict):
        tokens = next(
            (
                output[name]
                for name in ("compressed_tokens", "hidden_tokens", "tokens")
                if isinstance(output.get(name), Tensor)
            ),
            None,
        )
    else:
        tokens = next(
            (
                getattr(output, name)
                for name in ("compressed_tokens", "hidden_tokens", "tokens")
                if isinstance(getattr(output, name, None), Tensor)
            ),
            None,
        )
    if not isinstance(tokens, Tensor):
        raise AssertionError("LONGER output must expose its compressed tokens as a Tensor")
    if tokens.dim() < 2:
        raise AssertionError(f"compressed output must have rank >= 2, got shape {tuple(tokens.shape)}")
    return tokens.flatten(start_dim=1)


class LongerTokenMergerAlignmentTest(unittest.TestCase):
    def test_concat_merge_preserves_kd_width_and_slot_order(self) -> None:
        token_dim = 2
        merge_size = 2
        merger = LongerTokenMerger(
            token_dim=token_dim,
            num_heads=1,
            hidden_dim=8,
            merge_size=merge_size,
            inner_layers=0,
        )
        tokens = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]
        )
        mask = torch.ones(1, 4, dtype=torch.bool)

        merged, merged_mask = merger(tokens, mask)

        expected = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
        self.assertEqual(tuple(merged.shape), (1, 2, merge_size * token_dim))
        torch.testing.assert_close(merged, expected)
        torch.testing.assert_close(merged_mask, torch.ones(1, 2, dtype=torch.bool))

    def test_inner_transformer_preserves_all_k_slots_after_local_interaction(self) -> None:
        token_dim = 2
        merge_size = 2
        merger = LongerTokenMerger(
            token_dim=token_dim,
            num_heads=1,
            hidden_dim=8,
            merge_size=merge_size,
            inner_layers=1,
        )
        # A zero-parameter residual block is an identity.  The merger must then
        # expose all K transformed slots, not mean-pool them back to width d.
        _zero_parameters(merger.inner_blocks)
        tokens = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]
        )
        mask = torch.ones(1, 4, dtype=torch.bool)

        merged, _merged_mask = merger(tokens, mask)

        expected = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
        self.assertEqual(tuple(merged.shape), (1, 2, merge_size * token_dim))
        torch.testing.assert_close(merged, expected)

    def test_inner_transformer_mode_has_no_registered_dead_parameters(self) -> None:
        torch.manual_seed(7)
        merger = LongerTokenMerger(
            token_dim=4,
            num_heads=2,
            hidden_dim=8,
            merge_size=2,
            inner_layers=1,
        )
        tokens = torch.randn(2, 6, 4, requires_grad=True)
        mask = torch.ones(2, 6, dtype=torch.bool)

        merged, _merged_mask = merger(tokens, mask)
        merged.square().sum().backward()

        unused = [
            name
            for name, parameter in merger.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        self.assertEqual(unused, [], f"registered but unused merger parameters: {unused}")


class LongerSequenceEncoderAlignmentTest(unittest.TestCase):
    def _encoder(
        self,
        user_global_tokens: int = 0,
        activation_checkpoint: bool = False,
    ) -> LongerSequenceEncoder:
        return LongerSequenceEncoder(
            token_dim=4,
            num_heads=2,
            hidden_dim=8,
            query_token_count=2,
            self_layers=1,
            summary_tokens=2,
            token_merge=1,
            inner_layers=0,
            user_global_tokens=user_global_tokens,
            activation_checkpoint=activation_checkpoint,
        )

    def test_returns_full_global_and_recent_compressed_sequence(self) -> None:
        encoder = self._encoder().eval()
        # With identity residual blocks the exact compressed sequence must be
        # [global tokens; two most recent sequence tokens].
        _zero_parameters(encoder.cross_block)
        _zero_parameters(encoder.self_blocks)
        tokens = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ]
            ]
        )
        mask = torch.ones(1, 4, dtype=torch.bool)
        global_tokens = torch.tensor(
            [[[101.0, 102.0, 103.0, 104.0], [105.0, 106.0, 107.0, 108.0]]]
        )

        output = encoder(tokens, mask, global_tokens)

        expected = torch.cat([global_tokens, tokens[:, -2:, :]], dim=1).flatten(start_dim=1)
        actual = _compressed_tensor(output)
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        torch.testing.assert_close(actual, expected)

    def test_precomputed_cache_is_numerically_equivalent_and_candidate_reusable(self) -> None:
        torch.manual_seed(11)
        encoder = self._encoder().eval()
        tokens = torch.randn(2, 5, 4)
        mask = torch.tensor(
            [[True, True, True, True, True], [False, False, True, True, True]]
        )
        candidate_globals = [torch.randn(2, 2, 4), torch.randn(2, 2, 4)]

        with torch.no_grad():
            cache = encoder.precompute_cache(tokens, mask)
            for global_tokens in candidate_globals:
                uncached = _compressed_tensor(encoder(tokens, mask, global_tokens))
                cached = _compressed_tensor(encoder(tokens, mask, global_tokens, cache=cache))
                torch.testing.assert_close(cached, uncached, rtol=1.0e-5, atol=1.0e-6)

    def test_single_request_cache_expands_across_candidate_batch(self) -> None:
        torch.manual_seed(23)
        encoder = self._encoder().eval()
        request_tokens = torch.randn(1, 5, 4)
        request_mask = torch.tensor([[False, True, True, True, True]])
        candidate_count = 3
        global_tokens = torch.randn(candidate_count, 2, 4)
        repeated_tokens = request_tokens.expand(candidate_count, -1, -1)
        repeated_mask = request_mask.expand(candidate_count, -1)

        with torch.no_grad():
            cache = encoder.precompute_cache(request_tokens, request_mask)
            uncached = _compressed_tensor(
                encoder(repeated_tokens, repeated_mask, global_tokens)
            )
            cached = _compressed_tensor(
                encoder(request_tokens, request_mask, global_tokens, cache=cache)
            )

        torch.testing.assert_close(cached, uncached, rtol=1.0e-5, atol=1.0e-6)

    def test_user_globals_are_cacheable_but_candidates_are_isolated(self) -> None:
        torch.manual_seed(37)
        encoder = self._encoder(user_global_tokens=1).eval()
        tokens = torch.randn(1, 5, 4)
        mask = torch.ones(1, 5, dtype=torch.bool)
        user_global = torch.tensor([[[2.0, -1.0, 0.5, 3.0]]])
        candidate_a = torch.randn(1, 1, 4)
        candidate_b = torch.randn(1, 1, 4)

        with torch.no_grad():
            cache = encoder.precompute_cache(tokens, mask, user_global)
            uncached = encoder(
                tokens,
                mask,
                candidate_a,
                user_global_tokens=user_global,
            ).view(1, 4, 4)
            cached_a = encoder(tokens, mask, candidate_a, cache=cache).view(1, 4, 4)
            cached_b = encoder(tokens, mask, candidate_b, cache=cache).view(1, 4, 4)

        torch.testing.assert_close(cached_a, uncached, rtol=1.0e-5, atol=1.0e-6)
        # Output order is [cacheable user globals; candidate globals; recent queries].
        torch.testing.assert_close(cached_a[:, 0, :], cached_b[:, 0, :])
        torch.testing.assert_close(cached_a[:, 2:, :], cached_b[:, 2:, :])
        self.assertFalse(torch.allclose(cached_a[:, 1, :], cached_b[:, 1, :]))

    def test_user_global_changes_sequence_side_cache_state(self) -> None:
        torch.manual_seed(41)
        encoder = self._encoder(user_global_tokens=1).eval()
        tokens = torch.randn(1, 5, 4)
        mask = torch.ones(1, 5, dtype=torch.bool)
        user_a = torch.tensor([[[1.0, 0.0, -1.0, 2.0]]])
        user_b = torch.tensor([[[-2.0, 1.0, 0.0, 3.0]]])

        with torch.no_grad():
            cache_a = encoder.precompute_cache(tokens, mask, user_a)
            cache_b = encoder.precompute_cache(tokens, mask, user_b)

        self.assertFalse(
            torch.allclose(cache_a.cross_recent_output, cache_b.cross_recent_output)
        )

    def test_activation_checkpoint_covers_merge_cross_and_self_blocks(self) -> None:
        torch.manual_seed(43)
        encoder = LongerSequenceEncoder(
            token_dim=4,
            num_heads=2,
            hidden_dim=8,
            query_token_count=2,
            self_layers=1,
            summary_tokens=2,
            token_merge=1,
            inner_layers=1,
            user_global_tokens=1,
            activation_checkpoint=True,
        ).train()
        tokens = torch.randn(2, 5, 4, requires_grad=True)
        mask = torch.ones(2, 5, dtype=torch.bool)
        user_global = torch.randn(2, 1, 4, requires_grad=True)
        candidate_global = torch.randn(2, 1, 4, requires_grad=True)

        output = encoder(
            tokens,
            mask,
            candidate_global,
            user_global_tokens=user_global,
        )
        output.square().mean().backward()

        self.assertIsNotNone(tokens.grad)
        self.assertIsNotNone(user_global.grad)
        self.assertIsNotNone(candidate_global.grad)


class LongerInputGenerationAlignmentTest(unittest.TestCase):
    def _sequence(self, order: str) -> SequenceConfig:
        return SequenceConfig(
            name="hist",
            fields=[
                SequenceFieldConfig(
                    name="item",
                    kind="dense",
                    source="hist_item",
                    dimension=2,
                ),
                SequenceFieldConfig(
                    name="time_delta",
                    kind="dense",
                    source="hist_time_delta",
                ),
            ],
            max_length=2,
            encoder="longer",
            target_inputs=["candidate"],
            time_delta_field="time_delta",
            sequence_order=order,
        )

    def _encoder_bank(self, sequence: SequenceConfig) -> FeatureEncoderBank:
        bank = FeatureEncoderBank.__new__(FeatureEncoderBank)
        nn.Module.__init__(bank)
        bank.sequences_by_name = {sequence.name: sequence}
        bank.sequence_field_embedding_keys = {}
        bank.embeddings = nn.ModuleDict()
        bank.sequence_step_projectors = nn.ModuleDict(
            {"hist": nn.Linear(3, 2, bias=False)}
        )
        bank.sequence_position_embeddings = nn.ModuleDict(
            {"hist": nn.Embedding(2, 2)}
        )
        with torch.no_grad():
            bank.sequence_step_projectors["hist"].weight.copy_(
                torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            )
            bank.sequence_position_embeddings["hist"].weight.copy_(
                torch.tensor([[10.0, 20.0], [30.0, 40.0]])
            )
        return bank

    def _payload(self, newest_first: bool = False) -> dict[str, Any]:
        item = torch.tensor([[[1.0, 2.0], [4.0, 5.0]]])
        time_delta = torch.tensor([[[3.0], [6.0]]])
        if newest_first:
            item = item.flip(1)
            time_delta = time_delta.flip(1)
        return {
            "fields": {"item": item, "time_delta": time_delta},
            "lengths": torch.tensor([2]),
        }

    def test_position_is_added_before_time_delta_concat_and_projection(self) -> None:
        sequence = self._sequence("oldest_to_newest")
        bank = self._encoder_bank(sequence)

        tokens, mask = bank._multi_field_sequence_tokens(sequence, self._payload())

        expected = torch.tensor([[[11.0, 3.0], [34.0, 6.0]]])
        torch.testing.assert_close(tokens, expected)
        torch.testing.assert_close(mask, torch.ones(1, 2, dtype=torch.bool))

    def test_both_physical_sequence_orders_canonicalize_identically(self) -> None:
        oldest = self._sequence("oldest_to_newest")
        newest = self._sequence("newest_to_oldest")
        bank = self._encoder_bank(oldest)

        oldest_tokens, oldest_mask = bank._multi_field_sequence_tokens(
            oldest,
            self._payload(),
        )
        newest_tokens, newest_mask = bank._multi_field_sequence_tokens(
            newest,
            self._payload(newest_first=True),
        )

        torch.testing.assert_close(newest_tokens, oldest_tokens)
        torch.testing.assert_close(newest_mask, oldest_mask)

    def test_truncation_window_is_explicit_for_both_physical_orders(self) -> None:
        oldest = replace(self._sequence("oldest_to_newest"), truncation="tail")
        newest = replace(self._sequence("newest_to_oldest"), truncation="head")

        self.assertEqual(_sequence_bounds(5, oldest), (3, 5))
        self.assertEqual(_sequence_bounds(5, newest), (0, 2))


if __name__ == "__main__":
    unittest.main()
