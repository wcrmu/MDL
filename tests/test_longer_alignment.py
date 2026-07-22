from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from typing import Any

import torch
from torch import Tensor, nn

from src.config import SequenceConfig, SequenceFieldConfig
from src.dataloader import _sequence_bounds
from src.model import (
    FeatureEncoderBank,
    LongerSequenceAttentionBlock,
    LongerSequenceEncoder,
    LongerTokenMerger,
    _resolve_longer_chunk_rows,
)


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
    def test_token_budget_scales_rows_with_sequence_length(self) -> None:
        self.assertEqual(
            _resolve_longer_chunk_rows(
                1024, 2048, token_limit=262_144
            ),
            128,
        )
        self.assertEqual(
            _resolve_longer_chunk_rows(
                1024, 1024, token_limit=262_144
            ),
            256,
        )
        self.assertEqual(
            _resolve_longer_chunk_rows(
                1024, 256, token_limit=262_144
            ),
            1024,
        )
        self.assertEqual(
            _resolve_longer_chunk_rows(
                1024,
                256,
                row_limit=32,
                token_limit=262_144,
            ),
            32,
        )

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

    def test_chunked_inner_transformer_matches_unchunked_forward_and_backward(self) -> None:
        torch.manual_seed(11)
        reference = LongerTokenMerger(
            token_dim=4,
            num_heads=2,
            hidden_dim=8,
            merge_size=2,
            inner_layers=1,
        )
        chunked = LongerTokenMerger(
            token_dim=4,
            num_heads=2,
            hidden_dim=8,
            merge_size=2,
            inner_layers=1,
        )
        chunked.load_state_dict(reference.state_dict())
        chunked._INNER_ATTENTION_BATCH_LIMIT = 2
        reference_tokens = torch.randn(2, 6, 4, requires_grad=True)
        chunked_tokens = reference_tokens.detach().clone().requires_grad_(True)
        mask = torch.tensor(
            [[True, True, True, True, True, True], [False, True, True, True, True, True]]
        )

        reference_output, reference_mask = reference(reference_tokens, mask)
        chunked_output, chunked_mask = chunked(chunked_tokens, mask)
        torch.testing.assert_close(chunked_output, reference_output)
        torch.testing.assert_close(chunked_mask, reference_mask)

        reference_output.square().sum().backward()
        chunked_output.square().sum().backward()
        torch.testing.assert_close(chunked_tokens.grad, reference_tokens.grad)
        for (reference_name, reference_parameter), (chunked_name, chunked_parameter) in zip(
            reference.named_parameters(), chunked.named_parameters()
        ):
            self.assertEqual(chunked_name, reference_name)
            torch.testing.assert_close(chunked_parameter.grad, reference_parameter.grad)

    def test_leading_padding_compaction_preserves_merged_valid_groups(self) -> None:
        merger = LongerTokenMerger(
            token_dim=2,
            num_heads=1,
            hidden_dim=8,
            merge_size=4,
            inner_layers=0,
        )
        compact_tokens = torch.arange(10, dtype=torch.float32).view(1, 5, 2)
        compact_mask = torch.ones(1, 5, dtype=torch.bool)
        padded_tokens = torch.cat(
            [torch.zeros(1, 11, 2), compact_tokens],
            dim=1,
        )
        padded_mask = torch.cat(
            [torch.zeros(1, 11, dtype=torch.bool), compact_mask],
            dim=1,
        )

        compact, compact_merged_mask = merger(compact_tokens, compact_mask)
        padded, padded_merged_mask = merger(padded_tokens, padded_mask)

        torch.testing.assert_close(padded[:, -compact.size(1) :], compact)
        torch.testing.assert_close(
            padded_merged_mask[:, -compact_merged_mask.size(1) :],
            compact_merged_mask,
        )
        self.assertFalse(bool(padded_merged_mask[:, :-compact_merged_mask.size(1)].any()))


class FeatureEncoderSequenceCompactionTest(unittest.TestCase):
    def test_default_alignment_keeps_batch_physical_width(self) -> None:
        encoder = FeatureEncoderBank.__new__(FeatureEncoderBank)
        sequence = SequenceConfig(
            name="hist",
            fields=[
                SequenceFieldConfig(
                    name="value",
                    kind="dense",
                    source="hist_value",
                )
            ],
            max_length=2000,
            encoder="longer",
            time_delta_field="value",
        )
        inputs = torch.arange(20, dtype=torch.float32).view(2, 5, 2)
        lengths = torch.tensor([3, 5])

        aligned, mask = encoder._align_sequence_inputs(
            sequence,
            inputs,
            lengths,
        )

        self.assertEqual(tuple(aligned.shape), (2, 5, 2))
        torch.testing.assert_close(
            mask,
            torch.tensor(
                [[False, False, True, True, True], [True, True, True, True, True]]
            ),
        )


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

    def test_compacting_leading_padding_preserves_encoder_output(self) -> None:
        torch.manual_seed(5)
        encoder = LongerSequenceEncoder(
            token_dim=4,
            num_heads=2,
            hidden_dim=8,
            query_token_count=2,
            self_layers=1,
            summary_tokens=2,
            token_merge=2,
            inner_layers=1,
        ).eval()
        compact_tokens = torch.randn(2, 5, 4)
        compact_mask = torch.tensor(
            [
                [False, False, True, True, True],
                [True, True, True, True, True],
            ]
        )
        padded_tokens = torch.cat(
            [torch.zeros(2, 7, 4), compact_tokens],
            dim=1,
        )
        padded_mask = torch.cat(
            [torch.zeros(2, 7, dtype=torch.bool), compact_mask],
            dim=1,
        )
        candidate_globals = torch.randn(2, 2, 8)

        compact = encoder(compact_tokens, compact_mask, candidate_globals)
        padded = encoder(padded_tokens, padded_mask, candidate_globals)

        torch.testing.assert_close(compact, padded, rtol=1.0e-5, atol=1.0e-6)

    def test_mixed_visibility_contract_keeps_global_full_and_recent_causal(self) -> None:
        key_valid = torch.tensor(
            [
                [True, False, False, True, True, True],
                [True, False, False, False, False, True],
            ]
        )
        # Query layout is [one global; two recent]. The first recent query in
        # the second row is padding and must have no semantic visibility.
        query_valid = torch.tensor(
            [[True, True, True], [True, False, True]]
        )

        actual = LongerSequenceAttentionBlock.mixed_allowed_mask(
            key_valid, query_valid, global_query_count=1
        )
        expected = torch.tensor(
            [
                [
                    [True, False, False, True, True, True],
                    [True, False, False, True, True, False],
                    [True, False, False, True, True, True],
                ],
                [
                    [True, False, False, False, False, True],
                    [False, False, False, False, False, False],
                    [True, False, False, False, False, True],
                ],
            ]
        )
        torch.testing.assert_close(actual, expected)

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

    def test_full_checkpoint_drops_longer_kv_and_preserves_gradients(self) -> None:
        torch.manual_seed(45)
        baseline = LongerSequenceEncoder(
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
            drop_cached_kv=False,
        ).train()
        low_memory = LongerSequenceEncoder(
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
            drop_cached_kv=True,
        ).train()
        low_memory.load_state_dict(baseline.state_dict())
        mask = torch.tensor(
            [[True, True, True, True, True], [False, False, True, True, True]]
        )
        baseline_inputs = [
            torch.randn(2, 5, 4, requires_grad=True),
            torch.randn(2, 1, 4, requires_grad=True),
            torch.randn(2, 1, 4, requires_grad=True),
        ]
        low_memory_inputs = [value.detach().clone().requires_grad_() for value in baseline_inputs]

        baseline_cache = baseline.precompute_cache(
            baseline_inputs[0], mask, baseline_inputs[1]
        )
        low_memory_cache = low_memory.precompute_cache(
            low_memory_inputs[0], mask, low_memory_inputs[1]
        )
        self.assertGreater(baseline_cache.cross_cacheable_key.size(2), 0)
        self.assertEqual(low_memory_cache.cross_cacheable_key.size(2), 0)
        self.assertEqual(low_memory_cache.cross_cacheable_value.size(2), 0)
        self.assertEqual(low_memory_cache.self_layers[0].cacheable_key.size(2), 0)
        self.assertEqual(low_memory_cache.self_layers[0].cacheable_value.size(2), 0)

        baseline_output = baseline(
            baseline_inputs[0],
            mask,
            baseline_inputs[2],
            cache=baseline_cache,
        )
        low_memory_output = low_memory(
            low_memory_inputs[0],
            mask,
            low_memory_inputs[2],
            cache=low_memory_cache,
        )
        torch.testing.assert_close(
            low_memory_output, baseline_output, rtol=1.0e-5, atol=1.0e-6
        )

        baseline_output.square().mean().backward()
        low_memory_output.square().mean().backward()
        for low_grad, baseline_grad in zip(
            (value.grad for value in low_memory_inputs),
            (value.grad for value in baseline_inputs),
        ):
            torch.testing.assert_close(
                low_grad, baseline_grad, rtol=2.0e-4, atol=2.0e-5
            )

    def test_row_chunked_outer_checkpoint_matches_full_batch(self) -> None:
        torch.manual_seed(46)
        sequence = SequenceConfig(
            name="hist",
            fields=[
                SequenceFieldConfig(
                    name="time_delta",
                    kind="dense",
                    source="hist_time_delta",
                )
            ],
            encoder="longer",
            time_delta_field="time_delta",
            rankmixer_summary_tokens=1,
            longer_query_tokens=2,
            longer_self_layers=1,
            longer_token_merge=1,
            longer_inner_layers=0,
            longer_cls_tokens=1,
            longer_candidate_global_tokens=0,
            longer_output="summary",
        )
        baseline_projector = nn.Linear(3, 4)
        low_memory_projector = nn.Linear(3, 4)
        low_memory_projector.load_state_dict(baseline_projector.state_dict())
        baseline_encoder = LongerSequenceEncoder(
            token_dim=4,
            num_heads=2,
            hidden_dim=8,
            query_token_count=2,
            self_layers=1,
            summary_tokens=1,
            token_merge=1,
            inner_layers=0,
            user_global_tokens=1,
            summary_only=True,
        ).train()
        low_memory_encoder = LongerSequenceEncoder(
            token_dim=4,
            num_heads=2,
            hidden_dim=8,
            query_token_count=2,
            self_layers=1,
            summary_tokens=1,
            token_merge=1,
            inner_layers=0,
            user_global_tokens=1,
            activation_checkpoint=True,
            checkpoint_token_merger=True,
            drop_cached_kv=True,
            summary_only=True,
        ).train()
        low_memory_encoder.load_state_dict(baseline_encoder.state_dict())
        bank = FeatureEncoderBank.__new__(FeatureEncoderBank)
        nn.Module.__init__(bank)
        bank.config = SimpleNamespace(
            runtime=SimpleNamespace(sequence_encoder_chunk_rows=2)
        )
        bank.sequence_step_projectors = nn.ModuleDict(
            {"hist": low_memory_projector}
        )
        bank.sequence_longer_encoders = nn.ModuleDict(
            {"hist": low_memory_encoder}
        )
        mask = torch.tensor(
            [
                [True, True, True, True],
                [False, True, True, True],
                [False, False, True, True],
                [True, True, True, True],
                [False, False, False, True],
            ]
        )
        baseline_inputs = torch.randn(5, 4, 3, requires_grad=True)
        low_memory_inputs = baseline_inputs.detach().clone().requires_grad_()
        baseline_user = torch.randn(5, 1, 4, requires_grad=True)
        low_memory_user = baseline_user.detach().clone().requires_grad_()

        baseline_tokens = baseline_projector(baseline_inputs)
        baseline_tokens = baseline_tokens * mask.unsqueeze(-1)
        baseline_output = baseline_encoder(
            baseline_tokens,
            mask,
            baseline_tokens.new_zeros(5, 0, 4),
            user_global_tokens=baseline_user,
        )
        low_memory_output = bank._pool_checkpointed_longer_inputs(
            sequence,
            low_memory_inputs,
            mask,
            low_memory_user,
        )
        torch.testing.assert_close(low_memory_output, baseline_output)

        baseline_output.square().mean().backward()
        low_memory_output.square().mean().backward()
        torch.testing.assert_close(low_memory_inputs.grad, baseline_inputs.grad)
        torch.testing.assert_close(low_memory_user.grad, baseline_user.grad)
        for low_parameter, baseline_parameter in zip(
            low_memory_projector.parameters(), baseline_projector.parameters()
        ):
            torch.testing.assert_close(low_parameter.grad, baseline_parameter.grad)
        for low_parameter, baseline_parameter in zip(
            low_memory_encoder.parameters(), baseline_encoder.parameters()
        ):
            torch.testing.assert_close(
                low_parameter.grad,
                baseline_parameter.grad,
                rtol=2.0e-4,
                atol=2.0e-5,
            )

    def test_summary_only_exposes_one_history_conditioned_global_token(self) -> None:
        torch.manual_seed(47)
        encoder = LongerSequenceEncoder(
            token_dim=4,
            num_heads=2,
            hidden_dim=8,
            query_token_count=2,
            self_layers=1,
            summary_tokens=1,
            token_merge=1,
            inner_layers=0,
            user_global_tokens=1,
            summary_only=True,
        )
        tokens = torch.randn(2, 5, 4, requires_grad=True)
        mask = torch.tensor(
            [[True, True, True, True, True], [False, False, True, True, True]]
        )
        cls = torch.randn(2, 1, 4, requires_grad=True)
        no_candidate_globals = torch.zeros(2, 0, 4)

        output = encoder(
            tokens,
            mask,
            no_candidate_globals,
            user_global_tokens=cls,
        )

        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertEqual(encoder.output_dim, 4)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertTrue(torch.isfinite(cls.grad).all())


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
