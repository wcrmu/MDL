from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from src.benchmark import (
    _replace_id_embeddings_with_synthetic,
    _synthetic_feature_batch,
    _synthetic_vocab_maps,
)
from src.config import _mixformer_ui_head_split, load_app_config
from src.model import (
    MDLMixFormerModel,
    MixFormerFeatureHeadProjector,
    MixFormerModel,
    MixFormerTokenizer,
    ScenarioConditionedQueryRouter,
    build_model,
)
from src.modules.attention import varlen_attention_available
from src.modules.mixformer import (
    DenseSwiGLUFFN,
    MixFormerBlock,
    MixFormerCrossAttention,
    MixFormerHeadMixing,
    MixFormerQueryMixer,
    MixFormerRMSNorm,
    StackedPerHeadSwiGLUFFN,
    assemble_mixformer_heads,
)


ROOT = Path(__file__).resolve().parents[1]


class MixFormerPaperAlignmentTest(unittest.TestCase):
    def test_head_mixing_is_exact_reshape_transpose(self) -> None:
        module = MixFormerHeadMixing(num_heads=2, dim=4)
        values = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]
        )

        actual = module(values)

        expected = torch.tensor(
            [[[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]]]
        )
        torch.testing.assert_close(actual, expected)
        self.assertEqual(sum(parameter.numel() for parameter in module.parameters()), 0)

    def test_rmsnorm_matches_mean_square_formula(self) -> None:
        module = MixFormerRMSNorm(8)
        values = torch.randn(3, 5, 8)
        scale = values.pow(2).mean(dim=-1, keepdim=True).add(module.eps).rsqrt()
        torch.testing.assert_close(module(values), values * scale * module.weight)

    def test_rmsnorm_casts_fp32_weight_for_bf16_input(self) -> None:
        module = MixFormerRMSNorm(4)
        values = torch.randn(2, 4, dtype=torch.bfloat16)
        out = module(values)
        self.assertEqual(out.dtype, torch.bfloat16)
        expected = torch.nn.functional.rms_norm(
            values,
            (4,),
            module.weight.to(dtype=torch.bfloat16),
            module.eps,
        )
        torch.testing.assert_close(out, expected)

    def test_packed_per_head_swiglu_matches_two_gemm_reference(self) -> None:
        torch.manual_seed(4)
        module = StackedPerHeadSwiGLUFFN(num_heads=3, dim=5, hidden_dim=7)
        values = torch.randn(4, 3, 5)
        head_major = values.transpose(0, 1)
        up = torch.bmm(head_major, module.up_weight.transpose(1, 2))
        gate = torch.bmm(head_major, module.gate_weight.transpose(1, 2))
        expected = torch.bmm(
            up * torch.nn.functional.silu(gate),
            module.output_weight.transpose(1, 2),
        ).transpose(0, 1)
        torch.testing.assert_close(module(values), expected)

    def test_dense_swiglu_packed_matches_split_reference(self) -> None:
        torch.manual_seed(5)
        module = DenseSwiGLUFFN(dim=6, hidden_dim=5)
        values = torch.randn(4, 3, 6)
        packed_weight = module.up_gate_projection.weight
        up = values @ packed_weight[:5].t()
        gate = values @ packed_weight[5:].t()
        expected = (up * torch.nn.functional.silu(gate)) @ (
            module.output_projection.weight.t()
        )
        torch.testing.assert_close(module(values), expected)

    def test_dense_swiglu_loads_split_checkpoint_keys(self) -> None:
        torch.manual_seed(6)
        up = torch.nn.Linear(4, 5, bias=False)
        gate = torch.nn.Linear(4, 5, bias=False)
        output = torch.nn.Linear(5, 4, bias=False)
        module = DenseSwiGLUFFN(dim=4, hidden_dim=5)
        module.load_state_dict(
            {
                "up_projection.weight": up.weight,
                "gate_projection.weight": gate.weight,
                "output_projection.weight": output.weight,
            }
        )
        values = torch.randn(3, 4)
        expected = (
            (values @ up.weight.t())
            * torch.nn.functional.silu(values @ gate.weight.t())
        ) @ output.weight.t()
        torch.testing.assert_close(module(values), expected)

    def test_compact_history_marks_attention_density(self) -> None:
        tokens = torch.arange(8, dtype=torch.float32).view(2, 4, 1)
        dense_mask = torch.ones(2, 4, dtype=torch.bool)
        _, marked = MixFormerTokenizer.compact_selected_history(tokens, dense_mask)
        self.assertTrue(getattr(marked, "_mixformer_dense"))

        mixed = torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        )
        packed_tokens, packed_mask = MixFormerTokenizer.compact_selected_history(
            tokens,
            mixed,
        )
        self.assertEqual(tuple(packed_tokens.shape), (2, 3, 1))
        self.assertFalse(getattr(packed_mask, "_mixformer_dense"))

    def test_selected_history_does_not_pad_to_the_cap(self) -> None:
        tokens = torch.tensor(
            [
                [[1.0], [2.0], [0.0], [0.0], [0.0], [0.0]],
                [[0.0], [0.0], [0.0], [3.0], [4.0], [5.0]],
            ]
        )
        mask = torch.tensor(
            [
                [True, True, False, False, False, False],
                [False, False, False, True, True, True],
            ]
        )

        compact_tokens, compact_mask = MixFormerTokenizer.compact_selected_history(
            tokens,
            mask,
        )

        self.assertEqual(tuple(compact_tokens.shape), (2, 3, 1))
        torch.testing.assert_close(
            compact_tokens[:, :, 0],
            torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]]),
        )
        torch.testing.assert_close(
            compact_mask,
            torch.tensor([[True, True, False], [True, True, True]]),
        )

    def test_empty_selected_history_stays_length_zero(self) -> None:
        tokens = torch.zeros(2, 8, 4)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        compact_tokens, compact_mask = MixFormerTokenizer.compact_selected_history(
            tokens,
            mask,
        )
        self.assertEqual(tuple(compact_tokens.shape), (2, 0, 4))
        self.assertEqual(tuple(compact_mask.shape), (2, 0))

    def test_ui_head_mixing_masks_item_signal_from_user_outputs(self) -> None:
        module = MixFormerHeadMixing(
            num_heads=2,
            dim=4,
            user_head_count=1,
        )
        values = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]
        )

        actual = module(values)

        expected = torch.tensor(
            [[[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 7.0, 8.0]]]
        )
        torch.testing.assert_close(actual, expected)

    def test_user_only_head_mixing_matches_masked_full_tensor(self) -> None:
        module = MixFormerHeadMixing(
            num_heads=2,
            dim=4,
            user_head_count=1,
        )
        values = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]
        )

        full = module(values)[:, :1]
        user_only = module.mix_user_heads(values[:, :1])

        torch.testing.assert_close(user_only, full)

    def test_decoupled_query_mixer_matches_candidate_expanded_path(self) -> None:
        torch.manual_seed(9)
        module = MixFormerQueryMixer(
            num_heads=2,
            dim=4,
            hidden_dim=6,
            user_head_count=1,
        )
        user_heads = torch.randn(2, 1, 4)
        item_heads = torch.randn(4, 1, 4)
        row_indices = torch.tensor([0, 0, 1, 1])
        gathered = assemble_mixformer_heads(user_heads, item_heads, row_indices)

        user_query, item_query = module.forward_decoupled(
            user_heads,
            item_heads,
            row_indices,
        )
        full = module(gathered)

        torch.testing.assert_close(
            assemble_mixformer_heads(user_query, item_query, row_indices),
            full,
        )

    def test_decoupled_block_matches_candidate_expanded_path(self) -> None:
        torch.manual_seed(11)
        block = MixFormerBlock(
            num_heads=2,
            dim=4,
            hidden_dim=6,
            user_head_count=1,
        )
        user_heads = torch.randn(2, 1, 4)
        item_heads = torch.randn(4, 1, 4)
        history = torch.randn(2, 5, 8)
        valid_mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, False, False, False, False],
            ]
        )
        row_indices = torch.tensor([0, 0, 1, 1])
        gathered = assemble_mixformer_heads(user_heads, item_heads, row_indices)

        user_out, item_out = block.forward_decoupled(
            user_heads,
            item_heads,
            history,
            valid_mask,
            row_indices,
        )
        expanded = block(
            gathered,
            history.index_select(0, row_indices),
            valid_mask.index_select(0, row_indices),
        )

        torch.testing.assert_close(
            assemble_mixformer_heads(user_out, item_out, row_indices),
            expanded,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_decoupled_attention_keeps_request_major_user_queries(self) -> None:
        torch.manual_seed(15)
        attention = MixFormerCrossAttention(
            num_heads=2,
            dim=4,
            hidden_dim=6,
            user_head_count=1,
        )
        user_query = torch.randn(1, 1, 4)
        item_query = torch.randn(3, 1, 4)
        history = torch.randn(1, 5, 8)
        valid_mask = torch.tensor([[True, True, True, False, False]])
        row_indices = torch.zeros(3, dtype=torch.long)
        gathered = assemble_mixformer_heads(user_query, item_query, row_indices)

        user_out, item_out = attention.forward_decoupled(
            user_query,
            item_query,
            history,
            valid_mask,
        )
        expanded = attention(
            gathered,
            history.expand(3, -1, -1),
            valid_mask.expand(3, -1),
        )

        self.assertEqual(tuple(user_out.shape), (1, 1, 4))
        torch.testing.assert_close(
            assemble_mixformer_heads(user_out, item_out, row_indices),
            expanded,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_ui_head_split_uses_paper_formula_then_one_to_one(self) -> None:
        self.assertEqual(_mixformer_ui_head_split(600, 1200, 16, None), 6)
        self.assertEqual(_mixformer_ui_head_split(624, 1312, 16, None), 8)
        self.assertEqual(_mixformer_ui_head_split(624, 1312, 16, 8), 8)
        with self.assertRaises(ValueError):
            _mixformer_ui_head_split(624, 1312, 16, 6)

    def test_query_mixer_matches_pre_norm_residual_equations(self) -> None:
        torch.manual_seed(3)
        module = MixFormerQueryMixer(num_heads=2, dim=4, hidden_dim=7)
        with torch.no_grad():
            module.ffn.up_weight.zero_()
            module.ffn.gate_weight.zero_()
            module.ffn.output_weight.zero_()
        values = torch.randn(3, 2, 4)

        expected = values + module.head_mixing(module.input_norm(values))
        actual = module(values)

        torch.testing.assert_close(actual, expected)

    def test_cross_attention_scores_raw_query(self) -> None:
        attention = MixFormerCrossAttention(num_heads=2, dim=4, hidden_dim=6)
        self.assertFalse(hasattr(attention, "query_norm"))
        with torch.no_grad():
            attention.key_weight.copy_(torch.eye(4).expand(2, -1, -1))
        query = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]])
        projected = attention._project_query(query)
        torch.testing.assert_close(projected, query)

    def test_feature_embedding_is_even_split_then_independent_projection(self) -> None:
        projector = MixFormerFeatureHeadProjector(
            ["a", "b"],
            {"a": 2, "b": 2},
            num_heads=2,
            head_dim=3,
            init_std=0.02,
        )
        with torch.no_grad():
            projector.weight.zero_()
            projector.weight[0, :2, :] = torch.eye(2)
            projector.weight[1, :2, :] = torch.eye(2)

        output = projector(
            {
                "a": torch.tensor([[1.0, 2.0]]),
                "b": torch.tensor([[3.0, 4.0]]),
            }
        )

        expected = torch.tensor([[[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]])
        torch.testing.assert_close(output, expected)

    def test_reordered_cross_attention_matches_materialized_reference(self) -> None:
        torch.manual_seed(5)
        attention = MixFormerCrossAttention(
            num_heads=2,
            dim=4,
            hidden_dim=6,
        )
        query = torch.randn(5, 2, 4, requires_grad=True)
        history = torch.randn(3, 6, 8, requires_grad=True)
        valid_mask = torch.tensor(
            [
                [True, True, True, False, False, False],
                [True, False, False, False, False, False],
                [False, False, False, False, False, False],
            ]
        )
        row_indices = torch.tensor([0, 0, 1, 2, 2])

        optimized = attention(query, history, valid_mask, row_indices)
        reference = attention.forward_reference(
            query,
            history,
            valid_mask,
            row_indices,
        )

        torch.testing.assert_close(optimized, reference, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(optimized[3:], query[3:])
        self.assertTrue(bool(torch.isfinite(optimized).all()))
        gradient_targets = (query, history, *attention.parameters())
        optimized_gradients = torch.autograd.grad(
            optimized.square().mean(),
            gradient_targets,
        )
        reference_gradients = torch.autograd.grad(
            reference.square().mean(),
            gradient_targets,
        )
        for optimized_gradient, reference_gradient in zip(
            optimized_gradients,
            reference_gradients,
        ):
            torch.testing.assert_close(
                optimized_gradient,
                reference_gradient,
                rtol=1e-5,
                atol=1e-6,
            )

    def test_request_grouping_matches_candidate_expansion(self) -> None:
        torch.manual_seed(7)
        attention = MixFormerCrossAttention(
            num_heads=2,
            dim=4,
            hidden_dim=5,
        )
        query = torch.randn(4, 2, 4)
        history = torch.randn(3, 3, 8)
        valid_mask = torch.tensor(
            [
                [True, True, False],
                [False, False, False],
                [True, True, True],
            ]
        )
        # Interleaved targets plus a request with no target exercise the
        # general request-level packing path, not only contiguous candidates.
        row_indices = torch.tensor([2, 0, 2, 0])

        grouped = attention(query, history, valid_mask, row_indices)
        expanded = attention(
            query,
            history.index_select(0, row_indices),
            valid_mask.index_select(0, row_indices),
        )

        torch.testing.assert_close(grouped, expanded, rtol=1e-5, atol=1e-6)

    @unittest.skipUnless(
        torch.cuda.is_available() and varlen_attention_available(),
        "Dao flash_attn varlen is required",
    )
    def test_varlen_mixed_length_matches_masked_sdpa(self) -> None:
        torch.manual_seed(21)
        attention = (
            MixFormerCrossAttention(num_heads=2, dim=32, hidden_dim=64)
            .cuda()
            .to(dtype=torch.bfloat16)
        )
        query = torch.randn(3, 2, 32, device="cuda", dtype=torch.bfloat16)
        history = torch.randn(3, 6, 64, device="cuda", dtype=torch.bfloat16)
        valid_mask = torch.tensor(
            [
                [True, True, True, False, False, False],
                [True, False, False, False, False, False],
                [True, True, True, True, True, False],
            ],
            device="cuda",
        )
        varlen = attention(query, history, valid_mask)
        with patch.object(
            MixFormerCrossAttention, "_can_varlen_history", return_value=False
        ):
            masked = attention(query, history, valid_mask)
        torch.testing.assert_close(varlen, masked, rtol=2e-2, atol=2e-2)

    @unittest.skipUnless(
        torch.cuda.is_available() and varlen_attention_available(),
        "Dao flash_attn varlen is required",
    )
    def test_varlen_grouped_occupancy_matches_expanded(self) -> None:
        torch.manual_seed(22)
        attention = (
            MixFormerCrossAttention(num_heads=2, dim=32, hidden_dim=64)
            .cuda()
            .to(dtype=torch.bfloat16)
        )
        query = torch.randn(5, 2, 32, device="cuda", dtype=torch.bfloat16)
        history = torch.randn(3, 6, 64, device="cuda", dtype=torch.bfloat16)
        valid_mask = torch.tensor(
            [
                [True, True, True, False, False, False],
                [False, False, False, False, False, False],
                [True, True, True, True, True, False],
            ],
            device="cuda",
        )
        row_indices = torch.tensor([2, 0, 2, 0, 2], device="cuda")
        grouped = attention(query, history, valid_mask, row_indices)
        expanded = attention(
            query,
            history.index_select(0, row_indices),
            valid_mask.index_select(0, row_indices),
        )
        torch.testing.assert_close(grouped, expanded, rtol=2e-2, atol=2e-2)
        with patch.object(
            MixFormerCrossAttention, "_can_varlen_history", return_value=False
        ):
            masked = attention(query, history, valid_mask, row_indices)
        torch.testing.assert_close(grouped, masked, rtol=2e-2, atol=2e-2)

    def test_full_occupancy_grouping_matches_candidate_expansion(self) -> None:
        torch.manual_seed(8)
        attention = MixFormerCrossAttention(
            num_heads=2,
            dim=4,
            hidden_dim=5,
        )
        query = torch.randn(8, 2, 4)
        history = torch.randn(4, 3, 8)
        valid_mask = torch.ones(4, 3, dtype=torch.bool)
        row_indices = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        layout = MixFormerCrossAttention.request_layout(row_indices, 4)
        self.assertFalse(
            MixFormerCrossAttention._prefer_candidate_gather(
                candidate_count=8,
                request_count=layout.request_count,
                max_targets=layout.max_targets,
                history_length=3,
                history_heads=2,
                history_dim=4,
                element_size=4,
            )
        )
        grouped = attention(query, history, valid_mask, row_indices)
        expanded = attention(
            query,
            history.index_select(0, row_indices),
            valid_mask.index_select(0, row_indices),
        )
        torch.testing.assert_close(grouped, expanded, rtol=1e-5, atol=1e-6)

    def test_candidate_gather_skips_long_history_hbm(self) -> None:
        self.assertTrue(
            MixFormerCrossAttention._prefer_candidate_gather(
                candidate_count=448,
                request_count=320,
                max_targets=7,
                history_length=320,
                history_heads=4,
                history_dim=128,
                element_size=2,
            )
        )
        self.assertFalse(
            MixFormerCrossAttention._prefer_candidate_gather(
                candidate_count=448,
                request_count=320,
                max_targets=7,
                history_length=8000,
                history_heads=4,
                history_dim=128,
                element_size=2,
            )
        )

    def test_chunked_cross_attention_matches_eager(self) -> None:
        torch.manual_seed(13)
        eager = MixFormerCrossAttention(
            num_heads=2,
            dim=4,
            hidden_dim=6,
            sequence_chunk_tokens=0,
        )
        chunked = MixFormerCrossAttention(
            num_heads=2,
            dim=4,
            hidden_dim=6,
            # Force both length-chunked online softmax and request chunking.
            sequence_chunk_tokens=2,
        )
        chunked.load_state_dict(eager.state_dict())
        query = torch.randn(5, 2, 4, requires_grad=True)
        history = torch.randn(3, 7, 8, requires_grad=True)
        valid_mask = torch.tensor(
            [
                [True, True, True, False, True, False, False],
                [True, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
            ]
        )
        row_indices = torch.tensor([0, 0, 1, 2, 2])

        eager_query = query.detach().clone().requires_grad_(True)
        eager_history = history.detach().clone().requires_grad_(True)
        chunked_query = query.detach().clone().requires_grad_(True)
        chunked_history = history.detach().clone().requires_grad_(True)

        eager_out = eager(eager_query, eager_history, valid_mask, row_indices)
        chunked_out = chunked(
            chunked_query,
            chunked_history,
            valid_mask,
            row_indices,
        )
        torch.testing.assert_close(chunked_out, eager_out, rtol=1e-5, atol=1e-5)

        eager_out.square().mean().backward()
        chunked_out.square().mean().backward()
        torch.testing.assert_close(
            chunked_query.grad,
            eager_query.grad,
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            chunked_history.grad,
            eager_history.grad,
            rtol=1e-5,
            atol=1e-5,
        )
        for eager_parameter, chunked_parameter in zip(
            eager.parameters(),
            chunked.parameters(),
        ):
            torch.testing.assert_close(
                chunked_parameter.grad,
                eager_parameter.grad,
                rtol=1e-5,
                atol=1e-5,
            )

    def test_production_chunk_budget_keeps_request_batched_attention(self) -> None:
        attention = MixFormerCrossAttention(
            num_heads=8,
            dim=128,
            hidden_dim=512,
            sequence_chunk_tokens=8192,
        )
        self.assertEqual(
            attention._grouped_request_chunk(
                request_count=32,
                max_targets=16,
                sequence_length=8000,
            ),
            32,
        )
        self.assertEqual(attention._attention_length_chunk(8000), 8000)

    def test_fully_valid_history_matches_materialized_reference(self) -> None:
        torch.manual_seed(8)
        attention = MixFormerCrossAttention(num_heads=2, dim=4, hidden_dim=6)
        query = torch.randn(4, 2, 4, requires_grad=True)
        history = torch.randn(2, 5, 8, requires_grad=True)
        valid_mask = torch.ones(2, 5, dtype=torch.bool)
        row_indices = torch.tensor([0, 0, 1, 1])
        self.assertTrue(attention._history_is_dense(valid_mask))

        optimized = attention(query, history, valid_mask, row_indices)
        reference = attention.forward_reference(
            query,
            history,
            valid_mask,
            row_indices,
        )
        torch.testing.assert_close(optimized, reference, rtol=1e-5, atol=1e-6)


class MDLMixFormerInnovationTest(unittest.TestCase):
    def test_scenario_router_starts_as_exact_mixformer_identity(self) -> None:
        torch.manual_seed(11)
        router = ScenarioConditionedQueryRouter(num_heads=3, dim=4)
        queries = torch.randn(2, 3, 4)
        context = torch.randn(2, 4)

        output = router(queries, context)
        torch.testing.assert_close(output, queries)
        output.square().mean().backward()
        self.assertIsNotNone(router.context_delta.weight.grad)
        self.assertGreater(
            float(router.context_delta.weight.grad.abs().sum()),
            0.0,
        )

    def test_scenario_router_learns_head_gated_context(self) -> None:
        router = ScenarioConditionedQueryRouter(num_heads=2, dim=3)
        with torch.no_grad():
            router.context_delta.weight.copy_(torch.eye(3))
            router.head_gate.weight.zero_()
            router.head_gate.bias.zero_()
        queries = torch.zeros(2, 2, 3)
        context = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

        output = router(queries, context)

        self.assertFalse(torch.equal(output[0], output[1]))
        torch.testing.assert_close(output[:, 0], output[:, 1])


class MixFormerIntegrationTest(unittest.TestCase):
    @staticmethod
    def _compact_config(model_name: str):
        mdl = model_name == "mdl_mixformer"
        base_name = "mdl_mixformer.yaml" if mdl else "mixformer.yaml"
        config = load_app_config(ROOT / "configs" / base_name)
        config = replace(
            config,
            runtime=replace(
                config.runtime,
                attention_backend="sdpa",
                activation_checkpoint="none",
                cuda_graph_backbone=False,
                compile=False,
                require_compact_sequence_batches=False,
            ),
            tokenization=replace(
                config.tokenization,
                feature_tokenizer="rankmixer",
                feature_tokens=(),
                num_feature_tokens=4,
            ),
            model=replace(
                config.model,
                name=model_name,
                token_dim=32,
                num_layers=2,
                num_heads=4,
                hidden_dim=64,
                task_head_hidden_dim=64,
                first_domain_sequence_layer=None,
                mixformer_user_head_count=None,
                mixformer_user_item_decouple=False,
                mdl_mixformer_query_conditioning=True,
                experimental_model_acknowledged=mdl,
            ),
            training=replace(
                config.training,
                embedding_weight_dtype="fp32",
                gset=replace(config.training.gset, capacity=4096)
                if config.training.gset.enabled
                else config.training.gset,
            ),
        )
        config.validate()
        return config

    def test_compact_models_build_forward_and_backward_with_rlb(self) -> None:
        for model_name, expected_type in (
            ("mixformer", MixFormerModel),
            ("mdl_mixformer", MDLMixFormerModel),
        ):
            with self.subTest(model=model_name):
                config = self._compact_config(model_name)
                model = build_model(
                    config,
                    _synthetic_vocab_maps(config),
                    embedding_size_override=32,
                )
                self.assertIsInstance(model, expected_type)
                self.assertIsNotNone(model.tokenizer.sequence_type_embeddings)
                self.assertEqual(len(model.tokenizer.sep_tokens), 0)
                _replace_id_embeddings_with_synthetic(model)
                batch = _synthetic_feature_batch(
                    config,
                    torch.device("cpu"),
                    batch_size=4,
                    sequence_length=5,
                    seed=17,
                    candidates_per_request=2,
                )

                output = model(batch.features, batch.scenario_id)
                loss = output["logits"].square().mean()
                loss.backward()

                self.assertEqual(
                    tuple(output["logits"].shape),
                    (4, len(config.task_names)),
                )
                self.assertTrue(
                    all(
                        parameter.grad is None
                        or bool(
                            torch.isfinite(
                                parameter.grad._values()
                                if parameter.grad.is_sparse
                                else parameter.grad
                            ).all()
                        )
                        for parameter in model.parameters()
                    )
                )

    def test_compact_ui_mixformer_keeps_request_major_user_heads(self) -> None:
        config = self._compact_config("mixformer")
        config = replace(
            config,
            model=replace(
                config.model,
                mixformer_user_item_decouple=True,
            ),
        )
        config.validate()
        self.assertEqual(config.resolved.mixformer_user_head_count, 2)
        model = build_model(
            config,
            _synthetic_vocab_maps(config),
            embedding_size_override=32,
        )
        _replace_id_embeddings_with_synthetic(model)
        batch = _synthetic_feature_batch(
            config,
            torch.device("cpu"),
            batch_size=4,
            sequence_length=5,
            seed=19,
            candidates_per_request=2,
        )

        tokenized = model.tokenizer(batch.features)
        self.assertIsNotNone(tokenized.user_feature_heads)
        self.assertIsNotNone(tokenized.item_feature_heads)
        assert tokenized.user_feature_heads is not None
        assert tokenized.item_feature_heads is not None
        self.assertEqual(tokenized.user_feature_heads.size(0), 2)
        self.assertEqual(tokenized.item_feature_heads.size(0), 4)
        self.assertEqual(tokenized.user_feature_heads.size(1), 2)
        self.assertEqual(tokenized.item_feature_heads.size(1), 2)
        self.assertEqual(model.blocks[0].query_mixer.user_head_count, 2)

        output = model(batch.features, batch.scenario_id)
        loss = output["logits"].square().mean()
        loss.backward()
        self.assertEqual(tuple(output["logits"].shape), (4, len(config.task_names)))
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_compact_models_support_full_activation_checkpointing(self) -> None:
        for model_name in ("mixformer", "mdl_mixformer"):
            with self.subTest(model=model_name):
                config = self._compact_config(model_name)
                config = replace(
                    config,
                    runtime=replace(
                        config.runtime,
                        activation_checkpoint="full",
                    ),
                )
                config.validate()
                model = build_model(
                    config,
                    _synthetic_vocab_maps(config),
                    embedding_size_override=32,
                )
                _replace_id_embeddings_with_synthetic(model)
                batch = _synthetic_feature_batch(
                    config,
                    torch.device("cpu"),
                    batch_size=4,
                    sequence_length=5,
                    seed=23,
                    candidates_per_request=2,
                )

                loss = model(
                    batch.features,
                    batch.scenario_id,
                )["logits"].square().mean()
                loss.backward()

                self.assertTrue(bool(torch.isfinite(loss)))
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        for parameter in model.parameters()
                    )
                )

    def test_production_configs_preserve_current_data_contract(self) -> None:
        for config_name in (
            "mixformer.yaml",
            "mdl_mixformer.yaml",
            "mixformer_fine.yaml",
            "mdl_mixformer_fine.yaml",
        ):
            with self.subTest(config=config_name):
                config = load_app_config(ROOT / "configs" / config_name)
                resolved = config.resolved
                packed_width = sum(
                    resolved.encoded_input_dims[name]
                    for name in resolved.tokenization.feature_token_inputs
                )
                sequence_by_name = {
                    sequence.name: sequence
                    for sequence in config.sequences
                }
                active_sequence_names = {
                    input_name
                    for group in resolved.tokenization.sequence_token_groups
                    for input_name in group.input_refs
                }
                active_sequence_capacity = sum(
                    int(sequence_by_name[name].max_length or 0)
                    for name in active_sequence_names
                )
                self.assertEqual(packed_width, 1936)
                self.assertEqual(
                    resolved.tokenization.feature_token_count,
                    8,
                )
                self.assertEqual(
                    packed_width % resolved.tokenization.feature_token_count,
                    0,
                )
                self.assertEqual(
                    len(resolved.tokenization.sequence_token_groups),
                    8,
                )
                self.assertEqual(
                    config.model.sequence_fusion,
                    "timestamp_aware",
                )
                expected_global_limit = 8000
                self.assertEqual(
                    config.model.global_sequence_max_length,
                    expected_global_limit,
                )
                for name in active_sequence_names:
                    self.assertEqual(
                        sequence_by_name[name].tensor_max_length,
                        8000,
                    )
                for split in (config.data.train, config.data.test):
                    assert split is not None and split.adapter is not None
                    self.assertEqual(
                        split.adapter.options.get("global_sequence_max_length"),
                        expected_global_limit,
                    )
                self.assertFalse(config.model.use_sep_tokens)
                self.assertTrue(
                    all(
                        sequence_by_name[name].timestamp_field == "time"
                        and sequence_by_name[name].time_delta_field is None
                        for name in active_sequence_names
                    )
                )
                # Per-stream v3 windows sum to 41100. Timestamp-aware fusion
                # merges into an 8000 global window so a busy stream can borrow
                # unused capacity from empty ones.
                self.assertEqual(active_sequence_capacity, 41100)
                self.assertEqual(
                    config.model.token_dim
                    % resolved.tokenization.feature_token_count,
                    0,
                )
                self.assertEqual(config.model.token_dim, 128)
                self.assertEqual(config.model.num_layers, 4)
                self.assertEqual(config.model.hidden_dim, 512)
                self.assertEqual(config.model.num_heads, 8)
                self.assertEqual(config.training.dense_optimizer, "rmsprop")
                self.assertEqual(config.training.lr_dense, 1.0e-4)
                self.assertTrue(config.model.mixformer_user_item_decouple)
                self.assertEqual(resolved.mixformer_user_head_count, 4)
                if config_name in {"mixformer.yaml", "mixformer_fine.yaml"}:
                    self.assertEqual(config.data.train.reader.pack_unit, "agg_rows")
                    self.assertEqual(config.training.batch_size, 64)
                    self.assertEqual(config.training.gradient_accumulation_steps, 64)
                    self.assertEqual(
                        [bucket.batch_size for bucket in config.data.train.reader.length_buckets],
                        [64, 64, 64, 64, 64],
                    )
                    for split in (config.data.train, config.data.test):
                        assert split is not None and split.adapter is not None
                        self.assertTrue(
                            split.adapter.options.get("accumulate_agg_history")
                        )
                text = (ROOT / "configs" / config_name).read_text(
                    encoding="utf-8"
                )
                self.assertFalse(
                    any(
                        line.startswith("extends:")
                        for line in text.splitlines()
                    )
                )
                user_width = sum(
                    resolved.encoded_input_dims[name]
                    for name in resolved.mixformer_user_feature_inputs
                )
                item_width = sum(
                    resolved.encoded_input_dims[name]
                    for name in resolved.mixformer_item_feature_inputs
                )
                self.assertEqual(user_width, 624)
                self.assertEqual(item_width, 1312)
                self.assertEqual(user_width % 4, 0)
                self.assertEqual(item_width % 4, 0)


if __name__ == "__main__":
    unittest.main()
