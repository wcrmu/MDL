from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import torch

from src.benchmark import (
    _replace_id_embeddings_with_synthetic,
    _synthetic_feature_batch,
    _synthetic_vocab_maps,
)
from src.config import load_app_config
from src.model import (
    MDLMixFormerModel,
    MixFormerFeatureHeadProjector,
    MixFormerModel,
    ScenarioConditionedQueryRouter,
    build_model,
)
from src.modules.mixformer import (
    MixFormerCrossAttention,
    MixFormerHeadMixing,
    MixFormerQueryMixer,
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
                mdl_mixformer_query_conditioning=True,
                experimental_model_acknowledged=mdl,
            ),
            training=replace(
                config.training,
                embedding_weight_dtype="fp32",
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
                        or bool(torch.isfinite(parameter.grad).all())
                        for parameter in model.parameters()
                    )
                )

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
                self.assertEqual(packed_width, 3216)
                self.assertEqual(
                    resolved.tokenization.feature_token_count,
                    16,
                )
                self.assertEqual(
                    packed_width % resolved.tokenization.feature_token_count,
                    0,
                )
                self.assertEqual(
                    len(resolved.tokenization.sequence_token_groups),
                    9,
                )
                self.assertEqual(
                    config.model.sequence_fusion,
                    "timestamp_aware",
                )
                paper_mixformer = not config_name.startswith("mdl_")
                expected_global_limit = 2048 if paper_mixformer else None
                self.assertEqual(
                    config.model.global_sequence_max_length,
                    expected_global_limit,
                )
                for name in active_sequence_names:
                    expected_transport = (
                        2048
                        if paper_mixformer
                        else sequence_by_name[name].max_length
                    )
                    self.assertEqual(
                        sequence_by_name[name].tensor_max_length,
                        expected_transport,
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
                        sequence_by_name[name].time_delta_field
                        == "time_delta_log1p_seconds"
                        for name in active_sequence_names
                    )
                )
                self.assertEqual(active_sequence_capacity, 2048)
                self.assertEqual(
                    config.model.token_dim
                    % resolved.tokenization.feature_token_count,
                    0,
                )
                self.assertEqual(config.model.token_dim, 384)
                self.assertEqual(config.model.num_layers, 4)
                self.assertEqual(config.model.hidden_dim, 1024)
                self.assertEqual(config.training.dense_optimizer, "rmsprop")
                self.assertEqual(config.training.lr_dense, 0.01)


if __name__ == "__main__":
    unittest.main()
