from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import torch

from src.config import SequenceConfig, SequenceFieldConfig, load_app_config
from src.model import FeatureEncoderBank, RankMixerSliceTokenizer, build_model
from src.modules.stca import (
    STCASequenceCache,
    STCASequenceEncoder,
    SingleQueryTargetAttention,
    SwiGLUFFN,
)


ROOT = Path(__file__).resolve().parents[1]


class SwiGLUFFNTest(unittest.TestCase):
    def test_matches_paper_equation(self) -> None:
        module = SwiGLUFFN(2, expansion_ratio=1, bias=False).double()
        with torch.no_grad():
            module.up_projection.weight.copy_(
                torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float64)
            )
            module.gate_projection.weight.copy_(
                torch.tensor([[0.5, -1.0], [2.0, 1.0]], dtype=torch.float64)
            )
            module.output_projection.weight.copy_(
                torch.tensor([[1.5, -0.5], [0.25, 2.0]], dtype=torch.float64)
            )
        values = torch.tensor(
            [[1.0, -2.0], [0.5, 3.0]],
            dtype=torch.float64,
        )

        actual = module(values)

        up = values @ module.up_projection.weight.t()
        gate = values @ module.gate_projection.weight.t()
        expected = (up * (gate * torch.sigmoid(gate))) @ (
            module.output_projection.weight.t()
        )
        torch.testing.assert_close(actual, expected)


class SingleQueryTargetAttentionTest(unittest.TestCase):
    def test_reordered_attention_matches_materialized_kv_and_gradients(self) -> None:
        torch.manual_seed(11)
        attention = SingleQueryTargetAttention(
            dim=8,
            num_heads=2,
            bias=True,
        ).double()
        query = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
        history = torch.randn(3, 5, 8, dtype=torch.float64, requires_grad=True)
        valid_mask = torch.tensor(
            [
                [True, True, False, True, False],
                [False, True, True, True, True],
                [False, False, False, False, False],
            ]
        )
        probe = torch.randn(3, 8, dtype=torch.float64)
        parameters = tuple(attention.parameters())

        optimized = attention(query, history, valid_mask)
        optimized_gradients = torch.autograd.grad(
            (optimized * probe).sum(),
            (query, history, *parameters),
        )

        reference = attention.forward_reference(query, history, valid_mask)
        reference_gradients = torch.autograd.grad(
            (reference * probe).sum(),
            (query, history, *parameters),
        )

        torch.testing.assert_close(
            optimized,
            reference,
            rtol=1.0e-10,
            atol=1.0e-11,
        )
        for optimized_gradient, reference_gradient in zip(
            optimized_gradients,
            reference_gradients,
        ):
            torch.testing.assert_close(
                optimized_gradient,
                reference_gradient,
                rtol=1.0e-9,
                atol=1.0e-10,
            )

    def test_request_grouped_path_matches_explicit_rlb_replication_and_gradients(
        self,
    ) -> None:
        torch.manual_seed(14)
        attention = SingleQueryTargetAttention(
            dim=8,
            num_heads=2,
            bias=True,
        ).double()
        query = torch.randn(6, 8, dtype=torch.float64, requires_grad=True)
        history = torch.randn(3, 7, 8, dtype=torch.float64, requires_grad=True)
        valid_mask = torch.tensor(
            [
                [False, False, True, True, True, True, True],
                [False, True, True, True, False, True, True],
                [True, True, True, True, True, True, True],
            ]
        )
        # Deliberately unordered with unequal targets per request.
        row_indices = torch.tensor([2, 0, 2, 1, 0, 2])
        probe = torch.randn(6, 8, dtype=torch.float64)
        parameters = tuple(attention.parameters())

        grouped = attention(
            query,
            history,
            valid_mask,
            history_row_indices=row_indices,
        )
        grouped_gradients = torch.autograd.grad(
            (grouped * probe).sum(),
            (query, history, *parameters),
        )

        reference = attention.forward_reference(
            query,
            history,
            valid_mask,
            history_row_indices=row_indices,
        )
        reference_gradients = torch.autograd.grad(
            (reference * probe).sum(),
            (query, history, *parameters),
        )

        torch.testing.assert_close(grouped, reference, rtol=1.0e-10, atol=1.0e-11)
        for grouped_gradient, reference_gradient in zip(
            grouped_gradients,
            reference_gradients,
        ):
            torch.testing.assert_close(
                grouped_gradient,
                reference_gradient,
                rtol=1.0e-9,
                atol=1.0e-10,
            )

    def test_request_grouped_path_does_not_save_candidate_expanded_history(
        self,
    ) -> None:
        torch.manual_seed(15)
        attention = SingleQueryTargetAttention(dim=8, num_heads=2)
        query = torch.randn(5, 8, requires_grad=True)
        history = torch.randn(2, 7, 8, requires_grad=True)
        valid_mask = torch.ones(2, 7, dtype=torch.bool)
        row_indices = torch.tensor([1, 0, 1, 1, 0])
        saved_shapes: list[tuple[int, ...]] = []

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            saved_shapes.append(tuple(tensor.shape))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = attention(
                query,
                history,
                valid_mask,
                history_row_indices=row_indices,
            )
            output.square().sum().backward()

        self.assertNotIn((5, 7, 8), saved_shapes)

    def test_masked_and_empty_histories_have_defined_zero_contribution(self) -> None:
        torch.manual_seed(12)
        attention = SingleQueryTargetAttention(
            dim=8,
            num_heads=2,
            bias=True,
        )
        query = torch.randn(2, 8)
        history = torch.randn(2, 4, 8)
        no_valid_tokens = torch.zeros(2, 4, dtype=torch.bool)

        masked = attention(query, history, no_valid_tokens)
        empty = attention(
            query,
            history[:, :0, :],
            no_valid_tokens[:, :0],
        )

        torch.testing.assert_close(masked, torch.zeros_like(masked))
        torch.testing.assert_close(empty, torch.zeros_like(empty))
        self.assertTrue(bool(torch.isfinite(masked).all()))
        self.assertTrue(bool(torch.isfinite(empty).all()))

    def test_padding_values_cannot_change_output(self) -> None:
        torch.manual_seed(13)
        attention = SingleQueryTargetAttention(dim=8, num_heads=4)
        query = torch.randn(2, 8)
        history = torch.randn(2, 6, 8)
        valid_mask = torch.tensor(
            [
                [False, False, True, True, True, True],
                [False, True, True, False, True, True],
            ]
        )
        changed_padding = history.clone()
        changed_padding[~valid_mask] = (
            torch.randn_like(changed_padding[~valid_mask]) * 1.0e4
        )

        baseline = attention(query, history, valid_mask)
        changed = attention(query, changed_padding, valid_mask)

        torch.testing.assert_close(baseline, changed)


class STCASequenceEncoderTest(unittest.TestCase):
    def _encoder(
        self,
        *,
        checkpoint_layers: bool = False,
        history_chunk_tokens: int = 0,
    ) -> STCASequenceEncoder:
        return STCASequenceEncoder(
            dim=8,
            num_heads=2,
            num_layers=3,
            expansion_ratio=2,
            activation_checkpoint=checkpoint_layers,
            history_chunk_tokens=history_chunk_tokens,
        )

    def test_output_shape_fusion_widths_and_gradients(self) -> None:
        torch.manual_seed(21)
        encoder = self._encoder().double().train()
        history = torch.randn(
            4,
            7,
            8,
            dtype=torch.float64,
            requires_grad=True,
        )
        target = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
        valid_mask = torch.tensor(
            [
                [True, True, True, True, True, True, True],
                [False, False, True, True, True, True, True],
                [False, True, True, True, False, True, True],
                [False, False, False, False, False, False, False],
            ]
        )

        output = encoder(history, valid_mask, target)
        output.square().mean().backward()

        self.assertEqual(tuple(output.shape), (4, 8))
        self.assertEqual(
            [projection.in_features for projection in encoder.query_fusion_projections],
            [16, 24],
        )
        self.assertEqual(encoder.final_projection.in_features, 32)
        self.assertIsNotNone(history.grad)
        self.assertIsNotNone(target.grad)
        self.assertTrue(bool(torch.isfinite(history.grad).all()))
        self.assertTrue(bool(torch.isfinite(target.grad).all()))
        self.assertTrue(
            all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                for parameter in encoder.parameters()
            )
        )

    def test_full_stack_matches_literal_paper_equations(self) -> None:
        torch.manual_seed(20)
        encoder = self._encoder().double().eval()
        reference_encoder = self._encoder().double().eval()
        reference_encoder.load_state_dict(encoder.state_dict())
        history = torch.randn(
            3,
            5,
            8,
            dtype=torch.float64,
            requires_grad=True,
        )
        target = torch.randn(
            3,
            8,
            dtype=torch.float64,
            requires_grad=True,
        )
        reference_history = history.detach().clone().requires_grad_()
        reference_target = target.detach().clone().requires_grad_()
        valid_mask = torch.tensor(
            [
                [False, True, True, True, True],
                [True, True, True, True, True],
                [False, False, True, True, True],
            ]
        )

        actual = encoder(history, valid_mask, target)

        # Literal paper equations, deliberately written from raw parameters:
        # no STCA transform/attention/reference helper is called here.
        def literal_swiglu(
            module: SwiGLUFFN,
            values: torch.Tensor,
        ) -> torch.Tensor:
            up = torch.nn.functional.linear(
                values,
                module.up_projection.weight,
                module.up_projection.bias,
            )
            gate = torch.nn.functional.linear(
                values,
                module.gate_projection.weight,
                module.gate_projection.bias,
            )
            gated = up * (gate * torch.sigmoid(gate))
            return torch.nn.functional.linear(
                gated,
                module.output_projection.weight,
                module.output_projection.bias,
            )

        def literal_transform(layer, values: torch.Tensor) -> torch.Tensor:
            transformed = literal_swiglu(layer.input_ffn, values)
            return torch.nn.functional.layer_norm(
                transformed,
                (reference_encoder.dim,),
                layer.input_norm.weight,
                layer.input_norm.bias,
                layer.input_norm.eps,
            )

        def literal_materialized_attention(
            layer,
            query: torch.Tensor,
            transformed_history: torch.Tensor,
        ) -> torch.Tensor:
            attention = layer.attention
            projected_query = torch.nn.functional.linear(
                query,
                attention.query_projection.weight,
                attention.query_projection.bias,
            ).view(
                query.size(0),
                attention.num_heads,
                attention.head_dim,
            )
            keys = torch.nn.functional.linear(
                transformed_history,
                attention.key_projection.weight,
                attention.key_projection.bias,
            ).view(
                transformed_history.size(0),
                transformed_history.size(1),
                attention.num_heads,
                attention.head_dim,
            )
            values = torch.nn.functional.linear(
                transformed_history,
                attention.value_projection.weight,
                attention.value_projection.bias,
            ).view(
                transformed_history.size(0),
                transformed_history.size(1),
                attention.num_heads,
                attention.head_dim,
            )
            scores = torch.einsum(
                "bhd,blhd->bhl",
                projected_query,
                keys,
            ) / (attention.head_dim**0.5)
            weights = torch.softmax(
                scores.masked_fill(~valid_mask.unsqueeze(1), float("-inf")),
                dim=-1,
            )
            heads = torch.einsum("bhl,blhd->bhd", weights, values)
            return torch.nn.functional.linear(
                heads.flatten(start_dim=1),
                attention.output_projection.weight,
                attention.output_projection.bias,
            )

        outputs: list[torch.Tensor] = []
        query = literal_transform(
            reference_encoder.layers[0],
            reference_target,
        )
        for layer_index, layer in enumerate(reference_encoder.layers):
            transformed_history = literal_transform(
                layer,
                reference_history,
            )
            layer_output = literal_materialized_attention(
                layer,
                query,
                transformed_history,
            )
            outputs.append(layer_output)
            if layer_index + 1 < reference_encoder.num_layers:
                projection = reference_encoder.query_fusion_projections[layer_index]
                compressed = torch.nn.functional.linear(
                    torch.cat([*outputs, reference_target], dim=-1),
                    projection.weight,
                    projection.bias,
                )
                query = literal_transform(
                    reference_encoder.layers[layer_index + 1],
                    compressed,
                )
        final_input = torch.nn.functional.linear(
            torch.cat([*outputs, reference_target], dim=-1),
            reference_encoder.final_projection.weight,
            reference_encoder.final_projection.bias,
        )
        expected = literal_swiglu(reference_encoder.final_ffn, final_input)

        torch.testing.assert_close(actual, expected, rtol=1.0e-10, atol=1.0e-11)
        probe = torch.randn_like(actual)
        actual_gradients = torch.autograd.grad(
            (actual * probe).sum(),
            (history, target, *encoder.parameters()),
        )
        expected_gradients = torch.autograd.grad(
            (expected * probe).sum(),
            (
                reference_history,
                reference_target,
                *reference_encoder.parameters(),
            ),
        )
        for actual_gradient, expected_gradient in zip(
            actual_gradients,
            expected_gradients,
        ):
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                rtol=1.0e-9,
                atol=1.0e-10,
            )

    def test_request_row_mapping_matches_explicit_history_replication(self) -> None:
        torch.manual_seed(22)
        encoder = self._encoder().double().eval()
        request_history = torch.randn(2, 5, 8, dtype=torch.float64)
        request_mask = torch.tensor(
            [
                [False, True, True, True, True],
                [True, True, True, True, True],
            ]
        )
        row_indices = torch.tensor([1, 0, 1, 1, 0])
        targets = torch.randn(5, 8, dtype=torch.float64)

        request_batched = encoder(
            request_history,
            request_mask,
            targets,
            history_row_indices=row_indices,
        )
        explicitly_repeated = encoder(
            request_history.index_select(0, row_indices),
            request_mask.index_select(0, row_indices),
            targets,
        )

        torch.testing.assert_close(
            request_batched,
            explicitly_repeated,
            rtol=1.0e-10,
            atol=1.0e-11,
        )

    def test_request_cache_matches_uncached_forward_and_backward(self) -> None:
        torch.manual_seed(25)
        uncached = self._encoder().double().train()
        cached = self._encoder().double().train()
        cached.load_state_dict(uncached.state_dict())
        mask = torch.tensor(
            [
                [False, False, True, True, True, True],
                [False, True, True, True, True, True],
                [True, True, True, True, True, True],
            ]
        )
        row_indices = torch.tensor([2, 0, 2, 1, 0, 2])
        uncached_history = torch.randn(
            3,
            6,
            8,
            dtype=torch.float64,
            requires_grad=True,
        )
        uncached_target = torch.randn(
            6,
            8,
            dtype=torch.float64,
            requires_grad=True,
        )
        cached_history = uncached_history.detach().clone().requires_grad_()
        cached_target = uncached_target.detach().clone().requires_grad_()

        uncached_output = uncached(
            uncached_history,
            mask,
            uncached_target,
            history_row_indices=row_indices,
        )
        request_cache = cached.precompute_cache(cached_history, mask)
        self.assertIsInstance(request_cache, STCASequenceCache)
        self.assertTrue(
            all(
                tuple(item.shape) == (3, 6, 8)
                for item in request_cache.transformed_histories
            )
        )
        cached_output = cached(
            None,
            None,
            cached_target,
            history_row_indices=row_indices,
            cache=request_cache,
        )
        uncached_output.square().mean().backward()
        cached_output.square().mean().backward()

        torch.testing.assert_close(uncached_output, cached_output)
        torch.testing.assert_close(uncached_history.grad, cached_history.grad)
        torch.testing.assert_close(uncached_target.grad, cached_target.grad)
        for uncached_parameter, cached_parameter in zip(
            uncached.parameters(),
            cached.parameters(),
        ):
            torch.testing.assert_close(
                uncached_parameter.grad,
                cached_parameter.grad,
            )

    def test_checkpointed_layers_match_eager_forward_and_backward(self) -> None:
        torch.manual_seed(23)
        eager = self._encoder(checkpoint_layers=False).double().train()
        checkpointed = self._encoder(checkpoint_layers=True).double().train()
        checkpointed.load_state_dict(eager.state_dict())
        mask = torch.tensor(
            [
                [False, True, True, True, True],
                [True, True, True, True, True],
                [False, False, True, True, True],
            ]
        )
        eager_history = torch.randn(
            3,
            5,
            8,
            dtype=torch.float64,
            requires_grad=True,
        )
        eager_target = torch.randn(
            3,
            8,
            dtype=torch.float64,
            requires_grad=True,
        )
        checkpoint_history = eager_history.detach().clone().requires_grad_()
        checkpoint_target = eager_target.detach().clone().requires_grad_()

        eager_output = eager(eager_history, mask, eager_target)
        checkpoint_output = checkpointed(
            checkpoint_history,
            mask,
            checkpoint_target,
        )
        eager_output.sum().backward()
        checkpoint_output.sum().backward()

        torch.testing.assert_close(eager_output, checkpoint_output)
        torch.testing.assert_close(eager_history.grad, checkpoint_history.grad)
        torch.testing.assert_close(eager_target.grad, checkpoint_target.grad)
        for eager_parameter, checkpoint_parameter in zip(
            eager.parameters(),
            checkpointed.parameters(),
        ):
            torch.testing.assert_close(
                eager_parameter.grad,
                checkpoint_parameter.grad,
            )

    def test_chunked_history_swiglu_is_equation_and_gradient_preserving(self) -> None:
        torch.manual_seed(26)
        eager = self._encoder().double().train()
        chunked = self._encoder(history_chunk_tokens=5).double().train()
        chunked.load_state_dict(eager.state_dict())
        mask = torch.tensor(
            [
                [False, True, True, True, True, True],
                [True, True, True, True, True, True],
                [False, False, True, True, True, True],
            ]
        )
        eager_history = torch.randn(
            3,
            6,
            8,
            dtype=torch.float64,
            requires_grad=True,
        )
        eager_target = torch.randn(
            3,
            8,
            dtype=torch.float64,
            requires_grad=True,
        )
        chunked_history = eager_history.detach().clone().requires_grad_()
        chunked_target = eager_target.detach().clone().requires_grad_()

        eager_output = eager(eager_history, mask, eager_target)
        chunked_output = chunked(chunked_history, mask, chunked_target)
        eager_output.square().mean().backward()
        chunked_output.square().mean().backward()

        torch.testing.assert_close(eager_output, chunked_output)
        torch.testing.assert_close(eager_history.grad, chunked_history.grad)
        torch.testing.assert_close(eager_target.grad, chunked_target.grad)
        for eager_parameter, chunked_parameter in zip(
            eager.parameters(),
            chunked.parameters(),
        ):
            torch.testing.assert_close(eager_parameter.grad, chunked_parameter.grad)

    def test_empty_history_still_produces_finite_target_aware_token(self) -> None:
        torch.manual_seed(24)
        encoder = self._encoder().eval()
        history = torch.empty(2, 0, 8)
        mask = torch.empty(2, 0, dtype=torch.bool)
        target = torch.randn(2, 8)

        output = encoder(history, mask, target)

        self.assertEqual(tuple(output.shape), (2, 8))
        self.assertTrue(bool(torch.isfinite(output).all()))


class STCAConfigTest(unittest.TestCase):
    def test_sequence_config_requires_target_and_valid_hyperparameters(self) -> None:
        field = SequenceFieldConfig(
            name="item",
            kind="dense",
            source="hist_item",
            dimension=8,
        )
        missing_target = SequenceConfig(
            name="hist",
            fields=(field,),
            encoder="stca",
        )
        with self.assertRaisesRegex(ValueError, "requires target_inputs"):
            missing_target.validate({"target"})

        valid = replace(
            missing_target,
            target_inputs=("target",),
            stca_layers=4,
            stca_num_heads=8,
            stca_expansion_ratio=4,
        )
        valid.validate({"target"})

        attention_missing = SequenceConfig(
            name="hist",
            fields=(field,),
            encoder="attention_pool",
        )
        with self.assertRaisesRegex(ValueError, "requires target_inputs"):
            attention_missing.validate({"target"})
        replace(attention_missing, target_inputs=("target",)).validate({"target"})

        with self.assertRaisesRegex(ValueError, "stca_layers must be positive"):
            replace(valid, stca_layers=0).validate({"target"})
        with self.assertRaisesRegex(
            ValueError,
            "stca_expansion_ratio must be positive",
        ):
            replace(valid, stca_expansion_ratio=0).validate({"target"})
        time_only = replace(
            valid,
            fields=(
                SequenceFieldConfig(
                    name="time_delta",
                    kind="dense",
                    source="hist_time_delta",
                    dimension=1,
                ),
            ),
            time_delta_field="time_delta",
        )
        with self.assertRaisesRegex(ValueError, "STCA input requires"):
            time_only.validate({"target"})

    def test_rankmixer_config_accepts_stca_and_checks_head_divisibility(self) -> None:
        config = load_app_config(ROOT / "configs" / "reference" / "rankmixer.yaml")
        sequence = replace(
            config.sequences[0],
            encoder="stca",
            target_inputs=("item_id",),
            rankmixer_summary_tokens=1,
            longer_output="full",
            stca_layers=4,
            stca_num_heads=4,
            stca_expansion_ratio=4,
        )
        stca_config = replace(
            config,
            sequences=(sequence, *config.sequences[1:]),
        )

        stca_config.validate()
        self.assertEqual(
            stca_config.resolved.encoded_input_dims[sequence.name],
            sequence.stca_dim,
        )

        invalid = replace(
            stca_config,
            sequences=(
                replace(sequence, stca_num_heads=3),
                *config.sequences[1:],
            ),
        )
        with self.assertRaisesRegex(
            ValueError, "stca_dim must be divisible|resolved head count"
        ):
            invalid.validate()

    def test_parameter_group_shares_one_stca_stack_and_checks_signature(
        self,
    ) -> None:
        config = load_app_config(ROOT / "configs" / "reference" / "rankmixer.yaml")
        original = config.sequences[0]
        first = replace(
            original,
            encoder="stca",
            target_inputs=("item_id",),
            rankmixer_summary_tokens=1,
            stca_layers=2,
            stca_num_heads=4,
            stca_expansion_ratio=2,
            stca_parameter_group="shared_history",
        )
        time_delta = next(
            field
            for field in original.fields
            if field.name == original.time_delta_field
        )
        dense_side = next(
            field
            for field in original.fields
            if field.kind == "dense" and field.name != original.time_delta_field
        )
        second = replace(
            first,
            name="hist_copy",
            fields=(dense_side, time_delta),
            null_anchor_field=None,
        )
        shared_config = replace(config, sequences=(first, second))

        bank = FeatureEncoderBank(
            shared_config,
            {},
            shared_config.model.embedding_dim,
            embedding_size_override=32,
        )

        self.assertEqual(len(bank.sequence_stca_encoders), 1)
        self.assertEqual(
            bank.sequence_stca_encoder_keys[first.name],
            bank.sequence_stca_encoder_keys[second.name],
        )
        self.assertEqual(len(bank.sequence_query_projectors), 1)
        self.assertEqual(
            bank.sequence_query_projector_keys[first.name],
            bank.sequence_query_projector_keys[second.name],
        )

        incompatible = replace(
            shared_config,
            sequences=(first, replace(second, stca_layers=3)),
        )
        with self.assertRaisesRegex(ValueError, "incompatible"):
            incompatible.validate()

    def test_history_group_builds_one_globally_chronological_typed_history(
        self,
    ) -> None:
        config = load_app_config(ROOT / "configs" / "reference" / "rankmixer.yaml")
        signal = SequenceFieldConfig(
            name="signal",
            kind="dense",
            source="hist_signal",
            dimension=1,
        )
        time_delta = SequenceFieldConfig(
            name="time_delta",
            kind="dense",
            source="hist_time_delta",
            dimension=1,
        )
        owner = replace(
            config.sequences[0],
            fields=(signal, time_delta),
            max_length=3,
            encoder="stca",
            target_inputs=("item_id",),
            rankmixer_summary_tokens=1,
            stca_layers=2,
            stca_num_heads=4,
            stca_expansion_ratio=2,
            stca_parameter_group="merged",
            stca_history_group="merged",
            null_anchor_field=None,
        )
        auxiliary = replace(
            owner,
            name="hist_aux",
            fields=(
                replace(signal, source="hist_aux_signal"),
                replace(time_delta, source="hist_aux_time_delta"),
            ),
            max_length=2,
        )
        feature_inputs = tuple(
            feature.name
            for feature in config.features
            if feature.embedding_scope in {"feature", "shared"}
        ) + (owner.name,)
        grouped_config = replace(
            config,
            sequences=(owner, auxiliary),
            tokenization=replace(
                config.tokenization,
                feature_token_inputs=feature_inputs,
            ),
            vocab_strategy=replace(
                config.vocab_strategy,
                features={
                    name: strategy
                    for name, strategy in config.vocab_strategy.features.items()
                    if not name.startswith("hist.")
                },
            ),
        )
        grouped_config.validate()
        bank = FeatureEncoderBank(
            grouped_config,
            {},
            grouped_config.model.embedding_dim,
            embedding_size_override=32,
        )

        with torch.no_grad():
            for name, action_code in ((owner.name, 1.0), (auxiliary.name, 2.0)):
                key = bank._module_key(name)
                projector = bank.sequence_step_projectors[key]
                self.assertIsInstance(projector, torch.nn.Linear)
                projector.weight.zero_()
                projector.bias.zero_()
                projector.weight[0, 0] = 1.0
                action_type = bank.sequence_stca_type_embeddings[key]
                action_type.zero_()
                action_type[..., 1] = action_code
            position_key = bank.sequence_stca_history_position_keys[owner.name]
            positions = bank.sequence_position_embeddings[position_key].weight
            positions.zero_()
            positions[:, 2] = torch.arange(positions.size(0))

        tokens, mask, row_indices = bank._stca_history_group_tokens(
            owner.name,
            {
                owner.name: {
                    "fields": {
                        "signal": torch.tensor([[10.0, 20.0, 999.0]]),
                        "time_delta": torch.tensor([[5.0, 1.0, -999.0]]),
                    },
                    "lengths": torch.tensor([2]),
                },
                auxiliary.name: {
                    "fields": {
                        "signal": torch.tensor([[30.0, 40.0]]),
                        "time_delta": torch.tensor([[4.0, 2.0]]),
                    },
                    "lengths": torch.tensor([2]),
                },
            },
            None,
        )

        self.assertIsNone(row_indices)
        torch.testing.assert_close(
            mask,
            torch.tensor([[True, True, True, True, False]]),
        )
        torch.testing.assert_close(
            tokens[0, :, :3],
            torch.tensor(
                [
                    [10.0, 1.0, 0.0],
                    [30.0, 2.0, 1.0],
                    [40.0, 2.0, 2.0],
                    [20.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
        )


class STCARankMixerTokenizationTest(unittest.TestCase):
    def test_stca_z_occupies_one_dedicated_rankmixer_token(self) -> None:
        tokenizer = RankMixerSliceTokenizer(
            input_names=["regular", "z"],
            input_dims={"regular": 6, "z": 4},
            num_tokens=3,
            token_dim=4,
            dedicated_token_names=["z"],
        )
        assert tokenizer.projection is not None
        with torch.no_grad():
            tokenizer.projection.weight.zero_()
            tokenizer.projection.bias.zero_()
            tokenizer.projection.weight[:, :3, :].copy_(torch.eye(3).expand(2, -1, -1))
            tokenizer.dedicated_projections[0].weight.copy_(torch.eye(4))
            tokenizer.dedicated_projections[0].bias.zero_()
        output = tokenizer(
            {
                "regular": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
                "z": torch.tensor([[9.0, 8.0, 7.0, 6.0]]),
            }
        )
        expected = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 0.0],
                    [4.0, 5.0, 6.0, 0.0],
                    [9.0, 8.0, 7.0, 6.0],
                ]
            ]
        )
        torch.testing.assert_close(output, expected)


class STCAModelIntegrationTest(unittest.TestCase):
    @staticmethod
    def _config(model_name: str):
        config = load_app_config(ROOT / "configs" / "reference" / f"{model_name}.yaml")
        sequence = replace(
            config.sequences[0],
            encoder="stca",
            target_inputs=("item_id",),
            rankmixer_summary_tokens=1,
            stca_layers=2,
            stca_num_heads=4,
            stca_expansion_ratio=2,
        )
        config = replace(
            config,
            sequences=(sequence, *config.sequences[1:]),
            runtime=replace(
                config.runtime,
                device="cpu",
                precision="fp32",
                attention_backend="auto",
                activation_checkpoint="none",
                compile=False,
            ),
        )
        config.validate()
        return config

    @classmethod
    def _grouped_config(cls, model_name: str):
        config = cls._config(model_name)
        owner = replace(
            config.sequences[0],
            stca_parameter_group="merged_history",
            stca_history_group="merged_history",
        )
        auxiliary = replace(
            owner,
            name="hist_aux",
            fields=tuple(
                replace(field, source=f"hist_aux_{field.name}")
                for field in owner.fields
            ),
        )
        feature_inputs = tuple(
            feature.name
            for feature in config.features
            if feature.embedding_scope in {"feature", "shared"}
        ) + (owner.name,)
        vocab_features = dict(config.vocab_strategy.features)
        for field in auxiliary.fields:
            if field.kind != "categorical":
                continue
            vocab_features[field.qualified_name(auxiliary.name)] = replace(
                config.vocab_strategy.features[field.qualified_name(owner.name)],
                source=field.source,
            )
        grouped = replace(
            config,
            sequences=(owner, auxiliary, *config.sequences[1:]),
            tokenization=replace(
                config.tokenization,
                feature_token_inputs=feature_inputs,
            ),
            vocab_strategy=replace(
                config.vocab_strategy,
                features=vocab_features,
            ),
        )
        grouped.validate()
        return grouped

    @staticmethod
    def _request_indexed_features(
        config,
    ) -> tuple[dict[str, object], torch.Tensor]:
        candidate_count = 4
        request_count = 2
        sequence_length = 5
        row_indices = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        result: dict[str, object] = {}

        # IDs stay below embedding_size_override=32 so this remains a small,
        # deterministic CPU integration test independent of production tables.
        for feature in config.features:
            if feature.kind == "dense":
                result[feature.name] = torch.randn(
                    candidate_count,
                    feature.dimension,
                )
            elif feature.pooling == "mean":
                result[feature.name] = {
                    "values": torch.randint(1, 15, (candidate_count, 2)),
                    "lengths": torch.full(
                        (candidate_count,),
                        2,
                        dtype=torch.long,
                    ),
                }
            else:
                result[feature.name] = torch.randint(
                    1,
                    15,
                    (candidate_count,),
                )

        for sequence in config.sequences:
            fields: dict[str, torch.Tensor] = {}
            for field in sequence.fields:
                shape = (
                    (request_count, sequence_length)
                    if field.dimension == 1
                    else (
                        request_count,
                        sequence_length,
                        field.dimension,
                    )
                )
                fields[field.name] = (
                    torch.randint(1, 15, shape)
                    if field.kind == "categorical"
                    else torch.randn(shape)
                )
            result[sequence.name] = {
                "fields": fields,
                "lengths": torch.tensor([3, 5], dtype=torch.long),
                "row_indices": row_indices,
            }
        return result, row_indices

    @staticmethod
    def _expand_request_sequences(
        config,
        features: dict[str, object],
        row_indices: torch.Tensor,
    ) -> dict[str, object]:
        expanded = dict(features)
        for sequence in config.sequences:
            value = features[sequence.name]
            if not isinstance(value, dict):
                raise AssertionError("synthetic sequence must be a mapping")
            fields = value["fields"]
            if not isinstance(fields, dict):
                raise AssertionError("synthetic sequence fields must be a mapping")
            lengths = value["lengths"]
            if not isinstance(lengths, torch.Tensor):
                raise AssertionError("synthetic lengths must be a tensor")
            expanded[sequence.name] = {
                "fields": {
                    name: tensor.index_select(0, row_indices)
                    for name, tensor in fields.items()
                },
                "lengths": lengths.index_select(0, row_indices),
            }
        return expanded

    def test_rankmixer_families_forward_backward_and_request_batching(self) -> None:
        for model_name in ("rankmixer", "mdl_rankmixer"):
            with self.subTest(model=model_name):
                torch.manual_seed(31)
                config = self._config(model_name)
                indexed, row_indices = self._request_indexed_features(config)
                expanded = self._expand_request_sequences(
                    config,
                    indexed,
                    row_indices,
                )
                model = build_model(
                    config,
                    {},
                    embedding_size_override=32,
                )
                scenario_id = torch.zeros(4, dtype=torch.long)

                model.eval()
                with torch.no_grad():
                    request_batched = model(indexed, scenario_id)["logits"]
                    explicitly_repeated = model(expanded, scenario_id)["logits"]
                    request_cache = model.precompute_request_cache(indexed)
                    externally_cached = model(
                        indexed,
                        scenario_id,
                        request_cache=request_cache,
                    )["logits"]
                self.assertEqual(
                    tuple(request_batched.shape),
                    (4, len(config.task_names)),
                )
                self.assertIsInstance(
                    request_cache[config.sequences[0].name],
                    STCASequenceCache,
                )
                self.assertTrue(bool(torch.isfinite(request_batched).all()))
                torch.testing.assert_close(
                    request_batched,
                    explicitly_repeated,
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )
                torch.testing.assert_close(
                    request_batched,
                    externally_cached,
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )

                model.train()
                model.zero_grad(set_to_none=True)
                logits = model(indexed, scenario_id)["logits"]
                logits.square().mean().backward()
                stca_gradients = [
                    parameter.grad
                    for name, parameter in model.named_parameters()
                    if "sequence_stca_encoders" in name
                ]
                self.assertTrue(stca_gradients)
                self.assertTrue(
                    all(
                        gradient is not None and bool(torch.isfinite(gradient).all())
                        for gradient in stca_gradients
                    )
                )
                action_type_gradients = [
                    parameter.grad
                    for name, parameter in model.named_parameters()
                    if "sequence_stca_type_embeddings" in name
                ]
                self.assertTrue(action_type_gradients)
                self.assertTrue(
                    all(
                        gradient is not None and bool(torch.isfinite(gradient).all())
                        for gradient in action_type_gradients
                    )
                )

    def test_grouped_history_emits_one_z_for_both_rankmixer_families(self) -> None:
        for model_name in ("rankmixer", "mdl_rankmixer"):
            with self.subTest(model=model_name):
                torch.manual_seed(32)
                config = self._grouped_config(model_name)
                owner, auxiliary = config.sequences[:2]
                indexed, row_indices = self._request_indexed_features(config)
                expanded = self._expand_request_sequences(
                    config,
                    indexed,
                    row_indices,
                )
                model = build_model(
                    config,
                    {},
                    embedding_size_override=32,
                )
                scenario_id = torch.zeros(4, dtype=torch.long)

                self.assertEqual(len(model.encoder_bank.sequence_stca_encoders), 1)
                self.assertEqual(len(model.encoder_bank.sequence_query_projectors), 1)
                model.eval()
                with torch.no_grad():
                    encoded = model.encoder_bank(indexed)
                    request_batched = model(indexed, scenario_id)["logits"]
                    explicitly_repeated = model(expanded, scenario_id)["logits"]
                    request_cache = model.precompute_request_cache(indexed)
                    externally_cached = model(
                        indexed,
                        scenario_id,
                        request_cache=request_cache,
                    )["logits"]

                self.assertIn(owner.name, encoded)
                self.assertNotIn(auxiliary.name, encoded)
                self.assertEqual(set(request_cache), {owner.name})
                torch.testing.assert_close(
                    request_batched,
                    explicitly_repeated,
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )
                torch.testing.assert_close(
                    request_batched,
                    externally_cached,
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )

                model.train()
                model.zero_grad(set_to_none=True)
                model(indexed, scenario_id)["logits"].square().mean().backward()
                action_type_gradients = [
                    parameter.grad
                    for name, parameter in model.named_parameters()
                    if "sequence_stca_type_embeddings" in name
                ]
                self.assertEqual(len(action_type_gradients), 2)
                self.assertTrue(
                    all(
                        gradient is not None and bool(torch.isfinite(gradient).all())
                        for gradient in action_type_gradients
                    )
                )


if __name__ == "__main__":
    unittest.main()
