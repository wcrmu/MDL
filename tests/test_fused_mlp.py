from __future__ import annotations

import copy
import unittest

import torch

from src.modules.mlp import (
    PerTokenFFN,
    PerTokenLinear,
    SparseMoEPerTokenFFN,
    StackedPerTokenFFN,
)


class StackedPerTokenTest(unittest.TestCase):
    def test_linear_matches_independent_reference_layers(self) -> None:
        torch.manual_seed(7)
        module = PerTokenLinear(3, 4, 5)
        tokens = torch.randn(2, 3, 4, requires_grad=True)
        expected = torch.stack(
            [
                torch.nn.functional.linear(
                    tokens[:, index],
                    module.weight[index],
                    module.bias[index],
                )
                for index in range(3)
            ],
            dim=1,
        )
        actual = module(tokens)
        torch.testing.assert_close(actual, expected)

    def test_ffn_matches_module_list_forward_and_gradients(self) -> None:
        torch.manual_seed(11)
        reference = PerTokenFFN(
            4, 6, 9, dropout=0.0, activation="gelu"
        )
        fused = StackedPerTokenFFN(
            4, 6, 9, dropout=0.0, activation="gelu"
        )
        with torch.no_grad():
            for token_index, network in enumerate(reference.networks):
                fused.input_weight[token_index].copy_(network[0].weight)
                fused.input_bias[token_index].copy_(network[0].bias)
                fused.output_weight[token_index].copy_(network[3].weight)
                fused.output_bias[token_index].copy_(network[3].bias)

        reference_input = torch.randn(5, 4, 6, requires_grad=True)
        fused_input = reference_input.detach().clone().requires_grad_(True)
        reference_output = reference(reference_input)
        fused_output = fused(fused_input)
        torch.testing.assert_close(fused_output, reference_output)

        reference_output.square().sum().backward()
        fused_output.square().sum().backward()
        torch.testing.assert_close(fused_input.grad, reference_input.grad)
        for token_index, network in enumerate(reference.networks):
            torch.testing.assert_close(
                fused.input_weight.grad[token_index], network[0].weight.grad
            )
            torch.testing.assert_close(
                fused.output_weight.grad[token_index], network[3].weight.grad
            )

    def test_compatible_ffn_keeps_state_keys_and_matches_explicit_loop(self) -> None:
        torch.manual_seed(13)
        module = PerTokenFFN(3, 5, 7, dropout=0.0, activation="gelu")
        tokens = torch.randn(4, 3, 5)
        expected = torch.cat(
            [
                network(tokens[:, token_index, :]).unsqueeze(1)
                for token_index, network in enumerate(module.networks)
            ],
            dim=1,
        )

        actual = module(tokens)
        batched = module._forward_batched(tokens)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(batched, expected)
        self.assertIn("networks.0.0.weight", module.state_dict())
        self.assertIn("networks.2.3.bias", module.state_dict())

    def _assert_moe_batched_matches_looped(
        self, *, training: bool, dtsi_output, use_dtsi: bool
    ) -> None:
        torch.manual_seed(17)
        looped = SparseMoEPerTokenFFN(
            num_tokens=4,
            token_dim=6,
            hidden_dim=9,
            num_experts=3,
            dropout=0.0,
            use_dtsi=use_dtsi,
            dtsi_training_output=dtsi_output,
            inference_threshold=0.0,
        )
        batched = copy.deepcopy(looped)
        looped.train(training)
        batched.train(training)

        reference_input = torch.randn(5, 4, 6, requires_grad=True)
        batched_input = reference_input.detach().clone().requires_grad_(True)

        ref_out = looped._forward_looped(reference_input)
        bat_out = batched._forward_batched(batched_input)
        torch.testing.assert_close(bat_out, ref_out)
        torch.testing.assert_close(
            batched.regularization_loss(), looped.regularization_loss()
        )
        torch.testing.assert_close(batched.active_ratio(), looped.active_ratio())

        ref_out.square().sum().backward()
        bat_out.square().sum().backward()
        torch.testing.assert_close(batched_input.grad, reference_input.grad)
        ref_params = dict(looped.named_parameters())
        for name, param in batched.named_parameters():
            torch.testing.assert_close(
                param.grad, ref_params[name].grad, msg=f"grad mismatch for {name}"
            )

    def test_sparse_moe_batched_matches_looped_training_dense_router(self) -> None:
        self._assert_moe_batched_matches_looped(
            training=True, dtsi_output="dense_router", use_dtsi=True
        )

    def test_sparse_moe_batched_matches_looped_training_mean(self) -> None:
        self._assert_moe_batched_matches_looped(
            training=True, dtsi_output="mean", use_dtsi=True
        )

    def test_sparse_moe_batched_matches_looped_training_no_dtsi(self) -> None:
        self._assert_moe_batched_matches_looped(
            training=True, dtsi_output=None, use_dtsi=False
        )

    def test_sparse_moe_batched_matches_looped_inference(self) -> None:
        self._assert_moe_batched_matches_looped(
            training=False, dtsi_output="dense_router", use_dtsi=True
        )

    def test_batched_modules_keep_autocast_output_dtype(self) -> None:
        tokens = torch.randn(2, 3, 8)
        modules = (
            PerTokenLinear(3, 8, 8),
            PerTokenFFN(3, 8, 16),
            StackedPerTokenFFN(3, 8, 16),
        )

        with torch.autocast("cpu", dtype=torch.bfloat16):
            outputs = [module(tokens) for module in modules]

        self.assertTrue(
            all(output.dtype == torch.bfloat16 for output in outputs)
        )


if __name__ == "__main__":
    unittest.main()
