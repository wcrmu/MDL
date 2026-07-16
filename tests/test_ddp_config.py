from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.config import DDPConfig
from src.train import DistributedContext, _maybe_compile_model, _prepare_forward_model


class DDPConfigTest(unittest.TestCase):
    def test_safe_default_is_dynamic_with_unused_detection(self) -> None:
        config = DDPConfig()
        config.validate()
        self.assertFalse(config.static_graph)
        self.assertTrue(config.find_unused_parameters)

    def test_static_graph_cannot_contradict_unused_detection(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires find_unused_parameters=false"):
            DDPConfig(
                static_graph=True,
                find_unused_parameters=True,
                validated_static_graph=True,
            ).validate()

    def test_optimized_modes_require_recorded_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "validated_no_unused_parameters"):
            DDPConfig(find_unused_parameters=False).validate()
        with self.assertRaisesRegex(ValueError, "validated_static_graph"):
            DDPConfig(
                static_graph=True,
                find_unused_parameters=False,
            ).validate()
        DDPConfig(
            find_unused_parameters=False,
            validated_no_unused_parameters=True,
        ).validate()
        DDPConfig(
            static_graph=True,
            find_unused_parameters=False,
            validated_static_graph=True,
        ).validate()

    def test_compile_wraps_ddp_before_invoking_dynamo(self) -> None:
        base_model = nn.Linear(2, 1)
        ddp_wrapper = nn.Sequential(base_model)
        events: list[tuple[str, nn.Module]] = []
        config = SimpleNamespace(
            runtime=SimpleNamespace(compile=True),
            training=SimpleNamespace(
                ddp=DDPConfig(
                    static_graph=True,
                    find_unused_parameters=False,
                    validated_static_graph=True,
                )
            ),
        )
        context = DistributedContext(
            enabled=True,
            rank=0,
            local_rank=0,
            world_size=2,
            device=torch.device("cpu"),
        )

        def fake_ddp(model: nn.Module, **_kwargs: object) -> nn.Module:
            events.append(("ddp", model))
            return ddp_wrapper

        def fake_compile(_config: object, model: nn.Module) -> nn.Module:
            events.append(("compile", model))
            return model

        with patch("src.train.DistributedDataParallel", side_effect=fake_ddp), patch(
            "src.train._maybe_compile_model", side_effect=fake_compile
        ):
            actual = _prepare_forward_model(config, base_model, context)

        self.assertIs(actual, ddp_wrapper)
        self.assertEqual(events, [("ddp", base_model), ("compile", ddp_wrapper)])

    def test_reduce_overhead_compile_mode_is_forwarded(self) -> None:
        model = nn.Linear(2, 1)
        config = SimpleNamespace(
            runtime=SimpleNamespace(
                compile=True,
                compile_mode="reduce-overhead",
            )
        )
        with patch("src.train.torch.compile", return_value=model) as compile_model:
            actual = _maybe_compile_model(config, model)

        self.assertIs(actual, model)
        compile_model.assert_called_once_with(model, mode="reduce-overhead")


if __name__ == "__main__":
    unittest.main()
