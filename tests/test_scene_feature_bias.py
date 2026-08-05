from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import torch

from src.config import ModelConfig, load_app_config
from src.main import _apply_model_overrides, build_arg_parser
from src.model import SceneFeatureBias, build_model
from tests.test_build_production_configs import (
    _compact_production_config,
    _synthetic_model_features,
)


ROOT = Path(__file__).resolve().parents[1]


class SceneFeatureBiasConfigTest(unittest.TestCase):
    def test_schema_default_is_none(self) -> None:
        self.assertEqual(ModelConfig(name="mdl_rankmixer").scene_feature_bias, "none")
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        self.assertEqual(config.model.scene_feature_bias, "none")

    def test_rejects_non_mdl_models(self) -> None:
        with self.assertRaisesRegex(ValueError, "mdl_rankmixer or mdl_onetrans"):
            ModelConfig(name="rankmixer", scene_feature_bias="additive").validate()

    def test_requires_scenario_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "use_scenario_tokens"):
            ModelConfig(
                name="mdl_rankmixer",
                scene_feature_bias="film",
                use_scenario_tokens=False,
            ).validate()

    def test_cli_override(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        args = build_arg_parser().parse_args(
            [
                "train",
                "--config",
                str(ROOT / "configs" / "mdl_rankmixer.yaml"),
                "--scene-feature-bias",
                "film",
            ]
        )
        overridden = _apply_model_overrides(config, args)
        self.assertEqual(overridden.model.scene_feature_bias, "film")
        self.assertEqual(config.model.scene_feature_bias, "none")


class SceneFeatureBiasModuleTest(unittest.TestCase):
    def test_zero_init_is_identity(self) -> None:
        bias = SceneFeatureBias(token_dim=8, mode="additive")
        features = torch.randn(2, 3, 8)
        scenarios = torch.randn(2, 2, 8)
        mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        out = bias(features, scenarios, mask)
        self.assertTrue(torch.allclose(out, features))

        film = SceneFeatureBias(token_dim=8, mode="film")
        out_film = film(features, scenarios, mask)
        self.assertTrue(torch.allclose(out_film, features))

    def test_mdl_rankmixer_forward_with_modes(self) -> None:
        torch.manual_seed(0)
        base = _compact_production_config("mdl_rankmixer")
        features = _synthetic_model_features(base)
        scenario_id = torch.tensor([0, 1], dtype=torch.long)

        for mode in ("none", "additive", "film"):
            with self.subTest(mode=mode):
                config = replace(
                    base,
                    model=replace(base.model, scene_feature_bias=mode),
                )
                config.model.validate()
                model = build_model(config, {}, embedding_size_override=16).train()
                if mode == "none":
                    self.assertIsNone(model.scene_feature_bias)
                else:
                    self.assertIsNotNone(model.scene_feature_bias)
                    self.assertEqual(model.scene_feature_bias.mode, mode)
                logits = model(features, scenario_id=scenario_id)["logits"]
                self.assertEqual(tuple(logits.shape), (2, 3))
                self.assertTrue(bool(torch.isfinite(logits).all()))
                logits.square().mean().backward()


if __name__ == "__main__":
    unittest.main()
