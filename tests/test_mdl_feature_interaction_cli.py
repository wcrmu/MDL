from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from src.config import load_app_config
from src.main import _apply_model_overrides, _load_config, build_arg_parser


ROOT = Path(__file__).resolve().parents[1]


class MdlFeatureInteractionCliOverrideTest(unittest.TestCase):
    def test_parser_accepts_mdl_feature_interaction_on_train(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "train",
                "--config",
                "configs/mdl_rankmixer.yaml",
                "--mdl-feature-interaction",
                "residual_ffn",
            ]
        )
        self.assertEqual(args.mdl_feature_interaction, "residual_ffn")

    def test_model_override_replaces_yaml_value(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        self.assertEqual(config.model.mdl_feature_interaction, "direct_ffn")
        args = build_arg_parser().parse_args(
            [
                "train",
                "--config",
                str(ROOT / "configs" / "mdl_rankmixer.yaml"),
                "--mdl-feature-interaction",
                "residual_ffn",
            ]
        )
        overridden = _apply_model_overrides(config, args)
        self.assertEqual(overridden.model.mdl_feature_interaction, "residual_ffn")
        self.assertEqual(config.model.mdl_feature_interaction, "direct_ffn")

    def test_load_config_applies_mdl_feature_interaction_override(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "validate-config",
                "--config",
                str(ROOT / "configs" / "mdl_rankmixer.yaml"),
                "--mdl-feature-interaction",
                "residual_ffn",
            ]
        )
        with patch(
            "src.main.load_app_config",
            return_value=load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml"),
        ):
            config = _load_config(args)
        self.assertEqual(config.model.mdl_feature_interaction, "residual_ffn")

    def test_omitted_flag_leaves_yaml_unchanged(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        args = build_arg_parser().parse_args(
            [
                "train",
                "--config",
                str(ROOT / "configs" / "mdl_rankmixer.yaml"),
            ]
        )
        self.assertIsNone(args.mdl_feature_interaction)
        same = _apply_model_overrides(config, args)
        self.assertIs(same, config)


if __name__ == "__main__":
    unittest.main()
