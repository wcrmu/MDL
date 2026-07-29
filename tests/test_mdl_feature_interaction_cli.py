from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from src.config import ModelConfig, load_app_config
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
                "direct_ffn",
            ]
        )
        self.assertEqual(args.mdl_feature_interaction, "direct_ffn")

    def test_model_override_replaces_yaml_value(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        self.assertEqual(config.model.mdl_feature_interaction, "residual_ffn")
        args = build_arg_parser().parse_args(
            [
                "train",
                "--config",
                str(ROOT / "configs" / "mdl_rankmixer.yaml"),
                "--mdl-feature-interaction",
                "direct_ffn",
            ]
        )
        overridden = _apply_model_overrides(config, args)
        self.assertEqual(overridden.model.mdl_feature_interaction, "direct_ffn")
        self.assertEqual(config.model.mdl_feature_interaction, "residual_ffn")

    def test_load_config_applies_mdl_feature_interaction_override(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "validate-config",
                "--config",
                str(ROOT / "configs" / "mdl_rankmixer.yaml"),
                "--mdl-feature-interaction",
                "direct_ffn",
            ]
        )
        with patch(
            "src.main.load_app_config",
            return_value=load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml"),
        ):
            config = _load_config(args)
        self.assertEqual(config.model.mdl_feature_interaction, "direct_ffn")

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
        self.assertEqual(same.model.mdl_feature_interaction, "residual_ffn")

    def test_parser_and_override_accept_split_token_state(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        self.assertEqual(config.model.mdl_token_state, "coupled")
        args = build_arg_parser().parse_args(
            [
                "train",
                "--config",
                str(ROOT / "configs" / "mdl_rankmixer.yaml"),
                "--mdl-token-state",
                "split",
            ]
        )

        self.assertEqual(args.mdl_token_state, "split")
        overridden = _apply_model_overrides(config, args)
        self.assertEqual(overridden.model.mdl_token_state, "split")
        self.assertEqual(config.model.mdl_token_state, "coupled")

    def test_all_production_mdl_yamls_name_coupled_mode_explicitly(self) -> None:
        for config_name in (
            "mdl_rankmixer.yaml",
            "mdl_rankmixer_fine.yaml",
            "mdl_onetrans.yaml",
            "mdl_onetrans_fine.yaml",
        ):
            with self.subTest(config=config_name):
                config = load_app_config(ROOT / "configs" / config_name)
                self.assertEqual(config.model.mdl_token_state, "coupled")

    def test_load_config_applies_split_token_state(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "validate-config",
                "--config",
                str(ROOT / "configs" / "mdl_rankmixer.yaml"),
                "--mdl-token-state",
                "split",
            ]
        )
        config = _load_config(args)
        self.assertEqual(config.model.mdl_token_state, "split")

    def test_split_rejects_paths_that_reintroduce_prompt_bypass(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires mdl_rankmixer"):
            ModelConfig(name="rankmixer", mdl_token_state="split").validate()
        with self.assertRaisesRegex(
            ValueError,
            "use_task_feature_interaction=true",
        ):
            ModelConfig(
                name="mdl_rankmixer",
                mdl_token_state="split",
                use_task_feature_interaction=False,
            ).validate()
        with self.assertRaisesRegex(ValueError, "scene_feature_bias=none"):
            ModelConfig(
                name="mdl_rankmixer",
                mdl_token_state="split",
                scene_feature_bias="film",
            ).validate()


if __name__ == "__main__":
    unittest.main()
