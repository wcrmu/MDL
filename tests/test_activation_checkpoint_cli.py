from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from src.config import load_app_config
from src.main import _apply_runtime_overrides, _load_config, build_arg_parser


ROOT = Path(__file__).resolve().parents[1]


class ActivationCheckpointCliOverrideTest(unittest.TestCase):
    def test_parser_accepts_activation_checkpoint_on_train(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "train",
                "--config",
                "configs/rankmixer.yaml",
                "--activation-checkpoint",
                "selective",
            ]
        )
        self.assertEqual(args.activation_checkpoint, "selective")

    def test_runtime_override_replaces_yaml_value(self) -> None:
        config = load_app_config(ROOT / "configs" / "rankmixer.yaml")
        self.assertEqual(config.runtime.activation_checkpoint, "none")
        args = build_arg_parser().parse_args(
            [
                "train",
                "--config",
                str(ROOT / "configs" / "rankmixer.yaml"),
                "--activation-checkpoint",
                "full",
            ]
        )
        overridden = _apply_runtime_overrides(config, args)
        self.assertEqual(overridden.runtime.activation_checkpoint, "full")
        self.assertEqual(config.runtime.activation_checkpoint, "none")

    def test_load_config_applies_activation_checkpoint_override(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "validate-config",
                "--config",
                str(ROOT / "configs" / "rankmixer.yaml"),
                "--activation-checkpoint",
                "selective",
            ]
        )
        with patch(
            "src.main.load_app_config",
            return_value=load_app_config(ROOT / "configs" / "rankmixer.yaml"),
        ):
            config = _load_config(args)
        self.assertEqual(config.runtime.activation_checkpoint, "selective")

    def test_omitted_flag_leaves_yaml_unchanged(self) -> None:
        config = load_app_config(ROOT / "configs" / "rankmixer.yaml")
        args = build_arg_parser().parse_args(
            [
                "train",
                "--config",
                str(ROOT / "configs" / "rankmixer.yaml"),
            ]
        )
        self.assertIsNone(args.activation_checkpoint)
        same = _apply_runtime_overrides(config, args)
        self.assertIs(same, config)


if __name__ == "__main__":
    unittest.main()
