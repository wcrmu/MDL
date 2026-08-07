from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from src.config import load_app_config
from src.main import _apply_model_overrides, _load_config, build_arg_parser
from src.model import build_model
from src.train import _needs_padded_sdpa_flash


ROOT = Path(__file__).resolve().parents[1]


class MdlDomainReadCliOverrideTest(unittest.TestCase):
    def test_parser_accepts_mdl_domain_read_ns(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "train",
                "--config",
                "configs/mdl_onetrans.yaml",
                "--mdl-domain-read",
                "ns",
            ]
        )
        self.assertEqual(args.mdl_domain_read, "ns")

    def test_mdl_domain_read_ns_clears_sequence_layer(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_onetrans.yaml")
        self.assertEqual(config.model.first_domain_sequence_layer, 0)
        args = build_arg_parser().parse_args(
            [
                "train",
                "--config",
                str(ROOT / "configs" / "mdl_onetrans.yaml"),
                "--mdl-domain-read",
                "ns",
            ]
        )
        overridden = _apply_model_overrides(config, args)
        self.assertIsNone(overridden.model.first_domain_sequence_layer)
        self.assertEqual(config.model.first_domain_sequence_layer, 0)

    def test_mdl_domain_read_equal_sets_layer_zero(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_onetrans.yaml")
        ns_only = replace(
            config,
            model=replace(config.model, first_domain_sequence_layer=None),
        )
        args = build_arg_parser().parse_args(
            [
                "train",
                "--config",
                str(ROOT / "configs" / "mdl_onetrans.yaml"),
                "--mdl-domain-read",
                "equal",
            ]
        )
        overridden = _apply_model_overrides(ns_only, args)
        self.assertEqual(overridden.model.first_domain_sequence_layer, 0)

    def test_first_domain_sequence_layer_null_string(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_onetrans.yaml")
        args = build_arg_parser().parse_args(
            [
                "train",
                "--config",
                str(ROOT / "configs" / "mdl_onetrans.yaml"),
                "--first-domain-sequence-layer",
                "null",
            ]
        )
        overridden = _apply_model_overrides(config, args)
        self.assertIsNone(overridden.model.first_domain_sequence_layer)

    def test_load_config_applies_ns_domain_read(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "validate-config",
                "--config",
                str(ROOT / "configs" / "mdl_onetrans.yaml"),
                "--mdl-domain-read",
                "ns",
            ]
        )
        with patch(
            "src.main.load_app_config",
            return_value=load_app_config(ROOT / "configs" / "mdl_onetrans.yaml"),
        ):
            config = _load_config(args)
        self.assertIsNone(config.model.first_domain_sequence_layer)

    def test_conflicting_domain_read_flags_raise(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_onetrans.yaml")
        args = build_arg_parser().parse_args(
            [
                "train",
                "--config",
                str(ROOT / "configs" / "mdl_onetrans.yaml"),
                "--mdl-domain-read",
                "ns",
                "--first-domain-sequence-layer",
                "0",
            ]
        )
        with self.assertRaisesRegex(ValueError, "only one of"):
            _apply_model_overrides(config, args)

    def test_ns_only_builds_fixed_width_domain_attention(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_onetrans.yaml")
        config = replace(
            config,
            model=replace(config.model, first_domain_sequence_layer=None),
        )
        self.assertTrue(_needs_padded_sdpa_flash(config))
        model = build_model(config, {}, embedding_size_override=16)
        block = model.blocks[0]
        self.assertFalse(block.use_sequence_attention)
        self.assertIsNotNone(block.task_attention)
        self.assertIsNone(block.task_sequence_attention)

    def test_equal_read_skips_fixed_width_domain_attention(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_onetrans.yaml")
        self.assertEqual(config.model.first_domain_sequence_layer, 0)
        self.assertFalse(_needs_padded_sdpa_flash(config))
        model = build_model(config, {}, embedding_size_override=16)
        block = model.blocks[0]
        self.assertTrue(block.use_sequence_attention)
        self.assertIsNone(block.task_attention)
        self.assertIsNotNone(block.task_sequence_attention)


if __name__ == "__main__":
    unittest.main()
