from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from src.config import (
    DEAD_CONSTANT_FEATURE_NAMES,
    REQUEST_SCENE_FEATURE_NAMES,
    TokenizationConfig,
    filter_dead_constant_feature_names,
    filter_scene_related_feature_names,
    is_dead_constant_feature_name,
    is_request_scene_feature_name,
    is_scene_related_feature_name,
    load_app_config,
    resolve_app_config,
)
from src.main import _load_config, build_arg_parser
from src.model import _consumed_scalar_feature_names

ROOT = Path(__file__).resolve().parents[1]

REQUEST_SCENE_FIELDS = tuple(sorted(REQUEST_SCENE_FEATURE_NAMES))
CANDIDATE_SCENE_FIELDS = (
    "scene_adj_ctr_15d_hn",
    "scene_adj_cvr_15d_hn",
    "scene_adj_cartcvr_15d_hn",
    "scene_cart_cnt_15d_hn",
    "goods_scene_clk_cnt_15d_hn",
)
SCENARIO_IMPRESSION_PRIOR_FIELDS = (
    "scenario_prior_scene_impr_cnt_15d_hit_hn",
    "scenario_prior_scene_impr_cnt_15d_hn",
)


class ExcludeSceneFeatureTokensTest(unittest.TestCase):
    def test_scene_name_matcher(self) -> None:
        self.assertTrue(is_scene_related_feature_name("scene_id_hn"))
        self.assertTrue(is_scene_related_feature_name("scene_adj_ctr_15d_hn"))
        self.assertTrue(is_scene_related_feature_name("goods_scene_clk_cnt_15d_hn"))
        self.assertFalse(is_scene_related_feature_name("sku_id_hn"))
        self.assertTrue(is_request_scene_feature_name("scene_id_hn"))
        self.assertTrue(is_request_scene_feature_name("scene_impr_cnt_15d_hn"))
        self.assertFalse(is_request_scene_feature_name("scene_adj_ctr_15d_hn"))
        self.assertFalse(is_request_scene_feature_name("goods_scene_clk_cnt_15d_hn"))
        self.assertTrue(is_dead_constant_feature_name("c_adj_ctr_15d_hn"))
        self.assertTrue(is_dead_constant_feature_name("clk_7d_page_elsns_hn"))
        self.assertFalse(is_dead_constant_feature_name("idx_c_adj_ctr_15d_hn"))
        # omit drops request-axis only; candidate×scene crosses stay.
        self.assertEqual(
            filter_scene_related_feature_names(
                [
                    "sku_id_hn",
                    "scene_id_hn",
                    "plat_hn",
                    "scene_cart_cnt_15d_hn",
                    "scene_adj_ctr_15d_hn",
                ]
            ),
            ("sku_id_hn", "plat_hn", "scene_cart_cnt_15d_hn", "scene_adj_ctr_15d_hn"),
        )
        self.assertEqual(
            filter_dead_constant_feature_names(
                ["sku_id_hn", "c_adj_ctr_15d_hn", "clk_7d_page_elsns_hn"]
            ),
            ("sku_id_hn",),
        )

    def test_schema_and_production_default_omit_scene_features(self) -> None:
        self.assertTrue(TokenizationConfig().omit_scene_features)
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        self.assertTrue(config.tokenization.omit_scene_features)
        resolved = resolve_app_config(config)
        feature_names = {feature.name for feature in config.features}
        for name in REQUEST_SCENE_FIELDS:
            self.assertNotIn(name, resolved.tokenization.feature_token_inputs)
            if name in DEAD_CONSTANT_FEATURE_NAMES:
                self.assertNotIn(name, feature_names)
            else:
                # Request scene stays in the feature contract (adapter axis /
                # scenario-important sources) but is omitted from the pack.
                self.assertIn(name, feature_names)
        for name in CANDIDATE_SCENE_FIELDS:
            self.assertIn(name, resolved.tokenization.feature_token_inputs)
        for name in DEAD_CONSTANT_FEATURE_NAMES:
            self.assertNotIn(name, resolved.tokenization.feature_token_inputs)
            self.assertNotIn(name, feature_names)
        tokens = {
            token.name: token for token in config.tokenization.scenario_tokens or ()
        }
        for token_name in ("search", "recommendation"):
            self.assertIn(
                "scenario_important_scene_id_hn",
                tokens[token_name].important_inputs,
            )
            for name in SCENARIO_IMPRESSION_PRIOR_FIELDS:
                self.assertIn(name, tokens[token_name].prior_inputs)
                self.assertIn(name, feature_names)
        self.assertNotIn(
            "scenario_important_scene_id_hn",
            tokens["global"].important_inputs,
        )
        for name in SCENARIO_IMPRESSION_PRIOR_FIELDS:
            self.assertNotIn(name, tokens["global"].prior_inputs)
        count = resolved.tokenization.feature_token_count
        self.assertEqual(count, 32)
        self.assertEqual(config.model.token_dim % count, 0)

    def test_onetrans_ns_pack_honors_omit_and_dead_filters(self) -> None:
        for config_name in ("onetrans.yaml", "mdl_onetrans.yaml"):
            with self.subTest(config=config_name):
                config = load_app_config(ROOT / "configs" / config_name)
                resolved = resolve_app_config(config)
                self.assertEqual(
                    list(resolved.scalar_feature_names),
                    [
                        name
                        for name in resolved.tokenization.feature_token_inputs
                        if name
                        not in {sequence.name for sequence in config.sequences}
                    ],
                )
                self.assertEqual(
                    REQUEST_SCENE_FEATURE_NAMES & set(resolved.scalar_feature_names),
                    set(),
                )
                self.assertEqual(
                    DEAD_CONSTANT_FEATURE_NAMES & set(resolved.scalar_feature_names),
                    set(),
                )

    def test_consumed_scalars_keep_longer_scene_id_only(self) -> None:
        # Pure RankMixer: request scene_id stays via LONGER user-global.
        rankmixer = load_app_config(ROOT / "configs" / "rankmixer.yaml")
        rankmixer_included = _consumed_scalar_feature_names(rankmixer)
        self.assertIn("scene_id_hn", rankmixer_included)
        self.assertNotIn("scene_impr_cnt_15d_hn", rankmixer_included)
        self.assertNotIn("scene_impr_cnt_15d_hit_hn", rankmixer_included)

        # MDL-RankMixer: scene goes through scenario importants, not LONGER.
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        included = _consumed_scalar_feature_names(config)
        self.assertNotIn("scene_id_hn", included)
        self.assertNotIn("scene_impr_cnt_15d_hn", included)
        self.assertNotIn("scene_impr_cnt_15d_hit_hn", included)
        self.assertTrue(included.isdisjoint(DEAD_CONSTANT_FEATURE_NAMES))
        self.assertIn("scenario_important_scene_id_hn", included)
        self.assertIn("scenario_prior_scene_impr_cnt_15d_hn", included)
        self.assertIn("scenario_prior_scene_impr_cnt_15d_hit_hn", included)

    def test_disabling_omit_keeps_scene_in_feature_pack(self) -> None:
        config = load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml")
        disabled = replace(
            config,
            tokenization=replace(
                config.tokenization,
                omit_scene_features=False,
            ),
        )
        resolved = resolve_app_config(disabled)
        omitted = resolve_app_config(config)
        self.assertIn("scene_id_hn", resolved.tokenization.feature_token_inputs)
        self.assertNotIn("scene_id_hn", omitted.tokenization.feature_token_inputs)
        self.assertGreater(
            len(resolved.tokenization.feature_token_inputs),
            len(omitted.tokenization.feature_token_inputs),
        )

    def test_legacy_yaml_key_still_accepted(self) -> None:
        tokenization = TokenizationConfig.from_mapping(
            {"exclude_scene_features_from_feature_tokens": True}
        )
        self.assertTrue(tokenization.omit_scene_features)
        tokenization_off = TokenizationConfig.from_mapping(
            {"exclude_scene_features_from_feature_tokens": False}
        )
        self.assertFalse(tokenization_off.omit_scene_features)

    def test_cli_override_can_disable_omit(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "validate-config",
                "--config",
                str(ROOT / "configs" / "mdl_rankmixer.yaml"),
                "--no-omit-scene-features",
            ]
        )
        with patch(
            "src.main.load_app_config",
            return_value=load_app_config(ROOT / "configs" / "mdl_rankmixer.yaml"),
        ):
            config = _load_config(args)
        self.assertFalse(config.tokenization.omit_scene_features)
        resolved = resolve_app_config(config)
        self.assertIn("scene_id_hn", resolved.tokenization.feature_token_inputs)


if __name__ == "__main__":
    unittest.main()
