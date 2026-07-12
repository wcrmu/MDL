from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from src.config import load_app_config


class ModelConfigOverlayTest(unittest.TestCase):
    def test_all_model_profiles_extend_and_validate(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected = {
            "default.yaml": "mdl_rankmixer",
            "rankmixer.yaml": "rankmixer",
            "mdl_rankmixer.yaml": "mdl_rankmixer",
            "onetrans.yaml": "onetrans",
            "mdl_onetrans.yaml": "mdl_onetrans",
            "longer.yaml": "longer",
            "rankmixer_paper.yaml": "rankmixer",
            "mdl_rankmixer_paper.yaml": "mdl_rankmixer",
            "onetrans_paper.yaml": "onetrans",
            "longer_paper.yaml": "longer",
        }
        for filename, model_name in expected.items():
            with self.subTest(filename=filename):
                config = load_app_config(root / "configs" / filename)
                self.assertEqual(config.model.name, model_name)

        experimental = load_app_config(root / "configs" / "mdl_onetrans.yaml")
        self.assertTrue(experimental.model.experimental_model_acknowledged)

        rankmixer = load_app_config(root / "configs" / "rankmixer_paper.yaml")
        self.assertEqual(rankmixer.model.token_dim, 768)
        self.assertEqual(rankmixer.training.lr_dense, 0.01)
        self.assertEqual(rankmixer.tokenization.feature_tokenizer, "rankmixer")
        self.assertEqual(rankmixer.resolved.encoded_input_dims["hist"], 33_792)

        mdl = load_app_config(root / "configs" / "mdl_rankmixer_paper.yaml")
        self.assertEqual(len(mdl.scenarios.names), 3)
        self.assertEqual(len(mdl.task_names), 3)
        self.assertEqual(mdl.model.mdl_feature_interaction, "paper")
        self.assertEqual(mdl.training.loss_reduction, "sum")

        onetrans = load_app_config(root / "configs" / "onetrans_paper.yaml")
        self.assertEqual(onetrans.model.sequence_fusion, "timestamp_aware")
        self.assertEqual(onetrans.model.num_layers, 6)
        self.assertEqual(onetrans.model.token_dim, 256)
        self.assertEqual(onetrans.model.final_s_tokens, 12)

        longer = load_app_config(root / "configs" / "longer_paper.yaml")
        self.assertEqual(longer.sequences[0].sequence_order, "newest_to_oldest")
        self.assertEqual(longer.sequences[0].longer_token_merge, 8)
        self.assertEqual(longer.sequences[0].longer_query_tokens, 100)
        self.assertEqual(longer.resolved.encoded_input_dims["hist"], 26_368)

    def test_sparse_dtsi_requires_explicit_unpublished_output_policy(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_app_config(root / "configs" / "rankmixer.yaml")
        sparse_model = replace(
            config.model,
            rankmixer_ffn_type="sparse_moe",
            sparse_moe_use_dtsi=True,
            sparse_moe_dtsi_training_output=None,
        )

        with self.assertRaisesRegex(ValueError, "does not publish"):
            sparse_model.validate()


if __name__ == "__main__":
    unittest.main()
