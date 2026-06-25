import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mdl.data.encoded import EncodedTabularDataset, collate_tabular_batch, load_manifest
from adapters.tenrec import prepare_tenrec
from mdl.tokenization import FeatureCompilerConfig, FeatureTokenCompiler
from mdl.training import multitask_bce_loss
from mdl.tabular_model import TabularMDLModel, config_from_manifest


class TenrecPipelineTest(unittest.TestCase):
    def fixture_dir(self) -> Path:
        return Path(__file__).parent / "fixtures" / "tenrec"

    def test_prepare_tenrec_outputs_encoded_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = prepare_tenrec(self.fixture_dir(), tmpdir)

            self.assertEqual(manifest["total_rows"], 20)
            self.assertEqual(manifest["splits"], {"train": 16, "val": 2, "test": 2})
            self.assertEqual(manifest["scenario_names"], ["QK-article", "QK-video"])
            self.assertTrue((Path(tmpdir) / "manifest.json").exists())
            self.assertTrue((Path(tmpdir) / "train.csv").exists())
            self.assertEqual(manifest["tokenization"]["kind"], "encoder_registry")
            self.assertEqual(len(manifest["tokenization"]["features"]), 17)
            self.assertEqual(len(manifest["tokenization"]["token_specs"]), 4)


    def test_feature_token_compiler_uses_encoder_registry_config(self) -> None:
        config = FeatureCompilerConfig(
            feature_specs=[{"name": "image_embedding", "encoder": "dense_vector", "dim": 3}],
            token_specs=[{"token_id": 0, "projection": "linear", "inputs": ["image_embedding"]}],
            token_dim=5,
        )
        compiler = FeatureTokenCompiler(config)
        feature_tokens = compiler({"image_embedding": torch.randn(2, 3)})

        self.assertEqual(feature_tokens.shape, (2, 1, 5))

    def test_encoded_dataset_and_tabular_model_forward(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prepare_tenrec(self.fixture_dir(), tmpdir)
            manifest = load_manifest(tmpdir)
            dataset = EncodedTabularDataset(tmpdir, "train")
            loader = DataLoader(dataset, batch_size=4, collate_fn=collate_tabular_batch)
            batch = next(iter(loader))

            config = config_from_manifest(
                manifest,
                embedding_dim=8,
                token_dim=24,
                num_layers=1,
                num_heads=4,
                ffn_hidden_dim=24,
            )
            model = TabularMDLModel(config)
            feature_tokens = model.compile_feature_tokens(batch["features"])
            output = model.forward_tokens(feature_tokens, batch["scenario_id"])
            logits = output["logits"]

            self.assertEqual(feature_tokens.shape, (4, len(config.token_specs), 24))
            self.assertEqual(logits.shape, (4, len(manifest["task_names"])))
            loss = multitask_bce_loss(logits, batch["labels"], batch["label_mask"])
            loss.backward()
            self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
