from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from src.benchmark import (
    _replace_id_embeddings_with_synthetic,
    _synthetic_feature_batch,
    _synthetic_vocab_maps,
)
from src.config import load_app_config
from src.more_model import MOREModel
from src.uniformer_model import UniFormerModel
from src.model import build_model
from src.train import _maybe_compile_model


ROOT = Path(__file__).resolve().parents[1]


class CurrentFeatureContractTest(unittest.TestCase):
    def test_empty_history_backward_and_checkpoint_roundtrip(self) -> None:
        for name in ("uniformer", "more"):
            with self.subTest(model=name):
                config = _compact(name)
                model = build_model(config, _synthetic_vocab_maps(config), embedding_size_override=32)
                _replace_id_embeddings_with_synthetic(model)
                batch = _synthetic_feature_batch(config, torch.device("cpu"), 4, 3, 17, 2)
                for sequence in config.sequences:
                    batch.features[sequence.name]["lengths"].zero_()
                logits = model(batch.features, batch.scenario_id)["logits"]
                self.assertTrue(torch.isfinite(logits).all())
                logits.square().mean().backward()
                missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
                self.assertEqual(missing, [])
                restored = build_model(config, _synthetic_vocab_maps(config), embedding_size_override=32)
                _replace_id_embeddings_with_synthetic(restored)
                restored.load_state_dict(model.state_dict(), strict=True)
                torch.testing.assert_close(restored(batch.features, batch.scenario_id)["logits"], logits)

    def test_compile_keeps_embedding_and_dynamic_attention_eager(self) -> None:
        for name in ("uniformer", "more"):
            config = _compact(name)
            config = replace(config, runtime=replace(config.runtime, compile=True))
            model = build_model(config, _synthetic_vocab_maps(config), embedding_size_override=32)
            before = tuple(model.state_dict())
            with patch("torch.compile", side_effect=lambda function, **kwargs: function) as compile_fn:
                self.assertIs(_maybe_compile_model(config, model), model)
                self.assertGreater(compile_fn.call_count, 0)
                for call in compile_fn.call_args_list:
                    self.assertNotIsInstance(call.args[0], torch.nn.Module)
                    self.assertIn(call.args[0].__name__, {"feature_ffn", "task_ffn", "_mix", "_enhance"})
            self.assertEqual(tuple(model.state_dict()), before)

    def test_production_configs_keep_the_mixformer_feature_pack(self) -> None:
        mixformer = load_app_config(ROOT / "configs" / "mixformer.yaml")
        for name, model_type, shared_anchors in (
            ("uniformer", UniFormerModel, None),
            ("more", MOREModel, 5),
        ):
            with self.subTest(model=name):
                config = load_app_config(ROOT / "configs" / f"{name}.yaml")
                self.assertEqual(
                    config.resolved.tokenization.feature_token_inputs,
                    mixformer.resolved.tokenization.feature_token_inputs,
                )
                self.assertEqual(
                    tuple(sequence.name for sequence in config.sequences),
                    tuple(sequence.name for sequence in mixformer.sequences),
                )
                self.assertEqual(config.task_names, mixformer.task_names)
                self.assertTrue(config.resolved.mixformer_user_feature_inputs)
                self.assertTrue(config.resolved.mixformer_item_feature_inputs)
                self.assertEqual(config.resolved.mixformer_user_head_count, 4)
                self.assertEqual(config.resolved.tokenization.feature_token_count, 8)
                self.assertTrue(
                    all(sequence.encoder == "raw" for sequence in config.sequences)
                )
                if shared_anchors is not None:
                    width = 8 + shared_anchors + len(config.task_names)
                    self.assertEqual(config.model.token_dim % width, 0)

    def test_compact_models_match_candidate_labels(self) -> None:
        for name, model_type in (
            ("uniformer", UniFormerModel),
            ("more", MOREModel),
        ):
            with self.subTest(model=name):
                config = _compact(name)
                model = build_model(
                    config,
                    _synthetic_vocab_maps(config),
                    embedding_size_override=32,
                )
                self.assertIsInstance(model, model_type)
                _replace_id_embeddings_with_synthetic(model)
                batch = _synthetic_feature_batch(
                    config,
                    torch.device("cpu"),
                    batch_size=4,
                    sequence_length=5,
                    seed=11,
                    candidates_per_request=2,
                )
                output = model(batch.features, batch.scenario_id)
                self.assertEqual(
                    tuple(output["logits"].shape),
                    (4, len(config.task_names)),
                )
                loss = output["logits"].square().mean()
                loss.backward()
                self.assertTrue(bool(torch.isfinite(loss)))
                self.assertTrue(
                    all(
                        parameter.grad is None
                        or bool(
                            torch.isfinite(
                                parameter.grad._values()
                                if parameter.grad.is_sparse
                                else parameter.grad
                            ).all()
                        )
                        for parameter in model.parameters()
                    )
                )


def _compact(model_name: str):
    config = load_app_config(ROOT / "configs" / f"{model_name}.yaml")
    shared_anchors = 1 if model_name == "more" else config.model.more_num_shared_anchors
    config = replace(
        config,
        runtime=replace(
            config.runtime,
            attention_backend="sdpa",
            activation_checkpoint="none",
            cuda_graph_backbone=False,
            compile=False,
            require_compact_sequence_batches=False,
        ),
        tokenization=replace(
            config.tokenization,
            feature_tokenizer="rankmixer",
            feature_tokens=(),
            num_feature_tokens=8 if model_name == "uniformer" else 4,
        ),
        model=replace(
            config.model,
            token_dim=32,
            num_layers=1,
            num_heads=4,
            hidden_dim=64,
            task_head_hidden_dim=64,
            more_num_shared_anchors=shared_anchors,
            mixformer_user_item_decouple=True,
        ),
        training=replace(
            config.training,
            embedding_weight_dtype="fp32",
            gset=replace(config.training.gset, capacity=4096)
            if config.training.gset.enabled
            else config.training.gset,
        ),
    )
    config.validate()
    return config
