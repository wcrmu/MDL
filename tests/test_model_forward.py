from __future__ import annotations

import torch

from src.models import MDLConfig, MDLModel, ModelFromManifest, config_from_manifest


def test_mdl_forward_shapes() -> None:
    config = MDLConfig(
        num_feature_tokens=4,
        scenario_context_dim=8,
        task_context_dim=6,
        num_scenarios=3,
        num_tasks=2,
        token_dim=16,
        num_layers=1,
        num_heads=4,
        ffn_hidden_dim=32,
    )
    model = MDLModel(config)
    output = model(
        feature_tokens=torch.randn(5, 4, 16),
        scenario_context=torch.randn(5, 8),
        task_context=torch.randn(5, 6),
        scenario_mask=torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=torch.float32,
        ),
    )
    assert output["logits"].shape == (5, 2)


def test_model_from_manifest_forward() -> None:
    manifest = {
        "scenario_names": ["home", "search"],
        "task_names": ["click"],
        "tokenization": {
            "version": 2,
            "kind": "encoder_registry",
            "features": [
                {"name": "user_id", "encoder": "embedding", "vocab_size": 10},
                {"name": "score", "encoder": "identity", "dim": 1},
            ],
            "token_specs": [
                {"token_id": 0, "projection": "linear", "inputs": ["user_id"]},
                {"token_id": 1, "projection": "linear", "inputs": ["score"]},
            ],
        },
    }
    config = config_from_manifest(
        manifest,
        embedding_dim=8,
        token_dim=16,
        num_layers=1,
        num_heads=4,
        ffn_hidden_dim=32,
    )
    model = ModelFromManifest(config)
    output = model(
        {"user_id": torch.tensor([1, 2, 3]), "score": torch.tensor([0.1, 0.2, 0.3])},
        torch.tensor([0, 1, 0]),
    )
    assert output["logits"].shape == (3, 1)
