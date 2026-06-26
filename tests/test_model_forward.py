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


def test_model_from_manifest_with_din_sequence_encoder() -> None:
    manifest = {
        "scenario_names": ["default"],
        "task_names": ["click"],
        "tokenization": {
            "version": 2,
            "kind": "encoder_registry",
            "features": [
                {"name": "item_id", "encoder": "embedding", "vocab_size": 20},
                {
                    "name": "history_items",
                    "encoder": "din",
                    "vocab_size": 20,
                    "target_feature": "item_id",
                    "attention_hidden_dims": [8],
                },
            ],
            "token_specs": [
                {"token_id": 0, "projection": "linear", "inputs": ["item_id"]},
                {"token_id": 1, "projection": "linear", "inputs": ["history_items"]},
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
    din_encoder = model.feature_compiler.encoders[1]
    assert din_encoder.attention_normalization == "none"
    assert any(layer.__class__.__name__ == "Dice" for layer in din_encoder.activation_unit.network)
    output = model(
        {
            "item_id": torch.tensor([3, 4, 5]),
            "history_items": {
                "values": torch.tensor([[1, 2, 3], [4, 0, 0], [0, 0, 0]]),
                "lengths": torch.tensor([3, 1, 0]),
            },
        },
        torch.tensor([0, 0, 0]),
    )
    assert output["logits"].shape == (3, 1)
    assert torch.isfinite(output["logits"]).all()


def test_model_from_manifest_with_multifield_din_sequence_encoder() -> None:
    manifest = {
        "scenario_names": ["default"],
        "task_names": ["click"],
        "tokenization": {
            "version": 2,
            "kind": "encoder_registry",
            "features": [
                {"name": "item_id", "encoder": "embedding", "vocab_size": 20},
                {"name": "cate_id", "encoder": "embedding", "vocab_size": 8, "embedding_dim": 4},
                {"name": "price", "encoder": "identity", "dim": 1},
                {
                    "name": "history_behavior",
                    "encoder": "din",
                    "fusion": "concat",
                    "attention_hidden_dims": [8],
                    "sequence_features": [
                        {
                            "name": "hist_item_id",
                            "target_feature": "item_id",
                            "encoder": "embedding",
                            "vocab_size": 20,
                            "embedding_dim": 8,
                        },
                        {
                            "name": "hist_cate_id",
                            "target_feature": "cate_id",
                            "encoder": "embedding",
                            "vocab_size": 8,
                            "embedding_dim": 4,
                        },
                        {
                            "name": "hist_price",
                            "target_feature": "price",
                            "encoder": "identity",
                            "dim": 1,
                            "projection_dim": 2,
                        },
                    ],
                },
            ],
            "token_specs": [
                {"token_id": 0, "projection": "linear", "inputs": ["item_id"]},
                {"token_id": 1, "projection": "linear", "inputs": ["history_behavior"]},
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
    din_encoder = model.feature_compiler.encoders[3]
    assert din_encoder.output_dim == 14
    output = model(
        {
            "item_id": torch.tensor([3, 4]),
            "cate_id": torch.tensor([2, 5]),
            "price": torch.tensor([0.2, 0.5]),
            "hist_item_id": {
                "values": torch.tensor([[1, 2, 3], [4, 5, 0]]),
                "lengths": torch.tensor([3, 2]),
            },
            "hist_cate_id": {
                "values": torch.tensor([[2, 2, 1], [5, 4, 0]]),
                "lengths": torch.tensor([3, 2]),
            },
            "hist_price": {
                "values": torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.4, 0.0]]),
                "lengths": torch.tensor([3, 2]),
            },
        },
        torch.tensor([0, 0]),
    )
    assert output["logits"].shape == (2, 1)
    assert torch.isfinite(output["logits"]).all()


def test_model_from_manifest_with_multifield_sequence_mean_pooling() -> None:
    manifest = {
        "scenario_names": ["default"],
        "task_names": ["click"],
        "tokenization": {
            "version": 2,
            "kind": "encoder_registry",
            "features": [
                {
                    "name": "history_behavior",
                    "encoder": "sequence_mean_pooling",
                    "fusion": "concat",
                    "sequence_features": [
                        {
                            "name": "hist_item_id",
                            "encoder": "embedding",
                            "vocab_size": 20,
                            "embedding_dim": 8,
                        },
                        {
                            "name": "hist_cate_id",
                            "encoder": "embedding",
                            "vocab_size": 8,
                            "embedding_dim": 4,
                        },
                        {
                            "name": "hist_price",
                            "encoder": "identity",
                            "dim": 1,
                            "projection_dim": 2,
                        },
                    ],
                },
            ],
            "token_specs": [
                {"token_id": 0, "projection": "linear", "inputs": ["history_behavior"]},
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
    pooling_encoder = model.feature_compiler.encoders[0]
    assert pooling_encoder.output_dim == 14
    output = model(
        {
            "hist_item_id": {
                "values": torch.tensor([[1, 2, 3], [4, 5, 0]]),
                "lengths": torch.tensor([3, 2]),
            },
            "hist_cate_id": {
                "values": torch.tensor([[2, 2, 1], [5, 4, 0]]),
                "lengths": torch.tensor([3, 2]),
            },
            "hist_price": {
                "values": torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.4, 0.0]]),
                "lengths": torch.tensor([3, 2]),
            },
        },
        torch.tensor([0, 0]),
    )
    assert output["logits"].shape == (2, 1)
    assert torch.isfinite(output["logits"]).all()



def test_model_from_manifest_with_manifest_driven_domain_tokens() -> None:
    manifest = {
        "scenario_names": ["home", "search"],
        "task_names": ["click", "like"],
        "tokenization": {
            "version": 2,
            "kind": "encoder_registry",
            "features": [
                {"name": "user_id", "encoder": "embedding", "vocab_size": 10},
                {"name": "item_id", "encoder": "embedding", "vocab_size": 20},
                {"name": "score", "encoder": "identity", "dim": 1},
            ],
            "token_specs": [
                {"token_id": 0, "projection": "linear", "inputs": ["user_id"]},
                {"token_id": 1, "projection": "linear", "inputs": ["item_id", "score"]},
            ],
            "scenario_features": [
                {"name": "user_id", "encoder": "embedding", "vocab_size": 10},
                {"name": "item_id", "encoder": "embedding", "vocab_size": 20},
                {"name": "score", "encoder": "identity", "dim": 1},
            ],
            "scenario_token_specs": [
                {"token_id": 0, "inputs": ["user_id", "item_id"]},
                {"token_id": 1, "inputs": ["user_id", "score"]},
                {"token_id": 2, "inputs": ["user_id", "item_id", "score"]},
            ],
            "task_features": [
                {"name": "item_id", "encoder": "embedding", "vocab_size": 20},
                {"name": "score", "encoder": "identity", "dim": 1},
            ],
            "task_token_specs": [
                {"token_id": 0, "inputs": ["item_id"]},
                {"token_id": 1, "inputs": ["item_id", "score"]},
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
    assert model.scenario_token_compiler is not None
    assert model.scenario_embedding is None
    assert model.task_token_compiler is not None
    assert model.task_context is None

    output = model(
        {
            "user_id": torch.tensor([1, 2, 3]),
            "item_id": torch.tensor([4, 5, 6]),
            "score": torch.tensor([0.1, 0.2, 0.3]),
        },
        torch.tensor([0, 1, 0]),
        return_attention=True,
    )

    assert output["logits"].shape == (3, 2)
    assert torch.isfinite(output["logits"]).all()
    attention = output["attentions"][0]
    assert attention["scenario_feature"].shape == (3, 4, 3, 2)
    assert attention["task_feature"].shape == (3, 4, 2, 2)


def test_manifest_driven_domain_tokens_validate_token_counts() -> None:
    manifest = {
        "scenario_names": ["home"],
        "task_names": ["click"],
        "tokenization": {
            "version": 2,
            "kind": "encoder_registry",
            "features": [
                {"name": "user_id", "encoder": "embedding", "vocab_size": 10},
            ],
            "token_specs": [
                {"token_id": 0, "projection": "linear", "inputs": ["user_id"]},
            ],
            "scenario_features": [
                {"name": "user_id", "encoder": "embedding", "vocab_size": 10},
            ],
            "scenario_token_specs": [
                {"token_id": 0, "inputs": ["user_id"]},
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
    try:
        ModelFromManifest(config)
    except ValueError as error:
        assert "one token per scenario plus one global token" in str(error)
    else:
        raise AssertionError("expected scenario token count validation to fail")



def test_mdl_sparse_moe_forward_regularization_and_inference() -> None:
    config = MDLConfig(
        num_feature_tokens=4,
        scenario_context_dim=8,
        task_context_dim=6,
        num_scenarios=2,
        num_tasks=2,
        token_dim=16,
        num_layers=1,
        num_heads=4,
        ffn_hidden_dim=32,
        ffn_type="sparse_moe",
        sparse_moe_num_experts=3,
    )
    model = MDLModel(config)
    feature_tokens = torch.randn(3, 4, 16)
    scenario_context = torch.randn(3, 8)
    task_context = torch.randn(3, 6)
    scenario_mask = torch.tensor(
        [[1, 0], [0, 1], [1, 0]],
        dtype=torch.float32,
    )

    model.train()
    output = model(
        feature_tokens=feature_tokens,
        scenario_context=scenario_context,
        task_context=task_context,
        scenario_mask=scenario_mask,
    )
    assert output["logits"].shape == (3, 2)
    assert output["moe_regularization_loss"].ndim == 0
    assert output["moe_regularization_loss"].requires_grad
    assert 0.0 <= float(output["moe_active_ratio"]) <= 1.0
    (output["logits"].sum() + 0.01 * output["moe_regularization_loss"]).backward()
    first_infer_router = model.blocks[0].feature_ffn.infer_routers[0]
    assert first_infer_router.weight.grad is not None

    model.eval()
    with torch.no_grad():
        inference_output = model(
            feature_tokens=feature_tokens,
            scenario_context=scenario_context,
            task_context=task_context,
            scenario_mask=scenario_mask,
        )
    assert inference_output["logits"].shape == (3, 2)
    assert inference_output["moe_regularization_loss"].shape == ()


def test_model_from_manifest_sparse_moe_config_forward() -> None:
    manifest = {
        "scenario_names": ["default"],
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
        ffn_type="sparse_moe",
        sparse_moe_num_experts=2,
        sparse_moe_loss_weight=0.05,
    )
    assert config.ffn_type == "sparse_moe"
    assert config.sparse_moe_loss_weight == 0.05
    model = ModelFromManifest(config)
    output = model(
        {"user_id": torch.tensor([1, 2]), "score": torch.tensor([0.1, 0.2])},
        torch.tensor([0, 0]),
    )
    assert output["logits"].shape == (2, 1)
    assert output["moe_regularization_loss"].ndim == 0
