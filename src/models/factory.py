from __future__ import annotations

from typing import Any

from .mdl import ModelConfig, ModelFromManifest, config_from_manifest as mdl_config_from_manifest
from .rankmixer import RankMixerConfig, RankMixerFromManifest, rankmixer_config_from_manifest

MODEL_NAMES = {"mdl", "rankmixer"}
ManifestModelConfig = ModelConfig | RankMixerConfig
ManifestModel = ModelFromManifest | RankMixerFromManifest


def _validate_model_name(model_name: str) -> str:
    if model_name not in MODEL_NAMES:
        raise ValueError("model_name must be 'mdl' or 'rankmixer'")
    return model_name


def build_model_config_from_manifest(
    manifest: dict[str, Any],
    model_name: str = "mdl",
    embedding_dim: int = 32,
    token_dim: int = 36,
    num_layers: int = 2,
    num_heads: int = 4,
    ffn_hidden_dim: int = 64,
    dropout: float = 0.0,
    feature_backbone: str = "rankmixer",
    ffn_type: str = "dense",
    sparse_moe_num_experts: int = 4,
    sparse_moe_loss_weight: float = 0.0,
    sparse_moe_use_dtsi: bool = True,
    sparse_moe_dtsi_infer_weight: float = 0.5,
    sparse_moe_inference_threshold: float = 0.0,
    use_task_tokens: bool = True,
    use_scenario_tokens: bool = True,
    use_global_scenario_token: bool = True,
    use_task_feature_interaction: bool = True,
    use_scenario_feature_interaction: bool = True,
) -> ManifestModelConfig:
    model_name = _validate_model_name(model_name)
    if model_name == "mdl":
        return mdl_config_from_manifest(
            manifest,
            embedding_dim=embedding_dim,
            token_dim=token_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            dropout=dropout,
            feature_backbone=feature_backbone,
            ffn_type=ffn_type,
            sparse_moe_num_experts=sparse_moe_num_experts,
            sparse_moe_loss_weight=sparse_moe_loss_weight,
            sparse_moe_use_dtsi=sparse_moe_use_dtsi,
            sparse_moe_dtsi_infer_weight=sparse_moe_dtsi_infer_weight,
            sparse_moe_inference_threshold=sparse_moe_inference_threshold,
            use_task_tokens=use_task_tokens,
            use_scenario_tokens=use_scenario_tokens,
            use_global_scenario_token=use_global_scenario_token,
            use_task_feature_interaction=use_task_feature_interaction,
            use_scenario_feature_interaction=use_scenario_feature_interaction,
        )
    if feature_backbone != "rankmixer":
        raise ValueError("rankmixer model requires feature_backbone='rankmixer'")
    return rankmixer_config_from_manifest(
        manifest,
        embedding_dim=embedding_dim,
        token_dim=token_dim,
        num_layers=num_layers,
        ffn_hidden_dim=ffn_hidden_dim,
        dropout=dropout,
        ffn_type=ffn_type,
        sparse_moe_num_experts=sparse_moe_num_experts,
        sparse_moe_loss_weight=sparse_moe_loss_weight,
        sparse_moe_use_dtsi=sparse_moe_use_dtsi,
        sparse_moe_dtsi_infer_weight=sparse_moe_dtsi_infer_weight,
        sparse_moe_inference_threshold=sparse_moe_inference_threshold,
    )


def build_model_from_config(config: ManifestModelConfig) -> ManifestModel:
    if isinstance(config, RankMixerConfig):
        return RankMixerFromManifest(config)
    if isinstance(config, ModelConfig):
        return ModelFromManifest(config)
    raise TypeError("unsupported model config type")


def deserialize_model_config(payload: dict[str, Any]) -> ManifestModelConfig:
    model_name = payload.get("model_name", "mdl")
    model_name = _validate_model_name(str(model_name))
    if model_name == "rankmixer":
        return RankMixerConfig(**payload)
    return ModelConfig(**payload)
