from __future__ import annotations

import argparse

import torch

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.datasets import load_manifest
from src.models import ModelFromManifest, config_from_manifest
from src.trainers import evaluate_model
from src.utils import load_checkpoint


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a recommendation model.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--token-dim", type=int, default=36)
    parser.add_argument("--feature-backbone", choices=["rankmixer", "attention"], default="rankmixer")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-hidden-dim", type=int, default=64)
    parser.add_argument("--ffn-type", choices=["dense", "sparse_moe"], default="dense")
    parser.add_argument("--sparse-moe-num-experts", type=int, default=4)
    parser.add_argument("--sparse-moe-inference-threshold", type=float, default=0.0)
    parser.add_argument("--disable-sparse-moe-dtsi", dest="sparse_moe_use_dtsi", action="store_false")
    parser.set_defaults(sparse_moe_use_dtsi=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device)
    manifest = load_manifest(args.data_dir)
    if args.checkpoint_path:
        checkpoint = load_checkpoint(args.checkpoint_path, map_location=device)
        model = ModelFromManifest(checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model_config = config_from_manifest(
            manifest,
            embedding_dim=args.embedding_dim,
            token_dim=args.token_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ffn_hidden_dim=args.ffn_hidden_dim,
            feature_backbone=args.feature_backbone,
            ffn_type=args.ffn_type,
            sparse_moe_num_experts=args.sparse_moe_num_experts,
            sparse_moe_use_dtsi=args.sparse_moe_use_dtsi,
            sparse_moe_inference_threshold=args.sparse_moe_inference_threshold,
        )
        model = ModelFromManifest(model_config).to(device)

    result = evaluate_model(
        model,
        args.data_dir,
        args.split,
        manifest,
        args.batch_size,
        device,
        args.max_batches,
    )
    for line in result.format_lines():
        print(line)


if __name__ == "__main__":
    main()
