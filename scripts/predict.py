from __future__ import annotations

import argparse
import csv
import sys

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.datasets import ManifestDataset, collate_manifest_batch, load_manifest
from src.models import build_model_config_from_manifest, build_model_from_config
from src.trainers.evaluator import move_batch
from src.utils import load_checkpoint


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate prediction probabilities for a manifest split.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--model-name", choices=["mdl", "rankmixer"], default="mdl")
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
        model = build_model_from_config(checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model_config = build_model_config_from_manifest(
            manifest,
            model_name=args.model_name,
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
        model = build_model_from_config(model_config).to(device)

    dataset = ManifestDataset(args.data_dir, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_manifest_batch)
    task_names = manifest["task_names"]
    output_handle = open(args.output_path, "w", encoding="utf-8", newline="") if args.output_path else sys.stdout
    try:
        writer = csv.DictWriter(output_handle, fieldnames=["group_id", *task_names])
        writer.writeheader()
        model.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                batch = move_batch(batch, device)
                output = model(batch["features"], batch["scenario_id"])
                logits = output["logits"]
                if not isinstance(logits, Tensor):
                    raise TypeError("model output logits must be a tensor")
                probabilities = torch.sigmoid(logits).cpu().tolist()
                for group_id, scores in zip(batch["group_id"], probabilities):
                    row = {"group_id": group_id}
                    row.update({task_name: score for task_name, score in zip(task_names, scores)})
                    writer.writerow(row)
    finally:
        if args.output_path:
            output_handle.close()


if __name__ == "__main__":
    main()
