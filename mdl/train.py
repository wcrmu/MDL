from __future__ import annotations

import argparse
import math
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .data.manifest import ManifestDataset, collate_manifest_batch, load_manifest
from .utils import multitask_bce_loss
from .utils import binary_auc, qauc
from .ranking import RankingModel, config_from_manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MDL on a manifest dataset.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--token-dim", type=int, default=36)
    parser.add_argument("--feature-backbone", choices=["rankmixer", "attention"], default="rankmixer")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-hidden-dim", type=int, default=64)
    return parser


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    def move_value(value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move_value(child) for key, child in value.items()}
        return value

    return {key: move_value(value) for key, value in batch.items()}


@torch.no_grad()
def evaluate(
    model: RankingModel,
    data_dir: str,
    split: str,
    manifest: dict[str, Any],
    batch_size: int,
    device: torch.device,
    max_batches: int,
) -> None:
    dataset = ManifestDataset(data_dir, split)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_manifest_batch)
    task_names = manifest["task_names"]
    labels_by_task = [[] for _ in task_names]
    scores_by_task = [[] for _ in task_names]
    groups_by_task = [[] for _ in task_names]
    losses: list[float] = []

    model.eval()
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = move_batch(batch, device)
        output = model(batch["features"], batch["scenario_id"])
        logits = output["logits"]
        if not isinstance(logits, Tensor):
            raise TypeError("model output logits must be a tensor")
        losses.append(multitask_bce_loss(logits, batch["labels"], batch["label_mask"]).item())
        probabilities = torch.sigmoid(logits).cpu()
        labels = batch["labels"].cpu()
        masks = batch["label_mask"].cpu()
        for task_index, _task_name in enumerate(task_names):
            valid = masks[:, task_index] > 0
            labels_by_task[task_index].extend(labels[valid, task_index].tolist())
            scores_by_task[task_index].extend(probabilities[valid, task_index].tolist())
            groups_by_task[task_index].extend(
                group for group, keep in zip(batch["group_id"], valid.tolist()) if keep
            )

    loss_text = f"{sum(losses) / len(losses):.6f}" if losses else "nan"
    print(f"{split}_loss={loss_text}")
    for task_index, task_name in enumerate(task_names):
        labels = labels_by_task[task_index]
        scores = scores_by_task[task_index]
        if not labels:
            continue
        auc = binary_auc(labels, scores)
        qauc_result = qauc(labels, scores, groups_by_task[task_index])
        auc_text = f"{auc:.6f}" if auc is not None else "nan"
        qauc_text = f"{qauc_result.qauc:.6f}" if not math.isnan(qauc_result.qauc) else "nan"
        print(
            f"{split}_{task_name}_auc={auc_text} "
            f"{split}_{task_name}_qauc={qauc_text} "
            f"valid_groups={qauc_result.valid_groups}"
        )


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device)
    manifest = load_manifest(args.data_dir)
    config = config_from_manifest(
        manifest,
        embedding_dim=args.embedding_dim,
        token_dim=args.token_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_hidden_dim=args.ffn_hidden_dim,
        feature_backbone=args.feature_backbone,
    )
    model = RankingModel(config).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        dataset = ManifestDataset(args.data_dir, "train")
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_manifest_batch)
        model.train()
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["features"], batch["scenario_id"])
            logits = output["logits"]
            if not isinstance(logits, Tensor):
                raise TypeError("model output logits must be a tensor")
            loss = multitask_bce_loss(logits, batch["labels"], batch["label_mask"])
            loss.backward()
            optimizer.step()
            global_step += 1
            print(f"epoch={epoch} step={global_step} loss={loss.item():.6f}")
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        evaluate(
            model,
            args.data_dir,
            "val",
            manifest,
            args.batch_size,
            device,
            args.eval_max_batches,
        )
        if args.max_steps is not None and global_step >= args.max_steps:
            break


if __name__ == "__main__":
    main()

