from __future__ import annotations

import argparse

import torch

from .utils import multitask_bce_loss
from .utils import qauc
from .models import MDLConfig, MDLModel
from .utils import make_synthetic_batch


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a short MDL synthetic smoke test.")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device)
    config = MDLConfig(
        num_feature_tokens=8,
        scenario_context_dim=10,
        task_context_dim=6,
        num_scenarios=3,
        num_tasks=3,
        token_dim=32,
        num_layers=2,
        num_heads=4,
        ffn_hidden_dim=64,
        feature_backbone="rankmixer",
    )
    model = MDLModel(config).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

    last_batch = None
    last_logits = None
    for step in range(1, args.steps + 1):
        batch = make_synthetic_batch(config, args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch.feature_tokens,
            batch.scenario_context,
            batch.task_context,
            batch.scenario_mask,
        )
        logits = output["logits"]
        if not isinstance(logits, torch.Tensor):
            raise TypeError("model output logits must be a tensor")
        loss = multitask_bce_loss(logits, batch.labels, batch.label_mask)
        loss.backward()
        optimizer.step()
        print(f"step={step} loss={loss.item():.6f}")
        last_batch = batch
        last_logits = logits.detach()

    if last_batch is not None and last_logits is not None:
        scores = torch.sigmoid(last_logits[:, 0]).cpu().tolist()
        labels = last_batch.labels[:, 0].cpu().tolist()
        result = qauc(labels, scores, last_batch.query_ids)
        print(
            "task0_qauc="
            f"{result.qauc:.6f} valid_groups={result.valid_groups} "
            f"skipped_groups={result.skipped_groups}"
        )


if __name__ == "__main__":
    main()

