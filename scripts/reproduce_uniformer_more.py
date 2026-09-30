"""Train the paper-aligned UniFormer and MORE rankers on synthetic labels.

The Kuaishou and Momo logs are not public, so this does not reproduce their
GAUC or online lifts. It checks that the published stacks run and that a
fixed labeling rule is learnable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.modules.more import MORERanker, weighted_bce_with_logits as more_loss  # noqa: E402
from src.modules.uniformer import (
    GroupedSwiGLUTokenizer,
    UniFormerRanker,
    weighted_bce_with_logits as uniformer_loss,
)


def train_uniformer(steps: int = 40) -> tuple[float, float]:
    torch.manual_seed(7)
    tokenizer = GroupedSwiGLUTokenizer([8, 8, 8, 8], d_model=16, ffn_expansion=1)
    task_tokenizer = GroupedSwiGLUTokenizer([8, 8], d_model=16, ffn_expansion=1)
    model = UniFormerRanker(
        d_model=16,
        num_heads=4,
        num_ns_tokens=4,
        num_user_tokens=2,
        num_tasks=2,
        num_sequences=2,
        fim_layers=2,
        tim_layers=1,
        ffn_expansion=1,
    )
    groups = [torch.randn(32, 8) for _ in range(4)]
    task_groups = [torch.randn(32, 8) for _ in range(2)]
    with torch.no_grad():
        ns = tokenizer(groups)
        tasks = task_tokenizer(task_groups)
        positive = ns.mean(dim=(1, 2)) > 0
        labels = torch.stack([positive, ~positive], dim=1).float()
    sequences = [torch.randn(32, 6, 16), torch.randn(32, 4, 16)]
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(tokenizer.parameters()) + list(task_tokenizer.parameters()),
        lr=2e-2,
    )

    def loss() -> torch.Tensor:
        return uniformer_loss(model(tokenizer(groups), task_tokenizer(task_groups), sequences).logits, labels)

    initial = loss().detach().item()
    for _ in range(steps):
        optimizer.zero_grad()
        loss().backward()
        optimizer.step()
    return initial, loss().detach().item()


def train_more(steps: int = 40) -> tuple[float, float]:
    torch.manual_seed(8)
    model = MORERanker(
        d_model=16,
        num_ns_tokens=1,
        num_shared_anchors=1,
        num_private_anchors=2,
        num_heads=4,
        num_blocks=2,
        ffn_expansion=1,
    )
    sequence = torch.randn(2, 6, 16)
    nonseq = torch.randn(32, 1, 16)
    request_index = torch.arange(32) % 2
    with torch.no_grad():
        positive = nonseq.mean(dim=(1, 2)) > 0
        labels = torch.stack([positive, ~positive], dim=1).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)

    def loss() -> torch.Tensor:
        output = model(sequence, nonseq, request_index=request_index)
        return more_loss(output.logits, labels)

    initial = loss().detach().item()
    for _ in range(steps):
        optimizer.zero_grad()
        loss().backward()
        optimizer.step()
    return initial, loss().detach().item()


def main() -> None:
    uniformer_initial, uniformer_final = train_uniformer()
    more_initial, more_final = train_more()
    print(f"UniFormer synthetic BCE {uniformer_initial:.4f} -> {uniformer_final:.4f}")
    print(f"MORE synthetic BCE {more_initial:.4f} -> {more_final:.4f}")
    if uniformer_final >= uniformer_initial or more_final >= more_initial:
        raise SystemExit("synthetic loss did not decrease")


if __name__ == "__main__":
    main()
