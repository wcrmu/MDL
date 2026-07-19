from __future__ import annotations

import os
from pathlib import Path
import socket
import tempfile
import unittest

import torch
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp
from torch import nn

from src.checkpoint import load_model_checkpoint, save_model_checkpoint
from src.config import load_app_config
from src.embeddings import EmbeddingTableSpec, ShardedEmbedding, plan_embedding_shards


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _CheckpointToyModel(nn.Module):
    def __init__(self, world_size: int) -> None:
        super().__init__()
        table = EmbeddingTableSpec("item", 17, 3)
        plan = plan_embedding_shards(
            [table],
            world_size=world_size,
            strategy="row_wise",
            table_wise_max_rows=4,
        )
        self.embedding = ShardedEmbedding(
            table.num_embeddings,
            table.embedding_dim,
            table_name=table.name,
            shard_spec=plan.tables[table.name],
        )
        self.dense = nn.Linear(3, 2)


def _save_worker(
    rank: int,
    world_size: int,
    port: int,
    checkpoint_path: str,
    config_path: str,
) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch_dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        config = load_app_config(config_path)
        model = _CheckpointToyModel(world_size)
        full_weight = torch.arange(51, dtype=torch.float32).view(17, 3) / 7.0
        full_weight[0].zero_()
        model.embedding.load_full_weight_(full_weight)
        with torch.no_grad():
            model.dense.weight.fill_(0.25)
            model.dense.bias.copy_(torch.tensor([1.0, -1.0]))
        save_model_checkpoint(
            config,
            model,
            checkpoint_path,
            rank=rank,
            world_size=world_size,
        )
    finally:
        torch_dist.destroy_process_group()


class ShardedCheckpointTest(unittest.TestCase):
    def test_checkpoint_reshards_without_full_reconstruction(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config_path = root / "configs" / "reference" / "default.yaml"
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "model.sharded"
            torch_mp.start_processes(
                _save_worker,
                args=(
                    2,
                    _free_port(),
                    str(checkpoint_path),
                    str(config_path),
                ),
                nprocs=2,
                join=True,
                start_method="spawn",
            )
            self.assertTrue((checkpoint_path / "manifest.json").exists())
            self.assertTrue((checkpoint_path / "dense.pt").exists())
            self.assertEqual(
                len(list(checkpoint_path.glob("rank-*-of-*.pt"))), 2
            )

            config = load_app_config(config_path)
            model = _CheckpointToyModel(world_size=1)
            load_model_checkpoint(
                config,
                model,
                checkpoint_path,
                device=torch.device("cpu"),
            )
            expected = torch.arange(51, dtype=torch.float32).view(17, 3) / 7.0
            expected[0].zero_()
            torch.testing.assert_close(model.embedding.weight, expected)
            torch.testing.assert_close(
                model.dense.weight, torch.full_like(model.dense.weight, 0.25)
            )
            torch.testing.assert_close(
                model.dense.bias, torch.tensor([1.0, -1.0])
            )


if __name__ == "__main__":
    unittest.main()
