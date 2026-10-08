"""Bounded diagnostic probes; writes only to a new local output directory.

CPU mode records shared-infrastructure defects without changing model code.
NCCL mode checks accumulated compiled updates, not production data quality.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import socket
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verify_industrial_rankers import config_for, small_model
from src import train
from src.benchmark import _synthetic_feature_batch, _synthetic_vocab_maps
from src.checkpoint import load_model_checkpoint, stage_training_checkpoint
from src.embeddings import sharded_embedding_modules
from src.modules.rank_table import RankEmbeddingTable
from src.optim import ShardedRowWiseAdagrad


def write_json(path, value):
    with Path(path).open("x") as output:
        json.dump(value, output, indent=2)


def cpu_probe(output):
    torch.set_num_threads(1)
    table = RankEmbeddingTable(4)
    original = table.weight
    optimizer = ShardedRowWiseAdagrad([original], lr=.01)
    table.lookup(0, torch.tensor([7, 7])).sum().backward()
    train._coalesce_accumulated_sparse_gradients([original])
    result = {"coalesce_after_growth": {
        "parameter_replaced": original is not table.weight,
        "current_gradient_coalesced": table.weight.grad.is_coalesced(),
        "stored_entries": table.weight.grad._nnz(),
        "unique_entries": table.weight.grad.coalesce()._nnz(),
    }}
    table.lookup(0, torch.cat((torch.tensor([7]), torch.arange(100, 140)))).sum().backward()
    accumulated = table.weight.grad.coalesce().to_dense()[table.row_for(7)]
    torch.testing.assert_close(accumulated, torch.full_like(accumulated, 3.))
    result["gradient_preserved_across_second_growth"] = True
    groups = train._classify_model_parameters(table)
    result["replicated_table_classification"] = {
        "row_sharded": table.row_sharded,
        "sparse_sync_references": len(groups.sparse_sync),
        "sharded_ignored_references": len(groups.sharded_ddp_ignore),
    }
    # Exercise the actual staging API twice with disjoint rank-owned keys.
    # The current erroneous replicated branch has no collective, so sequential
    # invocation isolates its persistence behavior without a process group.
    config = config_for("mixformer", "cpu")
    config = replace(config, training=replace(config.training, embedding_distribution="sharded"))
    stage = output / "checkpoint_probe"
    files = []
    for rank, key in enumerate((2, 3)):
        local = RankEmbeddingTable(4, row_sharded=True)
        local.insert_ids(torch.tensor([key]))
        local_optimizer = ShardedRowWiseAdagrad([local.weight], lr=.01)
        local.lookup(0, torch.tensor([key])).sum().backward()
        local_optimizer.step()
        staged = stage_training_checkpoint(config, local, stage, step=1, rows=1,
            rank=rank, world_size=2, sharded_optimizer=local_optimizer,
            sparse_stream=True, cleanup_staging=False)
        files.append(list(staged.relative_files))
    restored = RankEmbeddingTable(4, row_sharded=True)
    load_model_checkpoint(config, restored, stage / "model.pt", device=torch.device("cpu"))
    metadata = json.loads((stage / "checkpoint.json").read_text())
    training_state = torch.load(stage / "train_state.pt", weights_only=False)
    result["checkpoint"] = {
        "files_by_rank": files,
        "manifest_sharded_embeddings": metadata["sharded_embeddings"],
        "restored_keys": sorted(restored._key_to_row),
        "rank1_key_preserved": restored.contains(3),
        "train_state_keys": sorted(training_state),
    }
    result["models"] = {}
    for name in ("uniformer", "more", "mixformer"):
        config = config_for(name, "cpu")
        config = replace(config, training=replace(config.training, embedding_distribution="sharded"))
        model = small_model(config, _synthetic_vocab_maps(config), torch.device("cpu"))
        bank = model.encoder_bank
        goods = bank.embeddings["goods_id_hn"]
        mall = bank.embeddings["mall_id_hn"]
        bank.gset_table.insert_ids(torch.tensor([7]))
        goods_value = goods(torch.tensor([7]))
        mall_value = mall(torch.tensor([7]))
        result["models"][name] = {
            "actual_table": type(model.encoder_bank.gset_table).__name__,
            "checkpoint_detected_shards": len(sharded_embedding_modules(model)),
            "train_inputs": list(config.data.train.inputs),
            "test_inputs": list(config.data.test.inputs),
            "key_mode": config.training.gset.key_mode,
            "distinct_views": goods is not mall,
            "goods_namespace": goods.namespace,
            "mall_namespace": mall.namespace,
            "goods7_mall7_same_embedding": torch.equal(goods_value, mall_value),
            "physical_key_count": len(bank.gset_table._key_to_row),
        }
    write_json(output / "cpu.json", result)
    print(json.dumps(result), flush=True)


def nccl_worker(rank, port, output, compile_enabled, accumulation, steps, models):
    torch.set_num_threads(1)
    torch._inductor.config.compile_threads = 1
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=2,
        init_method=f"tcp://127.0.0.1:{port}", timeout=timedelta(seconds=300))
    results = {}
    name, step, micro = None, None, None
    try:
        results = {"rank": rank, "backend": dist.get_backend(),
                   "compile": compile_enabled, "accumulation": accumulation,
                   "steps": steps, "models": {}}
        # Independent analytic oracle for owner routing and world-size scaling.
        table = RankEmbeddingTable(4, row_sharded=True).to(device)
        table.insert_ids(torch.tensor([2 + rank], device=device))
        (table.lookup(0, torch.tensor([2, 3, 2], device=device)).sum() * (rank + 1)).backward()
        gradient = table.weight.grad.coalesce().to_dense()[table.row_for(2 + rank)]
        torch.testing.assert_close(gradient, torch.full_like(gradient, 3. if rank == 0 else 1.5))
        results["sharded_gradient_analytic_oracle"] = True
        # Replicated-mode diagnostic: identical initial values, different local
        # gradients, then the same synchronizer invoked by the trainer.
        replica = RankEmbeddingTable(4).to(device)
        replica.insert_ids(torch.tensor([7], device=device))
        with torch.no_grad():
            replica.weight.fill_(.02)
        replica_groups = train._classify_model_parameters(replica)
        replica_optimizer = ShardedRowWiseAdagrad(replica_groups.sharded_optimizer, lr=.01,
                                                initial_accumulator_value=.1)
        (replica.lookup(0, torch.tensor([7], device=device)).sum() * (rank + 1)).backward()
        context = train.DistributedContext(True, rank, rank, 2, device)
        train._ReplicatedSparseGradientSynchronizer(context, replica_groups.sparse_sync).synchronize()
        replica_optimizer.step()
        reference = replica.weight.detach().clone()
        dist.broadcast(reference, src=0)
        results["replicated_weight_rank0_max_delta"] = float((replica.weight.detach()-reference).abs().max())
        write_json(Path(output) / f"shared_rank{rank}.json", results)
        for name in models:
            torch.manual_seed(1100)
            config = config_for(name, str(device))
            config = replace(config,
                runtime=replace(config.runtime, compile=(name == "mixformer")
                    if compile_enabled is None else compile_enabled),
                training=replace(config.training, embedding_distribution="sharded"),
                model=replace(config.model, token_dim=32, hidden_dim=64,
                              task_head_hidden_dim=64, num_heads=4, num_layers=1))
            base = small_model(config, _synthetic_vocab_maps(config), device)
            groups = train._classify_model_parameters(base)
            model = train._prepare_forward_model(config, base, context,
                (*groups.sparse_sync, *groups.sharded_ddp_ignore))
            dense = train._build_dense_optimizer(list(groups.dense_optimizer), config, device)
            sparse = ShardedRowWiseAdagrad(groups.sharded_optimizer, lr=.001,
                initial_accumulator_value=.1, track_dirty_rows=True)
            changes, losses = 0, []
            for step in range(steps):
                dense.zero_grad(set_to_none=True)
                sparse.zero_grad(set_to_none=True)
                for micro in range(accumulation):
                    batch = _synthetic_feature_batch(config, device, 4, 3,
                        900 + rank * 17 + step * accumulation + micro, 2)
                    if rank == (step + micro) % 2:
                        for sequence in config.sequences:
                            batch.features[sequence.name]["lengths"].zero_()
                    sync = micro == accumulation - 1 or (step == 0 and config.training.ddp.static_graph)
                    with train._gradient_sync_context(model, synchronize=sync):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            prediction = model(batch.features, batch.scenario_id)
                            loss = train._loss_terms_from_batch(prediction, batch,
                                loss_reduction=config.training.loss_reduction,
                                task_loss_weights=config.ordered_task_loss_weights)[0]
                        (loss / accumulation).backward()
                    assert torch.isfinite(loss)
                for parameter in groups.dense_optimizer:
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                table = base.encoder_bank.gset_table
                gradient = table.weight.grad.coalesce()
                assert torch.isfinite(gradient.values()).all()
                assert sparse.param_groups[0]["params"][0] is table.weight
                assert table._key_to_row and all(key % 2 == rank for key in table._key_to_row)
                train._clip_grad_norm(list(groups.dense_optimizer), 1.)
                train._clip_sparse_grad_norm([], list(groups.sharded_optimizer), 1.)
                gradient = table.weight.grad.coalesce()
                rows, values = gradient.indices()[0], gradient.values().float()
                before = table.weight.detach().index_select(0, rows).clone()
                accumulator = sparse.state[table.weight]["sum"].index_select(0, rows).float() + values.square().mean(1)
                expected = (before.float() - .001 * values / (accumulator.sqrt() + sparse.param_groups[0]["eps"])[:, None]).to(before.dtype)
                dense.step()
                sparse.step()
                actual = table.weight.detach().index_select(0, rows)
                torch.testing.assert_close(actual, expected, rtol=.01, atol=1e-8)
                changes += int(not torch.equal(before, actual))
                flat = torch.cat([p.detach().flatten() for p in groups.dense_optimizer])
                reference = flat.clone()
                dist.broadcast(reference, src=0)
                torch.testing.assert_close(flat, reference, atol=0, rtol=0)
                losses.append(float(loss.detach()))
            assert changes > 0
            results["models"][name] = {"last_microbatch_losses": losses,
                "sparse_changed_steps": changes, "sparse_reference_checked": True,
                "compile": config.runtime.compile,
                "static_graph": config.training.ddp.static_graph,
                "dense_exactly_synchronized": True, "table_rows": table.num_embeddings}
            write_json(Path(output) / f"{name}_rank{rank}.json", results["models"][name])
            print(f"rank={rank} model={name} passed", flush=True)
        write_json(Path(output) / f"rank{rank}.json", results)
    except Exception as error:
        results["error"] = {"model": name, "step": step, "microbatch": micro,
                            "type": type(error).__name__, "message": str(error)}
        write_json(Path(output) / f"error_rank{rank}.json", results)
        raise
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nccl", action="store_true")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                        help="Default: compile MixFormer only, matching production configs")
    parser.add_argument("--models", nargs="+", choices=("uniformer", "more", "mixformer"),
                        default=("uniformer", "more", "mixformer"))
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.nccl:
        if torch.cuda.device_count() != 2:
            raise ValueError("NCCL diagnostic requires exactly two visible GPUs")
        if args.accumulation < 1 or args.steps < 2:
            raise ValueError("require accumulation >= 1 and steps >= 2 to exercise no_sync")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(nccl_worker, args=(port, str(args.output), args.compile,
                                   args.accumulation, args.steps, args.models), nprocs=2, join=True)
    else:
        cpu_probe(args.output)


if __name__ == "__main__":
    main()
