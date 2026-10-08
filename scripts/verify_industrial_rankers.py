"""Bounded synthetic acceptance test; never writes production checkpoints.

Run with --device cpu first; use an explicitly available GPU for --device cuda:N.
All outputs use a new directory, and failures produce a nonzero exit status.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_agg_parquet import generate_synthetic_agg_dataset
from src import train
from src.benchmark import _synthetic_feature_batch, _synthetic_vocab_maps
from src.config import load_app_config
from src.model import build_model
from src.modules.gset import gset_batch_outcomes, iter_gset_tables
from src.optim import ShardedRowWiseAdagrad


def small_model(config, vocab_maps, device):
    model = build_model(config, vocab_maps, embedding_size_override=64).to(device)
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            module.register_forward_pre_hook(lambda m, a: (a[0].remainder(m.num_embeddings),))
    return model


def config_for(name, device):
    c = load_app_config(ROOT / "configs" / f"{name}.yaml")
    cuda = device.startswith("cuda")
    return replace(c,
        runtime=replace(c.runtime, device=device, distributed="none", compile=False,
                        attention_backend="flash" if cuda else "sdpa",
                        precision="bf16" if cuda else "fp32"),
        training=replace(c.training, embedding_distribution="replicated",
                         embedding_weight_dtype="bf16" if cuda else "fp32",
                         gset=replace(c.training.gset, capacity=16384)))


def update_probe(config, device, steps):
    torch.manual_seed(19)
    model = small_model(config, _synthetic_vocab_maps(config), device)
    batch = _synthetic_feature_batch(config, device, 8, 12, 19, 2)
    groups = train._classify_model_parameters(model)
    dense = list(groups.dense_optimizer)
    sparse = list(groups.embedding_optimizer) + list(groups.sharded_optimizer)
    optimizers = [train._build_dense_optimizer(dense, config, device)]
    if sparse:
        optimizers.append(ShardedRowWiseAdagrad(sparse, lr=config.training.lr_sparse,
            initial_accumulator_value=config.training.adagrad_initial_accumulator_value,
            eps=config.training.adagrad_eps))
    base_lrs = [[g["lr"] for g in o.param_groups] for o in optimizers]
    model = train._maybe_compile_model(config, model)
    losses = []
    shared_embedding_update_steps = 0
    shared_embedding_rounding_only_steps = 0
    for step in range(steps):
        train._set_optimizer_lrs(optimizers, base_lrs,
                                train._lr_schedule_multiplier(config, step + 1, None))
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        with gset_batch_outcomes(model, batch.labels, batch.label_mask,
                task_index=config.task_names.index(config.training.gset.score_task)), \
                torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(batch.features, batch.scenario_id)
            loss = train._loss_terms_from_batch(output, batch,
                loss_reduction=config.training.loss_reduction,
                task_loss_weights=config.ordered_task_loss_weights)[0]
        loss.backward()
        losses.append(loss.detach().float().item())
        missing, nonfinite = [], []
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is None:
                missing.append(name)
            elif p.grad is not None and not torch.isfinite(p.grad._values() if p.grad.is_sparse else p.grad).all():
                nonfinite.append(name)
        if missing or nonfinite:
            raise AssertionError({"missing_gradients": missing, "nonfinite_gradients": nonfinite})
        train._clip_grad_norm(dense, config.training.dense_clip_norm)
        train._clip_grad_norm(sparse, config.training.sparse_clip_norm)
        tables = list(iter_gset_tables(model))
        before = [table.weight.detach().clone() for table in tables]
        expected_updates = []
        for table in tables:
            parameter = table.weight
            owner = next((o, g) for o in optimizers for g in o.param_groups
                         if any(p is parameter for p in g["params"]))
            optimizer, group = owner
            gradient = parameter.grad.coalesce()
            rows, values = gradient.indices()[0], gradient.values().float()
            state = optimizer.state[parameter]
            accumulator = state["sum"].index_select(0, rows).float() + values.square().mean(1)
            lr = group["lr"] / (1 + float(state["step"]) * group["lr_decay"])
            original = parameter.detach().index_select(0, rows)
            rounded = (original.float() - lr * values / (accumulator.sqrt() + group["eps"])[:, None]).to(parameter.dtype)
            expected_updates.append((parameter, rows, rounded, torch.equal(original, rounded)))
        for optimizer in optimizers:
            optimizer.step()
        for parameter, rows, expected, _unchanged in expected_updates:
            torch.testing.assert_close(parameter.detach().index_select(0, rows), expected,
                                       rtol=.01 if parameter.dtype == torch.bfloat16 else 1e-5, atol=1e-9)
        if any(not torch.equal(old, table.weight) for old, table in zip(before, tables)):
            shared_embedding_update_steps += 1
        elif all(unchanged for _p, _r, _e, unchanged in expected_updates):
            shared_embedding_rounding_only_steps += 1
        else:
            raise AssertionError("shared embedding update disappeared despite a representable reference update")
    result = {"losses": losses, "shared_embedding_update_steps": shared_embedding_update_steps,
              "shared_embedding_rounding_only_steps": shared_embedding_rounding_only_steps,
              "sparse_update_reference_checked": True,
              "shared_tables": [{"type": type(t).__name__, "rows": t.weight.size(0),
                                 "step": t.stats().current_step} for t in iter_gset_tables(model)]}
    if not shared_embedding_update_steps:
        raise AssertionError("shared embeddings never changed: " + json.dumps(result))
    if not losses[-1] < losses[0]:
        raise AssertionError(f"synthetic loss did not decrease: {losses}")
    if config.model.name == "more":
        model.eval()
        for seq in config.sequences:
            raw = batch.features[seq.name]["fields"][seq.timestamp_field]
            raw.copy_(torch.arange(raw.size(1), device=device).expand_as(raw) + 10000)
        reversed_features = copy.deepcopy(batch.features)
        for seq in config.sequences:
            fields = reversed_features[seq.name]["fields"]
            fields[seq.timestamp_field] = fields[seq.timestamp_field].flip(1)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            first = model(batch.features, batch.scenario_id)["logits"]
            second = model(reversed_features, batch.scenario_id)["logits"]
        result["timestamp_reversal_max_logit_delta"] = (first - second).abs().max().item()
        if not result["timestamp_reversal_max_logit_delta"] > 0:
            raise AssertionError("MORE is insensitive to timestamp order")
    return result


def trainer_probe(config, data):
    reader = replace(config.data.train.reader, num_workers=0, adapter_workers=0,
                     prefetch_batches=0, host_prepare_prefetch=0, overlap_host_prepare=False,
                     pin_memory=False, length_buckets=(), shuffle_buffer_rows=0)
    c = replace(config,
        data=replace(config.data, train=replace(config.data.train, inputs=(str(data),), reader=reader)),
        scenarios=replace(config.scenarios, names=tuple(str(i) for i in range(32)), auto_discover=False),
        training=replace(config.training, batch_size=1, gradient_accumulation_steps=1,
            checkpoint=replace(config.training.checkpoint, dir=None, resume="none", data_window_hours=0),
            fixed_test_eval=replace(config.training.fixed_test_eval, enabled=False), log_every_steps=1))
    with patch.object(train, "_build_model_on_device", small_model):
        return asdict(train.train_mdl(c, max_steps=3, log_steps=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--compile", action="store_true", help="Verify the dense-only compiled path")
    parser.add_argument("--models", nargs="+", choices=("uniformer", "more", "mixformer"),
                        default=("uniformer", "more"))
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    if args.compile:
        torch._inductor.config.compile_threads = 1
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    data = args.output / "synthetic_agg"
    config = config_for(args.models[0], args.device)
    manifest = generate_synthetic_agg_dataset(config, data, files=1, raw_rows_per_file=4,
        requests_per_agg=2, candidates_per_request=2,
        sequence_lengths={s.name: 8 for s in config.sequences}, physical_column_count=630)
    results = {"device": str(device), "compile": args.compile, "torch": torch.__version__, "data": asdict(manifest),
               "scope": "synthetic only; ordinary ID tables capped; dynamic rank table grows from test IDs; no production checkpoints", "models": {}}
    for name in args.models:
        config = config_for(name, args.device)
        config = replace(config, runtime=replace(config.runtime, compile=args.compile))
        results["models"][name] = {"updates": update_probe(config, device, args.steps),
                                   "trainer": trainer_probe(config, data)}
        print(json.dumps(results["models"][name], default=str), flush=True)
    with (args.output / "results.json").open("x") as output:
        json.dump(results, output, indent=2, default=str)


if __name__ == "__main__":
    main()
