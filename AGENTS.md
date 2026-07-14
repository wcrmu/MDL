# AGENTS.md

## Scope

- These instructions apply to the entire repository.
- A nested `AGENTS.md` may define more specific instructions for its directory.
- Direct user instructions take precedence over this file.

## Project purpose

This repository trains and evaluates large-scale recommendation models from
Parquet data. It implements paper-aligned variants of RankMixer, MDL,
OneTrans, and LONGER behind one shared config, dataloader, and training stack.

Primary goals:

- reproduce published model behavior under a common data pipeline;
- support single- and multi-GPU (DDP) training, evaluation, and benchmarking;
- keep model implementations comparable without dataset-specific logic in model
  classes.

## Repository map

- `src/main.py`: CLI entry point for all commands.
- `src/config.py`: YAML loading (`extends` overlays), validation, and resolved
  feature/token specs.
- `src/dataloader.py`: Parquet-native scanning, sharding, prefetch, and batch
  construction.
- `src/features.py`: categorical encoding, vocab fitting/loading, hash buckets,
  and strategy fingerprints.
- `src/model.py`: all model implementations and forward paths.
- `src/train.py`: training loop, DDP/sparse-embedding sync, evaluate/predict.
- `src/embeddings.py`, `src/optim.py`, `src/checkpoint.py`, `src/benchmark.py`:
  sharded embeddings, optimizers, checkpoints, and synthetic benchmarks.
- `src/modules/`: reusable attention, MLP, and MoE blocks.
- `configs/`: version-controlled experiment configs. `*_paper.yaml` profiles
  target paper settings; `*_perf.yaml` profiles target throughput experiments.
- `tests/`: unit and alignment tests (`unittest` + `pytest` discovery).
- `scripts/`: reproducible benchmark entry points.
- `examples/`: sample `adapter_parquet` callables for environment-specific
  preprocessing.

Configs compose via `extends:` (see `configs/rankmixer_paper.yaml`). The base
template is `configs/default.yaml`; it expects secure-environment data paths
and must be adapted locally, not hard-coded in Python.

## Commands

Run all commands from the repository root.

### Setup

- Python: 3.11+
- Runtime dependencies: `torch>=2.2`, `PyYAML>=6.0`, `pyarrow>=14.0`
- Install (when manifests are available locally):
  `pip install torch PyYAML pyarrow`

### CLI smoke checks

```bash
python src/main.py validate-config --config configs/default.yaml
python src/main.py profile --config configs/default.yaml --split train --max-batches 10
python src/main.py fit-vocab --config configs/default.yaml
python src/main.py train --config configs/default.yaml --max-steps 10
```

### Distributed training

```bash
python src/main.py train --config configs/default.yaml --distributed ddp --nproc-per-node 4
```

DDP is launched via `torch.distributed.run` from `src/main.py`. Sparse
embedding parameters are synchronized separately from dense DDP parameters.

### Benchmark, predict, evaluate

```bash
python src/main.py benchmark --config configs/mdl_perf.yaml --mode end-to-end --steps 10
python src/main.py predict --config configs/default.yaml --max-batches 1 --allow-random-init
python src/main.py evaluate --config configs/default.yaml --split test --max-batches 1 --allow-random-init
```

`benchmark --mode` choices: `data`, `embedding`, `compute`, `end-to-end`.
`scripts/run_benchmark_matrix.sh` sweeps GPU counts and modes for perf configs.

### Validation

```bash
python -m pytest -q
python -m pytest tests/test_config_overlays.py -q
python -m pytest tests/test_model_alignment.py -q
```

`tests/test_sparse_ddp.py::SparseDDPTest::test_two_gpu_nccl_sparse_smoke` is
skipped unless two CUDA devices are available. Report CPU vs GPU when results
depend on hardware.

Parquet-backed `profile`, `fit-vocab`, and `train` commands require local data
at the paths configured in YAML. They are not part of the default unit-test
suite.

## Working rules

- Inspect the nearest implementation and tests before editing.
- Make the smallest change that fully addresses the task.
- Keep dataset layout, vocab strategy, bucket sizes, and flattening rules in
  YAML or adapter callables—not in model classes.
- Preserve backward compatibility of config keys unless the task explicitly
  migrates them.
- Do not silently change default hyperparameters or evaluation metrics.
- Preserve tensor shape, dtype, and feature-encoding semantics across public
  APIs.
- New model surfaces must register through `model.name` validation in
  `src/config.py` and have a config profile under `configs/`.
- `model.name=mdl_onetrans` is experimental; keep it gated behind
  `experimental_model_acknowledged`.
- Prefer extending existing modules (`src/modules/`, `src/embeddings.py`) over
  duplicating blocks inside `src/model.py`.

## Verification by change type

- **Documentation-only changes**: no Python test required; verify referenced
  commands and paths still exist.

- **Config changes**:
  - `python src/main.py validate-config --config <changed-config>`
  - `python -m pytest tests/test_config_overlays.py -q`
  - run a `--max-steps 10` training smoke test when runtime/training fields
    change.

- **Dataloader or feature-encoding changes**:
  - `python -m pytest tests/test_data_prefetch.py tests/test_direct_id.py -q`
  - exercise empty, single-row, and multi-batch paths when batching logic
    changes.
  - never reinterpret hashed or identity-encoded IDs as raw categorical values.

- **Model architecture changes**:
  - `python -m pytest tests/test_model_alignment.py tests/test_longer_alignment.py -q`
  - add or update a forward/backward smoke test; assert finite loss/gradients.
  - for public shape changes, update alignment tests that compare against
    reference paths in `tests/test_model_alignment.py`.

- **Embedding / optimizer / checkpoint changes**:
  - `python -m pytest tests/test_sharded_embedding.py tests/test_checkpoint.py tests/test_fused_mlp.py -q`

- **Distributed training changes**:
  - `python -m pytest tests/test_ddp_config.py tests/test_sparse_ddp.py -q`
  - run the two-GPU NCCL smoke test when CUDA with `>=2` devices is available.

- **Benchmark changes**:
  - `python -m pytest tests/test_benchmark.py -q`

## Machine learning correctness

- Preserve input feature semantics, tensor shapes, and dtypes across train,
  predict, and evaluate.
- Do not materialize full embedding tables during data loading.
- Do not change data splitting, negative sampling, or group metrics (`qauc`,
  `uauc`) without explicit task approval.
- Seed-controlled tests must remain deterministic.
- Distributed code must handle ranks receiving different batch counts; sparse
  shards may finish at different steps.
- Training smoke test for training-loop changes: 2 batches, one optimizer step,
  finite loss, and checkpoint save/reload when checkpoint logic changes.

## Boundaries

- Do not commit raw data, generated vocab artifacts, checkpoints, predictions,
  benchmark outputs, or environment-specific paths.
- Do not edit `paper/` (local paper sources; not part of the runtime tree).
- Do not weaken, delete, or skip tests to make a failing implementation pass.
- Do not add runtime dependencies without explicit approval.
- Do not modify CI secrets, credentials, or production endpoints.
- Protected generated/local paths (see `.gitignore`): `/data/`, `/artifacts/`,
  `__pycache__/`, `.pytest_cache/`.

## Git and pull requests

- Keep each change scoped to the requested task; do not reformat unrelated
  files.
- Do not create commits unless explicitly requested.
- Commit subjects use short imperative phrasing, often scoped by area
  (e.g. `train: synchronize sparse DDP embeddings`).
- PRs should state motivation, validation commands run, and any config or data
  migration impact.

## Definition of done

A task is complete when:

1. the requested behavior is implemented;
2. relevant tests are added or updated;
3. applicable validation commands pass;
4. unrelated files remain unchanged;
5. public behavior or config changes are reflected in the relevant config or
   docs when agents can access them;
6. limitations and checks that could not be run (missing Parquet data, GPU,
   multi-GPU) are reported explicitly.

## Additional documentation

Read only when relevant:

- `PAPER_ALIGNMENT.md` (local): model-surface definitions and known deviations
  from papers.
- `configs/*_paper.yaml`: paper-target hyperparameters and token layouts.
- `examples/parquet_identity_adapter.py`: `adapter_parquet` callable contract.
- `tests/test_model_alignment.py`: reference patterns for model smoke and
  alignment tests.
