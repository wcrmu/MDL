# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `src/`. Parquet-native loading and model-batch
construction live in `src/dataloader.py`; categorical encoding helpers,
vocab fitting/loading, strategy fingerprints, and hash buckets live in
`src/features.py`; related code is in `src/config.py`, `src/model.py`,
`src/train.py`, and `src/benchmark.py`. Reusable neural network blocks live in
`src/modules/`.

The root CLI is `mdl.py`. The core YAML template is `configs/default.yaml`; paper-alignment notes live in `PAPER_ALIGNMENT.md`, and local paper sources are under `paper/`.

## Build, Test, and Development Commands

- `pip install -r requirements.txt`: install runtime and test dependencies.
- `python mdl.py validate-config --config configs/default.yaml`: validate the YAML config.
- `python mdl.py profile --config configs/default.yaml --split train --max-batches 10`: inspect parquet schema, columns, and scan stats.
- `python mdl.py benchmark --config configs/default.yaml --split train --max-batches 10`: measure parquet scan and candidate decoding speed.
- `python mdl.py fit-vocab --config configs/default.yaml`: build configured vocab artifacts.
- `python mdl.py train --config configs/default.yaml --max-steps 10`: run a training smoke test.
- `python mdl.py train --config configs/default.yaml --distributed ddp --nproc-per-node 4`: launch single-node multi-GPU DDP training.
- `python mdl.py check-paper-alignment`: verify expected MDL and OneTrans paper-alignment markers.

## Coding Style & Naming Conventions

Use Python 3.11+ style, 4-space indentation, type hints for public functions, and dataclasses for structured config or data contracts. Keep dataset-specific rules in config or data adapters, not in model classes. Name tests `test_*.py`, modules with lowercase snake_case, classes with PascalCase, and config keys with snake_case. Prefer small, explicit helpers over deep package nesting.

## Testing Guidelines

No in-repository test suite is kept in the slim runtime tree. Validate changes with `python mdl.py validate-config --config configs/default.yaml`, `python mdl.py check-paper-alignment`, and a small secure-environment smoke run when parquet fixtures are available.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects, often scoped by area, such as `docs: clarify DIN target feature requirements`. Keep commits focused and mention user-visible behavior, config compatibility, or training/data implications when relevant. Pull requests should include a concise summary, tests run, linked issue or paper-alignment note when applicable, and any config or data migration steps.

## Security & Configuration Tips

Do not commit raw data, generated vocab artifacts, checkpoints, predictions, secrets, or secure environment paths. Keep data layout, vocab strategy, bucket sizes, and aggregation/flattening behavior in YAML so secure environments can adapt without model-code edits.
