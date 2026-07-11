# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `src/`. Parquet-native loading and model-batch
construction live in `src/dataloader.py`; categorical encoding helpers,
vocab fitting/loading, strategy fingerprints, and hash buckets live in
`src/features.py`; related code is in `src/config.py`, `src/model.py`,
`src/train.py`, and `src/main.py`. Reusable neural network blocks live in
`src/modules/`.

`src/main.py` is the application entry point. Run commands with
`python src/main.py ...` from the repository root. The core YAML template is
`configs/default.yaml`.

## Build, Test, and Development Commands

Install runtime dependencies in the secure environment (`torch`, `PyYAML`, `pyarrow`).

- `python src/main.py validate-config --config configs/default.yaml`: validate the YAML config.
- `python src/main.py profile --config configs/default.yaml --split train --max-batches 10`: inspect parquet schema, columns, and scan stats.
- `python src/main.py fit-vocab --config configs/default.yaml`: build configured vocab artifacts.
- `python src/main.py train --config configs/default.yaml --max-steps 10`: run a training smoke test.
- `python src/main.py train --config configs/default.yaml --distributed ddp --nproc-per-node 4`: launch single-node multi-GPU DDP training.

## Coding Style & Naming Conventions

Use Python 3.11+ style, 4-space indentation, type hints for public functions, and dataclasses for structured config or data contracts. Keep dataset-specific rules in config or data adapters, not in model classes. Name tests `test_*.py`, modules with lowercase snake_case, classes with PascalCase, and config keys with snake_case. Prefer small, explicit helpers over deep package nesting.

## Testing Guidelines

No in-repository test suite is kept in the slim runtime tree. Validate changes with `python src/main.py validate-config --config configs/default.yaml`, and a small secure-environment smoke run when parquet fixtures are available.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects, often scoped by area, such as `docs: clarify DIN target feature requirements`. Keep commits focused and mention user-visible behavior, config compatibility, or training/data implications when relevant. Pull requests should include a concise summary, tests run, linked issue when applicable, and any config or data migration steps.

## Security & Configuration Tips

Do not commit raw data, generated vocab artifacts, checkpoints, predictions, secrets, or secure environment paths. Keep data layout, vocab strategy, bucket sizes, and aggregation/flattening behavior in YAML so secure environments can adapt without model-code edits.
