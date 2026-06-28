# MDL Recommendation Project

This repository is organized as a standard recommendation-system project. The main implementation lives under `src/`; legacy files were moved out of this tree to a sibling legacy archive directory.

## Project Layout

```text
.
├── configs/
│   ├── default.yaml
│   ├── model/
│   │   ├── mdl.yaml
│   │   ├── rankmixer.yaml
│   │   └── deepfm.yaml
│   └── dataset/
│       └── manifest.yaml
├── data/
├── src/
│   ├── datasets/
│   │   ├── build_dataset.py
│   │   ├── feature_schema.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── base.py
│   │   ├── deepfm.py
│   │   ├── rankmixer.py
│   │   └── mdl.py
│   ├── modules/
│   │   ├── embedding.py
│   │   ├── tokenizer.py
│   │   ├── mlp.py
│   │   ├── attention.py
│   │   ├── loss.py
│   │   └── metrics.py
│   ├── trainers/
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── callbacks.py
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       ├── seed.py
│       └── checkpoint.py
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── experiments/
│   ├── runs/
│   ├── logs/
│   └── checkpoints/
├── tests/
└── notebooks/
```

## Data Contract

The core model package does not hard-code any dataset. Dataset-specific raw conversion should live in an adapter that writes the generic processed format. For full adapter implementation guidance, see [docs/adapter_development.md](docs/adapter_development.md). For agent execution prompts and step-by-step acceptance gates, see [docs/adapter_agent_playbook.md](docs/adapter_agent_playbook.md).

The processed format is:

```text
processed_dataset/
  manifest.json
  train.csv
  val.csv
  test.csv
  vocab__<feature_name>.json   # optional
```

The manifest declares scenario columns, group columns, labels, label masks, feature encoders, and token grouping. The current tokenization contract is:

```json
{
  "scenario_names": ["default"],
  "task_names": ["click"],
  "data_columns": {
    "scenario_id": "scene",
    "group_id": "query",
    "sample_weight": "sample_weight",
    "labels": {"click": "click_label"},
    "label_masks": {"click": "click_mask"}
  },
  "tokenization": {
    "version": 2,
    "kind": "encoder_registry",
    "features": [
      {"name": "user_id", "encoder": "embedding", "vocab_size": 100000, "source": {"type": "csv_column", "column": "user_id", "dtype": "int64"}},
      {"name": "score", "encoder": "identity", "dim": 1, "source": {"type": "csv_column", "column": "score", "dtype": "float32"}}
    ],
    "token_specs": [
      {"token_id": 0, "projection": "linear", "inputs": ["user_id", "score"]}
    ],
    "scenario_features": [
      {"name": "user_id", "encoder": "embedding", "vocab_size": 100000}
    ],
    "scenario_token_specs": [
      {"token_id": 0, "inputs": ["user_id"]},
      {"token_id": 1, "inputs": ["user_id"]}
    ],
    "task_features": [
      {"name": "score", "encoder": "identity", "dim": 1}
    ],
    "task_token_specs": [
      {"token_id": 0, "inputs": ["score"]}
    ]
  }
}
```

Built-in encoders are `embedding`, `identity`, multi-field `sequence_mean_pooling`, multi-field target-aware `din`, and long-sequence target-aware `sim`/`longer`. MDL manifests must declare `scenario_features/scenario_token_specs` and `task_features/task_token_specs`; model construction raises an error if any of these fields is missing. Single-scenario CSVs use `data_columns.scenario_id`; overlapping scenarios can use `data_columns.scenario_ids` with `scenario_ids_delimiter` such as `|`. If `data_columns.sample_weight` is declared, training and evaluation loss use it together with optional task/scenario weights.

## Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Validate a processed manifest dataset:

```bash
python scripts/preprocess.py --data-dir processed_dataset
```

Use `--max-rows N` to validate only the first `N` rows per split during fast adapter iteration.

Train MDL:

```bash
python scripts/train.py \
  --data-dir processed_dataset \
  --epochs 1 \
  --batch-size 256 \
  --max-steps 10 \
  --eval-max-batches 10 \
  --task-weights 1.0 \
  --scenario-weights 1.0
```

Enable RankMixer-style Sparse-MoE per-token FFNs:

```bash
python scripts/train.py \
  --data-dir processed_dataset \
  --ffn-type sparse_moe \
  --sparse-moe-num-experts 4 \
  --sparse-moe-loss-weight 1e-4 \
  --sparse-moe-target-active-ratio 0.25 \
  --sparse-moe-dtsi-infer-weight 0.5
```

Sparse-MoE uses ReLU routing, DTSI training by default, L1 regularization on the inference router, adaptive loss-weight control when `--sparse-moe-target-active-ratio` is set, configurable train/infer-router mixing via `--sparse-moe-dtsi-infer-weight`, and sparse expert execution during `eval()`/prediction. Training uses RMSProp for dense parameters and Adagrad for embedding-table parameters; override the embedding optimizer learning rate with `--sparse-lr`. Use `--disable-sparse-moe-dtsi` and the `--disable-*-tokens` / `--disable-*-feature-interaction` flags only for ablation.

Evaluate:

```bash
python scripts/evaluate.py \
  --data-dir processed_dataset \
  --split test \
  --checkpoint-path experiments/checkpoints/mdl.pt
```

Predict:

```bash
python scripts/predict.py \
  --data-dir processed_dataset \
  --split test \
  --checkpoint-path experiments/checkpoints/mdl.pt \
  --output-path experiments/runs/predictions.csv
```

## Testing

Run the focused tests:

```bash
python -m pytest tests
```
