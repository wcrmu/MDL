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

The core model package does not hard-code any dataset. Dataset-specific raw conversion should live in an adapter that writes the generic processed format. For full adapter implementation guidance, see [docs/adapter_development.md](docs/adapter_development.md).

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
  "data_columns": {
    "scenario_id": "scene",
    "group_id": "query",
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
    ]
  }
}
```

Built-in encoders are `embedding`, `identity`, and `sequence_mean_pooling`.

## Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Validate a processed manifest dataset:

```bash
python scripts/preprocess.py --data-dir processed_dataset
```

Train MDL:

```bash
python scripts/train.py \
  --data-dir processed_dataset \
  --epochs 1 \
  --batch-size 256 \
  --max-steps 10 \
  --eval-max-batches 10
```

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
