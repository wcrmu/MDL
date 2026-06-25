# Adapter Guide

This guide defines the output contract for dataset adapters. An adapter converts a raw
dataset into the standard processed format consumed by the generic MDL tabular training
pipeline.

## Pipeline Position

```text
raw dataset
  -> adapter
  -> processed splits + manifest
  -> mdl/data
  -> mdl/interfaces or tabular wrapper
  -> mdl/tokenization
  -> feature_tokens
  -> mdl/models
```

Adapters live outside the core `mdl` package. For example, Tenrec-specific code lives in
`adapters/tenrec/`.

## Required Output Directory

An adapter must write a processed data directory with this shape:

```text
processed_dataset/
  manifest.json
  train.csv
  val.csv
  test.csv
  vocab__<feature_name>.json   # optional, recommended for categorical features
```

The generic training command consumes the directory through `--data-dir`:

```bash
PYTHONNOUSERSITE=1 conda run -n torch python -m mdl.train_tabular \
  --data-dir processed_dataset
```

## Split CSV Format

For the current generic CSV reader, split CSV columns are not fixed by global prefixes. The adapter chooses column names and declares them in `manifest.json`.

Example:

```csv
scene,query,user_id,item_id,score,click_label,click_mask
0,123,42,991,0.137,1,1
```

The corresponding manifest maps semantic roles to physical CSV columns:

```json
{
  "data_columns": {
    "scenario_id": "scene",
    "group_id": "query",
    "labels": {"click": "click_label"},
    "label_masks": {"click": "click_mask"}
  },
  "tokenization": {
    "features": [
      {
        "name": "user_id",
        "encoder": "categorical_embedding",
        "vocab_size": 100000,
        "source": {"type": "csv_column", "column": "user_id"}
      },
      {
        "name": "score",
        "encoder": "numeric_value",
        "dim": 1,
        "source": {"type": "csv_column", "column": "score"}
      }
    ]
  }
}
```

Rules:

- The scenario column value is a zero-based integer index into `manifest.scenario_names`.
- The group column is used for QAUC grouping. Use a query/session/user id when available.
- Categorical feature source columns must already be integer encoded. Reserve `0` for padding/unknown.
- Numeric feature source columns should already be normalized if normalization is desired.
- Label mask columns must be `1` when the task label is valid for that row and `0` when unavailable.

## Manifest Format

`manifest.json` must use the latest tokenization contract:

```json
{
  "dataset": "my_dataset",
  "scenario_names": ["scene_a", "scene_b"],
  "task_names": ["click", "like"],
  "data_columns": {
    "scenario_id": "scene",
    "group_id": "query",
    "labels": {"click": "click_label", "like": "like_label"},
    "label_masks": {"click": "click_mask", "like": "like_mask"}
  },
  "tokenization": {
    "version": 2,
    "kind": "encoder_registry",
    "features": [
      {"name": "user_id", "encoder": "categorical_embedding", "vocab_size": 100000, "source": {"type": "csv_column", "column": "user_id"}},
      {"name": "item_id", "encoder": "categorical_embedding", "vocab_size": 500000, "source": {"type": "csv_column", "column": "item_id"}},
      {"name": "score", "encoder": "numeric_value", "dim": 1, "source": {"type": "csv_column", "column": "score"}}
    ],
    "token_specs": [
      {"token_id": 0, "projection": "linear", "inputs": ["user_id", "score"]},
      {"token_id": 1, "projection": "linear", "inputs": ["item_id"]}
    ]
  },
  "splits": {
    "train": 1000,
    "val": 100,
    "test": 100
  },
  "group_id": "query/session/user id used for QAUC"
}
```

Required manifest fields:

- `scenario_names`: ordered scenario names. CSV `scenario_id` indexes this list.
- `task_names`: ordered task names. `data_columns.labels` and `data_columns.label_masks` map each task to its physical CSV columns.
- `data_columns`: CSV columns for scenario, group id, labels, and label masks.
- `tokenization.version`: must be `2`.
- `tokenization.kind`: must be `"encoder_registry"`.
- `tokenization.features`: feature encoder declarations.
- `tokenization.token_specs`: feature-token grouping declarations.
- `splits`: row counts for `train`, `val`, and `test`.

## Tokenization Configuration

`tokenization.features` declares how each named feature is encoded. The main compiler
does not hard-code feature types; it builds encoders from this registry config.

Built-in encoders:

- `categorical_embedding`: reads one named feature tensor and applies `nn.Embedding`.
- `numeric_value`: reads one named feature tensor and projects one or more numeric dimensions.
- `sequence_mean_pooling`: pools padded categorical id sequences with a mask or lengths.
- `dense_vector`: consumes an already dense tensor from a custom interface.

`tokenization.token_specs` declares feature grouping:

```json
[
  {"token_id": 0, "projection": "linear", "inputs": ["user_id", "score"]},
  {"token_id": 1, "projection": "linear", "inputs": ["item_id"]}
]
```

Grouping is a manifest-only change as long as all input names exist in `features`.

## Current Generic CSV Support

The current `mdl.train_tabular` path reads feature values through `tokenization.features[*].source`. For CSV-backed features, use:

```json
"source": {"type": "csv_column", "column": "physical_column_name"}
```

That means `categorical_embedding` and `numeric_value` work end-to-end through the generic CSV reader today without any required `cat__` or `num__` prefix.

`sequence_mean_pooling` and `dense_vector` are supported by `FeatureTokenCompiler`, but the
generic CSV reader does not yet load sequence or dense tensors from disk. To use them today, add
a dataset-specific reader/collate function or interface that passes:

```python
features={
    "hist_ids": {
        "values": LongTensor[B, L],
        "lengths": LongTensor[B]  # or "mask": BoolTensor[B, L]
    },
    "image_embedding": FloatTensor[B, D]
}
```

## Adapter Checklist

1. Decide scenario names and encode each row's `scenario_id`.
2. Decide task names, emit physical label/mask columns, and map them in `data_columns`.
3. Encode categorical features into integer ids, reserving `0` for padding/unknown, and declare their physical CSV columns in `source.column`.
4. Normalize numeric features if needed and declare their physical CSV columns in `source.column`.
5. Write `train.csv`, `val.csv`, and `test.csv` with identical headers.
6. Write `manifest.json` with `tokenization.version = 2`.
7. Make sure every token input name exists in `tokenization.features`.
8. Run a smoke training command against the processed directory.

## Minimal Adapter Skeleton

```python
from pathlib import Path


def prepare_my_dataset(raw_dir: str | Path, out_dir: str | Path) -> dict[str, object]:
    # 1. Read raw files.
    # 2. Build categorical vocabularies.
    # 3. Compute numeric normalization statistics if needed.
    # 4. Write train.csv, val.csv, test.csv.
    # 5. Write manifest.json using tokenization version 2.
    # 6. Return the manifest dictionary.
    raise NotImplementedError
```

Use `adapters/tenrec/adapter.py` as a concrete example of a dataset adapter.
