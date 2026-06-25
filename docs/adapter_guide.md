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
        "encoder": "embedding",
        "vocab_size": 100000,
        "source": {"type": "csv_column", "column": "user_id", "dtype": "int64"}
      },
      {
        "name": "score",
        "encoder": "identity",
        "dim": 1,
        "source": {"type": "csv_column", "column": "score", "dtype": "float32"}
      }
    ]
  }
}
```

Rules:

- The scenario column value is a zero-based integer index into `manifest.scenario_names`.
- The group column is used for QAUC grouping. Use a query/session/user id when available.
- Feature CSV parsing is declared by each feature's `source`, not inferred from `encoder`.
- Scalar integer-like features should use `source.dtype = "int64"`; reserve `0` for padding/unknown when the encoder treats zero specially.
- Scalar numeric features should use `source.dtype = "float32"` and should already be normalized if normalization is desired.
- Vector/list features can use `source.shape = "vector"` plus an optional `source.delimiter`; rows must have a consistent padded length for the generic collate path.
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
      {"name": "user_id", "encoder": "embedding", "vocab_size": 100000, "source": {"type": "csv_column", "column": "user_id", "dtype": "int64"}},
      {"name": "item_id", "encoder": "embedding", "vocab_size": 500000, "source": {"type": "csv_column", "column": "item_id", "dtype": "int64"}},
      {"name": "score", "encoder": "identity", "dim": 1, "source": {"type": "csv_column", "column": "score", "dtype": "float32"}}
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

`tokenization.features` declares how each named feature is read and encoded. The
`source` object is owned by the input adapter and controls CSV parsing (`column`, `dtype`,
optional `shape`, optional `delimiter`). The `encoder` object controls model-side encoding.
The data reader must not infer parsing behavior from the encoder name.

Built-in encoders:

- `embedding`: reads one named id tensor and applies `nn.Embedding`.
- `identity`: reads one scalar/vector tensor, casts it to float, validates `dim`, and passes it through unchanged.
- `sequence_mean_pooling`: embeds padded id sequences and mean-pools them with a mask or lengths.

`tokenization.token_specs` declares feature grouping:

```json
[
  {"token_id": 0, "projection": "linear", "inputs": ["user_id", "score"]},
  {"token_id": 1, "projection": "linear", "inputs": ["item_id"]}
]
```

Grouping is defined by the input manifest. It is a manifest-only change as long as all input names exist in `features`; no core-code feature grouping should be hard-coded.

## Current Generic CSV Support

The current `mdl.train_tabular` path reads feature values through `tokenization.features[*].source`. For CSV-backed features, use:

```json
"source": {"type": "csv_column", "column": "physical_column_name", "dtype": "float32"}
```

The generic CSV reader supports scalar, fixed-length vector/list, and variable-length sequence cells based on `source.dtype` and `source.shape`; it does not branch on `embedding`, `identity`, or any other encoder name. There are no required `cat__` or `num__` prefixes.

Supported `source` fields:

- `type`: must be `"csv_column"` for the generic CSV reader.
- `column`: physical CSV column name.
- `dtype`: one of `"int64"`, `"float32"`, `"double"`, or `"bool"`. Aliases `"int"`, `"long"`, `"float"`, and `"boolean"` are accepted.
- `shape`: optional. Omit it for scalar cells; use `"vector"`/`"list"` for fixed-length dense cells; use `"sequence"` for variable-length sequence cells.
- `delimiter`: optional for vector/list/sequence cells. If omitted, whitespace splitting is used.
- `missing_value`: optional scalar or list used when a CSV cell is empty. Defaults are `0`, `0.0`, or `false` based on `dtype`.
- `padding_value`: optional sequence padding value used during collation. Defaults to the dtype missing value, usually `0` for id sequences.

Scalar cells produce tensors shaped `[B]` before encoder-specific reshaping. Example:

```json
{"type": "csv_column", "column": "score", "dtype": "float32"}
```

Fixed vector/list cells produce tensors shaped `[B, D]`. Every row in a batch must have the same length. Example:

```json
{"type": "csv_column", "column": "image_vec", "dtype": "float32", "shape": "vector", "delimiter": "|"}
```

```csv
image_vec
0.1|0.3|-0.2
0.0|0.5|0.4
```

A sequence feature is declared with `source.shape = "sequence"`. The CSV cell stores one row's sequence using the declared delimiter. The generic collate path pads each batch to the batch-local max length and passes `{"values": LongTensor[B, L], "lengths": LongTensor[B]}` to the encoder. Empty cells become length-0 sequences and are padded within the batch.

For `sequence_mean_pooling`, use integer ids (`dtype = "int64"`), provide `vocab_size` or `cardinality`, and reserve padding id `0` unless you override `padding_value` and the encoder supports that convention.

```json
{
  "name": "hist_item_ids",
  "encoder": "sequence_mean_pooling",
  "vocab_size": 500000,
  "source": {
    "type": "csv_column",
    "column": "hist_items",
    "dtype": "int64",
    "shape": "sequence",
    "delimiter": "|"
  }
}
```

```csv
hist_items
12|33|91
44

```

The two non-empty rows above collate as:

```python
features["hist_item_ids"] = {
    "values": LongTensor([[12, 33, 91], [44, 0, 0]]),
    "lengths": LongTensor([3, 1]),
}
```

For custom sequence or dense interfaces, a dataset-specific reader/collate function can also
pass explicit tensors directly:

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
3. Declare every feature source with `source.type`, `source.column`, and `source.dtype`; add `source.shape`/`source.delimiter` for vector or sequence cells when needed.
4. Encode categorical-like inputs into integer ids when their chosen encoder expects ids, reserving `0` for padding/unknown when relevant.
5. Normalize numeric-like inputs if needed before writing CSV.
6. Write `train.csv`, `val.csv`, and `test.csv` with identical headers.
7. Write `manifest.json` with `tokenization.version = 2`.
8. Define feature grouping in `tokenization.token_specs` and make sure every token input name exists in `tokenization.features`.
9. Run a smoke training command against the processed directory.

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
