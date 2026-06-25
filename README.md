# MDL Reproduction

## What Is Implemented

- Core MDL tensor model under `mdl/models/`: consumes precompiled feature tokens, scenario tokens, task tokens, domain-aware attention, domain-fused task/scenario aggregation, and per-task logits.
- RankMixer-style feature backbone: parameter-free multi-head TokenMixing plus per-token FFNs, selectable with `--feature-backbone rankmixer`.
- Registry-driven feature compiler under `mdl/tokenization/`: converts configured feature encoders into anonymous feature token slots.
- Generic processed-data reader under `mdl/data/`: reads adapter-produced CSV splits through a manifest-declared feature-token interface.
- Manifest training CLI: trains MDL on encoded recommendation data through the shared feature-token interface. Training helpers live under `mdl/utils/`.

## Environment

Use the existing Conda environment with PyTorch:

```bash
PYTHONNOUSERSITE=1 conda run -n torch python -c "import torch; print(torch.__version__)"
```

Before GPU training, inspect utilization and prefer GPUs `6` and `7`:

```bash
nvidia-smi
```

For a single-GPU run on GPU 6:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONNOUSERSITE=1 conda run -n torch python -c "import torch; print(torch.cuda.is_available())"
```

For a two-GPU-visible run:

```bash
CUDA_VISIBLE_DEVICES=6,7 PYTHONNOUSERSITE=1 conda run -n torch python -c "import torch; print(torch.cuda.device_count())"
```

Current training code uses one process/model. `CUDA_VISIBLE_DEVICES=6` is the safest default unless multi-GPU support is added.

## Dataset Adapters

Use this path when adapting a private or public dataset. Do not put dataset-specific logic under `mdl/`. Create a dataset adapter under `adapters/<dataset_name>/` that converts raw files into the generic processed format.

Full adapter contract: [docs/adapter_guide.md](docs/adapter_guide.md).

The adapter must write:

```text
processed_dataset/
  manifest.json
  train.csv
  val.csv
  test.csv
  vocab__<feature_name>.json   # optional, recommended for id features
```

The split CSV files can use adapter-chosen physical column names. The manifest tells `mdl.data.ManifestDataset` which columns represent scenario id, group id, labels, label masks, and feature sources.

The manifest must use the latest tokenization contract:

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
      {"name": "some_id", "encoder": "embedding", "vocab_size": 100000, "source": {"type": "csv_column", "column": "user_col", "dtype": "int64"}},
      {"name": "some_score", "encoder": "identity", "dim": 1, "source": {"type": "csv_column", "column": "score_col", "dtype": "float32"}},
      {"name": "history_ids", "encoder": "sequence_mean_pooling", "vocab_size": 500000, "source": {"type": "csv_column", "column": "history_col", "dtype": "int64", "shape": "sequence", "delimiter": "|"}}
    ],
    "token_specs": [
      {"token_id": 0, "projection": "linear", "inputs": ["some_id", "some_score"]},
      {"token_id": 1, "projection": "linear", "inputs": ["history_ids"]}
    ]
  }
}
```

Built-in feature encoders currently include `embedding`, `identity`, and `sequence_mean_pooling`. CSV parsing is declared by each feature's `source` (`column`, `dtype`, optional `shape`/`delimiter`) rather than inferred from the encoder name. Sequence CSV cells use `source.shape = "sequence"`; the generic collate path pads them and passes `values` plus `lengths` to `sequence_mean_pooling`. Feature grouping is declared by the input manifest in `tokenization.token_specs`.

## Training

After your adapter writes a processed directory, train with:

```bash
PYTHONNOUSERSITE=1 conda run -n torch python -m mdl.train \
  --data-dir processed_dataset \
  --epochs 1 \
  --batch-size 256 \
  --max-steps 10 \
  --eval-max-batches 10
```

GPU example:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONNOUSERSITE=1 conda run -n torch python -m mdl.train \
  --data-dir processed_dataset \
  --epochs 1 \
  --batch-size 2048 \
  --device cuda \
  --embedding-dim 32 \
  --token-dim 36 \
  --num-layers 2 \
  --num-heads 4 \
  --ffn-hidden-dim 64 \
  --feature-backbone rankmixer
```

## Smoke Test

Run a synthetic model smoke test:

```bash
PYTHONNOUSERSITE=1 conda run -n torch python -m mdl.train_smoke --steps 3
```

## Metrics

The training CLI reports:

- validation BCE loss;
- per-task AUC;
- per-task QAUC.
