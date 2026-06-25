# MDL Reproduction


## What Is Implemented

- Core MDL tensor model under `mdl/models/`: consumes precompiled feature tokens, scenario tokens, task tokens, domain-aware attention, domain-fused task/scenario aggregation, and per-task logits.
- RankMixer-style feature backbone: parameter-free multi-head TokenMixing plus per-token FFNs, selectable with `--feature-backbone rankmixer`.
- Registry-driven feature compiler under `mdl/tokenization/`: converts configured feature encoders into anonymous feature token slots.
- Tenrec adapter under `adapters/tenrec/`: converts local Tenrec CSV files into encoded train/val/test splits plus a `tokenization` spec.
- Tabular training CLI: trains MDL on encoded categorical/numeric recommendation data through the shared feature-token interface. Training helpers live under `mdl/training/`.
- Smoke tests with a tiny Tenrec-like fixture under `tests/fixtures/tenrec/`.

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

## Run Path 1: Public Tenrec Data

Use this path when reproducing the current public-data setup. The Tenrec adapter lives outside the core `mdl` package under `adapters/tenrec/`. It converts raw Tenrec CSV files into the generic processed format consumed by `mdl.train_tabular`.

### Download Tenrec

The code expects Tenrec files to be downloaded manually and placed on disk. The Tenrec paper describes the dataset as four scenarios with multiple feedback labels and true negatives, which is why it is the public dataset used here: <https://arxiv.org/abs/2210.10629>.

Create a raw data directory:

```bash
mkdir -p data/raw/tenrec
```

Download Tenrec from the official source linked by the Tenrec authors, then copy or symlink the CSV files into `data/raw/tenrec/`.

Expected filenames should contain one of these scenario names so the adapter can detect them:

- `QK-video`
- `QK-article`
- `QB-video`
- `QB-article`

Example layout:

```text
data/raw/tenrec/
  QK-video.csv
  QK-article.csv
  QB-video.csv
  QB-article.csv
```

The adapter also accepts nested directories and any `.csv` filename containing those scenario names.

### Prepare Tenrec

Encode the raw CSV files into train/validation/test splits:

```bash
PYTHONNOUSERSITE=1 conda run -n torch python -m adapters.tenrec.prepare \
  --raw-dir data/raw/tenrec \
  --out-dir data/processed/tenrec \
  --overwrite
```

This writes:

```text
data/processed/tenrec/
  manifest.json
  train.csv
  val.csv
  test.csv
  vocab__*.json
```

The split is deterministic by row order: 80% train, 10% validation, 10% test. This follows Tenrec's common 8:1:1 convention, but it is not the private Douyin Search split from the MDL paper.

For a quick preprocessing check on a small row cap:

```bash
PYTHONNOUSERSITE=1 conda run -n torch python -m adapters.tenrec.prepare \
  --raw-dir data/raw/tenrec \
  --out-dir /tmp/mdl-tenrec-small \
  --max-rows 10000 \
  --overwrite
```

### Train On Tenrec

Check GPU usage first:

```bash
nvidia-smi
```

Train on GPU 6:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONNOUSERSITE=1 conda run -n torch python -m mdl.train_tabular \
  --data-dir data/processed/tenrec \
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

Fast debug run:

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONNOUSERSITE=1 conda run -n torch python -m mdl.train_tabular \
  --data-dir data/processed/tenrec \
  --epochs 1 \
  --batch-size 256 \
  --max-steps 10 \
  --eval-max-batches 10 \
  --device cuda \
  --embedding-dim 16 \
  --token-dim 24 \
  --num-layers 1 \
  --num-heads 4 \
  --ffn-hidden-dim 32 \
  --feature-backbone rankmixer
```

CPU fallback:

```bash
PYTHONNOUSERSITE=1 conda run -n torch python -m mdl.train_tabular \
  --data-dir data/processed/tenrec \
  --epochs 1 \
  --batch-size 256 \
  --max-steps 10 \
  --eval-max-batches 10
```

Tenrec does not provide Douyin Search query IDs. This implementation uses encoded `user_id` as `group_id` for QAUC, so this is a public-data proxy rather than a faithful reproduction of the paper's query-level QAUC.

## Run Path 2: Private Dataset

Use this path when adapting a private or new public dataset. Do not put dataset-specific logic under `mdl/`. Create a dataset adapter under `adapters/<dataset_name>/` that converts raw files into the generic processed format.

Full adapter contract: [docs/adapter_guide.md](docs/adapter_guide.md).

The adapter must write:

```text
processed_dataset/
  manifest.json
  train.csv
  val.csv
  test.csv
  vocab__<feature_name>.json   # optional, recommended for categorical features
```

The split CSV files can use adapter-chosen physical column names. The manifest tells `mdl.data.EncodedTabularDataset` which columns represent scenario id, group id, labels, label masks, and feature sources.

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
      {"name": "some_id", "encoder": "categorical_embedding", "vocab_size": 100000, "source": {"type": "csv_column", "column": "user_col", "dtype": "int64"}},
      {"name": "some_score", "encoder": "numeric_value", "dim": 1, "source": {"type": "csv_column", "column": "score_col", "dtype": "float32"}}
    ],
    "token_specs": [
      {"token_id": 0, "projection": "linear", "inputs": ["some_id", "some_score"]}
    ]
  }
}
```

After your adapter writes a processed directory, train with the same generic command used for Tenrec:

```bash
PYTHONNOUSERSITE=1 conda run -n torch python -m mdl.train_tabular \
  --data-dir processed_dataset \
  --epochs 1 \
  --batch-size 256 \
  --max-steps 10 \
  --eval-max-batches 10
```

Built-in feature encoders currently include `categorical_embedding`, `numeric_value`, `sequence_mean_pooling`, and `dense_vector`. CSV parsing is declared by each feature's `source` (`column`, `dtype`, optional `shape`/`delimiter`) rather than inferred from the encoder name. Sequence CSV cells use `source.shape = "sequence"`; the generic collate path pads them and passes `values` plus `lengths` to `sequence_mean_pooling`. Feature grouping is declared by the input manifest in `tokenization.token_specs`. See [docs/adapter_guide.md](docs/adapter_guide.md) for details.

## Smoke Tests

Prepare the tiny Tenrec fixture:

```bash
PYTHONNOUSERSITE=1 conda run -n torch python -m adapters.tenrec.prepare \
  --raw-dir tests/fixtures/tenrec \
  --out-dir /tmp/mdl-tenrec-smoke \
  --overwrite
```

Run a short tabular train/eval pass:

```bash
PYTHONNOUSERSITE=1 conda run -n torch python -m mdl.train_tabular \
  --data-dir /tmp/mdl-tenrec-smoke \
  --epochs 1 \
  --batch-size 4 \
  --max-steps 2 \
  --eval-max-batches 2 \
  --embedding-dim 8 \
  --token-dim 24 \
  --num-layers 1 \
  --num-heads 4 \
  --ffn-hidden-dim 24 \
  --feature-backbone rankmixer
```

Run unit tests:

```bash
PYTHONNOUSERSITE=1 conda run -n torch python -m unittest tests.test_mdl_smoke tests.test_tenrec_pipeline
```

## Metrics

The training CLI reports:

- validation BCE loss;
- per-task AUC;
- per-task QAUC.

Tenrec does not provide Douyin Search query IDs. This implementation uses encoded `user_id` as `group_id` for QAUC, so this is a public-data proxy rather than a faithful reproduction of the paper’s query-level QAUC.
