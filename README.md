# MDL 推荐系统项目

## 项目结构

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

## 数据处理流程

数据集相关的原始数据转换逻辑放在 feature pipeline 中。

processed 数据格式如下：

```text
processed_dataset/
  manifest.json
  train.csv
  val.csv
  test.csv
  vocab__<feature_name>.json   # 可选
```

`manifest.json` 声明场景列、标签、标签 mask、特征 encoder 和 token 分组；可选声明 `group_id` 作为业务追踪字段。当前 tokenization 契约如下：

```json
{
  "scenario_names": ["default"],
  "task_names": ["click"],
  "data_columns": {
    "scenario_id": "scene",
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

内置 encoder 包括 `embedding`、`identity`、多字段 `sequence_mean_pooling`、多字段目标感知 `din`，以及长序列目标感知 `sim`/`longer`。MDL manifest 必须声明 `scenario_features/scenario_token_specs` 和 `task_features/task_token_specs`；如果缺少任意字段，模型构建会直接报错。纯 RankMixer baseline 只使用 `features/token_specs`，可用于 feature-only manifest。单场景 CSV 使用 `data_columns.scenario_id`；重叠场景可使用 `data_columns.scenario_ids`，并配合 `scenario_ids_delimiter`，例如 `|`。如果声明了 `data_columns.sample_weight`，训练和评估 loss 会同时使用该样本权重以及可选的 task/scenario 权重。

## 常用命令

安装依赖：

```bash
pip install -r requirements.txt
```

校验 processed manifest 数据集：

```bash
python scripts/preprocess.py --data-dir processed_dataset
```

在快速迭代 feature pipeline 时，可以用 `--max-rows N` 只校验每个 split 的前 `N` 行。

训练 MDL：

```bash
python scripts/train.py \
  --data-dir processed_dataset \
  --epochs 1 \
  --batch-size 256 \
  --max-steps 10 \
  --eval-max-batches 10 \
  --gradient-clip-norm 5.0 \
  --lr-scheduler cosine \
  --warmup-steps 2 \
  --min-lr-ratio 0.1 \
  --dense-weight-decay 1e-5 \
  --task-head-type mlp \
  --task-head-hidden-dim 64 \
  --task-head-dropout 0.0 \
  --auto-positive-class-weights \
  --task-weights 1.0 \
  --scenario-weights 1.0
```

`--task-head-type linear` 是 MDL 默认输出头，等价于每个 task token 接一层 `Linear(token_dim, 1)`；`--task-head-type mlp` 会改为每个任务一个两层 MLP head，便于和 RankMixer baseline 的 head 容量做更公平的 ablation。

训练纯 RankMixer baseline：

```bash
python scripts/train.py \
  --model-name rankmixer \
  --data-dir processed_dataset \
  --epochs 1 \
  --batch-size 256 \
  --max-steps 10
```

启用 RankMixer 风格的 Sparse-MoE per-token FFN：

```bash
python scripts/train.py \
  --data-dir processed_dataset \
  --ffn-type sparse_moe \
  --sparse-moe-num-experts 4 \
  --sparse-moe-loss-weight 1e-4 \
  --sparse-moe-target-active-ratio 0.25 \
  --sparse-moe-dtsi-infer-weight 0.5
```

Sparse-MoE 默认使用 ReLU routing 和 DTSI training，对 inference router 做 L1 正则；当设置 `--sparse-moe-target-active-ratio` 时，会启用自适应 loss weight 控制；`--sparse-moe-dtsi-infer-weight` 可配置训练/推理 router 的混合比例；在 `eval()`/prediction 阶段会使用稀疏 expert 执行。

评估：

```bash
python scripts/evaluate.py \
  --data-dir processed_dataset \
  --split test \
  --checkpoint-path experiments/checkpoints/mdl.pt
```

预测：

```bash
python scripts/predict.py \
  --data-dir processed_dataset \
  --split test \
  --checkpoint-path experiments/checkpoints/mdl.pt \
  --output-path experiments/runs/predictions.csv
```

## 测试

运行聚焦测试：

```bash
python -m pytest tests
```
