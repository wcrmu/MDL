# Feature Pipeline Data Layout Migration Guide

本文档给下游 feature pipeline 作者和 agent 使用，用于把已经实现或即将实现的 feature pipeline 统一迁移到 MDL 仓库内的 `data/` 目录。目标是统一数据工程代码和数据产物的位置，同时不改变 MDL core 的训练、评估和 manifest reader 逻辑。

如果需要直接分发给下游 agent，使用 [feature_pipeline_data_layout_migration_prompt.md](feature_pipeline_data_layout_migration_prompt.md)。

## 1. 新目录约定

标准位置：

```text
MDL/
  data/
    pipelines/<dataset_name>/
    processed/<dataset_name>/
    raw/<dataset_name>/
    fixtures/<dataset_name>/
```

职责边界：

- `data/pipelines/<dataset_name>/`: 数据集专属 feature pipeline 代码、配置、报告和自测。
- `data/processed/<dataset_name>/`: feature pipeline 生成的 MDL 训练输入，包括 `manifest.json`、`train.csv`、`val.csv`、`test.csv` 和可选 vocab 文件。
- `data/raw/<dataset_name>/`: 原始数据、本地软链或本地缓存，默认不提交。
- `data/fixtures/<dataset_name>/`: 可提交的小样本 fixture，用于 pipeline 单测。

不要把 dataset-specific 逻辑写入 `src/`、`scripts/`、`configs/` 或 `tests/`。这些目录属于 MDL core。

## 2. 从旧位置迁移

旧位置如果是：

```text
../MDL_feature_pipelines/<dataset_name>/
```

迁移到：

```text
<project-root>/data/pipelines/<dataset_name>/
```

输出目录从：

```text
../MDL_feature_pipelines/<dataset_name>/processed
```

改为：

```text
<project-root>/data/processed/<dataset_name>
```

如果原 pipeline 内部有 `raw/`、`processed/`、`reports/`：

- `reports/` 保留在 `data/pipelines/<dataset_name>/reports/`。
- `processed/` 不再放在 pipeline 根目录下，改为输出到 `data/processed/<dataset_name>/`。
- `raw/` 如果只是本地数据或软链，改为 `data/raw/<dataset_name>/`；不要提交大文件。

## 3. 下游 Agent 修改边界

允许修改：

```text
data/pipelines/<dataset_name>/**
data/processed/<dataset_name>/**
data/fixtures/<dataset_name>/**
```

通常不应该修改：

```text
src/**
scripts/**
configs/**
tests/**
```

例外：只有在用户明确要求改 MDL core、manifest reader、训练流程或通用 encoder 时，才可以修改这些 core 目录。

## 4. Feature Pipeline 仍然负责什么

迁移目录后，feature pipeline 的职责不变：

- raw parsing。
- sample construction。
- `label` / `label_mask` 定义。
- `group_id` 和 `scenario_id` 生成。
- train/val/test split。
- vocab fit 和 val/test unknown ID 映射。
- dense 特征清洗、clip、归一化。
- sequence 排序、截断、padding。
- `features` / `token_specs`、`scenario_features` / `scenario_token_specs`、`task_features` / `task_token_specs` 生成。
- `manifest.json` 和 split CSV 生成。

MDL core 仍然负责：

- manifest reader。
- collate。
- model forward。
- training loop。
- evaluation。
- checkpoint。

只要 processed dataset 的协议不变，MDL 训练代码不需要改。

## 5. 命令更新

校验 processed dataset：

```bash
cd <project-root>
python scripts/preprocess.py --data-dir data/processed/<dataset_name> --max-rows 1000
```

smoke train：

```bash
cd <project-root>
python scripts/train.py \
  --data-dir data/processed/<dataset_name> \
  --epochs 1 \
  --batch-size 32 \
  --max-steps 2 \
  --eval-max-batches 2
```

如果 feature pipeline 自带测试：

```bash
cd <project-root>/data/pipelines/<dataset_name>
python -m pytest tests
```

## 6. 迁移检查清单

迁移后必须检查旧路径和旧术语残留：

```bash
rg -n "MDL_feature_pipelines|<feature-pipeline-root>|adapter|Adapter|ADAPTER" \
  data/pipelines/<dataset_name>
```

如果搜索结果中的 `adapter` 属于第三方库、历史兼容说明或外部 URL，不要直接替换，先列入迁移报告。

确认 pipeline 输出路径：

```bash
ls data/processed/<dataset_name>/manifest.json
ls data/processed/<dataset_name>/train.csv
```

确认 MDL core 可以读取：

```bash
python scripts/preprocess.py --data-dir data/processed/<dataset_name> --max-rows 1000
```

## 7. 未来 HDFS/DDP 边界

本次目录迁移不要求实现 HDFS 或 DDP。

为未来大规模训练预留的边界是：

- feature pipeline 可以生成 sharded processed files，例如 `train/part-00000.csv`。
- feature pipeline 不要实现 rank、world size、worker id 分片逻辑。
- feature pipeline 不要直接封装 DDP lifecycle。
- 未来 HDFS/DDP 读取应由 MDL core 的通用 data I/O 层处理，例如 `src/dataio/`。

也就是说：

```text
feature pipeline: raw -> processed manifest dataset
MDL dataio: local/HDFS read + shard discovery + rank/worker partition
trainer: model training/evaluation/checkpoint
```

如果当前数据集必须从 HDFS 读取 raw 数据来生成 processed dataset，可以在 `data/pipelines/<dataset_name>/src/` 内实现 raw ingestion。但训练时的 HDFS streaming 和 DDP shard assignment 不应该放在 dataset-specific pipeline 中。

## 8. 迁移报告模板

迁移完成后，下游 agent 应输出：

```text
Feature pipeline data layout migration completed.

Dataset:
<dataset_name>

Moved:
- <old path> -> data/pipelines/<dataset_name>
- processed output -> data/processed/<dataset_name>

Updated:
- README paths
- config output_dir
- scripts CLI defaults
- reports references
- tests paths

Validation:
- old naming search: PASS/FAIL
- preprocess validation: PASS/FAIL
- smoke train: PASS/FAIL
- pipeline tests: PASS/FAIL or N/A

Notes:
- <any remaining user decision or framework change>
```

## 9. 停止条件

遇到以下情况，不要继续迁移：

- pipeline 代码硬编码了旧绝对路径且无法确认新路径。
- processed 数据太大，迁移会复制大量文件但用户没有明确授权。
- 需要修改 MDL core 才能让 pipeline 跑通。
- 发现 manifest 协议无法表达当前数据集。

输出：

```text
NEEDS_USER_DECISION: <具体问题>
```

或：

```text
NEEDS_FRAMEWORK_CHANGE: <需要 MDL core 支持什么>
```
