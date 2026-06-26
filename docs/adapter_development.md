# Adapter Development Guide

本文档说明如何为新的推荐系统数据集编写 adapter、adapter 应该放在哪里、目录如何组织，以及最终需要输出什么格式的数据，才能被当前 `MDL` 项目直接训练、评估和预测。

## 1. 核心原则

当前 `MDL` 仓库只保留通用训练框架、模型、数据读取器和脚本：

```text
<project-root>/
  src/
  scripts/
  configs/
  tests/
```

具体数据集的原始字段、清洗逻辑、负采样逻辑、ID 映射、时间切分、任务定义等，全部放在 adapter 里。adapter 不应该写进 `<project-root>/src`，也不应该在 `MDL` 内部硬编码某个数据集名称。

推荐把 adapter 放在 `MDL` 同级目录：

```text
  MDL/
  MDL_adapters/
    your_dataset/
```

adapter 的唯一职责是：

```text
raw dataset -> processed manifest dataset
```

也就是把任意原始数据集转换成 `MDL` 能读取的统一格式：

```text
processed_dataset/
  manifest.json
  train.csv
  val.csv
  test.csv
  vocab__<feature_name>.json    # optional
```

训练时，`MDL` 只接收这个 processed 目录：

```bash
cd <project-root>
python scripts/train.py --data-dir /path/to/processed_dataset
```

## 2. 推荐 Adapter 目录结构

建议每个数据集一个独立目录：

```text
<workspace>/MDL_adapters/
  your_dataset/
    README.md
    requirements.txt             # optional, only adapter-specific dependencies
    configs/
      default.yaml               # optional adapter config
    raw/                         # optional local raw data, usually not committed
      .gitkeep
    processed/                   # adapter output, usually not committed
      .gitkeep
    src/
      __init__.py
      preprocess.py              # main conversion logic
      schema.py                  # dataset-specific field names and feature definitions
      vocab.py                   # id mapping helpers
      split.py                   # train/val/test split helpers
      validate.py                # adapter-side validation
    scripts/
      preprocess.py              # CLI entrypoint
    tests/
      test_preprocess.py
```

小型 adapter 也可以更简单：

```text
<workspace>/MDL_adapters/
  your_dataset/
    README.md
    preprocess.py
```

但只要数据清洗逻辑较复杂，建议使用 `src/` 拆分，避免一个脚本变成不可维护的大文件。

## 3. Adapter 输出格式

### 3.1 必需文件

processed 目录至少应该包含：

```text
manifest.json
train.csv
```

如果要验证和测试模型，建议同时输出：

```text
val.csv
test.csv
```

`manifest.json` 中的 `splits` 字段必须和实际 CSV 文件对应：

```json
{
  "splits": ["train", "val", "test"]
}
```

### 3.2 CSV 文件规则

CSV 文件的物理列名由 adapter 自己决定，但必须在 `manifest.json` 里声明映射。

例如 CSV 可以长这样：

```csv
scene,query,click_label,click_mask,user_id,item_id,price,history_items
0,q1,1,1,12,331,19.9,4|8|15
0,q1,0,1,12,882,12.5,4|8|15
1,q2,1,1,44,102,8.0,3|9
```

要求：

- `scenario_id` 必须是从 `0` 开始的整数 ID。
- `group_id` 用于 QAUC 分组，通常是 query/session/request/user-session。
- 每个 task 都需要 label 列。
- 每个 task 都需要 label mask 列，mask 为 `1` 表示该样本该任务有效，`0` 表示忽略。
- 类别 ID 特征建议用整数，并保留 `0` 给 padding/unknown。
- 序列特征用一个字符串单元格保存，常见分隔符如 `|`，并在 manifest 中声明。

## 4. manifest.json 协议

一个完整 manifest 示例：

```json
{
  "splits": ["train", "val", "test"],
  "scenario_names": ["home", "search"],
  "task_names": ["click", "like"],
  "data_columns": {
    "scenario_id": "scene",
    "group_id": "query",
    "labels": {
      "click": "click_label",
      "like": "like_label"
    },
    "label_masks": {
      "click": "click_mask",
      "like": "like_mask"
    }
  },
  "tokenization": {
    "version": 2,
    "kind": "encoder_registry",
    "features": [
      {
        "name": "user_id",
        "encoder": "embedding",
        "vocab_size": 100000,
        "source": {
          "type": "csv_column",
          "column": "user_id",
          "dtype": "int64"
        }
      },
      {
        "name": "item_id",
        "encoder": "embedding",
        "vocab_size": 500000,
        "source": {
          "type": "csv_column",
          "column": "item_id",
          "dtype": "int64"
        }
      },
      {
        "name": "price",
        "encoder": "identity",
        "dim": 1,
        "source": {
          "type": "csv_column",
          "column": "price",
          "dtype": "float32"
        }
      },
      {
        "name": "history_items",
        "encoder": "sequence_mean_pooling",
        "vocab_size": 500000,
        "source": {
          "type": "csv_column",
          "column": "history_items",
          "dtype": "int64",
          "shape": "sequence",
          "delimiter": "|"
        }
      }
    ],
    "token_specs": [
      {
        "token_id": 0,
        "projection": "linear",
        "inputs": ["user_id"]
      },
      {
        "token_id": 1,
        "projection": "linear",
        "inputs": ["item_id", "price"]
      },
      {
        "token_id": 2,
        "projection": "linear",
        "inputs": ["history_items"]
      }
    ]
  }
}
```

### 4.1 顶层字段

`splits`
: 输出了哪些数据切分。当前通用读取器会按 `<split>.csv` 查找文件。

`scenario_names`
: 场景名称列表。CSV 中的 `scenario_id` 必须是这些名称的下标，例如 `home -> 0`，`search -> 1`。

`task_names`
: 多任务名称列表。label 和 label mask 都按这个顺序组成训练张量。

`data_columns`
: 声明 CSV 中哪些物理列对应场景、分组、label 和 mask。

`tokenization`
: 声明如何把 CSV 特征编码成匿名 feature tokens。

### 4.2 data_columns

```json
{
  "scenario_id": "scene",
  "group_id": "query",
  "labels": {"click": "click_label"},
  "label_masks": {"click": "click_mask"}
}
```

注意：

- `labels` 和 `label_masks` 的 key 必须覆盖 `task_names` 中的每个任务。
- label 建议使用 `0/1` 或可转成 float 的值。
- mask 建议使用 `0/1`。

### 4.3 feature source

当前通用 reader 支持 CSV column source：

```json
{
  "type": "csv_column",
  "column": "user_id",
  "dtype": "int64"
}
```

支持的 dtype：

- `int`, `int64`, `long`
- `float`, `float32`, `double`
- `bool`, `boolean`

标量特征默认：

```json
{
  "shape": "scalar"
}
```

序列特征：

```json
{
  "shape": "sequence",
  "delimiter": "|"
}
```

缺失值可以声明：

```json
{
  "missing_value": 0
}
```

序列 padding 可以声明：

```json
{
  "padding_value": 0
}
```

### 4.4 built-in encoders

当前内置 encoder 有三个：

`embedding`
: 用于整数 ID 特征。需要 `vocab_size` 或 `cardinality`。

```json
{
  "name": "user_id",
  "encoder": "embedding",
  "vocab_size": 100000
}
```

`identity`
: 用于连续数值特征。需要 `dim`。

```json
{
  "name": "price",
  "encoder": "identity",
  "dim": 1
}
```

`sequence_mean_pooling`
: 用于整数 ID 序列。需要 `vocab_size` 或 `cardinality`，输入 source 通常设置 `shape: sequence`。

```json
{
  "name": "history_items",
  "encoder": "sequence_mean_pooling",
  "vocab_size": 500000
}
```

### 4.5 token_specs

`features` 定义的是原始特征怎么编码；`token_specs` 定义的是哪些编码特征合成一个 feature token。

```json
{
  "token_id": 0,
  "projection": "linear",
  "inputs": ["user_id", "age", "gender"]
}
```

规则：

- `token_id` 从 `0` 开始，建议连续。
- `projection` 当前使用 `linear`。
- `inputs` 中的名字必须存在于 `features[*].name`。
- 一个 token 可以包含一个或多个 feature。
- RankMixer backbone 要求 `token_dim % num_feature_tokens == 0`。如果 token 数量变化，需要相应调整训练参数 `--token-dim`。

## 5. Adapter 实现步骤

### Step 1: 明确任务和粒度

先确定：

- 一行样本代表什么：曝光、点击候选、用户-物品交互、session item 等。
- 训练目标是什么：click、like、conversion、watch time binarization 等。
- 分组 ID 是什么：query、request、session、user-session。
- 场景有哪些：如果没有多场景，也定义一个场景，例如 `["default"]`，CSV 中 `scenario_id` 全部写 `0`。

### Step 2: 设计特征

把特征分成几类：

- sparse ID: user_id、item_id、category_id
- dense numeric: price、age、score、duration
- sequence: history_items、recent_categories
- context: device、hour、position、region

然后决定：

- 哪些 ID 需要建 vocab。
- 哪些连续值需要归一化。
- 哪些序列需要截断长度。
- 哪些字段缺失时如何填充。

### Step 3: 建立 ID 映射

推荐规则：

- `0` 保留给 unknown/padding。
- 从训练集构建 vocab，val/test 未见 ID 映射到 `0`。
- 输出 `vocab__feature_name.json` 方便复现和排查。

示例 vocab：

```json
{
  "<UNK>": 0,
  "raw_user_123": 1,
  "raw_user_456": 2
}
```

### Step 4: 切分 train/val/test

推荐根据数据性质选择：

- 有时间戳：按时间切分，避免未来信息泄漏。
- 无时间戳：按 user 或 group 做稳定随机切分。
- 排序/检索任务：同一个 query/session 的候选样本不要拆到不同 split。

### Step 5: 写 CSV

每个 split 写一个 CSV，字段名保持一致。不要把 Python 对象、JSON 对象直接塞进单元格；序列用明确 delimiter。

### Step 6: 写 manifest.json

manifest 应该由 adapter 代码生成，不建议手工维护。这样当特征、任务、vocab size 改变时，不容易不一致。

### Step 7: 验证输出

在 `MDL` 仓库中运行：

```bash
cd <project-root>
python scripts/preprocess.py --data-dir ../MDL_adapters/your_dataset/processed
python scripts/train.py --data-dir ../MDL_adapters/your_dataset/processed --epochs 1 --batch-size 32 --max-steps 2 --eval-max-batches 2
```

如果只是验证模型 forward，也可以先跑：

```bash
python -m pytest tests
```

## 6. 最小 Adapter 示例

下面是一个单文件 adapter 的骨架。实际项目建议拆到 `src/` 多文件。

```python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_vocab(values: list[str]) -> dict[str, int]:
    vocab = {"<UNK>": 0}
    for value in values:
        if value not in vocab:
            vocab[value] = len(vocab)
    return vocab


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def convert(raw_path: Path, output_dir: Path) -> None:
    # Replace this with real raw parsing.
    raw_rows = [
        {"user": "u1", "item": "i1", "label": 1, "group": "q1"},
        {"user": "u1", "item": "i2", "label": 0, "group": "q1"},
        {"user": "u2", "item": "i3", "label": 1, "group": "q2"},
    ]

    user_vocab = build_vocab([row["user"] for row in raw_rows])
    item_vocab = build_vocab([row["item"] for row in raw_rows])

    rows = []
    for row in raw_rows:
        rows.append(
            {
                "scene": 0,
                "query": row["group"],
                "click_label": row["label"],
                "click_mask": 1,
                "user_id": user_vocab.get(row["user"], 0),
                "item_id": item_vocab.get(row["item"], 0),
            }
        )

    fieldnames = ["scene", "query", "click_label", "click_mask", "user_id", "item_id"]
    write_csv(output_dir / "train.csv", rows, fieldnames)
    write_csv(output_dir / "val.csv", rows, fieldnames)
    write_csv(output_dir / "test.csv", rows, fieldnames)
    write_json(output_dir / "vocab__user_id.json", user_vocab)
    write_json(output_dir / "vocab__item_id.json", item_vocab)

    manifest = {
        "splits": ["train", "val", "test"],
        "scenario_names": ["default"],
        "task_names": ["click"],
        "data_columns": {
            "scenario_id": "scene",
            "group_id": "query",
            "labels": {"click": "click_label"},
            "label_masks": {"click": "click_mask"},
        },
        "tokenization": {
            "version": 2,
            "kind": "encoder_registry",
            "features": [
                {
                    "name": "user_id",
                    "encoder": "embedding",
                    "vocab_size": len(user_vocab),
                    "source": {
                        "type": "csv_column",
                        "column": "user_id",
                        "dtype": "int64",
                    },
                },
                {
                    "name": "item_id",
                    "encoder": "embedding",
                    "vocab_size": len(item_vocab),
                    "source": {
                        "type": "csv_column",
                        "column": "item_id",
                        "dtype": "int64",
                    },
                },
            ],
            "token_specs": [
                {"token_id": 0, "projection": "linear", "inputs": ["user_id"]},
                {"token_id": 1, "projection": "linear", "inputs": ["item_id"]},
            ],
        },
    }
    write_json(output_dir / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    convert(Path(args.raw_path), Path(args.output_dir))


if __name__ == "__main__":
    main()
```

运行：

```bash
cd ../MDL_adapters/your_dataset
python preprocess.py --raw-path raw/input.csv --output-dir processed

cd <project-root>
python scripts/preprocess.py --data-dir ../MDL_adapters/your_dataset/processed
python scripts/train.py --data-dir ../MDL_adapters/your_dataset/processed --epochs 1 --max-steps 2
```

## 7. 多任务样例

如果有 click 和 like 两个任务：

CSV：

```csv
scene,query,click_label,click_mask,like_label,like_mask,user_id,item_id
0,q1,1,1,0,1,12,331
0,q1,0,1,0,1,12,882
```

manifest：

```json
{
  "task_names": ["click", "like"],
  "data_columns": {
    "scenario_id": "scene",
    "group_id": "query",
    "labels": {
      "click": "click_label",
      "like": "like_label"
    },
    "label_masks": {
      "click": "click_mask",
      "like": "like_mask"
    }
  }
}
```

如果某个样本没有 like label，就写：

```csv
like_label=0,like_mask=0
```

mask 为 `0` 时，该任务 loss 和 metric 会忽略这个位置。

## 8. 序列特征样例

CSV：

```csv
history_items
1|2|3|4
7|8

```

manifest feature：

```json
{
  "name": "history_items",
  "encoder": "sequence_mean_pooling",
  "vocab_size": 100000,
  "source": {
    "type": "csv_column",
    "column": "history_items",
    "dtype": "int64",
    "shape": "sequence",
    "delimiter": "|",
    "padding_value": 0
  }
}
```

通用 collate 会把序列 padding 成 batch 内等长，并传入：

```python
{
  "values": LongTensor[batch, max_len],
  "lengths": LongTensor[batch]
}
```

`sequence_mean_pooling` 会基于 `lengths` 做 masked mean。

## 9. Adapter 测试建议

每个 adapter 至少做以下测试：

1. 能从一个小 raw fixture 生成 `manifest.json` 和 split CSV。
2. `manifest.json` 中 `splits` 对应的 CSV 都存在。
3. CSV 中 manifest 声明的列都存在。
4. `scenario_id` 不越界。
5. label 和 mask 可以转成 float。
6. embedding 特征 ID 在 `[0, vocab_size)` 范围内。
7. 序列特征中的每个 ID 在 `[0, vocab_size)` 范围内。
8. 用 `python scripts/train.py --max-steps 2` 能跑通。

## 10. 常见错误

`unknown feature encoder`
: manifest 中的 `encoder` 名称不是内置 encoder。当前支持 `embedding`、`identity`、`sequence_mean_pooling`。

`feature ... csv_column source must declare dtype`
: 每个 feature source 都必须声明 dtype。

`token_dim must be divisible by number of feature tokens`
: 使用 RankMixer backbone 时，`--token-dim` 必须能被 `token_specs` 数量整除。可以调整 token 数量，或训练时传入新的 `--token-dim`。

`label_mask must have the same shape as logits`
: `task_names`、`labels`、`label_masks` 没有一一对应。

训练 loss 为 `nan`
: 常见原因是 label 不是合法数值、mask 全为 0、连续特征存在空值但未声明 missing value。

QAUC 有大量 skipped groups
: 很多 group 内只有正样本或只有负样本。检查 group 定义是否合理，或者数据切分是否破坏了同一个 group 的候选集合。

## 11. 不应该做什么

不要把某个数据集的字段名写进 `<project-root>/src`。

不要把 raw 大文件提交进 `MDL` 仓库。

不要在 adapter 输出中依赖 Python pickle 作为训练输入；通用 reader 当前读取的是 manifest + CSV。

不要在 val/test 上重新 fit vocab 或 normalization 参数。应只使用 train 上得到的映射和统计量。

不要让同一个 query/session 的候选样本被拆到不同 split，除非你的任务定义明确允许。

## 12. 推荐交付清单

一个完整 adapter 最好交付：

```text
your_dataset/
  README.md
  requirements.txt
  configs/default.yaml
  src/
  scripts/preprocess.py
  tests/
```

并在 README 中写清楚：

- raw 数据应该放在哪里。
- 运行 adapter 的命令。
- 输出 processed 目录位置。
- 输出的任务和场景。
- 特征列表和 vocab 规则。
- 如何用 `<project-root>/scripts/train.py` 做 smoke train。

