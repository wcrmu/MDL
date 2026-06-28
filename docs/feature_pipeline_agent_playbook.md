# Weak Agent Feature Pipeline Playbook

本文档给 agent 使用。目标是让 agent 在内部机器上分析本地原始数据，并实现一个外部 dataset-specific feature pipeline，同时不修改 `MDL/` 仓库中的任何文件。

如果你是人类开发者，请先读 [feature_pipeline_development.md](feature_pipeline_development.md)。如果你要把任务交给一个 agent，请把本文档中的 prompt 和步骤交给它。

## 1. 总原则

agent 不应该自由发挥。它必须按阶段工作，每阶段只做一类事情，并在满足验收条件后再进入下一阶段。

最重要的边界：

- 不修改 `MDL/` 仓库中的任何文件。
- 只在外部 feature pipeline 目录中创建和修改文件，例如 `../MDL_feature_pipelines/<dataset_name>/`。
- 不把 raw 数据复制进 `MDL/` 仓库。
- 不把 raw 大文件提交到 git。
- 不改变 `MDL/src`、`MDL/scripts`、`MDL/configs`、`MDL/tests`。
- 不新增某个数据集专属逻辑到 `MDL`。
- 如果当前通用协议不够用，停止并报告，不要绕开协议硬写。

普通 feature pipeline 的唯一目标：

```text
raw dataset -> processed manifest dataset
```

最终输出：

```text
processed/
  manifest.json
  train.csv
  val.csv
  test.csv
  vocab__<feature_name>.json    # optional
```

## 2. 推荐给 Agent 的总 Prompt

把下面这段作为任务开头发给 agent。把尖括号变量替换成真实路径和数据集名称。

```text
你要为 MDL 项目实现一个外部 dataset-specific feature pipeline。你能力有限，所以必须严格按步骤执行，不允许自由发挥。

路径：
- MDL 仓库根目录：<project-root>
- feature pipeline 根目录：<feature-pipeline-root>
- 原始数据目录或文件：<raw-data-path>
- 数据集名称：<dataset-name>

硬性边界：
1. 禁止修改 <project-root> 下任何文件。
2. 只能在 <feature-pipeline-root> 下创建或修改文件。
3. 禁止把 raw 大文件复制到 <project-root>。
4. 禁止把数据集专属字段、路径、逻辑写进 MDL/src、MDL/scripts、MDL/configs 或 MDL/tests。
5. 如果发现 MDL 的通用 manifest 协议无法表达该数据集，立刻停止，输出 NEEDS_FRAMEWORK_CHANGE，不要修改 MDL。

你必须阅读：
- <project-root>/docs/feature_pipeline_development.md
- <project-root>/docs/feature_pipeline_agent_playbook.md
- <project-root>/docs/feature_engineering_checklist.md

工作方式：
每次只执行一个阶段。每个阶段结束后，必须输出：
- 本阶段做了什么
- 产生了哪些文件
- 验收命令和结果
- 是否可以进入下一阶段
- 如果不能继续，明确写 NEEDS_USER_DECISION 或 NEEDS_FRAMEWORK_CHANGE

最终目标：
在 <feature-pipeline-root>/processed 下生成 manifest.json、train.csv、val.csv、test.csv，并且从 <project-root> 运行以下命令通过：

python scripts/preprocess.py --data-dir <feature-pipeline-root>/processed
python scripts/train.py --data-dir <feature-pipeline-root>/processed --epochs 1 --batch-size 32 --max-steps 2 --eval-max-batches 2
```

## 3. Agent 工作目录约定

推荐 feature pipeline 目录：

```text
<workspace>/
  MDL/
  MDL_feature_pipelines/
    <dataset-name>/
      README.md
      configs/
      reports/
      raw/              # optional symlink or local ignored directory
      processed/
      src/
      scripts/
      tests/
```

弱 agent 允许创建或修改：

```text
<feature-pipeline-root>/README.md
<feature-pipeline-root>/configs/*
<feature-pipeline-root>/reports/*
<feature-pipeline-root>/processed/*
<feature-pipeline-root>/src/*
<feature-pipeline-root>/scripts/*
<feature-pipeline-root>/tests/*
```

弱 agent 禁止创建或修改：

```text
<project-root>/*
<project-root>/src/*
<project-root>/scripts/*
<project-root>/configs/*
<project-root>/tests/*
```

验收边界命令：

```bash
cd <project-root>
git status --short
```

执行 feature pipeline 任务前后，`MDL` 仓库都不应该因为 feature pipeline 工作产生新的变更。

## 4. 阶段 0：边界确认

目标：确认路径和权限，建立 feature pipeline 工作区。

给 agent 的阶段 prompt：

```text
阶段 0：只做边界确认。

任务：
1. 打印当前目录。
2. 确认 <project-root> 存在并且包含 scripts/train.py。
3. 确认 <raw-data-path> 存在。
4. 创建 <feature-pipeline-root> 目录结构。
5. 在 <feature-pipeline-root>/reports/boundary_check.md 写入路径、时间、允许修改范围、禁止修改范围。
6. 不要读取大量数据，不要写训练代码。

验收：
- <feature-pipeline-root>/reports/boundary_check.md 存在。
- <project-root> 下 git status --short 没有新增变更。

如果任何路径不存在，停止并输出 NEEDS_USER_DECISION。
```

验收清单：

- `scripts/train.py` 存在。
- raw path 存在。
- feature pipeline root 存在。
- MDL 工作区无变化。

## 5. 阶段 1：原始数据盘点

目标：只了解数据文件类型、大小、表头、行数估计，不做转换。

给 agent 的阶段 prompt：

```text
阶段 1：原始数据盘点。

任务：
1. 列出 <raw-data-path> 下的数据文件，不要递归打印过多内容。
2. 记录每个主要文件的格式、大小、可能的用途。
3. 对 CSV/TSV/JSONL/Parquet 等表格数据，只读取表头和少量样本。
4. 不要在回答中泄露敏感原始值。样本值需要截断或只描述类型。
5. 输出 <feature-pipeline-root>/reports/data_inventory.md。

禁止：
- 不要修改 MDL。
- 不要复制 raw 大文件。
- 不要开始写 feature pipeline 逻辑。

验收：
- data_inventory.md 包含文件清单、格式判断、候选主表。
- 如果无法判断主表，输出 NEEDS_USER_DECISION。
```

建议记录字段：

```text
file path
format
size
estimated rows
columns/header
candidate role: interactions / users / items / features / labels / unknown
```

## 6. 阶段 2：数据画像

目标：生成字段级报告，帮助确定 label、group、scenario 和 features。

给 agent 的阶段 prompt：

```text
阶段 2：数据画像。

任务：
1. 对候选主表做字段级 profile。
2. 每列统计：dtype、非空率、唯一值数量估计、最小/最大值或长度范围、少量脱敏样例。
3. 找出候选 label 列。
4. 找出候选 group_id 列，例如 query、request、session、user-session。
5. 找出候选 scenario 列。如果没有，建议使用单场景 default。
6. 找出候选 sparse ID、dense numeric、sequence 特征。
7. 写入 <feature-pipeline-root>/reports/data_profile.md。

禁止：
- 不要写最终 feature pipeline。
- 不要修改 MDL。
- 不要打印大量原始样本。

验收：
- data_profile.md 有字段表。
- 明确列出候选 labels、group_id、scenario、features。
- 无法确定 label 或 group_id 时，停止并输出 NEEDS_USER_DECISION。
```

字段画像表推荐格式：

```text
| column | inferred_type | non_null_rate | unique_estimate | example_masked | candidate_use |
| --- | --- | --- | --- | --- | --- |
```

candidate_use 只能使用这些值：

```text
label
label_mask
group_id
scenario
sparse_id
dense_numeric
sequence
ignore
unknown
```

## 7. 阶段 3：适配方案设计

目标：先写设计，不写代码。弱 agent 必须让方案可检查。

给 agent 的阶段 prompt：

```text
阶段 3：适配方案设计。

任务：
基于 data_inventory.md 和 data_profile.md，写 <feature-pipeline-root>/reports/feature_pipeline_design.md。

feature_pipeline_design.md 必须包含：
1. 一行样本代表什么。
2. task_names 和每个 task 的 label 列。
3. 每个 task 的 label_mask 规则。
4. scenario_names 和 scenario_id 生成规则。如果无多场景，使用 ["default"]。
5. group_id 使用哪一列，以及为什么。
6. 是否需要 sample_weight；如果需要，说明权重列和生成规则。
7. train/val/test 切分策略。
8. sparse ID 特征列表及 vocab 规则。
9. dense numeric 特征列表及缺失值/归一化规则。
10. sequence 特征列表、delimiter、截断长度、padding 规则，以及使用 `sequence_mean_pooling`、`din`、`sim` 还是 `longer`。
11. feature token_specs 设计。
12. scenario_features 和 scenario_token_specs 设计；如果没有多场景，也必须说明 default scenario token 和 global scenario token。
13. task_features 和 task_token_specs 设计。
14. 不能确定的问题。

禁止：
- 不要写 feature pipeline 代码。
- 不要修改 MDL。

验收：
- feature_pipeline_design.md 完整。
- 没有 NEEDS_USER_DECISION 项时才能进入下一阶段。
```

弱 agent 的判断规则：

- 不知道 label，就停止。
- 不知道 group_id，就停止。
- 不知道是否能按时间切分，就使用稳定随机切分，但要写清楚。
- 没有 scenario，就使用单场景 `default`。
- ID 特征无法可靠转整数时，建立 vocab。
- 未见过的 val/test ID 映射到 `0`。
- `0` 永远保留给 unknown/padding。
- 普通历史序列优先使用 `sequence_mean_pooling`。如果一个历史行为 step 内有多个字段，例如 item、category、shop、price、sales、time_gap，应在 `sequence_mean_pooling.sequence_features` 中逐个声明，并使用 `fusion: concat` 先融合 step 内字段，再做 masked mean。若存在候选 target ID，例如 `item_id`，且需要 target-aware 兴趣建模，可以使用 `din.sequence_features`；`din` 默认使用标准 DIN 的 Dice activation 和非 softmax weighted sum。长历史序列需要先检索候选位置时，使用 `sim` 或 `longer`，并声明 `top_k`/`search_top_k`。
- 完整 MDL manifest 必须优先设计三类 tokens：`features/token_specs` 生成 feature tokens，`scenario_features/scenario_token_specs` 生成 scenario tokens，`task_features/task_token_specs` 生成 task tokens。不要只写 feature `token_specs` 后就结束。
- `scenario_token_specs` 数量必须等于 `len(scenario_names) + 1`，最后一个 token 是 global scenario token。`task_token_specs` 数量必须等于 `len(task_names)`。
- scenario/task token specs 默认使用 per-token FFN 投影；不要显式写 `projection: "linear"`，除非 feature_pipeline_design.md 明确这是消融实验。
- 一个样本如果属于多个场景，在 manifest 的 `data_columns` 中使用 `scenario_ids` 和可选 `scenario_ids_delimiter`，CSV 值例如 `0|2`。只有单场景时使用 `scenario_id`。
- 如果无法用 manifest 表达所需 scenario/task token 输入，停止并输出 NEEDS_FRAMEWORK_CHANGE；不要修改 MDL 源码。

## 8. 阶段 4：Feature Pipeline 脚手架

目标：创建目录和空实现，但不要做复杂逻辑。

给 agent 的阶段 prompt：

```text
阶段 4：创建 feature pipeline 脚手架。

只在 <feature-pipeline-root> 下创建：
- README.md
- configs/default.yaml
- src/__init__.py
- src/schema.py
- src/vocab.py
- src/split.py
- src/preprocess.py
- src/validate.py
- scripts/preprocess.py
- tests/test_preprocess.py

要求：
1. README 写清楚 raw path、processed path、运行命令。
2. configs/default.yaml 写数据路径和切分配置。
3. schema.py 只放数据集专属字段配置，不导入 MDL。
4. scripts/preprocess.py 是 CLI 入口。
5. 不要修改 MDL。

验收：
- 文件存在。
- 从 <feature-pipeline-root> 运行 python scripts/preprocess.py --help 成功。
- <project-root> git status --short 无变化。
```

推荐 CLI：

```bash
python scripts/preprocess.py \
  --raw-path raw/input \
  --output-dir processed \
  --config configs/default.yaml
```

## 9. 阶段 5：实现转换逻辑

目标：把 raw 数据转换成 manifest + CSV。

给 agent 的阶段 prompt：

```text
阶段 5：实现转换逻辑。

任务：
1. 在 <feature-pipeline-root>/src 中实现读取 raw 数据。
2. 按 feature_pipeline_design.md 生成 train.csv、val.csv、test.csv。
3. 构建 vocab，只用 train split fit。
4. val/test 未见 ID 映射为 0。
5. 生成 manifest.json。
6. 生成 vocab__<feature_name>.json。
7. 不要修改 MDL。

实现要求：
- 代码必须有 main conversion function。
- CSV header 必须和 manifest data_columns/source column 一致。
- label、label_mask 和可选 sample_weight 必须可转 float。
- scenario_id 必须是 int，从 0 开始。
- sequence 用明确 delimiter，例如 |。
- 输出路径可配置。

验收：
- 运行 feature pipeline CLI 成功。
- processed/manifest.json 存在。
- processed/train.csv 存在。
- 如果设计中有 val/test，则对应 CSV 存在。
```

最小输出结构：

```text
<feature-pipeline-root>/processed/
  manifest.json
  train.csv
  val.csv
  test.csv
```

## 10. 阶段 6：Feature Pipeline 自检

目标：在 feature pipeline 侧检查 manifest 和 CSV 是否一致。

给 agent 的阶段 prompt：

```text
阶段 6：feature pipeline 自检。

任务：
1. 实现或运行 <feature-pipeline-root>/src/validate.py。
2. 检查 manifest 中 splits 对应文件存在。
3. 检查 data_columns 中声明的列都在 CSV header 中。
4. 检查 features[*].source.column 都在 CSV header 中。
5. 检查 task_names 和 labels/label_masks key 完全一致。
6. 检查 scenario_id 范围合法。
7. 检查 embedding ID 范围在 [0, vocab_size)。
8. 检查 sequence ID 范围在 [0, vocab_size)。
9. 检查 label/mask 可转 float。
10. 输出 <feature-pipeline-root>/reports/validation_report.md。

禁止：
- 不要修改 MDL。

验收：
- validation_report.md 显示 PASS。
- 有任何 FAIL 就修 feature pipeline，不要改 MDL。
```

推荐 validation report：

```text
| check | result | detail |
| --- | --- | --- |
| split files exist | PASS | train,val,test |
| columns exist | PASS | ... |
```

## 11. 阶段 7：使用 MDL 通用入口验收

目标：从 `MDL` 仓库调用通用脚本验证 processed 数据。

给 agent 的阶段 prompt：

```text
阶段 7：MDL 通用入口验收。

任务：
从 <project-root> 运行：

python scripts/preprocess.py --data-dir <feature-pipeline-root>/processed
python scripts/train.py --data-dir <feature-pipeline-root>/processed --epochs 1 --batch-size 32 --max-steps 2 --eval-max-batches 2

要求：
1. 不要修改 MDL。
2. 如果 preprocess 失败，修 feature pipeline 输出。
3. 如果 train 因 manifest/CSV/data type 失败，修 feature pipeline 输出。
4. 如果错误明确指向 MDL 缺少通用能力，停止并输出 NEEDS_FRAMEWORK_CHANGE。
5. 把命令和结果写入 <feature-pipeline-root>/reports/mdl_smoke_report.md。

验收：
- preprocess 命令成功。
- train smoke 命令至少完成 2 step 或正常到达 max-steps。
- mdl_smoke_report.md 记录命令、结果和关键日志。
```

常见处理：

- `unknown feature encoder`: feature pipeline manifest 写错 encoder，先修 feature pipeline。
- `unknown token input feature`: token_specs 引用了不存在 feature，修 feature pipeline。
- `token_dim must be divisible`: 调整 smoke train 的 `--token-dim` 或减少 token 数；优先在报告里写明推荐训练参数。
- `cannot parse ...`: CSV 值和 dtype 不一致，修 feature pipeline。

## 12. 阶段 8：最终交付

目标：产出可交接的 feature pipeline。

给 agent 的阶段 prompt：

```text
阶段 8：最终交付。

任务：
1. 更新 <feature-pipeline-root>/README.md。
2. 写清楚 raw 数据准备方式。
3. 写清楚 feature pipeline 运行命令。
4. 写清楚 processed 输出结构。
5. 写清楚 task_names、scenario_names、group_id、特征列表。
6. 写清楚 smoke train 命令。
7. 写清楚已知限制。
8. 确认没有修改 MDL。

验收：
- README 完整。
- reports/feature_pipeline_design.md 存在。
- reports/validation_report.md 显示 PASS。
- reports/mdl_smoke_report.md 显示 PASS。
- <project-root> git status --short 无 feature pipeline 引起的变更。
```

最终回复模板：

```text
Feature pipeline completed.

Feature pipeline root:
<feature-pipeline-root>

Processed data:
<feature-pipeline-root>/processed

Tasks:
...

Scenarios:
...

Feature tokens:
...

Validation:
- feature pipeline validation: PASS
- MDL preprocess: PASS
- MDL smoke train: PASS

MDL changes:
None

Known limitations:
...
```

## 13. 停止条件

agent 遇到以下情况必须停止，不要猜：

`NEEDS_USER_DECISION`
: 无法确定 label、group_id、样本粒度、切分策略、任务定义、敏感字段处理方式。

`NEEDS_FRAMEWORK_CHANGE`
: 当前 MDL manifest 协议不能表达该数据，例如需要新的 encoder、新的 reader、新的 label 类型、新的 loss 或非 CSV 输入。

`NEEDS_DATA_ACCESS`
: raw 数据路径不存在、权限不足、文件损坏、依赖不可用。

`NEEDS_DEPENDENCY_APPROVAL`
: 需要安装 pandas、pyarrow 等额外依赖才能读取数据。

停止时必须输出：

```text
Status: NEEDS_...
Blocking issue:
Evidence:
Options:
Recommended next action:
Files changed:
```

## 14. 禁止行为清单

agent 明确禁止：

- 修改 `MDL/` 仓库任何文件。
- 在 `MDL/src` 添加数据集专属 reader。
- 在 `MDL/scripts` 添加数据集专属 preprocess 脚本。
- 在 `MDL/configs` 添加只服务单个私有数据集的配置。
- 把 raw 数据提交到 git。
- 把敏感样本原文大量写进报告。
- 为了让 smoke train 通过而伪造 label。
- 在 val/test 上 fit vocab 或 normalization。
- 遇到协议不支持时绕过 manifest 直接改训练代码。
- 不写报告直接交付代码。

## 15. 人类验收清单

人类 reviewer 最少检查：

- `MDL` 仓库没有被 feature pipeline 修改。
- feature pipeline README 可以从零复现 processed 数据。
- `manifest.json` 中 `task_names`、`labels`、`label_masks` 一致。
- `scenario_id` 范围匹配 `scenario_names`。
- embedding/sequence ID 范围匹配 `vocab_size`。
- `0` 被保留给 unknown/padding。
- val/test 没有 fit 新 vocab。
- group_id 对 QAUC 有意义。
- smoke train 命令通过。
- reports 中没有泄露敏感原始数据。

## 16. 最小命令序列

当 feature pipeline 已经实现后，agent 最后只能按下面顺序验收：

```bash
cd <feature-pipeline-root>
python scripts/preprocess.py --raw-path <raw-data-path> --output-dir processed
python -m pytest tests

cd <project-root>
python scripts/preprocess.py --data-dir <feature-pipeline-root>/processed
python scripts/train.py --data-dir <feature-pipeline-root>/processed --epochs 1 --batch-size 32 --max-steps 2 --eval-max-batches 2
git status --short
```

最后一条 `git status --short` 不应该出现 feature pipeline 造成的 `MDL` 修改。

