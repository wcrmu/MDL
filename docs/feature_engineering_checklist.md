# Feature Engineering Checklist

本文档给下游 agent 使用，用于在实现 dataset-specific feature pipeline 前、中、后检查特征工程设计。目标不是追求特征越多越好，而是确保任务、场景、特征、序列和 MDL token 语义都能被稳定地写进 processed dataset 和 manifest。

下游 agent 在设计 feature pipeline 时必须先阅读：

- [feature_pipeline_development.md](feature_pipeline_development.md)
- [feature_pipeline_agent_playbook.md](feature_pipeline_agent_playbook.md)
- 本文档

## 1. 任务定义先于特征

必须先确定以下内容，再开始做特征：

- `task_names` 是什么，每个 task 的 label 来自哪一列。
- 每个 task 的 `label_mask` 规则是什么。
- 一行样本代表什么，例如曝光、候选 item、用户行为、session step。
- `group_id` 使用哪一列，以及为什么它适合 QAUC 分组。
- `scenario_names` 有哪些；如果没有多场景，也必须使用 `["default"]`。

检查点：

- 不知道 label 时停止，输出 `NEEDS_USER_DECISION`。
- 不知道 `group_id` 时停止，输出 `NEEDS_USER_DECISION`。
- 多任务中缺失 label 的样本必须用 `label_mask=0`，不要硬填负样本。

## 2. 数据泄漏检查

以下内容只能用 train split fit：

- vocab。
- dense 特征归一化参数。
- clip 阈值。
- frequency filter。
- ID 映射表。

禁止使用以下特征：

- 曝光之后、点击之后、转化之后才知道的字段。
- val/test 的统计信息。
- 未来行为序列。
- 全量数据 fit 出来的 target encoding 或 CTR prior，除非明确只用历史窗口计算。

检查点：

- val/test 未见 ID 必须映射到 `0`。
- 时间切分数据中，历史序列只能包含样本时间点之前的行为。
- 如果无法确认某字段是否泄漏，停止并输出 `NEEDS_USER_DECISION`。

## 3. ID 特征规范

ID 类特征必须满足：

- 转成整数 ID。
- `0` 永远保留给 unknown/padding。
- embedding 输入值在 `[0, vocab_size)`。
- train vocab fit 后，val/test 只 lookup，不重新 fit。

高基数特征必须明确策略：

- 建 vocab。
- hash。
- 过滤低频。
- 映射到 unknown。

检查点：

- 每个 `embedding` feature 必须声明 `vocab_size` 或 `cardinality`。
- feature pipeline 测试要覆盖未见 ID 映射到 `0`。
- 不允许把原始字符串 ID 直接写入 CSV 给 `embedding` encoder。

## 4. Dense 特征规范

连续特征必须明确：

- 缺失值如何处理。
- 是否需要 clip。
- 是否需要 log transform。
- 是否需要标准化或归一化。
- `identity` encoder 的 `dim`。

检查点：

- 所有 dense feature 都能转成 float。
- 极端长尾数值不能不处理就直接输入。
- `identity.dim` 必须和实际输出维度一致。

## 5. 序列特征规范

序列特征必须明确：

- 序列排序方向。
- 最大长度。
- 截断规则。
- padding 值。
- delimiter，例如 `|`。
- 每个序列元素是否来自样本时间点之前。

普通历史序列：

- 优先使用 `sequence_mean_pooling`。

Target-aware 兴趣建模：

- 使用 `din`。
- `target_feature` 必须指向当前 batch 中的候选 target 字段。
- 历史 ID 和 target ID 必须共用同一个 vocab 体系。

长历史序列：

- 使用 `sim` 或 `longer`。
- 必须声明 `top_k` 或 `search_top_k`，或接受默认 `50`。
- 它们先用 target 相似度检索 top-k，再做 DIN-style attention。

多字段行为序列：

- 例如 item/category/shop/price/time_gap。
- 必须在 `sequence_features` 中逐字段声明。
- 每个字段必须逐 step 对齐。
- 当前 `fusion` 使用 `concat`。

检查点：

- 序列 payload 会被 collate 成 `values` 和 `lengths`。
- 所有 sequence 字段同一行的长度必须语义对齐。
- 空序列必须能被 encoder 处理，不能让 feature pipeline 崩溃。

## 6. MDL Token 设计

manifest 不能只声明 feature tokens。完整 MDL feature pipeline 必须声明三类 tokens：

- `features` + `token_specs` 生成 feature tokens `T_f`。
- `scenario_features` + `scenario_token_specs` 生成 scenario tokens `T_s`。
- `task_features` + `task_token_specs` 生成 task tokens `T_t`。

硬性要求：

- `scenario_token_specs` 数量必须等于 `len(scenario_names) + 1`。
- 最后一个 scenario token 是 global scenario token。
- `task_token_specs` 数量必须等于 `len(task_names)`。
- 这四个字段缺一不可：`scenario_features`、`scenario_token_specs`、`task_features`、`task_token_specs`。

检查点：

- 不允许依赖 fallback；缺字段应该让校验或模型构建直接报错。
- scenario/task token specs 默认使用 per-token FFN-ReLU 投影。
- 不要显式写 `projection: "linear"`，除非 feature pipeline design 明确这是消融实验。
- `scenario_token_specs[*].inputs` 只能引用 `scenario_features`。
- `task_token_specs[*].inputs` 只能引用 `task_features`。

## 7. Scenario 设计

单场景：

- `scenario_names` 写 `["default"]`。
- CSV 中 `scenario_id` 全部写 `0`。
- `scenario_token_specs` 仍然需要 2 个 token：default token + global token。

多场景：

- `scenario_id` 必须是从 `0` 开始的整数 ID。
- ID 顺序必须和 `scenario_names` 下标一致。

多场景重叠样本：

- 使用 `data_columns.scenario_ids`。
- 使用 `data_columns.scenario_ids_delimiter`，默认可以是 `|`。
- CSV 示例：`0|2`。

检查点：

- `scenario_id` 或 `scenario_ids` 不能越界。
- 不能同时声明 `scenario_id` 和 `scenario_ids`。
- 如果场景定义不清楚，先使用 `default`，不要虚构业务场景。

## 8. 样本和训练权重

如果需要样本级权重：

- 在 manifest 中声明 `data_columns.sample_weight`。
- CSV 对应列必须能转成 float。
- loss 会按有效权重和归一化。

Task/scenario 权重：

- 训练时通过 CLI 传入 `--task-weights` 和 `--scenario-weights`。
- 不建议硬编码到 CSV 中。

检查点：

- `task_weights` 数量必须等于 `len(task_names)`。
- `scenario_weights` 数量必须等于 `len(scenario_names)`。
- 权重必须非负。

## 9. Token 数量和维度

使用 RankMixer backbone 时：

- `token_dim % num_feature_tokens == 0`。
- 如果修改 `token_specs` 数量，要同步检查 `--token-dim`。

设计建议：

- 不要把 token 拆得过细，避免维度约束和计算量膨胀。
- 不要把完全不同语义的特征全塞进一个 token。
- 常见分组可以按 user、item、context、history、cross/prior 组织。

检查点：

- 每个 token 的 `inputs` 都存在。
- 每个 token 的语义可解释。
- token 数量变化后必须跑 smoke train。

## 10. Manifest 和 CSV 校验

feature pipeline 产物必须通过：

```bash
python scripts/preprocess.py --data-dir data/processed/<dataset_name>
```

快速迭代时可以先扫部分行：

```bash
python scripts/preprocess.py --data-dir data/processed/<dataset_name> --max-rows 1000
```

校验覆盖：

- manifest 必需字段。
- domain tokenization 必需字段。
- token specs 引用。
- scenario/task token 数量。
- CSV header。
- `scenario_id` 范围。
- label、mask、sample_weight float 解析。
- feature source 值解析。

## 11. Smoke Train 验收

feature pipeline 完成后，必须从 MDL 仓库根目录跑：

```bash
python scripts/train.py \
  --data-dir data/processed/<dataset_name> \
  --epochs 1 \
  --batch-size 32 \
  --max-steps 2 \
  --eval-max-batches 2
```

如果使用 Sparse-MoE，也至少跑一次：

```bash
python scripts/train.py \
  --data-dir data/processed/<dataset_name> \
  --epochs 1 \
  --batch-size 32 \
  --max-steps 2 \
  --ffn-type sparse_moe \
  --sparse-moe-num-experts 4 \
  --sparse-moe-loss-weight 1e-4
```

验收标准：

- preprocess 校验通过。
- smoke train 能跑到 `max_steps`。
- 如果有 val split，能输出 task metrics 和 scenario-task metrics。
- 没有 shape、dtype、unknown encoder、token count 错误。

## 12. 常见停止条件

遇到以下情况不要绕开协议，必须停止：

- label 定义不清楚。
- group_id 定义不清楚。
- 字段是否泄漏无法判断。
- scenario/task token 输入无法用 manifest 表达。
- 需要新的 encoder、reader 或输入 shape。
- 原始数据缺少实现任务所需的核心字段。

输出格式：

```text
NEEDS_USER_DECISION: <具体缺什么决策>
```

或：

```text
NEEDS_FRAMEWORK_CHANGE: <当前 manifest/encoder/reader 无法表达什么>
```
