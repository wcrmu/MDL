# MDL 推荐系统项目

本仓库保留工业级 MDL 推荐系统核心实现，面向安全环境中的 parquet-native 训练和推理。默认数据契约是 `flat_parquet`：一行是一条训练样本。非 flat 的业务 Parquet 布局通过外部 `adapter_parquet` 预处理 adapter 接入，核心仓库不硬编码具体展开规则；具体字段、词表策略、bucket/hash 策略和训练参数都通过 YAML 配置。

## 项目结构

```text
.
├── configs/
│   └── mdl.yaml                 # 数据、词表、模型和训练模板
├── PAPER_ALIGNMENT.md           # MDL / OneTrans 论文对齐检查说明
├── paper/                       # 本地论文源码
├── mdl.py                       # 唯一对外 CLI
├── src/
│   ├── config.py                # YAML 契约：logical features、vocab_strategy、model、training
│   ├── features.py              # categorical 编码：词表 fit/load、策略指纹、hash bucket
│   ├── dataloader.py            # Parquet 读取、adapter、运行时列编码、FeatureBatch
│   ├── model.py                 # 模型实现
│   ├── train.py                 # 训练与预测
│   ├── benchmark.py             # 读取性能基准
│   └── modules/                 # 复用 attention、MLP 等
```

## 常用命令

安装依赖：

```bash
pip install -r requirements.txt
```

校验工业配置：

```bash
python mdl.py validate-config --config configs/mdl.yaml
```

检查 parquet schema、所需列和样例 batch：

```bash
python mdl.py profile \
  --config configs/mdl.yaml \
  --split train \
  --max-batches 10
```

做读取性能基准：

```bash
python mdl.py benchmark \
  --config configs/mdl.yaml \
  --split train \
  --max-batches 10
```

构建词表：

```bash
python mdl.py fit-vocab --config configs/mdl.yaml
```

短训练 smoke test：

```bash
python mdl.py train \
  --config configs/mdl.yaml \
  --max-steps 10
```

模型通过 `model.name` 切换：

- `rankmixer`：纯 RankMixer-style feature-token baseline，不使用 MDL 的场景/任务/domain 模块。
- `mdl_rankmixer`：MDL 场景/任务 token + RankMixer-style TokenMixing backbone。
- `onetrans`：论文版 OneTrans，包含 S/NS tokenizer、mixed causal attention、mixed FFN 和 pyramid stack。
- `mdl_onetrans`：OneTrans 产生 feature tokens，MDL 场景/任务 token 和 domain-aware attention 负责多场景/多任务输出。

默认 `configs/mdl.yaml` 面向大字段安全环境：`rankmixer` 和 `mdl_rankmixer` 使用 `tokenization.feature_tokenizer: rankmixer`，默认 feature tokens 是 `32 * 768`；`rankmixer` 不生成 scenario/task tokens，`mdl_rankmixer` 的 scenario/task tokens 不计入 feature-token 数。按 `feature_token_inputs` 的 YAML 顺序拼接所有 `embedding_scope: feature/shared` 的输入，并要求拼接后的维度严格等于 `num_feature_tokens * token_dim` 后直接 reshape；不再隐式 zero-pad。模板中 `user_id/item_id/shop_id` 提供 `3 * 32` 维，`rankmixer_context_dense` 提供 `672` 维，`hist` 的 LONGER multi-slice summary 提供 `31 * 768` 维，总计 `24576 = 32 * 768`。后续适配私有 schema 时，优先改 `features`、`sequences`、`vocab_strategy`、`scenario_tokens` 和 `task_tokens`；只有需要排除某些字段或控制切片顺序时才声明 `tokenization.feature_token_inputs`。`auto_split` 仍保留为显式 fallback，但不是默认 RankMixer 对齐路径。

多字段行为序列用 `sequences` 声明，每个序列 step 内可以包含多个 categorical/dense 字段。序列的 parquet 物理布局由数据侧处理，训练代码只消费 `fields[].source` 指向的字段级 list columns：

`FeatureConfig.kind: sequence` 只保留为 legacy 单列序列兼容入口，新配置应使用顶层 `sequences`。`encoder: longer` 在 RankMixer 路径中支持 target/global tokens、recent query 压缩、Token Merge、InnerTrans、time-delta side projection、cross-causal attention、后续 self-causal attention，以及 `rankmixer_summary_tokens` 多 slice 序列摘要。

```yaml
sequences:
  - name: hist
    max_length: 100
    truncation: tail
    encoder: longer
    target_inputs: [item_id]
    longer_query_tokens: 32
    longer_self_layers: 1
    fields:
      - name: item_id
        kind: categorical
        source: hist_item_id
      - name: action
        kind: categorical
        source: hist_action
      - name: age
        kind: dense
        source: hist_age
```

### Parquet 预处理 adapter

`flat_parquet` 是默认 identity 语义：原始 Parquet 已经满足一行一样本，训练、词表、profile 和 benchmark 都直接消费这些 flat 列。

当原始 Parquet 是一行多请求、一行一个请求但内含多个 item，或其他环境专属布局时，配置 `format: adapter_parquet` 并提供外部 callable：

```yaml
data:
  train:
    format: adapter_parquet
    inputs:
      - /secure/train/day=*/part-*.parquet
    adapter:
      callable: secure_pkg.mdl_adapters:flatten_requests
      input_columns: [request_id, user_id, candidates]
      options:
        candidate_field: candidates
    request_id: request_id
    group_id: request_id
    labels:
      click: click
```

Adapter 签名：

```python
def flatten_requests(table, *, context):
    ...
    return flat_table_or_iterable
```

`context` 包含 `split_name`、`required_columns` 和 YAML 中的 `adapter.options`。adapter 输入只假定是一个 raw `pyarrow.Table`；输出必须满足 flat 契约：

- 一行 = 一条训练样本。
- 标量列用于普通 features、labels、request/group id；已配置 `dimension > 1` 的 dense feature 可以用 fixed-width list 单元表示。
- list 列用于 `sequences[].fields[].source`，同一 sequence 内同行长度必须对齐。
- 如果使用 list-valued scenario mask，列名必须是 `scenarios.source`。
- 输出列名必须与 YAML 中的 `source`、labels、masks、request/group/scenario 配置一致；需要重命名时由 adapter 完成。

仓库提供 `examples.parquet_identity_adapter:adapt` 作为最小接入示例。它只返回输入 table，可用于验证 `adapter_parquet` 配置链路；真实业务展开逻辑应放在安全环境自己的包中。

单机多卡 DDP 训练可以由 CLI 自动启动：

```bash
python mdl.py train \
  --config configs/mdl.yaml \
  --distributed ddp \
  --nproc-per-node 4 \
  --max-steps 100
```

也可以直接使用生产环境常见的 `torchrun`：

```bash
torchrun --nproc_per_node=4 mdl.py train \
  --config configs/mdl.yaml \
  --distributed ddp \
  --max-steps 100
```

DDP 下 `reader.shard_unit: file` 会按文件分片；`row_group` 和 `record_batch` 会按扫描 batch 分片。`training.batch_size` 按每个进程解释，checkpoint 只由 rank 0 写入。

默认 `training.sparse_update_mode: ddp_synced_adagrad` 使用稀疏 embedding 梯度和 Adagrad，但 DDP 仍会同步梯度；这不是论文中“sparse 异步、dense 同步”的数百 GPU 参数服务器训练。安全环境若要对齐论文训练系统，需要配置：

```yaml
training:
  sparse_update_mode: external_parameter_server
  sparse_parameter_server_adapter: secure_pkg.mdl_ps_train:train
```

adapter 必须接管完整训练流程，并返回 `{"steps": int, "last_loss": float}` 或 `TrainResult`。

预测：

```bash
python mdl.py predict \
  --config configs/mdl.yaml \
  --checkpoint-path artifacts/checkpoints/mdl_rankmixer.pt \
  --output-path artifacts/runs/predictions.parquet
```

论文对齐检查：

```bash
python mdl.py check-paper-alignment
```

基础校验：

```bash
python mdl.py validate-config --config configs/mdl.yaml
python mdl.py check-paper-alignment
```

## 数据与安全

不要提交 raw parquet、词表产物、checkpoint、预测结果或安全环境路径。`configs/mdl.yaml` 中的路径是模板占位；进入安全环境后只需要替换 `data.*.inputs`、`features`、`vocab_strategy`、`tokenization` 和训练参数；训练输出默认写入 ignored 的 `artifacts/`。
