# MDL 推荐系统项目

本仓库提供 RankMixer、MDL、OneTrans 和 LONGER 的独立复现实现，面向安全环境中的
parquet-native 训练和推理。它不是论文作者的工业源码；`configs/*_paper.yaml` 固化论文
公开的方法与超参数，原有 `configs/*.yaml` 保持小规模架构与集成 smoke profile。两类配置
都不包含论文私有数据、数百个生产特征、集群拓扑或结果表复现条件。
默认数据契约是 `flat_parquet`：一行是一条训练样本。非 flat 业务布局通过外部
`adapter_parquet` 接入。

## 项目结构

```text
.
├── configs/
│   ├── default.yaml             # 小规模 mdl_rankmixer smoke 模板
│   ├── *_paper.yaml             # 论文公开方法/超参数 profile
│   └── *.yaml                   # 各模型小规模 smoke overlay
├── src/
│   ├── main.py                  # 应用入口：参数解析、子命令分发、DDP 启动
│   ├── config.py                # YAML 契约：logical features、vocab_strategy、model、training
│   ├── features.py              # categorical 编码：词表 fit/load、策略指纹、hash bucket
│   ├── dataloader.py            # Parquet 读取、adapter、运行时列编码、FeatureBatch
│   ├── model.py                 # 模型实现
│   ├── train.py                 # 训练、预测、AUC/QAUC/UAUC 评估
│   └── modules/                 # 复用 attention、MLP 等
└── tests/                       # 数学结构、缓存等价性和指标回归测试
```

## 常用命令

要求 Python 3.11+。GPU 环境应先安装与 CUDA 匹配且经过批准的 PyTorch：

```bash
python -m pip install -r requirements.txt
```

校验工业配置：

```bash
python src/main.py validate-config --config configs/default.yaml
```

检查 parquet schema、所需列和样例 batch：

```bash
python src/main.py profile \
  --config configs/default.yaml \
  --split train \
  --max-batches 10
```

构建词表：

```bash
python src/main.py fit-vocab --config configs/default.yaml
```

短训练 smoke test：

```bash
python src/main.py train \
  --config configs/default.yaml \
  --max-steps 10
```

运行回归测试：

```bash
python -m unittest discover -s tests -v
```

模型使用独立配置入口；overlay 通过 `extends: default.yaml` 复用公共契约，映射递归
合并、列表整体替换，并检测循环继承：

- `configs/rankmixer.yaml`：双残差/双 LayerNorm Block、语义分组投影、mean pooling，
  可选 ReLU Routing + adaptive L1 + DTSI Sparse-MoE。
- `configs/mdl_rankmixer.yaml`：feature/scenario/task token 逐层共同演化，重要 ID
  使用独立 scenario/task embedding table。
- `configs/onetrans.yaml`：S/NS mixed parameter、causal pyramid、MLP tokenizer 和逐层 K/V cache。
- `configs/longer.yaml`：端到端 LONGER，TokenMerge 保留 `Kd` 宽度和完整压缩序列。
- `configs/mdl_onetrans.yaml`：实验性逐层组合，不是论文已定义模型，必须设置
  `experimental_model_acknowledged: true`。
- `configs/rankmixer_paper.yaml`：RankMixer 100M dense 方法配置，`T=16, D=768, L=2`
  且 dense LR 为 `0.01`。
- `configs/mdl_rankmixer_paper.yaml`：MDL 论文公式路径和最小 3-scenario x 3-task
  对齐面；论文未公开的 L/D/H 显式标为 implementation choice。
- `configs/onetrans_paper.yaml`：OneTrans-S，`L=6, d=256, H=4`、
  `1190 -> 12` timestamp-aware pyramid。
- `configs/longer_paper.yaml`：LONGER `L=2000, d=32, K=8, recent-k=100` 主配置。

默认配置使用 4 个显式语义 token、32 维宽度和 2 层，仅用于快速验证，不冒充论文工业
配置。MDL feature self-interaction 默认严格执行论文 Eq. (6)；需要复现旧实现时必须显式
设置 `model.mdl_feature_interaction: rankmixer_full`。MDL 论文消融要求替换 tower 或
RankMixer interaction；原文未给出替换模块的
完整结构，因此仓库拒绝旧的清零式开关，而不是输出不可比较的 Table 2 数字。

多字段行为序列用 `sequences` 声明，每个序列 step 内可以包含多个 categorical/dense 字段。序列的 parquet 物理布局由数据侧处理，训练代码只消费 `fields[].source` 指向的字段级 list columns：

`encoder: longer` 要求固定 `max_length` 和标量 dense `time_delta_field`，支持
cacheable user/CLS globals、candidate globals、recent query、TokenMerge、InnerTrans、
hybrid attention 和候选复用缓存。LONGER 输入按“position 加到 item/side embedding，
再拼 time-delta，最后过 MLP”的论文顺序生成。OneTrans timestamp-aware 模式另外要求
`timestamp_field`；没有跨行为
时间戳时应明确使用 `intent_ordered`。

`sequence_order` 声明 parquet list 中有效事件的物理方向。模型在 causal attention 前统一
转成 `oldest_to_newest`；`truncation: head/tail` 仍针对物理 list，因此 newest-first 数据要
保留最近事件时应使用 `head`，oldest-first 数据则使用 `tail`。

```yaml
sequences:
  - name: hist
    max_length: 100
    truncation: tail
    sequence_order: oldest_to_newest
    encoder: longer
    time_delta_field: time_delta
    target_inputs: [item_id]
    longer_user_global_inputs: [user_id]
    longer_user_global_tokens: 1
    longer_cls_tokens: 1
    longer_candidate_global_tokens: 1
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
      - name: time_delta
        kind: dense
        source: hist_time_delta
```

### Parquet 预处理 adapter

`flat_parquet` 是默认 identity 语义：原始 Parquet 已经满足一行一样本，训练、词表和 profile 都直接消费这些 flat 列。

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
python src/main.py train \
  --config configs/default.yaml \
  --distributed ddp \
  --nproc-per-node 4 \
  --max-steps 100
```

也可以直接使用生产环境常见的 `torchrun`：

```bash
torchrun --nproc_per_node=4 src/main.py train \
  --config configs/default.yaml \
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
python src/main.py predict \
  --config configs/default.yaml \
  --checkpoint-path artifacts/checkpoints/mdl_rankmixer.pt \
  --output-path artifacts/runs/predictions.parquet
```

评估 split 必须按训练任务的同一顺序配置 labels，并配置 `group_id`。将其指向
query/request key 时报告 QAUC，指向 user key 时报告 UAUC：

```bash
python src/main.py evaluate \
  --config configs/mdl_rankmixer.yaml \
  --checkpoint-path artifacts/checkpoints/mdl_rankmixer.pt \
  --split test \
  --group-metric-name qauc
```

评估使用带 tie 平均秩的精确 AUC；QAUC/UAUC 是有效 group AUC 的非加权平均。

## 缓存与实现边界

- OneTrans cache 保存每层 S-side K/V、mask 和 pyramid output；LONGER cache 保存
  user/CLS + sequence K/V 和 sequence-side 压缩状态。candidate globals 不进入可复用状态。
- OneTrans 提供 `update_request_cache(features, previous_cache)` 做跨请求 append-only 更新。
  非追加修改会被拒绝；无 pyramid 时每层只投影新增 K/V。pyramid 尾窗移动导致较深层
  hidden state 改变时，该层会重建 K/V 以保持与完整重算数值等价，首层仍复用旧 K/V。
- RankMixer DTSI 论文没有公开双 router 的训练输出融合公式。启用 sparse DTSI 时必须
  显式设置 `sparse_moe_dtsi_training_output`；仓库不再静默采用 `0.5` 平均。
- `external_parameter_server` 是安全环境集成边界，仓库不实现工业异步参数服务。
- Sparse-MoE 不包含论文生产环境的自定义 sparse-GEMM kernel。
- next-batch evaluation、私有数据和完整特征定义不可获得，不能宣称复现论文指标表。

## 数据与安全

不要提交 raw parquet、词表产物、checkpoint、预测结果或安全环境路径。`configs/default.yaml` 中的路径是模板占位；进入安全环境后只需要替换 `data.*.inputs`、`features`、`vocab_strategy`、`tokenization` 和训练参数；训练输出默认写入 ignored 的 `artifacts/`。
