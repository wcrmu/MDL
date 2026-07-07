# MDL 推荐系统项目

本仓库保留工业级 MDL 推荐系统核心实现，面向安全环境中的 parquet-native 训练和推理。训练数据支持多个同 schema 的 agg parquet 文件，测试/推理数据支持多个同 schema 的 flat parquet 文件；具体字段、agg 解压方式、词表策略、bucket/hash 策略和训练参数都通过 YAML 配置。

## 项目结构

```text
.
├── configs/
│   └── mdl.yaml                 # 数据、词表、模型和训练模板
├── PAPER_ALIGNMENT.md           # MDL / OneTrans 论文对齐检查说明
├── paper/                       # 本地论文源码
├── mdl.py                       # 唯一对外 CLI
├── src/
│   ├── config.py                # YAML 配置 dataclass 和校验
│   ├── data.py                  # parquet discovery、schema 校验和 agg 解压
│   ├── features.py              # 词表策略和特征元信息
│   ├── vocab.py                 # vocab fit/load
│   ├── tensorize.py             # Arrow table 到 torch batch
│   ├── model.py                 # paper-aligned MDL 模型
│   ├── train.py                 # 训练和预测流程
│   ├── benchmark.py             # parquet 读取性能检查
│   └── modules/                 # 复用的 attention、MLP、loss、metrics 模块
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

- `mdl_rankmixer`：MDL 场景/任务 token + RankMixer-style TokenMixing backbone。
- `onetrans`：论文版 OneTrans，包含 S/NS tokenizer、mixed causal attention、mixed FFN 和 pyramid stack。
- `mdl_onetrans`：OneTrans 产生 feature tokens，MDL 场景/任务 token 和 domain-aware attention 负责多场景/多任务输出。

默认 `configs/mdl.yaml` 面向大字段安全环境：`mdl_rankmixer` 使用 `tokenization.feature_tokenizer: rankmixer`，默认 feature tokens 是 `32 * 768`，不包括另外生成的 scenario/task tokens。按 `feature_token_inputs` 的 YAML 顺序拼接所有 `embedding_scope: feature/shared` 的输入，并要求拼接后的维度严格等于 `num_feature_tokens * token_dim` 后直接 reshape；不再隐式 zero-pad。模板中 `user_id/item_id/shop_id` 提供 `3 * 32` 维，`rankmixer_context_dense` 提供 `672` 维，`hist` 的 LONGER multi-slice summary 提供 `31 * 768` 维，总计 `24576 = 32 * 768`。后续适配私有 schema 时，优先改 `features`、`sequences`、`vocab_strategy`、`scenario_tokens` 和 `task_tokens`；只有需要排除某些字段或控制切片顺序时才声明 `tokenization.feature_token_inputs`。`auto_split` 仍保留为显式 fallback，但不是默认 RankMixer 对齐路径。

多字段行为序列用 `sequences` 声明，每个序列 step 内可以包含多个 categorical/dense 字段：

`FeatureConfig.kind: sequence` 只保留为 legacy 单列序列兼容入口，新配置应使用顶层 `sequences`。`encoder: longer` 在 RankMixer 路径中支持 target/global tokens、recent query 压缩、Token Merge、InnerTrans、time-delta side projection、cross-causal attention、后续 self-causal attention，以及 `rankmixer_summary_tokens` 多 slice 序列摘要。

```yaml
sequences:
  - name: hist
    layout: parallel_lists
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

DDP 下 `reader.shard_unit: file` 会按文件分片；`row_group` 和 `record_batch` 会按扫描 batch 分片。`training.batch_size` 和 `reader.batch_size_candidates` 按每个进程解释，checkpoint 只由 rank 0 写入。

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
