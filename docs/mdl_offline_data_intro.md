# MDL 离线数据简介

> 简单介绍当前训练使用的离线数据：特征规模、来源、目标与场景，以及训练/测试如何选取。  
> 更细的字段合同见 [`DATA_FORMAT.md`](../DATA_FORMAT.md)、[`current_field_processing_report.md`](./current_field_processing_report.md)；实现总览见 [`mdl_data_adaptation_overview.md`](./mdl_data_adaptation_overview.md)。

## 1. 数据来源

离线训练读的是搜推侧 **小时级特征聚合 Parquet**，不是原始日志。

| 项 | 说明 |
|---|---|
| 存储 | HDFS / viewfs |
| 基路径 | `hdfs://temu-data-ns/apps/nothive/warehouse/searchrec/searchrec_cvr_allscene_agg_fgoutput_hour_dracarys_exp` |
| 分区 | `pt=YYYY-MM-DD/hr=HH/`（Hive 风格小时目录） |
| 文件 | 约 **500 个/小时**，后缀 `.gz.parquet`（Parquet 内 GZIP 列压缩） |
| 行语义 | 以 **request × candidate** 为主的聚合行（`agg` 布局）；一行里可含多个请求及其候选 |
| 规模量级 | 约 **1.7k agg 行/文件**，约 **86 万 agg 行/小时**（观测值，随分区波动） |

训练时用 `--data-base-dir` + `--train-start-hour` / `--train-end-hour` 展开成小时目录列表；代码递归发现 `*.parquet`，按列裁剪后喂给 adapter。

---

## 2. 特征数量（怎么数）

上游很宽，模型只扫其中一部分。常用口径如下（以当前生产配置为准）：

| 口径 | 数量 | 含义 |
|---|---:|---|
| 上游物理列 | **630** | Parquet schema 全宽；多数不进本模型 |
| Adapter 必扫 raw 列 | **约 260～280** | 训练必须投影的物理列（随可选列略变） |
| 主非序列逻辑字段 | **147** | **47 request + 100 candidate**（配置生成器 `EXPECTED_FEATURE_COUNT`） |
| 主 UPS 历史 | **9** | `impr` / `clk_long` / `view_long` / `cart_long` / `buy_long` / `semi_clk` / `srch_q2i` / `ups_clk_sku` / `flatten_query_hash` |
| UPS 原始属性 | **107** | 9 个绝对时间戳 + 98 个预编码类别属性 |
| 真正连续输入 | **9** | 每条历史由 `impr_time - event_time` 派生的 `time_delta_log1p_seconds` |
| 其余主特征 | 几乎全是 **categorical / pre_hashed** | 名称像价格、CTR、计数，实际仍走 embedding |

补充：

- MDL 还会为 Scenario/Task 增加 **scope 独立** 的 important / prior 逻辑字段（与主 147 共用物理 source，但常不共享参数）。
- RankMixer 侧最终压成 **32×768** Feature token；OneTrans 侧是 **变长 S + 32 NS**——那是模型消费方式，不是上游又多了一套特征表。

---

## 3. 目标（Tasks）与场景（Scenarios）

### 3.1 三个预测目标

每个 **candidate** 同时监督 3 个二分类标签（`{0,1}`）：

| 任务名 | 标签列 | 业务含义（简） |
|---|---|---|
| `fst_cart` | `label_fst_cart` | 首加购相关 |
| `upid_pay` | `upid_fst_trgt_noc_clk_pay_24h` | 支付 / 转化相关（24h 口径） |
| `cateid_filter` | `cateid_is_fst_scene_sp_filter` | 类目/场景过滤相关；**是预测目标，不是行过滤器** |

默认 loss：`mean_per_task`；任务权重约为 `fst_cart=0.5`，`upid_pay=cateid_filter=0.01`。

### 3.2 场景怎么定义

上游有大量细粒度 `scene_id`。当前 **coarse MDL** 不直接用几百个 scene 当 Domain token，而是映射成：

| Scenario | 规则 |
|---|---|
| `search` | `scene_id` 落在生产 **121** 个 search allowlist 内 |
| `recommendation` | 其余非负 `scene_id` |
| `global` | 额外的跨场景 common token（不是又一个业务流量场景） |

Adapter 生成：

- `coarse_scene_index`：路由用（0=search，1=recommendation）
- `coarse_scene_prior_id`：scenario prior embedding 用的小 identity

fine 配置可自动发现更多 scenario，但计算成本随场景数上升；日常串讲/长跑默认讲 coarse。

---

## 4. 训练数据如何选取

### 4.1 时间窗

通过 CLI 指定**左闭右开**小时窗：

```bash
--train-start-hour 2026-07-21-00 \
--train-end-hour   2026-07-22-01
```

会展开为：

```text
{base}/pt=2026-07-21/hr=00
...
{base}/pt=2026-07-22/hr=00
```

即包含 start，不包含 end。  
规划容量时常按 **约 500 files/hour × 训练小时数** 估算（例如约 24h → 约 12,000 文件），词表/桶大小也按该尺度外推，而不是只看几十个抽样文件。

### 4.2 样本与切分

- 单位：展开后的 **candidate 级样本**（同一 `search_id` 下多个候选共享 request/历史）。
- 分片：`shard_unit=file`，多卡按文件划分，避免同一请求被拆到不同 rank 重复解历史。
- 读取：列裁剪 + `agg_direct` 直接构造 `FeatureBatch`；失败文件可按 `on_hdfs_failure=skip` 跳过（需盯 skip 预算）。

YAML 里 `data.train.inputs` 默认为空，**以启动参数展开的小时目录为准**。

---

## 5. 测试数据如何选取

测试有两层含义：**时间窗上的 holdout 全集**，以及训练中 **fixed-test 实际跑的文件子集**。

### 5.1 Holdout 时间窗（默认）

若只给了 train 起止、未显式给 test：

```text
test = train_end 所在日期的「下一个自然日」整天
     = [次日 00:00, 再次日 00:00)
```

也可显式指定：

```bash
--test-start-hour ... --test-end-hour ...
```

硬约束：

- train / test 的小时目录 **必须不相交**（代码会直接报错）；
- 训练命令要求 `data.test` 存在，用于 fixed-test 评估。

### 5.2 训练内 fixed-test 抽样

完整 holdout 日文件量很大，训练中不会每轮扫完全部文件。流程是：

1. Rank 0 在 test 时间窗内 `discover` 全部 Parquet；
2. 按均匀间隔冻结一份 **immutable manifest**；
3. 数量 = `files_per_rank × world_size`（当前默认 **`files_per_rank=4`**）；
4. 广播给各 rank，之后每次评估重复读**同一批文件**。

因此：

- **可比性**：同一 job 内多次 eval 看的是同一 holdout 切片；
- **代表性**：切片覆盖 holdout 日的时间跨度，但不是全量离线评估；
- **完整效果验收**：仍需另跑全量/分组 QAUC 等离线评估，不能只用 fixed-test 宣称论文指标复现。

---

## 6. 一张图串起来

```text
HDFS 小时分区 (pt/hr, ~500 parquet/hour)
        │
        ├─ train 小时窗 ──► 全量文件训练（按文件分片）
        │
        └─ test 小时窗（默认=训练结束后次日整天）
                │
                └─ 均匀抽样 files_per_rank×#GPU 个文件
                        └─ 训练中周期性 fixed-test
```

---

## 7. 相关入口

| 内容 | 位置 |
|---|---|
| HDFS 合同与字段枚举 | [`DATA_FORMAT.md`](../DATA_FORMAT.md) |
| 字段处理全景 | [`current_field_processing_report.md`](./current_field_processing_report.md) |
| Domain 字段设计 | [`mdl_token_feature_design.md`](./mdl_token_feature_design.md) |
| 数据适配总览 | [`mdl_data_adaptation_overview.md`](./mdl_data_adaptation_overview.md) |
| 小时窗展开 / 不相交检查 | `src/main.py`（`_expand_hour_partition`、`_resolve_train_test_hour_window`） |
| fixed-test manifest | `src/train.py`（`_prepare_fixed_test_eval`） |
