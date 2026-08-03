# 我们如何在现有数据上实现 `mdl_rankmixer` 与 `mdl_onetrans`

> 面向算法/工程同学的实现总览。  
> 事实基线：仓库 `main@5a7f1c1` 及之后的 fixed-test 默认调整，截至 2026-08-03。  
> 配套细稿：
>
> - 离线数据简介：[`mdl_offline_data_intro.md`](./mdl_offline_data_intro.md)
> - 字段与 Domain 合同：[`mdl_token_feature_design.md`](./mdl_token_feature_design.md)
> - 全字段处理：[`current_field_processing_report.md`](./current_field_processing_report.md)
> - Embedding 容量：[`embedding_shape_24h_audit.md`](./embedding_shape_24h_audit.md)
> - OneTrans 适配难点：[`mdl_onetrans_adaptation_hardships.md`](./mdl_onetrans_adaptation_hardships.md)
> - 串讲总稿：[`mdl_reproduction_lecture_report.md`](./mdl_reproduction_lecture_report.md)

## 0. 一句话结论

论文只给了 **Feature / Scenario / Task 三类 token + Domain-aware Attention** 的结构合同，没有给业务字段表。我们做的不是“把论文字段名抄进 YAML”，而是：

1. 用现有推荐/搜索 **agg Parquet** 构造可训练样本；
2. 按论文结构约束做 **字段分组、词表、Domain prompt**；
3. 对论文未写清之处给出可验证的默认实现；
4. 对必须适配我们数据分布的地方做显式工程取舍，并写进配置与测试。

当前两条生产路径：

| 模型 | 角色 | 主干 | Domain |
|---|---|---|---|
| `mdl_rankmixer` | 更接近论文 MDL × RankMixer | 32×768 Feature token，2 层 | Scenario/Task 逐层读 Feature |
| `mdl_onetrans` | 线上 OneTrans 相似主干上的组合实验 | 9 路 S + 32 NS，6 层 ×256 | sidecar Domain 读 NS，后两层再读 S |

`mdl_onetrans` **不是** MDL 或 OneTrans 论文的公开模型；上线前必须单独归因。

---

## 1. 数据从哪来，怎样进模型

### 1.1 上游形态

训练输入是小时级 **adapter Parquet**（HDFS/viewfs），不是论文里的假想 schema：

- 上游总 schema 约 630 列；adapter 实际投影约 **280～294** 个训练必扫列。
- 一行是 **request × candidate** 聚合视图：同一 `search_id` 下可有多个候选。
- 监督任务固定 3 个：
  - `fst_cart` ← `label_fst_cart`
  - `upid_pay` ← `upid_fst_trgt_noc_clk_pay_24h`
  - `cateid_filter` ← `cateid_is_fst_scene_sp_filter`

### 1.2 Adapter 做了什么

`adapt_mdl_rankmixer_parquet`（两模型共用数据合同）负责把“表”变成“模型可读轴”：

```text
Parquet row
  -> request 轴：环境/页面/query/用户摘要等
  -> candidate 轴：商品、价格、相关性、label
  -> sequence 轴：9 条 UPS 历史（按 membership 选中）
  -> 派生列：时间差、coarse scene、candidate_position
  -> FeatureBatch（可 pin / host-prepare / device prefetch）
```

关键派生：

| 派生 | 作用 |
|---|---|
| `time_delta_log1p_seconds` | 用 `impr_time - event_time` 生成 9 路连续时间差 |
| `coarse_scene_index` | 把数百个 `scene_id` 压成 **search / recommendation** 两档路由 |
| `coarse_scene_prior_id` | 给 scenario prior 用的小 identity（含 global） |
| `candidate_position` | 候选位次，供 pack/评估 |

当前默认 `agg_direct_mode=direct`：在 Arrow 上直接构造 request/candidate/sequence，避免先落成宽 legacy 表再二次转换。

### 1.3 Batch 语义：按 candidate 出，按 request 活

- **训练 batch size 按 candidate 计**（如 RankMixer 默认 1536/rank，OneTrans 默认 1408/rank；length bucket 会再缩放）。
- **历史、部分上下文、S cache 按 request 计**：同一请求多个候选共享一份 UPS/S，不能把 candidate 随机打散后再各自重解历史。
- `RequestGroupBlock` 保持请求内候选顺序；pack 阶段再切成容量允许的 candidate batch；底层 Arrow/payload 的 refcount 必须活到该请求最后一个 slice。

这是数据适配里最容易静默做错、又不立刻改 logits 的地方。

### 1.4 null / 0 / 空序列合同

上游大量使用预哈希 ID，`0` 往往是合法哈希值，不是 padding。当前合同：

| 记号 | 含义 |
|---|---|
| `0` | 预哈希后的合法取值 |
| `null` | 缺失 |
| `[]` | 已知为空的 bag/sequence |

序列某步关键字段为 null 时删整条对齐 event，而不是单字段补 0。dense 用 presence bit 区分 real zero 与 missing。label 在 `trusted_input` 下仍要求 `{0,1}` 且无 null。

---

## 2. 词表与 Embedding 设计

### 2.1 编码方式

主特征与绝大多数序列 categorical 走 **`pre_hashed`**：上游已给出哈希桶，训练侧做 `(hash & (buckets-1)) + 1` 一类映射，并单独处理 null。  
少数派生 ID（如 `coarse_scene_prior_id`）走 **identity**，不再二次哈希。

### 2.2 为什么不能按“理想低碰撞”定桶

用 10/100 文件 profile 外推到 **24h × 约 500 files/hour ≈ 12,000 文件** 后：

- 若坚持 `load ≤ 0.5` 且保留理想 dim，coarse MDL 需要约 **399 GiB/GPU（2 卡）**，训不动；
- 生产采用显存约束：24h projected load 上限约 1.75、大表 dim 封顶 32、bucket 硬上限 \(2^{30}\)；
- **shared root 按 union 定桶**（如 `goods_id` 要合并主特征 + 各历史来源），禁止只看单列 distinct。

结果落到各 YAML：coarse MDL embedding+Adagrad 规划约 **65～66 GiB/GPU**，给 dense/激活留余量。细节见 [`embedding_shape_24h_audit.md`](./embedding_shape_24h_audit.md)。

### 2.3 参数共享 vs 复制

| 范围 | 策略 | 原因 |
|---|---|---|
| 主 Feature 与多条历史同语义 ID | 可共享 physical root | 省显存、统一长尾统计 |
| Scenario / Task important | **独立 scope 表** | 论文要求 Domain 用 extra embedding，不能与普通 Feature 混梯度 |
| Scenario history prior / Task prior | 独立 encoder + 常独立表 | 条件查询角色不同；同名 raw 不等于同参数 |
| search/recommendation/global 的同名字段 important | 共用一张 scenario-scope 表 | 差异来自字段组合、prior 与 PerToken FFN，不是 3 套全复制 |

粗算 coarse MDL 约 **290** 张物理表。Task 侧精确 `goods_id` 使用独立小 dim 的 task-extra 表，避免把 268M 主表再复制进 task scope。

### 2.4 序列：同一份物理历史，多种消费者

9 条主 UPS 历史（`impr / clk_long / view_long / cart_long / buy_long / semi_clk / srch_q2i / ups_clk_sku / flatten_query_hash`）在两个模型中角色不同：

| 模型 | 主历史怎么进主干 | Domain prior |
|---|---|---|
| `mdl_rankmixer` | 各历史 LONGER summary → 进入 9 个历史 Feature token | Scenario/Task 另做 mean/attention pool prior |
| `mdl_onetrans` | 各历史 **raw event** 拼成 S（上限 2048 + SEP） | prior 只初始化 Domain prompt，不进 S 因果链 |

因此“数据只读一次”不等于“模型只编码一次”。`S-only / prior-only / S+prior` 必须消融，否则无法区分多视图证据与重复计权。

---

## 3. 论文不清楚时，我们怎么落地

论文给结构、不给字段。下表是当前默认解释（均可被实验推翻）。

### 3.1 Scenario / Task token 由什么构成

论文只说 important raw features + related prior。我们的落地：

**Scenario（coarse：search / recommendation / global）**

- important：locale、页面/入口、`scene_id`（global 不用）、类目/价格等强锚；
- prior：coarse scene prior、scene 曝光统计、scene-conditioned click 历史摘要、global 的 impr/clk/view 摘要。

**Task（`fst_cart` / `upid_pay` / `cateid_filter`）**

- important：类目层级、价格、任务相关统计，以及可控的 identity（goods/mall/query 等）；
- prior：分别用 `cart_long` / `buy_long` / `srch_q2i` 做 **task-important 条件 attention pool**。

取舍原则：

- **高频、可学习、语义锚** 优先进 important；
- **行为统计/历史摘要** 进 prior；
- 单 epoch 下极稀疏的 `goods_id` 默认不进 scenario important，而在 task 侧用独立小表补精确身份（更长窗口再消融 on/off）。

完整字段表见 [`mdl_token_feature_design.md`](./mdl_token_feature_design.md)。

### 3.2 RankMixer Feature token：不是任意等宽切片

论文要求 semantically coherent clusters 再投影。我们不用“拼成长向量再静默切 32 段”，而用 **groupwise**：

- 23 个非序列语义组 + 9 个历史组 = **32 token**；
- 每组独立 `Linear(group_width → 768)`；
- 字段增删或 dim 调整不会让相邻 token 语义漂移。

`T=32,D=768` 是我们保留的容量契约，不是把模型静默缩成 8 token。

### 3.3 Domain 状态：`coupled` vs `split`

论文公式允许 important/prior 沿 residual 直达 logit（`coupled`，默认）。  
这会让“Attention 是否真的工作”难归因，所以额外提供诊断路径 `split`：prompt 只控 Query，readout 用独立 seed。两者 checkpoint 不混用。

### 3.4 Feature FFN：生产稳定变体 ≠ 逐式复现

生产 `mdl_rankmixer` 默认 `residual_ffn + coupled`。  
若要严格对齐论文 Feature Eq.6，应显式切到 `direct_ffn + coupled`。稳定变体不能冒充逐式复现。

### 3.5 多场景怎么接到我们的 scene 体系

论文假设清晰的 scenario 集合；我们线上有大量细粒度 `scene_id`。适配方式：

- **coarse MDL**：用 allowlist 把 scene 映射为 `search` / `recommendation`（约 121 个 search scene，其余进 recommendation），再加 **global**；
- **fine MDL**：可自动发现更多 scenario token，但计算仍随 \(N_{scene}\) 增长——mask 防错融合，不自动省算力。

request-side `scene_id` 不进 RankMixer/OneTrans 的 flat/NS pack；MDL 通过 scenario important / routing 消费。避免同一 scene 信号既在内容通道又在 Domain 通道时无法归因。

### 3.6 输出头与 loss

- 最终 **只由 Task token** 接 3 个任务 head，不做 \(N_s \times N_t\) 塔。
- 默认 `loss_reduction=mean_per_task`，任务权重 `fst_cart=0.5`，`upid_pay=cateid_filter=0.01`。
- 尾部任务弱，先查有效样本与权重，再谈梯度冲突算法。

### 3.7 评测

训练内 fixed-test：训练窗后的 holdout 日上均匀抽样冻结 manifest，周期性重跑同一文件集。  
当前默认 **`files_per_rank=4`**（缩短评估、降低平台 GPU-util 杀任务风险）；报告 AUC、logloss、prob/logit 矩与有效 label 数。完整 QAUC / task×scene 离线评估另有路径，训练内 quick eval 不宣称论文效果已复现。

---

## 4. 必须适配我们数据时，做了哪些特化

### 4.1 搜索 + 推荐混部，而不是论文示例站点

我们同时服务搜/推：

- Scenario 用 coarse 两档 + global，而不是抄论文场景名；
- search important 含 `search_method_hn`，recommendation 去掉；
- `cateid_filter` 的 prior 用 `srch_q2i`，不用购买历史——按标签语义推断，需 holdout 消融。

### 4.2 历史事件没有逐 event `scene_id`

无法构造严格 per-scene 原始历史。退而求其次：

```text
query = extra_embedding(scene_id_hn, page_sn_hn)
key/value = clk_long events
→ attention_pool
```

这是 **scene-conditioned history**，不是声称上游已提供 per-scene sequence。

### 4.3 候选深度与 request 级共享极强

推荐 batch 常有 \(B \gg R\)。因此：

- OneTrans **S cache 必须按 request 存**，只在当前层 gather 到 candidate；
- Domain prompt 含候选 important，不能整体塞进 request cache；
- `mdl_onetrans` 把 Domain 做成 **sidecar**，不插入 `[S;NS]`，以免破坏 cache 边界或反转 `Domain reads Feature`。

### 4.4 长序列与显存现实

- S 最长 2048；Domain 对 S 的直接 cross-attn 只开在最后两层（`first_domain_sequence_layer=4`），每层仍读 32 NS；
- RankMixer 历史走 LONGER 压缩进 Feature token，控制 32×768 宽度；
- OOM 由 length bucket、packing、cache、checkpoint、allocator 同相叠加决定，**降 batch 不单调更省**；生产回到验证过的 profile（如 OneTrans `batch_size=1408`、`activation_checkpoint=none`）。

### 4.5 分布式稀疏表

大 embedding 走 sharded + Row-Wise Adagrad；DDP 同步 dense；稀疏梯度按全局 batch 语义聚合。多卡还要处理 HDFS 打开错峰、host-prepare IPC、以及平台对 GPU util / host mem 的 protect 规则。

### 4.6 两条模型共享数据合同、分叉消费

```text
          同一 Adapter / FeatureBatch / 词表审计
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  mdl_rankmixer            mdl_onetrans
  groupwise 32×768         raw S + DCNv2 NS
  Domain 读 Feature        Domain 读 NS(/late S)
  LONGER summaries         event-level causal S
```

这样可以在同一数据窗上对比“MDL×RankMixer”与“MDL sidecar×OneTrans”，而不是各写一套不可比 ETL。

---

## 5. 端到端路径（便于串讲）

```text
小时 Parquet (HDFS)
  → discover + shard_unit=file
  → adapter: 轴对齐 / UPS 选择 / 时间差 / coarse scene
  → direct FeatureBatch (+ host prepare / pin / prefetch)
  → EmbeddingBank（pre_hashed + scope 隔离 + shared union）
  → Tokenizer
        RankMixer: 32 语义 Feature tokens
        OneTrans:  S 序列 + 32 NS
  → Backbone + Domain-aware updates
  → Task heads → BCE (mean_per_task + weights)
  → fixed-test holdout（冻结少量文件，周期性评估）
```

---

## 6. 明确还没闭环的事

1. 固定 holdout 上完整的 **MDL vs 基线质量曲线**（不能把“能训”写成“效果已复现”）。
2. task×scene 分解、grouped QAUC、校准曲线的训练内常态化。
3. `goods important`、`cateid_filter` prior 选择、`S+prior` 重复计权等单因子消融。
4. `mdl_onetrans` 相对线上 OneTrans 的 equal-readout / scene 通道 2×2 归因矩阵。

---

## 附录：关键入口

| 主题 | 位置 |
|---|---|
| 配置 | `configs/mdl_rankmixer.yaml`、`configs/mdl_onetrans.yaml`（及 `_fine`） |
| 配置生成 | `scripts/build_mdl_rankmixer_config.py` |
| Adapter / 数据 | `src/dataloader.py` |
| 训练与 fixed-test | `src/train.py` |
| 模型 | `src/model.py`、`src/modules/` |
| Domain 字段合同 | [`mdl_token_feature_design.md`](./mdl_token_feature_design.md) |
| OneTrans 适配难点 | [`mdl_onetrans_adaptation_hardships.md`](./mdl_onetrans_adaptation_hardships.md) |
