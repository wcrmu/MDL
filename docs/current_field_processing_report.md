# 当前字段与字段处理全景审计报告

> **状态：历史快照，不是当前生产契约。** 本报告记录的是 2026-07-27
> 11:22 UTC 的 168-field/旧 prior 配置。当前恢复后的权威来源是
> `configs/{rankmixer,mdl_rankmixer,onetrans,mdl_onetrans}{,_fine}.yaml`、
> `scripts/build_mdl_rankmixer_config.py` 与
> `docs/mdl_token_feature_design.md`：147 个主字段（47 request + 100
> candidate），RankMixer 为固定 32×768 语义 token，scenario/task prior
> 组成也已更新。请勿据本报告中的旧字段数或旧 prior 名称重新生成配置。

> 审计日期：2026-07-27
>
> 仓库：`/home/user/MDL`
>
> Git 基线：`b66d5516f81687c5e04dced7131a3eba01a9d21a`（`main`）
>
> 审计对象：**当前工作树**，不是仅审计 Git HEAD。审计时相关配置、数据处理、模型与 STCA 文件均包含尚未提交的改动。
>
> 配置快照：本报告编写期间，8 份生产 YAML 被外部进程重新生成；最终稳定写入时间为 2026-07-27 11:21:56–11:22:04 UTC。报告已切换到这次最终写入后的主字段 profile，报告本身没有修改任何 YAML。
>
> 当前可加载性：最终快照中的 8 份生产配置均可通过 `load_app_config`，并可使用小表 override 构造实际模型对象。

## 0. 报告目的、范围与结论口径

这份报告回答五类问题：

1. 现在到底有哪些字段，字段处于 request、candidate、sequence、scenario、task 中的哪一层；
2. 每个字段从 Parquet 到模型之间经历了哪些归一化、截断、编码、池化、复制或路由；
3. 哪些字段是真连续值，哪些字段只是名字像价格、CTR、计数，但实际上仍是离散类别；
4. RankMixer、OneTrans、MDL-RankMixer、MDL-OneTrans 对同一批字段分别怎样消费；
5. LONGER、mean pooling 与新增的 STCA 分别在字段层面拿到什么、产生什么，并指出当前默认配置与可选插件配置的区别。

本报告的事实来源按以下优先级解释：

1. 当前可被 `load_app_config` 和 `resolve_app_config` 成功解析的 8 份配置；
2. 当前 `src/dataloader.py`、`src/config.py`、`src/model.py`、`src/modules/stca.py` 的实际代码路径；
3. 当前 `scripts/build_mdl_rankmixer_config.py` 的生成规则；
4. `DATA_FORMAT.md` 中的上游数据观测与物理格式说明；
5. `docs/stca_sequence_encoder.md` 与 `paper/STCA/main.tex` 的 STCA 对齐说明。

需要强调两个口径：

- “声明了字段”不等于“字段一定进入最终 token”；
- “同一个物理列被多个逻辑输入引用”不等于“只编码一次或共享参数”。

报告中的“当前”均指上述 11:22:04 UTC 配置快照和本报告完成时的代码工作树。由于工作树不是 clean 状态，后续若重新生成 YAML 或继续修改模型，这些数字应重新执行审计脚本确认。

---

## 1. 先给结论

### 1.1 最重要的数字

当前四个模型族及其 fine 版本共享同一套主数据合同：

| 层次 | 数量 | 含义 |
|---|---:|---|
| 上游物理列总数 | 630 | 已观测 Parquet schema 宽度；大部分不会被本模型扫描 |
| 训练必扫列 | 280 | 当前 adapter 的 mandatory raw projection |
| 训练可选列 | 12 | agg membership 索引和一个可选 item 字段 |
| 测试可选列 | 13 | 比训练多 `example_ids` |
| 主非序列逻辑字段 | 168 | 50 个 request 轴字段 + 118 个 candidate 轴字段 |
| 主非序列标量字段 | 86 | 全部按 categorical 处理 |
| 主非序列 bag 字段 | 82 | 全部按 categorical + masked mean pooling 处理 |
| 主 UPS 历史 | 9 | `impr` 到 `flatten_query_hash` |
| 原始 UPS 属性 | 107 | 9 个绝对时间戳 + 98 个预编码类别属性 |
| 模型 UPS 字段 | 107 | 9 个派生连续时间差 + 98 个类别字段 |
| 真正连续的主模型输入 | 9 个物理来源 | 每条 UPS 的 `time_delta_log1p_seconds` |
| 训练任务 | 3 | `fst_cart`、`upid_pay`、`cateid_filter` |
| 固定 coarse scenario | 2 | `search`、`recommendation` |

### 1.2 168 个主非序列字段中没有连续特征

这是当前字段审计最容易被字段名误导的地方：

- `price_hn`
- `sales_hn`
- `c_adj_ctr_15d_hn`
- `goods_query_emb32v3_cos_hn`
- `rel_score_hn`
- 各种 `*_cnt_*`
- `auto_price_p05_dis`

这些字段在当前 YAML 中都声明为：

```yaml
kind: categorical
```

它们不是直接进入网络的浮点标量，而是先按预哈希类别 ID 映射到 embedding row，再使用 embedding 向量。字段名称表达业务含义，不表达模型 dtype。

`scripts/build_mdl_rankmixer_config.py::_main_features` 会把 168 个主字段统一生成为 categorical。当前 8 份配置中：

- top-level dense feature：**0**
- top-level categorical feature：**168**

MDL 固定场景配置另外增加 10 个 scenario/task scoped categorical feature；fine MDL 增加 9 个。它们也不是 dense。

### 1.3 真正的连续特征只来自时间差

每条 UPS 历史都有一个原始毫秒时间戳：

```text
{ups}_x_time
```

adapter 用 request 的 `impr_time` 派生：

```text
delta_seconds = (impr_time - event_time) / 1000
time_delta_log1p_seconds = log1p(delta_seconds)
```

输出为：

```text
{ups}_x_time_delta_log1p_seconds
```

这 9 个派生列在 sequence 中声明为：

```yaml
kind: dense
dimension: 1
```

因此：

- 物理上有 9 个独立连续来源；
- 基础模型有 9 个 logical dense sequence field；
- 固定场景 MDL 因 scenario/task prior 的逻辑复制共有 14 个 dense field 定义；
- fine MDL 有 13 个 dense field 定义；
- 这些复制仍读取同一组 9 个物理派生列，不是新增 14 个物理连续列。

`*_x_timegap_hn` 同时存在，但它是上游的粗粒度类别桶，仍走 embedding，不能代替连续时间差。

### 1.4 当前默认序列编码器

| 模型族 | 9 条主历史的当前默认 | domain prior |
|---|---|---|
| `rankmixer` | 9 × LONGER | 无 MDL domain prior |
| `mdl_rankmixer` | 9 × LONGER | scenario/task 使用 mean pooling |
| `onetrans` | 9 × raw S stream | 无 MDL domain prior |
| `mdl_onetrans` | 9 × raw S stream | scenario/task 使用 mean pooling |

STCA 已经接入 RankMixer 家族的可选 sequence encoder 代码路径，但当前 4 份 RankMixer YAML 仍然是 `encoder: longer`。也就是说：

- 代码支持 STCA；
- 生成器暴露了 `--sequence-encoder stca` 接口；
- 当前默认生产 YAML **没有启用 STCA**；
- 启用 STCA 不会修改 LONGER、mean pooling 或 raw 的实现路径，而是通过 `SequenceConfig.encoder` 选择独立模块。

本次最终复测中，synthetic STCA profile 的配置生成、模型构造、前向/反向与 grouped-history 参数共享检查均已通过。准确口径仍是：STCA 是已验证的可选路径，但 4 份当前 RankMixer 生产 YAML 默认继续使用 LONGER；并且论文中的完整训练系统策略不等同于仅切换 encoder。详见第 12.4、20.6 和 20.7 节。

### 1.5 当前 scene 信号的真实状态

8 份配置当前都设置：

```yaml
tokenization:
  omit_scene_features: true

model:
  scene_feature_bias: none
  mdl_feature_interaction: residual_ffn   # 默认；paper 对齐用 direct_ffn
  mdl_token_state: coupled                # 原始 MDL domain-token 状态传播
```

`mdl_token_state: coupled` 中 important/prior 初始化的 token 同时作为 Query、
residual state 和最终 readout，符合 MDL 原文。可用
`--mdl-token-state split` 切到 Query/readout 分离实现：字段与 prior encoder
不变，prompt 只控制 attention，独立 state 承接 Value update 并进入 logits。
两种模式均不改变 RankMixer 的 32×768 feature-token 容量；详见
`docs/mdl_token_feature_design.md` 第 7 节。

`omit_scene_features` **只排除 request 轴 4 个字段**（其中 `scene_clk_cnt_15d_hit_hn` 已作为 dead constant 从 features 删除）：

```text
scene_id_hn
scene_impr_cnt_15d_hn
scene_impr_cnt_15d_hit_hn
scene_clk_cnt_15d_hit_hn   # deleted
```

candidate×scene 交叉 **留在 item pack**：

```text
goods_scene_clk_cnt_15d_hn
scene_adj_cartcvr_15d_hn / scene_adj_ctr_15d_hn / scene_adj_cvr_15d_hn
scene_cart_cnt_15d_hn
```

当前实际执行：

- RankMixer / OneTrans 的 feature·NS pack 都走 `resolved.feature_token_inputs`（omit + dead 过滤后）；
- `scene_id_hn` 不进 flat/NS pack，但 RankMixer / MDL-RankMixer 仍作为每条 LONGER 的 user-global；
- `scene_impr_*` 物理表在 omit 下不进 pack；MDL 通过独立 `scenario_important_scene_*` 进入 scenario token；
- 无 additive/FiLM `scene_feature_bias`（默认 `none`）；
- MDL 另有 scenario routing、scenario token、prior 与 DomainFused。

详见第 20.1 / 20.2 节。

---

## 2. 需要先分清的五层“字段”

同一个名字在讨论中常被混用。建议统一使用下面五层术语。

### 2.1 物理 Parquet 列

这是文件中真实存在的列，例如：

```text
goods_id_hn
clk_long_x_goods_id_hn
clk_long_x_time
label_fst_cart
context_indices
```

上游总 schema 有 630 列，但当前 adapter 最多只投影 293/294 个。

### 2.2 adapter 归一化输出列

adapter 会：

- 将 agg 或 req 布局归一化；
- 将 request 字段扩展或保留在 request 轴；
- 将 item 和 label 对齐到 candidate 轴；
- 按 membership 为每个 request 选择 UPS；
- 生成 `candidate_position`；
- 生成 `coarse_scene_index` 和 `coarse_scene_prior_id`；
- 用绝对时间生成 9 个连续时间差列。

因此 adapter 输出中有上游不存在的派生列。

### 2.3 `FeatureConfig` / `SequenceFieldConfig` 逻辑输入

模型配置不直接按物理列名组织所有语义。每个逻辑输入包含：

- `name`：模型引用名；
- `source`：物理或 adapter 派生列；
- `kind`：categorical 或 dense；
- `embedding_scope`：feature/scenario/task/shared；
- pooling、max length、truncation；
- encoding、embedding dim 和可能的参数共享关系。

一个 source 可以被多个逻辑 name 引用。

### 2.4 编码后的向量

categorical 会变成 embedding 向量，bag 会变成池化向量，sequence 会变成：

- event token 序列；
- LONGER summary；
- mean-pooled summary；
- attention-pooled summary；
- 或 STCA 的目标感知 `z`。

### 2.5 模型 token

编码向量最终还会被 RankMixer/OneTrans/MDL 的 tokenizer 投影：

- RankMixer feature token；
- OneTrans S token；
- OneTrans NS token；
- MDL scenario token；
- MDL task token。

因此“字段进入模型”至少要进一步问：

1. 是否被扫描；
2. 是否被 adapter 输出；
3. 是否被 tensorize；
4. 是否做了 embedding/sequence encoding；
5. 是否真的进入某类 token；
6. 是否影响最终 logit。

---

## 3. 从 HDFS 到 logit 的端到端链路

```text
HDFS Parquet
  │
  ├─ 列裁剪：280 mandatory raw + optional columns
  │
  ├─ agg / req 布局识别
  │    ├─ agg: context_indices + target_indices 都存在
  │    └─ req: 二者都不存在
  │
  ├─ adapter 轴归一化
  │    ├─ request 轴：50 个主字段 + metadata + 9 UPS
  │    ├─ candidate 轴：118 item/cross/creative + 3 labels
  │    ├─ membership 选择
  │    ├─ head 截断
  │    ├─ coarse scene 派生
  │    └─ continuous time delta 派生
  │
  ├─ request 去重与 batch pack
  │    ├─ request 侧字段只保留 unique request
  │    └─ row_indices: candidate -> request
  │
  ├─ tensorization
  │    ├─ scalar categorical -> int64 IDs
  │    ├─ bag categorical -> flat values + lengths
  │    ├─ sequence -> padded fields + lengths + row_indices
  │    ├─ labels -> float32 [candidate, 3]
  │    └─ scenario -> index/mask
  │
  ├─ field encoding
  │    ├─ categorical -> embedding
  │    ├─ bag -> masked mean
  │    ├─ dense sequence value -> BF16/FP32 activation
  │    └─ sequence -> LONGER / raw / mean_pool / STCA
  │
  ├─ tokenization
  │    ├─ RankMixer: feature tokens
  │    ├─ OneTrans: S + NS tokens
  │    └─ MDL: scenario + task tokens
  │
  ├─ backbone / domain interaction
  │
  └─ 3 个 task logits -> BCEWithLogits
```

上图只说明模块边界，不足以回答“某个字段具体怎样变成 embedding、怎样又变成 token”。下面把这条链路展开到张量级。阅读后续各节时，必须始终区分三个概念：

- **ID tensor**：仍是离散整数，例如 `price_hn -> torch.long [B]`；
- **encoded vector**：字段查 embedding 或池化后的向量，例如 `[B,16]`；
- **model token**：多个 encoded vector 再经过 tokenizer 投影后的统一宽度向量，例如 RankMixer 的 `[B,32,768]`。

当前绝大多数字段并不是“一字段对应一个 token”。RankMixer 会先把很多字段向量拼成一条长向量，再等宽切片；OneTrans `auto_split` 会用一个全局 MLP 同时生成 32 个 NS token。只有 MDL scenario/task token，以及 STCA 的 dedicated token，才比较接近“一个有明确语义的逻辑组对应一个 token”。

### 3.1 本节统一使用的形状符号

| 符号 | 含义 |
|---|---|
| \(R\) | 当前 batch 中去重后的 request 数 |
| \(B\) | candidate/example 数，也是最终 logit 的 batch 维 |
| \(L_s\) | sequence \(s\) 在当前 batch 内 padding 后的物理长度；不是配置上限 |
| \(N_f\) | bag 字段 \(f\) 在整个 batch 中保留的元素总数，即 `sum(lengths)` |
| \(D_f\) | categorical 字段 \(f\) 的 embedding dim |
| \(E_s\) | sequence \(s\) 一步中所有 categorical embedding 与 dense 值拼接后的宽度 |
| \(T\) | backbone token dim；RankMixer 为 768，OneTrans 为 256 |
| \(K\) | token 数；当前 RankMixer feature token 为 32，OneTrans NS token 为 32 |
| `row_indices` | `[B]` 的 candidate-to-request 映射，满足 `row_indices[b] ∈ [0,R)` |

当 request 去重关闭或输入天然已是 candidate-flat 时，可把 \(R\) 理解为 \(B\)，相应的 gather 退化为恒等映射。当前 8 份生产配置都开启 request 去重，因此下面优先描述 \(R \ne B\) 的真实热路径。

### 3.2 不同字段类型的完整去向总表

| 字段类型 | `FeatureBatch` 中的 CPU 形态 | embedding/编码后 | 怎样变成 token | 当前代表字段 |
|---|---|---|---|---|
| request scalar categorical | `{"values": long[R], "row_indices": long[B]}` | lookup 得 `[R,D_f]`，再 gather 成 `[B,D_f]` | RankMixer flat pack；OneTrans NS MLP；也可能被 LONGER target/user-global 或 MDL projector 复用 | `currency_hn`、`scene_id_hn` |
| candidate scalar categorical | `long[B]` | lookup 得 `[B,D_f]` | RankMixer flat pack；OneTrans NS MLP；部分字段同时作为 LONGER/STCA target | `goods_id_hn`、`price_hn` |
| request categorical bag | `values: long[N_f]`、`lengths: long[R]`、`row_indices: long[B]` | lookup `[N_f,D_f]`，segment mean 得 `[R,D_f]`，再 gather `[B,D_f]` | 与 scalar 一样进入后续 flat pack/NS MLP | `origin_query_hash_hn` |
| candidate categorical bag | `values: long[N_f]`、`lengths: long[B]` | lookup `[N_f,D_f]`，segment mean 得 `[B,D_f]` | 与 scalar 一样进入后续 flat pack/NS MLP | `goods_name_bigram_hn`、`sku_id_hn` |
| top-level dense | `float32[R,d]` 或 `[B,d]`，可另带 presence | 不查 embedding；转 embedding activation dtype，并可拼 presence bit | 作为普通 encoded vector 进入 tokenizer | **代码支持，当前主配置为 0 个** |
| sequence categorical field | padded `long[R,L_s]` | lookup 得 `[R,L_s,D_f]` | 与同一步其他字段 concat，再走 LONGER/raw/mean pool/STCA | `clk_long.goods_id_hn` |
| sequence dense field | padded `float32[R,L_s,1]` | 不查 embedding；进入 encoder bank 后转 BF16 | 与同一步 categorical embedding concat | 9 条主历史的 `time_delta_log1p_seconds` |
| MDL scoped categorical copy | 与其 source 所在轴相同 | 用该 logical name 自己的 table lookup；除非显式 `share_with`，否则不共享 | scenario/task projector 的输入，不进入 RankMixer 主 pack | `scenario_important_currency_hn`、`task_important_currency_hn` |
| scenario routing ID | `scenario_id: long[B]` | 不做 feature embedding；转 one-hot/multi-hot mask | 选择有效 scenario state，并控制 DomainFused pooling | fixed 的 `coarse_scene_index` |
| label / label mask | `float32[B,3]` | 不查 embedding | 只进入 `BCEWithLogits` 和 loss reduction | `label_fst_cart` 等 |
| group/prediction metadata | Python list 或 metadata tensor | 不编码 | 不进入 forward；用于评估聚合或输出落盘 | request ID、`example_ids` |

这张表也解释了一个常见误区：**bag、sequence 和 token 是三个不同层次**。bag 在字段层先被压成一个 `[B,D_f]` 向量；sequence 保留 event 轴或被专用 encoder 压缩；tokenizer 最后才把这些输出统一到 backbone token 空间。

### 3.3 request 字段怎样从 \(R\) 行服务 \(B\) 个 candidate

以 request 级 scalar `currency_hn` 为例：

```text
Arrow request values
    │
    ├─ pre-hashed tensorization
    ▼
ids_currency: long[R]
    │
    ├─ embedding lookup W_currency
    ▼
request_embedding: bf16[R, 8]
    │
    ├─ index_select(dim=0, row_indices[B])
    ▼
candidate_aligned_embedding: bf16[B, 8]
```

`FeatureBatch` 不会先复制原始 ID 成 `[B]`。它保留：

```python
{
    "values": ids_currency,      # [R]
    "row_indices": row_indices,  # [B]
}
```

`FeatureEncoderBank.encode_scalar_features` 先在 \(R\) 行做 lookup，再通过 `_batch_gather_indexed_rows` 一次性把多个 request 字段 gather 到 \(B\) 行。这样相同 request 下的多个 candidate 不会重复做大表 lookup。

request bag 也遵循同一边界，只是 gather 发生在 bag mean 之后：

```text
flat IDs/lengths on R requests
  -> embedding flat elements
  -> mean pool to [R,D]
  -> row_indices gather to [B,D]
```

sequence payload 同样携带 `row_indices`。LONGER/STCA/OneTrans 可以先在 request 轴处理长历史，再仅把小 summary、query 结果或 cache 对齐到 candidate 轴。这是当前 request-level batching 的核心。

### 3.4 candidate 字段不需要 request gather

以 candidate 级 `goods_id_hn` 为例：

```text
Arrow int64[B]
  -> pre-hashed long[B]
  -> W_goods[id] = bf16[B,48]
```

这个 `[B,48]` encoded vector 当前会被多路复用：

1. 作为普通主字段进入 RankMixer flat pack；
2. 作为普通主字段进入 OneTrans NS MLP；
3. 与 `cat1_id_hn:[B,16]`、`price_hn:[B,16]` 拼成 `[B,80]`，分别进入 9 条 LONGER 的 candidate-global projector；
4. 启用 STCA 时，同一组 target fields 进入 STCA target projector；
5. 主 sequence 中配置了 `share_embedding: true` 的 `*.goods_id_hn` 使用同一张 `goods_id_hn` embedding table，但 lookup 的 ID 来自历史 event 列而不是当前 candidate 列。

“同一 encoded vector 被多个模块消费”和“多个 logical field 共享 embedding table”是两件不同的事：

- 第 1–4 条是当前 candidate 的 `[B,D]` tensor 复用；
- 第 5 条是 embedding 参数复用，event tensor 仍然单独 lookup。

### 3.5 scalar categorical：从业务值到 embedding row

当前 168 个主 top-level field 全是 categorical。其中 scalar categorical 有 86 个。以 `price_hn` 为例，虽然业务语义是价格，但配置为：

```yaml
kind: categorical
embedding_dim: 16
encoding:
  type: pre_hashed
  num_buckets: 1024
  padding_id: 0
```

对非 null signed int64 值 \(v\)，ID 变换是：

\[
\operatorname{id}(v)
=
(v\ \&\ (1024-1))+1
\in [1,1024]
\]

null 单独映射到 0，因此 embedding table 是：

\[
W_{\text{price}}\in\mathbb{R}^{1025\times16}
\]

且第 0 行是固定的 padding/null row。最终：

```text
price_hn physical int64
  -> long ID [B]
  -> embedding lookup [B,16]
  -> 不作为连续价格做归一化，也不保留数值大小关系
```

当前 `validate_prehashed_nonzero=false`。这表示热路径不会拒绝物理非 null 的数值 0；若上游真的给出非 null 0，低位变换会把它变成 embedding ID 1。Arrow null 才会保留为 ID 0。这个区别对数据质量审计很重要。

### 3.6 categorical bag：先 lookup 每个元素，再做 segment mean

以 request bag `origin_query_hash_hn` 为例，配置上限 46、`truncation: head`、embedding dim 32。tensorizer 不构造 `[R,46]` 的大 padded tensor，而是产生 CSR-like 结构：

```text
values:  long[N_f]    # 所有 request 保留元素首尾拼接
lengths: long[R]      # 每个 request 的保留元素数
row_indices: long[B]  # request 字段才有
```

embedding 后：

```text
W[values] -> [N_f,32]
```

若 `pooling_null_policy=exclude`，第 \(r\) 行的结果是：

\[
e_r=
\frac{
  \sum_{j\in\operatorname{segment}(r)}
  \mathbf{1}[\operatorname{id}_j\ne0]W[\operatorname{id}_j]
}{
  \max\left(
    \sum_j\mathbf{1}[\operatorname{id}_j\ne0],1
  \right)
}
\]

空 bag 或全 null bag 返回零向量。随后 request bag 才通过 `row_indices` gather 到 `[B,32]`。

当前 82 个 bag 中：

- 73 个使用 `exclude`；
- 9 个 SKU 对齐字段使用 `include_as_padding`。

`include_as_padding` 的差别不是让 padding embedding 变成非零，而是让 ID 0 也计入分母。由于 row 0 embedding 为零，它会降低均值幅度，从而保留“这个 SKU slot 存在但该属性缺失”的对齐语义。当前 9 个字段是：

```text
sku_id_hn
sku_price_v2_hn
sku_sales_hn
sku_spec_hash_hn
sku_spec_hn
sku_cart_cnt_7d_hn
sku_ordr_cnt_1m_hn
sku_price_dis_hn
sku_sales_dis_hn
```

`sku_spec_vids_hn` 虽然也是 SKU 相关 bag，但不在这 9 个按 128 个 SKU slot 对齐的 denominator 组内。

### 3.7 top-level dense：代码怎样处理，以及为什么当前看不到

代码支持 top-level dense：

```yaml
kind: dense
dimension: d
presence: true
```

对应过程是：

```text
Arrow numeric/null
  -> float32 values [R,d] 或 [B,d]，null 填 0
  -> presence [R,1] 或 [B,1]：真实值为 1，null 为 0
  -> concat 得 [*,d+1]
  -> 转为 embedding activation dtype
  -> 直接进入 tokenizer，不查 embedding table
```

presence bit 用来区分“真实数值正好为 0”和“缺失后填 0”。但是当前 8 份生产配置的 top-level dense 数量都是 0，所以这是框架能力，不是当前训练事实。

当前真正的 dense 输入只存在于 sequence 内部的 9 个派生时间差，sequence dense 不附加 top-level presence bit；它依靠 sequence mask 表示该 event 是否有效。

### 3.8 sequence 的每个 event 怎样形成 step vector

一条 sequence 是多个严格对齐的 list column。以 `clk_long` 为例，当前有 12 个字段，其中：

- 1 个 dense：`time_delta_log1p_seconds`；
- 11 个 categorical：goods/category/price/sales/page/timegap 等。

tensorizer 产生：

```text
fields["time_delta_log1p_seconds"]: float32[R,L_clk,1]
fields["goods_id_hn"]:              long[R,L_clk]
fields["cat1_id_hn"]:               long[R,L_clk]
...
lengths:                            long[R]
has_sequence:                       bool[R]
row_indices:                        long[B]
```

每个 categorical field 分别 lookup：

```text
goods_id_hn: long[R,L] -> bf16[R,L,48]
cat1_id_hn:  long[R,L] -> bf16[R,L,8]
...
```

dense 时间差不查表：

```text
float32[R,L,1] -> bf16[R,L,1]
```

然后按 YAML 中 `fields` 的稳定顺序拼接：

\[
x_{r,t}
=
\operatorname{Concat}
\left(
W_1[id_1],\ldots,W_m[id_m],
\operatorname{time\_delta}
\right)
\in\mathbb{R}^{E_s}
\]

当前 9 条 RankMixer 主历史的 \(E_s\) 为：

| sequence | fields | \(E_s\) |
|---|---:|---:|
| `impr` | 12 | 289 |
| `clk_long` | 12 | 289 |
| `view_long` | 30 | 433 |
| `cart_long` | 12 | 377 |
| `buy_long` | 13 | 393 |
| `semi_clk` | 6 | 161 |
| `srch_q2i` | 9 | 233 |
| `ups_clk_sku` | 10 | 297 |
| `flatten_query_hash` | 3 | 41 |

event 有效窗口会被规范成 oldest-to-newest，并右对齐；左侧 padding 由 bool mask 排除。后续 encoder 分支决定 \(x_{r,t}\) 是保留为 event token，还是压成 summary。

### 3.9 LONGER 分支：event embedding 怎样变成 96 维 sequence output

当前 RankMixer/MDL-RankMixer 的 9 条主历史都走 LONGER。对每条历史：

```text
step concat x: [R,L,E_s]
  │
  ├─ categorical 部分加 learned relative position
  ├─ continuous time delta 保持为单独连续通道后再拼回
  ▼
step MLP: E_s -> 64 -> 32
  ▼
history tokens: [R,L,32]
```

除此之外还构造三类 32 维 global token：

1. `scene_id_hn:[R,16] -> MLP 16 -> 64 -> 32`：1 个 request/user-global token；
2. learned CLS：1 个 request-global token；
3. `goods_id_hn + cat1_id_hn + price_hn:[B,80] -> MLP 80 -> 64 -> 32`：1 个 candidate-global token。

因此当前 `rankmixer_summary_tokens=3`，一条历史最终返回：

```text
[user scene global, learned CLS, candidate global]
    -> 3 × 32
    -> flatten
    -> encoded[sequence_name]: [B,96]
```

这里的具体 hidden state 已经经过 LONGER 的 token merge、cross attention 和 self layer；上面的名字描述 token 来源，不代表直接原样拼接输入。

request cache 保存 history projection/KV 与 request/CLS 侧状态；candidate global 仍按 \(B\) 行计算。因此输出 `[B,96]` 是 candidate-aware，也是 scene-aware 的。即使 `omit_scene_features=true` 从 RankMixer 主 scalar pack 排除了 `scene_id_hn`，它仍通过这条 LONGER user-global 路径影响 9 条 sequence output。

空历史也不是简单全零：learned CLS、scene user-global 和 candidate-global 路径仍然存在。

### 3.10 OneTrans raw 分支：event 怎样变成 S token

OneTrans 不先把主历史压成 summary。对 sequence \(s\)：

```text
event field embeddings + dense time
    -> concat [R,L_s,E_s]
    -> sequence 专属 MLP E_s -> 1024 -> 256
    -> [R,L_s,256]
```

当前是 9 个独立 sequence group，组间插入 8 个 learned separator token。`intent_ordered` 融合保持 sequence group 顺序；每组内部有效 event 已规范为 oldest-to-newest。

padding mask 经过 stable compact 后放在前缀，有效 event 和 separator 保持稳定相对顺序；可被整个 batch 丢弃的全无效前缀会进一步裁掉。随后 9 组 S token 与 32 个 NS token 拼接，并统一加 learned position embedding。

当前 OneTrans 最大位置数 2088 恰好来自：

```text
9 条主历史配置上限之和 = 2048
+ 8 个 separator            = 2056
+ 32 个 NS token            = 2088
```

需要特别纠正旧报告中的一句话：raw S token 不是“一个 Linear 后加 sequence position”。实际是 sequence 专属两层 MLP，之后使用 S+NS 共用的 unified position embedding。

### 3.11 domain mean-pool 分支：先融合 event fields，再平均

MDL 的独立 scenario/task prior sequence 使用：

```text
x: [R,L,E_s]
  -> Linear(E_s,32)
  -> 加 learned sequence position [R,L,32]
  -> mask
  -> valid-token mean
  -> [R,32]
  -> row_indices gather
  -> [B,32]
```

公式为：

\[
p_r=
\frac{
\sum_t m_{r,t}
\left(
W_sx_{r,t}+b_s+\operatorname{Pos}_{s,t}
\right)
}{
\max(\sum_t m_{r,t},1)
}
\]

空历史返回零向量。它不是逐字段分别 mean，也不是直接平均原始 categorical embedding；所有 event fields 已经先被 concat 和投影。

MDL-OneTrans 还有一条容易遗漏的双用途路径：主 raw `impr/clk_long/view_long` 一方面进入 S stream，另一方面为了 global scenario prior，会在 `FeatureEncoderBank` 中另走 `Linear(E_s,256)+position+masked mean`，得到三个 `[B,256]` summary。两条路径复用同一批主 sequence categorical embedding table，但 S projector 与 summary projector 不同。

### 3.12 STCA 分支：同一批字段怎样变成一个 dedicated target-aware token

STCA 不是当前 YAML 默认，但 RankMixer 家族已支持按 `SequenceConfig.encoder` 选择。生成器设计的 grouped-history 路径为：

```text
9 个物理 action-family sequence
  -> 各字段 embedding + dense time
  -> 每个 stream: Linear(E_s,256)
  -> 加 stream/action-type embedding
  -> 按 continuous time delta 合并并稳定排序
  -> 加合并历史的统一 position embedding
  -> history X: [R,L_all,256]

candidate goods/cat1/price:
  [B,80] -> Linear -> target q: [B,256]

(X, mask, q, row_indices)
  -> 4 层 STCA
  -> target-aware z: [B,256]
  -> dedicated Linear(256,768)
  -> 1 个 RankMixer dedicated token: [B,1,768]
```

history-side SwiGLU+LayerNorm 可保留在 request 轴；target-to-history attention 通过 `row_indices` 让多个 candidate 读取同一 request history，不显式物化 `[B,L_all,256]` 的完整历史副本。

若不使用 `stca_history_group`，每个 STCA sequence 可各自产生 dedicated token；当前生成器的论文对齐目标是 9 流合并为一条历史并只暴露一个 \(z\)。

### 3.13 RankMixer：encoded vector 不是怎样“一字段一 token”的

当前 RankMixer resolved feature input 是：

```text
159 个非 scene scalar/bag encoded vector：总宽 3488
+ 9 个 LONGER output：9 × 96 = 864
= concat [B,4352]
```

`RankMixerSliceTokenizer` 执行：

```text
[B,4352]
  -> reshape [B,32,136]
  -> 32 组独立权重的 PerTokenLinear(136,768)
  -> feature_tokens [B,32,768]
```

所以一个字段向量可能落在某个 slice 中，也可能跨越 slice 边界；一个 slice 也可能包含多个字段的尾部和头部。当前 32 个 token 是 **flat-coordinate slices**，不是 32 个具名字段组。

每个 RankMixer block：

```text
[B,32,768]
  -> view [B,32,32,24]
  -> 交换两个 token 轴
  -> residual + LayerNorm
  -> 每 token FFN 768 -> 1536 -> 768
  -> residual + LayerNorm
```

共 2 层。纯 RankMixer 最后对 32 个 token 求均值为 `[B,768]`，再进入三个独立 task head：

```text
768 -> 1536 -> GELU -> dropout(0) -> 1
```

得到 `logits:[B,3]`。

启用 grouped STCA 时，普通输入仍走 regular slices，STCA 的 `[B,256]` 输出作为 ordered suffix 走独立 `Linear(256,768)`，不会被切片拆开。

### 3.14 OneTrans：scalar/bag vector 怎样变成 32 个 NS token

OneTrans `auto_split` 与 RankMixer 一样消费 **resolved** pack（`omit_scene` + dead constant 过滤后）：

```text
resolved.scalar_feature_names:
  当前生产约 146 个字段
  （request scene 与 dead constants 已去掉；candidate×scene 仍在）
```

实际执行：

```text
N 个 encoded vector concat -> [B, D]
  -> Linear(D, hidden)
  -> GELU
  -> Linear(hidden, num_ns_tokens * token_dim)
  -> reshape [B,32,token_dim]
```

因此当前 OneTrans 也不是一字段一 token；每个 NS token 都是过滤后主字段集合的可学习混合，且不再吃 request scene。

S/NS 合并后，6 层 OneTrans 使用混合因果注意力：

- S token 共用 S-side Q/K/V 与 FFN 参数；
- 32 个 NS 位置各自有独立的 Q/K/V 和 FFN 参数；
- NS 位于 S 之后，因此可以读取历史 S token；
- pyramid 将 S query/state 逐层压到最终 12 个；
- backbone 最终只返回 32 个 NS token给纯模型 head，S 不直接 flatten 到 head，但已经通过注意力影响 NS。

纯 OneTrans 最终：

```text
32 × 256 -> flatten [B,8192]
  -> 三个独立 head: 8192 -> 1024 -> 1
  -> logits [B,3]
```

`scenario_id` 参数在纯 OneTrans forward 中被丢弃。omit 后 NS **不含** request scene，但仍含 candidate×scene；与 RankMixer（LONGER 吃 `scene_id_hn`）的 content-side scene 强度不对称，见 20.2。

### 3.15 MDL scenario/task logical field 怎样变成 domain token

以 fixed `search` scenario token 为例，它的输入由三组 logical encoded vector 组成：

```text
8 个 scenario_important_* categorical embedding
  （locale/page + scene_id / scene_impr_*）
+ scenario_search_prior_coarse_scene embedding
+ scenario_search_clk_long_prior mean-pool [B,32]
```

这些向量 concat 后进入该 token 自己的 projector：

```text
Concat[B,d_search]
  -> Linear(d_search, hidden_dim)
  -> GELU
  -> Dropout
  -> Linear(hidden_dim, token_dim)
  -> ReLU
  -> search token [B,token_dim]
```

每个 scenario/task spec 有独立 projector。fixed MDL 产生：

```text
scenario tokens:
  search, recommendation, global  -> [B,3,T]

task tokens:
  fst_cart, upid_pay, cateid_filter -> [B,3,T]
```

同一个物理 `currency_hn` 当前至少可对应：

```text
currency_hn                     # 主 feature logical input
scenario_important_currency_hn # scenario scoped copy
task_important_currency_hn     # task scoped copy
```

它们读取同一 source，但除非配置显式 `share_with/share_embedding`，会使用独立 embedding table。也就是说，“物理值相同”不等于“向量相同”。

fixed coarse prior 的两个 logical field 使用 identity encoding。`coarse_scene_prior_id` 已是边界内整数 ID，null/out-of-range 按 identity 规则处理，不再做 pre-hash 低位映射。

### 3.16 MDL-RankMixer：三类 token 后续怎样交互

初始化后有：

```text
feature_tokens:  [B,32,768]
scenario_tokens: [B,3,768]   # fixed 配置
task_tokens:     [B,3,768]
scenario_mask:   [B,2]       # search/recommendation active mask
```

每一层按当前配置执行：

1. RankMixer token mixing + feature FFN 更新 32 个 feature token；
2. scenario token 作 Q，feature token 作 K/V，得到 scenario update；
3. `scenario_hat + scenario_ffn(scenario_hat)` 更新 scenario state；
4. task token 作 Q，feature token 作 K/V，得到 task update；
5. 按 `scenario_mask` 选择 named scenario state，并包含 global state做均值；
6. 将该 scenario context 加到每个 task state；
7. `task_hat + task_ffn(task_hat)` 更新 task state。

2 层后，三个 task token 分别进入对应的：

```text
768 -> 1536 -> 1
```

head，得到 `[B,3]` logits。

注意当前 feature 内容路径并不“纯净无 scene”：

- `scene_id_hn` 已进入每条 LONGER 的 user-global；
- scenario/task token 又提供显式 domain path；
- 因此 MDL-RankMixer 当前存在内容侧 scene 条件化与 domain prompt 两条 scene 通道。

### 3.17 MDL-OneTrans：S、NS、scenario、task 四类 token 怎样交互

初始化阶段：

```text
S tokens:        9 条 raw histories + 8 separators，长度可变，dim=256
NS tokens:       [B,32,256]
scenario tokens: fixed 时 [B,3,256]
task tokens:     [B,3,256]
```

每个 layer 先执行 OneTrans step，使 S/NS 进行混合因果注意力；随后 MDL domain block 读取当前 NS：

1. scenario token 对 32 个 NS token 做 domain-aware attention；
2. task token 对 32 个 NS token 做 domain-aware attention；
3. 从 0-based layer 4 开始，也就是第 5、6 层，scenario/task token 额外对当前 S token 做 variable-length cross attention；
4. gate 输入为 `[原 domain state, NS update, S update]`，决定注入多少 S update；
5. task state 再融合 scenario state；
6. 最终三个 256 维 task token分别进入 `256 -> 1024 -> 1` 的 head。

这里同时存在三种历史使用方式：

- 主 9 条历史作为 event-level S；
- `impr/clk_long/view_long` 另做 compact mean summary 进入 global scenario token；
- cart/buy 等独立 logical prior 做 32 维 mean summary 进入 task token。

因此分析某条历史是否“重复编码”时，必须分别检查 physical payload、embedding table、step projector、token consumer 四层，不能只看 source 名相同就下结论。

### 3.18 dtype、embedding 分片与梯度更新

当前 8 份生产配置统一为：

```text
embedding_distribution = sharded
embedding_sparse_gradients = true
embedding_weight_dtype = bf16
sparse_optimizer = rowwise_adagrad
```

主要 dtype 流转：

```text
categorical Arrow int64
  -> CPU torch.long ID
  -> sharded embedding lookup
  -> BF16 embedding activation

sequence dense time
  -> CPU float32
  -> encoder bank 中转 BF16
  -> 与 BF16 embedding concat

label
  -> float32[B,3]
```

多个 scalar/sequence ID lookup 会尽量合并到 grouped sharded lookup。相同 table 的 alias 可复用同一张参数表和去重后的 ID 通信，但不同 logical input 若未显式共享，仍是独立 table。dense tokenizer/backbone 参数由 DDP 管理，embedding row 使用稀疏 Row-Wise Adagrad 更新。

### 3.19 从 logit 到 loss：字段链路的最后一步

四个模型最终都输出：

```text
logits: [B,3]
```

task 顺序由配置稳定确定：

```text
fst_cart
upid_pay
cateid_filter
```

labels 为 `float32[B,3]`。若显式 label mask 不存在，代码不会额外分配全 1 mask；若存在，则只对有效 label 计算。当前默认 `loss_reduction=mean_per_task`：

1. 每个 task 对有效 candidate 求 `BCEWithLogits` mean；
2. 再将三个 task mean 相加。

group ID、prediction key 和 request metadata 不参与 logit 计算；它们只用于评估切片、请求级聚合或预测结果回写。

### 3.20 十个代表字段的逐字段追踪卡

为了让上述规则可以直接用于代码审查，下面选十个具有不同语义的字段逐一追踪。

#### A. `currency_hn`：request scalar categorical

```text
physical source: currency_hn
axis: request
encoding: pre_hashed, 256 buckets
embedding dim: 16

Arrow int64[R]
  -> long ID[R]
  -> W_currency[257,16] lookup
  -> BF16[R,16]
  -> row_indices gather
  -> BF16[B,16]
```

消费者：

- RankMixer：进入 159 个 non-scene scalar/bag concat；
- OneTrans：进入实际 168-field NS concat；
- MDL：主 `currency_hn` 仍走上述内容路径；
- `scenario_important_currency_hn` 和 `task_important_currency_hn` 是另外两个 logical input，读取同一 source，但各自重新 lookup 自己的独立 table。

#### B. `goods_id_hn`：candidate scalar、target input、embedding root

```text
physical source: goods_id_hn
axis: candidate
encoding: pre_hashed, 134217728 buckets
embedding dim: 48

Arrow int64[B]
  -> long ID[B]
  -> BF16[B,48]
```

消费者：

- RankMixer/OneTrans 主内容 tokenizer；
- 9 条 LONGER 的 target concat `[goods48,cat1 16,price16]=80`；
- STCA 启用时的 target concat；
- 8 条历史的 `*.goods_id_hn` 通过 `share_embedding=true` 指向这张 root table，但 event ID tensor 是 `[R,L]`，并不复用 candidate lookup 结果。

#### C. `price_hn`：名字像连续值，实际是 candidate categorical

```text
physical source: price_hn
axis: candidate
encoding: pre_hashed, 1024 buckets
embedding dim: 16

业务价格值
  -> 上游已编码 int64 bit pattern
  -> low-bit bucket ID[B]
  -> BF16 embedding[B,16]
```

它不经过 min-max、z-score、log transform，也不以 float magnitude 进入网络。除了主 tokenizer，它还与 goods/category 一起构造 LONGER/STCA target。

#### D. `origin_query_hash_hn`：request categorical bag

```text
axis: request
max_length: 46
truncation: head
null policy: exclude
embedding dim: 32

Arrow list<int64>[R]
  -> truncate
  -> flat IDs[N] + lengths[R]
  -> embedding[N,32]
  -> exclude-ID0 segment mean[R,32]
  -> row_indices gather[B,32]
  -> main tokenizer
```

最终只有一个 32 维 encoded vector，不会产生 46 个 model token。

#### E. `sku_id_hn`：candidate bag，padding 计入 denominator

```text
axis: candidate
max_length: 128
truncation: head
null policy: include_as_padding
embedding dim: 48

Arrow SKU slots[B]
  -> flat IDs[N] + lengths[B]
  -> embedding[N,48]
  -> 包含 ID0 slot 数的 segment mean[B,48]
  -> main tokenizer
```

ID0 embedding 仍为零，但 slot 数进入分母。这与普通 query/item bag 的 `exclude` 均值不同。

#### F. `scene_id_hn`：同一字段在两个 backbone 中走不同路径

```text
axis: request
encoding: pre_hashed, 1024 buckets
embedding dim: 16

Arrow int64[R]
  -> long ID[R]
  -> BF16[R,16]
  -> candidate gather[B,16]
```

消费者：

- RankMixer：不进入 direct flat pack；作为 request/user-global 输入分别进入 9 个 LONGER projector；
- OneTrans：直接进入实际 168-field NS concat；
- fine MDL 还从同一物理 source 生成 scoped `scenario_prior_scene_id_hn`，该 logical input 使用自己的 embedding 参数；
- 它与控制 active domain state 的 raw `scene_id` / `scenario_id` 不是同一个 tensor。

#### G. `clk_long.goods_id_hn`：sequence categorical field

```text
physical source: clk_long_x_goods_id_hn
axis: request-event
encoding: pre_hashed
embedding: share with top-level goods_id_hn

Arrow list<int64>[R]
  -> membership selection + max_length 2048 + anchor compaction
  -> padded long[R,L_clk]
  -> shared goods table lookup
  -> BF16[R,L_clk,48]
  -> 与该 event 其余 11 个字段 concat
```

之后：

- RankMixer family：作为 \(E_{clk}=289\) 的一部分进入 LONGER step MLP；
- OneTrans family：作为同一个 \(E_{clk}=289\) 的一部分进入 raw S projector；
- 它共享 table 参数，但不共享 event position、sequence projector 或 history cache。

#### H. `clk_long.time_delta_log1p_seconds`：sequence dense

```text
physical source before adapter: clk_long_x_time
derived source: clk_long_x_time_delta_log1p_seconds
kind: dense
dimension: 1

(impr_time - event_time)/1000
  -> log1p
  -> float32[R,L,1]
  -> BF16[R,L,1]
  -> 与 event categorical embeddings concat
```

它没有 embedding table，也没有 top-level presence bit；event 是否存在由 sequence mask 决定。LONGER、raw、mean-pool、STCA 都能消费这一个连续通道。

#### I. `scenario_search_clk_long_prior`：同 source 的独立 MDL logical sequence

```text
physical sources: 与主 clk_long 相同
logical scope: scenario
encoder: mean_pool
pool_dim: 32
embedding relation: independent from main clk_long

同一组 Arrow history columns
  -> 自己的 categorical embedding tables
  -> 自己的 Linear(E,32)
  -> 自己的 position embedding
  -> masked mean[B,32]
  -> search DomainTokenProjector
  -> search scenario token[B,T]
```

它说明“物理数据只读一份”与“表示只算一份”不能画等号。

#### J. `label_fst_cart`：监督信号，不是 feature

```text
Arrow numeric[B]
  -> float32 label[B]
  -> 按 task 顺序 stack 为 labels[B,3]
  -> 与 logits[B,3] 的第 0 列计算 BCEWithLogits
```

它不会进入 `FeatureEncoderBank`、embedding、tokenizer 或 backbone。

---

## 4. 物理布局与轴语义

### 4.1 agg 布局

一个物理行可包含多个 request 和多个 candidate。

关键控制列：

```text
context_indices
target_indices
{ups}_x_indices
```

语义是：

- `context_indices[i]` 标识第 `i` 个 request/context 位置；
- `target_indices[j]` 标识第 `j` 个 candidate 属于哪个 request；
- `{ups}_x_indices[k]` 是第 `k` 个历史事件可被哪些 request 看见。

adapter 要求：

- context request ID 唯一；
- candidate 引用的 request 必须存在；
- 每个 candidate 字段和 label 的外层长度等于 candidate count；
- UPS 每个属性的 event 轴与 membership 轴等长；
- present event 的 membership 不能是空 orphan。

### 4.2 req 布局

一个物理行就是一个 request，内部保留多个 candidate：

- 没有 `context_indices`；
- 没有 `target_indices`；
- 没有 `{ups}_x_indices`；
- context 和 UPS 直接属于该 request；
- candidate 字段与 label 仍有 candidate 外层轴。

布局判断是结构性的：

- `context_indices` 和 `target_indices` 同时存在：agg；
- 二者同时不存在：req；
- 只存在一个：非法混合布局。

### 4.3 归一化后的三轴

当前 direct path 使用三轴 payload：

| 轴 | 内容 | 最终是否按 candidate 展开 |
|---|---|---|
| request | 50 个主字段、request metadata、UPS、派生场景字段 | 通过 `row_indices` 延迟映射 |
| candidate | 118 个 item/cross/creative、labels、prediction metadata | 已是 candidate-major |
| sequence/event | 每个 request 的 ragged UPS event | 保持 request-major，必要时模型内映射 |

这样做的目的不是改变数学结果，而是避免同一 request 下多个 candidate 重复复制巨大的历史。

---

## 5. 扫描投影、metadata、label 与生成列

### 5.1 扫描列数

8 份配置的 adapter projection 一致：

| split | mandatory | optional | 最大唯一投影 |
|---|---:|---:|---:|
| train | 280 | 12 | 292 |
| test | 280 | 13 | 293 |

测试多出的 optional 字段是 `example_ids`。

### 5.2 request metadata

| 字段 | 用途 | 是否是模型 categorical feature |
|---|---|---|
| `search_id` | request/group ID、去重与评估分组 | 否 |
| `scene_id` | 原始场景路由、分场景评估 | 否 |
| `impr_time` | request 毫秒时间，用于时间差 | 否 |
| `scene_id_hn` | 上游预编码的 scene 类别字段 | 是；RankMixer direct pack 排除但 LONGER 消费，OneTrans NS 直接消费 |

`scene_id` 和 `scene_id_hn` 不是同一个概念。

### 5.3 candidate metadata

| 字段 | 来源 | 用途 |
|---|---|---|
| `candidate_position` | adapter 生成 | request 内从 0 开始的 candidate 顺序 |
| `example_ids` | 上游可选 | 预测身份 |
| `goods_id_hn` | 主模型字段 | 同时可作为预测 identity key |

当前测试 prediction keys 为：

```text
search_id
candidate_position
example_id <- example_ids
goods_id_hn
```

### 5.4 三个 label

| task | 原始列 | adapter 后 | tensor 后 |
|---|---|---|---|
| `fst_cart` | `label_fst_cart` | candidate scalar int64 | float32 |
| `upid_pay` | `upid_fst_trgt_noc_clk_pay_24h` | candidate scalar int64 | float32 |
| `cateid_filter` | `cateid_is_fst_scene_sp_filter` | candidate scalar int64 | float32 |

当前生产合同：

- label 必须是完整 0/1；
- 没有 label mask 列；
- `FeatureBatch.label_mask=None`；
- 不为全 1 mask 分配额外 tensor；
- `cateid_filter` 是预测任务，不是数据过滤条件。

当前默认 loss reduction 是 `mean_per_task`：

1. 对每个 task 的有效 candidate 求 BCE mean；
2. 再将 task mean 相加。

生成器对 STCA profile 会切到 `mean_per_request_per_task`，用于 request-level batching 对齐；当前默认 LONGER YAML 仍是 `mean_per_task`。本次专项测试已覆盖 synthetic STCA profile 的配置生成、模型构造和前向/反向，因此这里既是生成规则，也是当前已执行验证的路径。

---

## 6. 168 个主非序列字段的结构

### 6.1 request 轴：50 个

request 轴由 24 个 context 字段和 26 个 user/request-shared 字段组成：

- 15 个 scalar categorical；
- 35 个 bag categorical。

#### 24 个 context 字段

```text
currency_hn
hash_language_site_hn
language_hn
page_elsn_hn
page_sn_hn
plat_hn
region_hn
scene_id_hn
site_id_hn
timezone_hn
origin_query_hash_hn
query_arr_hn
query_hash_hn
query_terms_hash_hn
query_tfidf_term_hash_list_hn
query_extend_translation_hash_hn
search_method_hn
sess_q2q_hash_list_hn
recall_merge_cate_levels_hn
recall_merge_cate1_ids_hn
recall_merge_cate_ids_hn
scene_clk_cnt_15d_hit_hn
scene_impr_cnt_15d_hn
scene_impr_cnt_15d_hit_hn
```

#### 26 个 user/request-shared 字段

```text
u_fst_ordr_cnt_mix_d_hn
clk_7d_page_sns_hn
clk_7d_page_elsns_hn
cart_7d_cat1_ids_hn
flip_mall_ids_hn
list_clk_cat1_ids_hn
list_clk_cat_ids_hn
ups_in_cart_2h_sku_cur_prices_hn
ups_in_cart_goods_hn_share
ups_incart_cat1_id_nc_hn
ups_in_cart_tg_hn
ups_query_term_hash_v2_hn
ups_query_tg_hn
ups_search_method_hash_hn
view_30m_cat1_ids_hn
view_7d_page_sns_hn
view_7d_page_elsns_hn
offline_outside_goods_id_list_hn_share
site_q2i_good_list_hn_share
main_goods_ids_hn_share
buy_long_spec_vids_hn
cart_long_spec_vids_hn
opt_id_hn
impr_3h_tg_hn
impr_all_tg_hn
query_pay_cnt_15d_hn
```

### 6.2 candidate 轴：118 个

candidate 轴由：

- 101 个 item 字段；
- 14 个 cross 字段；
- 3 个 creative 字段；

组成，共 71 个 scalar categorical 和 47 个 bag categorical。

#### 101 个 item 字段

```text
cat_id_hn
cat1_id_hn
cat2_id_hn
cat3_id_hn
cat4_id_hn
goods_id_hn
goods_name_bigram_hn
goods_ner_infos_hn
goods_scene_clk_cnt_15d_hn
goods_title_tfidf_term_hash_list_hn
goods_avlb_sku_num_dis_hn
goods_onsale_sku_num_dis_hn
goods_cluster_id_1w_hn
rev_ratings_cnt_crs_pos_hn
g_sku_spec_hn
g_sku_spec_hash_hn
g_sku_spec_unit_list_hn
g_prpty_val_id_list_hn
sku_id_hn
sku_price_v2_hn
sku_sales_hn
sku_spec_hash_hn
sku_spec_hn
sku_spec_vids_hn
sku_cart_cnt_7d_hn
sku_ordr_cnt_1m_hn
sku_price_dis_hn
sku_sales_dis_hn
price_hn
price_bef_coupon_hn
price_after_promotion_hn
price_after_promotion_div_hn
mkt_prc_hn
show_price_hn
is_promotion_hn
promotion_discount_hn
auto_price_p05_dis
auto_price_p10_dis_hn
ori_price_hn_share
sales_hn
auto_sales_p10_dis
c_adj_cart_cvr_15d_hn
c_adj_ctr_15d_hn
c_adj_ordr_cvr_15d_hn
c_cart_cnt_15d_hn
c_clk_cnt_15d_hn
c_impr_cnt_15d_hn
c_ordr_cnt_15d_hn
c_simi_adj_cart_cvr_15d_hn
c_simi_adj_ctr_15d_hn
c_simi_cart_cnt_15d_hn
c_simi_clk_cnt_15d_hn
c_simi_impr_cnt_15d_hn
idx_c_adj_cart_cvr_15d_hn
idx_c_adj_ctr_15d_hn
idx_c_adj_ordr_cvr_15d_hn
idx_c_cart_cnt_15d_hn
idx_c_clk_cnt_15d_hn
idx_c_impr_cnt_15d_hn
idx_c_ordr_cnt_15d_hn
idx_c_simi_adj_cart_cvr_15d_hn
idx_c_simi_adj_ctr_15d_hn
idx_c_simi_cart_cnt_15d_hn
idx_c_simi_clk_cnt_15d_hn
idx_c_simi_impr_cnt_15d_hn
scene_adj_cartcvr_15d_hn
scene_adj_ctr_15d_hn
scene_adj_cvr_15d_hn
scene_cart_cnt_15d_hn
nfk_sales_14d_hn
nfk_price_14d_hn
nfk_gmv_14d_hn
i2i2cat2_swing_hn
i2i_coclk_hn_share
i2i_list_amazoni2ifullgmv_hn_share
i2i_list_multimodal_hn_share
i2i_list_swingv3gmv_hn_share
i2i_hit_site_q2i_idx_hn
only_semi_swingi2i_cut60_hn_share
semi_swingi2i_cut30_hn_share
multimodal_i2i_hit_cart_size_hn
multimodal_i2i_hit_clk_size_hn
adj_cartcvr_hn
adj_ctr_hn
adj_cvr_hn
create_time_hn
mall_id_hn
sellr_type_hn
site_x_asian_code_hn
f_goods_view_times_tg_l1_hn
target_gs_last_cart_tg_hn
clk_cnt_1d_hn
clk_3d_cnt_hn
clk_1d_cat_cnt_hn
cart_cnt_1d_hn
cart_cnt_3d_hn
impr_clk_6h_cnt_hn
clk_long_goods_abs_timegap_1d_hn
impr_long_goods_abs_timegap_1d_hn
mid_goods_prc_list_dis
mid_cmprc_diff_list_dis
```

#### 14 个 cross 字段

```text
rel_score_hn
rel_level_hn
q_hit_good_correct_unigram_hn
q2c_cart_15d_hit_val_hn
tit_in_top_query_cnt_hn
goods_query_emb32v3_cos_hn
query_cat_hn
clk_hit_i2i_idx_hn
cart_hit_i2i_idx_hn
cart_long_hit_samestyle_i2i_idx_hn
ups_clkv2_i2i_goods_ids_hit_size
ups_clkv2_i2i_goods_ids_hit_all_size
us_ctr_price_dis50_hn
impr_cat_clk_goods_ids_cnt_1d_hn
```

#### 3 个 creative 字段

```text
ad_id_bin_hn
campaign_id_hn
idx_goods_creative_id_hn
```

---

## 7. categorical 编码规则

### 7.1 预哈希不是二次 hash

当前主输入和绝大多数 sequence categorical 都使用 `pre_hashed`。

上游值是 signed int64 bit pattern。bucket 数要求是 2 的幂，映射为：

```text
embedding_id = (signed_int64_value & (num_buckets - 1)) + 1
```

规则：

- 0 号 row 永远留给 null/padding；
- 实际非空 ID 范围是 `1..num_buckets`；
- embedding table 行数是 `num_buckets + 1`；
- 负数合法，按二进制低位映射；
- 不做 `abs()`；
- 不做字符串二次 hash；
- 不做 sample min offset；
- 不加 per-field salt；
- 非空原始值 0 是合同错误。

当前生产为了吞吐设置：

```text
validate_prehashed_nonzero = false
```

因此 tensorization 热路径不会每个 batch 扫描所有字段寻找非空 0。这个开关省计算，但也意味着对上游“真实非空值不为 0”的信任更强。

### 7.2 identity 编码

固定场景 MDL 中只有两个 coarse prior logical feature 使用 identity：

```text
scenario_search_prior_coarse_scene
scenario_recommendation_prior_coarse_scene
```

两者 source 都是 `coarse_scene_prior_id`，有效 ID 为：

```text
0 = padding
1 = search
2 = recommendation
```

两个 logical feature 使用独立 embedding table，因此同一个 prior ID 在 search token 和 recommendation token 中可以学习不同的向量。

### 7.3 6 个没有 `_hn` 的 categorical

以下名称没有 `_hn`，但当前仍按 pre-hashed categorical 处理：

```text
auto_price_p05_dis
auto_sales_p10_dis
mid_goods_prc_list_dis
mid_cmprc_diff_list_dis
ups_clkv2_i2i_goods_ids_hit_size
ups_clkv2_i2i_goods_ids_hit_all_size
```

是否 categorical 只能看配置，不能只看后缀。

### 7.4 null

- categorical null -> ID 0；
- embedding padding row 0 初始化为全零；
- scalar categorical 没有额外 presence bit；
- categorical 的 `presence: true` 配置值被忽略；
- dense top-level feature 若存在则可附加 presence bit，但当前 168 个主字段没有 dense。

---

## 8. bag 字段的处理

### 8.1 82 个 bag 的统一过程

每个 bag 经过：

1. top-level null 或 `[]` 归一化为零长度 bag；
2. 按字段 `max_length` 截断；
3. 当前 82 个 bag 都使用 `truncation: head`；
4. Arrow list flatten 成一维 ID；
5. 保留每行 `lengths`，形成 CSR-like payload：

```text
values:  [sum(lengths)]
lengths: [batch]
```

6. ID embedding；
7. segmented masked mean pooling。

不构造固定 `[batch, max_length]` 的巨大 bag pad tensor。

### 8.2 两种 null denominator 策略

当前 82 个 bag：

- 73 个 `exclude`；
- 9 个 `include_as_padding`。

`exclude`：

```text
mean = sum(non_padding embeddings) / count(non_padding IDs)
```

`include_as_padding`：

```text
mean = sum(all embeddings, padding row is zero) / physical bag length
```

空 bag 的 denominator clamp 到 1，输出严格零向量。

### 8.3 9 个 SKU bag

以下 9 个字段使用 `include_as_padding`：

```text
sku_id_hn
sku_price_v2_hn
sku_sales_hn
sku_spec_hash_hn
sku_spec_hn
sku_cart_cnt_7d_hn
sku_ordr_cnt_1m_hn
sku_price_dis_hn
sku_sales_dis_hn
```

其中 8 个（不包括 `sku_spec_hn`）被配置为一个 position-aligned group：

```text
sku_id_hn
sku_price_v2_hn
sku_sales_hn
sku_spec_hash_hn
sku_cart_cnt_7d_hn
sku_ordr_cnt_1m_hn
sku_price_dis_hn
sku_sales_dis_hn
```

adapter 会验证同一 candidate 上这 8 个数组长度一致。某个属性在合法 SKU 位置为 null 时，只将该属性位置编码为 padding，不删除整个 SKU 位置。

`sku_spec_hn` 允许上游整体为 null/empty，因此保持独立 bag，不参与 8 字段长度一致性约束。

### 8.4 bag 不是 sequence

bag 的顺序不参与 attention：

- 没有 event time；
- 没有因果 mask；
- 没有 sequence position embedding；
- 截断后直接均值池化。

所以 `query_arr_hn` 或 `sku_id_hn` 即使是 list，也不能等价地称为行为序列。

---

## 9. UPS 历史的原始字段

9 条历史原始共 107 个属性：

- 9 个绝对时间戳；
- 98 个 categorical int64。

### 9.1 `impr`：12 个

```text
impr_x_time
impr_x_cat1_id_hn
impr_x_cat2_id_hn
impr_x_cat3_id_hn
impr_x_cat4_id_hn
impr_x_cat_id_hn
impr_x_goods_id_hn
impr_x_mall_id_hn
impr_x_page_sn_hn
impr_x_sales_hn
impr_x_price_hn
impr_x_timegap_hn
```

### 9.2 `clk_long`：12 个

```text
clk_long_x_time
clk_long_x_cat1_id_hn
clk_long_x_cat2_id_hn
clk_long_x_cat3_id_hn
clk_long_x_cat4_id_hn
clk_long_x_cat_id_hn
clk_long_x_goods_id_hn
clk_long_x_mall_id_hn
clk_long_x_page_sn_hn
clk_long_x_sales_hn
clk_long_x_price_hn
clk_long_x_timegap_hn
```

### 9.3 `view_long`：30 个

```text
view_long_x_time
view_long_x_cat1_id_hn
view_long_x_cat2_id_hn
view_long_x_cat3_id_hn
view_long_x_cat4_id_hn
view_long_x_cat_id_hn
view_long_x_goods_id_hn
view_long_x_mall_id_hn
view_long_x_page_sn_hn
view_long_x_sales_hn
view_long_x_price_hn
view_long_x_timegap_hn
view_long_x_clk_bottom_img_hn
view_long_x_clk_cancel_wish_hn
view_long_x_clk_carousel_hn
view_long_x_clk_evaluate_hn
view_long_x_clk_more_hn
view_long_x_clk_svid_hn
view_long_x_clk_wish_hn
view_long_x_fvid_cv_hn
view_long_x_fvid_ratio_hn
view_long_x_vid_hn
view_long_x_share_hn
view_long_x_slide_bottom_detail_hn
view_long_x_slide_bottom_img_hn
view_long_x_slide_carousel_hn
view_long_x_slide_carousel_cnt_hn
view_long_x_stay_time_hn
view_long_x_switch_sku_hn
view_long_x_switch_sku_cnt_hn
```

### 9.4 `cart_long`：12 个

```text
cart_long_x_time
cart_long_x_cat1_id_hn
cart_long_x_cat2_id_hn
cart_long_x_cat3_id_hn
cart_long_x_cat4_id_hn
cart_long_x_cat_id_hn
cart_long_x_goods_id_hn
cart_long_x_mall_id_hn
cart_long_x_price_hn
cart_long_x_timegap_hn
cart_long_x_spec_hn
cart_long_x_sku_ids_hn
```

### 9.5 `buy_long`：13 个

```text
buy_long_x_time
buy_long_x_cat1_id_hn
buy_long_x_cat2_id_hn
buy_long_x_cat3_id_hn
buy_long_x_cat4_id_hn
buy_long_x_cat_id_hn
buy_long_x_goods_id_hn
buy_long_x_mall_id_hn
buy_long_x_sales_hn
buy_long_x_price_hn
buy_long_x_timegap_hn
buy_long_x_spec_hn
buy_long_x_sku_ids_hn
```

### 9.6 `semi_clk`：6 个

```text
semi_clk_x_time
semi_clk_x_cat_id_hn
semi_clk_x_goods_id_hn
semi_clk_x_mall_id_hn
semi_clk_x_page_sn_hn
semi_clk_x_timegap_hn
```

### 9.7 `srch_q2i`：9 个

```text
srch_q2i_x_time
srch_q2i_x_cat1_id_hn
srch_q2i_x_cat2_id_hn
srch_q2i_x_cat3_id_hn
srch_q2i_x_cat4_id_hn
srch_q2i_x_cat_id_hn
srch_q2i_x_goods_id_hn
srch_q2i_x_mall_id_hn
srch_q2i_x_timegap_hn
```

### 9.8 `ups_clk_sku`：10 个

```text
ups_clk_sku_x_time
ups_clk_sku_x_cat1_id_hn
ups_clk_sku_x_cat2_id_hn
ups_clk_sku_x_cat3_id_hn
ups_clk_sku_x_cat4_id_hn
ups_clk_sku_x_cat_id_hn
ups_clk_sku_x_goods_id_hn
ups_clk_sku_x_mall_id_hn
ups_clk_sku_x_timegap_hn
ups_clk_sku_x_spec_hn
```

### 9.9 `flatten_query_hash`：3 个

```text
flatten_query_hash_x_time
flatten_query_hash_x_flat_q_hash_hn
flatten_query_hash_x_timegap_hn
```

---

## 10. UPS 从原始列到模型输入

### 10.1 membership 选择与截断顺序

agg 布局的处理顺序是：

1. 使用 `{ups}_x_indices` 选择当前 request 可见的 event；
2. 保持上游 `newest_to_oldest` 顺序；
3. 使用 `truncation: head` 保留最近事件；
4. 只为保留事件派生时间差；
5. 所有对齐字段使用同一个选择窗口。

因此 head 截断在当前物理顺序中等价于保留 recent suffix。

### 10.2 当前两组长度上限

| UPS | RankMixer/LONGER | OneTrans/raw |
|---|---:|---:|
| `impr` | 1024 | 256 |
| `clk_long` | 2048 | 512 |
| `view_long` | 2048 | 512 |
| `cart_long` | 512 | 192 |
| `buy_long` | 256 | 128 |
| `semi_clk` | 128 | 64 |
| `srch_q2i` | 100 | 100 |
| `ups_clk_sku` | 200 | 128 |
| `flatten_query_hash` | 512 | 156 |

adapter 的 `sequence_max_lengths` 和 `SequenceConfig.max_length` 当前一致。tensorizer 仍会在 pack 边界执行幂等的窗口约束，保证其他 adapter 路径也遵守 sequence 配置。

### 10.3 时间检查

在完整校验路径中：

- event time 必须是整数；
- event time 必须是 newest-to-oldest；
- event time 不能晚于 request `impr_time`；
- 负 delta 是错误。

在 trusted hot path 中，首次检查后会跳过大量逐事件诊断，以 NumPy 向量化方式计算时间差。

### 10.4 null anchor

前 8 条以 `goods_id_hn` 为 `null_anchor_field`；`flatten_query_hash` 以 `flat_q_hash_hn` 为 anchor。

处理顺序：

1. 先应用 max length/truncation；
2. 再检查窗口内 anchor；
3. anchor 为 null 的 step 从所有字段一起删除；
4. 其他字段的 null 不删除 step：
   - categorical -> ID 0；
   - dense -> 0.0。

length bucket 使用的是 anchor compaction **之前**的截断长度；最终 sequence `lengths` 使用 compaction 后长度。这样 batch shape 规划保守，同时模型 mask 仍准确。

### 10.5 padding、对齐与顺序规范化

tensorizer 产生：

```text
fields[field_name]: [request_count, batch_max_length]
lengths:            [request_count]
row_indices:        [candidate_count]  # candidate -> request，可选但当前启用
```

有效 token 被右对齐，padding 在左侧。当前原始顺序是 newest-to-oldest；需要因果语义时模型只反转有效窗口，规范成 oldest-to-newest，padding mask 不变。

---

## 11. sequence categorical 的 embedding 共享

主 9 条历史共有 98 个 categorical logical fields。当前有 76 个 alias 复用 13 个 embedding namespace：

| embedding root | sequence alias |
|---|---|
| `cat1_id_hn` | 7 条历史的 `cat1_id_hn` |
| `cat2_id_hn` | 7 条历史的 `cat2_id_hn` |
| `cat3_id_hn` | 7 条历史的 `cat3_id_hn` |
| `cat4_id_hn` | 7 条历史的 `cat4_id_hn` |
| `cat_id_hn` | 8 条历史的 `cat_id_hn` |
| `goods_id_hn` | 8 条历史的 `goods_id_hn` |
| `mall_id_hn` | 8 条历史的 `mall_id_hn` |
| `page_sn_hn` | `impr`、`clk_long`、`view_long`、`semi_clk` |
| `price_hn` | `impr`、`clk_long`、`view_long`、`cart_long`、`buy_long` |
| `sales_hn` | `impr`、`clk_long`、`view_long`、`buy_long` |
| `impr.timegap_hn` | 其余 8 条历史的 `timegap_hn` |
| `cart_long.spec_hn` | `buy_long.spec_hn`、`ups_clk_sku.spec_hn` |
| `cart_long.sku_ids_hn` | `buy_long.sku_ids_hn` |

这里的“共享”是参数共享，不是 source 共享：

- 每条历史仍读取自己的物理列；
- 每条历史仍有自己的 sequence position；
- 只有 ID -> vector 的 embedding table 指向同一个 root。

未列入上表的 view action 字段、query hash、初始 `impr.timegap_hn`、cart spec/sku root 等使用独立 table。

MDL 的 scenario/task prior sequence 虽然复用同一物理历史列，但当前被生成成 **independent embeddings**，不会自动复用主 sequence 的 embedding table。这一点与主 9 条历史内部的 76 个 alias 共享要严格区分。

---

## 12. LONGER、raw、mean pooling 和 STCA 的字段消费

### 12.1 LONGER 当前默认

RankMixer 家族每条主历史：

- `longer_dim = 32`
- `longer_num_heads = 4`
- `longer_hidden_dim = 64`
- `longer_query_tokens = 32`
- `longer_self_layers = 1`
- `longer_token_merge = 1`
- `longer_output = summary`
- `longer_cls_tokens = 1`
- `longer_user_global_tokens = 1`
- `longer_candidate_global_tokens = 1`
- `rankmixer_summary_tokens = 3`

candidate target inputs：

```text
goods_id_hn
cat1_id_hn
price_hn
```

request/user-global input：

```text
scene_id_hn
```

一条历史最终产生：

```text
1 个 scene user-global + 1 个 learned CLS + 1 个 candidate-aware summary
= 3 × 32
= 96 维 packed sequence output
```

空历史仍保留 learned CLS/target global 路径，不会简单返回全零。

因为同时包含 scene user-global 与 candidate global，主 LONGER 输出既是 scene-aware，也是 candidate-aware。历史侧 K/V 和 request globals 可以按 request 缓存，但最终与 candidate query 的交互仍按 candidate 计算。

### 12.2 OneTrans `raw`

OneTrans 家族不先把每条历史压成一个 summary：

1. 每个 event 的 categorical field embedding 与连续 time delta concat；
2. 通过该 sequence 专属的 `E_s -> 1024 -> 256` MLP；
3. 9 条历史之间插入 8 个 learned separator，形成 S token stream；
4. S 与 NS 拼接后统一加 position embedding；
5. 实际代码把过滤后的主字段（约 146 个；不含 request scene / dead constants）投影成 32 个 NS token；
6. OneTrans backbone 联合建模 S/NS。

所以 `encoder: raw` 的“raw”不是原始 int64 直接进入 Transformer，而是保留 event-level token，不预先做 sequence summary pooling。

NS 输入与 RankMixer 一样来自 resolved `feature_token_inputs`（omit + dead 过滤），不再存在“resolver 排除、tokenizer 又加回”的口径差。

### 12.3 domain `mean_pool`

MDL scenario/task prior 使用 `pool_dim=32`：

1. 每步 categorical embedding + dense time delta concat；
2. 线性投影到 32 维；
3. 加 learned sequence position embedding；
4. 用有效 sequence mask 求均值；
5. 空历史输出零向量。

因此当前 mean pooling **不是**“对原始字段 embedding 直接无位置地平均”；它平均的是已经融合全部 event fields、经过 step projector 且加了位置向量的 event token。

### 12.4 STCA 可选插件

STCA 接收与 LONGER 相同类别的输入边界：

- 多字段历史 event；
- 连续 `time_delta_log1p_seconds`；
- candidate target inputs；
- valid mask；
- candidate-to-request `row_indices`。

默认参数：

```text
stca_dim = 256
stca_layers = 4
stca_expansion_ratio = 4
stca_num_heads = model.num_heads（未显式设置时）
rankmixer_summary_tokens = 1
```

它产生一个目标感知 `z`，并在 RankMixer tokenizer 中占据一个 dedicated token。`z` 不与普通字段的等宽 slice 混在一起。

生成器启用 STCA 时的设计规则，是把 9 个物理 action-family stream 放入同一个 `stca_history_group`：

1. 每条物理 stream 各自完成字段 embedding/projector；
2. 加各自 action-type embedding；
3. 按 continuous time delta 全局合并；
4. larger delta 表示更老，稳定降序得到 oldest-to-newest；
5. 加合并后的统一 position embedding；
6. 一套 STCA stack 编码；
7. 只由 group 的第一个 member 暴露一个最终 `z`。

这使“9 个物理列族”在 STCA 逻辑边界上恢复为论文的一条 `(item/video, action type)` 历史，而不是输出 9 个互相独立的 `z`。

上述 grouped-history 规则不只停留在设计层：本次 `test_builds_stca_rankmixer_variants_without_changing_task_priors` 已验证 RankMixer 与 MDL-RankMixer 两种 synthetic STCA profile 能完成配置生成、模型构造、前向/反向，并确认 9 个 grouped members 复用同一套 STCA stack、最终只暴露一个 dedicated \(z\) token。

STCA 当前没有自动带入以下训练系统策略：

- U-shaped Beta 随机长度课程；
- 全局 token-budget 重分配；
- 自定义 flattened ragged kernel；
- 512 -> 2048 分阶段预训练；
- sequence subnetwork 预训练 schedule。

这些是论文训练/系统策略，不是 `encoder: stca` 本身的数学定义。当前 STCA 代码路径采用静态 `max_length` 和现有 padded batch；即使配置生成和模型执行已经通过，这些额外训练策略也不会自动出现。

---

## 13. MDL 额外 scoped feature

### 13.1 固定场景 MDL：178 个 top-level feature

构成：

```text
168 feature scope
+ 7 scenario scope
+ 3 task scope
= 178
```

7 个 scenario feature：

| logical name | source | encoding | embedding | 是否独立 |
|---|---|---|---:|---|
| `scenario_important_currency_hn` | `currency_hn` | pre-hashed/256 | 16 | 是 |
| `scenario_important_hash_language_site_hn` | `hash_language_site_hn` | pre-hashed/4096 | 16 | 是 |
| `scenario_important_language_hn` | `language_hn` | pre-hashed/512 | 8 | 是 |
| `scenario_important_page_elsn_hn` | `page_elsn_hn` | pre-hashed/4096 | 16 | 是 |
| `scenario_important_page_sn_hn` | `page_sn_hn` | pre-hashed/8192 | 24 | 是 |
| `scenario_search_prior_coarse_scene` | `coarse_scene_prior_id` | identity/3 | 8 | 是 |
| `scenario_recommendation_prior_coarse_scene` | `coarse_scene_prior_id` | identity/3 | 8 | 是 |

3 个 task feature：

| logical name | source | encoding | embedding | 是否独立 |
|---|---|---|---:|---|
| `task_important_currency_hn` | `currency_hn` | pre-hashed/256 | 16 | 是 |
| `task_important_hash_language_site_hn` | `hash_language_site_hn` | pre-hashed/4096 | 16 | 是 |
| `task_important_language_hn` | `language_hn` | pre-hashed/512 | 8 | 是 |

“独立”表示即使 source 与主 feature 相同，也有独立参数，不与主 embedding table 共享。

### 13.2 fine MDL：177 个 top-level feature

构成：

```text
168 feature scope
+ 6 scenario scope
+ 3 task scope
= 177
```

fine 使用自动发现的 raw scenario，并用：

```text
scenario_prior_scene_id_hn <- scene_id_hn
```

替代固定 coarse 场景下的两个 identity prior，因此少一个 logical feature。

### 13.3 逻辑复制的意义

相同 source 的独立 scoped feature 允许：

- feature 内容空间；
- scenario prompt 空间；
- task prompt 空间；

分别学习不同 embedding。它的代价是：

- 参数量增加；
- 同一物理 ID 可能发生多次 lookup；
- 必须在报告和实验中区分 source reuse 与 representation reuse。

---

## 14. MDL scenario 路由与 token

### 14.1 固定 coarse 路由

adapter 对原始 `scene_id`：

- 在 121 个 search allowlist 中 -> `coarse_scene_index=0`；
- 其他非负 ID -> `coarse_scene_index=1`；
- 负 ID -> 错误；
- unlisted policy 当前是 `recommendation`。

同时生成：

```text
coarse_scene_prior_id = coarse_scene_index + 1
```

所以：

| 场景 | scenario index | prior embedding ID |
|---|---:|---:|
| search | 0 | 1 |
| recommendation | 1 | 2 |
| padding | 不适用 | 0 |

`coarse_scene_index` 用于 scenario mask；`coarse_scene_prior_id` 用于 scoped categorical prior embedding。

### 14.2 固定场景的 3 个 scenario token

#### search token

```text
important:
  scenario_important_currency_hn
  scenario_important_hash_language_site_hn
  scenario_important_language_hn
  scenario_important_page_elsn_hn
  scenario_important_page_sn_hn

prior:
  scenario_search_prior_coarse_scene
  scenario_search_clk_long_prior
```

#### recommendation token

```text
important:
  同上 5 个

prior:
  scenario_recommendation_prior_coarse_scene
  scenario_recommendation_clk_long_prior
```

#### global token

```text
important:
  同上 5 个

prior:
  impr
  clk_long
  view_long
```

### 14.3 scenario-specific history prior

固定场景配置有两条独立 logical sequence：

```text
scenario_search_clk_long_prior
scenario_recommendation_clk_long_prior
```

二者：

- 都读取 `clk_long` 的相同物理列；
- 都是 mean pooling；
- embedding_scope 是 scenario；
- categorical embeddings 相互独立，也独立于主 `clk_long`；
- 两个 token 都会被构造，active scenario 由 mask 决定。

这是“同一份物理历史，两个场景专属表示空间”，不是简单复用一次主 LONGER 输出。

### 14.4 global history prior 的细微语义

global token 直接引用主 logical sequence：

```text
impr
clk_long
view_long
```

因此在两个模型族中的含义不同：

- `mdl_rankmixer`：引用的是主 LONGER summary，其中包含 candidate global，故 global scenario token 会带 candidate 条件；
- `mdl_onetrans`：主 sequence 是 raw S stream，但 domain projector 需要 compact summary 时按有效 event 做 mean pool，不使用 candidate target，因此是 request-side history summary。

所以“global prior 复用主 sequence”是配置层复用 logical output，不应笼统解释为两种模型里都复用同一类 candidate-independent prior。

---

## 15. MDL task token 与双路历史

三个 task token：

| task | important inputs | prior |
|---|---|---|
| `fst_cart` | currency、language-site、language 的 task scoped copy | `task_fst_cart_prior` |
| `upid_pay` | 同上 | `task_upid_pay_prior` |
| `cateid_filter` | 同上 | `task_cateid_filter_prior` |

对应物理历史：

| task prior | 物理来源 | max length | encoder |
|---|---|---:|---|
| `task_fst_cart_prior` | `cart_long` | 512 | mean_pool |
| `task_upid_pay_prior` | `buy_long` | 256 | mean_pool |
| `task_cateid_filter_prior` | `buy_long` | 256 | mean_pool |

这里确实存在双路：

```text
cart_long:
  主 feature 路 -> LONGER（RankMixer）或 raw S（OneTrans）
  fst_cart task 路 -> independent mean_pool prior

buy_long:
  主 feature 路 -> LONGER（RankMixer）或 raw S（OneTrans）
  upid_pay task 路 -> independent mean_pool prior
  cateid_filter task 路 -> 另一套 independent mean_pool prior
```

应如何理解：

- 物理 payload 只读取一份；
- logical sequence 配置被复制；
- tensor payload 可能复用 source/selection；
- embedding 和 step projector 是独立参数；
- 主路服务通用 feature 表示；
- task prior 服务任务 prompt；
- 因此这是有意的多视角编码，而不是无条件的“重复 bug”；
- 但它确实有额外计算和参数成本，必须通过消融验证收益。

---

## 16. tokenization 与最终模型消费

### 16.1 RankMixer

当前 RankMixer 主 pack：

```text
159 个非 scene scalar/bag encoded outputs
+ 9 个 LONGER outputs
= 168 个 logical projector inputs
```

以 `mdl_rankmixer.yaml` 为例：

- 159 个 scalar/bag encoded width 总和：3488；
- 9 条 LONGER 每条 96 维：864；
- concat 总宽：4352；
- 32 个 feature token；
- 每个 input slice：`4352 / 32 = 136`；
- 每个 slice 用独立 per-token linear 投影到 `token_dim=768`。

当前默认 LONGER 没有 dedicated token，每条 sequence output 只是 concat 中的一段。STCA 启用后，最终 `z` 会被放到 ordered suffix 并用独立 linear 投影成一个 dedicated token，避免被等宽切片拆散。

`scene_id_hn` 不在 159 个 scalar/bag 中，但已进入每条 LONGER 的 user-global，所以不能据此推断 RankMixer 不含 scene。

### 16.2 OneTrans

当前 OneTrans：

- 9 条主历史保留 event-level S token；
- NS 输入 = resolved pack（约 146 个；request scene / dead 已过滤，candidate×scene 保留）；
- concat 后经 MLP reshape 成 32 个 NS token（`token_dim=256`）；
- 6 层、4 heads；
- 当前两族的主字段 embedding profile 已一致；
- 9 条历史之间有 8 个 separator，S+separator 与 NS 共用 position 容量。

OneTrans 不把 9 条历史 summary 混入 NS concat；它们作为单独的 S token stream 参与注意力。

### 16.3 MDL-RankMixer

每一层大致执行：

1. RankMixer 更新 feature tokens；
2. scenario tokens 作 query，与 feature tokens 做 domain-aware attention；
3. task tokens 作 query，与 feature tokens 做 domain-aware attention；
4. scenario state 通过 DomainFusedModule 融入 task state；
5. scenario/task FFN 更新；
6. 最终每个 task token 进入对应 head。

当前 `use_scenario_feature_interaction=true` 和 `use_task_feature_interaction=true`，因此使用 attention 路，而不是 fallback domain RankMixer。

### 16.4 MDL-OneTrans

MDL-OneTrans 的 domain token：

- 每层可与 NS feature tokens 交互；
- 当前 `first_domain_sequence_layer=4`；
- 在 0-based layer 4 和 5，domain token 额外通过 gated sequence interaction 读取 S stream；
- task 再融合 scenario；
- 最终 task token 进入 task head。

这比“只把 pooled history 塞进 prior”多一条后层 S-stream interaction。

---

## 17. request 去重、缓存与重复编码

### 17.1 当前配置

8 份配置 train/test 都启用：

```text
deduplicate_request_features = true
compact_request_lists = true
use_request_cache = true
```

### 17.2 `row_indices`

request side 的 scalar、bag 和 sequence 只在 unique request 轴 tensorize。每个 payload 携带：

```text
row_indices[candidate] = request_row
```

普通 scalar/bag 在 embedding/pooling 后批量 gather 到 candidate；sequence encoder可以在 request 侧先完成可复用部分。

### 17.3 LONGER 缓存边界

LONGER：

- history projection/KV、CLS 等 request-side 部分可缓存；
- candidate global 来自 `goods_id_hn/cat1_id_hn/price_hn`；
- candidate-aware summary 不能完全缓存；
- request cache 和 candidate query 必须分开理解。

### 17.4 STCA 缓存边界

STCA：

- 每层 history SwiGLU/LayerNorm 是 candidate-independent，可按 request 缓存；
- target-to-history attention 和最终 `z` 是 candidate-dependent；
- 实现用 `row_indices` 将同 request 的 candidate query 分组；
- 不显式复制 `[candidate, history_length, dim]` history；
- 最终 `z` 不能作为 request-only cache。

### 17.5 mean_pool prior

mean_pool prior 是 request-side、candidate-independent，可以按 request 计算后 gather。独立 prior embeddings 仍意味着不同 prior logical sequence 会分别做 lookup/project/pool。

---

## 18. 8 份当前 YAML 的声明统计与可加载性

| 配置 | top-level features | 主 sequence | domain sequence | categorical inputs | tokenization | 当前校验 |
|---|---:|---|---|---:|---|---|
| `rankmixer.yaml` | 168 | 9 LONGER | 0 | 266 pre-hashed | 32 RankMixer tokens | 通过 |
| `rankmixer_fine.yaml` | 168 | 9 LONGER | 0 | 266 pre-hashed | 32 RankMixer tokens | 通过 |
| `mdl_rankmixer.yaml` | 178 | 9 LONGER | 2 scenario + 3 task mean_pool | 333：331 pre-hashed + 2 identity | 32 feature + 3 scenario + 3 task | 通过 |
| `mdl_rankmixer_fine.yaml` | 177 | 9 LONGER | 1 scenario + 3 task mean_pool | 321 pre-hashed | 32 feature + runtime scenario/global + 3 task | 通过 |
| `onetrans.yaml` | 168 | 9 raw | 0 | 266 pre-hashed | 9 S streams + 8 SEP + 32 NS | 通过 |
| `onetrans_fine.yaml` | 168 | 9 raw | 0 | 266 pre-hashed | 9 S streams + 8 SEP + 32 NS | 通过 |
| `mdl_onetrans.yaml` | 178 | 9 raw | 2 scenario + 3 task mean_pool | 333：331 pre-hashed + 2 identity | S/NS + 3 scenario + 3 task | 通过 |
| `mdl_onetrans_fine.yaml` | 177 | 9 raw | 1 scenario + 3 task mean_pool | 321 pre-hashed | S/NS + runtime scenario/global + 3 task | 通过 |

补充：

- 非 MDL 配置解析器仍会生成兼容性的 fallback scenario/task spec，但纯模型类不构造也不消费 MDL domain projector，不能把这些 fallback spec 当成实际模型输入；
- fixed 版本 scenario names 为 `search/recommendation`；
- fine 版本配置中只有 `__auto__` placeholder，实际 scenario 数在训练前自动发现后确定；
- 所有配置都使用 3 个 task；
- 所有配置都使用 BF16 runtime；
- 当前默认 loss 都是 `mean_per_task`。

---

## 19. 当前校验策略

当前 8 份配置的实际 reader 值是：

```text
trusted_input = true
cardinality_audit_raw_rows = 1
eager_schema_validation = sample
schema_validation_samples = 1
validate_prehashed_nonzero = false
agg_direct_mode = direct
```

data schema policy：

```text
require_same_schema = true
allow_missing_nullable_columns = false
validate_before_train = true
```

即使 trusted：

- agg/req 结构仍必须合法；
- request/candidate 外层长度仍必须匹配；
- membership 仍必须合法；
- scalar 长度 > 1 仍是硬错误；
- sequence 对齐仍在关键边界检查；
- label 完整合同仍检查。

但逐 payload 的非零预哈希、时间顺序、详细诊断在 warm-up 后会显著减少。

注意：当前 `DATA_FORMAT.md` 第 11 节仍提到默认 256 个 cardinality audit raw rows 和更大的 schema sample 口径；这与当前 8 份 YAML 解析出的实际值 `1/1` 不一致。本报告以配置解析结果为准，并建议后续同步文档。

---

## 20. 当前值得关注的审计发现

### 20.1 `omit_scene_features`：request 轴 omit + dead constant 出包（已对齐）

当前策略（代码与生产 YAML 已落地）：

- **只从 feature/NS pack 排除 request 轴 scene**：`scene_id_hn`、`scene_impr_cnt_15d_hn`、`scene_impr_cnt_15d_hit_hn`（`scene_clk_cnt_15d_hit_hn` 为 dead constant，已从 features 删除）；
- **candidate×scene 交叉保留在 item pack**：`scene_adj_*`、`scene_cart_cnt_15d_hn`、`goods_scene_*`；
- **distinct≈1 dead constants**（`c_*` / `c_simi_*` / `idx_c_simi_*` / `clk_7d_page_elsns_hn` / `scene_clk_*` / near-constant `ad_id_bin_hn` / `ups_in_cart_2h_sku_cur_prices_hn`）已从 features 与 pack 移除；
- RankMixer 与 OneTrans `auto_split` 都消费 `resolved.tokenization.feature_token_inputs` / `resolved.scalar_feature_names`（omit + dead 过滤后），不再各自重枚举全量 feature/shared；
- encoder inclusion 按最终 consumer 裁剪：pack + LONGER `target`/`user_global` + MDL scenario/task 独立表；
  - pure RankMixer：request scene 里 **仅 `scene_id_hn` 因 LONGER user-global 保留 embedding**；
  - MDL-RankMixer：**不再**把 `scene_id_hn` 挂 LONGER user-global；request scene 只走 `scenario_important_scene_*`；
- MDL scenario token 的 important 侧挂 `scenario_important_scene_{id,impr_*}` 独立表（不复用 feature-scope 主表）；task importants 另含 locale + `cat1`/`goods`（goods 独立表桶封顶 16M）。

残留注意：`table_to_feature_batch` 仍遍历 `config.features`；已删除的 dead 字段不再出现。request `scene_impr_*` 仍声明在 features（adapter 轴 / scenario-important source），但默认 omit 下不进 pack；encoder 只吃 scenario_important 副本（MDL）或完全不吃（pure）。

### 20.2 纯基线 content-side scene 不对称（实验设计）

当前实际状态：

- pure RankMixer forward 丢弃 `scenario_id`，但 `scene_id_hn` 通过 LONGER user-global 进入；
- pure OneTrans forward 丢弃 `scenario_id`，且 omit 后 **NS 不再吃 request scene**；candidate×scene 仍在 NS；
- 两者都没有 MDL scenario token，也没有 additive/FiLM `scene_feature_bias`；
- MDL 在相同家族内容路径之外，额外拥有 scenario mask、scenario token、prior 和 DomainFused。

因此当前对比更接近：

```text
RankMixer baseline: scene_id through LONGER content path (+ candidate×scene in pack)
OneTrans baseline:  no request scene in NS; candidate×scene in NS
MDL:                family content path + explicit domain path (scenario important includes request scene)
```

这仍不是严格对称的 baseline。若实验要回答“MDL domain path 是否有效”，应固定并记录 content-side scene policy，至少做 `scene content on/off × MDL domain on/off` 的消融。

### 20.3 global scenario history：已改为 request-only mean_pool（已对齐）

旧实现：`global.prior_inputs = impr/clk_long/view_long` 复用主 LONGER summary（含 candidate global），global scenario token 不是纯 request prior。

当前实现（生产 YAML + builder）：

- `global.prior_inputs = scenario_global_{impr,clk_long,view_long}_prior`
- 三条独立 `embedding_scope: scenario` + `mean_pool` 克隆，**无** `target_inputs` / candidate LONGER
- 与 search/recommendation 的 `*_clk_long_prior` 同构：domain prior，不泄漏候选

代价：多 3 张独立高基 goods 表（≈12 GiB×3 @ dim 48 bf16）。若要压显存，优先消融「复用主 UPS embedding vs 独立表」，而不是再改回 LONGER summary。

### 20.4 scenario-specific prior 与 task prior 的重复有明确参数边界

当前不是简单的“一份 embedding 结果进两条路”：

- source 相同；
- logical sequence 不同；
- embedding 独立；
- projector 独立；
- pooling 独立。

这增强了表达能力，也增加参数和计算。建议消融：

```text
shared source + shared embedding
shared source + independent embedding（当前 specific/task/global）
reuse main sequence output（已废弃于 global）
independent mean_pool prior（当前）
```

另：concrete scenario prior 的 `goods_id` 仍是 **64 维**，global/pack `goods_id` 是 **48 维**；同 ID 空间不同维，消融/共享时要先对齐 dim。
### 20.5 validation 配置比文档描述更激进

当前仅 1 个 raw row 做 cardinality audit，schema sample 也是 1。对持续变化的 630 列上游来说，这更偏吞吐优先。

建议：

- 生产训练热路径保持 1；
- 在每日数据发布/预检 job 中跑更大 sample；
- 将“训练热路径校验”和“离线数据门禁”拆开；
- 修正文档，避免团队误以为训练前已抽检 64/256。

### 20.6 STCA 已对齐 encoder，但论文训练策略仍未自动化

不能只看 `encoder: stca` 就宣称“完整复现 STCA 论文训练系统”。准确表述应是：

- STCA encoder 方程与 request-level history reuse 已实现；
- 9 流全局历史合并已实现；
- one-z dedicated token 已实现；
- RLB loss reduction profile 已支持；
- U-shaped Beta curriculum、token budget redistribution、custom ragged kernel、staged pretraining 未自动实现。

### 20.7 专项测试结果与剩余的 reference 配置一致性问题

本报告完成后执行了两组专项测试。

第一组：

```text
tests/test_exclude_scene_feature_tokens.py
tests/test_stca.py
tests/test_mdl_rankmixer_adapter.py
```

结果：

```text
70 passed
```

这覆盖了 scene token 排除、STCA 数学/梯度/请求复用路径和主 adapter 合同。

第二组：

```text
tests/test_build_mdl_rankmixer_config.py
tests/test_config_overlays.py
```

结果：

```text
41 passed
5 failed
```

5 个失败属于同一类 reference/overlay 问题：

```text
scenario_single_column_history
encoder=attention_pool
缺少 target_inputs
```

当前 `SequenceConfig` 要求 `attention_pool` 必须有 target inputs，但旧 `configs/reference/*` profile 尚未同步。这些失败影响 reference profile 的全仓一致性，不影响本报告解析的 8 份当前生产 YAML。

随后又对 tokenizer/lookup 合同执行了定向测试，覆盖 RankMixer slice、OneTrans auto-split、sequence MLP、时间融合、padding mask 和 MDL/OneTrans lookup fusion：

```text
6 passed
1 failed
53 deselected
```

唯一失败的 `test_mdl_fuses_scalar_and_all_sequence_lookups_with_output_parity` 也是在测试主体执行前加载同一个 `configs/reference/mdl_perf.yaml` 时，被上述缺少 `target_inputs` 的校验问题拦截；没有出现新的 tokenizer 数值不一致。

第二组中原先暴露的 STCA ordinary-width 整除失败在当前最终工作树上已不再出现；STCA generator 测试现已通过。生产配置的构造、前向与反向也在该组成功项中通过。由于本次任务只新增本报告，没有修改 reference profile 或运行代码，因此没有在报告任务中顺手修复这 5 个既有 reference 配置失败；在宣布“全仓测试全绿”前仍应处理它们。

---

## 21. 建议的字段治理方式

建议把字段治理长期拆成四张机器可生成的表：

### 21.1 Physical schema registry

至少包含：

```text
physical_name
arrow_type
nullable
axis
producer
first_seen_partition
last_seen_partition
```

### 21.2 Logical feature registry

```text
logical_name
source
kind
embedding_scope
pooling
max_length
encoding
num_buckets
embedding_dim
share_with
consumer_tokens
```

### 21.3 Sequence registry

```text
sequence_name
physical_stream
fields
order
truncation
max_length
null_anchor
time_delta
encoder
target_inputs
history_group
parameter_group
```

### 21.4 Consumer graph

明确每个 logical input 最终进入：

```text
feature token
S token
NS token
scenario important
scenario prior
task important
task prior
prediction metadata
label
evaluation-only
```

这样可以自动发现：

- 扫描但不消费的字段；
- 编码但不进 token 的字段；
- 同 source 多 table；
- source/type 漂移；
- main/prior 间意外共享；
- scene feature 的职责重叠。

---

## 22. 如何回答团队最常见的几个问题

### Q1：我们有连续特征吗？

有，但只在 sequence 中：

- 9 个物理 `time_delta_log1p_seconds`；
- top-level 168 个主字段没有 dense。

### Q2：价格、CTR、count 为什么不是连续？

当前数据合同认为这些是上游离散化/哈希后的类别 bit pattern，配置明确是 categorical。名字不决定 dtype。

### Q3：mean pooling 是不是只支持 sequence？

不是：

- 82 个普通 categorical bag 用 masked mean；
- MDL domain prior sequence 用 event-token mean；
- `encoder=raw` 的主 OneTrans sequence在需要 domain compact summary 时也可走 mean summary；
- 三者输入形状和位置语义不同。

### Q4：同一个 `clk_long` 进多条路，是不是重复？

物理数据不重复扫描，但逻辑编码和参数可能重复：

- 主 LONGER/raw；
- search mean_pool prior；
- recommendation mean_pool prior；
- global token 复用主 output。

是否保留取决于要表达“共享表示”还是“场景专属表示”，不能只用“重复/不重复”判断。

### Q5：STCA 会替换所有旧代码吗？

不会。它是 `encoder: stca` 选择的插件路径：

- LONGER 不变；
- mean_pool 不变；
- raw 不变；
- tokenizer 对 STCA 的 `z` 做 dedicated token 处理。

### Q6：STCA 现在是默认吗？

不是。当前 RankMixer/MDL-RankMixer YAML 默认仍是 LONGER。

### Q7：scene 字段现在还在 feature pack 吗？

分轴：

- **request 轴**（`scene_id` / `scene_impr_*`）：`omit_scene_features=true` 时不进 RankMixer flat pack / OneTrans NS；`scene_clk_*` 已作为 dead 删除；
- **candidate×scene**（`scene_adj_*` / `goods_scene_*`）：留在 item pack；
- RankMixer / MDL-RankMixer：`scene_id_hn` 仍进每条 LONGER user-global；
- MDL：另有 `scenario_important_scene_*` 进 scenario token。

### Q8：scene 字段是不是完全不处理了？

不是。candidate×scene 仍走主 pack；request `scene_id_hn` 在 RankMixer 家族走 LONGER；MDL 还有 scenario important / prior / DomainFused。omit 后无 consumer 的 request scene 物理表不再进 encoder inclusion。

### Q9：MDL 的场景信息从哪里来？

固定版本来自：

- `coarse_scene_index` 的 active mask；
- 独立 coarse prior embedding；
- scenario-important（locale/page + `scene_id` / `scene_impr_*`）；
- scenario-specific `clk_long` mean_pool prior；
- global `impr/clk_long/view_long` prior；
- scenario 对 feature 的 domain-aware attention + per-scenario FFN。

### Q10：纯 RankMixer/OneTrans 还有显式场景吗？

不消费独立 `scenario_id` 参数。内容侧：

- RankMixer：LONGER `scene_id_hn` user-global + item pack 里的 candidate×scene；
- OneTrans：NS **不含** request scene；仍含 candidate×scene。

因此两边 content-side scene 强度仍不对称，MDL domain 消融需固定 scene 策略（见 20.2）。

---

## 23. 主要源码索引

| 主题 | 文件 |
|---|---|
| 权威数据合同 | `DATA_FORMAT.md` |
| 8 份生产配置 | `configs/*.yaml` |
| 配置 dataclass、校验、resolve | `src/config.py` |
| agg/req adapter、scene/time 派生 | `src/dataloader.py` |
| categorical 编码公式 | `src/features.py` |
| embedding 与分布式 lookup | `src/embeddings.py` |
| scalar/bag/sequence 编码与模型 | `src/model.py` |
| STCA encoder | `src/modules/stca.py` |
| 配置生成规则 | `scripts/build_mdl_rankmixer_config.py` |
| STCA 对齐说明 | `docs/stca_sequence_encoder.md` |
| STCA 原文 | `paper/STCA/main.tex` |

---

## 附录 A：168 个主字段的逐字段配置矩阵

说明：

- “轴”来自 adapter 的 `context_features`/`item_features`；
- “形态”来自 `pooling`；
- “max”只对 bag 有意义；
- “emb”与“buckets”取当前可解析的 `rankmixer.yaml`，并已核对与 `mdl_rankmixer.yaml` 的 168 个主字段一致；
- 当前 RankMixer 与 OneTrans 的 168 个主字段使用同一 embedding profile；
- “RankMixer 主 pack”列只描述 RankMixer 的 direct flat pack；`scene_id_hn` 仍可经 LONGER 间接进入；
- OneTrans `auto_split` 当前实际消费全部 168 个字段，因此表中的 scene 排除标记不能套用于 OneTrans；
- 序号沿用字段合同的历史稳定编号；当前不存在的原 `#25 uid_or_bg_hn` 被保留为空档，所以最后一个有效字段编号仍为 169，但有效行数是 168。

| 历史序号 | 字段 | 轴 | 形态 | max | emb | buckets | RankMixer direct pack |
|---:|---|---|---|---:|---:|---:|---|
| 1 | `currency_hn` | request | scalar | — | 16 | 256 | 进入 |
| 2 | `hash_language_site_hn` | request | scalar | — | 16 | 4096 | 进入 |
| 3 | `language_hn` | request | scalar | — | 8 | 512 | 进入 |
| 4 | `page_elsn_hn` | request | scalar | — | 16 | 4096 | 进入 |
| 5 | `page_sn_hn` | request | scalar | — | 24 | 8192 | 进入 |
| 6 | `plat_hn` | request | scalar | — | 8 | 64 | 进入 |
| 7 | `region_hn` | request | scalar | — | 16 | 2048 | 进入 |
| 8 | `scene_id_hn` | request | scalar | — | 16 | 1024 | 排除(scene) |
| 9 | `site_id_hn` | request | scalar | — | 16 | 1024 | 进入 |
| 10 | `timezone_hn` | request | scalar | — | 16 | 2048 | 进入 |
| 11 | `origin_query_hash_hn` | request | bag/mean | 46 | 32 | 1048576 | 进入 |
| 12 | `query_arr_hn` | request | bag/mean | 53 | 32 | 524288 | 进入 |
| 13 | `query_hash_hn` | request | bag/mean | 46 | 32 | 262144 | 进入 |
| 14 | `query_terms_hash_hn` | request | bag/mean | 30 | 32 | 524288 | 进入 |
| 15 | `query_tfidf_term_hash_list_hn` | request | bag/mean | 30 | 32 | 1048576 | 进入 |
| 16 | `query_extend_translation_hash_hn` | request | bag/mean | 15 | 24 | 262144 | 进入 |
| 17 | `search_method_hn` | request | scalar | — | 16 | 256 | 进入 |
| 18 | `sess_q2q_hash_list_hn` | request | bag/mean | 6 | 32 | 2097152 | 进入 |
| 19 | `recall_merge_cate_levels_hn` | request | bag/mean | 256 | 8 | 256 | 进入 |
| 20 | `recall_merge_cate1_ids_hn` | request | bag/mean | 24 | 16 | 256 | 进入 |
| 21 | `recall_merge_cate_ids_hn` | request | bag/mean | 256 | 32 | 262144 | 进入 |
| 22 | `scene_clk_cnt_15d_hit_hn` | request | scalar | — | 8 | 1024 | 已删除(dead) |
| 23 | `scene_impr_cnt_15d_hn` | request | bag/mean | 30 | 32 | 65536 | 排除(scene) |
| 24 | `scene_impr_cnt_15d_hit_hn` | request | scalar | — | 16 | 2048 | 排除(scene) |
| 26 | `u_fst_ordr_cnt_mix_d_hn` | request | bag/mean | 3 | 16 | 4096 | 进入 |
| 27 | `clk_7d_page_sns_hn` | request | bag/mean | 512 | 16 | 8192 | 进入 |
| 28 | `clk_7d_page_elsns_hn` | request | bag/mean | 512 | 8 | 1024 | 进入 |
| 29 | `cart_7d_cat1_ids_hn` | request | bag/mean | 512 | 16 | 256 | 进入 |
| 30 | `flip_mall_ids_hn` | request | bag/mean | 512 | 48 | 4194304 | 进入 |
| 31 | `list_clk_cat1_ids_hn` | request | bag/mean | 128 | 16 | 256 | 进入 |
| 32 | `list_clk_cat_ids_hn` | request | bag/mean | 128 | 32 | 262144 | 进入 |
| 33 | `ups_in_cart_2h_sku_cur_prices_hn` | request | bag/mean | 128 | 8 | 1024 | 进入 |
| 34 | `ups_in_cart_goods_hn_share` | request | bag/mean | 256 | 48 | 16777216 | 进入 |
| 35 | `ups_incart_cat1_id_nc_hn` | request | bag/mean | 5 | 24 | 8192 | 进入 |
| 36 | `ups_in_cart_tg_hn` | request | bag/mean | 256 | 8 | 1024 | 进入 |
| 37 | `ups_query_term_hash_v2_hn` | request | bag/mean | 128 | 32 | 2097152 | 进入 |
| 38 | `ups_query_tg_hn` | request | bag/mean | 128 | 8 | 1024 | 进入 |
| 39 | `ups_search_method_hash_hn` | request | bag/mean | 128 | 16 | 256 | 进入 |
| 40 | `view_30m_cat1_ids_hn` | request | bag/mean | 128 | 16 | 256 | 进入 |
| 41 | `view_7d_page_sns_hn` | request | bag/mean | 512 | 16 | 8192 | 进入 |
| 42 | `view_7d_page_elsns_hn` | request | bag/mean | 512 | 16 | 8192 | 进入 |
| 43 | `offline_outside_goods_id_list_hn_share` | request | bag/mean | 512 | 48 | 16777216 | 进入 |
| 44 | `site_q2i_good_list_hn_share` | request | bag/mean | 49 | 48 | 1048576 | 进入 |
| 45 | `main_goods_ids_hn_share` | request | bag/mean | 45 | 32 | 2097152 | 进入 |
| 46 | `buy_long_spec_vids_hn` | request | bag/mean | 512 | 48 | 8388608 | 进入 |
| 47 | `cart_long_spec_vids_hn` | request | bag/mean | 512 | 64 | 8388608 | 进入 |
| 48 | `opt_id_hn` | request | scalar | — | 24 | 16384 | 进入 |
| 49 | `impr_3h_tg_hn` | request | bag/mean | 256 | 8 | 1024 | 进入 |
| 50 | `impr_all_tg_hn` | request | bag/mean | 256 | 8 | 1024 | 进入 |
| 51 | `query_pay_cnt_15d_hn` | request | scalar | — | 16 | 4096 | 进入 |
| 52 | `cat_id_hn` | candidate | scalar | — | 32 | 262144 | 进入 |
| 53 | `cat1_id_hn` | candidate | scalar | — | 16 | 256 | 进入 |
| 54 | `cat2_id_hn` | candidate | scalar | — | 24 | 4096 | 进入 |
| 55 | `cat3_id_hn` | candidate | scalar | — | 24 | 32768 | 进入 |
| 56 | `cat4_id_hn` | candidate | scalar | — | 32 | 131072 | 进入 |
| 57 | `goods_id_hn` | candidate | scalar | — | 48 | 134217728 | 进入 |
| 58 | `goods_name_bigram_hn` | candidate | bag/mean | 85 | 48 | 8388608 | 进入 |
| 59 | `goods_ner_infos_hn` | candidate | bag/mean | 45 | 48 | 1048576 | 进入 |
| 60 | `goods_scene_clk_cnt_15d_hn` | candidate | bag/mean | 6 | 16 | 4096 | 进入(item×scene) |
| 61 | `goods_title_tfidf_term_hash_list_hn` | candidate | bag/mean | 6 | 32 | 4194304 | 进入 |
| 62 | `goods_avlb_sku_num_dis_hn` | candidate | scalar | — | 16 | 1024 | 进入 |
| 63 | `goods_onsale_sku_num_dis_hn` | candidate | scalar | — | 16 | 1024 | 进入 |
| 64 | `goods_cluster_id_1w_hn` | candidate | scalar | — | 32 | 524288 | 进入 |
| 65 | `rev_ratings_cnt_crs_pos_hn` | candidate | bag/mean | 5 | 24 | 4096 | 进入 |
| 66 | `g_sku_spec_hn` | candidate | bag/mean | 25 | 48 | 8388608 | 进入 |
| 67 | `g_sku_spec_hash_hn` | candidate | bag/mean | 25 | 48 | 8388608 | 进入 |
| 68 | `g_sku_spec_unit_list_hn` | candidate | bag/mean | 15 | 48 | 8388608 | 进入 |
| 69 | `g_prpty_val_id_list_hn` | candidate | bag/mean | 58 | 32 | 2097152 | 进入 |
| 70 | `sku_id_hn` | candidate | bag/mean | 128 | 48 | 67108864 | 进入 |
| 71 | `sku_price_v2_hn` | candidate | bag/mean | 128 | 16 | 4096 | 进入 |
| 72 | `sku_sales_hn` | candidate | bag/mean | 128 | 16 | 1024 | 进入 |
| 73 | `sku_spec_hash_hn` | candidate | bag/mean | 128 | 48 | 8388608 | 进入 |
| 74 | `sku_spec_hn` | candidate | bag/mean | 128 | 48 | 8388608 | 进入 |
| 75 | `sku_spec_vids_hn` | candidate | bag/mean | 256 | 48 | 8388608 | 进入 |
| 76 | `sku_cart_cnt_7d_hn` | candidate | bag/mean | 128 | 16 | 4096 | 进入 |
| 77 | `sku_ordr_cnt_1m_hn` | candidate | bag/mean | 128 | 16 | 4096 | 进入 |
| 78 | `sku_price_dis_hn` | candidate | bag/mean | 128 | 16 | 4096 | 进入 |
| 79 | `sku_sales_dis_hn` | candidate | bag/mean | 128 | 16 | 1024 | 进入 |
| 80 | `price_hn` | candidate | scalar | — | 16 | 1024 | 进入 |
| 81 | `price_bef_coupon_hn` | candidate | scalar | — | 16 | 1024 | 进入 |
| 82 | `price_after_promotion_hn` | candidate | scalar | — | 16 | 1024 | 进入 |
| 83 | `price_after_promotion_div_hn` | candidate | bag/mean | 5 | 24 | 16384 | 进入 |
| 84 | `mkt_prc_hn` | candidate | scalar | — | 16 | 1024 | 进入 |
| 85 | `show_price_hn` | candidate | scalar | — | 16 | 1024 | 进入 |
| 86 | `is_promotion_hn` | candidate | scalar | — | 8 | 16 | 进入 |
| 87 | `promotion_discount_hn` | candidate | scalar | — | 16 | 4096 | 进入 |
| 88 | `auto_price_p05_dis` | candidate | scalar | — | 16 | 4096 | 进入 |
| 89 | `auto_price_p10_dis_hn` | candidate | scalar | — | 16 | 4096 | 进入 |
| 90 | `ori_price_hn_share` | candidate | scalar | — | 16 | 1024 | 进入 |
| 91 | `sales_hn` | candidate | scalar | — | 16 | 2048 | 进入 |
| 92 | `auto_sales_p10_dis` | candidate | scalar | — | 16 | 4096 | 进入 |
| 93 | `c_adj_cart_cvr_15d_hn` | candidate | scalar | — | 8 | 1024 | 进入 |
| 94 | `c_adj_ctr_15d_hn` | candidate | scalar | — | 8 | 1024 | 进入 |
| 95 | `c_adj_ordr_cvr_15d_hn` | candidate | scalar | — | 8 | 1024 | 进入 |
| 96 | `c_cart_cnt_15d_hn` | candidate | scalar | — | 8 | 1024 | 进入 |
| 97 | `c_clk_cnt_15d_hn` | candidate | scalar | — | 8 | 1024 | 进入 |
| 98 | `c_impr_cnt_15d_hn` | candidate | scalar | — | 8 | 1024 | 进入 |
| 99 | `c_ordr_cnt_15d_hn` | candidate | scalar | — | 8 | 1024 | 进入 |
| 100 | `c_simi_adj_cart_cvr_15d_hn` | candidate | scalar | — | 8 | 1024 | 进入 |
| 101 | `c_simi_adj_ctr_15d_hn` | candidate | scalar | — | 8 | 1024 | 进入 |
| 102 | `c_simi_cart_cnt_15d_hn` | candidate | scalar | — | 8 | 1024 | 进入 |
| 103 | `c_simi_clk_cnt_15d_hn` | candidate | scalar | — | 8 | 1024 | 进入 |
| 104 | `c_simi_impr_cnt_15d_hn` | candidate | scalar | — | 8 | 1024 | 进入 |
| 105 | `idx_c_adj_cart_cvr_15d_hn` | candidate | scalar | — | 16 | 1024 | 进入 |
| 106 | `idx_c_adj_ctr_15d_hn` | candidate | scalar | — | 16 | 1024 | 进入 |
| 107 | `idx_c_adj_ordr_cvr_15d_hn` | candidate | scalar | — | 16 | 1024 | 进入 |
| 108 | `idx_c_cart_cnt_15d_hn` | candidate | scalar | — | 16 | 1024 | 进入 |
| 109 | `idx_c_clk_cnt_15d_hn` | candidate | scalar | — | 16 | 1024 | 进入 |
| 110 | `idx_c_impr_cnt_15d_hn` | candidate | scalar | — | 16 | 1024 | 进入 |
| 111 | `idx_c_ordr_cnt_15d_hn` | candidate | scalar | — | 16 | 1024 | 进入 |
| 112 | `idx_c_simi_adj_cart_cvr_15d_hn` | candidate | scalar | — | 8 | 1024 | 进入 |
| 113 | `idx_c_simi_adj_ctr_15d_hn` | candidate | scalar | — | 8 | 1024 | 进入 |
| 114 | `idx_c_simi_cart_cnt_15d_hn` | candidate | scalar | — | 8 | 1024 | 进入 |
| 115 | `idx_c_simi_clk_cnt_15d_hn` | candidate | scalar | — | 8 | 1024 | 进入 |
| 116 | `idx_c_simi_impr_cnt_15d_hn` | candidate | scalar | — | 8 | 1024 | 进入 |
| 117 | `scene_adj_cartcvr_15d_hn` | candidate | bag/mean | 6 | 16 | 4096 | 进入(item×scene) |
| 118 | `scene_adj_ctr_15d_hn` | candidate | bag/mean | 6 | 16 | 4096 | 进入(item×scene) |
| 119 | `scene_adj_cvr_15d_hn` | candidate | bag/mean | 6 | 16 | 4096 | 进入(item×scene) |
| 120 | `scene_cart_cnt_15d_hn` | candidate | bag/mean | 6 | 16 | 4096 | 进入(item×scene) |
| 121 | `nfk_sales_14d_hn` | candidate | scalar | — | 16 | 2048 | 进入 |
| 122 | `nfk_price_14d_hn` | candidate | scalar | — | 16 | 1024 | 进入 |
| 123 | `nfk_gmv_14d_hn` | candidate | scalar | — | 16 | 2048 | 进入 |
| 124 | `i2i2cat2_swing_hn` | candidate | bag/mean | 38 | 24 | 16384 | 进入 |
| 125 | `i2i_coclk_hn_share` | candidate | bag/mean | 30 | 64 | 16777216 | 进入 |
| 126 | `i2i_list_amazoni2ifullgmv_hn_share` | candidate | bag/mean | 30 | 64 | 16777216 | 进入 |
| 127 | `i2i_list_multimodal_hn_share` | candidate | bag/mean | 30 | 64 | 16777216 | 进入 |
| 128 | `i2i_list_swingv3gmv_hn_share` | candidate | bag/mean | 30 | 64 | 16777216 | 进入 |
| 129 | `i2i_hit_site_q2i_idx_hn` | candidate | bag/mean | 28 | 16 | 4096 | 进入 |
| 130 | `only_semi_swingi2i_cut60_hn_share` | candidate | bag/mean | 60 | 48 | 2097152 | 进入 |
| 131 | `semi_swingi2i_cut30_hn_share` | candidate | bag/mean | 30 | 48 | 2097152 | 进入 |
| 132 | `multimodal_i2i_hit_cart_size_hn` | candidate | scalar | — | 16 | 2048 | 进入 |
| 133 | `multimodal_i2i_hit_clk_size_hn` | candidate | scalar | — | 16 | 2048 | 进入 |
| 134 | `adj_cartcvr_hn` | candidate | scalar | — | 16 | 1024 | 进入 |
| 135 | `adj_ctr_hn` | candidate | scalar | — | 16 | 1024 | 进入 |
| 136 | `adj_cvr_hn` | candidate | scalar | — | 16 | 1024 | 进入 |
| 137 | `create_time_hn` | candidate | scalar | — | 8 | 1024 | 进入 |
| 138 | `mall_id_hn` | candidate | scalar | — | 48 | 4194304 | 进入 |
| 139 | `sellr_type_hn` | candidate | scalar | — | 8 | 256 | 进入 |
| 140 | `site_x_asian_code_hn` | candidate | scalar | — | 16 | 8192 | 进入 |
| 141 | `f_goods_view_times_tg_l1_hn` | candidate | scalar | — | 24 | 4096 | 进入 |
| 142 | `target_gs_last_cart_tg_hn` | candidate | scalar | — | 8 | 1024 | 进入 |
| 143 | `impr_clk_6h_cnt_hn` | candidate | scalar | — | 16 | 4096 | 进入 |
| 144 | `clk_long_goods_abs_timegap_1d_hn` | candidate | bag/mean | 37 | 8 | 1024 | 进入 |
| 145 | `impr_long_goods_abs_timegap_1d_hn` | candidate | bag/mean | 38 | 8 | 1024 | 进入 |
| 146 | `mid_goods_prc_list_dis` | candidate | bag/mean | 60 | 16 | 4096 | 进入 |
| 147 | `mid_cmprc_diff_list_dis` | candidate | bag/mean | 60 | 16 | 2048 | 进入 |
| 148 | `rel_score_hn` | candidate | scalar | — | 16 | 1024 | 进入 |
| 149 | `rel_level_hn` | candidate | scalar | — | 8 | 1024 | 进入 |
| 150 | `q_hit_good_correct_unigram_hn` | candidate | bag/mean | 14 | 24 | 32768 | 进入 |
| 151 | `q2c_cart_15d_hit_val_hn` | candidate | scalar | — | 24 | 32768 | 进入 |
| 152 | `tit_in_top_query_cnt_hn` | candidate | bag/mean | 19 | 16 | 4096 | 进入 |
| 153 | `goods_query_emb32v3_cos_hn` | candidate | scalar | — | 16 | 8192 | 进入 |
| 154 | `query_cat_hn` | candidate | bag/mean | 46 | 32 | 262144 | 进入 |
| 155 | `clk_hit_i2i_idx_hn` | candidate | bag/mean | 29 | 16 | 2048 | 进入 |
| 156 | `cart_hit_i2i_idx_hn` | candidate | bag/mean | 25 | 16 | 2048 | 进入 |
| 157 | `cart_long_hit_samestyle_i2i_idx_hn` | candidate | bag/mean | 16 | 16 | 2048 | 进入 |
| 158 | `ups_clkv2_i2i_goods_ids_hit_size` | candidate | scalar | — | 16 | 1024 | 进入 |
| 159 | `ups_clkv2_i2i_goods_ids_hit_all_size` | candidate | scalar | — | 16 | 1024 | 进入 |
| 160 | `us_ctr_price_dis50_hn` | candidate | scalar | — | 16 | 4096 | 进入 |
| 161 | `impr_cat_clk_goods_ids_cnt_1d_hn` | candidate | scalar | — | 16 | 4096 | 进入 |
| 162 | `ad_id_bin_hn` | candidate | scalar | — | 8 | 1024 | 进入 |
| 163 | `campaign_id_hn` | candidate | scalar | — | 32 | 4194304 | 进入 |
| 164 | `idx_goods_creative_id_hn` | candidate | scalar | — | 48 | 8388608 | 进入 |
| 165 | `clk_cnt_1d_hn` | candidate | bag/mean | 1 | 16 | 1024 | 进入 |
| 166 | `clk_3d_cnt_hn` | candidate | bag/mean | 1 | 16 | 2048 | 进入 |
| 167 | `clk_1d_cat_cnt_hn` | candidate | bag/mean | 1 | 16 | 4096 | 进入 |
| 168 | `cart_cnt_1d_hn` | candidate | bag/mean | 1 | 16 | 1024 | 进入 |
| 169 | `cart_cnt_3d_hn` | candidate | bag/mean | 1 | 16 | 1024 | 进入 |

---

## 附录 B：主 9 条 sequence 的逐字段逻辑映射

说明：

- 每条 sequence 的第一个字段都是 adapter 派生的连续 time delta；
- 其余字段为 pre-hashed categorical；
- “共享”表示 embedding 参数共享；
- “独立”表示该 logical input 自有 embedding table；
- 所有字段在同一 sequence 内必须 event 轴等长。

### B.1 `impr`

| logical field | source | kind/dim | embedding |
|---|---|---|---|
| `time_delta_log1p_seconds` | `impr_x_time_delta_log1p_seconds` | dense/1 | 不适用 |
| `cat1_id_hn` | `impr_x_cat1_id_hn` | categorical/8 | 共享 `cat1_id_hn` |
| `cat2_id_hn` | `impr_x_cat2_id_hn` | categorical/16 | 共享 `cat2_id_hn` |
| `cat3_id_hn` | `impr_x_cat3_id_hn` | categorical/24 | 共享 `cat3_id_hn` |
| `cat4_id_hn` | `impr_x_cat4_id_hn` | categorical/32 | 共享 `cat4_id_hn` |
| `cat_id_hn` | `impr_x_cat_id_hn` | categorical/32 | 共享 `cat_id_hn` |
| `goods_id_hn` | `impr_x_goods_id_hn` | categorical/48 | 共享 `goods_id_hn` |
| `mall_id_hn` | `impr_x_mall_id_hn` | categorical/32 | 共享 `mall_id_hn` |
| `page_sn_hn` | `impr_x_page_sn_hn` | categorical/24 | 共享 `page_sn_hn` |
| `sales_hn` | `impr_x_sales_hn` | categorical/16 | 共享 `sales_hn` |
| `price_hn` | `impr_x_price_hn` | categorical/16 | 共享 `price_hn` |
| `timegap_hn` | `impr_x_timegap_hn` | categorical/16 | 独立 root |

### B.2 `clk_long`

| logical field | source | kind/dim | embedding |
|---|---|---|---|
| `time_delta_log1p_seconds` | `clk_long_x_time_delta_log1p_seconds` | dense/1 | 不适用 |
| `cat1_id_hn` | `clk_long_x_cat1_id_hn` | categorical/8 | 共享 `cat1_id_hn` |
| `cat2_id_hn` | `clk_long_x_cat2_id_hn` | categorical/16 | 共享 `cat2_id_hn` |
| `cat3_id_hn` | `clk_long_x_cat3_id_hn` | categorical/24 | 共享 `cat3_id_hn` |
| `cat4_id_hn` | `clk_long_x_cat4_id_hn` | categorical/32 | 共享 `cat4_id_hn` |
| `cat_id_hn` | `clk_long_x_cat_id_hn` | categorical/32 | 共享 `cat_id_hn` |
| `goods_id_hn` | `clk_long_x_goods_id_hn` | categorical/48 | 共享 `goods_id_hn` |
| `mall_id_hn` | `clk_long_x_mall_id_hn` | categorical/32 | 共享 `mall_id_hn` |
| `page_sn_hn` | `clk_long_x_page_sn_hn` | categorical/24 | 共享 `page_sn_hn` |
| `sales_hn` | `clk_long_x_sales_hn` | categorical/16 | 共享 `sales_hn` |
| `price_hn` | `clk_long_x_price_hn` | categorical/16 | 共享 `price_hn` |
| `timegap_hn` | `clk_long_x_timegap_hn` | categorical/16 | 共享 `impr.timegap_hn` |

### B.3 `view_long`

| logical field | source | kind/dim | embedding |
|---|---|---|---|
| `time_delta_log1p_seconds` | `view_long_x_time_delta_log1p_seconds` | dense/1 | 不适用 |
| `cat1_id_hn` | `view_long_x_cat1_id_hn` | categorical/8 | 共享 `cat1_id_hn` |
| `cat2_id_hn` | `view_long_x_cat2_id_hn` | categorical/16 | 共享 `cat2_id_hn` |
| `cat3_id_hn` | `view_long_x_cat3_id_hn` | categorical/24 | 共享 `cat3_id_hn` |
| `cat4_id_hn` | `view_long_x_cat4_id_hn` | categorical/32 | 共享 `cat4_id_hn` |
| `cat_id_hn` | `view_long_x_cat_id_hn` | categorical/32 | 共享 `cat_id_hn` |
| `goods_id_hn` | `view_long_x_goods_id_hn` | categorical/48 | 共享 `goods_id_hn` |
| `mall_id_hn` | `view_long_x_mall_id_hn` | categorical/32 | 共享 `mall_id_hn` |
| `page_sn_hn` | `view_long_x_page_sn_hn` | categorical/24 | 共享 `page_sn_hn` |
| `sales_hn` | `view_long_x_sales_hn` | categorical/16 | 共享 `sales_hn` |
| `price_hn` | `view_long_x_price_hn` | categorical/16 | 共享 `price_hn` |
| `timegap_hn` | `view_long_x_timegap_hn` | categorical/16 | 共享 `impr.timegap_hn` |
| `clk_bottom_img_hn` | `view_long_x_clk_bottom_img_hn` | categorical/8 | 独立 |
| `clk_cancel_wish_hn` | `view_long_x_clk_cancel_wish_hn` | categorical/8 | 独立 |
| `clk_carousel_hn` | `view_long_x_clk_carousel_hn` | categorical/8 | 独立 |
| `clk_evaluate_hn` | `view_long_x_clk_evaluate_hn` | categorical/8 | 独立 |
| `clk_more_hn` | `view_long_x_clk_more_hn` | categorical/8 | 独立 |
| `clk_svid_hn` | `view_long_x_clk_svid_hn` | categorical/8 | 独立 |
| `clk_wish_hn` | `view_long_x_clk_wish_hn` | categorical/8 | 独立 |
| `fvid_cv_hn` | `view_long_x_fvid_cv_hn` | categorical/8 | 独立 |
| `fvid_ratio_hn` | `view_long_x_fvid_ratio_hn` | categorical/8 | 独立 |
| `vid_hn` | `view_long_x_vid_hn` | categorical/32 | 独立 |
| `share_hn` | `view_long_x_share_hn` | categorical/32 | 独立 |
| `slide_bottom_detail_hn` | `view_long_x_slide_bottom_detail_hn` | categorical/8 | 独立 |
| `slide_bottom_img_hn` | `view_long_x_slide_bottom_img_hn` | categorical/8 | 独立 |
| `slide_carousel_hn` | `view_long_x_slide_carousel_hn` | categorical/8 | 独立 |
| `slide_carousel_cnt_hn` | `view_long_x_slide_carousel_cnt_hn` | categorical/8 | 独立 |
| `stay_time_hn` | `view_long_x_stay_time_hn` | categorical/16 | 独立 |
| `switch_sku_hn` | `view_long_x_switch_sku_hn` | categorical/8 | 独立 |
| `switch_sku_cnt_hn` | `view_long_x_switch_sku_cnt_hn` | categorical/8 | 独立 |

### B.4 `cart_long`

| logical field | source | kind/dim | embedding |
|---|---|---|---|
| `time_delta_log1p_seconds` | `cart_long_x_time_delta_log1p_seconds` | dense/1 | 不适用 |
| `cat1_id_hn` | `cart_long_x_cat1_id_hn` | categorical/8 | 共享 `cat1_id_hn` |
| `cat2_id_hn` | `cart_long_x_cat2_id_hn` | categorical/16 | 共享 `cat2_id_hn` |
| `cat3_id_hn` | `cart_long_x_cat3_id_hn` | categorical/24 | 共享 `cat3_id_hn` |
| `cat4_id_hn` | `cart_long_x_cat4_id_hn` | categorical/32 | 共享 `cat4_id_hn` |
| `cat_id_hn` | `cart_long_x_cat_id_hn` | categorical/32 | 共享 `cat_id_hn` |
| `goods_id_hn` | `cart_long_x_goods_id_hn` | categorical/48 | 共享 `goods_id_hn` |
| `mall_id_hn` | `cart_long_x_mall_id_hn` | categorical/32 | 共享 `mall_id_hn` |
| `price_hn` | `cart_long_x_price_hn` | categorical/16 | 共享 `price_hn` |
| `timegap_hn` | `cart_long_x_timegap_hn` | categorical/16 | 共享 `impr.timegap_hn` |
| `spec_hn` | `cart_long_x_spec_hn` | categorical/48 | 独立 root |
| `sku_ids_hn` | `cart_long_x_sku_ids_hn` | categorical/48 | 独立 root |

### B.5 `buy_long`

| logical field | source | kind/dim | embedding |
|---|---|---|---|
| `time_delta_log1p_seconds` | `buy_long_x_time_delta_log1p_seconds` | dense/1 | 不适用 |
| `cat1_id_hn` | `buy_long_x_cat1_id_hn` | categorical/8 | 共享 `cat1_id_hn` |
| `cat2_id_hn` | `buy_long_x_cat2_id_hn` | categorical/16 | 共享 `cat2_id_hn` |
| `cat3_id_hn` | `buy_long_x_cat3_id_hn` | categorical/24 | 共享 `cat3_id_hn` |
| `cat4_id_hn` | `buy_long_x_cat4_id_hn` | categorical/32 | 共享 `cat4_id_hn` |
| `cat_id_hn` | `buy_long_x_cat_id_hn` | categorical/32 | 共享 `cat_id_hn` |
| `goods_id_hn` | `buy_long_x_goods_id_hn` | categorical/48 | 共享 `goods_id_hn` |
| `mall_id_hn` | `buy_long_x_mall_id_hn` | categorical/32 | 共享 `mall_id_hn` |
| `sales_hn` | `buy_long_x_sales_hn` | categorical/16 | 共享 `sales_hn` |
| `price_hn` | `buy_long_x_price_hn` | categorical/16 | 共享 `price_hn` |
| `timegap_hn` | `buy_long_x_timegap_hn` | categorical/16 | 共享 `impr.timegap_hn` |
| `spec_hn` | `buy_long_x_spec_hn` | categorical/48 | 共享 `cart_long.spec_hn` |
| `sku_ids_hn` | `buy_long_x_sku_ids_hn` | categorical/48 | 共享 `cart_long.sku_ids_hn` |

### B.6 `semi_clk`

| logical field | source | kind/dim | embedding |
|---|---|---|---|
| `time_delta_log1p_seconds` | `semi_clk_x_time_delta_log1p_seconds` | dense/1 | 不适用 |
| `cat_id_hn` | `semi_clk_x_cat_id_hn` | categorical/32 | 共享 `cat_id_hn` |
| `goods_id_hn` | `semi_clk_x_goods_id_hn` | categorical/48 | 共享 `goods_id_hn` |
| `mall_id_hn` | `semi_clk_x_mall_id_hn` | categorical/32 | 共享 `mall_id_hn` |
| `page_sn_hn` | `semi_clk_x_page_sn_hn` | categorical/24 | 共享 `page_sn_hn` |
| `timegap_hn` | `semi_clk_x_timegap_hn` | categorical/16 | 共享 `impr.timegap_hn` |

### B.7 `srch_q2i`

| logical field | source | kind/dim | embedding |
|---|---|---|---|
| `time_delta_log1p_seconds` | `srch_q2i_x_time_delta_log1p_seconds` | dense/1 | 不适用 |
| `cat1_id_hn` | `srch_q2i_x_cat1_id_hn` | categorical/8 | 共享 `cat1_id_hn` |
| `cat2_id_hn` | `srch_q2i_x_cat2_id_hn` | categorical/16 | 共享 `cat2_id_hn` |
| `cat3_id_hn` | `srch_q2i_x_cat3_id_hn` | categorical/24 | 共享 `cat3_id_hn` |
| `cat4_id_hn` | `srch_q2i_x_cat4_id_hn` | categorical/32 | 共享 `cat4_id_hn` |
| `cat_id_hn` | `srch_q2i_x_cat_id_hn` | categorical/32 | 共享 `cat_id_hn` |
| `goods_id_hn` | `srch_q2i_x_goods_id_hn` | categorical/48 | 共享 `goods_id_hn` |
| `mall_id_hn` | `srch_q2i_x_mall_id_hn` | categorical/32 | 共享 `mall_id_hn` |
| `timegap_hn` | `srch_q2i_x_timegap_hn` | categorical/16 | 共享 `impr.timegap_hn` |

### B.8 `ups_clk_sku`

| logical field | source | kind/dim | embedding |
|---|---|---|---|
| `time_delta_log1p_seconds` | `ups_clk_sku_x_time_delta_log1p_seconds` | dense/1 | 不适用 |
| `cat1_id_hn` | `ups_clk_sku_x_cat1_id_hn` | categorical/8 | 共享 `cat1_id_hn` |
| `cat2_id_hn` | `ups_clk_sku_x_cat2_id_hn` | categorical/16 | 共享 `cat2_id_hn` |
| `cat3_id_hn` | `ups_clk_sku_x_cat3_id_hn` | categorical/24 | 共享 `cat3_id_hn` |
| `cat4_id_hn` | `ups_clk_sku_x_cat4_id_hn` | categorical/32 | 共享 `cat4_id_hn` |
| `cat_id_hn` | `ups_clk_sku_x_cat_id_hn` | categorical/32 | 共享 `cat_id_hn` |
| `goods_id_hn` | `ups_clk_sku_x_goods_id_hn` | categorical/48 | 共享 `goods_id_hn` |
| `mall_id_hn` | `ups_clk_sku_x_mall_id_hn` | categorical/32 | 共享 `mall_id_hn` |
| `timegap_hn` | `ups_clk_sku_x_timegap_hn` | categorical/16 | 共享 `impr.timegap_hn` |
| `spec_hn` | `ups_clk_sku_x_spec_hn` | categorical/48 | 共享 `cart_long.spec_hn` |

### B.9 `flatten_query_hash`

| logical field | source | kind/dim | embedding |
|---|---|---|---|
| `time_delta_log1p_seconds` | `flatten_query_hash_x_time_delta_log1p_seconds` | dense/1 | 不适用 |
| `flat_q_hash_hn` | `flatten_query_hash_x_flat_q_hash_hn` | categorical/32 | 独立 |
| `timegap_hn` | `flatten_query_hash_x_timegap_hn` | categorical/16 | 共享 `impr.timegap_hn` |

---

## 附录 C：MDL sequence 复制矩阵

### C.1 固定场景 MDL-RankMixer

| logical sequence | scope | source family | encoder | max | categorical | dense |
|---|---|---|---|---:|---:|---:|
| `impr` | feature | impr | LONGER | 1024 | 11 | 1 |
| `clk_long` | feature | clk_long | LONGER | 2048 | 11 | 1 |
| `view_long` | feature | view_long | LONGER | 2048 | 29 | 1 |
| `cart_long` | feature | cart_long | LONGER | 512 | 11 | 1 |
| `buy_long` | feature | buy_long | LONGER | 256 | 12 | 1 |
| `semi_clk` | feature | semi_clk | LONGER | 128 | 5 | 1 |
| `srch_q2i` | feature | srch_q2i | LONGER | 100 | 8 | 1 |
| `ups_clk_sku` | feature | ups_clk_sku | LONGER | 200 | 9 | 1 |
| `flatten_query_hash` | feature | flatten_query_hash | LONGER | 512 | 2 | 1 |
| `scenario_search_clk_long_prior` | scenario | clk_long | mean_pool | 2048 | 11 | 1 |
| `scenario_recommendation_clk_long_prior` | scenario | clk_long | mean_pool | 2048 | 11 | 1 |
| `task_fst_cart_prior` | task | cart_long | mean_pool | 512 | 11 | 1 |
| `task_upid_pay_prior` | task | buy_long | mean_pool | 256 | 12 | 1 |
| `task_cateid_filter_prior` | task | buy_long | mean_pool | 256 | 12 | 1 |

合计：

```text
14 logical sequences
155 categorical sequence fields
14 dense sequence fields
107 unique physical/derived sequence sources
```

### C.2 固定场景 MDL-OneTrans

逻辑复制关系相同，但：

- 9 条主 sequence 的 encoder 是 raw；
- 主长度使用 OneTrans 较短的一组上限；
- scenario-specific `clk_long` prior max 为 512；
- task prior 上限仍是 cart 512、buy 256；
- domain prior 仍是 mean pooling。

### C.3 fine MDL

fine 只有一个自动场景 prior：

```text
scenario_auto_clk_long_prior
```

所以：

```text
13 logical sequences
144 categorical sequence fields
13 dense sequence fields
107 unique physical/derived sequence sources
```

---

## 附录 D：categorical input 与 embedding 统计

以当前固定场景 `mdl_rankmixer.yaml` 为例：

```text
333 categorical logical inputs
331 pre-hashed
2 identity
76 share aliases
13 alias roots
```

333 个 logical categorical input 的 embedding dim 分布：

| embedding dim | logical input 数 |
|---:|---:|
| 8 | 71 |
| 16 | 114 |
| 24 | 43 |
| 32 | 44 |
| 48 | 47 |
| 64 | 14 |

这里的统计包含：

- 168 个 top-level 主字段；
- MDL scenario/task scoped copy；
- 主 sequence categorical field；
- scenario/task prior sequence categorical field。

它不是“独立 embedding table 数”，因为 76 个 alias 会指向共享 root；也不是物理列数，因为多个 logical input 可引用同一 source。

bucket 分布覆盖：

```text
3
16
64
128
256
512
1024
2048
4096
8192
16384
32768
65536
131072
262144
524288
1048576
2097152
4194304
8388608
16777216
33554432
67108864
134217728
```

所有 pre-hashed bucket 都是 2 的幂；3 只属于 coarse scene identity table。

---

## 附录 E：审计复现检查清单

每次配置生成或字段调整后，建议至少验证：

### E.1 配置层

```text
[ ] 8 份 YAML 都能 load_app_config
[ ] 8 份 YAML 都能 resolve_app_config
[ ] 主 FeatureConfig 数仍符合预期
[ ] request/item 轴仍是 51/118
[ ] scalar/bag 仍是 86/82
[ ] sequence field 对齐无重复 logical name
[ ] pre-hashed bucket 是 2 的幂
[ ] identity ID 范围与 padding 不冲突
[ ] share_with target 的 bucket/dim 一致
```

### E.2 adapter 层

```text
[ ] agg/req 双布局测试
[ ] context/target/membership 外层长度测试
[ ] orphan membership 拒绝
[ ] top-level null/[] 语义
[ ] aligned SKU 长度一致
[ ] candidate_position request 内重置
[ ] scene allowlist 与 unlisted policy
[ ] event time 非未来且顺序正确
[ ] time delta 公式准确
```

### E.3 tensor 层

```text
[ ] signed int64 low-bit pre-hash
[ ] null -> 0
[ ] bag flat values + lengths
[ ] include/exclude padding mean
[ ] sequence truncate-then-anchor-compact
[ ] newest-to-oldest 到模型规范顺序
[ ] request row_indices 一致
[ ] label float32 与 task 顺序一致
```

### E.4 模型层

```text
[ ] resolved scene 过滤与实际 tokenizer input names 分模型核对
[ ] RankMixer concat width 可被 token count 整除
[ ] OneTrans S/NS count 正确
[ ] LONGER global token 数求和正确
[ ] mean_pool empty history 为零
[ ] STCA empty history 无 NaN
[ ] STCA grouped history 只输出一个 z
[ ] scenario/task token name 与 runtime scenario/task 对齐
[ ] fixed/fine scenario 行为分别测试
```

### E.5 实验层

```text
[ ] pure baseline 是否显式包含 scene 已明确
[ ] main/prior embedding 是否共享已明确
[ ] global scenario prior 是否允许 candidate-aware 已明确
[ ] dual-route history 的消融已定义
[ ] STCA 与 LONGER 使用可比长度/预算
[ ] STCA curriculum 是否启用单独记录
```

---

## 结语

当前字段系统的核心不是“168 个字段加 9 条序列”这么简单，而是：

```text
物理列
  -> request/candidate/event 三轴归一化
  -> categorical/dense 语义
  -> scalar/bag/sequence 形态
  -> embedding 参数共享或独立
  -> feature/scenario/task scope
  -> LONGER/raw/mean_pool/STCA
  -> RankMixer/OneTrans/MDL token
```

当前最需要团队统一认识的三件事是：

1. 主非序列字段全部是 categorical，连续信息只有 sequence time delta；
2. 相同物理 history 在主路、scenario prior、task prior 中可能拥有完全独立的表示参数；
3. scene 策略：request 轴 omit 出 pack；candidate×scene 留在 item；RankMixer LONGER 仍吃 `scene_id_hn`；OneTrans NS 不再吃 request scene；MDL 另叠加 scenario token / prior / DomainFused。`mdl_feature_interaction` 默认 `residual_ffn`。

这三点决定了后续 STCA/LONGER 对比、scene bias 讨论、prior 复用设计和性能优化应该怎样解释与消融。
