# MDL scenario/task token 字段设计

本文记录当前生产配置的算法契约。字段选择依据是
`paper/MDL/main.tex` 的 tokenization / Domain-aware Attention 定义，以及
`docs/profile.json` 的 10 个 Parquet、18,720 request、约 170,792 candidate
观测。它不是论文未公开字段表的“复原”，而是对论文结构约束的业务化实现。

## 1. 原文给出的硬约束

MDL 原文只规定结构，没有规定具体字段名：

1. scenario token 由两部分构成：
   - important raw features 的 **extra embeddings**；
   - scenario-related prior embeddings，例如场景相关用户行为序列。
2. task token 使用 **other important features** 和 task-related features。
3. 每个 scenario/task token 使用不同参数的 PerToken FFN。
4. scenario/task token 是 Domain-aware Attention 的 query；feature tokens 是
   key/value。
5. global scenario token 用于学习跨场景 common knowledge。

因此：

- important 是“当前样本的强锚点”，不等于 scene 字段全集；
- prior 是“该分布下的证据”，不是另一份无条件 feature pack；
- important embedding 必须与普通 feature embedding 分开；
- 原文没有要求 important embedding 再按每个场景复制一遍。当前是一套
  scenario-scope extra table 供 scenario tokens 使用，token 差异来自字段组合、
  prior 和 PerToken FFN；
- item 信息可以进入 important。论文明确举了 video id，但生产字段必须考虑
  频次与训练周期。

## 2. 从 profile 推出的取舍

| 字段 | distinct（样本） | top1 约占比 | 结论 |
|---|---:|---:|---|
| `goods_id_hn` | 144,708 | <1% | 单 epoch 下独立 extra 表更新过稀，不作为默认 important |
| `cat1_id_hn` | 23 | 48% | 可学习的候选语义锚 |
| `cat2_id_hn` | 266 | 34% | task query 的细一级类目锚 |
| `price_hn` | 20 | 8% | 高支持度候选价格带 |
| `scene_id_hn` | 104 | 31% | concrete scenario 的细场景身份 |
| `page_sn_hn` | 85 | 32% | 页面/入口上下文，也存在于历史事件中 |
| `search_method_hn` | 18 | 47% | search-only 请求意图 |
| `adj_cartcvr_hn` | 20 | 81% | 加购任务统计锚，虽偏斜但非常数 |
| `adj_cvr_hn` | 21 | 10% | 支付任务统计锚 |
| `cart_cnt_3d_hn` | 20 | 8% | 比 `is_promotion` 更直接的加购强度 |
| `rel_level_hn` / `rel_score_hn` | 5 / 18 | 60% / 72% | filter/relevance 任务锚 |

`goods_id_hn` 仍保留在普通 feature tokens 和各历史中。这里不复制 task/scenario
goods extra table，是针对“一轮训练 + 低重复 ID”的样本效率取舍，不是否定 item
identity 的价值。更长训练窗口下应重新做 `goods important on/off` 消融。

## 3. 当前 scenario token

### 3.1 search

important：

```text
currency_hn
hash_language_site_hn
plat_hn
region_hn
page_elsn_hn
page_sn_hn
scene_id_hn
search_method_hn
cat1_id_hn
price_hn
```

prior：

```text
scenario_search_prior_coarse_scene
scenario_prior_scene_impr_cnt_15d_hit_hn
scenario_prior_scene_impr_cnt_15d_hn
scenario_conditioned_clk_long_prior
```

### 3.2 recommendation

important 与 search 相比去掉 `search_method_hn`，其余相同。prior 使用
`scenario_recommendation_prior_coarse_scene`，其余三项与 search 共用。

### 3.3 global

important：

```text
currency_hn
hash_language_site_hn
plat_hn
region_hn
cat1_id_hn
price_hn
```

prior：

```text
scenario_global_impr_prior
scenario_global_clk_long_prior
scenario_global_view_long_prior
```

global 刻意不读 `scene_id/page/search_method`，避免 common token 与 concrete
token 共线；但保留类目/价格，因为 global 表示仍然是“当前样本上的跨场景共性”，
不是与样本无关的常数向量。

### 3.4 为什么 scene 统计放 prior

`scene_impr_cnt_15d_*` 是行为/分布统计，不是稳定身份锚，所以使用独立
scenario-scope embedding，但进入 `prior_inputs`。`scene_id_hn` 是身份字段，
进入 concrete important；global 不使用它。

`scenario_conditioned_clk_long_prior` 不再为 search/recommendation 各复制一条
内容完全相同的 mean pool。它用：

```text
query = extra_embedding(scene_id_hn, page_sn_hn)
key/value = clk_long events
```

做 `attention_pool`。这使历史摘要随当前 scene/page 变化。search 与
recommendation 的进一步差异由 coarse prior、important 组合和 PerToken FFN
提供。当前历史事件没有逐事件 `scene_id`，所以这里准确名称是
“scene-conditioned history”，不是声称原始数据已经提供严格 per-scene history。

## 4. 当前 task token

| task | important raw fields | task prior |
|---|---|---|
| `fst_cart` | `cat1_id_hn`, `cat2_id_hn`, `price_hn`, `adj_cartcvr_hn`, `cart_cnt_3d_hn`, `cat_id_hn`, `goods_cluster_id_1w_hn`, `mall_id_hn`, `goods_id_hn` | `cart_long` attention pool |
| `upid_pay` | `cat1_id_hn`, `cat2_id_hn`, `price_hn`, `sales_hn`, `adj_cvr_hn`, `cat_id_hn`, `goods_cluster_id_1w_hn`, `mall_id_hn`, `goods_id_hn` | `buy_long` attention pool |
| `cateid_filter` | `cat1_id_hn`, `cat2_id_hn`, `cat_id_hn`, `rel_level_hn`, `rel_score_hn`, `origin_query_hash_hn` | `srch_q2i` attention pool |

三条 task prior 都以本 task important embeddings 为 query。相比无条件 mean
pool，它们回答“与当前候选/任务条件相关的历史是什么”。`cateid_filter` 的标签
是 `cateid_is_fst_scene_sp_filter`，默认使用搜索 query-to-item 历史，而不再复用
购买历史；这是按字段语义做出的推断，必须通过固定 holdout 消融验证。

表中写的是 raw source；YAML 中对应 logical name 带
`task_important_` 前缀，并使用独立 task-scope embedding。

task important 不再复制 locale；在保留高支持度类目、价格和任务统计的基础上，
分两层加入 identity：

- locale 已由 active scenario token 经 DomainFused 注入；
- `fst_cart` / `upid_pay` 使用叶子类目、商品簇和商家作为低风险 identity，
  并使用独立的 32M×32 `goods_id` task-extra 表作为论文同型的精确商品身份；
- `cateid_filter` 已有叶子 `cat_id`，额外复用物理存在的
  `origin_query_hash_hn` bag 作为可部署 query identity；若上游补充 scalar
  current-query ID，应优先用 scalar 替换该 bag；
- 所有 identity extra embeddings 均与主 feature/history 表独立，避免把主
  `goods_id` union 的 268M 表复制进 task scope；
- 各 task 的字段组合、prior 和 PerToken FFN 都不同，避免三个 task token
  仅靠同一套输入硬分化。

## 5. embedding 与参数边界

- 普通 feature、scenario important、task important 是不同物理 embedding 表；
- 一项 `scenario_important_*` 由 search/recommendation/global 共用，不是
  `3 × scenario` 表；
- task important 同名字段在使用它的 task tokens 间共用一张 task-scope 表；
- task/scenario history prior 保留独立参数与独立 encoder；
- scenario history 的独立表尺寸跟随 profile 后 backbone 同 raw 字段的
  bucket/dim，独立不再隐含“回退到更大的估计维度”。

当前 coarse MDL 配置约 290 张物理表，规划约 66.24 GiB/GPU
（BF16 + Row-Wise Adagrad，2 GPU）；fine 配置约 288 张。

## 6. RankMixer feature token 边界

MDL 原文不是“把所有 embedding 拼起来后任意等宽切片”，而是明确要求用领域
知识构造 semantically coherent clusters，再逐组投影。当前 RankMixer 使用固定
`T=32, D=768`：

```text
23 个普通语义 token:
  request_environment
  request_page_scene
  query_identity
  query_semantic_expansion
  query_recall_context
  user_commerce_history
  user_click_view_history
  user_query_exposure_profile
  item_category_hierarchy
  item_goods_identity
  item_text_content
  item_sku_specification
  item_supply_quality
  sku_commerce
  item_price_promotion
  item_sales_value
  item_conversion_statistics
  item_scene_statistics
  item_temporal_commerce
  retrieval_i2i_candidates
  retrieval_hit_evidence
  query_item_relevance
  creative_recent_behavior

9 个主历史 token:
  impr / clk_long / view_long / cart_long / buy_long
  semi_clk / srch_q2i / ups_clk_sku / flatten_query_hash
```

每组拥有自己的 `Linear(group_width, 768)`，因此字段增删或 embedding dim 调整
不会悄悄改变相邻 token 的语义边界。投影参数量与对同一总宽度做切片投影基本
同阶；变化的是归纳偏置，而不是简单扩参。

这里的 `T=32,D=768` 是当前生产容量契约。RankMixer 原文的 scaling 表同时给出
`(D=768,T=16,L=2)` 的 100M 档和 `(D=1536,T=32,L=2)` 的 1B 档，并没有规定
MDL 只能取 16。我们保留既有 32-token 容量，只把 raw-width 切片替换为论文要求
的语义边界，避免把“特征工程改造”和“模型缩容”混成一个实验。

这同时避免了三个旧问题：

1. 为追求 raw width 整除而静默把 32 token 降成 8；
2. 小幅调整某字段 embedding dim 后，后续大量字段跨 token 漂移；
3. 把多个完整历史 summary 混入同一个 flat slice。

`RankMixerSliceTokenizer` 的 zero-padding 仍保留为兼容 fallback，但生产 LONGER
配置走 `groupwise`。LONGER 的 `d=32` 继续独立于 RankMixer 的 `D=768`。

## 7. Domain token 的两种状态传播

字段合同在两种模式下完全相同；important/prior encoder、独立 embedding、
scenario mask、PerToken FFN 和 32×768 feature-token 容量都不改变。唯一变化是
important/prior projector 的输出能否沿 residual 直接进入最终 readout。

### 7.1 `coupled`（默认，原始 MDL）

```yaml
model:
  mdl_token_state: coupled
```

该模式忠实保留 `paper/MDL/main.tex` 的 domain-token 状态传播：

```text
important + prior -> initial domain token
initial/current token -> attention Query
current token + attention update -> residual state
final task token -> logits
```

因此同一个 token 同时是 prompt/query、recurrent state 和 readout。即使 attention
update 被置零，important/prior 仍可通过 token residual 到达 logits；这是论文
公式本身允许的路径，不是旧实现接错。

这里的“忠实”特指 scenario/task tokenization、Domain-aware Attention residual、
DomainFused 和 final task-token readout。生产配置的
`mdl_feature_interaction: residual_ffn` 是此前选定的 RankMixer 稳定性变体；
严格复现论文 feature Eq. 6 还需使用 `direct_ffn`。`mdl_onetrans` 仍是实验组合，
不是论文公开模型。

### 7.2 `split`（消除 prior→readout residual bypass）

```yaml
model:
  mdl_token_state: split
```

或临时覆盖：

```bash
python -m src.main train --config configs/mdl_rankmixer.yaml \
  --mdl-token-state split \
  --train-start-hour 2026-07-22-22 --train-end-hour 2026-07-29-22
```

该模式保留原 important/prior projector 作为静态、样本相关的 domain prompt，
并为每个 scenario/task slot 新增一个样本无关的 learned readout seed：

```text
prompt p = Projector(important, prior)
state  h0 = learned readout seed
query  ql = p + hl
update Δl = DomainAwareAttention(q=ql, K/V=feature tokens)
state  h(l+1) = residual/FFN(h(l) + Δl)
logits = task head(hL)
```

prompt 不与 state concat，也不 residual-copy 到 state。Scenario→task 的
DomainFused 只融合 scenario evidence/readout state，不融合 scenario prompt。
对实验性的 MDL-OneTrans，prompt 可调制 NS/S attention 与 gate，但只有读取到的
Value update 能写入 state。

该模式**不会**把 prior 搬到 32 个 feature K/V 槽，也不会删除 prior。它只改变
prior summary 的消费者：从“Query + residual/readout 内容”改为“Query-only
routing control”。硬不变量是：

```text
固定 readout seed/state，若所有 domain attention Value update = 0，
替换 important/prior prompt 不得改变 final readout/logit。
```

为保证这条不变量，`split` 要求启用对应 Domain-aware Attention，且要求
`scene_feature_bias: none`；RankMixer domain-interaction fallback 和
additive/FiLM scene bias 会重新引入 prompt→content/readout 路径，配置校验会拒绝。

`split` 比 `coupled` 多出少量 readout seed 参数，二者 checkpoint 参数面不同；
应分别启动训练，不要把一种模式的 checkpoint 当作另一种模式的等价续训。

## 8. 必须配套的验证

结构修正不等于 AUC 必然提升。至少做以下单因子实验：

1. 旧 token 字段/mean pool vs 本文方案；
2. `goods_id` important off vs on（仅在更长窗口下）；
3. `cateid_filter`: `srch_q2i` vs `buy_long`;
4. shared conditioned click prior vs 真正由上游产出的 per-scene history；
5. `scene_feature_bias`: `none` vs `additive` vs `film`，默认仍为 `none`。

所有实验固定数据窗口、seed 和 holdout，按 task × scene 报告：

当前生产两卡配置从训练结束后的下一自然日 24 小时窗口中均匀冻结
`25 files/rank`，即总计 50 个 Parquet 文件；每 5000 step 重复评估同一 manifest。

```text
AUC, COPC, BCE, prob_mean, logit_mean/std, positives, examples
```

并同时记录 attention entropy 或 top-k mass；否则无法判断 uplift 是更好的特征
选择，还是 token/query 塌缩。
