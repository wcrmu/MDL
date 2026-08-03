# MDL 词表与 Embedding 设计

> 单独说明当前生产配置里：ID 怎么进表、桶和维度怎么定、哪些表共享/隔离、显存如何收住。  
> 容量审计细节见 [`embedding_shape_24h_audit.md`](./embedding_shape_24h_audit.md)；Domain 字段选型见 [`mdl_token_feature_design.md`](./mdl_token_feature_design.md)；离线数据口径见 [`mdl_offline_data_intro.md`](./mdl_offline_data_intro.md)。

## 0. 一句话结论

我们不在训练侧维护传统“字符串 → 连续 vocab id”词典，而是以 **上游预哈希 ID + 可训练 hash embedding 表** 为主；再用 **scope（feature / scenario / task）** 和 **shared union** 决定哪些逻辑字段共用一张物理表。桶大小与维度由 **24 小时训练尺度上的 projected distinct + 显存预算** 共同决定，而不是追求零碰撞。

当前 coarse MDL 量级：约 **290** 张物理表，BF16 权重 + Row-Wise Adagrad 规划约 **65～66 GiB/GPU（2 卡）**。

---

## 1. 词表是什么形态

### 1.1 不是经典 vocab 文件

| 方式 | 我们是否采用 | 说明 |
|---|---|---|
| 离线枚举全量 string vocab | 否 | goods/sku/query 长尾随文件数持续增长，无法闭集 |
| 上游预哈希 + 训练侧再取模 | **是（主路径）** | 字段多为 `*_hn` / `*_hn_share` |
| 小集合 identity ID | 是（少数） | 如 `coarse_scene_prior_id` |
| 连续 dense 数值直接进网 | 极少 | 主特征几乎全是 categorical；真正连续的主要是 9 路时间差 |

### 1.2 两种 encoding

**`pre_hashed`（绝大多数）**

```text
上游已给出 hash / 桶化 ID
  → null 单独处理
  → 合法值映射到 embedding row（形如 (id & (num_buckets-1)) + 1）
  → row 0 通常留给 padding / 缺失
```

注意：上游值 **`0` 常常是合法哈希**，不能当成 padding。

**`identity`（少数派生 ID）**

```text
adapter 已生成落在小范围内的整数（如 coarse_scene_prior_id ∈ {0,1,2}）
  → 不再二次 hash
  → 越界 / null 按 identity 规则处理
```

### 1.3 逻辑名 vs 物理表

配置里一个逻辑字段包含：

- `name`：模型引用名（可带 `scenario_important_` / `task_important_` 前缀）
- `source`：物理或 adapter 派生列
- `embedding_scope`：`feature` / `scenario` / `task` / `shared`
- `num_buckets` / `embedding_dim`
- 可选 `share_with`：显式挂到已有物理表

**同 source ≠ 同向量。** 除非显式共享，scenario/task important 各自独立查表。

---

## 2. Embedding 表怎么组织

### 2.1 Scope 隔离（MDL 硬约束）

论文要求 Domain 的 important 使用 **extra embeddings**。落地规则：

| 消费者 | 表策略 |
|---|---|
| 普通 Feature / 主历史 | feature（或 shared root）表 |
| Scenario important | **scenario-scope** 独立表；search/recommendation/global **共用**同一套 scenario important 表 |
| Task important | **task-scope** 独立表；同名字段在用到它的 task 间共用 |
| Scenario / Task history prior | 独立 encoder；表常独立，尺寸跟同 raw 字段的生产 bucket/dim，不偷偷放大 |

这样 Domain Query 的梯度不会直接污染主干 Feature 表，也避免“全复制 3×scene × 全字段”。

### 2.2 Shared root / union 定桶

同语义 ID 可出现在候选特征与多条 UPS 历史中。定桶时必须按 **`shared_embedding_groups` 的来源并集** 估 distinct，不能只看 root 单列。

典型例子：`goods_id_hn`

- 100 文件 union distinct ≈ 31.8M，不是候选列单独统计；
- 外推到 24h 后生产表约 **268.4M × 32**；
- 主 Feature 与多历史可共享这张大表；
- Task 侧若要精确商品身份，用 **独立小 dim 的 task-extra 表**，禁止再复制一整份 268M 主表。

### 2.3 一张表可以被几条路径消费

同一物理 embedding 仍可能进入不同 projector / token：

```text
物理表 goods_id
  ├─ RankMixer item_goods_identity token
  ├─ 某条 LONGER / S 历史事件字段
  └─（若 share）其它 alias
```

分析“是否重复编码”时要分四层：**physical payload → embedding table → projector → token consumer**。

---

## 3. 桶大小与维度怎么定

### 3.1 规划尺度

| 尺度 | 文件量 | 用途 |
|---|---:|---|
| 小 profile | 10 | 增长趋势 |
| 大 profile | 100 | 增长趋势 |
| **生产主尺度** | **≈12,000（24h × 500 files/hour）** | 定 `num_buckets` / `embedding_dim` |
| 压力尺度 | ≈84,000（7 天） | 只报告风险，不直接写进配置 |

只用单小时或 10 文件定桶，会系统性低估长尾，把碰撞推迟到长跑才爆发。

### 3.2 生产公式（摘要）

```text
用 10→100 文件拟合增长指数 alpha
外推 d_12000

ideal_bucket  = next_pow2(d_12000 / 0.5)     # 低碰撞理想值（常不可部署）
deploy_bucket = max(
    next_pow2(d_500 / 0.5),                  # 至少覆盖 1 小时
    next_pow2(d_12000 / 1.75),               # 24h，允许更高 load
)
deploy_bucket ≤ 2^30

若 projected distinct > 1e6：生产 dim 从理想 64 封顶到 32
```

完整逐字段结果：`docs/emb_bucket_recommendation_growth.json`；审计说明：[`embedding_shape_24h_audit.md`](./embedding_shape_24h_audit.md)。

### 3.3 为什么接受碰撞

若坚持 24h `load≤0.5` 且高基数仍用 64 维：

- coarse MDL ≈ **399 GiB/GPU（2 卡）**，不可训；
- 即使 4 卡仍约 **200 GiB/GPU**。

因此生产明确折中：

- 大表加桶、降维（多为 32）；
- 接受均匀哈希假设下数十个百分点的理论碰撞（如 `goods_id` ~41%）；
- 若碰撞伤效果，优先考虑 **高频 vocab + 长尾 hash**、remixed pre-hash 或 parameter server，而不是在单机 GPU 上无约束扩表。

### 3.4 若干生产 shape 例子

| 物理表 | 生产 shape（约） | 备注 |
|---|---:|---|
| `goods_id_hn` shared union | 268.4M × 32 | 按多来源 union |
| `cart_long.sku_ids_hn` | 536.9M × 32 | 历史 SKU 长尾极重 |
| `sku_id_hn` | 134.2M × 32 | |
| `mall_id_hn` shared union | 4.2M × 32 | |
| `page_sn_hn` shared union | 4,096 × 24 | 小表可保留更合适 dim |

---

## 4. 优化器与分布式放表

大表无法每卡复制完整副本，当前默认：

| 机制 | 作用 |
|---|---|
| **Sharded embedding** | 每卡只持有 owned rows；lookup/grad 走 all-to-all |
| **Row-Wise Adagrad** | 每行一个 FP32 accumulator，state 从 \(O(ND)\) 降到 \(O(N)\) |
| Dense 参数 | DDP 同步（RankMixer/OneTrans 主干、Domain FFN 等） |
| 权重 dtype | 生产 embedding 常用 **bf16** |

规划显存报的是 **embedding 权重 + sparse optimizer state**，不是完整训练峰值（激活、cache、fragmentation 另计）。

---

## 5. 进模型之后：表 → token

词表设计停在 embedding 向量；tokenizer 决定怎么拼：

| 模型 | Embedding 之后 |
|---|---|
| `mdl_rankmixer` | groupwise 语义组投影到 **32×768** Feature token；历史多为 LONGER summary |
| `mdl_onetrans` | 历史 raw event → 变长 **S**；非序列特征经 DCNv2 → **32 NS**（宽 256） |
| 两者的 Domain | Scenario/Task 用 **独立 scope 表** 的 important/prior 初始化，再做 Domain-aware Attention |

因此：

- 改某个字段的 `embedding_dim`，在 RankMixer groupwise 下主要改变该组 `Linear` 的输入宽，**不应**让相邻 token 语义漂移；
- OneTrans 的 NS 是 latent slot，与 RankMixer 的 32 个语义 token **同数不同义**。

---

## 6. 设计原则（决策时怎么选）

1. **先定 24h 可训，再谈低碰撞。** 不可部署的理想 shape 只留在报告里。  
2. **shared 看 union，独立看角色。** 同名 raw 在 Domain important/prior 上默认不共享主表。  
3. **高频锚点可升维/独立；超长尾优先共享大 hash 表。**  
4. **Task 精确 ID 用小 extra 表**，禁止复制主 `goods` 巨表。  
5. **dim 最终由效果实验定。** 32 是预算起点，不是理论最优；应用 32/48/64 对照 AUC/吞吐。  
6. **改 shape 必须连带改 shared alias / prior 副本**，并重跑显存规划。

---

## 7. 相关入口

| 内容 | 位置 |
|---|---|
| 24h 容量审计 | [`embedding_shape_24h_audit.md`](./embedding_shape_24h_audit.md) |
| 逐字段推荐 JSON | `docs/emb_bucket_recommendation_growth.json` |
| 推荐脚本 | `scripts/recommend_embedding_shapes.py` |
| Domain 字段与 scope | [`mdl_token_feature_design.md`](./mdl_token_feature_design.md) |
| 配置生成 | `scripts/build_mdl_rankmixer_config.py` |
| 生产 YAML | `configs/mdl_rankmixer.yaml`、`configs/mdl_onetrans.yaml` 等 |
| Sharded / sparse 测试 | `tests/test_sharded_embedding.py`、`tests/test_sparse_ddp.py` |
