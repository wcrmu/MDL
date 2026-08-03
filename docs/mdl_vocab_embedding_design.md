# MDL 词表与 Embedding 设计

> 细节审计见 [`embedding_shape_24h_audit.md`](./embedding_shape_24h_audit.md)；Domain 字段见 [`mdl_token_feature_design.md`](./mdl_token_feature_design.md)。

## 结论

- **没有闭集 vocab 文件**：主路径是上游预哈希 ID → `pre_hashed` embedding；少数派生 ID（如 `coarse_scene_prior_id`）走 `identity`。
- **`0` 常是合法哈希**，不是 padding；null 才是缺失。
- 桶 / dim 按 **24h ≈ 12,000 文件** 外推定，不是按 10 个抽样文件。
- 生产目标是 **能训**，不是零碰撞：约 **290** 张物理表，BF16 + Row-Wise Adagrad ≈ **65～66 GiB/GPU（2 卡）**。理想低碰撞要 ~399 GiB/GPU，不可部署。

## 编码与表

| 点 | 做法 |
|---|---|
| 主特征 | `pre_hashed`：`(id & (buckets-1)) + 1`，row 0 留给 pad/缺失 |
| Domain important | **独立 scope 表**（scenario / task）；与主 Feature 表不共享梯度 |
| 同语义 ID（如 goods） | 主特征 + 多历史按 **union** 定一张大表；Task 精确身份用**小 extra 表**，禁止再复制 268M 主表 |
| 同 source | ≠ 同向量；无 `share_with` 则各查各的表 |

## 定桶规则（要点）

```text
尺度：24h × ~500 files/hour ≈ 12,000 文件
deploy_bucket ≈ max(1h@load0.5, 24h@load1.75)，≤ 2^30
高基数（distinct > 1e6）：dim 封顶 32（理想 64 常放不下）
```

例子（生产 shape，BF16）：

| 物理表 | shape |
|---|---|
| `goods_id_hn` shared union | ≈ **268M × 32** |
| `cart_long.sku_ids_hn` | ≈ **537M × 32** |
| `sku_id_hn` | **134.2M × 32** |
| `mall_id_hn` | **4.2M × 32** |
| `page_sn_hn` | **4,096 × 24** |

## 怎么放下

大表无法在每张 GPU 上完整复制。静态 embedding 走 **分片 + 行式稀疏优化器**；RankMixer / OneTrans 主干、Domain FFN 等 dense 参数仍由 **DDP** 同步。下表只覆盖「权重 + sparse optimizer state」的规划占用，**不含**激活、cache 与临时通信峰值。

| 机制 | 作用 |
|---|---|
| Sharded embedding | 每卡只持有 owned rows；lookup / grad 走 all-to-all |
| Row-Wise Adagrad | 每一行一个 FP32 accumulator，state 从 \(O(ND)\) 降到 \(O(N)\) |
| Dense 参数 | DDP 同步（RankMixer / OneTrans 主干、Domain FFN 等） |
| 权重 dtype | 生产 embedding 常用 **bf16** |

粗粒度 MDL 约 **290** 张物理表，两卡规划约 **65～66 GiB/GPU**；理想低碰撞 shape（~399 GiB/GPU）不可部署。

## 表之后

Embedding lookup 之后，两模型**共用同一套表与 FeatureBatch 合同**，但消费路径分叉：RankMixer 把字段收成固定 Feature token；OneTrans 把历史留在变长 S、把非序列收成 NS。Domain（Scenario / Task）不混进这两条主干，而是用**独立 scope 表**初始化后再做 Domain-aware Attention。

| 模型 | Embedding 之后 |
|---|---|
| `mdl_rankmixer` | groupwise 语义组投影到 **32×768** Feature token；历史多为 **LONGER summary** |
| `mdl_onetrans` | 历史 raw event → 变长 **S**；非序列特征经 DCNv2 → **32 NS**（宽 256） |
| 两者的 Domain | Scenario / Task 用**独立表**的 important / prior 初始化，再做 Domain-aware Attention |
