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

例子：`goods_id` shared union ≈ **268M×32**；`cart_long.sku_ids` ≈ **537M×32**。

## 怎么放下

- **Sharded embedding** + **Row-Wise Adagrad**（state 从 \(O(ND)\) → \(O(N)\)）
- 报的规划显存 = 权重 + sparse state，**不含**激活 / cache 峰值

## 表之后

| 模型 | embedding 之后 |
|---|---|
| `mdl_rankmixer` | groupwise → **32×768** Feature token |
| `mdl_onetrans` | 历史 → 变长 **S**；非序列 → **32 NS**（256） |

两者 Domain 都用独立 scope 的 important/prior 初始化。
