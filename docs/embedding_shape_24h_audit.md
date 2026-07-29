# 24 小时 / 12,000 文件 Embedding Shape 审计

## 结论

`profile.json`（10 文件）和 `new_profile.json`（100 文件）足以给出逐字段增长量级，
但不能单独证明效果最优的 embedding dim。线上每小时约 500 文件，而一次训练覆盖
24 小时，因此生产主尺度应为 **12,000 文件**，不能只按 500 文件定值。

直接追求 24 小时窗口 `distinct / buckets <= 0.5` 并保留理想 dim，会把
coarse MDL embedding 推到约 **399.49 GiB/GPU（2 卡）**，即使使用 4 卡也
需要约 **199.74 GiB/GPU**，无法训练。本次采用显存约束方案：

- 24 小时 projected hash load 上限取 1.75，并向上取 2 次幂；
- 同时保留 1 小时 / 500 文件、load 0.5 的 bucket 作为下限；
- projected distinct 超过 100 万时，生产 dim 上限从理想档位 64 调为 32，
  用表示宽度换 hash 容量；
- shared root 必须使用 `shared_embedding_groups` 的来源并集，不使用 root 单列；
- 生产 bucket 硬上限为 2^30；超过预算的“理想 shape”仍在报告中保留，不伪装成
  已达到低碰撞率。

结果已落到 8 份生产 YAML。coarse MDL 的 BF16 embedding + Row-Wise Adagrad
规划为 **65.01 GiB/GPU**，低于内部 68 GiB/GPU embedding 规划线，给 dense
参数、激活和运行时保留约 15 GiB/GPU；这部分余量不是完整训练峰值的保证，仍需
用真实 batch 做 CUDA peak-memory 验证。

## 生成方式

逐字段证据、理想 shape 和实际生产 shape 全部在
`docs/emb_bucket_recommendation_growth.json`：

```bash
python scripts/recommend_embedding_shapes.py \
  --small-profile docs/profile.json \
  --large-profile docs/new_profile.json \
  --config configs/mdl_rankmixer.yaml \
  --files-per-hour 500 \
  --target-hours 24 \
  --stress-hours 168
```

`new_profile.json` 前 106 行是平台启动/进度日志，最后一行才是合法 JSON。生成器
会读取最后一个合法 JSON object，不会把日志当作 profile，也不会改写采样文件。

## 数据与估计口径

| 项目 | 小 profile | 大 profile | 生产主尺度 | 压力尺度 |
|---|---:|---:|---:|---:|
| Parquet 文件 | 10 | 100 | 12,000（24 小时） | 84,000（7 天） |
| agg rows | 18,720 | 187,190 | 按字段增长率外推 | 只报告，不进配置 |
| 采样关系 | 同一分区前缀 | 前 10 文件与小 profile 完全一致 | 500 文件/小时 | 检查长期风险 |

逐字段拟合：

```text
alpha = clamp(log(d100 / d10) / log(100 / 10), 0, 0.95)
d12000 = d100 * (12000 / 100) ^ alpha

ideal_bucket  = next_pow2(d12000 / 0.5)
deploy_bucket = max(
    next_pow2(d500 / 0.5),
    next_pow2(d12000 / 1.75),
)
```

低基数饱和字段保留 25% headroom；增长很慢但基数较高的字段至少保留 50%
headroom。报告中的 `ideal_num_buckets` / `ideal_embedding_dim` 是不考虑两卡
显存时的参考，`num_buckets` / `embedding_dim` 才是当前生产配置。

## 关键字段

下表的“旧 shape”是改动前生产配置；“理想 shape”是 24 小时 load 0.5 +
未压缩 dim 的结果。

| 物理表 / source | 旧 shape | 24h 生产 shape | 24h 理想 shape | projected distinct | 生产 load |
|---|---:|---:|---:|---:|---:|
| `goods_id_hn` shared union | 33.6M × 64 | **268.4M × 32** | 1,073.7M × 64 | 315.77M | 1.176 |
| `cart_long.sku_ids_hn` | 134.2M × 64 | **536.9M × 32** | 2,147.5M × 64 | 712.88M | 1.328 |
| `sku_id_hn` | 67.1M × 64 | **134.2M × 32** | 536.9M × 64 | 225.25M | 1.678 |
| `i2i_list_multimodal_hn_share` | 134.2M × 64 | **268.4M × 32** | 536.9M × 64 | 263.12M | 0.980 |
| `ups_in_cart_goods_hn_share` | 16.8M × 64 | **67.1M × 32** | 268.4M × 64 | 112.13M | 1.671 |
| `idx_goods_creative_id_hn` | 67.1M × 64 | **33.6M × 32** | 134.2M × 64 | 57.24M | 1.706 |
| `cart_long.spec_hn` | 67.1M × 64 | **67.1M × 32** | 268.4M × 64 | 92.72M | 1.382 |
| `main_goods_ids_hn_share` | 16.8M × 64 | **16.8M × 32** | 67.1M × 64 | 18.69M | 1.114 |
| `offline_outside_goods_id_list_hn_share` | 16.8M × 64 | **33.6M × 32** | 134.2M × 64 | 34.19M | 1.019 |
| `sess_q2q_hash_list_hn` | 8.4M × 64 | **8.4M × 32** | 33.6M × 64 | 9.45M | 1.126 |
| `mall_id_hn` shared union | 2.1M × 48 | **4.2M × 32** | 4.2M × 64 | 1.47M | 0.350 |
| `page_sn_hn` shared union | 512 × 16 | **4,096 × 24** | 4,096 × 24 | 1,966 | 0.480 |

`goods_id_hn` 必须按 9 个共享来源的 union 估算；100 文件 union distinct 是
31.77M，而不是候选 root 单列的值。`page_sn_hn`、`mall_id_hn` 和 category
roots 同样按 union 处理。

## 配置与显存变化

相对改动前的 coarse MDL，286 张物理表中有 68 张 shape 改变：

- 19 张涨 bucket；
- 9 张缩 bucket；
- 40 张 bucket 不变、主要将高基数 dim 64 调为 32；
- shared alias 只跟随物理 root，不重复计数。

生产设置为两卡、BF16 embedding、Row-Wise Adagrad：

| 配置族 | 改动前 | 24h 新值 |
|---|---:|---:|
| coarse/fine MDL | 68.05 GiB/GPU | **65.01 GiB/GPU** |
| 非 MDL | 54.14 GiB/GPU | **57.86 GiB/GPU** |
| 直接应用 24h 理想 shape 的 coarse MDL | — | **399.49 GiB/GPU（不可行）** |

如果改为 4×80 GiB H100 并进行四路 embedding shard：

| 4 卡方案 | Embedding 权重 + Row-Wise Adagrad / GPU | 结论 |
|---|---:|---|
| 当前 24h bucket + 32 dim | **32.50 GiB** | 容量充足 |
| 当前 24h bucket + 理想 dim（高基数恢复 64） | **62.94 GiB** | 数学上可容纳，需实测完整训练峰值 |
| 24h load 0.5 bucket + 理想 dim | **199.74 GiB** | 4 卡仍不可行 |

MDL 总量下降并不表示 bucket 变小：核心 ID/历史表大幅增桶，但大量独立 prior
和其它高基数表从 64 维改为 32 维，抵消了 bucket 增量。独立 task/scenario
prior 仍受各自的 bucket cap 约束，否则同一 namespace 的多份物理副本会耗尽
显存。

## 覆盖与限制

- profile 共 277 个字段；
- 248 个字段/shared root 生成了生产 shape；
- 19 个 `distinct <= 1` 字段建议排除；
- 10 个 time-delta / coarse-scene identity 字段不是 hashed embedding；
- 配置内其余 245 个 categorical source 均有推荐；
- 10 个 shared root 全部按 union 估算。

当前生产 load 不是“零碰撞”。例如均匀哈希假设下，`goods_id_hn` 的理论
distinct collision 约 41.2%，`sku_id_hn` 约 51.5%，
`cart_long.sku_ids_hn` 约 44.7%。这是两卡显存下的明确折中；若这些碰撞影响
AUC/GAUC，优先考虑高频 vocab + 长尾 hash、remixed pre-hash 或 embedding
parameter server，而不是继续在 GPU 上无约束扩表。

另外，10/100 文件来自同一小时前缀，向 24 小时外推默认增长指数跨小时稳定。
这是目前最可复现的保守估计，不等于真实 24 小时 union。上线前仍应补一份跨小时
的 500/12,000 文件 profile；若实际 overlap 更高，当前方案会偏保守，若存在
明显小时级 namespace 漂移，则可能继续低估。

Embedding dim 不能由 cardinality 唯一确定。32 维是预算约束下的生产起点，
最终应以 32/48/64 dim 的离线 AUC/GAUC、吞吐和线上实验决定。
