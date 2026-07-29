# Embedding bucket / dim 五档对照

对照基准：

- **当前配置**：`configs/mdl_rankmixer.yaml`（与 baseline-fine 一致）
- **抽样 profile**：`profile_prehashed_parquet.py`，`pt=2026-07-21/hr=00`，`--max-files 10`，约 **18.7k** agg 行
- **覆盖范围（重要）**：本文 **不是全量审查**。平台 stdout 在 `c_cart_cnt_15d_hn` 后被截断，下列五档表仅含完整解析到的 **25** 个字段（大致 a–c 前缀）。

### 配置规模 vs 本文覆盖

| 范围 | 配置内数量 | 本文有抽样对照 | 缺口 |
|------|------------|----------------|------|
| 非序列 categorical（均为物理表） | **179** | ~12 | **~167** |
| 序列 categorical（含 share alias） | **133**（物理约 25 + alias 约 108） | ~13 | **大部分未对照** |
| 序列条数 | 12 | — | — |

要对齐全部字段，需要：

1. 把 `profile_prehashed_parquet.py` 的完整 JSON **写到文件**（勿只依赖被截断的日志），例如 `--output artifacts/profile_prehashed.json`（或平台侧重定向到对象存储）；
2. 用完整 `fields.*` + `shared_embedding_groups.*` 对 179+ 物理表逐个对比 `num_buckets` / `embedding_dim`；
3. 序列侧优先审 **物理表 / share root**（~25 张），108 个 alias 跟物理表走，不必逐个别改。

读表约定：

- `cur/rec` ≫ 1：当前桶偏大，宜缩
- `cur/rec` ≪ 1 且高基数：抽样给的是理想防碰撞值，**不能直接当涨桶依据**
- sequence 字段若 `share_with` 非空，改动的是**物理表**，会连带其它序列

---

## 档1｜立即改（浪费极大 / 明显配错）

| 字段 | 当前 (phys) | 抽样推荐 | distinct≈ | 审查结论 |
|------|-------------|----------|-----------|----------|
| `ad_id_bin_hn` | **16M × 48**（独立表） | **1k × 8** | **2** | 桶大 **16384×**，dim 也偏大。bf16 权重约 **1.5 GiB → 可忽略**。无 share，改它不影响别的表。**立刻缩到 1k/8（或最多 4k/8）。** |

---

## 档2｜高优先（低基数偏大，该批量收）

| 字段 | 当前 | 抽样推荐 | distinct≈ | 审查结论 |
|------|------|----------|-----------|----------|
| `c_adj_cart_cvr_15d_hn` | 4k × 16 | 1k × 8 | **1** | 抽样全是同一 hash，表几乎常数。**缩到 1k×8**；全量若仍≈1，可考虑去掉或并入 bias。 |
| `c_adj_ctr_15d_hn` | 4k × 16 | 1k × 8 | **1** | 同上。 |
| `c_adj_ordr_cvr_15d_hn` | 4k × 16 | 1k × 8 | **1** | 同上。 |
| `c_cart_cnt_15d_hn` | 4k × 16 | 1k × ? | **1** | 同上（报告里 dim 截断，按 8 处理）。单表省显存很小，但属于**错误量级配置**，应和上面一起改。 |

---

## 档3｜中优先（离散特征略大，顺手改）

| 字段 | 当前 | 抽样推荐 | distinct≈ | 审查结论 |
|------|------|----------|-----------|----------|
| `adj_cartcvr_hn` | 4k × 16 | 1k × 16 | 20 | 桶 4× 偏大，dim 已合适。**→ 1k×16**。省显存很小，但配置更合理。 |
| `adj_ctr_hn` | 4k × 16 | 1k × 16 | 21 | 同上。 |
| `adj_cvr_hn` | 4k × 16 | 1k × 16 | 21 | 同上。 |
| `buy_long_x_price_hn` → `price_hn` | 4k × 16 | 1k × 16 | 20 | share 到物理 `price_hn`，改物理表会连带其它序列。抽样仅 buy_long；**可收到 1k–2k**，改前扫一眼其它 `*_x_price_hn`。 |
| `buy_long_x_timegap_hn` → `impr.timegap_hn` | 4k × 16 | 1k × 8 | 14 | share 到 impr 侧。桶可 **4k→1k/2k**；dim 16→8 要确认其它序列是否共用。 |
| `buy_long_x_sales_hn` → `sales_hn` | 4k × 16 | 2k × 16 | 25 | **可 4k→2k**；收益小，和 price 一批改即可。 |

---

## 档4｜低优先 / 可选（略紧或已接近，非必须）

| 字段 | 当前 | 抽样推荐 | distinct≈ | 审查结论 |
|------|------|----------|-----------|----------|
| `buy_long_x_cat1_id_hn` → `cat1_id_hn` | **256 × 8** | 2k × 16 | 24 | 桶/维都偏紧。全量 cat1 通常仍很小。**可选：512–2k，dim 8→16**。非性能热点。 |
| `buy_long_x_cat2_id_hn` → `cat2_id_hn` | 4k × 16 | 16k × 24 | 278 | 略紧。**可选抬到 8k–16k**；dim 16→24 收益不明，可先不动 dim。 |
| `buy_long_x_cat3_id_hn` → `cat3_id_hn` | 32k × 24 | 128k × 24 | 2.4k | 抽样建议 4× 更大；当前 load 仍可接受。**全量后再决定是否 64k–128k**。 |
| `buy_long_x_cat4_id_hn` → `cat4_id_hn` | 128k × 32 | 512k × 32 | 7.9k | 同上，**低优先**。 |
| `buy_long_x_cat_id_hn` → `cat_id_hn` | 256k × 32 | 1M × 32 | 20k | 同上；抬桶有成本，**等全量 shared-group 统计**。 |
| `auto_price_p05_dis` | 4k × 16 | 8k × 16 | 123 | 当前略紧（0.5×）。**可选 4k→8k**；碰撞风险低，不急。 |
| `auto_price_p10_dis_hn` | 4k × 16 | 4k × 16 | 53 | **已对齐，不动。** |
| `auto_sales_p10_dis` | 4k × 16 | 4k × 16 | 52 | **已对齐，不动。** |
| `buy_long_x_time_delta_log1p_seconds` | **配置无此字段** | 1k × 8 | **0** | 抽样 cells=0。**不要新建 embedding**；若语义是 dense，应走 dense，不是 hash 表。 |

---

## 档5｜维持 / 勿盲跟抽样上涨（高基数 / 共享大表）

| 字段 | 当前 (phys) | 抽样推荐 | distinct≈ | 审查结论 |
|------|-------------|----------|-----------|----------|
| `buy_long_x_goods_id_hn` → **`goods_id_hn`** | **134M × 48**（全局共享） | 67M × 64（仅 buy_long 样本） | ~1.1M（样本） | 抽样推荐**不能代表**全序列 union 基数。当前已是共享大表；**禁止因抽样再涨 dim/桶**。若要省显存，走 Phase2 cap，且必须用 **shared_embedding_groups.goods** 全量统计，不能用单序列样本。 |
| `buy_long_x_sku_ids_hn` → `cart_long.sku_ids_hn` | **16M × 48** | 128M × 64 | ~1.35M | 抽样要再大 **8×**——那是理想防碰撞，**跟涨会爆显存**。保持 cap；全量后再评估。 |
| `buy_long_x_mall_id_hn` → `mall_id_hn` | 4M × 32 | 16M × 48 | ~0.31M | **勿跟到 16M/48**；维持，全量复核。 |
| `buy_long_spec_vids_hn` | 8M × 48 | 32M × 48 | ~0.40M | **维持 8M**；勿跟涨。 |
| `buy_long_x_spec_hn` → `cart_long.spec_hn` | 8M × 48 | 32M × 48 | ~0.48M | 与 vids 同档，**维持共享 cap**。 |

---

## 执行顺序

1. **档1**：只改 `ad_id_bin_hn`（收益最大）。
2. **档2**：四个 `c_*_15d` 缩到 1k×8。
3. **档3**：`adj_*` + price/sales/timegap 物理表小幅缩桶。
4. **档4**：cat / auto_p05 可选微调；空 `time_delta` 不建表。
5. **档5**：goods/sku/mall/spec **只持稳或考虑 cap，禁止按本抽样涨**。

---

## 已落地（档1–档3）

已写入以下 **8** 个配置（物理表 + 对应 `share_with` 别名；MDL 系各 33 处，非 MDL 系各 28 处，因无 task prior 序列）：

- `configs/mdl_rankmixer.yaml`
- `configs/mdl_rankmixer_fine.yaml`
- `configs/mdl_onetrans.yaml`
- `configs/mdl_onetrans_fine.yaml`
- `configs/rankmixer.yaml`
- `configs/rankmixer_fine.yaml`
- `configs/onetrans.yaml`
- `configs/onetrans_fine.yaml`

并同步收紧 `scripts/build_mdl_rankmixer_config.py` 的 `_estimated_bucket` 启发式，避免下次无 profile 重建时再胀回去。

| 物理表 | 原 shape | 新 shape |
|--------|----------|----------|
| `ad_id_bin_hn` | 16M × 48 | **1k × 8** |
| `c_adj_cart_cvr_15d_hn` / `c_adj_ctr_15d_hn` / `c_adj_ordr_cvr_15d_hn` / `c_cart_cnt_15d_hn` | 4k × 16 | **1k × 8** |
| `adj_cartcvr_hn` / `adj_ctr_hn` / `adj_cvr_hn` | 4k × 16 | **1k × 16** |
| `price_hn`（及各序列 share） | 4k × 16 | **1k × 16** |
| `sales_hn`（及各序列 share） | 4k × 16 | **2k × 16** |
| `impr.timegap_hn`（及 share 到它的序列字段） | 4k × 16 | **1k × 8** |
| `cat1_id_hn`（及 share） | 256 × 8 | **2k × 16** |
| `cat2_id_hn`（及 share） | 4k × 16 | **16k × 16**（桶按抽样抬；dim 暂不升 24） |
| `cat3_id_hn`（及 share） | 32k × 24 | **128k × 24** |
| `cat4_id_hn`（及 share） | 128k × 32 | **512k × 32** |
| `cat_id_hn`（及 share） | 256k × 32 | **1M × 32** |
| `auto_price_p05_dis` | 4k × 16 | **8k × 16** |

未改：`auto_price_p10_dis_hn` / `auto_sales_p10_dis`（已对齐）、空 `time_delta`（不新建表）、档5 goods/sku/mall/spec。其余未覆盖字段等完整 profile JSON 后再审。

---

## 备注

- 同语义字段（其它序列的 price/timegap、更多 `c_*`）未出现在截断日志里时，应按同一档规则批量套用。
- `task_*_prior` 在 contract 中全 missing、filter 后 count=0，属于数据/接线问题，优先级不低于档1，但不在本 bucket/dim 对照表内。
- 定最终高基数桶之前，应再跑更大窗口（更多 pt/hr、更大 `--max-files/--max-rows`），并看 `shared_embedding_groups` 而非单序列字段。
