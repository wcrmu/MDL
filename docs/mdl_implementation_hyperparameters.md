# `mdl_rankmixer` / `mdl_onetrans` 实现超参数

> 来源：`configs/mdl_rankmixer.yaml`、`configs/mdl_onetrans.yaml`（生产 coarse 默认）。  
> CLI 覆盖（如 `--batch-size`、`--activation-checkpoint`）会改写下表中的对应项。

## 1. `mdl_rankmixer`

| 类别 | 超参数 | 值 |
|---|---|---|
| 结构 | Feature token | groupwise **32×768**（23 语义组 + 9 历史组） |
| | Backbone 层数 / heads | **2 / 12** |
| | FFN hidden / activation | 1536 / GELU |
| | Feature interaction | `residual_ffn`（论文对齐可用 `direct_ffn`） |
| | RankMixer FFN | dense |
| Domain | Scenario tokens | search + recommendation + **global** |
| | Task tokens | `fst_cart` / `upid_pay` / `cateid_filter` |
| | Token state | `coupled`（可选 `split`） |
| | Scene→feature bias | `none` |
| | Domain 读序列层 | 无（读 Feature tokens） |
| 历史 | 主历史 encoder | **LONGER**（`longer_dim=32`, heads=4, query_tokens=32） |
| | 主历史 max_length（例） | impr 1024；clk/view 2048；cart 512；buy 256 等 |
| | omit request scene 进主 pack | **true** |
| 训练 | batch / rank（默认 bucket） | **1536**（长度桶：1536/960/640/480/768） |
| | lr dense / sparse | 1e-4 / 1e-4 |
| | schedule / warmup | constant / **5000** |
| | dense opt | RMSprop（α=0.99999, momentum=0, fused） |
| | sparse opt | Row-Wise Adagrad（init=0.1, eps=1e-10）, sharded |
| | clip dense / sparse | 1.0 / 1.0 |
| | loss | `mean_per_task`；权重 0.5 / 0.01 / 0.01 |
| | emb dtype / dist | bf16 / sharded |
| 运行时 | precision | bf16 |
| | activation checkpoint | **none** |
| | cuda_graph_backbone | **true** |
| | attention / packing | flash / **compact** |
| | 默认 nproc | 2 |
| 评估 | fixed-test | every **5000** step；`files_per_rank=4`；auc_bins=4096 |

---

## 2. `mdl_onetrans`

| 类别 | 超参数 | 值 |
|---|---|---|
| 结构 | Token 宽 / 层数 / heads | **256 / 6 / 4** |
| | NS tokens | **32**（`feature_tokenizer=auto_split`，`ns_tokenizer=dcnv2`） |
| | omit request scene 进 NS pack | **true** |
| | DCNv2 | cross=3, deep_layers=2, deep_dim=1024, proj_dim=1024 |
| | 位置容量 | `max_position_embeddings=2088` |
| | Pyramid / 最终保留 S | `use_pyramid=true`；`final_s_tokens=12` |
| | SEP tokens | 开启（`use_sep_tokens=true`） |
| | FFN hidden / task head hidden | 1024 / 1024 |
| | experimental | **true**（组合模型，非论文公开结构） |
| Domain | Scenario / Task | 同 coarse：2+global / 3 task |
| | Token state | `coupled` |
| | Scene→feature bias | `none` |
| | 每层读 NS | 是 |
| | 读 S 起始层 | **`first_domain_sequence_layer=4`**（最后两层 + gate） |
| 历史 | 主历史 encoder | **raw S stream**（非 LONGER） |
| | 主历史 max_length（例） | impr 256；clk/view 512；cart 192；buy 128 等 |
| | S 总事件上限（工程口径） | 拼接待 SEP 后落在 2088 位置预算内 |
| | request cache | **true** |
| 训练 | batch / rank（默认 bucket） | **1408**（长度桶：1408/880/576/432/704） |
| | lr dense / sparse | 1e-4 / 1e-4 |
| | schedule / warmup | constant / **5000** |
| | dense opt | RMSprop（α=0.99999, momentum=0；默认未开 fused） |
| | sparse opt | Row-Wise Adagrad（init=0.1, eps=1e-10）, sharded |
| | clip dense / sparse | 1.0 / 1.0 |
| | loss | `mean_per_task`；权重 0.5 / 0.01 / 0.01 |
| | emb dtype / dist | bf16 / sharded |
| 运行时 | precision | bf16 |
| | activation checkpoint | **none** |
| | cuda_graph_backbone | false（配置未开） |
| | attention / packing | flash / **fixed** |
| | seq proj / encoder chunk | 81920 / 163840 tokens |
| | 默认 nproc | 2 |
| 评估 | fixed-test | every **5000** step；`files_per_rank=4`；auc_bins=4096 |

---

## 3. 对照（只列差异）

| 项 | `mdl_rankmixer` | `mdl_onetrans` |
|---|---|---|
| 主干 | 32×768 RankMixer ×2 | S/NS OneTrans ×6，宽 256 |
| 历史 | LONGER summary → Feature | raw event → causal S |
| Domain 读什么 | Feature tokens | 每层 NS + 后两层 S |
| 默认 batch | 1536 | 1408 |
| packing | compact | fixed |
| cuda graph | on | off |
| 标记 | 生产 MDL×RankMixer | experimental 组合 |

配置与字段细节：[数据适配总览](./mdl_data_adaptation_overview.md)、[OneTrans 难点](./mdl_onetrans_adaptation_hardships.md)、[词表](./mdl_vocab_embedding_design.md)。
