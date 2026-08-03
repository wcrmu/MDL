# MDL 适配 OneTrans：遇到的难点、根因与解法

> 面向串讲与线上迁移讨论。  
> 事实基线：仓库 `main@5a7f1c1`，截至 2026-08-03。  
> 配套讲稿：[`mdl_reproduction_lecture_report.md`](./mdl_reproduction_lecture_report.md)  
> 数据适配总览：[`mdl_data_adaptation_overview.md`](./mdl_data_adaptation_overview.md)  
> 字段与 Domain 合同：[`mdl_token_feature_design.md`](./mdl_token_feature_design.md)、[`current_field_processing_report.md`](./current_field_processing_report.md)

## 0. 这篇文档讲什么

线上模型与 OneTrans 高度相似，因此需要把 MDL 的 Scenario/Task Domain 接到 OneTrans 的 S/NS 主干上。本文只讲适配时真正卡住的难点，不讲普通配置搬运。

当前实现是实验性组合模型 `mdl_onetrans`，**不是** MDL 或 OneTrans 论文的公开模型：

| 项 | 当前生产口径 |
|---|---|
| 主干 | 9 条原始历史 → S；DCNv2 → 32 个 NS |
| 规模 | `token_dim=256`，6 层、4 heads，`max_position_embeddings=2088` |
| Domain | sidecar recurrent states，逐层读 NS；从 layer 4 起额外读压缩后的 S |
| 状态语义 | 默认 `mdl_token_state=coupled` |
| batch | 当前 `batch_size=1408`（多 profile 另有 880/576/432/704） |
| 边界 | `experimental_model_acknowledged=true` |

证据边界：仓库有结构一致性、缓存等价、多卡与性能记录，但**没有**可直接引用的固定 holdout 上完整 MDL vs OneTrans 质量对比曲线。因此本文不下“效果已复现”的结论。

---

## 1. 难点总览

| # | 难点 | 一句话 |
|---|---|---|
| 1 | Domain 往哪放 | 塞进 `[S; NS]` 会破坏 cache 边界或逼出第三种模型 |
| 2 | 读什么、读哪几层 | NS 便宜且含候选；S 最长 2048，必须受显存约束取舍 |
| 3 | coupled 难归因 | 强 prior 可沿 residual 绕过 Attention 到 logit |
| 4 | cache 按 R 还是按 B | 提前按候选展开会把峰值从 \(O(R)\) 放大到 \(O(B)\) |
| 5 | 增量 cache 不等于 append 可复用 | pyramid 会让深层旧 token 表示变化 |
| 6 | 长跑 OOM 不是单旋钮 | packing、cache、width、allocator、checkpoint 互相耦合 |
| 7 | 线上对齐变量太多 | scene 通道、readout 容量等会污染归因 |

下面按「问题 → 根因 → 解决方案 → 验证/边界」展开。

---

## 2. 难点详解

### 2.1 Domain token 不能直接进入 `[S; NS]` 因果序列

**问题：**  
最直觉的改法是把 Scenario/Task append 进 OneTrans 序列。两个插入位置都有结构性问题。

**根因：**

```text
request histories -> S tokens ----┐
                                  ├-> causal OneTrans blocks
candidate features -> 32 NS tokens┘
```

S 是请求级、可缓存；NS 是候选级。Domain prompt 含候选类目、价格、goods/query identity，不是纯请求级状态。

| 放置方式 | 会发生什么 | 为什么不采用 |
|---|---|---|
| 放在 NS 之前 | Domain 被当成共享 S，后续 NS 可读它 | 破坏 request cache 边界；也反转了 MDL 的 `Domain reads Feature` 方向 |
| 放在 NS 之后 | Domain 变成新增 token-specific slots | 必须扩展 NS 专属 Q/K/V、FFN、位置与缓存；Scenario/Task 顺序还会制造任意因果可见性 |
| 与 NS 混排 | 可做更复杂 mask | 需要重定义参数共享与 cache contract，得到第三种模型，不是清晰的 MDL 适配 |

**解决方案：**  
保留 OneTrans 的 S/NS 主干不变，把 Scenario/Task 做成 **sidecar recurrent states**。Domain 只读取每一层最新的 OneTrans 表示，**不写回** S/NS。

这保留三条不变量：

1. OneTrans 的 S 因果链与 request cache 仍成立；
2. MDL 仍是 Domain 作 Query、Feature 作 Key/Value；
3. 最终仍由 Task state 接任务 head，而不是重新引入 `scene × task` 输出塔。

**验证/边界：**  
结构测试应覆盖：Domain 不改变 S/NS 因果可见性；关闭 Domain 时主干与纯 OneTrans 路径可对齐（在相同 S/NS 合同下）。sidecar 是组合设计，效果需单独实验矩阵归因。

---

### 2.2 每层读 NS，但只在后两层读 S

**问题：**  
Domain 若对最长 2048 的 S 做全层 cross-attention，训练代价与峰值显存都会失控；若完全不读 S，又丢掉长历史直接证据。

**根因：**  
三类信号职责不同：

| 信号 | 特点 | 当前处理 |
|---|---|---|
| 32 个 NS | 已融合当前候选，长度固定 | Scenario/Task **每一层**都读 |
| 最长 2048 的 S | 请求共享、变长；早层偏原始事件 | OneTrans 先压缩 4 层，**最后两层**才允许 Domain 读 |
| 7 个 Domain prior summary | 面向 scenario/task 的紧凑历史摘要 | 只初始化 Domain prompt，不进原始 S 因果链 |

当前 YAML 有 16 个逻辑 sequence 配置：9 条 raw S stream + 7 条 Scenario/Task prior。它们复用上游物理历史，但消费者、pooling 与 embedding scope 不同。

逐层计算（记第 \(l\) 层后为 \(S^l,N^l,D^l\)）：

\[
(S^{l+1},N^{l+1})=\operatorname{OneTransBlock}_l(S^l,N^l)
\]

\[
\Delta^{l}_{D,NS}
=\operatorname{Attn}
\left(Q_l(D^l),K_l(N^{l+1}),V_l(N^{l+1})\right)
\]

从 zero-based layer 4 起（6 层中的最后两层）再读压缩后的 S：

\[
\Delta^{l}_{D,S}
=\operatorname{VLAttn}
\left(Q_l(D^l),S^{l+1},M_S^{l+1}\right)
\]

S 分支由残差门控控制，门控 bias 初始化为 \(-2\)（零输入时 \(g\approx0.119\)），避免训练初期长序列支路直接压过 NS：

\[
g^l=\sigma\left(W_g[D^l,\Delta^{l}_{D,NS},\Delta^{l}_{D,S}]+b_g\right)
\]

\[
\hat D^l=D^l+\Delta^{l}_{D,NS}+g^l\odot\Delta^{l}_{D,S}
\]

**解决方案：**  
`first_domain_sequence_layer=4`：先保证能训、能缓存；算法结论留给消融。

**验证/边界：**  
这是有显存约束的工程起点，不是论文结论。至少比较 `null`（不直接读 S）、`4`（最后两层）、`0`（全层读），并同时报告质量、延迟与峰值 HBM。`S-only / prior-only / S+prior` 也要单独做，否则无法区分“复用同一份行为”与“重复计权”。

---

### 2.3 `coupled` 让“Attention 是否真的在工作”很难归因

**问题：**  
线上若直接拿 `coupled` 的提升讲“Domain Attention 有效”，可能把 initializer bypass 算进去。

**根因：**  
默认 `coupled` 与 MDL 原始状态传播一致：

```text
important/prior -> Domain prompt
prompt == attention query == recurrent residual state == final readout state
```

即使把 Domain Attention 的 Value update 置零，strong initializer 仍能沿 residual 到达 logits。这是论文公式允许的路径，但会让收益来源含糊。

**解决方案：**  
额外实现诊断路径 `split`：

```text
prompt p = Projector(important, prior)      # 只控制 query
state h0 = learned readout seed             # 承接证据与最终 readout
query ql = p + hl
hl+1 = hl + Attention(q=ql, K/V=NS/S)
```

`split` 的硬验证：固定 state 并把所有 Value update 置零后，交换样本 prompt **不得**改变 logit。

**验证/边界：**  
- 线上对齐论文同型：先用 `coupled`；
- 要判断 bypass 贡献：再用 `split`；
- `coupled` 与 `split` checkpoint **不能混用**。

---

### 2.4 请求级缓存：不能把 R 份 S 在第 0 层就展开成 B 份

**问题：**  
开启 S cache 后，峰值显存随 candidates/request 急升，长跑易 OOM。

**根因：**  
推荐请求通常有 \(R\) 个 request、\(B\) 个 candidate，且 \(B\gg R\)。S cache 属于 request：

\[
C_l^S\in\mathbb{R}^{R\times L_l\times D},
\qquad
N_l\in\mathbb{R}^{B\times32\times D}
\]

早期实现为了方便，在进入第一层前按 `request_row_indices` 把**所有层**的 `s_input/K/V/output` 一次性扩成 candidate batch。数值正确，但峰值从

\[
O\left(\sum_l R L_lD\right)
\quad\text{变成}\quad
O\left(\sum_l B L_lD\right).
\]

**解决方案：**  
提交 `5e040d6`：cache 永远保留 request-sized；只在当前层的 `forward_cached_ns/step` 内做一次 layer-local gather。checkpoint 也只保存紧凑 request tensor。

线上铁律：**缓存键必须是 request；候选展开只是当前层的消费方式。** Domain state 含候选 important features，通常仍是 candidate-sized，不能整体塞进 request cache。

**验证/边界：**  
cache on/off 时 logits/grad 必须一致；性能收益应随 candidates/request 增长而显现。在线 batching 必须保留可靠的 `candidate -> request` 映射。

---

### 2.5 增量缓存不是“append 就一定能复用”

**问题：**  
增量 cache 命中率看起来很高，输出仍可能与 full recompute 不一致。

**根因：**  
OneTrans 支持增量更新 S cache，但 pyramid 会让更深层的保留窗口移动。即使原始历史只是 append，新窗口中的旧 token 也可能因上层输出变化而不再等于旧 cache。

**解决方案：**  
逐层检查 overlap 的 token 与 mask：

- overlap 完全一致：复用旧 K/V，只投影新追加部分；
- overlap 不一致：该层从头重建，保证与 full recompute 精确一致；
- full checkpoint 丢弃持久 K/V 时：明确重建，而不是伪装成 cache hit。

**验证/边界：**  
线上不能只看命中率；必须用 `incremental == full recompute` 的逐层张量测试约束正确性。

---

### 2.6 实际撞上的显存与长跑 OOM

**问题：**  
短 smoke 能过，长跑或换 profile 仍 OOM；把 batch 从 768 降到 512 也不保证安全。

**根因与修复：**

| 现场问题 | 根因 | 修复 |
|---|---|---|
| compact pack 阶段直接 OOM | boolean indexing 同时制造大 mask 与大拷贝 | 索引式 `index_select`，避免全宽临时副本（`cc92f47`） |
| cache 开启后峰值随 candidates/request 急升 | 所有层 S cache 被提前从 request 展开到 candidate | request-sized 持久 cache + layer-local gather（`5e040d6`） |
| token width 扩到 512 后 HBM 不安全 | S 长度远大于 NS，宽度翻倍同时放大 QKV、FFN 与 cache | 生产 profile 恢复到 256（`1eef46f`） |
| batch 768 在约 step 1300 仍 OOM | 除 live tensor 外约有 12 GiB reserved fragmentation；短 smoke 未覆盖高水位 | allocator 配置必须在 import torch 前生效，并联动 length bucket、projection chunk 与 packing（`304c1d0`） |
| 打开 full activation checkpoint 后仍 mid-run OOM | checkpoint、固定/变长 packing、cache K/V 与 batch profile 耦合；单一“省激活”旋钮非单调 | 回退到验证过的 `activation_checkpoint=none + fixed packing` 基线后重新调参（`f6fdb68`） |
| Flash/FFN 峰值叠加 | 冗余 contiguous、重复 mask metadata、长序列投影同时存活 | 复用 packing metadata、跳过冗余拷贝、按长度分块序列投影（`4861e0e`） |

**解决方案的结论：**  
当前已重新调到 `batch_size=1408`。历史上的“降到 512”不是终局答案。OneTrans 的安全 batch 必须在**真实长度分布、缓存策略、allocator、packing 与完整长跑**下联合验证，不能由短跑峰值或单一 batch 数字外推。

**验证/边界：**  
任何新 packing / checkpoint / width / cache 组合，都要跑到足以暴露 reserved fragmentation 的步数，并报告峰值 HBM、cache bytes/request、candidates/request 扩展斜率。

---

### 2.7 与线上 OneTrans 对齐时，必须先锁住的变量

**问题：**  
即使 sidecar 能跑，直接对比“线上 OneTrans vs mdl_onetrans”仍可能把多个变化混在一起。

**根因：**  
至少有 6 个会改变含义或成本的变量同时在动：

| 变量 | 当前 `mdl_onetrans` | 上线前要回答的问题 |
|---|---|---|
| Scene 通道 | request-side scene 从 NS 中 omit；candidate×scene crosses 保留；MDL 通过 Scenario prompt/routing 获取 request scene | 线上是否仍在 NS/S 中读 request scene？不同则必须做 `content scene on/off × Domain on/off` 2×2 |
| S/NS 合同 | 9 条 S、8 个 SEP、32 个 NS，位置容量 2088 | 历史顺序、截断、separator、padding、causal mask 是否逐项一致 |
| Domain initializer | 含 request 与 candidate important、7 个历史 prior | 哪些字段能在线稳定供给；是否引入 train/serve 不一致；强 prior 是否绕过 Attention |
| Cache 粒度 | S 为 request cache；NS/Domain 为 candidate state | 在线 batching 是否保留 `candidate -> request` 映射；更新时是否 full-recompute 等价 |
| Readout | MDL 用 1 个 256 维 Task state；纯 OneTrans 当前把 `32×256=8192` flatten 后进 task MLP | 直接比较会同时改变 readout 容量；需要 equal-readout control |
| 成本口径 | 每层 Domain 读 32 NS，最后两层再读 S | 除 samples/s 外还要报 request P99、candidate expansion slope、cache bytes/request、峰值 HBM |

**解决方案：**  
先锁合同，再开 Domain；用最小可归因实验矩阵推进，而不是一次全开。

**验证/边界：**  
没有 equal-readout 与固定 scene policy，就不能把结果归因给“MDL-OneTrans”。

---

## 3. 最小可归因实验矩阵

| 实验 | 唯一变化 | 要回答的问题 |
|---|---|---|
| A. OneTrans baseline | 保持线上字段、S/NS、缓存和 readout | 建立可信基线 |
| B. Equal-readout baseline | baseline 改成与 MDL 相同维度的 task-query/readout，不加 layer-wise Domain | 排除 8192→256 readout 变化 |
| C. NS-only MDL | Domain 每层只读 NS，`first_domain_sequence_layer=null` | Layer-wise Domain 是否有增益 |
| D. Late-S MDL | 仅最后两层增加 gated S read | 长历史直接证据是否有额外价值 |
| E. `coupled` vs `split` | 字段、参数预算和数据完全一致 | 收益来自 initializer bypass 还是动态读取 |
| F. Scene 2×2 | content-side scene on/off × Domain scene on/off | scene 信号换通道造成了多少收益 |
| G. Cache on/off | logits/grad 必须一致，只比较性能 | 请求缓存是否真正等价、收益是否随 candidates/request 增长 |

质量至少按 `task × scene` 报告 AUC/QAUC、logloss 与 calibration；系统至少报告 samples/s、P50/P99、峰值 HBM、host RSS、cache bytes/request。

---

## 4. 串讲时怎么讲这 18 分钟

建议顺序：**难点 → 错了会怎样 → 我们怎么解 → 还没闭环的验证**。

1. **开场（2 min）：** 为什么要接 OneTrans；当前 `mdl_onetrans` 是组合设计，不是论文公开模型。  
2. **结构难点（5 min）：** Domain 不能进序列 → sidecar；每层 NS / 后两层 S；coupled vs split。  
3. **系统难点（7 min）：** request-sized cache；pyramid 下增量复用；六类 OOM 现场。  
4. **收尾（4 min）：** 六个线上变量 + 实验矩阵；明确哪些结论现在还不能讲。

---

## 附录：证据索引

| 主题 | 材料 |
|---|---|
| 结构与 Domain 字段 | [`mdl_token_feature_design.md`](./mdl_token_feature_design.md) |
| 字段与 S/NS 处理 | [`current_field_processing_report.md`](./current_field_processing_report.md) |
| 串讲总稿 | [`mdl_reproduction_lecture_report.md`](./mdl_reproduction_lecture_report.md) |
| 配置 | `configs/mdl_onetrans.yaml`、`configs/mdl_onetrans_fine.yaml` |
| 结构测试 | `tests/test_model_alignment.py` |
| OneTrans HBM/cache 提交 | `cc92f47`、`1eef46f`、`4861e0e`、`5e040d6`、`304c1d0`、`f6fdb68` |
