# MDL 复现与 MDL-OneTrans 适配：问题、根因与解法

> 面向推荐系统算法与工程同学的串讲底稿
>
> 事实基线：仓库 `main@5a7f1c1`，截至 2026-08-03
>
> 材料范围：当前代码与生产 YAML、Git 提交、字段审计、Embedding 容量审计、数据链路与性能报告

## 0. 这次串讲的口径

这次只讲两类事情：一类是**会改变模型含义或实验结论的算法问题**，另一类是**会让长跑训练挂死、OOM、静默训错的工程问题**。普通接口适配、配置搬运和没有形成因果闭环的小优化不单列。

全文分成两条主线：

1. `mdl_rankmixer`：复现 MDL 时遇到的计算图、token、数据、稀疏训练和评测问题；
2. `mdl_onetrans`：考虑到线上模型与 OneTrans 高度相似，单独说明把 MDL 接到 OneTrans 时遇到的难点，以及缓存和显存解法；完整专题见 [`mdl_onetrans_adaptation_hardships.md`](./mdl_onetrans_adaptation_hardships.md)。

需要先声明证据边界：仓库目前有较完整的结构一致性测试、数据 oracle、多卡一致性测试和性能记录，但**没有一条可以直接引用的、固定 holdout 上完整的 MDL/OneTrans 质量对比曲线**。因此本文不会把“代码能跑”写成“论文效果已复现”，也不会编造 AUC 从 0.59 到 0.7、四层优于六层等结论。

## 1. 当前实现到底是什么

| 模型 | 当前主体结构 | 当前生产口径 | 必须注意的边界 |
|---|---|---|---|
| `mdl_rankmixer` | 32 个 Feature token，维度 768，2 层 RankMixer；3 个 Scenario state（search、recommendation、global）与 3 个 Task state 逐层读取 Feature | `groupwise` 语义分组；`residual_ffn + coupled` | 严格对齐 MDL Eq. 6 时应使用 `direct_ffn + coupled`；生产稳定变体不能冒充逐式复现 |
| `mdl_onetrans` | 9 条原始历史形成 S，DCNv2 形成 32 个 NS；维度 256，6 层、4 heads；Domain state 作为旁路逐层读取 OneTrans | S 总事件上限 2048，加 8 个 SEP 与 32 个 NS，位置容量 2088；最终保留 12 个 S；最后两层额外读取 S | `experimental_model_acknowledged=true`；这是我们的组合设计，不是 MDL 或 OneTrans 论文公开模型 |

`mdl_rankmixer` 的 32 个 token 是 **23 个非序列语义组 + 9 个历史组**；`mdl_onetrans` 的 32 个 NS 是 DCNv2 生成的 latent slots。两者虽然 token 数都为 32，但语义并不等价。

---

## 2. 重点专题：MDL-OneTrans 适配难点（摘要）

> 本节是独立主体，**不占后面 8 个核心问题的名额**。  
> 完整“难点—根因—解法—边界”版见 [`mdl_onetrans_adaptation_hardships.md`](./mdl_onetrans_adaptation_hardships.md)。

### 2.1 最关键的决定：Domain token 不进入 `[S; NS]` 因果序列

OneTrans 的原始信息流是：

```text
request histories -> S tokens ----┐
                                  ├-> causal OneTrans blocks
candidate features -> 32 NS tokens┘
```

其中 S 是请求级、可缓存的共享历史；NS 是候选级、包含当前候选信息的非共享 token。最直接的改法似乎是把 Scenario/Task token 一起 append 到序列里，但两个插入位置都有结构性问题。

| 放置方式 | 会发生什么 | 为什么不采用 |
|---|---|---|
| 放在 NS 之前 | Domain 被当成共享 S；后续 NS 可以读取它 | 当前 Domain prompt 含候选类目、价格、goods/query identity，不是请求级状态；写入 S 会破坏请求缓存边界，也改变 MDL 的 `Domain reads Feature` 方向 |
| 放在 NS 之后 | Domain 变成新增的 token-specific slots | 必须扩展 OneTrans 的 NS 专属 Q/K/V、FFN、位置与缓存；Scenario/Task 的先后顺序还会制造任意的 causal 可见性 |
| 与 NS 混排 | 可以构造更复杂 mask | 需要重新定义 S/NS 参数共享、因果顺序和 cache contract，得到的是第三种模型，而不是清晰的 MDL 适配 |

最终方案是：**保留 OneTrans 的 S/NS 主干不变，把 Scenario/Task 做成 sidecar recurrent states。** Domain 只读取每一层最新的 OneTrans 表示，不写回 S/NS。

这保留了三条重要不变量：

- OneTrans 的 S 因果链和 request cache 仍然成立；
- MDL 仍然是 Domain 作 Query、Feature 作 Key/Value；
- 最终仍由 Task state 连接任务 head，而不是重新引入 `scene × task` 输出塔。

### 2.2 当前逐层计算图

记第 $l$ 层后的 S、NS、Domain state 分别为 $S^l$、$N^l$、$D^l$。每层先走原生 OneTrans：

\[
(S^{l+1},N^{l+1})=\operatorname{OneTransBlock}_l(S^l,N^l)
\]

然后 Scenario 与 Task 分别读取当前 32 个 NS：

\[
\Delta^{l}_{D,NS}
=\operatorname{Attn}
\left(Q_l(D^l),K_l(N^{l+1}),V_l(N^{l+1})\right)
\]

当前配置从 zero-based layer 4 开始，也就是 6 层中的最后两层，再让 Domain 直接读取已经经过 pyramid 压缩的 S：

\[
\Delta^{l}_{D,S}
=\operatorname{VLAttn}
\left(Q_l(D^l),S^{l+1},M_S^{l+1}\right)
\]

S 分支不是无条件相加，而是由残差门控控制：

\[
g^l=\sigma\left(W_g[D^l,\Delta^{l}_{D,NS},\Delta^{l}_{D,S}]+b_g\right)
\]

\[
\hat D^l=D^l+\Delta^{l}_{D,NS}+g^l\odot\Delta^{l}_{D,S}
\]

门控 bias 初始化为 $-2$，零输入时 $g\approx0.119$。这样新增的长序列支路在训练初期不会直接压过 NS 支路。Scenario 先更新；当前 active Scenario 与 global Scenario 经 `DomainFused` 注入 Task；最终 Task state 进入任务 head。

### 2.3 为什么每层读 NS，但只在后两层读 S

这不是论文结论，而是当前工程假设，依据是计算职责与代价不同。

| 信号 | 特点 | 当前处理 |
|---|---|---|
| 32 个 NS | 已融合当前候选，长度固定，候选相关性强 | Scenario/Task 每一层都读取 |
| 最长 2048 的 S | 请求共享、变长、早层偏原始事件，直接 cross-attention 昂贵 | 先由 OneTrans 做 4 层压缩，最后 2 层才允许 Domain 读取 |
| 7 个 Domain prior summary | 面向 scenario/task 的紧凑历史摘要 | 用于初始化 Domain prompt，不进入原始 S 因果链 |

当前 YAML 一共有 16 个逻辑 sequence 配置：9 条 raw S stream，以及 7 条 Scenario/Task prior。它们复用上游物理历史，但消费者、pooling 方式和 embedding scope 不同。这里要区分“复用同一份原始行为”与“同一信号被模型使用两次”：后者可能带来收益，也可能造成先验过强或重复计权，必须通过 `S-only / prior-only / S+prior` 消融回答。

`first_domain_sequence_layer=4` 目前只是一个有显存约束的合理起点。要形成算法结论，至少要比较 `null`（不直接读 S）、`4`（最后两层）和 `0`（全层读取），同时报告质量、延迟和峰值 HBM。

### 2.4 `coupled` 与 `split`：线上适配时必须先选清楚的状态语义

当前默认 `coupled` 与 MDL 原始状态传播一致：

```text
important/prior -> Domain prompt
prompt == attention query == recurrent residual state == final readout state
```

因此，即使把 Domain Attention 的 Value update 置零，strong initializer 仍能沿 residual 到达 logits。这是论文公式允许的路径，但会让“提升究竟来自动态读取，还是来自 task-specific 强先验”难以归因。

我们额外实现的 `split` 用于诊断：

```text
prompt p = Projector(important, prior)      # 只控制 query
state h0 = learned readout seed             # 承接证据与最终 readout
query ql = p + hl
hl+1 = hl + Attention(q=ql, K/V=NS/S)
```

`split` 的硬验证是：固定 state 并把所有 Value update 置零后，交换样本 prompt 不得改变 logit。它不是论文默认，也不能与 `coupled` checkpoint 混用。对于线上模型，建议先以 `coupled` 做论文同型基线，再用 `split` 判断 initializer bypass 到底贡献了多少。

### 2.5 请求级缓存：不能把 R 份 S 在第 0 层就展开成 B 份

推荐请求通常有 $R$ 个 request、$B$ 个 candidate，且 $B\gg R$。S cache 属于 request：

\[
C_l^S\in\mathbb{R}^{R\times L_l\times D},
\qquad
N_l\in\mathbb{R}^{B\times32\times D}
\]

早期实现为了方便，在进入第一层前按 `request_row_indices` 把**所有层**的 `s_input/K/V/output` 一次性扩成 candidate batch。虽然数值正确，但峰值缓存近似从

\[
O\left(\sum_l R L_lD\right)
\]

变成

\[
O\left(\sum_l B L_lD\right),
\]

候选数越多越容易 OOM。

提交 `5e040d6` 的修复是：cache 永远保留 request-sized；只在当前层的 `forward_cached_ns/step` 内，根据 `request_row_indices` 做一次 layer-local gather。checkpoint 中也只保存紧凑 request tensor，不再同时持有所有层的 candidate 展开副本。

这条规则对线上尤为重要：**缓存键必须是 request，候选展开只是当前层的消费方式。** Domain state 因为含候选 important features，通常仍是 candidate-sized，不能整体塞进 request cache。

### 2.6 增量缓存不是“append 就一定能复用”

OneTrans 支持增量更新 S cache，但 pyramid 会让更深层的保留窗口发生移动。即使原始历史只是 append，新窗口中的旧 token 也可能因上层输出变化而不再等于旧 cache。

当前实现逐层检查 overlap 的 token 与 mask：

- overlap 完全一致：复用旧 K/V，只投影新追加部分；
- overlap 不一致：该层从头重建，保证与 full recompute 精确一致；
- full checkpoint 丢弃持久 K/V 时：明确重建，而不是伪装成 cache hit。

线上接入不能只看“命中率”，还要用 `incremental == full recompute` 的逐层张量测试约束正确性。

### 2.7 实际遇到的 OneTrans 显存问题

| 现场问题 | 根因 | 修复 |
|---|---|---|
| compact pack 阶段直接 OOM | boolean indexing 同时制造大 mask 与大拷贝 | 改为索引式 `index_select`，避免全宽临时副本（`cc92f47`） |
| cache 开启后峰值随 candidates/request 急升 | 所有层 S cache 被提前从 request 展开到 candidate | request-sized 持久 cache + layer-local gather（`5e040d6`） |
| OneTrans token width 扩到 512 后 HBM 不安全 | S 长度远大于 NS，宽度翻倍会同时放大 QKV、FFN 与 cache | 生产 profile 恢复到 256（`1eef46f`） |
| batch 768 在约 step 1300 仍 OOM | 除 live tensor 外约有 12 GiB reserved fragmentation；短 smoke test 未覆盖高水位 | allocator 配置必须在 import torch 前生效，并联动 length bucket、projection chunk 与 packing（`304c1d0`） |
| 打开 full activation checkpoint 后仍出现 mid-run OOM | checkpoint、固定/变长 packing、cache K/V 与 batch profile 存在耦合；单一“省激活”旋钮并非单调收益 | 回退到验证过的 `activation_checkpoint=none + fixed packing` 基线后重新调参（`f6fdb68`） |
| Flash/FFN 峰值叠加 | 冗余 contiguous、重复 mask metadata、长序列投影同时存活 | 复用 packing metadata、跳过冗余拷贝、按长度分块序列投影（`4861e0e`） |

当前配置已经重新调到 `batch_size=1408`，这说明历史上的“降到 512”不是最终答案。真正的结论是：OneTrans 的安全 batch 必须在**真实长度分布、缓存策略、allocator、packing 与完整长跑**下联合验证，不能由短跑峰值或单一 batch 数字外推。

### 2.8 与线上 OneTrans 类模型对齐时，必须控制的 6 个变量

| 变量 | 当前 `mdl_onetrans` | 上线前要回答的问题 |
|---|---|---|
| Scene 通道 | request-side scene 从 NS 中 omit；candidate×scene crosses 保留；MDL 通过 Scenario prompt/routing 获取 request scene | 线上 OneTrans 是否仍在 NS/S 中读取 request scene？若不同，必须做 `content scene on/off × Domain on/off` 2×2，而不能把换通道当成 MDL 收益 |
| S/NS 合同 | 9 条 S、8 个 SEP、32 个 NS，位置容量 2088 | 线上历史顺序、截断、separator、padding 和 causal mask 是否逐项一致 |
| Domain initializer | 含 request 与 candidate important features、7 个历史 prior | 哪些字段能在线稳定供给；是否引入训练—推理不一致；强 prior 是否绕过 Attention |
| Cache 粒度 | S 为 request cache，NS/Domain 为 candidate state | 在线 batching 是否保留可靠的 `candidate -> request` 映射；更新 cache 时是否能保证 full-recompute 等价 |
| Readout | MDL 用 1 个 256 维 Task state；纯 OneTrans 当前把 `32×256=8192` flatten 后进 task MLP | 直接比较会同时改变 readout 容量；需要增加 equal-readout control |
| 成本口径 | 每层 Domain 读 32 NS，最后两层再读 S | 除 samples/s 外还要报告 request P99、candidate expansion slope、cache bytes/request 与峰值 HBM |

### 2.9 最小可归因实验矩阵

建议按以下顺序推进，而不是一次把所有 Domain 功能全开：

| 实验 | 唯一变化 | 要回答的问题 |
|---|---|---|
| A. OneTrans baseline | 保持线上字段、S/NS、缓存和 readout | 建立可信基线 |
| B. Equal-readout baseline | 将 baseline 改成与 MDL 相同维度的 task-query/readout，但不加 layer-wise Domain | 排除 8192→256 readout 变化 |
| C. NS-only MDL | Domain 每层只读 NS，`first_domain_sequence_layer=null` | Layer-wise Domain 是否有增益 |
| D. Late-S MDL | 仅最后两层增加 gated S read | 长历史直接证据是否有额外价值 |
| E. `coupled` vs `split` | 字段、参数预算和数据完全一致 | 收益来自 initializer bypass 还是动态读取 |
| F. Scene 2×2 | content-side scene on/off × Domain scene on/off | scene 信号换通道造成了多少收益 |
| G. Cache on/off | logits/grad 必须一致，只比较性能 | 请求缓存是否真正等价、收益是否随 candidates/request 增长 |

质量至少按 `task × scene` 报告 AUC/QAUC、logloss 与 calibration；系统至少报告 samples/s、P50/P99、峰值 HBM、host RSS、cache bytes/request。没有 equal-readout 与固定 scene policy，就不能把结果归因给 MDL-OneTrans。

---

## 3. 复现过程中最值得讲的 8 个问题

### 3.1 多出来的一次 Add & Norm，为什么让“论文复现”变成了另一个模型？

**问题：** 早期 `MDLRankMixerBlock` 直接复用了完整 RankMixer block。代码能前向、能反向，但 Feature self-interaction 比 MDL Eq. 6 多了一条 residual 路径。

论文口径是：

\[
F_{mix}=\operatorname{LN}(\operatorname{TokenMix}(F)+F),
\qquad
F^{l+1}=\operatorname{FFN}(F_{mix})
\]

早期/当前生产稳定变体则是：

\[
F^{l+1}=\operatorname{LN}
\left(\operatorname{FFN}(F_{mix})+F_{mix}\right)
\]

**根因：** 实现时把“复用 RankMixer 成熟 block”与“逐式复现 MDL”当成了同一件事。这个差异不会触发 shape error，也不会阻止收敛，所以比 crash 更危险。

**解决方案：** 提交 `edaf19d` 把两条路径显式拆开。

| 模式 | 计算 | 用途 |
|---|---|---|
| `direct_ffn` | FFN 输出直接作为下一层 Feature state | MDL Eq. 6 对齐 |
| `residual_ffn` | FFN 外保留第二次 residual + norm | 当前生产稳定变体 |

同时用结构测试固定：Domain 是 Query，Feature 是 Key/Value；Scenario/Task 使用各自 Per-token FFN；Task 读取 active + global Scenario；只有 Task state 进入输出 head。

**影响：** 从此“论文同型实现”和“生产稳定实现”可以分别训练、分别报告，不再用同一模型名混口径。

**边界：** 仓库没有证明 `direct_ffn` 或 `residual_ffn` 的质量一定更好。我们解决的是**模型身份与实验归因**，不是替效果实验下结论。

### 3.2 只改一个 Embedding dim，为什么后面所有 Feature token 都可能变了？

**问题：** 早期 tokenizer 先拼接全部 embedding/sequence summary，再按总宽度等分。一个字段从 32 维改到 64 维，会移动后续所有 slice 边界；为满足整除，token 数甚至可能从 32 静默降为 8。

**根因：** token identity 由“当前 raw width 的等分位置”决定，而不是由业务语义决定。于是一次 embedding 容量调整，实际同时改了：

- 字段在哪个 token 内先融合；
- 一条历史是否被切成两半；
- 不相关字段是否被塞进同一投影；
- 后续 Token Mixing 的归纳偏置。

**解决方案：** 生产 `mdl_rankmixer` 改为固定 groupwise contract：

| 组成 | 数量 | 例子 |
|---|---:|---|
| 非序列语义组 | 23 | request environment、query identity、item category/goods/price、retrieval evidence、relevance 等 |
| 历史组 | 9 | `impr`、`clk_long`、`view_long`、`cart_long`、`buy_long`、`semi_clk`、`srch_q2i`、`ups_clk_sku`、`flatten_query_hash` |

每组使用独立 `Linear(group_width, 768)`，总体仍保持 `32×768`。字段增删只影响所属组，不再平移所有后续 token。

**验证：** 配置生成测试固定 32 个组及每组字段；模型测试检查字段宽度变化不会改变其它组边界；旧 slice tokenizer 只保留为兼容 fallback。

**影响：** 后续 embedding shape、字段增删和 MDL 算法消融可以解耦。这个改动首先提升实验可解释性，不应直接写成 AUC uplift。

### 3.3 14 个字段放错物理轴，为什么所有长度检查仍可能通过？

**问题：** agg Parquet 一条物理 row 同时包含 request、candidate 和 sequence/event 三个轴。曾有 14 个字段以及 coarse search/recommendation 路由被按错误轴展开，但产出的 tensor shape 仍可能合法。

**根因：** 旧逻辑用“字段属于 context 组还是 item 组”推断物理轴；实际物理存储轴与模型语义分组不是一回事。错误广播后，长度可能恰好等于候选数或请求数，因此不会 crash，只会把错误值喂给模型。

此外还有两类容易一起训错的空值语义：

- outer `null` / `[]` 表示 zero-length bag，不是结构错误；
- 脱敏 full-width row 中的 `0` 是活数据替代值，不是可按值删除的 padding；
- dense `null` 与真实数值 0 必须用 presence bit 区分；sequence 的 null anchor 才能删除整个 event step。

**解决方案：** 提交 `053257b`、`81a53d1`、`a729b00` 建立显式三轴合同。

| 层次 | 规则 |
|---|---|
| 物理读取 | request/candidate 轴由 adapter plan 明确定义，不由 token group 反推 |
| 派生字段 | `coarse_scene_*` 等请求级派生列显式标记 request axis |
| 打包 | 最后一步才按 membership 将 request 数据广播到 candidate |
| 空值 | `null`、`[]`、真实 0、padding 0 分别处理；同一序列所有字段共享 selection plan |
| 性能路径 | axis-separated direct path 不再重建整张 flat Arrow candidate table |

**验证：** 完整 mock 的第一批中，legacy 与 direct 对 **511 个 candidates、178 个 feature/sequence entries**，以及 labels、masks、scenario IDs、group IDs、prediction keys 全部逐值一致。

**影响：** 这是典型的“shape 正确、语义错误”。对于推荐数据链路，模型单测不能替代 axis-level oracle。

### 3.4 一个 HDFS `pread` 卡死，为什么最后表现成所有 GPU 一起不动？

**问题：** 多卡 HDFS 流式训练中，某个 reader 的 native `pread` 偶发永久不返回；该 rank 不再产生 batch，其它 rank 随后卡在 collective，日志只剩 GPU 空转或 NCCL timeout。任务结束时还可能卡在 reader/queue/process-group teardown。

**根因：** 这里不能直接套用常见的“fork 继承 libhdfs JVM”解释。仓库中的实际证据指向：

1. 并发 worker 复用 `HadoopFileSystem`/Parquet session，native handle 可能损坏；
2. timeout thread 仍持有 generator 时，再次 `next()` 会报 `generator already executing`；
3. 对仍有挂起 `pread` 的 poisoned session 调 native `close`，close 本身也可能永久挂起；
4. 训练前 scene discovery 在冷 HDFS 上耗时很长，若直接走 NCCL 广播，会把 IO 冷启动与 GPU collective 绑死；
5. **更根本的一层：每步取 batch 的等待没有预算。** 训练循环其实早就支持「某个 rank 这步没有 batch」——它照常投票、照常进集合通信，然后用上一个 batch 做零损失重放来保持 DDP 对齐（本来是给 epoch 末尾各 rank 先后读完准备的）。但 `next(batch_iterator)` 会一直阻塞，慢的 rank 永远走不到投票那一行，也就没机会告诉其它 rank「这步跳过我」。前面几条修的都是「如何发现这个阻塞不会结束」，而不是「阻塞本身为什么必须致命」。

**解决方案：**

| 手段 | 作用 |
|---|---|
| thread-local HDFS client | 不在长期 prefetch workers 间共享 native client |
| timed open/start/batch + controlled row-group concurrency | 给每个远程阶段独立预算，限制同时悬挂的 native 调用 |
| poisoned session quarantine | timeout 后不在原 generator 上重试，也不 native-close；新建 session，从未产出 batch时才安全重开 |
| `spawn` host-prepare child + process-group kill | startup/idle 超时后杀整个子进程树，避免 D-state 子进程拖住父 rank |
| `REMOTE_IO_STALL -> exit 70` | 让平台按可重启 IO 故障处理，而不是无限等 NCCL |
| Gloo control group | scene discovery 使用 Arrow unique 扫描后经 CPU control group 广播，避免冷 HDFS 阻塞 NCCL |
| bounded teardown | 先关 IPC pipe，再 join CUDA prefetch；删除无界 queue join，并限制 `destroy_process_group` 等待 |
| **每步读取预算 + 饥饿投票** | `reader.step_batch_budget_sec`（默认 30s）到期后返回 `BATCH_NOT_READY` 而非继续阻塞；该 rank 这步投「饥饿」，其它 rank 照常训练，它欠的那个 batch 下一步再消费，一条数据都不丢 |
| **三态投票（active / starved / exhausted）** | allreduce 从 0/1 计数改成 `[active, exhausted]` 两个计数器 |

**为什么必须是三态：** 原来 `active_ranks == 0` 的语义是「数据读完了」，直接 `break`。四个 rank 共用同一个 HDFS 集群、同一份内存压力，同时卡住完全可能——如果那时大家都因超时投 0，会被误判成 epoch 结束，**训练静默地提前终止**。所以「暂时没有」和「永远没有」必须分开：只有全员 exhausted 才结束；全员 starved 则一起重试；有人还在训练时，饥饿的 rank 做零损失重放。判定逻辑抽成了纯函数 `_supply_verdict()` 以便单测覆盖。

另外第 0 步的预算强制为 `None`——replicated sparse DDP 要求每个 rank 都提供首个 batch，此时只能等，由 startup timeout 兜底。

**一个此前无效的计时器：** `host_prepare_idle_timeout_sec` 曾把「距上次交付 batch」和「距上次子进程心跳」取 `min`。子进程每读一个 record batch 就心跳，`io_progress_pulses` 还每 15s 补一次，所以这个值恒 ≤ 15s，**idle timeout 从来没能触发过**，600s 的 step watchdog 是唯一后盾——而它会把四个 rank 全杀掉。现在 startup 仍认心跳（冷启动阶段 HDFS list/footer/adapt 确实只有心跳，且 JNI 挂死会让心跳停），idle 只认真正交付的 batch：首个 batch 之后，「活着但不产出」恰恰就是该中止的状态。

**唯一天花板：** 饥饿容忍不能是无限的，否则一个彻底卡死的 reader 会永远投饥饿（重试循环每轮都会 beat step watchdog，600s 那道后盾不再生效）。为避免两个上限互相抢先，这里只保留一个：`host_prepare_idle_timeout_sec`，默认从 300 提到 900。**提高默认值实际是收紧**——修复前它等效于无穷大。配置校验强制它大于 `step_batch_budget_sec`，防止配出「来不及饥饿一次就被判死」的组合。

**影响：** 修复目标不是“永远不遇到坏 HDFS”，而是分两层：先把不可观测的永久挂起转换成**有阶段、有 HBM 快照、有退出码、可自动重启**的失败；再把爆炸半径从「一个 rank 慢 → 全 job 死」降到「一个 rank 慢 → 吞吐掉一个坑」。代价是被跳过的那步在该 rank 上多做一次零损失前反向，远小于丢掉数万步进度重启。行/token 计数只在 `rank_active` 时累加，所以重放不会污染指标；`starved_steps` 非零时会打进 Train step 日志，避免降级完全静默。

**边界：** 这不会让 reader 变快，HDFS 抖动和内存压力仍在（见 3.5）。评测路径**不**做饥饿容忍——跳过一个 rank 会静默丢掉测试行、污染 AUC，所以 `_active_rank_count()` 保持阻塞语义。`host_prepare_prefetch=0` 的 in-process 路径没有预算读取，退回阻塞 + step watchdog。

**对应提交：** `705878d`、`48f243a`、`fd5edb6`、`457d2e6`、`8c31ae3`；饥饿容忍与 idle 计时器修复随 `50ed954` 提交（该 commit 的 message 只描述了 checkpoint resume，实际混入了本节与 3.5 的改动）。

### 3.5 Python 对象都释放了，为什么 DataLoader RSS 仍持续上涨？

**问题：** 长跑时 host-prepare/parent RSS 持续抬升；Python 引用已经释放，GC 也看不到等量存活对象。最终可能先耗尽 container RAM，而不是 HBM。

**根因：** 这里也不是示例中的 glibc arena 单一问题，而是多条独立的 high-water / 单调增长 path：

| 内存来源 | 为什么不下降 |
|---|---|
| `share_memory_` + pinned IPC | child 中先 pin 再 share，会让 `/dev/shm` handle 与 pinned page 跨 batch 存活；shared `FeatureBatch` 未及时私有化也会延长底层 storage 生命周期 |
| 变长 `pin_memory()` | 每个不同 batch/sequence shape 都申请新的 size class；CUDA host caching allocator 保留旧 slab，RSS 随新高水位只升不降 |
| `_PinnedHostBufferPool` 只扩不缩 | idle slot 按**历史峰值**保留，遇到一次大 batch 就永久上棘轮 |
| HDFS poisoned session quarantine | 为避免 native `close` 挂死而永久持有整个 `ParquetSession`，连同已 pre-buffer 的 Arrow 数据与 exception traceback 一起留住——**单调增长**，且与内存压力构成正反馈：越挤 → 越多 timeout → 越多 quarantine → 越挤 |
| 线程 churn × glibc arena | `scanner_batch_rows=128` 让每个小 batch 都起一个 timeout 线程，glibc 为每个线程新建 arena，碎片与虚拟保留一起抬 RSS |
| `_ARANGE_CACHE` / `_NP_ARANGE_CACHE` | 按 length 做 key 的 dict，`group_total` 每出现一个新长度就多留一个 buffer，长跑下 key 空间不收敛 |
| `PerFileLock._thread_locks` | class 级 dict 按 file URI 累积 `RLock`，每个新文件一把，永不回收 |
| parent 侧缺 Arrow / heap trim | 主循环只 flush PyTorch host cache，没有 `pa.default_memory_pool().release_unused()` 与 `malloc_trim`，in-process 数据路径的空闲页留在进程里 |

**解决方案：**

| 手段 | 具体处理 |
|---|---|
| 默认 IPC 改为 memfd | child 只写匿名 memfd，parent 读取后再 pin；不把 pinned tensor 直接放入 shared memory |
| parent 立即私有化 | shared FeatureBatch 出队后 clone/privatize，并删除 IPC payload 引用 |
| recycled pinned pool | 少量 slot 按 dtype 复用 storage；lease 归还后按**滑动高水位**丢掉过大 idle slab（可选 `MDL_PINNED_POOL_MAX_SLOT_BYTES`），避免跟历史尖峰永久上棘轮 |
| H2D 后释放 host refs | 防止 device prefetch 队列继续持有上一个 batch |
| 周期性清空 idle host allocator slab | 在 log cadence 触发 host cache release；pool grow/drop 时也主动归还旧 slab |
| 收紧 memfd/Arrow cleanup | 不生成第二份巨型 bytes copy，关闭 mmap/fd/table iterator |
| quarantine 分级释放 | 只永久保留可能被 native 调用触碰的 `filesystem` / `native_file`；timed operation 一结束就放掉 `batch_iterator`、`parquet_file`、`RecordBatch` 与 traceback（改存 `repr(failure)`），把「单调增长」降回「常数级 handle」 |
| timeout 线程池 | 用可复用的 daemon worker 池替代每次调用起一个线程；挂死的 worker 自行退休，不回池 |
| `MALLOC_ARENA_MAX=2` + `malloc_trim(0)` | 在 `import torch` 之前设好 arena 上限（父子进程都生效），并在 child 与 parent 的 `gc.collect()` 之后主动 trim |
| arange 缓存改为单 buffer | 不再按 length 建 dict，而是一个按需翻倍增长的 buffer，取 view 返回；已发出的 view 在增长后仍然有效 |
| `PerFileLock` 改用 `WeakValueDictionary` | 没有活的 `PerFileLock` 持有时，锁自动从 registry 清掉；同 key 的活实例仍共享同一把锁 |

**验证与边界：** `331b6ac` 与 `64ba075` 分别修复 IPC ratchet 和变长 pinned allocator ratchet；后续 pool 从 grow-only 改为滑动缩容（见 [`mdl_key_questions.md`](./mdl_key_questions.md) 问题 8）。上表后半部分的 quarantine 分级释放、线程池、arena/trim、arange 与 lock registry 随 `50ed954` 提交。回归覆盖 memfd 传输、pool 复用/尖峰缩容、quarantine 释放、线程池复用、arange 视图有效性、lock registry 回收、子进程退出和 host-prepare watchdog。仓库没有一条可引用的完整“修复前后 24 小时 RSS 曲线”，因此只陈述根因与回归，不编造下降百分比。

**为什么值得单独讲 quarantine：** 其余几条是「高水位不回落」，而 quarantine 是真正的**单调泄漏**，并且会自我放大——它是把「HDFS 偶发抖动」升级成「几小时后必然填满 250GB 内存、进而让 HDFS 读取更容易 timeout」的那个环节。3.4 的饥饿容忍处理的是卡顿的**表现**，这一条处理的是卡顿变频繁的**成因**，两者要一起看。

另一个重要 trade-off 来自 direct pipeline：同一 data-only 基准吞吐从 361.28 提升到 475.85 samples/s（+31.71%），但当时 peak host RSS 从 1.603 GB 增至 2.588 GB（+61.47%）。数据路径优化必须同时报告吞吐与内存 runway。

### 3.6 10 个文件算出来的“好表”，为什么放到 24 小时会需要 399.49 GiB/GPU？

**问题：** 在少量 Parquet profile 上按 observed distinct 直接定 bucket，短跑很容易装下；训练窗口扩到线上 24 小时、约 12,000 文件后，高基数 goods/sku/history 表继续增长，原 shape 会产生严重碰撞，而低碰撞理想 shape 又完全放不进 H100。

**根因：**

1. 10/100 文件没有覆盖长尾，不能假设 distinct 已饱和；
2. shared table 必须按所有 alias source 的 union 估算，不能只看 root 单列；
3. Embedding weight、optimizer state、临时通信和 dense activation 竞争同一 HBM；
4. 标准 Adagrad 的 accumulator 为 $O(ND)$，高基数表的状态可能与权重同量级。

**解决方案：** 以 10/100 文件观测拟合逐字段 power-law 增长，并外推到 12,000 文件；部署时在碰撞、维度和显存之间显式做预算。

| 约束 | 当前策略 |
|---|---|
| 长尾外推 | 每字段增长指数限制在 0～0.95，保留 1 小时 profile 下限 |
| bucket | 24h projected load 上限 1.75，向上取 2 的幂；硬上限 (2^{30}) |
| dim | projected distinct 超过 1M 时，高基数表生产 dim 上限 32 |
| shared table | 按所有来源 union 估算，只计算一次物理表 |
| optimizer | sharded Row-Wise Adagrad，FP32 state 从 $O(ND)$ 降为 $O(N)$ |
| 二阶段压缩 | task-prior 共享表以及 goods/uid/sku/query dim cuts |

**结果：**

- 24h load 0.5 + 理想 dim：两卡约 **399.49 GiB/GPU**；即使四卡仍约 **199.74 GiB/GPU**，不可训练；
- 首轮 24h 可部署审计：约 **65.01 GiB/GPU**；
- 加入 task-important identity 后，当前 coarse MDL 约 **66.24 GiB/GPU**（BF16 weight + Row-Wise Adagrad，2 GPU）。

**代价必须明说：** 该方案不是“无损压缩”。均匀哈希假设下，主 `goods_id`、`sku_id`、`cart_long.sku_ids` 的理论 distinct collision 分别约 **41.2% / 51.5% / 44.7%**。如果碰撞成为质量瓶颈，下一步应考虑高频精确词表 + 尾部 hash、remix 或参数服务器，而不是继续盲目放大本地表。

### 实际遇到的 RankMixer / MDL-RankMixer 显存问题

> 本节与 §2.7 OneTrans 表对照阅读，**不占第 3 节 8 个核心问题的名额**。  
> 静态表预算见 3.6；host RSS / pinned IPC 见 3.5（两族模型共用数据路径）。  
> 口径：`rankmixer` 与 `mdl_rankmixer` 同属浅层 32×768 / 2-layer 家族；MDL 额外 Domain 与 scoped embedding 抬高静态占用，但运行时峰值机制大体相同。

| 现场问题 | 根因 | 修复 |
|---|---|---|
| 24h 理想 embedding ≈399 GiB/GPU，两卡直接不可训 | 短 profile 低估长尾；标准 Adagrad state 为 \(O(ND)\)；shared root 若只按单列估会漏算 alias | load≤1.75、大表 dim 封顶 32、sharded Row-Wise Adagrad、shared union；粗粒度 MDL 落到约 65～66 GiB/GPU，非 MDL RankMixer 约 57.86 GiB/GPU（详 3.6；`859902e`、`1a0579d`、`f8e0dad`） |
| 把 per-rank batch 拉到 1024 后，LONGER / 序列投影峰值失控 | 9 条历史最长到 2048，整 batch 一次投影会让激活随 \(B\times L\) 线性涨；`device_prefetch_batches=2` 还会在执行当前 batch 时再钉住一份大序列 | 按 token 预算切块 LONGER（`sequence_encoder_chunk_tokens`）与序列投影；prefetch 降到 1；flash varlen + compact packing（`9529ae2`）。当前生产再调到 `mdl_rankmixer` batch **1536** + length bucket（1536/960/640/480/768） |
| 打开 full activation checkpoint 后，6/8 卡 util 反而崩 | 浅层 2-layer dense 重算成本高，且 full remat 挡住 `cuda_graph_backbone`，无法用 dense 计算掩盖 emb A2A | 曾为省激活全开 full remat（`659ad1c`），后对 RankMixer 族回到验证过的 `activation_checkpoint=none + cuda_graph_backbone`（`bd4a367`）。这与 OneTrans「full remat 仍 mid-run OOM」是同一旋钮的不同失败模式 |
| 8 卡时 NCCL / emb staging 挤占激活余量 | 大 world size 下 NCCL buffer/channel 与 emb A2A scratch 与激活同相；OneTrans 曾在 8×1024 backward OOM | ≥6 卡才 cap NCCL buffer/channel；emb A2A 设 `MDL_GROUPED_EMB_MAX_OUTPUT_MIB`；大 HBM 上 RankMixer 族默认保持满 per-rank batch，仅 ≤32 GiB 卡在 7–8 GPU 做 mild derate（`6a768f8`、`bd4a367`） |
| 为 HBM 加的安全旋钮反过来饿死 2–4 卡 util | NCCL cap 与强制 memfd 被无条件套到小 world；`/dev/shm` 充足时 RankMixer 仍走 memfd 拷贝 | NCCL 保险仅保留在 ≥6 GPU；host-prepare `auto` 在 shm 充裕时回 `share` IPC，小容器才回退 memfd（`bfaba79`） |
| CUDA graph 路径看起来在跑，dense 参数却可能没进 DDP | graph wrapper 用普通属性藏 live modules 时，`make_graphed_callables` 能执行，但 static_graph reducer 看不到完整参数面 | 注册精确 modules、预热全部 train bucket shape，并加两卡 graph/eager 交错回归（`3e53d18`）。这首先是正确性事故，但也锁死了 RankMixer「none + cuda graph」这条生产 HBM/util 剖面 |

**和 OneTrans 对照时的结论：**

| 维度 | RankMixer / `mdl_rankmixer` | OneTrans / `mdl_onetrans` |
|---|---|---|
| 静态大头 | embedding + Row-Wise state（MDL 因 Domain scope 更高） | 同左，另加长 S 的激活/cache |
| 动态大头 | LONGER 与序列投影、device prefetch、emb A2A scratch | varlen pack、多层 S cache、Domain×S cross-attn |
| 默认省激活旋钮 | **不要** full remat；用 none + CUDA graph | **不要**盲开 full remat；用 none + fixed packing 基线 |
| 当前生产 batch | 1536（`rankmixer` 1280） | 1408 |
| 不能外推的点 | 短跑 peak ≠ 长跑；world size 改变 NCCL/emb 占用 | 同左，且 candidates/request 会放大 cache |

真正的结论与 §2.7 相同：**安全 batch 必须在真实长度分布、embedding 预算、chunk/prefetch、allocator、world-size profile 与完整长跑下联合验证**；不能把「降 batch」或「开 checkpoint」当成单调更省的旋钮。

### 3.7 Loss 明明在下降，为什么每张卡可能训练的是不同 Embedding？

**问题：** 大表启用 `nn.Embedding(sparse=True)` 后，backward 产生 row-sparse COO grad。标准 NCCL DDP 不会像 dense grad 一样自动 all-reduce；如果只是把 sparse parameter 从 DDP reducer 中排除，每个 rank 会根据自己的样本更新不同的行，loss 仍可下降，但 replicas 已经静默分叉。

**根因：** “参数属于 embedding optimizer”与“梯度由 DDP 同步”是两套所有权。把 embedding 交给 Adagrad，并不等于 sparse COO 已经跨卡同步；强行走 dense all-reduce 又会把超大表梯度实体化，失去 sparse 的意义。

**解决方案演进：**

| 阶段 | 方案 | 解决的问题 |
|---|---|---|
| replicated sparse | rank 0 广播初始表；按稳定参数名校验 metadata；每步收集 touched rows、coalesce 重复行、跨 rank gather 后平均更新 | 保证数学上与全局 batch 的稀疏更新一致 |
| sharded embedding | 每张卡只持有 owned rows；ID 按 owner 路由，lookup/gradient 使用 all-to-all | 权重与状态不再每卡完整复制 |
| Row-Wise Adagrad | 每行一个 FP32 accumulator | optimizer state 从 $N\times D$ 降到 $N$ |
| collective fusion | 不同 dim 的表按精确宽度打包，约 20 次小 collective 合并为约 4 次 | 降低 latency 与 launch overhead |

**验证：** 两卡测试覆盖不均匀 touched rows、重复 ID、padding row、replica equality、sharded lookup 与 reference full table 的 forward/backward/update 一致性。CUDA OOM rank 还会 hard-abort，使 peers 快速失败而不是继续等 collective。

**影响：** 这类问题比显式 OOM 更危险：训练曲线可以正常、dense 参数也一致，只有 embedding 在悄悄分叉。多卡推荐训练必须把“稀疏参数同步正确性”作为独立 correctness gate。

**对应提交：** `cf7019b`、`1a0579d`、`079946d`。

### 3.8 AUC “稳态不动”时，为什么第一步不是调层数，而是先推翻评测窗口？

**问题：** 旧 `quick_eval` 每 1000 step 从 train split 抽最多 20 个 batch。不同 step 看到的样本、正例率、request/candidate mix 都可能不同，因此 AUC 曲线同时包含“模型变化”和“评测集变化”。这种曲线无法判断 plateau、collapse 或超参收益。

**根因：** 为追求低开销，把训练中监控当成了可比较 holdout。多卡下还存在 rank shard 长度不齐、稀有任务正负样本不足，以及评测 reader 状态污染训练 reader 的风险。

**解决方案：** 提交 `efac1a5` 将其替换为 fixed held-out eval。

| 项目 | 旧 quick eval | 当前 fixed test eval |
|---|---|---|
| 数据 | train split 上滚动 batch | 启动时冻结、按 rank 确定性分片的 test manifest |
| 频率 | 每 1000 step | 每 5000 step |
| 范围 | 最多 20 batch | 完整读取冻结 manifest；当前 4 files/rank |
| 多卡尾部 | 容易 rank 提前结束 | exhausted rank 重放 dummy batch 以保持 collective 对齐，指标只累计 active rank |
| 指标 | task histogram AUC | task AUC、logloss、prob mean、logit mean/std、examples/positives/negatives |

同一提交还把 dense/sparse LR 从 $10^{-3}$ 降到 $10^{-4}$，warmup 从 500 延长到 5000。这是结合近期 sweep 更新的训练默认值，但仓库没有保存足够的固定 holdout 对照，不能把它写成“已经证明解决 AUC 坍缩”。

**当前边界：**

- 当前训练中 fixed eval 仍只有 task-level 指标，没有 task×scene、QAUC/UAUC 或置信区间；
- `files_per_rank` 曾扩到 25，随后为平台 GPU-util 保护窗口缩回 4；低频任务的方差需要重新评估；
- 当前已删除 COPC，不能再引用旧文档中的 COPC 输出；
- 仓库没有可信的固定 holdout AUC 时间序列，所以不能声称“六层坍缩、四层到 0.7”。

**影响：** 在评测窗口固定之前，任何“层数、学习率、warmup 解决了 AUC 不动”的故事都缺少因果证据。先让曲线可比，再讨论优化问题。

---

## 4. 13 个备选的高吸引力问题

下面 13 个题目均来自现有实现或提交，可替换进主讲；每个都可以独立展开成 3～5 分钟。

| # | 题目 | 真正值得讲的 insight | 证据/切入点 |
|---:|---|---|---|
| 1 | **CUDA Graph 明明 replay 成功，为什么 Dense 参数可能一次都没被 DDP 同步？** | graph wrapper 用普通属性藏住 live modules 时，`make_graphed_callables` 能执行 forward/backward，但 DDP 看不到完整参数面；性能优化可以直接破坏梯度语义 | `3e53d18`：注册精确 modules、预热全部 bucket shape，并增加两卡 graph/eager 交错回归 |
| 2 | **Batch 从 768 降到 512，为什么 OneTrans 仍可能在长跑中 OOM？** | OOM 由真实长度分布、reserved fragmentation、packing、checkpoint 和 cache 生命周期共同决定；短 smoke 的 peak 不是长跑上界 | `304c1d0`、`f6fdb68`；约 step 1300、约 12 GiB reserved fragmentation |
| 3 | **Batch size 写的是 candidates，为什么 shuffle、cache 和内存生命周期必须按 requests 设计？** | 推荐 batch 同时存在 request 数 $R$ 与 candidate 数 $B$；S/history 是 $R$ 级，NS/logit 是 $B$ 级，把两者混成一个 batch 轴会重复计算并放大内存 | `request_row_indices`、request-sized S cache、axis-separated adapter |
| 4 | **一个 100 候选请求，真的应该比 1 候选请求贡献 100 倍梯度吗？** | 当前 `mean_per_task` 解决任务 mask/样本数归一化，但仍主要按 candidate 计权；若线上目标更接近 request-level，loss objective 需要显式 request normalization | 现有 `group_id`/request membership 与 loss reduction 合同；适合作为开放算法题 |
| 5 | **尾部任务学不动，是梯度冲突，还是它一开始就被乘了 0.01？** | 当前 task weights 为 `0.5 / 0.01 / 0.01`；讨论梯度冲突前，先审计静态缩放、正例率和有效 batch 规模 | 生产 YAML 的 `task_loss_weights`；应报告每任务未加权 loss/grad norm |
| 6 | **有 3 个 Task token，为什么模型仍然没有显式学习“加购→支付”的任务依赖？** | token 分开不等于建模 task dependency；当前 Task 都读 Feature/Scenario，但没有 Task-to-Task attention 或有向漏斗约束 | `MDLDomainBlock` 的信息流；可讨论 stop-gradient、有向 task graph 与 leakage |
| 7 | **价格、CTR、Count 明明是数字，为什么当前 147 个主非序列字段里没有一个连续特征？** | 字段名不等于模型 dtype；当前这些值是预哈希类别 ID，真正连续输入只来自 9 条历史的 `log1p(time_delta)` | [`current_field_processing_report.md`](./current_field_processing_report.md) 文件头的当前口径；适合讲错误 dtype 如何制造伪复现 |
| 8 | **0 到底是 padding，还是脱敏后的真实值？删错一次为什么整条历史都变短？** | full-width mock 的 0 是活数据；embedding row 0 又是 padding。必须在物理数据、编码 ID 与 mask 三层分清，不能 value-based compact | `a729b00`、`DATA_FORMAT.md` full-width contract |
| 9 | **`null`、`[]`、`[0]` 看起来都“空”，为什么必须严格三分？** | 它们分别可能表示缺失 bag、零长度、存在一个 padding slot；对 mean denominator、presence 和 sequence alignment 的影响不同 | `81a53d1` 与 null semantics tests |
| 10 | **最多 256 个 fine scenes，但每条样本只有一个 active scene，为什么还要算 257 个 Scenario states？** | MDL 减少的是输出 head 数，并未自动消除中间 Domain 参数与 FLOPs；active-token execution 是下一阶段系统问题 | fine config `max_discovered=256` + global token；需要 dense/active path 分开报告 |
| 11 | **增量 cache 命中率 100%，为什么输出仍可能和 full recompute 不一致？** | pyramid 改变深层窗口时，旧 token 的深层表示也会变化；只按 raw append 判断 K/V 可复用会留下 stale cache | `extend_precomputed_s` 的 overlap tensor/mask 校验与 rebuild 分支 |
| 12 | **数据路径快了 31.71%，为什么 GPU util 最终仍只有约 11%？** | Amdahl 定律：direct tensorization 消掉一段等待后，Arrow→Python adapter normalization 成为主瓶颈；局部 fast path 不等于端到端供数能力足够 | [`AGG_DIRECT_RESULTS.md`](../AGG_DIRECT_RESULTS.md)：独占 4090 E2E +16.6%，direct util 10.79%，adapter normalization 仍是最大 CPU 项 |
| 13 | **RankMixer 开了 full remat，为什么 6/8 卡 util 反而更差、也不一定更省 HBM？** | 浅层 2-layer 重算贵，且挡住 CUDA graph，dense 无法掩盖 emb A2A；同一「省激活」旋钮在 RankMixer 与 OneTrans 上失败模式不同 | `659ad1c`→`bd4a367`；对照 RankMixer/MDL 显存节与 §2.7 |

### 备选题选择建议

- 偏算法听众：4、5、6、10、11；
- 偏训练系统听众：1、2、3、12、13；
- 偏特征与数据平台听众：7、8、9；
- 若只有 10 分钟加餐，优先选 1、3、10、13：都属于“系统看起来正常，但模型、显存或成本口径已经错了”。

---

## 5. 建议的串讲顺序

一个 60 分钟版本可以这样组织：

| 时间 | 内容 | 目标 |
|---:|---|---|
| 5 min | 当前两个模型与证据边界 | 先区分论文复现、生产变体和实验性组合 |
| 18 min | MDL-OneTrans 专题 | 按难点讲：sidecar、读层取舍、coupled/split、request cache、OOM、线上变量；底稿见 [`mdl_onetrans_adaptation_hardships.md`](./mdl_onetrans_adaptation_hardships.md) |
| 28 min | 8 个核心问题中精选 6～8 个；穿插 RankMixer 显存表与 §2.7 对照 | 每题只讲现象、根因、方案、验证；强调两族模型 HBM 失败模式不同 |
| 6 min | 当前仍不能下的质量结论 | fixed holdout、equal-readout、task×scene 指标缺口 |
| 3 min | 讨论 | 聚焦线上迁移变量与下一轮实验 |

如果时间只有 30 分钟，建议保留：

1. Eq. 6 多一条 residual；
2. 三轴错位为何不 crash；
3. sparse DDP 静默分叉；
4. OneTrans sidecar 与 request-sized cache；
5. 399.49 GiB 的 24h Embedding 预算，以及 RankMixer「none + CUDA graph」与 OneTrans「none + fixed packing」两条不同的 HBM 剖面；
6. fixed holdout 之前不能讲 AUC 坍缩。

## 6. 当前结论与下一步

已经完成的，是一套能够继续做严肃实验的实现基础：

- MDL 论文同型计算图与生产稳定变体已显式分离；
- Feature token、Domain initializer、request/candidate/event 轴都有可检查合同；
- HDFS、host RSS、稀疏多卡与 OneTrans cache 的主要长跑故障已有针对性解法；
- RankMixer / `mdl_rankmixer` 已形成可训的静态 embedding 预算，并用 LONGER chunk、prefetch=1、`none + CUDA graph` 与 world-size profile 压住运行时峰值；
- `mdl_onetrans` 已形成 sidecar layer-wise Domain 方案，并解决了候选展开造成的 cache/HBM 放大；
- 训练中评测已经从滚动 train quick eval 切换到冻结 test manifest。

仍未完成、且不能用措辞掩盖的，是效果归因：

1. `mdl_rankmixer` 的 `direct_ffn` 与 `residual_ffn` 固定 holdout 对照；
2. `mdl_onetrans` 的 equal-readout、NS-only、late-S、`coupled/split` 与 scene 2×2；
3. task×scene AUC/QAUC、calibration、置信区间和低频任务样本量；
4. 24h hash collision 对质量的真实影响；
5. 线上 P99、cache bytes/request 与 candidates/request 扩展曲线。

这次复现最值得带走的结论不是某个 trick，而是：**推荐模型最危险的失败往往既不 crash，也不阻止 loss 下降。只有把计算图、数据轴、分布式状态、缓存粒度和评测窗口都写成可验证合同，结果才有资格被解释。**

## 附录：关键证据索引

| 主题 | 主要材料 |
|---|---|
| MDL 原理与整体叙事 | [`document.md`](../document.md)、`paper/MDL/main.tex` |
| Domain 字段、groupwise token、coupled/split | [`mdl_token_feature_design.md`](./mdl_token_feature_design.md) |
| MDL→OneTrans 适配难点 | [`mdl_onetrans_adaptation_hardships.md`](./mdl_onetrans_adaptation_hardships.md) |
| 关键问答（问题/根因/方案） | [`mdl_key_questions.md`](./mdl_key_questions.md) |
| 当前字段与 S/NS 处理 | [`current_field_processing_report.md`](./current_field_processing_report.md)；文件头已标出历史快照边界 |
| 三轴数据合同与空值 | [`DATA_FORMAT.md`](../DATA_FORMAT.md)、`tests/test_agg_direct.py`、`tests/test_null_semantics.py` |
| direct pipeline 正确性与性能 | [`AGG_DIRECT_RESULTS.md`](../AGG_DIRECT_RESULTS.md) |
| 24h Embedding 规划 | [`embedding_shape_24h_audit.md`](./embedding_shape_24h_audit.md)、`emb_bucket_recommendation_growth.json` |
| MDL/OneTrans 结构测试 | `tests/test_model_alignment.py` |
| sparse/sharded 多卡 | `tests/test_sparse_ddp.py`、`tests/test_sharded_embedding.py` |
| HDFS 与 watchdog | `tests/test_remote_io.py`、`tests/test_host_prepare_watchdog.py`、`tests/test_step_watchdog.py` |
| reader 饥饿容忍与三态投票 | `tests/test_batch_starvation.py`、`tests/test_active_rank_count_async.py` |
| pinned pool 缩容与 host heap | `tests/test_pinned_host_pool.py` |
| CUDA Graph 参数面 | `tests/test_cuda_graph_static_graph.py` |
| 固定 holdout | `tests/test_fixed_test_eval.py`、`tests/test_evaluation_metrics.py` |

| 问题 | 关键提交 |
|---|---|
| MDL Eq. 6 与论文路径 | `edaf19d` |
| 三轴与空值语义 | `053257b`、`81a53d1`、`a729b00` |
| HDFS 挂起与 teardown | `705878d`、`48f243a`、`fd5edb6`、`457d2e6`、`8c31ae3` |
| reader 饥饿容忍、idle 计时器、host RSS 二轮 | `50ed954`（message 只写了 checkpoint resume，实际混入 3.4/3.5 改动） |
| host RSS | `331b6ac`、`64ba075` |
| 24h Embedding 与 Row-Wise | `859902e`、`f8e0dad`、`1a0579d` |
| RankMixer / MDL-RankMixer HBM | `9529ae2`、`659ad1c`、`6a768f8`、`bd4a367`、`bfaba79`、`3e53d18` |
| sparse DDP 与 collective fusion | `cf7019b`、`079946d` |
| OneTrans HBM/cache | `cc92f47`、`1eef46f`、`4861e0e`、`5e040d6`、`304c1d0`、`f6fdb68` |
| CUDA Graph DDP 参数面 | `3e53d18` |
| fixed heldout 与训练默认值 | `efac1a5`、`c188c82`、`5a7f1c1` |
