# MDL 关键问题集：问题 · 根因 · 解决方案

> 口径对齐仓库生产配置与现有专题文档（截至 `main`）。  
> 完整字段见 [`mdl_token_feature_design.md`](./mdl_token_feature_design.md)；OneTrans 适配见 [`mdl_onetrans_adaptation_hardships.md`](./mdl_onetrans_adaptation_hardships.md)；串讲底稿见 [`mdl_reproduction_lecture_report.md`](./mdl_reproduction_lecture_report.md)。

统一体例：**问题 → 根因 → 解决方案**；必要时补 **边界**（什么还不能下结论）。

---

## 问题 1：Scenario / Task token 由什么构成？

**问题：**  
论文只说 Domain 由 important raw features + related prior 初始化，没有给出工业字段清单。直接抄论文示例字段会对不上我们的搜推混部数据。

**根因：**  
论文给结构、不给字段。线上 Scenario 要同时覆盖搜索/推荐，Task 要对齐三个监督目标；important 是语义锚，prior 挂与该 Domain 相关的行为历史，一起初始化 Domain prompt。主干如何编码历史是另一条合同（RankMixer 主 UPS 走 LONGER，OneTrans 主 UPS 走 raw），不回答「Domain 由什么字段构成」。

**解决方案：**  

**Scenario（coarse：`search` / `recommendation` / `global`）**

| 组成 | 内容 |
|---|---|
| important | locale、页面/入口、`scene_id`（**global 不用**）、类目/价格等强锚；search 另含 `search_method_hn`，recommendation 去掉 |
| prior | coarse scene prior、scene 曝光统计；search/recommendation 挂 conditioned click 历史；global 挂 impr / clk / view 历史 |

**Task（`fst_cart` / `upid_pay` / `cateid_filter`）**

| 任务 | important（示意） | prior（相关行为历史） |
|---|---|---|
| `fst_cart` | 类目/价格、加购统计，可控 identity（mall/goods 等） | `cart_long` |
| `upid_pay` | 类目/价格、转化/订单/GMV 锚与用户购买倾向；**不含** mall/goods | 主 `buy_long` + 辅 `ups_clk_sku`（补 buy_long 空窗） |
| `cateid_filter` | 类目、相关性、query；**不含** goods/mall/price | `srch_q2i` |

取舍原则：

- **高频、可学习、语义锚** → important；**相关行为历史** → prior；
- 单 epoch 下极稀疏的主 `goods_id` **默认不进** scenario important；task 侧需要 identity 时用独立低基数 extra 表，禁止再复制 268M 主表。

**边界：**  
完整字段表见 [`mdl_token_feature_design.md`](./mdl_token_feature_design.md)。prior 选哪条历史需 holdout 消融，不是论文规定。

---

## 问题 2：为什么 Feature token 不能做任意等宽切片？

**问题：**  
早期实现把全部 embedding / 序列 summary 拼成长向量，再按总宽度静默切成 \(T\) 段。字段改 dim 或增删后，看起来 shape 仍合法，但 token 语义已经漂了；为满足整除，token 数甚至可能从 32 静默降成 8。

**根因：**  
论文要求 **semantically coherent clusters** 再投影。等宽切片的 token identity 由“当前 raw width 的等分位置”决定，而不是由业务语义决定。于是一次 embedding 容量调整，会同时改：

- 字段落在哪个 token；
- 一条历史是否被切成两半；
- 不相关字段是否被塞进同一投影；
- 后续 Token Mixing 的归纳偏置。

这不是 RankMixer 论文的默认合同，而是我们早期 tokenizer 的实现错误。

**解决方案：**  
生产 `mdl_rankmixer` / `rankmixer` 改为固定 **groupwise** 合同：

- 23 个非序列语义组 + 9 个历史组 = **32 token**；
- 每组独立 `Linear(group_width → 768)`；
- 字段增删或 dim 调整只影响所属组，不平移相邻 token。

`T=32, D=768` 是保留的**容量契约**，不是把模型静默缩成 8 token。

**边界：**  
groupwise 解决的是 token 身份与实验可解释性；仓库没有证明某一组字段划分一定带来更高 AUC。

---

## 问题 3：Domain 状态为什么要区分 `coupled` 与 `split`？

**问题：**  
线上若直接拿 `coupled` 的提升讲“Domain Attention 有效”，可能把 initializer bypass 算进去。

**根因：**  
默认 `coupled` 与 MDL 原始状态传播一致：

```text
important/prior -> Domain prompt
prompt == attention query == recurrent residual state == final readout state
```

即使把 Domain Attention 的 Value update 置零，strong initializer 仍能沿 residual 到达 logits。这是论文公式允许的路径，但会让“收益来自动态读取，还是来自 task-specific 强先验”含糊。

**解决方案：**  
额外实现诊断路径 `split`：

```text
prompt p = Projector(important, prior)      # 只控制 query
state h0 = learned readout seed             # 承接证据与最终 readout
query ql = p + hl
hl+1 = hl + Attention(q=ql, K/V=Feature 或 NS/S)
```

硬验证：固定 state，并把所有 Value update 置零后，交换样本 prompt **不得**改变 logit。

使用约定：

- 对齐论文同型 / 线上默认：先用 `coupled`；
- 要拆 bypass 贡献：再用 `split`；
- 两者 checkpoint **不能混用**。

生产 `mdl_rankmixer` 当前默认是 `residual_ffn + coupled`；若要严格对齐 Feature Eq. 6，应显式切到 `direct_ffn + coupled`。

---

## 问题 4：Domain 每层读什么？

> 适用范围：`mdl_onetrans`。`mdl_rankmixer` 的 Domain 读的是 32 个 Feature token，没有 S/NS 拆分。

**问题：**  
接到 OneTrans 后，定长 NS 与变长 S 代价差很大。早期「每层读 NS、后两层读 S + S 残差门」给被选中 token 不同表决权，长历史支路会被门压住。

**根因：**  
进入读取集合的 token 应同等竞争。OneTrans 统一流已是 `[Q_S; NS]`，Domain 应直接读这个池。

**当前生产合同（`mdl_onetrans`）：**

| 信号 | 角色 |
|---|---|
| 当层 `[Q_S; NS]` | Domain **每层**唯一读取池（`first_domain_sequence_layer=0`） |
| Domain important + 相关行为 prior | **只初始化** prompt，不进 S 链、也不进该读取池 |

S 容量为九条 raw 流之和约 2048；Domain 读当层 pyramid 压缩后的 \(Q_S\)。prior 现为 8 条 Domain 专用序列（含 pay 的 `ups_clk_sku` 辅 prior）。

\[
\hat D^l = D^l + \operatorname{VLAttn}\left(Q_l(D^l),\ [\,Q_S^{l+1};N^{l+1}\,],\ [\,M_S^{l+1};\mathbf{1}\,]\right)
\]

无 S 分支门；读统一流、共享一次 memory 准备；定宽 `DomainAwareAttention` 不再构造。

**边界：**  
至少对比 `null` vs `0`，并报质量 / 延迟 / HBM；`S-only / prior-only / S+prior` 分开做。非 `null` 时关掉 `use_task_feature_interaction` 会被校验拒绝。

---

## 问题 5：多场景怎么接到我们的 scene 体系？

**问题：**  
论文假设清晰、少量的 scenario 集合；我们线上有大量细粒度 `scene_id`。若把每个 fine scene 都做成独立 Domain token，参数与算力会随 \(N_{scene}\) 膨胀；若 scene 同时进内容通道和 Domain 通道，收益也无法归因。

**根因：**  
工业 scene 粒度与论文示例不一致；MDL 减少的是输出头数，并不自动消除中间 Domain 的参数与 FLOPs。mask 只能防止错误融合，不能把 inactive scenario 的计算自动省掉。

**解决方案：**

| 配置 | 做法 |
|---|---|
| coarse MDL | allowlist 把 scene 映射为 `search` / `recommendation`（约 **121** 个 search scene，其余进 recommendation），再加 **global** |
| fine MDL | 可自动发现更多 scenario token，但计算仍随 \(N_{scene}\) 增长 |

通道隔离：

- request-side `scene_id` **不进** RankMixer flat pack / OneTrans NS pack（`omit_scene_features`）；
- MDL 通过 scenario important / routing 消费 request scene；
- 这样避免同一 scene 信号既在内容通道又在 Domain 通道时无法归因。

**边界：**  
即使 omit 了 flat/NS pack，RankMixer 的 LONGER **user-global** 路径仍可能消费 `scene_id_hn`（scene-aware summary）。比较“content scene on/off × Domain scene on/off”时必须把这条路径算进去，否则 2×2 不干净。fine 配置下 active scene 只有一个，仍可能维护大量 Scenario states——那是下一阶段 active-token execution 问题。

---

## 问题 6：实际遇到的显存问题有哪些？

**问题：**  
短 smoke 能过，换 profile、开 cache、开 checkpoint 或拉长跑仍可能 OOM / util 崩。把 batch 降下来也不保证安全。

**根因：**  
OOM 很少由单一张量决定，而是 **embedding 静态占用、激活、packing、cache、prefetch、NCCL/emb staging、allocator reserved fragmentation** 同相叠加；“降 batch”和“开 full remat”都不是单调更省的旋钮。OneTrans 与 RankMixer 的失败模式还不同。

**解决方案（现场表）：**

### OneTrans / `mdl_onetrans`

| 现场问题 | 根因 | 修复 |
|---|---|---|
| compact pack 阶段直接 OOM | boolean indexing 同时制造大 mask 与大拷贝 | 改为索引式 `index_select`，避免全宽临时副本（`cc92f47`） |
| cache 开启后峰值随 candidates/request 急升 | 所有层 S cache 被提前从 request 展开到 candidate | request-sized 持久 cache + layer-local gather（`5e040d6`） |
| token width 扩到 512 后 HBM 不安全 | S 远长于 NS，宽度翻倍同时放大 QKV、FFN 与 cache | 生产 profile 恢复到 **256**（`1eef46f`） |
| batch 768 在约 step 1300 仍 OOM | 除 live tensor 外约有 12 GiB reserved fragmentation；短 smoke 未覆盖高水位 | allocator 必须在 import torch 前生效，并联动 length bucket、projection chunk 与 packing（`304c1d0`） |
| 打开 full activation checkpoint 后仍 mid-run OOM | checkpoint、packing、cache K/V 与 batch profile 耦合 | 回退到验证过的 `activation_checkpoint=none + fixed packing` 基线后重调（`f6fdb68`） |
| Flash/FFN 峰值叠加 | 冗余 contiguous、重复 mask metadata、长序列投影同时存活 | 复用 packing metadata、跳过冗余拷贝、按长度分块序列投影（`4861e0e`） |

当前生产 `mdl_onetrans` batch 已回到 **1408**。历史上的“降到 512”不是终局答案。

### RankMixer / `mdl_rankmixer`（对照）

| 现场问题 | 根因 | 修复 |
|---|---|---|
| 24h 理想 embedding ≈399 GiB/GPU 不可训 | 短 profile 低估长尾；标准 Adagrad state \(O(ND)\) | load≤1.75、大表 dim≤32、sharded Row-Wise；粗粒度 MDL ≈65～66 GiB/GPU |
| batch 提到 1024 后 LONGER/投影峰值失控 | 整 batch 一次投影；prefetch=2 叠两份大序列 | token 预算切块 + prefetch=1；现生产 batch **1536** + length bucket |
| full remat 后 6/8 卡 util 崩 | 浅层 2-layer 重算贵，且挡住 CUDA graph | 回到 `activation_checkpoint=none + cuda_graph_backbone`（`bd4a367`） |
| 为 HBM 加的安全旋钮饿死 2–4 卡 util | NCCL cap / 强制 memfd 被套到小 world | NCCL 保险仅 ≥6 GPU；shm 充裕时回 share IPC（`bfaba79`） |

**边界：**  
安全 batch 必须在真实长度分布、缓存策略、allocator、packing、world-size profile 与完整长跑下联合验证。更完整的对照表见 [`mdl_reproduction_lecture_report.md`](./mdl_reproduction_lecture_report.md) §2.7 与 RankMixer 显存节。

---

## 问题 7：为什么一个 HDFS `pread` 卡住，会让所有 GPU 停住？

多卡训练要求所有 rank 每一步都一起做梯度同步。只要其中一个 rank 卡在 HDFS 读取、拿不到下一个 batch，其他 rank 即使已经算完，也只能停在 collective 等它，因此最终表现为所有 GPU 一起不动，甚至触发 NCCL timeout。

问题不只是 HDFS 偶发卡顿，而是超时后的处理方式：

- 多个 worker 复用同一个 HDFS session，可能相互影响；
- 已经超时的 generator 不能继续重试；
- 对仍有挂起读取的 session 调 `close()`，也可能卡住；
- 启动阶段的 HDFS 扫描若走 NCCL，会把冷 IO 和 GPU 通信绑在一起。

因此处理原则是：**超时后直接废弃旧 session，不重试、不强行关闭；在独立子进程中执行远程读取，超时后杀掉整个进程组，并以明确退出码结束，让平台自动重启。** 启动期 scene discovery 走 CPU control group（Gloo）广播，避免冷 HDFS 绑死 NCCL。细节与提交见 [`mdl_reproduction_lecture_report.md`](./mdl_reproduction_lecture_report.md)。

---

## 问题 8：为什么 RSS 会不断增长？

训练使用 pinned memory 缓存 CPU 到 GPU 的数据。旧版 buffer pool 只会扩容，不会缩容。

例如平时 batch 只需要 200 MB，偶尔一个超长 batch 需要 2 GB，pool 就会把 2 GB buffer 永久保留。之后即使 batch 恢复正常，实际只使用其中一小部分，整块 pinned memory 仍占着 RSS。

因此这不是普通的 Python 对象泄漏，而是 **pinned memory pool 被历史峰值撑大后没有回收**。

解决方式是让 pool 根据最近一段时间的 batch 大小动态缩容：

- 过大的空闲 buffer 直接丢弃；
- 降低扩容预留比例；
- 限制空闲 buffer 数量和单个 buffer 的保留上限。

核心变化是：**超长 batch 可以临时申请大内存，但处理完成后不再永久保留。** 可选硬上限示例：`MDL_PINNED_POOL_MAX_SLOT_BYTES=1073741824`（1 GiB）。长跑验收看尖峰后 RSS 是否回落；这不替代 memfd IPC 等更早一层的修复。

---

## 修订时相对原稿的主要修正

| 原稿问题 | 修正 |
|---|---|
| 任务写成 cart / pay / cate | 统一为 `fst_cart` / `upid_pay` / `cateid_filter` |
| “不用 RankMixer 默认等宽切片” | 改为：论文要求语义组；**我们早期 tokenizer** 做错了等宽切片 |
| `coupled`/`split` 方案写到一半 | 补全 `split` 公式、硬验证与 checkpoint 不混用 |
| “每层读 NS / 后两层读 S”表格挤成一行，且易被当成两模型通用 | 标明仅 `mdl_onetrans`；后续改为 `[Q_S; NS]` 等同待遇单池，去掉 S 分支门 |
| scene omit 说得过绝对 | 补上 LONGER user-global 仍可能吃 `scene_id` 的边界 |
| 显存表缺行，且 batch-768 行把修复写进了根因 | 补全 OneTrans 表，并加 RankMixer 对照；当前 batch 写回 1408 / 1536 |
| HDFS 解决方案表格无表头 | 恢复“手段 / 作用”表 |

## 相关文档

| 主题 | 文档 |
|---|---|
| 数据适配总览 | [`mdl_data_adaptation_overview.md`](./mdl_data_adaptation_overview.md) |
| Domain 字段设计 | [`mdl_token_feature_design.md`](./mdl_token_feature_design.md) |
| OneTrans 适配难点 | [`mdl_onetrans_adaptation_hardships.md`](./mdl_onetrans_adaptation_hardships.md) |
| 词表与 Embedding | [`mdl_vocab_embedding_design.md`](./mdl_vocab_embedding_design.md) |
| 串讲底稿 | [`mdl_reproduction_lecture_report.md`](./mdl_reproduction_lecture_report.md) |
