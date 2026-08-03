# MDL 关键问题集：问题 · 根因 · 解决方案

> 口径对齐仓库生产配置与现有专题文档（截至 `main`）。  
> 完整字段见 [`mdl_token_feature_design.md`](./mdl_token_feature_design.md)；OneTrans 适配见 [`mdl_onetrans_adaptation_hardships.md`](./mdl_onetrans_adaptation_hardships.md)；串讲底稿见 [`mdl_reproduction_lecture_report.md`](./mdl_reproduction_lecture_report.md)。

统一体例：**问题 → 根因 → 解决方案**；必要时补 **边界**（什么还不能下结论）。

---

## 问题 1：Scenario / Task token 由什么构成？

**问题：**  
论文只说 Domain 由 important raw features + related prior 初始化，没有给出工业字段清单。直接抄论文示例字段会对不上我们的搜推混部数据。

**根因：**  
论文给结构、不给字段。线上 Scenario 要同时覆盖搜索/推荐，Task 要对齐三个监督目标；important 与 prior 的职责不同——前者是语义锚，后者是行为摘要。若把极稀疏 identity 和长历史摘要混进同一条路径，既难训也难归因。

**解决方案：**  

**Scenario（coarse：`search` / `recommendation` / `global`）**

| 组成 | 内容 |
|---|---|
| important | locale、页面/入口、`scene_id`（**global 不用**）、类目/价格等强锚；search 另含 `search_method_hn`，recommendation 去掉 |
| prior | coarse scene prior、scene 曝光统计、scene-conditioned click 历史摘要；global 另用 impr / clk / view 摘要 |

**Task（`fst_cart` / `upid_pay` / `cateid_filter`）**

| 任务 | important（示意） | prior |
|---|---|---|
| `fst_cart` | 类目层级、价格、加购相关统计，以及可控 identity（goods/mall 等） | `cart_long` 的 **task-important 条件 attention pool** |
| `upid_pay` | 类目层级、价格、支付/转化相关统计，以及可控 identity | `buy_long` 的条件 attention pool |
| `cateid_filter` | 类目、相关性、query 等 | `srch_q2i` 的条件 attention pool（按标签语义选，不是买历史） |

取舍原则：

- **高频、可学习、语义锚** → important；
- **行为统计 / 历史摘要** → prior；
- 单 epoch 下极稀疏的主 `goods_id` **默认不进** scenario important；task 侧用**独立小 dim extra 表**补精确身份，禁止再复制 268M 主表。

**边界：**  
完整字段表与消融清单见 [`mdl_token_feature_design.md`](./mdl_token_feature_design.md)。prior 用哪条历史（尤其 `cateid_filter`）是当前默认解释，需 holdout 消融，不是论文规定。

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

## 问题 4：为什么每层读 NS，但只在后两层读 S？

> 适用范围：`mdl_onetrans`（Domain sidecar 读 OneTrans）。`mdl_rankmixer` 的 Domain 读的是 32 个 Feature token，没有这条 S/NS 分层合同。

**问题：**  
若 Scenario/Task 对最长约 2048 的 S 做**全层** cross-attention，训练代价与峰值显存都会失控；若完全不读 S，又丢掉长历史直接证据。

**根因：**  
三类信号职责与代价不同：

| 信号 | 特点 | 当前处理 |
|---|---|---|
| 32 个 NS | 已融合当前候选，长度固定，候选相关性强 | Scenario/Task **每一层**都读 |
| 最长约 2048 的 S | 请求共享、变长；早层偏原始事件，直接 cross-attn 昂贵 | OneTrans 先做 pyramid 压缩，**最后两层**才允许 Domain 读 |
| 7 个 Domain prior summary | 面向 scenario/task 的紧凑历史摘要 | **只初始化** Domain prompt，不进原始 S 因果链 |

逐层计算（记第 \(l\) 层后为 \(S^l,N^l,D^l\)）：先跑原生 OneTrans block 得到 \(S^{l+1},N^{l+1}\)，再让 Domain 读 NS；从 zero-based layer **4** 起（6 层中的最后两层）额外读压缩后的 S，并用残差门控（bias 初始化为 \(-2\)）避免训练初期长序列支路压过 NS。

**解决方案：**  
配置 `first_domain_sequence_layer=4`：先保证能训、能缓存、能保留 request-sized S cache。算法结论留给消融，而不是把该超参写成论文结论。

**边界：**  
至少比较 `null`（不直接读 S）、`4`（最后两层）、`0`（全层读），并同时报告质量、延迟与峰值 HBM。`S-only / prior-only / S+prior` 也要单独做，否则无法区分“复用同一份行为”与“重复计权”。

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

## 问题 7：一个 HDFS `pread` 卡死，为什么最后表现成所有 GPU 一起不动？

**问题：**  
多卡 HDFS 流式训练中，某个 reader 的 native `pread` 偶发永久不返回；该 rank 不再产生 batch，其它 rank 随后卡在 collective，日志只剩 GPU 空转或 NCCL timeout。任务结束时还可能卡在 reader / queue / process-group teardown。

**根因：**

1. 并发 worker 复用 `HadoopFileSystem` / Parquet session，native handle 可能损坏；
2. timeout thread 仍持有 generator 时，再次 `next()` 会报 `generator already executing`；
3. 对仍有挂起 `pread` 的 poisoned session 调 native `close`，close 本身也可能永久挂起；
4. 训练前 scene discovery 在冷 HDFS 上耗时很长，若直接走 NCCL 广播，会把 IO 冷启动与 GPU collective 绑死。

这里不能简单套用“fork 继承 libhdfs JVM”的常见故事；仓库证据指向的是 **session 复用、超时重试与 teardown 合同** 的问题。

**解决方案：**

| 手段 | 作用 |
|---|---|
| thread-local HDFS client | 不在长期 prefetch workers 间共享 native client |
| timed open/start/batch + controlled row-group concurrency | 给每个远程阶段独立预算，限制同时悬挂的 native 调用 |
| poisoned session quarantine | timeout 后不在原 generator 上重试，也不 native-close；新建 session；从未产出 batch 时才安全重开 |
| `spawn` host-prepare child + process-group kill | startup/idle 超时后杀整个子进程树，避免 D-state 子进程拖住父 rank |
| `REMOTE_IO_STALL -> exit 70` | 让平台按可重启 IO 故障处理，而不是无限等 NCCL |
| Gloo control group | scene discovery 用 Arrow unique 扫描后经 CPU control group 广播，避免冷 HDFS 阻塞 NCCL |
| bounded teardown | 先关 IPC pipe，再 join CUDA prefetch；删除无界 queue join，并限制 `destroy_process_group` 等待 |

**边界：**  
目标不是“永远不遇到坏 HDFS”，而是把不可观测的永久挂起转换成**有阶段、有 HBM 快照、有退出码、可自动重启**的失败。相关提交见串讲底稿附录（`705878d`、`48f243a`、`fd5edb6`、`457d2e6`、`8c31ae3` 等）。

---

## 问题 8：开了 recycled pinned pool，为什么 host RSS 仍可能跟历史尖峰一起爬？

**问题：**  
`331b6ac` / `64ba075` 之后，host-prepare 已用 memfd + recycled pinned pool，不再每个变长 shape 都 `pin_memory()` 新 slab。但长跑里偶发超长 pack（尖峰 batch）仍可能把 container RSS 顶上去，之后即使多数 batch 变短，**idle pinned 页也不回落**，RSS 继续贴着历史峰值斜着爬，最终可能先被平台 host-mem protect 杀掉。

**根因：**  
第一代 pool 是 **grow-only 复用**：lease 归还后按历史最大 `numel` 留着 buffer，以便下次免分配。这对“稳定长度分布”很省，但对“偶发长尾 pack”是单向下棘轮——一次尖峰把 slot 撑大，后续小 batch 只 `narrow` 使用前缀，**整块 pinned storage 仍被 CUDA host caching allocator 握着**。再叠加：

- 扩容余量原先偏大（约 25%），每次 grow 多锁一截；
- free slot 数若跟 `queue_size+2` 走，深 prefetch 会同时留住多份峰值级 idle slab。

Python 对象都释放了也看不见：泄漏在 allocator / pinned page，不在 `gc.get_objects()`。

**解决方案：**  
`_PinnedHostBufferPool` 改为**可缩的滑动高水位**，不再按全局历史峰值永久长大：

| 手段 | 作用 |
|---|---|
| 滑动窗口（默认 256）记近期请求 `numel` | lease 归还时若 `buf > 2× recent_hwm`，丢掉过大 idle slab，并 `_host_emptyCache` |
| 扩容余量 `25% → 12.5%` | 降低每次 grow 的锁页幅度 |
| `max_free_slots = min(4, max(2, queue_size))` | 深 queue 不再成倍堆峰值级 idle slot |
| 可选 `MDL_PINNED_POOL_MAX_SLOT_BYTES` | idle 保留硬上限；**live batch 仍保证能装下**（超 cap 时跳过 headroom，归还后再 trim） |

默认即可靠滑动缩容；更狠的 idle 上限示例：

```bash
export MDL_PINNED_POOL_MAX_SLOT_BYTES=1073741824  # 1GiB
```

**边界：**  
回归覆盖复用、尖峰后缩容、byte cap 与 env 解析（见 `tests/test_pinned_host_pool.py`）。仓库仍没有可引用的完整“修复前后 24h RSS 曲线”，长跑验收应看**偶发尖峰后 RSS 是否回落/平台**，而不是只看短 smoke 的峰值。这是对串讲 §3.5「recycled pinned pool」的增强，不替代 memfd / 私有化 / H2D 后放 ref 那几条。

---

## 修订时相对原稿的主要修正

| 原稿问题 | 修正 |
|---|---|
| 任务写成 cart / pay / cate | 统一为 `fst_cart` / `upid_pay` / `cateid_filter` |
| “不用 RankMixer 默认等宽切片” | 改为：论文要求语义组；**我们早期 tokenizer** 做错了等宽切片 |
| `coupled`/`split` 方案写到一半 | 补全 `split` 公式、硬验证与 checkpoint 不混用 |
| “每层读 NS / 后两层读 S”表格挤成一行，且易被当成两模型通用 | 标明仅 `mdl_onetrans`；补门控与消融边界 |
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
