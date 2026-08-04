# MDL 复现过程中的问题和解决方案

> 口径对齐仓库生产配置与代码（截至 `main`）。
> 本文与 [`mdl_key_questions.md`](./mdl_key_questions.md) 同源，为独立成文的压缩重写版。
> 完整字段见 [`mdl_token_feature_design.md`](./mdl_token_feature_design.md)；OneTrans 适配见 [`mdl_onetrans_adaptation_hardships.md`](./mdl_onetrans_adaptation_hardships.md)；串讲底稿见 [`mdl_reproduction_lecture_report.md`](./mdl_reproduction_lecture_report.md)。

本文记录 MDL 复现过程中最关键的八类问题：token 构成与切分（§1–2）、Domain 状态语义与 S/NS 分层（§3–4）、多场景接入（§5），以及显存、HDFS IO、Host RSS 三类系统工程问题（§6–8）。统一体例为 **问题 → 根因 → 解决方案**；必要时补 **边界**，说明哪些结论还不能下定论。

---

## 1. Scenario / Task token 由什么构成？

**问题：**
论文只说 Domain token 由 important raw features + related prior 初始化，没有给出工业字段清单。直接照抄论文示例字段，对不上我们的搜推混部数据。

**根因：**
论文给结构、不给字段。线上 Scenario 要同时覆盖搜索与推荐，Task 要对齐三个监督目标；important 与 prior 的职责不同——前者是语义锚，后者是行为摘要。若把极稀疏 identity 和长历史摘要混进同一条路径，既难训、也难归因。

**解决方案：**

**Scenario（coarse：`search` / `recommendation` / `global`）**

routing 只有 `search` / `recommendation` 两类；`global` 是每个样本都使用的常驻 Domain token，不参与 routing。

| 组成 | 内容 |
|---|---|
| important | locale（`currency` / `plat` / `region` / `hash_language_site`）、页面/入口（`page_elsn` / `page_sn`）、`scene_id`（**global 不用**）、`cat1_id`、`price` 等强锚；`search` 另含 `search_method_hn`，`recommendation` 去掉 |
| prior | coarse scene prior、scene 曝光统计（`impr_cnt_15d`）、scene-conditioned click 历史摘要（`scenario_conditioned_clk_long_prior`，attention pool 的 query 取 scenario important 的 `scene_id` + `page_sn`）；global 另用全局 impr / clk / view 摘要 |

**Task（`fst_cart` / `upid_pay` / `cateid_filter`）**

| 任务 | important | prior |
|---|---|---|
| `fst_cart` | 加购相关统计（`adj_cartcvr`、`cart_cnt_3d`），加可控 identity（goods / mall / cluster） | `cart_long` 的 **task-important 条件 attention pool** |
| `upid_pay` | 支付/转化相关统计（`sales`、`adj_cvr`），加可控 identity（goods / mall / cluster） | `buy_long` 的条件 attention pool |
| `cateid_filter` | 相关性与 query（`rel_level` / `rel_score` / `origin_query_hash`），**不含** goods / mall / price | `srch_q2i` 的条件 attention pool（按标签语义选，不是买历史） |

取舍原则：

- **高频、可学习、语义锚** → important；**行为统计 / 历史摘要** → prior；
- 单 epoch 下极稀疏的主 `goods_id`（\(2^{28}\approx\) 268M buckets）**默认不进** scenario important；task 侧用独立的**低基数** extra 表（\(2^{25}\approx\) 32M buckets，dim 与主表同为 32）补精确身份——缩小的是行数而不是维度，关键是**不再复制一张 268M 主表**。

**边界：**
完整字段表与消融清单见 [`mdl_token_feature_design.md`](./mdl_token_feature_design.md)。prior 用哪条历史（尤其 `cateid_filter`）是当前默认解释，需 holdout 消融，不是论文规定。

---

## 2. 为什么 Feature token 不能做任意等宽切片？

**问题：**
早期实现把全部 embedding / 序列 summary 拼成长向量，再按总宽度静默切成 \(T\) 段。字段改 dim 或增删后，shape 看起来仍合法，但 token 语义已经漂了；为满足整除，token 数甚至可能从 32 静默降成 8。

**根因：**
论文要求 **semantically coherent clusters** 再投影。等宽切片的 token identity 由“当前 raw width 的等分位置”决定，而不是由业务语义决定。于是一次 embedding 容量调整，会同时改变：

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

`T=32, D=768` 是保留的**容量契约**，不是把模型静默缩成 8 token。注意这只是 RankMixer 系的合同：`mdl_onetrans` 的 NS 走 `auto_split`（32 token、dim 256），不适用同一条 groupwise 约定。

**边界：**
groupwise 解决的是 token 身份与实验可解释性；仓库没有证明某一组字段划分一定带来更高 AUC。

---

## 3. Domain 状态是否可能造成捷径？

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

attention residual 之后，两种模式都仍会过各自的 Domain FFN。

硬验证：固定 state，并把所有 Value update 置零后，交换样本 prompt **不得**改变 logit。

使用约定：

- 对齐论文同型 / 线上默认：先用 `coupled`；
- 要拆 bypass 贡献：再用 `split`；
- 两者 checkpoint **不能混用**（`split` 多出 readout seed 参数，state_dict 结构不同）。

另注意：生产 `mdl_rankmixer` 当前默认是 `residual_ffn + coupled`；若要严格对齐论文 Feature Eq. 6，应显式切到 `direct_ffn + coupled`。

---

## 4. 为什么每层读 NS，但只在后两层读 S？

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

两点口径需要说清：

- “2048” 是九条原始行为流 `max_length` 之和，不是单一序列上限；
- pyramid 不是 pooling 级联，而是线性递减、按 32 取整的 keep-count schedule，配合 causal-suffix attention 逐层保留最新事件：\(2048 \to 1696 \to 1376 \to 1024 \to 704 \to 352 \to 12\)。

逐层计算（记第 \(l\) 层后为 \(S^l,N^l,D^l\)）：先跑原生 OneTrans block 得到 \(S^{l+1},N^{l+1}\)，再让 Domain 读 NS；从 zero-based layer **4** 起（6 层中的最后两层）额外读压缩后的 S，并用残差门控（bias 初始化为 \(-2\)，零输入时 sigmoid \(\approx 0.119\)）避免训练初期长序列支路压过 NS。

**解决方案：**
配置 `first_domain_sequence_layer=4`：先保证能训、能缓存、能保留 request-sized S cache。算法结论留给消融，而不是把该超参写成论文结论。

**边界：**
至少比较 `null`（不直接读 S）、`4`（最后两层）、`0`（全层读），并同时报告质量、延迟与峰值 HBM。`S-only / prior-only / S+prior` 也要单独做，否则无法区分“复用同一份行为”与“重复计权”。

---

## 5. 多场景怎么接到我们的 scene 体系？

**问题：**
论文假设清晰、少量的 scenario 集合；我们线上有大量细粒度 `scene_id`。若把每个 fine scene 都做成独立 Domain token，参数与算力会随 \(N_{scene}\) 膨胀；若 scene 同时进内容通道和 Domain 通道，收益也无法归因。

**根因：**
工业 scene 粒度与论文示例不一致；MDL 减少的是输出头数，并不自动消除中间 Domain 的参数与 FLOPs。mask 只能防止错误融合，不能把 inactive scenario 的计算自动省掉。

**解决方案：**

| 配置 | 做法 |
|---|---|
| coarse MDL | allowlist 把 scene 映射为 `search` / `recommendation`（`SEARCH_SCENE_IDS` 恰为 **121** 个 search scene，其余进 recommendation），再加常驻 `global` token |
| fine MDL | `auto_discover` 从 `scene_id` 自动发现 scenario token（上限 256），但计算仍随 \(N_{scene}\) 增长 |

通道隔离：

- request-side `scene_id` **不进** RankMixer flat pack / OneTrans NS pack（`omit_scene_features`，默认开）；
- MDL 通过 scenario important / routing 消费 request scene；
- 这样避免同一 scene 信号既在内容通道又在 Domain 通道时无法归因。

**边界：**
即使 omit 了 flat/NS pack，RankMixer 的 LONGER **user-global** 路径仍可能消费 `scene_id_hn`（scene-aware summary）。比较“content scene on/off × Domain scene on/off”时必须把这条路径算进去，否则 2×2 不干净。fine 配置下 active scene 只有一个，仍可能维护大量 Scenario states——那是下一阶段 active-token execution 问题。

---

## 6. 实际遇到的显存问题

**问题：**
短 smoke 能过，换 profile、开 cache、开 checkpoint 或拉长跑仍可能 OOM / util 崩；把 batch 降下来也不保证安全。

**根因：**
OOM 很少由单一张量决定，而是 **embedding 静态占用、激活、packing、cache、prefetch、NCCL/emb staging、allocator reserved fragmentation** 同相叠加；“降 batch”和“开 full remat”都不是单调更省的旋钮。OneTrans 与 RankMixer 的失败模式还不同。

**解决方案（现场表）：**

**OneTrans / `mdl_onetrans`**

| 现场问题 | 根因 | 修复 |
|---|---|---|
| compact pack 阶段直接 OOM | boolean indexing 同时制造大 mask 与大拷贝 | 改为索引式 `index_select`，避免全宽临时副本（`cc92f47`） |
| cache 开启后峰值随 candidates/request 急升 | 所有层 S cache 被提前从 request 展开到 candidate | request-sized 持久 cache + layer-local gather（`5e040d6`） |
| token width 扩到 512 后 HBM 不安全 | S 远长于 NS，宽度翻倍同时放大 QKV、FFN 与 cache | 生产 profile 恢复到 **256**（`1eef46f`） |
| batch 768 在约 step 1300 仍 OOM | 除 live tensor 外约有 12 GiB reserved fragmentation；短 smoke 未覆盖高水位 | allocator（`expandable_segments` + `max_split_size_mb`）必须在 import torch 前生效，并联动 length bucket、projection chunk 与 packing（`304c1d0`，当时先降到 512 兜底） |
| 打开 full activation checkpoint 后仍 mid-run OOM | checkpoint、packing、cache K/V 与 batch profile 耦合 | 回退到验证过的 `activation_checkpoint=none + fixed packing` 基线后重调，batch 回到 **1408**（`f6fdb68`） |
| Flash/FFN 峰值叠加 | 冗余 contiguous、重复 mask metadata、长序列 FFN 同时存活 | 复用 packing metadata、跳过冗余拷贝、按长度分块 MixFormer SwiGLU/attention（`4861e0e`） |

当前生产 `mdl_onetrans` batch 已回到 **1408**。历史上的“降到 512”不是终局答案。

**RankMixer / `mdl_rankmixer`（对照）**

| 现场问题 | 根因 | 修复 |
|---|---|---|
| 24h 理想 embedding ≈399.49 GiB/GPU 不可训 | 短 profile 低估长尾；标准 Adagrad state \(O(N\cdot D)\) | load ≤ 1.75、大表 dim ≤ 32、sharded Row-Wise Adagrad（每行一个 FP32 累加器）；粗粒度 MDL 部署 ≈65～66 GiB/GPU |
| batch 提到 1024 后 LONGER/投影峰值失控 | 整 batch 一次过序列编码与投影；prefetch=2 叠两份大序列 | token 预算切块（`sequence_encoder_chunk_tokens`）+ prefetch=1（`9529ae2`）；现生产 batch **1536** + length bucket |
| full remat 后 6/8 卡 util 崩 | 浅层 2-layer 重算贵，且挡住 CUDA graph | 回到 `activation_checkpoint=none + cuda_graph_backbone`（`bd4a367`） |
| 为 HBM 加的安全旋钮饿死 2–4 卡 util | NCCL cap / 强制 memfd 被套到小 world | NCCL 保险仅 ≥6 GPU；shm 充裕时回 share IPC（`bfaba79`） |

**边界：**
安全 batch 必须在真实长度分布、缓存策略、allocator、packing、world-size profile 与完整长跑下联合验证。更完整的对照表见 [`mdl_reproduction_lecture_report.md`](./mdl_reproduction_lecture_report.md) §2.7 与 RankMixer 显存节。

---

## 7. 一个 HDFS `pread` 卡死，为什么最后表现成所有 GPU 一起不动？

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
| timed open/start/batch + controlled row-group concurrency | 给每个远程阶段独立预算，限制同时悬挂的 native 调用（HDFS prefetch 每 rank 上限 4） |
| poisoned session quarantine | timeout 后不在原 generator 上重试，也不 native-close；新建 session；从未产出 batch 时才安全重开，否则直接放弃 |
| `spawn` host-prepare child + process-group kill | startup/idle 超时后杀整个子进程树（`killpg`），避免 D-state 子进程拖住父 rank |
| `REMOTE_IO_STALL -> exit 70` | 让平台按可重启 IO 故障处理，而不是无限等 NCCL |
| Gloo control group | scene discovery 用 Arrow unique 扫描后经 CPU control group 广播，避免冷 HDFS 阻塞 NCCL |
| bounded teardown | 先关 IPC pipe，再 join CUDA prefetch；删除无界 queue join，`destroy_process_group` 等待限 30s |

**边界：**
目标不是“永远不遇到坏 HDFS”，而是把不可观测的永久挂起转换成**有阶段、有 HBM 快照、有退出码、可自动重启**的失败。相关提交见串讲底稿附录（`705878d`、`48f243a`、`fd5edb6`、`457d2e6`、`8c31ae3` 等）。

---

## 8. 为什么 RSS 不断增长？

**问题：**
`331b6ac` / `64ba075` 之后，host-prepare 已用 memfd + recycled pinned pool，不再每个变长 shape 都 `pin_memory()` 新 slab。但长跑里偶发超长 pack（尖峰 batch）仍可能把 container RSS 顶上去；之后即使多数 batch 变短，**idle pinned 页也不回落**，RSS 继续贴着历史峰值斜着爬，最终可能先被平台 host-mem protect 杀掉。

**根因：**
第一代 pool 是 **grow-only 复用**：lease 归还后按历史最大 `numel` 留着 buffer，以便下次免分配。这对“稳定长度分布”很省，但对“偶发长尾 pack”是单向下棘轮——一次尖峰把 slot 撑大，后续小 batch 只 `narrow` 使用前缀，**整块 pinned storage 仍被 CUDA host caching allocator 握着**。再叠加：

- 扩容余量原先偏大（25%），每次 grow 多锁一截；
- free slot 数按 `max(4, queue_size + 2)` 走，深 prefetch 会同时留住多份峰值级 idle slab。

Python 对象都释放了也看不见：泄漏在 allocator / pinned page，不在 `gc.get_objects()`。

**解决方案：**
`4a3cda0` 把 `_PinnedHostBufferPool` 改为**可缩的滑动高水位**，不再按全局历史峰值永久长大：

| 手段 | 作用 |
|---|---|
| 滑动窗口（默认 256）记近期请求 `numel` | lease 归还时若 `buf > 2× recent_hwm`，丢掉过大 idle slab，并调 `torch._C._host_emptyCache` |
| 扩容余量 `25% → 12.5%` | 降低每次 grow 的锁页幅度 |
| `max_free_slots = min(4, max(2, queue_size))` | 深 queue 不再成倍堆峰值级 idle slot |
| 可选 `MDL_PINNED_POOL_MAX_SLOT_BYTES` | idle 保留硬上限；**live batch 仍保证能装下**（超 cap 时跳过 headroom，归还后再 trim） |

默认即可靠滑动缩容；更狠的 idle 上限示例：

```bash
export MDL_PINNED_POOL_MAX_SLOT_BYTES=1073741824  # 1 GiB
```

**边界：**
回归覆盖复用、尖峰后缩容、byte cap 与 env 解析（见 `tests/test_pinned_host_pool.py`）。仓库仍没有可引用的完整“修复前后 24h RSS 曲线”，长跑验收应看**偶发尖峰后 RSS 是否回落**，而不是只看短 smoke 的峰值。这是对串讲 §3.5「recycled pinned pool」的增强，不替代 memfd / 私有化 / H2D 后放 ref 那几条。

---

## 附：相对原稿的主要修正

| 原稿问题 | 修正 |
|---|---|
| “task 侧用独立小 dim extra 表” | 实为 **32M×32**：缩小的是行数（268M → 32M buckets），dim 与主表相同；关键约束是不复制主表 |
| locale 只有泛称 | 落实为 `currency` / `plat` / `region` / `hash_language_site` |
| Task important 统一写“类目层级、价格、……” | 按配置拆开：`fst_cart`（`adj_cartcvr`/`cart_cnt_3d`）、`upid_pay`（`sales`/`adj_cvr`）+ goods/mall/cluster；`cateid_filter` 为 `rel_level`/`rel_score`/`origin_query_hash`，不含 goods/mall/price |
| Feature token 切分未限定模型范围 | 23+9=32、`Linear→768` 仅 `mdl_rankmixer`/`rankmixer`；`mdl_onetrans` 为 `auto_split`、32 token、dim 256 |
| `coupled`/`split` 只有公式 | 补硬验证（置零 Value update 后交换 prompt 不得改变 logit）、使用约定与 checkpoint 不混用、`residual_ffn`/`direct_ffn` 说明 |
| “最长约 2048” 与 pyramid 含糊 | 注明 2048 为九条流上限之和；pyramid 为线性 keep-count schedule（…→352→12）；门控零输入 sigmoid≈0.119；补 `null/4/0` 与 `S-only/prior-only` 消融边界 |
| scene 通道隔离缺边界 | 补 LONGER user-global 仍消费 `scene_id_hn` 的 2×2 归因注意点；fine 自动发现上限 256 |
| 显存表缺行、个别根因/修复混淆 | 补 token width 512→256、full remat 回退两行；补精确数字 399.49 GiB 与当前 batch 1408/1536；`4861e0e` 第三项是**按长度分块 MixFormer SwiGLU/attention**（非“序列投影”）；补 commit 号 |
| HDFS / RSS 只有骨架 | 补精确 API（`torch._C._host_emptyCache`）、第一代 free-slot 公式 `max(4, queue_size+2)`、commit 号（`331b6ac`/`64ba075`/`4a3cda0`）与边界 |

## 相关文档

| 主题 | 文档 |
|---|---|
| 同内容详版（问题 · 根因 · 解决方案） | [`mdl_key_questions.md`](./mdl_key_questions.md) |
| Domain 字段设计 | [`mdl_token_feature_design.md`](./mdl_token_feature_design.md) |
| OneTrans 适配难点 | [`mdl_onetrans_adaptation_hardships.md`](./mdl_onetrans_adaptation_hardships.md) |
| 串讲底稿 | [`mdl_reproduction_lecture_report.md`](./mdl_reproduction_lecture_report.md) |
