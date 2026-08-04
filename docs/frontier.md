## 11. 前沿延伸：多分布推荐继续向哪里演进

MDL 将 Scenario 与 Task 从末端 gate / tower，提升为跨层保留、逐层读取 Feature states 的条件状态。主干扩大后，条件路径终于能与容量匹配。

但多分布推荐仍有两个未解问题：

| 轴 | 未解问题 |
|---|---|
| 任务 | 不同任务除了差异化读取 Feature，是否还需显式传递知识，并限制彼此损失对私有状态的干扰？ |
| 场景 | 当各业务连输入字段、行为序列、候选组织都无法对齐时，统一模型如何扩展？ |

2026 年的 OneRank 与 MTFM 分别回应这两轴。三者不是直接迭代关系，而是三种相互关联的问题意识。

> **MDL 管条件如何深入主干；OneRank 管任务如何交换知识与隔离梯度；MTFM 管异构场景如何进入同一可扩展主干。**

---

## 11.1 OneRank：把多任务推理放进 Transformer 内部

### 传统结构的瓶颈

多数 Transformer 排序仍是「共享编码器 + 多任务预测器」：

```text
共享 Transformer → task-agnostic representation → MMoE / PLE / 多塔
```

即使编码器再深，任务分化仍推迟到末端；多任务梯度又共同冲刷共享路径，容易出现 seesaw。OneRank 的判断是：多任务不应再挂在 Transformer 之后，而应进入其计算过程本身。

### OneRank 做了什么

| 机制 | 作用 |
|---|---|
| 任务专属 token + mutual invisibility | 早期任务通道相对独立，Click / ATC / Order 从底层起有各自状态 |
| Candidate-aware contextualization | 同请求内候选互相上下文，预测不只依赖单个 user–item pair |
| 可配置 attention mask | 显式控制 Task–Task 可见性：独立任务可互不可见，漏斗任务可定向传递 |
| Cross-task gradient detachment | 前向可读其他任务状态，反向梯度不进入对方私有参数 |
| Task-aware 动态匹配 | 输出端用任务全局表示与候选匹配，不再切回静态任务 MLP |

### 与 MDL 的关键差异：Task–Task

| | MDL | OneRank |
|---|---|---|
| Task 如何分化 | 各 Task token 分别读 Feature | 任务 token 更早进入，且可互读 |
| Task–Task | 无独立 mixing，关系经共享 Feature 间接形成 | 用 mask 显式设计信息流 |
| 梯度边界 | 各任务损失仍经 K/V 更新共享 Feature | 跨任务 attention 做 detach，形成 read-only knowledge transfer |
| Scenario | 有 Scenario / Global token 与实例级选择 | 更集中于多任务，无显式 Scenario 路径 |

MDL 主要缓解**前向共享表示瓶颈**；OneRank 继续追问：任务已有专属表示后，**哪些信息该共享，哪些梯度不该跨界**。

### 线上结果（Shopee 主排序，7 天 A/B，各 10% 流量）

| 指标 | 变化 |
|---|---:|
| GMV/UU | +1.01% |
| Paid Orders/UU | +1.17% |
| 广告收入 | +0.81% |
| Bad Query Rate | −2.29% |

OneRank 不是 MDL 的直接升级，论文也未以 MDL 作对照。更准确的定位是：

> **MDL 让 Scenario 与 Task 逐层读取共享 Feature capacity；OneRank 沿任务轴推进，显式建模 Task–Task，并把前向知识共享与反向梯度干扰拆开。**

---

## 11.2 MTFM：当不同场景的输入空间也无法对齐

OneRank 延伸任务轴；MTFM 转向场景与系统：不同业务不仅分布不同，**feature schema 本身也可能不一致**。

### 旧范式：harmonize-then-decompose

传统多场景通常先对齐统一输入模板，再拆共享 / 场景专属参数。字段差异小时可行；餐厅、菜品、券包等差异大时，对齐意味着大量 padding、丢异构字段，或维护越来越重的统一特征模板。MTFM 试图绕开输入预对齐。

### 核心：heterogeneous tokenization

| Token | 含义 |
|---|---|
| H-token | 历史行为 |
| R-token | 实时行为 |
| T-token | 各场景曝光候选 |

各场景可保留各自 tokenizer 与原始字段，只需投影到同一隐藏维，再组成可变长异构序列。**统一发生在 latent 层，而不是原始 schema。**

### 与 MDL tokenization 目的不同

| | MDL | MTFM |
|---|---|---|
| Token 组织 | Feature / Scenario / Task，按职责 | H / R / T，按数据来源与时间角色 |
| 核心问题 | 场景与任务怎样读取主干 | 异构场景数据怎样进入同一主干 |
| 跨层 Domain 状态 | 有独立 Scenario / Task 状态 | 无 MDL 式跨层 Domain 状态 |
| Task–Task / 梯度边界 | 无显式 Task–Task | 非重点；预测端仍接场景多任务模块 |

### 算力与系统

序列变长后，每层 Full Attention 代价过高。MTFM 的应对：

| 手段 | 作用 |
|---|---|
| Hybrid Target Attention | 每 Block 少量 Full Attention 抓全局，多数层只用 Target Attention 更新候选 |
| Grouped-Query Attention | 降低 K/V heads 的存储与计算 |
| 按用户聚合多场景曝光 | 减少重复用户上下文编码 |
| Kernel fusion / CPU–GPU pipeline / 动态稀疏 | 抬高训练与推理吞吐 |

### 线上结果（美团，两场景）

| 场景结果 | 订单量 | 推理延迟 |
|---|---:|---:|
| 场景 A | +2.98% | −5 ms |
| 场景 B | +1.45% | −6 ms |

论文还报告：随 GFLOPs 与多场景训练数据增加，效果持续改善。MTFM 不是 MDL 的替代，而是把统一从「统一模型参数」推进到「统一异构数据接口」，并验证工业系统能否跟着扩展。

> **MDL 解决条件信息如何深入主干；MTFM 解决输入空间不对齐时，怎样仍共享一个可扩展主干。**

---

## 11.3 从扩大共享主干，到重新设计信息与梯度边界

三篇论文各自卡住一个问题：

| 工作 | 核心问题 |
|---|---|
| MDL | 条件信息能读取哪些模型容量 |
| OneRank | 任务之间如何换信息，梯度能跨哪些边界 |
| MTFM | 什么数据能进同一模型，统一计算如何可扩展 |

趋势已不只是把主干做深做宽，而是同时重设三件事：

1. 不同信息以什么 token / 状态进入模型；
2. 哪些状态能读取或影响其他状态；
3. 哪些梯度可以在任务与场景之间传播。

> **大模型化工业推荐，不只是扩大共享容量，而是让信息流、任务关系和计算边界都能随规模一起扩展。**
