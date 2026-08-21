# OxygenREC 模型介绍

论文链接：OxygenREC: An Instruction-Following Generative Framework for E-commerce Recommendation，arXiv:2512.22386

论文团队：京东 JD.com

传统推荐系统通常采用 Recall → Pre-Rank → Rank → Re-Rank 的级联架构，不同阶段分别优化召回率、CTR、CVR、GMV 等目标，容易产生多阶段目标不一致和误差累积。Generative Recommendation（GR）进一步把推荐改写成序列生成问题：先把商品编码成 Semantic ID（SID），再让模型直接根据用户历史自回归生成目标商品 SID，从而尝试在一个模型中统一候选检索与排序。

但 OxygenREC 认为，现有 GR 在工业场景中仍有两个核心问题。第一，模型主要依赖历史行为进行归纳式匹配：用户过去看过什么，就继续生成相似商品；对于需要结合时间、地点、用户画像、世界知识和行为上下文才能推断出的潜在需求，模型缺乏 deductive reasoning。第二，同一个电商平台存在 Homepage、Channel Feed、I2I、购物车、Checkout 等大量推荐场景，如果每个场景分别训练一个生成模型，会带来巨大的训练和部署成本；如果简单共享一个模型，又容易出现不同场景之间的负迁移。

OxygenREC 的基础主干仍然非常简单：

**用户历史 → Encoder → Decoder → Semantic ID → 商品**

在这条主线上，它增加了两类 Instruction：

* Scenario Instruction (I_s)：告诉模型“现在在哪个场景推荐”；
* Contextual Reasoning Instruction (I_r)：告诉模型“根据更复杂的上下文推理，这个用户现在可能真正想要什么”。

于是模型从普通生成推荐的

$$
P(Y\mid X)
$$

变成：

$$
P(Y\mid X,I_s,I_r).
$$

同时，OxygenREC 并不在线调用大 LLM。它采用 **Fast-Slow Thinking**：near-line LLM 提前完成复杂推理并生成 (I_r)，在线 Encoder–Decoder 只读取提前计算好的 instruction embedding 并完成低延迟 SID 生成。对于过长的用户历史，Instruction-Guided Retrieval（IGR）进一步根据当前 instruction 只选择相关的历史行为；Q2I Loss 则负责训练 instruction 与 item 的共享检索空间。最后，预训练后的 Generator 再通过统一 Reward Mapping 和 SA-GCPO 强化学习，对齐 GMV、Conversion、相关性和多样性等真实业务目标。

因此 OxygenREC 可以概括为：

> **用 Slow Thinking 解决“用户现在真正想要什么”，用 Scenario Instruction 解决“现在应该按什么场景推荐”，用 Encoder–Decoder GR 完成实时商品生成，再用统一 RL 把同一个 Generator 对齐到不同场景的业务目标。**

## 1.1 模型背景

### 1.1.1 从级联推荐到生成式推荐

传统工业推荐一般经历：

$$
\text{Recall}
\rightarrow
\text{Pre-Rank}
\rightarrow
\text{Rank}
\rightarrow
\text{Re-Rank}.
$$

每一层使用不同模型和不同目标。Recall 关注是否覆盖潜在相关商品，Rank 更关注 CTR/CVR，Re-Rank 还可能加入多样性、业务规则和 GMV 等目标。因此上游阶段丢掉的商品无法被下游恢复，而且不同阶段的最优目标并不一致。

Generative Recommendation 将问题重新写成：

$$
P(Y\mid X),
$$

其中 (X) 表示用户及其历史行为，(Y) 不再是一个 binary label，而是目标商品对应的 Semantic ID 序列。

例如某个商品最终被编码成：

$$
Y=(s_1,s_2,s_3),
$$

Decoder 依次预测：

$$
P(s_1\mid X),
$$

$$
P(s_2\mid X,s_1),
$$

$$
P(s_3\mid X,s_1,s_2).
$$

通过 Beam Search 可以同时搜索多条高概率 SID 路径，再映射回真实 SKU。OxygenREC 使用这种 Encoder–Decoder 生成式推荐作为整个系统的基础骨架。

### 1.1.2 仅靠用户历史，推不出所有需求

普通 GR 主要学习：

$$
\text{Past Behavior}
\rightarrow
\text{Future Item}.
$$

这种方法本质上仍然是从日志中学习统计共现。

例如用户最近反复比较两款拍照手机，纯行为模型可能认为用户同时喜欢这两款手机，于是继续推荐相似商品；但真正的潜在需求可能是“用户非常重视手机摄影能力，但现有两款都没有完全满足需求”。

类似地，时间、地点和用户画像的组合也可能形成日志中很少出现的新需求。例如某个地区、某种天气和特定家庭结构共同决定当前购买需求，这些关系不一定能直接从历史点击中学出来。

OxygenREC 因此区分：

* **inductive matching**：从过去发生过的行为中总结统计模式；
* **deductive reasoning**：借助世界知识与上下文推出日志中没有直接出现的潜在需求。

但如果直接让大型 LLM 在线参与每次推荐请求，又无法满足工业推荐的严格延迟和成本要求。

### 1.1.3 Fast-Slow Thinking：把推理与生成拆开

OxygenREC 的解决方案不是把在线 Generator 直接替换成大 LLM，而是把整个系统拆成两条不同时间尺度的路径。

**Slow Thinking：**

near-line LLM 读取用户近期行为、搜索、时间、地点和画像等信息，提前推理当前潜在需求，并生成 Contextual Reasoning Instruction：

$$
\text{Context + Behavior}
\rightarrow
\text{LLM}
\rightarrow
I_r.
$$

**Fast Thinking：**

在线 Encoder–Decoder 不再做复杂世界知识推理，而是直接读取提前生成好的 (I_r)：

$$
(X,I_s,I_r)
\rightarrow
\text{Encoder--Decoder}
\rightarrow
Y.
$$

Slow Thinking 可以使用更强、计算更慢的模型；Fast Thinking 只负责高吞吐实时生成。

在线 serving 阶段不需要调用 LLM，而是直接根据 User ID 读取提前存储的 instruction embedding，因此复杂推理不会直接进入推荐请求的 latency critical path。论文还使用短时间窗口聚合同一用户的行为变化，例如在一个约 5 分钟滑动窗口内只重新生成一次 instruction，以减少 near-line 写入压力。

### 1.1.4 一个模型如何同时服务不同推荐场景？

京东 App 中不同推荐场景的输入和业务目标存在明显差异，例如：

| 场景                         | 当前上下文                  | 典型目标              |
| -------------------------- | ---------------------- | ----------------- |
| Homepage                   | 用户本身，没有明确 trigger item | 激发兴趣、Click        |
| Channel Feed               | 用户点击某个入口 SKU 后进入 Feed  | 深入浏览、Cart / Order |
| I2I Related Recommendation | 当前正在查看一个主商品            | 找相关/补充商品          |
| Add-to-Cart Overlay        | 用户刚将商品加入购物车            | 转化、追加购买           |
| Checkout Add-on            | 用户准备支付                 | 强购买意图下的附加销售       |

如果分别训练：

$$
Scenario_1\rightarrow Model_1,
$$

$$
Scenario_2\rightarrow Model_2,
$$

$$
\cdots
$$

会产生大量独立 checkpoint、训练链路和在线服务。

OxygenREC 不为每个场景复制 Generator，而是把场景本身也变成 Instruction：

$$
I_s={\text{Scenario Information},\text{Optional Trigger Item}}.
$$

例如 Homepage 没有 trigger item，而 Channel Feed、I2I、购物车等场景可以将当前商品作为 trigger。

于是同一个模型学习：

$$
P(Y\mid X,I_s,I_r),
$$

由 (I_s) 改变不同场景下的候选分布和生成行为，从而实现论文所称的：

> **train-once-deploy-everywhere。**

## 1.2 模型收益

### 1.2.1 实验设置

**训练数据：**

论文使用京东 App 搜索与多个推荐场景的真实工业数据进行联合训练，包括 Search、Homepage、Channel Feed、I2I 等流量。论文没有公开具体样本数量或日志时间跨度，因此不应该自行补充数据规模。搜索数据提供天然文本 query；推荐场景在训练时没有文本 (I_r) 时使用 learnable default instruction embedding。

**模型评估指标：**

* HR@K：Top-K Beam Search 结果中是否存在与 Ground Truth 完全匹配的 SID 路径；
* Recall@K：用户真实正反馈商品中，有多少被 Top-K 生成结果覆盖。

Semantic ID 还单独评估：

* Codebook Coverage；
* Semantic Cluster Purity；
* SID Collision；
* Codebook Load Balance。

**模型规模：**

论文从 0.1B 一直扩展到 3.0B，总参数量和实际激活参数量如下：

| 模型          | Encoder Layers | Decoder Layers | Hidden / Intermediate | Experts Total / Active |
| ----------- | -------------: | -------------: | --------------------: | ---------------------: |
| 0.1B / 0.1B |              4 |              4 |            1024 / 512 |                  2 / 1 |
| 0.4B / 0.3B |              4 |              6 |           2048 / 1024 |                  4 / 2 |
| 0.7B / 0.4B |              4 |              8 |           2048 / 1024 |                  8 / 2 |
| 1.5B / 0.4B |              4 |              8 |           2048 / 1024 |                 24 / 2 |
| 3.0B / 0.6B |              4 |             16 |           2048 / 1024 |                 24 / 2 |

训练系统部署于 128 张 NVIDIA H800 GPU，并在统一 PyTorch sparse+dense 训练框架下达到约 40% MFU。

### 1.2.2 Backbone Scaling

随着 Generative Backbone 从 0.1B 增加到 3.0B，HR 和 Recall 整体持续提升：

| 模型规模 |  HR@1 |  HR@10 | Recall@10 | Recall@30 |
| ---- | ----: | -----: | --------: | --------: |
| 0.1B | 3.99% | 13.17% |    10.10% |    15.11% |
| 0.4B | 4.42% | 15.03% |    11.38% |    17.34% |
| 0.7B | 4.84% | 16.33% |    12.32% |    18.71% |
| 1.6B | 4.92% | 16.61% |    12.51% |    19.01% |
| 3.0B | 5.02% | 16.99% |    12.78% |    19.53% |

从 0.1B 扩展到 3.0B 后，HR@10 从 13.17% 提升到 16.99%。

值得注意的是，0.7B 到约 1.5/1.6B 之间出现一定平台期：两者每个 token 实际都只激活 2 个 expert，仅扩大 MoE expert pool 并没有同比增加每次前向的有效计算。3.0B 又增加到 16 层 Decoder 后，性能继续改善。因此 OxygenREC 的实验也说明，GR 的 scaling 不只是“总参数更多”，实际激活参数与计算路径深度同样重要。

### 1.2.3 Instruction 与 IGR 消融

Instruction Token 的插入位置对效果存在明显影响：

| Instruction 方式          |      HR@1 |      HR@10 | Recall@10 |  Recall@30 |
| ----------------------- | --------: | ---------: | --------: | ---------: |
| No Instruction          |     2.78% |     10.38% |     8.18% |     13.01% |
| Replace BOS             |     3.30% |     12.08% |     9.12% |     14.38% |
| Add to BOS              |     3.50% |     12.59% |     9.52% |     14.93% |
| Insert Left of BOS      |     3.33% |     12.17% |     9.21% |     14.50% |
| **Insert Right of BOS** | **3.53%** | **12.68%** | **9.58%** | **14.91%** |

最终模型保留标准 BOS，然后紧接一个 Instruction Token，再开始生成 SID：

$$
[\mathrm{BOS},\mathrm{Instruction},s_1,s_2,s_3,\ldots].
$$

论文进一步发现，在 (I_s) 内部，仅使用 Scenario ID 或 Trigger Item 都不如二者联合；将 Scenario 与 Trigger 先融合成一个统一 token 的效果最好。

IGR 与 Q2I 的消融结果：

| 配置          |      HR@1 |      HR@10 |  Recall@10 |  Recall@30 |
| ----------- | --------: | ---------: | ---------: | ---------: |
| w/o IGR/Q2I |     3.76% |     12.20% |      9.87% |     15.53% |
| + IGR       |     4.02% |     12.91% |     10.25% |     15.95% |
| + IGR + Q2I | **4.19%** | **13.38%** | **10.52%** | **16.23%** |

IGR 本身通过去除不相关长期行为取得提升；加入 Q2I 后进一步提高，说明 instruction 与 item space 的显式对齐能够改善历史检索质量。

### 1.2.4 Unified Model 与独立场景模型

论文比较：

* Pretrain + 每场景单独 SFT；
* 一个统一 Instruction-Following Model。

统一模型在六个核心场景中均明显优于 Independent SFT：

| Metric | Model           |         S1 |         S2 |         S3 |         S4 |         S5 |         S6 |
| ------ | --------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| HR@1   | Independent SFT |      6.39% |      8.17% |      1.12% |      1.83% |      7.22% |      5.29% |
| HR@1   | Unified         | **15.39%** | **20.75%** | **17.24%** |  **6.34%** | **10.54%** | **25.75%** |
| HR@10  | Independent SFT |     23.29% |     29.05% |      5.22% |      8.44% |     29.84% |     19.38% |
| HR@10  | Unified         | **46.73%** | **55.02%** | **53.57%** | **29.89%** | **37.90%** | **62.62%** |

因此 OxygenREC 的统一并不只是为了省部署成本；至少在论文报告的场景中，跨场景共享训练本身也带来了明显的知识迁移收益。

### 1.2.5 SA-GCPO

Post-training 以预训练 OxygenREC-0.7B MoE 为 warm start，并使用 policy 生成 candidate，再由 Unified Ranking Model 对 synthetic samples 打 reward。

在 synthetic data 比例为 33% 的实验中：

| 方法          |       HR@1 |      HR@10 |
| ----------- | ---------: | ---------: |
| GRPO        |     23.85% |     62.15% |
| GSPO        |     24.13% |     62.88% |
| **SA-GCPO** | **25.58%** | **65.95%** |

SA-GCPO 相比普通 GRPO 和 GSPO 都获得更高 HR，并且在改变 synthetic data 比例时表现更加稳定。

### 1.2.6 在线 A/B 实验

OxygenREC 最终部署覆盖三个用户生命周期阶段：

1. Homepage：Interest Triggering；
2. Channel Feed：Deep Exploration；
3. Checkout Path：Immediate Conversion。

实验组和对照组各占总流量的 10%。首次上线时的业务提升如下：

| 场景              |   UCTR | UCTCVR | Order Volume |     GMV |
| --------------- | -----: | -----: | -----------: | ------: |
| Homepage S1     | +0.68% | +2.71% |       +2.81% |  +4.52% |
| Homepage S2     | +3.55% | +2.26% |       +2.21% |  +8.40% |
| Channel Feed S3 | -0.25% | +7.89% |       +8.03% |  +1.46% |
| Channel Feed S4 | +0.78% | +2.17% |       +1.49% |  +1.66% |
| Checkout S5     | +0.40% | +4.21% |       +4.28% | +11.80% |
| Checkout S6     | +3.29% | +3.00% |       +2.92% |  +4.15% |

Homepage 和 Checkout 场景报告约 50ms latency，Channel Feed 为约 80ms。Scenario 3 中 OxygenREC 已经端到端替换原有 Recall 与 Pre-Rank，传统 Ranking 模型则保留为 Reward Mapping Service。

## 1.3 模型架构

### 1.3.1 总体架构

理解 OxygenREC 时，最重要的是不要把 SID、IGR、Q2I、Slow Thinking 和 SA-GCPO 看成几个平级网络。

真正的生成主干只有：

$$
\boxed{
\text{User History}
\rightarrow
\text{Encoder}
\rightarrow
\text{Decoder}
\rightarrow
\text{SID}
\rightarrow
\text{Item}
}
$$

其他模块分别服务这条主线：

| 模块                          | 解决的问题                           |
| --------------------------- | ------------------------------- |
| Multimodal SID              | 商品如何变成 Decoder 可以生成的离散 ID       |
| Slow Thinking               | (I_r) 从哪里来                      |
| Scenario Instruction (I_s)  | 当前在哪个场景生成                       |
| Reasoning Instruction (I_r) | 当前用户可能真正需要什么                    |
| IGR                         | 用户长期历史太长，Encoder 应该看哪些          |
| Q2I                         | IGR 如何判断某条历史是否与 instruction 相关  |
| NTP                         | Decoder 怎么学会生成 Ground Truth SID |
| Reward Mapping              | 怎么把多个业务目标转换成统一 reward           |
| SA-GCPO                     | 怎么让预训练 Generator 进一步符合业务目标      |

因此 OxygenREC 的完整逻辑不是：

> SID → IGR → Q2I → Encoder → NTP → RL

这样的串行结构。

而是：

$$
\text{SID / Slow Thinking / IGR}
\rightarrow
\boxed{\text{Encoder--Decoder Generator}}
\leftarrow
\text{Instruction},
$$

预训练以后再：

$$
\boxed{\text{Generator}}
\rightarrow
\text{Reward}
\rightarrow
\text{SA-GCPO}.
$$

### 1.3.2 商品表示：Multimodal Semantic ID

Decoder 不直接面对十亿级 SKU vocabulary。

OxygenREC 首先为商品建立统一 Semantic ID。

商品包含：

* textual metadata；
* product image。

文本和图像分别通过独立 encoder：

$$
x_i^{text}\rightarrow E_{text},
$$

$$
x_i^{image}\rightarrow E_{image}.
$$

不同模态先投影到统一表示空间，再经过 Q-Former 和 MLP 建模 cross-modal interaction，最终得到一个 256 维连续商品 embedding：

$$
e_i\in\mathbb R^{256}.
$$

随后利用 RQ-KMeans 做三级 residual quantization：

$$
e_i
\rightarrow
(s_i^1,s_i^2,s_i^3).
$$

每一级 codebook：

$$
|V_l|=8192,
$$

总深度：

$$
L=3.
$$

因此一个商品最终可以表示成：

$$
SID_i=(s_i^1,s_i^2,s_i^3).
$$

例如：

$$
\text{Item A}
\rightarrow
(351,827,6192).
$$

三级 code 并不是三个独立标签，而是一条 coarse-to-fine 路径。Decoder 最终学习的就是这种 SID sequence。

论文经历了四代 SID：

* V1：Textual；
* V2：MiniCPM multimodal；
* V3：Fusion；
* V4：Multi-Source。

最终 V4 的一级 Cluster Purity 达到 92.80%，P999 collision 降低到 35。论文也发现，与使用大型端到端多模态 backbone 相比，将 Qwen3 text encoder 与 CLIP image encoder 分开提取，再使用专门 Q-Former+MLP 做融合，不仅 SID 质量更高，还能获得最高约 32× 的 embedding inference 加速。

### 1.3.3 Encoder：把用户过去做过什么编码成用户状态

Encoder 输入 (X_{\text{enc}}) 包含三类用户侧信息：

1. User Profile；
2. Short-term Behavior；
3. 经过 IGR 筛选后的 Long-term Behavior。

可以抽象写成：

$$
X_{\text{enc}}
==============

[
X_{\text{profile}};
X_{\text{recent}};
X_{\text{related}}
].
$$

其中 Recent Behavior 保留近期兴趣变化；Long-term History 不直接全部进入 Encoder，而是由 IGR 根据当前 instruction 筛选。

经过 Transformer Encoder：

$$
Z_{\text{enc}}
==============

Encoder(X_{\text{enc}}),
$$

得到 Decoder 可以 cross-attend 的用户历史表示。

需要注意：OxygenREC 原文明确说明 item inputs 使用 multimodal SID 表示，但**没有完整公开每个历史 item 的三级 SID 在 Encoder 内部究竟如何聚合、一个 item 对应几个 Encoder positions、各种 profile/behavior feature 的完整 serialization 方式**。因此这部分不能进一步自行假设成“每个历史 item 必然展开成三个 SID token”。

### 1.3.4 Slow Thinking：生成 (I_r)

Contextual Reasoning Instruction 并不是由在线 Generator 自己推出来的，而是来自独立 near-line LLM pipeline。

论文主要构建三类 reasoning：

**Spatiotemporal and Profile Reasoning**

结合：

* 时间；
* 地点；
* 节日；
* 天气；
* 用户画像；

推断当前环境下可能产生的购买需求。

**User Query Rewrite**

对错误、截断或语义不完整 query 做补全和标准化。

论文使用 DeepSeek-R1 构造部分合成训练数据，并对 Qwen3-0.6B SFT；人工评估通过率达到 95.33%。

**User Intent Reasoning**

根据用户行为序列和偏好寻找更深层 latent intent。训练数据利用 Qwen3-32B 做 intent aggregation / behavior filtering，再由 DeepSeek-R1 生成 rationale pseudo-label，最终 Qwen3-0.6B 的输出在人工评估中达到 72% usability。

最终 textual instruction 再经过 Adapter 投影为：

$$
I_r.
$$

论文明确将 (I_r) 定义为：

> 从 textual instruction 投影得到的 dense embedding。

线上只读取这一 embedding，不在线运行 Slow LLM。

### 1.3.5 Dual Instruction：(I_s) 和 (I_r)

最终 Decoder 同时接受两类控制变量。

#### Scenario Instruction (I_s)

由：

$$
I_s=(s,z)
$$

组成，其中：

* (s)：scenario information；
* (z)：optional trigger item。

没有 trigger 时使用：

$$
z=z_{\mathrm{def}}.
$$

例如：

| 场景           | Scenario     | Trigger    |
| ------------ | ------------ | ---------- |
| Search       | Search       | Default    |
| Homepage     | Homepage     | Default    |
| Channel Feed | Channel      | Entry Item |
| I2I          | Related Rec. | Main Item  |

#### Contextual Reasoning Instruction (I_r)

表示用户当前 latent intent。

训练阶段存在一个重要的 train-serving gap：

* Search 有天然 query，因此用 rewritten/normalized query 训练 (I_r)；
* Homepage、Channel、I2I 等推荐场景通常没有文本 reasoning label，因此训练时使用 learnable default embedding；
* 线上 serving 时再读取 Slow Thinking pipeline 真正产生的 near-line instruction。

因此搜索日志实际上充当了：

> **训练 Generator 学会“如何服从文本语义 instruction”的天然监督数据。**

最终模型形式为：

$$
P(Y\mid X,I_s,I_r).
$$

论文将 Composite Instruction Token 放到 BOS 右侧：

$$
[\mathrm{BOS},\mathrm{Instruction},s_1,s_2,\ldots].
$$

这里要注意：论文明确说明 composite prompt 包含 (I_s) 和 (I_r)，但没有完整公开两者最终融合为 Decoder instruction representation 的全部算子细节；不能简单写死为 (Concat(I_s,I_r))。

### 1.3.6 IGR：Instruction 决定 Encoder 看哪些长期历史

假设一个用户长期历史包含：

* 手机；
* 零食；
* 登山鞋；
* 洗衣液；
* 帐篷；
* 冲锋衣。

当前 (I_r) 表示：

> 用户最近可能准备户外徒步。

如果全部送给 Encoder，手机和洗衣液等旧兴趣可能干扰当前生成。

IGR 的目标就是：

$$
\text{Long History}
\xrightarrow{I_s,I_r}
\text{Top-K Relevant History}.
$$

但 Instruction 和历史商品原本不在同一个 embedding space，所以模型先分别构造 Query、Target Item、History Item 表示。

Instruction：

$$
e_q=
Concat[
\phi_{\text{scn}}(I_s),
g^{train}(I_r^{text})
],
$$

$$
q=\psi_q(e_q).
$$

训练阶段 Target Item：

$$
e_t=
Concat[
\phi_{\text{item}}(v_t),
\phi_{\text{side}}(u_t),
g^{train}(x_t)
],
$$

$$
t=\psi_i(e_t).
$$

历史商品：

$$
e_h=
Concat[
\phi_{\text{item}}(v_h),
\phi_{\text{side}}(u_h),
g^{frozen}(x_h)
],
$$

$$
h=\psi_i(e_h).
$$

注意 Target Item 与 History Item 共用：

$$
\psi_i.
$$

因此只要训练：

$$
q\approx t,
$$

就可以让 query 进入同一 item retrieval space；线上没有 target item 时，仍可以使用：

$$
sim(q,h_i)
$$

检索长期历史。

### 1.3.7 Q2I：训练 IGR 的检索空间

Q2I Loss 使用 Ground Truth Target Item 对齐 instruction query。

论文定义：

$$
\mathcal L_{\text{Q2I}}
=======================

-\frac1B\sum_{i=1}^{B}q_i\cdot t_i
+
\lambda_r
\left(
-\log[\operatorname{Var}(Q)\operatorname{Var}(T)]
\right)
+
\lambda_d
\frac{1}{B^2-B}
\sum_{i\ne j}(q_i^\top q_j)^2.
$$

第一项负责 instruction–target alignment；

第二项避免 embedding dimension collapse；

第三项降低 batch 内表示冗余。

因此：

> **Q2I 不是另一个推荐任务，而是在训练“Instruction 到商品空间的检索坐标系”。**

训练完成以后：

$$
q
\rightarrow
TopK({h_i})
\rightarrow
X_{\text{related}},
$$

IGR 才能真正根据当前 instruction 筛选 Long-term History。

### 1.3.8 Pre-training：NTP + Q2I

OxygenREC 将 Search、Homepage、Channel Feed、I2I 等场景混合训练。

统一目标：

$$
\mathcal L
==========

\mathcal L_{\text{NTP}}
+
\lambda\mathcal L_{\text{Q2I}}.
$$

其中：

* Q2I：训练 instruction 与 item space 对齐；
* NTP：训练 Decoder 生成真实 Target Item 的 SID。

假设 Target Item：

$$
SID^*=(351,827,6192),
$$

Decoder 就学习：

$$
351\rightarrow827\rightarrow6192.
$$

OxygenREC 使用 **Weighted NTP**，让高价值用户行为拥有更大的训练权重：

$$
Purchase > Cart > Click.
$$

这里非常重要：

> v1 中 Behavior 主要用于决定某条生成监督“有多重要”，而不是作为显式 Decoder condition。

论文没有公开 Purchase、Cart、Click 的具体权重数值。

### 1.3.9 Post-training：为什么预训练以后还需要 RL？

NTP 解决的是：

> 根据历史日志，能不能生成用户真实交互过的商品？

但工业系统最终还需要优化：

* Format；
* Relevance；
* Conversion；
* GMV；
* Diversity。

因此预训练完成以后，当前 policy 针对同一请求生成一组 candidate：

$$
{y_i}_{i=1}^{G}.
$$

这些商品进入 Reward Mapping Service：

$$
Generator
\rightarrow
Candidates
\rightarrow
Reward Mapping
\rightarrow
R_i.
$$

Reward 包含四类：

1. Format Reward：SID 是否有效；
2. Relative Reward：商品与当前用户上下文 / query 是否相关；
3. Ranking Reward：Unified Ranking Model 对 GMV、Conversion 等目标的预测；
4. Diversity Reward：整组生成结果是否足够多样。

总 Reward 是这些信号的 scenario-aware 加权组合。

### 1.3.10 Unified Ranking Model：RL 的外部业务价值判断器

OxygenREC 没有为每个场景维护单独 Reward Model，而是训练一个统一多场景 Ranking Model。

Ranking Model 将 heterogeneous features 构造成统一 representation tokens，通过共享 Transformer 做跨场景 feature interaction；同时利用 label packing 把传统 point-wise sample 重组成按用户 request trajectory 排列的 list-wise sample，并用 causal masking 建模用户行为路径。

它最终作为在线 Reward Mapping Service：

$$
(User,Candidate)
\rightarrow
Ranking\ Reward.
$$

因此 v1 的 Generator 并没有完全自行掌握 Click、Cart、Order、GMV 等 discriminative value，而是在生成 candidate 后由一个外部 ranking model 给出业务价值反馈。

### 1.3.11 SA-GCPO：多场景统一 RL

对于同一输入 (x)，old policy 生成：

$$
{y_i}_{i=1}^{G}.
$$

首先根据 reward 做组内标准化：

$$
\widehat A_i
============

\frac{
R_i-\operatorname{mean}({R_i})
}{
\operatorname{std}({R_i})
}.
$$

普通 GRPO 会把：

$$
\widehat A_i>0
$$

直接视为需要强化的 candidate。

但这存在一个问题：

如果这一组 candidate 全都很差，其中“最不差”的一个仍可能得到正 advantage。

OxygenREC 因此引入真实 Target Item 的 reward：

$$
R_g^*.
$$

当：

$$
\widehat A_i>0
$$

但：

$$
R_i<R_g^*,
$$

SA-GCPO 将这一 positive advantage 置零：

$$
\Gamma^{adv}
============

0.

$$

也就是：

> **不能因为一个商品是“这一批生成结果里最好”，就认为它真的足够好；至少还要经过真实用户反馈对应 Target 的 reward 门槛。**

第二，SA-GCPO 不采用传统 hard clipping，而定义 soft adaptive gate：

$$
f_{i,t}(\rho)
=============

\sigma\left(
\tau_{i,t}(\rho-1)
\right)
\frac{4}{\tau_{i,t}}.
$$

对应梯度权重：

$$
w_{i,t}
=======

4p_{i,t}(1-p_{i,t}).
$$

当新 policy 与 old policy 接近，即：

$$
r_{i,t}\approx1,
$$

梯度权重最大；随着 policy 偏离越来越大，梯度连续、平滑衰减。

因此可以把：

* PPO / GRPO hard clip 理解成“撞墙”；
* SA-GCPO 理解成“逐渐踩刹车”。

第三，SA-GCPO 为正负 advantage 设置不同温度：

$$
\tau_{\text{pos}},
\qquad
\tau_{\text{neg}}.
$$

实验发现：

$$
\tau_{\text{pos}}>\tau_{\text{neg}}
$$

时训练更稳定；较低的 negative temperature 可以让负样本梯度更快衰减，降低 RL collapse 风险。

因此 SA-GCPO 的核心不是重新定义业务 reward，而是解决：

> **同一个 Generator 同时面对多个场景、多个 reward 时，如何稳定地把“真正好的生成路径”概率提高，而又避免 policy 被 noisy / relative reward 带偏。**

### 1.3.12 在线 Serving

最终线上请求不调用 Slow LLM。

Near-line：

$$
\text{User Context}
\rightarrow
LLM
\rightarrow
I_r
\rightarrow
\text{Store}.
$$

在线请求：

$$
UserID
\rightarrow
\text{读取 }I_r.
$$

然后：

$$
\text{User Profile}
+
\text{Recent Behavior}
+
\text{IGR Related History}
\rightarrow
Encoder,
$$

同时：

$$
I_s+I_r
\rightarrow
Decoder\ Instruction.
$$

Decoder 自回归生成 SID，并使用 prefix-constrained Beam Search：

$$
(s_1,s_2,s_3)
\rightarrow
Item.
$$

Prefix Constraint 用于保证 SID 路径合法，同时执行不同场景的 candidate pool 和业务规则约束。

论文线上使用的 GR workload 具有“长用户历史输入 + 短生成输出 + 大 Beam”的特点，典型 beam size 达到 256–512，因此专门基于 xLLM 开发 xGR serving engine，对 scheduling、KV Cache、Attention 和 Beam Search 排序进行优化。

最终 OxygenREC 可以压缩成三层：

**第一层：Generative Recommendation 主干**

$$
\boxed{
用户历史
\rightarrow
Encoder
\rightarrow
Decoder
\rightarrow
SID
\rightarrow
商品
}
$$

**第二层：Instruction Control**

$$
\boxed{
I_s=\text{现在在哪个场景}
}
$$

$$
\boxed{
I_r=\text{用户现在可能真正想要什么}
}
$$

二者直接控制 Decoder 的生成分布。

**第三层：围绕主干的训练和辅助机制**

$$
\boxed{
\text{IGR + Q2I}
\rightarrow
\text{让 Encoder 看对历史}
}
$$

$$
\boxed{
\text{NTP}
\rightarrow
\text{让 Decoder 学会生成真实商品}
}
$$

$$
\boxed{
\text{Reward Mapping + SA-GCPO}
\rightarrow
\text{让 Generator 更符合真实业务目标}
}
$$

所以 OxygenREC 最值得记住的并不是某一个复杂模块，而是：

> **它把“复杂推理”“场景控制”“用户历史建模”“商品生成”和“业务目标对齐”拆到了不同时间尺度和不同计算位置：Slow LLM 负责推理，Instruction 负责控制，Encoder 负责理解用户历史，Decoder 负责生成商品 SID，RL 再负责把生成策略对齐到真实工业价值。**
