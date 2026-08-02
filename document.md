# MDL：当排序主干变大，场景与任务如何参与每一层计算？

工业推荐系统很少只面对一个数据分布。同一套搜索或推荐服务，可能同时覆盖单列、双列、内搜等展示场景，并预测点击、点赞、收藏等多个行为。长期以来，我们用 SharedBottom、MMoE、STAR、PEPNet 等结构回答同一个问题：哪些知识应该共享，哪些能力应该按场景或任务区分？

MDL 提出了一个更适合大型排序主干的问题：**当大部分参数和计算已经进入深层 Feature backbone，场景与任务信息仍只在末端的 gate、tower 或 head 中出现，新增容量还能否被充分利用？**

它的回答是“Tokenize-and-Interact”：把 Feature、Scenario 和 Task 表示为三类 token。Feature token 负责共享特征交互；Scenario/Task token 以 Query 的身份逐层读取 Feature token；当前场景状态再进入各 Task token，最终由 Task token 产生预测。[MDL 原文](https://arxiv.org/html/2602.07520v2)

> **MDL 的核心不是“增加几个 token”，而是让 Scenario 与 Task 从末端控制信号变成跨层保留、逐层读取共享 Feature 状态的条件化表示。**

<!-- IMAGE_PLACEHOLDER_START: FIG_01 -->
> **【插图占位｜图 1】大主干下的条件路径**
>
> **内容：** 对比场景与任务只在末端生效，以及它们在每一层读取 Feature states 的两种计算逻辑，说明当主要参数和计算进入深层 backbone 后，条件信息覆盖范围为何比“有没有条件模块”更重要。
>
> **图注：** 主干越深，场景与任务越需要参与中间表示的形成，而不是只在输出端分流。
<!-- IMAGE_PLACEHOLDER_END: FIG_01 -->

---

## 1. 多场景、多任务真正难在哪里

考虑两个搜索展示场景：单列搜索与双列搜索。系统同时预测两个任务：Click 与 Favorite。

|  | Click | Favorite |
|---|---|---|
| Single-column | $$D_{single,click}$$ | $$D_{single,fav}$$ |
| Double-column | $$D_{double,click}$$ | $$D_{double,fav}$$ |

<!-- IMAGE_PLACEHOLDER_START: FIG_02 -->
> **【插图占位｜图 2】Scenario × Task 形成的多分布问题**
>
> **内容：** 说明场景改变样本产生方式，任务改变监督目标，因此每个 Scenario–Task 组合都是相关但不同的预测分布；同时对比完全独立训练与完全共享训练的收益和代价。
>
> **图注：** 多场景多任务学习需要在共享统计信息与保留分布差异之间取得平衡。
<!-- IMAGE_PLACEHOLDER_END: FIG_02 -->

这四个格子不是四个普通类别，而是四个相关但并不相同的预测分布。场景影响样本怎样产生：同屏候选数量、位置偏置、浏览节奏、曝光竞争关系和上下文特征都可能变化。任务则改变监督信号的定义：Click 通常更高频、更接近即时相关性；Favorite 更稀疏，也更依赖长期兴趣和内容价值。

对来自场景 $$s$$ 的样本 $$x$$，任务 $$t$$ 的预测可写为：

$$
\hat y_{s,t}=p_\Theta(y_t=1\mid x,s).
$$

Scenario 改变样本如何产生，例如 $$p(x\mid s)$$；Task 改变标签定义以及 $$p(y_t\mid x,s)$$。前者描述输入分布，后者描述预测目标，两者都需要共享知识，也都需要保留差异。[MDL §2](https://arxiv.org/html/2602.07520v2)

最直接的方案是训练四套独立模型。这样每个组合都能拥有专属参数，但会重复特征工程、训练、部署和监控链路，稀疏任务也难以利用其他分布的统计信息。另一个极端是完全共享：所有样本通过同一主干，仅在末端分出不同 head。它具有较高统计效率，但共享表示容易受高流量场景与高频任务主导，分布差异只能由较浅的输出模块恢复。

所以，多场景多任务学习的第一性问题不是“是否共享”，而是：

> **共享什么，区分什么，以及在计算图的什么位置开始区分。**

---

## 2. 条件路径为何跟不上大主干

过去几类模型不断把 Task 或 Scenario 信息向计算图内部推进，但它们作用的范围并不相同：

| 模型 | 条件如何进入计算 | 仍然受限的部分 |
|---|---|---|
| [SharedBottom](https://people.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/caruana-1997.pdf) | 共享表示，任务差异留给末端 head | 主干前向不感知任务 |
| [MMoE](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/) | 每个任务用独立 gate 组合 shared experts | 条件只覆盖 expert module |
| [STAR](https://dl.acm.org/doi/10.1145/3459637.3481941) / [PEPNet](https://dl.acm.org/doi/10.1145/3580305.3599884) | 场景信息改变参数或激活 | 多场景与多任务通常仍由不同模块处理 |
| [RankMixer](https://arxiv.org/html/2507.15551v1) | 把主要容量放进深层 Feature-interaction blocks | 本身不提供逐层的 Scenario/Task 状态 |

<!-- IMAGE_PLACEHOLDER_START: FIG_03 -->
> **【插图占位｜图 3】不同模型的条件信息进入了多深**
>
> **内容：** 比较 SharedBottom、MMoE、STAR/PEPNet、RankMixer 与 MDL 中 Task/Scenario 信息的作用位置和覆盖范围，重点说明条件信息能够影响主干的哪些层、哪些参数和哪些计算过程。
>
> **图注：** 模型差异不仅在于是否使用场景或任务信息，还在于条件路径覆盖了多少模型容量。
<!-- IMAGE_PLACEHOLDER_END: FIG_03 -->

真正的转折来自 RankMixer：规则、可堆叠的 Feature blocks 让 dense 参数和 FLOPs 大量进入深层 backbone。模型虽然变大了，但如果 Scenario/Task 仍只出现在末端 gate 或 tower，新增容量在前向中依旧是 scenario-agnostic、task-agnostic 的。

因此，问题不再只是“共享多少参数”，而是：**条件路径是否覆盖了模型的主要容量。** MDL 正是从这里切入。

---

## 3. MDL 的答案：共享 Feature，逐层读取

MDL 的设计判断是：当主要容量位于多层 Feature backbone 时，Scenario/Task 也应该拥有与主干深度相匹配的状态演化路径。它们不必为每个任务重新计算整套 Feature backbone，但需要在每一层重新决定“当前场景或任务应从 Feature states 中聚合什么”。[MDL Introduction](https://arxiv.org/html/2602.07520v2)

Feature、Scenario、Task 被映射为固定数量、固定宽度的 latent tokens，共用 attention、residual、Per-token FFN 等规则计算模块。“统一”指统一计算接口，不是让三类 token 共享语义或职责。

### 3.1 为什么要保留独立的 Domain states？

一个样本通常只有一个当前 Scenario，因此把 Scenario embedding 拼入输入、用 FiLM 调制每层或加到每个 Feature token，都可以让 Feature update 直接依赖场景。[FiLM 原文](https://arxiv.org/abs/1709.07871)

Task 更麻烦：同一曝光样本要同时预测 Click、Like、Favorite。如果把单个 task ID 作为整个 backbone 的条件，一次前向只能表达一个任务；若希望 backbone 对每个任务产生不同 Feature states，要么按任务重复运行，要么在模型内部维护并行的 task-specific states。MDL 选择后者，但只让这些 states **读取**共享 Feature backbone，从而保留一次 Feature computation。

几种条件化方式的计算差异如下：

| 条件化位置 | 条件是否直接改写 Feature update | 是否能一次输出全部任务 | 条件是否跨层保留独立状态 |
|---|---|---|---|
| 输入拼接 Scenario embedding | 是 | 是 | 否，条件被吸收到共享激活中 |
| 末端 tower/head | 否 | 是 | 否 |
| 局部 MMoE gate | 只改变所在 expert module | 是 | 通常没有独立递推状态 |
| MDL Domain token | 否；逐层读取 Feature | 是 | 是 |
| Feature/Domain 联合双向 self-attention | 是；Domain 可写回 Feature | 是 | 是 |

<!-- IMAGE_PLACEHOLDER_START: FIG_04 -->
> **【插图占位｜图 4】末端、读取侧与写入侧条件化**
>
> **内容：** 对比条件只在输出端生效、Domain state 逐层读取共享 Feature、条件直接改写 Feature 三种机制，说明它们在条件影响范围、主干复用、重复计算和任务干扰上的差异。
>
> **图注：** MDL 选择读取侧条件化：保留共享 Feature computation，同时让不同 Domain states 形成持续演化的专属表示。
<!-- IMAGE_PLACEHOLDER_END: FIG_04 -->

MDL 优先保持昂贵 Feature path 的共享，用 $$N_s+N_t$$ 条较窄的 Domain states 获得分布特定读取；相应代价是 Feature states 在单次前向中仍不具备 task-specific 写入。它的关键不在 token 形式本身，而在于让条件路径覆盖已经迁入深层 backbone 的主要容量。

---

## 4. 三类 token 与有向信息流

MDL 的完整状态可以写为：

$$
F^{(l)}\in\mathbb{R}^{N_f\times d_f},\qquad
S^{(l)}\in\mathbb{R}^{(N_s+1)\times d_s},\qquad
T^{(l)}\in\mathbb{R}^{N_t\times d_t}.
$$

其中 $$N_f$$ 是 Feature token 数量，$$N_s$$ 是业务场景数，额外的 1 对应 Global Scenario token，$$N_t$$ 是预测任务数。三类状态在交互前通过 projection 对齐到相同的 attention width。

<!-- IMAGE_PLACEHOLDER_START: FIG_05 -->
> **【插图占位｜图 5】MDL 整体模型架构**
>
> **内容：** 完整说明 Feature、Scenario、Task 三类 token 的初始化与跨层传播：Feature 先完成共享交互；Scenario/Task 分别以 Query 读取 Feature；active Scenario 与 Global Scenario 聚合后进入所有 Task states；最终每个 Task state 连接对应输出层。还需体现不存在独立的 Task–Task 或 Scenario–Scenario mixing，最终输出数为 $$N_t$$ 而非 $$N_sN_t$$。
>
> **图注：** MDL 共享 Feature backbone，并用逐层演化的 Scenario/Task states 完成多分布条件化。
<!-- IMAGE_PLACEHOLDER_END: FIG_05 -->

信息流分为三类：

第一类是 `Feature → Feature`。Feature tokens 经过共享 Feature backbone，不断形成高阶特征交互。

第二类是 `Feature → Scenario/Task`。Scenario/Task token 作为 Query，Feature token 作为 Key/Value。不同 Domain states 对同一组 Feature states 形成不同权重和聚合结果。

第三类是 `Selected Scenario → Task`。每个样本只选取与其对应的 Scenario token，同时保留 Global Scenario token；二者聚合后进入所有 Task tokens。最终只有 Task tokens 连接输出 head。[MDL §3.2–3.3](https://arxiv.org/html/2602.07520v2)

论文称这一部分为 Domain-aware All-Token Interaction，但它不是把三类 token 拼在一起做普通 self-attention。MDL 的“统一”体现在 latent 表示宽度和重复交互协议；Feature、Scenario、Task 仍承担不同的计算角色，信息流具有明确方向。

### 4.1 完整前向

完整计算可以写成：

```python
F = feature_tokenizer(raw_features)          # [B, Nf, d]
S = scenario_tokenizer(domain_features)      # [B, Ns + 1, d]
T = task_tokenizer(task_features)            # [B, Nt, d]

for block in blocks:
    F = block.feature_interaction(F)

    S_read = block.scenario_attn(S, F) + S
    S_next = block.scenario_ffn(S_read) + S_read

    T_read = block.task_attn(T, F) + T
    s_active = masked_mean(
        S_read, scenario_mask, include_global=True
    )                                        # [B, d]
    T_fused = T_read + s_active[:, None, :]
    T_next = block.task_ffn(T_fused) + T_fused

    S, T = S_next, T_next

logits = [head_t(T[:, t]) for t in range(Nt)]
```

`S_read` 同时承担两项职责：一条路径经 Scenario FFN 形成下一层状态 `S_next`，另一条路径选出 active Scenario 并写入当前层的 Task states。[MDL §3.3](https://arxiv.org/html/2602.07520v2) `scenario_mask` 决定哪些 Scenario states 进入 Task；`label_mask` 则决定哪些 Task logits 进入监督损失，两者属于不同计算环节。

<!-- IMAGE_PLACEHOLDER_START: FIG_06 -->
> **【插图占位｜图 6】Specific/Global Scenario 与两类 Mask**
>
> **内容：** 说明 specific Scenario state 只由对应场景样本更新，Global Scenario state 由所有场景共享更新；scenario mask 负责选择进入 Task 的 active Scenario，label mask 负责决定哪些任务损失有效，两种 mask 相互独立。若样本属于多个重叠场景，需要同时表达多个 specific states 与 Global state 的聚合逻辑。
>
> **图注：** 场景路由与标签可用性是两个不同的计算问题。
<!-- IMAGE_PLACEHOLDER_END: FIG_06 -->

---

## 5. Tokenization 不是形式变化：三类 token 从哪里来

### 5.1 Feature token：把异构字段组织成固定计算槽位

工业推荐输入通常包括 user、item、query、context、统计交叉和多个行为序列。离散字段先经过 embedding table，序列字段通过 DIN、LONGER 等序列模块编码，最终得到宽度不一的表示。MDL 沿用 RankMixer 的 semantic tokenization：人工按业务语义把字段分成 $$N_f$$ 组，每组拼接后投影到固定维度。[MDL Eq. 2–3](https://arxiv.org/html/2602.07520v2)；[LONGER 原文](https://arxiv.org/abs/2505.04421)

$$
e_{input}=[e_1;e_2;\ldots;e_{N_f}],
\qquad
t_j=\operatorname{Proj}(e_j),
$$

$$
F^{(0)}=[t_1;t_2;\ldots;t_{N_f}].
$$

在这个最小例子中，可以将数百个字段压缩为四个语义组：

| Feature token | 可能包含的字段 |
|---|---|
| User | user ID、画像、活跃度、长期统计 |
| Query | query ID、query category、query statistics |
| Item | item/video ID、author、内容类别、质量统计 |
| History | 点击序列、搜索序列及其编码结果 |

这不是语言模型中的词表分词，也不是向量量化，更不是自动发现 feature clusters。token boundary 由业务语义与人工设计决定。它决定哪些字段先在 token 内融合、哪些信息必须通过后续 Token Mixing 交换，也决定不同 Per-token FFN 分别负责什么子空间。因此，Feature grouping 不是无关紧要的预处理，而是模型结构的一部分。

论文使用“discrete latent tokens”描述结果，但公式中的 token 是连续向量。这里的“discrete”更适合理解为有限个、位置明确的 latent slots，而不是离散码或 vocabulary ID。

### 5.2 Scenario token：不是场景 ID 的简单查表

论文以 Scenario token 为例解释 Domain token initializer。每个 Scenario token 的输入由两部分组成：

1. 若干 important raw features 的额外 embedding，例如 user ID、video ID；这些 embedding 与 Feature 路径使用相同 raw field，但参数不同；
2. scenario-specific prior，例如该场景下的用户行为序列。

随后，每个 Scenario token 经过自己的 Per-token FFN：[MDL Eq. 4](https://arxiv.org/html/2602.07520v2)

$$
s_k^{(0)}=\operatorname{ReLU}
\left(\operatorname{FFN}_k^s
(\hat e_{imp}\oplus\hat e_{spec}^{k})\right).
$$

如果有 $$N_s$$ 个业务场景，模型构造 $$N_s$$ 个 specific Scenario tokens，并额外构造一个 Global Scenario token：

$$
S^{(0)}=[s_1^{(0)};\ldots;s_{N_s}^{(0)};s_{global}^{(0)}].
$$

Global token 用于学习跨场景共性；specific token 负责保留各场景差异。

Global token 是跨场景共享槽位，每个样本都会使用；specific token 则只服务对应场景。

### 5.3 Task token：样本相关的任务状态

Task token 采用相似构造：task-related features 与 important features 经过独立 Per-token FFN，得到每个任务的初始状态。[MDL §3.1.2](https://arxiv.org/html/2602.07520v2)

这意味着 Task token 不是固定的 Click/Favorite ID embedding，而是由当前样本特征生成的动态状态。同一个 Click task 面对不同用户、query 和 candidate，会得到不同的初始 Task token。

论文使用 LLM prompt 解释 Scenario/Task token 的作用，这个类比只成立一部分。P-Tuning v2 等 deep prompt tuning 方法通常在冻结的预训练语言模型中，把少量连续 prompt 放入多层；MDL 则与 Feature backbone 联合训练，而且 token 由当前样本的 raw features 与 prior 生成。因此，MDL 的 Domain token 更接近 **sample-conditioned latent query**，而不是经典 PEFT 中用于调用冻结知识的固定 prompt。[P-Tuning v2 原文](https://arxiv.org/abs/2110.07602)；[Prefix-Tuning 原文](https://arxiv.org/abs/2101.00190)

### 5.4 Initializer 决定 Domain token 携带什么

同一个 raw field 可以在两条路径中使用两套 embedding。以 video ID 为例，一路进入 Item Feature token，另一路以额外 embedding 进入某个 Scenario/Task initializer。二者接收不同梯度，并通过不同带宽到达输出。

因此，MDL 相比基线新增的并不只是几个 attention queries，至少还包括：

- important features 的额外 embedding；
- scenario/task-specific prior；
- 每个 Domain token 的独立 FFN；
- 多层 residual state；
- Domain-to-Feature cross-attention。

这会直接影响论文消融的解释。删除整个 Task token 会同时删除额外信息、额外参数和交互路径；它可以证明完整 Task-token path 有效，却不能单独证明 layer-wise cross-attention 是全部收益来源。

Feature token 是共享证据的结构化表示；Scenario/Task token 是由样本和分布先验初始化的条件状态。两类 token 形状相似，但信息职责不同。

<!-- IMAGE_PLACEHOLDER_START: FIG_07 -->
> **【插图占位｜图 7】三类 Token 的初始化来源与职责**
>
> **内容：** 说明 Feature token 由按业务语义组织的原始字段构成，Scenario/Task token 由身份信息、重要字段的独立 embedding 和特定先验初始化；同一 raw field 可以通过不同参数进入不同路径。重点区分共享证据、场景条件和任务条件三种职责，并说明 Domain token 是样本相关的动态状态，而非固定 prompt。
>
> **图注：** 三类 token 使用统一计算接口，但承载不同信息并由不同参数路径初始化。
<!-- IMAGE_PLACEHOLDER_END: FIG_07 -->

---

## 6. 一个 MDL Block：严格按执行顺序拆开

设第 $$l$$ 层输入为 $$F^{(l)},S^{(l)},T^{(l)}$$。每层按三步执行。

<!-- IMAGE_PLACEHOLDER_START: FIG_08 -->
> **【插图占位｜图 8】一个 MDL Block 的完整执行逻辑**
>
> **内容：** 严格表达一个 block 内的先后关系：Feature 先更新；Scenario 和 Task 分别以 Query 读取新的 Feature states；Scenario residual state 一路经 Scenario FFN 传播到下一层，另一路选取 active Scenario 与 Global Scenario并聚合；聚合结果进入所有 Task states；每个 Task 再经自己的 FFN 形成下一层状态。
>
> **图注：** 每层先生成共享 Feature states，再由 Scenario/Task 完成条件化读取与融合。
<!-- IMAGE_PLACEHOLDER_END: FIG_08 -->

### 6.1 第一步：Feature tokens 先完成共享特征交互

抽象地写，Feature 路径为：

$$
F^{(l+1)}=B_l(F^{(l)}).
$$

默认实现采用 RankMixer：Token Mixing 负责跨 Feature tokens 交换信息，Per-token FFN 负责每个语义子空间的非线性变换。[MDL Eq. 6](https://arxiv.org/html/2602.07520v2)

$$
F^{(l+1)}=\operatorname{PerTokenFFN}
\left(\operatorname{LN}
(\operatorname{TokenMixing}(F^{(l)})+F^{(l)})\right).
$$

这部分可以替换为其他 Feature-interaction methods；MDL 只要求每层输出一组新的 Feature states。[MDL §3.2.1](https://arxiv.org/html/2602.07520v2)

### 6.2 第二步：Scenario 与 Task 以 Query 读取 Feature

对任意一组 Domain tokens $$D\in\{S,T\}$$，Domain-aware Attention 可写成标准 cross-attention 形式：

$$
Q=D^{(l)}W_Q,\qquad
K=F^{(l+1)}W_K,\qquad
V=F^{(l+1)}W_V,
$$

$$
\operatorname{Attn}(D,F)=
\operatorname{softmax}
\left(\frac{QK^\top}{\sqrt {d_h}}\right)V.
$$

其中 $$d_h$$ 是单个 attention head 的 Key 维度。

Q/K/V 的方向定义了模型语义：Scenario/Task 是 Query，Feature 是 Key/Value。[MDL Eq. 7–8](https://arxiv.org/html/2602.07520v2)

Q/K/V projection 采用 Per-token FFN 形式，使不同 Domain states 可以学习不同的读取映射。

加入 batch 与多头维度后，关键 shape 为：

| 张量 | 单头简化 shape | 含义 |
|---|---:|---|
| Feature state $$F$$ | $$[B,N_f,d]$$ | Key/Value 来源 |
| Scenario state $$S$$ | $$[B,N_s+1,d]$$ | Scenario Query |
| Task state $$T$$ | $$[B,N_t,d]$$ | Task Query |
| Scenario attention map | $$[B,N_s+1,N_f]$$ | 每个 Scenario 对 Feature 的读取权重 |
| Task attention map | $$[B,N_t,N_f]$$ | 每个 Task 对 Feature 的读取权重 |
| Scenario mask | $$[B,N_s]$$ | 选择 active Scenario；Global 始终参与 |

若有 $$H$$ 个 attention heads，权重张量相应变为 $$[B,H,N_d,N_f]$$，其中 $$N_d$$ 分别取 $$N_s+1$$ 或 $$N_t$$。这张 shape 表也是最直接的单元测试清单：Query 轴与 Feature 轴一旦互换，模型语义就已经不再是论文中的 MDL。

<!-- IMAGE_PLACEHOLDER_START: FIG_09 -->
> **【插图占位｜图 9】Domain-aware Cross-Attention 的张量与信息方向**
>
> **内容：** 说明 Query 来自 Scenario/Task states，Key 和 Value 来自 Feature states，attention 输出只更新 Domain states。需要体现 $$[B,H,N_d,N_f]$$ 中 Domain Query 轴与 Feature 轴的含义，并解释不同 Scenario/Task 可以对同一组 Feature states 形成不同读取分布；交换 Q 与 K/V 将改变模型语义。
>
> **图注：** MDL 的关键方向是 Domain 读取 Feature，而不是 Domain 写回 Feature。
<!-- IMAGE_PLACEHOLDER_END: FIG_09 -->

Click 与 Favorite 面对同一组 Feature states，可以产生不同 attention 分布；Single 与 Double 也可以读取不同 Feature 子空间。完成 attention 后，Scenario path 加 residual，再通过 Per-scenario FFN：[MDL Eq. 11–12](https://arxiv.org/html/2602.07520v2)

$$
\hat S^{(l+1)}=\operatorname{DomainAwareAttn}
(S^{(l)},F^{(l+1)})+S^{(l)},
$$

$$
S^{(l+1)}=\operatorname{PerTokenFFN}_s
(\hat S^{(l+1)})+\hat S^{(l+1)}.
$$

Task path 首先采用相同读取方向：

$$
\hat T^{(l+1)}=\operatorname{DomainAwareAttn}
(T^{(l)},F^{(l+1)})+T^{(l)}.
$$

### 6.3 第三步：把当前 Scenario state 融入所有 Task states

Scenario 与 Task 不会一直平行传播。对每个样本，模型根据 scenario information 选出对应的 specific Scenario token，同时保留 Global token。若一个样本属于重叠场景，也可以选出多个 specific tokens。这里被选中的是 attention 与 residual 之后、Scenario FFN 之前的 $$\hat S^{(l+1)}$$。论文使用简单 Mean Pool：[MDL Eq. 9](https://arxiv.org/html/2602.07520v2)

$$
s_{avg}=\operatorname{MeanPool}
(\{\hat s_{active,1},\ldots,\hat s_{active,m},\hat s_{global}\}).
$$

随后把同一个 $$s_{avg}$$ 加入所有 Task tokens：[MDL Eq. 10](https://arxiv.org/html/2602.07520v2)

$$
\tilde T^{(l+1)}=\hat T^{(l+1)}+s_{avg}.
$$

每个 Task token 再经过自己的 FFN 与 residual：[MDL Eq. 13–15](https://arxiv.org/html/2602.07520v2)

$$
T^{(l+1)}=\operatorname{PerTokenFFN}_t
(\tilde T^{(l+1)})+\tilde T^{(l+1)}.
$$

堆叠 $$L$$ 层后，第 $$t$$ 个 Task token 连接第 $$t$$ 个 logits layer：[MDL Eq. 16](https://arxiv.org/html/2602.07520v2)

$$
\hat y_t=\operatorname{LogitsLayer}_t(T_t^{(L)}).
$$

论文列出的三类 interaction 是 Feature–Feature、Feature–Scenario/Task 和 Scenario–Task，并没有独立的 Task–Task interaction。因此，Task token 解决的是 task-aware representation；它不会自动表达 Click→Purchase 等任务之间的方向性依赖。

最终输出数量是 $$N_t$$，而不是 $$N_sN_t$$。场景差异已经在每一层通过 selected Scenario state 进入 Task state，因此 Click head 可以跨场景复用。

“All-Token Interaction”并不是所有 token 两两 self-attention，其信息方向如下：

| 被更新的状态 | 读取 Feature | 读取 Scenario | 读取 Task |
|---|---|---|---|
| Feature | 是：Feature self-interaction | 否 | 否 |
| Scenario | 是：Scenario=Q，Feature=K/V | 仅同一 token 的 residual 与 FFN；没有 Scenario–Scenario mixing | 否 |
| Task | 是：Task=Q，Feature=K/V | 是：只读 active Scenario 与 Global 的均值 | 仅同一 token 的 residual 与 FFN；没有 Task–Task mixing |

这也澄清了 Global Scenario token 的来源：它不是由所有 specific Scenario tokens 聚合得到的。它是一条独立 token path，因为每个样本都选择它，所以会接收所有场景样本的梯度；“global”来自更新覆盖范围，而不是显式 Scenario–Scenario aggregation。[MDL Eq. 9–15](https://arxiv.org/html/2602.07520v2)

一个 MDL Block 的顺序是：**Feature 先计算；Scenario 和 Task 分别读取 Feature；当前 Scenario 再进入 Task；三类状态继续传播到下一层。**

### 6.4 哪些是 MDL 核心，哪些属于默认 RankMixer 实现

Feature self-interaction 可以替换，但需要保留 MDL 的核心计算契约。

| 层次 | 内容 | 变更后的性质 |
|---|---|---|
| MDL 核心契约 | 三类 token；Domain=Query、Feature=K/V；逐层状态；active Scenario + Global 融入 Task；Task 输出 head | 改变任一项都会改变 MDL 的主要方法 |
| 论文默认实现 | RankMixer Token Mixing、Feature Per-token FFN、人工语义 Feature grouping | 论文允许替换 Feature interaction，但实验结论基于该默认实现 |

---

## 7. 用最小实例走完两层前向与梯度

设隐藏维度 $$d=8$$，使用四个 Feature tokens、两个业务场景加一个 Global token、两个 Task tokens：

| 张量 | Shape | 槽位 |
|---|---:|---|
| $$F^{(0)}$$ | $$[4,8]$$ | User、Query、Item、History |
| $$S^{(0)}$$ | $$[3,8]$$ | Single、Double、Global |
| $$T^{(0)}$$ | $$[2,8]$$ | Click、Favorite |

对一个 Single-column 样本，第一层先更新四个 Feature tokens。User slot 可以结合 Query、Item 和 History；Item slot 也可以吸收用户与搜索上下文。随后 Click 和 Favorite 读取同一组更新后的 Feature states。设第一层得到以下平均 attention 权重：

| Task Query | User | Query | Item | History |
|---|---:|---:|---:|---:|
| Click | 0.15 | 0.35 | 0.30 | 0.20 |
| Favorite | 0.20 | 0.10 | 0.45 | 0.25 |

这些数值是教学构造，不来自论文实验。它们表达一种可检验行为：Click 可能更关注 query relevance，Favorite 可能更关注 item content 与长期兴趣。Scenario states 也读取相同 Feature states，但学习的是单列、双列展示条件下不同的信息组合。

当前样本属于 Single，因此融合阶段只选 Single 与 Global：

$$
s_{avg}^{(1)}=\frac{\hat s_{single}^{(1)}+\hat s_{global}^{(1)}}{2}.
$$

这里的两个向量都已经读取第一层 Feature states 并完成 residual，但尚未经过 Scenario FFN。同一份 $$\hat S^{(1)}$$ 一方面经 Scenario FFN 形成下一层输入 $$S^{(1)}$$，另一方面在当前层完成 Scenario–Task fusion。这一融合向量分别加入 Click 与 Favorite。进入第二层时，Task Query 已不再只是 initializer 输出：它已经包含第一层读取的样本证据以及当前场景状态。第二层 attention 因此可以在更深 Feature states 上执行新的条件化聚合。这就是论文所说 bottom-up、layer-wise interaction 的具体含义。

如果把同一用户—候选对放到 Double 场景，即使暂时假设普通 Feature inputs 完全相同，Scenario fusion 也会改为 Double + Global。Task heads 仍只有 Click 和 Favorite 两个；场景差异通过 Task state 到达共享的任务输出映射。

<!-- IMAGE_PLACEHOLDER_START: FIG_10 -->
> **【插图占位｜图 10】两层 MDL 前向的最小实例**
>
> **内容：** 使用 User、Query、Item、History 四个 Feature slots，Single、Double、Global 三个 Scenario states，以及 Click、Favorite 两个 Task states，走完一个 Single-column 样本的两层前向。第一层需表达 Click 与 Favorite 对同一 Feature states 的不同读取，以及 Single + Global 如何进入两个 Task；第二层需说明 Task Query 已包含 initializer、第一层证据和场景状态，因此会在更深 Feature states 上重新读取。切换到 Double 场景时，Task heads 不变，只改变进入 Task 的场景状态。
>
> **图注：** Task state 在每一层累积样本证据与场景信息，而不是只在末端区分任务。
<!-- IMAGE_PLACEHOLDER_END: FIG_10 -->

### 7.1 参数如何共享：必须沿 loss 反向看

前向图说明信息怎样流动，梯度图才说明知识怎样共享。参数归属可以概括为：

| 状态或参数 | 主要更新来源 |
|---|---|
| Feature embeddings/backbone | 全部场景、全部任务 loss 共享更新 |
| Global Scenario path | 全部场景共享更新 |
| Specific Scenario path | 被选中场景的样本更新 |
| Task token / Task FFN | 对应任务 loss 更新，跨场景复用 |
| Task logit head | 对应任务 loss 更新 |

论文特别指出，某场景样本只会更新被选中的 specific Scenario token；Global token 则是跨场景共享路径。[MDL §3.3](https://arxiv.org/html/2602.07520v2)

这里必须区分两个经常被混为一谈的问题。

**表示冲突**指不同任务需要不同信息，却只能使用同一份最终共享表示。MDL 为不同任务保留跨层 Task states，并允许它们形成不同 Feature 读取模式，因此直接针对这一瓶颈。

**梯度冲突**指不同任务对共享参数产生方向不一致的更新，例如 $$g_i^\top g_j<0$$。所有 Task loss 仍会通过 attention 的 K/V 和 Feature path 回传到共享 backbone，MDL 没有显式 gradient projection、normalization 或 detach。因此，不能从 Task token 的存在推出梯度冲突已经被解决。

2026 年提出的 OneRank 同时设计 task-private forward channels 与 cross-task gradient detachment，恰好说明 forward specialization 与 backward isolation 是两个独立设计维度。[OneRank 原文](https://arxiv.org/abs/2606.16838)

因此，MDL 让任务在前向中“读到不同表示”，但共享 Feature backbone 在反向中仍接受多个任务的梯度。Forward specialization 与 backward isolation 是两个独立维度。

<!-- IMAGE_PLACEHOLDER_START: FIG_11 -->
> **【插图占位｜图 11】前向专属化与反向共享更新**
>
> **内容：** 同时表达两条逻辑：前向中不同 Task/Scenario states 形成不同读取表示；反向中所有任务损失仍会经 attention 的 Key/Value 路径更新共享 Feature backbone。还要区分 Global Scenario由全部场景更新、specific Scenario由对应场景更新、Task private path由对应任务损失更新，并指出MDL没有显式梯度投影或detach。
>
> **图注：** MDL缓解共享表示瓶颈，但前向专属化不等于反向梯度隔离。
<!-- IMAGE_PLACEHOLDER_END: FIG_11 -->

---

## 8. MDL 的四个结构边界

### 8.1 MDL 是读取侧条件化

观察 Feature 更新式：

$$
F^{(l+1)}=B_l(F^{(l)}).
$$

右侧没有 $$S^{(l)}$$ 或 $$T^{(l)}$$。从该层前向依赖关系看：

$$
\frac{\partial F^{(l+1)}}{\partial S^{(l)}}=0,
\qquad
\frac{\partial F^{(l+1)}}{\partial T^{(l)}}=0.
$$

随后 Scenario/Task 才用 $$F^{(l+1)}$$ 作为 Key/Value 更新自己。于是，MDL 的主要结构不是“不同任务重新生成不同 Feature”，而是：

$$
\text{共享主干生成 Feature states}
\quad+\quad
\text{不同 Domain states 逐层执行不同读取}.
$$

这是一种 **read-side conditioning**：Scenario/Task 进入了逐层交互，却不直接改写同层的 Feature path。

这个零 Jacobian 只描述单个 Block 的前向依赖，不表示三类状态统计独立。它们可以由相同 raw features 初始化，也会通过同一组 loss 耦合；所有任务 loss 仍会沿 Key/Value 路径更新 Feature backbone。

昂贵的 Feature computation 因此可以保持共享，不必为每个任务重跑完整 backbone；每个 Domain 只增加较小的、逐层演化的读取状态。这与 FiLM 或双向 self-attention 的 write-side modulation 不同。[FiLM](https://arxiv.org/abs/1709.07871)

### 8.2 MDL 对 Scenario × Task 空间做了因子化

若为每个组合维护独立 tower，需要 $$N_sN_t$$ 套场景—任务输出路径。MDL 的组合近似为：

$$
H_{s,t}=\phi_t
\left(T_t+\operatorname{Mean}(S_s,S_{global})\right),
$$

其中 $$\phi_t$$ 是 Task-specific nonlinear mapping。联合差异由 Task state、Scenario state 及 Task-specific FFN 组合表达，而不是每个 $$(s,t)$$ 都拥有完整独立参数。

这个因子化带来两个结果。第一，Task head 可以跨场景复用，最终输出数量不随场景数相乘。第二，模型施加了一个强假设：复杂 pair-specific interaction 能够被低维状态相加与后续 Task FFN 充分恢复。

Mean + Add 也隐含两项具体假设：被选中的 Scenario states 权重相同；同一个 Scenario summary 以相同方式先进入所有 Task tokens，再由各 Task FFN 产生差异。

当一个样本同时属于多个场景时，Mean Pool 还会稀释 Global token 与单个 specific token 的权重。因此，MDL 的融合更适合作为低成本、低方差的 factorized baseline；复杂场景关系可能仍需要 task-conditioned aggregation。

### 8.3 输出成本解耦，不等于中间成本消失

MDL 将输出 head 数量从潜在的 $$N_sN_t$$ 降到 $$N_t$$。但按论文公式直接实现，$$N_s+1$$ 个 Scenario tokens 与 $$N_t$$ 个 Task tokens 都要先读取 $$N_f$$ 个 Feature tokens，之后才选择 active Scenario。cross-attention 的主要交互规模近似为：

$$
O\big((N_s+N_t)N_fd\big),
$$

不同 Domain token 的 Per-token FFN 参数和计算也大致随 $$N_s+N_t$$ 增长。

工程实现可以在 Domain-aware Attention 之前只 gather 当前 active Scenario 与 Global token，从而避免计算所有无关 Scenario states；但这是由稀疏执行得到的系统优化，不是论文已经描述和测量的实现。正式比较时应分别报告“按论文公式的 dense token path”和“active-token path”的成本。

所以，更准确的结论是：**MDL 解耦了场景数与最终输出头数量，但没有完全解耦场景/任务数量与中间参数、FLOPs 和访存。** 当任务从 3 个扩展到几十个时，active-token execution、shared/private FFN、task clustering 或稀疏容量才会成为新的系统问题。

### 8.4 固定 token slots 带来的是闭集组合，不是零样本扩展

MDL 初始化固定的 $$N_s+1$$ 个 Scenario slots 和 $$N_t$$ 个 Task slots；不同 slot 拥有 Per-token FFN，最终每个 Task slot 还连接独立 LogitsLayer。[MDL Eq. 4、16](https://arxiv.org/html/2602.07520v2) 因此，任务或场景身份不仅编码在输入特征中，也编码在参数槽位和输出索引中。

这使已知 Scenario × Task 的自由组合很便宜：新增组合不必新增一套完整 tower。但“组合已知槽位”与“处理从未训练过的新场景或新任务”是两个问题。若上线一个新的展示形态，原论文没有给出如何凭描述或少量样本生成新 Scenario token、如何把它映射到已有 Per-token FFN，或如何在不重训的情况下校准各 Task head。类似地，新增任务仍需要新的 Task initializer、Task FFN、LogitsLayer 和监督信号。

所以，MDL 的扩展性应准确表述为：**对训练时定义好的 Scenario/Task 闭集进行因子化组合**。它没有直接解决 unseen scenario、隐式子场景发现或 zero-shot task generalization。工程上若场景 ID 只是粗粒度枚举，还应检查同一 slot 内是否存在显著的流量来源、页面形态或人群子分布差异。

<!-- IMAGE_PLACEHOLDER_START: FIG_12 -->
> **【插图占位｜图 12】MDL 的结构收益与边界**
>
> **内容：** 集中说明三项边界：输出 head 从潜在的 $$N_sN_t$$ 因子化为 $$N_t$$，但中间 cross-attention 与私有 FFN 成本仍随 Scenario/Task 数量增长；固定 Scenario/Task slots 支持训练时已知组合，却不等于新场景或新任务的零样本扩展。需要把“输出解耦”“中间成本仍存在”“闭集组合”三个结论同时讲清楚。
>
> **图注：** MDL减少组合式输出重复，但没有消除中间计算扩展，也没有自然获得开放集泛化。
<!-- IMAGE_PLACEHOLDER_END: FIG_12 -->

---

## 9. 论文证据及其边界

MDL 的证据链包含工业离线主实验、组件消融、规模趋势、attention 可视化和一个月线上 A/B test，但不同实验支持的结论强度并不相同。

### 9.1 数据与对照

论文使用 Douyin Search 连续两个月日志，覆盖单列、双列、内搜三个场景、20 多个预测任务和 500 多个 user、item、sequence、cross features，并保留 1% 数据用于评估。[MDL §4.1.1](https://arxiv.org/html/2602.07520v2)

离线主要报告 Click、Like、Favorite 在三个场景上的 QAUC：先在每个 UID-query group 内计算 AUC，再对 group 平均。

$$
\operatorname{QAUC}=\frac{1}{N}\sum_{i=1}^{N}\operatorname{AUC}_i.
$$

比较方法包括 RankMixer、SharedBottom、MMoE、STAR、HMoE、PEPNet。除纯 RankMixer 外，其他多分布结构都接入 RankMixer backbone，总参数控制在约 0.5B。[MDL §4.1.3–4.1.4](https://arxiv.org/html/2602.07520v2) 这保证了近似等参数比较，但不等同于 activated FLOPs、访存和线上时延完全一致。

### 9.2 主实验：九个组合方向一致

Table 1 的 `Improv.` 行以每个场景—任务格子的最强 baseline 为参照，直接报告 MDL 的相对 QAUC 提升：

| 场景 | Click | Like | Favorite |
|---|---:|---:|---:|
| Single-column | +0.31% | +0.23% | +0.42% |
| Double-column | +0.27% | +0.63% | +0.77% |
| Inner Search | +0.25% | +0.34% | +0.71% |

<!-- IMAGE_PLACEHOLDER_START: FIG_13 -->
> **【插图占位｜图 13】九个场景—任务组合的相对 QAUC 提升**
>
> **内容：** 准确呈现三个场景、三个任务相对于各自最强 baseline 的九项相对提升，突出所有组合均为正，以及 Like、Favorite、Double-column、Inner Search 中部分弱势或复杂分布收益更大的现象。必须说明数值是相对提升率而非 QAUC 绝对百分点，且论文未提供置信区间。
>
> **图注：** MDL 在九个场景—任务组合上均取得正向相对提升。
<!-- IMAGE_PLACEHOLDER_END: FIG_13 -->

九个场景—任务组合全部提升，是论文最强的离线证据。Like、Favorite 以及双列、内搜的收益通常更大，说明相对弱势分布更可能从逐层共享读取中获益。[MDL Table 1 与 §4.2](https://arxiv.org/html/2602.07520v2) 表中数字是相对提升率，不是 QAUC 绝对百分点，也没有给出置信区间。

### 9.3 消融：Domain state 的证据强于具体算子的证据

论文在三个场景的 Click QAUC 上报告五项消融：

| 变体 | Single | Double | Inner |
|---|---:|---:|---:|
| w/o Task token | -0.12% | -0.11% | -0.09% |
| w/o Task–Feature interaction | -0.04% | -0.05% | +0.03% |
| w/o Scenario token | -0.17% | -0.16% | -0.15% |
| w/o Global Scenario token | -0.04% | -0.06% | -0.05% |
| w/o Scenario–Feature interaction | -0.05% | -0.04% | -0.05% |

<!-- IMAGE_PLACEHOLDER_START: FIG_14 -->
> **【插图占位｜图 14】组件消融及其证据强度**
>
> **内容：** 准确比较五项消融在三个场景上的 Click QAUC 变化。重点表达删除完整 Scenario/Task token 的下降更稳定，而替换 Feature interaction 的变化较小，Task–Feature interaction 在 Inner Search 上还出现 +0.03%。同时说明删除整个 token 会连带改变 initializer、embedding、private capacity 和交互路径，因此实验更强地支持显式 Domain states，而不是证明某个 cross-attention 算子的唯一性。
>
> **图注：** 消融对 Domain state 价值的支持强于对具体交互算子的支持。
<!-- IMAGE_PLACEHOLDER_END: FIG_14 -->

删除全部 Task token 或 Scenario token 在三个场景都造成稳定下降，支持显式 Domain state 的价值。Global Scenario token 的方向也一致。

但替换 Task–Feature interaction 后，Inner Search 反而为 `+0.03%`，其他 interaction 变体的幅度也较小。实验更有力地支持显式 Scenario/Task states，而不是某个特定 cross-attention 算子的唯一性。

同时，`w/o Task token` 使用 task tower 替代，`w/o Task–Feature interaction` 则替换为 RankMixer interaction。它们并没有在完全相同 initializer、完全相同 private capacity 下只改变“是否逐层读取”，因此不能独立拆出 extra embedding、prior、Per-token FFN 和 layer-wise attention 的各自贡献。[MDL Table 2 与 §4.3](https://arxiv.org/html/2602.07520v2)

### 9.4 Scaling：支持趋势，但还不是完整定律

论文改变 hidden dimension $$d$$ 与层数 $$L$$，比较 MDL 和 MMoE 随参数量、FLOPs 增加时的 Single-column Click QAUC gain。MDL 曲线持续高于 MMoE，而且差距随规模扩大。[MDL Figure 2](https://arxiv.org/html/2602.07520v2)

<!-- IMAGE_PLACEHOLDER_START: FIG_15 -->
> **【插图占位｜图 15】MDL 与 MMoE 的 Scaling 趋势（基于原文 Figure 2）**
>
> **内容：** 使用论文原始数据比较 hidden dimension 或层数增加、参数量和 FLOPs 上升时，MDL 与 MMoE 的 Single-column Click QAUC gain。需要表达两者均可能随规模增长，但 MDL 始终更高且差距扩大。不得补造实验点，并需限制结论为当前实验范围内的 scaling trend，而非普适 scaling law。
>
> **图注：** 当主干规模扩大，逐层 Domain states 比局部任务条件更能利用新增 Feature capacity。
<!-- IMAGE_PLACEHOLDER_END: FIG_15 -->

这个结果直接支持论文的核心动机：相比局部的 MMoE，layer-wise Domain states 更可能利用新增 Feature capacity。但从严格术语看，它更适合称为 **scaling trend**，还不是语言模型文献意义上的完整 scaling law：实验点数有限，只展示一个场景的 Click，没有幂律拟合、置信区间、数据量轴或 compute-optimal frontier。

因此，该实验支持“MDL 相对 MMoE 的优势随当前实验规模扩大”，尚不足以证明统一的 scaling law。

### 9.5 Attention 图：行为证据，不是因果证明

论文展示不同 Task/Scenario tokens 在不同层对 Feature tokens 的平均 attention distribution。不同任务、场景和层形成了不同权重，说明这些 Domain states 没有完全坍缩为相同读取行为。[MDL Figure 3–4](https://arxiv.org/html/2602.07520v2)

<!-- IMAGE_PLACEHOLDER_START: FIG_16 -->
> **【插图占位｜图 16】Task–Feature Attention 的跨任务与跨层差异（基于原文 Figure 3）**
>
> **内容：** 使用原文结果比较不同 Task tokens 对同一组 Feature tokens 的平均读取分布，并比较同一任务在不同层的读取变化。需要说明 Task state 从浅层到深层持续调整关注的信息，不同任务没有坍缩为同一种读取行为；不要为抽象 Feature token 添加原文未确认的业务语义。
>
> **图注：** 不同任务在不同层形成了不同的 Feature 读取模式。
<!-- IMAGE_PLACEHOLDER_END: FIG_16 -->

<!-- IMAGE_PLACEHOLDER_START: FIG_17 -->
> **【插图占位｜图 17】Scenario–Feature Attention 的跨场景与跨层差异（基于原文 Figure 4）**
>
> **内容：** 使用原文结果比较不同 specific Scenario 与 Global Scenario 对 Feature tokens 的平均读取分布，以及这些分布随层数的变化。需要表达场景差异在多层计算中持续积累，Global 与 specific Scenario 承担不同读取作用，并且Scenario states没有完全坍缩。
>
> **图注：** 不同场景在各层形成差异化 Feature 读取，Global Scenario 保留跨场景共享路径。
<!-- IMAGE_PLACEHOLDER_END: FIG_17 -->

Attention 图说明不同 Domain tokens 学出了不同读取分布，但不能单独证明这些差异就是性能提升的原因。输出还取决于 Value projection、向量范数、residual、Per-token FFN 和后续层。[Attention is not Explanation](https://aclanthology.org/N19-1357/)；[Attention is Not Only a Weight](https://aclanthology.org/2020.emnlp-main.574/)

### 9.6 线上结果：效果落地成立，系统成本仍未披露

论文在 Douyin Search 进行一个月 A/B test，线上 baseline 为 RankMixer + MMoE。LT30 指用户最近 30 天的平均活跃天数，越高越好；Change Query Rate 是发生手动 query reformulation 的 distinct UID-query 数占总 distinct UID-query 数的比例，论文把更低的数值解释为更高的搜索满意度。[MDL Eq. 18 与 §4.1.2](https://arxiv.org/html/2602.07520v2)

Table 3 报告的是相对变化，而不是指标绝对值：

| 范围 | Change Query Rate | LT30 |
|---|---:|---:|
| ALL | -0.3267% | +0.0626% |
| Single-column Search | -0.2678% | +0.0520% |
| Double-column Search | -0.5079% | +0.0674% |
| Inner Search | -0.5492% | +0.0630% |

<!-- IMAGE_PLACEHOLDER_START: FIG_18 -->
> **【插图占位｜图 18】一个月线上 A/B Test 结果**
>
> **内容：** 准确表达 ALL、Single-column、Double-column、Inner Search 四个范围上的 Change Query Rate 与 LT30 相对变化，并明确两个指标的有利方向分别是下降和上升。需要突出所有范围方向一致，以及 Double-column、Inner Search 的 Query 改写率下降更明显；同时指出结果不包含时延、吞吐和资源成本信息。
>
> **图注：** MDL 的离线收益转化为方向一致的线上用户行为改善，但系统成本仍未披露。
<!-- IMAGE_PLACEHOLDER_END: FIG_18 -->

两个指标在三个场景中都朝论文定义的有利方向变化。作者称模型已经全量部署，为数亿用户服务。[MDL Table 3 与 §4.5](https://arxiv.org/html/2602.07520v2)

这证明离线收益能够转化为生产业务指标，也是 MDL 很有价值的工业证据。但论文没有报告实验流量比例、置信区间、显著性检验、分桶样本量和 guardrail 指标，也没有报告相对 baseline 的 P50/P99 latency、QPS、MFU、峰值显存、通信量或 serving cost。因此目前能确认“作者报告了方向一致并最终全量的线上收益”，不能独立判断效应不确定性，也不能据此断言“在严格相同系统成本下更优”。

---

## 10. 复现前需要回答的五个工程问题

论文已经给出有效结构，但工程复现不能止于“把框图搭出来”。新增信息、新增参数、新增交互和新增成本需要分开测量。

### 10.1 增益来自 initializer，还是逐层读取？

最有信息量的对照不是简单删除整个 token，而是保持 initializer、输入字段和 private capacity 完全一致，只改变交互位置：

$$
\begin{aligned}
A&:\ \text{Same initializer + final tower},\\
B&:\ \text{Same initializer + one-shot cross-attention},\\
C&:\ \text{Same initializer + layer-wise cross-attention}.
\end{aligned}
$$

在这一控制下，$$B-A$$ 衡量条件化读取的价值，$$C-B$$ 衡量逐层读取的增量。还需要加入等字段、等参数的 input-concat 或 FiLM 基线，判断收益究竟来自 read-side factorization，还是额外条件信息本身。

<!-- IMAGE_PLACEHOLDER_START: FIG_19 -->
> **【插图占位｜图 19】拆分 Initializer、条件读取与逐层读取的控制实验**
>
> **内容：** 比较三组保持 initializer、输入字段和 private capacity 相同的模型：A 只在最终 tower 使用条件状态，B 只做一次 cross-attention，C 在每层执行 cross-attention。需要明确 $$B-A$$ 衡量条件化读取本身，$$C-B$$ 衡量逐层交互的增量，并补充等字段、等参数的 input-concat 或 FiLM 对照，用于区分信息、参数和交互位置的贡献。
>
> **图注：** 只有保持输入和容量一致，才能识别逐层读取的独立价值。
<!-- IMAGE_PLACEHOLDER_END: FIG_19 -->

### 10.2 Domain initializer 是否形成预测捷径？

论文明确允许 important raw features、specific priors 和独立 embedding 进入 Domain token。若 Task initializer 已直接包含 item ID、类目、价格、近期点击率或强转化统计，它可能在 cross-attention 之前就具备很强预测能力。

这不是说强 prior 一定有害，而是要区分两种模型：一种 Task state 主要表达“要读取什么”；另一种 Task state 已经携带大量“答案证据”。二者都可能提升 QAUC，但机制、泛化风险和部署依赖完全不同。

建议把字段按计算职责审计：

| 角色 | 定义 | 示例 |
|---|---|---|
| Control | 表示任务或场景身份 | scenario ID、task ID |
| Query-only | 帮助决定读取什么，但单独预测力有限 | 展示布局、请求类型 |
| Evidence | 直接携带用户—候选匹配证据 | item ID、内容特征、行为序列 |
| Wide/Prior | 强统计先验 | 历史 CTR、CVR、频次统计 |

最小诊断包括 initializer-only、Feature-only、Feature masking/shuffle，以及输出对 $$T^{(0)}$$ 与 $$F^{(l)}$$ 的梯度敏感度。

<!-- IMAGE_PLACEHOLDER_START: FIG_20 -->
> **【插图占位｜图 20】Domain Initializer 的信息职责与捷径审计**
>
> **内容：** 区分 Control、Query-only、Evidence、Wide/Prior 四类 initializer 字段，并解释两种可能机制：Domain state主要负责决定“读取什么”，或在cross-attention之前已经携带大量“答案证据”。需要结合 initializer-only、Feature-only、Feature masking/shuffle 和梯度敏感度等诊断，判断收益究竟来自读取机制还是initializer中的强预测信息。
>
> **图注：** Initializer既可能提供条件查询，也可能形成预测捷径，二者需要通过受控诊断区分。
<!-- IMAGE_PLACEHOLDER_END: FIG_20 -->

### 10.3 Mean + Add 是否足以表达业务关系？

MDL 对所有 active Scenario states 等权平均，再把同一结果加入所有 Task states。这在计算和部署上非常简单，也降低了小场景学习复杂 gate 的方差。但它要求 Task-specific FFN 能够吸收所有 Scenario–Task pair 差异。

对 Click、Like、Favorite 这类并行反馈，这个假设可能合理。对 impression→click→purchase 这类严格漏斗，MDL 没有显式 Task–Task probability decomposition 或 cascade information flow；“拥有 Task token”不等于“已经建模任务依赖”。

工程上应检查分场景 PCOC/ECE、logit distribution、标签窗口和采样机制。如果共享 Task head 的校准差异过大，可以先尝试 scenario-specific bias 或 task-conditioned scenario aggregation，而不是立刻退回 $$N_sN_t$$ 套完整 towers。

### 10.4 前向专属化是否仍受梯度冲突限制？

先区分两种现象：高流量任务因为样本数或 loss reduction 获得更大梯度，是**权重不平衡**；两个任务在共享参数上的梯度方向相反，才是**方向冲突**。二者可能同时发生，但解决方法不同。前者应先核对 sampling、label mask、task weight 与归一化，后者才需要梯度投影或隔离。

建议在共享 backbone 的代表层常驻监控各任务有效样本数、loss scale、梯度 norm、cosine similarity、负冲突率与训练速率。只有观察到稳定方向冲突并确认与尾部任务回退相关，才值得引入 GradNorm、PCGrad 或 OneRank 式 detach；否则逐任务梯度处理可能带来很高训练开销而缺少收益归因。[GradNorm](https://proceedings.mlr.press/v80/chen18a.html)；[PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)；[OneRank](https://arxiv.org/abs/2606.16838)

### 10.5 参数相同是否真的意味着生产成本相同？

MDL 引入更多 embedding tables、QKV projections、Per-token FFN、residual states 和 batch masking。即使总参数控制在 0.5B，也可能产生不同的 activated FLOPs、HBM bytes、kernel 数量和通信模式。

生产比较至少应同时报告：

| 类别 | 指标 |
|---|---|
| 模型质量 | 分场景/任务 QAUC、calibration、最差分布收益 |
| 计算 | Activated FLOPs、step time、MFU |
| 内存 | 峰值显存、HBM bytes/request、embedding cache miss |
| 在线 | QPS/GPU、P50/P95/P99 latency、timeout/fallback |
| 扩展 | 随 $$N_s,N_t,N_f,L,d$$ 增长的成本曲线 |

RankMixer 已经说明 Parameter Count、FLOPs 与实际系统成本不是同一个量；MDL 的公平评估也必须沿用这一系统视角。[RankMixer 原文](https://arxiv.org/html/2507.15551v1)

这些检查用于区分收益究竟来自信息、参数、交互、优化还是系统实现。

> **复现顺序应保持单变量：先证明 condition 有用，再证明 token 载体有用，最后验证 layer-wise interaction。**

---

## 结语：MDL 真正改变的是条件信息的地位

MDL 可以被简化成一个 Feature/Scenario/Task 三 token 模型，但这种描述没有抓住设计核心。

第一，Scenario 与 Task 不再只是静态 ID、gate input 或输出索引，而是由当前样本和分布先验初始化、可以跨层保留和更新的状态。

第二，它们不只在末端生效，而是在每一层以 Query 的身份读取 Feature states。Feature backbone 仍保持共享，因此模型不用为每个任务重复昂贵的 Feature computation。

第三，Scenario 与 Task 以 latent state 进行因子化组合：Scenario 表达当前输入分布，Task 表达预测目标；当前 Scenario 被聚合进 Task，输出规模只随任务数增长。

第四，这种组合仍建立在固定 Scenario/Task slots 上。MDL 降低了已知组合的结构重复，但没有自然获得 unseen scenario 或 zero-shot task 能力。

论文在九个场景—任务组合、组件消融、规模趋势和一个月线上 A/B test 中给出了较强效果证据。[MDL 原文](https://arxiv.org/html/2602.07520v2) 但工程复现仍应继续分离 initializer 容量、强 prior shortcut、layer-wise reading 的独立贡献、共享 backbone 的梯度冲突以及真实系统成本。

归根结底：

> **当共享 backbone 成为模型的主要容量载体时，条件信息不能只负责在末端选择答案，还需要拥有与主干深度匹配的信息读取路径。**

MDL 由此把多分布学习从末端分流推进到深层主干中的持续条件化读取。

---

## 参考文献

1. Mu, S., Jiang, Y., Wu, S., et al. [MDL: A Unified Multi-Distribution Learner in Large-scale Industrial Recommendation through Tokenization](https://arxiv.org/abs/2602.07520). KDD 2026（[会议论文列表](https://kdd2026.kdd.org/papers/)）。
2. Caruana, R. [Multitask Learning](https://people.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/caruana-1997.pdf). *Machine Learning*, 1997.
3. Ma, J., Zhao, Z., Yi, X., et al. [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/). KDD 2018.
4. Sheng, X.-R., Zhao, L., Zhou, G., et al. [One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction](https://dl.acm.org/doi/10.1145/3459637.3481941). CIKM 2021.
5. Chang, J., Zhang, C., Hui, Y., et al. [PEPNet: Parameter and Embedding Personalized Network for Infusing with Personalized Prior Information](https://dl.acm.org/doi/10.1145/3580305.3599884). KDD 2023.
6. Zhu, J., Fan, Z., Zhu, X., et al. [RankMixer: Scaling Up Ranking Models in Industrial Recommenders](https://arxiv.org/abs/2507.15551). 2025.
7. Liu, X., Ji, K., Fu, Y., et al. [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/abs/2110.07602). 2021.
8. Li, X. L., Liang, P. [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190). 2021.
9. Tang, J., Dai, S., Wang, K., et al. [OneRank: Unified Transformer-Native Ranking Architecture for Multi-Task Recommendation](https://arxiv.org/abs/2606.16838). 2026.
10. Jain, S., Wallace, B. C. [Attention is not Explanation](https://aclanthology.org/N19-1357/). NAACL 2019.
11. Kobayashi, G., Kuribayashi, T., Yokoi, S., Inui, K. [Attention is Not Only a Weight: Analyzing Transformers with Vector Norms](https://aclanthology.org/2020.emnlp-main.574/). EMNLP 2020.
12. Chai, Z., Ren, Q., Xiao, X., et al. [LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders](https://arxiv.org/abs/2505.04421). RecSys 2025.
13. Chen, Z., Badrinarayanan, V., Lee, C.-Y., Rabinovich, A. [GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://proceedings.mlr.press/v80/chen18a.html). ICML 2018.
14. Yu, T., Kumar, S., Gupta, A., et al. [Gradient Surgery for Multi-Task Learning](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html). NeurIPS 2020.
15. Perez, E., Strub, F., de Vries, H., Dumoulin, V., Courville, A. [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871). AAAI 2018.
