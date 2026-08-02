# MDL：当排序主干变大，场景与任务如何参与每一层计算？

同一套搜索或推荐系统，往往既要服务单列、双列、内搜等不同场景，又要同时预测点击、点赞、收藏等多个行为。SharedBottom、MMoE、STAR、PEPNet 等结构一直在处理同一个矛盾：既要共享统计信息，又要保留场景和任务差异。当排序模型的大部分参数与计算逐渐进入深层 Feature backbone，新的问题随之出现：**如果场景与任务信息仍只在末端的 gate、tower 或 head 中生效，扩大的主干是否真的学到了分布差异？** MDL 的做法是把 Feature、Scenario 和 Task 都表示为 token：Feature token 构成共享主干，Scenario/Task token 在每一层以 Query 的身份读取新的 Feature states，当前场景状态再融入各 Task state，最终由 Task token 输出预测。[MDL 原文](https://arxiv.org/html/2602.07520v2)

> **MDL 的核心不是“增加几个 token”，而是让 Scenario 与 Task 从末端控制信号变成跨层保留、逐层读取共享 Feature 状态的条件化表示。**

---

## 1. 多场景、多任务真正难在哪里

考虑两个搜索展示场景：单列搜索与双列搜索。系统同时预测两个任务：Click 与 Favorite。

|  | Click | Favorite |
|---|---|---|
| Single-column | $$D_{single,click}$$ | $$D_{single,fav}$$ |
| Double-column | $$D_{double,click}$$ | $$D_{double,fav}$$ |

这四个格子不是四个普通类别，而是四个相互关联却并不相同的预测分布。场景改变样本的产生方式，同屏候选数量、位置偏置、浏览节奏、曝光竞争和上下文特征都可能随之变化；任务则改变监督目标，Click 通常更高频、更接近即时相关性，Favorite 更稀疏，也更依赖长期兴趣和内容价值。对来自场景 $$s$$ 的样本 $$x$$，任务 $$t$$ 的预测可以写为：

$$
\hat y_{s,t}=p_\Theta(y_t=1\mid x,s).
$$

Scenario 主要改变 $$p(x\mid s)$$，Task 主要改变标签定义和 $$p(y_t\mid x,s)$$；前者对应输入分布，后者对应预测目标。[MDL §2](https://arxiv.org/html/2602.07520v2) 如果为每个组合训练独立模型，虽然可以保留充分差异，却会重复特征工程、训练、部署和监控链路，稀疏任务也难以借用其他分布的统计信息；如果完全共享主干、只在末端设置不同 head，统计效率更高，但共享表示容易被高流量场景和高频任务主导，复杂差异只能留给浅层输出模块补救。多场景多任务学习真正要决定的不是“是否共享”，而是：

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

<!-- IMAGE_PLACEHOLDER_START: FIG_01 -->
> **【插图占位｜图 1】不同模型的条件信息进入了多深**
>
> **内容：** 比较 SharedBottom、MMoE、STAR/PEPNet、RankMixer 与 MDL 中 Task/Scenario 信息的作用位置和覆盖范围，重点说明条件信息能够影响主干的哪些层、哪些参数和哪些计算过程。
>
> **图注：** 模型差异不仅在于是否使用场景或任务信息，还在于条件路径覆盖了多少模型容量。
<!-- IMAGE_PLACEHOLDER_END: FIG_01 -->

RankMixer 通过规则、可堆叠的 Feature blocks，把大量 dense 参数和 FLOPs 放进深层 backbone。主干虽然变大了，但如果 Scenario/Task 仍只出现在末端 gate 或 tower，新增计算在前向中依然不感知场景和任务。问题由此从“共享多少参数”转向了更关键的一点：**条件路径能否覆盖模型的主要容量。** MDL 正是沿着这个方向设计的。

---

## 3. MDL 的答案：共享 Feature，逐层读取

MDL 让 Scenario/Task 拥有与 Feature backbone 深度相匹配的状态路径：它们不为每个任务重复计算整套主干，而是在每一层重新判断当前场景或任务应从 Feature states 中聚合什么。[MDL Introduction](https://arxiv.org/html/2602.07520v2) 为此，Feature、Scenario、Task 都被映射为固定数量、固定宽度的 latent tokens，并通过 attention、residual、Per-token FFN 等规则模块持续更新。这里的“统一”是计算接口统一，而不是三类 token 的语义和职责相同。

### 3.1 为什么要保留独立的 Domain states？

一个样本通常只有一个当前 Scenario，因此无论把 Scenario embedding 拼入输入、用 FiLM 调制各层，还是直接加入每个 Feature token，都可以让 Feature update 感知场景。[FiLM 原文](https://arxiv.org/abs/1709.07871) Task 的处理更困难：同一曝光样本往往要同时预测 Click、Like、Favorite。如果用单个 task ID 条件化整个 backbone，一次前向只能服务一个任务；若要为各任务生成不同 Feature states，就必须按任务重复运行主干，或在模型内部维护并行的 task-specific states。MDL 采用后者，但只让这些 states **读取**共享 Feature backbone，从而保留一次 Feature computation。几种方式的差异如下：

| 条件化位置 | 条件是否直接改写 Feature update | 是否能一次输出全部任务 | 条件是否跨层保留独立状态 |
|---|---|---|---|
| 输入拼接 Scenario embedding | 是 | 是 | 否，条件被吸收到共享激活中 |
| 末端 tower/head | 否 | 是 | 否 |
| 局部 MMoE gate | 只改变所在 expert module | 是 | 通常没有独立递推状态 |
| MDL Domain token | 否；逐层读取 Feature | 是 | 是 |
| Feature/Domain 联合双向 self-attention | 是；Domain 可写回 Feature | 是 | 是 |

MDL 用 $$N_s+N_t$$ 条较窄的 Domain states 对共享 Feature path 进行分布特定读取，避免按任务重复昂贵的主干计算；代价是单次前向中的 Feature states 仍没有 task-specific 写入。它的关键不在 token 这一形式本身，而在于条件路径能够逐层触达深层 backbone 的主要容量。

---

## 4. 三类 token 与有向信息流

MDL 的完整状态可以写为：

$$
F^{(l)}\in\mathbb{R}^{N_f\times d_f},\qquad
S^{(l)}\in\mathbb{R}^{(N_s+1)\times d_s},\qquad
T^{(l)}\in\mathbb{R}^{N_t\times d_t}.
$$

其中 $$N_f$$ 是 Feature token 数量，$$N_s$$ 是业务场景数，额外的 1 对应 Global Scenario token，$$N_t$$ 是预测任务数。三类状态在交互前通过 projection 对齐到相同的 attention width。

<!-- IMAGE_PLACEHOLDER_START: FIG_02 -->
> **【插图占位｜图 2】MDL 整体模型架构**
>
> **内容：** 完整说明 Feature、Scenario、Task 三类 token 的初始化与跨层传播：Feature 先完成共享交互；Scenario/Task 分别以 Query 读取 Feature；active Scenario 与 Global Scenario 聚合后进入所有 Task states；最终每个 Task state 连接对应输出层。还需体现不存在独立的 Task–Task 或 Scenario–Scenario mixing，最终输出数为 $$N_t$$ 而非 $$N_sN_t$$。
>
> **图注：** MDL 共享 Feature backbone，并用逐层演化的 Scenario/Task states 完成多分布条件化。
<!-- IMAGE_PLACEHOLDER_END: FIG_02 -->

三类 token 之间存在清晰的有向信息流。Feature tokens 先在共享 backbone 中完成 `Feature → Feature` 的高阶交互；随后 Scenario 与 Task 分别作为 Query，以 Feature states 为 Key/Value，形成各自的读取权重和聚合结果；最后，模型为当前样本选出 active Scenario，并与 Global Scenario 聚合后写入所有 Task states，只有 Task tokens 连接输出 head。[MDL §3.2–3.3](https://arxiv.org/html/2602.07520v2) 论文将这套机制称为 Domain-aware All-Token Interaction，但它并不是把三类 token 拼接后做普通 self-attention。“统一”体现在 latent 宽度和重复交互协议，三类 token 的职责和信息方向仍然严格区分。

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

这不是语言模型中的词表分词，也不是向量量化，更不是自动发现 feature clusters。token boundary 由业务语义与人工设计决定。它决定哪些字段先在 token 内融合、哪些信息必须通过后续 Token Mixing 交换，也决定不同 Per-token FFN 分别负责什么子空间。因此，Feature grouping 不是无关紧要的预处理，而是模型结构的一部分。 论文使用“discrete latent tokens”描述结果，但公式中的 token 是连续向量。这里的“discrete”更适合理解为有限个、位置明确的 latent slots，而不是离散码或 vocabulary ID。

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

Global token 是每个样本都会使用的跨场景共享槽位，用于学习共性；specific token 只服务对应场景，用于保留差异。

### 5.3 Task token：样本相关的任务状态

Task token 采用相似构造：task-related features 与 important features 经过独立 Per-token FFN，得到每个任务的初始状态。[MDL §3.1.2](https://arxiv.org/html/2602.07520v2) 这意味着 Task token 不是固定的 Click/Favorite ID embedding，而是由当前样本特征生成的动态状态。同一个 Click task 面对不同用户、query 和 candidate，会得到不同的初始 Task token。

论文使用 LLM prompt 解释 Scenario/Task token 的作用，这个类比只成立一部分。P-Tuning v2 等 deep prompt tuning 方法通常在冻结的预训练语言模型中，把少量连续 prompt 放入多层；MDL 则与 Feature backbone 联合训练，而且 token 由当前样本的 raw features 与 prior 生成。因此，MDL 的 Domain token 更接近 **sample-conditioned latent query**，而不是经典 PEFT 中用于调用冻结知识的固定 prompt。[P-Tuning v2 原文](https://arxiv.org/abs/2110.07602)；[Prefix-Tuning 原文](https://arxiv.org/abs/2101.00190)

### 5.4 Initializer 决定 Domain token 携带什么

同一个 raw field 可以通过两套 embedding 同时进入不同路径。以 video ID 为例，一路进入 Item Feature token，另一路以额外 embedding 进入 Scenario/Task initializer；两套参数接收不同梯度，也以不同路径影响输出。因此，MDL 相比基线增加的不只是几个 attention queries，还包括：

- important features 的额外 embedding；
- scenario/task-specific prior；
- 每个 Domain token 的独立 FFN；
- 多层 residual state；
- Domain-to-Feature cross-attention。

这会直接影响消融实验的解释：删除整个 Task token，会同时删除额外信息、额外参数和交互路径，因此只能说明完整 Task-token path 有效，不能把全部收益单独归因于 layer-wise cross-attention。Feature token 组织共享证据，Scenario/Task token 则承载由样本与分布先验初始化的条件状态；它们形状相近，信息职责却不同。

<!-- IMAGE_PLACEHOLDER_START: FIG_03 -->
> **【插图占位｜图 3】三类 Token 的初始化来源与职责**
>
> **内容：** 说明 Feature token 由按业务语义组织的原始字段构成，Scenario/Task token 由身份信息、重要字段的独立 embedding 和特定先验初始化；同一 raw field 可以通过不同参数进入不同路径。重点区分共享证据、场景条件和任务条件三种职责，并说明 Domain token 是样本相关的动态状态，而非固定 prompt。
>
> **图注：** 三类 token 使用统一计算接口，但承载不同信息并由不同参数路径初始化。
<!-- IMAGE_PLACEHOLDER_END: FIG_03 -->

---

## 6. 一个 MDL Block：严格按执行顺序拆开

设第 $$l$$ 层输入为 $$F^{(l)},S^{(l)},T^{(l)}$$。每层按三步执行。

<!-- IMAGE_PLACEHOLDER_START: FIG_04 -->
> **【插图占位｜图 4】一个 MDL Block 的完整执行逻辑**
>
> **内容：** 严格表达一个 Block 内的先后关系与张量方向：Feature 先更新；Scenario 和 Task 分别作为 Query，以新的 Feature states 作为 Key/Value，attention 输出只更新 Domain states；Scenario residual state 一路经 Scenario FFN 传播到下一层，另一路选取 active Scenario 与 Global Scenario 并聚合；聚合结果进入所有 Task states，每个 Task 再经自己的 FFN 形成下一层状态。需要明确这不是 `Feature → Scenario → Task` 的串行读取，也不是三类 token 的双向 self-attention。
>
> **图注：** 每层先生成共享 Feature states，再由 Scenario/Task 完成条件化读取与融合。
<!-- IMAGE_PLACEHOLDER_END: FIG_04 -->

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

其中 $$d_h$$ 是单个 attention head 的 Key 维度。Q/K/V 的方向决定了模型语义：Scenario/Task 是 Query，Feature 是 Key/Value；Q/K/V projection 采用 Per-token FFN，使不同 Domain states 能够学习不同的读取映射。[MDL Eq. 7–8](https://arxiv.org/html/2602.07520v2) 加入 batch 与多头维度后，关键 shape 为：

| 张量 | 单头简化 shape | 含义 |
|---|---:|---|
| Feature state $$F$$ | $$[B,N_f,d]$$ | Key/Value 来源 |
| Scenario state $$S$$ | $$[B,N_s+1,d]$$ | Scenario Query |
| Task state $$T$$ | $$[B,N_t,d]$$ | Task Query |
| Scenario attention map | $$[B,N_s+1,N_f]$$ | 每个 Scenario 对 Feature 的读取权重 |
| Task attention map | $$[B,N_t,N_f]$$ | 每个 Task 对 Feature 的读取权重 |
| Scenario mask | $$[B,N_s]$$ | 选择 active Scenario；Global 始终参与 |

若有 $$H$$ 个 attention heads，权重张量相应变为 $$[B,H,N_d,N_f]$$，其中 $$N_d$$ 分别取 $$N_s+1$$ 或 $$N_t$$。这张 shape 表也是最直接的单元测试清单：Query 轴与 Feature 轴一旦互换，模型语义就已经不再是论文中的 MDL。

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

论文只定义了 Feature–Feature、Feature–Scenario/Task 和 Scenario–Task 三类 interaction，没有独立的 Task–Task interaction。因此，Task token 负责形成 task-aware representation，却不会自动表达 Click→Purchase 等任务之间的方向性依赖。场景差异已经逐层通过 selected Scenario state 进入 Task state，最终只需要 $$N_t$$ 个输出，而不是 $$N_sN_t$$ 个场景—任务 head。“All-Token Interaction”也并非所有 token 两两 self-attention，其信息方向如下：

| 被更新的状态 | 读取 Feature | 读取 Scenario | 读取 Task |
|---|---|---|---|
| Feature | 是：Feature self-interaction | 否 | 否 |
| Scenario | 是：Scenario=Q，Feature=K/V | 仅同一 token 的 residual 与 FFN；没有 Scenario–Scenario mixing | 否 |
| Task | 是：Task=Q，Feature=K/V | 是：只读 active Scenario 与 Global 的均值 | 仅同一 token 的 residual 与 FFN；没有 Task–Task mixing |

Global Scenario token 也不是由所有 specific Scenario tokens 聚合而来，而是一条独立路径；由于每个样本都会选择它，它能够接收所有场景样本的梯度。“global”来自更新覆盖范围，而不是显式 Scenario–Scenario aggregation。[MDL Eq. 9–15](https://arxiv.org/html/2602.07520v2) 至此，一个 Block 的依赖关系可以压缩为：**Feature 先更新，Scenario 与 Task 并行读取新的 Feature，active Scenario 与 Global 随后融入 Task，三类状态再进入下一层。**

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

这些数值只是教学构造，用来说明一种可检验的行为：Click 可能更关注 query relevance，Favorite 可能更关注 item content 与长期兴趣；Scenario states 虽然读取同一组 Feature states，学习的却是单列、双列等展示条件下不同的信息组合。当前样本属于 Single，因此融合阶段只选择 Single 与 Global：

$$
s_{avg}^{(1)}=\frac{\hat s_{single}^{(1)}+\hat s_{global}^{(1)}}{2}.
$$

这两个向量已经读取第一层 Feature states 并完成 residual，但尚未经过 Scenario FFN。同一份 $$\hat S^{(1)}$$ 一路经 Scenario FFN 形成下一层输入 $$S^{(1)}$$，另一路在当前层完成 Scenario–Task fusion，并分别加入 Click 与 Favorite。进入第二层时，Task Query 已包含 initializer、第一层样本证据和当前场景状态，因此会在更深的 Feature states 上重新执行条件化聚合，这就是 bottom-up、layer-wise interaction 的具体含义。若把同一用户—候选对切换到 Double 场景，即使普通 Feature inputs 保持不变，融合项也会改为 Double + Global；Click 和 Favorite 两个 Task heads 无需变化，场景差异通过 Task state 到达共享的任务输出映射。

### 7.1 参数如何共享：必须沿 loss 反向看

前向图说明信息怎样流动，梯度图才说明知识怎样共享。参数归属可以概括为：

| 状态或参数 | 主要更新来源 |
|---|---|
| Feature embeddings/backbone | 全部场景、全部任务 loss 共享更新 |
| Global Scenario path | 全部场景共享更新 |
| Specific Scenario path | 被选中场景的样本更新 |
| Task token / Task FFN | 对应任务 loss 更新，跨场景复用 |
| Task logit head | 对应任务 loss 更新 |

某场景样本只更新被选中的 specific Scenario token，Global token 则由所有场景共享更新。[MDL §3.3](https://arxiv.org/html/2602.07520v2) 这里需要区分表示冲突与梯度冲突：前者指不同任务需要不同信息，却只能使用同一份最终共享表示；后者指不同任务对共享参数产生方向不一致的更新，例如 $$g_i^\top g_j<0$$。MDL 通过跨层 Task states 和差异化 Feature 读取直接缓解表示冲突，但所有 Task loss 仍会沿 attention 的 K/V 与 Feature path 回传到共享 backbone，模型也没有显式的 gradient projection、normalization 或 detach。换言之，MDL 让不同任务在前向中读到不同表示，却没有隔离共享主干在反向中接收的多任务梯度。OneRank 同时设计 task-private forward channels 与 cross-task gradient detachment，也说明 forward specialization 与 backward isolation 是两个独立维度。[OneRank 原文](https://arxiv.org/abs/2606.16838)

<!-- IMAGE_PLACEHOLDER_START: FIG_05 -->
> **【插图占位｜图 5】前向专属化与反向共享更新**
>
> **内容：** 同时表达两条逻辑：前向中不同 Task/Scenario states 形成不同读取表示；反向中所有任务损失仍会经 attention 的 Key/Value 路径更新共享 Feature backbone。还要区分 Global Scenario 由全部场景更新、specific Scenario 由对应场景更新、Task private path 由对应任务损失更新，并指出 MDL 没有显式梯度投影或 detach。
>
> **图注：** MDL 缓解共享表示瓶颈，但前向专属化不等于反向梯度隔离。
<!-- IMAGE_PLACEHOLDER_END: FIG_05 -->

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

这是一种 **read-side conditioning**：Scenario/Task 参与逐层交互，却不直接改写同层 Feature path。上面的零 Jacobian 只刻画单个 Block 的前向依赖，并不表示三类状态统计独立；它们可以由相同 raw features 初始化，也会被同一组 loss 耦合，所有任务损失仍可沿 Key/Value 路径更新 Feature backbone。这样的选择使昂贵的 Feature computation 保持共享，每个 Domain 只增加较窄的跨层读取状态，但也放弃了 FiLM 或双向 self-attention 那种 write-side modulation。[FiLM](https://arxiv.org/abs/1709.07871)

### 8.2 MDL 对 Scenario × Task 空间做了因子化

若为每个组合维护独立 tower，需要 $$N_sN_t$$ 套场景—任务输出路径。MDL 的组合近似为：

$$
H_{s,t}=\phi_t
\left(T_t+\operatorname{Mean}(S_s,S_{global})\right),
$$

其中 $$\phi_t$$ 是 Task-specific nonlinear mapping。Task state、Scenario state 与 Task-specific FFN 共同表达联合差异，因此 Task head 可以跨场景复用，输出数量不必随场景数相乘；与此同时，模型也假设复杂的 $$(s,t)$$ pair-specific interaction 能由低维状态相加和后续 Task FFN 恢复。Mean + Add 进一步假设所有 active Scenario states 权重相同，并让同一份 Scenario summary 先以相同方式进入各 Task tokens，再由 Task FFN 产生差异。当一个样本属于多个场景时，Mean Pool 还会降低 Global token 与单个 specific token 的相对权重。因而，这种融合是低成本、低方差的 factorized baseline，但复杂的场景—任务关系可能仍需要 task-conditioned aggregation。

### 8.3 输出成本解耦，不等于中间成本消失

MDL 将输出 head 数量从潜在的 $$N_sN_t$$ 降到 $$N_t$$，但按论文公式直接实现时，$$N_s+1$$ 个 Scenario tokens 与 $$N_t$$ 个 Task tokens 都要先读取 $$N_f$$ 个 Feature tokens，随后才选择 active Scenario。cross-attention 的主要交互规模近似为：

$$
O\big((N_s+N_t)N_fd\big),
$$

不同 Domain token 的 Per-token FFN 参数和计算也大致随 $$N_s+N_t$$ 增长。工程上可以在 Domain-aware Attention 之前只 gather 当前 active Scenario 与 Global token，避免计算无关 Scenario states，但这属于稀疏执行优化，并不是论文已经描述和测量的实现；正式比较时应分别报告 dense token path 与 active-token path 的成本。更准确地说，**MDL 解耦了场景数与最终输出头数量，却没有完全解耦场景/任务数量与中间参数、FLOPs 和访存。** 当任务从 3 个扩展到几十个时，active-token execution、shared/private FFN、task clustering 或稀疏容量会成为新的系统问题。

### 8.4 固定 token slots 带来的是闭集组合，不是零样本扩展

MDL 初始化固定的 $$N_s+1$$ 个 Scenario slots 和 $$N_t$$ 个 Task slots；不同 slot 拥有 Per-token FFN，每个 Task slot 最终还连接独立 LogitsLayer。[MDL Eq. 4、16](https://arxiv.org/html/2602.07520v2) 场景与任务身份因此不仅存在于输入特征中，也被编码进参数槽位和输出索引。已知 Scenario × Task 的新组合不必再增加完整 tower，但组合已有槽位并不等于处理从未训练过的新场景或新任务：原论文没有说明如何根据描述或少量样本生成新的 Scenario token、将它映射到已有 Per-token FFN，或在不重训的情况下校准各 Task head；新增任务也仍需要新的 Task initializer、Task FFN、LogitsLayer 与监督信号。MDL 的扩展性更准确地说是**对训练时定义好的 Scenario/Task 闭集进行因子化组合**，并不直接覆盖 unseen scenario、隐式子场景发现或 zero-shot task generalization。若业务场景 ID 只是粗粒度枚举，还需要检查同一 slot 内是否混合了显著不同的流量来源、页面形态或人群分布。

---

## 9. 论文证据及其边界

MDL 的证据链包含工业离线主实验、组件消融、规模趋势、attention 可视化和一个月线上 A/B test，但不同实验支持的结论强度并不相同。

### 9.1 数据与对照

论文使用 Douyin Search 连续两个月日志，覆盖单列、双列、内搜三个场景、20 多个预测任务和 500 多个 user、item、sequence、cross features，并保留 1% 数据用于评估。[MDL §4.1.1](https://arxiv.org/html/2602.07520v2) 离线主要报告 Click、Like、Favorite 在三个场景上的 QAUC：先在每个 UID-query group 内计算 AUC，再对 group 平均。

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

九个场景—任务组合全部提升，是论文最强的离线证据；Like、Favorite 以及双列、内搜的收益通常更大，说明相对弱势分布可能更容易从逐层共享读取中获益。[MDL Table 1 与 §4.2](https://arxiv.org/html/2602.07520v2) 需要注意，表中数字是相对提升率，而非 QAUC 绝对百分点，论文也没有给出置信区间。

### 9.3 消融：Domain state 的证据强于具体算子的证据

论文在三个场景的 Click QAUC 上报告五项消融：

| 变体 | Single | Double | Inner |
|---|---:|---:|---:|
| w/o Task token | -0.12% | -0.11% | -0.09% |
| w/o Task–Feature interaction | -0.04% | -0.05% | +0.03% |
| w/o Scenario token | -0.17% | -0.16% | -0.15% |
| w/o Global Scenario token | -0.04% | -0.06% | -0.05% |
| w/o Scenario–Feature interaction | -0.05% | -0.04% | -0.05% |

删除全部 Task token 或 Scenario token 会在三个场景中造成稳定下降，Global Scenario token 的消融方向也一致，因此实验较有力地支持显式 Domain states 的价值。相比之下，替换 Task–Feature interaction 后，Inner Search 反而为 `+0.03%`，其他 interaction 变体的变化也较小，尚不足以证明某个 cross-attention 算子具有唯一性。更重要的是，`w/o Task token` 用 task tower 替代，`w/o Task–Feature interaction` 则改用 RankMixer interaction，并没有在相同 initializer 和 private capacity 下只改变“是否逐层读取”，所以 extra embedding、prior、Per-token FFN 与 layer-wise attention 的独立贡献仍未被拆开。[MDL Table 2 与 §4.3](https://arxiv.org/html/2602.07520v2)

### 9.4 Scaling：支持趋势，但还不是完整定律

论文改变 hidden dimension $$d$$ 与层数 $$L$$，比较 MDL 和 MMoE 随参数量、FLOPs 增加时的 Single-column Click QAUC gain。MDL 曲线持续高于 MMoE，而且差距随规模扩大。[MDL Figure 2](https://arxiv.org/html/2602.07520v2)

<!-- IMAGE_PLACEHOLDER_START: FIG_06 -->
> **【插图占位｜图 6】MDL 与 MMoE 的 Scaling 趋势（基于原文 Figure 2）**
>
> **内容：** 使用论文原始数据比较 hidden dimension 或层数增加、参数量和 FLOPs 上升时，MDL 与 MMoE 的 Single-column Click QAUC gain。需要表达两者均可能随规模增长，但 MDL 始终更高且差距扩大。不得补造实验点，并需限制结论为当前实验范围内的 scaling trend，而非普适 scaling law。
>
> **图注：** 当主干规模扩大，逐层 Domain states 比局部任务条件更能利用新增 Feature capacity。
<!-- IMAGE_PLACEHOLDER_END: FIG_06 -->

这一结果支持论文的核心动机：相比条件作用较局部的 MMoE，layer-wise Domain states 更可能利用新增 Feature capacity。不过，它更适合被称为 **scaling trend**，还不是完整的 scaling law，因为实验点数有限，只展示了一个场景的 Click，也没有幂律拟合、置信区间、数据量轴或 compute-optimal frontier。现有证据能够说明 MDL 相对 MMoE 的优势在该实验范围内随规模扩大，不能外推为普适定律。

### 9.5 Attention 图：行为证据，不是因果证明

论文展示不同 Task/Scenario tokens 在不同层对 Feature tokens 的平均 attention distribution。不同任务、场景和层形成了不同权重，说明这些 Domain states 没有完全坍缩为相同读取行为。[MDL Figure 3–4](https://arxiv.org/html/2602.07520v2)

Attention 图说明不同 Domain tokens 学出了不同读取分布，但不能单独证明这些差异就是性能提升的原因。输出还取决于 Value projection、向量范数、residual、Per-token FFN 和后续层。[Attention is not Explanation](https://aclanthology.org/N19-1357/)；[Attention is Not Only a Weight](https://aclanthology.org/2020.emnlp-main.574/)

### 9.6 线上结果：效果落地成立，系统成本仍未披露

论文在 Douyin Search 进行一个月 A/B test，线上 baseline 为 RankMixer + MMoE。LT30 指用户最近 30 天的平均活跃天数，越高越好；Change Query Rate 是发生手动 query reformulation 的 distinct UID-query 数占总 distinct UID-query 数的比例，论文把更低的数值解释为更高的搜索满意度。[MDL Eq. 18 与 §4.1.2](https://arxiv.org/html/2602.07520v2) Table 3 报告的是相对变化，而不是指标绝对值：

| 范围 | Change Query Rate | LT30 |
|---|---:|---:|
| ALL | -0.3267% | +0.0626% |
| Single-column Search | -0.2678% | +0.0520% |
| Double-column Search | -0.5079% | +0.0674% |
| Inner Search | -0.5492% | +0.0630% |

两个指标在三个场景中都朝论文定义的有利方向变化，作者也称模型已经全量部署并服务数亿用户。[MDL Table 3 与 §4.5](https://arxiv.org/html/2602.07520v2) 这说明离线收益能够转化为生产业务指标，是 MDL 很有价值的工业证据；但论文没有报告实验流量比例、置信区间、显著性检验、分桶样本量和 guardrail 指标，也没有披露相对 baseline 的 P50/P99 latency、QPS、MFU、峰值显存、通信量或 serving cost。因而，目前可以确认的是作者报告了方向一致并最终全量的线上收益，尚不能独立判断效应不确定性，也不能据此断言模型在严格相同的系统成本下更优。

---

## 10. 复现前需要回答的五个工程问题

论文已经给出了一套有效结构，但工程复现不能止于搭出框图；新增信息、新增参数、新增交互与新增成本必须分开测量，才能知道收益究竟来自哪里。

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

### 10.2 Domain initializer 是否形成预测捷径？

论文允许 important raw features、specific priors 和独立 embedding 进入 Domain token。若 Task initializer 已经直接包含 item ID、类目、价格、近期点击率或强转化统计，它在 cross-attention 之前就可能具备较强预测能力。问题不在于强 prior 本身，而在于需要区分两种机制：Task state 是主要表达“要读取什么”，还是已经携带大量“答案证据”。两者都可能提升 QAUC，但泛化风险、部署依赖和对 MDL 结构的解释完全不同，因此应先按计算职责审计字段：

| 角色 | 定义 | 示例 |
|---|---|---|
| Control | 表示任务或场景身份 | scenario ID、task ID |
| Query-only | 帮助决定读取什么，但单独预测力有限 | 展示布局、请求类型 |
| Evidence | 直接携带用户—候选匹配证据 | item ID、内容特征、行为序列 |
| Wide/Prior | 强统计先验 | 历史 CTR、CVR、频次统计 |

最小诊断包括 initializer-only、Feature-only、Feature masking/shuffle，以及输出对 $$T^{(0)}$$ 与 $$F^{(l)}$$ 的梯度敏感度。

### 10.3 Mean + Add 是否足以表达业务关系？

MDL 对所有 active Scenario states 等权平均，再把同一结果加入所有 Task states，计算与部署都很简单，也能降低小场景学习复杂 gate 的方差，但它要求 Task-specific FFN 吸收全部 Scenario–Task pair 差异。对于 Click、Like、Favorite 这类并行反馈，这个假设可能成立；对于 impression→click→purchase 这类严格漏斗，MDL 没有显式的 Task–Task probability decomposition 或 cascade information flow，“拥有 Task token”并不等于已经建模任务依赖。工程上应检查分场景 PCOC/ECE、logit distribution、标签窗口和采样机制；如果共享 Task head 的校准差异过大，可以先尝试 scenario-specific bias 或 task-conditioned scenario aggregation，而不是立即退回 $$N_sN_t$$ 套完整 towers。

### 10.4 前向专属化是否仍受梯度冲突限制？

高流量任务因样本数或 loss reduction 获得更大梯度，属于**权重不平衡**；两个任务在共享参数上的梯度方向相反，才属于**方向冲突**。二者可以同时出现，但前者应先核对 sampling、label mask、task weight 与归一化，后者才需要考虑梯度投影或隔离。建议在共享 backbone 的代表层持续监控各任务有效样本数、loss scale、梯度 norm、cosine similarity、负冲突率与训练速率；只有稳定的方向冲突确实与尾部任务回退相关时，才值得引入 GradNorm、PCGrad 或 OneRank 式 detach，否则逐任务梯度处理可能带来较高训练开销，却难以解释收益来源。[GradNorm](https://proceedings.mlr.press/v80/chen18a.html)；[PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)；[OneRank](https://arxiv.org/abs/2606.16838)

### 10.5 参数相同是否真的意味着生产成本相同？

MDL 增加了 embedding tables、QKV projections、Per-token FFN、residual states 和 batch masking。即使总参数同样控制在 0.5B，activated FLOPs、HBM bytes、kernel 数量和通信模式也可能不同，因此生产比较至少应同时报告：

| 类别 | 指标 |
|---|---|
| 模型质量 | 分场景/任务 QAUC、calibration、最差分布收益 |
| 计算 | Activated FLOPs、step time、MFU |
| 内存 | 峰值显存、HBM bytes/request、embedding cache miss |
| 在线 | QPS/GPU、P50/P95/P99 latency、timeout/fallback |
| 扩展 | 随 $$N_s,N_t,N_f,L,d$$ 增长的成本曲线 |

Parameter Count、FLOPs 与实际系统成本并不是同一个量，MDL 的公平评估也必须采用这一系统视角，才能区分收益来自信息、参数、交互、优化还是工程实现。[RankMixer 原文](https://arxiv.org/html/2507.15551v1)

> **复现顺序应保持单变量：先证明 condition 有用，再证明 token 载体有用，最后验证 layer-wise interaction。**

---

## 结语：MDL 真正改变的是条件信息的地位

MDL 不只是把 Feature、Scenario、Task 都表示为 token，而是重新安排了条件信息在计算图中的位置。Scenario 与 Task 由当前样本和分布先验初始化，不再只是静态 ID、gate input 或输出索引；它们作为跨层状态，在每个 Block 中重新读取 Feature states。与此同时，Feature backbone 仍保持共享，模型无需为每个任务重复昂贵的主干计算。active Scenario 与 Global Scenario 逐层融入 Task，使已知场景与任务能够通过 latent states 因子化组合，最终输出规模只随任务数增长；这种能力仍依赖训练时预先定义的固定 slots，并不自然延伸到 unseen scenario 或 zero-shot task。

九个场景—任务组合、组件消融、规模趋势和一个月线上 A/B test 为这套结构提供了较强证据。[MDL 原文](https://arxiv.org/html/2602.07520v2) 工程复现时仍需把 initializer 容量、强 prior shortcut、layer-wise reading、共享 backbone 的梯度冲突与真实系统成本分别测量。归根结底：

> **当共享 backbone 成为模型的主要容量载体时，条件信息不能只负责在末端选择答案，还需要拥有与主干深度匹配的信息读取路径。**

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
