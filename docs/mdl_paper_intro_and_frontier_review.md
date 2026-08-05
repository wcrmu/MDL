# MDL 论文介绍与前沿论文串讲

> 论文数字均对照原文核对（MDL：arXiv:2602.07520v2；OneRank：arXiv:2606.16838v1；MTFM：arXiv:2602.11235v1）。
> MDL 机制的深入解读另见 [`document.md`](../document.md)；复现工程问题见 [`mdl_reproduction_issues_solutions.md`](./mdl_reproduction_issues_solutions.md)。

---

## 一、MDL 模型介绍

论文链接：<https://arxiv.org/abs/2602.07520>　　论文团队：字节跳动

同一套搜索或推荐系统，往往既要服务多个场景，又要同时预测多种行为。SharedBottom、MMoE、STAR、PEPNet 等结构处理的是同一个矛盾：既要共享统计信息，又要保留场景和任务差异。当排序模型的大部分参数与计算逐渐进入深层主干，一个新的问题随之出现：**如果场景与任务信息仍只在末端的 gate、tower 或 head 中生效，扩大的主干是否真的学到了分布差异？**

MDL 的做法是把特征、场景和任务都表示为 token：Feature token 构成共享主干，Scenario/Task token 在每一层以 Query 的身份读取新的 Feature states，当前场景状态再融入各 Task state，最终由 Task token 输出预测。

> **MDL 的核心不是"增加几个 token"，而是让场景与任务从末端控制信号，变成跨层保留、逐层读取共享 Feature 状态的条件化表示。**

### 1.1 模型背景

#### 1.1.1 多场景、多目标任务

考虑两个搜索展示场景：单列搜索与双列搜索。模型同时预测两个任务：Click 与 Favorite。

|  | Click | Favorite |
|---|---|---|
| 单列 | \(D_{single,click}\) | \(D_{single,fav}\) |
| 双列 | \(D_{double,click}\) | \(D_{double,fav}\) |

这四个格子不是四个普通类别，而是四个相互关联却并不相同的预测分布：

- **场景改变样本的产生方式**：同屏候选数量、位置偏置、浏览节奏、曝光竞争和上下文特征都可能随之变化；
- **任务改变监督目标**：Click 通常更高频、更接近即时相关性，Favorite 更稀疏，也更依赖长期兴趣和内容价值。

对来自场景 \(s\) 的样本 \(x\)，任务 \(t\) 的预测可以写为：

\[
\hat y_{s,t}=p_\Theta(y_t=1\mid x,s).
\]

场景主要改变 \(p(x\mid s)\)，任务主要改变标签定义和 \(p(y_t\mid x,s)\)；前者对应输入分布，后者对应预测目标。两条极端路线都不理想：

- **每个组合训练独立模型**：可以保留充分差异，但特征工程、训练、部署和监控链路都要重复，稀疏任务也难以借用其他分布的统计信息；
- **完全共享主干、只在末端设置不同 head**：效率更高，但共享表示容易被高流量场景和高频任务主导，复杂差异只能留给浅层输出模块补救。

多场景多任务学习真正要决定的是：**共享什么，区分什么，以及在计算图的什么位置开始区分。**

#### 1.1.2 条件路径跟不上大主干

过去几类模型不断把 Task 或 Scenario 信息向计算图内部推进，但它们作用的范围并不相同：

| 模型               | 差异如何进入前向计算                                                                                                  | 条件未覆盖的计算                                                                                               |
| ---------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **SharedBottom** | 所有任务先经过同一个 Shared Bottom，再分别进入各自的 Task Tower                                                                | Shared Bottom 使用同一函数生成共享表示；任务差异只从各自的 Tower 分支开始体现                                                      |
| **MMoE**         | 每个任务配置独立 Gate；Gate 根据当前输入，对同一组 Shared Experts 的输出进行加权组合，再送入对应的 Task Tower                                   | 任务差异影响 Expert mixture 和后续 Tower，但不改变每个 Shared Expert 内部的前向计算 ([Google Research][1])                    |
| **STAR**         | Domain 决定专属的归一化统计与参数，并在 FCN 的每一层组合 Shared 与 Domain-specific 参数；辅助网络还直接读取 Domain ID                          | 场景条件覆盖 PN、整个 FCN 和输出旁路，但大规模 Embedding 层仍跨 Domain 共享；原文处理的是多 Domain 下的单一 CTR 任务 ([arXiv][2])            |
| **PEPNet**       | EPNet依据 Domain-side priors 门控共享 Embedding 的输出；PPNet依据 user/item/author priors，为各 Task Tower 的每一层生成分任务隐藏单元门控 | Embedding Table 本身仍然共享；条件通过门控缩放表示和 Tower hidden units，原文未定义独立且跨层递归更新的 Scenario/Task state ([arXiv][3]) |
| **RankMixer**    | 输入被组织为 Feature Tokens，经多层 Token Mixing 与 Per-token FFN 迭代，最终 mean pooling 后用于不同任务预测                         | 深层 RankMixer Blocks 迭代的是 Feature Tokens；原文未定义专门的 Scenario/Task token、条件 Gate 或逐层条件状态 ([arXiv][4])      |

RankMixer 通过规则、可堆叠的 Feature blocks，把大量 dense 参数和 FLOPs 放进深层 backbone。主干虽然变大了，但如果 Scenario/Task 仍只出现在末端 gate 或 tower，新增计算在前向中依然不感知场景和任务。问题由此从"共享多少参数"转向更关键的一点：**条件路径能否覆盖模型的主要容量。**

#### 1.1.3 MDL：共享 Feature，逐层读取

MDL 让 Scenario/Task 拥有与主干深度相匹配的状态路径：它们不为每个任务重复计算整套主干，而是在每一层重新判断当前场景或任务应从 Feature states 中聚合什么。为此，特征、场景、任务都被映射为固定数量、固定宽度的 latent token，并通过 attention、residual、Per-token FFN 等规则模块持续更新。这里的"统一"是计算接口统一，而不是三类 token 的语义和职责相同。

### 1.2 模型收益

#### 1.2.1 实验设置

**实验数据：**

- 离线：抖音搜索连续 2 个月日志，覆盖单列、双列、内搜三个场景、20 多个预测任务和 500 多个 user、item、sequence、cross features，涉及数十亿用户、数亿文档；保留 1% 数据用于评估，不参与训练。
- 在线：抖音搜索 A/B 实验，为期 1 个月。

**评估指标：**

- 离线实际训练 20 多个任务，论文只报告三个代表任务 Click、Like、Favorite 在三个场景上的 QAUC：先在每个 UID-query pair 内计算 \(\operatorname{AUC}_i\)，再等权平均：

\[
\operatorname{QAUC}=\frac{1}{N}\sum_{i=1}^{N}\operatorname{AUC}_i.
\]

- 在线主要报告 LT30 和 Change Query Rate。LT30 是用户最近 30 天的平均活跃天数，越高越好；Change Query Rate 是发生手动 query 改写的 distinct UID-query 数占全部 distinct UID-query 数的比例，论文把更低的改写率解释为更高的搜索满意度。

**基线方法：** RankMixer、SharedBottom、MMoE、STAR、HMoE、PEPNet。除纯 RankMixer 外，其他多分布结构都接入 RankMixer backbone，总参数量控制在约 0.5B——保证了近似等参数比较，但 activated FLOPs、访存和线上时延并不因此完全一致。

#### 1.2.2 离线实验

论文以每个场景—任务格子的最强 baseline 为参照，报告 MDL 的相对 QAUC 提升：

| 场景 | Click | Like | Favorite |
|---|---:|---:|---:|
| 单列 | +0.31% | +0.23% | +0.42% |
| 双列 | +0.27% | +0.63% | +0.77% |
| 内搜 | +0.25% | +0.34% | +0.71% |

九个场景—任务组合全部提升，是论文最强的离线证据。Like、Favorite 以及双列、内搜的收益通常更大，说明相对弱势的分布可能更容易从逐层共享读取中获益。注意表中数字是**相对提升率**，不是 QAUC 绝对百分点，论文也没有给出置信区间。

#### 1.2.3 在线 A/B 实验

在抖音搜索进行一个月 A/B 实验，线上基线为 RankMixer + MMoE。表中为相对变化：

| 范围 | Change Query Rate | LT30 |
|---|---:|---:|
| 全部 | -0.3267% | +0.0626% |
| 单列 | -0.2678% | +0.0520% |
| 双列 | -0.5079% | +0.0674% |
| 内搜 | -0.5492% | +0.0630% |

两个指标在三个场景中都朝有利方向变化，作者称模型已全量部署并服务数亿用户。这说明离线收益能够转化为生产业务指标；但论文没有报告实验流量比例、置信区间、显著性检验和 guardrail 指标，也没有披露相对 baseline 的延迟、QPS 或 serving cost。

#### 1.2.4 消融实验

论文在三个场景的 Click QAUC 上报告五项消融：

| 变体 | 单列 | 双列 | 内搜 |
|---|---:|---:|---:|
| w/o Task token | -0.12% | -0.11% | -0.09% |
| w/o Task–Feature interaction | -0.04% | -0.05% | +0.03% |
| w/o Scenario token | -0.17% | -0.16% | -0.15% |
| w/o Global Scenario token | -0.04% | -0.06% | -0.05% |
| w/o Scenario–Feature interaction | -0.05% | -0.04% | -0.05% |

删除全部 Task token 或 Scenario token 在三个场景中造成稳定下降，Global Scenario token 的消融方向也一致，因此实验较有力地支持**显式 Domain states** 的价值。但两点需要保留：其一，替换 Task–Feature interaction 后内搜反而 +0.03%，说明某个具体 cross-attention 算子的唯一性证据不足；其二，`w/o Task token` 是用 task tower 整体替代，extra embedding、prior、Per-token FFN 与逐层读取的独立贡献并未拆开。

#### 1.2.5 Scaling

论文改变 token 维度 \(d\) 与层数 \(L\)，比较 MDL 和 MMoE 随参数量、FLOPs 增加时的单列 Click QAUC gain：MDL 曲线持续高于 MMoE，且差距随规模扩大。这支持论文的核心动机——相比条件作用较局部的 MMoE，逐层 Domain states 更能利用新增 Feature capacity。不过实验点有限、只展示单场景单任务，也没有幂律拟合，更适合称为 **scaling trend**，而不是完整的 scaling law。

<!-- IMAGE_PLACEHOLDER_START: FIG_01 -->
> **【插图占位｜图 1】MDL 与 MMoE 的 Scaling 趋势（基于原文 Figure 2）**
>
> **内容：** 使用论文原始数据比较 hidden dimension 或层数增加时，MDL 与 MMoE 的单列 Click QAUC gain；两者均随规模增长，但 MDL 始终更高且差距扩大。不得补造实验点。
>
> **图注：** 当主干规模扩大，逐层 Domain states 比局部任务条件更能利用新增 Feature capacity。
<!-- IMAGE_PLACEHOLDER_END: FIG_01 -->

### 1.3 模型架构

#### 1.3.1 总体架构

三类 token 在交互前通过 projection 对齐到相同的 attention 宽度。Feature token 先在共享主干中完成高阶交互；随后 Scenario 与 Task 分别作为 Query，以 Feature states 为 Key/Value，形成各自的读取权重和聚合结果；最后，模型选出当前样本所在的 Scenario token，与 Global Scenario token 聚合后写入所有 Task states，只有 Task token 连接输出 head。论文将这套机制称为 **Domain-aware All-Token Interaction**，但它并不是把三类 token 拼接后做普通 self-attention："统一"体现在 latent 宽度和重复交互协议，三类 token 的职责和信息方向仍然严格区分。

<!-- IMAGE_PLACEHOLDER_START: FIG_02 -->
> **【插图占位｜图 2】MDL 整体模型架构**
>
> **内容：** 完整说明 Feature、Scenario、Task 三类 token 的初始化与跨层传播：Feature 先完成共享交互；Scenario/Task 分别以 Query 读取 Feature；active Scenario 与 Global Scenario 聚合后进入所有 Task states；最终每个 Task state 连接对应输出层。需体现不存在独立的 Task–Task 或 Scenario–Scenario mixing，输出数为 \(N_t\) 而非 \(N_sN_t\)。
>
> **图注：** MDL 共享 Feature backbone，并用逐层演化的 Scenario/Task states 完成多分布条件化。
<!-- IMAGE_PLACEHOLDER_END: FIG_02 -->

#### 1.3.2 Tokenization：三类 token 从哪里来

MDL 的完整状态可以写为：

\[
F^{(l)}\in\mathbb{R}^{N_f\times d_f},\qquad
S^{(l)}\in\mathbb{R}^{(N_s+1)\times d_s},\qquad
T^{(l)}\in\mathbb{R}^{N_t\times d_t}.
\]

其中 \(N_f\) 是 Feature token 数量，\(N_s\) 是业务场景数，额外的 1 对应 Global Scenario token，\(N_t\) 是预测任务数，\(l\) 为层号。

**Feature token。** 工业搜索输入通常包括 user、item、query、context、统计交叉和多个行为序列。离散字段先经过 embedding table，序列字段通过 DIN、LONGER 等序列模块编码，最终得到宽度不一的表示。MDL 沿用 RankMixer 的 semantic tokenization：人工按业务语义把字段分成 \(N_f\) 组，每组拼接后投影到固定维度：

\[
e_{input}=[e_1;e_2;\ldots;e_{N_f}],
\qquad
t_j=\operatorname{Proj}(e_j),
\qquad
F^{(0)}=[t_1;t_2;\ldots;t_{N_f}].
\]

token 边界由业务语义与人工设计决定：它决定哪些字段先在 token 内融合、哪些信息必须通过后续 Token Mixing 交换。因此 Feature grouping 不是无关紧要的预处理，而是模型结构的一部分。

**Scenario token。** 每个 Scenario token 的输入由两部分组成：

1. 若干 important raw features 的额外 embedding（例如 user ID、video ID）——与 Feature 路径使用相同 raw field，但单独开一张 embedding 表；
2. scenario-specific prior，例如该场景下的用户行为序列。

随后每个 Scenario token 经过自己的 Per-token FFN：

\[
s_k^{(0)}=\operatorname{ReLU}
\left(\operatorname{FFN}_k^s
(\hat e_{imp}\oplus\hat e_{spec}^{k})\right).
\]

若有 \(N_s\) 个业务场景，模型构造 \(N_s\) 个 specific Scenario token，并额外构造一个 Global Scenario token：

\[
S^{(0)}=[s_1^{(0)};\ldots;s_{N_s}^{(0)};s_{global}^{(0)}].
\]

Global token 是每个样本都会使用的跨场景共享槽位，用于学习共性；specific token 只服务对应场景，用于保留差异。

**Task token。** 采用相似构造：important features 与 task-related features 经过独立 Per-token FFN，得到每个任务的初始状态。这意味着 Task token 不是固定的 Click/Favorite ID embedding，而是由当前样本特征生成的动态状态——同一个 Click task 面对不同用户、query 和 candidate，会得到不同的初始 Task token。

值得注意的是，同一个 raw field 可以通过两套 embedding 同时进入不同路径。以 video ID 为例：一路进入 Item Feature token，另一路以额外 embedding 进入 Scenario/Task initializer；两套参数接收不同梯度，也以不同路径影响输出。因此 MDL 相比基线增加的不只是几个 attention query，还包括 important features 的额外 embedding、scenario/task-specific prior、每个 Domain token 的独立 FFN、多层 residual state 和 Domain-to-Feature cross-attention——这会直接影响消融实验的解释。

#### 1.3.3 单层 MDL Block

设第 \(l\) 层输入为 \(F^{(l)},S^{(l)},T^{(l)}\)。每层按三步执行：**先生成共享 Feature states，再由 Scenario/Task 完成条件化读取与融合。**

**第一步：Feature token 先完成共享特征交互。**

\[
F^{(l+1)}=B_l(F^{(l)}).
\]

默认实现采用 RankMixer：Token Mixing 负责跨 Feature token 交换信息，Per-token FFN 负责每个语义子空间的非线性变换：

\[
F^{(l+1)}=\operatorname{PerTokenFFN}
\left(\operatorname{LN}
(\operatorname{TokenMixing}(F^{(l)})+F^{(l)})\right).
\]

这部分可以替换为其他特征交互方法；MDL 只要求每层输出一组新的 Feature states。

**第二步：Scenario 与 Task 以 Query 读取 Feature。** 对任意一组 Domain token \(D\in\{S,T\}\)，Domain-aware Attention 可写成标准 cross-attention 形式——Domain token 是 Query，Feature token 是 Key/Value；Q/K/V projection 采用 Per-token FFN，使不同 Domain state 能学习不同的读取映射：

\[
Q=D^{(l)}W_Q,\qquad
K=F^{(l+1)}W_K,\qquad
V=F^{(l+1)}W_V,
\]

\[
\operatorname{Attn}(D,F)=
\operatorname{softmax}
\left(\frac{QK^\top}{\sqrt {d_h}}\right)V.
\]

不同任务面对同一组 Feature states，可以产生不同的 attention 分布。完成 attention 后，Scenario path 加 residual，再通过 Per-scenario FFN：

\[
\hat S^{(l+1)}=\operatorname{DomainAwareAttn}
(S^{(l)},F^{(l+1)})+S^{(l)},
\]

\[
S^{(l+1)}=\operatorname{PerTokenFFN}_s
(\hat S^{(l+1)})+\hat S^{(l+1)}.
\]

Task path 采用相同读取方向：

\[
\hat T^{(l+1)}=\operatorname{DomainAwareAttn}
(T^{(l)},F^{(l+1)})+T^{(l)}.
\]

**第三步：把当前 Scenario state 融入所有 Task states。** Scenario 与 Task 不会一直平行传播。对每个样本，模型根据场景信息选出对应的 specific Scenario token，同时保留 Global token；若样本属于重叠场景，也可选出多个 specific token。被选中的是 attention 与 residual 之后、Scenario FFN 之前的 \(\hat S^{(l+1)}\)，论文使用简单 Mean Pool：

\[
s_{avg}=\operatorname{MeanPool}
(\{\hat s_{active,1},\ldots,\hat s_{active,m},\hat s_{global}\}).
\]

随后把同一个 \(s_{avg}\) 加入所有 Task token：

\[
\tilde T^{(l+1)}=\hat T^{(l+1)}+s_{avg}.
\]

每个 Task token 再经过自己的 FFN 与 residual：

\[
T^{(l+1)}=\operatorname{PerTokenFFN}_t
(\tilde T^{(l+1)})+\tilde T^{(l+1)}.
\]

堆叠 \(L\) 层后，第 \(t\) 个 Task token 连接第 \(t\) 个 logits layer 输出预测：

\[
\hat y_t=\operatorname{LogitsLayer}_t(T_t^{(L)}).
\]

论文只定义了 Feature–Feature、Feature–Scenario/Task 和 Scenario–Task 三类 interaction，没有独立的 Task–Task interaction。因此 Task token 负责形成 task-aware representation，却不会自动表达 Click→Cart 这类任务之间的方向性依赖。三类 interaction 的信息方向如下：

| 被更新的状态 | 读取 Feature | 读取 Scenario | 读取 Task |
|---|---|---|---|
| Feature | 是：Feature self-interaction | 否 | 否 |
| Scenario | 是：Scenario=Q，Feature=K/V | 仅同一 token 的 residual 与 FFN；没有 Scenario–Scenario mixing | 否 |
| Task | 是：Task=Q，Feature=K/V | 是：只读 active Scenario 与 Global 的均值 | 仅同一 token 的 residual 与 FFN；没有 Task–Task mixing |

Global Scenario token 也不是由所有 specific token 聚合而来，而是一条独立路径；由于每个样本都会选择它，它能接收所有场景样本的梯度——"global"来自更新覆盖范围，而不是显式的 Scenario–Scenario aggregation。

---

## 二、复现工作及工程部署问题

我们基于原始论文，在 RankMixer 和 OneTrans 两种主干上分别实现了 Scenario/Task token 机制：

| 模型 | 主干 | 说明 |
|---|---|---|
| `mdl_rankmixer` | RankMixer | 论文默认机制的复现：32 个 groupwise Feature token（23 个语义组 + 9 个历史组），Domain 逐层读取 Feature |
| `mdl_onetrans` | OneTrans | MDL 机制的 OneTrans 适配：Domain sidecar 每层读 32 个 NS token，仅最后两层读 pyramid 压缩后的 S |

复现过程中遇到的 token 构成、状态语义、场景接入与显存/IO/RSS 工程问题，见 [`mdl_reproduction_issues_solutions.md`](./mdl_reproduction_issues_solutions.md)（详版见 [`mdl_key_questions.md`](./mdl_key_questions.md)、[`mdl_onetrans_adaptation_hardships.md`](./mdl_onetrans_adaptation_hardships.md)）。

---

## 三、前沿论文串讲

MDL 把场景和任务从末端 gate / tower，提升为跨层保留、逐层读取 Feature 的条件状态。但它隐含了一个前提：不同场景、不同任务**共用同一套 raw feature schema**，需要决定的只是读取方式该不该分化。顺着这条线往下，还有两篇工作各进一步：

| 工作 | 问题 |
|---|---|
| OneRank | 就算 Task 已经能分化读取，Transformer 编码完之后，为什么还要切换到一套外挂的静态 MLP 预测器？ |
| MTFM | 如果不同场景连 schema 都不一样（有的字段别的场景根本没有），"共用一套输入"这个前提还成立吗？ |

概括地说：**MDL 解决条件信息如何深入主干；OneRank 解决编码完之后还要不要切出去做预测；MTFM 解决 schema 都对不齐时怎么还共享一个主干。**

### 3.1 OneRank：把多任务预测放进 Transformer 内部

论文链接：<https://arxiv.org/abs/2606.16838>　　论文团队：Shopee

#### 3.1.1 模型背景

多数工业排序模型仍是"共享编码器 + 多任务预测器"：\(\mathbf{Z}=\mathcal{F}(\mathbf{X})\)，\(\hat y=\mathcal{G}(\mathbf{Z})\)。这些年只是把 \(\mathcal{F}\) 换成了更强的编码器，\(\mathcal{G}\)（MMoE / PLE / 多塔）仍是外挂的静态 MLP。OneRank 指出，encoder–predictor 分离这个结构本身有三个问题：

| 问题 | 具体表现 |
|---|---|
| 信息瓶颈 | \(\mathbf{Z}\) 是所有任务共享的融合表示，任务差异只能留到预测端才去解耦 |
| 跷跷板现象 | 多任务梯度共同作用于共享参数 \(\mathcal{F}\)，容易互相拖累 |
| Dataflow 断层 | Transformer 内部是 attention 式、随上下文变化的动态路由；外挂 \(\mathcal{G}\) 对动态用户上下文的适应能力有限 |

和 MDL 的背景差异在于：MDL 的问题是 Scenario/Task 信息只在末端生效，条件路径跟不上主干深度；OneRank 的问题更彻底——它认为只要还存在"编码器算完、交给外部预测器"这个动作，无论编码器多深，预测那一刻都会发生一次 dataflow 断层。OneRank 要做的不是把 Task 状态挪进 Transformer 早期（MDL 已经做到这一点），而是让 Transformer 从输入到打分全程不出模型。

#### 3.1.2 模型收益

**离线**（Shopee 2025 年 12 月连续 30 天日志）：

| #User | #Item | #Query | #Impression | #Click | #Add-to-Cart | #Order |
|---:|---:|---:|---:|---:|---:|---:|
| 33M | 118M | 105M | 26.6B | 1.05B | 251M | 40M |

主结果（完整对比覆盖 DNN / MTGR / OneTrans 三种 encoder × NSE / MMoE / PLE / DCMT / ResFlow 五种 predictor，此处列最强基线组合）：

| Encoder + Predictor | Params | FLOPs | C-AUC | C-GAUC | A-AUC | A-GAUC | O-AUC | O-GAUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OneTrans + PLE（最强基线） | 6.4M | 823.5M | 0.7770 | 0.7775 | 0.8371 | 0.7982 | 0.8996 | 0.8336 |
| **OneRank** | **4.9M** | **1.0G** | **0.7910** | **0.7843** | **0.8463** | **0.8036** | **0.9024** | **0.8350** |

参数更少但 FLOPs 略高——预测端算力从外部 MLP 挪回了 Transformer 内部；三个任务的 AUC 与 GAUC 六列全部超过所有 encoder × predictor 组合。

消融要点：去掉任务专属 token，A-AUC 从 0.8463 掉到 0.8424；把 situational descriptor 换成随机参数伤害最大，C-AUC 掉到 0.7872（O-GAUC 掉到 0.8318）；去掉梯度隔离后指标接近（A-AUC 0.8460 vs 0.8463），但在 add-to-cart 任务上出现训练不稳定。

**线上**（Shopee 主排序场景，2026-01-08 至 01-14，实验/对照各 10% 流量）：

| GMV/UU | Paid GMV/UU | 广告收入/UU（AR/UU） | Bad Query Rate |
|---:|---:|---:|---:|
| +1.01% | +1.17% | +0.81% | -2.29% |

#### 3.1.3 模型架构

OneRank 分四步，把"任务分化 → 候选竞争 → 任务关系 → 打分"全部留在 Transformer 内部：

| 步骤 | 做法 |
|---|---|
| ① 结构化输入 + 任务专属 token | 每个候选 \(c_i\) 组成一个 token 组：候选 embedding + \(K\) 个共享参数的任务 token（Click / ATC / Order）。注意力掩码让不同候选组互不可见；同组内不同任务 token 也互不可见（mutual invisibility），只能看用户上下文（interaction history + preference anchors，因果掩码）和本组候选 |
| ② 候选感知的请求级聚合 | 用 Situational Descriptor（用户画像 / query / session 元信息）经任务专属投影得到 Query，对全体候选的任务表示 \(\{r_k^i\}\) 做 task-specific cross-attention，聚合出该任务在整个候选集合上的全局表示 \(h_k\)——弥补 point-wise 训练与请求级排序之间的 gap |
| ③ 受控的 Task–Task 交互 | 在 \(K\) 个任务全局表示 \(\{h_k\}\) 之间做 mask 可配置的 self-attention：Parallel（互不可见）/ Null（全互见）/ Cascade（Click→ATC→Order 单向）/ Hybrid。前向允许信息流动，反向用 gradient detachment 只保留对角线梯度，把跨任务 attention 变成 read-only 的知识迁移 |
| ④ 动态匹配打分 | 抛弃静态 MLP 头：任务全局向量经 FFN + residual 得到 \(z_k\)，直接与候选任务表示做内积 \(s_k^i=\mathbf{z}_k^\top\mathbf{r}_k^i\)，打分随会话上下文动态变化 |

训练目标是 list-wise InfoNCE 与 point-wise 损失的混合，与动态匹配打分保持一致。

**和 MDL 架构的差异：**

|  | MDL | OneRank |
|---|---|---|
| Task 状态挂在哪 | 样本级、与候选无关的独立 Task token，逐层读 Feature | 挂在每个候选上（\(K\) 个 task token 随候选复制），天然自带候选身份 |
| Task 之间是否交互 | 无独立 mixing；关系只经共享 Feature 间接形成 | 显式 Task–Task attention，mask 可配置（parallel / cascade / hybrid） |
| 梯度处理 | 各任务损失仍经 attention 的 K/V 更新共享 Feature，无跨任务梯度隔离 | Cross-task attention 做 gradient detachment：前向共享、反向对角 |
| 候选竞争建模 | 无；本质 point-wise，不显式建模同请求候选间的竞争 | 有；situational descriptor 把整批候选聚合进任务表示，专门弥补训练-serving gap |
| 打分方式 | 每个 Task token 连接固定 logits head | 任务全局向量与候选表示做内积，动态匹配 |
| Scenario 轴 | 有独立 Scenario/Global token 与实例级选择 | 未涉及，论文聚焦多任务而非多场景 |

MDL 让 Scenario 与 Task 逐层读取共享 Feature capacity；OneRank 在此基础上继续追问：任务拿到专属通道之后，候选之间的竞争、任务之间的知识流动、梯度边界、打分方式，能不能也留在 Transformer 内部一次性解决。

### 3.2 MTFM：当不同场景的输入空间也无法对齐

论文链接：<https://arxiv.org/abs/2602.11235>　　论文团队：美团

#### 3.2.1 模型背景

美团把推荐 Foundation Model 需要具备的三个属性定义为 **Scalability**（模型和数据越大，效果稳定变好）、**Extensibility**（低成本接入新场景）、**Efficiency**（多场景数据量暴增时算力可控）。而现有多场景方法几乎都遵循 "harmonize-then-decompose" 范式：先把各场景数据整理成统一模板，再把参数拆成 domain-invariant / domain-specific（典型如 STAR、M3oE、MLoRA）。MTFM 指出这个范式在三点上撑不住：

| 局限 | 具体表现 |
|---|---|
| Extensibility 差 | 餐厅推荐和菜品推荐的 feature schema 天生不同（不同供给类型、不同 UI），字段强行对齐必然丢信息，或者靠 padding 硬凑 |
| 结构不可扩展 | 现有结构大多由专家针对固定几个场景手工设计，换个场景就要重新设计，无法享受 scaling law |
| 算力不可持续 | 多场景数据暴力拼接后，训练成本随数据量线性增长，工业上不可持续 |

和 MDL 的背景差异在于：MDL 默认所有场景共用同一套 raw feature（Feature token 分组对所有场景一致），分歧只在场景/任务该怎么读取这份共享表示；MTFM 面对的问题更前置——不同场景的原始字段本身就对不上，schema 层面的统一根本无法达成，需要先解决"怎么进模型"，再谈"怎么共享"。

#### 3.2.2 模型收益

**离线数据**（美团外卖推荐三个场景；用户与商品数做了匿名化处理）：

| 场景 | 业务 | #Exposure | #Click | #Purchase |
|---|---|---:|---:|---:|
| HP | 首页餐厅推荐 | 18.53B | 1.08B | 176.77M |
| PHF | 拼好饭菜品推荐 | 15.29B | 359.14M | 104.73M |
| SQS | 神枪手券包推荐 | 2.24B | 85.34M | 9.92M |

**离线总结**：MTFM 相对最强基线的 GAUC 提升（pp 为百分点）——

| 任务 | 平均 GAUC 提升 | 最大 GAUC 提升 |
|---|---:|---:|
| CTR | +0.36pp | +0.76pp |
| CTCVR | +0.29pp | +0.53pp |

完整主结果如下（基线分三类：通用排序 DCNv2 / MMoE / RankMixer，生成式排序 OneTrans / MTGR，多场景方法 STAR / PEPNet；其中通用与生成式方法用单场景数据训练，多场景方法与 MTFM 用多场景数据训练）。

HP / PHF 两个场景：

| 类别 | 方法 | HP CTR AUC | HP CTR GAUC | HP CTCVR AUC | HP CTCVR GAUC | PHF CTR AUC | PHF CTR GAUC | PHF CTCVR AUC | PHF CTCVR GAUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 通用 | DCNv2 | 0.7664 | 0.6853 | 0.8780 | 0.6451 | 0.7683 | 0.7236 | 0.8586 | 0.7555 |
| 通用 | MMoE | 0.7664 | 0.6857 | 0.8782 | 0.6454 | 0.7718 | 0.7282 | 0.8640 | 0.7597 |
| 通用 | RankMixer | 0.7665 | 0.6860 | 0.8789 | 0.6464 | 0.7711 | 0.7270 | 0.8628 | 0.7590 |
| 生成式 | OneTrans | 0.7672 | 0.6944 | 0.8774 | 0.6497 | 0.7832 | 0.7373 | 0.8827 | 0.7735 |
| 生成式 | MTGR | 0.7679 | 0.6951 | 0.8776 | 0.6491 | 0.7883 | 0.7398 | 0.8879 | 0.7771 |
| 多场景 | STAR | 0.7669 | 0.6882 | 0.8780 | 0.6482 | 0.7821 | 0.7298 | 0.8688 | 0.7660 |
| 多场景 | PEPNet | 0.7672 | 0.6895 | 0.8790 | 0.6489 | 0.7866 | 0.7328 | 0.8721 | 0.7693 |
| **本文** | **MTFM** | **0.7689** | **0.6954** | **0.8806** | **0.6507** | **0.7940** | **0.7474** | **0.8892** | **0.7824** |

SQS 场景（额外报告券包核销指标 IMD = 30 分钟内核销、WRITE = 24 小时内核销）：

| 类别 | 方法 | CTR AUC | CTR GAUC | CTCVR AUC | CTCVR GAUC | IMD AUC | IMD GAUC | WRITE AUC | WRITE GAUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 通用 | DCNv2 | 0.8290 | 0.7789 | 0.9057 | 0.8227 | 0.9074 | 0.8254 | 0.9072 | 0.8222 |
| 通用 | MMoE | 0.8449 | 0.7842 | 0.9073 | 0.8243 | 0.9082 | 0.8252 | 0.9056 | 0.8230 |
| 通用 | RankMixer | 0.8472 | 0.7881 | 0.9094 | 0.8281 | 0.9105 | 0.8268 | 0.9080 | 0.8279 |
| 生成式 | OneTrans | 0.8454 | 0.7994 | 0.9091 | 0.8271 | 0.9089 | 0.8279 | 0.9055 | 0.8248 |
| 生成式 | MTGR | 0.8454 | 0.7997 | 0.9097 | 0.8282 | 0.9095 | 0.8291 | 0.9059 | 0.8258 |
| 多场景 | STAR | 0.8470 | 0.7863 | 0.9079 | 0.8245 | 0.9081 | 0.8257 | 0.9055 | 0.8233 |
| 多场景 | PEPNet | 0.8511 | 0.7892 | 0.9081 | 0.8251 | 0.9089 | 0.8268 | 0.9066 | 0.8239 |
| **本文** | **MTFM** | **0.8624** | **0.8027** | **0.9119** | **0.8301** | **0.9117** | **0.8319** | 0.9079 | **0.8288** |

两点值得注意：其一，论文口径是 **nearly all** 指标最优——唯一的例外是 SQS WRITE AUC，MTFM 以 0.9079 对 0.9080 的毫厘之差略低于 RankMixer；其二，STAR / PEPNet 存在明显 see-saw（PEPNet 在 HP / PHF 领先 RankMixer，却在 SQS 上回落），而 MTFM 没有这个问题。离线超参：\(d_{model}=768\)，\(B=4\) 个 block，每块 \(K=3\) 层 Target Attention，GQA 为 \(H=3\)、\(G=1\)。

**效率**（Hybrid Target Attention，7 天样本、单卡 A100 实测；配置记法 \((K:P)\times B\) = 每块 \(K\) 层 Target、\(P\) 层 Full，共 \(B\) 块）：

| 配置 (Target:Full)×B | Batch | AUC | GAUC | 吞吐 (samples/s) | 显存 (GB) |
|---|---:|---:|---:|---:|---:|
| 纯 Full (0:16)×1 | 1× | 0.7514 | 0.6818 | 390 | 66.97 |
| 纯 Target (16:0)×1 | 1× | 0.7497 | 0.6806 | 575 | 32.64 |
| (1:1)×8 | 1× | 0.7508 | 0.6820 | 497 | 38.00 |
| (3:1)×4 | 1× | 0.7506 | 0.6821 | 547 | 34.08 |
| (5:1)×3 | 1× | 0.7504 | 0.6811 | 535 | 37.52 |
| (3:1)×4（去 GQA） | 2× | 0.7506 | 0.6822 | 660 | 70.16 |
| **(3:1)×4（MTFM）** | **2×** | **0.7506** | **0.6822** | **780** | **67.49** |

读法：同 batch 下 (3:1)×4 把显存从 66.97 GB 降到 34.08 GB、吞吐提升 40%，质量不降；省下的显存换成 2× batch 后吞吐达 780 samples/s，恰好是纯 Full 的 2 倍，显存持平。纯 Target（退化为 lazy decoder）质量开始明显下滑（-0.12pp），说明 Full Attention 层不可省；GQA 在 2× batch 下再贡献 660 → 780 的吞吐。

**系统协同优化：**

| 环节 | 手段 | 效果 |
|---|---|---|
| 训练 | CPU-GPU pipeline：消除同步点、合并 D2D 拷贝 | +20% 吞吐 |
| 训练 | 自定义 kernel：改进 FlashAttention-2（动态 mask 连续对齐布局 + shared memory 复用），Triton 融合 GLN 与动态 mask 构建 | 再 +57% 吞吐 |
| 推理 | HSTU 的 UVQK / Output 投影做 2:4 结构化稀疏（Sparse Tensor Core） | +10% 吞吐，-0.2ms 延迟 |
| 推理 | 细粒度计算跳过（padding 剪枝 + 跳过无效注意力） | 约 +5% 吞吐 |
| 推理 | 场景感知子图部署 + BF16 + M-Falcon | 消除无关场景算力 |

**可扩展性**：模型规模从 10× 扩到 70× 时，三个场景的 CTCVR GAUC 都呈现稳定的 scaling law 斜率（对照 MMoE）；同一模型规模下训练 token 越多效果越好，且大小模型之间的差距随数据量增加而拉大。

**线上**（两个场景 A/B，实验流量为千万级日曝光，基线为线上持续优化多年的 SOTA 模型）：

| 场景 | CTR | UV_CTCVR | 订单量 | 延迟 |
|---|---:|---:|---:|---:|
| SQS 券包推荐 | +1.89% | +2.46% | +2.98% | -5ms |
| PHF 菜品推荐 | +1.53% | +1.03% | +1.45% | -6ms |

论文指出，这一订单提升相当于该业务领域通常 2–3 轮模型迭代的累计收益。

#### 3.2.3 模型架构

| 环节 | 做法 |
|---|---|
| Heterogeneous tokenization | 三类 token：H-token（历史行为，跨场景共享，按时间排序）、R-token（实时行为，跨场景共享）、T-token（当前场景曝光候选，场景专属，由 \(\mathrm{MLP}_s(\mathrm{Emb}(U^s) \,\|\, \mathrm{Emb}(C_i^s) \,\|\, \mathrm{Emb}(I_i^s))\) 生成）。各类各场景用各自的 tokenizer，只需投影到统一的 \(d_{model}\)，拼成一条变长序列 \(\mathbf{X}^{(0)}=(\mathbf{H};\mathbf{R};\mathbf{T})\) |
| Hybrid Target Attention | 每个 Block 内 1 层 Full Attention + \(K\) 层 Target Attention 交替：Full 层对 H/R/T 全部 token 做 GQA self-attention（按序列/场景分组做 Group LayerNorm）；Target 层只更新 T-token——Q 取自 T-token，K/V 用全部 token，H/R token 靠 shortcut 直达下一层。复杂度从 \(O(N^2)\) 降到 \(O\!\left(\frac{KNL_T+N^2}{K+1}\right)\)，\(L_T \ll N\) |
| Dynamic mask | H-token 对所有 token 可见；R-token 只对时间戳更晚的 token 可见（防泄漏）；T-token 只能看到自己——包括同场景的其他候选和其他场景的候选都不可见 |
| 系统协同设计 | 训练侧按用户聚合多场景曝光（沿用 MTGR 的 user-level 压缩），减少样本冗余；推理侧按场景切出独立子图，只算该场景相关的 T-token 与共享 H/R，跳过其他场景的专属参数 |
| 任务预测端 | 最终层 T-token 表示送入传统 MMoE，输出各场景下 CTR / CTCVR 等多目标预测 |

**和 MDL 架构的差异：**

|  | MDL | MTFM |
|---|---|---|
| Token 组织依据 | 按语义职责分三类：Feature（共享证据）/ Scenario / Task（条件） | 按数据来源与时间角色分三类：H（历史）/ R（实时）/ T（当前场景候选） |
| 谁代表"场景" | 独立于 Feature 之外的 Scenario token，逐层从 Feature 读取信息再融合进 Task | 场景信息直接嵌进 T-token 本身（scenario-specific tokenizer + 专属参数），没有独立于候选之外的 Scenario 状态 |
| 主干自更新范围 | Feature self-interaction 只更新 Feature，Domain 不参与主干自更新 | Full Attention 层里 H/R/T 全部一起做 self-attention 更新，Target 层才收窄到只更新 T |
| 跨场景/跨任务 token 是否互见 | Scenario 之间、Task 之间都无独立 mixing，只能借共享 Feature 间接联系 | T-token 之间（包括同场景候选）任何一层都互不可见，跨场景知识只能借共享 H/R 间接传递——与 MDL"禁止 Domain-Domain 直接 mixing"的思路一致，但落在完全不同的 token 类型上 |
| 任务预测端 | Task token 各自连接固定 logits head，Task 状态本身就是条件化产物 | 候选表示算完后仍交给外部 MMoE 做多任务预测——任务这一维没有被重新设计，重点全在场景/输入这一维 |
| 效率手段 | 无稀疏 attention 设计，依赖 RankMixer 式 Per-token FFN 控制主干成本 | Hybrid Target Attention + GQA 专门解决异构长序列的二次复杂度问题 |

MDL 解决的是"字段能对齐时，场景与任务该怎样读取共享主干"；MTFM 解决的是"字段对不齐时，怎样先把异构数据变成能进同一个主干的 token，再把算力控制住"。

---

## 四、分析总结

| 工作 | 核心问题 | 解法落点 |
|---|---|---|
| MDL | 条件信息能读取哪些模型容量 | Scenario/Task 逐层读 Feature，不重复计算主干 |
| OneRank | 编码完之后还要不要切到外部预测器；任务之间的信息与梯度能不能分开处理 | 任务 token 挂在候选上、候选级聚合、Task–Task 受控 attention + 梯度隔离、动态匹配打分，全程留在 Transformer 内 |
| MTFM | 什么样的异构数据能进同一个模型；进去之后算力能不能扩展 | H/R/T 异构 tokenization + Hybrid Target Attention + 系统级协同优化 |

三篇工作其实回答的是同一个总问题的三个层次：**条件信息（场景 / 任务）应该在大主干的什么位置、以什么形式参与计算。** MDL 把它从末端推进到逐层读取；OneRank 进一步把预测端也收回 Transformer 内部；MTFM 则把问题前置到输入层——先解决异构 schema 如何进同一个模型，再用稀疏化结构把算力压在可扩展范围内。

---

## 附：本次校对相对原稿的主要修正

| 原稿问题 | 修正 |
|---|---|
| OneRank 数据集写成"3.3 亿曝光" | 原文 Table 2 为 **26.6B 曝光**（266 亿）；33M 是用户数 |
| MTFM 写成"三场景合计曝光 36 亿+" | 原文 Table 1 为 HP 18.53B + PHF 15.29B + SQS 2.24B，合计约 **36B（360 亿量级）** |
| OneRank 离线表只保留 C/A/O-AUC 三列 | 补回 C/A/O-**GAUC** 三列（如 OneRank O-GAUC 0.8350 vs 基线 0.8336），并注明完整对比含 16 组 encoder×predictor |
| MTFM 效率一句话（"显存从 66.97 降到 67.49 打平但吞吐翻倍"）自相矛盾 | 恢复原文 Table 4 完整 7 配置：同 batch 下显存 66.97→34.08 GB，2× batch 后吞吐 390→780 samples/s、显存持平 67.49 GB |
| MTFM 推理优化漏项 | 补回细粒度计算跳过（约 +5% 吞吐）与场景感知子图部署、BF16、M-Falcon |
| MTFM 主结果只有 pp 摘要 | 补完整 8 方法 ×（AUC/GAUC）主表（HP/PHF 与 SQS，SQS 含 IMD/WRITE 核销指标） |
| "MTFM 全面超过所有基线" | 论文口径为 **nearly all**：SQS WRITE AUC 上 MTFM 0.9079 略低于 RankMixer 0.9080 |
| MTFM 场景名"美团首推餐厅推荐" | 原文为 **Homepage（HP）首页餐厅推荐** |
| 条件路径对比表丢 RankMixer 行 | 补回（RankMixer 把主要容量放进深层 blocks，但不提供逐层 Scenario/Task 状态）——这正是 MDL 动机的落点 |
| 公式排版错误 | 修复 \(\operatorname{FFN}_k^s\)、\(\mathrm{MLP}_s\)、encoder–predictor 公式（\(\mathbf{Z}=\mathcal{F}(\mathbf{X}),\ \hat y=\mathcal{G}(\mathbf{Z})\)）等 |

## 相关文档

| 主题 | 文档 |
|---|---|
| MDL 论文精读（机制、证据边界、复现问题清单） | [`document.md`](../document.md) |
| 复现过程中的问题和解决方案 | [`mdl_reproduction_issues_solutions.md`](./mdl_reproduction_issues_solutions.md) |
| 关键问题集（详版） | [`mdl_key_questions.md`](./mdl_key_questions.md) |
| OneTrans 适配难点 | [`mdl_onetrans_adaptation_hardships.md`](./mdl_onetrans_adaptation_hardships.md) |
| 复现串讲底稿 | [`mdl_reproduction_lecture_report.md`](./mdl_reproduction_lecture_report.md) |
