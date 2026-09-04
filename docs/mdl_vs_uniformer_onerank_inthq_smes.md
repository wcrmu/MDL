# IntHQ、SMES、UniFormer、OneRank 与 MDL：核心机制对比

> 说明：题目中实际列出了五个方法；`SEMS` 应为 **SMES**。本文只依据五篇原始论文分析，不把论文中的宣传性命名直接当作结构事实。

## 1. 先给结论

这五个方法并不是同一个问题上的五种可互换解法。

- **SMES** 解决的是：多任务 MoE 如何增加总参数而不让在线计算量同步爆炸。它本质上仍是“共享 Backbone + MoE + Task Head”。
- **UniFormer** 解决的是：如何把序列建模、特征交互和多任务建模放进一套可扩容的工业排序架构。它先完成特征建模，再进入任务建模。
- **MDL** 解决的是：一个大规模排序模型如何同时适配多个场景和多个任务。核心是让场景 Token、任务 Token 在每层读取特征 Token。
- **OneRank** 解决的是：Transformer 排序模型不应先产出统一表示、再外挂 MLP 多任务头。它让每个候选的任务 Token 从输入层开始形成任务表示，并在候选集合级别完成任务交互和动态打分。
- **IntHQ** 解决的是：生成式推荐中的多个耦合决策如何同时获得任务专属表示、显式任务关系和不同深度的信息。它使用参数分离的 Context Stream 与 Task Stream，并增加跨层读取。

一句话概括：

| 方法 | 真正的核心 |
|---|---|
| SMES | **稀疏执行哪些 Expert** |
| UniFormer | **先建模 Feature Space，再建模 Task Space** |
| MDL | **Scenario/Task Token 逐层读取 Feature Token** |
| OneRank | **每个候选从底层拥有隔离的任务表示，最后做候选集合级任务交互与匹配打分** |
| IntHQ | **Context/Task 双流分参，任务逐层读上下文、逐层交互，最后再跨层选信息** |

因此：

- 研究多场景统一建模，MDL 最直接。
- 研究任务 Token 如何深度进入排序模型，OneRank、IntHQ、MDL 更相关。
- 研究显式任务依赖，OneRank、IntHQ、UniFormer 更相关，但三者发生交互的位置完全不同。
- 研究模型扩容与线上延迟，SMES、UniFormer 更相关。
- 研究不同任务需要不同网络深度，只有 IntHQ 直接解决。

## 2. 总体对比

| 维度 | IntHQ | SMES | UniFormer | OneRank | MDL |
|---|---|---|---|---|---|
| 主要范式 | 生成式多任务推荐 | 判别式多任务排序 | 判别式多任务排序 | Transformer-native 多任务排序 | 多场景、多任务排序 |
| 主要对象 | 用户历史中的多种出行决策 | 多任务 MoE Expert | 序列、非序列特征和任务 | 用户上下文、候选集合和任务 | 特征、场景和任务 |
| 任务信号首次进入位置 | 输入层建立独立 Task Stream | Backbone 后的 Router/Head | FIM 完成后进入 TIM | 输入序列中的 Candidate-Task Group | 输入层构造 Task Token |
| 场景建模 | Scenario 是历史 Context Token，不是独立多场景路由轴 | 无专门设计 | 可作为 Context/Task Feature，但无显式 Scenario Token 机制 | Situational Descriptor 表示当前情境，但不是多场景参数隔离 | 显式 Scenario Token + Global Scenario Token |
| 任务读特征 | 每层 Task→Context Cross-Attention | Task Gate 选择 Expert | TIM 中 Task→最终 FIM 表示 | 编码器内任务 Token 读取共享上下文和本候选 | 每层 Task→Feature Cross-Attention |
| 任务间交互 | 每层 Task Self-Attention，因果 Mask | 无显式 Task→Task 表示交互；通过共享 Expert 间接共享 | 每层 TIM 的 Task Self-Attention，论文未给任务依赖 Mask | 编码后做一次可配置 Mask 的 Task Self-Attention | 无显式 Task Self-Attention |
| 场景与任务融合 | 主要由同一时序结构和因果 Mask 体现 | 无 | Task Feature 可含上下文，但没有独立融合模块 | Situational Descriptor 先按任务聚合候选 | 当前 Scenario Token 均值直接加到所有 Task Token |
| 跨层利用 | 显式 HQ，对每个任务自适应聚合各层 Task State | 无 | 无；TIM 默认重复读取同一份最终 FIM KV | 无；读取编码器最终层 | 隐式逐层更新，但最终只用最后一层 Task Token |
| 任务私有参数 | Context Core 与 Task Core 分离；任务之间仍共享 Task Core；最终 Head 私有 | Task Router、Task Head；Expert 池共享 | Task Token、Per-task T-FFN、Head | Task Token；任务专属 SD Projection 和 Cross-Candidate Attention；编码器参数共享 | Per-task Token FFN/QKV、Head；Per-scenario FFN/QKV |
| 梯度隔离 | 无显式跨任务 Detach | 无 | 无 | 最终跨任务注意力对非对角路径 Detach | 无 |
| 输出 | 分类或大词表检索；按任务 InfoNCE/Softmax | Task Head 概率 | Task-specific FFN + Sigmoid | 全局任务向量与候选任务向量内积 | 最终 Task Token 接 Logits Layer |
| 扩容手段 | 增加双流深度、宽度 | 增加 Expert 总数，保持稀疏激活 | 增加 FIM/TIM 层数和多视角 FFN 宽度 | 增加 Transformer 深度、宽度 | 增加 MDL Block 深度、宽度 |
| 最独特能力 | 任务级跨层选择 | 总参数与激活计算解耦 | 工业特征空间与任务空间共同扩容 | 候选集合级动态排序 + 前向共享/反向隔离 | 场景 × 任务组合式建模 |

## 3. 五个方法分别在做什么

### 3.1 IntHQ

原论文：[IntHQ: Task-Interactive Hierarchical Query on Dual-Stream Representations for Generative Recommendation](https://arxiv.org/abs/2608.09634)

#### 解决的问题

IntHQ 面向多任务生成式推荐。论文将传统方法的问题归纳为三类：

1. **Source Collapse**：先用任务无关 Backbone 压缩历史，再由任务头读取；一旦压缩时丢掉某个任务需要的信息，后面的头无法恢复。
2. **Relational Collapse**：任务要么仅通过共享 Backbone 隐式联系，要么使用固定漏斗关系。
3. **Hierarchical Collapse**：不同深度表示不同粒度的信息，而所有任务只读最后一层或固定层权重。

#### 真实信息流

输入由两条序列组成：

- Context Stream：用户 Profile Token，以及每个 Session 的 Scenario、Item、Feedback Token。
- Task Stream：每个 Session 都实例化 `when / where / how / via` 四个可学习任务 Token。

两条流每层进行：

\[
\widetilde H_{ctx}^{(l)}=\operatorname{Attn}_{ctx}(H_{ctx}^{(l)},H_{ctx}^{(l)})
\]

\[
\widetilde H_q^{(l)}=\operatorname{Attn}_{q}(H_q^{(l)},H_{ctx}^{(l)})
\]

\[
\widehat H_q^{(l)}=\operatorname{Attn}_{q}(H_q^{(l)},H_q^{(l)})
\]

\[
H_q^{(l+1)}=H_q^{(l)}+\widetilde H_q^{(l)}+\widehat H_q^{(l)}
\]

即：Context 自己编码历史；Task Token 在每层读取当前层 Context，同时在 Task Stream 内交互。三次 Attention 读取同一层输入，可并行计算。

#### DSD 的准确含义

“Dual-Stream Decoupling”不是每个任务拥有一个独立网络，而是：

- `Attn_ctx` 与 `Attn_q` 参数不共享；
- Context Token 和 Task Token 不拼成同一个序列；
- 但所有任务 Token 仍共享同一套 `Attn_q` 参数。

因此 DSD 隔离的是“上下文建模”和“任务建模”两种计算角色，不是彻底隔离各任务梯度。

#### TIM 的准确含义

Task Self-Attention 使用因果可见性：后续决策可以看前序决策，反向不可见。这比固定 ESMM 式漏斗更灵活，因为 Attention 权重会随输入变化。

但论文摘要使用了“condition on realized outcomes”的强表述。按 Method 和 Algorithm 1，TIM 直接读取的是**前序任务 Token 的隐藏表示**，算法没有显式把真实标签或已经预测出的离散结果 `y_j` 再输入后续任务。因此更准确的说法是：

> IntHQ 做的是带因果约束的任务隐藏状态交互，而不是严格意义上显式喂入前序预测结果的自回归任务链。

#### HQ 的准确含义

对任务 \(k\)，收集全部层的 Task State：

\[
Z_k=[H_{q,k}^{(1)};\ldots;H_{q,k}^{(L)}]
\]

再用原始可学习任务身份 \(q_k\) 形成 Query，各层状态形成 Key/Value：

\[
Q_k=W_Qq_k,\quad K_l=W_KZ_k[l]+d_l,\quad V_l=W_VZ_k[l]
\]

\[
z_k=\operatorname{LN}\left(H_{q,k}^{(L)}+\sum_l\alpha_l^{(k)}V_l\right)
\]

关键点：

- Query 是任务身份 `q_k`，不是最终层 `H_q^(L)`。
- Key/Value 是**该任务在各层的 Task State**，不是原始 Context State。
- 权重不仅依赖任务身份，也依赖当前样本各层状态，所以会随用户和 Session 变化。
- 最后一层仍通过残差保留，再加跨层加权结果。

#### 优点与边界

优点：

- 任务从输入层开始读取历史，不等到最终 Head。
- Context/Task 分参，避免两类 Token 共用同一 Attention Core 的角色冲突。
- 任务交互发生在每一层。
- HQ 真正解决不同任务依赖不同深度的问题。

边界：

- 结构为生成式多决策建模，不是标准 CTR Point-wise Ranker。
- Context Stream 始终是任务无关的；任务只读取 Context，没有反向改写 Context。
- 各任务共享 Task Core，仍可能在共享参数上发生梯度冲突。
- 因果依赖顺序仍需由 `c(·)` 事先定义；自适应的是交互强度，不是任意学习任务拓扑。
- 实验只覆盖出行领域的四个耦合决策，能否迁移到点击、加购、支付等排序任务仍需验证。

### 3.2 SMES

原论文：[SMES: Towards Scalable Multi-Task Recommendation via Expert Sparsity](https://arxiv.org/abs/2602.09386)

#### 解决的问题

普通 Dense MMoE 的所有 Expert 都会执行，计算量随 Expert 数 \(E\) 线性增长。直接让每个任务独立 Top-K 又会出现：

- 多个任务选中不同 Expert，所有任务的 Expert 并集接近全部 Expert，稀疏执行失效；
- 少数热门 Expert 同时承受多个任务的流量和梯度，其他 Expert 训练不足。

#### 真实信息流

SMES 保留传统结构：

\[
x\rightarrow F(x)=h\rightarrow \text{Sparse MoE}\rightarrow h_t\rightarrow \phi_t(h_t)
\]

每个任务先产生对全部 Expert 的 Router 分数。随后分两阶段选 Expert：

1. 将各任务概率加权求和，联合选择 \(K_s\) 个 Task-shared Expert：

\[
s_e=\sum_tw_tp_{t,e},\qquad \mathcal S=\operatorname{TopK}(s,K_s)
\]

2. 每个任务从剩余 Expert 中再选择 \(K_a\) 个 Task-adaptive Expert：

\[
\mathcal A_t=\operatorname{TopK}(z_{t,e}:e\notin\mathcal S,K_a)
\]

最终：

\[
\mathcal K_t=\mathcal S\cup\mathcal A_t,\qquad K=K_s+K_a
\]

所有任务所需 Expert 取并集后，每个 Expert 对同一样本最多执行一次：

\[
\mathcal U=\bigcup_t\mathcal K_t,\qquad |\mathcal U|\le K_s+TK_a
\]

#### Load Balance

SMES 同时统计所有样本、所有任务对 Expert 的选择频率和概率质量：

\[
\mathcal L_{lb}=\frac{E}{K}\sum_e\bar f_e\bar p_e
\]

这比给每个 Router 单独做平衡更贴近实际系统负载，因为线上瓶颈取决于所有任务合并后的 Expert 流量。

#### 优点与边界

优点：

- 总参数量由 \(E\) 决定，单样本计算只与激活并集 \(|\mathcal U|\) 有关。
- 共享选择保证最少有 \(K_s\) 个 Expert 被各任务共同使用。
- Task-adaptive 部分允许任务选择不同 Expert。
- 去重执行、Grouped GEMM 和 Workspace 管理把算法稀疏性转化为实际延迟收益。

边界：

- 它没有改变“共享 Backbone + 多任务头”的基本范式。
- 没有任务 Token、任务 Self-Attention、显式任务依赖或多场景机制。
- 每个任务每个样本仍固定激活 \(K_s+K_a\) 个 Expert。所谓“异构任务容量自适应”主要表现为**选择哪些 Expert**，不是每个任务自动决定激活多少 Expert。
- \(|\mathcal U|\le K_s+TK_a\) 仍随任务数 \(T\) 线性增长；它抑制爆炸，但没有让成本对任务数保持常数。
- Expert 稀疏路由不能自动解决共享 Backbone 中的信息瓶颈和梯度冲突。

### 3.3 UniFormer

原论文：[UniFormer: Efficient and Unified Model-Centric Scaling for Industrial Recommendation](https://arxiv.org/abs/2606.27058)

#### 解决的问题

传统工业模型分别堆叠序列模块、特征交互模块和多任务模块。UniFormer 希望统一这些计算算子，并同时扩展 Feature Space 与 Task Space。

#### 真实信息流

UniFormer 不是把全部 Token 丢进一个 Full Self-Attention，而是明确分成两个阶段：

1. `FIM × M`：序列与非序列特征交互。
2. `TIM × N`：任务读取最终特征，并进行任务间交互。

##### FIM

非序列特征按语义分组形成 Query Token。每个序列单独提供 KV：

\[
H_{seq}^{(l)}=\operatorname{CA}(Q_{cross}^{(l-1)},K_{seq}^{(l)},V_{seq}^{(l)})+Q_{cross}^{(l-1)}
\]

短期、长期等不同序列分别 Cross-Attention、分别经过 Sequence-specific FFN，再进行全局或个性化加权融合。融合结果与上一层非序列表示拼接后做 Self-Attention，并对不同 Feature Slice 使用独立 FFN。

这部分的核心不是“用了 Transformer”，而是：

- 不同序列不先拼接，避免某一序列主导 Attention；
- 每种序列、每组非序列语义拥有专属 FFN；
- 用 Cross-Attention 避免序列内部全量两两交互；
- 利用语义分组实现 User-Item Decoupling 和请求级复用。

##### TIM

第一个 TIM Layer 以 Task Feature Token 为 Query，读取最终 FIM 表示：

\[
H_{cross}^{task,(l)}=\operatorname{CA}(Q_{task}^{(l-1)},K_{feat},V_{feat})+Q_{task}^{(l-1)}
\]

随后 Task Token 做 Self-Attention，再分别经过 Per-task FFN：

\[
H_{self}^{task,(l)}=\operatorname{SA}(H_{cross}^{task,(l)})+H_{cross}^{task,(l)}
\]

\[
f_i^{task,(l)}=\operatorname{FFN}_i(h_i^{task,(l)})+h_i^{task,(l)}
\]

该过程重复 \(N\) 层，最终每个任务接独立 FFN Head。

#### 容易误读的点

- “Unified”不表示 Task Token 从 FIM 底层参与特征建模。论文结构是**先 FIM，后 TIM**。
- TIM 默认把最终 `F^(feat,M)` 作为 KV，并可在所有 TIM Layer 共享这份 KV。它不是像 IntHQ 那样逐层读取不同深度的 Context State。
- Task Self-Attention 没有给出业务依赖 Mask 或梯度 Detach，默认是任务间自由双向交互。
- 真正支撑扩容的很大一部分参数来自 Sequence-specific、Feature-specific、Task-specific FFN，而不仅是 Attention。

#### 优点与边界

优点：

- 同时覆盖长短序列、非序列特征交互、多任务建模和系统优化。
- Task Token 不仅加权读取特征，还反复进行任务交互和私有 FFN 变换。
- 用户侧 Token、序列 KV 可以按请求复用；论文报告 512 候选设置下 QPS 提升 48%。
- 适合需要大规模工业特征与严格延迟约束的排序模型。

边界：

- 任务信息仍晚于完整 FIM；任务无法改变底层特征提取过程。
- 无显式多场景参数隔离。
- 无任务关系约束时，Task Self-Attention 可能学习有害或不符合业务因果方向的传递。
- 没有跨层任务选择；不同任务只能从同一份最终 FIM 表示开始。

### 3.4 OneRank

原论文：[OneRank: Unified Transformer-Native Ranking Architecture for Multi-Task Recommendation](https://arxiv.org/abs/2606.16838)

#### 解决的问题

OneRank 反对“Transformer Encoder 产生统一表示，再外挂 MMoE/PLE/MLP Head”的两段式设计。它希望在 Transformer 内部形成任务表示，并把最终打分也改成基于上下文的 Matching。

#### 真实信息流

输入包括：

- Interaction History；
- 检索得到的 Preference Anchors；
- 对每个候选 \(c_i\) 构造一组 `[Candidate, Task_1, ..., Task_K]`。

任务 Token 参数在所有候选间共享；每个候选只是实例化同一组任务模板。因此不是“每个候选拥有一套独立任务参数”。

编码 Mask 满足：

- User Context 使用 Causal Attention；
- 不同 Candidate Group 互不可见；
- 一个 Candidate Group 内，不同任务 Token 互不可见；
- 每个任务 Token 只能读取共享 User Context、本候选和自己。

经过共享 Transformer 后得到每个候选、每个任务的表示：

\[
r_k^i=\operatorname{Extract}(X^{(L)},t_k^{(i)})
\]

随后才发生候选集合级建模：

1. 当前用户、Query、时间、地点等形成 Situational Descriptor \(s\)。
2. 每个任务使用独立投影与 Cross-Attention，从全部候选 \(\{r_k^i\}\) 聚合一个全局向量 \(h_k\)。
3. 全部 \(h_k\) 做带可配置 Mask 的跨任务 Self-Attention。
4. 得到全局任务向量 \(z_k\)，再与每个候选任务向量内积：

\[
s_k^i=z_k^\top r_k^i
\]

因此 OneRank 的任务交互发生在**候选集合已经按任务聚合之后**，而不是输入 Transformer 的每一层。

#### Gradient Detachment

任务 \(k\) 前向可以读取任务 \(j\) 的 \(h_j\)，但任务 \(k\) 的损失不能沿非对角 Attention 路径更新 \(h_j\)。它把跨任务表示当作“可读、不可由本任务反向修改”的 Memory。

准确边界：

- Detach 只隔离最终 Cross-task Attention 的非对角梯度。
- 前面的 Transformer Encoder 仍由所有任务共同训练；任务 Token 互不可见只阻止它们在前向时直接交换表示，并没有让共享 MHSA/FFN 参数免受多任务梯度冲突。
- 因此论文所说的 Task-private Channel 是表示路径私有，并不等于整条编码参数完全私有。

#### Mask 的意义

OneRank 可配置：

- Parallel：任务互不可见；
- Null：任务全可见；
- Cascade：下游任务只读取自己和上游任务；
- Hybrid：人工定义任意部分关系。

它比 IntHQ 的 Session 因果顺序更适合 CTR/加购/支付等业务漏斗，也比 UniFormer 无约束 Self-Attention 更可控。

#### 训练目标

OneRank 同时优化：

- 候选集合上的 InfoNCE/List-wise Loss；
- 每个候选上的 BCE/Point-wise Loss。

所以它既学习相对排序，也保持工业系统所需的概率估计。

#### 优点与边界

优点：

- 从输入层就为每个候选建立任务专属表示。
- Candidate Group Mask 支持单用户、多候选并行。
- 显式建模候选集合竞争关系，避免只看孤立 `user-item` Pair。
- Task Mask 控制业务依赖；Detach 将前向知识迁移与部分反向优化隔离。
- 动态全局任务向量与候选向量 Matching，比分离的静态 MLP Head 更接近实际排序。

边界：

- 编码器参数仍然共享，并未彻底解决 Backbone 梯度冲突。
- 跨任务交互只发生在编码后的集合级全局表示上。
- Task-specific Cross-Candidate Attention 随任务数增加参数和计算。
- 分数依赖当前候选集合；候选池发生变化时，同一 `user-item` Pair 的分数也可能变化。这是 Set-wise Ranking 的能力，也是部署和缓存上的约束。
- 论文主要使用 Shopee 私有数据，缺少公共数据复现。

### 3.5 MDL

原论文：[MDL: A Unified Multi-Distribution Learner in Large-scale Industrial Recommendation through Tokenization](https://arxiv.org/abs/2602.07520)

#### 解决的问题

MDL 将不同 Scenario 与不同 Task 都看成数据分布差异。它反对只在 Backbone 末端加场景塔、任务塔或 Gate，希望场景和任务从底层开始逐层读取大规模 Feature Interaction Backbone。

#### Token 的真实来源

- Feature Token：按语义人工分组 Feature Embedding，再投影到统一维度。
- Scenario Token：重要原始特征的额外 Embedding + 场景相关特征，经 Per-scenario FFN 得到。
- Global Scenario Token：学习跨场景共性。
- Task Token：重要特征与任务相关特征经 Per-task FFN 得到。

因此 MDL 的 Scenario/Task Token 不是只有一个静态 ID Embedding，而是**样本相关、特征条件化的表示**。

#### 每个 MDL Block 的真实信息流

1. Feature Token 自己做 RankMixer 风格 `TokenMixing + Per-token FFN`：

\[
T_f^{(l+1)}=\operatorname{PerTokenFFN}(\operatorname{LN}(\operatorname{TokenMixing}(T_f^{(l)})+T_f^{(l)}))
\]

2. Scenario Token 以自己为 Query，读取更新后的 Feature Token：

\[
\widehat T_s^{(l+1)}=\operatorname{CrossAttn}(T_s^{(l)},T_f^{(l+1)})+T_s^{(l)}
\]

3. Task Token 同样读取 Feature Token：

\[
\widehat T_t^{(l+1)}=\operatorname{CrossAttn}(T_t^{(l)},T_f^{(l+1)})+T_t^{(l)}
\]

4. 只选择当前样本所属 Scenario Token，加上 Global Scenario Token，Mean Pooling 后直接加到所有 Task Token：

\[
t_{s,avg}=\operatorname{MeanPool}(\text{active scenario tokens},t_{s,global})
\]

\[
\widetilde T_t^{(l+1)}=\widehat T_t^{(l+1)}+t_{s,avg}
\]

5. Scenario Token 与 Task Token 分别经过 Per-token FFN。最终只取最后一层 Task Token 接 Logits Layer。

#### “All-Token Interaction”容易造成的误解

MDL 不是所有 Token 做一次 Full Self-Attention。按论文公式：

- Feature Token 只和 Feature Token 交互；
- Scenario Token 单向读取 Feature Token；
- Task Token 单向读取 Feature Token；
- Feature Token 不读取 Scenario/Task Token；
- Task Token 之间没有 Self-Attention；
- Scenario 与 Task 的交互只是“选择 Scenario → Mean Pool → 加到每个 Task Token”。

因此更准确的描述是：

> MDL 是以共享 Feature Stream 为只读知识源，Scenario/Task Query 在每层分别读取，再把当前场景偏置注入任务表示；不是三类 Token 的完全双向联合建模。

#### 优点与边界

优点：

- 五个方法中唯一把多 Scenario 作为一等建模对象的方法。
- Scenario/Task 从每层读取 Feature Token，比只在顶层 Gate 或 Head 中使用更深。
- Per-scenario/Per-task 参数提供分布专属容量。
- 当前场景选择使无关 Scenario Token 不参与输出，支持 Scenario × Task 的组合式预测，而无需为每个组合建一套 Head。
- Feature Self-Interaction 可替换，不强绑定 RankMixer。

边界：

- Feature Stream 不接收 Scenario/Task 反馈，因此不能称为真正的场景条件化 Backbone。
- 不建模 Task→Task 依赖，支付无法显式读取加购等任务表示。
- 场景到任务只用 Mean Pool + Addition，简单高效，但表达力弱于 Cross-Attention 或 Gate。
- 最终只使用最后一层 Task Token，没有显式跨层选择。
- 多任务、多场景仍会通过共享 Feature Stream 和 Embedding 产生梯度干扰。

## 4. 最容易混淆的结构差异

### 4.1 都叫 Task Token，但含义不同

| 方法 | Task Token 初始值 | 是否每个候选一份 | 是否每个 Session 一份 | 是否读取特征/上下文 | 是否任务间交互 |
|---|---|---:|---:|---|---|
| IntHQ | 可学习任务身份 | 否 | 是 | 每层读 Context Stream | 每层因果 Self-Attention |
| UniFormer | Task ID、Task Bias 等特征经 SwiGLU | 否 | 否 | TIM 读取最终 FIM 表示 | 每层 TIM Self-Attention |
| OneRank | 可学习任务模板 | 是，但参数跨候选共享 | 否 | 编码器内读 User Context 和本候选 | 编码时禁止；预测前允许 |
| MDL | 重要特征 + Task Feature 经 Per-task FFN | 否 | 否 | 每个 MDL Block 读取 Feature Token | 没有直接交互 |
| SMES | 无 Task Token | — | — | Task Router 读取共享表示 | 只通过共享 Expert 间接联系 |

### 4.2 都有 Cross-Attention，但读取对象不同

| 方法 | Query | Key/Value | 发生位置 |
|---|---|---|---|
| IntHQ DSD | 当前层 Task State | 同层 Context State | 每个双流层 |
| IntHQ HQ | 原始 Task Identity | 该任务全部层的 Task State | 编码结束后 |
| UniFormer FIM | 语义分组的非序列 Feature Token | 各类行为序列 | 每个 FIM Layer |
| UniFormer TIM | Task Token | 最终 FIM Feature | 每个 TIM Layer；KV 默认可共享 |
| OneRank Candidate Context | Task-specific Situational Descriptor | 同任务的全部候选表示 | 编码结束后、任务交互前 |
| MDL | Scenario Token 或 Task Token | 当前层 Feature Token | 每个 MDL Block |

### 4.3 任务交互发生的位置

- **IntHQ**：Task Stream 每一层交互，随后还有 HQ。
- **UniFormer**：FIM 完成后，Task 在 TIM 的每一层交互。
- **OneRank**：共享 Transformer 编码阶段任务互不可见；编码结束、按任务聚合候选后才交互。
- **MDL**：没有直接 Task→Task 交互。
- **SMES**：没有任务表示交互，只共同选择/使用 Expert。

### 4.4 “任务信息进入得早”不等于“任务改写 Backbone”

这是五篇论文中最容易被忽略的共同点：

- IntHQ：Task 每层读取 Context，但 Context 不读取 Task。
- MDL：Task/Scenario 每层读取 Feature，但 Feature 不读取 Task/Scenario。
- OneRank：Task Token 与 Context 在同一 Transformer 中计算并共享参数，但 Mask 下 User Context 不需要读取 Task；任务主要形成自己的读取路径。
- UniFormer：Task 在完整 FIM 后才出现。
- SMES：Task 在共享 Backbone 后通过 Router 出现。

因此 IntHQ、MDL 的“早期任务注入”更准确地说是：**早期建立任务专属读取通道**，不是把共享特征 Backbone 变成双向任务条件化网络。

## 5. 参数共享与梯度冲突

| 方法 | 主要共享参数 | 主要私有参数 | 对梯度冲突的真实处理 |
|---|---|---|---|
| IntHQ | Context Core 被所有任务共享；Task Core 也被所有任务共享 | Context Core 与 Task Core 彼此分离；Task Head | 隔离两类计算角色，但不隔离各任务在 Task Core/Context Core 上的梯度 |
| SMES | Backbone、Expert 池 | Router、Head；每样本选择不同 Expert | 通过路由减少同时更新的 Expert，但无显式梯度处理 |
| UniFormer | FIM Attention、TIM Attention | Sequence/Feature/Task-specific FFN、Head | 私有 FFN 提供容量隔离；无 Mask/Detach |
| OneRank | Transformer MHSA/FFN | Task Token、Task-specific SD Projection/MHCA | 最终跨任务 Attention 非对角 Detach；共享编码器梯度仍混合 |
| MDL | Feature Interaction、Embedding | Per-task/Per-scenario FFN 与投影、Head | 依靠专属 Token/FFN 分担差异；无显式梯度隔离 |

如果目标是严格控制“支付任务不能反向破坏加购任务”，OneRank 的非对角 Detach 最直接；但它只覆盖最后的跨任务模块。若要覆盖 Backbone，还需 PCGrad、GradNorm、任务专属 Adapter/LoRA、分层 Detach 或更强参数隔离。

## 6. 扩容与线上效率

| 方法 | 参数如何增加 | 单样本计算如何变化 | 主要工程手段 |
|---|---|---|---|
| IntHQ | 增加双流层数、宽度 | 两条流都增大；HQ Bank 随层数增加 | 历史离线组装、前缀缓存、POI ANN 检索 |
| SMES | 增加 Expert 总数 \(E\) | 理想情况下只执行 \(|\mathcal U|\ll E\) | Expert 去重、Reindexed Grouped GEMM、预分配 Workspace |
| UniFormer | 增加 FIM/TIM 和多视角 FFN | Dense 计算增加，但通过结构拆分控制 | Lazy KV、Variable-length FlashAttention、User-Item Decoupling、请求级复用 |
| OneRank | 增加 Transformer 深度、宽度 | 随序列长度、候选数、任务数增加 | Candidate Group Mask、单用户多候选、User Context KV Cache |
| MDL | 增加 MDL Block 和 Token/FFN 宽度 | Dense Token Mixing、Cross-Attention 增加 | Token 数固定、简单 Scenario Mean Pool；论文系统优化披露较少 |

核心区别：

- SMES 是 **Sparse Scaling**：大量参数不在每个样本上执行。
- UniFormer、OneRank、IntHQ、MDL 主要是 **Dense Structured Scaling**：通过合理的信息流让新增参数更有效，计算通常仍随深度和宽度增加。

## 7. 实验结果应该怎样看

五篇论文没有在同一数据、同一任务、同一参数统计口径下直接对比，不能根据线上提升数字排序。

| 方法 | 离线范围 | 论文报告的线上结果 | 不能直接比较的原因 |
|---|---|---|---|
| IntHQ | IntTravel：约 41 亿交互、1.63 亿用户、730 万 POI；When/Where/How/Via | Amap UVCTR 相对 +1.60%，平均延迟 <40 ms | 生成式分类/检索任务，不是 CTR 多目标排序 |
| SMES | KuaiRand-1K + Kuaishou 工业数据；多种时长/互动任务 | Watch Time +0.31%；Like +0.64%；Follow +1.56%；Comment +2.45%；相对 Dense MoE 延迟 -50% | 更换的是 MoE 模块，重点是稀疏扩容 |
| UniFormer | Kuaishou 私有数据；Effective-view/Long-view/Like/Follow | Kuaishou：App Stay Time +0.101%、Watch Time +0.729%；Lite 收益更高 | 完整大规模 Ranker，参数和特征系统远多于单一多任务头 |
| OneRank | Shopee 私有数据；Click/Add-to-cart/Order | GMV/UU +1.01%、Paid GMV/UU +1.17%、AR/UU +0.81%、Bad Query Rate -2.29% | 电商业务指标、候选集合训练，Baseline 不同 |
| MDL | Douyin Search 私有数据；3 场景、20+任务 | LT30 +0.0626%、Change Query Rate -0.3267% | 搜索多场景长期指标，任务与流量结构不同 |

还要注意以下论文证据边界：

1. **IntHQ** 的 TIM 消融对 When 完全无变化，对 Where 仅有极小变化，主要收益在依赖上游决策的 How/Via；这说明 TIM 不是所有任务都同等需要。
2. **SMES** 固定每任务每样本的 \(K_s+K_a\)。论文图中“稠密任务激活更多 Expert”更可能描述跨样本累计覆盖或负载，而不是单样本自动使用更多 Expert。
3. **UniFormer** 的关键消融主要以图展示，正文缺少完整数值；公开可审计性弱于表格消融。
4. **OneRank** 去掉 Gradient Detachment 后 A-AUC 从 0.8463 变为 0.8460，差距很小。该实验支持“有一定帮助”，但不足以单独证明共享编码器的梯度冲突已经解决。
5. **MDL** 表 2 中 `w/o task-feature interaction` 在 Inner Search 上标为 `+0.03%`，与正文“去掉每个组件都会下降”的概括不完全一致。不能据此断言 Task-Feature Attention 在所有场景都稳定有效。
6. 参数量统计口径可能不同。UniFormer 表中明确报告 Dense Network 参数；OneRank 的几百万参数、SMES/MDL/UniFormer 的数亿参数不能横向视为完整模型总量差异。

## 8. 哪些方法最相似，哪些只是表面相似

### 8.1 IntHQ 与 OneRank

相同：

- 都反对“任务无关 Encoder + 后置多任务 Head”。
- 都从较早位置建立 Task-specific Representation。
- 都支持显式 Task Mask。

不同：

- IntHQ 是 Context/Task 两套 Attention 参数；OneRank 的输入编码仍是一套共享 Transformer。
- IntHQ 每层做 Task→Context 和 Task→Task；OneRank 编码期任务互不可见，最后才交互。
- IntHQ 按 Session 组织生成式多决策；OneRank 按 Candidate Group 组织集合排序。
- IntHQ 用 HQ 聚合各层；OneRank 只取编码器最终层。
- OneRank 有 Cross-task Gradient Detach；IntHQ 没有。

### 8.2 UniFormer 与 MDL

相同：

- 都把任务表示为 Token。
- 都让 Task Token 通过 Cross-Attention 读取 Feature Token。
- 都使用 Per-task FFN 提供任务专属容量。

不同：

- UniFormer 先完成全部 FIM，再运行 TIM；MDL 在每个 Block 中交替更新 Feature 与 Task。
- UniFormer 有 Task Self-Attention；MDL 没有。
- MDL 有显式 Scenario Token；UniFormer 没有。
- UniFormer 的任务 KV 默认来自同一最终 FIM 表示；MDL 每层读取当层更新后的 Feature Token。
- UniFormer 更强调序列/非序列特征与线上推理系统；MDL 更强调 Scenario × Task 分布组合。

### 8.3 SMES 与其他四个

SMES 只有“多任务”和“Scaling”两个词与其他方法重合，结构目标不同。它最适合作为其他架构中的容量模块：例如把 UniFormer 的 T-FFN、MDL 的 Per-task FFN 或传统 Task Head 替换成稀疏 Expert 结构，而不是替代 Task Token、Task Attention 或 Scenario Modeling。

## 9. 如何选择

| 需求 | 更合适的方法 |
|---|---|
| 一个模型统一多个场景和任务 | MDL |
| 点击→加购→支付等显式业务依赖 | OneRank；若是序列化多决策，可参考 IntHQ |
| 不同任务需要不同深度特征 | IntHQ 的 HQ |
| 任务关系需要逐层交互 | IntHQ；或 UniFormer 的多层 TIM |
| 候选之间存在竞争，需要 Set-wise 排序 | OneRank |
| 长短行为序列、非序列特征和任务一起扩容 | UniFormer |
| 增大模型参数但严格限制在线 FLOPs | SMES |
| 需要最强场景专属参数路径 | MDL 的 Scenario Token/Per-scenario FFN，可再加强为场景 Adapter/Expert |
| 需要控制跨任务反向干扰 | OneRank 的 Detach，但建议同时检查共享 Backbone 梯度 |

## 10. 对统一多场景、多目标排序模型的可复用组合

如果目标是多场景、多目标工业 Ranker，这五篇论文最值得组合的不是完整模型，而是以下五个互补思想：

1. **UniFormer 的 Feature Space**：先把长短序列与非序列特征高效建模，并做好 User-Item Decoupling。
2. **MDL 的 Scenario/Task Token**：场景和任务从底层逐层读取 Feature Representation，而不是只在 Head 使用。
3. **OneRank/IntHQ 的 Task Mask**：明确限定允许的信息方向，例如 `支付 Query → 加购 KV`，禁止反向泄漏。
4. **IntHQ 的 HQ**：最终按任务聚合不同深度的 Task State，避免所有任务强制使用最后一层。
5. **SMES 的 Sparse Expert**：当 Dense 模型的信息流验证有效后，再用于扩大 Per-task FFN/Expert 容量并控制延迟。

一个更合理的顺序是：

\[
\text{Feature Modeling}
\rightarrow
\text{Layer-wise Scenario/Task Query}
\rightarrow
\text{Masked Task Interaction}
\rightarrow
\text{Task-wise Cross-Layer Aggregation}
\rightarrow
\text{Head}
\]

Gradient Detach 不应默认全开。先测 Task Gradient Cosine、Task-wise AUC/GAUC 和 Seesaw，再决定只 Detach 哪些边。Sparse Expert 也应最后加入，因为它解决容量/计算问题，不解决信息流是否正确。

## 11. 最终判断

五篇论文可以放在一条演进线上理解：

1. **SMES**：在传统多任务框架内，把容量扩展做成稀疏执行。
2. **UniFormer**：把特征与任务都纳入统一的可扩展 Ranker，但仍是 Feature→Task 的阶段式结构。
3. **MDL**：把场景和任务提前为逐层 Query，实现多分布组合建模，但没有显式任务关系。
4. **OneRank**：任务表示从输入层进入，并增加候选集合建模、任务 Mask 和部分梯度隔离，但任务交互发生较晚。
5. **IntHQ**：把 Context/Task 计算分参，在每层同时做任务读上下文和任务交互，并显式聚合不同深度；代价是结构更偏生成式多决策，且没有任务级完整梯度隔离。

如果只看“任务 Token 前移”会把 OneRank、MDL、IntHQ 误认为同类；真正决定模型能力的是四件事：

- Task Token 在哪一层开始读取信息；
- Feature/Context 是否反向读取 Task；
- Task→Task 在哪里发生、是否有 Mask；
- 不同任务的梯度究竟在哪些共享参数上相遇。

这四个问题比论文使用的 “Unified”“Prompt”“Interactive”“Native” 等命名更能揭示结构本质。

## 原始论文

1. [IntHQ: Task-Interactive Hierarchical Query on Dual-Stream Representations for Generative Recommendation](https://arxiv.org/abs/2608.09634)
2. [SMES: Towards Scalable Multi-Task Recommendation via Expert Sparsity](https://arxiv.org/abs/2602.09386)
3. [UniFormer: Efficient and Unified Model-Centric Scaling for Industrial Recommendation](https://arxiv.org/abs/2606.27058)
4. [OneRank: Unified Transformer-Native Ranking Architecture for Multi-Task Recommendation](https://arxiv.org/abs/2606.16838)
5. [MDL: A Unified Multi-Distribution Learner in Large-scale Industrial Recommendation through Tokenization](https://arxiv.org/abs/2602.07520)
