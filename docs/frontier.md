## 11. 前沿延伸：多分布推荐继续向哪里演进

MDL 将 Scenario 与 Task 从末端的 gate、tower 或 head，提升为能够跨层保留并持续读取 Feature states 的条件状态。这一变化解决了一个重要问题：当排序主干不断扩大时，场景与任务信息不应只在计算链末端生效，而需要拥有与主干深度相匹配的信息路径。

但 MDL 并没有穷尽多分布推荐中的所有问题。沿任务维度看，不同任务除了读取不同的 Feature 信息，是否还需要显式传递知识，并限制彼此损失对私有状态的干扰？沿场景维度看，当不同业务场景连输入字段、行为序列和候选组织方式都无法对齐时，统一模型又该如何扩展？2026 年公开的 OneRank 与 MTFM，分别从这两个方向给出了新的回答。MDL、OneRank 和 MTFM 并不是直接的迭代关系，而是代表了 Transformer 化工业推荐中三种相互关联的问题意识。

### 11.1 OneRank：把多任务推理放进 Transformer 内部

传统的 Transformer 排序模型通常仍然沿用“共享编码器 + 多任务预测器”的结构：前面的 Transformer 先将输入压缩成一份 task-agnostic representation，随后再交给 MMoE、PLE 或多个任务塔进行预测。OneRank 认为，即使共享编码器变得更深、更大，这种结构仍然存在任务无关的信息瓶颈。不同任务真正需要的信息直到预测端才开始分化，而多个任务的梯度又会共同作用于共享编码路径，容易形成 seesaw phenomenon。OneRank 因此不再把多任务学习看成 Transformer 之后附加的预测模块，而是将 task-specific representation learning 直接放入 Transformer 的计算过程。

OneRank 首先在输入阶段引入任务专属 token，并通过 mutual invisibility 保持早期任务通道的相对独立。也就是说，Click、Add-to-Cart 和 Order 不再只能接收同一个共享表示，再由末端 MLP 完成区分；不同任务从底层开始就拥有自己的状态和参数路径，并能够根据任务目标选择所需信息。模型随后利用 candidate-aware contextualization 聚合一个请求中不同候选之间的关系，使预测不仅依赖单个 user–item pair，也能够考虑同批候选之间的竞争和相对偏好。

不过，OneRank 与 MDL 最重要的差异，不只是任务状态进入得更早，而是它进一步设计了显式的 **Task–Task interaction**。MDL 中各 Task token 会分别读取 Feature states，但不同 Task token 之间没有独立的 mixing；任务关系主要通过共享 Feature backbone 间接形成。OneRank 则利用可配置的 attention mask 控制任务之间的信息可见性。对于相对独立的任务，可以限制相互读取；对于存在漏斗关系的任务，则可以构造 Click、Add-to-Cart、Order 之间的定向信息传递。任务之间“是否共享”不再完全由一个统一的共享主干隐式决定，而成为可以显式设计的信息流。

更进一步，OneRank 将**前向知识传递**与**反向梯度更新**区分开来。一个任务在前向过程中读取另一个任务的状态，并不意味着它的损失也应该反向修改对方的私有通道。为此，OneRank 在 cross-task attention 中使用 gradient detachment：任务可以读取其他任务提供的信息，但跨任务梯度不会沿同一路径反向进入对方的私有参数。论文将这种机制解释为一种 read-only knowledge transfer——前向允许借用知识，反向则维持任务优化边界。

这正好补充了 MDL 的一个结构边界。MDL 通过不同 Task token 对共享 Feature states 进行差异化读取，主要缓解的是前向中的共享表示瓶颈；但所有任务损失仍然会通过 attention 的 Key/Value 路径更新 Feature backbone，模型没有显式的跨任务梯度隔离。OneRank 则进一步追问：当任务已经拥有专属表示后，哪些信息应当共享，哪些梯度又不应跨越任务边界？

在输出端，OneRank 也没有重新回到传统的静态任务 MLP，而是使用 task-aware global representation 与候选表示进行动态匹配。这样，任务专属信息选择、候选集合建模、跨任务交互和最终打分能够保持在相对统一的 Transformer 表示空间中，而不是在编码结束后突然切换到一套固定的 feed-forward prediction tower。

OneRank 在 Shopee 主排序场景进行了 7 天线上 A/B test，实验组和对照组分别使用 10% 流量。论文报告 GMV/UU 提升 1.01%、Paid Orders/UU 提升 1.17%、广告收入提升 0.81%，同时 Bad Query Rate 下降 2.29%。这些结果说明，将多任务推理和任务优化边界共同纳入 Transformer，并不只是结构上的统一，也能够在工业排序链路中落地。

需要注意的是，OneRank 并不是 MDL 的直接升级版本。OneRank 更集中于多任务排序，没有 MDL 中显式的 Scenario token、Global Scenario token 与实例级场景选择；其论文实验也没有将 MDL 作为直接对照。因此，更准确的理解是：

> **MDL 让 Scenario 与 Task 逐层读取共享 Feature capacity；OneRank 则沿任务轴继续推进，进一步显式建模 Task–Task 关系，并将前向知识共享与反向梯度干扰分开处理。**

### 11.2 MTFM：当不同场景的输入空间也无法对齐

OneRank 延伸的是任务轴，MTFM 关注的则是另一个更偏场景和系统的问题：不同业务场景不仅数据分布不同，其 feature schema 本身也可能不一致。

传统多场景模型通常需要先把各场景整理成统一输入模板，再在共享参数与场景专属参数之间进行拆分。对于字段差异较小的场景，这种方式仍然可行；但在餐厅推荐、菜品推荐、券包推荐等差异较大的业务中，一个场景可能拥有另一个场景根本不存在的字段、序列和上下文。强行对齐意味着增加大量 padding、舍弃异构字段，或者持续维护越来越复杂的统一特征模板。MTFM 将这一范式概括为“harmonize-then-decompose”，并试图绕开输入预对齐。

MTFM 的核心做法是 **heterogeneous tokenization**。模型将历史行为表示为 H-token，将实时行为表示为 R-token，将不同场景中的曝光候选表示为 T-token。不同历史序列、实时序列和业务场景可以拥有各自的 tokenizer 与原始字段，只需要最终投影到相同的隐藏维度，再组成一条可变长度的异构 token 序列。这样，统一发生在 latent representation 层，而不是要求所有场景在原始 feature schema 上完全一致。

这里的 tokenization 与 MDL 具有不同目的。MDL 将 Feature、Scenario 和 Task 转换为职责明确的三类状态，重点是控制它们之间的有向信息流；MTFM 的 H/R/T tokens 主要按照数据来源与时间角色组织，重点是让不同场景的异构数据能够进入一个统一 Transformer。MDL 更关心“场景和任务怎样读取主干”，MTFM 更关心“不同场景的数据怎样进入同一个主干”。

异构数据被放进统一序列后，新的问题是计算成本。多场景行为与候选聚合会显著增加序列长度，若每一层都执行 Full Attention，计算量和显存开销将快速增长。MTFM 因此采用 Hybrid Target Attention：每个 Block 保留少量 Full Attention 层捕获全局依赖，而多数层只使用更高效的 Target Attention 更新目标候选 token；同时结合 Grouped-Query Attention，减少 Key/Value heads 带来的存储和计算开销。

MTFM 还将数据组织、模型结构和执行系统放在一起设计。训练时，模型按用户聚合多个场景中的曝光行为，减少重复的用户上下文编码；系统侧则结合 kernel fusion、CPU–GPU pipeline 和动态稀疏计算，提高训练与推理吞吐。论文在三个美团场景上观察到随模型 GFLOPs 和多场景训练数据增加而持续改善的趋势，并在两个线上场景报告订单量分别提升 2.98% 和 1.45%，推理延迟分别下降 5 ms 和 6 ms。

MTFM 同样不能被简单理解为 MDL 的替代。它没有像 MDL 那样维护独立的 Scenario/Task 跨层状态，也没有像 OneRank 那样重点处理 Task–Task interaction 和梯度边界；其任务预测端仍然连接场景对应的多任务模块。它更重要的贡献，是把统一多场景建模从“统一模型参数”推进到“统一异构数据接口”，并同时考虑这种统一是否能够在工业系统中高效扩展。

> **MDL 解决的是条件信息如何深入主干；MTFM 解决的是不同场景连输入空间都不一致时，怎样仍然共享一个可扩展主干。**

### 11.3 从扩大共享主干，到重新设计信息与梯度边界

将三篇论文放在一起看，可以看到近期工业推荐架构正在发生一个更深层的变化。

MDL 关心的是 **条件信息能够读取哪些模型容量**：Scenario 与 Task 不再只停留在末端，而是逐层读取 Feature states。OneRank 关心的是 **任务之间如何交换信息，以及梯度能够跨越哪些边界**：任务可以在前向中共享知识，同时通过 gradient detachment 保护各自的优化通道。MTFM 关心的是 **什么样的数据能够进入同一个模型，以及统一计算怎样保持可扩展性**：不同场景不必预先对齐全部字段，而是通过异构 tokenization 和稀疏注意力进入共享 Transformer。

因此，最新趋势已经不只是把推荐主干做得更深、更宽，而是在同时重新设计三个问题：

1. 不同信息以什么 token 或状态进入模型；
2. 哪些状态能够读取或影响其他状态；
3. 哪些梯度可以在任务与场景之间传播。

> **大模型化的工业推荐，不只是扩大共享容量，而是让信息流、任务关系和计算边界都能够随模型规模一起扩展。**
