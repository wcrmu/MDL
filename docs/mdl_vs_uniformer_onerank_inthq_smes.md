# IntHQ、SMES、UniFormer、OneRank 与 MDL 对比

## 1. 总体结论

这五个方法都面向**多任务推荐 / 多场景多任务排序**，但解决的问题并不完全相同。

可以先用一句话区分：

* **MDL**：重点解决**多场景 + 多任务信息如何逐层交互和融合**。
* **UniFormer**：重点解决**共享 Feature 表征与 Task-aware 表征如何统一建模**。
* **OneRank**：重点解决**任务之间存在依赖关系时，如何显式控制任务间信息流动**。
* **IntHQ**：重点解决**任务表示如何从不同 Backbone 深度获取信息**，强调跨层信息聚合。
* **SMES**：重点解决**多任务模型扩展后参数量和计算量过大**，通过稀疏专家提升容量。

因此可以把它们理解为解决五个不同层面的问题：

```text
MDL
↓
场景和任务如何共同建模

UniFormer
↓
Feature Space 和 Task Space 如何统一

OneRank
↓
不同任务之间应该如何交互

IntHQ
↓
一个任务应该从哪些网络层获取信息

SMES
↓
任务越来越多时，模型容量如何低成本扩展
```

---

# 2. 核心结构对比

| 方法        | 核心问题                            | Task 信息进入位置                   | Task 间交互              | 跨层融合     | Expert / MoE | 多场景 |
| --------- | ------------------------------- | ----------------------------- | --------------------- | -------- | ------------ | --- |
| MDL       | 多场景多任务联合建模                      | Backbone 早期，以 Task Token 形式进入 | 有                     | 隐式逐层更新   | 非核心          | 强   |
| UniFormer | Feature-space → Task-space 统一建模 | Feature 编码后                   | 有，Task Self-Attention | 非核心      | per-task FFN | 可支持 |
| OneRank   | 显式建模任务依赖                        | Backbone 内 Task Token         | 有，带 Mask              | 可结合深层表示  | 非核心          | 可扩展 |
| IntHQ     | 不同任务需要不同深度信息                    | Task Query                    | 有限/间接                 | **核心模块** | 非核心          | 可扩展 |
| SMES      | 多任务模型扩容成本                       | Expert Router                 | 通过共享/稀疏 Expert        | 非核心      | **核心**       | 可支持 |

最本质的区别可以写成：

```text
MDL      = Scene × Task × Layer
UniFormer = Feature Space → Task Space
OneRank   = Task Dependency
IntHQ     = Task × Layer
SMES      = Task × Expert
```

---

# 3. MDL

## 3.1 MDL 要解决什么问题

传统多任务模型通常只考虑：

```text
Input Feature
    ↓
Shared Backbone
    ↓
Task Heads
```

但真实推荐系统往往同时存在：

```text
多个场景
×
多个任务
```

例如：

```text
场景：
单列
双列
内搜

任务：
Click
Like
Favorite
```

MDL 的核心思想是：

> 不要等 Backbone 最后一层才区分任务和场景，而是让 **Task Token 和 Scene Token 从较早阶段就参与网络计算**。

---

## 3.2 核心结构

可以简化为：

```text
Feature Tokens
Task Tokens
Scene Tokens
     │
     ▼
┌───────────────┐
│   MDL Block   │
└───────────────┘
     │
     ▼
┌───────────────┐
│   MDL Block   │
└───────────────┘
     │
    ...
     │
     ▼
Task Representation
     │
     ▼
Task Head
```

一个 MDL Block 中主要包含：

```text
Feature
  │
  ├── Task Attention
  │      Q = Task Token
  │      K,V = Feature
  │
  └── Scene Attention
         Q = Scene Token
         K,V = Feature
```

因此 Task Token 和 Scene Token 都可以主动从 Feature 中提取信息。

---

## 3.3 Task Attention

对于任务 \(t\)：

```text
Task Token_t
     │ Q
     ▼
Attention
     ▲
     │ K,V
Feature Tokens
```

即：

$$
T_t' = \mathrm{Attention}(T_t, X, X)
$$

含义：

> 每个任务主动寻找“对自己最有用的 Feature”。

例如：

```text
Click Token
→ 更关注曝光、兴趣、视觉等信息

Favorite Token
→ 更关注长期兴趣

Purchase Token
→ 更关注价格、意向等信息
```

---

## 3.4 Scene Attention

同样：

$$
S_s' = \mathrm{Attention}(S_s, X, X)
$$

不同场景可以从共享特征中抽取不同的信息。

因此：

```text
Feature
 ├── 被 Task Token 查询
 └── 被 Scene Token 查询
```

---

## 3.5 MDL 最大特点

MDL 的 Task/Scene Representation 是**逐层更新的**：

```text
T^(0)
 ↓
Block 1
 ↓
T^(1)
 ↓
Block 2
 ↓
T^(2)
 ↓
...
 ↓
T^(L)
```

所以任务信息不是最后才加入，而是在整个网络深度中不断参与表示学习。

---

# 4. UniFormer

## 4.1 UniFormer 要解决什么问题

很多多任务模型存在两种极端：

### 完全共享

```text
Feature
  ↓
Shared Backbone
  ↓
Task Heads
```

优点：

```text
参数少
知识共享充分
```

问题：

```text
不同任务差异无法充分表达
```

---

### 完全 Task-specific

```text
Task A Backbone
Task B Backbone
Task C Backbone
```

优点：

```text
任务个性化强
```

问题：

```text
参数量大
无法有效共享
```

UniFormer 希望同时兼顾：

```text
共享 Feature Representation
+
Task-specific Representation
```

---

## 4.2 UniFormer 的整体结构

可以理解成两个阶段：

```text
Feature Space
    │
    ▼
Task Cross-Attention
    │
    ▼
Task Space
    │
    ▼
Task Self-Attention
    │
    ▼
per-task FFN
    │
    ▼
Task Heads
```

核心变化发生在：

```text
Feature Space
        ↓
Task Tokens Query Features
        ↓
Task Space
```

---

## 4.3 Feature → Task

Task Token 作为 Query：

```text
Task Tokens ──Q──┐
                 │
                 ▼
           Cross-Attention
                 ▲
                 │
Feature Tokens ─K,V
```

得到：

```text
Task-aware Features
```

即每个 Task Token 获得与自身任务相关的信息。

---

## 4.4 Task 间 Self-Attention

之后：

```text
Task A ─┐
Task B ─┼─ Self-Attention
Task C ─┘
```

不同任务之间进行信息交换。

例如：

```text
Click
 ↓
Cart
 ↓
Purchase
```

Purchase 的预测可以利用 Click / Cart 中的信息。

---

## 4.5 per-task FFN

Self-Attention 后：

```text
Task A → FFN_A
Task B → FFN_B
Task C → FFN_C
```

即：

```text
Attention 负责共享
FFN 负责个性化
```

这是 UniFormer 非常重要的设计。

---

# 5. OneRank

## 5.1 OneRank 要解决什么问题

UniFormer 中：

```text
Task A
Task B
Task C
   │
Self-Attention
```

如果完全自由交互，可能产生：

```text
不合理任务信息泄漏
负迁移
优化冲突
```

OneRank 更强调：

> **任务之间的关系应该显式建模，而不是全部自由通信。**

---

## 5.2 核心思想

可以抽象为：

```text
Task-specific Encoding
        │
        ▼
Task Tokens
        │
        ▼
Masked Task Attention
        │
        ▼
Task Representation
```

---

## 5.3 Task-specific Transformer

底层阶段：

```text
Task A Token → Task A Path
Task B Token → Task B Path
Task C Token → Task C Path
```

不同任务先形成自己的表示。

关键思想：

> 底层不要过早让任务彼此污染。

例如：

```text
Task A Token ✕ Task B Token
Task A Token ✕ Task C Token
```

它们可以共享 Feature，但 Task Token 之间先保持隔离。

---

## 5.4 跨任务关系 Attention

之后再进行：

```text
Task A
Task B
Task C
   │
   ▼
Task Relation Attention
```

但这里通常不是完全自由 Self-Attention，而是使用 **Mask** 控制任务之间的信息流。

例如业务关系：

```text
Click → Cart → Purchase
```

可以设置：

```text
Purchase 可以看 Cart
Purchase 可以看 Click

Click 不看 Purchase
```

Mask 类似：

```text
          K
        C   A   P
Q  C    ✓   ✕   ✕
   A    ✓   ✓   ✕
   P    ✓   ✓   ✓
```

因此任务关系是：

```text
有方向
有约束
可解释
```

---

## 5.5 Gradient Detachment

OneRank 还会考虑一个问题：

即使 Forward 中限制了任务交互，Backward 梯度仍可能造成任务之间相互影响。

因此可以使用：

```text
Gradient Detachment
```

例如：

```text
Click Representation
        │
        ├────────→ Purchase
        │
       stop-gradient
```

Purchase 可以利用 Click 表示：

```text
Forward：可以
Backward：不反向修改 Click
```

这样可以减轻任务梯度冲突。

---

# 6. IntHQ

## 6.1 IntHQ 要解决什么问题

普通 Transformer 通常：

```text
Layer 1
 ↓
Layer 2
 ↓
Layer 3
 ↓
Layer 4
 ↓
Final Layer
 ↓
Task Head
```

最终 Task Head 主要使用：

```text
h^(L)
```

但不同深度的表示具有不同含义：

```text
浅层
→ 基础 Feature Interaction

中层
→ 中阶组合关系

深层
→ 高阶语义 / Task-specific Representation
```

不同任务未必都最适合使用最终层。

因此 IntHQ 的核心问题是：

> **不同任务应该如何从不同网络深度选择信息？**

---

## 6.2 核心结构

保存 Backbone 的不同层：

```text
h^(1)
h^(2)
h^(3)
...
h^(L)
```

然后通过 Task Query：

```text
        h1
        h2
Task Q ─h3── Attention
        ...
        hL
```

得到：

$$
z_t =
\sum_l \alpha_{t,l} h^{(l)}
$$

其中：

$$
\alpha_{t,l}
=
\operatorname{softmax}
(q_t^\top h^{(l)})
$$

也就是说：

> 每个任务学习自己应该更关注哪一层。

---

## 6.3 与普通残差连接的区别

普通 Residual：

```text
h^(l+1) = F(h^l) + h^l
```

虽然理论上深层已经包含浅层信息，但这种信息是：

```text
被动传播
```

而 IntHQ 是：

```text
Task Query 主动选择不同层
```

即：

```text
Residual：
Layer → Layer

IntHQ：
Task → Layer
```

这是两件不同的事。

---

## 6.4 一个直观例子

假设任务：

```text
CTR
CVR
GMV
```

可能学出：

```text
CTR:
Layer2  0.45
Layer3  0.35
Layer4  0.20

CVR:
Layer2  0.10
Layer3  0.35
Layer4  0.55

GMV:
Layer2  0.05
Layer3  0.20
Layer4  0.75
```

表示不同任务对网络深度需求不同。

---

# 7. SMES

## 7.1 SMES 与前面四个方法关注点不同

前面四篇主要在研究：

```text
Task 怎么获取信息
Task 怎么交互
Task 怎么跨层读取
Scene 怎么参与
```

SMES 更关注：

> **任务数不断增加时，模型参数量和计算量如何保持可控？**

---

## 7.2 普通多任务 Expert 问题

例如：

```text
Task A → Expert A
Task B → Expert B
Task C → Expert C
...
```

任务越来越多：

```text
Expert 数量 ↑
参数量 ↑
计算量 ↑
```

---

## 7.3 SMES 思路

建立一个大 Expert Pool：

```text
Expert 1
Expert 2
Expert 3
...
Expert N
```

但每个样本 / Task 只激活其中少数 Expert：

```text
Router
  │
  ├── Expert 2
  └── Expert 7
```

而不是：

```text
全部 Expert 同时计算
```

因此形成：

```text
大模型容量
+
小计算成本
```

---

## 7.4 与 MMoE 的区别

MMoE：

```text
Input
 │
 ├── Expert1 ─┐
 ├── Expert2 ─┼→ Task Gate → Task
 ├── Expert3 ─┤
 └── Expert4 ─┘
```

通常所有 Expert 都要先计算。

SMES 更强调：

```text
Router
 ↓
只选择 Top-K Expert
```

即：

```text
Dense Expert
→
Sparse Expert
```

核心目标偏向：

```text
Scalability
```

而不只是任务信息交互。

---

# 8. MDL vs UniFormer

这两个最容易混淆。

## 相同点

都有：

```text
Task Token
+
Attention
+
Task-specific Representation
```

Task Token 都会主动从共享信息中获取任务相关表示。

---

## 最大区别

### MDL

Task Token 从早期开始反复参与：

```text
Feature
 +
Task
 +
Scene
 ↓
Block 1
 ↓
Feature / Task / Scene
 ↓
Block 2
 ↓
...
```

因此：

```text
Task Representation
```

是在 Backbone 中逐层演化的。

---

### UniFormer

更强调两个空间之间的转换：

```text
Feature Space
      ↓
Cross-Attention
      ↓
Task Space
      ↓
Self-Attention
```

结构更加明确地拆成：

```text
先 Feature 建模
再 Task 建模
```

---

可以概括为：

```text
MDL：
Task / Scene 与 Feature 一起逐层演化

UniFormer：
先 Feature Representation
再转换为 Task Representation
```

---

# 9. UniFormer vs OneRank

二者都存在：

```text
Task Tokens
    ↓
Task Interaction
```

但核心区别是：

```text
UniFormer
=
任务之间自由交流

OneRank
=
任务之间受控交流
```

UniFormer：

```text
Task A ↔ Task B
Task A ↔ Task C
Task B ↔ Task C
```

OneRank：

```text
Task A → Task B
Task B → Task C
Task A → Task C
```

使用 Mask 显式规定业务依赖关系。

因此：

```text
UniFormer
更强调统一建模

OneRank
更强调任务关系建模
```

---

# 10. OneRank vs IntHQ

两者都在使用 Task Representation 主动获取信息，但查询对象完全不同。

## OneRank

主要查询：

```text
其他 Task
```

即：

```text
Task → Task
```

解决：

```text
任务之间有什么依赖？
```

---

## IntHQ

主要查询：

```text
不同 Backbone Layer
```

即：

```text
Task → Layer
```

解决：

```text
一个任务更需要哪一层的信息？
```

因此：

```text
OneRank = 横向 Task Interaction

IntHQ = 纵向 Layer Interaction
```

这是二者最核心的区别。

---

# 11. IntHQ vs MDL

两者都涉及：

```text
Layer
+
Task
```

但方向相反。

### MDL

Task Representation 会逐层传播：

```text
T0
 ↓
T1
 ↓
T2
 ↓
T3
```

任务和 Feature 一起向深层走。

---

### IntHQ

保留不同层：

```text
h1
h2
h3
h4
```

最后：

```text
Task Query
   ↓
选择不同 Layer
```

即：

```text
MDL：
Layer-by-layer Task Evolution

IntHQ：
Cross-layer Task Retrieval
```

---

# 12. SMES vs 其他方法

SMES 与其他四篇不是直接替代关系。

例如完全可以组合：

```text
MDL
+
SMES
```

变成：

```text
Scene / Task Token
       ↓
MDL Blocks
       ↓
Sparse Expert
       ↓
Task Head
```

也可以：

```text
OneRank
+
SMES
```

```text
Task-specific Representation
        ↓
Masked Task Interaction
        ↓
Sparse Expert
        ↓
Prediction
```

因为：

```text
MDL / UniFormer / OneRank / IntHQ
主要解决 Representation / Interaction

SMES
主要解决 Capacity / Scalability
```

---

# 13. 从「信息从哪里来」理解五个模型

这是最容易记的方法。

## MDL

Task 信息来自：

```text
Feature
+
Scene
+
不同层的持续交互
```

核心：

```text
Task ← Feature
Scene ← Feature
```

---

## UniFormer

Task 信息来自：

```text
Feature
+
其他 Task
```

核心：

```text
Feature → Task
Task ↔ Task
```

---

## OneRank

Task 信息来自：

```text
Feature
+
允许依赖的其他 Task
```

核心：

```text
Feature → Task
Task → Task
```

但 Task→Task 受 Mask 控制。

---

## IntHQ

Task 信息来自：

```text
不同深度 Layer
```

核心：

```text
Layer1 ┐
Layer2 ├→ Task
Layer3 ┤
Layer4 ┘
```

---

## SMES

Task 信息来自：

```text
不同 Expert
```

核心：

```text
Expert1 ┐
Expert2 ├→ Router → Task
Expert3 ┤
Expert4 ┘
```

---

# 14. 最统一的视角

可以把一个多场景多任务模型拆成四个维度：

```text
Scene
Task
Layer
Expert
```

这五个模型分别重点解决：

```text
                 主要关注

MDL        Scene × Task × Layer

UniFormer      Feature × Task

OneRank          Task × Task

IntHQ            Task × Layer

SMES             Task × Expert
```

因此它们实际上具有很强的互补性。

一个更完整的统一模型甚至可以设计成：

```text
Feature Tokens
Scene Tokens
Task Tokens
      │
      ▼
Scene-aware / Task-aware Backbone
      │
      ├─────────────┐
      │             │
      ▼             ▼
Layer 1           Task Token
Layer 2              │
Layer 3              ▼
Layer 4        Masked Task Attention
      │             │
      └──────┬──────┘
             ▼
      Cross-Layer Attention
             │
             ▼
        Sparse Experts
             │
             ▼
         Task Heads
```

对应：

```text
MDL
→ Scene / Task Token

UniFormer
→ Feature-to-Task Transformation

OneRank
→ Masked Task Interaction

IntHQ
→ Cross-Layer Aggregation

SMES
→ Sparse Expert Capacity
```

---

# 15. 如果用于实际多场景多任务排序，怎么选择

## 只有普通多任务问题

例如：

```text
CTR
CVR
CTCVR
```

可以优先考虑：

```text
UniFormer / OneRank
```

如果任务存在明确业务依赖：

```text
Click → Cart → Purchase
```

则 OneRank 更自然。

---

## 多场景 + 多任务

例如：

```text
首页
搜索
推荐

×

CTR
CVR
GMV
```

MDL 更适合，因为：

```text
Scene
+
Task
```

是模型显式建模对象。

---

## 不同任务明显依赖不同网络深度

例如：

```text
CTR 偏浅层
Purchase 偏深层
```

可以加入 IntHQ 的：

```text
Cross-Layer Task Query
```

---

## Task 数量特别多

例如几十个甚至上百个任务：

```text
Task 1
Task 2
...
Task 100
```

则需要考虑 SMES 这种：

```text
Sparse Expert
```

否则 Task-specific 参数量很容易爆炸。

---

# 16. 最后速记

只记下面五句话即可：

```text
MDL
= Task 和 Scene 从 Backbone 内部逐层参与建模。

UniFormer
= 先学 Feature，再通过 Task Token 把 Feature Space 转到 Task Space。

OneRank
= Task 之间不是随便交流，而是根据任务依赖用 Mask 控制信息流。

IntHQ
= 每个 Task 主动从不同 Backbone 深度选择自己需要的信息。

SMES
= 用 Sparse Expert 解决任务规模增大后的模型容量和计算成本问题。
```

进一步压缩成：

```text
MDL        ：Task × Scene
UniFormer  ：Feature → Task
OneRank    ：Task → Task
IntHQ      ：Task → Layer
SMES       ：Task → Expert
```
