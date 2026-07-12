## 结论

**通常应该共用。**

更准确地说：

> 只要 target item 中的 ID 和行为序列中的 ID，表示的是**同一种实体、使用同一套 ID 空间**，就应该默认共用同一个 embedding 表。

例如：

```text
target_item_id = 103
history_item_ids = [25, 103, 87]
```

这里的 `103` 无论出现在目标位置还是历史位置，都是同一件商品，因此应满足：

[
e_{\text{target}}(103)=e_{\text{history}}(103)=E_{\text{item}}[103]
]

实现上就是：

```python
item_embedding = nn.Embedding(num_items, dim, sparse=True)

target_emb = item_embedding(target_item_id)
history_emb = item_embedding(history_item_ids)
```

SASRec 明确共享输入和预测层的 item embedding；BERT4Rec 同样使用共享的输入、输出 item embedding 矩阵。NVIDIA 的 SIM 实现也明确指出，target item、短期行为序列和长期行为序列中的 item 特征共享 embedding 表。([arXiv][1])

---

## 为什么默认要共享？

### 1. 同一个商品应具有同一个基础表示

假设历史序列为：

```text
历史：[手机壳、充电器、耳机]
目标：耳机
```

DIN 需要计算目标耳机和历史行为之间的相关性：

[
a_i=f(e_{\text{target}},e_{\text{history},i})
]

如果目标耳机来自表 (E_t)，历史耳机来自另一张表 (E_h)：

[
E_t[\text{耳机}] \ne E_h[\text{耳机}]
]

那么模型必须额外学习如何将两个 embedding 空间对齐。

共用表时，同一个商品天然位于同一个空间：

```text
target 耳机 ─┐
             ├── E_item[耳机]
history 耳机 ┘
```

这尤其适合 DIN、DIEN、BST 等 target-aware 模型，因为这些模型要用 target item 对行为序列进行匹配或注意力计算。DIN 的核心就是计算候选商品和历史商品的相关性；DIEN 的兴趣演化同样以目标商品为条件。([GitHub][2])

### 2. 两类样本共同训练同一件商品

假设商品 103 在训练数据中：

```text
作为历史行为出现：10,000 次
作为 target 出现：500 次
```

共享 embedding 后，商品 103 的向量可以同时收到两类梯度：

[
\nabla E[103]
=============

\nabla_{\text{history}} E[103]
+
\nabla_{\text{target}} E[103]
]

这通常能提高低频 target item 的训练充分度。

从你上一问的 sparse update 角度看，同一次迭代中可以把 target 和 sequence 访问的 ID 合并：

```text
target IDs:  [3, 8]
history IDs: [1, 3, 5, 8]

实际更新行：
[1, 3, 5, 8]
```

其中 ID 3 和 ID 8 的不同来源梯度在更新前进行聚合。分片训练时，同一个 item ID 也只需要对应一个 shard owner。

### 3. 节省 embedding 参数和优化器状态

如果有：

```text
1 亿件商品
embedding_dim = 64
FP32
```

一张表约为：

[
10^8 \times 64 \times 4 \approx 25.6\text{ GB}
]

target 和 history 分开建表就会变成约 51.2 GB。若使用 Adam，还要额外维护一阶、二阶状态，内存差异更大。

---

# 不只是 item_id：同名属性也通常成对共享

例如目标商品有：

```text
target_item_id
target_category_id
target_brand_id
target_shop_id
```

序列中每个历史商品有：

```text
hist_item_id
hist_category_id
hist_brand_id
hist_shop_id
```

通常建立：

```python
item_table     # target_item_id  <-> hist_item_id
category_table # target_category <-> hist_category
brand_table    # target_brand    <-> hist_brand
shop_table     # target_shop     <-> hist_shop
```

即：

| Target 特征            | Sequence 特征        | 是否共享 |
| -------------------- | ------------------ | ---: |
| `target_item_id`     | `hist_item_id`     |    是 |
| `target_category_id` | `hist_category_id` |    是 |
| `target_brand_id`    | `hist_brand_id`    |    是 |
| `target_shop_id`     | `hist_shop_id`     |    是 |
| `target_item_id`     | `hist_category_id` |    否 |

最后一行不能共享。虽然二者底层可能都是整数，例如都出现数字 `23`，但：

```text
item_id = 23
category_id = 23
```

表示不同类型的实体，必须属于不同 vocabulary 和 embedding 表。

阿里 EasyRec 的序列组件也要求 target item 和 history sequence 的对应子特征按相同次序配对，这反映的正是 item、category、brand 等同类型属性之间的对应关系。([GitHub][3])

---

# 共享基础表，不等于后续处理必须完全相同

实践中最稳妥的设计通常是：

```text
共享基础 ID embedding
        ↓
target 和 history 使用不同的角色变换
```

例如：

[
e_t=W_t E[i]
]

[
e_h=W_h E[i]
]

代码可以是：

```python
base_target = item_embedding(target_item_id)
base_history = item_embedding(history_item_ids)

target_emb = target_projection(base_target)
history_emb = history_projection(base_history)
```

或者加入角色 embedding：

```python
target_emb = item_embedding(target_id) + target_role_embedding
history_emb = item_embedding(history_ids) + history_role_embedding
```

这样同时保留：

1. 同一个商品具有统一的基础语义；
2. target 是“待判断对象”，history 是“用户已发生行为”，二者角色不同。

一个简单类比是 NLP：

```text
同一个词使用同一个 token embedding
+
通过 position/type embedding 表示它出现在什么位置、承担什么角色
```

但在推荐系统中，通常不必一上来就加复杂角色变换。最小基线应先使用共享表，观察是否确有角色冲突，再决定是否解耦。

---

# 哪些情况下不应该直接共享？

## 1. ID 空间实际上不同

例如：

```text
target_item_id：广告创意 ID
history_item_id：商品 SPU ID
```

即使字段都叫 item ID，也不是同一种实体，不能共享。

又例如跨域系统：

```text
domain A 的 item_id=100：一双鞋
domain B 的 item_id=100：一首歌
```

如果未先构造全局唯一 ID，就不能共用 embedding 表。

正确判断不是看字段名字，而是看：

[
\text{ID}=i
]

在两个字段中是否始终指向同一个实体。

---

## 2. Target 和 history 使用不同粒度

例如：

```text
target：SKU ID
history：SPU ID
```

一件衣服可能：

```text
SPU 100：某款 T 恤
SKU 10001：白色 M
SKU 10002：黑色 L
```

SKU 和 SPU 不是一一相同的 ID 空间，不能直接共表。

可以采用：

```text
target SKU embedding
target 对应的 SPU embedding
history SPU embedding
```

其中 target 的 SPU 与 history SPU 可以共享，但 SKU 表独立。

---

## 3. 特意使用双空间或双塔结构

某些模型会有意区分：

```text
query/target embedding space
behavior/candidate embedding space
```

例如：

[
q_i=E_q[i],\qquad k_i=E_k[i]
]

这种设计相当于 Transformer 中不同的 Query/Key 投影。它能增加表达能力，但也会：

* 增加参数量；
* 降低低频 ID 的数据复用；
* 使两个空间的对齐完全依赖训练；
* 增加分布式 embedding 的存储和通信复杂度。

因此它通常是有实验动机的结构选择，而不是默认选择。近期一些长序列模型也专门研究了将 attention embedding 与 representation embedding 解耦，说明“不共享”是可以成立的，但需要明确解决某种共享造成的优化冲突。([arXiv][4])

---

## 4. 目标 ID 与历史 ID 的训练任务严重冲突

例如一个 embedding 同时承担：

```text
历史序列编码
召回候选打分
排序特征表示
辅助分类任务
```

不同损失可能对同一向量提出冲突要求。

此时不一定要立刻复制完整 embedding 表，可以从简单到复杂依次测试：

```text
方案 1：完全共享
方案 2：共享表 + 不同 projection
方案 3：共享主 embedding + 各任务 residual embedding
方案 4：完全独立 embedding 表
```

通常方案 2 或方案 3 已经能提供角色差异，而不必承担两份超大表的成本。

---

# 一个容易混淆的点：共享不会自动造成标签泄漏

假设 next-item 训练样本是：

```text
history = [A, B, C]
target = D
```

历史和目标都查同一张 embedding 表：

```text
E[A], E[B], E[C], E[D]
```

这本身不是泄漏。共享的是模型参数，而不是把 `D` 填进历史输入。

真正的泄漏是：

```text
history = [A, B, C, D]
target = D
```

或者在构造 target 前的序列特征时，使用了目标行为发生之后才能知道的信息。

BERT4Rec 之所以采用掩码任务，正是为了避免被预测 item 直接出现在可见上下文中，而不是因为共享 embedding 表存在问题。([arXiv][5])

---

# 推荐的默认配置

对于典型的 DIN、DIEN、BST、SASRec、Transformer 序列模型，我会采用：

```python
class ItemEmbeddingModule(nn.Module):
    def __init__(self, num_items: int, dim: int):
        super().__init__()
        self.item_embedding = nn.Embedding(
            num_items,
            dim,
            padding_idx=0,
            sparse=True,
        )

    def forward(self, target_item_id, history_item_ids):
        target_emb = self.item_embedding(target_item_id)
        history_emb = self.item_embedding(history_item_ids)
        return target_emb, history_emb
```

如果需要角色差异，再加入轻量变换：

```python
self.target_proj = nn.Linear(dim, dim, bias=False)
self.history_proj = nn.Linear(dim, dim, bias=False)
```

即：

```text
同一实体、同一 ID 空间
    → 共享 embedding 表

相同实体但承担不同角色
    → 共享基础表 + 不同 projection/role embedding

不同实体或不同 ID 空间
    → 不同 embedding 表
```

一句话概括：

> **Embedding 表应按照“实体语义和 ID namespace”划分，而不是按照“它来自 target 字段还是 sequence 字段”划分。**

[1]: https://arxiv.org/pdf/1808.09781?utm_source=chatgpt.com "Self-Attentive Sequential Recommendation"
[2]: https://github.com/shenweichen/DeepCTR/blob/master/docs/source/Features.md?utm_source=chatgpt.com "DeepCTR/docs/source/Features.md at master"
[3]: https://github.com/alibaba/EasyRec/blob/master/docs/source/component/sequence.md?utm_source=chatgpt.com "EasyRec/docs/source/component/sequence.md at master"
[4]: https://arxiv.org/html/2410.02604v3?utm_source=chatgpt.com "Long-Sequence Recommendation Models Need ..."
[5]: https://arxiv.org/abs/1904.06690?utm_source=chatgpt.com "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer"
