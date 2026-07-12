这里要修正一个表述：

> **不是“推荐系统用了 sparse 更新，就完全不能用 DDP”**，而是不能把整个含 `sparse=True Embedding` 的模型直接交给 **NCCL-DDP，期待 NCCL 原生对稀疏梯度做 AllReduce**。

PyTorch 的 DDP Reducer 内部确实存在处理 sparse gradient 的分支，但 `ProcessGroupNCCL` 的原生 sparse AllReduce 仍不是常规支持路径；PyTorch 官方仓库对应的 NCCL sparse-allreduce issue 目前仍处于开放状态。Gloo 曾实现 sparse AllReduce，但性能和跨机通信能力通常不适合大型 GPU 推荐训练。([GitHub][1])

工业上的主流答案是：

## Embedding 不走 DDP，Dense 部分继续走 DDP

把模型拆成两部分：

```text
Sparse 部分：Embedding tables
    → 模型并行 / 参数服务器 / 分片更新

Dense 部分：MLP、Attention、CrossNet、Tower
    → DDP + NCCL AllReduce
```

也就是一种 **Hybrid Parallelism**：

```text
embedding model parallel
        +
dense data parallel
```

PyTorch 官方的混合 DDP + RPC 教程也是这个结构：大 embedding 放到参数服务器，FC 层复制到各 trainer 上并使用 DDP。([PyTorch 文档][2])

---

# 1. GPU 推荐系统最主流：Embedding Sharding

Embedding 表不是每张 GPU 各复制一份，而是切开存放。

例如有一个表：

```text
Embedding[10000, 64]
```

两张 GPU 做 row-wise sharding：

```text
GPU 0：row 0    ~ 4999
GPU 1：row 5000 ~ 9999
```

假设本轮输入是：

```text
GPU 0 batch IDs: [3, 7000]
GPU 1 batch IDs: [8, 7000]
```

执行过程是：

```text
1. 按 ID 的 owner 路由

   ID 3    → GPU 0
   ID 8    → GPU 0
   ID 7000 → GPU 1

2. 使用 All-to-All 把 ID 发给对应 owner

3. 每张 GPU 只查询自己持有的 embedding 行

4. 再把 embedding vector 发回原始请求 GPU

5. Dense 网络在每张 GPU 上正常计算

6. 反向传播时，embedding gradient 再路由给 owner

7. owner 对同一个 ID 的梯度求和，然后只更新本地行
```

这里没有进行：

```text
sparse gradient AllReduce
```

而是在进行：

```text
ID / embedding / gradient 的 All-to-All 路由
```

NCCL 完全可以传输这些普通的 dense tensor：

```text
indices: int64 tensor
values:  float tensor
```

所以并不要求 NCCL 理解 `torch.sparse_coo_tensor`。

同时 Dense MLP 仍然执行正常的：

```text
NCCL AllReduce(dense_gradients)
```

这就是大型 GPU 推荐系统最典型的结构。

TorchRec 的 `DistributedModelParallel` 正是为这种情况设计的，支持：

* table-wise
* row-wise
* column-wise
* table-row-wise
* data-parallel

并通过 planner 自动选择 embedding placement。([PyTorch 文档][3])

---

# 2. 不同规模通常怎么选

| 场景               | Embedding 处理                         | Dense 网络 |
| ---------------- | ------------------------------------ | -------- |
| 表很小，例如几十万到几百万参数  | `sparse=False`，直接 Dense DDP          | DDP      |
| 多个中等 embedding 表 | Table-wise sharding                  | DDP      |
| 单张超大 user/item 表 | Row-wise sharding                    | DDP      |
| 表非常大且放不进 GPU     | CPU Parameter Server / CPU embedding | GPU DDP  |
| ID 动态增长、在线增量训练   | Parameter Server / KV store          | DDP      |
| 只是实验原型           | 手工同步 sparse indices/values           | DDP      |

PyTorch 官方 TorchRec 教程将 table-wise 描述为多个中小表常见的分片方式，而 row-wise 用于单设备无法容纳的大表。([PyTorch 文档][3])

---

# 3. 最简单的处理：直接关闭 sparse gradient

假设表并不大：

```python
self.embedding = nn.Embedding(
    num_embeddings=100_000,
    embedding_dim=32,
    sparse=False,
)
```

这样 embedding 的梯度变成普通 dense tensor，可以直接：

```python
model = DistributedDataParallel(model)
```

问题是通信量由：

```text
本轮实际访问的行数 × embedding_dim
```

变成：

```text
整个 embedding 表大小
```

例如：

```text
表：1 亿行 × 64 维 × FP32
大小：约 25.6 GB

本轮只访问 10 万行
实际有效梯度：约 25.6 MB
```

如果 densify，DDP 需要处理接近整个 25.6 GB 的梯度，而不是有效的 25.6 MB，所以大型推荐表不能这样处理。

因此它只适用于：

* 小词表；
* embedding 参数量远小于 dense 网络；
* 单机实验；
* 通信不是瓶颈的情况。

---

# 4. 参数服务器方案

传统工业推荐系统常采用：

```text
GPU Trainer:
    Dense model
    DDP + NCCL

CPU Parameter Server:
    Embedding table
    Sparse pull / sparse push
```

一次训练大致是：

```text
Trainer 把 IDs 发给 PS
PS 返回对应 embedding rows
Trainer 完成 forward/backward
Trainer 把 (ID, gradient) 发回 PS
PS 更新这些行
```

它的优点：

* 表可以远大于单张甚至所有 GPU 显存；
* 只传输本轮访问的行；
* 支持动态 ID 和超大 KV 表；
* optimizer state 也可以放在 CPU 或分布式内存。

代价是：

* RPC 延迟；
* 网络带宽压力；
* 可能出现异步更新和参数陈旧；
* PS 容易成为热点，特别是高频 ID。

PyTorch 官方给出的 DDP + RPC 示例，就是 embedding 放参数服务器、FC 层使用 DDP 的混合模式。([PyTorch 文档][2])

从发展历史看，早期大规模推荐训练更常使用参数服务器；随着 GPU 显存、NVLink、All-to-All 集合通信和 fused embedding kernel 成熟，GPU embedding sharding 逐渐成为 GPU 集群上的主要方案。TorchRec 就是这一类 GPU 分片体系的代表。

---

# 5. 小规模可以手工同步 sparse gradient

还有一种折中方案：每张 GPU 仍然复制完整 embedding 表，但不使用 sparse AllReduce。

每个 rank 得到：

```text
indices = [3, 8, 8]
values  = [g3, g8_a, g8_b]
```

然后：

```text
1. AllGather 每个 rank 的 nnz
2. AllGather / AllToAll indices
3. AllGather / AllToAll values
4. concat
5. 按 ID coalesce
6. 每张 GPU 应用完全相同的更新
```

通信的是普通 tensor：

```python
indices: Tensor[int64]
values: Tensor[float]
```

不是 `SparseTensor`，所以 NCCL 可以处理。

但它通常只适合：

* 表能在每张 GPU 上完整复制；
* world size 较小；
* 每轮访问 ID 数量较少；
* 快速验证算法。

主要问题是，假设有 (P) 个 rank，每个 rank 访问 (K) 行，所有 rank 都需要获得约 (P K) 个梯度项，扩展性通常不如 owner-based sharding。

此外使用 Adam、Adagrad 等带状态的优化器时，必须保证：

```text
所有 rank 对重复 ID 做相同的聚合
所有 rank 对每个 ID 只执行一次 optimizer update
所有 rank 的 optimizer state 保持完全一致
```

否则 embedding 副本会逐渐分叉。

---

# 6. Optimizer 也要拆开

典型写法不是一个 optimizer 管整个模型，而是：

```python
dense_optimizer = torch.optim.AdamW(
    dense_parameters,
    lr=1e-3,
)

embedding_optimizer = SparseOrFusedOptimizer(
    embedding_parameters,
    lr=1e-2,
)
```

普通 `nn.Embedding(..., sparse=True)` 产生的是 sparse gradient；PyTorch 官方文档指出，原生支持这类 sparse gradient 的 optimizer 范围有限，主要包括 SGD、SparseAdam 和 Adagrad。([PyTorch 文档][4])

工业框架通常不会直接依赖普通的 `SparseAdam`，而是使用：

* fused embedding optimizer；
* row-wise Adagrad；
* optimizer-in-backward；
* 每个 shard 本地维护 optimizer state。

例如 row-wise sharding 下：

```text
GPU 0 只保存和更新 row 0~4999 的 optimizer state
GPU 1 只保存和更新 row 5000~9999 的 optimizer state
```

因此 optimizer state 也不需要 AllReduce。

---

# 7. 推荐的项目结构

如果你现在在写 PyTorch 推荐框架，比较合理的结构是：

```python
class RecModel(nn.Module):
    def __init__(self):
        self.sparse_arch = ShardedEmbeddingCollection(...)
        self.dense_arch = DenseTower(...)

    def forward(self, sparse_features, dense_features):
        sparse_emb = self.sparse_arch(sparse_features)
        return self.dense_arch(sparse_emb, dense_features)
```

分布式封装：

```text
sparse_arch
    → TorchRec DistributedModelParallel
      或自定义 embedding sharding

dense_arch
    → DDP / NCCL
```

而不是：

```python
DDP(entire_model_with_sparse_embedding)
```

---

# 最核心的判断

可以压缩成三句话：

1. **小 embedding 表**：关闭 `sparse=True`，整个模型直接 DDP。
2. **GPU 上的大 embedding 表**：embedding 做 table-wise/row-wise model parallel，dense 部分做 DDP。
3. **远超 GPU 容量或动态 ID 表**：embedding 放 parameter server，dense 部分做 DDP。

因此，推荐系统中一般不是解决“如何让 NCCL AllReduce sparse gradient”，而是改变并行结构：

```text
不要同步整个稀疏梯度；
把每一行 embedding 指定唯一 owner，
将访问和梯度路由到 owner 更新。
```

这才是问题结构决定的解法。

[1]: https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/reducer.cpp?utm_source=chatgpt.com "pytorch/torch/csrc/distributed/c10d/reducer.cpp at main"
[2]: https://docs.pytorch.org/tutorials/advanced/rpc_ddp_tutorial.html "Combining Distributed DataParallel with Distributed RPC Framework — PyTorch Tutorials 2.13.0+cu130 documentation"
[3]: https://docs.pytorch.org/tutorials/advanced/sharding.html "Exploring TorchRec sharding — PyTorch Tutorials 2.13.0+cu130 documentation"
[4]: https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.sparse.Embedding.html?utm_source=chatgpt.com "Embedding — PyTorch 2.12 documentation"
