# M-FALCON 与 Kraken GSET/rAdaGrad

本仓库现在提供两条彼此独立的机制：

- M-FALCON 用于 `evaluate`、`predict` 和训练中的 fixed-test evaluation。它只改变推理调度，不改变训练目标或候选顺序。
- GSET 用于在线训练式的动态稀疏表管理；与 Kraken 的 row-wise AdaGrad（rAdaGrad）一起使用。**生产配置默认打开 GSET。**

## M-FALCON

启用方式：

```yaml
runtime:
  mfalcon_microbatch_size: 256
```

也可以在命令行覆盖：

```bash
python -m src.main predict ... --mfalcon-microbatch-size 256
```

正值表示每次处理的候选数；`0` 保持原推理路径。一次候选批会被切成
`ceil(candidate_count / microbatch_size)` 个微批，但 request-only sequence
projection、attention K/V 和可缓存的 request state 只预计算一次。携带
`row_indices` 的请求去重 payload 只切候选到请求的映射，不复制请求 tensor；
flat CSR bag 则同时正确切分 `lengths` 与对应的 `values`。
直接使用 `MFalconInferenceEngine.run` 时，返回值同时包含本次 request cache；
`update_request_cache` 会优先调用模型的增量更新接口，否则安全地完整重建，因而同一
会话的后续 request 也能复用缓存生命周期。

`src.mfalcon.build_mfalcon_attention_layout` 还提供论文中的显式拼接布局，供将
多个候选物理追加到 HSTU 序列的实现使用：

- history 保持 causal；
- history 不读取 candidate；
- 每个 candidate 读取全部有效 history 和自己；
- candidate 之间互不可见；
- 所有 candidate 使用同一个有效 position 和 query timestamp，因此 position
  bias 与 timestamp bias 对 history 完全一致。

当前模型本来就把 candidate 放在 batch 轴上，因此候选隔离由 batch 语义天然
保证；服务路径实际需要补的是上述 request cache 生命周期和有界微批调度。
M-FALCON 是严格的 eval-only 路径，训练态调用会直接报错。

## Kraken GSET

生产默认（`scripts/build_production_configs.py`）如下：

```yaml
runtime:
  distributed: ddp
  compile: false
  cuda_graph_backbone: false

training:
  embedding_distribution: sharded
  embedding_sparse_gradients: true
  sparse_optimizer: rowwise_adagrad
  sparse_update_mode: ddp_synced_adagrad
  adagrad_weight_decay: 0.0
  gset:
    enabled: true
    capacity: 80000000
    compress_dim: 16
    key_mode: namespace
    missing_raw_id: 0
    eviction_policy: score
    admission_probability: 1.0
    score_decay: 0.1
    positive_weight: 1.0
    score_task: upid_pay
    score_update_interval: 1
    eviction_enabled: true
    seed: 2025
```

`capacity` 是每个 rank 上所有 categorical namespace 合用的物理行数；实现另保留不可训练的
row 0 作为 padding、missing、未准入 ID 的零向量。开启后所有 categorical 的
`embedding_dim` 被压到 `compress_dim`，这样它们能共享一张物理表。全局 mapper
的逻辑 key 是 `(feature namespace, logical ID)`，所以不同特征的同一整数 ID
不会碰撞；配置为 `share_embedding` 的字段则复用 owning base feature 的同一个
namespace。因此共享 alias 的策略要写在 owning base feature 上。TTL 的单位是
成功的稀疏 optimizer step，不是墙钟秒。

生产默认 `embedding_distribution: sharded`：按线上规则 `id % world_size == rank`
把逻辑 ID 切到各 GPU。每个 rank 只准入/淘汰自己拥有的 ID，lookup 用 all-to-all
交换 ID 和 embedding 行。`capacity` 是**每卡**物理行数，不是全局再除以 GPU 数。
`embedding_distribution: replicated` 仍可用：整表复制，并在 DDP 下对并集做
lockstep 准入。

Dataclass 默认仍是 `enabled: false`，避免 paper / 单测在未声明 capacity 时
分配生产级大表。生产 YAML 和 `build_config()` 会打开它。

多卡 DDP 下 GSET 默认按 `id % world_size` 分片：每个 rank 先收集本 microbatch
的 `(namespace, id)` 计数，再 all-to-all 到 ID 的 owner，由 owner 做准入/淘汰/打分，
随后的 embedding lookup 把逻辑 ID 路由到 owner 再取行。Mapper 因此按 residue
class 分裂，不再需要全 rank 复制。`compile` 和 `cuda_graph_backbone` 仍然不能开。

完整替换顺序是：

1. 未见 ID 按 namespace 的概率做 Bernoulli admission；批内重复仍保持几何等待
   时间对应的总准入概率。
2. 表满时先回收超过 TTL 的 ID。
3. 没有过期 ID 时，在低优先级 namespace 中按最低 feature score 淘汰，并用
   last-access 和 slot ID 做确定性 tie-break。
4. score 按论文公式更新：
   `S(t+1) = (1-beta) S(t) + beta (r*c_positive + c_negative)`。

训练器会使用 `score_task` 的 label 和 label mask 自动产生正负计数；未设置时使用
第一个 task。计数单位是“包含该 ID 的样本”，而不是 token 出现次数：重复序列 ID、
CSR bag，以及多个 candidate 共用 request row 的情况都会先按样本去重。评估和预测
只读 mapper，不准入新 ID，也不延长 TTL 或修改 feature score。

槽位在一个 optimizer boundary 内会被 pin，避免梯度累积期间两个逻辑 key 写入同一
物理行。槽位复用时会重新随机初始化 embedding row，并在应用新梯度前重置对应的
rAdaGrad 累加器。rAdaGrad 对每个 embedding row 只保存一个 FP32 累加值，更新量为
该行梯度平方的维度均值。

GSET mapper、淘汰 metadata、计数器和 admission RNG state 都随模型 `state_dict`
保存；row-wise optimizer state 由训练 checkpoint 一并保存。因此恢复后逻辑 key、
物理 slot、淘汰次序和随机准入序列可连续运行。

## 主要代码与测试

- `src/mfalcon.py`：显式 mask/position/time layout、候选切片、cache-aware 推理引擎。
- `src/modules/gset.py`：全局 mapper、准入、score/TTL/priority 淘汰、checkpoint state。
- `src/optim.py`：rAdaGrad 与 GSET 槽位复用时的 optimizer-state reset。
- `src/model.py`、`src/train.py`：模型 embedding bank、标签计数和实际训练/推理接线。
- `tests/test_mfalcon.py`、`tests/test_gset.py`：机制级数值、隔离、淘汰、恢复和集成测试。
