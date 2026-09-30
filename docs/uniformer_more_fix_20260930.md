# UniFormer / MORE 修复与验收（2026-09-30）

本轮在已有未提交实现上继续修复，没有回退其他工作区修改、提交 Git、访问生产 checkpoint 或启动真实业务训练。结论是：已修复可复现的训练和表示问题，通过 CPU 合成训练；还不能宣称论文完全复现、真实数据完全适配或效率最佳。

## 已完成

- 正式训练日志分支使用正确的 `batch_iterator`，开启每步日志可完成训练。
- coarse/fine 的 UniFormer/MORE 分别使用 `uniformer_v2`、`uniformer_fine_v2`、`more_v2`、`more_fine_v2` checkpoint 空间，避免混用 MixFormer 或修复前结构。原有旧断点未删除。
- UniFormer 用 8 个显式完整字段语义组替代维度硬切；验证输入无重复、无遗漏、不跨请求/候选轴，并接入独立 SwiGLU。
- MORE 在时间排序和全局选取后加入可学习位置编码，仅按有效事件计数；门控恢复 `2*sigmoid`。
- 修复请求数等于候选数时忽略非恒等 `row_indices` 的错误。
- 序列 K/V 保留在请求轴，分组候选 Q 读取共享 KV；每次 forward 中复用压紧的 KV。SDPA 参考路径不复制长历史，Flash 路径使用 varlen；严格 Flash 不再静默退回其他后端。
- UniFormer 用户/候选可见性掩码拆成两个语义等价的无掩码注意力调用，兼容严格 Flash；RMSNorm 权重转换到激活计算 dtype。
- 独立 token SwiGLU 与 MORE gate 改为批量矩阵乘法，保留独立权重。提供局部 dense-only 编译入口，默认仍关闭；未支持的缓存、activation checkpoint、CUDA graph 选项明确报错。
- 增加 Adam/AdamW dense optimizer 选项与隔离实验配置。生产默认训练策略不变，不将其包装成原论文超参复现。
- 去掉 UniFormer FLOPs 的 512-token 硬上限，补入主要 FFN/序列投影项。估算仍不是精确端到端 MFU：不含数据读取/稀疏通信，调用口径缺少完整请求复用分布。

## 验收中新发现的动态 embedding 问题

当前工作区实际使用 `RankEmbeddingTable`，并不是固定容量、带淘汰策略的原始 GSET。旧命名和 `training.gset.capacity` 不能证明它有容量上限。保留该已有设计，不擅自切换回其他表实现。

本轮修复了：

1. 扩容后的 checkpoint 无法加载回初始单行表；恢复前按保存尺寸分配权重，恢复 ID 映射和训练步数。
2. 动态表未向 optimizer 注册扩容通知，导致优化器仍指向旧 Parameter；现在两个自定义 Adagrad 优化器在构造时绑定动态表。
3. 新增行的累加器应使用配置的 initial accumulator，而不是无条件填零。
4. 梯度裁剪的初始参数列表可能过期；裁剪会解析到当前动态表权重。

新增测试检查实际权重变化、扩容后的参数引用、裁剪范数、权重/ID 映射恢复及 optimizer 续训等价。仅看到 step 计数增加或总 loss 下降，不足以证明 sparse embedding 真正更新；新验收脚本逐步验证了这一点。

## 已验证范围

环境：`mdl-cu128`，PyTorch 2.10.0+cu128；本轮 CPU FP32、单线程，不占用当前已有任务的 GPU。

| 验证 | UniFormer | MORE |
| --- | --- | --- |
| 生产宽度、20 步合成更新，保留 warmup/clip | loss 0.361498 → 0.330419 | loss 0.361172 → 0.202052 |
| 动态 embedding 实际发生更新 | 20/20 步 | 20/20 步 |
| 梯度缺失/非有限 | 均无 | 均无 |
| 正式 trainer，630 列合成 agg Parquet，逐步日志开启 | 3 步 / 12 candidates | 3 步 / 12 candidates |
| 两进程 CPU/Gloo，完整模型和动态分片 embedding，交替空历史 | 3 步通过 | 3 步通过 |
| 两进程 CPU/Gloo，独立骨干、空历史、dense 参数同步 | 3 步通过 | 3 步通过 |
| 空历史、state dict 严格恢复 | 通过 | 通过 |

动态表由测试 ID 扩至 8192 行；其他普通 ID 表为测试缩到 64 行。没有零向量替身代替上述训练探针的 embedding。单测中部分接口检查使用仓库自带 synthetic embedding helper，不能与训练探针混为一谈。

另已验证：共享 KV 与候选展开的前向/梯度等价，批量 FFN/gate 与逐 token 实现等价，用户/候选可见性等价，padding 不改变有效位置，严格 Flash 在 CPU 报错。CPU `aot_eager/fullgraph` 已验证 UniFormer feature/task FFN 与 MORE mixing 的编译前向和反向；这不是 CUDA Inductor 性能验收。

最终综合回归（14 个测试文件）：**198 passed、128 subtests passed、1 failed**。
另跑 `tests/test_industrial_rankers_ddp.py`：**2 passed**。`git diff --check` 通过。
唯一失败为 `tests/test_gset.py::test_model_bank_uses_one_physical_gset_and_shared_alias_namespace`：
该既有测试调用 `force_score_update()`，要求有界 GSET 的计分契约，而当前
FeatureEncoderBank 使用的 append-only RankEmbeddingTable 没有此接口。这是尚未解决的
实现/配置/测试契约分歧，不应删测试或添加无效空方法使之假通过。需确认是保留动态表并
正式调整契约，还是恢复有界 GSET；本轮没有替用户决定这一已有设计变更。

MORE 只反转 timestamp、保留其他输入时，输出最大差约 `1.25e-6`，不再严格排列不变。该小幅差异只证明通路存在，不证明位置特征已学出业务收益。

最新训练结果：`artifacts/uniformer_more_fix_20260930_cpu_v2/results.json`。首次修复验收位于不带 `_v2` 的目录，早于动态表更新修复，不能替代最新版。

复跑（输出目录必须不存在）：

```bash
conda run -n mdl-cu128 python scripts/verify_industrial_rankers.py \
  --device cpu --steps 20 --output artifacts/ranker_verify_new
```

获准使用空闲 GPU 后，把 `--device` 改为 `cuda:N` 即可跑 BF16 + strict Flash 训练验收。性能比较还须使用同一 GPU、负载、精度、warmup 和数据分布单独测量，不能直接与之前审查的 GPU 数值相除。

## 尚未完成，不能当作“全部已修好”

- 四份生产配置仍缺真实训练/验证输入；需用户提供路径，才能验证实际字段值、label mask、时间切分/泄漏、三任务离线收益及真实长度分布。
- MORE Cartesian action 组合尚用 stream-type 代理。需要逐事件的 item 关联键、行为语义和合并时间窗口，不能把 hash 值直接当原始布尔动作或跨时间凑组合。
- UniFormer 的任务先验、8 流融合、`srch_q2i` 是否依赖当前候选仍是本数据适配选择；论文未给出的结构细节不能伪称严格复现。
- 当前所有 GPU 有其他任务，本轮未完成 CUDA BF16/varlen 实测、NCCL 多卡、8k 长历史吞吐/显存、CUDA Inductor、长期稳定性及生产 checkpoint 全流程。CPU/Gloo 通过不能替代这些验收。
- 跨 forward/microbatch request cache、UniFormer 用户侧 FIM 去重还未实现；仍有进一步减少计算和 host 同步的空间，因此不能声称效率最佳。
- 动态表无界增长与容量/淘汰策略需要单独决策；本轮没有将已有 append-only 设计改成有界 GSET。

需要用户补充：真实训练集/验证集路径、可用 GPU 编号、MORE 行为组合的数据语义，以及
有界 GSET / append-only 动态表的设计选择。收到后可继续完成生产数据与 GPU 验收。
