# MixFormer 修复与验收（2026-10-05）

本轮延续 2026-09-30 开始的修复。服务重启后核查发现，前三项修复及回归测试已经包含在工作区最新提交 `5f8b1a7` 中；没有回退或重复覆盖该提交。本轮补齐实际 GPU 验证，发现并修复了混合精度类型不匹配，扩展了验收脚本。

## 修复内容

1. **编译时丢弃 padding mask。** 原 `_history_is_dense()` 在编译时无条件返回 True，将较短请求的 padding 当作事件。修复前最小反例正确输出 `1.0`，编译输出 `0.880797`。现在未知密度保留掩码，动态 attention dispatch 明确处于编译图外，混合长度和候选占位由 varlen/SDPA 正确处理。Query Mixer、序列投影、Output Fusion 仍可编译。
2. **空历史断开梯度。** 原零长度分支直接返回 query，使 key/value、sequence norm、sequence FFN 共 5 个参数没有梯度。现在为这些参数及上游序列输入保留零值计算图连接，空历史返回值不变，但梯度是零而不是 None。已测试与 `static_graph=True, find_unused_parameters=False` 的组合。
3. **Flash 配置未落实。** MixFormer/MDL-MixFormer 构造器现在传递 `runtime.attention_backend`；strict Flash 不允许静默回退，混合长度走 flash-attn varlen，完整密集历史走仅允许 Flash 的 SDPA。预检同时检查 varlen 和 padded SDPA 能力。`sdpa` 不主动调用外部 flash-attn。严格 Flash 时不走手写在线 softmax 的长度分块分支，序列 FFN 分块仍可使用。
4. **BF16 query 与 FP32 历史残差不匹配。** GPU 编译回归发现，autocast 可能只将投影 query 转成 BF16，历史残差仍为 FP32；外部 flash-attn 不会自动转换。现在在注意力入口将历史转换到 query dtype，保留可微转换。

这些是正确性修复，不改变论文的 Query Mixer / Cross Attention / Output Fusion 公式，也没有修改模型权重键名。旧 checkpoint 可加载不代表之前受错误 mask 影响的训练结果已自动纠正，建议独立对照或重新训练。

## 验收结果

环境：`mdl-cu128`，PyTorch 2.10.0+cu128。使用当时空闲的物理 GPU 4（通过 `CUDA_VISIBLE_DEVICES=4` 映射为 cuda:0）。没有停止其他任务、访问生产 checkpoint 或启动真实业务训练。

| 验证 | 结果与范围 |
| --- | --- |
| CPU 综合回归 | 133 passed、130 subtests passed；1 个显式 GPU 测试跳过、2 个 GPU varlen 测试暂不选中 |
| 上述 3 个 GPU 测试单独执行 | 3 passed；包含 BF16 + 实际 CUDA Inductor 编译、混合长度、空历史、候选分组和输出/梯度对照 |
| CPU/Gloo 两进程完整模型 | MixFormer + 动态分片 embedding，static_graph=True，空历史/非空历史交替更新通过；dense 参数跨 rank 一致 |
| 双 GPU/NCCL 两进程完整模型 | 物理 GPU 0、5（`CUDA_VISIBLE_DEVICES=0,5`），UniFormer/MORE/MixFormer 各 3 步；包含一侧全空历史、dense 参数同步和动态分片 embedding owner 检查，`1 passed` / 33.16 秒 |
| CPU 20 步合成更新 | loss 0.427754 → 0.009141；动态 embedding 每步实际变化；10 月 5 日逐步 Adagrad 参考对照复测通过 |
| GPU 20 步合成更新 | BF16、strict Flash、dense-only compile；loss 0.369074 → 0.011135；无缺失或非有限梯度 |
| GPU 正式 trainer | 630 列合成聚合 Parquet，编译开启、逐步日志开启；完成 3 步 / 12 candidates，末步 loss 0.329391 |
| 空历史 checkpoint | state dict 严格恢复、恢复后输出一致 |

测试数据非常小：普通 ID 表限制到 64 行，动态 RankEmbeddingTable 由测试 ID 扩到 8192 行。上述 loss 是同批合成样本的可学习性检查，不是验证集指标。正式 trainer 的梯度累积为 1，不能替代生产大 batch、64 microbatch 累积及长时间稳定性验收。

### BF16 embedding 检查口径

GPU 探针最初以“每一步权重张量都必须变化”为条件，20 步中 5 步因此触发失败。进一步验证表明，这 5 步的 FP32 Adagrad 参考更新写回 BF16 后也等于原值；这是当前 warmup、小学习率与 BF16 权重存储下的舍入，不是旧 Parameter 未更新的问题。

验收脚本现在逐步检查：optimizer 确实持有当前表权重，按稀疏梯度和累加器计算的参考更新与实际更新一致；如果实际权重不变，参考计算在写回目标 dtype 后也必须不变。最终 15/20 步实际改变权重，5/20 步由参考计算确认是舍入。没有把 optimizer 的 step 计数当作唯一更新证据。该结果也提示：生产 sparse LR / warmup / embedding 存储精度仍需实数据信号验证，本轮未擅自改成 FP32 master weights 或修改学习率。

完整结果：`artifacts/mixformer_fix_20261005_cuda4_compiled_v2/results.json`。
CPU 参考对照：`artifacts/mixformer_fix_20261005_cpu_reference/results.json`。
无 `_v2` 的同名前缀目录是首次 GPU 验收留下的合成数据，不是成功报告。

## 复跑

CPU 综合验证（不占用 GPU）：

```bash
conda run -n mdl-cu128 python -c 'import torch,pytest; torch.set_num_threads(1); raise SystemExit(pytest.main(["-q","tests/test_mixformer.py","tests/test_mixformer_regressions.py","tests/test_attention_preflight.py","tests/test_industrial_rankers_ddp.py","tests/test_rank_table_checkpoint.py","tests/test_ddp_config.py","tests/test_config_overlays.py","tests/test_checkpoint_resume.py","-k","not varlen_mixed_length and not varlen_grouped_occupancy"]))'
```

选择当时可用的 GPU，显式启用 GPU 回归或编译训练探针（输出目录必须不存在）：

```bash
conda run -n mdl-cu128 env CUDA_VISIBLE_DEVICES=4 MDL_TEST_MIXFORMER_CUDA=1 python -c 'import torch,pytest; torch.set_num_threads(1); torch._inductor.config.compile_threads=1; raise SystemExit(pytest.main(["-q","tests/test_mixformer_regressions.py::MixFormerRegressionTest::test_cuda_bf16_compiled_mixed_masks_match_sdpa","tests/test_mixformer.py::MixFormerPaperAlignmentTest::test_varlen_mixed_length_matches_masked_sdpa","tests/test_mixformer.py::MixFormerPaperAlignmentTest::test_varlen_grouped_occupancy_matches_expanded"]))'
conda run -n mdl-cu128 env CUDA_VISIBLE_DEVICES=4 python scripts/verify_industrial_rankers.py \
  --models mixformer --device cuda:0 --steps 20 --compile --output artifacts/mixformer_verify_new

# 双 GPU NCCL（只在确认两张卡可共享时运行；可见设备必须恰好两张）
conda run -n mdl-cu128 env CUDA_VISIBLE_DEVICES=0,5 MDL_TEST_RANKERS_NCCL=1 \
  python -m pytest -q tests/test_industrial_rankers_nccl.py
```

## 未宣称完成的部分

- 生产配置训练输入仍为空，未验证真实数据效果、时间切分/泄漏、长尾和标签分布。
- 已完成小规模双 GPU/NCCL 正确性验收，但尚未完成 NCCL+compile 联合验收、生产规模 8k 长历史与动态表长期增长、HDFS checkpoint 全链路以及端到端吞吐基准；不能声称效率最佳。正确处理 padding 的 graph break 可能影响吞吐，需重新测量，不能沿用此前错误无掩码路径的速度结论。
- 共享 embedding 仍是已有的无界 append-only RankEmbeddingTable，不提供原有 GSET 的容量/计分/淘汰契约。相关旧 GSET 测试冲突没有通过删测试或伪造接口掩盖；是否恢复有界 GSET 仍需明确设计选择。
- MDL-MixFormer 获得共享注意力修复与 backend 传递，但本报告完整 GPU 训练结果只针对独立 MixFormer，不是 MDL-MixFormer 的全部生产验收。
