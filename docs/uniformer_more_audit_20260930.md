# UniFormer / MORE 实现审查（2026-09-30）

> 本文保留修复前的审查记录，不代表当前代码状态。后续修改、验证结果及仍未验收的部分见 [修复报告](uniformer_more_fix_20260930.md)。

结论：两者的主要交互骨干已实现，缩小 embedding 容量后能进行 BF16 + GSET 更新；当前正式训练循环存在可复现的日志崩溃，不能直接判定为可稳定上线训练。输入表示与论文仍有重要差异，效率也未优化到位。数据适配已达到字段、请求/候选轴和标签可连通的程度，尚未完成真实数据训练效果验收。

审查对象是当前工作区，包括审查前已有的未提交实现；HEAD 为 `7d5c9d2609361eb5ce7c2b5ef94523d81ae5d35f`。本次仅新增论文副本、诊断脚本、结果和报告，没有修改模型、生产配置或训练代码。

## 论文与本地文件

| 论文 | 正式来源 | 本次下载 |
| --- | --- | --- |
| UniFormer: Efficient and Unified Model-Centric Scaling for Industrial Recommendation，快手 | [arXiv 2606.27058v1](https://arxiv.org/abs/2606.27058v1) | [PDF](/home/user/MDL/artifacts/uniformer_more_audit_20260930/uniformer-2606.27058v1.pdf) |
| Task-Blind No MORE: Multi-Task Information Flow in Unified Ranking Backbones，Hello Group / Momo | [arXiv 2609.07273v1](https://arxiv.org/abs/2609.07273v1) | [PDF](/home/user/MDL/artifacts/uniformer_more_audit_20260930/more-2609.07273v1.pdf) |

两份 PDF 均重新从 arXiv 下载，与 `papers/` 下既有文件的 SHA-256 完全一致。版本、来源、校验值见 [归档清单](/home/user/MDL/artifacts/uniformer_more_audit_20260930/archive-manifest.jsonl)。本次审查以论文与本地实现为准，未将任何第三方实现视为官方标准。

## 已证实的问题，按优先级排列

### P0：正式训练在日志分支崩溃，影响两个模型

[src/train.py:8344](/home/user/MDL/src/train.py:8344) 调用 `_host_memory_report(host_batch_iterator)`，但函数中实际维护的变量是 `batch_iterator`；全文件没有定义 `host_batch_iterator`。

本次使用正式 `train_mdl`、BF16、真实的 adapter_parquet 适配流程、项目生成器创建的 630 列合成聚合 Parquet，将日志间隔设为 1。两个模型都完成第一个参数更新，随后报同一个 `NameError`。生产配置的 `log_every_steps=100`，在正常启用日志且执行到该分支时会在第 100 步触发，而非一定在模型第一次前向触发。

使用已有 `log_steps=False` 开关隔离该分支后，两者均完成 3 步、12 个候选样本的正式训练流程。证据：[失败记录](/home/user/MDL/artifacts/uniformer_more_audit_20260930/e2e_results_no_windows.json)、[关闭日志的对照](/home/user/MDL/artifacts/uniformer_more_audit_20260930/e2e_results_no_windows_no_log.json)。这是共享训练器的问题，不能归因于某一个新增模型，也不能用模型单测通过来排除。

建议先修复变量引用并增加“真实 trainer 跨过日志间隔”的回归检查。

### P1：fine 配置沿用了 MixFormer 的 checkpoint 命名空间

`uniformer_fine.yaml`、`more_fine.yaml` 均继承 `mixformer_fine.yaml`，没有覆盖 [run_name: mixformer_fine](/home/user/MDL/configs/mixformer_fine.yaml:3729)。[训练器](/home/user/MDL/src/train.py:6741) 优先采用显式 `run_name`，而不是当前模型名；还继承了 `resume: auto` 和相同远端目录。

因此两个 fine 实验会指向 MixFormer fine 的存储空间，可能选中不兼容的断点，并共享保存/保留策略作用的目录。尚未对远端存储执行读取、恢复或写入来验证实际冲突。coarse 配置的 `run_name=null` 会回退到各自模型名，没有这个具体命名问题。

建议分别设置 `uniformer_fine` / `more_fine` 的独立 run name，并明确断点恢复策略。

### P1：UniFormer 的生产 tokenizer 没有实现论文的语义分组 + SwiGLU

论文 §4.2.2 要求按语义组织非序列特征，再分别使用 SwiGLU 投影。实际 [UniFormerTokenizer](/home/user/MDL/src/uniformer_model.py:71) 使用 `_EvenHeadProjector`：用户侧、候选侧分别拼接，再按总维度均分，最后做线性投影。当前 624/1312 维拆成 4+4 个 token；切分点不必落在字段或语义组边界。

仓库确有 [GroupedSwiGLUTokenizer](/home/user/MDL/src/modules/uniformer.py:231)，但实际 `UniFormerModel` 未接入它；它用于独立模块/合成复现脚本。因而“骨干公式大体对齐”与“生产输入层对齐”必须分开判断。

任务初始 token 采用 learned task prior 加用户 token 均值，再经过独立 SwiGLU；这是当前数据上的自定义设计，不能声称与原文的任务特征输入完全一致。8 条行为流的 softmax 融合是合理的多流扩展，尚需消融验证。

### P1：MORE 缺少论文的位置编码，时间排序没有等效作用

论文 Eq.(3) 包含可学习的 `e_time`。实际 [MORETokenizer](/home/user/MDL/src/more_model.py:114) 复用 OneTrans 的时间排序/合流；[SequenceConfig.event_fields](/home/user/MDL/src/config.py:1406) 从事件向量中排除原始 timestamp，而该模型也没有添加 OneTrans 主模型中的统一位置 embedding。

MORE 骨干只对序列逐位置变换，再做池化与 cross-attention；没有位置编码时，对同一组事件的排列不敏感。实测保持商品、行为类型、timegap 等内容不变，仅反转 timestamp，在全部 96 个事件都保留的条件下，输出 logits 的最大绝对差为 **0.0**。证据：[semantic_probe.json](/home/user/MDL/artifacts/uniformer_more_audit_20260930/semantic_probe.json)。

这不意味着模型完全没有时间信息：`timegap_hn` 等字段仍能传递粗粒度时效性，timestamp 也影响超过上限时的事件选择。但它没有实现论文中的顺序位置表示。建议在最终保留、排序后的事件上加入明确的位置或时间编码，并验证顺序敏感性。

### P1：MORE 的 action 编码只完成了代理适配

论文 Eq.(3) 使用每个 item 的多行为组合 ID。当前输入为 8 个行为族，使用 stream-type embedding 表示来源，没有将同一 item 对应的多任务行为组合编码成 Cartesian ID。[cartesian_sequence_token](/home/user/MDL/src/modules/more.py:83) 只是独立 helper，生产 tokenizer 未调用。

目前没有证据证明本数据具有构造所有原文组合所需的逐事件标签与时间对应关系，不能直接照抄组合逻辑。需要先确定行为组合发生的时间窗口、同物品事件合并规则和可用字段，再比较组合 ID 与当前 stream-type 代理的效果。

### P2：MORE 门控范围与论文图 2 不一致

论文图 2 明确画出 `2*Sigmoid`，图中 gate 数值也包含大于 1 的值。[TokenGate.forward](/home/user/MDL/src/modules/more.py:186) 仅调用 `sigmoid`，输出范围为 (0,1)，不能主动放大通道。将门控网络参数清零时，代码输出 0.5，而图示函数应输出 1。

代码将图示的“2”解释为隐藏层宽度 `2*head_dim`，这个解释不能替代输出乘 2。Eq.(11) 没有完整规定 gate MLP 内部结构，隐藏宽度和是否跨 token 联合计算仍属论文细节不充分，不能据此另行认定整个 MLP 结构错误。

## 骨干与实验设置对齐程度

| 项目 | 当前判断 |
| --- | --- |
| UniFormer 多流 cross-attention、S-FFN、concat/self-attention、独立 NS-FFN | 主要结构已实现 |
| UniFormer Lazy KV、FIM→TIM、任务独立 FFN/head、用户不能读候选的 mask | 已实现核心计算；请求级复用仍不完整 |
| UniFormer 深度和宽度 | 采用 2 FIM + 1 TIM、d=128；原文“3 层”和 d_head=1280 的细节不足，不能把当前规模称作原实验复现 |
| MORE Shared/Private Anchor、task prior、逐层 S/K/V、register | 已实现核心结构 |
| MORE task boundary mask、per-token FFN、FiLM 增强 | 主体实现与方法描述对应；gate 存在上述差异 |
| MORE 宽度与任务数 | 当前 M=8/K=5/T=3/d=128；原文默认 M=15/K=8/T=9/d=256。为三任务与 token-mixing 整除约束适配是合理选择，但未调优证明最优 |
| 优化器 | 当前继承 RMSprop + rowwise Adagrad，lr=1e-4；UniFormer 原文为 AdamW，MORE 为 Adam。属于本项目训练策略，不是原文超参复现 |
| 指标口径 | 当前 group_id 为 search_id，即请求分组；论文报告用户分组 GAUC。不能直接比较绝对值和收益 |

原文也存在复现信息不足：UniFormer 未明确 FIM/TIM 的默认层数拆分；MORE 实验部分同时出现 13 个总任务和 9 个核心 private anchors，任务塔细节也不完整。上述不确定性不应被包装成已严格复现。

## 能否训练：本次实测边界

| 验证 | UniFormer | MORE |
| --- | --- | --- |
| 现有模型测试 | 两模型合计 17 passed、4 subtests passed | 同左 |
| 现有独立合成可学习性脚本 | BCE 0.6947→0.0045 | BCE 0.6872→约 0 |
| 保留生产宽度、20 步 BF16 + RMSprop + GSET/rAdaGrad + 原 warmup/clip | 加权 loss 0.3575→0.3161 | 0.3634→0.2715 |
| 上述测试末步缺梯度/非有限梯度 | 均无；GSET step=20 | 均无；GSET step=20 |
| 正式 trainer + 合成 agg Parquet + 开日志 | 第一次日志触发 NameError | 同样失败 |
| 正式 trainer + 同样输入 + 关闭日志 | 完成 3 步/12 candidates | 完成 3 步/12 candidates |

环境为 PyTorch 2.10.0+cu128、单张空闲 RTX 4090、BF16；小样本测试将 GSET capacity 从 8000 万缩为 16384，将其他 ID embedding 上限缩为 64，并映射测试 ID 到小表。未使用零向量替身替换这些训练验证中的 embedding。合成测试没有提供泛化效果、原始大表容量、长时间训练稳定性或多卡通信正确性的证据。

20 步结果见 [cuda_0_bf16_warmup_clip.json](/home/user/MDL/artifacts/uniformer_more_audit_20260930/cuda_0_bf16_warmup_clip.json)。早期 `cpu_fp32.json` / `cuda_0_bf16.json` 探针未包含生产 warmup/clip，其损失增长不能用作生产训练不稳定的结论，保留仅供审计。正式 trainer 初次尝试还因合成目录不是小时分区而被 `data_window_hours=8` 拦截；本地验证显式设为 0，生产小时分区策略未修改。

## 效率：已测出改进空间，尚无“最佳”证据

两个骨干的注意力直接调用 PyTorch SDPA，没有接入当前配置选择的严格 backend 路径；`attention_backend: flash` 的启动日志不能证明每个 attention 实际使用 FlashAttention。本次 profiler 显示，带 bool padding mask 的序列 cross-attention 主要走 memory-efficient attention；UniFormer 部分其他 attention 使用 FlashAttention。

在所有位置都有效的同一负载下，把“全 True 的 padding mask”省略具有相同注意力语义。对照结果如下。均为 **backbone 前向+反向**，不含 tokenizer、embedding、数据读取、优化器或通信；4 个请求、32 个候选、d=128、BF16，3 次预热后取 8 次中位数。

| 模型 | 每请求总历史 token | 现有全 True mask | 省略冗余 mask | 比值 |
| --- | ---: | ---: | ---: | ---: |
| UniFormer | 1024 | 44.83 ms | 46.69 ms | 无明显收益 |
| UniFormer | 8192 | 61.52 ms | 61.82 ms | 无明显收益 |
| MORE | 1024 | 80.34 ms | 83.30 ms | 无明显收益 |
| MORE | 8192 | 170.35 ms | 82.48 ms | 2.07× |

8192 是骨干接口的长序列对照，不是生产 tokenizer 的 8000 上限设置。单机上其他 GPU 有任务，上述数据用于定位本实现瓶颈，不是独占机器上的最终吞吐评级，也不能推广为真实数据全面提速 2 倍。真实含 padding 的 batch 不能直接删除 mask；应接入 varlen、长度分桶或经过验证的合法 attention kernel。证据：[当前路径](/home/user/MDL/artifacts/uniformer_more_audit_20260930/backbone_profile.json)、[等价输入对照](/home/user/MDL/artifacts/uniformer_more_audit_20260930/backbone_profile_no_padding_mask.json)。

其他明确优化点：

- [UniFormer KV 展开](/home/user/MDL/src/modules/uniformer.py:520)：请求级 memory 完成后，通过两次 `index_select` 将同一 K/V 分别复制到候选批次；源头 K=V 的存储共享被打破。用户 token 也在 FIM 前展开，用户侧 cross-attention/FFN 仍逐候选重复。
- [MORE KV 展开](/home/user/MDL/src/modules/more.py:354)：序列 FFN 已在请求轴计算，这是正确的复用；每个 block 仍将 K/V 复制到候选轴，显存和搬运随候选数增长。两模型的 `request_cache` 参数目前均直接丢弃，不能跨候选 microbatch 复用缓存。
- 大量 `ModuleList` 中的逐 token FFN/gate 调用仍为 Python 循环。采样的一次前后向中，UniFormer/MORE 分别有 665/971 次 `aten::mm` 和 938/1491 次 `aten::copy_`，需要 grouped/batched GEMM 与 dense-only compile 验证，不能简单把所有模型含 GSET 一起 compile。
- UniFormer BF16 RMSNorm 出现输入/权重 dtype 不匹配导致无法使用 fused 实现的运行警告；值得单独验证融合和精度策略。
- MORE 合流包含动态选择、排序和 `bincount(...).tolist()` 的 CPU 同步。它也有“只投影选中事件”的已有优化，不能笼统认定为全部无效计算。
- [FLOPs 估算](/home/user/MDL/src/benchmark.py:355) 对 UniFormer 将长度上限写死为 512，且没有完整计算序列投影与多组 FFN；MORE 也是粗略式。现有估算不足以用 MFU/TFLOPs 证明最佳效率，应先修正口径。

## 对“我们的数据”的适配判断

已接通并验证的部分：现有 context/item 特征分轴、8 条 raw 行为流、request `row_indices` 到 candidate 的展开、`fst_cart` / `upid_pay` / `cateid_filter` 三标签、真实聚合 Parquet 的 adapter/direct 路径，以及 BF16/GSET 稀疏更新。原始时间戳不直接输入 MLP，避免大数值支配投影；MORE 用时间排序与全局截断处理历史，UniFormer 保留分流读取。

四份模型配置的 `data.train.inputs` 均为空。本次使用了明确标识的合成数据，没有对真实业务训练集、验证集或线上效果作实验，因此不能声称“已完全适配”或“效果已对齐论文”。尚需验证真实 label mask/正例率、时间切分与候选泄漏边界、长尾与空历史、真实请求候选数/长度分布、8k 历史与全容量 GSET、两卡 DDP/分片及 checkpoint 恢复，并完成三任务离线对照。

`srch_q2i` 在本配置中属于请求级历史流，没有 target-attention 输入。不能仅凭名字将它等同于 UniFormer 原文的 candidate-dependent 检索序列而强行合并；需要根据该列的生成链路确认是否依赖当前候选。

建议处理顺序：先修共享训练器和 fine checkpoint 隔离；再补齐/明确 UniFormer tokenizer、MORE 时间与行为编码、gate；随后用真实数据完成可训练性与离线效果验收；最后集中优化 varlen/request-level attention、KV 复用与 grouped FFN，并重新测端到端吞吐和显存。

## 复现材料

所有诊断位于 [artifacts/uniformer_more_audit_20260930](/home/user/MDL/artifacts/uniformer_more_audit_20260930)。脚本使用排他写入保护结果，重复执行前需为结果文件选择新名字。主要命令：

```bash
conda run -n mdl-cu128 python -c 'import torch,pytest; torch.set_num_threads(1); raise SystemExit(pytest.main(["-q","tests/test_uniformer.py","tests/test_more.py","tests/test_industrial_rankers.py"]))'
conda run -n mdl-cu128 python artifacts/uniformer_more_audit_20260930/probe.py --device cuda:0 --precision bf16 --steps 20
conda run -n mdl-cu128 python artifacts/uniformer_more_audit_20260930/e2e.py
conda run -n mdl-cu128 python artifacts/uniformer_more_audit_20260930/e2e.py --no-log
conda run -n mdl-cu128 python artifacts/uniformer_more_audit_20260930/semantic_probe.py
conda run -n mdl-cu128 python artifacts/uniformer_more_audit_20260930/backbone_profile.py
conda run -n mdl-cu128 python artifacts/uniformer_more_audit_20260930/backbone_profile.py --no-padding-mask
```
