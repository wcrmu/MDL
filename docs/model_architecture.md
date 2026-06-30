# 模型结构说明

本文档说明当前项目中已经实现的模型结构，重点覆盖 `MDL` 主模型、`RankMixer` baseline、`DeepFM` baseline，以及训练时使用的 loss 和可选 Sparse-MoE 结构。相关实现主要位于：

- `src/models/mdl.py`：MDL 主模型、manifest 驱动的模型封装。
- `src/models/rankmixer.py`：纯 RankMixer baseline。
- `src/models/deepfm.py`：DeepFM baseline。
- `src/modules/tokenizer.py`：把 manifest 中声明的特征编译成 token。
- `src/modules/embedding.py`：各类 feature encoder。
- `src/modules/attention.py`：RankMixer token mixing、domain-aware attention、场景融合。
- `src/modules/mlp.py`：逐 token FFN、Sparse-MoE FFN、上下文 tokenizer。

## 1. 总体数据流

项目采用 manifest 驱动的建模方式。训练脚本读取 processed dataset 中的 `manifest.json`，然后根据 `tokenization` 配置构造模型。

在 `model.name=mdl` 时，一条 batch 的前向传播可以概括为：

```text
features + scenario_id
  -> FeatureTokenCompiler
       feature_tokens:  [B, F, D]
  -> ScenarioTokenCompiler
       scenario_tokens: [B, S + 1, D]
  -> TaskTokenCompiler
       task_tokens:     [B, T, D]
  -> scenario_id 转 mask
       scenario_mask:   [B, S]
  -> L 个 MDLBlock
       feature_tokens:  [B, F, D]
       scenario_tokens: [B, S + 1, D]
       task_tokens:     [B, T, D]
  -> 每个 task token 独立输出头
       logits:          [B, T]
```

其中：

- `B` 是 batch size。
- `F` 是普通 feature token 数量，即 `tokenization.token_specs` 的长度。
- `S` 是场景数量，即 `scenario_names` 的长度。
- `S + 1` 中多出来的 1 是全局场景 token。
- `T` 是任务数量，即 `task_names` 的长度。
- `D` 是 `token_dim`。
- `L` 是 `num_layers`。

`RankMixer` baseline 使用相同的 feature token 编译方式，但不使用 scenario token 和 task token；它把所有 feature token 混合后 flatten，再接多任务输出头。

`DeepFM` baseline 是独立的 dense field 输入模型，目前没有接入 manifest 主训练路径，适合作为需要 `[B, num_fields, field_dim]` 输入的传统 DeepFM 槽位。

## 2. Manifest 到 Token 的编译

### 2.1 三类 token 配置

MDL 需要 manifest 同时声明三组 token：

```text
tokenization.features + tokenization.token_specs
  -> 普通 feature tokens

tokenization.scenario_features + tokenization.scenario_token_specs
  -> 场景 tokens，数量必须是 num_scenarios + 1

tokenization.task_features + tokenization.task_token_specs
  -> 任务 tokens，数量必须是 num_tasks
```

如果训练 MDL 时缺少 `scenario_features`、`scenario_token_specs`、`task_features` 或 `task_token_specs`，模型构建会直接报错。这样做的原因是：当前实现不是使用可学习的固定场景/任务 embedding，而是让场景 token 和任务 token 也由样本特征编译出来，使场景、任务与样本上下文绑定。

RankMixer baseline 只需要 `features` 和 `token_specs`，因此可以用于 feature-only manifest。

### 2.2 FeatureTokenCompiler 的工作方式

`FeatureTokenCompiler` 的逻辑是：

1. 按 `feature_specs` 为每个原始特征构建 encoder。
2. 每个 encoder 把原始输入转成一个 dense 向量。
3. 每个 `token_spec` 声明一个 token 使用哪些 encoded features。
4. 把这些 encoded features 拼接起来。
5. 通过 projection 投影到统一的 `token_dim`。

伪代码如下：

```python
encoded = {}
for feature_spec in feature_specs:
    encoded[feature_name] = encoder(feature_spec)(batch)

tokens = []
for token_spec in sorted(token_specs, key=token_id):
    parts = [encoded[name] for name in token_spec.inputs]
    token_input = concat(parts, dim=1)
    token = projection(token_input)  # [B, token_dim]
    tokens.append(token)

return stack(tokens, dim=1)  # [B, num_tokens, token_dim]
```

projection 支持两种：

- `linear`：一层 `Linear(input_dim, token_dim)`。
- `ffn_relu`：`Linear(input_dim, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, token_dim) -> ReLU`。

普通 feature tokens 默认使用 `linear`，scenario/task tokens 默认使用 `ffn_relu`。

### 2.3 支持的 feature encoder

当前 encoder registry 支持以下输入编码：

| encoder | 输入 | 输出 | 说明 |
| --- | --- | --- | --- |
| `embedding` | 类别 ID `[B]` | `[B, embedding_dim]` | 使用 `nn.Embedding`，`padding_idx=0` |
| `identity` | 数值特征 `[B]` 或 `[B, dim]` | `[B, dim]` | 直接转 float |
| `sequence_mean_pooling` | 序列字段 | `[B, sum(field_dims)]` | 对有效位置做 mask mean pooling |
| `din` | 序列字段 + target feature | `[B, sum(field_dims)]` | 使用 DIN activation unit 做目标感知加权 |
| `sim` / `longer` | 长序列字段 + target feature | `[B, sum(field_dims)]` | 先用 target 相似度取 top-k，再执行 DIN 加权 |

序列特征既可以是单字段，也可以是多字段。例如用户历史行为可以同时包含 `hist_item_id`、`hist_cate_id`、`hist_price`，先分别编码，再在最后一维拼接。

### 2.4 Token 编译示例

假设 manifest 中有：

```json
{
  "features": [
    {"name": "user_id", "encoder": "embedding", "vocab_size": 100000},
    {"name": "item_id", "encoder": "embedding", "vocab_size": 500000},
    {"name": "price", "encoder": "identity", "dim": 1}
  ],
  "token_specs": [
    {"token_id": 0, "projection": "linear", "inputs": ["user_id"]},
    {"token_id": 1, "projection": "linear", "inputs": ["item_id", "price"]}
  ]
}
```

如果 `embedding_dim=8`、`token_dim=16`，那么：

- `user_id` 被 embedding 成 `[B, 8]`，再投影为第 0 个 token `[B, 16]`。
- `item_id` 是 `[B, 8]`，`price` 是 `[B, 1]`，拼接成 `[B, 9]`，再投影为第 1 个 token `[B, 16]`。
- 最终 `feature_tokens` 形状是 `[B, 2, 16]`。

## 3. MDL 主模型

MDL 由 `ModelFromManifest` 和内部的 `MDLModel` 两层组成。

`ModelFromManifest` 负责把原始 batch 编译为三组 token：

```text
features -> feature_tokens
features -> scenario_tokens
features -> task_tokens
scenario_id -> scenario_mask
```

然后把这些张量传给 `MDLModel`。`MDLModel` 本身只关心 token 张量，不关心原始 CSV 字段。

### 3.1 关键配置

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `embedding_dim` | 32 | 默认类别 embedding 维度 |
| `token_dim` | 36 | 所有 token 的统一维度 |
| `num_layers` | 2 | MDLBlock 堆叠层数 |
| `num_heads` | 4 | domain-aware attention 的 head 数 |
| `ffn_hidden_dim` | 64 | token FFN 隐层维度 |
| `feature_backbone` | `rankmixer` | feature token 混合方式，可选 `rankmixer` 或 `attention` |
| `task_head_type` | `linear` | task 输出头，可选 `linear` 或 `mlp` |
| `ffn_type` | `dense` | token FFN 类型，可选 `dense` 或 `sparse_moe` |
| `use_task_tokens` | true | ablation 开关，是否使用任务 token |
| `use_scenario_tokens` | true | ablation 开关，是否使用场景 token |
| `use_global_scenario_token` | true | 是否使用全局场景 token |
| `use_task_feature_interaction` | true | 是否让任务 token attend feature tokens |
| `use_scenario_feature_interaction` | true | 是否让场景 token attend feature tokens |

重要约束：

- `token_dim` 必须能被 `num_heads` 整除。
- 当 `feature_backbone=rankmixer` 时，`token_dim` 必须能被 feature token 数量 `F` 整除。
- `scenario_token_specs` 必须恰好有 `S + 1` 个 token。
- `task_token_specs` 必须恰好有 `T` 个 token。

### 3.2 MDLBlock 的结构

每个 `MDLBlock` 包含三条交互路径：

```text
feature_tokens
  -> FeatureInteraction
  -> feature FFN

scenario_tokens + feature_tokens
  -> DomainAwareAttention
  -> scenario FFN

task_tokens + feature_tokens + scenario_tokens + scenario_mask
  -> DomainAwareAttention
  -> DomainFusedModule
  -> task FFN
```

更具体地说，单层 block 的执行顺序是：

```text
1. mixed_features = FeatureInteraction(feature_tokens)
2. feature_tokens = feature_ffn(LayerNorm(feature_tokens + mixed_features))

3. scenario_update = DomainAwareAttention(
       query=scenario_tokens,
       key=feature_tokens,
       value=feature_tokens
   )
4. scenario_hat = scenario_tokens + scenario_update
5. scenario_tokens = scenario_hat + scenario_ffn(scenario_hat)

6. task_update = DomainAwareAttention(
       query=task_tokens,
       key=feature_tokens,
       value=feature_tokens
   )
7. task_hat = task_tokens + task_update
8. task_tokens = task_hat + selected_scenario_average(scenario_hat)
9. task_tokens = task_tokens + task_ffn(task_tokens)
```

注意：当前实现中，任务分支融合的是 `scenario_hat`，也就是场景 token 加上 scenario-feature attention 更新后的表示；`scenario_ffn` 的输出会传给下一层 block 使用。

如果关闭相应 ablation 开关，场景 token、任务 token 或两类 domain-feature attention 会被置零或跳过。

### 3.3 FeatureInteraction

feature token 之间的混合有两种实现。

#### RankMixer token mixing

默认方式是 `RankMixerTokenMixing`。它不引入注意力参数，而是通过 reshape 和 permute 在 token 之间交换维度块。

要求 `D` 能被 `F` 整除，令 `H = D / F`。输入形状是 `[B, F, D]`：

```text
[B, F, D]
  -> view 为 [B, F, F, H]
  -> permute 为 [B, F, F, H]
  -> flatten 回 [B, F, D]
```

一个简单例子：`F=3`、`D=6`、`H=2`，三个输入 token 分别是：

```text
t0 = [a0 a1 | a2 a3 | a4 a5]
t1 = [b0 b1 | b2 b3 | b4 b5]
t2 = [c0 c1 | c2 c3 | c4 c5]
```

RankMixer 混合后得到：

```text
out0 = [a0 a1 | b0 b1 | c0 c1]
out1 = [a2 a3 | b2 b3 | c2 c3]
out2 = [a4 a5 | b4 b5 | c4 c5]
```

直观理解：每个输出 token 都从所有输入 token 中取同一段维度块，因此可以在很低成本下完成跨 token 信息交换。

#### Attention feature interaction

当 `feature_backbone=attention` 时，feature tokens 会进入标准 `nn.MultiheadAttention`，即：

```text
Q = K = V = feature_tokens
mixed_features = MultiHeadAttention(Q, K, V)
```

如果前向传播传入 `return_attention=True`，attention backbone 会返回 feature self-attention 权重；RankMixer backbone 没有 attention 权重，因此该字段为 `None`。

### 3.4 DomainAwareAttention

`DomainAwareAttention` 用于两处：

- scenario tokens attend feature tokens。
- task tokens attend feature tokens。

它和普通多头注意力的主要区别是：`Q`、`K`、`V` 不是共享线性层投影，而是通过 `PerTokenFFN` 分别投影。也就是说，每个场景 token、任务 token、feature token 都有自己的一套小 FFN 投影参数。

设 domain token 数量为 `M`，feature token 数量为 `F`，token 维度为 `D`，head 数为 `A`，单 head 维度为 `d = D / A`。

```text
domain_tokens:  [B, M, D]
feature_tokens: [B, F, D]

Q = PerTokenFFN(domain_tokens)  -> [B, A, M, d]
K = PerTokenFFN(feature_tokens) -> [B, A, F, d]
V = PerTokenFFN(feature_tokens) -> [B, A, F, d]

scores  = Q @ K^T / sqrt(d)    -> [B, A, M, F]
weights = softmax(scores)      -> [B, A, M, F]
update  = weights @ V          -> [B, M, D]
```

举例来说，如果有 `S=2` 个业务场景、1 个全局场景 token、`T=2` 个任务、`F=2` 个 feature tokens、`num_heads=4`：

- scenario-feature attention 权重形状是 `[B, 4, 3, 2]`。
- task-feature attention 权重形状是 `[B, 4, 2, 2]`。

### 3.5 场景融合 DomainFusedModule

`DomainFusedModule` 把当前样本所属场景的信息注入每个 task token。

输入：

- `task_tokens`: `[B, T, D]`
- `scenario_tokens`: `[B, S + 1, D]`
- `scenario_mask`: `[B, S]`

`scenario_mask` 不包含全局场景 token。例如 `scenario_names=["home", "search"]`：

```text
scenario_id = 0       -> scenario_mask = [1, 0]
scenario_id = 1       -> scenario_mask = [0, 1]
scenario_id = [0, 1]  -> scenario_mask = [1, 1]
```

如果启用全局场景 token，模块会在 mask 后面补一个 1：

```text
full_mask = concat(scenario_mask, [1])
```

然后对被选中的场景 token 求平均：

```text
selected_scenario_average =
    sum(scenario_tokens * full_mask) / sum(full_mask)
```

最后把这个场景平均向量加到每个 task token 上：

```text
task_tokens = task_tokens + selected_scenario_average.unsqueeze(1)
```

例子：如果样本属于 `home` 场景，并启用全局场景 token，那么注入任务 token 的向量是：

```text
(home_scenario_token + global_scenario_token) / 2
```

如果样本同时属于 `home` 和 `search`，注入向量是：

```text
(home_scenario_token + search_scenario_token + global_scenario_token) / 3
```

### 3.6 输出头

MDL 为每个任务维护一个独立输出头。

当 `task_head_type=linear`：

```text
logit_i = Linear(D, 1)(task_token_i)
```

当 `task_head_type=mlp`：

```text
logit_i = Linear(D, H) -> GELU -> Dropout -> Linear(H, 1)
```

所有任务的输出拼接后得到：

```text
logits: [B, T]
```

训练和评估阶段通常再对 `logits` 执行 sigmoid 得到每个任务的概率。

## 4. Sparse-MoE Per-Token FFN

`ffn_type` 可以选择：

- `dense`：每个 token 一套普通两层 FFN。
- `sparse_moe`：每个 token 一组 experts 和 router。

Dense FFN 的结构是：

```text
Linear(D, hidden_dim) -> GELU/ReLU -> Dropout -> Linear(hidden_dim, D)
```

Sparse-MoE FFN 对每个 token 单独维护：

- `num_experts` 个 expert，每个 expert 是两层 FFN。
- 一个 training router。
- 一个 inference router。

router 输出经过 ReLU 得到非负 gate：

```text
gates = ReLU(router(token_values))  # [B, num_experts]
```

训练时默认启用 DTSI，输出是 training router 和 inference router 的混合：

```text
train_output = sum(train_gate_k * expert_k(x))
infer_output = sum(infer_gate_k * expert_k(x))

output = (1 - sparse_moe_dtsi_infer_weight) * train_output
       + sparse_moe_dtsi_infer_weight       * infer_output
```

推理时使用 inference router，并根据 `sparse_moe_inference_threshold` 跳过 gate 不大于阈值的 expert：

```text
for expert_k:
    if infer_gate_k > threshold:
        output += infer_gate_k * expert_k(x)
```

模型输出中会额外包含：

- `moe_regularization_loss`：inference router gate 总激活的正则项。
- `moe_active_ratio`：超过阈值的 expert 比例。

训练器会在 `sparse_moe_loss_weight > 0` 时把正则项加到主 loss 上：

```text
loss = multitask_bce_loss + sparse_moe_loss_weight * moe_regularization_loss
```

如果设置 `sparse_moe_target_active_ratio`，训练器会根据当前 `moe_active_ratio` 自适应调整 `sparse_moe_loss_weight`。

## 5. RankMixer Baseline

`RankMixerFromManifest` 是一个 feature-only baseline。它的输入仍然是 manifest features，但只使用：

```text
tokenization.features
tokenization.token_specs
```

前向传播流程：

```text
features
  -> FeatureTokenCompiler
       tokens: [B, F, D]
  -> L 个 RankMixerBlock
       tokens: [B, F, D]
  -> flatten
       [B, F * D]
  -> 每个任务一个 MLP head
       logits: [B, T]
```

每个 `RankMixerBlock` 的结构是：

```text
tokens = LayerNorm(tokens + RankMixerTokenMixing(tokens))
tokens = LayerNorm(tokens + PerTokenFFN(tokens))
```

与 MDL 相比，RankMixer baseline 的特点是：

- 不使用 scenario token。
- 不使用 task token。
- 不使用 domain-aware attention。
- `scenario_id` 在 forward 中会被忽略。
- 输出头默认是两层 MLP：`Linear(F * D, hidden) -> GELU -> Dropout -> Linear(hidden, 1)`。

因此它适合用来比较“只做 feature token 混合”与“引入场景/任务 token 和 domain-aware attention”的效果差异。

## 6. DeepFM Baseline

`DeepFM` 接收已经准备好的 field embeddings：

```text
field_embeddings: [B, num_fields, field_dim]
```

它包含三部分：

### 6.1 Linear 部分

先把 field embeddings flatten：

```text
flattened = field_embeddings.flatten(start_dim=1)  # [B, num_fields * field_dim]
linear_logit = Linear(num_fields * field_dim, 1)(flattened)
```

### 6.2 FM 二阶交互部分

FM 二阶项使用经典公式：

```text
summed = sum(field_embeddings, dim=1)
squared_sum = summed^2
sum_squared = sum(field_embeddings^2, dim=1)

fm_second_order = 0.5 * sum(squared_sum - sum_squared)
```

这个公式等价于枚举所有 field pair 的内积，但计算更高效。

### 6.3 Deep 部分

deep tower 是多层 MLP：

```text
Linear(input_dim, 128) -> ReLU -> Dropout
Linear(128, 64)        -> ReLU -> Dropout
Linear(64, 1)
```

最终：

```text
logits = linear_logit + fm_second_order + deep_logit
```

当前 DeepFM 文件是通用模块槽位。项目默认训练入口主要构建 `mdl` 或 `rankmixer`，如果要训练 DeepFM，需要额外提供把数据转成 dense field embeddings 的 pipeline 或封装。

## 7. 训练目标与评估

模型输出统一是：

```text
{"logits": logits}
```

如果启用 Sparse-MoE，还会包含：

```text
{
  "moe_regularization_loss": scalar,
  "moe_active_ratio": scalar
}
```

主训练目标是多任务二分类 BCE：

```text
loss = BCEWithLogits(logits, labels)
```

loss 支持以下权重：

- `label_mask`：按样本、按任务屏蔽无效标签。
- `task_weights`：按任务加权。
- `sample_weight`：按样本加权，来自 manifest 中可选的 `data_columns.sample_weight`。
- `scenario_weights`：按样本所属场景加权，多场景样本取所属场景权重平均。
- `positive_class_weights`：只对正样本加权。

最终 loss 是加权 loss 的和除以有效权重和：

```text
weighted_loss = sum(raw_bce * weights) / max(sum(weights), 1)
```

评估阶段会输出：

- 整体 task AUC。
- 按场景、按任务拆分的 scenario-task AUC。
- 加权 BCE loss。

## 8. 端到端形状示例

假设：

```text
B = 2
scenario_names = ["home", "search"]  -> S = 2
task_names = ["click", "like"]       -> T = 2
feature token 数量 F = 2
token_dim D = 16
num_layers L = 2
num_heads = 4
```

一条 batch 的 `scenario_id` 是：

```text
[0, 1]
```

则 `_scenario_mask` 会生成：

```text
[
  [1, 0],
  [0, 1]
]
```

三组 token 编译结果：

```text
feature_tokens:  [2, 2, 16]
scenario_tokens: [2, 3, 16]  # home/search/global
task_tokens:     [2, 2, 16]  # click/like
```

进入第 1 个 MDLBlock：

```text
FeatureInteraction:
  [2, 2, 16] -> [2, 2, 16]

Scenario DomainAwareAttention:
  query = scenario_tokens [2, 3, 16]
  key/value = feature_tokens [2, 2, 16]
  weights = [2, 4, 3, 2]
  update = [2, 3, 16]

Task DomainAwareAttention:
  query = task_tokens [2, 2, 16]
  key/value = feature_tokens [2, 2, 16]
  weights = [2, 4, 2, 2]
  update = [2, 2, 16]

DomainFusedModule:
  selected scenario average [2, 16]
  task_tokens [2, 2, 16]
```

第 2 个 MDLBlock 形状保持不变。最后每个 task token 分别过输出头：

```text
click_logit: [2, 1]
like_logit:  [2, 1]
logits:      [2, 2]
```

如果第一条样本属于 `home`，它的 task tokens 会融合：

```text
(home_scenario_token + global_scenario_token) / 2
```

如果第二条样本属于 `search`，它的 task tokens 会融合：

```text
(search_scenario_token + global_scenario_token) / 2
```

这就是 MDL 与普通多任务模型的核心区别：每个任务的预测不仅依赖 feature tokens，还显式注入了当前样本所属场景的 token 表示。

## 9. 模型选择建议

在当前项目中可以按实验目标选择模型：

- 使用 `mdl`：验证多场景、多任务建模，尤其关注场景差异、任务差异、跨场景泛化。
- 使用 `rankmixer`：作为 feature-only baseline，衡量 MDL 中 scenario/task token 和 domain-aware attention 的增益。
- 使用 `deepfm`：作为传统 FM + deep tower 结构参考，但需要额外接入 dense field embedding 数据路径。

常见消融实验可以通过配置开关完成：

```yaml
use_task_tokens: false
use_scenario_tokens: false
use_global_scenario_token: false
use_task_feature_interaction: false
use_scenario_feature_interaction: false
feature_backbone: attention
task_head_type: mlp
ffn_type: sparse_moe
```

这些开关可以分别回答：

- 任务 token 是否带来收益。
- 场景 token 是否带来收益。
- 全局场景 token 是否稳定跨场景共享信息。
- task-feature attention 和 scenario-feature attention 各自贡献多大。
- RankMixer mixing 与 self-attention feature backbone 哪个更合适。
- Sparse-MoE 是否能在保持效果的同时减少推理时激活 expert 数量。
