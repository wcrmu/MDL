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

### 2.3 Tokenization 阶段的完整执行路径

这里的 tokenization 不是简单地把字段 ID 做 embedding，而是一条从 CSV 字段到模型 token 的编译链路。以 `ModelFromManifest.forward(features, scenario_id)` 为例，进入模型前后会经历四层转换：

```text
CSV row
  -> ManifestDataset 逐行解析 source
  -> collate_manifest_batch 组 batch、padding 序列
  -> FeatureTokenCompiler 调用 encoder 得到 dense feature vectors
  -> token_specs 把一个或多个 dense feature vectors 投影成 token
```

这几层的职责不同：

| 阶段 | 输入 | 输出 | 主要职责 |
| --- | --- | --- | --- |
| CSV parsing | CSV 单元格字符串 | Python scalar/list | 按 `source.dtype`、`source.shape` 和 `delimiter` 解析原始值 |
| collate | 多行 Python scalar/list | Tensor 或 sequence payload | 把 scalar 堆成 tensor，把 sequence padding 成 batch 内等长 |
| feature encoding | batch features | `[B, feature_output_dim]` | embedding、数值投影、序列池化或目标注意力 |
| token projection | 若干 encoded features 拼接 | `[B, token_dim]` | 按 `token_specs.inputs` 组合特征并统一维度 |

MDL 有三套独立但机制相同的 tokenization：

```text
feature_compiler:
  features + token_specs -> feature_tokens [B, F, D]

scenario_token_compiler:
  scenario_features + scenario_token_specs -> scenario_tokens [B, S + 1, D]

task_token_compiler:
  task_features + task_token_specs -> task_tokens [B, T, D]
```

这意味着同一个原始 CSV 字段可以被普通 feature token、scenario token 和 task token 复用。只要对应的 feature spec 有 `source`，数据 reader 就会把它读进 batch；如果同名字段已经在其他 feature group 中声明过 `source`，后续同名 spec 可以复用这个 batch 字段。

### 2.4 Source 解析与 batch payload

每个需要从 CSV 读取的 feature spec 都通过 `source` 声明物理列：

```json
{
  "name": "hist_item_id",
  "encoder": "embedding",
  "vocab_size": 100000,
  "source": {
    "type": "csv_column",
    "column": "hist_item_id",
    "dtype": "int64",
    "shape": "sequence",
    "delimiter": "|"
  }
}
```

当前 reader 支持的关键字段：

- `type`：当前必须是 `csv_column`。
- `column`：CSV 中的物理列名。
- `dtype`：支持 `int`、`int64`、`long`、`float`、`float32`、`double`、`bool`、`boolean`。
- `shape`：默认是 `scalar`；序列字段使用 `sequence`，代码里也把 `vector`、`list` 当作逐行 list 解析。
- `delimiter`：序列分隔符；如果不声明，则按空白字符 split。
- `missing_value`：空字符串时使用的缺失值；不声明时 int 默认 `0`、float 默认 `0.0`、bool 默认 `False`。
- `padding_value`：batch padding 时使用的值；不声明时使用对应 dtype 的默认缺失值。

scalar 字段会直接堆成 tensor：

```text
CSV:
user_id
3
4

batch["features"]["user_id"] = LongTensor([3, 4])  # [B]
```

sequence 字段会先按行解析成 list，再在 collate 时 padding 成 batch 内最大长度：

```text
CSV:
history
1|2|3
2|5

逐行解析:
row0 -> [1, 2, 3]
row1 -> [2, 5]

collate 后:
batch["features"]["history"] = {
  "values":  LongTensor([[1, 2, 3],
                         [2, 5, 0]]),
  "lengths": LongTensor([3, 2])
}
```

encoder 侧统一通过 `sequence_values_and_mask(payload)` 取序列值和 mask：

```text
如果 payload 是 Tensor:
  values = payload
  mask = values != 0

如果 payload 是 dict 且包含 mask:
  values = payload["values"]
  mask = payload["mask"]

如果 payload 是 dict 且包含 lengths:
  values = payload["values"]
  mask[b, pos] = pos < lengths[b]

否则:
  mask = values != 0
```

因此，通过当前通用 CSV reader 进入模型的序列默认是 `values + lengths`。padding 位置即使值是 `0`，也主要由 `lengths` 生成的 mask 控制；手工构造 batch 时也可以直接传 `mask`。

### 2.5 Feature encoder 总览

当前 encoder registry 支持以下输入编码：

| encoder | 输入 | 输出 | 说明 |
| --- | --- | --- | --- |
| `embedding` | 类别 ID `[B]` | `[B, embedding_dim]` | 使用 `nn.Embedding`，`padding_idx=0` |
| `identity` | 数值特征 `[B]` 或 `[B, dim]` | `[B, dim]` | 直接转 float |
| `sequence_mean_pooling` | 序列字段 | `[B, sum(field_dims)]` | 对有效位置做 mask mean pooling |
| `din` | 序列字段 + target feature | `[B, sum(field_dims)]` | 使用 DIN activation unit 做目标感知加权 |
| `sim` / `longer` | 长序列字段 + target feature | `[B, sum(field_dims)]` | 先用 target 相似度取 top-k，再执行 DIN 加权 |

普通 scalar encoder 的输出很直接：

```text
embedding:
  input  user_id: [B]
  output user_emb: [B, embedding_dim]

identity:
  input  price: [B] 或 [B, dim]
  output price_value: [B, dim]
```

序列 encoder 的输出不是 `[B, L, D]`，而是已经聚合好的 `[B, output_dim]`。随后它会像普通 dense feature 一样被 `token_specs` 引用，并投影为 token。

### 2.6 单字段序列的编码流程

最简单的序列配置可以把 feature 本身声明为序列：

```json
{
  "name": "history_items",
  "encoder": "sequence_mean_pooling",
  "vocab_size": 100000,
  "source": {
    "type": "csv_column",
    "column": "history_items",
    "dtype": "int64",
    "shape": "sequence",
    "delimiter": "|"
  }
}
```

内部会被 `_sequence_field_specs` 展开为一个序列字段：

```text
sequence field:
  name = history_items
  encoder = sequence_encoder or embedding
  vocab_size = 100000
```

如果没有显式写 `sequence_encoder`，默认使用 `embedding`。于是计算过程是：

```text
payload = {
  values:  [B, Lmax],
  lengths: [B]
}

values, mask = sequence_values_and_mask(payload)
step_embeddings = Embedding(values)       # [B, Lmax, embedding_dim]
masked_steps = step_embeddings * mask     # padding 位置清零
output = sum(masked_steps, dim=1) / max(sum(mask), 1)
```

例子：

```text
history_items:
  row0 = 1|2|3
  row1 = 4

values = [[1, 2, 3],
          [4, 0, 0]]
lengths = [3, 1]
mask = [[1, 1, 1],
        [1, 0, 0]]

row0_output = (emb(1) + emb(2) + emb(3)) / 3
row1_output = emb(4) / 1
```

如果序列是数值序列，可以使用 `sequence_encoder: "identity"` 或在多字段形式中写 `encoder: "identity"`。数值序列会先变成 `[B, L, dim]`，再按需要用 `projection_dim` 投影到新的 step 维度。

### 2.7 多字段序列的编码流程

真实行为序列通常每个 step 有多个字段，例如：

```text
第 k 个历史行为 = (hist_item_id[k], hist_cate_id[k], hist_price[k])
```

这类配置使用 `sequence_features`：

```json
{
  "name": "history_behavior",
  "encoder": "sequence_mean_pooling",
  "fusion": "concat",
  "sequence_features": [
    {
      "name": "hist_item_id",
      "encoder": "embedding",
      "vocab_size": 100000,
      "embedding_dim": 8,
      "source": {"type": "csv_column", "column": "hist_item_id", "dtype": "int64", "shape": "sequence", "delimiter": "|"}
    },
    {
      "name": "hist_cate_id",
      "encoder": "embedding",
      "vocab_size": 5000,
      "embedding_dim": 4,
      "source": {"type": "csv_column", "column": "hist_cate_id", "dtype": "int64", "shape": "sequence", "delimiter": "|"}
    },
    {
      "name": "hist_price",
      "encoder": "identity",
      "dim": 1,
      "projection_dim": 2,
      "source": {"type": "csv_column", "column": "hist_price", "dtype": "float32", "shape": "sequence", "delimiter": "|"}
    }
  ]
}
```

每个子字段先独立编码：

```text
hist_item_id -> item_step_emb:  [B, L, 8]
hist_cate_id -> cate_step_emb:  [B, L, 4]
hist_price   -> price_step_vec: [B, L, 2]
```

然后按最后一维拼接：

```text
sequence_embeddings = concat([item_step_emb, cate_step_emb, price_step_vec], dim=-1)
                    = [B, L, 14]
```

当前 `fusion` 只支持 `concat`。多字段序列还有两个重要约束：

- 所有 `sequence_features` 在同一个样本中应表示同一条行为序列，因此 batch 后的 padded shape 必须一致，例如都是 `[B, L]`。
- 多个字段的 mask 会逐位取 AND；只要某个字段在该 step 无效，该 step 就整体无效。

对于上面的例子，`history_behavior.output_dim = 8 + 4 + 2 = 14`。它输出的不是每个 step 的表示，而是经过 pooling 或 attention 后的 `[B, 14]` 向量。

### 2.8 sequence_mean_pooling 细节

`sequence_mean_pooling` 的目标是把变长行为序列压缩成一个固定维度向量。单字段和多字段都会先得到：

```text
sequence_embeddings: [B, L, E]
mask:                [B, L]
```

其中 `E` 是所有 step 字段编码维度之和。然后执行 masked mean：

```text
weighted = sequence_embeddings * mask.unsqueeze(-1)
denominator = max(mask.sum(dim=1), 1)
output = weighted.sum(dim=1) / denominator
```

如果某条样本的序列为空，`denominator` 会被 clamp 到 1，输出为零向量或 padding 后对应的零贡献，避免除零。

一个多字段例子：

```text
hist_item_id = 1|2|3
hist_cate_id = 7|7|8
hist_price   = 0.1|0.2|0.3

step0 = concat(item_emb(1), cate_emb(7), price_proj(0.1))
step1 = concat(item_emb(2), cate_emb(7), price_proj(0.2))
step2 = concat(item_emb(3), cate_emb(8), price_proj(0.3))

history_behavior = (step0 + step1 + step2) / 3
```

这个输出之后可以被 token 引用：

```json
{"token_id": 1, "projection": "linear", "inputs": ["history_behavior"]}
```

如果 `history_behavior.output_dim=14`、`token_dim=16`，这个 token 的 projection 就是：

```text
Linear(14, 16)(history_behavior) -> token_1 [B, 16]
```

### 2.9 DIN 序列编码细节

`din` 用于目标感知的行为序列建模。它不仅看历史行为本身，还看当前候选 item、类目、价格等 target feature，让不同候选样本对同一段历史产生不同的权重。

DIN 配置示例：

```json
{
  "name": "history_behavior",
  "encoder": "din",
  "fusion": "concat",
  "attention_hidden_dims": [80, 40],
  "activation": "dice",
  "attention_normalization": "none",
  "sequence_features": [
    {
      "name": "hist_item_id",
      "target_feature": "item_id",
      "encoder": "embedding",
      "vocab_size": 100000,
      "embedding_dim": 8,
      "source": {"type": "csv_column", "column": "hist_item_id", "dtype": "int64", "shape": "sequence", "delimiter": "|"}
    },
    {
      "name": "hist_cate_id",
      "target_feature": "cate_id",
      "encoder": "embedding",
      "vocab_size": 5000,
      "embedding_dim": 4,
      "source": {"type": "csv_column", "column": "hist_cate_id", "dtype": "int64", "shape": "sequence", "delimiter": "|"}
    },
    {
      "name": "hist_price",
      "target_feature": "price",
      "encoder": "identity",
      "dim": 1,
      "projection_dim": 2,
      "source": {"type": "csv_column", "column": "hist_price", "dtype": "float32", "shape": "sequence", "delimiter": "|"}
    }
  ]
}
```

每个 `sequence_features[*].target_feature` 必须能在当前 batch features 中找到。ID 类历史字段和 target 字段通常应该共用同一 vocab，因为它们会经过同一个 field encoder 的 embedding 表。例如：

```text
hist_item_id 使用 vocab_size=100000 的 embedding
target_feature=item_id 也通过这张 embedding 表编码
```

#### Target side info 缺失时的当前行为

当前实现不会自动补齐 target side info，也不会在 target side info 缺失时自动退化成只用 ID。规则是：DIN 里声明了几个历史字段，就要为这几个历史字段分别声明可用的 target 字段。

具体行为如下：

- 如果某个 `sequence_features[*]` 没有声明 `target_feature`，模型构建阶段会直接报错：`din sequence field '<name>' must declare target_feature`。
- 如果声明了 `target_feature`，但 forward 时 batch 中没有这个字段，会报错：`din sequence field '<name>' target_feature '<target>' is missing from batch features`。
- 如果 embedding target 存在但不是 scalar，即 shape 不是 `[B]` 或 `[B, 1]`，会报错：`din target_feature '<target>' must be scalar`。
- 如果 numeric target 存在但不是 `[B, dim]`，或最后一维与 `dim` 不一致，也会报错。
- 只有历史序列字段本身缺失时，DIN encoder 会返回该 DIN feature 的零向量；这不适用于 target 缺失。

因此，只有候选 item ID 而没有候选类目、价格等 side info 时，不应该把 `hist_cate_id`、`hist_price` 也放进同一个 DIN `sequence_features` 中。可行配置是只保留有 target 对应项的历史字段：

```json
{
  "name": "history_items",
  "encoder": "din",
  "vocab_size": 100000,
  "target_feature": "item_id",
  "source": {"type": "csv_column", "column": "history_items", "dtype": "int64", "shape": "sequence", "delimiter": "|"}
}
```

或者使用显式单字段形式：

```json
{
  "name": "history_behavior",
  "encoder": "din",
  "sequence_features": [
    {
      "name": "hist_item_id",
      "target_feature": "item_id",
      "encoder": "embedding",
      "vocab_size": 100000,
      "source": {"type": "csv_column", "column": "hist_item_id", "dtype": "int64", "shape": "sequence", "delimiter": "|"}
    }
  ]
}
```

如果确实希望历史类目、价格参与 DIN，当前数据中就必须提供对应 target 字段，例如 `cate_id`、`price`，并保证这些字段被 reader 读入 batch。target 字段可以来自普通 `features` 中同名 spec 的 `source`，也可以在 sequence field 中通过对象形式的 `target_feature` 或 `target_source` 单独声明。

DIN 的前向计算分为五步。

第一步，编码历史序列每个 step：

```text
hist_item_id -> [B, L, 8]
hist_cate_id -> [B, L, 4]
hist_price   -> [B, L, 2]

sequence_embeddings = concat(...) -> [B, L, 14]
```

第二步，用相同字段编码逻辑编码 target：

```text
item_id -> [B, 8]
cate_id -> [B, 4]
price   -> [B, 2]

target_embeddings = concat(...) -> [B, 14]
```

第三步，把 target 扩展到每个历史位置，并构造 activation unit 输入：

```text
expanded_target = target_embeddings.unsqueeze(1).expand_as(sequence_embeddings)

activation_input = concat([
  sequence_embeddings,
  expanded_target,
  sequence_embeddings - expanded_target,
  sequence_embeddings * expanded_target,
], dim=-1)

activation_input: [B, L, 4 * 14]
```

第四步，activation unit 输出每个历史 step 的权重分数：

```text
scores = MLP(activation_input).squeeze(-1)  # [B, L]
```

MLP 的隐藏层来自 `attention_hidden_dims`，激活函数由 `activation` 控制，支持：

- `dice`：默认值，使用 DIN 常见的 Dice activation。
- `prelu`。
- `relu`。

第五步，按 mask 加权求和：

```text
if attention_normalization == "softmax":
    scores = scores.masked_fill(~mask, -1e9)
    weights = softmax(scores, dim=1)
    weights = weights * mask
    weights = weights / max(weights.sum(dim=1), 1e-8)
else:
    weights = scores * mask

output = sum(sequence_embeddings * weights.unsqueeze(-1), dim=1)  # [B, 14]
```

注意：当前默认 `attention_normalization="none"`，也就是不做 softmax，直接使用 activation unit 的原始分数乘 mask 后加权求和。只有显式配置 `softmax` 时，才会把有效历史位置归一化成概率分布。

DIN 输出仍然是一个普通 dense feature，可以在 `token_specs` 中和其他特征一起组成 token：

```json
{
  "token_id": 2,
  "projection": "linear",
  "inputs": ["item_id", "history_behavior", "price"]
}
```

如果 `item_id` 输出 8 维、`history_behavior` 输出 14 维、`price` 输出 1 维，那么 token projection 的输入维度就是 23：

```text
concat(item_emb, history_behavior, price) -> [B, 23]
Linear(23, token_dim) -> [B, token_dim]
```

### 2.10 SIM / Longer 序列编码细节

`sim` 和 `longer` 继承 DIN 的字段编码和 activation unit，但在 DIN 加权前增加了 top-k 检索步骤，适合更长的历史序列。

配置上与 DIN 类似，多一个 `top_k` 或 `search_top_k`：

```json
{
  "name": "history_items",
  "encoder": "sim",
  "vocab_size": 100000,
  "target_feature": "item_id",
  "attention_hidden_dims": [80, 40],
  "attention_normalization": "softmax",
  "top_k": 50,
  "source": {
    "type": "csv_column",
    "column": "history_items",
    "dtype": "int64",
    "shape": "sequence",
    "delimiter": "|"
  }
}
```

内部流程是：

```text
1. 和 DIN 一样得到：
   sequence_embeddings: [B, L, E]
   target_embeddings:   [B, E]
   mask:                [B, L]

2. 用点积做粗排检索：
   search_scores = sum(sequence_embeddings * target_embeddings.unsqueeze(1), dim=-1)
                 = [B, L]

3. padding 位置设成很小的分数：
   masked_search_scores = search_scores.masked_fill(~mask, -1e9)

4. 取 top_k 个历史位置：
   k = min(search_top_k, L)
   top_indices = topk(masked_search_scores, k)

5. gather 出 selected_embeddings 和 selected_mask：
   selected_embeddings: [B, k, E]
   selected_mask:       [B, k]

6. 在 top-k 子序列上执行 DIN activation unit 和 weighted sum：
   output: [B, E]
```

`longer` 当前实现与 `sim` 相同，是 `SIMSequenceEncoder` 的别名子类。

### 2.11 序列特征进入 token_specs 的方式

无论使用 `sequence_mean_pooling`、`din`、`sim` 还是 `longer`，encoder 的最终输出都是 `[B, output_dim]`。从 `FeatureTokenCompiler` 看，它和普通 embedding/identity 没有区别：

```text
encoded_features["history_behavior"] = sequence_encoder(batch)  # [B, E]
```

然后 token_specs 决定如何把它变成 token：

```json
{
  "token_specs": [
    {"token_id": 0, "projection": "linear", "inputs": ["user_id"]},
    {"token_id": 1, "projection": "linear", "inputs": ["item_id", "history_behavior"]},
    {"token_id": 2, "projection": "ffn_relu", "inputs": ["history_behavior", "price"]}
  ]
}
```

对应的形状可能是：

```text
user_id          -> [B, 8]
item_id          -> [B, 8]
history_behavior -> [B, 14]
price            -> [B, 1]

token 0 input = user_id                      -> [B, 8]  -> Linear(8, D)
token 1 input = concat(item_id, history)     -> [B, 22] -> Linear(22, D)
token 2 input = concat(history, price)       -> [B, 15] -> FFN(15, hidden, D)

feature_tokens = concat token 0/1/2 on dim=1 -> [B, 3, D]
```

对于 MDL，scenario tokens 和 task tokens 也完全遵守同样规则。也就是说，序列特征不仅可以进入普通 feature tokens，也可以作为场景 token 或任务 token 的输入。例如：

```json
{
  "scenario_token_specs": [
    {"token_id": 0, "inputs": ["user_profile", "history_behavior"]},
    {"token_id": 1, "inputs": ["search_context"]},
    {"token_id": 2, "inputs": ["user_profile", "history_behavior", "search_context"]}
  ],
  "task_token_specs": [
    {"token_id": 0, "inputs": ["item_id", "history_behavior"]},
    {"token_id": 1, "inputs": ["item_id", "price"]}
  ]
}
```

这里 `scenario_token_specs` 的数量仍然必须是 `num_scenarios + 1`，`task_token_specs` 的数量仍然必须是 `num_tasks`。序列 encoder 只负责把历史压缩成 dense feature；最终它影响哪个 token、影响多少 token，完全由这些 token specs 决定。

### 2.12 序列 tokenization 的完整例子

假设一条样本包含当前候选商品和用户历史：

```csv
item_id,cate_id,price,hist_item_id,hist_cate_id,hist_price
3,2,0.2,1|2|3,2|2|1,0.1|0.2|0.3
```

manifest 中声明：

```json
{
  "features": [
    {"name": "item_id", "encoder": "embedding", "vocab_size": 20, "embedding_dim": 8,
     "source": {"type": "csv_column", "column": "item_id", "dtype": "int64"}},
    {"name": "cate_id", "encoder": "embedding", "vocab_size": 8, "embedding_dim": 4,
     "source": {"type": "csv_column", "column": "cate_id", "dtype": "int64"}},
    {"name": "price", "encoder": "identity", "dim": 1,
     "source": {"type": "csv_column", "column": "price", "dtype": "float32"}},
    {
      "name": "history_behavior",
      "encoder": "din",
      "fusion": "concat",
      "attention_hidden_dims": [8],
      "sequence_features": [
        {"name": "hist_item_id", "target_feature": "item_id", "encoder": "embedding", "vocab_size": 20, "embedding_dim": 8,
         "source": {"type": "csv_column", "column": "hist_item_id", "dtype": "int64", "shape": "sequence", "delimiter": "|"}},
        {"name": "hist_cate_id", "target_feature": "cate_id", "encoder": "embedding", "vocab_size": 8, "embedding_dim": 4,
         "source": {"type": "csv_column", "column": "hist_cate_id", "dtype": "int64", "shape": "sequence", "delimiter": "|"}},
        {"name": "hist_price", "target_feature": "price", "encoder": "identity", "dim": 1, "projection_dim": 2,
         "source": {"type": "csv_column", "column": "hist_price", "dtype": "float32", "shape": "sequence", "delimiter": "|"}}
      ]
    }
  ],
  "token_specs": [
    {"token_id": 0, "projection": "linear", "inputs": ["item_id"]},
    {"token_id": 1, "projection": "linear", "inputs": ["item_id", "history_behavior", "price"]}
  ]
}
```

tokenization 阶段的形状流是：

```text
CSV 解析:
  item_id = 3
  cate_id = 2
  price = 0.2
  hist_item_id = [1, 2, 3]
  hist_cate_id = [2, 2, 1]
  hist_price = [0.1, 0.2, 0.3]

collate:
  item_id: LongTensor[B]
  cate_id: LongTensor[B]
  price: FloatTensor[B]
  hist_item_id: {values: LongTensor[B, L], lengths: LongTensor[B]}
  hist_cate_id: {values: LongTensor[B, L], lengths: LongTensor[B]}
  hist_price: {values: FloatTensor[B, L], lengths: LongTensor[B]}

feature encoding:
  item_id -> [B, 8]
  cate_id -> [B, 4]
  price -> [B, 1]
  history_behavior:
    sequence side = concat([B, L, 8], [B, L, 4], [B, L, 2]) -> [B, L, 14]
    target side   = concat([B, 8],    [B, 4],    [B, 2])    -> [B, 14]
    DIN weighted sum -> [B, 14]

token projection:
  token 0 input = item_id -> [B, 8]
  token 0 = Linear(8, D) -> [B, D]

  token 1 input = concat(item_id, history_behavior, price) -> [B, 23]
  token 1 = Linear(23, D) -> [B, D]

final:
  feature_tokens = stack([token0, token1], dim=1) -> [B, 2, D]
```

这个例子里，DIN 的 target 是当前候选商品、类目和价格；因此同一个用户历史在不同候选 item 下会得到不同的 `history_behavior` 向量，进而影响后续 token 和最终预测。

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
