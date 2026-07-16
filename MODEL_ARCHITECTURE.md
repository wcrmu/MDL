# 当前模型设计、默认配置与评估结果记录

本文档描述当前工作区中已经实现的推荐模型、公共输入与训练栈、版本化默认配置，以及离线质量和性能结果的填写位置。内容以当前的 [src/model.py](src/model.py)、[src/config.py](src/config.py)、[src/train.py](src/train.py) 和 [configs/](configs/) 为准。

本文档中的“默认”有三种不同含义，阅读时必须区分：

1. **代码字段缺省值**：在 YAML 没有提供某个可选字段时，由 [src/config.py](src/config.py) 中的 dataclass 使用。
2. **默认模板**：仓库实际入口 [configs/default.yaml](configs/default.yaml)，覆盖了许多代码字段缺省值，是一个小规模、可快速验证的 MDL-RankMixer 模板。
3. **论文配置**：各个 configs/\*_paper.yaml，尽量固定论文公开的方法与超参数；论文没有公开的部分会明确标记为实现选择。

所有版本化模型配置都使用 extends 继承。映射递归合并，列表整体替换而不是追加。因此，一个子配置只要重写 features、sequences 或 token 列表，就必须写出希望保留的完整列表。

---

## 1. 当前模型入口

build_model 当前支持五个 model.name：

| model.name | 定位 | 主干 | 输出表示 | 推荐配置入口 |
|---|---|---|---|---|
| rankmixer | 独立 RankMixer 基线 | 特征 token 的无参数 TokenMixing + per-token FFN | 所有特征 token 均值池化 | [configs/rankmixer.yaml](configs/rankmixer.yaml)、[configs/rankmixer_paper.yaml](configs/rankmixer_paper.yaml) |
| mdl_rankmixer | 当前默认模型；MDL 多场景、多任务模型 | RankMixer 特征主干 + scenario/task domain interaction | 每个任务 token 单独进入任务头 | [configs/default.yaml](configs/default.yaml)、[configs/mdl_rankmixer_paper.yaml](configs/mdl_rankmixer_paper.yaml) |
| onetrans | 独立 OneTrans | 统一 S/NS token、mixed-parameter causal block、pyramid | 最终全部 NS token 展平 | [configs/onetrans.yaml](configs/onetrans.yaml)、[configs/onetrans_paper.yaml](configs/onetrans_paper.yaml) |
| longer | 独立 LONGER | 事件 MLP、TokenMerge、InnerTrans、global/recent mixed attention | global token 与 recent query 输出展平 | [configs/longer.yaml](configs/longer.yaml)、[configs/longer_paper.yaml](configs/longer_paper.yaml) |
| mdl_onetrans | 实验性组合，不是论文定义模型 | 每层 OneTrans 后接双路 MDL domain interaction：固定 NS memory + 可变长 masked S memory | MDL task token | [configs/mdl_onetrans.yaml](configs/mdl_onetrans.yaml) |

mdl_onetrans 必须显式设置 experimental_model_acknowledged: true。它只能用于清楚标注的工程实验，不能写成 MDL 或 OneTrans 论文方法的复现结果。

---

## 2. 端到端公共架构

当前五个模型共用同一条数据、特征、训练和评估链路：

~~~text
Parquet / adapter_parquet
        │
        ▼
按 YAML 读取标量列、序列 list 列、label、mask、scenario、group_id
        │
        ▼
FeatureBatch
  ├─ categorical: vocab / hash / identity / shared_vocab
  ├─ dense: float tensor
  ├─ sequence: 对齐字段、截断、方向归一化、padding mask
  ├─ labels: [B, M]
  └─ scenario_id: [B] 或多热 [B, S]
        │
        ▼
FeatureEncoderBank
  ├─ 标量 embedding / dense passthrough
  ├─ raw event inputs
  ├─ mean/attention pooled sequence
  └─ LONGER sequence representation
        │
        ▼
模型特定 tokenization 与 backbone
        │
        ▼
每任务独立 TaskHead
        │
        ▼
logits [B, M] → masked BCEWithLogits → AUC / QAUC / UAUC
~~~

其中：

- B 是 batch size。
- M 是训练 labels 的任务数，任务顺序严格等于 data.train.labels 的 YAML 顺序。
- S 是 scenarios.names 的场景数，不包含 MDL 内部可选的 global scenario token。
- 所有模型最终都返回 logits，形状必须与 labels 完全一致。
- rankmixer、onetrans、longer 不使用 scenario_id 改变预测；mdl_rankmixer 和 mdl_onetrans 使用它选择或融合场景表示。
- 一维 scenario_id 即使由浮点 tensor 提供，也必须是有限且精确的整数值；0.9、-0.1、NaN 和 Inf 会在转成 long 前报错。二维场景 mask 必须严格为 0/1。

### 2.1 标量特征

- categorical 特征进入 embedding。
- dense 特征保持浮点向量，dimension 必须与输入宽度一致。
- embedding_scope 控制特征语义：
  - feature：普通特征 token 或 OneTrans NS token。
  - scenario：MDL scenario token 专用输入或 prior。
  - task：MDL task token 专用输入或 prior。
  - shared：可以被多个 token 家族复用。

### 2.2 类别编码和 embedding 共享

当前支持：

| encoding | 语义 |
|---|---|
| vocab | 从训练 split 拟合或加载显式词表 |
| hash | 使用固定 bucket 和 salt 做稳定哈希 |
| identity | 输入已经是有界整数 ID，直接索引 |
| shared_vocab | 与 share_with 使用相同值到 ID 的映射 |

share_embedding 与 shared_vocab 是两个不同概念：

- share_embedding: true 表示实际复用同一张 embedding 权重表。
- share_embedding: false 表示只共享 ID 映射，仍建立独立参数表。
- 独立表的行数从 shared_vocab 链最终解析到的根 ID/vocab 命名空间计算，不按 alias 名称读取可能缺失或过期的映射。

MDL 的 important ID 默认采用后一种方式：原始 user/item 值与普通特征使用同一套词表映射，但 scenario/task token 拥有独立 embedding 权重。

### 2.3 序列公共语义

每条序列由 fields 中多个对齐的 list 列组成。同一样本中这些字段必须具有相同事件数。

- truncation 针对 Parquet 中的物理列表方向。
- sequence_order 声明物理列表是 oldest_to_newest 还是 newest_to_oldest。
- 模型在 causal attention 前统一转换为 oldest_to_newest。
- 序列在 batch 中右对齐，padding 位于有效事件之前。
- categorical 序列字段进入 embedding，dense 序列字段按 dimension 校验后直接拼接。
- 同一个 OneTrans S-token group 内的多个序列必须具有逐样本一致的逻辑长度；各序列和 timestamp 先对齐到组级最大长度，再沿最后一维拼接，因此允许成员声明不同的 max_length。
- encoder=raw 仅允许 OneTrans 家族使用。
- encoder=longer 要求固定 max_length 和标量 dense time_delta_field。

---

## 3. 公共任务头与训练目标

每个任务有一个独立两层 MLP：

~~~text
输入表示
  → Linear(input_dim, task_head_hidden_dim)
  → GELU 或 ReLU
  → Dropout
  → Linear(hidden_dim, 1)
  → logit
~~~

默认模板中 task_head_hidden_dim=64、dropout=0、activation=gelu。

训练使用逐元素 BCEWithLogits，并乘 label_mask。loss_reduction 支持：

- sum：对所有有效样本和任务直接求和。这是默认值，也是 MDL 公式对应路径。DDP 下会补偿 DDP 的跨 rank 梯度平均，使梯度等价于全局求和。
- mean_per_task：每个任务先除以自己的全局有效标签数，再把任务均值相加，适合显式需要任务平衡的工程实验。

如果 RankMixer FFN 使用 Sparse-MoE，总损失还可以加入：

~~~text
total_loss = prediction_loss
           + sparse_moe_loss_weight × moe_regularization_loss
~~~

---

## 4. RankMixer

### 4.1 特征 tokenization

RankMixer 家族支持三种特征 tokenizer：

#### groupwise

每个 YAML 语义组独立拼接并线性投影：

~~~text
x_t = Linear_t(concat(encoded[input] for input in group_t))
X ∈ R[B, T, D]
~~~

这是 default.yaml 和 mdl_rankmixer_paper.yaml 使用的模式，便于明确区分 user、item、context 和 history。

#### rankmixer

先按 YAML 顺序拼接全部输入，均匀切成 T 份，再由每个切片自己的线性层投影到 D：

~~~text
z = concat(all ordered encoded inputs) ∈ R[B, W]
z → reshape [B, T, W/T]
x_t = Linear_t(z_t) ∈ R[D]
~~~

要求 W 能被 T 整除。rankmixer_paper.yaml 使用这一模式。

#### auto_split

使用一个共享线性层把完整输入投影成 T×D，再 reshape 为 T 个 token。这是工程兼容路径，不是 RankMixer paper profile 的默认 tokenizer。

### 4.2 TokenMixing

输入 X 的形状为 [B, T, D]，要求 D 能被 T 整除。实现把最后一维拆成 T 个 head，每个 head 宽度为 D/T，然后交换“输入 token 轴”和“head 轴”：

~~~text
X: [B, T, D]
 → view [B, T, T, D/T]
 → permute [B, T(head), T(token), D/T]
 → view [B, T, D]
~~~

这个操作没有 Q/K/V、softmax 或可学习参数。它通过确定性的 reshape/permute 在 token 之间交换子空间。

### 4.3 RankMixer Block

独立 RankMixer 每层保留两组残差和 LayerNorm：

~~~text
M_l = LayerNorm(TokenMixing(X_l-1) + X_l-1)
X_l = LayerNorm(PerTokenFFN(M_l) + M_l)
~~~

PerTokenFFN 为每个 token 使用独立权重。dense 路径使用两个 batched GEMM 实现；输出数学上仍是互相独立的 token FFN。

### 4.4 Sparse-MoE 可选路径

rankmixer_ffn_type=sparse_moe 时，每个 token 使用自己的 router 和多专家 FFN：

- router 使用 ReLU sparse gate。
- 训练时保留 dense softmax router，使所有专家可以获得梯度。
- DTSI 打开时，训练输出策略必须显式选择 dense_router 或 mean，因为论文未公开 DTSI 的训练输出融合公式。
- 推理只执行 gate 超过 inference_threshold 的专家。
- target_active_ratio 和自适应 L1 系数控制期望激活比例。

默认模板使用 dense，不会启用 MoE。

### 4.5 输出

经过 L 层后，对 T 个特征 token 做 mean pooling：

~~~text
h = mean(X_L, dim=token)
logit_m = TaskHead_m(h)
~~~

因此独立 RankMixer 不包含 scenario token、task token 或 domain-aware attention。

---

## 5. MDL-RankMixer

mdl_rankmixer 是当前 default.yaml 的默认模型。它同时维护三类状态：

| 状态 | 形状 | 来源 |
|---|---|---|
| feature tokens F | [B, T, D] | groupwise/rankmixer/auto_split tokenizer |
| scenario tokens S | [B, S(+global), D] | 每个场景独立 DomainTokenProjector |
| task tokens Q | [B, M, D] | 每个任务独立 DomainTokenProjector |

### 5.1 Scenario/task token 初始化

每个 domain token 有独立 MLP：

~~~text
domain_token =
  ReLU(
    Linear(
      activation(
        Linear(concat(inputs, important_inputs, prior_inputs))
      )
    )
  )
~~~

- important_inputs 一般是独立的 user/item embedding。
- prior_inputs 一般是场景相关或任务相关的历史表示。
- 不同 token 使用不同 projector 参数。
- 多场景或多任务 paper profile 要求每个非 global token 至少有一个与自身 domain scope 对应、且不被其他同类 token 复用的 prior。
- prior 必须只来自预测时刻以前的行为，不能从当前行 label 构造。

### 5.2 Feature interaction

每个 MDL-RankMixer block 先做：

~~~text
M_l = LayerNorm(TokenMixing(F_l-1) + F_l-1)
U_l = PerTokenFFN(M_l)
~~~

当前默认 mdl_feature_interaction=direct_ffn：

~~~text
F_l = U_l
~~~

也就是 MDL 论文式 feature self-interaction 路径没有第二个残差和 LayerNorm。

显式消融 residual_ffn 使用：

~~~text
F_l = LayerNorm(U_l + M_l)
~~~

这是原始 RankMixer block 风格的第二个 Add & Norm。兼容旧 YAML 时，paper 会映射为 direct_ffn，rankmixer_full 会映射为 residual_ffn；新配置应直接使用新名称。

### 5.3 Domain-aware attention

scenario 和 task token 分别作为 query，feature token 作为 key/value。Q/K/V 不是共享线性层，而是每个 domain/feature token 各自拥有 PerTokenFFN：

~~~text
S_hat = S_l-1 + MultiHeadAttention(
  Q=PerTokenFFN_s(S_l-1),
  K=PerTokenFFN_f(F_l),
  V=PerTokenFFN_f(F_l)
)

S_l = S_hat + PerTokenFFN_scenario(S_hat)
~~~

task 路径先计算：

~~~text
Q_hat = Q_l-1 + MultiHeadAttention(
  Q=PerTokenFFN_t(Q_l-1),
  K=PerTokenFFN_f(F_l),
  V=PerTokenFFN_f(F_l)
)
~~~

随后将当前样本所属场景的 S_hat 与可选 global token 做非加权均值，并加到每个 task token：

~~~text
scenario_context = mean(selected scenario S_hat and optional global S_hat)
Q_fused = Q_hat + scenario_context
Q_l = Q_fused + PerTokenFFN_task(Q_fused)
~~~

scenario_id 可以是单个场景整数，也可以是多热场景 mask。单值 ID 必须是有限、整数值且位于 `[0, scenario_count)`；不会把小数静默截断到其他场景。多热 mask 必须严格二值，所有激活场景与 global token 一起做均值。

### 5.4 Interaction 消融

use_scenario_feature_interaction=false 或 use_task_feature_interaction=false 不表示把更新置零，而是用 RankMixerDomainInteraction 替代 attention：

1. 拼接 [feature; domain] token。
2. 如果 D 不能被总 token 数整除，右侧补零到最近的合法宽度。
3. 执行 RankMixer TokenMixing、残差和 LayerNorm。
4. 只保留 domain token 对应的输出并裁回 D。

因此关闭 attention 后，domain token 仍然依赖 feature token。

### 5.5 Token 消融与输出

| 开关 | false 时的真实行为 |
|---|---|
| use_task_tokens | 删除 task projector、task attention、task FFN；最终使用每任务独立顶层 MLP tower |
| use_scenario_tokens | 删除 scenario projector、scenario attention、scenario FFN；使用每场景独立 ScenarioTower，并按 scenario mask 融合 |
| use_global_scenario_token | 真正删除 global token 及其 projector/FFN 参数 |

正常完整路径使用最终 task token：

~~~text
logit_m = TaskHead_m(Q_L[:, m, :])
~~~

当 task token 被删除时，任务头改为消费最终 mean-pooled feature 表示，并按配置加入 scenario token 或 ScenarioTower 上下文。

### 5.6 末层 scenario FFN

完整 MDL 路径在最后一层仍创建并执行 scenario FFN，以保持所有 block 的传播公式一致。正常 use_task_tokens=true 时，task 融合使用 FFN 前的 S_hat；最后一层 FFN 后的 scenario 状态不再影响 logits，因此这组末层参数可能没有梯度。默认 DDP 设置 find_unused_parameters=true 正是为了安全处理这种结构。

---

## 6. OneTrans

### 6.1 S token 与 NS token

OneTrans 不先把行为序列编码成 pooled/LONGER 表示。所有 encoder=raw 的事件直接进入 S tokenizer：

~~~text
每个事件：
  categorical field embedding
  + dense side fields
  + 可选同组标量上下文
  → concat
  → 两层 MLP
  → 一个 S token
~~~

NS token 表示非序列特征，支持：

- auto_split：把所有 feature/shared 标量编码拼接，经一个 MLP 一次生成 N 个 NS token。
- groupwise：每个 ns_tokens 语义组使用自己的 MLP。

### 6.2 多序列融合

sequence_fusion 支持：

- timestamp_aware：给不同 S-token group 加 learned type embedding，按真实 timestamp 做稳定全局排序。
- intent_ordered：按 YAML group 顺序拼接；多个 group 之间可插入 learned SEP token。

timestamp_aware 要求每个参与的序列声明 timestamp_field。没有跨序列可比较时间戳时应使用 intent_ordered。

一个 S-token group 可以把多个行为序列和标量上下文融合成同一事件 token。组内序列即使配置了不同 max_length，也统一使用组级最大长度进行右对齐；组内逐样本逻辑长度不一致仍会报错，避免把不同事件错误拼成一个 step。timestamp_aware 的 timestamp 使用同一目标长度。

### 6.3 统一位置编码

S 与 NS 拼成 [S; NS] 后、进入第一层前，加入同一张 learned absolute position embedding：

- padding 仍保持在前缀，但 padding 位置的 position input 被清零。
- 位置 ID 只按有效 token 累计，不受同 batch 其他样本 padding 长度影响。
- 第一个 NS token 的逻辑位置紧跟最后一个有效 S token。
- max_position_embeddings 为空时，只要所有 S 序列都有 max_length，配置层会精确推导最大 S+NS 容量。

learned absolute 是当前明确的实现选择；论文图示包含 Pos Emb，但未公开其精确形式。

### 6.4 Mixed-parameter causal block

统一 token 序列顺序为 [S; NS]。每层使用 pre-RMSNorm：

~~~text
H = X_query + MixedCausalAttention(RMSNorm(X))
Y = H + MixedFFN(RMSNorm(H))
~~~

参数共享规则：

- 所有 S token 共享一套 Q/K/V 和 FFN。
- 每个 NS token 各自拥有独立 Q/K/V 和 FFN。
- 所有 query 使用 bottom-right causal mask，只能看到自身及之前的有效 token。
- NS token 位于序列末尾，因此能够逐步汇聚全部历史 S token 和更早的 NS token。

### 6.5 Pyramid

use_pyramid=true 时，每层只保留 S 序列的后缀作为下一层 query；NS token 每层全部保留。

- final_s_tokens 为空时，最终 S 数量为 min(NS 数量, 初始 S 数量)。
- 中间层按从初始 S 数量到 final_s_tokens 的线性进度计算。
- 大于 pyramid_round_to 时，数量舍入到该粒度。
- 最后一层强制达到 final_s_tokens。

独立 `onetrans` 的最终预测不直接使用剩余 S token，而是取全部最终 NS token。实验性 `mdl_onetrans` 还会在选定的中间/末层让 domain token 直接读取当层 S states，见第 7 节。

~~~text
h = flatten(NS_L) ∈ R[B, N_NS × D]
logit_m = TaskHead_m(h)
~~~

### 6.6 Request cache

OneTrans cache 保存：

- 原始 S token 与 mask。
- 每层 S-side K/V。
- 每层 pyramid 后的 S output 与 mask。

候选相关 NS token 不进入可复用 cache。update_request_cache 仅接受 append-only 更新：新 S 序列必须完整保留旧序列为精确前缀；如果 pyramid 窗口变化使某层旧状态不再等价，该层会重建而不是复用陈旧 K/V。

当前 cache 是模型内、本地调用的复用能力，不是跨机器的生产缓存服务。对 `mdl_onetrans`，这里缓存的是 OneTrans S-side state；domain S-attention 使用另一套表示空间，其缓存边界见 7.4。

---

## 7. 实验性 MDL-OneTrans

### 7.1 逐层 S/NS 双路 memory

`mdl_onetrans` 不是单纯的 “OneTrans 压缩到 NS 后再接 MDL”。每个 OneTrans block 更新完成后，状态按动态 `s_count` 拆成两类 memory：

~~~text
初始化 S、NS、scenario、task token
        │
        ▼
OneTrans block l
        │
        ├── S_l + valid mask ── variable-length sequence attention ─┐
        │                                                           ├─ scenario/task update
        └── NS_l ───────────── existing MDL NS interaction ─────────┘
        │
        ▼
下一层 OneTrans / MDL block
        │
        ▼
MDL task heads
~~~

对 scenario 或 task domain token `D`，当前层计算：

~~~text
D_ns = ExistingMDLInteraction(D, NS_l)
delta_ns = D_ns - D
U_s = VariableLengthDomainAttention(D, S_l, M_l)
g = sigmoid(Linear(concat(D, delta_ns, U_s)))
D_hat = D_ns + g * U_s
~~~

`ExistingMDLInteraction` 保持原有固定槽位 NS 语义，可以是 per-token DomainAwareAttention，也可以是配置消融选择的 RankMixerDomainInteraction。S 分支不压缩或替换 NS 分支，只作为 gated residual 增量加入。

scenario 和 task 各自拥有独立的 sequence attention 与 gate；对应 domain token 路径关闭时也不会创建该分支参数。scenario 先得到 `scenario_hat` 并执行 scenario FFN；task 独立读取 NS/S 后，再融合 FFN 前的 `scenario_hat`，最后执行 task FFN，顺序与原 MDL block 一致。

### 7.2 可变长 sequence attention 与 gate

S attention 的结构与固定槽位 DomainAwareAttention 不同：

- query 和 S memory 分别做 LayerNorm；Q/K/V 和 output 使用共享线性投影。
- K/V 参数跨 S 位置共享，不为动态事件位置创建 token-specific 参数。
- `M_l` 作为 key padding mask；被 mask 的 token 值不会影响输出。
- 某个样本没有任何有效历史时，`U_s` 显式为零且不产生 NaN，因此该样本精确退化为 NS-only domain 路径。
- 不使用额外 causal mask：S states 已由 OneTrans causal block 生成，domain query 只读取而不回写 S。
- 同一模块可处理 pyramid 产生的任意 S 长度，例如 128→64→32。

gate 输入为 `D`、`delta_ns` 和 `U_s`，线性层 bias 初始化为 `-2`；在零输入处其门值为 `sigmoid(-2)≈0.12`，实际初值还会受到随机初始化权重和输入的影响。这使新分支开始训练时偏向接近原 NS-only 结构，同时保留通向 S tokenizer、S states 和 sequence-attention 参数的梯度。把 gate 输出置零时，结果与旧 NS-only block 精确一致。

### 7.3 启用层配置

`model.first_domain_sequence_layer` 只对 `mdl_onetrans` 有效：

- `null`：不创建直接 S 分支，保留旧 NS-only domain 行为。
- 非负层索引：从该层开始（包含该层）创建 scenario/task S attention 与 gate。
- 当某层 `s_count=0` 时，即使该层在启用范围内也产生零 S update。

当前 `configs/mdl_onetrans.yaml` 是两层 smoke profile，设置为 `0`，所以两层都直接读取 S。更深的实验可以设置为 `num_layers - 2`，只让最后两个仍有 S token 的层启用；如果 pyramid 会提前把 S 降到 0，应把起始层前移。

### 7.4 Request cache 语义

缓存模式复用 OneTrans 每层生成的 S states 和 mask，并在候选 fan-out 时扩展到候选 batch。domain sequence attention 目前采用正确性优先实现：

- scenario/task attention 各自重新计算自己的 S-side K/V projection。
- 不直接复用 `OneTransLayerCache.s_key/s_value`，因为 LayerNorm、K/V projection 和 head 表示空间不同。
- 不缓存 `Attn(domain_query, S)` 的最终输出，因为 query 依赖 scenario/task token 和候选路径。
- cached 与 uncached logits 必须数值等价；当前测试覆盖候选 fan-out 和动态 pyramid 长度。

因此当前 cache 能避免 OneTrans S backbone 重算，但尚未消除多个候选对 domain-attention S K/V 的重复投影。后续性能缓存只能保存每套 domain attention 自己的 K/V，不能借用 OneTrans K/V。

### 7.5 数据与建模边界

- 同一历史只作为 OneTrans S token 建模一次；scenario/task token 不允许再把这些序列放进 `prior_inputs`。
- scenario/task token 的初始化继续使用独立的非序列 embedding。
- OneTrans 层数与 MDL domain block 数都等于 `model.num_layers`。
- task/scenario 现在可以在共享 S states 上学习不同的事件选择，但 S states 本身仍由共享 OneTrans causal backbone 生成；该结构应描述为“共享 OneTrans backbone + layer-wise gated domain S/NS interaction”，而不是论文定义的 domain-aware OneTrans。
- `first_domain_sequence_layer` 从 `null` 切换为整数会新增 sequence-attention 和 gate 参数；旧 NS-only `mdl_onetrans` checkpoint 不能按 strict 模式直接加载到启用分支的模型，应重新训练或执行显式迁移。

该组合仅用于实验，结果表必须标注“experimental”。

---

## 8. LONGER

### 8.1 事件输入

对每个历史事件：

1. 拼接 item 和 side-information embedding/dense 字段，但先排除 time_delta。
2. 对有效事件的相对位置加入 learned position embedding。
3. 再拼接绝对 time_delta 标量。
4. 经过两层 MLP 投影到基础宽度 d。

也就是当前实现顺序为：

~~~text
MLP([item/side embedding + position ; time_delta])
~~~

不是先把 time_delta 与其他字段相加，也不是在 MLP 后再加位置。

### 8.2 TokenMerge 与 InnerTrans

设合并因子为 K，基础宽度为 d：

- 序列左侧补齐到 K 的整数倍，保持最近事件在右侧对齐。
- 每 K 个连续事件组成一个局部分组。
- longer_inner_layers>0 时，在每个 K-token 分组内部执行局部 Transformer。
- 最后把 K 个位置按原顺序拼接，不做平均或投影压缩。

因此每个 merged token 的宽度是 Kd，完整保留 K 个槽位：

~~~text
[x_1, x_2, ..., x_K] → concat → R[Kd]
~~~

### 8.3 Global 与 recent query

LONGER 把固定输出 token 分为：

- cacheable user global token：由 longer_user_global_inputs 投影得到。
- learned CLS token：属于 cacheable sequence side。
- candidate global token：由 target_inputs 投影得到，不进入可复用 cache。
- recent query：从 merged sequence 的最后 longer_query_tokens 个 token 取得，不足时左侧补零。

前三类 global token 总数必须等于 rankmixer_summary_tokens。这里的字段名沿用了 RankMixer 接口，但在独立 LONGER 中表示 global token 总数。

### 8.4 Cross block、self block 与可见性

Cross block 使用完整 merged sequence 作为历史 K/V 上下文：

- user/CLS global query 可以看到所有有效 key。
- recent query 使用 bottom-right causal attention，只能看到自己对应时间及以前的 key。
- candidate global 单独计算，可以看到 candidate、自身可复用 user/CLS 和完整历史。

随后执行 longer_self_layers 个 self block：

- user/CLS 和 recent 的 sequence-side 状态可缓存。
- candidate 状态每个候选重新计算。
- global query 保持 full visibility。
- recent query 保持 causal visibility。

Flash 路径不会把两种语义合并成一个错误的统一 mask，而是把 global full attention 与 recent causal attention 拆成两个 packed varlen 调用，再按原顺序合并。

### 8.5 输出与 cache

最终输出顺序固定为：

~~~text
[cacheable user/CLS globals ;
 candidate globals ;
 recent query outputs]
~~~

输出展平后进入每任务独立 TaskHead。完整 merged sequence 保留在 K/V 上下文和 cache 中，但最终显式输出只保留固定数量的 globals 与 recent query。

独立 LONGER 要求恰好配置一条 encoder=longer 的序列。model.num_layers 计数为：

~~~text
1 个 cross block + longer_self_layers
~~~

TokenMerge 内部的 longer_inner_layers 不计入 model.num_layers。

---

## 9. configs/default.yaml 的实际默认设置

### 9.1 模型与 token 布局

| 配置项 | 默认值 | 作用 |
|---|---:|---|
| model.name | mdl_rankmixer | 默认模型入口 |
| embedding_dim | 32 | categorical 默认 embedding 宽度 |
| token_dim | 32 | feature/scenario/task token 宽度 D |
| num_layers | 2 | MDL-RankMixer block 数 |
| num_heads | 4 | domain attention 和 LONGER attention 头数 |
| hidden_dim | 64 | configurable FFN 隐藏宽度 |
| init_std | 0.02 | learned token/position/embedding 初始化标准差 |
| ffn_activation | gelu | 模型 FFN 激活 |
| task_head_hidden_dim | 64 | 每任务任务头隐藏宽度 |
| task_head_dropout | 0.0 | 任务头 dropout |
| rankmixer_ffn_type | dense | 默认不使用 Sparse-MoE |
| mdl_feature_interaction | direct_ffn | MDL feature FFN 后无第二个 Add & Norm |
| use_task_tokens | true | 创建并逐层更新 task token |
| use_scenario_tokens | true | 创建并逐层更新 scenario token |
| use_global_scenario_token | true | 加入 global scenario token |
| use_task_feature_interaction | true | task 使用 domain-aware attention |
| use_scenario_feature_interaction | true | scenario 使用 domain-aware attention |
| use_request_cache | false | 训练/普通推理不自动预计算 cache |
| first_domain_sequence_layer | null | 仅 mdl_onetrans；null 为 NS-only，整数表示从该层启用 gated S attention |

默认任务和场景：

| 项目 | 默认值 |
|---|---|
| task_names | [click]，来自 data.train.labels |
| scenarios.names | [default] |
| scenarios.source | null，单场景时自动使用场景 0 |
| feature token 数 T | 4 |
| scenario token 数 | 2，即 default + global |
| task token 数 | 1，即 click |

默认四个 feature token：

| token 名称 | 输入 | 投影前宽度 | 输出宽度 |
|---|---|---:|---:|
| user_profile | user_id | 32 | 32 |
| item_profile | item_id + shop_id | 64 | 32 |
| context | rankmixer_context_dense | 16 | 32 |
| long_history | hist 的 LONGER 表示 | 1408 | 32 |

默认 scenario token：

- default：scenario_user_id + scenario_item_id + hist。
- global：scenario_user_id + scenario_item_id + hist。

默认 task token：

- click：task_user_id + task_item_id + hist。

scenario_user_id、scenario_item_id、task_user_id、task_item_id 与普通 user/item 共用词表映射，但 share_embedding=false，因此各自有独立参数表。

### 9.2 默认 hist LONGER 子编码器

| 配置项 | 默认值 |
|---|---:|
| max_length | 100 |
| truncation | tail |
| sequence_order | oldest_to_newest |
| encoder | longer |
| fields | item_id、shop_id、action、age、time_delta |
| 基础 token 宽度 d | 32 |
| longer_token_merge K | 4 |
| merged 宽度 Kd | 128 |
| longer_inner_layers | 1 |
| longer_query_tokens | 8 |
| longer_self_layers | 1 |
| user global | 1，由 user_id 投影 |
| CLS | 1，learned |
| candidate global | 1，由 item_id 投影 |
| global 总数 | 3 |
| 最终序列表示宽度 | (3 + 8) × 128 = 1408 |

默认 hist.item_id 与当前 item_id 共享 embedding 权重；hist.shop_id 和 hist.action 使用各自的 hash embedding。

### 9.3 默认类别策略

| 输入 | encoding | 规模/共享 |
|---|---|---|
| user_id | vocab | min_count=5，max_size=5,000,000 |
| item_id | vocab | min_count=3，max_size=10,000,000 |
| scenario/task user/item | shared_vocab | 共用 ID 映射，独立 embedding |
| hist.item_id | shared_vocab | 与 item_id 共用 ID 映射和 embedding |
| shop_id | hash | 1,000,000 buckets |
| hist.shop_id | hash | 1,000,000 buckets |
| hist.action | hash | 1,024 buckets |

padding_id 和 oov_id 都是 0。

### 9.4 默认运行时

| 配置项 | 默认模板值 |
|---|---|
| device | cuda |
| precision | bf16 |
| compile | false |
| allow_tf32 | true |
| activation_checkpoint | none |
| attention_backend | auto |
| distributed | none |

这是安全 GPU 环境模板，不保证在 CPU-only 机器上直接运行。CPU 调试需要本地 overlay 将 device 改为 cpu，并通常将 precision 改为 fp32。

### 9.5 默认训练设置

| 配置项 | 默认值 |
|---|---:|
| batch_size | 2048，每进程 |
| embedding_distribution | replicated |
| dense_distribution | ddp |
| dense optimizer | RMSProp |
| lr_dense | 0.005 |
| lr_sparse | null，实际回退到 lr_dense |
| lr_schedule | cosine |
| warmup / decay | 1,000 / 100,000 steps |
| lr_min_ratio | 0.1 |
| RMSProp alpha / momentum | 0.99999 / 0.0 |
| sparse optimizer | Adagrad |
| Adagrad initial accumulator | 0.1 |
| Adagrad eps | 1e-10 |
| dense / sparse clip norm | 90 / 120 |
| loss_reduction | sum |
| embedding_sparse_gradients | true |
| checkpoint_path | artifacts/checkpoints/mdl_rankmixer.pt |

DDP 默认采用审计安全设置：

| 配置项 | 默认值 |
|---|---|
| static_graph | false |
| find_unused_parameters | true |
| gradient_as_bucket_view | true |
| bucket_cap_mb | 25 |
| audit_steps | 10 |

只有代表性训练已经证明无 unused parameter 或图静态后，才能设置相应 validated 证据位并改变这些开关。

### 9.6 默认数据模板的限制

- train/test 路径是 /secure/... 占位路径，必须由本地 overlay 适配。
- flat_parquet 要求一行一个样本；其他物理布局通过 adapter_parquet 转成该逻辑契约。
- 默认 train 定义 click label。
- 默认 test **没有配置 labels**，所以不能直接产出正式 test 评估结果。
- 默认 test.group_id=request_id，适合作为 QAUC 分组；如果要报告 UAUC，应让 group_id 指向 user key。
- 所有 smoke/paper overlay 如果没有覆盖 checkpoint_path，都会继承 mdl_rankmixer.pt。跨模型实验应显式传入不同的 --checkpoint-path，避免覆盖或误载。

---

## 10. 代码字段缺省值与默认模板的区别

以下差异最容易造成误解：

| 字段 | src/config.py 字段缺省值 | configs/default.yaml 实际值 |
|---|---:|---:|
| model.name | 必填，无缺省 | mdl_rankmixer |
| token_dim | 768 | 32 |
| num_layers | 6 | 2 |
| num_heads | 12 | 4 |
| hidden_dim | 1536 | 64 |
| task_head_hidden_dim | null，回退到 hidden_dim | 64 |
| runtime.device | cpu | cuda |
| runtime.precision | fp32 | bf16 |
| training.lr_schedule | constant | cosine |
| lr_warmup_steps | 0 | 1000 |
| lr_decay_steps | null | 100000 |
| lr_min_ratio | 0.0 | 0.1 |
| dense/sparse clip norm | null / null | 90 / 120 |

版本化配置通过 extends: default.yaml 继承的是右列，而不是重新回到 dataclass 缺省值。

---

## 11. Smoke profile 的默认行为

| 配置 | 相对 default.yaml 的关键变化 | 实际核心规模 |
|---|---|---|
| default.yaml | 无 | MDL-RankMixer，T=4、D=32、L=2、H=4 |
| mdl_rankmixer.yaml | 只确认 model.name=mdl_rankmixer | 与 default 相同 |
| rankmixer.yaml | model.name=rankmixer | T=4、D=32、L=2；无 MDL domain 模块 |
| onetrans.yaml | hist 改为 encoder=raw；intent_ordered；max_position_embeddings=104 | 最多 100 S + 自动推导 4 NS，D=32、L=2、H=4 |
| longer.yaml | model.name=longer | max length 100、d=32、K=4、recent=8、cross+1 self |
| mdl_onetrans.yaml | 继承 onetrans；加入实验 ack；不使用 hist prior；first_domain_sequence_layer=0 | 2 层 OneTrans 与 2 层 gated S/NS MDL domain block 逐层交替 |

OneTrans smoke 中 auto_split 的 4 个 NS token 来自四个 feature/shared 标量输入：user_id、item_id、shop_id、rankmixer_context_dense。位置表容量 104 对应最多 100 个 S token 加 4 个 NS token。只有一组 S sequence，因此虽然 use_sep_tokens 默认是 true，实际不会插入 SEP。`mdl_onetrans` smoke 的两层 scenario/task block 都同时读取当层固定 NS slots 和带 mask 的 pyramid S states。

---

## 12. Paper profile

### 12.1 关键参数对照

| 配置 | 论文方法面 | 主参数 | 当前公开实现选择 |
|---|---|---|---|
| rankmixer_paper.yaml | RankMixer 100M dense 方法 | T=16、D=768、L=2、dense LR=0.01 | embedding=32、hidden=1536 即 k=2、task head=64、输入 schema 和 batch 未由论文完整公开 |
| mdl_rankmixer_paper.yaml | 3 场景 × 3 任务的 MDL 信息流 | feature T=4、D=32、L=2、H=4、hidden=64、direct_ffn | 论文公开生产面为 3 场景和 20+ 任务、约 0.5B；L/D/H、私有 prior schema 和完整任务表未公开 |
| onetrans_paper.yaml | OneTrans-S | max S=1190、NS=12、D=256、L=6、H=4、pyramid→12 | hidden=1024、embedding 宽度、learned absolute position 形式和 LR schedule 是实现选择或未决项 |
| longer_paper.yaml | LONGER 主配置 | max L=2000、d=32、K=8、recent-k=100、InnerTrans=1、cross+1 self | H=4、hidden=64、激活、优化器、LR 未由论文完整公开 |

### 12.2 RankMixer paper profile 的输入宽度

该配置继承默认输入编码器：

- user_id：32
- item_id：32
- shop_id：32
- rankmixer_context_dense：16
- hist LONGER：1408

总宽度 1520，被切成 16 个宽度 95 的 slice，每个 slice 独立投影到 D=768。RankMixer 核心的“head 数”由 T=16 隐式决定；model.num_heads=12 主要供继承的 LONGER 历史编码器使用，不参与 RankMixer TokenMixing。

### 12.3 MDL paper profile 的 domain 面

场景索引：

| scenario index | 名称 | 专属 prior |
|---:|---|---|
| 0 | single_column | scenario_single_column_history |
| 1 | double_column | scenario_double_column_history |
| 2 | inner_search | scenario_inner_search_history |
| 内部额外 token | global | 通用 hist |

任务：

| task index | 名称 | 专属 prior |
|---:|---|---|
| 0 | click | task_click_history |
| 1 | like | task_like_history |
| 2 | favorite | task_favorite_history |

六条专属 prior 序列都使用独立 embedding 表和 attention_pool。它们是可执行的公开数据契约，不应被描述为论文未公开生产 schema 的逐字段复原。

### 12.4 OneTrans paper profile

- hist 使用 raw event token，max_length=1190。
- timestamp_field=timestamp，必须由数据侧提供有意义的真实时间戳。
- sequence_fusion=timestamp_aware。
- use_sep_tokens=false。
- N_NS=12，位置表容量=1190+12=1202。
- pyramid 最终保留 12 个 S token。
- 最终预测输入宽度是 12×256=3072。

### 12.5 LONGER paper profile

- 物理输入方向 newest_to_oldest。
- truncation=head，因此保留物理列表开头的最近 2000 个事件。
- 模型内部转换为 oldest_to_newest 后执行 causal attention。
- d=32，K=8，因此 merged token 宽度为 256。
- 3 个 global token：1 user、1 CLS、1 candidate。
- 100 个 recent query。
- 最终任务头输入宽度为 (3+100)×256=26368。

---

## 13. Performance profile

perf 配置用于吞吐和资源测试，不是新的模型定义：

| 配置 | 继承 | 关键覆盖 |
|---|---|---|
| rankmixer_perf.yaml | rankmixer_paper.yaml | CUDA BF16、identity ID、sharded embedding |
| mdl_perf.yaml | mdl_rankmixer_paper.yaml | CUDA BF16、batch=512、identity ID、sharded embedding |
| onetrans_perf.yaml | onetrans_paper.yaml | CUDA BF16、Flash attention、batch=32、sharded embedding |
| longer_perf.yaml | longer_paper.yaml | CUDA BF16、Flash attention、selective checkpoint、batch=64、sharded embedding |
| longer_5000_perf.yaml | longer_perf.yaml | max length=5000、batch=16、较小 reader/prefetch |

identity profile 中 0 保留为 padding，所有 ID 必须满足 0 ≤ id < num_buckets。越界默认报错。hist.item_id 继续与 item_id 共享 embedding，其余 MDL scenario/task 表保持独立。

---

## 14. Embedding、分布式和缓存边界

### 14.1 Embedding 分布

- replicated：每个 rank 保存完整 embedding 权重和 Adagrad accumulator；适合小表正确性基线。
- sharded：按表大小自动选择 table-wise 或 row-wise；每个 rank 只保存 owner 行和对应优化器状态。
- sharded 路径先做本地重复 ID 合并，再用 owner-based variable all-to-all 路由请求和梯度。
- 稠密参数继续由标准 DDP 同步。

### 14.2 DDP 不等长 shard

如果不同 rank 读到的 batch 数不同，已经耗尽的 rank 会复用最后一个 batch 做零权重 loss 的 forward/backward，继续参加同步，直到最长 shard 完成。若某个 rank 从第一步就是空 shard，需要减少 world size 或使用更细的 shard_unit。

### 14.3 Cache 复用前提

训练不会仅凭 request_id 相同自动复用 cache。只有数据契约能证明同一 request 下所有 cacheable 历史、时间和用户特征完全一致时，才能安全地把一次 request cache 展开给多个候选。

`mdl_onetrans` 的 fan-out 只复用 OneTrans S-side state；scenario/task 的 S K/V 目前按候选重新投影。该重复计算是已知性能边界，不影响 cached/uncached 数值等价性。

---

## 15. 离线评估口径

### 15.1 正式评估前的配置要求

evaluate 要求：

1. split 必须是 train 或 test。
2. 评估 split 的 labels 必须与 data.train.labels 具有相同任务名和相同顺序。
3. split.group_id 必须存在。
4. 正式结果必须加载匹配模型和配置的 checkpoint。
5. 不要把 --allow-random-init 的输出填写为模型结果。

默认 test 至少需要在本地 overlay 中补充：

~~~yaml
data:
  test:
    labels:
      click: click
~~~

MDL paper profile 需要：

~~~yaml
data:
  test:
    labels:
      click: click
      like: like
      favorite: favorite
~~~

### 15.2 命令

QAUC：

~~~bash
python src/main.py evaluate \
  --config configs/mdl_rankmixer.yaml \
  --checkpoint-path artifacts/checkpoints/mdl_rankmixer.pt \
  --split test \
  --group-metric-name qauc \
  --auc-bins 65536
~~~

UAUC：

~~~bash
python src/main.py evaluate \
  --config configs/mdl_rankmixer.yaml \
  --checkpoint-path artifacts/checkpoints/mdl_rankmixer.pt \
  --split test \
  --group-metric-name uauc \
  --auc-bins 65536
~~~

group-metric-name 只决定输出名称；真正的分组语义由 data.test.group_id 指向的列决定：

- group_id 是 query/request key：报告 QAUC。
- group_id 是 user key：报告 UAUC。

### 15.3 指标定义

evaluate 对每个任务分别输出：

- auc：全部有效标签的 AUC。
- qauc 或 uauc：所有有效 group 的非加权 group AUC 均值。
- scenario_i_auc：属于场景 i 的有效样本 AUC。
- scenario_i_qauc 或 scenario_i_uauc：场景 i 内的非加权 group AUC 均值。

实现细节：

- 先对 logits 做 sigmoid。
- label_mask=false 的位置不进入任何指标。
- 普通 AUC 使用固定分数直方图流式累计，默认 65,536 bins。
- 同 bin 的正负样本按 0.5 tie credit 处理。
- 没有正样本或没有负样本的 AUC 返回 NA。
- group 内只有单一类别时，该 group 被跳过。
- group 指标是有效 group 的非加权平均，不按 group 大小加权。
- 多热场景样本会进入每个激活场景的分场景指标。
- 当前 CLI 不自动计算“跨任务宏平均 AUC”；如需该值，必须在实验协议中另行定义和记录。

终端输出格式：

~~~text
evaluate_result rows=... group_metric=qauc auc_histogram_bins=65536
evaluate_task task=click auc=... qauc=...
  scenario_0_auc=... scenario_0_qauc=...
~~~

---

## 16. 评估结果填写区

以下所有“待填写”单元格都应在真实 checkpoint、固定数据快照和固定评估参数下填写。

### 16.1 实验元数据

| 项目 | 结果 |
|---|---|
| 实验名称 | 待填写 |
| 评估日期（UTC） | 待填写 |
| Git commit / 工作区说明 | 待填写 |
| 配置文件 | 待填写 |
| 配置 overlay / 修改摘要 | 待填写 |
| 数据集名称与版本 | 待填写 |
| train 时间范围 | 待填写 |
| validation/test 时间范围 | 待填写 |
| 数据 schema / adapter 版本 | 待填写 |
| vocab_strategy_hash | 待填写 |
| checkpoint 路径 | 待填写 |
| checkpoint step / epoch | 待填写 |
| 随机种子 | 待填写 |
| 设备型号与数量 | 待填写 |
| precision / attention backend | 待填写 |
| world size / per-rank batch / global batch | 待填写 |
| group_id 语义 | request/query/user，待填写 |
| group metric | QAUC 或 UAUC，待填写 |
| auc_bins | 65536 或待填写 |
| 是否完整 test split | 是/否，待填写 |
| 备注 | 待填写 |

### 16.2 主模型质量对比

单场景 click 配置可以直接填写下表。scenario_0 对应 default。

| 模型 | 配置 | checkpoint | rows | AUC | QAUC/UAUC | scenario_0 AUC | scenario_0 QAUC/UAUC | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---|
| RankMixer smoke | configs/rankmixer.yaml | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| MDL-RankMixer smoke | configs/mdl_rankmixer.yaml | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| OneTrans smoke | configs/onetrans.yaml | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| LONGER smoke | configs/longer.yaml | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| MDL-OneTrans experimental | configs/mdl_onetrans.yaml | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | gated S/NS 双路；必须标 experimental |
| RankMixer paper profile | configs/rankmixer_paper.yaml | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| OneTrans paper profile | configs/onetrans_paper.yaml | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| LONGER paper profile | configs/longer_paper.yaml | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

### 16.3 MDL paper profile：总体与分场景结果

场景映射：

| CLI key | 场景 |
|---|---|
| overall | 全部场景 |
| scenario_0 | single_column |
| scenario_1 | double_column |
| scenario_2 | inner_search |

质量结果：

| 任务 | 范围 | AUC | QAUC/UAUC | 备注 |
|---|---|---:|---:|---|
| click | overall | 待填写 | 待填写 | 待填写 |
| click | single_column | 待填写 | 待填写 | 待填写 |
| click | double_column | 待填写 | 待填写 | 待填写 |
| click | inner_search | 待填写 | 待填写 | 待填写 |
| like | overall | 待填写 | 待填写 | 待填写 |
| like | single_column | 待填写 | 待填写 | 待填写 |
| like | double_column | 待填写 | 待填写 | 待填写 |
| like | inner_search | 待填写 | 待填写 | 待填写 |
| favorite | overall | 待填写 | 待填写 | 待填写 |
| favorite | single_column | 待填写 | 待填写 | 待填写 |
| favorite | double_column | 待填写 | 待填写 | 待填写 |
| favorite | inner_search | 待填写 | 待填写 | 待填写 |

如果需要跨任务汇总，请先写明汇总公式：

| 汇总项 | 定义 | 结果 |
|---|---|---:|
| Macro AUC | 待填写，例如三个任务 AUC 的算术平均 | 待填写 |
| Macro QAUC/UAUC | 待填写，例如三个任务 group AUC 的算术平均 | 待填写 |
| 业务加权 AUC | 待填写权重与来源 | 待填写 |

### 16.4 MDL 消融结果

每行除“变更项”外应保持数据、seed、训练步数、batch 和优化器一致。

| 实验 | 变更项 | 参数量 | AUC | QAUC/UAUC | 相对主模型 ΔAUC | 相对主模型 ΔGroup AUC | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| 主模型 | direct_ffn，全部 token/interaction 开启 | 待填写 | 待填写 | 待填写 | 0 | 0 | 待填写 |
| Feature residual | mdl_feature_interaction=residual_ffn | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| No task token | use_task_tokens=false | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 顶层 per-task tower |
| No scenario token | use_scenario_tokens=false | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | ScenarioTower |
| No global scenario | use_global_scenario_token=false | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| Task RankMixer replacement | use_task_feature_interaction=false | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 不是零更新 |
| Scenario RankMixer replacement | use_scenario_feature_interaction=false | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 不是零更新 |
| Sparse-MoE | rankmixer_ffn_type=sparse_moe | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 填写 DTSI policy |

### 16.5 OneTrans 消融结果

| 实验 | 关键变化 | AUC | QAUC/UAUC | samples/s | peak HBM | 备注 |
|---|---|---:|---:|---:|---:|---|
| Paper profile | 1190 S、12 NS、6 层、pyramid | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| No pyramid | use_pyramid=false | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| Intent ordered | sequence_fusion=intent_ordered | 待填写 | 待填写 | 待填写 | 待填写 | 仅在数据语义允许时比较 |
| Groupwise NS | ns_tokenizer=groupwise | 待填写 | 待填写 | 待填写 | 待填写 | 填写 NS 分组 |
| No request cache | use_request_cache=false | 待填写 | 待填写 | 待填写 | 待填写 | 质量应等价 |
| Request cache | 显式复用 S-side cache | 待填写 | 待填写 | 待填写 | 待填写 | 记录候选复用数 |

### 16.6 MDL-OneTrans 消融结果

除结构开关外，数据、seed、训练步数、优化器和 task/scenario 汇总方式必须一致。当前公共配置只实现 gated residual S 分支；“直接相加、不使用 gate”不是现有 YAML 开关，若实验需要必须先增加明确实现与测试。

| 实验 | first_domain_sequence_layer | S 读取范围 | AUC | QAUC/UAUC | samples/s | peak HBM | 备注 |
|---|---:|---|---:|---:|---:|---:|---|
| NS-only baseline | null | 无直接 S attention | 待填写 | 待填写 | 待填写 | 待填写 | 与旧 domain block 精确一致 |
| Gated S, all layers | 0 | 每个仍有 S 的层 | 待填写 | 待填写 | 待填写 | 待填写 | 当前两层 smoke 配置 |
| Gated S, late layers | num_layers - 2 | 最后两个仍有 S 的层 | 待填写 | 待填写 | 待填写 | 待填写 | 若 S 提前归零需前移 |
| Gated S + request cache | 与主实验相同 | 同上 | 待填写 | 待填写 | 待填写 | 待填写 | domain S K/V 仍按候选重算 |

除总体指标外，应分别记录每个 task/scenario、长短历史分桶、候选 fan-out、延迟和 cache 内存；如通过额外 instrumentation 导出 attention，应同时记录任务间分布差异与 entropy。

### 16.7 LONGER 消融结果

| 实验 | max L | K | InnerTrans 层 | recent-k | self 层 | AUC | QAUC/UAUC | samples/s | peak HBM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper profile | 2000 | 8 | 1 | 100 | 1 | 待填写 | 待填写 | 待填写 | 待填写 |
| No InnerTrans | 2000 | 8 | 0 | 100 | 1 | 待填写 | 待填写 | 待填写 | 待填写 |
| K=4 | 2000 | 4 | 1 | 100 | 1 | 待填写 | 待填写 | 待填写 | 待填写 |
| Short history | 待填写 | 8 | 1 | 100 | 1 | 待填写 | 待填写 | 待填写 | 待填写 |
| Long stress | 5000 | 8 | 1 | 100 | 1 | 待填写 | 待填写 | 待填写 | 待填写 |

### 16.8 性能结果

性能测试命令示例：

~~~bash
python src/main.py benchmark \
  --config configs/longer_perf.yaml \
  --mode end-to-end \
  --warmup-steps 20 \
  --steps 100 \
  --profile-steps 3 \
  --distributed ddp \
  --nproc-per-node 8 \
  --output artifacts/benchmarks/longer/end_to_end_8gpu.json
~~~

主性能表：

| 模型/配置 | mode | GPU 数 | per-rank batch | samples/s | tokens/s | mean step ms | p95 step ms | padding ratio | peak HBM/rank | GPU util | MFU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RankMixer perf | data | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| RankMixer perf | end-to-end | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| MDL perf | end-to-end | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| OneTrans perf | compute | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| OneTrans perf | end-to-end | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| MDL-OneTrans experimental | end-to-end | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| LONGER perf | compute | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| LONGER perf | end-to-end | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| LONGER 5000 | compute | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

阶段与通信明细：

| 模型/配置 | dataloader wait ms | H2D ms | forward ms | backward ms | sparse sync ms | optimizer ms | sparse payload bytes/step/rank-max | profiler communication ms/rank-max | attention kernel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

### 16.9 可复现性和验证记录

| 检查项 | 命令/证据 | 结果 |
|---|---|---|
| 配置校验 | python src/main.py validate-config --config ... | 待填写 |
| MDL-OneTrans S 分支、mask、空历史与梯度 | python -m pytest tests/test_model_alignment.py::MDLOneTransSequenceAttentionTest -q | 待填写 |
| MDL-OneTrans cached/uncached 等价 | 同上候选 fan-out + pyramid 用例 | 待填写 |
| vocab hash 固定 | validate-config 输出 | 待填写 |
| 模型前向/反向有限值 | 对齐测试或 smoke train | 待填写 |
| checkpoint 保存/加载 | 训练与 evaluate 使用同一配置 | 待填写 |
| 完整 test，无 max-batches 截断 | evaluate 命令 | 待填写 |
| AUC bins 固定 | --auc-bins | 待填写 |
| group_id 语义审计 | 数据说明 | 待填写 |
| CPU/GPU 环境 | 运行日志 | 待填写 |
| 单卡/多卡等价性 | 对比实验 | 待填写 |
| Flash kernel 观测 | benchmark attention_kernels | 待填写 |
| 未运行项及原因 | 数据/GPU/权限限制 | 待填写 |

---

## 17. 结果解读时必须保留的边界

1. paper profile 只固定公开方法面和公开超参数，不包含论文私有数据、完整生产特征、集群拓扑或未公开训练细节。
2. default.yaml 是小规模 smoke 模板，不是论文工业规模。
3. mdl_rankmixer_paper.yaml 的 3×3 面只覆盖代表性任务；不能据此宣称复现论文 20+ 任务或约 0.5B 模型。
4. MDL 的六条专属 prior 是公开仓库的数据契约，不是未披露生产 schema。
5. OneTrans learned absolute position、部分 hidden/embedding 宽度是显式实现选择。
6. LONGER 的 H、FFN 宽度、优化器和 LR 是显式实现选择。
7. mdl_onetrans 是带 layer-wise gated S/NS domain interaction 的实验组合，不是 MDL 或 OneTrans 论文定义模型；任何结果都必须单独标注。
8. 默认评估 AUC 是 65,536-bin 直方图近似；不同 auc_bins 的结果不能在没有说明时直接混用。
9. QAUC/UAUC 的名称必须与 group_id 的真实业务语义一致。
10. cache 结果只有在输入可缓存性契约成立时才能用于性能结论。
11. 不同模型应使用独立 checkpoint 路径；结构或配置不匹配时不应非严格加载后继续报告结果。

---

## 18. 实现索引

| 内容 | 位置 |
|---|---|
| 配置加载、继承、验证、resolved token 规格 | [src/config.py](src/config.py) |
| 公共特征编码、LONGER、OneTrans、所有模型前向 | [src/model.py](src/model.py) |
| RankMixer TokenMixing、固定槽位 domain attention、可变长 domain S attention、scenario fusion | [src/modules/attention.py](src/modules/attention.py) |
| Per-token FFN、batched FFN、Sparse-MoE | [src/modules/mlp.py](src/modules/mlp.py) |
| 训练、masked BCE、DDP、AUC/QAUC/UAUC | [src/train.py](src/train.py) |
| Parquet 读取、序列截断/批构建 | [src/dataloader.py](src/dataloader.py) |
| 类别编码与 vocab | [src/features.py](src/features.py) |
| sharded embedding | [src/embeddings.py](src/embeddings.py) |
| benchmark 指标 | [src/benchmark.py](src/benchmark.py) |
| 模型数学和缓存对齐测试 | [tests/test_model_alignment.py](tests/test_model_alignment.py) |
| LONGER 对齐测试 | [tests/test_longer_alignment.py](tests/test_longer_alignment.py) |
| 评估指标测试 | [tests/test_evaluation_metrics.py](tests/test_evaluation_metrics.py) |
