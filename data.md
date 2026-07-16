# 1. 数据处理

## 1.1 fgout 输出数据：结构与语义

### 1.1.1 两种数据格式

fgout 输出的 Parquet 文件存在两种格式，分别用于训练和预测：

| 格式           | 目录              | 特点               | 结构                                                          |
| ------------ | --------------- | ---------------- | ----------------------------------------------------------- |
| 多请求合并格式（agg） | `mock_base_agg` | 一行 = 一个用户的多次请求合并 | 有 `context_indices`、`target_indices`、`{ups_type}_x_indices` |
| 单请求格式（req）   | `mock_base_req` | 一行 = 一个请求        | 无上述 indices 列，UPS、Context、Item 直接属于该请求                      |

训练数据利用合并格式加速：同一用户连续请求的 UPS 特征重复度高，合并后使用 indices 隔离不同请求。

预测数据是单条请求，无需合并。

> ★ **自动检测**
>
> `ParquetReader._parse_row` 通过检查 `context_indices` / `target_indices` 列是否存在，自动选择对应的解析路径。

---

### 1.1.2 列结构与数据类型

#### 多请求合并格式（agg）列结构

| 类别              | 列名模式                                 | 数据类型                   | 说明                  |
| --------------- | ------------------------------------ | ---------------------- | ------------------- |
| UPS 特征          | `{ups_type}_x_{feature_name}_hn`     | `array<bigint>`        | 每种行为的序列特征           |
| UPS indices     | `{ups_type}_x_indices`               | `array<array<bigint>>` | 每个 UPS token 属于哪些请求 |
| Context 特征      | `{feature_name}_hn`                  | `array<bigint>`        | 每个请求一个值的上下文特征       |
| Context indices | `context_indices`                    | `array<bigint>`        | 每个 context 值属于哪个请求  |
| Item 特征         | `{feature_name}_hn`                  | `array<array<bigint>>` | 每个候选物品一组值           |
| Item indices    | `target_indices`                     | `array<bigint>`        | 每个 item 属于哪个请求      |
| Creative 特征     | `{feature_name}_hn`                  | `array<array<bigint>>` | 每个候选物品的创意信息         |
| Meta            | `search_id`、`scene_id`、`example_ids` | `array`                | 元信息                 |
| Label           | `label_click` 等                      | `array<bigint>`        | 每个 item 的标签         |

#### 单请求格式（req）列结构

| 类别          | 列名模式                             | 数据类型                   | 说明             |
| ----------- | -------------------------------- | ---------------------- | -------------- |
| UPS 特征      | `{ups_type}_x_{feature_name}_hn` | `array<bigint>`        | 同 agg，但全部属于该请求 |
| Context 特征  | `{feature_name}_hn`              | `array<bigint>` 或标量    | 只有一个请求的值       |
| Item 特征     | `{feature_name}_hn`              | `array<array<bigint>>` | 可能仍有多个 item    |
| Creative 特征 | `{feature_name}_hn`              | `array<array<bigint>>` | 可选             |
| Meta        | `search_id`、`scene_id`           | 标量或 `array`            | 单请求元信息         |
| Label       | `label_click` 等                  | `array<bigint>` 或标量    | 可能为空，预测数据通常无标签 |

> ★ **关键差异**
>
> req 格式没有 indices 列，因此：
>
> * UPS 特征不需要按照 indices 过滤，全部 token 都属于该请求；
> * Context 特征不需要按照 `req_idx` 索引，直接取值；
> * Item 特征的处理逻辑相同，即每个 item 对应一组特征值。

---

### 1.1.3 不同 `ups_type` 有不同的特征集合

fgout 数据中，每种用户行为（`ups_type`）包含的特征不同。这是由业务语义决定的。例如：

* 点击行为有 `page_sn`（页面序号）、`price`（价格）等特征；
* 搜索行为有 `flat_q_hash`（搜索 Query 哈希），但没有 `page_sn`、`price`；
* 加购行为没有 `page_sn` 和 `sales`。

#### 测试配置（2 个 `ups_type`）

| `ups_type`           | discover 发现的特征                                                                                                                             | 数量 |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | -: |
| `clk_long`           | `cat1_id_hn`、`cat2_id_hn`、`cat3_id_hn`、`cat4_id_hn`、`cat_id_hn`、`goods_id_hn`、`mall_id_hn`、`page_sn_hn`、`price_hn`、`sales_hn`、`timegap_hn` | 13 |
| `flatten_query_hash` | `flat_q_hash_hn`、`timegap_hn`                                                                                                              |  2 |

> **校验提示：**上述 `clk_long` 特征列表中实际列出了 11 个特征，与“数量 13”不一致，需要结合真实 Parquet Schema 确认是否遗漏了两个特征。

#### 生产配置（6 个 `ups_type`）

| `ups_type`           | 特征数量 | 拥有的特征                                                                      | 缺失的特征（`miss_feas`）              |
| -------------------- | ---: | -------------------------------------------------------------------------- | ------------------------------- |
| `impr`               |   11 | `timegap`、`page_sn`、`price`、`sales`、`mall_id`、`cat_id`、`cat1-4`、`goods_id` | `flat_q_hash`                   |
| `clk_long`           |   11 | 同上                                                                         | `flat_q_hash`                   |
| `view_long`          |   11 | 同上                                                                         | `flat_q_hash`                   |
| `cart_long`          |    9 | `timegap`、`price`、`mall_id`、`cat_id`、`cat1-4`、`goods_id`                   | `page_sn`、`sales`、`flat_q_hash` |
| `buy_long`           |   10 | `timegap`、`price`、`sales`、`mall_id`、`cat_id`、`cat1-4`、`goods_id`           | `page_sn`、`flat_q_hash`         |
| `flatten_query_hash` |    2 | `timegap`、`flat_q_hash`                                                    | 其余 10 个特征全部缺失                   |

> **注意**
>
> `miss_feas` 是主力模型中的特征配置概念，OneTrans 不需要处理。
>
> OneTrans 的做法是：从 Parquet Schema 中自动发现每种 `ups_type` 实际包含哪些特征，有什么就使用什么，不对不存在的特征进行 zero-fill。

---

### 1.1.4 indices 语义详解

#### UPS indices：`{ups_type}_x_indices`

* **类型：**`array<array<bigint>>`
* **含义：**记录每个 UPS token 属于哪些请求。

示例：

```text
[
  [2],
  [2],
  [1, 2],
  [1, 2],
  [1, 2],
  [1],
  [0, 1],
  [0, 1],
  [0, 1],
  [0]
]
```

解释：

* token 0 只属于请求 2；
* token 2 同时属于请求 1 和请求 2，即该点击事件对两个请求都可见；
* token 9 只属于请求 0。

#### Context indices：`context_indices`

* **类型：**`array<bigint>`
* **含义：**记录每个 context 值属于哪个请求。

示例：

```text
[0, 1, 2]
```

表示存在 3 个 context，它们分别属于请求 0、请求 1 和请求 2。

#### Target indices：`target_indices`

* **类型：**`array<bigint>`
* **含义：**记录每个 item 属于哪个请求。

示例：

```text
[0, 0, 0, 3, 3, 3, 3, 4, 5, 5, 5]
```

表示 11 个 item 的请求归属关系：

* 前 3 个 item 属于请求 0；
* 接下来的 4 个 item 属于请求 3；
* 第 8 个 item 属于请求 4；
* 最后 3 个 item 属于请求 5。

---

### 1.1.5 UPS 数据的时间排序

fgout 数据中的 UPS 序列按照**从新到旧**排列，即最新行为位于数组最前面。

在 `ParquetReader` 解析时，会使用：

```python
sequence[::-1]
```

将序列反转为**从旧到新**，从而保证因果 Attention 的时间顺序正确。

即：

```text
fgout 原始顺序：最新行为 → 较早行为 → 最早行为
解析后顺序：    最早行为 → 较早行为 → 最新行为
```
