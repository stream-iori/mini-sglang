# Mini-SGLang 核心数据结构与批处理逻辑

本文档详细介绍了 `Req` (请求) 和 `Batch` (批处理) 对象的设计，以及系统如何处理多请求并发推理。

---

## 1. Req (Request) 对象

`Req` 是表示单个推理请求的核心对象。它在 CPU 上维护状态，并在推理步进中更新。

### 1.1 关键长度属性对比

| 属性             | 含义               | 计算/来源                     | 动态特性                                             |
| :--------------- | :----------------- | :---------------------------- | :--------------------------------------------------- |
| `device_len`     | **当前逻辑总长度** | `len(input_ids)`              | **增加**：每生成一个新 Token，该值 +1。              |
| `max_device_len` | **终止长度上限**   | `Prompt长度 + max_new_tokens` | **固定**：初始化后不再改变。                         |
| `cached_len`     | **已入缓存长度**   | 上一步结束时的 `device_len`   | **增加**：记录已存入 KV Cache 的 Token。             |
| `remain_len`     | **剩余可生成数**   | `max_device_len - device_len` | **减少**：当降为 0 时，请求结束。                    |
| `extend_len`     | **本次待计算长度** | `device_len - cached_len`     | **固定/变动**：Prefill 时为输入长，Decode 时恒为 1。 |

### 1.2 input_ids (CPU 侧)

- **内容**：包含从原始 Prompt 到当前已生成的**所有** Token ID。
- **位置**：始终保存在 CPU 内存中，以节省昂贵的显存（GPU 侧只保存当前步需要的输入）。
- **追加**：通过 `append_host` 方法，将 GPU 预测出的最新 Token 不断拼接。

---

## 2. Batch (批处理) 对象

`Batch` 是连接调度器与模型引擎的桥梁，负责将多个 `Req` 打包进行高效的并行计算。

### 2.1 核心字段解析

- **`phase`**: 标记当前是 `prefill`（预填充，处理 Prompt）还是 `decode`（解码，生成新 Token）。
- **`input_ids` (GPU 侧)**:
  - 这是一个**扁平化的一维 Tensor**。
  - 它将当前 Batch 中所有请求本次需要计算的 Token 紧挨着拼在一起。
  - 例如：ReqA 需要计算 3 个词，ReqB 需要计算 2 个词，则 `input_ids` 长度为 5。
- **`out_loc` (输出位置索引)**:
  - 由于 `input_ids` 是一维拼接的，模型 Forward 后会输出一个很长的 Logits。
  - `out_loc` 记录了每个请求**最后一个 Token** 在一维数组中的下标。
  - 系统根据 `out_loc` 从长数组中提取出每个请求的“预测结果”，忽略中间的计算产物。
- **`attn_metadata`**:
  - 包含 Page Table (页表)、Sequence Lengths 等底层信息。
  - 它是给底层 Attention Kernel (如 FlashAttention) 使用的“说明书”，指明了每个请求的 KV Cache 在显存中的具体物理位置。

---

## 3. 连续批处理详解：隔离与提取

本节深入探讨系统如何在一维数据流中同时实现**计算隔离**（互不干扰）与**精准提取**（找到归属）。

### 3.1 为什么一维拼接不会搞混？（计算隔离）

虽然 ReqA 和 ReqB 的 Token 在物理内存（一维数组）中是相邻的，但它们在计算过程中是**逻辑隔离**的：

1.  **非 Attention 层（MLP, LayerNorm）**: 这些层是逐 Token 计算的，Token 之间本身就没有数据依赖，拼在一起只会增加并行度，不会互相干扰。
2.  **Attention 层 (核心机制)**:
    - **Attention Mask**: 系统会生成掩码（或者通过 `cu_seqlens` 约束计算范围）。
    - 在计算 ReqA 的注意力时，GPU Kernel 会被限制只能读取 ReqA 对应的 KV Cache 区域，物理上“看不见”ReqB 的数据。
    - **结果**: ReqA 只能关注到 ReqA 的历史，ReqB 只能关注到 ReqB 的历史。

### 3.2 out_loc 的全生命周期（结果提取）

`out_loc` (Output Locator) 是连接扁平化计算结果与独立请求对象的导航图。它解决了“在混合的一堆输出中，哪一个是属于我的下一个词预测结果”的问题。

#### 场景一：Prefill (预填充) 阶段

在 Prefill 阶段，模型一次性处理 Prompt 中的多个 Token，但只有**最后一个 Token** 的输出包含对“下一个词”的预测。

**例子**:
假设 Batch 中有两个请求：

- **Req A**: Prompt 为 `[Why, is, the]` (长度 3)
- **Req B**: Prompt 为 `[Hello, world]` (长度 2)

**1. 准备阶段 (Prepare Input)**:
系统将它们拼接到 `input_ids`:
`Index:      0     1    2      3      4`
`input_ids: [Why, is, the, Hello, world]`

此时，调度器计算 `out_loc`。我们需要的是 Req A 的最后一位（Index 2）和 Req B 的最后一位（Index 4）。

- **`out_loc` = `[2, 4]`**

**2. 模型计算 (Forward)**:
模型输出 `logits`，形状为 `[5, Vocab_Size]`。这代表了 5 个位置各自对“下一个词”的预测概率。

- **Index 0, 1 (`Why`, `is`) 的输出**：
  - 虽然模型计算了这两个位置的预测概率（例如 Index 0 预测出了 `is`），但因为这些词已经存在于 Prompt 中，我们**不需要**这些预测。
  - **重要价值**：虽然丢弃了它们的预测结果，但它们在计算中产生的 **KV Cache** 被存入了显存。当计算 Index 2 时，模型会通过 Attention 机制“读取”这些缓存，从而让 Index 2 携带了完整的上下文信息。
- **Index 2 (`the`) 的输出**：
  - **关键数据**：这是 Req A 的末尾。它对“下一个词”的预测才是我们真正需要的第一个生成词（例如 `sky`）。
- **Index 3 (`Hello`) 的输出**：Req B 的中间过程，同理丢弃。
- **Index 4 (`world`) 的输出**：
  - **关键数据**：这是 Req B 的末尾，包含对 Req B 下一个词的预测。

**3. 提取阶段 (Extract)**:
系统执行 `logits[out_loc]` 操作（即 `logits[[2, 4]]`）：

- 提取 `logits[2]` -> 赋给 Req A 进行采样。
- 提取 `logits[4]` -> 赋给 Req B 进行采样。

---

#### 场景二：Decode (解码) 阶段

在 Decode 阶段，每个请求每次只生成 **1个** 新 Token。所有的输入都是“最后一个 Token”。

**例子**:
假设 Batch 中有两个请求正在生成：

- **Req A**: 上一步生成了 `sky`
- **Req B**: 上一步生成了 `!`

**1. 准备阶段 (Prepare Input)**:
拼接 `input_ids`:
`Index:      0    1`
`input_ids: [sky, !]`

调度器计算 `out_loc`。每个位置都是各自请求的末尾。

- **`out_loc` = `[0, 1]`**

**2. 提取阶段 (Extract)**:
模型输出 `logits` 形状 `[2, Vocab_Size]`。
直接提取：

- `logits[0]` -> Req A 的下一个词。
- `logits[1]` -> Req B 的下一个词。

### 3.3 总结：处理流程图

1.  **调度器** 挑选多个 `Req`，计算各自的 `extend_len`。
2.  **准备数据**：拼成一维 `input_ids`，计算 `out_loc` 索引（指向每个 extend 片段的末尾），准备 `attn_metadata`。
3.  **GPU Forward**：
    - 模型根据 `attn_metadata` 隔离各请求。
    - 模型计算出一维的总输出。
4.  **提取结果**：根据 `out_loc` 精准地把每个请求的预测值（Next Token Logits）拿出来，丢弃中间 Token 的输出。
5.  **更新状态**：每个请求在 CPU 上更新自己的 `device_len` 和 `cached_len`。

---

## 4. out_loc 的深度解析：从逻辑位置到物理存储

`out_loc` (Output Location) 是 Mini-SGLang 中连接“计算产物”与“物理存储”的关键纽带。虽然在逻辑概念上它常被理解为结果提取的导航图，但在底层代码实现中，它承载着更核心的显存管理职责。

### 4.1 物理本质：物理页索引 (Physical Page Indices)

在 `Scheduler._prepare_batch` 阶段，系统会为当前批次中所有需要计算的新 Token 分配物理显存空间。

- **分配逻辑**：`needed_size = sum(req.extend_len for req in batch.reqs)`。
- **存储内容**：`out_loc` 是一个 1D Tensor，存储了这些新 Token 对应在 KV Cache 池中的 **物理页号 (Physical Page Index)**。
- **映射关系**：这些索引随后被写入全局 `page_table`，完成逻辑位置（第几个 Token）到物理地址（显存哪个位置）的映射。

### 4.2 双重职责：KV 写入与结果定位

在推理循环中，`out_loc` 的概念在不同层级有不同的表现：

1.  **KV Cache 存储定位 (Storage)**：
    - 这是代码中 `batch.out_loc` 的直接用途。
    - 模型计算出 K/V 向量后，Attention Backend 调用 `store_kv(k, v, batch.out_loc, layer_id)`，将结果存入显存池。
2.  **结果提取定位 (Logit Extraction)**：
    - 在 Prefill 阶段，模型输出长序列 Logits，我们需要提取每个请求最后一个 Token 的输出。
    - **实现细节**：在当前代码中，这部分索引由 `attn_metadata.get_last_indices()` 提供。虽然在本文档的逻辑描述中将其统称为 `out_loc`，但在代码中它是通过 `cu_seqlens - 1` 动态计算得到的。

### 4.3 为什么需要 out_loc？

如果没有 `out_loc` 进行动态物理映射，系统将面临以下问题：

- **碎片化严重**：必须分配连续显存，无法处理不确定长度的请求。
- **无法实现 Prefix Caching**：物理存储位置被固定，无法在不同请求间灵活复用公共前缀。
- **显存利用率低**：无法根据实际计算需求（Extend Len）按需申请空间。

通过 `out_loc`，Mini-SGLang 实现了 **PagedAttention** 的核心理念：**计算在逻辑上是连续的（1D 拼接），而数据在物理上是离散的（Paged 存储）**。
