# Scheduler Forward Logic Analysis

本文档详细解析 `minisgl/scheduler/scheduler.py` 中的 `_forward` 方法逻辑，以及它如何与 `Engine`、`Model` (以 Qwen3 为例) 和底层数据结构进行交互。

## 1. 核心概念与数据结构

在深入逻辑之前，需要理解几个关键的数据容器，它们在 Scheduler 和 Engine 之间传递信息。

### 1.1 `ForwardInput` (NamedTuple)

这是 `Scheduler` 准备好的一组数据，包含了执行一次 Forward 所需的所有信息。

- **`batch` (`Batch`)**: 包含当前批次的所有请求 (`Req` 对象列表)，以及用于 Attention 的元数据 (如 `out_loc` KV Cache 指针)。
- **`sample_args` (`BatchSamplingArgs`)**: 采样所需的参数（如温度、top_p 等），由 `Engine.sampler` 准备。
- **`load_indices` (`torch.Tensor`)**: 1D 索引张量。用于从全局 `token_pool` (2D Table) 中“gather”出当前 Batch 需要的 Input Token IDs。
- **`write_indices` (`torch.Tensor`)**: 1D 索引张量。用于将模型生成的 Next Token ID “scatter”回全局 `token_pool` 的正确位置。

### 1.2 `ForwardOutput` (NamedTuple)

这是 `Engine` 执行完计算后的返回结果。

- **`next_tokens_gpu`**: 在 GPU 上生成的下一个 Token ID 列表。
- **`next_tokens_cpu`**: 异步拷贝到 CPU 的 Token ID 列表 (用于 Scheduler 逻辑判断，如 EOS 检查)。
- **`copy_done_event`**: CUDA Event，用于同步 GPU->CPU 的拷贝操作。

### 0.3 `token_pool` (2D Tensor)

`TableManager` 管理的一个大张量，存储了所有活跃请求的 Token ID 历史。

- **结构**: `[max_reqs, max_seq_len]`。
- **作用**: 作为 CPU (Scheduler 逻辑) 和 GPU (Model 计算) 之间的共享数据缓冲区。

### 1.4 `out_loc` (KV Cache 物理指针)

`out_loc` 是 `Batch` 对象中的一个关键字段，用于管理 KV Cache 的显存写入位置。

- **类型**: `torch.Tensor` (1D, Int32)
- **长度**: 等于当前 Batch 中所有请求**本次需要计算**的 Token 总数 (Prefill 阶段为 Prompt 长度，Decode 阶段为 1)。
- **含义**: 数组中的每个元素代表一个 Token 对应的 KV Cache 在显存池中的**物理页号 (Physical Page Index)**。
  - 这正是 [`memory_management.md`](./memory_management.md) 中定义的 **1.2 Page Table** 里的值，也是 **1.5 KV Cache** 的物理索引。
  - 它指向显存中实际存储 K/V 数据的物理块 (Block/Page)。在 Mini-SGLang 中通常 Page Size = 1，这意味着一个 Page Index 唯一对应显存池中的一个 Token 槽位。(一个Token槽位里存储的是什么东西)
- **生成方式**:
  - 在 `Scheduler._prepare_batch` 中，根据当前 Batch 需要的新增 Token 数，调用 `self.cache_manager.allocate(needed_size)` (见 [`memory_management.md`](./memory_management.md) 的 **CacheManager** 部分) 分配物理空间。
  - 分配得到的物理索引被存入 `out_loc`。
- **作用**:
  1. **更新页表**: `Scheduler` 将 `out_loc` 中的物理索引填入全局 `page_table`，建立逻辑位置 (Req, Token Index) 到物理位置 (KV Slot) 的映射。
  2. **KV 写入**: 在 `Engine` 执行 Forward 时，模型 Attention 层调用 `kvcache.store_kv(k, v, batch.out_loc, ...)`。底层的 CUDA Kernel 会根据 `out_loc` 指示的地址，将计算好的 Key/Value 向量直接写入显存池的对应位置，供后续的 Attention 算子使用。

### 1.5 `load_indices` & `write_indices` (数据搬运指挥棒)

这两个 `Tensor` 是 Scheduler 精确控制数据在全局 `token_pool` (2D) 和当前计算 `Batch` (1D) 之间流动的“指挥棒”。

- **本质**: 它们都是一维的 `Int32` 索引数组，指向 flatten 后的 `token_pool` 中的特定位置。
- **生成逻辑**: 通过辅助函数 `_make_2d_indices` 生成。该函数将一组 2D 切片范围 `(行号, 起始列, 结束列)` 转换为一维物理内存偏移量：
  `offset = 行号 * max_seq_len + 列号`

#### 1. `load_indices` (Gather 指令)

- **用途**: 定义了**当前 Forward 需要读取哪些 Token 作为输入**。
- **构建规则**: `[(r.table_idx, r.cached_len, r.device_len) ...]`
  - **Prefill 阶段**: `cached_len`=0, `device_len`=Prompt长度。选取该请求的**整个 Prompt**。
  - **Decode 阶段**: `cached_len`=总长-1, `device_len`=总长。只选取该请求的**最后一个 Token** (即上一步生成的 Token)。
- **执行动作**: `batch.input_ids = token_pool.view(-1)[load_indices]`

#### 2. `write_indices` (Scatter 指令)

- **用途**: 定义了**模型生成的 Next Token 应该存放在哪里**。
- **构建规则**: `[(r.table_idx, r.device_len, r.device_len + 1) ...]`
  - 指向每个请求序列当前末尾的**下一个空位**。
- **执行动作**: `token_pool.view(-1)[write_indices] = next_tokens_gpu`

这种设计使得 Model 和 Engine 不需要知道 Token 存储的复杂 2D 结构，只需要对着一维的 `input_ids` 计算，并吐出一维的 `next_tokens`，由 Scheduler 负责将它们“归位”。

---

## 2. `_forward` 方法流程详解

`_forward` 方法是 Scheduler 驱动计算的核心步骤。它连接了数据准备、模型执行和结果回写。

```python
def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
    """执行模型前向传播"""
    # 1. 加载数据
    self._load_token_ids(forward_input)
    batch, sample_args = forward_input.batch, forward_input.sample_args

    # 2. 执行引擎计算
    forward_output = self.engine.forward_batch(batch, sample_args)

    # 3. 写回结果
    self._write_token_ids(forward_input, forward_output)

    # 4. 重新调度
    self.decode_manager.add_reqs(forward_input.batch.reqs)

    return forward_output
```

### 步骤 1: `_load_token_ids`

- **目的**: 填充 `batch.input_ids`。
- **逻辑**: 使用 `forward_input.load_indices` 从 `self.token_pool` 中提取 Token IDs。
- **代码**: `input.batch.input_ids = self.token_pool.view(-1)[input.load_indices]`
- **对于 Prefill 请求**: 提取完整的 Prompt Tokens。
- **对于 Decode 请求**: 通常只提取最后一个生成的 Token。

### 步骤 2: `self.engine.forward_batch`

这是调用底层推理引擎的入口。

- **Context 设置**: 使用 `with self.ctx.forward_batch(batch):` 将当前 Batch 设置为全局上下文。这允许模型内部 (如 `Qwen3ForCausalLM`) 通过 `get_global_ctx()` 访问 `input_ids`，而不需要显式传参。
- **模型执行**: 调用 `self.model.forward()`。
- **状态更新**: 调用 `req.complete_one()` 更新每个请求的状态 (如剩余生成长度减 1)。
- **采样**: 调用 `self.sampler.sample(...)` 生成下一个 Token。

### 步骤 3: `_write_token_ids`

- **目的**: 保存生成结果。
- **逻辑**: 将 `forward_output.next_tokens_gpu` 写入 `self.token_pool`。
- **位置**: 由 `forward_input.write_indices` 指定，通常是每个请求序列的下一个空白位置。

### 步骤 4: 重新调度

- `self.decode_manager.add_reqs(...)`: 将刚刚运行完的请求重新加入 Decode 队列。如果请求是 Prefill 阶段结束，它现在正式进入 Decode 阶段。

---

## 3. Qwen3 模型调用链路

当 `engine.forward_batch` 调用 `self.model.forward()` 时，针对 Qwen3 模型，调用栈如下：

1.  **`Qwen3ForCausalLM.forward()`** (`minisgl/models/qwen3.py`)
    - **获取输入**: `get_global_ctx().batch.input_ids`。注意这里是从全局 Context 获取的，正是 Scheduler 在 `_load_token_ids` 中准备的数据。
    - **调用 Backbone**: `self.model.forward(input_ids)`。

2.  **`Qwen3Model.forward(input_ids)`**
    - **Embedding**: `self.embed_tokens.forward(input_ids)`。
    - **Layer 循环**: 遍历 `self.layers` (类型为 `OPList[Qwen3DecoderLayer]`)。
    - **Norm**: `self.norm.forward(x)`.

3.  **`Qwen3DecoderLayer.forward(x, residual)`**
    - **Input Norm**: `RMSNormFused`。
    - **Self Attention**: `Qwen3Attn`。这里会用到 `Engine` 初始化时创建的 `attn_backend` (如 FlashInfer)，利用 `Batch` 中的元数据 (Page Table, KV Cache 指针) 进行 Attention 计算。
    - **Post Attn Norm**: `RMSNormFused`。
    - **MLP**: `Qwen3MLP`。

4.  **回到 `Qwen3ForCausalLM`**
    - **LM Head**: `self.lm_head.forward(output)` 计算 Logits。

---

## 4. 关联关系总结

| 组件          | 方法                       | 作用           | 数据流向                           |
| :------------ | :------------------------- | :------------- | :--------------------------------- |
| **Scheduler** | `_prepare_batch`           | 准备索引和内存 | `Req`s -> `ForwardInput` (indices) |
| **Scheduler** | `_load_token_ids`          | 准备输入数据   | `token_pool` -> `Batch.input_ids`  |
| **Engine**    | `forward_batch`            | 协调模型与采样 | `Batch` -> `ForwardOutput`         |
| **Model**     | `Qwen3ForCausalLM.forward` | 神经网络计算   | `Batch.input_ids` -> `Logits`      |
| **Sampler**   | `sample`                   | 采样生成 Token | `Logits` -> `Next Tokens`          |
| **Scheduler** | `_write_token_ids`         | 持久化结果     | `Next Tokens` -> `token_pool`      |

这一整套逻辑确保了 Scheduler 可以在不关心具体模型架构的情况下，高效地管理内存和批处理，而模型实现只需要关注计算本身，通过全局 Context 获取输入。
