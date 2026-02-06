# Token IDs 和 Input IDs 解析

本文档详细解析 `mini-sglang` 中 `token_ids` 和 `input_ids` 的概念、结构、数值类型以及它们在系统中的流转过程。

## 1. 核心概念

在 LLM 推理系统中，模型无法直接理解文本字符串，只能处理数字。`token_ids` 和 `input_ids` 本质上都是指代这些数字索引。

- **Token**: 文本被切分后的最小单位（如单词、子词或字符）。
- **Token ID**: 词表中每个 Token 对应的唯一整数索引。
- **Vocabulary Size (词表大小)**: Token ID 的取值范围，通常为 `[0, vocab_size - 1]`。

在代码实现中，这两个术语通常指代不同上下文下的同一类数据：

- **`token_ids`**: 这是一个**通用术语**，通常指代一串逻辑上的 Token ID 列表。例如用户输入的 Prompt 被 Tokenize 后的结果，或者模型生成的输出结果。
- **`input_ids`**: 这是一个**特定术语**，特指**输入给模型进行计算**的那部分 Token IDs。它的具体内容会随着推理阶段（Prefill vs Decode）而变化。

## 2. 数据结构与数值

### 2.1 数值类型

- **类型**: 整数 (Integer)。
- **Python 类型**: `int`。
- **PyTorch 类型**: `torch.int32` (通常不使用 int64 以节省显存，且词表大小通常 < 2^31)。
- **范围**: `0` 到 `vocab_size - 1`。
  - Llama 2: ~32000
  - Qwen: ~152000
- **特殊值**: 包含特殊的控制 Token，如 `EOS` (End of Sentence), `BOS` (Begin of Sentence), `PAD` (Padding)。

### 2.2 形状 (Shape)

它们的形状取决于所在的容器和处理阶段。

#### A. 在 `Req` 对象中 (逻辑请求)

每个 `Req` 对象代表一个独立的请求。

- **`req.input_ids`**:
  - **类型**: `torch.Tensor` (CPU)
  - **形状**: 1D Tensor, `[current_seq_len]`
  - **内容**: 包含了该请求**迄今为止的所有 Token** (Prompt + Generated)。
  - **变化**: 每生成一个新 Token，会被 `append` 到末尾。
  - **示例**: `tensor([101, 2034, 55, 908], dtype=torch.int32)`

#### B. 在 `Batch` 对象中 (计算批次)

`Batch` 对象用于一次模型 Forward 计算。此时的 `input_ids` 是经过拼接和处理的。

- **`batch.input_ids`**:
  - **类型**: `torch.Tensor` (GPU)
  - **形状**: 1D Tensor, `[total_batch_tokens]`
  - **内容**:
    - **Prefill 阶段**: 所有请求的 Prompt 拼接在一起。
      `[ReqA_Prompt, ReqB_Prompt, ...]`
    - **Decode 阶段**: 所有请求的**最后一个生成的 Token** 拼接在一起。
      `[ReqA_LastToken, ReqB_LastToken, ...]`
  - **示例**:
    - Prefill: `tensor([1, 2, 3, 4, 1, 5, 6], device='cuda')` (ReqA长4, ReqB长3)
    - Decode: `tensor([99, 88], device='cuda')` (ReqA的最新词99, ReqB的最新词88)

## 3. 详细流转过程

以下是一个从用户输入到模型输出的完整生命周期，展示 `token_ids` 和 `input_ids` 的变化。

### 3.1 输入处理 (Frontend)

1.  **Tokenization**:
    - 用户输入: `"Hello world"`
    - Tokenizer 处理: `["Hello", "world"]` -> `[15496, 995]`
    - 变量名: 这里的 `[15496, 995]` 通常被称为 `token_ids` 或 `prompt_token_ids`。

2.  **创建 Request**:
    - 系统将其转换为 Tensor 并封装进 `Req` 对象。
    - `req.input_ids = tensor([15496, 995], device='cpu')`

### 3.2 第一次推理 (Prefill Phase)

这是处理 Prompt 的阶段。

1.  **Scheduler 准备**:
    - 调度器选中该 `Req`。
    - 因为是第一次运行，需要计算整个 Prompt。
    - 构造 `ForwardInput`。
2.  **生成 `batch.input_ids`**:
    - Scheduler 根据 `load_indices` 从 `token_pool` (或者直接从 `req`，视实现细节) 提取数据。
    - **`batch.input_ids` = `tensor([15496, 995], device='cuda')`**
    - 注意：如果 Batch 里有多个请求，这里会是长拼接。
3.  **Model Forward**:
    - 模型接收这个 1D Tensor。
    - Embedding 层查表: `Embed(15496)`, `Embed(995)`。
    - 输出 Logits。

### 3.3 后续推理 (Decode Phase)

假设模型预测的第一个词是 `"!"` (ID: 0)。

1.  **结果回写**:
    - 生成的 `0` 被追加到 `Req` 对象中。
    - `req.input_ids` 变为 `tensor([15496, 995, 0], device='cpu')`。
2.  **Scheduler 准备 (下一轮)**:
    - 调度器再次选中该 `Req`。
    - 这次是 Decode 阶段，只需要计算最新的 Token。
3.  **生成 `batch.input_ids`**: - 此时只提取最后一个词。- **`batch.input_ids` = `tensor([0], device='cuda')`**
    > 这一块是重点,Decode Phase,的batch.input_ids只有最后一个token_id,但是连接着其他req的token_id,所以也叫input_ids
4.  **Model Forward**:
    - 模型只接收 `[0]` 作为输入。
    - 结合 KV Cache (里面存了 "Hello world" 的信息) 计算下一个词。

## 4. 关键代码位置

- **`python/minisgl/llm/llm.py`**:
  - `_tokenize_one`: 将文本转为 `input_ids` (Tensor)。
  - `RequestStatus.input_ids`: 存储原始输入的 `List[int]`。
  - `RequestStatus.output_ids`: 存储生成结果的 `List[int]`。

- **`python/minisgl/core.py`**:
  - `Req.input_ids`: 维护单个请求完整历史的 CPU Tensor。
  - `Batch.input_ids`: 模型计算时使用的 GPU Tensor (Field definition)。

- **`python/minisgl/scheduler/scheduler.py`**:
  - `_load_token_ids`: 核心逻辑，利用 `load_indices` 从 `token_pool` 提取出当前 Batch 需要的 `input_ids`。

## 5. 总结

| 术语                  | 位置         | 类型        | 形状                                                      | 内容                      | 作用                           |
| :-------------------- | :----------- | :---------- | :-------------------------------------------------------- | :------------------------ | :----------------------------- |
| **User Prompt IDs**   | Frontend     | `List[int]` | `[L]`                                                     | 原始 Prompt               | 用户输入的起点                 |
| **`req.input_ids`**   | CPU (Req)    | `Tensor`    | `[L + Gen]`                                               | Prompt + 历史生成         | 维护请求的完整状态             |
| **`batch.input_ids`** | GPU (Batch)  | `Tensor`    | `[Batch_Size]` (Decode) 或 `[Sum(Prompt_Lens)]` (Prefill) | **本次**需要计算的 Tokens | 喂给模型进行 Embedding         |
| **`token_pool`**      | GPU (Global) | `Tensor`    | `[Max_Reqs, Max_Len]`                                     | 所有活跃请求的全部 Token  | 全局数据源，避免 CPU->GPU 拷贝 |

## 6. 关联文档

- [Memory Management (显存管理)](./memory_management.md): 详细介绍了 `token_pool` 的结构以及 `input_ids` 如何与 KV Cache 配合。
- [Scheduler Forward Logic (调度器前向逻辑)](./scheduler_forward_logic.md): 深入解析了 `_load_token_ids` 如何通过 `load_indices` 从 `token_pool` 提取数据。
- [Batch and Request Logic (Batch 与 Request 逻辑)](./batch_and_request_logic.md): 解释了 `Req` 和 `Batch` 对象的更广泛的生命周期管理。
