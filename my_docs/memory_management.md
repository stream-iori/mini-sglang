# 显存管理机制：Slots, Tables, 和 Radix Cache

本文档汇总了 `mini-sglang` 中显存管理的分析，重点关注 `TableManager`、`CacheManager`、`RadixCache` 之间的交互，以及 `page_table` 和 `token_pool` 等核心数据结构。

## 1. 核心概念与术语

要理解系统如何管理并发请求的 GPU 显存，我们必须区分“逻辑资源”（Slots）和“物理资源”（Pages）。

### 1.1 Slot (槽位)

- **定义**：分配给正在运行的请求的逻辑标识符（整数）。
- **范围**：`[0, max_running_reqs - 1]`。
- **作用**：作为全局元数据张量（`page_table`、`token_pool`）的 **行索引 (Row Index)**。
- **生命周期**：请求开始时分配，执行期间持有，请求结束时释放。

### 1.2 Page Table (页表)

- **定义**：GPU 上的 2D Tensor，形状为 `[max_running_reqs, max_seq_len]`。
- **结构**：
  - **行 (Row)**：对应一个 **Slot**。
  - **值 (Value)**：存储 **物理页索引 (Physical Page Indices)**。
- **示例**：`page_table[slot_id][:cached_len]` 返回该请求 KV 数据存储的物理页列表。

### 1.3 Token Pool (Token 池)

- **定义**：GPU 上的 2D Tensor，形状与 `page_table` 相同。
- **结构**：
  - **行 (Row)**：对应一个 **Slot**。
  - **值 (Value)**：存储实际的 **Token IDs**（整数）。
- **目的**：允许 GPU kernel 通过 Slot ID 以合并访问（coalesced access）的方式读取所有活跃请求的输入 token。

### 1.4 Physical Page (物理页 / KV Cache 块)

- **定义**：全局 `kv_cache` 张量中的一块连续内存。
- **标识符**：`Page Index` (例如 55)。
- **容量**：存储固定数量 token 的 KV 对（由 `page_size` 定义）。
- **映射**：`Physical Page 55` -> `kv_cache[..., 55, ...]`.

---

## 2. 架构与关系

下图展示了这些组件如何交互。

### 2.1 组件关系图

```mermaid
classDiagram
    class Scheduler {
        +TableManager table_manager
        +CacheManager cache_manager
        +schedule()
    }

    class TableManager {
        +List~int~ _free_slots
        +Tensor page_table
        +Tensor token_pool
        +allocate() int
        +free(slot)
    }

    class CacheManager {
        +List~int~ free_list
        +RadixCache radix_cache
        +allocate(num_pages)
        +free_and_cache_finished_req(...)
    }

    class RadixCache {
        +TreeNode root
        +match_prefix(tokens)
        +insert(tokens, page_indices)
        +evict(num)
    }

    class Req {
        +int table_idx (Slot ID)
        +List~int~ input_ids
    }

    Scheduler --> TableManager : 1. 分配 Slot
    Scheduler --> CacheManager : 2. 分配 Pages
    TableManager "1" -- "N" Req : 管理 ID
    CacheManager *-- RadixCache : 用于复用
    Req --> TableManager : 索引 page_table[slot]
```

### 2.2 "Slot" 桥梁 (概念视图)

**Slot** 是连接 Request 对象与其 GPU 资源的中心枢纽。

```mermaid
graph TD
    UserReq[Request 对象] -->|持有| SlotID[Slot ID: 5]

    SlotID -->|行索引| PageTable[Page Table 张量]
    SlotID -->|行索引| TokenPool[Token Pool 张量]

    PageTable -->|包含| PageIndices[物理页索引: 10, 88, 99]
    TokenPool -->|包含| TokenIDs[Token IDs: 101, 202, 303...]

    PageIndices -->|映射到| KVCache[物理 KV Cache 显存]
```

---

## 3. 工作流分析

### 3.1 分配与执行流程

此时序图跟踪一个新请求 `[A, B, C]` 进入系统的过程。

1.  **Slot 分配**：请求获得 Slot `5`。
2.  **显存查找**：系统检查 `[A, B]` 是否之前出现过。
3.  **映射**：物理页号写入 `page_table` 的第 `5` 行。
4.  **回收**：请求结束时，页面被发送到 `RadixCache` 而不是直接销毁。

```mermaid
sequenceDiagram
    participant S as Scheduler
    participant TM as TableManager
    participant CM as CacheManager
    participant RC as RadixCache
    participant GPU_PT as PageTable (GPU)
    participant GPU_KV as KV Cache (GPU)

    Note over S: 新请求: [A, B, C]

    %% 1. Slot Allocation
    S->>TM: allocate()
    TM-->>S: 返回 Slot ID = 5

    %% 2. Prefix Matching
    S->>CM: match_req([A, B, C])
    CM->>RC: match_prefix([A, B, C])

    alt 前缀命中 (发现 A, B)
        RC-->>CM: 找到索引 [10, 20] (对应 A, B)
        Note right of CM: 复用页面 10, 20
    else 前缀未命中
        RC-->>CM: None
    end

    %% 3. Allocation of Remainder
    CM-->>S: 返回已缓存的 [10, 20]
    S->>CM: allocate(还需要 'C')
    CM-->>S: 返回新页面 [99]

    %% 4. GPU Mapping
    Note over S: 请求 5 使用页面 [10, 20, 99]
    S->>GPU_PT: 将 [10, 20, 99] 写入第 5 行
    S->>TM: 将 [A, B, C] 写入 TokenPool 第 5 行

    %% 5. Execution
    S->>GPU_KV: 模型 Forward (读取 PageTable[5])

    %% 6. Recycling
    Note over S: 请求完成
    S->>TM: free(5) (Slot 5 现已空闲)

    S->>CM: free_and_cache(tokens=[A,B,C], pages=[10,20,99])
    CM->>RC: insert([A,B,C], [10,20,99])
    Note right of RC: 树已更新。<br/>下一个请求 [A, B, C, ...] 将复用 10, 20, 99。
```

### 3.2 Radix Cache 与驱逐 (Eviction) 流程

当显存已满时，`CacheManager` 和 `RadixCache` 如何协作。

```mermaid
sequenceDiagram
    participant Allocator as PrefillAdder
    participant CM as CacheManager
    participant RC as RadixCache

    Allocator->>CM: allocate(需要 10 页)

    alt 空闲列表页数 < 10
        CM->>RC: evict(需要 10)

        loop 直到满足需求
            RC->>RC: 查找 LRU 叶子节点
            RC->>RC: 移除节点 & 剥离页面
            RC-->>CM: 返回页面 [55, 56...]
        end

        CM->>CM: 将 [55, 56...] 加入空闲列表
    end

    CM-->>Allocator: 返回 10 个页面
```

## 4. 关键总结

1.  **Slot $\neq$ 物理内存**：Slot 只是一个目录条目。它通过 Page Table 指向物理内存。
2.  **Page Table 是 2D 的**：它映射 `(Request, Logical_Pos)` -> `Physical_Page`。
3.  **Token Pool 镜像了 Page Table**：它映射 `(Request, Logical_Pos)` -> `Token_ID`。
4.  **Radix Cache 维护状态**：物理页很少被传统意义上的“释放”。它们在“活跃请求” -> “Radix 树 (缓存)” -> “空闲列表 (被驱逐)” -> “活跃请求” 之间流转。

```

```

