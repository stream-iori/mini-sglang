# Mini-SGLang Codebase Study Guide

This guide provides a structured path for understanding the [Mini-SGLang](https://github.com/sgl-project/mini-sglang) codebase. It is designed to take you from a high-level user perspective down to the low-level optimizations that make this framework fast.

## Prerequisites

- Familiarity with Python and PyTorch.
- Basic understanding of LLM inference (Transformer architecture, KV Cache, Tokenization).
- Knowledge of asynchronous programming (asyncio) is helpful for the server part.

---

## Phase 0: Orientation & Setup

Before diving into the code, ensure the environment is set up and you understand the project's goal.

1.  **Read `README.md`**: Understand the features (Radix Cache, Tensor Parallelism, etc.) and installation steps.
2.  **Read `docs/structures.md`**: This is a crucial document. It explains the process separation (Tokenizer, Scheduler, Detokenizer) and the data flow. **Keep the diagram in this file open while you read the code.**
3.  **Run a Benchmark**:
    - Run `python benchmark/offline/bench.py`.
    - This will verify your setup and give you a sense of the system's output.

---

## Phase 1: High-Level Abstractions (The Interface)

Goal: Understand how a user request interacts with the system.

### 1. The Offline Interface

- **File**: `python/minisgl/llm/llm.py`
- **Focus**:
  - The `LLM` class.
  - `generate()` method: This is the entry point for offline batch inference.
  - Notice how it creates a `Scheduler` and acts as a simplified client, sending `UserMsg` directly.

### 2. Core Data Structures

- **File**: `python/minisgl/core.py`
- **Focus**:
  - `Req` (Request): Represents a single sequence generation task.
  - `Batch`: A collection of requests running together.
  - `SamplingParams`: What the user controls (temperature, top_p, etc.).

---

## Phase 2: The Brain (Scheduler)

Goal: Understand how the system manages resources and decides what to compute next.

### 1. The Scheduler Loop

- **File**: `python/minisgl/scheduler/scheduler.py`
- **Focus**:
  - `Scheduler` class.
  - `overlap_loop()`: This is the heart of the system. It overlaps CPU work (scheduling, signal processing) with GPU work (model execution).
  - `_forward()`: The core execution step (see [Scheduler Forward Logic](scheduler_forward_logic.md)).
  - **Continuous Batching**: See [Continuous Batching Logic](continuous_batching_logic.md) for details on iteration-level scheduling.
  - `run_forever()`: The main event loop.
  - `_process_one_msg()`: Handling new requests.

### 2. Batch Construction

- **Files**: `python/minisgl/scheduler/prefill.py` & `python/minisgl/scheduler/decode.py`
- **Focus**:
  - `PrefillManager`: How new prompts are batched. Look for "Chunked Prefill" logic.
  - `DecodeManager`: How the system selects requests for the next generation step.

---

## Phase 3: The Muscle (Engine)

Goal: Understand how the model is actually executed on the GPU.

### 1. The Engine

- **File**: `python/minisgl/engine/engine.py`
- **Documentation**: `my_docs/engine_architecture.md` (Deep dive into initialization and execution)
- **Focus**:
  - `Engine` class initialization: Loading models, allocating memory.
  - `forward_batch()`: The function that calls the model.
  - `_determine_num_pages()`: How it calculates how much memory to use for the KV cache.
  - `GraphRunner` usage: CUDA Graph capturing for optimization.

### 2. Model Implementation

- **File**: `python/minisgl/models/llama.py` (or `qwen3.py`)
- **Focus**:
  - `LlamaForCausalLM`: The top-level model class.
  - `LlamaDecoderLayer`: A single Transformer layer.
  - Notice the usage of `VocabParallelEmbedding` and `ParallelLMHead` for Tensor Parallelism support.

### 3. Layers & Ops

- **Directory**: `python/minisgl/layers/`
- **Focus**:
  - `attention.py`: How FlashAttention/FlashInfer is integrated.
  - `linear.py`: Tensor Parallel linear layers (`ColumnParallelLinear`, `RowParallelLinear`).

---

## Phase 4: The Key Optimization (Radix Cache)

Goal: Understand how Mini-SGLang reuses memory for shared prefixes (e.g., system prompts, few-shot examples).

### 1. Radix Cache Manager

- **File**: `python/minisgl/kvcache/radix_manager.py`
- **Focus**:
  - `RadixTreeNode`: A node in the prefix tree.
  - `insert_prefix()`: Adding a new sequence to the cache.
  - `match_prefix()`: Finding the longest reusable prefix for a new request.
  - `evict()`: LRU eviction policy when memory is full.

---

## Phase 5: Advanced Topics (Optional)

### 1. Distributed System

- **File**: `python/minisgl/distributed/`
- **Focus**:
  - `pynccl.py`: Python bindings for NCCL (NVIDIA Collective Communications Library).
  - How the system initializes process groups for multi-GPU inference.

### 2. Custom Kernels

- **File**: `python/minisgl/kernel/`
- **Focus**:
  - Custom CUDA kernels (C++ source in `csrc/`).
  - Python wrappers for invoking these kernels.

### 3. Server Architecture

- **File**: `python/minisgl/server/`
- **Focus**:
  - `api_server.py`: FastAPI implementation using asyncio.
  - `launch.py`: How multiple processes (Tokenizer, Detokenizer, Scheduler) are spawned and connected via ZMQ.

---

## Suggested Exercises

1.  **Trace a Request**: Add log statements in `scheduler.py` inside `_process_one_msg` and `_process_last_data` to see a request enter and leave the system.
2.  **Disable Radix Cache**: Try to modify the config to disable Radix Cache and observe the performance difference on a few-shot benchmark.
3.  **Add a Metric**: Try to add a counter for "tokens generated per second" in the `Scheduler` class and print it periodically.
