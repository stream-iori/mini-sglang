# Instructions to Apply CPU Patch and Run Benchmark

This patch enables `mini-sglang` to run on macOS (CPU-only) by bypassing CUDA dependencies and implementing a naive attention mechanism.

## 1. Apply the Patch
The necessary code changes have already been applied to the codebase. These include:
- Added `NaiveBackend` in `minisgl/attention/naive.py` for CPU attention.
- Registered `naive` backend in `minisgl/attention/__init__.py`.
- Modified `EngineConfig` default to use `naive` backend and disable PyNCCL.
- Modified `Engine` to initialize on CPU and bypass CUDA graph/streams.
- Modified `Scheduler` to remove CUDA stream dependencies.
- Added CPU fallback for RoPE in `minisgl/layers/rotary.py`.

## 2. Verify Installation
Run the test script to ensure the engine initializes correctly on CPU:

```bash
python test_cpu_patch.py
```

Expected output:
```
Testing Engine Initialization on CPU...
...
Engine initialized successfully!
Device: cpu
Attention Backend: <class 'minisgl.attention.naive.NaiveBackend'>
```

## 3. Run Benchmark

### Run simple benchmark script
```bash
python benchmark/offline/bench_cpu.py
```

### Launch API Server on CPU
To launch an OpenAI-compatible server on your Mac (CPU):
```bash
python -m minisgl \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --dtype float32 \
    --attn naive \
    --disable-pynccl \
    --dummy-weight
```
*Note: Remove `--dummy-weight` if you want real model outputs (requires downloading weights).*

### Interactive Shell on CPU
```bash
python -m minisgl \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --dtype float32 \
    --attn naive \
    --disable-pynccl \
    --dummy-weight \
    --shell
```
