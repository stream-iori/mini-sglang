import os
import sys

sys.path.append(os.path.join(os.getcwd(), "python"))

import time

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM


def benchmark_cpu(model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    print(f"Benchmarking on CPU with model: {model_path}")

    # Initialize LLM (this handles Engine/Scheduler creation with our patched config)
    llm = LLM(
        model_path=model_path,
        dtype=torch.float32,
        attention_backend="naive",
        use_pynccl=False,
        use_dummy_weight=True,  # Avoid downloading large weights if just testing flow
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "What is the meaning of life?",
    ]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)  # Short generation

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    for output in outputs:
        # Since we use dummy weights, the output text will be garbage, but the flow should work.
        # We print a bit of the output to show it ran.
        print(f"Output Len: {len(output['token_ids'])} | Generated: {output['text'][:50]}...")

    print(f"Total time: {end_time - start_time:.2f}s")
    print("Benchmark finished successfully!")


if __name__ == "__main__":
    benchmark_cpu()
