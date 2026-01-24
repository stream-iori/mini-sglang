import os
import sys

import torch

# Ensure we can import minisgl
sys.path.append(os.path.join(os.getcwd(), "python"))

from minisgl.distributed import DistributedInfo
from minisgl.engine.config import EngineConfig
from minisgl.engine.engine import Engine


def test_engine_init():
    print("Testing Engine Initialization on CPU...")
    try:
        config = EngineConfig(
            model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Use a small model
            tp_info=DistributedInfo(0, 1),
            dtype=torch.float32,  # CPU usually prefers float32
            attention_backend="naive",
            use_pynccl=False,
            use_dummy_weight=True,  # Use dummy weights to avoid downloading
        )

        engine = Engine(config)
        print("Engine initialized successfully!")
        print(f"Device: {engine.device}")
        print(f"Attention Backend: {type(engine.attn_backend)}")

    except Exception as e:
        print(f"Engine initialization failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_engine_init()
