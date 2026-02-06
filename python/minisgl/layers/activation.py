from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU 激活函数与乘法操作的融合实现。
    通常用于 SwiGLU (Swish-Gated Linear Unit) 结构中。
    
    计算公式: out = x_gate * SiLU(x_gate) * x_val
    但在 FlashInfer 的实现中，通常输入 x 已经被切分为两半 (Gate 和 Value)，
    或者该函数处理的是 split 后的逻辑。
    
    Args:
        x (torch.Tensor): 输入张量。
        
    Returns:
        torch.Tensor: 激活后的输出。
    """
    from flashinfer import silu_and_mul

    return silu_and_mul(x)


__all__ = ["silu_and_mul"]
