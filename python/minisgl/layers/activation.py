from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU 激活函数与乘法操作的融合实现。
    
    这是 SwiGLU (Swish-Gated Linear Unit) 的核心计算单元。
    SwiGLU 的定义为: SwiGLU(x, W, V, b, c) = (SiLU(xW + b) * (xV + c))
    
    在 Mini-SGLang 中，通常 LinearColParallelMerged 会将 Gate 和 Up 投影合并，
    产生一个形状为 [total_tokens, 2 * intermediate_size] 的输出。
    本算子将该输出沿最后一个维度切分，应用 SiLU(gate) * up。
    
    Args:
        x (torch.Tensor): 输入张量，其最后一个维度包含拼接好的 Gate 和 Up 数据。
        
    Returns:
        torch.Tensor: 激活与逐元素相乘后的输出，维度减半。
    """
    from flashinfer import silu_and_mul

    return silu_and_mul(x)


__all__ = ["silu_and_mul"]
