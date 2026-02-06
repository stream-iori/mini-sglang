from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    与传统的 LayerNorm 不同，RMSNorm 移除了均值中心化（Mean Centering）步骤，
    只保留了按方差缩放。这在计算上更高效，且在 Transformer 模型中表现良好。
    
    数学公式: y = x / sqrt(mean(x^2) + eps) * weight
    """
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：调用高性能 FlashInfer Kernel 计算 RMSNorm。
        """
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        """
        原地 (Inplace) 计算：直接修改输入张量 x，减少显存分配。
        常用于 Q/K-Normalization。
        """
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    """
    融合的 Add-RMSNorm (Fused Add-RMSNorm)。
    
    在 Transformer 的残差连接中，常见的操作是: x = x + Attention(x) -> x = Norm(x)。
    RMSNormFused 将 "加法(residual add)" 和 "归一化(norm)" 两个操作合并成一个 CUDA Kernel。
    
    优点：
    1. 减少内存带宽占用：只需要读取/写入一次数据到显存。
    2. 减少算子调度开销。
    """
    def __init__(self, size: int, eps: float) -> None:
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        try:
            from flashinfer import fused_add_rmsnorm, rmsnorm

            self.fused_add_rmsnorm = fused_add_rmsnorm
            self.rmsnorm = rmsnorm
        except ImportError:
            self.fused_add_rmsnorm = None
            self.rmsnorm = None

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。
        
        Args:
            x: 当前层的输出。
            residual: 上一层的残差输入。
            
        Returns:
            Tuple: (归一化后的输出, 更新后的残差)
        """
        if x.device.type == "cpu":
            # Naive CPU implementation (用于调试或 Mac 环境)
            if residual is not None:
                residual += x
                x = residual
            else:
                residual = x

            # RMSNorm: x * weight / sqrt(mean(x^2) + eps)
            variance = x.pow(2).mean(-1, keepdim=True)
            output = x * torch.rsqrt(variance + self.eps) * self.weight

            return output, residual

        # GPU 模式下尝试使用融合 Kernel
        if residual is not None:
            return self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        else:
            return self.rmsnorm(x, self.weight, self.eps), x
