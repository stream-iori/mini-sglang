from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    """
    Root Mean Square Layer Normalization (RMSNorm)。
    
    这是 Transformer 中常用的归一化层，相比 LayerNorm 少了减去均值的步骤，
    计算更简单且效果相当。
    
    Formula:
        x = x * weight / sqrt(mean(x^2) + eps)
    """
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        # 使用 FlashInfer 的优化 Kernel
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        """原地执行归一化，节省显存"""
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    """
    融合的 RMSNorm (Fused Add + RMSNorm)。
    
    在 Transformer 的残差连接结构中，通常模式是：
    x = x + residual
    residual = x
    x = RMSNorm(x)
    
    将 "Add Residual" 和 "RMSNorm" 两个操作融合到一个 CUDA Kernel 中，
    可以减少一次显存读写 (Memory Access)，显著提升带宽受限场景下的性能。
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
            output: 归一化后的结果 (用于下一层计算)。
            residual: 更新后的残差 (用于传给下下层)。
        """
        if x.device.type == "cpu":
             # Naive CPU implementation (用于调试或非 GPU 环境)
            if residual is not None:
                residual += x
                x = residual
            else:
                residual = x
            
            # RMSNorm: x * weight / sqrt(mean(x^2) + eps)
            variance = x.pow(2).mean(-1, keepdim=True)
            output = x * torch.rsqrt(variance + self.eps) * self.weight
            
            return output, residual
        
        # GPU 融合实现
        if residual is not None:
            # 同时完成: residual = residual + x; output = rmsnorm(residual)
            return self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        else:
            # 如果没有残差输入 (如第一层)，则退化为普通 RMSNorm
            return self.rmsnorm(x, self.weight, self.eps), x
