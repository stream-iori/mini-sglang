from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
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
        if x.device.type == "cpu":
            # Naive CPU implementation
            if residual is not None:
                residual += x
                x = residual
            else:
                residual = x

            # RMSNorm: x * weight / sqrt(mean(x^2) + eps)
            variance = x.pow(2).mean(-1, keepdim=True)
            output = x * torch.rsqrt(variance + self.eps) * self.weight

            return output, residual

        if residual is not None:
            return self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        else:
            return self.rmsnorm(x, self.weight, self.eps), x
