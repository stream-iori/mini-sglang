from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


class RotaryEmbedding(StateLessOP):
    """
    Rotary Position Embedding (RoPE).
    
    RoPE 是一种相对位置编码方法，通过将 Token 的 Query 和 Key 向量在复平面上旋转
    一定的角度来注入位置信息。旋转的角度取决于 Token 在序列中的绝对位置。
    
    优点:
    - 能够自然地处理相对位置关系。
    - 具有良好的外推性 (Extrapolation)，即可以处理比训练时更长的序列。
    """
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        
        # 预计算频率 (Inverse Frequency)
        # theta_i = 10000 ^ (-2(i-1)/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        
        # 可选的后处理 (例如用于 Llama 3 的 scaling)
        if post_process is not None:
            inv_freq = post_process(inv_freq)
            
        # 生成位置索引 [0, 1, ..., max_pos-1]
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        
        # 外积计算所有位置在所有频率上的角度: m * theta_i
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        
        # 计算 cos 和 sin
        cos = freqs.cos()
        sin = freqs.sin()
        
        # 缓存 cos/sin 表，避免每次 forward 重复计算
        # 形状: [max_pos, rotary_dim] (cat 后 dim 翻倍? 不，这里是 cat(cos, sin) dim=-1，如果是 interleave 可能会不同，需看具体实现)
        # 这里的实现是将 cos 和 sin 拼接到一起，供 FlashInfer Kernel 使用
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        assert self.head_size in [64, 128, 256, 512]

        try:
            from flashinfer import apply_rope_with_cos_sin_cache_inplace
            self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace
        except ImportError:
            self.apply_rope_with_cos_sin_cache_inplace = None

    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用 RoPE 到 q 和 k 上。
        
        Args:
            positions: 每个 Token 的位置索引 [batch_tokens]
            q: Query 向量
            k: Key 向量
        """
        if q.device.type == "cpu":
            # Naive CPU implementation (用于调试)
            dim = self.rotary_dim
            inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
            t = positions.float()
            freqs = torch.outer(t, inv_freq)
            # Create cos/sin
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            
            # Apply rotary
            # q: (total_tokens, num_heads, head_dim)
            def apply_rotary(x, cos, sin):
                x_rot = x[..., :dim]
                x_pass = x[..., dim:]
                
                x1 = x_rot[..., :dim//2]
                x2 = x_rot[..., dim//2:]
                
                # Standard rotary rotation
                # [-x2, x1] * sin + [x1, x2] * cos
                x_rotated = torch.cat((-x2, x1), dim=-1) * sin.unsqueeze(1) + x_rot * cos.unsqueeze(1)
                
                return torch.cat((x_rotated, x_pass), dim=-1)
                
            return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        else:
            # 使用 FlashInfer Kernel 进行高性能计算
            from flashinfer import apply_rope_with_cos_sin_cache_inplace

            self.cos_sin_cache = self.cos_sin_cache.to(q.device)
            apply_rope_with_cos_sin_cache_inplace(
                q, k, self.cos_sin_cache, self.is_neox, positions, False
            )
            return q, k


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
) -> RotaryEmbedding:
    """内部工厂函数，处理 Scaling 逻辑"""
    if rope_scaling is None:
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)
    
    # Llama 3 特有的 RoPE Scaling 策略
    match rope_scaling["rope_type"]:
        case "llama3":
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                """
                Llama 3 Scaling 逻辑:
                对不同频段的波长应用不同的缩放因子，以更好地支持长上下文。
                """
                # no smooth if low_freq_factor == high_freq_factor
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

    raise ValueError(f"Unsupported {rope_scaling = }")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    """设置 RoPE 的默认设备 (用于缓存管理)"""
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@lru_cache()
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbedding:
    """
    获取 RotaryEmbedding 实例的公共接口。
    使用 lru_cache 缓存实例，避免重复创建相同的 RoPE 对象。
    """
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]
