from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import divide_even

from .base import BaseKVCache, KVCacheLayout


class MHAKVCache(BaseKVCache):
    """
    Multi-Head Attention KV Cache 实现。
    这是一个巨大的显存池，物理上预分配了所有可能的 KV Cache 空间。
    
    特点：
    1. **预分配**: 初始化时一次性申请整个显存块，避免运行时的碎片和分配开销。
    2. **5D/6D Tensor**: 内部维护一个巨大的 Tensor 来存储所有层、所有页的 K 和 V。
    3. **物理索引**: 通过物理索引 (index) 直接访问显存位置，不关心逻辑请求 ID。
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        dtype: torch.dtype,
        kv_layout: KVCacheLayout,
        device: torch.device,
    ):
        # 获取 Tensor Parallel 信息，计算本地需要负责的 Head 数量
        tp_info = get_tp_info()
        local_kv_heads = divide_even(num_kv_heads, tp_info.size)
        
        # 根据指定的布局初始化巨大的 Tensor
        match kv_layout:
            case KVCacheLayout.PageFirst:
                # [2, num_pages, num_layers, local_kv_heads, head_dim]
                # 这种布局可能对某些特定的 Kernel 访问模式更友好
                kv_buffer = torch.empty(
                    (2, num_pages, num_layers, local_kv_heads, head_dim),
                    device=device,
                    dtype=dtype,
                ).permute(0, 2, 1, 3, 4) # 调整为标准视图以便统一访问
            case KVCacheLayout.LayerFirst:
                # [2, num_layers, num_pages, local_kv_heads, head_dim]
                # 这是最常用的布局，方便按层切片
                kv_buffer = torch.empty(
                    (2, num_layers, num_pages, local_kv_heads, head_dim),
                    device=device,
                    dtype=dtype,
                )
            case _:
                raise ValueError(f"Unsupported kv_layout: {kv_layout}")
        
        # 保存为统一的 6D 视图: [2, layers, pages, 1, heads, head_dim]
        # 这里的 '1' 是 page_size (在当前实现中固定为 1)
        self._kv_buffer = kv_buffer.view(2, num_layers, num_pages, 1, local_kv_heads, head_dim)
        self._num_layers = num_layers
        
        # 分离 Key 和 Value 的 Buffer，方便独立访问
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        
        # 存储形状 [pages, heads, dim]，用于 store_kv 时的 view
        self._storage_shape = (num_pages, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        """获取第 index 层的所有 Key Cache"""
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        """获取第 index 层的所有 Value Cache"""
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        """
        调用 CUDA Kernel 将计算结果存储到显存池中。
        
        Args:
            k: [num_tokens, num_heads, head_dim]
            v: [num_tokens, num_heads, head_dim]
            out_loc: [num_tokens] 存储目标在显存池中的物理索引
            layer_id: 当前层号
        """
        from minisgl.kernel import store_cache

        # 这里的 view 操作将多维 tensor 展平，配合 store_cache kernel 的输入要求
        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
