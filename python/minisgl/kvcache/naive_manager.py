from __future__ import annotations

from typing import Tuple

import torch

from .base import BaseCacheHandle, BaseCacheManager, SizeInfo


class NaiveCacheHandle(BaseCacheHandle):
    pass


class NaiveCacheManager(BaseCacheManager):
    """
    朴素的 KV Cache 管理器实现。
    
    特点：
    1. **无复用**: 不支持 Prefix Caching，每次请求都视为全新的。
    2. **无驱逐**: 不支持 LRU Eviction，只能依赖上层调用者显式释放。
    3. **简单**: 仅用于测试、基准对比或不需要缓存复用的场景。
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        super().__init__()

    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[NaiveCacheHandle, torch.Tensor]:
        """始终返回空匹配"""
        _ = input_ids  # unused
        return NaiveCacheHandle(0), self.empty_tensor

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """无操作 (因为没有缓存树需要维护)"""
        _ = handle, unlock  # unused

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
        """假装插入了，实际上不存储任何结构"""
        assert len(indices) == len(input_ids)
        return len(indices)

    def evict(self, size: int) -> torch.Tensor:
        """不支持驱逐，只能清空"""
        if size == 0:
            return self.empty_tensor
        raise NotImplementedError("NaiveCacheManager does not support eviction.")

    def reset(self) -> None:
        pass

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=0, protected_size=0)

    def check_integrity(self) -> None:
        pass
