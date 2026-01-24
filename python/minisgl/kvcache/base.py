from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Tuple

import torch


class BaseKVCache(ABC):
    """
    Key-Value Cache 基类。
    定义了 KV Cache 的物理存储接口。
    """

    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None: 
        """将计算出的 K/V 写入 Cache"""
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...


class KVCacheLayout(enum.Enum):
    """Cache 内存布局"""
    LayerFirst = enum.auto() # [layers, ...]
    PageFirst = enum.auto()  # [pages, ...]


@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    """Cache 句柄基类"""
    cached_len: int


class SizeInfo(NamedTuple):
    evictable_size: int  # 可被驱逐的大小 (空闲或LRU)
    protected_size: int  # 受保护的大小 (正在使用)

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size


class BaseCacheManager(ABC):
    """
    KV Cache 管理器基类。
    定义了逻辑上的 Cache 管理接口 (分配、释放、查找)。
    """
    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[BaseCacheHandle, torch.Tensor]:
        """
        匹配输入序列的前缀。
        不修改 Cache 状态。
        在使用返回的 indices 之前必须锁定 handle。

        Args:
            input_ids (torch.Tensor): 输入 Token IDs
        Returns:
            handle: Cache 句柄
            indices: 匹配到的物理索引
        """

    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        锁定或解锁句柄。
        锁定后的句柄对应的数据不能被驱逐。
        """

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
        """
        插入新前缀到 Cache 中 (修改状态)。
        
        Args:
            input_ids: Token IDs
            indices: 对应的物理索引
        Returns:
            int: 已经存在的前缀长度 (这部分不需要插入)
        """

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """
        驱逐指定大小的 Cache 以释放空间。
        
        Args:
            size: 需要释放的大小
        Returns:
            torch.Tensor: 被释放的物理索引
        """

    @abstractmethod
    def reset(self) -> None:
        """重置管理器"""

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        """获取当前容量信息"""

    @abstractmethod
    def check_integrity(self) -> None:
        """检查内部状态一致性"""