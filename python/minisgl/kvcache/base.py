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
    所有的 KV Cache 实现（如 MHAKVCache）都必须继承此类。
    
    KV Cache 是 LLM 推理中用于存储 Attention 历史 Key/Value 值的核心组件，
    用于避免重复计算，实现增量解码。
    """

    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor:
        """获取指定层的 Key Cache 张量"""
        ...

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor:
        """获取指定层的 Value Cache 张量"""
        ...

    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None: 
        """
        将计算出的新 K/V 数据写入 Cache。
        
        Args:
            k: 当前步计算出的 Key 张量 [batch_size, num_heads, head_dim]
            v: 当前步计算出的 Value 张量 [batch_size, num_heads, head_dim]
            out_loc: 写入的目标位置索引 (Flatten 后的物理地址) [batch_size]
            layer_id: 当前是第几层
        """
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
    """
    Cache 内存布局枚举。
    不同的布局会影响内存访问模式和 Kernel 效率。
    """
    LayerFirst = enum.auto() # 形状: [layers, num_pages, ...]
    PageFirst = enum.auto()  # 形状: [num_pages, layers, ...]


@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    """
    Cache 句柄基类。
    用于逻辑层 (CacheManager) 追踪和管理分配给请求的缓存资源。
    具体实现可能包含树节点引用或其他元数据。
    """
    cached_len: int  # 该句柄管理的缓存长度


class SizeInfo(NamedTuple):
    evictable_size: int  # 可被驱逐的大小 (空闲或LRU，单位通常为 Token 或 Page)
    protected_size: int  # 受保护的大小 (正在被活跃请求使用，不可驱逐)

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size


class BaseCacheManager(ABC):
    """
    KV Cache 管理器基类。
    定义了逻辑上的 Cache 管理接口，负责：
    1. 维护 Token ID 到 物理存储索引 的映射。
    2. 实现缓存复用策略 (如 Radix Tree 前缀匹配)。
    3. 管理显存的分配与回收 (Eviction)。
    """
    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[BaseCacheHandle, torch.Tensor]:
        """
        匹配输入序列的前缀。
        在树中查找是否已经存在该输入序列的前缀缓存。
        注意：此操作不修改 Cache 状态。
        在使用返回的 indices 之前，调用者必须负责锁定 handle。

        Args:
            input_ids (torch.Tensor): 当前请求的完整输入 Token IDs
        Returns:
            handle: 匹配到的最长前缀对应的 Cache 句柄 (用于后续引用计数管理)
            indices: 匹配路径上所有物理索引拼接成的 Tensor (用于写入 PageTable)
        """

    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        锁定或解锁句柄。
        
        锁定 (unlock=False):
            增加引用计数。表示有请求正在使用该缓存节点，防止被 LRU 策略驱逐。
        解锁 (unlock=True):
            减少引用计数。表示请求已结束或不再依赖该节点。当计数归零时，该节点变为可驱逐状态。
            
        Args:
            handle: 要操作的句柄
            unlock: True 为解锁，False 为锁定
        """

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
        """
        插入新生成的前缀到 Cache 管理结构中 (修改状态)。
        当请求计算出新的 KV Cache 后，调用此方法将其注册到树中，以便未来复用。
        
        Args:
            input_ids: Token IDs 序列
            indices: 对应的物理存储索引
        Returns:
            int: 已经存在的前缀长度 (这部分不需要重复插入)
        """

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """
        驱逐指定大小的 Cache 以释放空间。
        通常采用 LRU (Least Recently Used) 策略。
        
        Args:
            size: 需要释放的 Token 数量 (或 Page 数，取决于具体实现)
        Returns:
            torch.Tensor: 被释放的物理索引列表 (归还给内存池)
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