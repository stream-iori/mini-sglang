from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .base import BaseCacheHandle, BaseCacheManager, SizeInfo


class RadixTreeNode:
    """
    Radix Tree (基数树/前缀树) 节点。
    用于高效存储和检索 Token 序列的前缀。
    每个节点代表一段 Token 序列片段。
    """
    counter: int = 0

    def __init__(self, tic: int | None = None) -> None:
        self.children: Dict[int, RadixTreeNode] = {} # 子节点 Map (Key: Token ID)
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0  # 引用计数 (有多少个请求正在使用该节点)
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns() # LRU 时间戳

        # 这些字段稍后更新
        self._key: torch.Tensor   # 该节点存储的 Token ID 序列
        self._value: torch.Tensor # 对应的 KV Cache 物理索引
        self._length: int

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """设置节点的键值对"""
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    def set_parent(self, parent: RadixTreeNode) -> None:
        """连接父节点"""
        self._parent = parent
        parent.children[int(self._key[0].item())] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        """
        计算输入序列与当前节点 Key 的最大匹配长度。
        使用 C++ 优化的 fast_compare_key。
        """
        from minisgl.kernel import fast_compare_key
        return fast_compare_key(self._key, input_ids)

    def _split_at(self, pos: int) -> RadixTreeNode:
        """
        在指定位置分裂当前节点。
        当只有部分前缀匹配时使用。
        例如：Node(ABC) -> Split(1) -> Node(A) -> Node(BC)
        """
        assert 0 < pos < self.length
        parent = self.parent

        # 创建前半部分的新节点
        new_node = RadixTreeNode(self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        # 将当前节点更新为后半部分
        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        # 用于 heapq 比较 (按时间戳)
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    """Radix Cache 的句柄，持有对节点的引用"""
    node: RadixTreeNode


class RadixCacheManager(BaseCacheManager):
    """
    基于 Radix Tree 的 KV Cache 管理器。
    实现了 Prefix Caching (前缀缓存) 机制，允许不同请求共享相同的 Prompt 前缀缓存。
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        super().__init__()
        self.root_node = RadixTreeNode()
        self.root_node.ref_count = 1  # 根节点始终受保护
        self.evictable_size = 0 # 可驱逐的总 Token 数
        self.protected_size = 0 # 受保护（正在使用）的总 Token 数

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        锁定/解锁句柄。
        锁定会增加引用计数，防止节点被 LRU 驱逐。
        """
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[RadixCacheHandle, torch.Tensor]:
        """
        匹配最长前缀。
        Args:
            input_ids: 输入 Token 序列
        Returns:
            handle: 匹配到的最末端节点的句柄
            indices: 匹配路径上所有节点对应的 Cache 索引拼接成的 Tensor
        """
        node, prefix_len = self._walk(input_ids)
        if prefix_len == 0:
            assert node.is_root() and node is self.root_node and prefix_len == 0
            return RadixCacheHandle(prefix_len, node), self.empty_tensor
        
        # 收集路径上所有的 cache indices
        value_list: List[torch.Tensor] = []
        matched_node = node
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        return RadixCacheHandle(prefix_len, matched_node), torch.cat(value_list)

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
        """插入新的前缀到树中"""
        node, prefix_len = self._walk(input_ids)
        assert prefix_len <= len(input_ids)
        if prefix_len < len(input_ids):
            # 如果有未匹配的部分，创建新节点挂载
            new_node = RadixTreeNode()
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:])
            new_node.set_parent(node)
            self.evictable_size += new_node.length
        return prefix_len

    def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        """在树上行走，寻找最长匹配路径"""
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            this_id = int(input_ids[prefix_len].item())
            if this_id not in node.children:
                return node, prefix_len

            node = node.children[this_id]

            # 比较当前节点的 key
            match_len = node.get_match_len(input_ids[prefix_len:])
            prefix_len += match_len

            # 如果没有完全匹配当前节点，说明需要分裂
            if match_len != node.length:
                node = node._split_at(match_len)
                return node, prefix_len

            # 更新访问时间戳 (用于 LRU)
            node.timestamp = tic

        return node, prefix_len

    def evict(self, size: int) -> torch.Tensor:
        """
        驱逐最久未使用的节点以释放显存。
        Args:
            size: 需要释放的 Token 数量
        Returns:
            释放的物理 Cache 索引
        """
        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        # 收集所有引用计数为 0 的叶子节点
        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes) # 按时间戳构建小顶堆
        
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            
            # 弹出最老的节点
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            
            # 从父节点断开
            parent = node.parent
            del parent.children[int(node._key[0].item())]
            
            # 如果父节点变成了新的可驱逐叶子，加入堆中
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        """收集所有可驱逐的叶子节点"""
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        pass