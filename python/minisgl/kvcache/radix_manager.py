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
    
    设计目的：
    高效存储和检索 Token 序列的前缀，实现 Shared Prefix Caching。
    不同于普通的 Trie 树（每条边一个字符），Radix Tree 的每个节点可以存储一段 Token 序列（key），
    从而压缩路径，减少深度。

    属性:
    - _key: 该节点存储的 Token ID 序列片段 (e.g., [TokenA, TokenB])
    - _value: 对应的 KV Cache 物理显存索引 (e.g., [Page1, Page2])
    - children: 子节点映射 {First_Token_ID -> Node}
    - ref_count: 引用计数。表示当前有多少个活跃请求正在依赖该节点的数据。
                 ref_count > 0 时，该节点是受保护的 (Protected)，不可被驱逐。
    - timestamp: 最后访问时间戳，用于 LRU (Least Recently Used) 驱逐策略。
    """
    counter: int = 0

    def __init__(self, tic: int | None = None) -> None:
        self.children: Dict[int, RadixTreeNode] = {} # 子节点 Map (Key: Token ID)
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0  
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
        使用 C++ 优化的 fast_compare_key 进行快速比对。
        """
        from minisgl.kernel import fast_compare_key
        return fast_compare_key(self._key, input_ids)

    def _split_at(self, pos: int) -> RadixTreeNode:
        """
        在指定位置分裂当前节点。
        当只有部分前缀匹配时使用。
        
        示例:
        当前节点 Key: [A, B, C]
        输入序列: [A, B, D]
        
        操作:
        1. 在位置 2 (Token C 处) 分裂。
        2. 原节点变为父节点 Key: [A, B]
        3. 新创建子节点 Key: [C], 继承原有的 children 和 value。
        4. 输入序列后续会挂在父节点 [A, B] 下面，形成新的分支 [D]。
        
        Args:
            pos: 分裂点的位置索引
        Returns:
            new_node: 分裂出来的父节点（原节点变成了子节点）-- 等等，根据代码看逻辑是：
            代码逻辑是：
            1. 创建 new_node 作为父节点 (Key: 0~pos)
            2. 将 self (当前节点) 修改为子节点 (Key: pos~end)
            3. 返回 new_node
        """
        assert 0 < pos < self.length
        parent = self.parent

        # 创建前半部分的新节点 (作为新的父节点)
        new_node = RadixTreeNode(self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count # 继承引用计数

        # 将当前节点更新为后半部分 (作为子节点)
        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        # 用于 heapq 比较 (按时间戳)，决定谁先被 LRU 驱逐
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    """Radix Cache 的句柄，持有对节点的引用"""
    node: RadixTreeNode


class RadixCacheManager(BaseCacheManager):
    """
    基于 Radix Tree 的 KV Cache 管理器。
    
    核心功能：
    1. **前缀复用**: 自动识别不同请求的公共前缀 (如 System Prompt)，复用显存。
    2. **动态管理**: 随请求生成动态插入新节点，随显存压力动态驱逐旧节点。
    3. **LRU 策略**: 维护 LRU 堆，优先回收最久未使用的缓存。
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        super().__init__()
        self.root_node = RadixTreeNode()
        self.root_node.ref_count = 1  # 根节点始终受保护，不可删除
        self.evictable_size = 0 # 可驱逐的总 Token 数 (ref_count=0 的节点总长)
        self.protected_size = 0 # 受保护的总 Token 数 (ref_count>0 的节点总长)

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        锁定/解锁句柄。
        
        逻辑:
        从目标节点回溯到根节点，沿途更新所有父节点的引用计数。
        - 只要有一个子节点被引用，其所有父节点也必须被保护 (ref_count > 0)。
        - 只有当 ref_count 降为 0 时，显存大小才从 protected 转移到 evictable。
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
        在 Radix Tree 中查找输入 Token 序列的最长匹配前缀。
        
        流程:
        1. 调用 _walk 找到最后匹配的节点和长度。
        2. 回溯路径，收集所有父节点的 value (物理索引)。
        3. 拼接所有物理索引返回。
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
        """
        将新生成的 KV Cache 索引插入树中。
        
        场景:
        当请求进行 Prefill 或 Decode 后，会产生新的 KV 数据。
        我们需要将这些新数据的物理索引记录到树中，以便未来复用。
        """
        node, prefix_len = self._walk(input_ids)
        assert prefix_len <= len(input_ids)
        
        if prefix_len < len(input_ids):
            # 如果有未匹配的部分，创建新节点挂载到最后匹配的节点下
            # 新节点存储 input_ids[prefix_len:] 和 indices[prefix_len:]
            new_node = RadixTreeNode()
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:])
            new_node.set_parent(node)
            self.evictable_size += new_node.length
        return prefix_len

    def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        """
        在树上行走，寻找最长匹配路径。
        这是 Radix Tree 的核心查找逻辑。
        
        Returns:
            node: 最后匹配到的节点 (可能是完全匹配，也可能是部分匹配需分裂)
            prefix_len: 匹配的总长度
        """
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            this_id = int(input_ids[prefix_len].item())
            if this_id not in node.children:
                return node, prefix_len

            node = node.children[this_id]

            # 比较当前节点的 key 与剩余 input_ids
            match_len = node.get_match_len(input_ids[prefix_len:])
            prefix_len += match_len

            # 如果没有完全匹配当前节点 (match_len < node.length)，说明遇到了分叉点
            # 需要对当前节点进行分裂 (Split)，以便插入新的分支
            if match_len != node.length:
                node = node._split_at(match_len)
                return node, prefix_len

            # 更新访问时间戳 (用于 LRU)
            node.timestamp = tic

        return node, prefix_len

    def evict(self, size: int) -> torch.Tensor:
        """
        驱逐最久未使用的节点以释放显存 (LRU)。
        
        流程:
        1. 收集所有 ref_count=0 的叶子节点。
        2. 构建小顶堆 (按 timestamp 排序)。
        3. 循环弹出最老的节点，回收其物理索引，直到满足释放大小。
        4. 如果删除节点导致父节点变成了新的空闲叶子，将父节点加入堆中 (级联删除)。
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