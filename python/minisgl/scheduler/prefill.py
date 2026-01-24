from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    """
    分块请求 (Chunked Request)。
    当一个 Prefill 请求太长，超过一次 Forward 允许的 token 预算时，
    会被切分成多个 ChunkedReq 分多次处理。
    """

    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should be sampled")

    def can_decode(self) -> bool:
        # 只要还在分块 Prefill 阶段，就不能进入 Decode 阶段
        return False


@dataclass
class PrefillAdder:
    """
    辅助类，用于尝试将待处理请求添加到当前 Batch 中。
    负责检查显存预算、Token 预算，并进行资源分配。
    """

    token_budget: int  # 剩余 Token 预算
    reserved_size: int  # 预留空间 (用于 Decode 阶段的 inflight tokens)
    cache_manager: CacheManager
    table_manager: TableManager

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        """尝试为一个新请求分配资源"""
        if self.table_manager.available_size == 0:
            return None

        # 尝试在 Radix Cache 中匹配前缀
        handle, match_indices = self.cache_manager.match_req(req)
        cached_len = handle.cached_len
        # TODO: 更好的显存估算策略
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        # 检查显存是否足够容纳该请求的整个生命周期 (输入 + 输出)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        self.cache_manager.lock(handle)
        # Double check after locking (可能会触发 eviction)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        # 分配 Table Index
        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # 如果有缓存命中，设置缓存部分
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(match_indices)

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        """
        实际添加请求到 Batch。
        如果请求太长，会自动创建一个 ChunkedReq。
        """
        remain_len = pending_req.input_len - cached_len
        # 计算本次能处理的长度 (受限于 budget)
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req

        # 扣除预算
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + pending_req.output_len

        # 拷贝 Input Token IDs 到 GPU Pool
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx][_slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)

        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        """尝试添加一个请求"""
        if self.token_budget <= 0:
            return None

        # 如果已经是分块请求，继续处理下一块
        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        # 如果是新请求，先尝试分配资源
        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None


@dataclass
class PrefillManager:
    """
    预填充 (Prefill) 阶段管理器。
    负责管理等待队列，并将请求调度到 Batch 中进行处理。
    """

    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    # pending_list 是“待处理任务队列”（Waiting Queue）。
    # 它的核心作用：
    # 1. 任务排队：存放所有新进来的、尚未开始 GPU 计算的用户请求。
    # 2. 分块追踪：如果一个长请求被切分为多个 Chunk (Chunked Prefill)，未完成的部分会保留在这里。
    # 3. 优先级管理：调度器总是从头部取任务。未完成的 ChunkedReq 会被放回头部，确保长请求优先被连续处理完。
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        """接收新用户请求,先存储起来"""
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        """调度下一个 Prefill Batch"""
        if len(self.pending_list) == 0:
            return None

        # 初始化 PrefillAdder (预填充添加器)。
        # 该工具类负责将等待队列中的请求“塞”进当前的计算批次(Batch)。
        adder = PrefillAdder(
            # token_budget: 本次 Forward 允许处理的最大 Token 数量 (计算预算)。
            # 用于 Chunked Prefill，防止单次 Prefill 耗时过长或占用过多显存。
            token_budget=prefill_budget,
            # reserved_size: 已经处于 Decode 阶段的请求未来可能产生的总 Token 数 (显存预留)。
            # 这是一个悲观策略：在开启新的 Prefill 任务前，必须确保所有正在运行的请求
            # 都有足够的显存长到它们的最大长度 (max_tokens)，从而避免在生成过程中因显存耗尽(OOM)而被迫挂掉。
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []

        # 遍历等待队列
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    # 如果是分块请求，记录下来，下次继续优先处理
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break  # 无法添加更多请求（显存或 budget 耗尽）

        if len(reqs) == 0:
            return None

        # 更新等待队列：分块未完成的 + 未被调度的
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
