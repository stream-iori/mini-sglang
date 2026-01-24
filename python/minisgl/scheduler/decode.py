from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    """
    解码 (Decode) 阶段管理器。
    负责管理正在生成 Token 的活跃请求。
    """

    # NOTE: default_factory
    # 每次创建一个新的类实例时，请运行 set() 这个函数，把生成的新集合赋给这个变量
    #
    running_reqs: Set[Req] = field(default_factory=set)

    def add_reqs(self, reqs: Iterable[Req]) -> None:
        """添加请求到运行集合 (过滤掉不能 Decode 的，如 ChunkedReq)"""
        self.running_reqs.update(req for req in reqs if req.can_decode())

    def remove_req(self, req: Req) -> None:
        """移除已完成的请求"""
        self.running_reqs.discard(req)

    @property
    def inflight_tokens(self) -> int:
        """
        当前正在进行的解码请求总数。
        这用于估算显存占用：每个 decoding req 下一步都会产生一个新的 KV Cache block。
        """
        return sum(req.remain_len for req in self.running_reqs)

    def schedule_next_batch(self) -> Batch | None:
        """
        调度下一个 Decode Batch。
        Decode 阶段通常很简单：把所有能跑的都跑了 (Continuous Batching)。
        """
        if not self.runnable:
            return None
        return Batch(reqs=list(self.running_reqs), phase="decode")

    @property
    def runnable(self) -> bool:
        return bool(self.running_reqs)
