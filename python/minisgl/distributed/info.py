from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    """
    分布式环境信息类。
    存储当前进程在分布式组中的 Rank (ID) 和 World Size (总进程数)。
    """
    rank: int
    size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        """是否为主节点 (Rank 0)"""
        return self.rank == 0


_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    """设置全局 TP 信息（仅初始化时调用一次）"""
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    """获取全局 TP 信息，如果未设置则报错"""
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    """尝试获取全局 TP 信息，如果未设置则返回 None"""
    return _TP_INFO


__all__ = ["DistributedInfo", "set_tp_info", "get_tp_info", "try_get_tp_info"]