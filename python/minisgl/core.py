from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle


@dataclass
class SamplingParams:
    """
    采样参数配置类。
    用于控制模型生成的随机性和多样性。
    """

    temperature: float = 0.0  # 温度系数，越高越随机，<=0 表示贪婪解码
    top_k: int = -1  # Top-K 采样，-1 表示禁用
    top_p: float = 1.0  # Top-P (Nucleus) 采样
    ignore_eos: bool = False  # 是否忽略 EOS token（强制生成到最大长度）
    max_tokens: int = 1024  # 最大生成 token 数

    @property
    def is_greedy(self) -> bool:
        """判断是否为贪婪解码模式"""
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    """
    请求对象 (Request)。
    表示一个正在处理中的推理请求。
    """

    input_ids: torch.Tensor  # CPU 上的输入 Token ID Tensor
    table_idx: int  # 在 PageTable 中的索引（用于 KV Cache 管理）
    cached_len: int  # 已经进入 KV Cache 的长度（之前的轮次处理过的）
    output_len: int  # 期望生成的总长度（包括输入）？不，通常是 max_new_tokens，需确认上下文
    uid: int  # 请求唯一 ID
    sampling_params: SamplingParams  # 该请求的采样参数
    cache_handle: BaseCacheHandle  # KV Cache 的句柄

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)  # 当前在设备上需要处理的长度
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        """还可以生成的 Token 数量"""
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        """
        本次需要进行 Forward 计算的新增 Token 数量。
        Extend Len = 当前总长度 - 已缓存长度
        对于 Prefill 阶段，这是输入 Prompt 的长度。
        对于 Decode 阶段，通常为 1。
        """
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        """完成一步解码，更新状态"""
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        """将生成的新 Token 追加到 Host 端记录中"""
        self.input_ids = torch.cat([self.input_ids, next_token])

    def can_decode(self) -> bool:
        """是否还可以继续解码"""
        return self.remain_len > 0

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    """
    批处理对象 (Batch)。
    包含一组同时进行 Forward 计算的请求。
    """

    reqs: List[Req]  # 请求列表
    phase: Literal["prefill", "decode"]  # 当前阶段：预填充 (prefill) 或 解码 (decode)

    # 以下字段由 Scheduler 在 prepare 阶段设置,属于计算层
    input_ids: torch.Tensor = field(init=False)  # 拼接后的输入 Token IDs (GPU)
    out_loc: torch.Tensor = field(init=False)  # 输出位置索引 (GPU)
    padded_reqs: List[Req] = field(init=False)  # 填充后的请求列表 (可能包含 padding 的 dummy reqs)
    # 以下字段由 Attention Backend 设置
    attn_metadata: BaseAttnMetadata = field(
        init=False
    )  # 注意力机制需要的元数据（如 flash_attn 的参数）

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        """实际请求数量"""
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        """填充后的总数量（用于 CUDA Graph 固定 Batch Size）"""
        return len(self.padded_reqs)


@dataclass
class Context:
    """
    全局上下文对象
    用于在模型 Forward 过程中隐式传递当前的 Batch 信息
    类似于 Flask 的 request context
    """

    page_size: int
    attn_backend: BaseAttnBackend
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        """上下文管理器，用于在 Forward 期间设置当前 Batch"""
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
