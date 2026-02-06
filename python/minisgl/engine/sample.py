from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    """
    批处理采样参数张量。
    将一个 Batch 中所有请求的采样配置（温度、Top-K、Top-P）打包成 GPU 张量，
    以便在自定义 CUDA Kernel 中高效处理。
    """

    temperatures: torch.Tensor | None  # [batch_size]
    top_k: torch.Tensor | None = None  # [batch_size]
    top_p: torch.Tensor | None = None  # [batch_size]


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    高效创建设备张量的辅助函数。
    使用 pin_memory=True 可以加速从主机 (CPU) 到设备 (GPU) 的数据拷贝。
    """
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    """
    采样逻辑的具体实现。
    根据参数组合调用 FlashInfer 提供的各种高性能采样 Kernel。
    """
    import flashinfer.sampling as sampling

    # 1. 计算 Softmax 概率分布
    # PDL (Pre-fill Decode Layout) 优化仅在 SM90 (H100+) 上支持
    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())

    # 2. 根据不同的参数组合选择 Kernel
    # 场景 A: 仅温度采样 (Multinomial)
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    # 场景 B: Top-K 采样
    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    # 场景 C: Top-P (Nucleus) 采样
    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    # 场景 D: 混合采样 (Top-K + Top-P)
    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    """
    采样器类。
    负责将抽象的采样参数转换为 GPU 计算所需的张量，并执行最终的采样操作。
    """

    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        """
        准备阶段：将 Batch 中每个 Req 的 SamplingParams 提取并打包。
        """
        params = [r.sampling_params for r in batch.reqs]

        # 如果 Batch 内所有请求都是贪婪解码，则不需要复杂的采样计算
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        # 最小值保护，防止除以 0 或计算异常
        MIN_P = MIN_T = 1e-6

        # 提取参数列表
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        # 如果 top_k 无效，则设为整个词表大小
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]

        # 转换为 GPU 张量
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None

        # 仅当有请求使用了有效的 top_k/top_p 时才创建张量，节省开销
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)

        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        """
        执行采样：在指定的 logits 上根据参数选出 Next Token IDs。
        """
        with torch.cuda.nvtx.range("Sampler"):
            # 贪婪采样逻辑：直接取最大 Logits 的索引
            if args.temperatures is None:
                return torch.argmax(logits, dim=-1)

            # 复杂采样逻辑：调用 FlashInfer 内核
            # 注意：采样前通常需要将 logits 转换为 float32 以保证数值稳定性
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
