from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import divide_even

from .base import BaseOP


class _LinearTPImpl(BaseOP):
    """
    张量并行 (Tensor Parallelism, TP) 线性层的基类实现。
    
    它负责管理分布式环境下的权重切分和基础的前向计算。
    在 TP 模式下，一个巨大的线性层被切分到多个 GPU 上，每个 GPU 只持有权重的一部分。
    
    Args:
        full_isize (int): 原始的全局输入维度。
        full_osize (int): 原始的全局输出维度。
        local_isize (int): 当前 GPU 负责的本地输入维度 (切分后)。
        local_osize (int): 当前 GPU 负责的本地输出维度 (切分后)。
        has_bias (bool): 是否包含偏置项。
    """

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        # 初始化本地权重，形状取决于切分方式
        self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        基础的前向传播。
        注意：这只是局部计算，是否需要 All-Reduce 取决于具体的并行策略 (Row vs Col)。
        """
        return F.linear(x, self.weight, self.bias)


class LinearColParallelMerged(_LinearTPImpl):
    """
    列并行 (Column Parallel) 线性层 —— 针对多个输出合并的情况。
    
    在列并行中，权重矩阵按列切分。
    输入 x 会被广播到所有 GPU，每个 GPU 计算输出的一部分，
    最后的结果是所有 GPU 输出的拼接 (All-Gather，如果需要的话)。
    但在 Transformer 的 MLP 层中，通常列并行的输出直接作为下一个行并行层的输入，
    因此这里不需要立即通信。
    
    典型用途: MLP 的 Gate + Up projection (SwiGLU)。
    """
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int], # 例如 [hidden_dim, hidden_dim] 用于 Gate+Up
        has_bias: bool,
    ):
        # 检查所有输出尺寸是否能被 TP 大小整除
        tp_info = get_tp_info()
        tp_output_sizes = [divide_even(size, tp_info.size) for size in output_sizes]
        
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        
        # Col Parallel: 输入不切分，输出按 TP 切分
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)


class LinearQKVMerged(_LinearTPImpl):
    """
    专门针对 Attention QKV 投影的列并行层。
    
    处理了 GQA (Grouped Query Attention) 和 MQA (Multi-Query Attention) 的复杂性。
    在 GQA 中，KV Head 的数量少于 Q Head，这使得简单的切分变得棘手。
    
    这个类确保每个 GPU 分配到正确比例的 Q, K, V Head，保持它们在物理内存上的连续性，
    以便后续的 Attention Kernel 能高效读取。
    """
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()

        # 计算 GQA 比率 (每个 KV Head 对应多少个 Q Head)
        GQA_ratio = divide_even(num_qo_heads, num_kv_heads)
        # 计算当前 GPU 分到的 KV Head 数量
        local_num_kv = divide_even(num_kv_heads, tp_info.size)
        
        full_isize = hidden_size
        # 全局输出大小 = (Q + K + V) * head_dim
        # Q 的数量是 K 的 GQA_ratio 倍
        full_osize = (GQA_ratio + 2) * num_kv_heads * head_dim
        
        local_isize = hidden_size
        # 本地输出大小 = 本地负责的 KV 组数 * (Q+K+V per group) * head_dim
        local_osize = (GQA_ratio + 2) * local_num_kv * head_dim
        
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)


class LinearOProj(_LinearTPImpl):
    """
    Attention 的输出投影层 (Output Projection) —— 行并行 (Row Parallel)。
    
    这通常是 Attention 模块的最后一层。
    在行并行中，权重矩阵按行切分。
    输入 x 是切分过的 (来自上面的 Col Parallel QKV 和 Attention 计算)，
    每个 GPU 计算输出的一部分，然后通过 All-Reduce 累加得到最终的完整输出。
    """
    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        # Row Parallel: 输入按 TP 切分，输出是完整的
        local_isize = divide_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        # Row Parallel 必须进行 All-Reduce (Sum) 来合并结果
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    """
    通用的行并行 (Row Parallel) 线性层。
    
    用途: MLP 的 Down Projection。
    原理同 LinearOProj：接收切分的输入，输出完整的张量 (需要 All-Reduce)。
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()
        # Row Parallel: 输入按 TP 切分
        local_input_size = divide_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        # All-Reduce 聚合结果
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
