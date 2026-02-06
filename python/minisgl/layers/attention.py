from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import divide_even

from .base import StateLessOP
from .rotary import get_rope

if TYPE_CHECKING:
    from minisgl.layers import RMSNorm
    from minisgl.models import RotaryConfig


class AttentionLayer(StateLessOP):
    """
    通用 Attention 层实现。
    
    作为一个 "无状态算子" (StateLessOP)，它本身不持有权重 (权重在 LinearQKV 和 LinearOProj 中)。
    它的主要职责是协调 Attention 计算的各个步骤：
    1. QKV 切分 (Splitting)
    2. 归一化 (Normalization, 可选)
    3. 位置编码 (Rotary Embedding)
    4. 核心 Attention 计算 (通过 FlashInfer 或其他后端)
    """
    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None, # Q-Normalization (部分模型如 Qwen 使用)
        k_norm: RMSNorm | None = None, # K-Normalization
    ):
        # 验证 Grouped Query Attention (GQA) 参数合法性
        # Q Head 数量必须是 KV Head 数量的整数倍
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        
        # 计算当前 Tensor Parallel (TP) Rank 负责的 Head 数量
        # divide_even 确保能被整除，否则抛出错误
        self.num_qo_heads = divide_even(num_qo_heads, tp_size)
        self.num_kv_heads = divide_even(num_kv_heads, tp_size)
        
        # 计算本地维度的总大小 (Num Heads * Head Dim)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        
        # 初始化 Rotary Embedding (位置编码)
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
        )
        self.q_norm = q_norm
        self.k_norm = k_norm

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            qkv: 融合的 QKV Tensor，来自 LinearQKVMerged 的输出。
                 形状为 [batch_tokens, hidden_size + 2 * kv_hidden_size] (本地切分后的大小)。
                 
        Returns:
            torch.Tensor: Attention 的输出结果，形状 [batch_tokens, hidden_size] (本地切分后)。
        """
        ctx = get_global_ctx()
        metadata = ctx.batch.attn_metadata
        
        # 1. 切分 Q, K, V
        # input qkv 是拼接在一起的，需要根据维度切开
        # split 的大小基于本地计算的维度
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        
        # 2. 可选的 QK Norm (如 Qwen/Cohere 等模型使用)
        # 在应用 RoPE 之前对 Q 和 K 进行归一化
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
            
        # 3. 应用 Rotary Embedding (RoPE)
        # 这是一个原地 (inplace) 或非原地操作，取决于具体实现
        if self.rotary:
            q, k = self.rotary.forward(metadata.positions, q, k)
            
        # 4. 调用 Attention Backend 执行计算
        # 这是核心计算步骤，通过全局上下文委托给 FlashInfer 或 Naive 实现
        # 这里传入了 q, k, v 以及 layer_id (用于索引 KV Cache) 和 batch 信息
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        
        # 5. 重塑输出
        # 将 [batch_tokens, num_heads, head_dim] 展平回 [batch_tokens, hidden_size]
        return o.view(-1, self.qo_attn_dim)