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
    负责 QKV 切分、Rotary Embedding 应用，并调用后端 (FlashInfer/Naive) 执行 Attention 计算。
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
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        
        # 计算当前 TP Rank 负责的 Head 数量
        self.num_qo_heads = divide_even(num_qo_heads, tp_size)
        self.num_kv_heads = divide_even(num_kv_heads, tp_size)
        
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        
        # 初始化 Rotary Embedding
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
        Args:
            qkv: 融合的 QKV Tensor，形状为 [batch_tokens, hidden_size + 2 * kv_hidden_size]
        """
        ctx = get_global_ctx()
        metadata = ctx.batch.attn_metadata
        
        # 1. 切分 Q, K, V
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        
        # 2. 可选的 QK Norm (如 Qwen)
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
            
        # 3. 应用 Rotary Embedding (RoPE)
        if self.rotary:
            q, k = self.rotary.forward(metadata.positions, q, k)
            
        # 4. 调用 Attention Backend 执行计算
        # 这是核心计算步骤，通过全局上下文委托给 FlashInfer 或 Naive 实现
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        
        # 5. 重塑输出
        return o.view(-1, self.qo_attn_dim)