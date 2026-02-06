from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from transformers import LlamaConfig


@dataclass(frozen=True)
class RotaryConfig:
    """Rotary Positional Embedding (RoPE) 配置"""

    head_dim: int  # Head 维度
    rotary_dim: int  # 实际应用旋转的维度（通常等于 head_dim）
    max_position: int  # 最大位置索引
    base: float  # RoPE base (theta)
    scaling: Dict[str, float] | None  # 缩放因子配置（用于长上下文扩展）


@dataclass(frozen=True)
class ModelConfig:
    """
    统一的模型架构配置类。
    抽象了不同 HF 模型配置（如 LlamaConfig, QwenConfig）的差异。
    """

    num_layers: int
    num_qo_heads: int  # Query Heads 数量
    num_kv_heads: int  # Key/Value Heads 数量 (GQA/MQA)
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int  # FFN 中间层大小
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str  # 激活函数类型 (silu, gelu, etc.)
    tie_word_embeddings: bool  # 是否共享输入输出 Embedding 权重

    @classmethod
    def from_hf(cls, config: LlamaConfig) -> ModelConfig:
        """从 HuggingFace Config 创建 ModelConfig"""
        # 处理 GQA (Grouped Query Attention)
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                scaling=getattr(config, "rope_scaling", None),
            ),
        )

