from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as LlamaMLP
from .utils import RopeAttn as LlamaAttn

if TYPE_CHECKING:
    from .config import ModelConfig


class LlamaDecoderLayer(BaseOP):
    """
    LLaMA 模型的一个 Transformer Decoder 层。
    包含 Self-Attention 和 MLP，以及 LayerNorms。
    """
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = LlamaAttn(config, layer_id)
        self.mlp = LlamaMLP(config)
        # 使用 Fused RMS Norm 进行优化
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id

    # NVTX 注解用于 NVIDIA Nsight Systems 性能分析
    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。
        支持 Pre-Norm 架构：x = x + Layer(Norm(x))
        residual 参数用于传递残差连接，减少内存读写 (Fused Add-Norm)。
        """
        # 1. Self-Attention Block
        # Norm -> Attention -> Add Residual
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        
        # 2. MLP Block
        # Norm -> MLP -> Add Residual
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class LlamaModel(BaseOP):
    """
    LLaMA 模型主体 (不含 LM Head)。
    """
    def __init__(self, config: ModelConfig):
        # 词表并行 Embedding
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        # 堆叠 Decoder Layers
        self.layers = OPList(
            [LlamaDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 1. Embedding
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        
        # 2. Layers Loop
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
            
        # 3. Final Norm
        return self.norm.forward(x, residual)[0]


class LlamaForCausalLM(BaseLLMModel):
    """
    完整的 LLaMA 因果语言模型 (Causal LM)。
    包含 Model Body 和 LM Head。
    """
    def __init__(self, config: ModelConfig):
        self.model = LlamaModel(config)
        # 词表并行 LM Head (输出层)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        # 从全局上下文获取当前 Batch 的输入
        output = self.model.forward(get_global_ctx().batch.input_ids)
        # 计算 Logits
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["LlamaForCausalLM"]