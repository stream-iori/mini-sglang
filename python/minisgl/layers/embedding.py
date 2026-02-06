from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import divide_up, nvtx_annotate

from .base import BaseOP


class VocabParallelEmbedding(BaseOP):
    """
    词表并行 (Vocabulary Parallelism) 的 Embedding 层。
    
    为了节省显存，Embedding 矩阵通常非常大 (vocab_size * hidden_size)，
    因此我们将它按 "词表维度" 切分到多个 GPU 上。
    每个 GPU 只存储一部分词表对应的 Embedding 向量。
    
    例如: vocab_size=32000, tp_size=2
    GPU 0 存储词 ID [0, 16000)
    GPU 1 存储词 ID [16000, 32000)
    
    在前向传播时，每个 GPU 只处理属于自己范围内的输入 Token。
    最后通过 All-Reduce (Sum) 将结果合并。
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        
        # 计算每个 GPU 分到的词表大小 (向上取整)
        self.num_embeddings_tp = divide_up(num_embeddings, self.tp_size)
        
        # 计算当前 GPU 负责的词 ID 范围
        start_idx = self.num_embeddings_tp * tp_rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx - start_idx)
        
        # 初始化本地权重
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        self._comm = DistributedCommunicator()

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from minisgl.kernel import indexing

        # 使用 CUDA Kernel 进行索引
        # 如果当前 Token ID 不在 vocab_range 内，indexing 会返回全 0
        y = indexing(
            weights=self.weight,
            indices=x,
            vocab_range=self.vocab_range if self.tp_size > 1 else None,
        )

        # 通过 All-Reduce (Sum) 聚合所有 GPU 的结果
        # 因为非负责区域返回的是 0，所以 Sum 操作能得到正确结果
        return self._comm.all_reduce(y) if self.tp_size > 1 else y


class ParallelLMHead(VocabParallelEmbedding):
    """
    并行 LM Head (Language Model Head)。
    通常作为模型的最后一层，将隐藏状态投影回词表维度，计算 Logits。
    
    它继承自 VocabParallelEmbedding，因为它的权重矩阵形状也是 (vocab_size, hidden_size)。
    
    特点:
    1. 支持 Weight Tying (与 Embedding 层共享权重)。
    2. 输出处理逻辑复杂：它需要计算所有 Token 在整个词表上的概率分布 (Logits)。
       由于采用列并行 (按 vocab 维度切分)，每个 GPU 计算得到的 Logits 只是完整 Logits 的一部分。
       通常需要 Gather 操作将完整 Logits 收集起来 (或者只收集需要的 Top-K)。
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None
        self.tied_embedding = tied_embedding
        # 确保如果开启 weight tying，必须传入对应的 embedding 层实例
        assert (tied_embedding is not None) == tie_word_embeddings

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """
        加载权重。
        如果启用了 Weight Tying，LM Head 实际上不持有权重，
        它直接复用 Embedding 层的权重，因此从 state_dict 中移除相关键以避免重复加载。
        """
        if not self.tied_embedding:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        else:
            # pop the lm_head.weights and lm_head.bias if they exist
            possible_weight = f"{prefix}.weight"
            possible_bias = f"{prefix}.bias"
            if possible_weight in state_dict:
                state_dict.pop(possible_weight)
            if possible_bias in state_dict:
                state_dict.pop(possible_bias)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.tied_embedding:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 Logits。
        
        Args:
            x: 隐藏状态 [batch_size, hidden_dim]
            
        Returns:
            Logits: [batch_size, vocab_size] (可能是分布式的)
        """
        ctx = get_global_ctx()
        batch = ctx.batch
        bs = batch.size
        
        # 优化: 仅计算最后一个 Token 的 Logits (用于生成)
        # 除非是为了 debug 或训练，否则推理时不需要中间 Token 的 Logits
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
            del indices

        module = self.tied_embedding or self
        # 线性投影: [batch, hidden] @ [hidden, vocab_tp] -> [batch, vocab_tp]
        # 结果 logits 也是按 vocab 维度切分的
        logits = F.linear(x, module.weight, self.bias)
        
        if self.tp_size == 1:
            return logits
        
        # 分布式处理: All-Gather
        # 将各个 GPU 上的 partial logits 收集起来拼成完整的 [batch, vocab]
        input_shape = logits.shape
        output_tensor = self._comm.all_gather(logits)

        if bs == 1:
            return output_tensor.view(1, -1)[:, : self.num_embeddings]

        # 复杂的 Reshape 逻辑，处理多 Batch 多卡 Gather 后的数据排列
        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.movedim(0, -1)
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        return output_tensor[:, : self.num_embeddings]
