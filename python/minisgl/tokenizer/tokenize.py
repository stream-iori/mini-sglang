from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
from minisgl.message import TokenizeMsg

if TYPE_CHECKING:
    from transformers import LlamaTokenizer


class TokenizeManager:
    """
    Tokenize 管理器。
    运行在 Tokenizer 进程中，负责将前端收到的文本请求转换为 Token IDs。
    """
    def __init__(self, tokenizer: LlamaTokenizer) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        """批量处理 Tokenize 请求"""
        results: List[torch.Tensor] = []
        # TODO: 使用 batch_encode_plus 进行真正的批量加速
        for msg in msgs:
            if isinstance(msg.text, list):
                # 如果是列表，认为是 Chat 格式 [{"role": "user", "content": "..."}]
                # 使用模型的 Chat Template 进行格式化
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                assert isinstance(prompt, str)
            else:
                # 否则认为是原始文本 Prompt
                prompt = msg.text
                
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="pt")
            )
            # 转为 1D int32 Tensor (MiniSGL 内部标准格式)
            results.append(input_ids.view(-1).to(torch.int32))
        return results