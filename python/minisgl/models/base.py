from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from minisgl.layers import BaseOP

if TYPE_CHECKING:
    import torch


class BaseLLMModel(ABC, BaseOP):
    """
    所有 LLM 模型的基类。
    继承自 BaseOP (ModuleWrapper)，支持权重的自动加载和管理。
    """
    @abstractmethod
    def forward(self) -> torch.Tensor:
        """
        前向传播函数。
        注意：在 MiniSGL 中，forward 通常不接受参数。
        输入数据（如 input_ids, positions, kv_cache）通过全局上下文 `get_global_ctx()` 获取。
        这种设计是为了简化层与层之间的参数传递。
        """
        ...