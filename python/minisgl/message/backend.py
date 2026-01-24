from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    """后端消息基类"""
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    """批量后端消息"""
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    """退出信号"""
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    """用户请求消息 (包含 Input ID, 采样参数等)"""
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams