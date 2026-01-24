from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .utils import deserialize_type, serialize_type


@dataclass
class BaseFrontendMsg:
    """前端消息基类 (返回给前端的消息)"""
    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    """批量前端消息"""
    data: List[BaseFrontendMsg]


@dataclass
class UserReply(BaseFrontendMsg):
    """用户回复消息 (包含生成的文本片段)"""
    uid: int
    incremental_output: str # 增量输出的文本
    finished: bool          # 是否生成结束