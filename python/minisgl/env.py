from __future__ import annotations

import os
from functools import partial
from typing import Callable, Generic, TypeVar


class BaseEnv:
    """所有环境变量类的基类"""
    def _init(self, name: str) -> None:
        raise NotImplementedError


T = TypeVar("T")


class EnvVar(BaseEnv, Generic[T]):
    """
    泛型环境变量封装类。
    
    Args:
        default_value: 默认值。
        fn: 用于将字符串环境变量转换为目标类型的转换函数。
    """
    def __init__(self, default_value: T, fn: Callable[[str], T]):
        self.value = default_value
        self.fn = fn
        super().__init__()

    def _init(self, name: str) -> None:
        """从系统环境变量中读取值并初始化"""
        env_value = os.getenv(name)
        if env_value is not None:
            try:
                self.value = self.fn(env_value)
            except Exception:
                pass

    def __bool__(self):
        return bool(self.value)

    def __str__(self):
        return str(self.value)


# 辅助函数：将字符串转换为布尔值
_TO_BOOL = lambda x: x.lower() in ("1", "true", "yes")


def _PARSE_MEM_BYTES(mem: str) -> int:
    """解析内存大小字符串，如 '1G', '512M' 为字节数"""
    mem = mem.strip().upper()
    if not mem[-1].isalpha():
        return int(mem)
    if mem.endswith("B"):
        mem = mem[:-1]
    UNIT_MAP = {"K": 1024, "M": 1024**2, "G": 1024**3}
    return int(float(mem[:-1]) * UNIT_MAP[mem[-1]])


MINISGL_ENV_PREFIX = "MINISGL_"
# 预定义的类型化环境变量构造器
EnvInt = partial(EnvVar[int], fn=int)
EnvFloat = partial(EnvVar[float], fn=float)
EnvBool = partial(EnvVar[bool], fn=_TO_BOOL)
EnvOption = partial(EnvVar[bool | None], fn=_TO_BOOL, default_value=None)
EnvMem = partial(EnvVar[int], fn=_PARSE_MEM_BYTES)


class EnvClassSingleton:
    """
    全局环境配置单例类。
    所有配置项都可以通过 MINISGL_{ATTR_NAME} 环境变量覆盖。
    例如 SHELL_MAX_TOKENS 可以通过 export MINISGL_SHELL_MAX_TOKENS=4096 修改。
    """
    _instance: EnvClassSingleton | None = None

    # shell 相关配置
    SHELL_MAX_TOKENS = EnvInt(2048)
    SHELL_TOP_K = EnvInt(-1)
    SHELL_TOP_P = EnvFloat(1.0)
    SHELL_TEMPERATURE = EnvFloat(0.6)

    # 后端运行时配置
    # 是否在 FlashInfer 中使用 Tensor Cores
    FLASHINFER_USE_TENSOR_CORES = EnvOption()
    # 是否禁用通信与计算的重叠调度（Overlap Scheduling）
    DISABLE_OVERLAP_SCHEDULING = EnvBool(False)
    # PyNCCL 通信缓冲区最大大小
    PYNCCL_MAX_BUFFER_SIZE = EnvMem(1024**3)

    def __new__(cls):
        # 实现单例模式
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # 自动初始化所有 EnvVar 类型的属性
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            assert isinstance(attr_value, BaseEnv)
            # 加上前缀 MINISGL_ 进行初始化
            attr_value._init(f"{MINISGL_ENV_PREFIX}{attr_name}")


# 全局唯一的环境变量配置实例
ENV = EnvClassSingleton()