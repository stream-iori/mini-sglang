from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from .utils import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@lru_cache(maxsize=None)
def _load_radix_module() -> Module:
    """
    加载 Radix Tree 的 C++ 扩展模块。
    使用 AOT (Ahead-of-Time) 编译或加载。
    """
    return load_aot("radix", cpp_files=["radix.cpp"])


def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    """
    快速比较两个 1D Int CPU Tensor 的前缀匹配长度。
    
    Args:
        x: Tensor A
        y: Tensor B
        
    Returns:
        匹配的长度 (common prefix length)
    """
    return _load_radix_module().fast_compare_key(x, y)