from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Generic, List, TypeAlias, TypeVar

import torch

# 定义状态字典的类型别名，方便后续使用
_STATE_DICT: TypeAlias = Dict[str, torch.Tensor]


def _concat_prefix(prefix: str, name: str) -> str:
    """
    辅助函数：用于拼接参数名称的前缀。
    例如：prefix="layer1", name="weight" -> "layer1.weight"
    """
    return f"{prefix}.{name}" if prefix else name


class BaseOP:
    """
    基础算子类 (Base Operator)。
    这是 Mini-SGLang 中所有模型层 (Layer) 的基类。

    设计目的:
    虽然 PyTorch 提供了强大的 `nn.Module`，但在高性能推理引擎中，我们需要对参数加载、
    设备移动 (CPU -> GPU)、以及分布式状态管理有更精细的控制。
    因此，我们实现了一个轻量级的、类似于 `nn.Module` 的基类，去除了自动求导等推理不需要的开销。

    主要功能:
    1. `to(device)`: 递归地将层内的参数移动到指定设备。
    2. `state_dict()`: 递归地收集所有参数，支持自定义前缀。
    3. `load_state_dict()`: 递归地加载参数，支持严格的形状和类型检查。
    """

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        前向传播方法，所有子类必须实现此方法以定义具体的计算逻辑。
        """
        ...

    def to(self, device: torch.device):
        """
        将该算子内的所有 Tensor 和子算子移动到指定设备 (如 CPU 或 GPU)。

        Args:
            device: 目标设备。

        Returns:
            self: 返回自身以便链式调用。
        """
        # 遍历实例的所有属性
        for key, value in self.__dict__.items():
            if isinstance(value, BaseOP):
                # 递归调用子算子的 to 方法
                value.to(device)
            elif isinstance(value, torch.Tensor):
                # 如果是 Tensor，移动到指定设备并更新属性
                self.__dict__[key] = value.to(device)
            elif isinstance(value, torch.nn.Parameter):
                # 如果是 nn.Parameter，同样移动到指定设备
                self.__dict__[key] = value.to(device)
        return self

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        """
        获取该算子的状态字典 (包含所有权重参数)。
        用于模型保存或权重加载。

        逻辑:
        递归遍历所有属性，如果属性是 Tensor，则记录；如果是子 BaseOP，则递归调用。

        Args:
            prefix: 参数名称的前缀 (用于递归调用时构建层级名称)。
            result: 用于收集结果的字典 (可选，用于递归传递)。

        Returns:
            包含所有参数名称和对应 Tensor 的字典。
        """
        result = result if result is not None else {}

        # 遍历所有属性
        for name, param in self.__dict__.items():
            # 跳过私有属性 (以 _ 开头)
            if name.startswith("_"):
                continue

            if isinstance(param, torch.Tensor):
                # 如果是 Tensor，加入结果字典，key 为带前缀的全名
                result[_concat_prefix(prefix, name)] = param
            elif isinstance(param, BaseOP):
                # 如果是子算子，递归调用其 state_dict 方法
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)

        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """
        从状态字典中加载权重参数。

        特点:
        实现了“弹出式”加载 —— 加载完一个参数就从字典中移除。
        这样可以在加载结束后检查字典是否为空，以发现多余的键。

        Args:
            state_dict: 包含权重数据的字典。注意：加载过的键会从字典中 pop 移除。
            prefix: 当前算子的参数前缀。
            _internal: 内部标志，用于指示是否为递归调用的内部过程。

        Raises:
            RuntimeError: 如果参数形状/类型不匹配，或 state_dict 中存在多余的键。
        """
        for name, param in self.__dict__.items():
            # 跳过私有属性
            if name.startswith("_"):
                continue

            if isinstance(param, torch.Tensor):
                # 构造完整的键名
                key = _concat_prefix(prefix, name)
                # 从 state_dict 中弹出对应的权重数据
                if key not in state_dict:
                    # 如果 key 不存在，可能是因为 tied embedding 或者参数名不匹配
                    # 这里可以添加更详细的日志
                    continue
                item = state_dict.pop(key)

                assert isinstance(item, torch.Tensor)
                # 严格检查形状和数据类型是否一致，确保模型定义的参数与加载的权重匹配
                assert param.shape == item.shape and param.dtype == item.dtype
                # 直接使用 setattr 更新属性，替换为新加载的 Tensor
                setattr(self, name, item)
            elif isinstance(param, BaseOP):
                # 如果是子算子，递归加载
                param.load_state_dict(
                    state_dict, prefix=_concat_prefix(prefix, name), _internal=True
                )

        # 如果不是内部递归调用 (即最外层调用)，且 state_dict 还不为空，
        # 说明提供的权重字典里有一些键在模型中没找到对应位置，抛出错误。
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


class StateLessOP(BaseOP):
    """
    无状态算子类 (Stateless Operator)。
    继承自 BaseOP，但明确表示该算子不包含任何需要保存或加载的参数。

    典型用途:
    - 激活函数 (SiLU, GELU)
    - 仅包含逻辑的组合层 (AttentionLayer 本身不含权重，权重在 LinearQKV 中)
    - Rotary Embedding (只有预计算的 buffer，没有可学习参数)
    """

    def __init__(self):
        super().__init__()

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """
        无状态算子的加载方法。
        由于没有参数，它不从 state_dict 中取任何东西。
        仅在最外层调用时检查是否有意外的残留键。
        """
        if not _internal and state_dict:
            _ = prefix
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        """
        无状态算子的保存方法。
        直接返回空字典 (或传入的 result)。
        """
        _ = prefix
        return result if result is not None else {}


# 定义泛型变量 T，必须是 BaseOP 的子类
T = TypeVar("T", bound=BaseOP)


class OPList(BaseOP, Generic[T]):
    """
    算子列表类 (Operator List)。
    类似于 PyTorch 的 `nn.ModuleList`，用于按顺序管理一组相同类型的算子。

    典型用途:
    - Transformer 的多层 DecoderLayer (layers.0, layers.1, ...)
    """

    def __init__(self, ops: List[T]):
        super().__init__()
        self.op_list = ops  # 存储算子列表

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        """
        获取列表中所有算子的状态字典。
        参数键名会加上数字索引作为前缀 (例如 "0.weight", "1.weight")。
        """
        result = result if result is not None else {}
        for i, op in enumerate(self.op_list):
            # 递归调用每个子算子的 state_dict，前缀为数字索引
            op.state_dict(prefix=_concat_prefix(prefix, str(i)), result=result)
        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """
        加载列表中所有算子的权重。
        """
        for i, op in enumerate(self.op_list):
            # 递归调用每个子算子的 load_state_dict，前缀为数字索引
            op.load_state_dict(state_dict, prefix=_concat_prefix(prefix, str(i)), _internal=True)

        # 检查是否有多余的键
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
