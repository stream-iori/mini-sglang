from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    """
    引擎配置类。
    包含了模型推理运行时的所有静态配置信息。
    """
    model_path: str                 # 模型路径（本地文件夹或 HF Repo ID）
    tp_info: DistributedInfo        # Tensor Parallel 分布式信息
    dtype: torch.dtype             # 计算数据类型 (fp16, bf16, fp32)
    max_running_req: int = 256     # 最大并发运行请求数（超过此数量的请求会排队）
    attention_backend: str = "naive" # Attention 后端实现 ("naive", "flashinfer", "triton" 等)
    cuda_graph_bs: List[int] | None = None # 启用的 CUDA Graph Batch Sizes 列表
    cuda_graph_max_bs: int | None = None   # CUDA Graph 支持的最大 Batch Size
    page_size: int = 1            # PagedAttention 的页大小 (block_size)
    memory_ratio: float = 0.9     # KV Cache 占用显存的比例
    distributed_timeout: float = 60.0 # 分布式操作超时时间
    use_dummy_weight: bool = False # 是否使用随机初始化的假权重（用于调试/无权重文件测试）
    use_pynccl: bool = False       # 是否使用 PyNCCL 进行通信
    max_seq_len_override: int | None = None # 强制覆盖模型的最大序列长度
    num_page_override: int | None = None  # 强制覆盖 KV Cache 的总页数（用于测试显存限制）

    @cached_property
    def hf_config(self):
        """加载 Hugging Face 的 config.json"""
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        """
        将 HF Config 转换为 MiniSGL 内部使用的通用 ModelConfig。
        这使得代码可以解耦具体的 HF 模型类。
        """
        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        """模型支持的最大序列长度（上下文窗口）"""
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        """一次 Forward 允许的最大长度"""
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        """分布式初始化的 TCP 地址"""
        return "tcp://127.0.0.1:23333"