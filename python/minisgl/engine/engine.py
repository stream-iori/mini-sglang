from __future__ import annotations

from datetime import timedelta
from typing import Dict, NamedTuple, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from minisgl.kvcache import create_kvcache
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_hf_weight
from minisgl.utils import divide_even, init_logger, torch_dtype

from .config import EngineConfig
from .graph import mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    """模型 Forward 的输出结果"""

    next_tokens_gpu: torch.Tensor  # GPU 上的采样结果
    next_tokens_cpu: torch.Tensor  # 拷贝到 CPU 上的采样结果 (用于逻辑处理)
    copy_done_event: torch.cuda.Event | None  # 异步拷贝完成事件


def create_page_table(shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
    """创建页表 (Page Table)，用于 PagedAttention"""
    return torch.zeros(shape, dtype=torch.int32, device=device)


def _align_up_32(num: int) -> int:
    """将数字向上对齐到 32 的倍数 (为了内存对齐或 Kernel 性能)"""
    return (num + 31) // 32 * 32


class Engine:
    """
    推理引擎核心类。
    负责：
    1. 初始化模型、KV Cache、分布式环境。
    2. 管理显存。
    3. 执行模型 Forward 计算。
    """

    def __init__(self, config: EngineConfig):
        self.model_config = config.model_config
        # 设置全局分布式信息 (TP Rank)
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)

        # 确保 CUDA 尚未初始化 (避免 Fork 问题，但在 Spawn 模式下不是必须)
        # assert not torch.cuda.is_initialized()
        self.device = torch.device("cpu")
        # torch.cuda.set_device(self.device)
        # self.stream = torch.cuda.Stream()
        # torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype

        # 初始化分布式通信组 (Process Group)
        self.tp_cpu_group = self._init_communication(config)

        # 获取初始显存 (Mock 2GB 用于演示/非 GPU 环境)
        # init_free_memory = self._sync_get_memory()[1]
        init_free_memory = 2 * 1024 * 1024 * 1024  # Mock 2GB
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # 加载模型权重
        set_rope_device(self.device)
        # 在 CPU 上加载并转换权重类型，然后移动到 GPU
        with torch_dtype(config.dtype):
            self.model = create_model(config.model_path, config.model_config)
            self.model.load_state_dict(self._load_weight_state_dict(config))
            self.model.to(self.device)

        # 确定 KV Cache 的页数 (基于剩余显存)
        self.num_pages = self.dummy_page = self._determine_num_pages(init_free_memory, config)

        # 创建 KV Cache 物理存储
        self.kv_cache = create_kvcache(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 为 dummy page (用于 padding)
            device=self.device,
            dtype=self.dtype,
        )

        # 创建页表 (Page Table)
        # 行数 = 最大并发请求数，列数 = 最大序列长度 (按 32 对齐)
        self.max_seq_len = _align_up_32(min(config.max_seq_len, self.num_pages))
        self.page_table = create_page_table(  # + 1 for dummy request
            (config.max_running_req + 1, self.max_seq_len),
            device=self.device,
        )

        # 初始化 Attention Backend (如 FlashInfer)
        self.attn_backend = create_attention_backend(
            config.attention_backend,
            config.model_config,
            self.kv_cache,
            self.page_table,
        )

        # 设置全局上下文 (用于隐式参数传递)
        self.ctx = Context(page_size=1, attn_backend=self.attn_backend)
        set_global_ctx(self.ctx)

        # 初始化采样器
        self.sampler = Sampler(self.device, self.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # 初始化 CUDA Graph 相关的 Dummy Request
        # 作用：
        # 1. Padding (填充): 当使用 CUDA Graph 时，要求 Batch Size 固定。
        #    如果实际请求数不足（如只有 5 个，但图是为 8 个编译的），
        #    就需要用 3 个 Dummy Request 填充进去凑数。
        # 2. Safety (安全哨兵): 作为 Page Table 的最后一行，防止 Kernel 越界访问。
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,  # 占用 Page Table 的最后一行
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        # 将 Dummy Request 在页表中对应的整行都填充为 dummy_page 的物理索引。
        # 这样，无论 Kernel 访问该请求的哪个位置，都会被重定向到这个安全的 Dummy Page。
        # 确保了"假计算"不会读写到其他正常请求的显存数据。
        self.page_table[self.dummy_req.table_idx].fill_(self.dummy_page)
        self.graph_runner = None  # GraphRunner(...) # 暂时禁用 Graph Runner

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        """初始化 NCCL/Gloo 分布式通信"""
        if config.tp_info.size == 1 or config.use_pynccl:
            # 单卡或使用 PyNCCL 时，使用 Gloo 作为控制平面
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            # 否则使用标准 Torch NCCL
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        """加载权重字典"""
        if config.use_dummy_weight:
            # 使用随机权重 (测试用)
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            # 加载真实权重
            # 使用字典推导式处理所有权重：
            # 1. load_hf_weight 从磁盘加载权重到指定设备
            # 2. v.to(self.dtype) 将每个权重张量转换为目标推理精度 (如 fp16)
            return {
                k: v.to(self.dtype)
                for k, v in load_hf_weight(config.model_path, self.device).items()
            }

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        """
        计算并确定 GPU 显存中可以分配给 KV Cache 的物理页数。

        参数:
            old_free_memory: 加载模型权重之前的剩余显存 (单位: Byte)。
            config: 引擎配置对象，包含 memory_ratio 等参数。
        """
        # 1. 获取当前（加载完模型后）的剩余显存
        # _sync_get_memory 会同步所有 Rank (TP) 的状态，取最小值
        new_free_memory = self._sync_get_memory()[1]

        # 2. 计算“单个 Token 的一个物理页”在所有层中占用的显存大小
        # 注意：在 Mini-SGLang 中，page_size 通常为 1，即 1 Page = 1 Token
        cache_per_page = (
            2  # 乘 2：因为每个 Token 都要存 Key 和 Value 两个向量
            * self.model_config.head_dim  # 每个 Head 的维度 (如 128)
            # 计算当前 Rank (张量并行分块后) 负责的 KV Head 数量
            * divide_even(self.model_config.num_kv_heads, config.tp_info.size)
            * config.page_size  # 每页包含的 Token 数量 (默认为 1)
            * self.dtype.itemsize  # 每个元素的字节数 (fp16/bf16 为 2 字节)
            * self.model_config.num_layers  # 模型总层数 (每一层都有 KV Cache)
        )

        # 3. 优先级处理：如果用户在命令行手动指定了页数，则直接使用
        num_pages = config.num_page_override
        if num_pages is None:
            # 4. 自动计算逻辑：
            # 计算模型权重实际占用了多少显存
            model_memory = old_free_memory - new_free_memory
            # 计算总预算：用户允许使用的显存上限 (显存总量 * 比例)
            # 例如 80GB * 0.9 = 72GB
            total_budget = int(config.memory_ratio * old_free_memory)
            # 计算剩余可用给 KV Cache 的空间
            # 可用空间 = 总预算 - 模型权重占用的空间
            available_memory = total_budget - model_memory
            # 最终页数 = 可用空间 // 单页大小 (向下取整)
            num_pages = available_memory // cache_per_page

        # 5. 安全检查：如果页数 <= 1，说明显存已经被模型塞满了，无法进行推理
        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-tokens"
        # 6. 打印统计信息，方便用户知晓当前的显存分配状态
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_pages} pages for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """同步获取所有 Rank 中最小和最大的可用显存"""
        # torch.cuda.synchronize(self.device)
        # torch.cuda.empty_cache()
        # torch.cuda.reset_peak_memory_stats(self.device)
        # free_memory = get_free_memory(self.device)
        free_memory = 2 * 1024 * 1024 * 1024  # Mock 2GB

        min_free_memory = free_memory
        max_free_memory = free_memory

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        """执行一个 Batch 的 Forward 计算"""
        # assert torch.cuda.current_stream() == self.stream

        # 使用上下文管理器设置当前 Batch，以便 Model 内部能访问到
        with self.ctx.forward_batch(batch):
            logits = self.model.forward()

        # 更新请求状态 (完成一步 Decoding)
        for req in batch.reqs:
            req.complete_one()

        # 采样生成 Next Token
        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        next_tokens_cpu = next_tokens_gpu.to("cpu")
        # copy_done_event = torch.cuda.Event()
        # copy_done_event.record()
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, None)

    def shutdown(self) -> None:
        if self.graph_runner:
            self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()
