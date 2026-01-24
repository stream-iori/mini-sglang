from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
import torch.nn.functional as F
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    BaseBackendMsg,
    BatchBackendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger
from transformers import AutoTokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)


def _make_2d_indices(table_2d: torch.Tensor, ranges: List[Tuple[int, int, int]]) -> torch.Tensor:
    """
    辅助函数：将给定的 2D 范围列表转换为 1D 索引张量。
    用于从 2D Token Pool 中批量读取或写入数据。

    示例：如果 2D 表 (3, 4) 的底层索引是：
        [[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]]
    范围 [(0, 1, 3), (2, 0, 2)] 意味着：
    - 第 0 行，列 [1, 3) -> 索引 [1, 2]
    - 第 2 行，列 [0, 2) -> 索引 [8, 9]
    返回结果 [1, 2, 8, 9]。

    Args:
        table_2d (torch.Tensor): 2D 数据表张量 (用于获取 stride).
        ranges (List[Tuple[int, int, int]]): 范围列表 (行号, 起始列, 结束列).
    Returns:
        torch.Tensor: 1D 索引张量.
    """
    assert table_2d.dim() == 2 and table_2d.is_contiguous()
    STRIDE = table_2d.stride(0)
    needed_size = sum(end - begin for _, begin, end in ranges)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for entry, begin, end in ranges:
        length = end - begin
        offset += length
        torch.arange(
            begin + entry * STRIDE,
            end + entry * STRIDE,
            dtype=torch.int32,
            out=indices_host[offset - length : offset],
        )
    return indices_host.to(table_2d.device, non_blocking=True)


# 为了实现 Overlap Scheduling (重叠调度)，我们需要缓存一些输入数据
# 以便在 GPU 计算的同时，CPU 处理下一批数据或上一批结果
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    load_indices: torch.Tensor  # 用于加载 input_ids 的索引
    write_indices: torch.Tensor  # 用于写入 output_ids 的索引


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    """
    核心调度器类。
    负责协调内存管理、批处理调度、模型执行和结果处理。
    """

    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)
        # 初始化 IO Mixin，处理 ZMQ 通信
        super().__init__(config, self.engine.tp_cpu_group)

        # 使用独立的 CUDA Stream 来实现元数据处理与计算的重叠
        self.device = self.engine.device
        self.stream = None  # torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.no_grad()  # torch.cuda.stream(self.engine.stream)
        # torch.cuda.set_stream(self.stream)

        # 初始化各个管理器
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(self.device, self.engine.num_pages, config.cache_type)
        self.decode_manager = DecodeManager()
        # Prefill Manager 需要与其他管理器协作
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        self.tp_info = config.tp_info
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.page_table = self.engine.page_table
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens

    def _process_last_data(
        self, last_data: ForwardData | None, ongoing_data: ForwardData | None
    ) -> None:
        """
        处理上一批次 (Last Batch) 的计算结果。
        包括：将生成的 Token 复制回 CPU，判断是否结束，发送结果，释放资源。
        """
        if last_data is None:
            return
        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        # 等待数据从 GPU 拷贝到 CPU 完成
        if copy_done:
            copy_done.synchronize()
        reply = BatchTokenizerMsg(data=[])

        max_seq_len = self.engine.max_seq_len
        for i, req in enumerate(batch.reqs):
            # 忽略已完成的或仍在分块处理中的请求
            if req in self.finished_reqs or isinstance(req, ChunkedReq):
                continue

            # 获取生成的 Token ID
            next_token_id = next_tokens_cpu[i]
            # 更新 Host 端的请求状态
            req.append_host(next_token_id.unsqueeze(0))
            next_token = int(next_token_id.item())

            # 判断是否结束 (达到最大长度或生成 EOS)
            finished = req.remain_len <= 0
            if not req.sampling_params.ignore_eos:
                finished |= next_token == self.eos_token_id

            # 安全检查：防止超过模型最大上下文长度
            if req.device_len >= max_seq_len - 1:
                finished = True
                logger.warning_rank0(f"Request {req.uid} reached {max_seq_len = }, dropped.")

            # 添加到回复消息中
            reply.data.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

            # 如果请求完成，标记并移除
            if finished:
                self.finished_reqs.add(req)
                self.decode_manager.remove_req(req)
                logger.debug_rank0("Request %s is finished", req)

        # 释放已完成请求的资源
        # 注意：需要排除掉 ongoing_data (当前正在 GPU 上跑的) 中的请求
        ongoing_reqs = ongoing_data[0].batch.reqs if ongoing_data else []
        for req in self.finished_reqs.difference(ongoing_reqs):
            self.table_manager.free(req.table_idx)
            # 释放 KV Cache 并尝试将其加入 Radix Cache 以供未来复用
            self.cache_manager.free_and_cache_finished_req(
                req.cache_handle,
                req.input_ids[: req.cached_len],
                self.page_table[req.table_idx, : req.cached_len],
            )

        # 清理 finished_reqs 集合，只保留那些还在 ongoing 中的（ rare case?）
        self.finished_reqs.intersection_update(ongoing_reqs)
        # 发送结果给 Detokenizer
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        """处理一条来自前端的消息"""
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            if input_len >= max_seq_len:
                return logger.warning_rank0(
                    f"Input sequence len {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            max_output_len = max_seq_len - input_len
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            # 将新请求加入 Prefill 队列
            self.prefill_manager.add_one_req(msg)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        """
        为即将进行的 Forward 准备数据。
        包括分配 KV Cache 页面、计算索引、准备 Attention 元数据等。
        """
        # 计算所需的总显存空间 (Token 数)
        needed_size = sum(r.extend_len for r in batch.reqs)
        # 分配 KV Cache 页面
        batch.out_loc = self.cache_manager.allocate(needed_size)

        # 处理 CUDA Graph 的 Padding
        padding_size = 0
        if self.engine.graph_runner:
            # 确认padding_size
            padding_size = self.engine.graph_runner.pad_batch(batch)

        if padding_size:
            # 从batch,out_loc开始补, 左0(head)，右padding_size(tail), 补dummy_page这种假值
            # F.pad 的操作会产生新的值,需要返回给原值
            batch.out_loc = F.pad(batch.out_loc, (0, padding_size), value=self.engine.dummy_page)

        # 准备用于加载和写入 Token ID 的索引
        load_indices = _make_2d_indices(
            self.token_pool, [(r.table_idx, r.cached_len, r.device_len) for r in batch.padded_reqs]
        )
        write_indices = _make_2d_indices(
            self.token_pool, [(r.table_idx, r.device_len, r.device_len + 1) for r in batch.reqs]
        )

        # 将分配好的物理页地址写入 Page Table
        self.page_table.view(-1)[load_indices] = batch.out_loc
        # 准备 Attention Backend (如 FlashInfer) 需要的元数据
        self.engine.attn_backend.prepare_metadata(batch)

        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            load_indices=load_indices,
            write_indices=write_indices,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        """
        调度下一批次。
        策略：优先调度 Prefill 请求，如果没有则调度 Decode 请求。
        """
        # TODO: 支持其他调度策略 (如优先 Decode 以降低延迟)
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _load_token_ids(self, input: ForwardInput) -> None:
        """将 Input Token IDs 从 Pool 加载到 Batch 对象中 (用于 Model Forward)"""
        input.batch.input_ids = self.token_pool.view(-1)[input.load_indices]

    def _write_token_ids(self, input: ForwardInput, output: ForwardOutput) -> None:
        """将 Model 生成的 Next Token IDs 写回 Pool"""
        self.token_pool.view(-1)[input.write_indices] = output.next_tokens_gpu

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        """执行模型前向传播"""
        self._load_token_ids(forward_input)
        batch, sample_args = forward_input.batch, forward_input.sample_args
        # 调用 Engine 执行计算
        forward_output = self.engine.forward_batch(batch, sample_args)
        # 写回结果
        self._write_token_ids(forward_input, forward_output)
        # 将刚才 forward 的请求加入 Decode 队列 (如果是 prefill 完成，则转入 decode)
        self.decode_manager.add_reqs(forward_input.batch.reqs)
        return forward_output

    def run_when_idle(self) -> None:
        """空闲时执行的后台任务 (如 KV Cache 整理)"""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        重叠调度循环 (Overlap Loop)。
        它尝试重叠：
        1. 当前 Batch 的 GPU 计算
        2. 上一个 Batch 的结果处理 (CPU)
        3. 下一个 Batch 的调度准备 (CPU)
        这能有效隐藏 CPU 开销，提高 GPU 利用率。
        """
        # 决定是否阻塞等待新消息
        blocking = not (
            last_data  # 如果有上一批数据要处理，不阻塞
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        # 接收并处理所有待处理消息
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # 调度下一批次
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # 在 Engine 的 Stream 中执行 Forward
                # self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        # 处理上一批次的结果 (此时 GPU 可能正在跑 ongoing_data)
        self._process_last_data(last_data, ongoing_data)
        return ongoing_data

    def normal_loop(self) -> None:
        """非重叠循环 (串行执行)"""
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data, None)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        """主循环入口"""
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                # self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            # assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        # torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

