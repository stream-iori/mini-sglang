from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import (
    BaseBackendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    UserMsg,
)
from minisgl.scheduler import Scheduler, SchedulerConfig


# 这是一个自定义异常，用于在离线模式下通知 Scheduler 所有请求都处理完了
# 当待处理队列为空且没有正在运行的请求时抛出，用于打破 run_forever 循环
class RequestAllFinished(Exception):
    pass


# 这是一个简单的数据类，用于跟踪每个请求的状态
@dataclass
class RequestStatus:
    uid: int  # 请求的唯一 ID
    input_ids: List[int]  # 输入的 Token ID 列表
    output_ids: List[int]  # 生成的 Token ID 列表 (模型输出)


# LLM 类是 MiniSGL 的离线推理接口
# 它继承自 Scheduler，意味着它不仅是一个简单的包装器，而是直接作为一个调度器运行
# 这使得用户可以在本地 Python 脚本中直接调用 generate() 进行推理，而不需要启动服务器
class LLM(Scheduler):
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        # 初始化配置
        # offline_mode=True 是关键：它告诉 Scheduler 不要尝试连接 ZMQ 网络端口，
        # 而是调用我们重写的 offline_receive_msg 和 offline_send_result 方法
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),  # 默认单卡模式
            dtype=dtype,
            offline_mode=True,
            **kwargs,
        )
        super().__init__(config)
        # pending_requests 存储用户提交但尚未开始处理的 (Prompt, 参数) 对
        self.pending_requests: List[Tuple[List[int] | str, SamplingParams]] = []
        # status_map 用于存储正在处理或已完成的请求的结果
        # Key 是请求 UID
        self.status_map: Dict[int, RequestStatus] = {}
        self.counter = 0  # 用于生成唯一的 UID

    # 辅助函数：将输入的 Prompt (字符串或 Token ID 列表) 转换为 Tensor
    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        if isinstance(prompt, str):
            # 如果是字符串，使用加载的 tokenizer 进行编码
            # .view(-1) 确保是 1D Tensor
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        else:
            # 如果已经是 ID 列表，直接转换为 Tensor
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    # 重写 Scheduler 的接收消息方法
    # 在在线模式下，这个方法会从 ZMQ 接收网络请求
    # 在离线模式下，我们从 self.pending_requests 列表中“拉取”请求
    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        # 如果要求阻塞等待 (blocking=True) 且没有待处理请求，
        # 说明所有任务都做完了，抛出异常来结束 run_forever 循环
        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()

        results: List[BaseBackendMsg] = []
        added, sum_input_len = 0, 0

        # 遍历待处理队列，尽量多地取出请求，只要不超过 prefill_budget (预填充预算)
        # 这是一个简单的批处理策略
        for i, (tokens_or_prompt, sampling_params) in enumerate(self.pending_requests):
            if sum_input_len >= self.prefill_budget:
                break

            # 对单个 Prompt 进行 Tokenize
            input_ids = self._tokenize_one(tokens_or_prompt)
            sum_input_len += len(input_ids)

            # 生成 UID 并计数
            uid, added = self.counter + added, added + 1

            # 创建 UserMsg，这是系统内部传递请求的标准格式
            results.append(UserMsg(uid=uid, input_ids=input_ids, sampling_params=sampling_params))

            # 初始化该请求的状态记录
            self.status_map[uid] = RequestStatus(
                uid=i,  # 注意这里存的是原始索引，用于最后按顺序返回结果
                input_ids=(
                    input_ids.tolist() if isinstance(tokens_or_prompt, str) else tokens_or_prompt
                ),
                output_ids=[],
            )

        # 更新计数器和待处理队列
        self.counter += added
        self.pending_requests = self.pending_requests[added:]
        return results

    # 重写 Scheduler 的发送结果方法
    # 在在线模式下，这个方法会将结果通过 ZMQ 发送回前端
    # 在离线模式下，我们直接将生成的 Token ID 存入 self.status_map
    def offline_send_result(self, reply: BatchTokenizerMsg) -> None:
        for msg in reply.data:
            assert isinstance(msg, DetokenizeMsg)
            status = self.status_map[msg.uid]
            # 如果请求没结束，就将新生成的 Token 加入结果列表
            if not msg.finished:
                status.output_ids.append(msg.next_token)

    # 供用户调用的主入口函数
    def generate(
        self,
        prompts: List[str] | List[List[int]],  # 支持批量 Prompt
        sampling_params: List[SamplingParams] | SamplingParams,  # 采样参数
    ) -> List[str]:
        # 1. 重置状态
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0

        # 2. 统一采样参数格式 (如果是单个对象，则广播到所有 Prompt)
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)

        # 3. 将所有请求加入待处理队列
        for prompt, sp in zip(prompts, sampling_params):
            self.pending_requests.append((prompt, sp))

        # 4. 启动 Scheduler 的主循环
        # 这个循环会不断调用 offline_receive_msg 获取任务，并调用 offline_send_result 存结果
        # 直到抛出 RequestAllFinished 异常
        try:
            self.run_forever()
        except RequestAllFinished:
            pass  # 正常退出

        # 5. 整理最终结果
        results = []
        # 按原始 Prompts 的顺序返回结果
        for i in range(len(prompts)):
            status = self.status_map[i]
            # 将输出的 Token IDs 解码回文本字符串
            output_text = self.tokenizer.decode(status.output_ids)
            results.append({"text": output_text, "token_ids": status.output_ids})

        return results
