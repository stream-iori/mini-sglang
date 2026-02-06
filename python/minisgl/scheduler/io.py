from __future__ import annotations

# 引入类型检查相关的工具
from typing import TYPE_CHECKING, Final, List

# 引入 PyTorch，用于分布式通信（Tensor 广播等）
import torch
# 引入消息类，定义了后端和 Tokenizer 之间通信的数据结构
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg
# 引入 ZMQ 队列包装器（Pub/Sub, Pull/Push）和日志初始化工具
from minisgl.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger

# 仅在类型检查阶段导入 SchedulerConfig，避免运行时循环导入
if TYPE_CHECKING:
    from .config import SchedulerConfig

# 初始化当前模块的 logger
logger = init_logger(__name__)


class SchedulerIOMixin:
    """
    Scheduler I/O 操作的 Mixin 类。

    该类主要负责 Scheduler（调度器）与 Tokenizer（分词器/前端）之间的通信。
    它处理消息的接收（从 Tokenizer）和结果的发送（回传给 Tokenizer）。
    同时也处理多卡（Tensor Parallelism）环境下，Rank 之间的消息同步。

    公开工具方法:
        receive_msg: 用于从 tokenizer 接收消息的函数（根据配置动态绑定具体的实现）。
        send_result: 用于将结果发送回 tokenizer 的函数（根据配置动态绑定具体的实现）。
        sync_all_ranks: 用于在 CPU 侧同步所有 Rank 的函数。
    """

    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        """
        初始化 SchedulerIOMixin。

        Args:
            config: 调度器配置对象，包含通信地址、TP 信息等。
            tp_cpu_group: PyTorch 的进程组，用于 CPU 侧的通信（如 barrier, broadcast）。
        """
        tp_info = config.tp_info
        # 将 CPU 进程组标记为 Final，表示初始化后不应修改
        self.tp_cpu_group: Final = tp_cpu_group
        
        # 如果是离线模式（Offline Mode），使用离线特定的收发函数
        # 离线模式通常用于 benchmark 或测试，不涉及真实的网络通信
        if config.offline_mode:
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return  # 早期退出，不再初始化网络队列

        # 如果当前进程是主 Rank (Rank 0)
        # 主 Rank 负责直接与外部 Tokenizer 进行 ZMQ 通信
        if tp_info.is_primary():
            # 初始化接收队列 (Pull)，用于从 Tokenizer 接收请求
            # create=True 表示由后端创建该队列
            self._recv_from_tokenizer: Final = ZmqPullQueue(
                config.zmq_backend_addr,
                create=True,
                decoder=BaseBackendMsg.decoder,
            )
            # 初始化发送队列 (Push)，用于向 Tokenizer 发送响应（Detokenizer）
            # create=config.backend_create_detokenizer_link 决定是否由后端创建连接
            self._send_into_tokenizer: Final = ZmqPushQueue(
                config.zmq_detokenizer_addr,
                create=config.backend_create_detokenizer_link,
                encoder=BaseTokenizerMsg.encoder,
            )

        # 默认使用单 Rank 的收发逻辑
        recv = self._recv_msg_single_rank
        send = self._reply_tokenizer_rank0

        # 如果存在多个 Rank (TP size > 1)，需要处理 Rank 间的通信
        if tp_info.size > 1:
            if tp_info.is_primary():
                # 主 Rank (Rank 0)
                # 使用多 Rank 场景下的 Rank 0 接收逻辑：接收外部请求并广播给其他 Rank
                recv = self._recv_msg_multi_rank0
                # 初始化广播队列 (Pub)，用于将收到的请求广播给其他 Rank
                self._send_into_ranks: Final = ZmqPubQueue(
                    config.zmq_scheduler_broadcast_addr, create=True, encoder=BaseBackendMsg.encoder
                )
            else:
                # 非主 Rank (Rank > 0)
                # 使用多 Rank 场景下的非 Rank 0 接收逻辑：接收 Rank 0 的广播
                recv = self._recv_msg_multi_rank1
                # 非主 Rank 不直接回复 Tokenizer，使用空操作
                send = self._reply_tokenizer_rank1
                # 初始化订阅队列 (Sub)，用于订阅 Rank 0 的广播消息
                self._recv_from_rank0: Final = ZmqSubQueue(
                    config.zmq_scheduler_broadcast_addr,
                    create=False,
                    decoder=BaseBackendMsg.decoder,
                )

        # 绑定最终确定的接收和发送函数
        self.receive_msg = recv
        self.send_result = send

    def run_when_idle(self):
        """
        当调度器空闲时运行的钩子函数。
        需要在子类或混入该 Mixin 的主类中实现。
        """
        raise NotImplementedError("should be implemented")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """离线模式下的消息接收接口（需实现）。"""
        raise NotImplementedError("should be implemented")

    def offline_send_result(self, reply: BatchTokenizerMsg) -> None:
        """离线模式下的结果发送接口（需实现）。"""
        raise NotImplementedError("should be implemented")

    def sync_all_ranks(self) -> None:
        """
        同步所有 Rank。
        使用 PyTorch 的 barrier 确保所有进程运行到此处时会等待，直到所有进程都到达。
        """
        self.tp_cpu_group.barrier().wait()

    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """
        单 Rank 模式下的消息接收逻辑。
        
        Args:
            blocking: 是否阻塞等待。如果为 True，且队列为空，会先调用 run_when_idle 直到有消息。
        
        Returns:
            接收到的消息列表。
        """
        pending_msgs: List[BaseBackendMsg] = []
        # 如果是阻塞模式
        if blocking:
            # 在等待消息时执行空闲任务（如垃圾回收、状态检查等）
            self.run_when_idle()
            # 阻塞获取一条消息
            pending_msgs.append(self._recv_from_tokenizer.get())
        
        # 继续非阻塞地获取队列中剩余的所有消息
        while not self._recv_from_tokenizer.empty():
            pending_msgs.append(self._recv_from_tokenizer.get())
        return pending_msgs

    def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """
        多 Rank 模式下，主 Rank (Rank 0) 的消息接收逻辑。
        它负责从 Tokenizer 接收消息，并将原始字节广播给其他 Rank。
        """
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            # 阻塞获取原始字节数据
            raw = self._recv_from_tokenizer.get_raw()
            # 将原始数据放入内部广播队列，发给其他 Rank
            self._send_into_ranks.put_raw(raw)
            # 解码并添加到本地待处理消息列表
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))

        pending_raw_msgs: List[bytes] = []
        # 获取所有剩余的原始消息
        while not self._recv_from_tokenizer.empty():
            pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

        # 1. 广播消息数量给所有 Rank，确保大家知道接下来要接收多少条消息
        # src_tensor 包含消息数量
        src_tensor = torch.tensor(len(pending_raw_msgs))
        # 使用 PyTorch 分布式广播
        self.tp_cpu_group.broadcast(src_tensor, root=0).wait()

        # 2. 遍历并分发每一条消息
        for raw in pending_raw_msgs:
            # 通过 ZMQ 广播原始数据
            self._send_into_ranks.put_raw(raw)
            # 本地解码
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))
        return pending_msgs

    def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """
        多 Rank 模式下，非主 Rank (Rank > 0) 的消息接收逻辑。
        它从 Rank 0 接收广播过来的消息。
        """
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            # 阻塞等待从 Rank 0 接收一条消息
            pending_msgs.append(self._recv_from_rank0.get())

        # 1. 接收 Rank 0 广播的消息数量
        # 初始化一个 tensor 用于接收
        dst_tensor = torch.tensor(-1)
        self.tp_cpu_group.broadcast(dst_tensor, root=0).wait()
        # 获取需要接收的消息总数
        dst_length = int(dst_tensor.item())

        # 2. 根据数量循环接收剩余消息
        for _ in range(dst_length):
            pending_msgs.append(self._recv_from_rank0.get())
        return pending_msgs

    def _reply_tokenizer_rank0(self, reply: BatchTokenizerMsg) -> None:
        """
        主 Rank (Rank 0) 回复 Tokenizer 的逻辑。
        只有 Rank 0 有权限向 Tokenizer 发送数据。
        """
        num_reply = len(reply.data)
        logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")
        # 如果只有一条回复，直接发送对象
        if num_reply == 1:
            self._send_into_tokenizer.put(reply.data[0])
        # 如果有多条回复，发送整个 Batch 对象
        elif num_reply > 1:
            self._send_into_tokenizer.put(reply)

    def _reply_tokenizer_rank1(self, reply: BatchTokenizerMsg) -> None:
        """
        非主 Rank 的回复逻辑。
        什么都不做，因为非主 Rank 不直接与 Tokenizer 通信。
        """
        _ = reply  # 占位，避免未使用变量警告