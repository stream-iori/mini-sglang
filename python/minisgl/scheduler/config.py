from __future__ import annotations

from dataclasses import dataclass, field

from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    """
    调度器配置类。
    继承自 EngineConfig，增加了调度和网络相关的配置。
    """
    max_extend_tokens: int = 8192  # 预填充阶段最大处理的 token 数 (Chunk Prefill)
    cache_type: str = "radix"      # KV Cache 管理策略 ("radix" 或 "naive")
    offline_mode: bool = False     # 是否离线模式（不启动 ZMQ 网络服务，用于本地库调用）

    # 网络配置
    # 唯一后缀，用于区分同一台机器上运行的多个实例的 IPC socket
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        """后端（Scheduler）接收 Tokenizer 消息的 IPC 地址"""
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        """Detokenizer 接收 Token ID 消息的 IPC 地址"""
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        """Scheduler 向其他组件广播消息的 IPC 地址"""
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        """
        一次 Forward 最大处理长度。
        这里覆盖了 EngineConfig 的定义，限制为 Chunk Prefill 的大小，
        防止 OOM 或长时间阻塞。
        """
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        """后端是否负责创建到 Detokenizer 的连接"""
        return True