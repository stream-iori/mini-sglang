from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


# _run_scheduler 是后端推理核心进程的入口函数
# 它在一个独立的进程中运行，负责加载模型、管理显存和执行推理计算
def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    # 在函数内部导入 torch 和 Scheduler，而不是在文件开头导入
    # 这是为了避免在主进程中初始化 CUDA 上下文，防止多进程启动时出现 CUDA 错误
    import torch
    from minisgl.scheduler import Scheduler

    # 开启推理模式：禁用梯度计算，减少显存占用并提高计算速度
    with torch.inference_mode():
        # 初始化调度器（Scheduler），这是 MiniSGL 的核心引擎
        scheduler = Scheduler(args)

        # 多卡分布式运行时，同步所有进程，确保所有 Rank 都已就绪
        scheduler.sync_all_ranks()

        # 只有主节点（Rank 0）向主进程发送“准备就绪”的信号
        # 避免多个进程同时发消息导致混乱，且通常只要 Rank 0 成功，集群状态即正常
        if args.tp_info.is_primary():
            ack_queue.put("Scheduler is ready")

        # 如果设置了静默输出，则关闭 INFO 级别的日志，保持控制台清爽
        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            # 启动调度器的主循环，开始不断地处理推理请求
            # 这个函数通常是一个死循环，直到接收到关闭信号
            scheduler.run_forever()
        except KeyboardInterrupt:
            # 捕获 Ctrl+C 中断信号，优雅退出
            logger = init_logger(__name__)
            if scheduler.tp_info.is_primary():
                print()  # 打印换行，美观
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


# 启动服务器的主函数
# run_shell: 是否以交互式 Shell 模式启动，而不是启动 HTTP API 服务器
def launch_server(run_shell: bool = False) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    # 1. 解析命令行参数
    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")

    # 定义启动子进程的内部函数
    # 这个函数会被传递给 api_server，在 API 服务器（前端）初始化好 ZMQ 后调用
    def start_subprocess() -> None:
        import multiprocessing as mp

        from minisgl.tokenizer import tokenize_worker

        # 设置多进程启动方式为 'spawn'
        # 在 PyTorch/CUDA 环境下必须使用 'spawn'，因为默认的 'fork' 会复制父进程的内存空间（包括 CUDA 上下文），
        # 这会导致 CUDA 初始化错误或死锁。'spawn' 会启动一个全新的干净解释器进程。
        mp.set_start_method("spawn", force=True)

        world_size = server_args.tp_info.size
        # 创建一个多进程队列，用于接收子进程的“启动完成”确认 (ACK)
        # 这样主进程可以阻塞等待，直到所有子进程都准备好服务
        ack_queue: mp.Queue[str] = mp.Queue()

        # 2. 启动 Scheduler 进程（推理引擎）
        # 根据并行度 (Tensor Parallel Size) 启动对应数量的进程
        for i in range(world_size):
            # 为每个进程创建一个新的配置副本，设置对应的 Rank ID
            new_args = replace(
                server_args,
                tp_info=DistributedInfo(i, world_size),
            )
            mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                daemon=False,  # 设置为非守护进程，确保主进程退出时它们能受控地清理
                name=f"minisgl-TP{i}-scheduler",
            ).start()

        # 3. 启动 DeTokenizer 进程（解码器）
        # 负责将模型生成的 Token ID 转换回人类可读的文本
        # 这是一个独立的进程，避免繁重的字符串处理阻塞推理或 API 循环
        num_tokenizers = server_args.num_tokenizer
        # DeTokenizer 只需要 1 个
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,  # 接收 Token ID 的地址
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,  # 发送解码文本回前端的地址
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,  # 是否负责创建 ZMQ socket
                "tokenizer_id": num_tokenizers,  # ID 排在 tokenizer 之后
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="minisgl-detokenizer-0",
        ).start()

        # 4. 启动 Tokenizer 进程（编码器）
        # 负责将用户输入的文本转换为 Token ID
        # 如果请求量大，可以启动多个 tokenizer 进程并行处理
        for i in range(num_tokenizers):
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,  # 接收用户文本的地址
                    "backend_addr": server_args.zmq_backend_addr,  # 发送 Token ID 给 Scheduler 的地址
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"minisgl-tokenizer-{i}",
            ).start()

        # 5. 等待所有子进程启动完毕
        # 我们需要等待:
        # - 1 个 Scheduler ACK (只有 Rank 0 发送)
        # - num_tokenizers 个 Tokenizer ACK
        # - 1 个 DeTokenizer ACK
        # 总共 = num_tokenizers + 2
        # 这是一个简单的“屏障”，确保服务完全可用前不接受请求
        for _ in range(num_tokenizers + 2):
            logger.info(ack_queue.get())

    # 6. 运行前端 API 服务器 (FastAPI) 或 Shell
    # 并将 start_subprocess 作为回调传入，在 API 层初始化完毕后触发后端启动
    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
