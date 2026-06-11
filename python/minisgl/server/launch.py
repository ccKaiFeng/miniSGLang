from __future__ import annotations

# 这个文件负责启动整个 miniSGLang 服务。
#
# 它不是 HTTP API 的实现，而是“进程编排器”：
# 1. 解析命令行参数；
# 2. 启动 TP scheduler 进程，每个 TP rank 一个；
# 3. 启动 detokenizer worker；
# 4. 可选启动多个 tokenizer worker；
# 5. 等所有后端子进程 ready 后，再启动 API Server 或 shell。

import logging
import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    """单个 Scheduler 子进程的入口函数。

    多 GPU Tensor Parallel 时，每个 TP rank 都会启动一个 scheduler 进程。
    这个函数运行在子进程里，不运行在主进程里。
    """

    import torch
    from minisgl.scheduler import Scheduler

    # inference_mode 会关闭 autograd，减少推理时的额外开销。
    with torch.inference_mode():
        # 创建 scheduler。Scheduler 内部会创建 Engine、模型、KV cache 等对象。
        scheduler = Scheduler(args)

        # 多 TP rank 时，等待所有 rank 初始化到同一阶段。
        scheduler.sync_all_ranks()

        if args.tp_info.is_primary():
            # 只让 rank 0 通知主进程“Scheduler ready”，避免 world_size 个重复日志。
            ack_queue.put("Scheduler is ready")

        if args.silent_output:
            # shell 模式下减少后端日志，避免干扰终端聊天显示。
            logging.disable(logging.INFO)

        try:
            # 进入主循环，持续从 tokenizer 接收请求并驱动 Engine 推理。
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            # 释放通信队列、distributed/NCCL 等资源。
            scheduler.shutdown()


def launch_server(run_shell: bool = False) -> None:
    """miniSGLang 服务启动入口。

    `python -m minisgl` 和 `python -m minisgl.shell` 最终都会调用这里。

    参数：
    - run_shell=False：启动 HTTP API Server；
    - run_shell=True：启动终端交互式 shell。
    """

    from .api_server import run_api_server
    from .args import parse_args

    # 把命令行参数转成 ServerArgs。
    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    logger = init_logger(__name__, "initializer")

    def start_subprocess() -> None:
        """启动所有后端 worker 子进程。

        这个函数会被 api_server.run_api_server() 调用。这样可以先建立前端 ZMQ
        队列，再启动后端进程，避免消息链路创建顺序混乱。
        """

        import multiprocessing as mp

        from minisgl.tokenizer import tokenize_worker

        # spawn 表示每个子进程重新 import Python 模块启动。
        # 这比 fork 更适合 CUDA/PyTorch 多进程场景。
        mp.set_start_method("spawn", force=True)

        world_size = server_args.tp_info.size
        # a multiprocessing queue to receive ack from subprocesses
        # so that we can guarantee all subprocesses are ready
        # ack_queue 是主进程和子进程之间的同步信号队列。
        # 子进程初始化完成后 put 一条字符串，主进程等到足够数量的 ack 才继续。
        ack_queue: mp.Queue[str] = mp.Queue()

        for i in range(world_size):
            # 为每个 TP rank 复制一份配置，只替换 rank 编号。
            new_args = replace(
                server_args,
                tp_info=DistributedInfo(i, world_size),
            )
            # 启动一个 scheduler 进程。多卡时会有多个 scheduler。
            mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                daemon=False,
                name=f"minisgl-TP{i}-scheduler",
            ).start()

        num_tokenizers = server_args.num_tokenizer
        # DeTokenizer, only 1
        # detokenizer worker 负责把后端生成的 token id 转回文本。
        # 即使 num_tokenizer > 0，也始终只启动一个 detokenizer。
        mp.Process(
            target=tokenize_worker,
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="minisgl-detokenizer-0",
        ).start()

        # 可选启动额外 tokenizer worker。
        # 它们只处理前端文本 -> token id 的请求，用来提升 tokenize 吞吐。
        for i in range(num_tokenizers):
            mp.Process(
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"minisgl-tokenizer-{i}",
            ).start()

        # Wait for acknowledgments from all worker processes:
        # - world_size schedulers (but only primary rank sends ack)
        # - num_tokenizers tokenizers
        # - 1 detokenizer
        # Total acks expected: 1 + num_tokenizers + 1 = num_tokenizers + 2
        # 注意 scheduler 虽然可能有 world_size 个，但只有 primary rank 发送 ack。
        for _ in range(num_tokenizers + 2):
            logger.info(ack_queue.get())

    # 运行 API Server。它内部会调用 start_subprocess() 拉起后端。
    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    # 允许直接执行 python python/minisgl/server/launch.py。
    launch_server()
