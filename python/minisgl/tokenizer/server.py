from __future__ import annotations

# 这个文件实现的是 tokenizer worker 进程。
#
# 在 miniSGLang 的整体链路里，它夹在 API Server 和 Scheduler 中间：
#
#   API Server  --TokenizeMsg-->  tokenizer worker  --UserMsg-->  Scheduler
#   API Server  <--UserReply---   tokenizer worker  <--DetokenizeMsg-- Scheduler
#
# 它主要做三件事：
# 1. 收到前端传来的文本请求后，把文本转成 token id，也就是模型能理解的数字序列；
# 2. 收到后端传来的输出 token 后，把 token id 转回人能读懂的文本；
# 3. 收到用户取消请求后，把取消消息继续转发给后端 Scheduler。
#
# 注意：这里不执行模型推理，也不管理 KV cache。真正跑模型的是 Scheduler/Engine。

import multiprocessing as mp
from typing import List

import torch
from minisgl.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger, load_tokenizer


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    """把“单条消息”或“一批消息”统一展开成 Python list。

    miniSGLang 为了减少进程间通信开销，有时会把多条 tokenizer 相关消息
    打包成一个 BatchTokenizerMsg。后面的处理逻辑希望统一遍历 list，所以
    这里做一次格式统一：

    - 如果输入是 BatchTokenizerMsg，就取出里面的 msg.data；
    - 如果输入本来就是单条消息，就包装成只包含一个元素的 list。
    """

    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    return [msg]


@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    model_source: str = "huggingface",
    ack_queue: mp.Queue[str] | None = None,
) -> None:
    """tokenizer worker 进程主函数。

    这个函数会被 `server/launch.py` 作为一个单独的子进程启动。它启动后会
    一直阻塞在 ZMQ 队列上等待消息，然后根据消息类型分别处理。

    参数说明：
    - tokenizer_path：tokenizer 的路径或 HuggingFace 模型名。
    - addr：本 worker 监听的 ZMQ 地址。API Server 和 Scheduler 会把
      tokenizer/detokenizer/abort 消息发到这里。
    - create：是否由本 worker 创建 `addr` 对应的 ZMQ endpoint。
    - backend_addr：发往 Scheduler 后端的 ZMQ 地址。
    - frontend_addr：发往 API Server 前端的 ZMQ 地址。
    - local_bs：本 worker 一次最多聚合处理多少条消息。这里的 bs 是
      batch size 的缩写。
    - tokenizer_id：日志里使用的 worker 编号。
    - model_source：当前函数签名保留了这个参数，但本文件内没有直接使用。
    - ack_queue：可选的 multiprocessing Queue，用来通知父进程“我已经启动好”。

    函数不会正常返回；除非收到 KeyboardInterrupt，否则会一直运行。
    """

    # 发送到后端 Scheduler 的队列。
    # 例如文本已经变成 token id 后，会被包装成 UserMsg 发给 Scheduler。
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)

    # 发送到前端 API Server 的队列。
    # 例如输出 token 已经变回文本后，会被包装成 UserReply 发给 API Server。
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)

    # 本 worker 自己监听的输入队列。
    # 它接收三类消息：
    # - TokenizeMsg：前端发来的文本，需要转成 token id；
    # - DetokenizeMsg：后端发来的 token id，需要转回文本；
    # - AbortMsg：前端发来的取消请求，需要通知后端停止对应 uid。
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)

    assert local_bs > 0

    # 加载 HuggingFace tokenizer。tokenizer 的职责是文本 <-> token id。
    tokenizer = load_tokenizer(tokenizer_path)
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    # 这里放在函数内部 import，是为了避免进程启动/import 阶段出现循环依赖，
    # 也能让 worker 进程在需要时再加载这两个 manager。
    from .detokenize import DetokenizeManager
    from .tokenize import TokenizeManager

    # TokenizeManager：负责把 prompt/messages 转成 torch.Tensor 形式的 token id。
    tokenize_manager = TokenizeManager(tokenizer)

    # DetokenizeManager：负责把模型输出 token id 增量解码成文本。
    detokenize_manager = DetokenizeManager(tokenizer)

    # 通知父进程：tokenizer worker 已经初始化完成，可以开始接收请求。
    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            # 至少阻塞等待一条消息。
            # recv_listener.get() 会一直等到有消息到来。
            pending_msg = _unwrap_msg(recv_listener.get())

            # 如果队列里已经积压了更多消息，就继续拿出来，直到达到 local_bs。
            # 这样可以把多条请求聚合成小 batch，一次性 tokenize/detokenize，
            # 减少 Python 循环和进程通信开销。
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            # 按消息类型拆分。
            #
            # DetokenizeMsg：Scheduler/Engine 已经生成 token，需要转回文本给前端。
            # TokenizeMsg：API Server 收到用户文本，需要转成 token 给后端。
            # AbortMsg：用户取消请求，需要继续通知后端。
            detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
            abort_msg = [m for m in pending_msg if isinstance(m, AbortMsg)]

            # 确认所有消息都属于上面三类之一。
            # 如果这里 assert 失败，说明出现了 tokenizer worker 不认识的消息类型。
            assert len(detokenize_msg) + len(tokenize_msg) + len(abort_msg) == len(pending_msg)

            if len(detokenize_msg) > 0:
                # 把一批输出 token 转成文本。
                # replies 和 detokenize_msg 一一对应。
                replies = detokenize_manager.detokenize(detokenize_msg)

                # 生成发给 API Server 的回复。
                # UserReply 中：
                # - uid：标识是哪一个用户请求；
                # - incremental_output：本次新生成的文本片段；
                # - finished：这个请求是否已经结束。
                batch_output = BatchFrontendMsg(
                    data=[
                        UserReply(
                            uid=msg.uid,
                            incremental_output=reply,
                            finished=msg.finished,
                        )
                        for msg, reply in zip(detokenize_msg, replies, strict=True)
                    ]
                )

                # 如果只有一条回复，就直接发单条 UserReply；
                # 如果有多条回复，就发 BatchFrontendMsg。
                # 这是一个通信层面的优化，接收端能同时解析两种格式。
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_frontend.put(batch_output)

            if len(tokenize_msg) > 0:
                # 把用户输入文本/messages 转成 token id tensor。
                # tensors 和 tokenize_msg 一一对应。
                tensors = tokenize_manager.tokenize(tokenize_msg)

                # 生成发给 Scheduler 的 UserMsg。
                # Scheduler 后续会根据 input_ids 进行 prefill/decode 调度。
                batch_output = BatchBackendMsg(
                    data=[
                        UserMsg(
                            uid=msg.uid,
                            input_ids=t,
                            sampling_params=msg.sampling_params,
                        )
                        for msg, t in zip(tokenize_msg, tensors, strict=True)
                    ]
                )

                # 同样地，单条消息直接发 UserMsg，多条消息打包成 BatchBackendMsg。
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)

            if len(abort_msg) > 0:
                # 把前端的 AbortMsg 转成后端认识的 AbortBackendMsg。
                # 后端 Scheduler 收到后，会停止对应 uid 的请求并释放相关状态。
                batch_output = BatchBackendMsg(
                    data=[AbortBackendMsg(uid=msg.uid) for msg in abort_msg]
                )

                # 单条取消消息直接发 AbortBackendMsg，多条取消消息打包发送。
                if len(batch_output.data) == 1:
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)
    except KeyboardInterrupt:
        # 父进程或用户中断时，worker 直接退出。
        # 这里没有额外清理逻辑，因为 ZMQ 队列和子进程生命周期由上层 launch 逻辑管理。
        pass
