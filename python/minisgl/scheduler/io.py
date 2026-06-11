from __future__ import annotations

# 这个文件封装 Scheduler 的进程间通信。
#
# Scheduler 需要和两个方向通信：
# 1. tokenizer/detokenizer：
#    - 收 UserMsg/AbortBackendMsg；
#    - 发 DetokenizeMsg。
# 2. 多 TP rank 之间：
#    - rank 0 从 tokenizer 收请求；
#    - rank 0 把请求广播给其他 rank；
#    - 只有 rank 0 把结果发回 detokenizer。
#
# SchedulerIOMixin 被 Scheduler 继承，用来把通信逻辑和调度逻辑分开。

from typing import TYPE_CHECKING, Final, List

import torch
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg, DetokenizeMsg
from minisgl.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger

if TYPE_CHECKING:
    from .config import SchedulerConfig

logger = init_logger(__name__)


class SchedulerIOMixin:
    """
    Scheduler I/O 混入类。

    它处理 scheduler 与 tokenizer、detokenizer、其他 TP rank 的通信。

    Public Utilities:
        receive_msg: 接收 tokenizer/rank0 广播来的消息。
        send_result: 把结果发回 detokenizer。
        sync_all_ranks: CPU 侧同步所有 TP rank。
    """

    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        """根据 offline/online、单 rank/多 rank 配置通信函数。"""

        tp_info = config.tp_info
        self.tp_cpu_group: Final = tp_cpu_group
        if config.offline_mode:
            # offline 模式由 LLM 类自己实现 receive/send，不使用 ZMQ。
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return  # early exit

        if tp_info.is_primary():
            # rank 0 从 tokenizer 接收后端消息。
            self._recv_from_tokenizer: Final = ZmqPullQueue(
                config.zmq_backend_addr,
                create=True,
                decoder=BaseBackendMsg.decoder,
            )

            # rank 0 向 detokenizer 发送输出 token。
            self._send_into_tokenizer: Final = ZmqPushQueue(
                config.zmq_detokenizer_addr,
                create=config.backend_create_detokenizer_link,
                encoder=BaseTokenizerMsg.encoder,
            )

        recv = self._recv_msg_single_rank
        send = self._reply_tokenizer_rank0
        if tp_info.size > 1:
            if tp_info.is_primary():
                # 多 rank 时，rank 0 需要把收到的请求广播给其他 rank。
                recv = self._recv_msg_multi_rank0
                self._send_into_ranks: Final = ZmqPubQueue(
                    config.zmq_scheduler_broadcast_addr, create=True, encoder=BaseBackendMsg.encoder
                )
            else:
                # 非 rank0 不直接连 tokenizer，只订阅 rank0 广播。
                recv = self._recv_msg_multi_rank1
                send = self._reply_tokenizer_rank1
                self._recv_from_rank0: Final = ZmqSubQueue(
                    config.zmq_scheduler_broadcast_addr,
                    create=False,
                    decoder=BaseBackendMsg.decoder,
                )

        self.receive_msg = recv
        self.send_result = send

    def run_when_idle(self):
        """Scheduler 空闲等待消息时可执行的钩子，由 Scheduler 实现。"""

        raise NotImplementedError("should be implemented")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """offline 模式接收消息，由 LLM 子类实现。"""

        raise NotImplementedError("should be implemented")

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        """offline 模式发送结果，由 LLM 子类实现。"""

        raise NotImplementedError("should be implemented")

    def sync_all_ranks(self) -> None:
        """所有 TP rank 在 CPU 侧做 barrier 同步。"""

        self.tp_cpu_group.barrier().wait()

    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """单 rank 模式从 tokenizer 接收消息。"""

        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            # blocking=True 表示如果现在没有请求，就进入等待。
            self.run_when_idle()
            pending_msgs.append(self._recv_from_tokenizer.get())

        # 非阻塞地把当前队列里已有消息全部取出。
        while not self._recv_from_tokenizer.empty():
            pending_msgs.append(self._recv_from_tokenizer.get())
        return pending_msgs

    def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """多 rank 模式下 rank 0 接收 tokenizer 消息并广播给其他 rank。"""

        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            raw = self._recv_from_tokenizer.get_raw()

            # raw bytes 直接广播，避免先 decode 再 encode 的额外开销。
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))

        pending_raw_msgs: List[bytes] = []
        while not self._recv_from_tokenizer.empty():
            pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

        # broadcast the number of raw messages to all ranks
        # 先广播消息数量，确保所有 rank 本轮处理相同数量的请求。
        src_tensor = torch.tensor(len(pending_raw_msgs))
        self.tp_cpu_group.broadcast(src_tensor, root=0).wait()

        for raw in pending_raw_msgs:
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))
        return pending_msgs

    def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """多 rank 模式下非 rank0 从 rank0 广播接收消息。"""

        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_rank0.get())

        # ensure all ranks have the same number of raw messages
        # 接收 rank0 广播的消息数量。
        dst_tensor = torch.tensor(-1)
        self.tp_cpu_group.broadcast(dst_tensor, root=0).wait()
        dst_length = int(dst_tensor.item())

        # 按数量从 pub/sub 队列中取消息。
        for _ in range(dst_length):
            pending_msgs.append(self._recv_from_rank0.get())
        return pending_msgs

    def _reply_tokenizer_rank0(self, reply: List[DetokenizeMsg]) -> None:
        """rank 0 把输出 token 发给 detokenizer。"""

        num_reply = len(reply)
        logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")
        if num_reply == 1:
            self._send_into_tokenizer.put(reply[0])
        elif num_reply > 1:
            self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))  # type: ignore

    def _reply_tokenizer_rank1(self, reply: List[DetokenizeMsg]) -> None:
        """非 rank0 不向 detokenizer 回复，避免重复输出。"""

        _ = reply  # do nothing for non-primary ranks
