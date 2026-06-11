from __future__ import annotations

# 这个文件定义“发给后端 Scheduler”的消息。
#
# 在 miniSGLang 中，API Server、tokenizer、scheduler、detokenizer 是不同进程。
# 不同进程之间不能直接传 Python 对象引用，所以需要把消息对象序列化后通过 ZMQ 发送。
#
# backend.py 里的消息主要由 tokenizer worker 发给 Scheduler：
# - UserMsg：一个新用户请求，已经完成 tokenize，携带 input_ids；
# - AbortBackendMsg：取消某个用户请求；
# - ExitMsg：通知后端退出；
# - BatchBackendMsg：把多条后端消息打包发送，减少通信次数。

from dataclasses import dataclass
from typing import Dict, List

import torch
from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    """后端消息基类。

    所有发给 Scheduler 的消息都继承这个类。它提供统一的编码/解码接口，
    供 ZMQ 队列在发送前序列化、接收后反序列化。
    """

    def encoder(self) -> Dict:
        """把当前消息对象编码成 dict，便于 msgpack/ZMQ 传输。"""

        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        """把收到的 dict 还原成具体的消息对象。"""

        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    """多条后端消息的批量封装。"""

    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    """通知后端退出的消息。"""

    pass


@dataclass
class UserMsg(BaseBackendMsg):
    """一个已经 tokenize 完成的新用户请求。

    字段含义：
    - uid：API Server 分配的用户请求编号，用于后续匹配回复；
    - input_ids：CPU 上的一维 int tensor，表示 prompt 对应的 token id；
    - sampling_params：这个请求的生成参数。
    """

    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    """取消某个用户请求的消息。"""

    uid: int
