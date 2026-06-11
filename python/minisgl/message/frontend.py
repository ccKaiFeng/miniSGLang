from __future__ import annotations

# 这个文件定义“发给前端 API Server”的消息。
#
# Scheduler/Engine 生成 token 后，会先交给 tokenizer worker 做 detokenize。
# tokenizer worker 把 token id 转成文本后，通过这里定义的 UserReply 发回 API Server。
# API Server 再把文本片段通过 HTTP 流式响应返回给用户。

from dataclasses import dataclass
from typing import Dict, List

from .utils import deserialize_type, serialize_type


@dataclass
class BaseFrontendMsg:
    """前端消息基类。

    所有发给 API Server 的消息都继承它，并复用同一套序列化/反序列化逻辑。
    """

    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        """把前端消息编码成 dict，便于通过 ZMQ 发送。"""

        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        """把 ZMQ 收到的 dict 还原成前端消息对象。"""

        return deserialize_type(globals(), json)


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    """多条前端回复的批量封装。"""

    data: List[BaseFrontendMsg]


@dataclass
class UserReply(BaseFrontendMsg):
    """返回给 API Server 的用户请求回复。

    字段含义：
    - uid：对应哪个用户请求；
    - incremental_output：本次新生成的一小段文本；
    - finished：这个请求是否已经生成结束。

    因为大模型通常是边生成边返回，所以同一个 uid 会收到多条 UserReply。
    """

    uid: int
    incremental_output: str
    finished: bool
