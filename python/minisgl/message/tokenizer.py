from __future__ import annotations

# 这个文件定义“发给 tokenizer worker”的消息。
#
# tokenizer worker 同时承担两个方向的转换：
# - tokenize：把用户文本转成 token id，送给 Scheduler；
# - detokenize：把 Scheduler 生成的 token id 转回文本，送给 API Server。
#
# 所以这里的消息包括：
# - TokenizeMsg：前端发来的文本请求；
# - DetokenizeMsg：后端发来的单个输出 token；
# - AbortMsg：用户取消请求；
# - BatchTokenizerMsg：多条 tokenizer 消息的批量封装。

from dataclasses import dataclass
from typing import Dict, List

from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    """tokenizer worker 消息基类。"""

    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        """把 tokenizer 消息编码成 dict，便于通过 ZMQ 发送。"""

        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        """把收到的 dict 还原成具体 tokenizer 消息对象。"""

        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    """多条 tokenizer 消息的批量封装。"""

    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    """请求 tokenizer worker 把一个输出 token 转成文本。

    字段含义：
    - uid：对应哪个用户请求；
    - next_token：模型刚刚生成的 token id；
    - finished：这个 token 之后，请求是否结束。
    """

    uid: int
    next_token: int
    finished: bool


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    """请求 tokenizer worker 把用户输入转成 token id。

    字段含义：
    - uid：API Server 分配的用户请求编号；
    - text：可以是普通 prompt 字符串，也可以是 OpenAI chat messages 格式；
    - sampling_params：用户指定的生成参数。
    """

    uid: int
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams


@dataclass
class AbortMsg(BaseTokenizerMsg):
    """通知 tokenizer worker 和后端取消某个用户请求。"""

    uid: int
