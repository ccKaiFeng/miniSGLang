from __future__ import annotations

# 这个文件封装基于 ZMQ + msgpack 的进程间队列。
#
# miniSGLang 的 API Server、tokenizer、scheduler、detokenizer 是不同进程。
# 它们通过这里的 Push/Pull/Pub/Sub 队列传递序列化后的消息对象。

from typing import Callable, Dict, Generic, TypeVar

import msgpack
import zmq
import zmq.asyncio

T = TypeVar("T")


class ZmqPushQueue(Generic[T]):
    """同步 PUSH 队列，负责发送消息。"""

    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        """创建 PUSH socket。

        addr 是 ZMQ 地址；create=True 时 bind，False 时 connect；encoder 把消息对象转成 dict。
        """

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put(self, obj: T):
        """编码并发送一个对象。"""

        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        """关闭 socket 并释放 ZMQ context。"""

        self.socket.close()
        self.context.term()


class ZmqAsyncPushQueue(Generic[T]):
    """asyncio 版本的 PUSH 队列，用于 FastAPI 前端。"""

    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        """创建 asyncio PUSH socket，参数含义同 ZmqPushQueue。"""

        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    async def put(self, obj: T):
        """异步编码并发送一个对象。"""

        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        await self.socket.send(event, copy=False)

    def stop(self):
        """关闭 socket 并释放 asyncio ZMQ context。"""

        self.socket.close()
        self.context.term()


class ZmqPullQueue(Generic[T]):
    """同步 PULL 队列，负责接收消息。"""

    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        """创建 PULL socket。

        decoder 把 msgpack 解出的 dict 转回具体消息对象。
        """

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    def get(self) -> T:
        """阻塞接收一条消息并解码。"""

        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def get_raw(self) -> bytes:
        """阻塞接收原始 msgpack bytes，不做解码。"""

        return self.socket.recv()

    def decode(self, raw: bytes) -> T:
        """把 get_raw() 得到的 bytes 解码成消息对象。"""

        return self.decoder(msgpack.unpackb(raw, raw=False))

    def empty(self) -> bool:
        """非阻塞检查当前 socket 是否没有可读消息。"""

        return self.socket.poll(timeout=0) == 0

    def stop(self):
        """关闭 socket 并释放 ZMQ context。"""

        self.socket.close()
        self.context.term()


class ZmqAsyncPullQueue(Generic[T]):
    """asyncio 版本的 PULL 队列。"""

    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        """创建 asyncio PULL socket，参数含义同 ZmqPullQueue。"""

        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    async def get(self) -> T:
        """异步接收一条消息并解码。"""

        event = await self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def stop(self):
        """关闭 socket 并释放 asyncio ZMQ context。"""

        self.socket.close()
        self.context.term()


class ZmqPubQueue(Generic[T]):
    """同步 PUB 队列，一条消息可广播给多个 SUB。"""

    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        """创建 PUB socket，addr/create/encoder 含义同 PUSH 队列。"""

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put_raw(self, raw: bytes):
        """直接发送已经序列化好的 bytes。"""

        self.socket.send(raw, copy=False)

    def put(self, obj: T):
        """编码并广播一个对象。"""

        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        """关闭 socket 并释放 ZMQ context。"""

        self.socket.close()
        self.context.term()


class ZmqSubQueue(Generic[T]):
    """同步 SUB 队列，订阅 PUB 队列的所有消息。"""

    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        """创建 SUB socket，并订阅空 topic 表示接收全部消息。"""

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.decoder = decoder

    def get(self) -> T:
        """阻塞接收一条广播消息并解码。"""

        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def empty(self) -> bool:
        """非阻塞检查当前是否没有可读广播消息。"""

        return self.socket.poll(timeout=0) == 0

    def stop(self):
        """关闭 socket 并释放 ZMQ context。"""

        self.socket.close()
        self.context.term()
