from __future__ import annotations

# 这个文件实现 miniSGLang 的前端 API Server。
#
# 它负责直接面对用户或 OpenAI client，但它不直接运行模型。它的职责是：
# 1. 接收 HTTP 请求或 shell 输入；
# 2. 给每个用户请求分配 uid；
# 3. 把文本请求包装成 TokenizeMsg，通过 ZMQ 发给 tokenizer worker；
# 4. 从 detokenizer/tokenizer worker 接收 UserReply；
# 5. 把增量文本以 SSE(streaming) 或普通 JSON 返回给用户；
# 6. 如果用户断开连接，发送 AbortMsg 取消后端请求。
#
# 简化链路：
#   用户 / OpenAI client
#       -> FastAPI route
#       -> FrontendManager
#       -> ZMQ TokenizeMsg
#       -> tokenizer / scheduler / engine / detokenizer
#       -> ZMQ UserReply
#       -> StreamingResponse / JSON response

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    TokenizeMsg,
    UserReply,
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None


def get_global_state() -> FrontendManager:
    """取得全局 FrontendManager。

    FastAPI 的 route 函数没有显式传入 FrontendManager，所以这里使用模块级
    全局变量保存当前 server 的前端状态。
    """

    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    """把前端收到的单条/批量消息统一展开成 UserReply list。"""

    if isinstance(msg, BatchFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, UserReply)
            result.append(reply)
        return result
    assert isinstance(msg, UserReply)
    return [msg]


class GenerateRequest(BaseModel):
    """`/generate` 简化接口的请求格式。"""

    prompt: str
    max_tokens: int
    ignore_eos: bool = False


class Message(BaseModel):
    """OpenAI chat messages 中的一条消息。"""

    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    # 客户端指定的模型名。本工程当前只返回/使用启动时加载的模型。
    model: str

    # prompt 用于 completion 风格；messages 用于 chat-completions 风格。
    prompt: str | None = None
    messages: List[Message] | None = None

    # 采样参数。当前只把其中一部分真正传给 SamplingParams。
    max_tokens: int = 16
    temperature: float = 1.0

    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: List[str] = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    ignore_eos: bool = False


class ModelCard(BaseModel):
    """OpenAI `/v1/models` 返回的单个模型条目。"""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-sglang"
    root: str


class ModelList(BaseModel):
    """OpenAI `/v1/models` 返回的模型列表。"""

    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


@dataclass
class FrontendManager:
    """API Server 的请求状态管理器。

    这个类把 HTTP 层和 ZMQ 后端链路连接起来：
    - HTTP route 调用 send_one() 把请求送到 tokenizer；
    - listen() 后台任务持续接收 detokenizer 返回的 UserReply；
    - wait_for_ack() 按 uid 把回复交还给对应 HTTP 请求；
    - stream_*() 把 UserReply 转成前端需要的流式字节。
    """

    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)

    def new_user(self) -> int:
        """创建一个新的用户请求 uid，并初始化等待队列。"""

        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    async def listen(self):
        """后台接收循环。

        这个协程会一直从 recv_tokenizer 读取 UserReply。收到回复后，根据 uid
        放进对应请求的 ack_map，并 set 对应 event，唤醒正在等待的 HTTP route。
        """

        while True:
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                if msg.uid not in self.ack_map:
                    # 如果 uid 已经被取消或清理，丢弃迟到的后端回复。
                    continue
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()

    def _create_listener_once(self):
        """确保 listen() 后台任务只创建一次。"""

        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        """向 tokenizer worker 发送一条消息。"""

        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        """等待指定 uid 的后端回复。

        这是一个 async generator。每收到一批后端回复，就逐条 yield 给调用者。
        当某条 ack.finished=True 时，说明请求结束，清理 uid 状态后退出。
        """

        event = self.event_map[uid]

        while True:
            # 等待 listen() 收到该 uid 的新回复后 set event。
            await event.wait()
            event.clear()

            pending = self.ack_map[uid]
            self.ack_map[uid] = []
            ack = None
            for ack in pending:
                yield ack
            if ack and ack.finished:
                break

        # 请求结束后释放前端状态。
        del self.ack_map[uid]
        del self.event_map[uid]

    async def stream_generate(self, uid: int):
        """把 `/generate` 的回复转成简单 SSE 字节流。"""

        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int):
        """把回复转成 OpenAI chat-completions streaming 格式。"""

        first_chunk = True
        async for ack in self.wait_for_ack(uid):
            delta = {}
            if first_chunk:
                # OpenAI streaming 的第一个 chunk 通常带 role=assistant。
                delta["role"] = "assistant"
                first_chunk = False
            if ack.incremental_output:
                delta["content"] = ack.incremental_output

            chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

            if ack.finished:
                break

        # send final finish_reason
        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        """包装 streaming generator，检测客户端断开连接。

        如果浏览器/curl/OpenAI client 提前断开，后端继续生成会浪费 GPU。
        因此这里检测 disconnect，并异步发送 AbortMsg。
        """

        try:
            async for chunk in generator:
                # detect if the client has disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected for user %s", uid)
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            asyncio.create_task(self.abort_user(uid))
            raise

    async def abort_user(self, uid: int):
        """取消一个用户请求并通知后端。"""

        # 稍等 0.1 秒，避免正常完成和取消清理之间的竞态。
        await asyncio.sleep(0.1)
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]
        logger.warning("Aborting request for user %s", uid)
        await self.send_one(AbortMsg(uid=uid))

    def shutdown(self):
        """关闭前端 ZMQ 队列。"""

        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """FastAPI 生命周期回调。

    yield 前是启动阶段，yield 后是关闭阶段。
    本文件只在关闭阶段释放 ZMQ 队列。
    """

    yield
    # shutdown code here
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()


app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    """简化版生成接口。

    请求格式比 OpenAI 接口简单，只需要 prompt 和 max_tokens。
    返回值始终是 text/event-stream。
    """

    logger.debug("Received generate request %s", req)
    state = get_global_state()
    uid = state.new_user()

    # 把 HTTP 请求转换成 tokenizer worker 能理解的 TokenizeMsg。
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        media_type="text/event-stream",
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    """OpenAI 兼容根路径探活接口。"""

    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    """OpenAI 兼容 chat-completions 接口。

    支持 stream=True 的流式返回，也支持 stream=False 的一次性 JSON 返回。
    """

    state = get_global_state()
    if req.messages:
        # chat-completions 标准输入：messages list。
        prompt = [msg.model_dump() for msg in req.messages]
    else:
        # 兼容 prompt 风格输入。
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt

    # TODO: support more sampling parameters
    uid = state.new_user()

    # 把 OpenAI 请求转换成内部 TokenizeMsg。
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    if req.stream:
        # 流式模式：边生成边返回 SSE chunk。
        return StreamingResponse(
            state.stream_with_cancellation(state.stream_chat_completions(uid), request, uid),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all chunks and return a single JSON response
    # 非流式模式：先收集完整文本，再一次性返回。
    full_content = ""
    async for ack in state.wait_for_ack(uid):
        full_content += ack.incremental_output
        if ack.finished:
            break

    return {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            # 当前尚未统计真实 token 数，因此这里先填 0。
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/v1/models")
async def available_models():
    """OpenAI 兼容模型列表接口。"""

    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


async def shell_completion(req: OpenAICompletionRequest):
    """shell 模式内部使用的 completion 函数。

    它复用和 HTTP chat-completions 相同的 TokenizeMsg/stream_generate 链路，
    只是调用者是本进程的 shell()，不是外部 HTTP client。
    """

    state = get_global_state()
    assert req.messages is not None, "Shell completion only supports chat-completions"
    prompt = [msg.model_dump() for msg in req.messages]

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    async def _abort():
        """shell 请求结束后的兜底取消。"""

        await state.abort_user(uid)

    return StreamingResponse(
        state.stream_generate(uid),
        media_type="text/event-stream",
        background=BackgroundTask(lambda: _abort),
    )



async def shell():
    """交互式终端聊天主循环。"""

    commands = ["/exit", "/reset"]
    completer = WordCompleter(commands)
    session = PromptSession("$ ", completer=completer)

    try:
        # history 保存多轮对话，每个元素是 (user_msg, assistant_msg)。
        history: List[Tuple[str, str]] = []
        while True:
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/reset":
                    # 清空上下文，相当于新开一个会话。
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                # 把历史对话转成 OpenAI messages 格式，让模型看到上下文。
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # send to server
            req = OpenAICompletionRequest(
                model="",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                top_k=ENV.SHELL_TOP_K.value,
                top_p=ENV.SHELL_TOP_P.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                stream=True,
            )
            cur_msg = ""
            async for chunk in (await shell_completion(req)).body_iterator:
                # shell_completion 返回 SSE 格式，这里把 data: 前缀拆掉后打印纯文本。
                msg = chunk.decode()  # type: ignore
                assert msg.startswith("data: "), msg
                msg = msg[6:]
                assert msg.endswith("\n"), msg
                msg = msg[:-1]
                if msg == "[DONE]":
                    continue
                cur_msg += msg
                print(msg, end="", flush=True)
            print("", flush=True)
            history.append((cmd, cur_msg))
    except EOFError:
        # user pressed Ctrl-D
        pass
    finally:
        # shell 退出时清理 ZMQ 队列，并杀掉本进程启动的后端子进程。
        print("Exiting shell...")
        await asyncio.sleep(0.1)
        get_global_state().shutdown()
        # then kill all the subprocesses
        import psutil

        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()


def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    """
    运行前端 API Server，并通过 ZMQ 连接 tokenizer worker。

    Args:
        config: 服务配置，包括 host/port、ZMQ IPC 地址等。
        start_backend: 启动后端 worker 进程的回调函数。
        run_shell: True 时运行终端 shell；False 时启动 uvicorn HTTP server。
    """

    global _GLOBAL_STATE

    if run_shell:
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"

    # 创建全局前端状态。这里同时创建两条 ZMQ 队列：
    # - recv_tokenizer：从 detokenizer/tokenizer worker 收 UserReply；
    # - send_tokenizer：向 tokenizer worker 发 TokenizeMsg/AbortMsg。
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # start the backend here
    # 启动 scheduler、tokenizer、detokenizer 等子进程。
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        # 普通服务模式：启动 HTTP Server。
        uvicorn.run(app, host=host, port=port)
    else:
        # shell 模式：不监听 HTTP 端口，直接进入终端交互。
        asyncio.run(shell())
