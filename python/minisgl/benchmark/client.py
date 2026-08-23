from __future__ import annotations

# 这个文件是 online benchmark 的公共客户端工具。
#
# 它使用 OpenAI 兼容 API 向 miniSGLang server 发送请求，记录每个 token 到达的
# 时间戳，然后统计吞吐、延迟、TTFT(Time To First Token) 等指标。

import asyncio
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, overload

from minisgl.utils import UNSET, Unset, init_logger
from openai import AsyncOpenAI as OpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm

logger = init_logger(__name__)


@dataclass(frozen=True)
class BenchmarkTrace:
    """一条 benchmark 请求轨迹。"""

    timestamp: float  # unit (second)
    message: str
    output_length: int  # output length in tokens
    input_length: int | None = None  # input length in tokens, optional


@dataclass(frozen=True)
class BenchOneResult:
    """单个请求的测量结果。"""

    tics: List[float]
    input_len: int
    output_len: int

    def as_json(self) -> List[float]:
        """转成便于保存的 list 格式。"""

        return [self.input_len, self.output_len] + self.tics

    @staticmethod
    def from_json(raw: List[float]) -> BenchOneResult:
        """从保存的 list 恢复 BenchOneResult。"""

        # check raw[0] and raw[1] are integers
        assert raw[0].is_integer() and raw[1].is_integer()
        return BenchOneResult(tics=raw[2:], input_len=int(raw[0]), output_len=int(raw[1]))


@dataclass(frozen=True)
class RawResult:
    """原始请求结果，后续会被汇总成 BenchmarkResult。"""

    input_len: int | None
    output_len: int
    message: str
    tics: List[float]


@dataclass
class Counter:
    """简单计数器，同时记录历史最大值。"""

    current: int = 0
    history_max: int = 0

    def inc(self, n=1):
        """计数增加 n。"""

        self.current += n
        self.history_max = max(self.history_max, self.current)

    def dec(self, n=1):
        """计数减少 n，并检查不会变成负数。"""

        self.current -= n
        assert self.current >= 0


@dataclass
class Console:
    """管理 benchmark 运行时的多个 tqdm 进度条。"""

    input_pbar: tqdm
    output_pbar: tqdm
    prefill_pbar: tqdm
    decode_pbar: tqdm
    disabled: bool
    inflight_counter: Counter = field(default_factory=Counter)
    queue_counter: Counter = field(default_factory=Counter)

    def update_input(self, n=1):
        """记录新请求进入系统。"""

        self.input_pbar.update(n)
        self.input_pbar.refresh()
        self.inflight_counter.inc(n)
        self.queue_counter.inc(n)

    def update_output(self, n=1):
        """记录一个请求完成输出。"""

        self.output_pbar.update(n)
        self.output_pbar.refresh()
        self.inflight_counter.dec(n)

    def update_prefill(self, n=1):
        """记录一个请求收到首 token，相当于完成一次 prefill/TTFT。"""

        self.prefill_pbar.update(n)
        self.prefill_pbar.refresh()
        self.queue_counter.dec(n)

    def update_decode(self, n=1):
        """记录 decode 阶段新增 n 个输出 token。"""

        self.decode_pbar.update(n)

    @contextmanager
    def inflight(self, n=1):
        """请求生命周期上下文：进入时计入 inflight，退出时计为完成。"""

        self.update_input(n)
        yield
        self.update_output(n)

    @contextmanager
    def log_stats(self):
        """benchmark 结束时关闭进度条并打印最大并发/排队数。"""

        yield
        self.input_pbar.close()
        self.output_pbar.close()
        self.prefill_pbar.close()
        self.decode_pbar.close()
        if not self.disabled:
            max_inflight = self.inflight_counter.history_max
            max_queue = self.queue_counter.history_max
            logger.info(f"Max inflight requests: {max_inflight}, Max queued requests: {max_queue}")


@dataclass(frozen=True)
class BenchmarkResult:
    """一组 benchmark 请求的可序列化结果。"""

    raw_data: List[BenchOneResult]

    def as_json(self) -> List[List[float]]:
        """把所有请求结果转成 JSON 友好的二维 list。"""

        return [r.as_json() for r in self.raw_data]

    @staticmethod
    def from_json(raw: List[List[float]]) -> BenchmarkResult:
        """从二维 list 恢复 BenchmarkResult。"""

        return BenchmarkResult(raw_data=[BenchOneResult.from_json(r) for r in raw])


def make_console(num_requests: int, sum_output_length: int, use_pbar: bool = True) -> Console:
    """创建 benchmark 进度条集合。

    num_requests 是请求数；sum_output_length 是所有请求 max_tokens 之和。
    prefill token 数按每请求 1 个首 token 估算，剩余计入 decode token。
    """

    BAR_FORMAT_0 = (
        "{desc:<10} {percentage:3.0f}%|{bar}|"
        " {n_fmt:>5}/{total_fmt} "
        "[{rate_fmt:>12} {elapsed:>8}/{remaining:<8}]"
    )
    BAR_FORMAT_1 = BAR_FORMAT_0
    n_fmt_align = 5
    prefill_tokens = num_requests
    decode_tokens = sum_output_length - prefill_tokens

    if len(str(decode_tokens)) > n_fmt_align:
        n_fmt_align = len(str(decode_tokens))
        BAR_FORMAT_0 = BAR_FORMAT_0.replace("{n_fmt:>5}", "{n_fmt:>" + str(n_fmt_align) + "}")
        BAR_FORMAT_1 = BAR_FORMAT_0

    if len(str(prefill_tokens)) < len(str(decode_tokens)):
        old_align_str = "{n_fmt:>" + str(n_fmt_align) + "}"
        n_fmt_align += len(str(decode_tokens)) - len(str(prefill_tokens))
        BAR_FORMAT_0 = BAR_FORMAT_0.replace(old_align_str, "{n_fmt:>" + str(n_fmt_align) + "}")

    disabled = not use_pbar
    input_pbar = tqdm(
        total=num_requests,
        desc="Requests sent",
        position=0,
        bar_format=BAR_FORMAT_0,
        disable=disabled,
    )
    output_pbar = tqdm(
        total=num_requests,
        desc="Requests done",
        position=1,
        bar_format=BAR_FORMAT_0,
        disable=disabled,
    )
    prefill_pbar = tqdm(
        total=prefill_tokens,
        desc="Prefill token",
        position=2,
        bar_format=BAR_FORMAT_0,
        disable=disabled,
    )
    decode_pbar = tqdm(
        total=decode_tokens,
        desc="Decode token ",
        position=3,
        bar_format=BAR_FORMAT_1,
        disable=disabled,
    )
    return Console(
        input_pbar=input_pbar,
        output_pbar=output_pbar,
        prefill_pbar=prefill_pbar,
        decode_pbar=decode_pbar,
        disabled=disabled,
    )


def generate_prompt(tokenizer: Any, n: int) -> str:
    """Generate a prompt of approximately `n` tokens using the provided tokenizer."""
    vocab_size = tokenizer.vocab_size // 2
    token_ids = [random.randint(0, vocab_size) for _ in range(n)]

    for _ in range(64):
        prompt = tokenizer.decode(token_ids)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) == n:
            return prompt
        if len(token_ids) < n:
            need = n - len(token_ids)
            token_ids.extend([random.randint(0, vocab_size) for _ in range(need)])
        else:
            token_ids = token_ids[:n]

    raise ValueError("Failed to generate a message of the desired length.")


async def benchmark_one(
    client: OpenAI,
    prompt: str,
    output_length: int,
    model: str,
    *,
    pbar: Console | bool = True,
    extra_body: Dict[str, Any] | None = None,
    input_length: int | None = None,  # a hack to force input length
) -> RawResult:
    """发送单个 OpenAI chat completion streaming 请求并记录 token 到达时间。

    返回 RawResult.tics，其中 tics[0] 是请求发出前时间，tics[1] 是首个 chunk 到达时间，
    后续元素对应后续 streaming chunk 到达时间。
    """

    if isinstance(pbar, bool):
        pbar = make_console(1, output_length, use_pbar=pbar)
    with pbar.inflight(1):
        kwargs = {
            "ignore_eos": True,
            "top_k": 1,
        }
        # this is an internal kwargs that might work for our system
        if input_length is not None:
            kwargs["input_length_override"] = input_length
        kwargs.update(extra_body or {})  # can override kwargs
        response = await client.chat.completions.create(
            model=model,
            stream=True,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=output_length,
            temperature=0.0,
            extra_body=kwargs,
        )
        tics = [time.perf_counter()]
        async for _ in response:
            tics.append(time.perf_counter())
            if len(tics) == 2:
                pbar.update_prefill()
            elif len(tics) <= output_length + 1:
                pbar.update_decode()
        return RawResult(
            input_len=input_length,
            output_len=output_length,
            message=prompt,
            tics=tics,
        )


async def benchmark_one_batch(
    client: OpenAI,
    prompts: List[str],
    output_lengths: List[int] | int,
    model: str,
    *,
    extra_body: Dict[str, Any] | None = None,
    input_lengths: List[int | None] | None = None,
    pbar: Console | bool = True,
) -> List[RawResult]:
    """并发发送一批 prompt，并返回每个请求的 RawResult。

    output_lengths 可以是单个整数或逐请求列表；input_lengths 用于 trace replay 时强制
    记录原始输入长度，而不必重新 tokenizer 统计。
    """

    if isinstance(output_lengths, int):
        output_lengths = [output_lengths] * len(prompts)
    if isinstance(pbar, bool):
        pbar = make_console(len(prompts), sum(output_lengths), use_pbar=pbar)
    if input_lengths is None:
        l: List[int | None] = [None] * len(prompts)
        input_lengths = l  # work-around for typing bug

    tasks = [
        benchmark_one(
            client=client,
            prompt=prompt,
            output_length=output_length,
            model=model,
            pbar=pbar,
            extra_body=extra_body,
            input_length=input_length,
        )
        for prompt, output_length, input_length in zip(
            prompts, output_lengths, input_lengths, strict=True
        )
    ]
    with pbar.log_stats():
        return await asyncio.gather(*tasks)


async def benchmark_trace(
    client: OpenAI,
    msgs: List[BenchmarkTrace],
    model: str,
    *,
    pbar: Console | bool = True,
) -> List[RawResult]:
    """按 trace timestamp 回放一组请求。

    msg.timestamp 单位为秒；函数会按最早 timestamp 对齐到当前时间后异步 sleep，
    近似模拟线上请求到达过程。
    """

    if isinstance(pbar, bool):
        sum_output_len = sum(msg.output_length for msg in msgs)
        pbar = make_console(len(msgs), sum_output_len, use_pbar=pbar)
    start = time.perf_counter()
    offset = min(msg.timestamp for msg in msgs) - 1

    async def benchmark_timed(msg: BenchmarkTrace):
        """等待到 trace 目标时间后发起单个请求。"""

        target = start + msg.timestamp - offset
        await asyncio.sleep(max(0, target - time.perf_counter()))
        return await benchmark_one(
            client, msg.message, msg.output_length, model, pbar=pbar, input_length=msg.input_length
        )

    tasks = [benchmark_timed(msg) for msg in msgs]
    with pbar.log_stats():
        return await asyncio.gather(*tasks)


@overload
def process_benchmark_results(raw_data: List[RawResult], tokenizer: Any) -> BenchmarkResult:
    """传入 tokenizer 时返回可序列化 BenchmarkResult。"""
    ...


@overload
def process_benchmark_results(raw_data: List[RawResult]) -> None:
    """不传 tokenizer 时只打印统计日志，不返回结果。"""
    ...


def process_benchmark_results(
    raw_data: List[RawResult],
    tokenizer: Any = UNSET,
) -> BenchmarkResult | None:
    """汇总 benchmark 原始时间戳并打印延迟/吞吐指标。

    TTFT 使用首 chunk 延迟；TPOT 使用首 token 之后相邻 chunk 间隔；
    E2E 使用每个请求最后一个 chunk 与请求开始时间之差。
    如果传入 tokenizer，则额外返回可保存的 BenchmarkResult。
    """

    accum_times: List[float] = []
    first_times: List[float] = []
    results = [r.tics for r in raw_data]
    for tics in results:
        deltas: List[float] = []
        for i in range(len(tics) - 1):
            diff = tics[i + 1] - tics[i]
            deltas.append(diff)
        first_times.append(deltas[0])
        accum_times.extend(deltas[1:])

    e2e_times = [tics[-1] - tics[0] for tics in results]
    first_times.sort()
    accum_times.sort()
    e2e_times.sort()

    def _print_stats(times: List[float], scale: float = 1.0) -> Tuple[float, ...]:
        """返回 avg/p50/p90/p99/max，scale 用于秒到毫秒转换。"""

        assert len(times) > 0
        return (
            scale * sum(times) / len(times),  # avg
            scale * times[int(len(times) * 0.5)],  # p50
            scale * times[int(len(times) * 0.9)],  # p90
            scale * times[int(len(times) * 0.99)],  # p99
            scale * max(times),  # max
        )

    def _fmt(x: float) -> str:
        """把数值格式化成固定宽度字符串，便于日志对齐。"""

        if x >= 1000:
            return f"{int(x):>6}"
        elif x >= 10:
            return f"{x:>6.2f}"
        else:
            return f"{x:>6.4f}"

    avg_ttft, p50_ttft, p90_ttft, p99_ttft, max_ttft = _print_stats(first_times, 1000)
    avg_tpot, p50_tpot, p90_tpot, p99_tpot, max_tpot = _print_stats(accum_times, 1000)
    avg_e2e, p50_e2e, p90_e2e, p99_e2e, max_e2e = _print_stats(e2e_times)

    min_time = min(min(r) for r in results)
    max_time = max(max(r) for r in results)
    dur = max_time - min_time
    assert dur > 0, "Duration must be positive"

    num_tokens = sum(len(tic) for tic in results)
    num_requests = len(results)

    logger.info(f"Num requests: #{num_requests}, Num tokens: #{num_tokens}")
    logger.info(
        f"TTFT: {_fmt(avg_ttft)} ms (p50: {_fmt(p50_ttft)} ms, p90: {_fmt(p90_ttft)} ms,"
        f" p99: {_fmt(p99_ttft)} ms, max: {_fmt(max_ttft)} ms)"
    )
    logger.info(
        f"TPOT: {_fmt(avg_tpot)} ms (p50: {_fmt(p50_tpot)} ms, p90: {_fmt(p90_tpot)} ms,"
        f" p99: {_fmt(p99_tpot)} ms, max: {_fmt(max_tpot)} ms)"
    )
    logger.info(
        f"E2E:  {_fmt(avg_e2e) }  s (p50: {_fmt(p50_e2e) }  s, p90: {_fmt(p90_e2e) }  s,"
        f" p99: {_fmt(p99_e2e) }  s, max: {_fmt(max_e2e) }  s)"
    )
    logger.info(f"Duration: {_fmt(dur)} s")
    logger.info(f"Throughput: {_fmt(num_tokens / dur)} token/s, {_fmt(num_requests / dur)} req/s")

    # normalize the time to start from zero
    results = [[r - min_time for r in tics] for tics in results]
    if isinstance(tokenizer, Unset):
        return None

    return BenchmarkResult(
        raw_data=[
            BenchOneResult(
                tics=r.tics,
                input_len=(
                    r.input_len
                    if r.input_len is not None
                    else len(tokenizer.encode(r.message, add_special_tokens=False))
                ),
                output_len=r.output_len,
            )
            for r in raw_data
        ]
    )


def read_qwen_trace(
    file_path: str,
    tokenizer: Any,
    n: int | None = None,
    dummy: bool = False,
) -> List[BenchmarkTrace]:
    """读取 Qwen trace JSONL 并转换成 BenchmarkTrace 列表。

    n 限制读取前 n 行；dummy=True 时复用一段最长 prompt 切片，减少随机生成差异。
    timestamp 单位保持原 trace 的秒。
    """

    class JSONInput(BaseModel):
        """Qwen trace 单行 JSON schema。"""

        chat_id: int
        parent_chat_id: int
        timestamp: float
        input_length: int
        output_length: int
        type: str  # unused
        turn: int  # unused
        hash_ids: List[int]  # unused

    with open(file_path, "r") as f:
        lines = f.readlines()
        if n is not None:
            lines = lines[:n]
    objs = [JSONInput.model_validate_json(line) for line in lines]
    if dummy:
        prompt = generate_prompt(tokenizer, max(obj.input_length for obj in objs))
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        _get_prompt = lambda obj: tokenizer.decode(ids[: obj.input_length])
    else:
        _get_prompt = lambda obj: generate_prompt(tokenizer, obj.input_length)
    return [
        BenchmarkTrace(
            timestamp=obj.timestamp,
            message=_get_prompt(obj),
            input_length=obj.input_length,
            output_length=obj.output_length,
        )
        for obj in objs
    ]


def read_mooncake_trace(
    file_path: str,
    tokenizer: Any,
    n: int | None = None,
    dummy: bool = False,
) -> List[BenchmarkTrace]:
    """读取 Mooncake trace JSONL 并转换成 BenchmarkTrace 列表。

    原始 timestamp 单位是毫秒，返回时除以 1000 转成秒。
    """

    class JSONInput(BaseModel):
        """Mooncake trace 单行 JSON schema。"""

        timestamp: int
        input_length: int
        output_length: int
        hash_ids: List[int]  # unused for now

    with open(file_path, "r") as f:
        lines = f.readlines()
        if n is not None:
            lines = lines[:n]
    objs = [JSONInput.model_validate_json(line) for line in lines]
    if dummy:
        prompt = generate_prompt(tokenizer, max(obj.input_length for obj in objs))
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        _get_prompt = lambda obj: tokenizer.decode(ids[: obj.input_length])
    else:
        _get_prompt = lambda obj: generate_prompt(tokenizer, obj.input_length)
    return [
        BenchmarkTrace(
            timestamp=obj.timestamp / 1000,
            message=_get_prompt(obj),
            input_length=obj.input_length,
            output_length=obj.output_length,
        )
        for obj in objs
    ]


def scale_traces(
    traces: List[BenchmarkTrace],
    scale: float,
) -> List[BenchmarkTrace]:
    """按比例缩放 trace 到达时间。

    先把最早 timestamp 平移到 0，再乘以 scale；scale<1 表示请求到达更密集。
    """

    min_tic = min(trace.timestamp for trace in traces)
    return sorted(
        [
            BenchmarkTrace(
                timestamp=(trace.timestamp - min_tic) * scale,
                message=trace.message,
                input_length=trace.input_length,
                output_length=trace.output_length,
            )
            for trace in traces
        ],
        key=lambda x: x.timestamp,
    )


async def get_model_name(client: OpenAI) -> str:
    """从 OpenAI-compatible /models 接口读取第一个模型名。"""

    async for model in client.models.list():
        return model.id
    raise ValueError("No models available")
