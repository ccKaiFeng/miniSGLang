from __future__ import annotations

# 这个文件定义 miniSGLang 运行时最核心的数据结构。
#
# 可以把它理解成整个系统的“公共数据协议”：
# - SamplingParams：用户希望模型怎样生成，比如最多生成多少 token、是否随机采样；
# - Req：一个用户请求在后端调度和推理过程中的状态；
# - Batch：Scheduler 选出来、准备送进 Engine/Model 的一批请求；
# - Context：Engine 运行时的全局上下文，保存当前 batch、KV cache、attention backend 等。
#
# 后续的 server、tokenizer、scheduler、engine、layers 都会直接或间接使用这些对象。

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Literal

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from minisgl.moe import BaseMoeBackend


@dataclass
class SamplingParams:
    """用户侧采样参数。

    大模型不是一次性输出完整文本，而是一次生成一个 token。生成每个 token 时，
    模型会给出“每个词的概率分布”，采样参数决定如何从这个分布里选下一个 token。

    字段含义：
    - temperature：温度。越大越随机；小于等于 0 时基本等价于只选概率最高的 token。
    - top_k：只从概率最高的 k 个 token 中选择；-1 表示不限制。
    - top_p：只从累计概率达到 p 的 token 集合里选择；1.0 表示不限制。
    - ignore_eos：是否忽略 EOS(end of sequence，结束符)。
    - max_tokens：最多生成多少个新 token。
    """

    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        """判断当前是否是 greedy 解码。

        greedy 解码就是每一步都选概率最高的 token，不做随机采样。
        条件是：
        - temperature <= 0，或者 top_k == 1；
        - 并且 top_p 没有限制，也就是 top_p == 1.0。
        """

        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    """一个用户请求在后端推理过程中的状态。

    这个对象不是 HTTP 请求本身，而是 Scheduler/Engine 内部使用的运行状态。
    一个 Req 会从 prefill 阶段进入 decode 阶段，直到生成结束。

    关键长度概念：
    - input_ids：CPU 上保存的完整 token 序列，包含 prompt token 和已经生成的 token。
    - cached_len：已经写入 KV cache、后续可复用的 token 数。
    - device_len：当前已经准备送到设备侧/模型侧处理的 token 数。
    - max_device_len：prompt 长度 + 允许生成的最大输出长度。
    - remain_len：还能继续生成多少 token。
    - extend_len：本轮还需要送进模型计算、尚未 cache 的 token 数。
    """

    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle

    def __post_init__(self) -> None:
        """dataclass 初始化完成后的检查和派生字段计算。"""

        # input_ids 必须在 CPU 上，因为 tokenizer 输出和调度侧 token 列表都放在 CPU。
        assert self.input_ids.is_cpu

        # device_len 表示当前请求已经被后端纳入计算范围的 token 数。
        self.device_len = len(self.input_ids)

        # 最大可处理长度 = prompt 长度 + 用户允许生成的 token 数。
        self.max_device_len = len(self.input_ids) + self.output_len

        # cached_len 必须小于 device_len，device_len 不能超过最大长度。
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        """还可以继续生成的 token 数。"""

        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        """当前还有多少 token 需要做 prefill/extend 计算。"""

        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        """完成一次模型 forward 后，更新请求长度状态。

        decode 阶段每 forward 一次通常会生成一个新 token：
        - 旧的 device_len 位置已经完成计算，所以 cached_len 更新到 device_len；
        - 新 token 会被追加到请求尾部，所以 device_len 增加 1。
        """

        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        """把新生成的 token 追加到 CPU 侧 input_ids 后面。"""

        self.input_ids = torch.cat([self.input_ids, next_token])

    @property
    def can_decode(self) -> bool:
        """判断这个请求是否还能继续 decode。"""

        return self.remain_len > 0

    def __repr__(self) -> str:
        """打印 Req 时显示关键调度状态，便于日志调试。"""

        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    """Scheduler 选出来的一批请求。

    Batch 是 Engine/Model 一次 forward 的基本单位。Scheduler 会把多个 Req
    合成一个 Batch，然后补齐模型输入所需的 tensor 字段。

    字段含义：
    - reqs：真实请求列表；
    - phase：当前 batch 属于 prefill 还是 decode；
    - input_ids：本次送进模型的 token id；
    - positions：每个 token 在序列中的位置；
    - out_loc：模型输出/KV cache 写入的位置；
    - padded_reqs：为了 CUDA graph 或固定 batch size 可能补齐后的请求列表；
    - attn_metadata：attention backend 根据 batch 生成的元数据。
    """

    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    padded_reqs: List[Req] = field(init=False)
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)

    @property
    def is_prefill(self) -> bool:
        """当前 batch 是否处于 prefill 阶段。"""

        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        """当前 batch 是否处于 decode 阶段。"""

        return self.phase == "decode"

    @property
    def size(self) -> int:
        """真实请求个数。"""

        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        """补齐后的请求个数。没有 padding 时等于 size。"""

        return len(self.padded_reqs)


@dataclass
class Context:
    """Engine 内部的全局运行上下文。

    很多 layer 在 forward 时不希望每层都手工传一大堆参数，所以通过全局 Context
    读取当前 batch、KV cache、attention backend、MoE backend 等运行时资源。

    注意：Context 是进程内全局变量，不是跨进程共享对象。
    每个 Scheduler/Engine 进程会有自己的 Context。
    """

    page_size: int
    # NOTE: this table always treat page_size = 1
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    moe_backend: BaseMoeBackend = field(init=False)
    kv_cache: BaseKVCachePool = field(init=False)
    zipcache_manager: Any | None = field(default=None, init=False)
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        """取得当前正在 forward 的 batch。

        只有在 `with ctx.forward_batch(batch):` 的代码块内部才能访问。
        如果没有活跃 batch，说明调用位置不在模型 forward 上下文中。
        """

        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        """临时设置当前 forward 使用的 batch。

        这是一个 Python context manager，用法类似：

            with ctx.forward_batch(batch):
                model(...)

        进入 with 块时设置 self._batch，离开 with 块时清空。
        这样 attention layer、embedding layer 等就能通过 get_global_ctx().batch
        读取当前 batch。
        """

        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    """设置当前进程的全局 Context。

    Engine 初始化完成后调用一次。这里用 assert 防止重复设置，因为重复设置
    可能说明同一进程里错误地创建了多个 Engine。
    """

    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    """读取当前进程的全局 Context。"""

    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
