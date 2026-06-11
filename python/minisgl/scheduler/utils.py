from __future__ import annotations

# 这个文件放 Scheduler 内部使用的小型数据结构。

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from minisgl.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    """等待被 Scheduler 接纳的新请求。

    tokenizer 发来的 UserMsg 会先变成 PendingReq。它还没有正式分配 table slot、
    KV cache handle，也还没有进入 running decode 集合。
    """

    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams

    # 如果请求太长，PrefillManager 可能会把它切成多个 chunk。
    # chunked_req 用来保存切块后的中间状态。
    chunked_req: ChunkedReq | None = None

    @property
    def input_len(self) -> int:
        """prompt token 数。"""

        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        """这个请求最多要生成的 token 数。"""

        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    """PrefillManager 一轮调度的输出结果。

    - reqs：这轮被选中进入 prefill 的请求；
    - output_indices：每个请求本轮 forward 后需要取 logits 的位置。
    """

    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
