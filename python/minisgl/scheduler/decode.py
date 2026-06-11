from __future__ import annotations

# 这个文件管理 decode 阶段的请求集合。
#
# prefill 处理完 prompt 后，请求会进入 decode。decode 阶段通常每轮为每个请求
# 生成一个新 token，直到达到 max_tokens 或遇到 EOS。

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    """维护当前还在 decode 的请求。"""

    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        """加入新请求，并过滤掉已经不能继续 decode 的请求。"""

        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        """从 decode 集合中移除一个请求。"""

        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        """根据 uid 取消一个正在 decode 的请求。"""

        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        """估计当前 decode 请求还会占用多少 token 预算。

        每个 running req 还会继续生成 remain_len 个 token。因为 KV cache 按 page
        管理，还额外保守预留 page_size - 1 个 token。
        """

        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        """把所有可运行 decode 请求组成下一个 decode batch。"""

        if not self.runnable:
            return None
        # 按 uid 排序，保证 batch 顺序稳定，便于调试和多 rank 对齐。
        return Batch(reqs=sorted(self.running_reqs, key=lambda req: req.uid), phase="decode")

    @property
    def runnable(self) -> bool:
        """当前是否存在可 decode 的请求。"""

        return len(self.running_reqs) > 0
