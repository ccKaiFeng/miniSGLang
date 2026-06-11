from __future__ import annotations

# 这个文件管理 prefill 阶段。
#
# prefill 指处理用户 prompt 的阶段：模型需要把 prompt 中已有 token 全部跑一遍，
# 建立 KV cache，并得到第一个可采样位置的 logits。
#
# 如果 prompt 很长，一次性 prefill 会占用大量显存，所以这里支持 chunked prefill：
# 把长 prompt 切成多个 chunk 分多轮处理。

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    """表示一个还没有完成完整 prefill 的切块请求。

    ChunkedReq 只是中间状态，不应该进入 decode，也不应该采样。
    """

    def append_host(self, next_token: torch.Tensor) -> None:
        """chunked prefill 中间请求不允许追加采样 token。"""

        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        """ChunkedReq 不能 decode，必须等完整 prompt prefill 完成。"""

        return False  # avoid being added to decode manager


@dataclass
class PrefillAdder:
    """尝试把 pending 请求加入本轮 prefill batch 的辅助类。

    它同时检查三类资源：
    - token_budget：本轮 prefill 还能处理多少 token；
    - table_manager：还有没有请求槽位；
    - cache_manager：KV cache 空间是否足够。
    """

    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        """尝试为一个新请求分配 table slot 和 prefix cache handle。"""

        if self.table_manager.available_size == 0:
            return None

        # TODO: consider host cache match case
        # 查询 prefix cache，看看 prompt 前缀是否已经有 KV cache 可复用。
        handle = self.cache_manager.match_req(req).cuda_handle
        cached_len = handle.cached_len

        # TODO: better estimate policy
        # extend_len 是这次 prompt 中仍需实际计算的长度。
        extend_len = req.input_len - cached_len

        # estimated_len 粗略估计这个请求未来总共会占多少 token cache。
        estimated_len = extend_len + req.output_len

        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None

        # lock 之后，handle 对应的 prefix cache 不会被其他分配驱逐。
        self.cache_manager.lock(handle)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            # lock 后再检查一次，防止 available_size 因锁定状态变化而不够。
            return self.cache_manager.unlock(handle)

        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            # 如果前缀已经命中 cache，需要把已命中的 token id 和 page table 信息
            # 写入这个请求的 table slot。
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(handle.get_matched_indices())

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        """把一个 pending request 转成 Req/ChunkedReq，并写入本轮 token。"""

        remain_len = pending_req.input_len - cached_len

        # 本轮最多只能处理 token_budget 个未缓存 token。
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req

        self.token_budget -= chunk_size

        # reserved_size 记录已经被本轮接纳的请求未来可能占用的 KV cache。
        self.reserved_size += remain_len + pending_req.output_len

        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)

        # input_ids 只截到本轮 chunk 末尾。若是 ChunkedReq，后续还会继续处理剩余 prompt。
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        """尝试把一个 pending request 加入本轮 prefill。

        返回 Req/ChunkedReq 表示加入成功；返回 None 表示资源不足或 token budget 用完。
        """

        if self.token_budget <= 0:
            return None

        if chunked_req := pending_req.chunked_req:
            # 这个请求之前已经被切块处理过，本轮继续使用原来的 cache_handle/table_idx。
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        if resource := self._try_allocate_one(pending_req):
            # 新请求首次进入 prefill，需要先分配资源。
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None


@dataclass
class PrefillManager:
    """维护等待 prefill 的请求队列，并生成下一轮 prefill batch。"""

    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        """把 tokenizer 发来的新请求放入 pending 队列。"""

        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        """根据 prefill_budget 选择下一批要 prefill 的请求。"""

        if len(self.pending_list) == 0:
            return None

        # estimated offset due to in-flight decode
        # decode 中还没完成的请求会占用未来 KV cache，所以这里先预留。
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    # 没有完整 prefill 完，放回 pending 队列最前面，下轮继续。
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None

        # 新 pending 队列 = 未完成 chunk 的请求 + 本轮没有处理到的请求。
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        """取消还在 pending/prefill 阶段的请求。"""

        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req
        return None

    @property
    def runnable(self) -> bool:
        """当前是否有等待 prefill 的请求。"""

        return len(self.pending_list) > 0
