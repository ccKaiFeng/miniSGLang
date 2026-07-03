from __future__ import annotations

# 这个文件负责 Scheduler 侧的 KV cache page 分配和 prefix cache 管理。
#
# KV cache 可以理解成模型 attention 的历史 K/V 数据缓存。为了避免每次都重新
# 计算 prompt，系统会把已经算过的 token 对应的 K/V 保存下来。
#
# CacheManager 主要做：
# - 给请求分配新的 KV cache page；
# - 查询 prefix cache，看 prompt 前缀是否已有缓存可复用；
# - 请求完成或被驱逐时释放 page；
# - 把逻辑 token 位置写入 page_table，供 attention kernel 找到物理 KV 位置。

from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Req
from minisgl.kvcache import BaseCacheHandle, MatchResult, create_prefix_cache
from minisgl.utils import div_ceil

if TYPE_CHECKING:
    from .utils import PendingReq


class CacheManager:
    """Scheduler 侧 KV cache 管理器。"""

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        page_table: torch.Tensor,
        type: str,
        zipcache_manager=None,
    ):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        device = page_table.device

        # free_slots 保存空闲 page 的起始 token index。
        # 如果 page_size=4，第 3 页起始 token index 是 3*4=12。
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size

        # prefix_cache 可以是 radix cache 或 naive cache。
        self.prefix_cache = create_prefix_cache(device=device, type=type)
        self.device = device
        self.num_pages = num_pages

        # page_table[table_idx, token_pos] = KV cache 物理 token index。
        self.page_table = page_table
        self.page_size = page_size
        self.zipcache_manager = zipcache_manager

    def match_req(self, req: PendingReq) -> MatchResult:
        """查询请求 prompt 是否有可复用前缀缓存。

        只匹配到 input_len - 1，是因为最后一个 token 通常还需要作为本次 prefill
        的输出位置参与计算。
        """

        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        result = self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])
        if self.zipcache_manager is None or not hasattr(self.zipcache_manager, "materialize_match"):
            return result
        self.prefix_cache.lock_handle(result.cuda_handle)
        try:
            handle = self.zipcache_manager.materialize_match(
                self.prefix_cache,
                result.cuda_handle,
                self,
            )
        finally:
            self.prefix_cache.lock_handle(result.cuda_handle, unlock=True)
        return MatchResult(handle)

    @property
    def available_size(self) -> int:
        """当前可用 token 容量估计。

        包括：
        - prefix cache 中可被驱逐的 token；
        - free_slots 中完全空闲的 page。
        """

        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        """锁定一个 cache handle，防止其对应缓存被驱逐。"""

        if (
            self.zipcache_manager is not None
            and hasattr(self.zipcache_manager, "lock_handle")
            and self.zipcache_manager.lock_handle(self.prefix_cache, handle, unlock=False)
        ):
            return
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        """解锁一个 cache handle，允许其在需要时被驱逐。"""

        if (
            self.zipcache_manager is not None
            and hasattr(self.zipcache_manager, "lock_handle")
            and self.zipcache_manager.lock_handle(self.prefix_cache, handle, unlock=True)
        ):
            return
        self.prefix_cache.lock_handle(handle, unlock=True)

    def release_handle_resources(self, handle: BaseCacheHandle) -> None:
        """释放 v3 这类 handle 附带的临时 normal page。"""

        if self.zipcache_manager is not None and hasattr(
            self.zipcache_manager, "release_handle_resources"
        ):
            self.zipcache_manager.release_handle_resources(handle, self)

    def carry_handle_resources(
        self, old_handle: BaseCacheHandle, new_handle: BaseCacheHandle
    ) -> BaseCacheHandle:
        """把旧 handle 的临时资源转交给新 handle。

        普通 cache 没有附加资源，直接返回 new_handle。
        """

        if self.zipcache_manager is not None and hasattr(
            self.zipcache_manager, "carry_handle_resources"
        ):
            return self.zipcache_manager.carry_handle_resources(old_handle, new_handle)
        return new_handle

    def allocate_paged(self, reqs: List[Req]) -> None:
        """为一批请求分配缺失的 KV cache page。

        根据每个 req 的 cached_len 和 device_len 判断哪些 token 位置还没有 page，
        然后统一分配 page，并写入 page_table。
        """

        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            # first_page 是已经有 cache 的末尾 page。
            first_page = div_ceil(req.cached_len, self.page_size)

            # last_page 是当前 device_len 需要覆盖到的末尾 page。
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            # _allocate 返回 page 起始 index；_page_to_token 展开成每个 token index。
            allocated = self._page_to_token(self._allocate(needed_pages))
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    def cache_req(self, req: Req, *, finished: bool) -> None:
        """把一个请求已经计算过的前缀插入 prefix cache，并释放多余 page。

        这个函数通常在请求完成一段 prefill/decode 后调用。
        """

        # ==================================== valid cache region ====================================
        # [0, req.cached_len)                       This part is valid for attention kernel read/write.
        # [0, old_handle.cached_len)                This part is in the prefix cache before prefill.
        # [old_handle.cached_len, req.cached_len)   This part is allocated by cache manager for this request.
        # ================================== allocated cache region ==================================
        # [old_handle.cached_len, cached_len)       This part was not in the prefix cache when prefill,
        #                                           but later cached by other requests.
        #                                           We must free them to avoid memory leak.
        # [cached_len, new_handle.cached_len)       This part is newly inserted into the prefix cache.
        # [new_handle.cached_len, req.cached_len)   This part is tailing part that can not inserted into the prefix cache.
        #                                           We should free it if the request has finished.
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle

        # 尝试把 [0, req.cached_len) 插入 prefix cache。
        # 如果已有其他请求插入过相同前缀，cached_len 可能大于 old_handle.cached_len。
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)

        # unlock until all operations on handle is done
        # 老 handle 不再由当前请求持有，先解锁。
        self.unlock(old_handle)

        # this part is already in the prefix cache, free it
        # 这一段已被 prefix cache 复用，不需要当前请求独占这些 page。
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:  # this tail part should be freed
            # 请求结束，不能插入 prefix cache 的尾部也释放。
            self._free(page_indices[new_handle.cached_len :])
            should_demote = False
            if self.zipcache_manager is not None and hasattr(self.zipcache_manager, "demote_node"):
                config = self.zipcache_manager.config
                should_demote = bool(
                    getattr(config, "zipcache_v3_demote_on_finish", False)
                )
            if should_demote and hasattr(new_handle, "node"):
                # v3 的 normal pool 被刻意缩小为工作区。如果只压缩 finished
                # prefix 的叶子节点，radix tree 中可能留下 normal 父节点 + compressed
                # 子节点：父节点会被计入 evictable_size，但由于它不再是叶子，原
                # radix evict() 无法真正释放它，normal pool 紧张时会触发断言。
                #
                # 因此 v3 在请求结束时尝试把整条已解锁路径上的 normal 节点都
                # demote 到 compressed pool。
                if hasattr(self.prefix_cache, "path_nodes"):
                    nodes_to_demote = list(reversed(self.prefix_cache.path_nodes(new_handle)))
                else:
                    nodes_to_demote = [new_handle.node]

                demoted_parts: List[torch.Tensor] = []
                for node in nodes_to_demote:
                    demoted = self.zipcache_manager.demote_node(node)
                    if demoted is not None:
                        self.prefix_cache.mark_node_compressed(
                            node,
                            self.zipcache_manager.entry_by_node_uuid[node.uuid],
                        )
                        demoted_parts.append(demoted)
                if demoted_parts:
                    self._free(torch.cat(demoted_parts))
            self.release_handle_resources(old_handle)
        else:  # keep the tail part, update the handle
            # 请求还要继续 decode，更新 handle 并锁定，防止被驱逐。
            req.cache_handle = self.carry_handle_resources(old_handle, new_handle)
            self.lock(req.cache_handle)

    def check_integrity(self) -> None:
        """检查 free page + cache page 数是否等于总 page 数。"""

        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free_region(self):
        """延迟释放 page 的上下文管理器。

        在某些批量操作中，立即把 page 拼回 free_slots 会产生很多 torch.cat。
        lazy_free_region 会临时把 _free 替换成“先记录，最后统一释放”。
        """

        def lazy_free(indices: torch.Tensor) -> None:
            # indices 是 token index，按 page_size 取每页起点。
            lazy_free_list.append(indices[:: self.page_size])

        lazy_free_list: List[torch.Tensor] = []
        try:
            # 临时覆盖 self._free。
            self._free = lazy_free
            yield
        finally:
            del self._free
            # 退出上下文时一次性把所有释放 page 拼回 free_slots。
            self.free_slots = torch.cat([self.free_slots] + lazy_free_list)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        """分配 needed_pages 个 page，不够时从 prefix cache 驱逐。"""

        if needed_pages > (free_pages := len(self.free_slots)):
            # free page 不够，驱逐可驱逐的 prefix cache。
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."

        # 从 free_slots 头部取出需要的 page。
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        """释放一段 token indices 对应的 page。"""

        if len(indices) > 0:
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def allocate_token_indices(self, length: int) -> torch.Tensor:
        """为 restore 分配 length 个 token 位置，并返回可直接写入 page_table 的 indices。"""

        if length <= 0:
            return torch.empty(0, dtype=torch.int32, device=self.device)
        needed_pages = div_ceil(length, self.page_size)
        return self._page_to_token(self._allocate(needed_pages))[:length]

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        """把 page 起始 index 展开成 token index。"""

        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    """把新分配的物理 KV token index 写入 page_table。

    allocation_info 的每个元素是：
    - table_idx：请求槽位；
    - first_page：从哪个逻辑 page 开始写；
    - last_page：写到哪个逻辑 page 之前。
    """

    needed_tokens = len(allocated)

    # 先在 pinned CPU memory 上构造索引，再异步拷贝到 GPU。
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos

        # table_idx_host 记录 page_table 第一维，也就是请求槽位。
        table_idx_host[offset : offset + length].fill_(table_idx)

        # positions_host 记录 page_table 第二维，也就是该请求内的 token 位置。
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)

    # 最终写入：page_table[请求槽位, token位置] = 物理KV token index。
    page_table[table_idxs, offsets] = allocated
