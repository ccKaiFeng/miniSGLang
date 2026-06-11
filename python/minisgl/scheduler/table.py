import torch

# 这个文件管理每个 running request 使用的 table slot。
#
# page_table/token_pool 的第一维是请求槽位，也就是 table_idx。
# Scheduler 每接纳一个请求，就需要 allocate 一个空槽位；请求完成后 free。


class TableManager:
    """管理请求槽位和 token/page 表。"""

    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._max_running_reqs = max_running_reqs

        # 空闲槽位列表。槽位编号范围是 [0, max_running_reqs)。
        self._free_slots = list(range(max_running_reqs))

        # page_table[table_idx, token_pos] = KV cache 中对应 token 的物理位置。
        self.page_table = page_table

        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        # token_pool 用来保存每个请求当前位置对应的 token id。
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        """当前还剩多少个空闲请求槽位。"""

        return len(self._free_slots)

    def allocate(self) -> int:
        """分配一个空闲 table slot。"""

        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        """释放一个 table slot，供后续请求复用。"""

        self._free_slots.append(slot)
