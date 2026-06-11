import torch

# 这个文件实现最简单的 prefix cache：什么前缀都不复用。
#
# 它主要用于关闭 radix cache 做对比实验，或者在调试 cache 逻辑时排除前缀复用
# 的影响。注意它不等于“不使用 KV cache”：模型 decode 过程中仍会使用 KV cache，
# 只是不同请求之间不共享 prompt 前缀。

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo


class NaiveCacheHandle(BaseCacheHandle):
    """naive cache 的 handle，永远表示命中长度为 0。"""

    empty_tensor: torch.Tensor  # should be set by NaivePrefixCache

    def __init__(self):
        super().__init__(cached_len=0)

    def get_matched_indices(self) -> torch.Tensor:
        """返回空 tensor，表示没有任何 prefix cache 命中。"""

        return self.empty_tensor


class NaivePrefixCache(BasePrefixCache):
    """不做前缀复用的 BasePrefixCache 实现。"""

    def __init__(self, device: torch.device):
        """创建一个空 cache。"""

        self.device = device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        NaiveCacheHandle.empty_tensor = self.empty_tensor
        super().__init__()

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """naive cache 没有真实节点，所以 lock/unlock 什么也不做。"""

        pass

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """永远返回 0 长度命中。"""

        return MatchResult(NaiveCacheHandle())

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """不保存任何前缀，只返回 0 长度插入结果。"""

        return InsertResult(0, NaiveCacheHandle())

    def evict(self, size: int) -> torch.Tensor:
        """naive cache 没有可驱逐内容；只能安全驱逐 0。"""

        if size == 0:
            return self.empty_tensor
        raise NotImplementedError("NaiveCacheManager does not support eviction.")

    def reset(self) -> None:
        """没有内部状态需要重置。"""

        pass

    @property
    def size_info(self) -> SizeInfo:
        """没有可驱逐或受保护前缀。"""

        return SizeInfo(evictable_size=0, protected_size=0)

    def check_integrity(self) -> None:
        """没有复杂结构需要检查。"""

        pass
