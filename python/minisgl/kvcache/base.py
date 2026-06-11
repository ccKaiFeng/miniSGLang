from __future__ import annotations

# 这个文件定义 KV cache 和 prefix cache 的抽象接口。
#
# KV cache 是 LLM 推理里的关键优化：模型生成后续 token 时，不需要反复计算
# 过去 token 的 Key/Value，而是把每层 attention 的 K/V 缓存在显存里。
#
# prefix cache 是更高一层的缓存：如果两个请求有相同前缀，例如系统 prompt
# 一样，就可以复用前缀对应的 KV cache，减少 prefill 计算。

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import torch


class BaseKVCachePool(ABC):
    """KV cache 显存池抽象接口。

    具体实现负责分配真实的 K/V tensor，并提供按 layer 访问和写入的接口。
    """

    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...


@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    """prefix cache 命中结果的句柄。

    handle 可以理解为“我命中了 cache 里的哪一段前缀”。Scheduler 后续通过它
    取出已经缓存好的 KV 位置，并在使用期间 lock，避免被驱逐。
    """

    cached_len: int

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor: ...


class SizeInfo(NamedTuple):
    """prefix cache 当前容量状态。"""

    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size


class InsertResult(NamedTuple):
    """插入 prefix cache 后返回的信息。"""

    cached_len: int  # length already in cache before insertion (should be freed)
    handle: BaseCacheHandle  # cache handle for the inserted prefix


class MatchResult(NamedTuple):
    """查询 prefix cache 后返回的信息。"""

    cuda_handle: BaseCacheHandle
    # TODO: support HiCache


class BasePrefixCache(ABC):
    """prefix cache 抽象接口。

    Scheduler 只依赖这些方法，不关心底层是 naive cache 还是 radix tree cache。
    """

    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """锁定或解锁一个 cache handle。

        被锁定的 cache 不能被 evict，因为当前请求还在使用它。
        unlock=True 表示使用结束，可以重新变成可驱逐状态。
        """

    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """查找 input_ids 有多少前缀已经在 cache 中。

        这个操作只查询，不修改 cache。返回的 handle 在真正使用前需要 lock。
        """

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """把新前缀插入 prefix cache。

        input_ids 是 token 序列，indices 是这些 token 对应的 KV cache 物理位置。
        """

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """驱逐一部分可驱逐 cache，返回被释放的 KV cache 位置。

        实际驱逐数量可能大于请求的 size，因为 cache 通常按 page 或树节点管理。
        """

    @abstractmethod
    def reset(self) -> None:
        """重置 prefix cache。"""

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        """返回当前可驱逐/受保护 cache 的大小。"""

    @abstractmethod
    def check_integrity(self) -> None:
        """检查 cache 内部结构是否一致；发现错误时抛异常。"""
