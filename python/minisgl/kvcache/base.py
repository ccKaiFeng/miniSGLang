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

    本工程的常用维度符号：
    - L: transformer layer 数；
    - P: normal KV pool 的 page 数；
    - S: page_size，每个 page 中 token 数；
    - H_kv: 本 TP rank 上的 KV head 数；
    - D: head_dim；
    - T: 当前 batch 本轮实际计算的新 token 数。
    """

    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor:
        """返回第 index 层 K cache，典型 shape [P, S, H_kv, D]。"""
        ...

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor:
        """返回第 index 层 V cache，典型 shape [P, S, H_kv, D]。"""
        ...

    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        """把本轮新算出的 K/V 写入 cache。

        k/v shape 可以是 [T, H_kv, D] 或已经展平的 [T, H_kv*D]；
        store kernel 会按每个 token 一整行拷贝。out_loc shape [T]，
        每个元素是物理 token index。
        """
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """KV cache 所在设备，通常是当前 TP rank 的 CUDA device。"""
        ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """KV cache tensor 的 dtype，例如 torch.float16 或 torch.bfloat16。"""
        ...

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """KV cache 覆盖的 transformer layer 数 L。"""
        ...


@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    """prefix cache 命中结果的句柄。

    handle 可以理解为“我命中了 cache 里的哪一段前缀”。Scheduler 后续通过它
    取出已经缓存好的 KV 位置，并在使用期间 lock，避免被驱逐。
    """

    cached_len: int

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor:
        """返回命中前缀对应的物理 token indices，shape [cached_len]。"""
        ...


class SizeInfo(NamedTuple):
    """prefix cache 当前容量状态。"""

    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        """当前 prefix cache 总 token 数 = 可驱逐 + 受保护。"""

        return self.evictable_size + self.protected_size


class InsertResult(NamedTuple):
    """插入 prefix cache 后返回的信息。"""

    cached_len: int  # 已存在 cache 中的 token 数，单位 token，不是 page。
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
