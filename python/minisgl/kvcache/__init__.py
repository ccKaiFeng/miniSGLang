from __future__ import annotations

# 这个文件是 kvcache 子包的统一出口和工厂函数。
#
# 外部代码通常不直接 import mha_pool/radix_cache/naive_cache，而是调用这里的
# create_kvcache_pool() 或 create_prefix_cache()。这样后续新增 cache 实现时，
# 调用方不需要大改。

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry

if TYPE_CHECKING:
    import torch
    from minisgl.models import ModelConfig

from .base import (
    BaseCacheHandle,
    BaseKVCachePool,
    BasePrefixCache,
    MatchResult,
    SizeInfo,
)


class CacheManagerCreator(Protocol):
    """prefix cache 创建函数的类型约束。"""

    def __call__(self, device: torch.device) -> BasePrefixCache: ...


SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> BaseKVCachePool:
    """创建真正存放 K/V tensor 的显存池。

    当前实现只支持 MHAKVCache。它会根据模型层数、KV head 数、page 数、
    page size 等参数分配 K/V cache tensor。
    """

    from .mha_pool import MHAKVCache  # TODO: support other variants (e.g. MLA)

    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(device: torch.device):
    """注册并创建 naive prefix cache。"""

    from .naive_cache import NaivePrefixCache

    return NaivePrefixCache(device=device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device):
    """注册并创建 radix prefix cache。"""

    from .radix_cache import RadixPrefixCache

    return RadixPrefixCache(device=device)


def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    """按名字创建 prefix cache。

    type 通常来自命令行参数 `--cache`，例如 "naive" 或 "radix"。
    """

    return SUPPORTED_CACHE_MANAGER[type](device)


__all__ = [
    "create_kvcache_pool",
    "create_prefix_cache",
    "BaseKVCachePool",
    "BaseCacheHandle",
    "BasePrefixCache",
    "SizeInfo",
    "MatchResult",
    "SUPPORTED_CACHE_MANAGER",
]
