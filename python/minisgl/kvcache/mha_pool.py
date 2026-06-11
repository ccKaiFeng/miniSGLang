from __future__ import annotations

# 这个文件实现真正存放 attention K/V 的显存池。
#
# MHAKVCache 会分配一个大 tensor，形状大致是：
#   [K/V, layer, page, page_size, local_kv_heads, head_dim]
#
# Scheduler 负责决定某个 token 写到哪个 out_loc；AttentionLayer 负责产生每层
# 的 k/v；本类负责把 k/v 写进对应 layer 的 cache。

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """Multi-Head Attention 使用的 KV cache pool。"""

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """分配 K/V cache 显存。

        num_kv_heads 会按 TP size 切分到本 rank；allow_replicate=True 表示
        KV head 数小于 TP size 时允许复制。
        """

        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        """返回第 index 层的 K cache。"""

        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        """返回第 index 层的 V cache。"""

        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        """把当前 layer 新算出的 k/v 写入 KV cache。

        out_loc 是 Scheduler 生成的物理位置索引，store_cache 是自定义 CUDA kernel。
        """

        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
