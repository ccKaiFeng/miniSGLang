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

        参数维度：
        - num_kv_heads: 全模型 KV head 数，切分后得到 local_kv_heads；
        - num_layers: transformer 层数 L；
        - head_dim: 每个 head 的通道数 D；
        - num_pages/page_size: normal KV pool 容量是 num_pages * page_size 个 token。
        """

        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        # _kv_buffer shape [2, L, P, S, H_kv_local, D]：
        #   2 = K/V 两类；L = layer；P = page；S = page_size。
        # self._k_buffer[index] 和 self._v_buffer[index] 的 shape 都是 [P, S, H_kv_local, D]。
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        # 展平后按物理 token index 访问：shape [P*S, H_kv_local, D]。
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
        - k/v shape: [T, H_kv_local, D] 或 [T, H_kv_local*D]，kernel 按行拷贝；
        - out_loc shape: [T]，每个元素范围是 [0, P*S)；
        - layer_id: 当前 transformer layer 编号。
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
        """返回 KV cache 所在设备。"""

        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """返回 K/V buffer 的 dtype。"""

        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        """返回 KV cache 包含的 transformer layer 数。"""

        return self._num_layers
