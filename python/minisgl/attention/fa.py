from __future__ import annotations

# 这个文件实现 FlashAttention backend。
#
# 它负责把 batch 信息整理成 FlashAttention 需要的 cu_seqlens/page_table 等元数据，
# 并在 forward 时先写 KV cache，再调用 FA kernel 完成 attention。

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.utils import is_sm100_supported

from .base import BaseAttnBackend, BaseAttnMetadata
from .mixed import MixedAttnMetadata, maybe_prepare_mixed_metadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class FACaptureData(BaseCaptureData):
    """FlashAttention CUDA graph capture 使用的固定 buffer。"""

    pass


@dataclass
class FAMetadata(BaseAttnMetadata):
    """FlashAttention 每个 batch 的元数据。"""

    # cu_seqlens_* shape [padded_bs + 1]，dtype int32，表示 ragged batch 的前缀和。
    # 对第 i 个请求：
    #   q 范围是 [cu_seqlens_q[i], cu_seqlens_q[i+1])；
    #   k 范围是 [cu_seqlens_k[i], cu_seqlens_k[i+1])。
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    # cache_seqlens shape [padded_bs]，每个元素是 req.device_len。
    cache_seqlens: torch.Tensor
    max_seqlen_k: int
    max_seqlen_q: int

    # page_table shape [padded_bs, ceil(max_seqlen_k / page_size)]。
    # FlashAttention 需要 page id；page_size>1 时由 global raw token index 除以 page_size 得到。
    page_table: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        """返回每个请求最后一个 query token 在扁平 tensor 中的位置。"""

        return self.cu_seqlens_q[1 : 1 + bs] - 1


class FlashAttentionBackend(BaseAttnBackend):
    """基于 FlashAttention 的 attention backend。"""

    def __init__(self, config: ModelConfig):
        """初始化 FlashAttention backend。

        config 提供模型 head 数、head_dim 等静态结构信息；KV cache/page_size 从
        global context 读取。capture 相关字段默认为空，只有开启 CUDA Graph 时才填充。
        """

        ctx = get_global_ctx()
        self.config = config
        self.kvcache = ctx.kv_cache
        self.page_size = ctx.page_size
        self.capture: FACaptureData | None = None
        self.max_graph_bs = 0
        self.capture_bs: List[int] = []
        self.scale = config.head_dim**-0.5
        self.version = 4 if is_sm100_supported() else 3

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """写入本层新 K/V，然后调用 FlashAttention paged KV kernel。

        q shape [T_q, H_q_local, D]；k/v shape [T_new, H_kv_local*D]
        或等价展平行向量，写入 KV cache 后按 [physical_token, H_kv_local, D] 读取；
        返回 shape [T_q, H_q_local, D]。
        """

        metadata = batch.attn_metadata
        if isinstance(metadata, MixedAttnMetadata):
            from minisgl.kernel import mixed_paged_attention

            self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
            return mixed_paged_attention(
                q=q,
                normal_k=self.kvcache.k_cache(layer_id),
                normal_v=self.kvcache.v_cache(layer_id),
                metadata=metadata,
                layer_id=layer_id,
            )
        assert isinstance(metadata, FAMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        return _fa_sgl_impl(
            q=q,
            k_cache=self.kvcache.k_cache(layer_id),
            v_cache=self.kvcache.v_cache(layer_id),
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            softmax_scale=self.scale,
            version=self.version,
        )

    def prepare_metadata(self, batch: Batch) -> None:
        """把 Batch 的请求长度和 page_table 转成 FlashAttention metadata。

        padded_size = len(batch.padded_reqs)，可能大于真实 batch.size。
        seqlens_q[i] = req.extend_len；seqlens_k[i] = req.device_len。
        """

        if maybe_prepare_mixed_metadata(batch):
            return

        reqs = batch.padded_reqs

        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_k = max(seqlens_k)
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.kvcache.device
        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS)
        cache_seqlens = cache_seqlens.to(device, non_blocking=True)
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

        if max_seqlen_q == 1:
            cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            cu_seqlens_q = cu_seqlens_k
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
            cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)

        page_table = get_global_ctx().page_table
        new_page_table = torch.stack(  # NOTE: global page table treat page_size = 1, we need slice
            [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )
        if self.page_size > 1:
            new_page_table.div_(self.page_size, rounding_mode="floor")
        batch.attn_metadata = FAMetadata(
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """为 decode CUDA Graph 准备固定大小的 FA metadata buffer。

        max_seq_len 是单请求最大 device_len；FA page_table 以 page 为单位，
        因此第二维长度是 max_seq_len // page_size。
        bs_list 是允许 capture/replay 的 batch size 列表。
        """

        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FACaptureData.create(max_bs, max_seq_len // self.page_size, self.kvcache.device)
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        """capture decode graph 前，把 batch metadata 绑定到固定 capture buffer。

        batch 必须是 decode 阶段，真实 batch size 必须在 capture_bs 中。
        这里不拷贝动态数据，只构造指向 capture buffer slice 的 FAMetadata。
        """

        if getattr(batch, "_mixed_kv_spec", None) is not None:
            raise RuntimeError(
                "Mixed KV Attention currently supports eager execution only; "
                "CUDA Graph capture needs fixed-capacity descriptor buffers"
            )
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        metadata = FAMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
        )
        batch.attn_metadata = metadata

    def prepare_for_replay(self, batch: Batch) -> None:
        """CUDA Graph replay 前，把本次 batch 的长度和 page_table 拷入固定 buffer。

        metadata.page_table shape [bs, table_len]；capture.page_table 预分配到最大长度，
        replay 前只覆盖当前请求实际需要的前 table_len 列。
        """

        metadata, bs = batch.attn_metadata, batch.padded_size
        if isinstance(metadata, MixedAttnMetadata):
            raise RuntimeError("Mixed KV Attention CUDA Graph replay is not implemented")
        assert isinstance(metadata, FAMetadata)
        assert self.capture is not None and bs in self.capture_bs
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode (i.e. no-op)
        table_len = metadata.page_table.size(1)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)


def _fa_sgl_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    version: int,
    sm_margin: int = 0,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    softcap: float = 0.0,  # 0.0 means deactivated
    num_splits: int = 0,  # Can be tuned for speed
    pack_gqa: bool | None = None,  # Can be tuned for speed
    causal: bool = True,
) -> torch.Tensor:
    """调用 sgl-kernel FlashAttention paged KV kernel。

    输入维度：
    - q shape [T_q, H_q_local, D]；
    - k_cache/v_cache shape [num_pages, page_size, H_kv_local, D]；
    - page_table shape [padded_bs, max_num_pages]，元素是 page id；
    - cache_seqlens shape [padded_bs]，每个请求当前可见 KV token 数；
    - cu_seqlens_q/cu_seqlens_k shape [padded_bs + 1]，ragged batch 前缀和。

    返回 shape [T_q, H_q_local, D]。这个函数只做 Python wrapper，不改变 KV cache。
    """

    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
    except ImportError as e:
        raise ImportError(
            "sgl_kernel.flash_attn is not found. Please install it with `pip install sgl-kernel`.\n"
            "If you're sure it's correctly installed, try `apt update && apt install libnuma1`."
        ) from e

    return flash_attn_with_kvcache(  # type: ignore
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=softmax_scale,
        sm_margin=sm_margin,
        window_size=window_size,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        causal=causal,
        ver=version,  # TODO: support FA4 on blackwell
    )
