from __future__ import annotations

# 这个文件实现 TensorRT-LLM FMHA backend。
#
# 它使用 flashinfer 暴露的 TensorRT-LLM attention 接口，并为 prefill/decode
# 准备所需 metadata 和 workspace buffer。

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.core import Batch, get_global_ctx

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class TRTLLMCaptureData(BaseCaptureData):
    """TensorRT-LLM backend 的 CUDA graph capture buffer。"""

    pass


@dataclass
class TRTLLMMetadata(BaseAttnMetadata):
    """TensorRT-LLM backend 每个 batch 的元数据。"""

    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    max_seqlen_k: int
    max_seqlen_q: int

    page_table: torch.Tensor

    def get_last_indices(self, bs: int) -> torch.Tensor:
        """返回每个真实请求最后一个 query token 的扁平位置。

        cu_seqlens_q shape [padded_bs + 1]；取 [1:1+bs]-1 得到 shape [bs]。
        """

        return self.cu_seqlens_q[1 : 1 + bs] - 1


class TensorRTLLMBackend(BaseAttnBackend):
    """基于 TensorRT-LLM FMHA 的 attention backend。"""

    def __init__(self, config: ModelConfig):
        """初始化 TensorRT-LLM attention backend。

        config 提供 head_dim 等静态模型参数；KV cache/page_size 从 global context 获取。
        workspace_buffer 是 TensorRT-LLM FMHA kernel 的临时工作区，dtype uint8。
        """

        ctx = get_global_ctx()
        self.config = config
        self.kvcache = ctx.kv_cache
        self.page_size = ctx.page_size
        self.capture: TRTLLMCaptureData | None = None
        self.max_graph_bs = 0
        self.capture_bs: List[int] = []
        self.scale = config.head_dim**-0.5
        self.workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.kvcache.device
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """写入本层新 KV，并按 prefill/decode 调用 TensorRT-LLM kernel。

        q shape [T_q, H_q_local, D]；k/v shape [T_new, H_kv_local, D] 或等价展平；
        batch.out_loc shape [T_new]，指向 normal KV pool 的物理 token index。
        返回 shape [T_q, H_q_local, D]。
        """

        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
        from flashinfer.prefill import trtllm_batch_context_with_kv_cache

        metadata = batch.attn_metadata
        assert isinstance(metadata, TRTLLMMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))

        if batch.is_prefill:
            return trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                block_tables=metadata.page_table,
                seq_lens=metadata.cache_seqlens,
                max_q_len=metadata.max_seqlen_q,
                max_kv_len=metadata.max_seqlen_k,
                bmm1_scale=self.scale,
                bmm2_scale=1.0,
                cum_seq_lens_q=metadata.cu_seqlens_q,
                cum_seq_lens_kv=metadata.cu_seqlens_k,
                kv_layout="NHD",
                batch_size=batch.size,
                out_dtype=q.dtype,
            )
        else:
            return trtllm_batch_decode_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                block_tables=metadata.page_table,
                seq_lens=metadata.cache_seqlens,
                max_seq_len=metadata.max_seqlen_k,
                bmm1_scale=self.scale,
                bmm2_scale=1.0,
                kv_layout="NHD",
                out_dtype=q.dtype,
            )

    def prepare_metadata(self, batch: Batch) -> None:
        """把 Batch 转成 TensorRT-LLM FMHA metadata。

        seqlens_q[i]=req.extend_len，seqlens_k[i]=req.device_len。
        page_table shape [padded_bs, ceil(max_seqlen_k/page_size)]，page_size>1 时
        从全局物理 token index 转为 page id。
        """

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
        batch.attn_metadata = TRTLLMMetadata(
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """为 decode CUDA Graph 预分配固定 metadata buffer。

        max_seq_len 是最大可见上下文 token 数；TRT-LLM block table 以 page 为单位，
        因此 capture.page_table 的列数是 max_seq_len // page_size。
        """

        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = TRTLLMCaptureData.create(
            max_bs, max_seq_len // self.page_size, self.kvcache.device
        )
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        """capture decode graph 前，让 batch metadata 指向固定 capture buffer。"""

        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        metadata = TRTLLMMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
        )
        batch.attn_metadata = metadata

    def prepare_for_replay(self, batch: Batch) -> None:
        """replay 前拷贝本次 batch 的动态长度和 page_table。

        只覆盖当前 table_len 列，capture buffer 剩余列保持预分配内容。
        """

        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, TRTLLMMetadata)
        assert self.capture is not None and bs in self.capture_bs
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode (i.e. no-op)
        table_len = metadata.page_table.size(1)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)
