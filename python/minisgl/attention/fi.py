from __future__ import annotations

# 这个文件实现 FlashInfer attention backend。
#
# FlashInfer 对 paged KV cache 和 decode 阶段有专门优化，本 backend 负责准备
# wrapper、indices、page table、seq lens 等元数据。

import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Literal

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.env import ENV
from minisgl.utils import div_even, init_logger

from .base import BaseAttnBackend, BaseAttnMetadata
from .mixed import MixedAttnMetadata, maybe_prepare_mixed_metadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    )
    from minisgl.models import ModelConfig


def _next_power_of_2(n: int) -> int:
    """返回大于等于 n 的最小 2 的幂。"""

    if n <= 1:
        return 1
    return 1 << math.ceil(math.log2(n))


logger = init_logger(__name__)


@dataclass
class FICaptureData(BaseCaptureData):
    """FlashInfer CUDA graph capture 使用的固定 buffer。"""

    @property
    def one_tensor(self) -> torch.Tensor:
        """返回 shape [max_bs] 的全 1 buffer。

        FlashInfer page_size 固定为 1，因此 last_page_len 对每个请求恒为 1。
        capture 时复用 seq_lens 这块 int32 buffer 的前 bs 个元素表达该信息。
        """

        return self.seq_lens

    @property
    def indices(self) -> torch.Tensor:
        """返回 capture 用的一维 paged_kv_indices buffer。

        FlashInfer page_size=1 时，page id 等价于 normal KV pool 的物理 token index。
        """

        return self.page_table


@dataclass
class FIMetadata(BaseAttnMetadata):
    """FlashInfer 每个 batch 的元数据。"""

    # fmt: off
    cu_seqlens_q_cpu:   torch.Tensor  # CPU int32, shape [padded_bs + 1], query ragged 前缀和。
    cu_seqlens_k_cpu:   torch.Tensor  # CPU int32, shape [padded_bs + 1], KV/page ragged 前缀和。
    cu_seqlens_q_gpu:   torch.Tensor  # GPU int32, shape [padded_bs + 1]，用于 get_last_indices。
    indices:            torch.Tensor  # GPU int32, shape [sum(req.device_len)]，paged KV indices。
    last_page_len_cpu:  torch.Tensor  # CPU int32, shape [padded_bs]；page_size=1 时全为 1。
    num_qo_heads:       int
    num_kv_heads:       int
    head_dim:           int
    page_size:          Literal[1] # currently only support page_size=1
    pos_encoding_mode:  str
    seq_lens_cpu:       torch.Tensor  # CPU int32, shape [padded_bs]，每个请求 req.device_len。
    dtype:              torch.dtype
    wrapper:            BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper
    initialized:        bool = False
    # fmt: on

    def __post_init__(self) -> None:
        """校验 FlashInfer metadata 的 device placement。

        FlashInfer plan() 的 indptr/last_page_len/seq_lens 当前从 pinned CPU buffer
        异步拷贝；indices 和 cu_seqlens_q_gpu 则已经在 GPU 上。
        """

        assert self.page_size == 1, "Currently only page_size=1 is supported."
        assert (
            self.cu_seqlens_k_cpu.is_cpu
            and self.cu_seqlens_q_cpu.is_cpu
            and self.cu_seqlens_q_gpu.is_cuda
            and self.indices.is_cuda
            and self.last_page_len_cpu.is_cpu
            and self.seq_lens_cpu.is_cpu
        )

    def get_last_indices(self, bs: int) -> torch.Tensor:
        """返回每个真实请求最后一个 query 在扁平 q/output tensor 中的位置。

        cu_seqlens_q_gpu shape [padded_bs + 1]；取 [1:1+bs]-1 后得到 shape [bs]。
        """

        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class FlashInferBackend(BaseAttnBackend):
    """基于 FlashInfer 的 paged KV attention backend。"""

    def __init__(self, config: ModelConfig) -> None:
        """初始化 FlashInfer wrapper 和 workspace。

        config 提供 head 数/head_dim；KV cache 从 global context 获取。FlashInfer 需要
        float_workspace_buffer 和 int_workspace_buffer，prefill/decode wrapper 共享
        同一块 int workspace，避免重复分配。
        """

        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead
        )
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            use_tensor_cores=self.use_tensor_cores,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead
        )

        # NOTE: some hack to reuse the int_workspace_buffer
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer

        # initialize some data members
        tp_size = get_tp_info().size
        self.qo_head_local = div_even(self.config.num_qo_heads, tp_size)
        self.kv_head_local = div_even(self.config.num_kv_heads, tp_size, allow_replicate=True)

        self.cached_ones_cpu: torch.Tensor = torch.tensor([], dtype=torch.int32, pin_memory=True)
        # for cuda graph
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.graph_wrappers: Dict[int, CUDAGraphBatchDecodeWithPagedKVCacheWrapper] = {}
        self.capture: FICaptureData | None = None
        self.last_event = torch.cuda.Event()
        self.last_event.record()

    def _initialize_metadata_once(self, metadata: FIMetadata) -> None:
        """对一个 FIMetadata 执行一次 FlashInfer plan()。

        plan() 会根据 indptr、indices、head 数和 dtype 准备内部 kernel 元数据。
        initialized=True 后同一个 metadata 不再重复 plan。由于 FlashInfer 会异步
        读取 pinned CPU staging buffer，这里先等待上一次 plan 的 event，避免 host
        buffer 被下一次 plan 提前覆盖。
        """

        if metadata.initialized:
            return

        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        metadata.initialized = True
        # FlashInfer planning reuses a pinned host staging buffer and launches an
        # async H2D copy. Wait here before the next plan mutates that host buffer.
        self.last_event.synchronize()
        if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.dtype,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
            )
        else:
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim_qk=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
                causal=True,
            )
        self.last_event.record()

    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        """返回 pinned CPU 上长度为 bs 的 int32 全 1 tensor。

        用于 last_page_len_cpu。内部按 2 的幂扩容缓存，减少频繁分配 pinned memory。
        """

        if bs <= len(self.cached_ones_cpu):
            return self.cached_ones_cpu[:bs]
        # padding to next pow of 2
        next_len = _next_power_of_2(bs)
        self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, pin_memory=True)
        return self.cached_ones_cpu[:bs]

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """写入本层新 K/V，然后调用 FlashInfer wrapper.run()。

        输入：
        - q shape [T_q, H_q_local, D]；
        - k/v shape [T_new, H_kv_local, D] 或等价展平形式；
        - batch.out_loc shape [T_new]，指出每个新 token 写入 normal KV pool 的物理 index。

        输出 shape [T_q, H_q_local, D]。FlashInfer 从 kv_cache 的 paged layout 中按
        metadata.indices/page indptr 读取历史 KV。
        """

        def _flatten_cache(cache: torch.Tensor) -> torch.Tensor:  # treat page = 1
            """把 MHAKVCache 单层 cache 转成 FlashInfer paged KV layout。"""

            # MHAKVCache layer cache 原 shape [P, 1, H_kv_local, D]。
            # FlashInfer 这里要求 paged KV shape [num_pages, page_size, H_kv_local, D]。
            return cache.view(-1, 1, cache.shape[2], cache.shape[3])

        metadata = batch.attn_metadata
        if isinstance(metadata, MixedAttnMetadata):
            # 本步新产生的 K/V 仍属于 normal pool。mixed kernel 会在同一个
            # CUDA stream 上随后读取它们和 compressed pool，不会执行完整 restore。
            from minisgl.kernel import mixed_paged_attention

            self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
            return mixed_paged_attention(
                q=q,
                normal_k=self.kvcache.k_cache(layer_id),
                normal_v=self.kvcache.v_cache(layer_id),
                metadata=metadata,
                layer_id=layer_id,
            )
        assert isinstance(metadata, FIMetadata)
        self._initialize_metadata_once(metadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
        kv_cache = (_flatten_cache(kv_cache[0]), _flatten_cache(kv_cache[1]))
        return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)

    def prepare_metadata(self, batch: Batch) -> None:
        """把 Batch 转成 FlashInfer wrapper.plan() 需要的 metadata。

        当前 FlashInfer backend 只支持 page_size=1，所以 page_table 中的 raw token index
        可以直接作为 paged_kv_indices。indices shape 是所有 req.device_len 拼接后的
        [sum_k]，其中 sum_k=sum(req.device_len for req in padded_reqs)。
        """

        if maybe_prepare_mixed_metadata(batch):
            return

        reqs = batch.padded_reqs

        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.device
        seq_len_cpu = torch.tensor(seqlens_k, **CPU_KWARGS)
        cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        if max_seqlen_q == 1:  # decode with all extend_len = 1
            cu_seqlens_q_cpu = torch.arange(0, padded_size + 1, **CPU_KWARGS)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            cu_seqlens_q_cpu = cu_seqlens_k_cpu
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)

        page_table = get_global_ctx().page_table
        batch.attn_metadata = FIMetadata(
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            cu_seqlens_q_gpu=cu_seqlens_q_cpu.to(device, non_blocking=True),
            indices=torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs]),
            last_page_len_cpu=self._get_ones_cpu(padded_size),
            num_qo_heads=self.qo_head_local,
            num_kv_heads=self.kv_head_local,
            head_dim=self.config.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens_cpu=seq_len_cpu,
            dtype=self.kvcache.dtype,
            wrapper=self.decode_wrappers if batch.is_decode else self.prefill_wrapper,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """为 FlashInfer decode CUDA Graph 预分配固定 buffer。

        capture.page_table 初始是二维 [max_bs, max_seq_len]，FlashInfer 的
        paged_kv_indices 是 ragged 一维数组，因此这里 view 成一维连续 buffer。
        """

        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FICaptureData.create(max_bs, max_seq_len, self.kvcache.device)
        capture.page_table = capture.page_table.view(-1)  # use 1D as ragged indices
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    @cached_property
    def use_tensor_cores(self) -> bool:
        """判断 FlashInfer decode 是否启用 tensor core 路径。

        环境变量 FLASHINFER_USE_TENSOR_CORES 可以强制覆盖；否则按 GQA ratio 判断，
        num_qo_heads / num_kv_heads >= 4 时启用。
        """

        if (overriden_value := ENV.FLASHINFER_USE_TENSOR_CORES.value) is not None:
            logger.warning(f"Overriding FlashInfer tensor core usage to {overriden_value}")
            return overriden_value
        GQA = self.config.num_qo_heads // self.config.num_kv_heads
        return GQA >= 4

    def prepare_for_capture(self, batch: Batch) -> None:
        """capture decode graph 前创建对应 batch size 的 graph wrapper。

        wrapper 直接绑定 capture buffer：indptr_buffer shape [bs+1]，
        indices_buffer 是一维最大容量 buffer，last_page_len_buffer shape [bs]。
        随后调用 prepare_metadata() 并将 metadata.wrapper 替换成 graph wrapper。
        """

        if getattr(batch, "_mixed_kv_spec", None) is not None:
            raise RuntimeError(
                "Mixed KV Attention currently supports eager execution only; "
                "CUDA Graph capture needs fixed-capacity descriptor buffers"
            )

        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        bs = batch.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture
        capture = self.capture
        self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self.use_tensor_cores,
            indptr_buffer=capture.cu_seqlens_k[: bs + 1],
            indices_buffer=capture.indices,
            last_page_len_buffer=capture.one_tensor[:bs],
        )
        self.graph_wrappers[bs]._backend = "fa2"
        self.graph_wrappers[bs]._int_workspace_buffer = self.int_workspace_buffer
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)

    def prepare_for_replay(self, batch: Batch) -> None:
        """CUDA Graph replay 前复用已创建的 graph wrapper 并重新 plan metadata。

        batch.attn_metadata 已由 prepare_metadata() 生成，其中 indices/indptr 描述本次
        batch。这里把 wrapper 换成 capture 时创建的固定 wrapper，再执行 plan。
        """

        metadata, bs = batch.attn_metadata, batch.padded_size
        if isinstance(metadata, MixedAttnMetadata):
            raise RuntimeError("Mixed KV Attention CUDA Graph replay is not implemented")
        assert isinstance(metadata, FIMetadata) and not metadata.initialized
        assert self.capture is not None and bs in self.capture_bs
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)
