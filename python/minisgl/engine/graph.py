from __future__ import annotations

# 这个文件实现 CUDA Graph capture/replay。
#
# decode 阶段每轮计算很小，但会频繁调用很多 CUDA kernel。CPU 每次 launch kernel
# 都有开销。CUDA Graph 可以把一组固定形状的 CUDA 操作预先 capture 成图，
# 运行时直接 replay，从而减少 CPU launch overhead。
#
# 本文件只对 decode batch 使用 CUDA Graph；prefill 长度变化大，一般不使用。

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import init_logger
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
    from minisgl.models import BaseLLMModel

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    """CUDA Graph capture/replay 使用的固定输入输出 buffer。

    CUDA Graph 要求 replay 时 tensor 地址不变，所以不能每次都换新的 input_ids。
    解决办法是预先分配固定 buffer，每次把 batch 数据 copy 到 buffer 里。
    """

    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        """为最大 graph batch size 分配 buffer。"""

        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        """让 batch 的输入字段指向 capture buffer。"""

        _slice = slice(batch.padded_size)
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]

    def copy_from(self, batch: Batch) -> None:
        """replay 前把真实 batch 数据拷贝进固定 buffer。"""

        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    """决定要 capture 哪些 batch size 的 CUDA graph。"""

    if cuda_graph_bs is not None:
        # 用户显式指定时直接使用。
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    # 小 batch 1/2/4 单独 capture；8 之后每 8 一个档位。
    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))


def mem_GB(size: int) -> str:
    """把字节数格式化成 GiB 字符串。"""

    return f"{size / (1024**3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    """读取当前 GPU 空闲显存。"""

    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    """管理 CUDA Graph 的 capture、padding 和 replay。"""

    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
    ) -> None:
        """初始化 GraphRunner 并立即 capture 所需 graph。"""

        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        """capture 多个 batch size 对应的 decode CUDA graph。"""

        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        # attention backend 也需要为 capture 准备固定 metadata buffer。
        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        # 最大 bs 的共享 buffer，较小 bs 使用前缀 slice。
        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()

            # 用 dummy_req 构造固定形状 decode batch。
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            with get_global_ctx().forward_batch(batch):
                # capture 前先跑一次 warmup。
                self.buffer.logits[:bs] = model.forward()
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    # 这里的 forward 会被 capture 成 CUDA graph。
                    self.buffer.logits[:bs] = model.forward()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph

        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        """判断当前 batch 是否能使用 CUDA Graph。"""

        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        """用已 capture 的 graph 执行一次 decode forward。"""

        assert self.can_use_cuda_graph(batch)

        # 把真实输入拷贝到 capture 时固定的 buffer 地址。
        self.buffer.copy_from(batch)

        # padded_size 是实际 replay 使用的 graph batch size。
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()

        # 返回真实请求对应的 logits，不返回 padding dummy 请求的 logits。
        return self.buffer.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        """把 decode batch 补齐到已 capture 的某个 batch size。"""

        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        # 用 dummy_req 补齐，保证 graph replay 的 batch shape 固定。
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        """释放 CUDA Graph 对象。

        需要在释放 NCCL 等资源前调用，否则某些环境下可能因为 graph 内部持有资源
        导致退出 hang 住。
        """

        del self.graph_map
        gc.collect()
