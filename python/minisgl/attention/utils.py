from dataclasses import dataclass

# 这个文件保存 attention backend 在 CUDA graph capture/replay 时复用的数据结构。
#
# CUDA graph 要求 replay 时很多 tensor 地址稳定，所以这里提前创建固定形状的
# seq_lens、positions、page_table 等 buffer。

import torch


@dataclass
class BaseCaptureData:
    """CUDA graph capture/replay 期间复用的 attention 元数据 buffer。"""

    seq_lens: torch.Tensor
    positions: torch.Tensor
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    page_table: torch.Tensor

    @classmethod
    def create(cls, max_bs: int, max_seq_len: int, device: torch.device, **kwargs):
        """按最大 batch size 和最大序列长度创建 capture buffer。"""

        return cls(
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            positions=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),
            **kwargs,
        )
