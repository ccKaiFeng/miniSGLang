from __future__ import annotations

# 这个文件实现支持 Tensor Parallel(TP) 的线性层。
#
# 普通线性层是 y = x @ W^T + b。TP 场景下，W 会按输入维或输出维切到多张 GPU：
# - column parallel：每个 rank 负责一部分输出通道；
# - row parallel：每个 rank 负责一部分输入通道，最后 all_reduce 合并；
# - replicated：每个 rank 都保存完整权重。

from typing import List

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_even

from .base import BaseOP


class _LinearTPImpl(BaseOP):
    """TP 线性层的基础实现，保存本 rank 的局部权重。"""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ):
        """记录完整尺寸和本 rank 局部尺寸，并创建占位权重。"""

        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行本 rank 上的线性计算。

        x shape [T, local_input_size] 或 [..., local_input_size]；
        返回 shape [T, local_output_size] 或 [..., local_output_size]。
        """

        return F.linear(x, self.weight, self.bias)


class LinearReplicated(_LinearTPImpl):
    """权重不切分的线性层，每个 TP rank 都保存完整矩阵。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        """创建 replicated linear。

        input_size/output_size 是完整模型维度；每个 TP rank 都保存完整
        weight shape [output_size, input_size]，输出也不需要通信合并。
        """

        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
        )


class LinearColParallelMerged(_LinearTPImpl):
    """按输出维切分的 merged 线性层，常用于 MLP 的 gate/up 合并投影。

    输入 shape [T, hidden_size]；输出 shape [T, sum(output_sizes)/tp_size]。
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
    ):
        """创建 column-parallel merged linear。

        input_size 是完整 hidden_size；output_sizes 是多个逻辑输出投影的完整尺寸，
        例如 gate_proj/up_proj。每个 TP rank 保存每个输出投影的 1/tp_size 切片，
        本地 weight shape [sum(output_sizes)/tp_size, input_size]。
        """

        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)


class LinearQKVMerged(_LinearTPImpl):
    """Attention 的 Q/K/V 合并投影层。

    输入 shape [T, hidden_size]；输出 shape
    [T, (local_q_heads + 2*local_kv_heads) * head_dim]。
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        """创建合并 QKV 投影。

        hidden_size 是输入通道数；num_qo_heads/num_kv_heads 是全局 head 数。
        TP 后每个 rank 输出 local_q_heads 个 Q 和 local_kv_heads 个 K/V：
        local output size = (local_q_heads + 2*local_kv_heads) * head_dim。
        """

        tp_info = get_tp_info()

        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        full_isize = hidden_size
        full_osize = (num_qo_heads + 2 * num_kv_heads) * head_dim
        local_isize = hidden_size
        local_osize = (local_num_qo + 2 * local_num_kv) * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)


class LinearOProj(_LinearTPImpl):
    """Attention 输出投影层，输入维按 TP 切分，输出需要 all_reduce。

    输入 shape [T, hidden_size/tp_size]；all_reduce 后输出 shape [T, hidden_size]。
    """

    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        """创建 attention output projection。

        input_size/output_size 是全局 hidden size。每个 TP rank 只接收
        input_size/tp_size 个输入通道，局部线性输出 shape [..., output_size]，
        多卡时再 all_reduce 求和得到完整输出。
        """

        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = div_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """本 rank 先做局部线性层，多卡时再 all_reduce 合并。"""

        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    """按输入维切分的线性层，输出需要 all_reduce。

    输入 shape [T, input_size/tp_size]；all_reduce 后输出 shape [T, output_size]。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        """创建 row-parallel linear。

        input_size 是全局输入维度；每个 TP rank 保存 local_input_size=input_size/tp_size。
        局部输出 shape [..., output_size]，多卡时所有 rank 的输出相加。
        """

        tp_info = get_tp_info()
        local_input_size = div_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """本 rank 计算局部输入贡献，多卡时所有 rank 求和。"""

        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
