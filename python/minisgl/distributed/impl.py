from __future__ import annotations

# 这个文件封装 Tensor Parallel 进程之间的通信。
#
# 模型并行时，某些层会把计算结果分散在多个 GPU/rank 上。为了得到正确结果，
# 需要做 all_reduce、all_gather 这类 collective 通信：
# - all_reduce：所有 rank 的 tensor 相加，并让每个 rank 都拿到相加结果；
# - all_gather：把每个 rank 的 tensor 拼起来，让每个 rank 都拿到完整 tensor。
#
# 本文件提供两套实现：
# - TorchDistributedImpl：使用 PyTorch 自带 torch.distributed；
# - PyNCCLDistributedImpl：使用项目自定义 PyNCCL wrapper，减少部分通信开销。

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from minisgl.distributed import DistributedInfo
    from minisgl.kernel import PyNCCLCommunicator


@dataclass
class DistributedImpl(ABC):
    """分布式通信后端的抽象接口。"""

    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    """基于 torch.distributed 的默认通信实现。"""

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """对所有 TP rank 的 x 求和，结果原地写回 x。"""

        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """收集所有 TP rank 的 x，并沿第 0 维拼接。"""

        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    """基于自定义 PyNCCL communicator 的通信实现。"""

    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """通过 NCCL 对所有 rank 的 x 求和。"""

        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """通过 NCCL 收集所有 rank 的 x，并沿第 0 维拼接。"""

        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result


class DistributedCommunicator:
    """统一的通信入口。

    代码里调用 `DistributedCommunicator().all_reduce(x)` 时，不需要关心底层
    现在用的是 torch.distributed 还是 PyNCCL。plugins 列表最后一个元素是
    当前生效的实现。
    """

    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)


def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """启用 PyNCCL 通信后端。

    单卡时不需要 NCCL 通信，直接返回。多卡时初始化 PyNCCL communicator，
    并把它追加到 plugins 末尾，让后续 all_reduce/all_gather 走 PyNCCL。
    """

    if tp_info.size == 1:
        return
    from minisgl.kernel import init_pynccl

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))


def destroy_distributed() -> None:
    """销毁通信插件列表。

    进程退出或 scheduler shutdown 时调用，避免保留旧 communicator。
    """

    DistributedCommunicator.plugins = []
