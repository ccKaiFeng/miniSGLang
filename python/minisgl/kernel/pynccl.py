from __future__ import annotations

# 这个文件是自定义 PyNCCL 扩展的 Python wrapper。
#
# 它通过 TVM FFI 加载 csrc/src/pynccl.cu，并创建 NCCL communicator。上层
# distributed/impl.py 会用它实现 all_reduce/all_gather。

import functools
from typing import TYPE_CHECKING, Any, Literal

from minisgl.env import ENV

from .utils import load_aot

if TYPE_CHECKING:
    from abc import abstractmethod

    import torch
    from tvm_ffi import Module

    class PyNCCLCommunicator:
        @abstractmethod
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...
        @abstractmethod
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...
        @abstractmethod
        def get_buffer(self) -> int: ...

else:
    PyNCCLCommunicator = Any


@functools.cache
def _load_nccl_module() -> Module:
    """加载 PyNCCL CUDA 扩展。"""

    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=["-lnccl"])


@functools.cache
def _get_pynccl_wrapper_cls():
    """注册并返回 TVM FFI 的 NCCLWrapper Python 类。"""

    import tvm_ffi

    @tvm_ffi.register_object("minisgl.NCCLWrapper")
    class PyNCCLImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return PyNCCLImpl


def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator:
    """初始化 NCCL communicator。

    rank 0 创建 NCCL unique id，然后通过 torch.distributed 广播给其他 rank。
    所有 rank 拿到同一个 id 后才能加入同一个 NCCL 通信域。
    """

    import torch

    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()
    cls = _get_pynccl_wrapper_cls()

    if tp_rank == 0:
        id_list = [module.create_nccl_uid()]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )
    else:
        id_list = [None]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )

    nccl_id = id_list[0]
    assert not nccl_id is None, f"Failed to get NCCL unique ID on {tp_rank = }"

    # bypass type checking for the FFI object
    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)  # type: ignore
