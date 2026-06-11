from __future__ import annotations

# 这个文件判断当前 CUDA/GPU 架构是否满足某些优化路径要求。

import functools
from typing import Tuple


@functools.cache
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    """读取当前 GPU compute capability，例如 H100 是 (9, 0)。"""

    import torch
    import torch.version

    if not torch.cuda.is_available() or not torch.version.cuda:
        return None
    return torch.cuda.get_device_capability()


def is_arch_supported(major: int, minor: int = 0) -> bool:
    """判断当前 GPU 架构是否大于等于指定 SM 版本。"""

    arch = _get_torch_cuda_version()
    if arch is None:
        return False
    return arch >= (major, minor)


def is_sm90_supported() -> bool:
    """是否支持 SM90，例如 NVIDIA Hopper。"""

    return is_arch_supported(9, 0)


def is_sm100_supported() -> bool:
    """是否支持 SM100。"""

    return is_arch_supported(10, 0)
