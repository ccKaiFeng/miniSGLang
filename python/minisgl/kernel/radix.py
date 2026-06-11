from __future__ import annotations

# 这个文件是 csrc/src/radix.cpp 的 Python wrapper。
#
# radix cache 在 CPU 上比较 token 前缀时，需要快速找出两个 int tensor 第一个
# 不同的位置。fast_compare_key() 就是这个 C++ 实现的 Python 入口。

import functools
from typing import TYPE_CHECKING

from .utils import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@functools.cache
def _load_radix_module() -> Module:
    """加载 radix C++ 扩展。"""

    return load_aot("radix", cpp_files=["radix.cpp"])


def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    """比较两个 1D CPU int tensor，返回共同前缀长度。"""

    # compare 2 1-D int cpu tensors for equality
    return _load_radix_module().fast_compare_key(x, y)
