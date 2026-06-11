from __future__ import annotations

# 这个文件是 csrc/src/tensor.cpp 的测试 wrapper。
#
# 它主要用于验证 TVM FFI tensor 传参是否正常，不属于主推理链路。

import functools
from typing import TYPE_CHECKING

from .utils import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@functools.cache
def _load_test_tensor_module() -> Module:
    """加载 tensor 测试扩展。"""

    return load_aot("test_tensor", cpp_files=["tensor.cpp"])


def test_tensor(x: torch.Tensor, y: torch.Tensor) -> int:
    """调用 C++ 测试函数，检查 tensor binding。"""

    return _load_test_tensor_module().test(x, y)
