from __future__ import annotations

# 这个文件封装 fused activation。
#
# LLM 的 MLP 常见结构是 gate_proj 和 up_proj 两路输出，然后做：
#   silu(gate) * up
# 或：
#   gelu(gate) * up
#
# 这里复用 flashinfer 的 fused 实现，一次完成激活和乘法，减少中间 tensor。

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    """执行 SiLU(x 前半) * (x 后半) 的 fused 计算。"""

    from flashinfer import silu_and_mul

    return silu_and_mul(x, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    """执行 GELU(x 前半) * (x 后半) 的 fused 计算。"""

    from flashinfer import gelu_and_mul

    return gelu_and_mul(x, out=out)


__all__ = ["silu_and_mul", "gelu_and_mul"]
