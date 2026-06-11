from typing import Tuple

# 这个文件封装 RMSNorm。
#
# RMSNorm 是 Llama/Qwen 等模型常用的归一化层。RMSNormFused 还把 residual add
# 和 norm 融合到一个 kernel 里，减少访存。

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    """普通 RMSNorm。"""

    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回归一化后的新 tensor。"""

        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        """原地归一化 x，常用于 q/k norm。"""

        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    """融合 residual add 的 RMSNorm。"""

    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行 RMSNorm。

        如果 residual 为空，返回 norm(x) 和新的 residual=x。
        如果 residual 已存在，则先把 x 加到 residual，再对结果做 norm。
        """

        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
