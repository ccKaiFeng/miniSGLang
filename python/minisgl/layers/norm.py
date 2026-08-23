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
        """创建 RMSNorm 参数。

        size 是最后一维通道数，例如 hidden_size 或 head_dim；weight shape [size]。
        eps 是防止均方根分母为 0 的小常数。
        """

        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回归一化后的新 tensor。

        x 最后一维必须等于 size；常见 shape 是 [T, hidden_size] 或 [T, heads, head_dim]。
        """

        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        """原地归一化 x，常用于 q/k norm。

        x shape 通常是 [T, local_heads, head_dim]，最后一维等于初始化 size。
        """

        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    """融合 residual add 的 RMSNorm。"""

    def __init__(self, size: int, eps: float) -> None:
        """创建 fused add + RMSNorm 参数。

        size 是 hidden_states 最后一维；weight shape [size]。fused kernel 会把
        residual += x 和 RMSNorm 合并，常用于 transformer block 的残差路径。
        """

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

        x/residual shape 通常都是 [T, hidden_size]；返回的两个 tensor shape 不变。
        """

        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
