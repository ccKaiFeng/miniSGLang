from abc import ABC, abstractmethod

# 这个文件定义 MoE backend 的抽象接口。
#
# MoE layer 只负责保存 expert 权重和 router logits；具体如何做 top-k、token
# 重排、expert GEMM、结果合并，由 backend 实现。

import torch


class BaseMoeBackend(ABC):
    """MoE 计算后端接口。"""

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor: ...
