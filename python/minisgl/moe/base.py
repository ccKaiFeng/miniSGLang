from abc import ABC, abstractmethod

# 这个文件定义 MoE backend 的抽象接口。
#
# MoE layer 只负责保存 expert 权重和 router logits；具体如何做 top-k、token
# 重排、expert GEMM、结果合并，由 backend 实现。

import torch


class BaseMoeBackend(ABC):
    """MoE 计算后端接口。

    维度约定：
    - hidden_states shape [T, hidden_size]；
    - w1 shape [E, 2*I_local, hidden_size]，E 是 expert 数，I_local=intermediate/tp_size；
    - w2 shape [E, hidden_size, I_local]；
    - gating_output shape [T, E]；
    - 返回 shape [T, hidden_size]。
    """

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
    ) -> torch.Tensor:
        """执行一次 MoE layer 计算。

        输入：
        - hidden_states shape [T, hidden_size]；
        - w1/w2 是 expert 权重，E 维是 expert id；
        - gating_output shape [T, E]，每个 token 对每个 expert 的 router logits；
        - topk 表示每个 token 选择多少个 expert；
        - renormalize=True 时 top-k 权重会重新归一化；
        - activation 是 expert FFN 激活函数名；
        - apply_router_weight_on_input 控制 router weight 乘在第一段还是第二段 GEMM。

        返回 shape [T, hidden_size]。
        """
        ...
