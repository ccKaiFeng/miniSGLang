from __future__ import annotations

# 这个文件放多个模型共享的 block。
#
# Llama/Qwen/Mistral 的整体结构很接近：Embedding -> 多层 DecoderLayer ->
# Norm -> LMHead。DecoderLayer 内部通常包含 Attention 和 MLP。这里把共享的
# MLP、MoE MLP、RoPE Attention 抽出来，减少各模型文件重复代码。

from typing import TYPE_CHECKING

from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearReplicated,
    LinearRowParallel,
    MoELayer,
    RMSNorm,
    gelu_and_mul,
    silu_and_mul,
)
from minisgl.models import ModelConfig
from minisgl.utils import nvtx_annotate

if TYPE_CHECKING:
    import torch


class GatedMLP(BaseOP):
    """dense 模型使用的 gated MLP。"""

    def __init__(self, config: ModelConfig):
        """创建 gate/up 合并投影、激活函数和 down projection。"""

        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
        )

        FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}
        act_fn = FN_MAP.get(config.hidden_act, None)
        if act_fn is None:
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")
        self.act_fn = act_fn
        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 MLP：gate/up 投影 -> fused activation -> down projection。

        x shape [T, hidden_size]；gate_up shape [T, 2*intermediate_size/tp_size]；
        fused activation 后 shape [T, intermediate_size/tp_size]；
        down_proj all_reduce 后返回 [T, hidden_size]。
        """

        gate_up = self.gate_up_proj.forward(x)
        del x
        y = self.act_fn(gate_up)
        del gate_up
        return self.down_proj.forward(y)


class MoEMLP(BaseOP):
    """MoE 模型使用的 MLP。"""

    def __init__(self, config: ModelConfig):
        """创建 expert 层和 router/gate 线性层。"""

        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
        )
        self.gate = LinearReplicated(
            config.hidden_size,
            config.num_experts,
            has_bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """先用 gate 算每个 token 应该去哪些 expert，再执行 expert 计算。

        hidden_states shape [T, hidden_size]；router_logits shape [T, num_experts]；
        返回 shape [T, hidden_size]。
        """

        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        final_hidden_states = self.experts.forward(
            hidden_states=hidden_states, router_logits=router_logits
        )
        final_hidden_states = final_hidden_states.view(num_tokens, hidden_dim)
        return final_hidden_states


class RopeAttn(BaseOP):
    """带 RoPE 的 attention block。"""

    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        has_attn_bias: bool = False,
        has_qk_norm: bool = False,
    ):
        """创建一个带 RoPE 的 attention block。

        参数：
        - config：模型结构配置，提供 hidden_size、head_dim、head 数和 RoPE 参数；
        - layer_id：当前层号，AttentionLayer 用它选择 KV cache[layer_id]；
        - has_attn_bias：QKV merged projection 是否带 bias，Qwen2 为 True；
        - has_qk_norm：是否对 q/k 按 head_dim 做 RMSNorm，Qwen3 为 True。

        维度：
        - 输入 x shape [T, hidden_size]；
        - qkv_proj 输出 [T, (local_q_heads + 2*local_kv_heads) * head_dim]；
        - attention 输出 [T, local_q_heads * head_dim]；
        - o_proj 输出 [T, hidden_size]。
        """

        head_dim = config.head_dim
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=has_attn_bias,
        )
        self.has_qk_norm = has_qk_norm
        if has_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        self.o_proj = LinearOProj(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 QKV projection、attention backend 和输出投影。

        x shape [T, hidden_size]；
        qkv shape [T, (local_q_heads + 2*local_kv_heads)*head_dim]；
        attention 输出 shape [T, local_q_heads*head_dim]；
        o_proj all_reduce 后返回 [T, hidden_size]。
        """

        qkv = self.qkv_proj.forward(x)
        del x
        o = self.attn.forward(qkv)
        return self.o_proj.forward(o)


__all__ = ["GatedMLP", "RopeAttn", "MoEMLP"]
