from __future__ import annotations

# 这个文件实现模型层里的 attention 封装。
#
# 它不直接实现 FlashAttention/FlashInfer kernel，而是负责：
# 1. 把 qkv 合并投影结果拆成 q/k/v；
# 2. 做可选 q_norm/k_norm；
# 3. 应用 RoPE 位置编码；
# 4. 调用全局 Context 中选择好的 attention backend；
# 5. 把输出 reshape 回 hidden 维度。

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import StateLessOP
from .rotary import get_rope

if TYPE_CHECKING:
    from minisgl.layers import RMSNorm
    from minisgl.models import RotaryConfig


class AttentionLayer(StateLessOP):
    """单层 attention 的通用封装。"""

    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
    ):
        """记录 head 数、head_dim、RoPE 和可选 norm。"""

        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
        )
        self.q_norm = q_norm
        self.k_norm = k_norm

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        """执行 attention 前后的张量整理和 backend 调用。

        输入 qkv shape [T, local_qkv_dim]，T 是本轮扁平 token 数：
        - local_qkv_dim = local_q_heads*D + 2*local_kv_heads*D；
        - split 后 q shape [T, local_q_heads*D]，k/v shape [T, local_kv_heads*D]；
        - RoPE 在 q/k 的展平行向量上原地生效；随后 q view 成
          [T, local_q_heads, D] 传给 attention backend；
        - k/v 保持 [T, local_kv_heads*D]，store_cache 按整行写入 KV pool；
        - 输出 reshape 回 [T, local_q_heads*D]，再交给 o_proj 做 TP all_reduce。
        """

        ctx = get_global_ctx()
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        # q/k/v 此时仍是二维 [T, head_count * head_dim]，norm 需要临时 view 成三维。
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        # positions shape [T]，RoPE 按每个 token 的绝对位置旋转 q/k。
        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        return o.view(-1, self.qo_attn_dim)
