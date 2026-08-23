from __future__ import annotations

# 这个文件实现 Qwen2 causal language model。
#
# Qwen2 和 Llama 主体结构类似，但 attention projection 带 bias，且不启用 q/k norm。

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as Qwen2MLP
from .utils import RopeAttn as Qwen2Attn

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen2DecoderLayer(BaseOP):
    """一层 Qwen2 decoder block。"""

    def __init__(self, config: ModelConfig, layer_id: int):
        """创建第 layer_id 层 Qwen2 decoder。

        Qwen2 attention 带 qkv bias，不启用 q/k norm。输入/输出 hidden_states
        shape 都是 [T, hidden_size]，layer_id 用于访问同编号 KV cache。
        """

        self.self_attn = Qwen2Attn(config, layer_id, has_qk_norm=False, has_attn_bias=True)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行一层 Qwen2 decoder。

        x/residual shape 是 [T, hidden_size]；返回 shape 不变。
        """

        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class Qwen2Model(BaseOP):
    """不带 lm_head 的 Qwen2 主体。"""

    def __init__(self, config: ModelConfig):
        """创建 Qwen2 embedding、decoder layers 和 final RMSNorm。

        input_ids shape [T] 经 embedding 后变为 [T, hidden_size]；
        layers 数量为 config.num_layers。
        """

        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen2DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """从 token id 得到最后一层 hidden states。

        input_ids shape [T]；返回 hidden states shape [T, hidden_size]。
        """

        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class Qwen2ForCausalLM(BaseLLMModel):
    """完整 Qwen2 因果语言模型。"""

    def __init__(self, config: ModelConfig):
        """创建 Qwen2 backbone 和 vocab parallel lm_head。

        lm_head 将 [T, hidden_size] 投影到 vocab 维度；tie_word_embeddings=True
        时复用 self.model.embed_tokens 的权重。
        """

        self.model = Qwen2Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        """读取当前 batch.input_ids 并输出 logits。

        ctx.batch.input_ids shape [T]；返回 logits shape [batch.size, vocab_size]。
        """

        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Qwen2ForCausalLM"]
