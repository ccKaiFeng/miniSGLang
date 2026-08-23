from __future__ import annotations

# 这个文件实现 Llama 系列 causal language model。
#
# 结构是：
#   token ids -> embedding -> N 层 LlamaDecoderLayer -> final norm -> lm_head -> logits

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as LlamaMLP
from .utils import RopeAttn as LlamaAttn

if TYPE_CHECKING:
    from .config import ModelConfig


class LlamaDecoderLayer(BaseOP):
    """一层 Llama decoder block：RMSNorm + Attention + RMSNorm + MLP。"""

    def __init__(self, config: ModelConfig, layer_id: int):
        """创建第 layer_id 层 decoder。

        config.hidden_size 决定 hidden_states 最后一维；layer_id 会传入 AttentionLayer，
        用于选择 KV cache 的对应层。该层输入/输出 shape 都是 [T, hidden_size]。
        """

        self.self_attn = LlamaAttn(config, layer_id)
        self.mlp = LlamaMLP(config)
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
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行一层 decoder，并返回新的 hidden states 和 residual。

        x shape [T, hidden_size]；residual 为 None 或同 shape。
        返回的 x/residual shape 仍是 [T, hidden_size]。
        """

        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class LlamaModel(BaseOP):
    """不带 lm_head 的 Llama 主体。"""

    def __init__(self, config: ModelConfig):
        """创建 embedding、decoder layers 和 final norm。

        embed_tokens 把 input_ids shape [T] 映射到 [T, hidden_size]；
        layers 长度为 config.num_layers；norm 输出仍是 [T, hidden_size]。
        """

        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [LlamaDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """从 token id 得到最后一层 hidden states。

        input_ids shape [T]；embedding 后 x shape [T, hidden_size]；
        返回最后 norm 后的 hidden states，shape [T, hidden_size]。
        """

        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class LlamaForCausalLM(BaseLLMModel):
    """完整 Llama 因果语言模型，用于预测下一个 token。"""

    def __init__(self, config: ModelConfig):
        """创建 Llama backbone 和并行 lm_head。

        lm_head 输入 shape [T, hidden_size]，输出完整 vocab logits。若
        tie_word_embeddings=True，lm_head 复用 embedding 权重。
        """

        self.model = LlamaModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        """从全局 Context 当前 batch 中读取 input_ids，输出 logits。

        ctx.batch.input_ids shape [T]；lm_head 返回 logits shape [batch.size, vocab_size]。
        """

        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["LlamaForCausalLM"]
