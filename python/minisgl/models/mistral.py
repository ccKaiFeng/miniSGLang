from __future__ import annotations

# 这个文件实现 Mistral causal language model。
#
# 结构与 Llama 类似，主要差异来自 HuggingFace config 和 RoPE 参数。

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as MistralMLP
from .utils import RopeAttn as MistralAttn

if TYPE_CHECKING:
    from .config import ModelConfig


class MistralDecoderLayer(BaseOP):
    """一层 Mistral decoder block。"""

    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = MistralAttn(config, layer_id)
        self.mlp = MistralMLP(config)
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
        """执行一层 Mistral decoder。"""

        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class MistralModel(BaseOP):
    """不带 lm_head 的 Mistral 主体。"""

    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [MistralDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """从 token id 得到最后一层 hidden states。"""

        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class MistralForCausalLM(BaseLLMModel):
    """完整 Mistral 因果语言模型。"""

    def __init__(self, config: ModelConfig):
        self.model = MistralModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        """读取当前 batch.input_ids 并输出 logits。"""

        ids = get_global_ctx().batch.input_ids
        output = self.model.forward(ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["MistralForCausalLM"]
