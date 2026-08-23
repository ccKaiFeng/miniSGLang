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
        """创建第 layer_id 层 Mistral decoder。

        结构与 Llama decoder 类似：attention + gated MLP + 两个 fused RMSNorm。
        输入/输出 hidden_states shape 都是 [T, hidden_size]。
        """

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
        """执行一层 Mistral decoder。

        x/residual shape 是 [T, hidden_size]；返回 shape 不变。
        """

        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class MistralModel(BaseOP):
    """不带 lm_head 的 Mistral 主体。"""

    def __init__(self, config: ModelConfig):
        """创建 Mistral embedding、decoder layers 和 final RMSNorm。

        embed_tokens 把 input_ids [T] 转为 [T, hidden_size]；layers 长度为
        config.num_layers。
        """

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
        """从 token id 得到最后一层 hidden states。

        input_ids shape [T]；返回 hidden states shape [T, hidden_size]。
        """

        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class MistralForCausalLM(BaseLLMModel):
    """完整 Mistral 因果语言模型。"""

    def __init__(self, config: ModelConfig):
        """创建 Mistral backbone 和 lm_head。

        lm_head 输入 [T, hidden_size]，输出当前 batch 需要采样的 logits；
        tie_word_embeddings=True 时复用 embedding 权重。
        """

        self.model = MistralModel(config)
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

        ids = get_global_ctx().batch.input_ids
        output = self.model.forward(ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["MistralForCausalLM"]
