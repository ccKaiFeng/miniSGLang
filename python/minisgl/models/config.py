from __future__ import annotations

# 这个文件把 HuggingFace 的模型配置统一转换成 miniSGLang 内部使用的配置。
#
# 不同模型的 HF config 字段名可能略有差异。ModelConfig.from_hf() 会把这些差异
# 整理成统一字段，后面的 model/layer/engine 代码只依赖 ModelConfig。

from dataclasses import dataclass
from typing import Any, Dict
from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    """RoPE 旋转位置编码相关配置。"""

    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    """miniSGLang 内部统一使用的模型结构配置。"""

    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]

    @property
    def is_moe(self) -> bool:
        """根据 model_type 判断是否是 MoE 模型。"""

        return "moe" in self.model_type

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        """从 HuggingFace PretrainedConfig 构造 ModelConfig。"""

        if hasattr(config, "text_config") and config.text_config is not None:
            # 有些模型把真正的文本模型配置放在 text_config 内部。
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        # Llama/Qwen: rope_theta is a direct attr; Mistral: it's inside rope_scaling dict
        # Llama/Qwen 通常直接有 rope_theta；Mistral 可能放在 rope_scaling 字典里。
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_theta = getattr(config, "rope_theta", None) or rope_scaling["rope_theta"]

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
        )
