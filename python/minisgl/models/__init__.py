# models 包导出模型配置、模型创建和权重加载入口。

from .base import BaseLLMModel
from .config import ModelConfig, RotaryConfig
from .register import get_model_class
from .weight import load_weight


def create_model(model_config: ModelConfig) -> BaseLLMModel:
    """根据 HuggingFace architecture 名字创建模型对象。

    model_config.architectures[0] 例如 "LlamaForCausalLM"、"Qwen2ForCausalLM"。
    get_model_class() 负责名字到具体类的映射，返回的模型实现 BaseLLMModel.forward()。
    """

    return get_model_class(model_config.architectures[0], model_config)


__all__ = ["create_model", "load_weight", "RotaryConfig"]
