import importlib

# 这个文件根据 HuggingFace config 中的 architecture 名称选择具体模型类。
#
# 例如 architecture 为 "Qwen3ForCausalLM" 时，会动态导入 .qwen3 里的
# Qwen3ForCausalLM。这样 Engine 不需要写一堆 if/else。

from .config import ModelConfig

_MODEL_REGISTRY = {
    "LlamaForCausalLM": (".llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": (".qwen2", "Qwen2ForCausalLM"),
    "Qwen3ForCausalLM": (".qwen3", "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": (".qwen3_moe", "Qwen3MoeForCausalLM"),
    "MistralForCausalLM": (".mistral", "MistralForCausalLM"),
    "Mistral3ForConditionalGeneration": (".mistral", "MistralForCausalLM"),
}


def get_model_class(model_architecture: str, model_config: ModelConfig):
    """创建指定 architecture 对应的模型对象。"""

    if model_architecture not in _MODEL_REGISTRY:
        raise ValueError(f"Model architecture {model_architecture} not supported")
    module_path, class_name = _MODEL_REGISTRY[model_architecture]
    module = importlib.import_module(module_path, package=__package__)
    model_cls = getattr(module, class_name)
    return model_cls(model_config)


__all__ = ["get_model_class"]
