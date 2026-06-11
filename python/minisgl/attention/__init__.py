from __future__ import annotations

# 这个文件是 attention 子包的统一出口和 backend registry。
#
# 用户通过 `--attn fa`、`--attn fi`、`--attn trtllm` 或 `--attn fa,fi`
# 选择 attention backend。这里负责校验名字、创建 backend，并支持 prefill/decode
# 使用不同 backend 的 hybrid 模式。

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry, init_logger

from .base import BaseAttnBackend, BaseAttnMetadata, HybridBackend

if TYPE_CHECKING:
    from minisgl.models import ModelConfig

logger = init_logger(__name__)


class BackendCreator(Protocol):
    """attention backend 创建函数的类型约束。"""

    def __call__(self, config: ModelConfig) -> BaseAttnBackend: ...


SUPPORTED_ATTENTION_BACKENDS = Registry[BackendCreator]("Attention Backend")


@SUPPORTED_ATTENTION_BACKENDS.register("trtllm")
def create_trtllm_backend(config: ModelConfig):
    """创建 TensorRT-LLM attention backend。"""

    from .trtllm import TensorRTLLMBackend

    return TensorRTLLMBackend(config)


@SUPPORTED_ATTENTION_BACKENDS.register("fi")
def create_fi_backend(config: ModelConfig):
    """创建 FlashInfer attention backend。"""

    from .fi import FlashInferBackend

    return FlashInferBackend(config)


@SUPPORTED_ATTENTION_BACKENDS.register("fa")
def create_fa_backend(config: ModelConfig):
    """创建 FlashAttention backend。"""

    from .fa import FlashAttentionBackend

    return FlashAttentionBackend(config)


def validate_attn_backend(backend: str, allow_auto: bool = True):
    """检查 backend 名称是否合法。"""

    if backend != "auto":
        required_backends = backend.split(",") if "," in backend else [backend]
        SUPPORTED_ATTENTION_BACKENDS.assert_supported(required_backends)
    else:
        assert allow_auto, "auto is not allowed here"
    return backend


def create_attention_backend(
    backend: str,
    config: ModelConfig,
) -> BaseAttnBackend:
    """按配置创建 attention backend。

    如果 backend 形如 "fa,fi"，表示 prefill 用 fa，decode 用 fi。
    """

    validate_attn_backend(backend, allow_auto=False)
    if "," in backend:
        assert backend.count(",") == 1, "Only one comma is allowed in hybrid backend"
        p_backend, d_backend = backend.split(",", 1)
        if p_backend != d_backend:
            logger.info(f"Using hybrid attention backend: prefill={p_backend}, decode={d_backend}")
            p_backend = create_attention_backend(p_backend, config)
            d_backend = create_attention_backend(d_backend, config)
            return HybridBackend(p_backend, d_backend)
        backend = p_backend  # both are the same, fall through to single backend
        logger.warning(f"P/D attention backends are the same: {backend}, using single backend.")

    return SUPPORTED_ATTENTION_BACKENDS[backend](config)


__all__ = [
    "BaseAttnMetadata",
    "BaseAttnBackend",
    "create_attention_backend",
    "SUPPORTED_ATTENTION_BACKENDS",
    "validate_attn_backend",
]
