from __future__ import annotations

# 这个文件是 MoE backend 的 registry。
#
# 当前只注册 fused backend，后续如果增加别的 MoE 实现，可以继续注册到
# SUPPORTED_MOE_BACKENDS。

from typing import Protocol

from minisgl.utils import Registry, init_logger

from .base import BaseMoeBackend

logger = init_logger(__name__)


class MoeBackendCreator(Protocol):
    """MoE backend 创建函数的类型约束。"""

    def __call__(self) -> BaseMoeBackend: ...


SUPPORTED_MOE_BACKENDS = Registry[MoeBackendCreator]("MoE Backend")


@SUPPORTED_MOE_BACKENDS.register("fused")
def create_fused_moe_backend():
    """创建 fused MoE backend。"""

    from .fused import FusedMoe

    return FusedMoe()


def create_moe_backend(backend: str) -> BaseMoeBackend:
    """按名字创建 MoE backend。"""

    return SUPPORTED_MOE_BACKENDS[backend]()


__all__ = [
    "BaseMoeBackend",
    "create_moe_backend",
    "SUPPORTED_MOE_BACKENDS",
]
