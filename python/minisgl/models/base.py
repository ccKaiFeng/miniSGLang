from __future__ import annotations

# 这个文件定义所有 LLM 模型类的共同抽象接口。
#
# 具体模型如 Llama/Qwen/Mistral 都需要继承 BaseLLMModel，并实现 forward()。

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from minisgl.layers import BaseOP

if TYPE_CHECKING:
    import torch


class BaseLLMModel(ABC, BaseOP):
    """完整因果语言模型的抽象基类。"""

    @abstractmethod
    def forward(self) -> torch.Tensor: ...
