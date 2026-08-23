from __future__ import annotations

# 这个文件定义 miniSGLang 自己的“轻量模块系统”。
#
# 它没有直接继承 torch.nn.Module，而是用 BaseOP 管理参数、state_dict 和
# load_state_dict。这样模型代码更轻，也方便手工控制 Tensor Parallel 权重切分。

from abc import abstractmethod
from typing import Any, Dict, Generic, List, TypeAlias, TypeVar

import torch

_STATE_DICT: TypeAlias = Dict[str, torch.Tensor]


def _concat_prefix(prefix: str, name: str) -> str:
    """把父模块名前缀和当前字段名拼成 state_dict key。"""

    return f"{prefix}.{name}" if prefix else name


class BaseOP:
    """所有自定义算子、层、模型 block 的基类。"""

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """执行算子前向计算。

        具体输入输出由子类决定。miniSGLang 的模型层通常接收 shape
        [T, hidden_size] 的 hidden_states，其中 T 是当前 batch 展平后的 token 数。
        """
        ...

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        """递归收集当前对象中的 torch.Tensor 参数。"""

        result = result if result is not None else {}

        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                result[_concat_prefix(prefix, name)] = param
            elif isinstance(param, BaseOP):
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)

        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """把 state_dict 中的权重递归加载到当前对象。"""

        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                item = state_dict.pop(_concat_prefix(prefix, name))
                assert isinstance(item, torch.Tensor)
                assert param.shape == item.shape and param.dtype == item.dtype
                setattr(self, name, item)
            elif isinstance(param, BaseOP):
                param.load_state_dict(
                    state_dict, prefix=_concat_prefix(prefix, name), _internal=True
                )

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


class StateLessOP(BaseOP):
    """没有可训练参数的算子基类。"""

    def __init__(self):
        """初始化无状态算子。

        无状态表示没有 torch.Tensor 参数需要加载/保存，例如激活函数、RoPE factory wrapper 等。
        """

        super().__init__()

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """无状态算子不消费任何权重。

        顶层调用时如果 state_dict 仍有 key，说明 checkpoint 与模型结构不匹配。
        """

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        """无状态算子返回空权重集合。"""

        return result if result is not None else {}


T = TypeVar("T", bound=BaseOP)


class OPList(BaseOP, Generic[T]):
    """BaseOP 的列表容器，作用类似 torch.nn.ModuleList。"""

    def __init__(self, ops: List[T]):
        """保存一组 BaseOP 子模块。

        ops 通常是 transformer layers 列表；state_dict key 会使用数字下标作为层号。
        """

        super().__init__()
        self.op_list = ops

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        """按数字下标递归收集每个子 OP 的权重。"""

        result = result if result is not None else {}
        for i, op in enumerate(self.op_list):
            op.state_dict(prefix=_concat_prefix(prefix, str(i)), result=result)
        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """按数字下标递归加载每个子 OP 的权重。"""

        for i, op in enumerate(self.op_list):
            op.load_state_dict(state_dict, prefix=_concat_prefix(prefix, str(i)), _internal=True)

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
