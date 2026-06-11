from __future__ import annotations

# 这个文件实现进程间消息的序列化和反序列化。
#
# miniSGLang 的多个进程通过 ZMQ 传消息。ZMQ 不能直接传 Python 对象，
# 所以发送前要把 dataclass、list、dict、torch.Tensor 等对象转换成普通 dict/bytes；
# 接收后再根据 "__type__" 字段还原成原来的消息对象。
#
# 这里的实现是一个轻量级“对象 <-> dict”转换器，不是通用 pickle。
# 它只支持本工程消息里会用到的类型。

from typing import Any, Dict, Type

import numpy as np
import torch


def _serialize_any(value: Any) -> Any:
    """递归序列化任意支持的 Python 值。

    递归的意思是：如果 value 是 list/dict，里面的每个元素也要继续序列化。
    这样嵌套对象才能完整变成可传输的数据。
    """

    if isinstance(value, dict):
        # dict 逐个序列化 value，key 保持不变。
        return {k: _serialize_any(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        # list/tuple 逐个序列化元素，并保持原来的容器类型。
        return type(value)(_serialize_any(v) for v in value)
    elif isinstance(value, (int, float, str, type(None), bool, bytes)):
        # 这些是 msgpack/ZMQ 可以直接传输的基础类型。
        return value
    else:
        # 其他对象必须是本工程定义的 dataclass 或 torch.Tensor，
        # 继续交给 serialize_type 处理。
        return serialize_type(value)


def serialize_type(self) -> Dict:
    """把一个对象序列化成 dict。

    输出 dict 一定带有 "__type__" 字段，用来记录原始对象类型。
    反序列化时会根据这个字段找到对应 class，然后重新构造对象。
    """

    # find all member variables
    serialized = {}

    if isinstance(self, torch.Tensor):
        # 当前只支持一维 tensor，因为消息里只传 token id 这种 1D 数据。
        assert self.dim() == 1, "we can only serialize 1D tensor for now"
        serialized["__type__"] = "Tensor"

        # Tensor 先转 numpy，再转 bytes，这样可以跨进程传输。
        serialized["buffer"] = self.numpy().tobytes()

        # dtype 也要记录下来，否则接收端不知道 bytes 应该按 int32/int64 等哪种格式解释。
        serialized["dtype"] = str(self.dtype)
        return serialized

    # normal type
    # 普通 dataclass：记录类名，然后把每个成员变量递归序列化。
    serialized["__type__"] = self.__class__.__name__
    for k, v in self.__dict__.items():
        serialized[k] = _serialize_any(v)
    return serialized


def _deserialize_any(cls_map: Dict[str, Type], data: Any) -> Any:
    """递归反序列化一个值。"""

    if isinstance(data, dict):
        if "__type__" in data:
            # 带 "__type__" 说明这是之前序列化过的对象。
            return deserialize_type(cls_map, data)
        else:
            # 普通 dict 递归处理每个 value。
            return {k: _deserialize_any(cls_map, v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        # list/tuple 递归还原每个元素，并保持容器类型。
        return type(data)(_deserialize_any(cls_map, d) for d in data)
    elif isinstance(data, (int, float, str, type(None), bool, bytes)):
        # 基础类型直接返回。
        return data
    else:
        raise ValueError(f"Cannot deserialize type {type(data)}")


def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any:
    """把带 "__type__" 的 dict 还原成对象。

    cls_map 通常传 globals()，也就是当前 message 文件里的所有类名到类对象的映射。
    例如 "__type__" == "UserMsg" 时，就会找到 UserMsg 这个 class 并调用构造函数。
    """

    type_name = data["__type__"]
    # we can only serialize 1D tensor for now
    if type_name == "Tensor":
        # 还原 tensor 时，需要先按 dtype 把 bytes 解释成 numpy array。
        buffer = data["buffer"]
        dtype_str = data["dtype"].replace("torch.", "")
        np_dtype = getattr(np, dtype_str)
        assert isinstance(buffer, bytes)
        np_tensor = np.frombuffer(buffer, dtype=np_dtype)

        # np.frombuffer 得到的数组可能引用原始 bytes，这里 copy 一份再转 torch，
        # 避免后续数据生命周期问题。
        return torch.from_numpy(np_tensor.copy())

    # 普通消息对象：根据类型名找到 class。
    cls = cls_map[type_name]
    kwargs = {}
    for k, v in data.items():
        if k == "__type__":
            continue
        # 每个字段都递归反序列化后，作为构造参数传给 class。
        kwargs[k] = _deserialize_any(cls_map, v)
    return cls(**kwargs)
