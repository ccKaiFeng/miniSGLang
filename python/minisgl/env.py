from __future__ import annotations

# 这个文件集中管理 miniSGLang 使用的环境变量。
#
# 环境变量统一加 MINISGL_ 前缀，例如 MINISGL_SHELL_MAX_TOKENS。
# EnvVar 会保存默认值，并在 ENV 初始化时尝试从 os.environ 读取覆盖值。

import os
from functools import partial
from typing import Callable, Generic, TypeVar


class BaseEnv:
    """环境变量项的基类。"""

    def _init(self, name: str) -> None:
        raise NotImplementedError


T = TypeVar("T")


class EnvVar(BaseEnv, Generic[T]):
    """带类型转换函数的环境变量封装。"""

    def __init__(self, default_value: T, fn: Callable[[str], T]):
        self.value = default_value
        self.fn = fn
        super().__init__()

    def _init(self, name: str) -> None:
        """从真实环境变量中读取 name，并用 fn 转成目标类型。"""

        env_value = os.getenv(name)
        if env_value is not None:
            try:
                self.value = self.fn(env_value)
            except Exception:
                pass

    def __bool__(self):
        return self.value

    def __str__(self):
        return str(self.value)


_TO_BOOL = lambda x: x.lower() in ("1", "true", "yes")


def _PARSE_MEM_BYTES(mem: str) -> int:
    """把 '1G'、'512M'、'1024' 这类字符串解析成字节数。"""

    mem = mem.strip().upper()
    if not mem[-1].isalpha():
        return int(mem)
    if mem.endswith("B"):
        mem = mem[:-1]
    UNIT_MAP = {"K": 1024, "M": 1024**2, "G": 1024**3}
    return int(float(mem[:-1]) * UNIT_MAP[mem[-1]])


MINISGL_ENV_PREFIX = "MINISGL_"
EnvInt = partial(EnvVar[int], fn=int)
EnvFloat = partial(EnvVar[float], fn=float)
EnvBool = partial(EnvVar[bool], fn=_TO_BOOL)
EnvOption = partial(EnvVar[bool | None], fn=_TO_BOOL, default_value=None)
EnvMem = partial(EnvVar[int], fn=_PARSE_MEM_BYTES)


class EnvClassSingleton:
    """全局唯一的环境变量配置对象。"""

    _instance: EnvClassSingleton | None = None

    # shell
    SHELL_MAX_TOKENS = EnvInt(2048)
    SHELL_TOP_K = EnvInt(-1)
    SHELL_TOP_P = EnvFloat(1.0)
    SHELL_TEMPERATURE = EnvFloat(0.6)

    # backend runtime
    FLASHINFER_USE_TENSOR_CORES = EnvOption()
    DISABLE_OVERLAP_SCHEDULING = EnvBool(False)
    PYNCCL_MAX_BUFFER_SIZE = EnvMem(1024**3)

    def __new__(cls):
        """保证 ENV 只有一个实例。"""

        # single instance
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """遍历类中的 EnvVar 字段，并用 MINISGL_ 前缀初始化。"""

        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            assert isinstance(attr_value, BaseEnv)
            attr_value._init(f"{MINISGL_ENV_PREFIX}{attr_name}")


ENV = EnvClassSingleton()
