from typing import Callable, Generic, Iterable, List, TypeVar

# 这个文件实现一个简单 registry。
#
# attention backend、MoE backend、cache manager 都使用它按字符串名字注册和创建
# 具体实现。

T = TypeVar("T")


class Registry(Generic[T]):
    """名字到对象/创建函数的映射表。"""

    def __init__(self, type: str):
        self._registry = {}
        self._type = type

    def register(self, name: str) -> Callable[[T], None]:
        """把某个对象注册为 name。通常作为装饰器使用。"""

        if name in self._registry:
            raise KeyError(f"{self._type} '{name}' is already registered.")

        def decorator(item: T) -> None:
            self._registry[name] = item

        return decorator

    def __getitem__(self, name: str) -> T:
        """按名字取出注册对象。"""

        if name not in self._registry:
            raise KeyError(f"Unsupported {self._type}: {name}")
        return self._registry[name]

    def supported_names(self) -> List[str]:
        """返回当前支持的名字列表。"""

        return list(self._registry.keys())

    def assert_supported(self, names: str | Iterable[str]) -> None:
        """检查一个或多个名字是否已注册。"""

        if isinstance(names, str):
            names = [names]
        for name in names:
            if name not in self._registry:
                from argparse import ArgumentTypeError

                raise ArgumentTypeError(
                    f"Unsupported {self._type}: {name}. "
                    f"Supported items: {self.supported_names()}"
                )
