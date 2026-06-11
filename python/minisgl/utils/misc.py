from __future__ import annotations

# 这个文件放一些小工具函数。


def call_if_main(name: str = "__main__", discard: bool | None = None):
    """装饰器：只在脚本作为 main 运行时调用函数。"""
    if name != "__main__":
        discard = False if discard is None else discard
        if discard:
            return lambda _: None
        else:
            return lambda f: f
    else:
        discard = True if discard is None else discard
        if discard:
            return lambda f: (f() or True) and None
        else:
            return lambda f: (f() and None) or f


def div_even(a: int, b: int, allow_replicate: bool = False) -> int:
    """要求 a 能被 b 整除后相除；可选允许复制型切分。"""
    if allow_replicate and b > a:
        assert b % a == 0, f"{b = } must be divisible by {a = } for KV head replication"
        return 1
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    return a // b


def div_ceil(a: int, b: int) -> int:
    """向上取整除法。"""
    return (a + b - 1) // b


def align_ceil(a: int, b: int) -> int:
    """把 a 向上对齐到 b 的倍数。"""
    return div_ceil(a, b) * b


def align_down(a: int, b: int) -> int:
    """把 a 向下对齐到 b 的倍数。"""
    return (a // b) * b


class Unset:
    """表示“用户没有设置该值”的哨兵类型。"""

    pass


UNSET = Unset()
