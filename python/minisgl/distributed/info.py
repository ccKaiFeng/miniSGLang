from __future__ import annotations

# 这个文件保存当前进程的 Tensor Parallel(TP) 身份信息。
#
# TP 可以理解成“把一个大模型切到多张 GPU 上一起跑”。每个进程负责其中一张
# GPU，也就是一个 TP rank。本文件记录：
# - rank：当前进程是第几个 rank；
# - size：总共有多少个 rank。

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    """一个 TP 进程的身份信息。

    frozen=True 表示对象创建后字段不能再改，避免运行中 rank/size 被误改。
    """

    rank: int
    size: int

    def __post_init__(self):
        """创建后检查 rank 是否在合法范围内。"""

        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        """rank 0 是主 rank，通常负责和外部进程通信或打印一次性日志。"""

        return self.rank == 0


_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    """设置当前进程的 TP 身份。

    每个进程只允许设置一次。重复设置通常说明初始化流程有问题。
    """

    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    """读取当前进程的 TP 身份；未初始化时直接报错。"""

    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    """尝试读取 TP 身份；未初始化时返回 None，而不是报错。"""

    return _TP_INFO


__all__ = ["DistributedInfo", "set_tp_info", "get_tp_info", "try_get_tp_info"]
