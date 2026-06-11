from __future__ import annotations

# 这个文件统一初始化 miniSGLang 的日志格式。
#
# 日志会带时间戳、可选 pid、可选 TP rank，并对不同等级使用颜色，便于多进程
# server 调试。

from functools import partial
from typing import TYPE_CHECKING

_LOG_LEVEL = None


def init_logger(
    name: str,
    suffix: str = "",
    *,
    strip_file: bool = True,
    level: str | None = None,
    use_pid: bool | None = None,
    use_tp_rank: bool | None = None,
):
    """初始化一个带颜色和固定格式的 logger。"""

    import logging
    import os
    import sys

    global _LOG_LEVEL
    if _LOG_LEVEL is None:
        # LOG_LEVEL 环境变量可以覆盖默认 INFO。
        LEVEL_MAP = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        level = level or os.getenv("LOG_LEVEL", "").upper()
        _LOG_LEVEL = LEVEL_MAP.get(level, logging.INFO)

    if strip_file:
        suffix = os.path.basename(suffix)

    if suffix:
        suffix = f"|{suffix}"

    if use_pid is None:
        use_pid = os.getenv("LOG_PID", "0").lower() in ("1", "true", "yes")

    if use_pid:
        # 多进程调试时可以打开 LOG_PID，看清每条日志来自哪个进程。
        pid = os.getpid()
        suffix = f"|pid={pid}{suffix}"

    tp_info = None

    # Color formatter class
    class ColorFormatter(logging.Formatter):
        """给日志等级上色的 formatter。"""

        # ANSI color codes
        COLORS = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
        }
        RESET = "\033[0m"
        BOLD = "\033[1m"

        def format(self, record):
            """把一条 logging record 格式化成最终字符串。"""

            from minisgl.distributed import try_get_tp_info

            # Format timestamp like SGLang: [YYYY-MM-DD|HH:MM:SS|pid=1234]
            timestamp = self.formatTime(record, "[%Y-%m-%d|%H:%M:%S{suffix}]")
            nonlocal tp_info
            tp_info = tp_info or try_get_tp_info()
            if tp_info is not None and use_tp_rank is not False:
                real_suffix = f"{suffix}|core|rank={tp_info.rank}"
            else:
                real_suffix = suffix
            timestamp = timestamp.format(suffix=real_suffix)

            # Get color for log level
            level_color = self.COLORS.get(record.levelname, "")

            # Format the message
            colored_level = f"{level_color}{record.levelname:<8}{self.RESET}"
            message = record.getMessage()

            # Pretty format: [timestamp] LEVEL message
            return f"{self.BOLD}{timestamp}{self.RESET} {colored_level} {message}"

    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = ColorFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    def _call_rank0(msg, *args, _which, **kwargs):
        from minisgl.distributed import get_tp_info

        nonlocal tp_info
        tp_info = tp_info or get_tp_info()
        assert tp_info is not None, "TP info not set yet"
        if tp_info.is_primary():
            getattr(logger, _which)(msg, *args, **kwargs)

    if TYPE_CHECKING:

        class WrapperLogger(logging.Logger):
            """Custom logger to handle the color formatter."""

            def info_rank0(self, msg, *args, **kwargs): ...
            def warning_rank0(self, msg, *args, **kwargs): ...
            def debug_rank0(self, msg, *args, **kwargs): ...
            def critical_rank0(self, msg, *args, **kwargs): ...

        return WrapperLogger(name)
    else:
        logger.info_rank0 = partial(_call_rank0, _which="info")
        logger.debug_rank0 = partial(_call_rank0, _which="debug")
        logger.critical_rank0 = partial(_call_rank0, _which="critical")
        logger.warning_rank0 = partial(_call_rank0, _which="warning")
        return logger
