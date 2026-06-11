# server 包对外只暴露 launch_server。
# 其他模块可以通过 `from minisgl.server import launch_server` 启动服务。

from .launch import launch_server

__all__ = ["launch_server"]
