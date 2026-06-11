from .server import launch_server

# 这个文件是 `python -m minisgl` 的入口。
# Python 执行一个 package 时，会运行该 package 下的 __main__.py。

assert __name__ == "__main__"

# 启动默认 HTTP API Server 模式。
launch_server()
