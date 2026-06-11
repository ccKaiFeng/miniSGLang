from .server import launch_server

# 这个文件是交互式 shell 的入口。
# 执行 `python -m minisgl.shell --model ...` 时，会启动后端进程，
# 但不启动 HTTP Server，而是在当前终端里进入聊天模式。

if __name__ == "__main__":
    launch_server(run_shell=True)
