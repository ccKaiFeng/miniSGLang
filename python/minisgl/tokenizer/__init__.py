# tokenizer 包导出 tokenizer/detokenizer worker 进程入口。

from .server import tokenize_worker

__all__ = ["tokenize_worker"]
