from __future__ import annotations

# 这个文件定义 Scheduler 的配置。
#
# SchedulerConfig 继承 EngineConfig，所以它既包含模型/Engine 配置，也包含
# 调度层自己的参数，例如 chunked prefill 最大长度、cache 类型、ZMQ 地址等。

from dataclasses import dataclass, field

from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    """生成带进程 pid 的后缀，避免多个 miniSGLang 实例复用同一个 ipc 地址。"""

    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    """Scheduler 运行参数。"""

    # 一次 prefill/extend 最多处理多少 token，用于 chunked prefill。
    max_extend_tokens: int = 8192

    # KV cache 管理策略，例如 radix 或 naive。
    cache_type: str = "radix"

    # 实验功能：KV cache 即将释放/驱逐时，先在 Python 层保存压缩归档。
    enable_compressed_kv_cache: bool = False
    compressed_kv_cache_dir: str = "/root/autodl-tmp/kv_archive"
    compressed_kv_cache_codec: str = "mock"
    compressed_kv_cache_max_size_mb: int = 4096
    compressed_kv_cache_restore_policy: str = "cost"

    # offline_mode 用于 LLM 本地接口，区别于多进程 online serving。
    offline_mode: bool = False

    # networking config
    # 每次启动生成独立 suffix，拼到 ZMQ ipc 地址后面。
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        """tokenizer 向 scheduler 发送 UserMsg/AbortBackendMsg 的地址。"""

        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        """scheduler 向 detokenizer 发送 DetokenizeMsg 的地址。"""

        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        """rank 0 scheduler 向其他 TP rank 广播请求的地址。"""

        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        """Scheduler 单次 forward 允许的最大 token 数。"""

        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        """默认由后端创建 detokenizer 链路。ServerArgs 会在多 tokenizer 配置下覆盖。"""

        return True
