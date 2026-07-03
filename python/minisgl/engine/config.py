from __future__ import annotations

# 这个文件定义 Engine 的配置。
#
# Engine 是真正持有模型、KV cache、attention backend 并执行 forward 的对象。
# EngineConfig 保存它初始化需要的参数。

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    """单个 Engine/TP rank 的配置。"""

    # 模型权重目录或 HuggingFace repo id。
    model_path: str

    # Tensor Parallel 信息：当前 rank 和总 rank 数。
    tp_info: DistributedInfo

    # 模型权重和激活使用的数据类型。
    dtype: torch.dtype

    # 同时运行的最大请求数，也决定 page_table 第一维大小。
    max_running_req: int = 256

    # attention/MoE backend 选择。
    attention_backend: str = "auto"
    moe_backend: str = "auto"

    # CUDA graph capture 的 batch size 列表或上限。
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None

    # KV cache page size。
    page_size: int = 1

    # 用于 KV cache 的显存比例。
    memory_ratio: float = 0.9

    # distributed 初始化超时时间。
    distributed_timeout: float = 60.0

    # 测试模式：使用假权重，不加载真实模型。
    use_dummy_weight: bool = False

    # 是否使用自定义 PyNCCL 通信。
    use_pynccl: bool = True

    # 可选覆盖最大序列长度和 KV page 数。
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages

    # ZipCache v1 experimental runtime compression. Disabled by default.
    enable_zipcache_v1: bool = False
    enable_zipcache_v2: bool = False
    enable_zipcache_v3: bool = False
    enable_zipcache_v4: bool = False
    enable_zipcache_cuda_graph: bool = False
    zipcache_unimportant_ratio: float = 0.4
    zipcache_k_important_bit: int = 4
    zipcache_k_unimportant_bit: int = 2
    zipcache_v_important_bit: int = 4
    zipcache_v_unimportant_bit: int = 2
    zipcache_streaming_gap: int = 100
    zipcache_protect_recent_tokens: int = 1
    zipcache_stats_interval: float = 30.0
    zipcache_v2_demote_on_finish: bool = True
    zipcache_v2_compressed_pool_mb: int = 0
    zipcache_v2_compressed_pool_ratio: float = 0.35
    zipcache_v3_demote_on_finish: bool = True
    zipcache_v3_normal_pool_pages: int = 0
    zipcache_v3_compressed_pool_mb: int = 0
    zipcache_v3_compressed_pool_ratio: float = 1.0
    zipcache_v3_q4_pool_ratio: float = 0.45
    zipcache_v3_q2_pool_ratio: float = 0.15
    zipcache_v3_scale_pool_ratio: float = 0.25
    zipcache_v3_ids_pool_ratio: float = 0.15
    zipcache_v3_keep_compressed_after_restore: bool = True
    zipcache_v3_min_restore_tokens: int = 0
    zipcache_v4_demote_on_finish: bool = True
    zipcache_v4_normal_pool_pages: int = 0
    zipcache_v4_compressed_pool_mb: int = 0
    zipcache_v4_compressed_pool_ratio: float = 1.0
    zipcache_v4_q4_pool_ratio: float = 0.45
    zipcache_v4_q2_pool_ratio: float = 0.15
    zipcache_v4_scale_pool_ratio: float = 0.25
    zipcache_v4_ids_pool_ratio: float = 0.15
    zipcache_v4_keep_compressed_after_restore: bool = True
    zipcache_v4_min_restore_tokens: int = 0
    zipcache_v4_use_kernel_compress: bool = True
    zipcache_v4_use_kernel_restore: bool = True

    @cached_property
    def hf_config(self):
        """加载 HuggingFace config，并缓存结果。"""

        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        """把 HuggingFace config 转成 miniSGLang 内部 ModelConfig。"""

        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        """模型允许的最大序列长度。"""

        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        """单次 forward 最大长度。Engine 默认等于模型最大序列长度。"""

        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        """单机默认 distributed 地址。ServerArgs 会覆盖这个地址。"""

        return "tcp://127.0.0.1:2333"
