from __future__ import annotations

# 这个文件定义 attention backend 的抽象接口。
#
# miniSGLang 支持多种 attention 实现，例如 FlashAttention、FlashInfer、
# TensorRT-LLM。Scheduler/Engine 只依赖这里的统一接口，不直接关心底层 kernel。

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch
    from minisgl.core import Batch


@dataclass
class BaseAttnMetadata(ABC):
    """attention backend 为当前 batch 准备的元数据基类。

    metadata 的共同目标是描述“扁平 q tensor 中每个请求的边界”和“历史 KV
    在 paged cache 中的位置”。不同 backend 字段不同，但 get_last_indices()
    都返回每个真实请求最后一个 query token 的扁平下标。
    """

    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor:
        """返回真实 batch 中每个请求最后一个 query token 的扁平下标。

        参数：
        - bs：真实请求数，不包含 padding request。

        返回：
        - shape [bs]，dtype int32/int64，device 取决于 backend；
        - 第 i 个元素指向 q/output 扁平 tensor 中第 i 个请求的最后一个 query。
        """
        ...


class BaseAttnBackend(ABC):
    """attention backend 抽象接口。

    forward() 的 q/k/v 维度约定：
    - q shape [T_q, H_q_local, D]；
    - k/v shape 在 layer 内通常是 [T_new, H_kv_local*D]，store 到 cache 后按
      [physical_token, H_kv_local, D] 理解；
    - T_q/T_new 通常等于 sum(req.extend_len)，decode 时等于 batch size；
    - layer_id 指当前 transformer layer，用于选择 KV cache 的第几层。
    """

    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """执行一层 attention。

        输入：
        - q shape [T_q, H_q_local, D]，当前 batch 需要计算 attention 的 query；
        - k/v shape [T_new, H_kv_local, D] 或等价展平形式，表示本步新产生的 KV；
        - layer_id：当前 transformer layer 编号，用于访问对应层 KV cache；
        - batch：包含 req 列表、page_table 位置、prefill/decode phase 和 metadata。

        输出：
        - shape [T_q, H_q_local, D]，与 q 的 token/head 维度一致。
        """
        ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None:
        """根据 Batch 中的长度/page_table 信息生成 backend 专用 metadata。"""
        ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """为 CUDA Graph replay 预分配固定 shape buffer。

        max_seq_len 是 capture 支持的最大 device_len，bs_list 是允许 capture 的 batch size 集合。
        """
        ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None:
        """capture 前把 batch metadata 指向固定 buffer，并完成 backend plan。"""
        ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None:
        """replay 前把当前 batch 的动态内容拷入 capture buffer。"""
        ...


class HybridBackend(BaseAttnBackend):
    """prefill 和 decode 使用不同 backend 的组合封装。"""

    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        """保存 prefill/decode 各自的 backend。"""

        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """根据 batch.phase 选择 prefill 或 decode backend。"""

        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch: Batch) -> None:
        """把 metadata 准备工作转发给当前阶段对应的 backend。"""

        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """Hybrid 模式只 capture decode backend。

        prefill 的 token 数和 ragged shape 变化较大；当前 miniSGLang 只对 decode
        做 CUDA Graph，所以这里直接转发给 decode_backend。
        """

        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        """把 decode capture 准备动作转发给 decode_backend。"""

        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        """把 decode replay 准备动作转发给 decode_backend。"""

        self.decode_backend.prepare_for_replay(batch)
