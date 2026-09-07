from __future__ import annotations

"""ZipCache mixed-precision KV Attention 的输入描述协议。

这个模块只负责把请求使用的 normal/compressed KV 段整理成 GPU descriptor。
它不会反量化 KV，也不会修改 radix cache、page table 或 compressed pool。
"""

from dataclasses import dataclass
from typing import Any, Dict, Sequence, TypeAlias

import torch
from minisgl.core import Batch

from .base import BaseAttnMetadata

NORMAL_SEGMENT = 0
QUANTIZED_SEGMENT = 1


@dataclass(frozen=True)
class NormalKVSegment:
    """请求中仍保存在 normal KV pool 的一段逻辑连续 KV。

    参数：
    - position_base：本段第一个 token 在请求中的逻辑位置；
    - indices：shape ``[M]`` 的 GPU int32/int64 tensor，保存 M 个 normal pool
      物理 token index。物理地址不要求连续。
    """

    position_base: int
    indices: torch.Tensor

    @property
    def length(self) -> int:
        """返回本段 token 数 M。"""

        return self.indices.numel()


@dataclass(frozen=True)
class CompressedKVSegment:
    """请求中直接从 compressed pool 读取的一个 radix entry。

    entry 是 ZipCache manager 保存的 compressed entry。它必须提供 ``length`` 和
    ``layers[layer_id].k/v``；每个 K/V 又包含 important/unimportant QuantizedPart。
    position_base 是该 entry 第一个 token 在请求中的逻辑位置。
    """

    position_base: int
    entry: Any

    @property
    def length(self) -> int:
        """返回 compressed entry 覆盖的 token 数 N。"""

        return int(self.entry.length)


MixedKVSource: TypeAlias = NormalKVSegment | CompressedKVSegment


@dataclass(frozen=True)
class MixedCompressedBuffers:
    """mixed Attention 只读访问的四块 ZipCache GPU buffer。"""

    q4: torch.Tensor
    q2: torch.Tensor
    scale: torch.Tensor
    ids: torch.Tensor

    @classmethod
    def from_pool(cls, pool: Any) -> MixedCompressedBuffers:
        """从 ``_V3CompressedPool`` 提取固定 buffer，不复制数据。"""

        buffers = cls(
            q4=pool.q4_buffer,
            q2=pool.q2_buffer,
            scale=pool.scale_buffer,
            ids=pool.ids_buffer,
        )
        buffers.validate()
        return buffers

    def validate(self) -> None:
        """校验 buffer 的 device、dtype 和一维连续布局。"""

        tensors = (self.q4, self.q2, self.scale, self.ids)
        if any(t.dim() != 1 or not t.is_contiguous() for t in tensors):
            raise ValueError("Compressed buffers must be contiguous 1-D tensors")
        if self.q4.dtype != torch.uint8 or self.q2.dtype != torch.uint8:
            raise TypeError("q4/q2 compressed buffers must use torch.uint8")
        if self.scale.dtype != torch.float16:
            raise TypeError("compressed scale buffer must use torch.float16")
        if self.ids.dtype != torch.int64:
            raise TypeError("compressed ids buffer must use torch.int64")
        if len({t.device for t in tensors}) != 1:
            raise ValueError("All compressed buffers must be on the same device")
        if self.q4.device.type != "cuda":
            raise ValueError("Mixed KV Attention requires CUDA compressed buffers")

    @property
    def device(self) -> torch.device:
        """返回 compressed pool 所在 CUDA device。"""

        return self.q4.device


@dataclass(frozen=True)
class MixedLayerMetadata:
    """某一 transformer layer 的 KV segment descriptor。

    - segment_indptr shape ``[B+1]``，第 b 个请求使用 descriptor
      ``[indptr[b], indptr[b+1])``；
    - segment_meta shape ``[G,5]``，每行为
      ``kind,count,position_base,k_storage_bit,v_storage_bit``；
    - segment_offsets shape ``[G,8]``，每行为 K/V q/min/step、共享 ids 和
      normal index 的 offset。offset 单位是各自 buffer 的元素个数。
    """

    segment_indptr: torch.Tensor
    segment_meta: torch.Tensor
    segment_offsets: torch.Tensor


@dataclass
class MixedAttnMetadata(BaseAttnMetadata):
    """FI/FA 共用的 mixed KV Attention batch metadata。

    qo_indptr shape ``[B+1]``；q_positions shape ``[Tq]``；kv_lens shape
    ``[B]``；normal_indices shape ``[Nnormal]``。layers 中每一项只描述当前
    layer 的 compressed 分组，因为 ZipCache 会逐层选择 important token。
    """

    qo_indptr: torch.Tensor
    q_positions: torch.Tensor
    kv_lens: torch.Tensor
    normal_indices: torch.Tensor
    layers: Dict[int, MixedLayerMetadata]
    compressed_buffers: MixedCompressedBuffers
    has_compressed: bool
    compressed_num_kv_heads: int
    compressed_head_dim: int

    def get_last_indices(self, bs: int) -> torch.Tensor:
        """返回真实 batch 中每个请求最后一个 query 的扁平下标，shape ``[bs]``。"""

        return self.qo_indptr[1 : 1 + bs] - 1

    def for_layer(self, layer_id: int) -> MixedLayerMetadata:
        """返回 layer_id 对应的 segment descriptor。"""

        try:
            return self.layers[layer_id]
        except KeyError as exc:
            raise KeyError(f"Missing mixed KV metadata for layer {layer_id}") from exc


@dataclass(frozen=True)
class MixedKVBatchSpec:
    """尚未编码成 GPU descriptor 的 batch 输入说明。"""

    request_sources: tuple[tuple[MixedKVSource, ...], ...]
    compressed_buffers: MixedCompressedBuffers
    num_layers: int


def attach_mixed_kv_sources(
    batch: Batch,
    request_sources: Sequence[Sequence[MixedKVSource]],
    *,
    compressed_pool: Any,
    num_layers: int,
) -> None:
    """给 batch 附加 mixed KV 输入，供 FI/FA 的 ``prepare_metadata`` 消费。

    request_sources 必须与 ``batch.padded_reqs`` 一一对应。每个请求的 source
    区间必须恰好、不重叠地覆盖 ``[0, req.device_len)``。该函数只保存引用，
    compressed entry 及其 pool slice 必须至少存活到 Attention kernel 执行结束。
    """

    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    spec = MixedKVBatchSpec(
        request_sources=tuple(tuple(sources) for sources in request_sources),
        compressed_buffers=MixedCompressedBuffers.from_pool(compressed_pool),
        num_layers=num_layers,
    )
    setattr(batch, "_mixed_kv_spec", spec)


def clear_mixed_kv_sources(batch: Batch) -> None:
    """移除测试或后续 runtime 附加到 batch 的 mixed KV 输入说明。"""

    if hasattr(batch, "_mixed_kv_spec"):
        delattr(batch, "_mixed_kv_spec")


def maybe_prepare_mixed_metadata(batch: Batch) -> bool:
    """如果 batch 附带 MixedKVBatchSpec，则构建 metadata 并返回 True。"""

    spec = getattr(batch, "_mixed_kv_spec", None)
    if spec is None:
        return False
    if not isinstance(spec, MixedKVBatchSpec):
        raise TypeError("batch._mixed_kv_spec must be a MixedKVBatchSpec")
    batch.attn_metadata = build_mixed_metadata(
        batch,
        spec.request_sources,
        compressed_buffers=spec.compressed_buffers,
        num_layers=spec.num_layers,
    )
    return True


def build_mixed_metadata(
    batch: Batch,
    request_sources: Sequence[Sequence[MixedKVSource]],
    *,
    compressed_buffers: MixedCompressedBuffers,
    num_layers: int,
) -> MixedAttnMetadata:
    """把每请求 KV source 编码成 mixed Attention 使用的 GPU tensor。

    本函数只读取 Python 对象中的 shape、bit、storage_offset 等 CPU metadata，
    不把 compressed q/min/step/ids 拷回 CPU，也不创建完整反量化 KV。
    """

    compressed_buffers.validate()
    reqs = batch.padded_reqs
    if len(request_sources) != len(reqs):
        raise ValueError(
            f"request_sources has {len(request_sources)} requests, expected {len(reqs)}"
        )
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    device = compressed_buffers.device
    qo_lengths = [int(req.extend_len) for req in reqs]
    kv_lengths = [int(req.device_len) for req in reqs]
    total_q = sum(qo_lengths)
    if batch.positions.numel() != total_q:
        raise ValueError(
            f"batch.positions has {batch.positions.numel()} values, expected {total_q}"
        )

    normal_tensors: list[torch.Tensor] = []
    normal_offsets: Dict[tuple[int, int], int] = {}
    next_normal_offset = 0
    has_compressed = False
    for request_id, (sources, kv_len) in enumerate(zip(request_sources, kv_lengths)):
        _validate_request_sources(sources, kv_len, request_id)
        for source_id, source in enumerate(sources):
            if isinstance(source, NormalKVSegment):
                if source.indices.dim() != 1:
                    raise ValueError(
                        f"normal indices must have shape [M], got {tuple(source.indices.shape)}"
                    )
                if source.indices.dtype not in (torch.int32, torch.int64):
                    raise TypeError("normal indices must use torch.int32 or torch.int64")
                indices = source.indices.reshape(-1)
                if indices.device != device:
                    indices = indices.to(device=device, non_blocking=True)
                indices = indices.to(dtype=torch.int32)
                normal_offsets[(request_id, source_id)] = next_normal_offset
                next_normal_offset += indices.numel()
                normal_tensors.append(indices)
            elif isinstance(source, CompressedKVSegment):
                has_compressed = True
            else:
                raise TypeError(f"Unsupported mixed KV source: {type(source)!r}")

    normal_indices = (
        torch.cat(normal_tensors)
        if normal_tensors
        else torch.empty(0, dtype=torch.int32, device=device)
    )

    layers: Dict[int, MixedLayerMetadata] = {}
    compressed_shape: tuple[int, int] | None = None
    for layer_id in range(num_layers):
        indptr = [0]
        meta_rows: list[list[int]] = []
        offset_rows: list[list[int]] = []
        for request_id, sources in enumerate(request_sources):
            for source_id, source in enumerate(sources):
                if isinstance(source, NormalKVSegment):
                    meta_rows.append(
                        [NORMAL_SEGMENT, source.length, int(source.position_base), 0, 0]
                    )
                    offset_rows.append(
                        [0, 0, 0, 0, 0, 0, 0, normal_offsets[(request_id, source_id)]]
                    )
                    continue

                try:
                    layer_entry = source.entry.layers[layer_id]
                except (AttributeError, KeyError) as exc:
                    raise ValueError(f"Compressed entry has no data for layer {layer_id}") from exc
                part_count = int(layer_entry.k.important.ids.numel()) + int(
                    layer_entry.k.unimportant.ids.numel()
                )
                if part_count != source.length:
                    raise ValueError(
                        f"Compressed entry layer {layer_id} has {part_count} grouped tokens, "
                        f"expected {source.length}"
                    )
                for part_name in ("important", "unimportant"):
                    k_part = getattr(layer_entry.k, part_name)
                    v_part = getattr(layer_entry.v, part_name)
                    count = int(k_part.ids.numel())
                    if count == 0:
                        continue
                    _validate_part_pair(
                        k_part,
                        v_part,
                        count=count,
                        buffers=compressed_buffers,
                        request_id=request_id,
                        layer_id=layer_id,
                        part_name=part_name,
                    )
                    current_shape = (int(k_part.q_shape[1]), int(k_part.q_shape[2]))
                    if compressed_shape is None:
                        compressed_shape = current_shape
                    elif compressed_shape != current_shape:
                        raise ValueError(
                            "All compressed parts must use the same [Hkv,D] shape; "
                            f"got {compressed_shape} and {current_shape}"
                        )
                    k_offsets = _part_offsets(k_part)
                    v_offsets = _part_offsets(v_part)
                    meta_rows.append(
                        [
                            QUANTIZED_SEGMENT,
                            count,
                            int(source.position_base),
                            int(k_part.storage_bit),
                            int(v_part.storage_bit),
                        ]
                    )
                    offset_rows.append(
                        [
                            k_offsets[0],
                            k_offsets[1],
                            k_offsets[2],
                            v_offsets[0],
                            v_offsets[1],
                            v_offsets[2],
                            int(k_part.ids.storage_offset()),
                            0,
                        ]
                    )
            indptr.append(len(meta_rows))

        layers[layer_id] = MixedLayerMetadata(
            segment_indptr=torch.tensor(indptr, dtype=torch.int32, device=device),
            segment_meta=torch.tensor(meta_rows, dtype=torch.int32, device=device).reshape(-1, 5),
            segment_offsets=torch.tensor(offset_rows, dtype=torch.int64, device=device).reshape(
                -1, 8
            ),
        )

    qo_indptr = torch.tensor([0] + qo_lengths, dtype=torch.int32).cumsum_(0).to(device)
    return MixedAttnMetadata(
        qo_indptr=qo_indptr,
        q_positions=batch.positions.to(device=device, dtype=torch.int32, non_blocking=True),
        kv_lens=torch.tensor(kv_lengths, dtype=torch.int32, device=device),
        normal_indices=normal_indices,
        layers=layers,
        compressed_buffers=compressed_buffers,
        has_compressed=has_compressed,
        compressed_num_kv_heads=compressed_shape[0] if compressed_shape else 0,
        compressed_head_dim=compressed_shape[1] if compressed_shape else 0,
    )


def _validate_request_sources(
    sources: Sequence[MixedKVSource], kv_len: int, request_id: int
) -> None:
    """校验 source 的逻辑区间恰好覆盖请求的全部可见 KV。"""

    ranges = sorted((int(source.position_base), source.length) for source in sources)
    cursor = 0
    for position_base, length in ranges:
        if length < 0 or position_base != cursor:
            raise ValueError(
                f"request {request_id} KV sources must partition [0, {kv_len}); "
                f"expected segment at {cursor}, got start={position_base}, length={length}"
            )
        cursor += length
    if cursor != kv_len:
        raise ValueError(
            f"request {request_id} KV sources cover {cursor} tokens, expected {kv_len}"
        )


def _part_offsets(part: Any) -> tuple[int, int, int]:
    """返回 QuantizedPart 的 q/min/step storage offset。"""

    return (
        int(part.q.storage_offset()),
        int(part.min.storage_offset()),
        int(part.step.storage_offset()),
    )


def _validate_part_pair(
    k_part: Any,
    v_part: Any,
    *,
    count: int,
    buffers: MixedCompressedBuffers,
    request_id: int,
    layer_id: int,
    part_name: str,
) -> None:
    """校验同一组 K/V QuantizedPart 是否能由共享 descriptor 读取。"""

    prefix = f"request={request_id}, layer={layer_id}, part={part_name}"
    if int(v_part.ids.numel()) != count:
        raise ValueError(f"K/V ids length mismatch for {prefix}")
    if tuple(k_part.q_shape) != tuple(v_part.q_shape):
        raise ValueError(f"K/V q_shape mismatch for {prefix}")
    if len(k_part.q_shape) != 3 or int(k_part.q_shape[0]) != count:
        raise ValueError(f"Expected q_shape [M,H,D] for {prefix}, got {k_part.q_shape}")
    for name, part in (("K", k_part), ("V", v_part)):
        if int(part.storage_bit) not in (2, 4):
            raise ValueError(f"{name} storage_bit must be 2 or 4 for {prefix}")
        q_buffer = buffers.q2 if int(part.storage_bit) == 2 else buffers.q4
        if part.q.dtype != torch.uint8 or not part.q.is_contiguous():
            raise TypeError(f"{name} q must be contiguous uint8 for {prefix}")
        if part.min.dtype != torch.float16 or part.step.dtype != torch.float16:
            raise TypeError(f"{name} min/step must be FP16 for {prefix}")
        if tuple(part.min.shape) != tuple(part.q_shape[:2]) + (1,):
            raise ValueError(f"{name} min shape must be [M,H,1] for {prefix}")
        if tuple(part.step.shape) != tuple(part.min.shape):
            raise ValueError(f"{name} step shape must match min for {prefix}")
        logical_numel = int(part.q_shape[0] * part.q_shape[1] * part.q_shape[2])
        if int(part.logical_numel) != logical_numel:
            raise ValueError(f"{name} logical_numel mismatch for {prefix}")
        expected_packed = (logical_numel * int(part.storage_bit) + 7) // 8
        if part.q.numel() != expected_packed:
            raise ValueError(f"{name} packed q length mismatch for {prefix}")
        if part.q.untyped_storage().data_ptr() != q_buffer.untyped_storage().data_ptr():
            raise ValueError(f"{name} q tensor does not belong to the expected pool for {prefix}")
        if part.min.untyped_storage().data_ptr() != buffers.scale.untyped_storage().data_ptr():
            raise ValueError(f"{name} min tensor does not belong to scale_buffer for {prefix}")
        if part.step.untyped_storage().data_ptr() != buffers.scale.untyped_storage().data_ptr():
            raise ValueError(f"{name} step tensor does not belong to scale_buffer for {prefix}")
        if part.ids.dtype != torch.int64 or not part.ids.is_contiguous():
            raise TypeError(f"{name} ids must be contiguous int64 for {prefix}")
        if part.ids.untyped_storage().data_ptr() != buffers.ids.untyped_storage().data_ptr():
            raise ValueError(f"{name} ids tensor does not belong to ids_buffer for {prefix}")
        views = (
            (part.q, q_buffer, "q"),
            (part.min, buffers.scale, "min"),
            (part.step, buffers.scale, "step"),
            (part.ids, buffers.ids, "ids"),
        )
        for view, buffer, field in views:
            if view.storage_offset() + view.numel() > buffer.numel():
                raise ValueError(f"{name} {field} slice exceeds its pool buffer for {prefix}")


__all__ = [
    "NORMAL_SEGMENT",
    "QUANTIZED_SEGMENT",
    "NormalKVSegment",
    "CompressedKVSegment",
    "MixedCompressedBuffers",
    "MixedLayerMetadata",
    "MixedAttnMetadata",
    "MixedKVBatchSpec",
    "attach_mixed_kv_sources",
    "clear_mixed_kv_sources",
    "build_mixed_metadata",
    "maybe_prepare_mixed_metadata",
]
