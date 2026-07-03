from __future__ import annotations

import time
import weakref
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from minisgl.kvcache import BaseCacheHandle
from minisgl.utils import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class _PoolSlice:
    """compressed pool 内一段连续 buffer 的归属信息。"""

    buffer_name: str
    offset: int
    length: int


@dataclass
class _QuantizedPart:
    """一组 token 的低比特量化结果。

    ZipCache v3 会把 token 分成 important 和 unimportant 两组。每组都记录：
    - ids：这些 token 在当前 radix node 内的相对位置；
    - q：packed 后的 uint8 量化值；
    - min/step：按 token/head 保存的反量化参数；
    - bit：算法语义上的量化 bit，例如 4bit 或 2bit；
    - storage_bit：实际 packed 存储 bit，当前只支持 4bit/2bit。
    """

    ids: torch.Tensor
    q: torch.Tensor
    min: torch.Tensor
    step: torch.Tensor
    bit: int
    storage_bit: int
    logical_numel: int
    q_shape: Tuple[int, ...]
    slices: Tuple[_PoolSlice, ...]

    @property
    def estimated_bytes(self) -> int:
        q_bits = self.logical_numel * self.bit
        scale_bytes = (self.min.numel() + self.step.numel()) * self.min.element_size()
        id_bytes = self.ids.numel() * self.ids.element_size()
        return (q_bits + 7) // 8 + scale_bytes + id_bytes

    @property
    def storage_bytes(self) -> int:
        return (
            self.q.numel() * self.q.element_size()
            + self.min.numel() * self.min.element_size()
            + self.step.numel() * self.step.element_size()
            + self.ids.numel() * self.ids.element_size()
        )


@dataclass
class _CompressedTensor:
    """一个 K 或 V tensor 的 compressed 表示。"""

    shape: Tuple[int, int, int]
    dtype: torch.dtype
    important: _QuantizedPart
    unimportant: _QuantizedPart

    @property
    def estimated_bytes(self) -> int:
        return self.important.estimated_bytes + self.unimportant.estimated_bytes

    @property
    def storage_bytes(self) -> int:
        return self.important.storage_bytes + self.unimportant.storage_bytes


@dataclass
class _LayerEntry:
    """一个 transformer layer 的 compressed K/V。"""

    k: _CompressedTensor
    v: _CompressedTensor
    original_bytes: int
    estimated_compressed_bytes: int
    storage_bytes: int


@dataclass
class _CompressedEntry:
    """一个 radix node 对应的完整 compressed KV entry。"""

    entry_id: int
    node_uuid: int
    token_ids: torch.Tensor
    length: int
    old_indices: torch.Tensor
    layers: Dict[int, _LayerEntry]
    original_bytes: int
    estimated_compressed_bytes: int
    storage_bytes: int
    created_time: float
    last_access_time: float
    hit_count: int = 0


@dataclass(frozen=True)
class _V3MaterializedHandle(BaseCacheHandle):
    """v3 prefix 命中后返回给 scheduler 的临时 handle。

    radix tree 里的 compressed node 不会永久恢复成 normal node。v3 只为当前
    请求分配一段 normal KV page，把 compressed pool 里的内容解压进去，然后
    通过这个 handle 把临时 indices 写入请求的 page_table。
    """

    base_handle: Any
    matched_indices: torch.Tensor
    temporary_indices: torch.Tensor

    def get_matched_indices(self) -> torch.Tensor:
        return self.matched_indices


@dataclass(frozen=True)
class _V3OwnedHandle(BaseCacheHandle):
    """请求 decode 期间持有临时 restored page 的 wrapper handle。"""

    base_handle: Any
    temporary_indices: torch.Tensor

    @property
    def node(self) -> Any:
        return self.base_handle.node

    def get_matched_indices(self) -> torch.Tensor:
        return self.base_handle.get_matched_indices()


class _SegmentAllocator:
    """固定 buffer 内的简单 first-fit allocator。"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.free_segments: List[Tuple[int, int]] = [(0, capacity)]

    def allocate(self, length: int) -> int | None:
        if length == 0:
            return 0
        for idx, (offset, free_len) in enumerate(self.free_segments):
            if free_len < length:
                continue
            alloc_offset = offset
            remain = free_len - length
            if remain == 0:
                self.free_segments.pop(idx)
            else:
                self.free_segments[idx] = (offset + length, remain)
            return alloc_offset
        return None

    def free(self, offset: int, length: int) -> None:
        if length == 0:
            return
        self.free_segments.append((offset, length))
        self.free_segments.sort()
        merged: List[Tuple[int, int]] = []
        for seg_offset, seg_len in self.free_segments:
            if not merged:
                merged.append((seg_offset, seg_len))
                continue
            last_offset, last_len = merged[-1]
            if last_offset + last_len == seg_offset:
                merged[-1] = (last_offset, last_len + seg_len)
            else:
                merged.append((seg_offset, seg_len))
        self.free_segments = merged

    @property
    def free_size(self) -> int:
        return sum(length for _, length in self.free_segments)

    @property
    def used_size(self) -> int:
        return self.capacity - self.free_size


class _V3CompressedPool:
    """ZipCache v3 固定大小 GPU compressed pool。

    v3 把量化值拆成 q4/q2 两个 packed uint8 buffer：
    - bit <= 2 的 token 进入 q2 buffer；
    - bit > 2 且 bit <= 4 的 token 进入 q4 buffer；
    - min/step 和 token ids 分别进入 scale/ids buffer。
    """

    def __init__(
        self,
        total_bytes: int,
        device: torch.device,
        *,
        q4_ratio: float,
        q2_ratio: float,
        scale_ratio: float,
        ids_ratio: float,
    ):
        total_bytes = max(total_bytes, 1024 * 1024)
        ratios = [
            max(q4_ratio, 0.0),
            max(q2_ratio, 0.0),
            max(scale_ratio, 0.0),
            max(ids_ratio, 0.0),
        ]
        ratio_sum = sum(ratios)
        if ratio_sum <= 0:
            ratios = [0.45, 0.15, 0.25, 0.15]
            ratio_sum = sum(ratios)
        ratios = [r / ratio_sum for r in ratios]

        q4_bytes = max(int(total_bytes * ratios[0]), 1)
        q2_bytes = max(int(total_bytes * ratios[1]), 1)
        scale_bytes = max(int(total_bytes * ratios[2]), 2)
        ids_bytes = max(total_bytes - q4_bytes - q2_bytes - scale_bytes, 8)

        self.q4_buffer = torch.empty(q4_bytes, dtype=torch.uint8, device=device)
        self.q2_buffer = torch.empty(q2_bytes, dtype=torch.uint8, device=device)
        self.scale_buffer = torch.empty(scale_bytes // 2, dtype=torch.float16, device=device)
        self.ids_buffer = torch.empty(ids_bytes // 8, dtype=torch.long, device=device)

        self.q4_allocator = _SegmentAllocator(self.q4_buffer.numel())
        self.q2_allocator = _SegmentAllocator(self.q2_buffer.numel())
        self.scale_allocator = _SegmentAllocator(self.scale_buffer.numel())
        self.ids_allocator = _SegmentAllocator(self.ids_buffer.numel())
        self.capacity_bytes = (
            self.q4_buffer.numel() * self.q4_buffer.element_size()
            + self.q2_buffer.numel() * self.q2_buffer.element_size()
            + self.scale_buffer.numel() * self.scale_buffer.element_size()
            + self.ids_buffer.numel() * self.ids_buffer.element_size()
        )

    def allocate_part(
        self,
        *,
        ids: torch.Tensor,
        q: torch.Tensor,
        min_val: torch.Tensor,
        step: torch.Tensor,
        bit: int,
    ) -> _QuantizedPart:
        if bit > 4:
            raise ValueError("ZipCacheV3 packed pool supports bit width <= 4")

        allocated: List[_PoolSlice] = []
        storage_bit = 2 if bit <= 2 else 4
        q_buffer = self.q2_buffer if storage_bit == 2 else self.q4_buffer
        q_allocator = self.q2_allocator if storage_bit == 2 else self.q4_allocator
        q_buffer_name = "q2" if storage_bit == 2 else "q4"

        ids_flat = ids.reshape(-1).to(device=self.ids_buffer.device, dtype=torch.long)
        q_shape = tuple(q.shape)
        q_flat = q.reshape(-1).to(device=q_buffer.device, dtype=torch.uint8)
        logical_numel = q_flat.numel()
        q_packed = _pack_lowbit(q_flat, storage_bit)
        min_flat = min_val.reshape(-1).to(device=self.scale_buffer.device, dtype=torch.float16)
        step_flat = step.reshape(-1).to(device=self.scale_buffer.device, dtype=torch.float16)

        try:
            ids_offset = self._allocate(self.ids_allocator, "ids", ids_flat.numel(), allocated)
            q_offset = self._allocate(q_allocator, q_buffer_name, q_packed.numel(), allocated)
            min_offset = self._allocate(self.scale_allocator, "scale", min_flat.numel(), allocated)
            step_offset = self._allocate(self.scale_allocator, "scale", step_flat.numel(), allocated)

            ids_view = self.ids_buffer[ids_offset : ids_offset + ids_flat.numel()]
            q_view = q_buffer[q_offset : q_offset + q_packed.numel()]
            min_view = self.scale_buffer[min_offset : min_offset + min_flat.numel()].view(
                min_val.shape
            )
            step_view = self.scale_buffer[step_offset : step_offset + step_flat.numel()].view(
                step.shape
            )
            ids_view.copy_(ids_flat)
            q_view.copy_(q_packed)
            min_view.reshape(-1).copy_(min_flat)
            step_view.reshape(-1).copy_(step_flat)
            return _QuantizedPart(
                ids=ids_view,
                q=q_view,
                min=min_view,
                step=step_view,
                bit=bit,
                storage_bit=storage_bit,
                logical_numel=logical_numel,
                q_shape=q_shape,
                slices=tuple(allocated),
            )
        except Exception:
            self.free_slices(allocated)
            raise

    def free_part(self, part: _QuantizedPart) -> None:
        self.free_slices(part.slices)

    def free_slices(self, slices: Tuple[_PoolSlice, ...] | List[_PoolSlice]) -> None:
        for pool_slice in slices:
            if pool_slice.buffer_name == "q4":
                self.q4_allocator.free(pool_slice.offset, pool_slice.length)
            elif pool_slice.buffer_name == "q2":
                self.q2_allocator.free(pool_slice.offset, pool_slice.length)
            elif pool_slice.buffer_name == "scale":
                self.scale_allocator.free(pool_slice.offset, pool_slice.length)
            elif pool_slice.buffer_name == "ids":
                self.ids_allocator.free(pool_slice.offset, pool_slice.length)

    def stats(self) -> Dict[str, int | float]:
        q4_used = self.q4_allocator.used_size * self.q4_buffer.element_size()
        q2_used = self.q2_allocator.used_size * self.q2_buffer.element_size()
        scale_used = self.scale_allocator.used_size * self.scale_buffer.element_size()
        ids_used = self.ids_allocator.used_size * self.ids_buffer.element_size()
        used_bytes = q4_used + q2_used + scale_used + ids_used
        return {
            "compressed_pool_capacity_bytes": self.capacity_bytes,
            "compressed_pool_used_bytes": used_bytes,
            "compressed_pool_free_bytes": self.capacity_bytes - used_bytes,
            "compressed_pool_utilization": used_bytes / self.capacity_bytes,
            "compressed_pool_q_used_bytes": q4_used + q2_used,
            "compressed_pool_q4_used_bytes": q4_used,
            "compressed_pool_q2_used_bytes": q2_used,
            "compressed_pool_scale_used_bytes": scale_used,
            "compressed_pool_ids_used_bytes": ids_used,
            "compressed_pool_q4_capacity_bytes": self.q4_buffer.numel()
            * self.q4_buffer.element_size(),
            "compressed_pool_q2_capacity_bytes": self.q2_buffer.numel()
            * self.q2_buffer.element_size(),
            "compressed_pool_scale_capacity_bytes": self.scale_buffer.numel()
            * self.scale_buffer.element_size(),
            "compressed_pool_ids_capacity_bytes": self.ids_buffer.numel()
            * self.ids_buffer.element_size(),
        }

    def _allocate(
        self,
        allocator: _SegmentAllocator,
        buffer_name: str,
        length: int,
        allocated: List[_PoolSlice],
    ) -> int:
        offset = allocator.allocate(length)
        if offset is None:
            raise RuntimeError(f"ZipCacheV3 compressed pool is full: buffer={buffer_name}")
        allocated.append(_PoolSlice(buffer_name, offset, length))
        return offset


class ZipCacheV3Manager:
    """ZipCache v3 GPU prefix compression manager.

    v3 让 compressed pool 成为长期 KV archive，normal KV pool 只作为当前请求
    attention 前的 fp16/bf16 工作区：
    - 请求结束后，把 radix prefix cache 中 ref_count == 0 的 normal node 压缩；
    - 压缩成功后释放原 normal page，避免 normal pool 泄漏；
    - 未来命中 compressed radix node 时，临时解压到 normal pool；
    - page_table 仍然只写 normal pool indices，所以不改 attention kernel。
    """

    def __init__(self, config: Any, kv_pool: Any, page_table: torch.Tensor):
        self.config = config
        self.kv_pool = kv_pool
        self.page_table = page_table
        self.original_kv_pool_bytes = _estimate_kv_pool_bytes(kv_pool)
        self.pool = _V3CompressedPool(
            total_bytes=self._choose_compressed_pool_bytes(),
            device=kv_pool.device,
            q4_ratio=self._pool_ratio("q4", 0.45),
            q2_ratio=self._pool_ratio("q2", 0.15),
            scale_ratio=self._pool_ratio("scale", 0.25),
            ids_ratio=self._pool_ratio("ids", 0.15),
        )
        self.entries: Dict[int, _CompressedEntry] = {}
        self.entry_by_node_uuid: Dict[int, int] = {}
        self._next_entry_id = 1
        self._last_stats_log = 0.0
        self._released_handle_refs: Dict[int, weakref.ReferenceType[Any]] = {}
        self._stats: Dict[str, int | float] = {
            "num_demotions": 0,
            "num_demote_failures": 0,
            "num_compressed_entries": 0,
            "num_compressed_hits": 0,
            "num_restore_attempts": 0,
            "num_restore_success": 0,
            "num_restore_fallback": 0,
            "num_compressed_freed": 0,
            "num_temporary_restore_pages": 0,
            "num_restore_pages_released": 0,
            "num_restore_rejected_small_prefix": 0,
            "original_estimated_bytes": 0,
            "compressed_estimated_bytes_4bit": 0,
            "compressed_storage_bytes": 0,
            "active_original_estimated_bytes": 0,
            "active_compressed_estimated_bytes_4bit": 0,
            "active_compressed_storage_bytes": 0,
            "last_estimated_compression_ratio": 1.0,
            "last_storage_compression_ratio": 1.0,
            "num_demote_rejected_pool_full": 0,
        }
        logger.info_rank0(
            "[ZipCacheV3] compressed pool initialized: capacity=%s bytes, "
            "normal_kv_pool=%s bytes",
            self.pool.capacity_bytes,
            self.original_kv_pool_bytes,
        )

    def enabled(self) -> bool:
        return bool(getattr(self.config, "enable_zipcache_v3", False))

    def before_attention(self, **_: Any) -> None:
        """v3 不在 attention 热路径前做额外处理。"""

    def after_attention(self, **_: Any) -> None:
        """v3 不在 attention 热路径后压缩活跃请求。"""

    def free_request(self, req_uid: int) -> None:
        """compressed prefix 不按请求 uid 释放，保留给未来 shared-prefix 命中。"""

    def demote_node(self, node: Any) -> torch.Tensor | None:
        """把一个 radix node 的 normal KV page 压缩到 GPU compressed pool。

        成功时返回原 normal indices，调用者负责把这些 page 归还 normal pool。
        失败时返回 None，调用者保持原 miniSGLang 行为。
        """

        if not self.enabled() or node.is_root() or node.is_compressed:
            return None
        if node.ref_count != 0 or node.length <= 0:
            return None

        layers: Dict[int, _LayerEntry] = {}
        try:
            indices = node.value.detach().to(self.kv_pool.device, non_blocking=True)
            token_ids = node._key.detach().to("cpu", non_blocking=True)
            original_total = 0
            estimated_total = 0
            storage_total = 0
            for layer_id in range(self.kv_pool.num_layers):
                flat_k = _flatten_layer_cache(self.kv_pool.k_cache(layer_id))
                flat_v = _flatten_layer_cache(self.kv_pool.v_cache(layer_id))
                k_tensor = flat_k[indices]
                v_tensor = flat_v[indices]
                unimportant = _select_unimportant_ids(
                    k_tensor,
                    v_tensor,
                    ratio=float(self.config.zipcache_unimportant_ratio),
                    protect_recent=int(self.config.zipcache_protect_recent_tokens),
                )
                compressed_k = None
                try:
                    compressed_k = _quantize_mixed_gpu(
                        k_tensor,
                        unimportant,
                        important_bit=int(self.config.zipcache_k_important_bit),
                        unimportant_bit=int(self.config.zipcache_k_unimportant_bit),
                        pool=self.pool,
                    )
                    compressed_v = _quantize_mixed_gpu(
                        v_tensor,
                        unimportant,
                        important_bit=int(self.config.zipcache_v_important_bit),
                        unimportant_bit=int(self.config.zipcache_v_unimportant_bit),
                        pool=self.pool,
                    )
                except Exception:
                    if compressed_k is not None:
                        self.pool.free_part(compressed_k.important)
                        self.pool.free_part(compressed_k.unimportant)
                    raise

                original_bytes = (k_tensor.numel() + v_tensor.numel()) * k_tensor.element_size()
                estimated_bytes = compressed_k.estimated_bytes + compressed_v.estimated_bytes
                storage_bytes = compressed_k.storage_bytes + compressed_v.storage_bytes
                layers[layer_id] = _LayerEntry(
                    k=compressed_k,
                    v=compressed_v,
                    original_bytes=original_bytes,
                    estimated_compressed_bytes=estimated_bytes,
                    storage_bytes=storage_bytes,
                )
                original_total += original_bytes
                estimated_total += estimated_bytes
                storage_total += storage_bytes

            entry_id = self._next_entry_id
            self._next_entry_id += 1
            entry = _CompressedEntry(
                entry_id=entry_id,
                node_uuid=node.uuid,
                token_ids=token_ids,
                length=int(node.length),
                old_indices=indices.detach().clone(),
                layers=layers,
                original_bytes=original_total,
                estimated_compressed_bytes=estimated_total,
                storage_bytes=storage_total,
                created_time=time.time(),
                last_access_time=time.time(),
            )
            self.entries[entry_id] = entry
            self.entry_by_node_uuid[node.uuid] = entry_id
            self._stats["num_demotions"] += 1
            self._stats["num_compressed_entries"] = len(self.entries)
            self._stats["original_estimated_bytes"] += original_total
            self._stats["compressed_estimated_bytes_4bit"] += estimated_total
            self._stats["compressed_storage_bytes"] += storage_total
            self._stats["last_estimated_compression_ratio"] = (
                original_total / estimated_total if estimated_total > 0 else 1.0
            )
            self._stats["last_storage_compression_ratio"] = (
                original_total / storage_total if storage_total > 0 else 1.0
            )
            self._update_active_stats()
            logger.debug_rank0(
                "[ZipCacheV3] demoted: entry_id=%s node=%s tokens=%s "
                "original=%s estimated_4bit=%s gpu_storage=%s",
                entry_id,
                node.uuid,
                node.length,
                original_total,
                estimated_total,
                storage_total,
            )
            return indices
        except Exception as exc:
            self._free_layers(layers)
            self._stats["num_demote_failures"] += 1
            if "compressed pool is full" in str(exc):
                self._stats["num_demote_rejected_pool_full"] += 1
            logger.exception("[ZipCacheV3] demote failed: node=%s", getattr(node, "uuid", None))
            return None

    def materialize_match(self, prefix_cache: Any, handle: Any, cache_manager: Any) -> Any:
        """把 compressed radix 命中临时恢复到 normal pool。

        返回的 handle 只属于当前请求。radix tree 节点仍保持 compressed 状态，
        后续请求仍然可以继续命中 compressed entry。
        """

        if not self.enabled() or handle.cached_len == 0:
            return handle
        try:
            from minisgl.kvcache.radix_cache import RadixCacheHandle

            nodes = prefix_cache.path_nodes(handle)
            if not any(node.is_compressed for node in nodes):
                return handle

            min_restore = int(getattr(self.config, "zipcache_v3_min_restore_tokens", 0))
            keep_compressed = bool(
                getattr(self.config, "zipcache_v3_keep_compressed_after_restore", True)
            )
            materialized_len = 0
            last_safe_node = prefix_cache.root_node
            matched_parts: List[torch.Tensor] = []
            temporary_parts: List[torch.Tensor] = []

            for node in nodes:
                if not node.is_compressed:
                    matched_parts.append(node.value)
                    materialized_len += node.length
                    last_safe_node = node
                    continue

                if node.length < min_restore:
                    self._stats["num_restore_rejected_small_prefix"] += 1
                    self._stats["num_restore_fallback"] += 1
                    logger.debug_rank0(
                        "[ZipCacheV3] restore skipped: node=%s tokens=%s min_restore=%s",
                        node.uuid,
                        node.length,
                        min_restore,
                    )
                    self._release_temporary_parts(cache_manager, temporary_parts)
                    return RadixCacheHandle(materialized_len, last_safe_node)

                entry_id = node.compressed_id or self.entry_by_node_uuid.get(node.uuid)
                entry = self.entries.get(entry_id) if entry_id is not None else None
                if entry is None:
                    self._stats["num_restore_fallback"] += 1
                    logger.warning_rank0(
                        "[ZipCacheV3] restore fallback: missing entry node=%s", node.uuid
                    )
                    self._release_temporary_parts(cache_manager, temporary_parts)
                    return RadixCacheHandle(materialized_len, last_safe_node)

                self._stats["num_compressed_hits"] += 1
                self._stats["num_restore_attempts"] += 1
                new_indices = cache_manager.allocate_token_indices(node.length)
                try:
                    self._restore_entry_to_indices(entry, new_indices)
                except Exception:
                    cache_manager._free(new_indices)
                    raise

                matched_parts.append(new_indices)
                if keep_compressed:
                    temporary_parts.append(new_indices)
                else:
                    prefix_cache.mark_node_restored(node, new_indices)
                    self._free_entry(entry.entry_id)
                materialized_len += node.length
                last_safe_node = node
                self._stats["num_restore_success"] += 1
                restored_pages = int(
                    (len(new_indices) + cache_manager.page_size - 1)
                    // cache_manager.page_size
                )
                if keep_compressed:
                    self._stats["num_temporary_restore_pages"] += restored_pages
                    logger.debug_rank0(
                        "[ZipCacheV3] restored temporary: entry_id=%s node=%s tokens=%s",
                        entry.entry_id,
                        node.uuid,
                        node.length,
                    )
                else:
                    logger.debug_rank0(
                        "[ZipCacheV3] restored permanent: entry_id=%s node=%s tokens=%s",
                        entry.entry_id,
                        node.uuid,
                        node.length,
                    )

            if not matched_parts:
                return RadixCacheHandle(0, prefix_cache.root_node)
            if not temporary_parts:
                return handle
            return _V3MaterializedHandle(
                cached_len=materialized_len,
                base_handle=handle,
                matched_indices=torch.cat(matched_parts),
                temporary_indices=torch.cat(temporary_parts),
            )
        except Exception:
            if "temporary_parts" in locals():
                self._release_temporary_parts(cache_manager, temporary_parts)
            self._stats["num_restore_fallback"] += 1
            logger.exception("[ZipCacheV3] restore failed; fallback to recompute")
            try:
                from minisgl.kvcache.radix_cache import RadixCacheHandle

                return RadixCacheHandle(0, prefix_cache.root_node)
            except Exception:
                return handle

    def lock_handle(self, prefix_cache: Any, handle: Any, *, unlock: bool = False) -> bool:
        if isinstance(handle, (_V3MaterializedHandle, _V3OwnedHandle)):
            prefix_cache.lock_handle(handle.base_handle, unlock=unlock)
            return True
        return False

    def carry_handle_resources(self, old_handle: Any, new_handle: Any) -> Any:
        """prefill 后把临时 restored page 从 match handle 转交给 decode handle。"""

        if isinstance(old_handle, (_V3MaterializedHandle, _V3OwnedHandle)):
            return _V3OwnedHandle(
                cached_len=new_handle.cached_len,
                base_handle=new_handle,
                temporary_indices=old_handle.temporary_indices,
            )
        return new_handle

    def release_handle_resources(self, handle: Any, cache_manager: Any) -> None:
        if not isinstance(handle, (_V3MaterializedHandle, _V3OwnedHandle)):
            return
        handle_id = id(handle)
        released_ref = self._released_handle_refs.get(handle_id)
        if released_ref is not None:
            released_handle = released_ref()
            if released_handle is handle:
                return
            if released_handle is None:
                self._released_handle_refs.pop(handle_id, None)
        self._released_handle_refs[handle_id] = weakref.ref(handle)
        if handle.temporary_indices.numel() == 0:
            return
        cache_manager._free(handle.temporary_indices)
        released_pages = int(
            (len(handle.temporary_indices) + cache_manager.page_size - 1)
            // cache_manager.page_size
        )
        self._stats["num_restore_pages_released"] += released_pages
        logger.debug_rank0(
            "[ZipCacheV3] released temporary restore pages: tokens=%s pages=%s",
            len(handle.temporary_indices),
            released_pages,
        )

    def stats(self) -> Dict[str, int | float]:
        self._update_active_stats()
        stats = dict(self._stats)
        estimated = int(stats["active_compressed_estimated_bytes_4bit"])
        storage = int(stats["active_compressed_storage_bytes"])
        original = int(stats["active_original_estimated_bytes"])
        stats["active_estimated_compression_ratio"] = (
            original / estimated if estimated > 0 else 1.0
        )
        stats["active_storage_compression_ratio"] = original / storage if storage > 0 else 1.0
        if torch.cuda.is_available():
            device = self.kv_pool.device
            stats["gpu_memory_allocated_bytes"] = torch.cuda.memory_allocated(device)
            stats["gpu_memory_reserved_bytes"] = torch.cuda.memory_reserved(device)
            stats["gpu_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            stats["gpu_max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        stats.update(self.pool.stats())
        stats["normal_pool_capacity_bytes"] = self.original_kv_pool_bytes
        stats["estimated_effective_kv_capacity_bytes"] = (
            self.original_kv_pool_bytes
            + self.pool.capacity_bytes
            * float(stats.get("active_storage_compression_ratio", 1.0))
        )
        stats["estimated_capacity_gain_vs_normal_pool"] = (
            stats["estimated_effective_kv_capacity_bytes"] / self.original_kv_pool_bytes
            if self.original_kv_pool_bytes > 0
            else 1.0
        )
        return stats

    def log_stats(self) -> None:
        if self.enabled():
            logger.info_rank0("[ZipCacheV3] stats: %s", self.stats())

    def maybe_log_stats(self) -> None:
        interval = float(self.config.zipcache_stats_interval)
        if interval <= 0:
            return
        now = time.monotonic()
        if now - self._last_stats_log >= interval:
            self._last_stats_log = now
            self.log_stats()

    def _restore_entry_to_indices(self, entry: _CompressedEntry, indices: torch.Tensor) -> None:
        indices = indices.to(self.kv_pool.device, non_blocking=True)
        for layer_id, layer_entry in entry.layers.items():
            flat_k = _flatten_layer_cache(self.kv_pool.k_cache(layer_id))
            flat_v = _flatten_layer_cache(self.kv_pool.v_cache(layer_id))
            flat_k[indices] = _dequantize_mixed_gpu(layer_entry.k, self.kv_pool.device)
            flat_v[indices] = _dequantize_mixed_gpu(layer_entry.v, self.kv_pool.device)
        entry.hit_count += 1
        entry.last_access_time = time.time()

    def _free_entry(self, entry_id: int) -> None:
        entry = self.entries.pop(entry_id, None)
        if entry is None:
            return
        self.entry_by_node_uuid.pop(entry.node_uuid, None)
        self._stats["num_compressed_freed"] += 1
        self._stats["num_compressed_entries"] = len(self.entries)
        self._free_layers(entry.layers)
        self._update_active_stats()

    def _free_layers(self, layers: Dict[int, _LayerEntry]) -> None:
        for layer_entry in layers.values():
            self.pool.free_part(layer_entry.k.important)
            self.pool.free_part(layer_entry.k.unimportant)
            self.pool.free_part(layer_entry.v.important)
            self.pool.free_part(layer_entry.v.unimportant)

    def _update_active_stats(self) -> None:
        self._stats["active_original_estimated_bytes"] = sum(
            entry.original_bytes for entry in self.entries.values()
        )
        self._stats["active_compressed_estimated_bytes_4bit"] = sum(
            entry.estimated_compressed_bytes for entry in self.entries.values()
        )
        self._stats["active_compressed_storage_bytes"] = sum(
            entry.storage_bytes for entry in self.entries.values()
        )

    def _choose_compressed_pool_bytes(self) -> int:
        pool_mb = int(getattr(self.config, "zipcache_v3_compressed_pool_mb", 0))
        if pool_mb > 0:
            return pool_mb * 1024 * 1024
        ratio = float(getattr(self.config, "zipcache_v3_compressed_pool_ratio", 1.0))
        ratio = max(ratio, 0.01)
        return int(self.original_kv_pool_bytes * ratio)

    def _pool_ratio(self, name: str, default: float) -> float:
        return float(getattr(self.config, f"zipcache_v3_{name}_pool_ratio", default))

    def _release_temporary_parts(self, cache_manager: Any, parts: List[torch.Tensor]) -> None:
        if not parts:
            return
        indices = torch.cat(parts)
        cache_manager._free(indices)
        self._stats["num_restore_pages_released"] += int(
            (len(indices) + cache_manager.page_size - 1) // cache_manager.page_size
        )


def _flatten_layer_cache(cache: torch.Tensor) -> torch.Tensor:
    return cache.view(-1, cache.shape[-2], cache.shape[-1])


def _estimate_kv_pool_bytes(kv_pool: Any) -> int:
    per_layer = kv_pool.k_cache(0).numel() + kv_pool.v_cache(0).numel()
    return int(per_layer * kv_pool.num_layers * kv_pool.dtype.itemsize)


def _select_unimportant_ids(
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    *,
    ratio: float,
    protect_recent: int,
) -> torch.Tensor:
    length = k_tensor.shape[0]
    if length <= 1 or ratio <= 0:
        return torch.empty(0, dtype=torch.long, device=k_tensor.device)
    num_unimportant = int(length * ratio)
    protect_recent = min(max(protect_recent, 0), length)
    num_unimportant = min(num_unimportant, max(length - protect_recent, 0))
    if num_unimportant <= 0:
        return torch.empty(0, dtype=torch.long, device=k_tensor.device)

    scores = k_tensor.float().abs().mean(dim=(1, 2)) + v_tensor.float().abs().mean(
        dim=(1, 2)
    )
    if protect_recent > 0:
        scores[-protect_recent:] = torch.inf
    return torch.topk(scores, k=num_unimportant, largest=False).indices


def _split_ids(
    length: int, unimportant_ids: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    if unimportant_ids.numel() == 0:
        important = torch.arange(length, device=device, dtype=torch.long)
        return important, unimportant_ids.to(device=device, dtype=torch.long)
    unimportant = torch.unique(unimportant_ids.to(device=device, dtype=torch.long))
    unimportant = unimportant[(unimportant >= 0) & (unimportant < length)]
    mask = torch.ones(length, device=device, dtype=torch.bool)
    mask[unimportant] = False
    important = torch.arange(length, device=device, dtype=torch.long)[mask]
    return important, unimportant


def _quantize_mixed_gpu(
    x: torch.Tensor,
    unimportant_ids: torch.Tensor,
    *,
    important_bit: int,
    unimportant_bit: int,
    pool: _V3CompressedPool,
) -> _CompressedTensor:
    important_ids, unimportant_ids = _split_ids(x.shape[0], unimportant_ids, x.device)
    important = None
    try:
        important = _quantize_part_gpu(x, important_ids, important_bit, pool)
        unimportant = _quantize_part_gpu(x, unimportant_ids, unimportant_bit, pool)
    except Exception:
        if important is not None:
            pool.free_part(important)
        raise
    return _CompressedTensor(
        shape=tuple(x.shape),  # type: ignore[arg-type]
        dtype=x.dtype,
        important=important,
        unimportant=unimportant,
    )


def _quantize_part_gpu(
    x: torch.Tensor,
    ids: torch.Tensor,
    bit: int,
    pool: _V3CompressedPool,
) -> _QuantizedPart:
    if bit > 4:
        raise ValueError("ZipCacheV3 packed pool supports bit width <= 4")
    ids = ids.to(device=x.device, dtype=torch.long)
    if ids.numel() == 0:
        empty_q = torch.empty(0, dtype=torch.uint8, device=x.device)
        empty_scale = torch.empty(0, dtype=torch.float16, device=x.device)
        return pool.allocate_part(
            ids=ids,
            q=empty_q,
            min_val=empty_scale,
            step=empty_scale,
            bit=bit,
        )

    selected = x[ids].float()
    qmax = float((1 << bit) - 1)
    min_val = selected.amin(dim=-1, keepdim=True)
    max_val = selected.amax(dim=-1, keepdim=True)
    step = ((max_val - min_val) / qmax).clamp_min(1e-6)
    q = torch.round((selected - min_val) / step).clamp_(0, qmax).to(torch.uint8)
    return pool.allocate_part(ids=ids, q=q, min_val=min_val, step=step, bit=bit)


def _dequantize_mixed_gpu(data: _CompressedTensor, device: torch.device) -> torch.Tensor:
    out = torch.empty(data.shape, dtype=data.dtype, device=device)
    _dequantize_part_gpu_into(out, data.important, data.dtype)
    _dequantize_part_gpu_into(out, data.unimportant, data.dtype)
    return out


def _dequantize_part_gpu_into(
    out: torch.Tensor, part: _QuantizedPart, dtype: torch.dtype
) -> None:
    if part.ids.numel() == 0:
        return
    q = _unpack_lowbit(part.q, part.storage_bit, part.logical_numel, part.q_shape)
    out[part.ids] = (q.float() * part.step.float() + part.min.float()).to(dtype)


def _pack_lowbit(q: torch.Tensor, storage_bit: int) -> torch.Tensor:
    """把 uint8 量化值按 2bit 或 4bit 打包到 uint8 buffer。"""

    if storage_bit not in (2, 4):
        raise ValueError(f"Unsupported packed bit width: {storage_bit}")
    mask = (1 << storage_bit) - 1
    values_per_byte = 8 // storage_bit
    q = q.reshape(-1).to(torch.uint8).clamp_(0, mask)
    if q.numel() == 0:
        return q
    pad = (-q.numel()) % values_per_byte
    if pad:
        q = torch.cat([q, torch.zeros(pad, dtype=torch.uint8, device=q.device)])
    chunks = q.view(-1, values_per_byte).to(torch.int16)
    shifts = (
        torch.arange(values_per_byte, device=q.device, dtype=torch.int16) * storage_bit
    )
    return ((chunks << shifts).sum(dim=1) & 0xFF).to(torch.uint8)


def _unpack_lowbit(
    packed: torch.Tensor,
    storage_bit: int,
    logical_numel: int,
    shape: Tuple[int, ...],
) -> torch.Tensor:
    """把 2bit/4bit packed uint8 buffer 解包成原量化值 tensor。"""

    if storage_bit not in (2, 4):
        raise ValueError(f"Unsupported packed bit width: {storage_bit}")
    if logical_numel == 0:
        return torch.empty(shape, dtype=torch.uint8, device=packed.device)
    values_per_byte = 8 // storage_bit
    mask = (1 << storage_bit) - 1
    shifts = (
        torch.arange(values_per_byte, device=packed.device, dtype=torch.int16) * storage_bit
    )
    packed_i16 = packed.to(torch.int16).unsqueeze(1)
    out = ((packed_i16 >> shifts) & mask).to(torch.uint8).reshape(-1)
    return out[:logical_numel].view(shape)
