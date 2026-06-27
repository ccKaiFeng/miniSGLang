from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.utils import init_logger

logger = init_logger(__name__)


@dataclass
class _QuantizedPart:
    ids: torch.Tensor
    q: torch.Tensor
    min: torch.Tensor
    step: torch.Tensor
    bit: int

    @property
    def estimated_bytes(self) -> int:
        q_bits = self.q.numel() * self.bit
        scale_bytes = (self.min.numel() + self.step.numel()) * self.min.element_size()
        return (q_bits + 7) // 8 + scale_bytes


@dataclass
class _CompressedTensor:
    shape: Tuple[int, int, int]
    dtype: torch.dtype
    important: _QuantizedPart
    unimportant: _QuantizedPart

    @property
    def estimated_bytes(self) -> int:
        return self.important.estimated_bytes + self.unimportant.estimated_bytes


@dataclass
class _LayerEntry:
    req_uid: int
    layer_id: int
    indices: torch.Tensor
    k: _CompressedTensor
    v: _CompressedTensor
    seq_len: int
    unimportant_ids: torch.Tensor
    original_bytes: int
    compressed_bytes: int
    created_time: float
    last_access_time: float


class ZipCacheV1Manager:
    """ZipCache v1 runtime manager for miniSGLang.

    v1 does not change attention kernels or the paged KV table format. It stores a
    compressed CPU copy of each active request/layer after attention, and restores
    it back to the original GPU KV pool before the next attention call.
    """

    def __init__(self, config: Any, kv_pool: Any, page_table: torch.Tensor):
        self.config = config
        self.kv_pool = kv_pool
        self.page_table = page_table
        self.entries: Dict[Tuple[int, int], _LayerEntry] = {}
        self.layer_saliency: Dict[Tuple[int, int], torch.Tensor] = {}
        self.layer_steps: Dict[Tuple[int, int], int] = {}
        self._last_stats_log = 0.0
        self._stats: Dict[str, int | float] = {
            "num_probe_runs": 0,
            "num_probe_tokens": 0,
            "num_salient_updates": 0,
            "num_compressions": 0,
            "num_decompressions": 0,
            "num_restore_failures": 0,
            "num_freed_entries": 0,
            "original_estimated_bytes": 0,
            "compressed_estimated_bytes": 0,
            "active_original_estimated_bytes": 0,
            "active_compressed_estimated_bytes": 0,
            "max_active_original_estimated_bytes": 0,
            "max_active_compressed_estimated_bytes": 0,
            "last_compression_ratio": 1.0,
        }

    def enabled(self) -> bool:
        return bool(getattr(self.config, "enable_zipcache_v1", False))

    def before_attention(
        self,
        *,
        q: torch.Tensor,
        layer_id: int,
        batch: Batch,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> None:
        if not self.enabled():
            return
        flat_k = _flatten_layer_cache(k_cache)
        flat_v = _flatten_layer_cache(v_cache)
        self._restore_batch(layer_id, batch, flat_k, flat_v)
        self._update_saliency(q, layer_id, batch, flat_k)

    def after_attention(
        self,
        *,
        layer_id: int,
        batch: Batch,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> None:
        if not self.enabled():
            return
        flat_k = _flatten_layer_cache(k_cache)
        flat_v = _flatten_layer_cache(v_cache)
        for req in batch.reqs:
            if req.uid < 0:
                continue
            self._compress_req_layer(req, layer_id, flat_k, flat_v)
        self._update_active_stats()
        self._maybe_log_stats()

    def free_request(self, req_uid: int) -> None:
        if not self.enabled():
            return
        stale_keys = [key for key in self.entries if key[0] == req_uid]
        for key in stale_keys:
            del self.entries[key]
            self._stats["num_freed_entries"] += 1
        stale_saliency = [key for key in self.layer_saliency if key[0] == req_uid]
        for key in stale_saliency:
            del self.layer_saliency[key]
        stale_steps = [key for key in self.layer_steps if key[0] == req_uid]
        for key in stale_steps:
            del self.layer_steps[key]
        self._update_active_stats()

    def stats(self) -> Dict[str, int | float]:
        self._update_active_stats()
        stats = dict(self._stats)
        active_compressed = int(stats["active_compressed_estimated_bytes"])
        active_original = int(stats["active_original_estimated_bytes"])
        stats["active_compression_ratio"] = (
            active_original / active_compressed if active_compressed > 0 else 1.0
        )
        stats["num_active_entries"] = len(self.entries)
        if torch.cuda.is_available():
            device = self.kv_pool.device
            stats["gpu_memory_allocated_bytes"] = torch.cuda.memory_allocated(device)
            stats["gpu_memory_reserved_bytes"] = torch.cuda.memory_reserved(device)
            stats["gpu_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            stats["gpu_max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        return stats

    def log_stats(self) -> None:
        if self.enabled():
            logger.info_rank0("[ZipCacheV1] stats: %s", self.stats())

    def _restore_batch(
        self,
        layer_id: int,
        batch: Batch,
        flat_k: torch.Tensor,
        flat_v: torch.Tensor,
    ) -> None:
        for req in batch.reqs:
            key = (req.uid, layer_id)
            entry = self.entries.get(key)
            if entry is None:
                continue
            try:
                indices = entry.indices.to(flat_k.device, non_blocking=True)
                flat_k[indices] = _dequantize_mixed(entry.k, flat_k.device)
                flat_v[indices] = _dequantize_mixed(entry.v, flat_v.device)
                entry.last_access_time = time.time()
                self._stats["num_decompressions"] += 1
            except Exception:
                self._stats["num_restore_failures"] += 1
                logger.exception("[ZipCacheV1] restore failed: req=%s layer=%s", req.uid, layer_id)

    def _update_saliency(
        self,
        q: torch.Tensor,
        layer_id: int,
        batch: Batch,
        flat_k: torch.Tensor,
    ) -> None:
        q_offset = 0
        for req in batch.padded_reqs:
            extend_len = req.extend_len
            q_req = q[q_offset : q_offset + extend_len]
            q_offset += extend_len
            if req.uid < 0 or extend_len <= 0:
                continue

            key = (req.uid, layer_id)
            step = self.layer_steps.get(key, 0)
            should_probe = extend_len > 1 or key not in self.layer_saliency
            if step > 0 and step % int(self.config.zipcache_streaming_gap) == 0:
                should_probe = True
            self.layer_steps[key] = step + 1
            if not should_probe:
                continue

            indices = self.page_table[req.table_idx, : req.device_len]
            k_seq = flat_k[indices]
            unimportant = self._select_unimportant_ids(q_req, k_seq, req)
            self.layer_saliency[key] = unimportant.detach().to("cpu")
            self._stats["num_salient_updates"] += 1

    def _select_unimportant_ids(
        self, q_req: torch.Tensor, k_seq: torch.Tensor, req: Req
    ) -> torch.Tensor:
        seq_len = k_seq.shape[0]
        if seq_len <= 1 or q_req.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=q_req.device)

        probe_ids = _make_probe_ids(q_req.shape[0], q_req.device)
        probe_q = q_req[probe_ids]
        scores = _normalized_attention_scores(probe_q, k_seq)
        ratio = float(self.config.zipcache_unimportant_ratio)
        num_unimportant = int(seq_len * ratio)
        num_unimportant = max(0, min(num_unimportant, seq_len - 1))
        if num_unimportant == 0:
            return torch.empty(0, dtype=torch.long, device=q_req.device)

        # 保护当前刚写入的尾部 token，避免 decode 马上量化最新 token。
        protect_tail = min(int(self.config.zipcache_protect_recent_tokens), seq_len)
        num_unimportant = min(num_unimportant, max(seq_len - protect_tail, 0))
        if num_unimportant == 0:
            return torch.empty(0, dtype=torch.long, device=q_req.device)
        if protect_tail > 0:
            scores[-protect_tail:] = torch.inf

        unimportant = torch.topk(scores, k=num_unimportant, largest=False).indices
        self._stats["num_probe_runs"] += 1
        self._stats["num_probe_tokens"] += int(len(probe_ids))
        logger.debug_rank0(
            "[ZipCacheV1] probe: req=%s seq=%s probes=%s unimportant=%s",
            req.uid,
            seq_len,
            len(probe_ids),
            len(unimportant),
        )
        return unimportant

    def _compress_req_layer(
        self,
        req: Req,
        layer_id: int,
        flat_k: torch.Tensor,
        flat_v: torch.Tensor,
    ) -> None:
        indices = self.page_table[req.table_idx, : req.device_len]
        if len(indices) == 0:
            return

        key = (req.uid, layer_id)
        unimportant = self.layer_saliency.get(key)
        if unimportant is None:
            unimportant = torch.empty(0, dtype=torch.long)
        unimportant_gpu = unimportant.to(flat_k.device, non_blocking=True)

        k_tensor = flat_k[indices]
        v_tensor = flat_v[indices]
        compressed_k = _quantize_mixed(
            k_tensor,
            unimportant_gpu,
            important_bit=int(self.config.zipcache_k_important_bit),
            unimportant_bit=int(self.config.zipcache_k_unimportant_bit),
        )
        compressed_v = _quantize_mixed(
            v_tensor,
            unimportant_gpu,
            important_bit=int(self.config.zipcache_v_important_bit),
            unimportant_bit=int(self.config.zipcache_v_unimportant_bit),
        )
        original_bytes = (k_tensor.numel() + v_tensor.numel()) * k_tensor.element_size()
        compressed_bytes = compressed_k.estimated_bytes + compressed_v.estimated_bytes
        self.entries[key] = _LayerEntry(
            req_uid=req.uid,
            layer_id=layer_id,
            indices=indices.detach().to("cpu", non_blocking=True),
            k=compressed_k,
            v=compressed_v,
            seq_len=int(req.device_len),
            unimportant_ids=unimportant.detach().to("cpu"),
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
            created_time=time.time(),
            last_access_time=time.time(),
        )
        self._stats["num_compressions"] += 1
        self._stats["original_estimated_bytes"] += original_bytes
        self._stats["compressed_estimated_bytes"] += compressed_bytes
        self._stats["last_compression_ratio"] = (
            original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
        )

    def _update_active_stats(self) -> None:
        original = sum(entry.original_bytes for entry in self.entries.values())
        compressed = sum(entry.compressed_bytes for entry in self.entries.values())
        self._stats["active_original_estimated_bytes"] = original
        self._stats["active_compressed_estimated_bytes"] = compressed
        self._stats["max_active_original_estimated_bytes"] = max(
            int(self._stats["max_active_original_estimated_bytes"]), original
        )
        self._stats["max_active_compressed_estimated_bytes"] = max(
            int(self._stats["max_active_compressed_estimated_bytes"]), compressed
        )

    def _maybe_log_stats(self) -> None:
        interval = float(self.config.zipcache_stats_interval)
        if interval <= 0:
            return
        now = time.monotonic()
        if now - self._last_stats_log >= interval:
            self._last_stats_log = now
            self.log_stats()


def _flatten_layer_cache(cache: torch.Tensor) -> torch.Tensor:
    return cache.view(-1, cache.shape[-2], cache.shape[-1])


def _make_probe_ids(length: int, device: torch.device) -> torch.Tensor:
    if length <= 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    recent = max(1, int(length * 0.05))
    random_count = max(1, int(length * 0.05))
    recent_ids = torch.arange(length - recent, length, device=device, dtype=torch.long)
    prefix_len = max(length - recent, 1)
    rand_ids = torch.randint(0, prefix_len, (random_count,), device=device, dtype=torch.long)
    return torch.unique(torch.cat([recent_ids, rand_ids]), sorted=False)


def _normalized_attention_scores(probe_q: torch.Tensor, k_seq: torch.Tensor) -> torch.Tensor:
    # probe_q: [P, Hq, D], k_seq: [S, Hkv, D]. For GQA, Hq is grouped by Hkv.
    p, hq, d = probe_q.shape
    s, hkv, _ = k_seq.shape
    if hq % hkv != 0:
        probe_q = probe_q[:, :hkv]
        hq = hkv
    groups = hq // hkv
    q_grouped = probe_q.view(p, hkv, groups, d).permute(1, 2, 0, 3)
    k_heads = k_seq.permute(1, 0, 2)
    attn = torch.einsum("hgpd,hsd->hgps", q_grouped.float(), k_heads.float()) * (d**-0.5)
    attn = torch.softmax(attn, dim=-1)
    token_scores = attn.sum(dim=(0, 1, 2))
    normalizer = torch.flip(
        torch.arange(1, s + 1, device=k_seq.device, dtype=token_scores.dtype), dims=(0,)
    )
    return token_scores / normalizer.clamp_min(1)


def _split_ids(length: int, unimportant_ids: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if unimportant_ids.numel() == 0:
        important = torch.arange(length, device=device, dtype=torch.long)
        return important, unimportant_ids.to(device=device, dtype=torch.long)
    unimportant = torch.unique(unimportant_ids.to(device=device, dtype=torch.long))
    unimportant = unimportant[(unimportant >= 0) & (unimportant < length)]
    mask = torch.ones(length, device=device, dtype=torch.bool)
    mask[unimportant] = False
    important = torch.arange(length, device=device, dtype=torch.long)[mask]
    return important, unimportant


def _quantize_mixed(
    x: torch.Tensor,
    unimportant_ids: torch.Tensor,
    *,
    important_bit: int,
    unimportant_bit: int,
) -> _CompressedTensor:
    important_ids, unimportant_ids = _split_ids(x.shape[0], unimportant_ids, x.device)
    important = _quantize_part(x, important_ids, important_bit)
    unimportant = _quantize_part(x, unimportant_ids, unimportant_bit)
    return _CompressedTensor(
        shape=tuple(x.shape),  # type: ignore[arg-type]
        dtype=x.dtype,
        important=important,
        unimportant=unimportant,
    )


def _quantize_part(x: torch.Tensor, ids: torch.Tensor, bit: int) -> _QuantizedPart:
    if ids.numel() == 0:
        empty_q = torch.empty(0, dtype=torch.uint8, device="cpu")
        empty_scale = torch.empty(0, dtype=torch.float16, device="cpu")
        return _QuantizedPart(ids=ids.detach().to("cpu"), q=empty_q, min=empty_scale, step=empty_scale, bit=bit)

    selected = x[ids].float()
    qmax = float((1 << bit) - 1)
    min_val = selected.amin(dim=-1, keepdim=True)
    max_val = selected.amax(dim=-1, keepdim=True)
    step = ((max_val - min_val) / qmax).clamp_min(1e-6)
    q = torch.round((selected - min_val) / step).clamp_(0, qmax).to(torch.uint8)
    return _QuantizedPart(
        ids=ids.detach().to("cpu", non_blocking=True),
        q=q.detach().to("cpu", non_blocking=True),
        min=min_val.to(torch.float16).detach().to("cpu", non_blocking=True),
        step=step.to(torch.float16).detach().to("cpu", non_blocking=True),
        bit=bit,
    )


def _dequantize_mixed(data: _CompressedTensor, device: torch.device) -> torch.Tensor:
    out = torch.empty(data.shape, dtype=data.dtype, device=device)
    _dequantize_part_into(out, data.important, data.dtype, device)
    _dequantize_part_into(out, data.unimportant, data.dtype, device)
    return out


def _dequantize_part_into(
    out: torch.Tensor, part: _QuantizedPart, dtype: torch.dtype, device: torch.device
) -> None:
    if part.ids.numel() == 0:
        return
    ids = part.ids.to(device, non_blocking=True)
    q = part.q.to(device, non_blocking=True).float()
    min_val = part.min.to(device, non_blocking=True).float()
    step = part.step.to(device, non_blocking=True).float()
    out[ids] = (q * step + min_val).to(dtype)
