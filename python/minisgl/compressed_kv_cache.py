from __future__ import annotations

# Compressed KV Cache Demotion 的 Python 层实验原型。
#
# 这个模块刻意不修改 attention kernel，也不改变 attention 数学逻辑。它只在
# Scheduler/CacheManager 即将释放 GPU KV cache 前，把“这段 KV 曾经存在过”的
# 元数据写到磁盘。未来新请求如果出现相同 token 前缀，可以记录 compressed hit，
# 然后尝试 restore；第一版 mock codec 不恢复真实 tensor，所以一定 fallback 到
# 原始 prefill/recompute 路径，保证生成正确性不受影响。

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable

import torch


class CompressedKVCacheManager:
    """管理被 demote 的 KV cache 元数据和实验统计。

    第一版重点是把 runtime 路径跑通：
    - demote() 在释放/驱逐 GPU KV cache 前调用；
    - maybe_restore() 在请求命中 compressed archive 时调用；
    - mock codec 只保存 json metadata，不保存真实 K/V tensor；
    - restore 失败时返回 None，调用方继续走原始 prefill。
    """

    def __init__(self, args, kv_pool=None, logger=None):
        self.args = args
        self.kv_pool = kv_pool
        self.logger = logger
        self.codec = getattr(args, "compressed_kv_cache_codec", "mock")
        self.policy = getattr(args, "compressed_kv_cache_restore_policy", "cost")
        self.max_size_bytes = int(
            getattr(args, "compressed_kv_cache_max_size_mb", 4096) * 1024 * 1024
        )
        self.archive_dir = Path(
            getattr(args, "compressed_kv_cache_dir", "/root/autodl-tmp/kv_archive")
        )
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._token_hash_to_entry: Dict[str, Dict[str, Any]] = {}
        self._stats: Dict[str, int] = {
            "num_evictions": 0,
            "num_demotions": 0,
            "num_mock_demotions": 0,
            "num_int8_demotions": 0,
            "num_compressed_entries": 0,
            "compressed_bytes": 0,
            "original_estimated_bytes": 0,
            "num_compressed_hits": 0,
            "num_restore_attempts": 0,
            "num_restore_success": 0,
            "num_restore_fallback": 0,
            "saved_prefill_tokens_estimated": 0,
        }

        if self.enabled():
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            self._load_existing_metadata()
            self._log(
                "info",
                "[CompressedKV] enabled: dir=%s codec=%s max_size_mb=%s policy=%s",
                self.archive_dir,
                self.codec,
                getattr(args, "compressed_kv_cache_max_size_mb", 4096),
                self.policy,
            )

    def enabled(self) -> bool:
        """feature flag 是否开启。关闭时调用方应该保持原行为。"""

        return bool(getattr(self.args, "enable_compressed_kv_cache", False))

    def demote(self, entry) -> bool:
        """在 GPU KV cache 即将释放/淘汰前保存压缩归档。

        entry 是调用方组装的 dict，通常包含 request_id、token_ids、indices、
        num_tokens、page_size 等信息。mock codec 只写 json metadata。int8_cpu
        第一版不做真实 tensor 拷贝和恢复，只保留 TODO metadata，并按 mock 路径
        fallback，避免在不明确写回 KV pool 的情况下破坏原推理。
        """

        if not self.enabled():
            return False

        try:
            token_ids = entry.get("token_ids")
            indices = entry.get("indices")
            num_tokens = int(entry.get("num_tokens") or _safe_len(token_ids) or _safe_len(indices))
            if num_tokens <= 0:
                return False

            self._stats["num_evictions"] += 1
            token_ids_hash = entry.get("token_ids_hash") or self.token_ids_hash(token_ids)
            entry_id = entry.get("entry_id") or self._make_entry_id(token_ids_hash)
            original_bytes = int(
                entry.get("original_estimated_bytes")
                or self._estimate_original_bytes(num_tokens, entry)
            )
            codec = self.codec
            actual_codec = codec
            storage_path = self.archive_dir / f"{entry_id}.json"

            self._log(
                "info",
                "[CompressedKV] eviction detected: entry_id=%s, num_tokens=%s, estimated_bytes=%s",
                entry_id,
                num_tokens,
                original_bytes,
            )

            if codec == "mock":
                compressed_bytes = 0
                self._stats["num_mock_demotions"] += 1
            elif codec == "int8_cpu":
                # TODO: 这里未来可以从 kv_pool 按 indices 读取每层 K/V tensor，
                # 做 int8 对称量化后保存 entry_xxx.pt。当前原型没有安全的
                # 写回 restore 路径，因此不保存真实 tensor，避免误导调用方。
                actual_codec = "mock"
                compressed_bytes = 0
                self._stats["num_mock_demotions"] += 1
            else:
                self._log(
                    "warning",
                    "[CompressedKV] unsupported codec=%s, fallback to mock metadata",
                    codec,
                )
                actual_codec = "mock"
                compressed_bytes = 0
                self._stats["num_mock_demotions"] += 1

            now = time.time()
            meta = {
                "entry_id": entry_id,
                "request_id": str(entry.get("request_id", "")),
                "token_ids_hash": token_ids_hash,
                "num_tokens": num_tokens,
                "num_layers": self._num_layers(entry),
                "dtype": self._dtype(entry),
                "codec": actual_codec,
                "requested_codec": codec,
                "state": "COLD",
                "created_time": now,
                "last_access_time": now,
                "hit_count": 0,
                "original_estimated_bytes": original_bytes,
                "compressed_estimated_bytes": compressed_bytes,
                "storage_path": str(storage_path),
                "reason": entry.get("reason", "unknown"),
                "page_size": entry.get("page_size"),
                "kv_indices": _tensor_to_list(indices, limit=128),
                "kv_indices_numel": _safe_len(indices),
                "device": str(getattr(self.kv_pool, "device", entry.get("device", ""))),
                "layout": entry.get("layout", "token_indices"),
                "restore_status": "mock_fallback",
            }
            self._write_json(storage_path, meta)
            self._entries[entry_id] = meta
            if token_ids_hash:
                self._token_hash_to_entry[token_ids_hash] = meta

            self._stats["num_demotions"] += 1
            self._stats["num_compressed_entries"] = len(self._entries)
            self._stats["compressed_bytes"] += compressed_bytes
            self._stats["original_estimated_bytes"] += original_bytes
            self._enforce_size_limit()

            self._log(
                "info",
                "[CompressedKV] demoted: entry_id=%s, codec=%s, original_bytes=%s, compressed_bytes=%s",
                entry_id,
                actual_codec,
                original_bytes,
                compressed_bytes,
            )
            return True
        except Exception as exc:
            self._log("warning", "[CompressedKV] demote failed, ignore and free GPU KV: %s", exc)
            return False

    def maybe_restore(self, entry_or_key):
        """尝试恢复 compressed entry。

        mock codec 没有真实 tensor，因此这里只记录日志和统计，并返回 None。调用方
        看到 None 后必须 fallback 到原始 prefill/recompute。
        """

        if not self.enabled():
            return None
        meta = self._resolve_entry(entry_or_key)
        if meta is None:
            return None

        self._stats["num_restore_attempts"] += 1
        self._log(
            "info",
            "[CompressedKV] restore attempt: entry_id=%s, policy=%s",
            meta["entry_id"],
            self.policy,
        )
        self._stats["num_restore_fallback"] += 1
        self._log(
            "info",
            "[CompressedKV] restore fallback to recompute: entry_id=%s",
            meta["entry_id"],
        )
        return None

    def should_restore(self, meta, recompute_tokens: int) -> bool:
        """简单代价模型，决定是否值得尝试 restore。

        当前 mock codec 不可能真实恢复，所以只用于记录实验决策。cost 策略下，
        长前缀更倾向于尝试；always 策略下只要命中就尝试。
        """

        if not self.enabled():
            return False
        if self.policy == "always":
            return True
        if self.policy == "never":
            return False
        restore_cost = 512
        recompute_cost = max(0, int(recompute_tokens))
        return recompute_cost > restore_cost

    def find_match(self, input_ids: torch.Tensor):
        """按完整前缀 hash 查找 compressed archive。

        这里不是完整 prefix cache，只是 request-level archive 原型：从最长前缀
        开始查 hash，命中后打印日志并让调用方决定是否 maybe_restore。
        """

        if not self.enabled():
            return None
        for length in range(len(input_ids), 0, -1):
            key = self.token_ids_hash(input_ids[:length])
            meta = self._token_hash_to_entry.get(key)
            if meta is None:
                continue
            meta["last_access_time"] = time.time()
            meta["hit_count"] = int(meta.get("hit_count", 0)) + 1
            self._stats["num_compressed_hits"] += 1
            self._stats["saved_prefill_tokens_estimated"] += int(meta.get("num_tokens", length))
            self._log(
                "info",
                "[CompressedKV] compressed hit: entry_id=%s, num_tokens=%s",
                meta["entry_id"],
                meta.get("num_tokens", length),
            )
            return meta
        return None

    def token_ids_hash(self, token_ids) -> str:
        """把 token ids 转成稳定 hash，用作 request-level archive key。"""

        if token_ids is None:
            return ""
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.detach().cpu().to(torch.int32).contiguous()
            payload = ids.numpy().tobytes()
        else:
            payload = ",".join(str(int(x)) for x in token_ids).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def stats(self) -> dict:
        """返回当前统计信息。"""

        result = dict(self._stats)
        result["num_compressed_entries"] = len(self._entries)
        return result

    def log_stats(self) -> None:
        """把统计信息打印到日志。"""

        if self.enabled():
            self._log("info", "[CompressedKV] stats: %s", self.stats())

    def _resolve_entry(self, entry_or_key):
        if isinstance(entry_or_key, dict):
            return entry_or_key
        return self._entries.get(str(entry_or_key)) or self._token_hash_to_entry.get(str(entry_or_key))

    def _estimate_original_bytes(self, num_tokens: int, entry: Dict[str, Any]) -> int:
        dtype_size = int(entry.get("dtype_size") or getattr(getattr(self.kv_pool, "dtype", None), "itemsize", 2))
        num_layers = self._num_layers(entry)
        local_kv_heads = int(entry.get("local_kv_heads") or 1)
        head_dim = int(entry.get("head_dim") or 1)
        return 2 * num_layers * num_tokens * local_kv_heads * head_dim * dtype_size

    def _num_layers(self, entry: Dict[str, Any]) -> int:
        return int(entry.get("num_layers") or getattr(self.kv_pool, "num_layers", 0) or 0)

    def _dtype(self, entry: Dict[str, Any]) -> str:
        return str(entry.get("dtype") or getattr(self.kv_pool, "dtype", "unknown"))

    def _make_entry_id(self, token_ids_hash: str) -> str:
        suffix = token_ids_hash[:12] if token_ids_hash else uuid.uuid4().hex[:12]
        return f"entry_{int(time.time() * 1000)}_{suffix}"

    def _load_existing_metadata(self) -> None:
        for path in self.archive_dir.glob("entry_*.json"):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            entry_id = meta.get("entry_id")
            token_hash = meta.get("token_ids_hash")
            if not entry_id:
                continue
            self._entries[entry_id] = meta
            if token_hash:
                self._token_hash_to_entry[token_hash] = meta
        self._stats["num_compressed_entries"] = len(self._entries)

    def _write_json(self, path: Path, meta: Dict[str, Any]) -> None:
        path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    def _enforce_size_limit(self) -> None:
        if self.max_size_bytes <= 0 or self._stats["compressed_bytes"] <= self.max_size_bytes:
            return
        # mock codec 不占 tensor 文件空间；这里保留入口，未来 int8_cpu 可在这里按
        # last_access_time 删除冷归档文件。
        self._log("warning", "[CompressedKV] archive size limit reached; pruning TODO")

    def _log(self, level: str, msg: str, *args) -> None:
        if self.logger is None:
            print(msg % args if args else msg)
            return
        fn = getattr(self.logger, f"{level}_rank0", None) or getattr(self.logger, level)
        fn(msg, *args)


def _safe_len(value) -> int:
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


def _tensor_to_list(value, *, limit: int) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().flatten()[:limit]
        return [int(x) for x in flat.tolist()]
    if isinstance(value, Iterable):
        result = []
        for i, item in enumerate(value):
            if i >= limit:
                break
            result.append(int(item))
        return result
    return []
