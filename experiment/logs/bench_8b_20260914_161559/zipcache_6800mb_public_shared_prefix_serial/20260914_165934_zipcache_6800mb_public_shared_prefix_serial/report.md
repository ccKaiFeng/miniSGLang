# miniSGLang Experiment Report: zipcache_6800mb_public_shared_prefix_serial

- mode: `zipcache_6800mb_public_shared_prefix_serial`
- base_url: `http://127.0.0.1:31000`
- run_dir: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/zipcache_6800mb_public_shared_prefix_serial/20260914_165934_zipcache_6800mb_public_shared_prefix_serial`
- started_at: `2026-09-14 16:59:34`
- git_branch: `working-tree`
- git_commit: `b3ece67ca5cd4682f7fe929a5cad93b9bc169780`

## Server Check

```json
{
  "ok": true,
  "url": "http://127.0.0.1:31000/v1",
  "latency_s": 0.00868341326713562,
  "response": "{\"status\":\"ok\"}"
}
```

## Experiments

| experiment | ok/total | maxed | rps | chunks/s | ttft avg | ttft p90 | e2e avg | tpot avg | gpu max MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| public_shared_prefix_serial | 192/192 | 0 | 0.5311 | 67.45 | 0.3468 | 0.4418 | 1.883 | 0.01219 | 37375 |

## Result Files

- `public_shared_prefix_serial` results: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/zipcache_6800mb_public_shared_prefix_serial/20260914_165934_zipcache_6800mb_public_shared_prefix_serial/public_shared_prefix_serial.jsonl`
- `public_shared_prefix_serial` summary: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/zipcache_6800mb_public_shared_prefix_serial/20260914_165934_zipcache_6800mb_public_shared_prefix_serial/public_shared_prefix_serial_summary.json`
- `public_shared_prefix_serial` log: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/zipcache_6800mb_public_shared_prefix_serial/20260914_165934_zipcache_6800mb_public_shared_prefix_serial/public_shared_prefix_serial.log`

## ZipCache Stats

```json
{
  "num_stats": 33,
  "versions": [
    "ZipCacheV3"
  ],
  "last": {
    "num_demotions": 2,
    "num_demote_failures": 0,
    "num_compressed_entries": 2,
    "num_compressed_hits": 0,
    "num_restore_attempts": 0,
    "num_restore_success": 0,
    "num_restore_fallback": 0,
    "num_compressed_freed": 0,
    "num_temporary_restore_pages": 0,
    "num_restore_pages_released": 0,
    "num_restore_rejected_small_prefix": 0,
    "original_estimated_bytes": 738902016,
    "compressed_estimated_bytes_4bit": 162219456,
    "compressed_storage_bytes": 162219456,
    "active_original_estimated_bytes": 738902016,
    "active_compressed_estimated_bytes_4bit": 162219456,
    "active_compressed_storage_bytes": 162219456,
    "last_estimated_compression_ratio": 4.555160142348755,
    "last_storage_compression_ratio": 4.555160142348755,
    "num_demote_rejected_pool_full": 0,
    "active_estimated_compression_ratio": 4.55495311240595,
    "active_storage_compression_ratio": 4.55495311240595,
    "gpu_memory_allocated_bytes": 37550551552,
    "gpu_memory_reserved_bytes": 38621151232,
    "gpu_max_memory_allocated_bytes": 37953323520,
    "gpu_max_memory_reserved_bytes": 38621151232,
    "compressed_pool_capacity_bytes": 7130316800,
    "compressed_pool_used_bytes": 162219456,
    "compressed_pool_free_bytes": 6968097344,
    "compressed_pool_utilization": 0.022750666001263787,
    "compressed_pool_q_used_bytes": 147787776,
    "compressed_pool_q4_used_bytes": 110850048,
    "compressed_pool_q2_used_bytes": 36937728,
    "compressed_pool_scale_used_bytes": 11545344,
    "compressed_pool_ids_used_bytes": 2886336,
    "compressed_pool_q4_capacity_bytes": 4706009088,
    "compressed_pool_q2_capacity_bytes": 1568669696,
    "compressed_pool_scale_capacity_bytes": 641728512,
    "compressed_pool_ids_capacity_bytes": 213909504,
    "normal_pool_capacity_bytes": 13861011456,
    "estimated_effective_kv_capacity_bytes": 46339270156.60043,
    "estimated_capacity_gain_vs_normal_pool": 3.3431377142785355,
    "_zipcache_version": "ZipCacheV3"
  },
  "max_active_compression_ratio": 4.55495311240595,
  "last_active_compression_ratio": 4.55495311240595,
  "max_active_original_estimated_bytes": 738902016,
  "max_active_compressed_estimated_bytes": 162219456,
  "max_num_compressions_or_demotions": 2,
  "max_num_decompressions_or_restores": 0
}
```
