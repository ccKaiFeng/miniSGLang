# miniSGLang Experiment Report: zipcache_v3_8b_20gb

- mode: `zipcache_v3_8b_20gb`
- base_url: `http://127.0.0.1:30001`
- run_dir: `experiment/logs/20260914_133848_zipcache_v3_8b_20gb`
- started_at: `2026-09-14 13:38:48`
- git_branch: `zipcache-v3-quick-results`
- git_commit: `c643ae0`

## Server Check

```json
{
  "ok": true,
  "url": "http://127.0.0.1:30001/v1",
  "latency_s": 0.009790219366550446,
  "response": "{\"status\":\"ok\"}"
}
```

## Experiments

| experiment | ok/total | maxed | rps | chunks/s | ttft avg | ttft p90 | e2e avg | tpot avg | gpu max MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| longbench_long_context_pressure | 32/32 | 0 | 1.019 | 259.7 | 2.221 | 3.558 | 7.851 | 0.02217 | 44295 |

## Result Files

- `longbench_long_context_pressure` results: `experiment/logs/20260914_133848_zipcache_v3_8b_20gb/longbench_long_context_pressure.jsonl`
- `longbench_long_context_pressure` summary: `experiment/logs/20260914_133848_zipcache_v3_8b_20gb/longbench_long_context_pressure_summary.json`
- `longbench_long_context_pressure` log: `experiment/logs/20260914_133848_zipcache_v3_8b_20gb/longbench_long_context_pressure.log`

## ZipCache Stats

```json
{
  "num_stats": 33,
  "versions": [
    "ZipCacheV3"
  ],
  "last": {
    "num_demotions": 148,
    "num_demote_failures": 539,
    "num_compressed_entries": 148,
    "num_compressed_hits": 182,
    "num_restore_attempts": 182,
    "num_restore_success": 182,
    "num_restore_fallback": 0,
    "num_compressed_freed": 0,
    "num_temporary_restore_pages": 3318,
    "num_restore_pages_released": 3318,
    "num_restore_rejected_small_prefix": 0,
    "original_estimated_bytes": 42767400960,
    "compressed_estimated_bytes_4bit": 9389868480,
    "compressed_storage_bytes": 9389868480,
    "active_original_estimated_bytes": 42767400960,
    "active_compressed_estimated_bytes_4bit": 9389868480,
    "active_compressed_storage_bytes": 9389868480,
    "last_estimated_compression_ratio": 4.515603799185889,
    "last_storage_compression_ratio": 4.515603799185889,
    "num_demote_rejected_pool_full": 539,
    "active_estimated_compression_ratio": 4.55463258629156,
    "active_storage_compression_ratio": 4.55463258629156,
    "gpu_memory_allocated_bytes": 44682251264,
    "gpu_memory_reserved_bytes": 45877297152,
    "gpu_max_memory_allocated_bytes": 45085167104,
    "gpu_max_memory_reserved_bytes": 45877297152,
    "compressed_pool_capacity_bytes": 14260633600,
    "compressed_pool_used_bytes": 9389868480,
    "compressed_pool_free_bytes": 4870765120,
    "compressed_pool_utilization": 0.6584467943976907,
    "compressed_pool_q_used_bytes": 8554567680,
    "compressed_pool_q4_used_bytes": 6417285120,
    "compressed_pool_q2_used_bytes": 2137282560,
    "compressed_pool_scale_used_bytes": 668240640,
    "compressed_pool_ids_used_bytes": 167060160,
    "compressed_pool_q4_capacity_bytes": 6417285120,
    "compressed_pool_q2_capacity_bytes": 2139095040,
    "compressed_pool_scale_capacity_bytes": 3565158400,
    "compressed_pool_ids_capacity_bytes": 2139095040,
    "normal_pool_capacity_bytes": 13861011456,
    "estimated_effective_kv_capacity_bytes": 78812957951.72432,
    "estimated_capacity_gain_vs_normal_pool": 5.685945661462435,
    "_zipcache_version": "ZipCacheV3"
  },
  "max_active_compression_ratio": 4.554808716985427,
  "last_active_compression_ratio": 4.55463258629156,
  "max_active_original_estimated_bytes": 42767400960,
  "max_active_compressed_estimated_bytes": 9389868480,
  "max_num_compressions_or_demotions": 148,
  "max_num_decompressions_or_restores": 182
}
```
