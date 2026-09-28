# miniSGLang Experiment Report: zipcache_13600mb_longbench_long_context_pressure

- mode: `zipcache_13600mb_longbench_long_context_pressure`
- base_url: `http://127.0.0.1:31000`
- run_dir: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/zipcache_13600mb_longbench_long_context_pressure/20260914_164008_zipcache_13600mb_longbench_long_context_pressure`
- started_at: `2026-09-14 16:40:08`
- git_branch: `working-tree`
- git_commit: `b3ece67ca5cd4682f7fe929a5cad93b9bc169780`

## Server Check

```json
{
  "ok": true,
  "url": "http://127.0.0.1:31000/v1",
  "latency_s": 0.008904244750738144,
  "response": "{\"status\":\"ok\"}"
}
```

## Experiments

| experiment | ok/total | maxed | rps | chunks/s | ttft avg | ttft p90 | e2e avg | tpot avg | gpu max MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| longbench_long_context_pressure | 32/32 | 0 | 0.7983 | 203.6 | 2.661 | 4.103 | 10.02 | 0.02897 | 44249 |

## Result Files

- `longbench_long_context_pressure` results: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/zipcache_13600mb_longbench_long_context_pressure/20260914_164008_zipcache_13600mb_longbench_long_context_pressure/longbench_long_context_pressure.jsonl`
- `longbench_long_context_pressure` summary: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/zipcache_13600mb_longbench_long_context_pressure/20260914_164008_zipcache_13600mb_longbench_long_context_pressure/longbench_long_context_pressure_summary.json`
- `longbench_long_context_pressure` log: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/zipcache_13600mb_longbench_long_context_pressure/20260914_164008_zipcache_13600mb_longbench_long_context_pressure/longbench_long_context_pressure.log`

## ZipCache Stats

```json
{
  "num_stats": 3,
  "versions": [
    "ZipCacheV3"
  ],
  "last": {
    "num_demotions": 52,
    "num_demote_failures": 0,
    "num_compressed_entries": 52,
    "num_compressed_hits": 32,
    "num_restore_attempts": 32,
    "num_restore_success": 32,
    "num_restore_fallback": 0,
    "num_compressed_freed": 0,
    "num_temporary_restore_pages": 400,
    "num_restore_pages_released": 400,
    "num_restore_rejected_small_prefix": 0,
    "original_estimated_bytes": 22429532160,
    "compressed_estimated_bytes_4bit": 4924222848,
    "compressed_storage_bytes": 4924222848,
    "active_original_estimated_bytes": 22429532160,
    "active_compressed_estimated_bytes_4bit": 4924222848,
    "active_compressed_storage_bytes": 4924222848,
    "last_estimated_compression_ratio": 3.710144927536232,
    "last_storage_compression_ratio": 3.710144927536232,
    "num_demote_rejected_pool_full": 0,
    "active_estimated_compression_ratio": 4.554938485188557,
    "active_storage_compression_ratio": 4.554938485188557,
    "gpu_memory_allocated_bytes": 44682275840,
    "gpu_memory_reserved_bytes": 45829062656,
    "gpu_max_memory_allocated_bytes": 45084713984,
    "gpu_max_memory_reserved_bytes": 45829062656,
    "compressed_pool_capacity_bytes": 14260633600,
    "compressed_pool_used_bytes": 4924222848,
    "compressed_pool_free_bytes": 9336410752,
    "compressed_pool_utilization": 0.3453018278234145,
    "compressed_pool_q_used_bytes": 4486146048,
    "compressed_pool_q4_used_bytes": 3364909056,
    "compressed_pool_q2_used_bytes": 1121236992,
    "compressed_pool_scale_used_bytes": 350461440,
    "compressed_pool_ids_used_bytes": 87615360,
    "compressed_pool_q4_capacity_bytes": 9412018176,
    "compressed_pool_q2_capacity_bytes": 3137339392,
    "compressed_pool_scale_capacity_bytes": 1283457024,
    "compressed_pool_ids_capacity_bytes": 427819008,
    "normal_pool_capacity_bytes": 13861011456,
    "estimated_effective_kv_capacity_bytes": 78817320263.81303,
    "estimated_capacity_gain_vs_normal_pool": 5.686260379627309,
    "_zipcache_version": "ZipCacheV3"
  },
  "max_active_compression_ratio": 4.554938485188557,
  "last_active_compression_ratio": 4.554938485188557,
  "max_active_original_estimated_bytes": 22429532160,
  "max_active_compressed_estimated_bytes": 4924222848,
  "max_num_compressions_or_demotions": 52,
  "max_num_decompressions_or_restores": 32
}
```
