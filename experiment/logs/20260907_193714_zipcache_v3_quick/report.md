# miniSGLang Experiment Report: zipcache_v3_quick

- mode: `zipcache_v3_quick`
- base_url: `http://127.0.0.1:30001`
- run_dir: `experiment/logs/20260907_193714_zipcache_v3_quick`
- started_at: `2026-09-07 19:37:14`
- git_branch: ``
- git_commit: `37baefd`

## Server Check

```json
{
  "ok": true,
  "url": "http://127.0.0.1:30001/v1",
  "latency_s": 0.012277323752641678,
  "response": "{\"status\":\"ok\"}"
}
```

## Experiments

| experiment | ok/total | maxed | rps | chunks/s | ttft avg | ttft p90 | e2e avg | tpot avg | gpu max MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsm8k_public_correctness | 64/64 | 0 | 0.1111 | 111.9 | 1.268 | 0.02775 | 8.998 | 0.007785 | 46503 |
| longbench_long_context_pressure | 32/32 | 0 | 2.129 | 270.4 | 0.2825 | 0.3776 | 1.878 | 0.01266 | 46821 |
| public_shared_prefix_serial | 64/64 | 0 | 1.065 | 101.1 | 0.1172 | 0.139 | 0.9392 | 0.008745 | 46821 |

## Result Files

- `gsm8k_public_correctness` results: `experiment/logs/20260907_193714_zipcache_v3_quick/gsm8k_public_correctness.jsonl`
- `gsm8k_public_correctness` summary: `experiment/logs/20260907_193714_zipcache_v3_quick/gsm8k_public_correctness_summary.json`
- `gsm8k_public_correctness` log: `experiment/logs/20260907_193714_zipcache_v3_quick/gsm8k_public_correctness.log`
- `gsm8k_public_correctness` correctness eval: `experiment/logs/20260907_193714_zipcache_v3_quick/gsm8k_public_correctness_eval.json`
- `longbench_long_context_pressure` results: `experiment/logs/20260907_193714_zipcache_v3_quick/longbench_long_context_pressure.jsonl`
- `longbench_long_context_pressure` summary: `experiment/logs/20260907_193714_zipcache_v3_quick/longbench_long_context_pressure_summary.json`
- `longbench_long_context_pressure` log: `experiment/logs/20260907_193714_zipcache_v3_quick/longbench_long_context_pressure.log`
- `public_shared_prefix_serial` results: `experiment/logs/20260907_193714_zipcache_v3_quick/public_shared_prefix_serial.jsonl`
- `public_shared_prefix_serial` summary: `experiment/logs/20260907_193714_zipcache_v3_quick/public_shared_prefix_serial_summary.json`
- `public_shared_prefix_serial` log: `experiment/logs/20260907_193714_zipcache_v3_quick/public_shared_prefix_serial.log`

## Correctness Evaluation

| experiment | judged | correct | accuracy |
| --- | ---: | ---: | ---: |
| gsm8k_public_correctness | 64 | 44 | 0.6875 |

## ZipCache Stats

```json
{
  "num_stats": 40,
  "versions": [
    "ZipCacheV3"
  ],
  "last": {
    "num_demotions": 308,
    "num_demote_failures": 0,
    "num_compressed_entries": 308,
    "num_compressed_hits": 104,
    "num_restore_attempts": 104,
    "num_restore_success": 104,
    "num_restore_fallback": 0,
    "num_compressed_freed": 0,
    "num_temporary_restore_pages": 1761,
    "num_restore_pages_released": 1761,
    "num_restore_rejected_small_prefix": 0,
    "original_estimated_bytes": 63783157760,
    "compressed_estimated_bytes_4bit": 14004403392,
    "compressed_storage_bytes": 14004403392,
    "active_original_estimated_bytes": 63783157760,
    "active_compressed_estimated_bytes_4bit": 14004403392,
    "active_compressed_storage_bytes": 14004403392,
    "last_estimated_compression_ratio": 4.554676184349775,
    "last_storage_compression_ratio": 4.554676184349775,
    "num_demote_rejected_pool_full": 0,
    "active_estimated_compression_ratio": 4.5545073199216795,
    "active_storage_compression_ratio": 4.5545073199216795,
    "gpu_memory_allocated_bytes": 48080026624,
    "gpu_memory_reserved_bytes": 48530194432,
    "gpu_max_memory_allocated_bytes": 48200625152,
    "gpu_max_memory_reserved_bytes": 48530194432,
    "compressed_pool_capacity_bytes": 42949672960,
    "compressed_pool_used_bytes": 14004403392,
    "compressed_pool_free_bytes": 28945269568,
    "compressed_pool_utilization": 0.3260654255747795,
    "compressed_pool_q_used_bytes": 12758638592,
    "compressed_pool_q4_used_bytes": 9571487744,
    "compressed_pool_q2_used_bytes": 3187150848,
    "compressed_pool_scale_used_bytes": 996611840,
    "compressed_pool_ids_used_bytes": 249152960,
    "compressed_pool_q4_capacity_bytes": 19327352832,
    "compressed_pool_q2_capacity_bytes": 6442450944,
    "compressed_pool_scale_capacity_bytes": 10737418240,
    "compressed_pool_ids_capacity_bytes": 6442450944,
    "normal_pool_capacity_bytes": 3758211072,
    "estimated_effective_kv_capacity_bytes": 199372810956.56223,
    "estimated_capacity_gain_vs_normal_pool": 53.049923790060674,
    "_zipcache_version": "ZipCacheV3"
  },
  "max_active_compression_ratio": 4.5545073199216795,
  "last_active_compression_ratio": 4.5545073199216795,
  "max_active_original_estimated_bytes": 63783157760,
  "max_active_compressed_estimated_bytes": 14004403392,
  "max_num_compressions_or_demotions": 308,
  "max_num_decompressions_or_restores": 104
}
```
