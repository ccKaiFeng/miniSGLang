# miniSGLang Experiment Report: zipcache_v4_quick

- mode: `zipcache_v4_quick`
- base_url: `http://127.0.0.1:30001`
- run_dir: `experiment/logs/20260703_161134_zipcache_v4_quick`
- started_at: `2026-07-03 16:11:34`
- git_branch: `ZipCache`
- git_commit: `98f0030`

## Server Check

```json
{
  "ok": true,
  "url": "http://127.0.0.1:30001/v1",
  "latency_s": 0.009906943887472153,
  "response": "{\"status\":\"ok\"}"
}
```

## Experiments

| experiment | ok/total | maxed | rps | chunks/s | ttft avg | ttft p90 | e2e avg | tpot avg | gpu max MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsm8k_public_correctness | 64/64 | 0 | 0.1202 | 121 | 0.04786 | 0.0223 | 8.318 | 0.008298 | 46499 |
| longbench_long_context_pressure | 32/32 | 0 | 1.168 | 148.3 | 1.761 | 12.35 | 3.425 | 0.0132 | 46819 |
| public_shared_prefix_serial | 64/64 | 0 | 1.055 | 100.2 | 0.07413 | 0.09318 | 0.9478 | 0.009295 | 46819 |

## Result Files

- `gsm8k_public_correctness` results: `experiment/logs/20260703_161134_zipcache_v4_quick/gsm8k_public_correctness.jsonl`
- `gsm8k_public_correctness` summary: `experiment/logs/20260703_161134_zipcache_v4_quick/gsm8k_public_correctness_summary.json`
- `gsm8k_public_correctness` log: `experiment/logs/20260703_161134_zipcache_v4_quick/gsm8k_public_correctness.log`
- `gsm8k_public_correctness` correctness eval: `experiment/logs/20260703_161134_zipcache_v4_quick/gsm8k_public_correctness_eval.json`
- `longbench_long_context_pressure` results: `experiment/logs/20260703_161134_zipcache_v4_quick/longbench_long_context_pressure.jsonl`
- `longbench_long_context_pressure` summary: `experiment/logs/20260703_161134_zipcache_v4_quick/longbench_long_context_pressure_summary.json`
- `longbench_long_context_pressure` log: `experiment/logs/20260703_161134_zipcache_v4_quick/longbench_long_context_pressure.log`
- `public_shared_prefix_serial` results: `experiment/logs/20260703_161134_zipcache_v4_quick/public_shared_prefix_serial.jsonl`
- `public_shared_prefix_serial` summary: `experiment/logs/20260703_161134_zipcache_v4_quick/public_shared_prefix_serial_summary.json`
- `public_shared_prefix_serial` log: `experiment/logs/20260703_161134_zipcache_v4_quick/public_shared_prefix_serial.log`

## Correctness Evaluation

| experiment | judged | correct | accuracy |
| --- | ---: | ---: | ---: |
| gsm8k_public_correctness | 64 | 44 | 0.6875 |

## ZipCache Stats

```json
{
  "num_stats": 20,
  "versions": [
    "ZipCacheV4"
  ],
  "last": {
    "num_demotions": 322,
    "num_demote_failures": 0,
    "num_compressed_entries": 322,
    "num_compressed_hits": 111,
    "num_restore_attempts": 111,
    "num_restore_success": 111,
    "num_restore_fallback": 0,
    "num_compressed_freed": 0,
    "num_temporary_restore_pages": 1901,
    "num_restore_pages_released": 1901,
    "num_restore_rejected_small_prefix": 0,
    "original_estimated_bytes": 67133194240,
    "compressed_estimated_bytes_4bit": 14739941440,
    "compressed_storage_bytes": 14739941440,
    "active_original_estimated_bytes": 67133194240,
    "active_compressed_estimated_bytes_4bit": 14739941440,
    "active_compressed_storage_bytes": 14739941440,
    "last_estimated_compression_ratio": 4.554903419823263,
    "last_storage_compression_ratio": 4.554903419823263,
    "num_demote_rejected_pool_full": 0,
    "num_kernel_restore_calls": 12376,
    "num_kernel_restore_fallback": 0,
    "kernel_restore_tokens": 1901,
    "kernel_restore_elements": 109010944,
    "active_estimated_compression_ratio": 4.554508884127562,
    "active_storage_compression_ratio": 4.554508884127562,
    "gpu_memory_allocated_bytes": 48080237056,
    "gpu_memory_reserved_bytes": 48532291584,
    "gpu_max_memory_allocated_bytes": 48200348672,
    "gpu_max_memory_reserved_bytes": 48532291584,
    "compressed_pool_capacity_bytes": 42949672960,
    "compressed_pool_used_bytes": 14739941440,
    "compressed_pool_free_bytes": 28209731520,
    "compressed_pool_utilization": 0.3431910052895546,
    "compressed_pool_q_used_bytes": 13428746240,
    "compressed_pool_q4_used_bytes": 10074193920,
    "compressed_pool_q2_used_bytes": 3354552320,
    "compressed_pool_scale_used_bytes": 1048956160,
    "compressed_pool_ids_used_bytes": 262239040,
    "compressed_pool_q4_capacity_bytes": 19327352832,
    "compressed_pool_q2_capacity_bytes": 6442450944,
    "compressed_pool_scale_capacity_bytes": 10737418240,
    "compressed_pool_ids_capacity_bytes": 6442450944,
    "normal_pool_capacity_bytes": 3758211072,
    "estimated_effective_kv_capacity_bytes": 199372878138.6933,
    "estimated_capacity_gain_vs_normal_pool": 53.04994166615379,
    "_zipcache_version": "ZipCacheV4"
  },
  "max_active_compression_ratio": 4.554508884127562,
  "last_active_compression_ratio": 4.554508884127562,
  "max_active_original_estimated_bytes": 67133194240,
  "max_active_compressed_estimated_bytes": 14739941440,
  "max_num_compressions_or_demotions": 322,
  "max_num_decompressions_or_restores": 111
}
```
