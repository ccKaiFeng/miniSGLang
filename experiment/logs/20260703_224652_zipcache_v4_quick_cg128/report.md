# miniSGLang Experiment Report: zipcache_v4_quick_cg128

- mode: `zipcache_v4_quick_cg128`
- base_url: `http://127.0.0.1:30001`
- run_dir: `experiment/logs/20260703_224652_zipcache_v4_quick_cg128`
- started_at: `2026-07-03 22:46:52`
- git_branch: `ZipCache`
- git_commit: `1fedece`

## Server Check

```json
{
  "ok": true,
  "url": "http://127.0.0.1:30001/v1",
  "latency_s": 0.009100053459405899,
  "response": "{\"status\":\"ok\"}"
}
```

## Experiments

| experiment | ok/total | maxed | rps | chunks/s | ttft avg | ttft p90 | e2e avg | tpot avg | gpu max MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsm8k_public_correctness | 64/64 | 0 | 0.4096 | 412.3 | 0.01558 | 0.01588 | 2.441 | 0.002448 | 46731 |
| longbench_long_context_pressure | 32/32 | 0 | 1.393 | 176.9 | 1.743 | 12.18 | 2.871 | 0.008953 | 47003 |
| public_shared_prefix_serial | 64/64 | 0 | 2.389 | 227 | 0.07507 | 0.09574 | 0.4185 | 0.003653 | 47051 |

## Result Files

- `gsm8k_public_correctness` results: `experiment/logs/20260703_224652_zipcache_v4_quick_cg128/gsm8k_public_correctness.jsonl`
- `gsm8k_public_correctness` summary: `experiment/logs/20260703_224652_zipcache_v4_quick_cg128/gsm8k_public_correctness_summary.json`
- `gsm8k_public_correctness` log: `experiment/logs/20260703_224652_zipcache_v4_quick_cg128/gsm8k_public_correctness.log`
- `gsm8k_public_correctness` correctness eval: `experiment/logs/20260703_224652_zipcache_v4_quick_cg128/gsm8k_public_correctness_eval.json`
- `longbench_long_context_pressure` results: `experiment/logs/20260703_224652_zipcache_v4_quick_cg128/longbench_long_context_pressure.jsonl`
- `longbench_long_context_pressure` summary: `experiment/logs/20260703_224652_zipcache_v4_quick_cg128/longbench_long_context_pressure_summary.json`
- `longbench_long_context_pressure` log: `experiment/logs/20260703_224652_zipcache_v4_quick_cg128/longbench_long_context_pressure.log`
- `public_shared_prefix_serial` results: `experiment/logs/20260703_224652_zipcache_v4_quick_cg128/public_shared_prefix_serial.jsonl`
- `public_shared_prefix_serial` summary: `experiment/logs/20260703_224652_zipcache_v4_quick_cg128/public_shared_prefix_serial_summary.json`
- `public_shared_prefix_serial` log: `experiment/logs/20260703_224652_zipcache_v4_quick_cg128/public_shared_prefix_serial.log`

## Correctness Evaluation

| experiment | judged | correct | accuracy |
| --- | ---: | ---: | ---: |
| gsm8k_public_correctness | 64 | 44 | 0.6875 |

## ZipCache Stats

```json
{
  "num_stats": 7,
  "versions": [
    "ZipCacheV4"
  ],
  "last": {
    "num_demotions": 222,
    "num_demote_failures": 0,
    "num_compressed_entries": 222,
    "num_compressed_hits": 61,
    "num_restore_attempts": 61,
    "num_restore_success": 61,
    "num_restore_fallback": 0,
    "num_compressed_freed": 0,
    "num_temporary_restore_pages": 901,
    "num_restore_pages_released": 901,
    "num_restore_rejected_small_prefix": 0,
    "original_estimated_bytes": 39559790592,
    "compressed_estimated_bytes_4bit": 8686023808,
    "compressed_storage_bytes": 8686023808,
    "active_original_estimated_bytes": 39559790592,
    "active_compressed_estimated_bytes_4bit": 8686023808,
    "active_compressed_storage_bytes": 8686023808,
    "last_estimated_compression_ratio": 4.555160142348755,
    "last_storage_compression_ratio": 4.555160142348755,
    "num_demote_rejected_pool_full": 0,
    "num_kernel_restore_calls": 6776,
    "num_kernel_restore_fallback": 0,
    "kernel_restore_tokens": 901,
    "kernel_restore_elements": 51666944,
    "active_estimated_compression_ratio": 4.554418853372777,
    "active_storage_compression_ratio": 4.554418853372777,
    "gpu_memory_allocated_bytes": 48173765632,
    "gpu_memory_reserved_bytes": 48643440640,
    "gpu_max_memory_allocated_bytes": 48294919680,
    "gpu_max_memory_reserved_bytes": 48643440640,
    "compressed_pool_capacity_bytes": 42949672960,
    "compressed_pool_used_bytes": 8686023808,
    "compressed_pool_free_bytes": 34263649152,
    "compressed_pool_utilization": 0.20223725140094756,
    "compressed_pool_q_used_bytes": 7913371648,
    "compressed_pool_q4_used_bytes": 5936795648,
    "compressed_pool_q2_used_bytes": 1976576000,
    "compressed_pool_scale_used_bytes": 618121728,
    "compressed_pool_ids_used_bytes": 154530432,
    "compressed_pool_q4_capacity_bytes": 19327352832,
    "compressed_pool_q2_capacity_bytes": 6442450944,
    "compressed_pool_scale_capacity_bytes": 10737418240,
    "compressed_pool_ids_capacity_bytes": 6442450944,
    "normal_pool_capacity_bytes": 3758211072,
    "estimated_effective_kv_capacity_bytes": 199369011347.21896,
    "estimated_capacity_gain_vs_normal_pool": 53.048912774641245,
    "_zipcache_version": "ZipCacheV4"
  },
  "max_active_compression_ratio": 4.554418853372777,
  "last_active_compression_ratio": 4.554418853372777,
  "max_active_original_estimated_bytes": 39559790592,
  "max_active_compressed_estimated_bytes": 8686023808,
  "max_num_compressions_or_demotions": 222,
  "max_num_decompressions_or_restores": 61
}
```
