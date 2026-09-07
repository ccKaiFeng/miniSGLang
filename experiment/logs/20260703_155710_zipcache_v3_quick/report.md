# miniSGLang Experiment Report: zipcache_v3_quick

- mode: `zipcache_v3_quick`
- base_url: `http://127.0.0.1:30001`
- run_dir: `experiment/logs/20260703_155710_zipcache_v3_quick`
- started_at: `2026-07-03 15:57:10`
- git_branch: `ZipCache`
- git_commit: `98f0030`

## Server Check

```json
{
  "ok": true,
  "url": "http://127.0.0.1:30001/v1",
  "latency_s": 0.00922766700387001,
  "response": "{\"status\":\"ok\"}"
}
```

## Experiments

| experiment | ok/total | maxed | rps | chunks/s | ttft avg | ttft p90 | e2e avg | tpot avg | gpu max MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsm8k_public_correctness | 64/64 | 0 | 0.1206 | 121.4 | 0.04788 | 0.02198 | 8.307 | 0.008282 | 46499 |
| longbench_long_context_pressure | 32/32 | 0 | 2.105 | 267.4 | 0.2468 | 0.3537 | 1.9 | 0.01312 | 46817 |
| public_shared_prefix_serial | 64/64 | 0 | 1.046 | 99.34 | 0.08746 | 0.106 | 0.9562 | 0.009242 | 46817 |

## Result Files

- `gsm8k_public_correctness` results: `experiment/logs/20260703_155710_zipcache_v3_quick/gsm8k_public_correctness.jsonl`
- `gsm8k_public_correctness` summary: `experiment/logs/20260703_155710_zipcache_v3_quick/gsm8k_public_correctness_summary.json`
- `gsm8k_public_correctness` log: `experiment/logs/20260703_155710_zipcache_v3_quick/gsm8k_public_correctness.log`
- `gsm8k_public_correctness` correctness eval: `experiment/logs/20260703_155710_zipcache_v3_quick/gsm8k_public_correctness_eval.json`
- `longbench_long_context_pressure` results: `experiment/logs/20260703_155710_zipcache_v3_quick/longbench_long_context_pressure.jsonl`
- `longbench_long_context_pressure` summary: `experiment/logs/20260703_155710_zipcache_v3_quick/longbench_long_context_pressure_summary.json`
- `longbench_long_context_pressure` log: `experiment/logs/20260703_155710_zipcache_v3_quick/longbench_long_context_pressure.log`
- `public_shared_prefix_serial` results: `experiment/logs/20260703_155710_zipcache_v3_quick/public_shared_prefix_serial.jsonl`
- `public_shared_prefix_serial` summary: `experiment/logs/20260703_155710_zipcache_v3_quick/public_shared_prefix_serial_summary.json`
- `public_shared_prefix_serial` log: `experiment/logs/20260703_155710_zipcache_v3_quick/public_shared_prefix_serial.log`

## Correctness Evaluation

| experiment | judged | correct | accuracy |
| --- | ---: | ---: | ---: |
| gsm8k_public_correctness | 64 | 44 | 0.6875 |

## ZipCache Stats

```json
{
  "num_stats": 18,
  "versions": [
    "ZipCacheV3"
  ],
  "last": {
    "num_demotions": 264,
    "num_demote_failures": 0,
    "num_compressed_entries": 264,
    "num_compressed_hits": 82,
    "num_restore_attempts": 82,
    "num_restore_success": 82,
    "num_restore_fallback": 0,
    "num_compressed_freed": 0,
    "num_temporary_restore_pages": 1321,
    "num_restore_pages_released": 1321,
    "num_restore_rejected_small_prefix": 0,
    "original_estimated_bytes": 50724552704,
    "compressed_estimated_bytes_4bit": 11137304640,
    "compressed_storage_bytes": 11137304640,
    "active_original_estimated_bytes": 50724552704,
    "active_compressed_estimated_bytes_4bit": 11137304640,
    "active_compressed_storage_bytes": 11137304640,
    "last_estimated_compression_ratio": 4.555160142348755,
    "last_storage_compression_ratio": 4.555160142348755,
    "num_demote_rejected_pool_full": 0,
    "active_estimated_compression_ratio": 4.5544729486720765,
    "active_storage_compression_ratio": 4.5544729486720765,
    "gpu_memory_allocated_bytes": 48079749120,
    "gpu_memory_reserved_bytes": 48530194432,
    "gpu_max_memory_allocated_bytes": 48200496128,
    "gpu_max_memory_reserved_bytes": 48530194432,
    "compressed_pool_capacity_bytes": 42949672960,
    "compressed_pool_used_bytes": 11137304640,
    "compressed_pool_free_bytes": 31812368320,
    "compressed_pool_utilization": 0.2593105807900429,
    "compressed_pool_q_used_bytes": 10146590720,
    "compressed_pool_q4_used_bytes": 7612043264,
    "compressed_pool_q2_used_bytes": 2534547456,
    "compressed_pool_scale_used_bytes": 792571136,
    "compressed_pool_ids_used_bytes": 198142784,
    "compressed_pool_q4_capacity_bytes": 19327352832,
    "compressed_pool_q2_capacity_bytes": 6442450944,
    "compressed_pool_scale_capacity_bytes": 10737418240,
    "compressed_pool_ids_capacity_bytes": 6442450944,
    "normal_pool_capacity_bytes": 3758211072,
    "estimated_effective_kv_capacity_bytes": 199371334722.63257,
    "estimated_capacity_gain_vs_normal_pool": 53.04953098776688,
    "_zipcache_version": "ZipCacheV3"
  },
  "max_active_compression_ratio": 4.5544729486720765,
  "last_active_compression_ratio": 4.5544729486720765,
  "max_active_original_estimated_bytes": 50724552704,
  "max_active_compressed_estimated_bytes": 11137304640,
  "max_num_compressions_or_demotions": 264,
  "max_num_decompressions_or_restores": 82
}
```
