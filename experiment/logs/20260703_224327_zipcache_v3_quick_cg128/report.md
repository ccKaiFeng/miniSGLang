# miniSGLang Experiment Report: zipcache_v3_quick_cg128

- mode: `zipcache_v3_quick_cg128`
- base_url: `http://127.0.0.1:30001`
- run_dir: `experiment/logs/20260703_224327_zipcache_v3_quick_cg128`
- started_at: `2026-07-03 22:43:27`
- git_branch: `ZipCache`
- git_commit: `1fedece`

## Server Check

```json
{
  "ok": true,
  "url": "http://127.0.0.1:30001/v1",
  "latency_s": 0.008681941777467728,
  "response": "{\"status\":\"ok\"}"
}
```

## Experiments

| experiment | ok/total | maxed | rps | chunks/s | ttft avg | ttft p90 | e2e avg | tpot avg | gpu max MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsm8k_public_correctness | 64/64 | 0 | 0.412 | 414.8 | 0.01453 | 0.0145 | 2.427 | 0.00243 | 46731 |
| longbench_long_context_pressure | 32/32 | 0 | 2.944 | 373.9 | 0.258 | 0.3781 | 1.358 | 0.008733 | 47033 |
| public_shared_prefix_serial | 64/64 | 0 | 2.351 | 223.4 | 0.08821 | 0.1091 | 0.4253 | 0.003586 | 47081 |

## Result Files

- `gsm8k_public_correctness` results: `experiment/logs/20260703_224327_zipcache_v3_quick_cg128/gsm8k_public_correctness.jsonl`
- `gsm8k_public_correctness` summary: `experiment/logs/20260703_224327_zipcache_v3_quick_cg128/gsm8k_public_correctness_summary.json`
- `gsm8k_public_correctness` log: `experiment/logs/20260703_224327_zipcache_v3_quick_cg128/gsm8k_public_correctness.log`
- `gsm8k_public_correctness` correctness eval: `experiment/logs/20260703_224327_zipcache_v3_quick_cg128/gsm8k_public_correctness_eval.json`
- `longbench_long_context_pressure` results: `experiment/logs/20260703_224327_zipcache_v3_quick_cg128/longbench_long_context_pressure.jsonl`
- `longbench_long_context_pressure` summary: `experiment/logs/20260703_224327_zipcache_v3_quick_cg128/longbench_long_context_pressure_summary.json`
- `longbench_long_context_pressure` log: `experiment/logs/20260703_224327_zipcache_v3_quick_cg128/longbench_long_context_pressure.log`
- `public_shared_prefix_serial` results: `experiment/logs/20260703_224327_zipcache_v3_quick_cg128/public_shared_prefix_serial.jsonl`
- `public_shared_prefix_serial` summary: `experiment/logs/20260703_224327_zipcache_v3_quick_cg128/public_shared_prefix_serial_summary.json`
- `public_shared_prefix_serial` log: `experiment/logs/20260703_224327_zipcache_v3_quick_cg128/public_shared_prefix_serial.log`

## Correctness Evaluation

| experiment | judged | correct | accuracy |
| --- | ---: | ---: | ---: |
| gsm8k_public_correctness | 64 | 44 | 0.6875 |

## ZipCache Stats

```json
{
  "num_stats": 7,
  "versions": [
    "ZipCacheV3"
  ],
  "last": {
    "num_demotions": 288,
    "num_demote_failures": 0,
    "num_compressed_entries": 288,
    "num_compressed_hits": 94,
    "num_restore_attempts": 94,
    "num_restore_success": 94,
    "num_restore_fallback": 0,
    "num_compressed_freed": 0,
    "num_temporary_restore_pages": 1561,
    "num_restore_pages_released": 1561,
    "num_restore_rejected_small_prefix": 0,
    "original_estimated_bytes": 58180419584,
    "compressed_estimated_bytes_4bit": 12774286784,
    "compressed_storage_bytes": 12774286784,
    "active_original_estimated_bytes": 58180419584,
    "active_compressed_estimated_bytes_4bit": 12774286784,
    "active_compressed_storage_bytes": 12774286784,
    "last_estimated_compression_ratio": 4.555160142348755,
    "last_storage_compression_ratio": 4.555160142348755,
    "num_demote_rejected_pool_full": 0,
    "active_estimated_compression_ratio": 4.5544945536115495,
    "active_storage_compression_ratio": 4.5544945536115495,
    "gpu_memory_allocated_bytes": 48174664192,
    "gpu_memory_reserved_bytes": 48725229568,
    "gpu_max_memory_allocated_bytes": 48295197696,
    "gpu_max_memory_reserved_bytes": 48725229568,
    "compressed_pool_capacity_bytes": 42949672960,
    "compressed_pool_used_bytes": 12774286784,
    "compressed_pool_free_bytes": 30175386176,
    "compressed_pool_utilization": 0.29742454141378405,
    "compressed_pool_q_used_bytes": 11637950464,
    "compressed_pool_q4_used_bytes": 8730796032,
    "compressed_pool_q2_used_bytes": 2907154432,
    "compressed_pool_scale_used_bytes": 909069056,
    "compressed_pool_ids_used_bytes": 227267264,
    "compressed_pool_q4_capacity_bytes": 19327352832,
    "compressed_pool_q2_capacity_bytes": 6442450944,
    "compressed_pool_scale_capacity_bytes": 10737418240,
    "compressed_pool_ids_capacity_bytes": 6442450944,
    "normal_pool_capacity_bytes": 3758211072,
    "estimated_effective_kv_capacity_bytes": 199372262647.71722,
    "estimated_capacity_gain_vs_normal_pool": 53.04977789382587,
    "_zipcache_version": "ZipCacheV3"
  },
  "max_active_compression_ratio": 4.5544945536115495,
  "last_active_compression_ratio": 4.5544945536115495,
  "max_active_original_estimated_bytes": 58180419584,
  "max_active_compressed_estimated_bytes": 12774286784,
  "max_num_compressions_or_demotions": 288,
  "max_num_decompressions_or_restores": 94
}
```
