# miniSGLang Experiment Report: zipcache_13600mb_gsm8k_public_correctness

- mode: `zipcache_13600mb_gsm8k_public_correctness`
- base_url: `http://127.0.0.1:31000`
- run_dir: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/zipcache_13600mb_gsm8k_public_correctness/20260914_164101_zipcache_13600mb_gsm8k_public_correctness`
- started_at: `2026-09-14 16:41:01`
- git_branch: `working-tree`
- git_commit: `b3ece67ca5cd4682f7fe929a5cad93b9bc169780`

## Server Check

```json
{
  "ok": true,
  "url": "http://127.0.0.1:31000/v1",
  "latency_s": 0.008607462048530579,
  "response": "{\"status\":\"ok\"}"
}
```

## Experiments

| experiment | ok/total | maxed | rps | chunks/s | ttft avg | ttft p90 | e2e avg | tpot avg | gpu max MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsm8k_public_correctness | 64/64 | 0 | 0.08737 | 82.62 | 0.0642 | 0.03457 | 11.45 | 0.01201 | 43729 |

## Result Files

- `gsm8k_public_correctness` results: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/zipcache_13600mb_gsm8k_public_correctness/20260914_164101_zipcache_13600mb_gsm8k_public_correctness/gsm8k_public_correctness.jsonl`
- `gsm8k_public_correctness` summary: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/zipcache_13600mb_gsm8k_public_correctness/20260914_164101_zipcache_13600mb_gsm8k_public_correctness/gsm8k_public_correctness_summary.json`
- `gsm8k_public_correctness` log: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/zipcache_13600mb_gsm8k_public_correctness/20260914_164101_zipcache_13600mb_gsm8k_public_correctness/gsm8k_public_correctness.log`
- `gsm8k_public_correctness` correctness eval: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/zipcache_13600mb_gsm8k_public_correctness/20260914_164101_zipcache_13600mb_gsm8k_public_correctness/gsm8k_public_correctness_eval.json`

## Correctness Evaluation

| experiment | judged | correct | accuracy |
| --- | ---: | ---: | ---: |
| gsm8k_public_correctness | 64 | 60 | 0.9375 |

## ZipCache Stats

```json
{
  "num_stats": 40,
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
    "original_estimated_bytes": 97615872,
    "compressed_estimated_bytes_4bit": 21444480,
    "compressed_storage_bytes": 21444480,
    "active_original_estimated_bytes": 97615872,
    "active_compressed_estimated_bytes_4bit": 21444480,
    "active_compressed_storage_bytes": 21444480,
    "last_estimated_compression_ratio": 4.53585030634211,
    "last_storage_compression_ratio": 4.53585030634211,
    "num_demote_rejected_pool_full": 0,
    "active_estimated_compression_ratio": 4.552027934461456,
    "active_storage_compression_ratio": 4.552027934461456,
    "gpu_memory_allocated_bytes": 44680800768,
    "gpu_memory_reserved_bytes": 45283803136,
    "gpu_max_memory_allocated_bytes": 44698253312,
    "gpu_max_memory_reserved_bytes": 45283803136,
    "compressed_pool_capacity_bytes": 14260633600,
    "compressed_pool_used_bytes": 21444480,
    "compressed_pool_free_bytes": 14239189120,
    "compressed_pool_utilization": 0.001503753662109375,
    "compressed_pool_q_used_bytes": 19537920,
    "compressed_pool_q4_used_bytes": 14671872,
    "compressed_pool_q2_used_bytes": 4866048,
    "compressed_pool_scale_used_bytes": 1525248,
    "compressed_pool_ids_used_bytes": 381312,
    "compressed_pool_q4_capacity_bytes": 9412018176,
    "compressed_pool_q2_capacity_bytes": 3137339392,
    "compressed_pool_scale_capacity_bytes": 1283457024,
    "compressed_pool_ids_capacity_bytes": 427819008,
    "normal_pool_capacity_bytes": 13861011456,
    "estimated_effective_kv_capacity_bytes": 78775813966.31964,
    "estimated_capacity_gain_vs_normal_pool": 5.683265915794338,
    "_zipcache_version": "ZipCacheV3"
  },
  "max_active_compression_ratio": 4.552027934461456,
  "last_active_compression_ratio": 4.552027934461456,
  "max_active_original_estimated_bytes": 97615872,
  "max_active_compressed_estimated_bytes": 21444480,
  "max_num_compressions_or_demotions": 2,
  "max_num_decompressions_or_restores": 0
}
```
