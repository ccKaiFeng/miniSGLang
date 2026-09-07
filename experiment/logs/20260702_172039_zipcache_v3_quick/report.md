# miniSGLang Experiment Report: zipcache_v3_quick

- mode: `zipcache_v3_quick`
- base_url: `http://127.0.0.1:30001`
- run_dir: `experiment/logs/20260702_172039_zipcache_v3_quick`
- started_at: `2026-07-02 17:20:39`
- git_branch: `ZipCache`
- git_commit: `98f0030`

## Server Check

```json
{
  "ok": true,
  "url": "http://127.0.0.1:30001/v1",
  "latency_s": 0.009458685293793678,
  "response": "{\"status\":\"ok\"}"
}
```

## Experiments

| experiment | ok/total | maxed | rps | chunks/s | ttft avg | ttft p90 | e2e avg | tpot avg | gpu max MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsm8k_public_correctness | 64/64 | 0 | 0.1191 | 119.9 | 0.05133 | 0.02218 | 8.398 | 0.008366 | 46499 |
| longbench_long_context_pressure | 32/32 | 0 | 2.074 | 263.5 | 0.2544 | 0.3619 | 1.928 | 0.01328 | 46817 |
| public_shared_prefix_serial | 64/64 | 0 | 1.038 | 98.6 | 0.08866 | 0.1074 | 0.9635 | 0.009306 | 46817 |

## Result Files

- `gsm8k_public_correctness` results: `experiment/logs/20260702_172039_zipcache_v3_quick/gsm8k_public_correctness.jsonl`
- `gsm8k_public_correctness` summary: `experiment/logs/20260702_172039_zipcache_v3_quick/gsm8k_public_correctness_summary.json`
- `gsm8k_public_correctness` log: `experiment/logs/20260702_172039_zipcache_v3_quick/gsm8k_public_correctness.log`
- `gsm8k_public_correctness` correctness eval: `experiment/logs/20260702_172039_zipcache_v3_quick/gsm8k_public_correctness_eval.json`
- `longbench_long_context_pressure` results: `experiment/logs/20260702_172039_zipcache_v3_quick/longbench_long_context_pressure.jsonl`
- `longbench_long_context_pressure` summary: `experiment/logs/20260702_172039_zipcache_v3_quick/longbench_long_context_pressure_summary.json`
- `longbench_long_context_pressure` log: `experiment/logs/20260702_172039_zipcache_v3_quick/longbench_long_context_pressure.log`
- `public_shared_prefix_serial` results: `experiment/logs/20260702_172039_zipcache_v3_quick/public_shared_prefix_serial.jsonl`
- `public_shared_prefix_serial` summary: `experiment/logs/20260702_172039_zipcache_v3_quick/public_shared_prefix_serial_summary.json`
- `public_shared_prefix_serial` log: `experiment/logs/20260702_172039_zipcache_v3_quick/public_shared_prefix_serial.log`

## Correctness Evaluation

| experiment | judged | correct | accuracy |
| --- | ---: | ---: | ---: |
| gsm8k_public_correctness | 64 | 44 | 0.6875 |

## ZipCache Stats

```json
{
  "num_stats": 0,
  "message": "No [ZipCacheV1]/[ZipCacheV2]/[ZipCacheV3]/[ZipCacheV4] stats found."
}
```
