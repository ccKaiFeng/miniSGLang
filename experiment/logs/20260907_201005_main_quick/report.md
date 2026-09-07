# miniSGLang Experiment Report: main_quick

- mode: `main_quick`
- base_url: `http://127.0.0.1:30000`
- run_dir: `experiment/logs/20260907_201005_main_quick`
- started_at: `2026-09-07 20:10:05`
- git_branch: `main`
- git_commit: `2aae790`

## Server Check

```json
{
  "ok": true,
  "url": "http://127.0.0.1:30000/v1",
  "latency_s": 0.011582087725400925,
  "response": "{\"status\":\"ok\"}"
}
```

## Experiments

| experiment | ok/total | maxed | rps | chunks/s | ttft avg | ttft p90 | e2e avg | tpot avg | gpu max MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsm8k_public_correctness | 64/64 | 0 | 0.4399 | 440.1 | 0.01838 | 0.01949 | 2.273 | 0.002226 | 73563 |
| longbench_long_context_pressure | 32/32 | 0 | 4.615 | 586.1 | 0.256 | 0.357 | 0.8664 | 0.004844 | 73679 |
| public_shared_prefix_serial | 64/64 | 0 | 3.493 | 331.8 | 0.06155 | 0.06414 | 0.2862 | 0.00239 | 73727 |

## Result Files

- `gsm8k_public_correctness` results: `experiment/logs/20260907_201005_main_quick/gsm8k_public_correctness.jsonl`
- `gsm8k_public_correctness` summary: `experiment/logs/20260907_201005_main_quick/gsm8k_public_correctness_summary.json`
- `gsm8k_public_correctness` log: `experiment/logs/20260907_201005_main_quick/gsm8k_public_correctness.log`
- `gsm8k_public_correctness` correctness eval: `experiment/logs/20260907_201005_main_quick/gsm8k_public_correctness_eval.json`
- `longbench_long_context_pressure` results: `experiment/logs/20260907_201005_main_quick/longbench_long_context_pressure.jsonl`
- `longbench_long_context_pressure` summary: `experiment/logs/20260907_201005_main_quick/longbench_long_context_pressure_summary.json`
- `longbench_long_context_pressure` log: `experiment/logs/20260907_201005_main_quick/longbench_long_context_pressure.log`
- `public_shared_prefix_serial` results: `experiment/logs/20260907_201005_main_quick/public_shared_prefix_serial.jsonl`
- `public_shared_prefix_serial` summary: `experiment/logs/20260907_201005_main_quick/public_shared_prefix_serial_summary.json`
- `public_shared_prefix_serial` log: `experiment/logs/20260907_201005_main_quick/public_shared_prefix_serial.log`

## Correctness Evaluation

| experiment | judged | correct | accuracy |
| --- | ---: | ---: | ---: |
| gsm8k_public_correctness | 64 | 45 | 0.7031 |

## ZipCache Stats

```json
{
  "num_stats": 0,
  "message": "No [ZipCacheV1]/[ZipCacheV2]/[ZipCacheV3]/[ZipCacheV4] stats found."
}
```
