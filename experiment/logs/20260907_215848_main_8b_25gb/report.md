# miniSGLang Experiment Report: main_8b_25gb

- mode: `main_8b_25gb`
- base_url: `http://127.0.0.1:30000`
- run_dir: `experiment/logs/20260907_215848_main_8b_25gb`
- started_at: `2026-09-07 21:58:48`
- git_branch: `main`
- git_commit: `2d4f201`

## Server Check

```json
{
  "ok": true,
  "url": "http://127.0.0.1:30000/v1",
  "latency_s": 0.010673969984054565,
  "response": "{\"status\":\"ok\"}"
}
```

## Experiments

| experiment | ok/total | maxed | rps | chunks/s | ttft avg | ttft p90 | e2e avg | tpot avg | gpu max MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| longbench_long_context_pressure | 32/32 | 0 | 1.042 | 265.6 | 2.216 | 3.484 | 7.679 | 0.02151 | 43247 |

## Result Files

- `longbench_long_context_pressure` results: `experiment/logs/20260907_215848_main_8b_25gb/longbench_long_context_pressure.jsonl`
- `longbench_long_context_pressure` summary: `experiment/logs/20260907_215848_main_8b_25gb/longbench_long_context_pressure_summary.json`
- `longbench_long_context_pressure` log: `experiment/logs/20260907_215848_main_8b_25gb/longbench_long_context_pressure.log`

## ZipCache Stats

```json
{
  "num_stats": 0,
  "message": "No [ZipCacheV1]/[ZipCacheV2]/[ZipCacheV3]/[ZipCacheV4] stats found."
}
```
