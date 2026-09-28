# miniSGLang Experiment Report: main_longbench_long_context_pressure

- mode: `main_longbench_long_context_pressure`
- base_url: `http://127.0.0.1:31000`
- run_dir: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/main_longbench_long_context_pressure/20260914_162134_main_longbench_long_context_pressure`
- started_at: `2026-09-14 16:21:34`
- git_branch: `main`
- git_commit: `2d4f20134e252cfad2bcb5db1c79e1ce9110fc62`

## Server Check

```json
{
  "ok": true,
  "url": "http://127.0.0.1:31000/v1",
  "latency_s": 0.009863175451755524,
  "response": "{\"status\":\"ok\"}"
}
```

## Experiments

| experiment | ok/total | maxed | rps | chunks/s | ttft avg | ttft p90 | e2e avg | tpot avg | gpu max MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| longbench_long_context_pressure | 32/32 | 0 | 1.047 | 266.9 | 2.204 | 3.444 | 7.642 | 0.02141 | 43245 |

## Result Files

- `longbench_long_context_pressure` results: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/main_longbench_long_context_pressure/20260914_162134_main_longbench_long_context_pressure/longbench_long_context_pressure.jsonl`
- `longbench_long_context_pressure` summary: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/main_longbench_long_context_pressure/20260914_162134_main_longbench_long_context_pressure/longbench_long_context_pressure_summary.json`
- `longbench_long_context_pressure` log: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/main_longbench_long_context_pressure/20260914_162134_main_longbench_long_context_pressure/longbench_long_context_pressure.log`

## ZipCache Stats

```json
{
  "num_stats": 0,
  "message": "No [ZipCacheV1]/[ZipCacheV2]/[ZipCacheV3]/[ZipCacheV4] stats found."
}
```
