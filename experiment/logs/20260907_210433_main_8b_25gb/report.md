# miniSGLang Experiment Report: main_8b_25gb

- mode: `main_8b_25gb`
- base_url: `http://127.0.0.1:30000`
- run_dir: `experiment/logs/20260907_210433_main_8b_25gb`
- started_at: `2026-09-07 21:04:33`
- git_branch: `main`
- git_commit: `2d4f201`

## Server Check

```json
{
  "ok": true,
  "url": "http://127.0.0.1:30000/v1",
  "latency_s": 0.009664684534072876,
  "response": "{\"status\":\"ok\"}"
}
```

## Experiments

| experiment | ok/total | maxed | rps | chunks/s | ttft avg | ttft p90 | e2e avg | tpot avg | gpu max MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| public_shared_prefix_serial | 192/192 | 0 | 0.6647 | 84.42 | 0.07352 | 0.07363 | 1.504 | 0.01135 | 43247 |

## Result Files

- `public_shared_prefix_serial` results: `experiment/logs/20260907_210433_main_8b_25gb/public_shared_prefix_serial.jsonl`
- `public_shared_prefix_serial` summary: `experiment/logs/20260907_210433_main_8b_25gb/public_shared_prefix_serial_summary.json`
- `public_shared_prefix_serial` log: `experiment/logs/20260907_210433_main_8b_25gb/public_shared_prefix_serial.log`

## ZipCache Stats

```json
{
  "num_stats": 0,
  "message": "No [ZipCacheV1]/[ZipCacheV2]/[ZipCacheV3]/[ZipCacheV4] stats found."
}
```
