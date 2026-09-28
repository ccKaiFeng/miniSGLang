# miniSGLang Experiment Report: main_public_shared_prefix_serial

- mode: `main_public_shared_prefix_serial`
- base_url: `http://127.0.0.1:31000`
- run_dir: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/main_public_shared_prefix_serial/20260914_161626_main_public_shared_prefix_serial`
- started_at: `2026-09-14 16:16:26`
- git_branch: `main`
- git_commit: `2d4f20134e252cfad2bcb5db1c79e1ce9110fc62`

## Server Check

```json
{
  "ok": true,
  "url": "http://127.0.0.1:31000/v1",
  "latency_s": 0.00972696766257286,
  "response": "{\"status\":\"ok\"}"
}
```

## Experiments

| experiment | ok/total | maxed | rps | chunks/s | ttft avg | ttft p90 | e2e avg | tpot avg | gpu max MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| public_shared_prefix_serial | 192/192 | 0 | 0.6603 | 83.85 | 0.06983 | 0.07271 | 1.514 | 0.01147 | 43247 |

## Result Files

- `public_shared_prefix_serial` results: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/main_public_shared_prefix_serial/20260914_161626_main_public_shared_prefix_serial/public_shared_prefix_serial.jsonl`
- `public_shared_prefix_serial` summary: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/main_public_shared_prefix_serial/20260914_161626_main_public_shared_prefix_serial/public_shared_prefix_serial_summary.json`
- `public_shared_prefix_serial` log: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/main_public_shared_prefix_serial/20260914_161626_main_public_shared_prefix_serial/public_shared_prefix_serial.log`

## ZipCache Stats

```json
{
  "num_stats": 0,
  "message": "No [ZipCacheV1]/[ZipCacheV2]/[ZipCacheV3]/[ZipCacheV4] stats found."
}
```
