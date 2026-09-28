# miniSGLang Experiment Report: main_gsm8k_public_correctness

- mode: `main_gsm8k_public_correctness`
- base_url: `http://127.0.0.1:31000`
- run_dir: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/main_gsm8k_public_correctness/20260914_162221_main_gsm8k_public_correctness`
- started_at: `2026-09-14 16:22:21`
- git_branch: `main`
- git_commit: `2d4f20134e252cfad2bcb5db1c79e1ce9110fc62`

## Server Check

```json
{
  "ok": true,
  "url": "http://127.0.0.1:31000/v1",
  "latency_s": 0.008439779281616211,
  "response": "{\"status\":\"ok\"}"
}
```

## Experiments

| experiment | ok/total | maxed | rps | chunks/s | ttft avg | ttft p90 | e2e avg | tpot avg | gpu max MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsm8k_public_correctness | 64/64 | 0 | 0.09559 | 91 | 0.02948 | 0.02863 | 10.46 | 0.01096 | 42847 |

## Result Files

- `gsm8k_public_correctness` results: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/main_gsm8k_public_correctness/20260914_162221_main_gsm8k_public_correctness/gsm8k_public_correctness.jsonl`
- `gsm8k_public_correctness` summary: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/main_gsm8k_public_correctness/20260914_162221_main_gsm8k_public_correctness/gsm8k_public_correctness_summary.json`
- `gsm8k_public_correctness` log: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/main_gsm8k_public_correctness/20260914_162221_main_gsm8k_public_correctness/gsm8k_public_correctness.log`
- `gsm8k_public_correctness` correctness eval: `/root/autodl-tmp/miniSGLang/experiment/logs/bench_8b_20260914_161559/main_gsm8k_public_correctness/20260914_162221_main_gsm8k_public_correctness/gsm8k_public_correctness_eval.json`

## Correctness Evaluation

| experiment | judged | correct | accuracy |
| --- | ---: | ---: | ---: |
| gsm8k_public_correctness | 64 | 61 | 0.9531 |

## ZipCache Stats

```json
{
  "num_stats": 0,
  "message": "No [ZipCacheV1]/[ZipCacheV2]/[ZipCacheV3]/[ZipCacheV4] stats found."
}
```
