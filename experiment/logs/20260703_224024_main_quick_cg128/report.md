# miniSGLang Experiment Report: main_quick_cg128

- mode: `main_quick_cg128`
- base_url: `http://127.0.0.1:30000`
- run_dir: `experiment/logs/20260703_224024_main_quick_cg128`
- started_at: `2026-07-03 22:40:24`
- git_branch: `main`
- git_commit: `2aae790`

## Server Check

```json
{
  "ok": true,
  "url": "http://127.0.0.1:30000/v1",
  "latency_s": 0.008507885038852692,
  "response": "{\"status\":\"ok\"}"
}
```

## Experiments

| experiment | ok/total | maxed | rps | chunks/s | ttft avg | ttft p90 | e2e avg | tpot avg | gpu max MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsm8k_public_correctness | 64/64 | 0 | 0.4334 | 433.6 | 0.01368 | 0.01331 | 2.307 | 0.00227 | 73511 |
| longbench_long_context_pressure | 32/32 | 0 | 4.542 | 576.8 | 0.2424 | 0.364 | 0.8804 | 0.005064 | 73627 |
| public_shared_prefix_serial | 64/64 | 0 | 3.88 | 368.6 | 0.03021 | 0.02983 | 0.2577 | 0.00242 | 73675 |

## Result Files

- `gsm8k_public_correctness` results: `experiment/logs/20260703_224024_main_quick_cg128/gsm8k_public_correctness.jsonl`
- `gsm8k_public_correctness` summary: `experiment/logs/20260703_224024_main_quick_cg128/gsm8k_public_correctness_summary.json`
- `gsm8k_public_correctness` log: `experiment/logs/20260703_224024_main_quick_cg128/gsm8k_public_correctness.log`
- `gsm8k_public_correctness` correctness eval: `experiment/logs/20260703_224024_main_quick_cg128/gsm8k_public_correctness_eval.json`
- `longbench_long_context_pressure` results: `experiment/logs/20260703_224024_main_quick_cg128/longbench_long_context_pressure.jsonl`
- `longbench_long_context_pressure` summary: `experiment/logs/20260703_224024_main_quick_cg128/longbench_long_context_pressure_summary.json`
- `longbench_long_context_pressure` log: `experiment/logs/20260703_224024_main_quick_cg128/longbench_long_context_pressure.log`
- `public_shared_prefix_serial` results: `experiment/logs/20260703_224024_main_quick_cg128/public_shared_prefix_serial.jsonl`
- `public_shared_prefix_serial` summary: `experiment/logs/20260703_224024_main_quick_cg128/public_shared_prefix_serial_summary.json`
- `public_shared_prefix_serial` log: `experiment/logs/20260703_224024_main_quick_cg128/public_shared_prefix_serial.log`

## Correctness Evaluation

| experiment | judged | correct | accuracy |
| --- | ---: | ---: | ---: |
| gsm8k_public_correctness | 64 | 45 | 0.7031 |
