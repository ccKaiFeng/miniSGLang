# miniSGLang Experiment Report: main_quick

- mode: `main_quick`
- base_url: `http://127.0.0.1:30000`
- run_dir: `experiment/logs/20260702_150321_main_quick`
- started_at: `2026-07-02 15:03:21`
- git_branch: `main`
- git_commit: `2aae790`

## Server Check

```json
{
  "ok": true,
  "url": "http://127.0.0.1:30000/v1",
  "latency_s": 0.010028654709458351,
  "response": "{\"status\":\"ok\"}"
}
```

## Experiments

| experiment | ok/total | maxed | rps | chunks/s | ttft avg | ttft p90 | e2e avg | tpot avg | gpu max MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gsm8k_public_correctness | 64/64 | 0 | 0.1264 | 126.5 | 0.04847 | 0.02185 | 7.911 | 0.007845 | 73273 |
| longbench_long_context_pressure | 32/32 | 0 | 2.853 | 362.4 | 0.2273 | 0.3364 | 1.402 | 0.009321 | 73437 |
| public_shared_prefix_serial | 64/64 | 0 | 1.234 | 117.2 | 0.04081 | 0.04418 | 0.8104 | 0.008187 | 73437 |

## Result Files

- `gsm8k_public_correctness` results: `experiment/logs/20260702_150321_main_quick/gsm8k_public_correctness.jsonl`
- `gsm8k_public_correctness` summary: `experiment/logs/20260702_150321_main_quick/gsm8k_public_correctness_summary.json`
- `gsm8k_public_correctness` log: `experiment/logs/20260702_150321_main_quick/gsm8k_public_correctness.log`
- `gsm8k_public_correctness` correctness eval: `experiment/logs/20260702_150321_main_quick/gsm8k_public_correctness_eval.json`
- `longbench_long_context_pressure` results: `experiment/logs/20260702_150321_main_quick/longbench_long_context_pressure.jsonl`
- `longbench_long_context_pressure` summary: `experiment/logs/20260702_150321_main_quick/longbench_long_context_pressure_summary.json`
- `longbench_long_context_pressure` log: `experiment/logs/20260702_150321_main_quick/longbench_long_context_pressure.log`
- `public_shared_prefix_serial` results: `experiment/logs/20260702_150321_main_quick/public_shared_prefix_serial.jsonl`
- `public_shared_prefix_serial` summary: `experiment/logs/20260702_150321_main_quick/public_shared_prefix_serial_summary.json`
- `public_shared_prefix_serial` log: `experiment/logs/20260702_150321_main_quick/public_shared_prefix_serial.log`

## Correctness Evaluation

| experiment | judged | correct | accuracy |
| --- | ---: | ---: | ---: |
| gsm8k_public_correctness | 64 | 45 | 0.7031 |
