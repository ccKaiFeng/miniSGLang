# miniSGLang / ZipCache 公开数据集实验说明

本目录用于在云服务器上对比 `main`、ZipCache v1/v2/v3/v4 等版本的推理性能、显存占用、ZipCache 压缩/恢复统计和输出正确性。

当前默认测试数据已经切换到公开数据集派生的 workload：

```text
experiment/workloads/
```

旧的手工构造数据仍保留在：

```text
experiment/data/
```

但一键测试脚本默认不再使用 `experiment/data/`。

## 1. 数据目录

```text
experiment/
├── gsm8k/                         # openai/gsm8k 本地 Arrow 数据
├── cmmlu/                         # haonan-li/cmmlu 本地 Arrow 数据
├── longbench/                     # THUDM/LongBench 本地 Arrow 数据
├── ruler/squad.json               # RULER helper 下载得到的 SQuAD 数据
├── synthetic/                     # generated-shared-prefix / random token ids
├── workloads/                     # 默认 benchmark JSONL，公开数据派生
├── run_all_experiments.py         # 一键运行多个 workload
├── bench_openai_stream.py         # OpenAI chat-completions streaming 压测
├── evaluate_correctness.py        # 数字/选择题/文本包含式正确性评估
├── compare_results.py             # main vs ZipCache 单项结果对比
└── parse_zipcache_log.py          # 解析 [ZipCacheV1/V2/V3/V4] stats
```

`experiment/workloads/` 当前包含：

| workload | 来源 | 用途 |
| --- | --- | --- |
| `gsm8k_public_correctness.jsonl` | GSM8K test | 数学推理正确性，数字答案 accuracy |
| `cmmlu_public_correctness.jsonl` | CMMLU 多科目 test | 中文选择题正确性，A/B/C/D accuracy |
| `longbench_public_qa.jsonl` | LongBench | 长上下文问答，近似文本包含式正确性 |
| `longbench_long_context_pressure.jsonl` | LongBench 长样本 | 长上下文性能和 KV cache 显存压力 |
| `public_shared_prefix.jsonl` | LongBench 派生 | 公开数据共享前缀，观察 radix/prefix cache 与 compressed restore |
| `ruler_squad_qa.jsonl` | RULER SQuAD helper | 检索式问答正确性近似评估 |
| `synthetic_shared_prefix.jsonl` | generated-shared-prefix | 强 shared-prefix 压测 |

`synthetic/random_token_ids.jsonl` 目前不进入默认 HTTP 一键测试，因为 miniSGLang 当前 HTTP API 只暴露文本 prompt / chat messages，没有直接暴露 `prompt_token_ids` 路径。

## 2. 重新生成 workload

如果公开 Arrow 数据已经在 `experiment/` 下，可以重新生成默认 JSONL：

```bash
python experiment/prepare_public_workloads.py \
  --root experiment \
  --output-dir experiment/workloads
```

该脚本需要：

```bash
python -m pip install datasets pyarrow
```

云服务器只运行 benchmark 时不需要重新生成，直接使用已提交的 `experiment/workloads/*.jsonl` 即可。

## 3. 启动服务

建议同一台机器、同一个模型、同一套参数分别测试。

main 分支示例：

```bash
git switch main

PYTHONPATH=python python -m minisgl \
  --model-path /root/autodl-tmp/modelscope-cache/models/Qwen/Qwen3-0___6B \
  --host 0.0.0.0 \
  --port 30000 \
  --cache-type radix \
  --max-running-requests 16 \
  --max-prefill-length 4096 \
  2>&1 | tee main_server.log
```

ZipCache v1 示例：

```bash
git switch ZipCache

PYTHONPATH=python python -m minisgl \
  --model-path /root/autodl-tmp/modelscope-cache/models/Qwen/Qwen3-0___6B \
  --host 0.0.0.0 \
  --port 30001 \
  --cache-type radix \
  --max-running-requests 16 \
  --max-prefill-length 4096 \
  --enable-zipcache-v1 \
  --zipcache-stats-interval 10 \
  2>&1 | tee zipcache_v1_server.log
```

ZipCache v2/v3/v4 的启动参数见对应版本说明文档。

## 4. 一键测试

main：

```bash
python experiment/run_all_experiments.py \
  --mode main \
  --base-url http://127.0.0.1:30000 \
  --log-root experiment/logs \
  --gpu-sample-interval 0.5
```

ZipCache：

```bash
python experiment/run_all_experiments.py \
  --mode zipcache_v3 \
  --base-url http://127.0.0.1:30001 \
  --server-log zipcache_v3_server.log \
  --log-root experiment/logs \
  --gpu-sample-interval 0.5
```

结果保存到：

```text
experiment/logs/<时间>_<mode>/
```

其中 `report.md` 汇总每个 workload 的：

```text
RPS
chunks/s
TTFT
E2E
TPOT
gpu max MB
正确性 accuracy
ZipCache stats
```

## 5. 只跑部分测试

查看可用 workload：

```bash
python experiment/run_all_experiments.py \
  --mode list \
  --base-url http://127.0.0.1:30000 \
  --skip-server-check \
  --list-experiments
```

只跑正确性：

```bash
python experiment/run_all_experiments.py \
  --mode main_correctness \
  --base-url http://127.0.0.1:30000 \
  --only gsm8k_public_correctness,cmmlu_public_correctness,ruler_squad_qa
```

正确性 workload 会自动使用更大的 `max_tokens`，并向服务端发送
`ignore_eos=false`。这样 Qwen3 这类会输出 `<think>` 的模型不容易在推理中途被
`max_tokens` 截断；性能压测 workload 仍保留固定长度输出，便于比较吞吐。
如果 `report.md` 中某个正确性实验的 `maxed` 仍然较大，说明仍有请求打满了
`max_tokens`，需要继续提高该 workload 的生成上限。

只跑长上下文和 shared-prefix：

```bash
python experiment/run_all_experiments.py \
  --mode zipcache_v3_pressure \
  --base-url http://127.0.0.1:30001 \
  --server-log zipcache_v3_server.log \
  --only longbench_long_context_pressure,public_shared_prefix,public_shared_prefix_serial,synthetic_shared_prefix
```

## 6. 对比 main 和 ZipCache

假设两次测试目录是：

```text
experiment/logs/2026xxxx_xxxxxx_main/
experiment/logs/2026xxxx_xxxxxx_zipcache_v3/
```

查看总报告：

```bash
cat experiment/logs/2026xxxx_xxxxxx_main/report.md
cat experiment/logs/2026xxxx_xxxxxx_zipcache_v3/report.md
```

单项对比：

```bash
python experiment/compare_results.py \
  --baseline experiment/logs/2026xxxx_xxxxxx_main/public_shared_prefix.jsonl \
  --candidate experiment/logs/2026xxxx_xxxxxx_zipcache_v3/public_shared_prefix.jsonl
```

正确性结果：

```bash
cat experiment/logs/2026xxxx_xxxxxx_main/gsm8k_public_correctness_eval.json
cat experiment/logs/2026xxxx_xxxxxx_zipcache_v3/gsm8k_public_correctness_eval.json
```

ZipCache stats：

```bash
python experiment/parse_zipcache_log.py \
  --log zipcache_v3_server.log
```

## 7. 主要对比指标

性能：

- `request_throughput_rps`
- `output_chunk_throughput_cps`
- `ttft_avg_s / ttft_p90_s`
- `e2e_avg_s / e2e_p90_s`
- `tpot_avg_s / tpot_p90_s`

显存：

- `gpu_memory_used_mb_max`
- ZipCache 日志中的 normal/compressed pool 使用率
- `active_original_estimated_bytes`
- `active_compressed_*_bytes`
- `active_*_compression_ratio`

正确性：

- GSM8K：数字答案 accuracy
- CMMLU：选项 accuracy
- LongBench/RULER：近似 answer contains accuracy
- main 与 ZipCache 输出是否出现明显语义退化

v3/v4 的核心优势不要只看进程总 `nvidia-smi` 是否下降。更重要的是在相同或更低 KV 显存预算下，ZipCache 是否能保存更多历史 prefix、触发更多 compressed hits/restores，并保持正确性接近 main。
