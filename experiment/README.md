# miniSGLang main vs ZipCache 实验说明

本目录用于在云服务器上对比 `main` 分支和 `ZipCache` 分支的推理性能、显存占用、压缩率和输出正确性。

推荐做法是：同一台机器、同一个模型、同一套参数，分别启动两个服务端口：

```text
main 分支:     http://127.0.0.1:30000
ZipCache 分支: http://127.0.0.1:30001
```

然后用本目录的同一套 workload 对两个服务发请求，生成 JSONL 结果，最后运行对比脚本。

## 1. 目录内容

```text
experiment/
├── README.md
├── run_all_experiments.py
├── bench_openai_stream.py
├── compare_results.py
├── parse_zipcache_log.py
├── data/
│   ├── shared_prefix.jsonl
│   ├── mixed_length.jsonl
│   └── correctness.jsonl
└── results/
```

文件作用：

- `run_all_experiments.py`：一键运行多组实验，把结果统一保存到 `experiment/logs/<时间>_<模式>/`。
- `bench_openai_stream.py`：对 OpenAI 兼容 `/v1/chat/completions` 发流式请求，统计 TTFT、总延迟、吞吐和显存。
- `compare_results.py`：对比 main 和 ZipCache 两份 benchmark 结果。
- `parse_zipcache_log.py`：从 ZipCache 分支服务日志中提取 `[ZipCacheV1] stats`，汇总压缩率。
- `data/shared_prefix.jsonl`：共享长前缀 workload，用于测试 prefix cache 和 ZipCache 的长上下文表现。
- `data/mixed_length.jsonl`：混合长度 workload，用于模拟普通在线服务。
- `data/correctness.jsonl`：小规模确定性问题，用于初步检查输出是否明显错误。

## 2. 对比指标

### 2.1 延迟指标

| 指标 | 含义 |
| --- | --- |
| TTFT | Time To First Token，发出请求到收到第一个非空 token 的时间。 |
| E2E latency | 请求总耗时，从发出请求到收到 `[DONE]`。 |
| TPOT | Time Per Output Token，近似 `(E2E - TTFT) / (输出 token 数 - 1)`。 |
| output tokens/s | 按请求实际收到的流式 chunk 数估算输出吞吐。 |
| request throughput | 每秒完成请求数。 |

注意：miniSGLang 当前 API 没有返回真实 token usage，所以脚本用“收到的非空 streaming chunk 数”近似输出 token 数。它适合做同版本对比，但不是严格 tokenizer token 数。

### 2.2 显存指标

脚本会周期性调用：

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
```

记录：

- `gpu_memory_used_mb_min`
- `gpu_memory_used_mb_max`
- `gpu_memory_used_mb_avg`

ZipCache v1 的重要限制：

当前 v1 不改 kernel，也不重构 miniSGLang 的预分配 KV pool，因此 `nvidia-smi` 的显存使用不一定按压缩率下降。v1 的显存对比主要观察：

- ZipCache 额外临时显存开销；
- 是否有显存峰值异常；
- 后续 v2 做压缩页/解压 workspace 前的 baseline。

### 2.3 ZipCache 压缩率指标

ZipCache 分支服务端会输出：

```text
[ZipCacheV1] stats: {...}
```

用：

```bash
python experiment/parse_zipcache_log.py --log zipcache_server.log
```

可以提取：

- `active_original_estimated_bytes`
- `active_compressed_estimated_bytes`
- `active_compression_ratio`
- `max_active_original_estimated_bytes`
- `max_active_compressed_estimated_bytes`
- `num_compressions`
- `num_decompressions`

## 3. 启动服务

### 3.1 main 分支

在 main 分支目录：

```bash
git switch main

PYTHONPATH=python python -m minisgl \
  --model-path /path/to/model \
  --host 0.0.0.0 \
  --port 30000 \
  --cache-type radix \
  --max-running-requests 16 \
  --max-prefill-length 4096 \
  2>&1 | tee main_server.log
```

### 3.2 ZipCache 分支

在 ZipCache 分支目录：

```bash
git switch ZipCache

PYTHONPATH=python python -m minisgl \
  --model "/root/autodl-tmp/modelscope-cache/models/Qwen/Qwen3-0___6B" \
  --host 0.0.0.0 \
  --port 30001 \
  --cache-type radix \
  --max-running-requests 16 \
  --max-prefill-length 4096 \
  --enable-zipcache-v1 \
  --zipcache-unimportant-ratio 0.4 \
  --zipcache-k-important-bit 4 \
  --zipcache-k-unimportant-bit 2 \
  --zipcache-v-important-bit 4 \
  --zipcache-v-unimportant-bit 2 \
  --zipcache-stats-interval 10 \
  2>&1 | tee zipcache_server.log
```

如果你只有一个工作目录，先测试 main，保存结果；再切换 ZipCache，启动同样模型和参数测试。

## 4. 一键运行全部实验

你只需要先启动 miniSGLang 服务，然后在另一个终端执行一键脚本。

### 4.1 测 main 分支

如果 main 服务跑在 `30000`：

```bash
python experiment/run_all_experiments.py \
  --mode main \
  --base-url http://127.0.0.1:30000
```

结果会保存到：

```text
experiment/logs/<时间>_main/
```

目录中包含：

```text
manifest.json
shared_prefix.jsonl
shared_prefix_summary.json
shared_prefix.log
mixed_length.jsonl
mixed_length_summary.json
mixed_length.log
correctness.jsonl
correctness_summary.json
correctness.log
all_results_summary.json
report.md
```

其中 `report.md` 会说明当前运行在哪个模式下，以及每个实验的核心结果。

### 4.2 测 ZipCache 分支

如果 ZipCache 服务跑在 `30001`，并且服务端日志保存为 `zipcache_server.log`：

```bash
python experiment/run_all_experiments.py \
  --mode zipcache \
  --base-url http://127.0.0.1:30001 \
  --server-log zipcache_server.log  \
  --gpu-sample-interval 0.5 
```

结果会保存到：

```text
experiment/logs/<时间>_zipcache/
```

如果传入 `--server-log`，脚本还会自动解析 `[ZipCacheV1] stats`，生成：

```text
zipcache_stats_summary.json
parse_zipcache_log.log
```

### 4.3 常用参数

```bash
python experiment/run_all_experiments.py \
  --mode main \
  --base-url http://127.0.0.1:30000 \
  --log-root experiment/logs \
  --gpu-sample-interval 0.5 \
  --timeout 600
```

参数说明：

- `--mode`：本次实验模式标签，例如 `main` 或 `zipcache`。
- `--base-url`：已经启动的 miniSGLang 服务地址。
- `--log-root`：结果根目录，默认 `experiment/logs`。
- `--server-log`：可选，ZipCache 服务端日志路径。
- `--gpu-sample-interval`：显存采样间隔，单位秒；设为 `0` 可关闭。
- `--timeout`：单个请求超时时间。

## 5. 单独运行 benchmark

如果你只想单独跑某一组 workload，可以直接调用 `bench_openai_stream.py`。

### 5.1 shared-prefix 性能测试

main：

```bash
python experiment/bench_openai_stream.py \
  --base-url http://127.0.0.1:30000 \
  --dataset experiment/data/shared_prefix.jsonl \
  --output experiment/results/main_shared_prefix.jsonl \
  --summary experiment/results/main_shared_prefix_summary.json \
  --concurrency 4 \
  --repeat 2 \
  --max-tokens 128 \
  --gpu-sample-interval 0.5
```

ZipCache：

```bash
python experiment/bench_openai_stream.py \
  --base-url http://127.0.0.1:30001 \
  --dataset experiment/data/shared_prefix.jsonl \
  --output experiment/results/zipcache_shared_prefix.jsonl \
  --summary experiment/results/zipcache_shared_prefix_summary.json \
  --concurrency 4 \
  --repeat 2 \
  --max-tokens 128 \
  --gpu-sample-interval 0.5
```

对比：

```bash
python experiment/compare_results.py \
  --baseline experiment/results/main_shared_prefix.jsonl \
  --candidate experiment/results/zipcache_shared_prefix.jsonl
```

### 5.2 mixed-length 性能测试

把 dataset 换成：

```text
experiment/data/mixed_length.jsonl
```

其他命令相同。

### 5.3 正确性冒烟测试

建议低并发、greedy、较短输出：

```bash
python experiment/bench_openai_stream.py \
  --base-url http://127.0.0.1:30000 \
  --dataset experiment/data/correctness.jsonl \
  --output experiment/results/main_correctness.jsonl \
  --summary experiment/results/main_correctness_summary.json \
  --concurrency 1 \
  --repeat 1 \
  --max-tokens 96

python experiment/bench_openai_stream.py \
  --base-url http://127.0.0.1:30001 \
  --dataset experiment/data/correctness.jsonl \
  --output experiment/results/zipcache_correctness.jsonl \
  --summary experiment/results/zipcache_correctness_summary.json \
  --concurrency 1 \
  --repeat 1 \
  --max-tokens 96

python experiment/compare_results.py \
  --baseline experiment/results/main_correctness.jsonl \
  --candidate experiment/results/zipcache_correctness.jsonl \
  --show-text
```

## 6. 实验记录建议

每组实验至少记录：

- git 分支；
- git commit；
- 模型路径；
- GPU 型号；
- 启动命令；
- workload；
- concurrency；
- max_tokens；
- main summary；
- ZipCache summary；
- ZipCache stats；
- 服务器日志。

建议每组实验至少跑 3 次，取均值和 p50/p90/p99。

## 7. 结果解读

如果 ZipCache 分支：

- TTFT 变大：可能来自 probe attention 和首次压缩。
- TPOT 变大：可能来自每层解压/压缩 CPU-GPU 传输。
- 压缩率较高但 `nvidia-smi` 显存没有明显下降：符合 v1 预期，因为原始 KV pool 仍预分配。
- 输出与 main 有差异：可能来自 KV 量化误差，需要用任务指标判断是否可接受。

v1 的核心目标是验证“算法链路和压缩率”，不是最终性能最优版本。
