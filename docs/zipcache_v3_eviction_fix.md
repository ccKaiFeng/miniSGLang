# ZipCache V3 缓存回收与实验运行

## 修复范围

normal KV pool 缺页时，RadixCache 现在可以先删除无引用的 compressed 叶子，
释放其 compressed entry，再继续驱逐变成叶子的 normal 父节点。
compressed 节点保留的旧 normal indices 已经失效，绝不能再次放回 free_slots。
只有 normal 节点的 indices 会作为 `evict()` 的返回值。

compressed pool 的任何子池分配失败时，manager 先回滚本次节点的所有部分分配，
再尝试按 radix 时间戳驱逐一个无引用的 compressed 叶子并重试。
当前 demote 路径的节点通过 uuid 排除，正在使用的节点通过 ref_count 保护。
没有可回收叶子时保留 normal 节点，返回 None。整条路径可以部分压缩成功，
由上述混合节点驱逐规则保证 normal 父节点后续仍然可回收。

部分匹配 compressed 节点时仍不拆分压缩数据，但插入操作不会覆盖同 key 的旧子树。
本次未插入的尾部继续由请求持有，请求结束时释放，避免丢失旧子树资源。

默认 Q4/Q2/scale/ids 容量比例调整为 0.66/0.22/0.09/0.03。
这是面向当前 4/2 bit workload 的配置，可用原有 CLI 参数覆盖，并非所有模型的最优值。
`num_demote_rejected_pool_full` 统计池满重试次数；`num_demote_failures` 统计最终失败节点次数。
池满重试不是服务崩溃；`num_compressed_freed` 应在持续压力下增长。

## 运行实验

在 GPU 服务器上激活原有 miniSGLang 环境，从仓库根目录执行：

```bash
bash experiment/scripts/run_zipcache_benchmark.sh /absolute/path/to/Qwen3-8B
```

默认复制当前工作区的 python 源码作为 ZipCache 快照，包括未提交修复；
Main 使用 main 提交的源码快照。脚本不会 checkout/stash，也不会结束其他 miniSGLang 服务。
可通过第二个位置参数指定 ZipCache 已提交的分支或 commit：

```bash
bash experiment/scripts/run_zipcache_benchmark.sh /absolute/path/to/Qwen3-8B zipcache-v3-quick-results
```

指定分支时，未提交修改不会进入实验。额外参数包括 `--main-ref`、`--port`、
`--startup-timeout`、`--request-timeout` 和 `--workload-timeout`。
默认单请求超时 120 秒，单 workload 总超时 3600 秒。

8 个 workload 分别为 Main 与 ZipCache 13600 MiB 的 shared-prefix、LongBench、GSM8K，
以及 ZipCache 3400/6800 MiB 的 shared-prefix。每个 workload 启动全新服务，
shared-prefix 内部仍执行 64 个样本重复 3 次，共 192 请求。
这种隔离适合独立性能比较；不能代替跨 workload 的长期服务稳定性实验。
normal pool 默认仍为 94000 pages，Main memory_ratio 为 0.52。
两者总显存并非严格相等，应以记录的实测显存为准。

## 日志与完成条件

全部输出放在 `experiment/logs/bench_8b_时间戳/`：

- `status.json`：整组状态及已完成 workload；失败时保存错误。
- `hardware.csv`：GPU 型号、总显存、驱动版本。
- `zipcache_source.tar.gz`、`working_tree.patch`：默认工作区模式的代码留档。
- 每个 workload 子目录：`server.log`、`client.log`、`server_metadata.json`、`status.json`。
- workload 内的时间戳目录：逐请求 JSONL、summary、report、正确性评估。

检测到 scheduler traceback、server 退出、请求失败、评估缺失或超时即停止整组运行。
只有全部 8 项的请求数与成功数符合预期、GSM8K 评估完整，才生成 `summary.txt`，
其中包含 `BENCHMARK COMPLETE`。HTTP 200 或客户端 exit code=0 本身不代表实验通过。
请求结果逐条 flush 到 JSONL，中断后已完成请求仍可查看。

旧、新目录结构都可用汇总脚本读取；指定单组目录可避免混入其他运行结果：

```bash
python experiment/scripts/summarize_benchmark.py \
  --log-root experiment/logs/bench_8b_时间戳 --pattern "" --output /tmp/comparison.md
```

## 回归测试

```bash
PYTHONPATH=python python -m pytest -o addopts= \
  tests/core/test_zipcache_eviction.py tests/core/test_benchmark_runner.py -q
```

这些 CPU 测试使用真实 PyTorch 量化和 allocator；只替换 CUDA 前缀比较函数。
覆盖 mixed radix 驱逐、锁定与路径保护、page_size 1/4、部分 compressed 匹配、
压缩池反复耗尽、失败回滚、请求结束与临时 restore 的资源回收，以及实验结果校验。
CUDA attention、真实 Qwen3-8B 输出正确性及吞吐量仍需要服务器重跑确认。
