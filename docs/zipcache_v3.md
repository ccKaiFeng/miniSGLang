# ZipCache V3 源码阅读与项目说明

本文档对应 `ZipCacheV3` 分支。这个分支以 `main` 为基线，只保留 ZipCache v3 的代码路径，便于和原 miniSGLang 对比阅读。

## 1. 整体架构

miniSGLang 原始 KV cache 路径可以简化为：

```text
请求 prompt
  -> radix prefix cache 查找可复用前缀
  -> CacheManager 分配 normal KV pool page
  -> prefill/decode 写入 page_table
  -> attention backend 按 page_table 读取 fp16/bf16 KV
  -> 请求结束后把可复用前缀插入 radix tree
```

原始设计里，radix tree node 保存的是 normal KV pool 的物理 token index。只要 prefix cache 还保留，这些 KV 就长期占用 normal pool 显存。

ZipCache v3 的核心思路是把 KV cache 分成两级：

```text
normal KV pool:
  小容量 fp16/bf16 工作区
  只保存当前 attention 必须直接读取的 KV
  也保存 compressed hit 后临时恢复出的 KV

compressed pool:
  大容量 GPU 压缩归档
  长期保存已经冷却的 radix prefix KV
  采用 important token 4bit、unimportant token 2bit 的 packed 存储

radix tree:
  仍然按 token prefix 命中
  node 可以处于 fp16 或 compressed 两种状态
  compressed node 命中后先恢复到 normal pool，再进入原 attention 路径
```

关键边界是：

```text
page_table 只允许写 normal KV pool 的物理 index。
compressed pool 的地址不直接交给 attention kernel。
attention 计算公式和 attention kernel 不改。
```

因此 v3 的命中路径是：

```text
radix match compressed node
  -> ZipCacheV3Manager 分配临时 normal pages
  -> compressed pool 解包 + 反量化
  -> 写回 normal KV pool
  -> CacheManager 把 restored indices 写入 page_table
  -> 原 attention backend 正常运行
  -> 请求结束后释放临时 restored pages
```

请求结束时的 demote 路径是：

```text
finished request
  -> radix cache 插入可复用 prefix
  -> ZipCacheV3Manager 读取 node.value 对应的 normal KV
  -> 计算 unimportant token
  -> K/V 分别按 4bit/2bit 量化并 packed 到 compressed pool
  -> radix node 标记为 compressed
  -> 原 normal pages 释放回 normal pool
```

## 2. 与 main 的主要区别

### 2.1 新增 ZipCacheV3Manager

文件：`python/minisgl/zipcache/manager.py`

这个文件是 v3 的核心实现，负责：

- 创建固定大小 GPU compressed pool；
- 按 q4/q2/scale/ids 四类 buffer 管理压缩数据；
- 在请求结束时 demote radix node；
- 在 prefix 命中时 temporary restore；
- 统计压缩条目数、命中数、恢复成功数、压缩比、pool 使用率和显存占用。

当前实现不包含 v1/v2/v4 类，也不依赖自定义 CUDA kernel，阅读时可以把它当成 v3 的唯一 runtime manager。

### 2.2 参数层增加 v3 开关

文件：

- `python/minisgl/engine/config.py`
- `python/minisgl/server/args.py`

新增主要参数：

```bash
--enable-zipcache-v3
--zipcache-v3-normal-pool-pages 32768
--zipcache-v3-compressed-pool-mb 40960
--zipcache-unimportant-ratio 0.4
--zipcache-k-important-bit 4
--zipcache-k-unimportant-bit 2
--zipcache-v-important-bit 4
--zipcache-v-unimportant-bit 2
--zipcache-v3-min-restore-tokens 0
--zipcache-stats-interval 10
```

默认关闭 ZipCache。不开 `--enable-zipcache-v3` 时，main 的原始路径不变。

### 2.3 Engine 接入 compressed pool

文件：`python/minisgl/engine/engine.py`

差异：

- `enable_zipcache_v3=True` 时创建 `ZipCacheV3Manager`；
- `--zipcache-v3-normal-pool-pages` 可以覆盖 normal KV pool 页数；
- 关闭服务时输出一次 `[ZipCacheV3] stats`；
- 默认关闭 CUDA Graph，便于观察压缩/恢复开销；
- 可用 `--enable-zipcache-cuda-graph --cuda-graph-max-bs N` 做 CUDA Graph 对比实验。

### 2.4 radix tree 增加 compressed node 语义

文件：`python/minisgl/kvcache/radix_cache.py`

main 中 radix node 只表示 normal KV cache index。v3 增加：

- `value_kind`：`fp16` 或 `compressed`；
- `compressed_id`：指向 compressed pool entry；
- `mark_compressed()`：node demote 成 compressed；
- `mark_restored()`：node 恢复成 normal；
- `path_nodes()`：返回命中路径上的所有 node；
- compressed node 不再计入 normal pool 的 evictable pages。

这样 radix tree 仍负责 prefix match，但它不再假设命中的 KV 一定已经在 normal pool 里。

### 2.5 CacheManager 接入 demote/restore

文件：`python/minisgl/scheduler/cache.py`

main 的 `match_req()` 直接返回 radix handle。v3 中：

```text
match_req()
  -> radix match
  -> ZipCacheV3Manager.materialize_match()
  -> 如果命中 compressed node，则临时 restore 到 normal pool
```

main 的 `cache_req(finished=True)` 只把 finished prefix 插入 radix cache 并释放尾部。v3 中：

```text
cache_req(finished=True)
  -> 插入 radix cache
  -> 对命中路径上的 normal node 做 demote
  -> demote 成功后释放原 normal pages
  -> 释放本次请求 temporary restored pages
```

额外增加：

- `allocate_token_indices()`：给 restore 分配指定 token 数的 normal indices；
- `release_handle_resources()`：释放 temporary restore pages；
- `carry_handle_resources()`：prefill 结束后把 temporary pages 转交给 decode handle。

### 2.6 Scheduler 传递 manager

文件：`python/minisgl/scheduler/scheduler.py`

差异：

- 初始化 `CacheManager` 时传入 `engine.zipcache_manager`；
- scheduler 空闲时定期输出 ZipCache stats；
- 请求资源释放时调用 `zipcache_manager.free_request()`，当前 v3 中该函数为 no-op，保留接口是为了生命周期清晰。

### 2.7 Prefill 失败路径释放临时资源

文件：`python/minisgl/scheduler/prefill.py`

如果某个请求已经 restore 了 compressed prefix，但后续因为 cache 空间不足无法进入 prefill，v3 会释放刚分配的 temporary pages，避免 normal pool 泄漏。

## 3. 为什么这样设计

### 3.1 不直接修改 attention kernel

v3 的目标是先做 runtime 层原型，验证压缩缓存是否能带来显存容量优势。attention kernel 仍读取 fp16/bf16 KV，可以保证：

- attention 数学逻辑不变；
- page_table 格式不变；
- FlashAttention / FlashInfer / TensorRT-LLM backend 不需要同步改造；
- restore 失败时可以回退到 recompute，不破坏生成正确性。

### 3.2 normal pool 作为工作区

main 的 normal pool 同时承担“当前计算”和“长期 prefix cache”两种职责。v3 把长期保存职责转移到 compressed pool，使 normal pool 更像工作区：

```text
normal pool 保存当前活跃 batch 必须读写的 fp16/bf16 KV；
compressed pool 保存已经冷却但未来可能复用的历史 KV。
```

因此同等显存预算下，v3 不一定追求 `nvidia-smi` 立即下降，而是追求：

```text
同样显存预算可以缓存更多 prefix / 更长上下文 / 更多 finished request KV。
```

### 3.3 compressed pool 使用固定大小

固定 pool 的原因：

- 启动时显存预算明确；
- 避免运行中频繁 `torch.empty` 造成碎片；
- 统计压缩容量和利用率更清晰；
- 便于和 main 在固定显存预算下做对比。

内部不是一个纯线性 buffer，而是拆成：

```text
q4_buffer: 保存 4bit packed important token
q2_buffer: 保存 2bit packed unimportant token
scale_buffer: 保存 min/step
ids_buffer: 保存 token 相对位置
```

这样比全部用 4bit 存储更省显存，也比完全独立两个 compressed pool 更容易统一配置总预算。

### 3.4 为什么 restore 后仍释放 normal page

v3 默认 `--zipcache-v3-keep-compressed-after-restore` 为 true。命中 compressed prefix 时，只为当前请求临时恢复一份 normal KV；请求结束后释放临时 normal pages，但 compressed entry 继续保留。

这样能避免同一个 shared prefix 被多个请求命中后反复从 compressed pool 中删除，又能让 normal pool 保持小容量工作区语义。

## 4. 启动与测试

main 启动示例：

```bash
PYTHONPATH=python python -m minisgl \
  --model-path /path/to/model \
  --host 0.0.0.0 \
  --port 30000 \
  --cache-type radix \
  --max-running-requests 16 \
  --max-prefill-length 4096
```

ZipCache v3 启动示例：

```bash
PYTHONPATH=python python -m minisgl \
  --model-path /path/to/model \
  --host 0.0.0.0 \
  --port 30001 \
  --cache-type radix \
  --max-running-requests 16 \
  --max-prefill-length 4096 \
  --enable-zipcache-v3 \
  --zipcache-v3-normal-pool-pages 32768 \
  --zipcache-v3-compressed-pool-mb 40960 \
  --zipcache-unimportant-ratio 0.4 \
  --zipcache-k-important-bit 4 \
  --zipcache-k-unimportant-bit 2 \
  --zipcache-v-important-bit 4 \
  --zipcache-v-unimportant-bit 2 \
  --zipcache-v3-min-restore-tokens 0 \
  --zipcache-stats-interval 10
```

如果要和 main 的 CUDA Graph 版本对齐，可以额外加：

```bash
--enable-zipcache-cuda-graph --cuda-graph-max-bs 64
```

重点观察日志：

```text
[ZipCacheV3] compressed pool initialized
[ZipCacheV3] stats: {...}
```

关键指标：

- `num_demotions`：有多少 radix node 被压缩；
- `num_compressed_hits`：有多少次命中 compressed node；
- `num_restore_success`：有多少次成功恢复到 normal pool；
- `active_storage_compression_ratio`：真实 GPU 存储压缩比；
- `compressed_pool_utilization`：compressed pool 使用率；
- `estimated_capacity_gain_vs_normal_pool`：估算的 KV 容量放大倍数；
- `gpu_max_memory_allocated_bytes`：PyTorch 视角的峰值显存。

## 5. 阅读源码建议

推荐按下面顺序阅读：

1. `python/minisgl/kvcache/radix_cache.py`
   先理解 radix node 如何从 `fp16` 状态变成 `compressed` 状态。

2. `python/minisgl/scheduler/cache.py`
   看 `match_req()` 如何在命中后 materialize compressed KV；再看 `cache_req()` 如何在请求结束时 demote。

3. `python/minisgl/zipcache/manager.py`
   看 `_V3CompressedPool` 的四个 buffer，再看 `demote_node()` 和 `materialize_match()`。

4. `python/minisgl/engine/engine.py`
   看启动时如何创建 manager，以及 normal pool pages 如何被 v3 参数覆盖。

5. `python/minisgl/server/args.py`
   看所有 v3 参数如何从命令行传入。

## 6. 设计收益与局限

收益：

- 在不改 attention kernel 的前提下实现 KV cache 压缩复用；
- 显著降低长期 prefix cache 对 normal fp16/bf16 KV pool 的占用；
- 同等 KV 显存预算下可以保存更多历史 prefix；
- 压缩、命中、恢复、释放都有可观测统计，便于实验分析；
- feature flag 关闭时保持 main 行为。

局限：

- restore 在 attention 前发生，会带来额外 PyTorch kernel launch 和 temporary pages；
- normal pool 仍必须容纳当前 batch 的工作集，不能无限缩小；
- 2bit/4bit 量化会带来精度风险，需要 GSM8K、CMMLU、LongBench 等任务验证；
- 当前 saliency 使用 K/V 幅值启发式，不等同于论文里的完整 probe attention 评分。

## 7. 可写入简历的项目说明

项目名称：miniSGLang 推理框架 ZipCache KV Cache 压缩优化

简历描述：

- 基于 miniSGLang 实现 ZipCache V3 原型，在不修改 attention kernel 和 page table 语义的前提下，引入 GPU compressed KV archive，将 radix prefix cache 中冷 KV 从 fp16/bf16 normal pool demote 为 4bit/2bit packed 存储。
- 设计 normal pool + compressed pool 两级 KV cache 架构，使 normal pool 仅作为当前 attention 工作区，compressed pool 作为长期 prefix cache 后端，支持 compressed radix node 命中后的临时 restore 和请求结束后的 page 回收。
- 扩展 radix tree/cache manager/scheduler 生命周期，新增 compressed node 状态、prefix 命中恢复、finished request demotion、temporary page 释放和完整统计指标，实现 feature flag 关闭时与原 miniSGLang 行为保持一致。
- 构建显存、吞吐、TTFT、TPOT、压缩比和正确性对比实验，验证在共享前缀与长上下文负载下，ZipCache 能以一定恢复开销换取更高 KV cache 有效容量。

面试讲解重点：

- 为什么 page_table 不能直接指向 compressed pool；
- 为什么 v3 不修改 attention kernel；
- normal pool 和 compressed pool 的职责边界；
- radix node 从 fp16 到 compressed 的状态转换；
- compressed hit 如何恢复并保证请求结束后不泄漏 normal pages；
- 性能下降主要来自 restore/dequant/unpack/scatter，而收益体现在同等显存预算下缓存更多历史 KV。
