# miniSGLang ZipCache v1 实现说明

本文档说明当前仓库中将 ZipCache 部署到 miniSGLang 的 v1 版本。v1 的目标是先把 ZipCache 的关键算法链路跑通：

1. 计算 probe token；
2. 根据 probe attention score 识别 salient / unimportant token；
3. 对 KV cache 做 Key/Value 分别配置的混合 bit 量化；
4. 在 attention kernel 执行前把压缩 KV 解压回 miniSGLang 原始 GPU KV pool；
5. 不修改 CUDA kernel；
6. 不修改 attention 数学公式；
7. feature flag 关闭时保持原始路径不变。

## 1. v1 总体设计

miniSGLang 当前的 KV cache 是启动时一次性分配的 paged GPU KV pool：

```text
MHAKVCache._kv_buffer:
[K/V, layer, page, page_size, local_kv_heads, head_dim]
```

attention backend 会把每层新生成的 K/V 写入这个 pool，然后 FlashAttention / FlashInfer / TensorRT-LLM kernel 通过 page table 读取普通 fp16/bf16 KV。

因此 v1 不直接替换 KV pool，也不让 kernel 读取 int2/int4 压缩数据，而是在 Python runtime 层做：

```text
store_kv(k, v)
  -> ZipCacheV1.before_attention()
       如果该 req/layer 之前已有压缩状态，则解压回原 GPU KV pool
       根据 probe token 计算 attention score
       选择 unimportant token
  -> 原始 attention kernel
  -> ZipCacheV1.after_attention()
       gather 当前 req/layer 的 KV
       对 salient token 用高 bit，对 unimportant token 用低 bit
       将压缩状态保存到 CPU
```

下一次同一个请求、同一层执行 attention 前，v1 会把 CPU 压缩状态解压回 GPU KV pool，然后继续走原始 attention kernel。

## 2. 当前 v1 已实现功能

### 2.1 probe token 计算

实现位置：

```text
python/minisgl/zipcache/manager.py
```

函数：

```python
_make_probe_ids()
```

策略与 ZipCache 论文/原代码保持一致的方向：

- 从当前 query 序列中取最近约 5% token；
- 再随机取约 5% token；
- 合并后作为 probe token；
- 序列很短时至少取 1 个 token。

### 2.2 salient / unimportant token 识别

实现位置：

```python
ZipCacheV1Manager._select_unimportant_ids()
_normalized_attention_scores()
```

输入：

- 当前层 probe query：`probe_q`
- 当前层完整 KV pool 中的 K：`k_seq`

计算方式：

```text
score = softmax(Q_probe @ K^T / sqrt(head_dim))
token_score = 对 probe/head/group 求和
token_score = 按 token 可见长度做归一化
score 最小的一部分 token -> unimportant token
```

GQA 情况下，v1 会把 Q head 按 KV head 分组，按 KV head 统计 token 分数。

参数：

```text
--zipcache-unimportant-ratio
```

默认值：

```text
0.4
```

也就是约 40% token 会被选为低 bit 量化对象。

为避免刚写入的新 token 马上低精度化，v1 默认保护最近 token：

```text
--zipcache-protect-recent-tokens 1
```

### 2.3 KV cache 多精度压缩

实现位置：

```python
_quantize_mixed()
_quantize_part()
_dequantize_mixed()
```

压缩对象：

```text
当前请求 req.uid
当前 layer_id
当前请求 page_table 指向的物理 KV indices
```

压缩粒度：

```text
每个 token / 每个 KV head / head_dim
```

量化方式：

```text
min_val = x.min(dim=-1)
max_val = x.max(dim=-1)
step = (max_val - min_val) / (2^bit - 1)
q = round((x - min_val) / step)
```

重要 token 和不重要 token 使用不同 bit 数：

```text
Key important:      --zipcache-k-important-bit      默认 4
Key unimportant:    --zipcache-k-unimportant-bit    默认 2
Value important:    --zipcache-v-important-bit      默认 4
Value unimportant:  --zipcache-v-unimportant-bit    默认 2
```

压缩状态保存在 CPU 中，包含：

- token ids 位置；
- quantized uint8 数据；
- min；
- step；
- bit width；
- 原始 estimated bytes；
- 压缩 estimated bytes。

注意：v1 的 `q` 张量物理上用 `uint8` 存储，但统计压缩率时按 bit width 估算有效压缩大小。例如 2-bit 量化按 `numel * 2 / 8` 估算。

### 2.4 attention 前解压

实现位置：

```python
ZipCacheV1Manager.before_attention()
ZipCacheV1Manager._restore_batch()
```

每次 attention backend 调用 kernel 前：

1. 根据 `(req.uid, layer_id)` 查找压缩状态；
2. 将 CPU 压缩数据搬回 GPU；
3. 按 `q * step + min` 解压；
4. scatter 回原始 `MHAKVCache` 的 flat token 位置；
5. 原始 kernel 继续从 fp16/bf16 KV pool 读取。

这保证了：

- 不改 kernel；
- 不改 page table；
- 不改 attention 公式；
- restore 失败时记录日志并继续使用当前 KV pool 内容。

### 2.5 统计信息

v1 统计项包括：

```text
num_probe_runs
num_probe_tokens
num_salient_updates
num_compressions
num_decompressions
num_restore_failures
num_freed_entries
original_estimated_bytes
compressed_estimated_bytes
active_original_estimated_bytes
active_compressed_estimated_bytes
max_active_original_estimated_bytes
max_active_compressed_estimated_bytes
last_compression_ratio
active_compression_ratio
num_active_entries
gpu_memory_allocated_bytes
gpu_memory_reserved_bytes
gpu_max_memory_allocated_bytes
gpu_max_memory_reserved_bytes
```

日志前缀：

```text
[ZipCacheV1]
```

周期性输出间隔：

```text
--zipcache-stats-interval 30
```

设置为 `<= 0` 可关闭周期日志。

## 3. 修改文件清单

### 3.1 新增文件

```text
python/minisgl/zipcache/__init__.py
python/minisgl/zipcache/manager.py
ZipCache/ZipCache/miniSGLang_zipcache_v1.md
```

### 3.2 修改文件

```text
python/minisgl/engine/config.py
python/minisgl/server/args.py
python/minisgl/core.py
python/minisgl/engine/engine.py
python/minisgl/scheduler/scheduler.py
python/minisgl/attention/fa.py
python/minisgl/attention/fi.py
python/minisgl/attention/trtllm.py
```

## 4. 各文件修改说明

### 4.1 `python/minisgl/engine/config.py`

新增 ZipCache v1 配置字段：

```python
enable_zipcache_v1
zipcache_unimportant_ratio
zipcache_k_important_bit
zipcache_k_unimportant_bit
zipcache_v_important_bit
zipcache_v_unimportant_bit
zipcache_streaming_gap
zipcache_protect_recent_tokens
zipcache_stats_interval
```

默认 `enable_zipcache_v1=False`，所以不带参数启动时原行为不变。

### 4.2 `python/minisgl/server/args.py`

新增命令行参数：

```text
--enable-zipcache-v1
--zipcache-unimportant-ratio
--zipcache-k-important-bit
--zipcache-k-unimportant-bit
--zipcache-v-important-bit
--zipcache-v-unimportant-bit
--zipcache-streaming-gap
--zipcache-protect-recent-tokens
--zipcache-stats-interval
```

当启用 `--enable-zipcache-v1` 时，自动设置：

```text
cuda_graph_max_bs = 0
```

原因：v1 在 attention 前后有 Python/CPU 压缩恢复逻辑，不适合被 CUDA Graph capture 固化。

### 4.3 `python/minisgl/core.py`

`Context` 中新增：

```python
zipcache_manager
```

使 attention backend 能通过全局 context 找到 ZipCache v1 manager。

### 4.4 `python/minisgl/engine/engine.py`

在 KV pool 和 page table 创建后，如果启用 ZipCache v1，则创建：

```python
ZipCacheV1Manager(config, kv_pool, page_table)
```

并挂到：

```python
self.zipcache_manager
self.ctx.zipcache_manager
```

同时在 `_adjust_config()` 中禁用 CUDA Graph。

`shutdown()` 时会输出最终 ZipCache v1 stats。

### 4.5 `python/minisgl/scheduler/scheduler.py`

修改点：

1. `run_when_idle()` 中打印 ZipCache stats；
2. `_free_req_resources()` 中调用：

```python
self.engine.zipcache_manager.free_request(req.uid)
```

用于释放该请求对应的 CPU 压缩状态，避免长时间运行后状态泄漏。

### 4.6 `python/minisgl/attention/fa.py`

在 FlashAttention backend 中：

```text
store_kv()
  -> manager.before_attention()
  -> flash_attn_with_kvcache()
  -> manager.after_attention()
```

kernel 调用参数没有改变。

### 4.7 `python/minisgl/attention/fi.py`

在 FlashInfer backend 中加入同样 hook：

```text
store_kv()
  -> manager.before_attention()
  -> FlashInfer wrapper.run()
  -> manager.after_attention()
```

### 4.8 `python/minisgl/attention/trtllm.py`

在 TensorRT-LLM backend 中加入同样 hook。

原来的 prefill/decode 分支由直接 `return` 改为先保存 `output`，执行压缩 hook 后统一 `return output`。

## 5. 启动方式

示例：

```bash
python -m minisgl \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 30000 \
  --cache-type radix \
  --enable-zipcache-v1 \
  --zipcache-unimportant-ratio 0.4 \
  --zipcache-k-important-bit 4 \
  --zipcache-k-unimportant-bit 2 \
  --zipcache-v-important-bit 4 \
  --zipcache-v-unimportant-bit 2 \
  --zipcache-streaming-gap 100 \
  --zipcache-stats-interval 10
```

原始版本对比启动：

```bash
python -m minisgl \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 30000 \
  --cache-type radix
```

不带 `--enable-zipcache-v1` 时，ZipCache manager 不创建，attention backend 不执行压缩/解压逻辑。

## 6. 如何对比显存利用率

v1 stats 中会输出：

```text
gpu_memory_allocated_bytes
gpu_memory_reserved_bytes
gpu_max_memory_allocated_bytes
gpu_max_memory_reserved_bytes
```

也可以同时使用：

```bash
nvidia-smi
```

重要限制：

miniSGLang 当前 KV pool 是启动时预分配的 fp16/bf16 大 tensor。v1 没有重构 KV pool，也没有修改 kernel 让它直接读取压缩 KV，所以 GPU reserved memory 不会按压缩率下降。v1 的显存统计主要用于观察：

- ZipCache v1 额外 CPU/GPU 临时开销；
- 是否出现峰值显存异常；
- 原始预分配 KV pool 下的真实运行开销。

如果目标是让 `nvidia-smi` 中 KV cache 显存显著下降，后续版本需要引入：

1. 压缩页存储；
2. 解压 workspace；
3. page 状态机；
4. 或者支持压缩 KV 的 attention kernel。

这些都超出了 v1“不大改 KV 框架、不改 kernel”的边界。

## 7. 如何观察压缩率

查看日志：

```text
[ZipCacheV1] stats: {
  ...
  "active_original_estimated_bytes": ...,
  "active_compressed_estimated_bytes": ...,
  "active_compression_ratio": ...
}
```

含义：

- `active_original_estimated_bytes`：当前活跃压缩状态如果用原 fp16/bf16 KV 表示，需要多少字节；
- `active_compressed_estimated_bytes`：按当前 mixed precision bit width 估算的压缩字节；
- `active_compression_ratio`：二者比值，越大表示压缩越明显。

默认 4-bit important + 2-bit unimportant，且 40% token 低 bit，理论上会比 fp16 KV 小很多，但实际还包含 min/step metadata。

## 8. 如何验证正确率

建议用 greedy decode 对比：

1. 原始 miniSGLang，不启用 ZipCache v1；
2. 启用 ZipCache v1；
3. 使用相同模型、相同 prompt、相同采样参数；
4. 设置 `temperature=0` 或 greedy；
5. 对比输出文本和 benchmark 指标。

由于 v1 确实会把历史 KV 经过量化再解压，输出可能与原始 fp16/bf16 路径存在差异。这是预期现象。正确率验证应使用任务数据集，例如 GSM8K、LongBench 或你自己的 shared-prefix 测试集。

## 9. 当前限制

1. v1 不修改 kernel，所以 kernel 仍读取 fp16/bf16 KV pool。
2. v1 不重构 KV page allocator，所以 GPU KV pool 仍会完整预分配。
3. v1 的压缩状态保存在 CPU，解压时会产生 CPU/GPU 传输开销。
4. v1 主要验证算法正确性和压缩率，不是最终性能版本。
5. v1 saliency 统计使用额外 PyTorch attention score 计算，会增加运行时间。
6. v1 默认关闭 CUDA Graph。
7. v1 的 mixed precision 量化是工程化 affine min/step 版本，不是完整照搬 ZipCache 原仓库所有 packing/GEAR/outlier 变体。

## 10. 已完成的静态验证

已执行：

```bash
python -m compileall -q python/minisgl
```

语法检查通过。

真实 GPU 推理需要本地具备：

- CUDA；
- PyTorch；
- sgl-kernel / flashinfer 等 miniSGLang 运行依赖；
- 可加载的模型权重。

## 11. 后续 v2 建议

如果 v1 正确率和压缩率符合预期，v2 应重点解决真实显存下降：

1. 给 KV page 增加状态：`FP_GPU`、`COMPRESSED_CPU`、`RESTORED_GPU`；
2. 让部分冷 page 从 GPU KV pool 迁移到压缩 CPU/GPU archive；
3. attention 前只把当前 batch 需要的压缩 page 解压到有限 workspace；
4. page_table 指向 workspace 或临时 restored page；
5. 与 radix prefix cache 的 eviction/restore 路径结合；
6. 最后再考虑 kernel 直接读取低 bit KV。

v1 的意义是先验证 ZipCache 算法在 miniSGLang runtime 中可插入、可运行、可统计，并明确后续显存优化需要改动的边界。
