# miniSGLang ZipCache v4 方案设计

## 1. v4 目标

v3 的性能下降主要来自 compressed prefix 命中后的 restore 路径：

```text
compressed pool
  -> Python/PyTorch unpack
  -> Python/PyTorch dequant
  -> 写回 normal KV pool
  -> FlashAttention / FlashInfer attention
```

这条路径虽然没有修改 attention kernel，但会产生额外临时 tensor、多个 PyTorch
kernel launch、advanced indexing 写回和 Python 调度开销。v4 的目标是：

```text
不修改 FlashAttention 的 attention 数学逻辑；
不让 FlashAttention 直接读取 int2/int4 KV；
新增 miniSGLang 自己的 ZipCache CUDA kernel；
把 demote 时的 min/max + quantize + pack 放到 CUDA kernel；
把 compressed KV 的 unpack + dequant + scatter 写回 normal KV pool 融合到一个 kernel；
然后继续调用原 FlashAttention / FlashInfer。
```

因此 v4 的核心不是“重写 FlashAttention”，而是把 v3 的解压恢复路径从
Python/PyTorch tensor 运算改成专用 CUDA kernel。

## 2. 为什么不直接修改 FlashAttention

ZipCache 论文和原代码的关键特点是：

```text
1. 用 probe token 识别 salient / unimportant token；
2. 对不同 token 使用不同 bit 数量化；
3. 真实 attention 输出仍然使用 FlashAttention；
4. 不改变 attention 公式。
```

原 ZipCache 代码中，compressed `past_key_value` 在 attention 前通过
`decompress()` 恢复成普通 tensor，再交给 FlashAttention。也就是说，它没有让
FlashAttention 在内部直接读取压缩 KV。

miniSGLang v4 也保持这个原则。原因：

```text
1. 直接改 FlashAttention / FlashInfer 内部读 KV 的逻辑，工程风险很高；
2. FlashAttention 的 tile、softmax、GQA/MQA、causal mask、split-k 等优化复杂；
3. 直接在 attention kernel 里混合读取 fp16 normal page 和 int2/int4 compressed page，
   会引入新的 page table 语义和访存分支；
4. v4 当前更应该先把 v3 最大的 Python/PyTorch restore 开销去掉。
```

所以 v4 修改的是 miniSGLang 自己的 kernel 包，而不是外部 FlashAttention 计算内核。

## 3. v4 数据路径

v3：

```text
radix compressed hit
  -> allocate normal pages
  -> Python/PyTorch dequantize compressed entry
  -> normal KV pool
  -> FlashAttention / FlashInfer
```

v4：

```text
radix node demote
  -> ZipCache fused CUDA compress kernel
       compute min / step
       quantize salient token to 4bit
       quantize unimportant token to 2bit
       pack q into q4/q2 compressed pool
  -> free normal pages

radix compressed hit
  -> allocate normal pages
  -> ZipCache fused CUDA restore kernel
       unpack 4bit / 2bit
       load min / step
       dequantize to fp16 / bf16
       scatter to normal KV pool physical indices
  -> FlashAttention / FlashInfer
```

注意：v4 仍然需要 normal KV working pages，因为现有 attention backend 的输入仍是：

```text
k_cache(layer_id): normal fp16/bf16 tensor
v_cache(layer_id): normal fp16/bf16 tensor
page_table: normal physical token index
```

v4 只是让“恢复到 normal pool”这一步更快。

## 4. v4 修改范围

### 4.1 Python 配置

新增参数：

```text
--enable-zipcache-v4
--zipcache-v4-normal-pool-pages
--zipcache-v4-compressed-pool-mb
--zipcache-v4-compressed-pool-ratio
--zipcache-v4-demote-on-finish / --no-zipcache-v4-demote-on-finish
--zipcache-v4-use-kernel-compress / --no-zipcache-v4-use-kernel-compress
--zipcache-v4-use-kernel-restore / --no-zipcache-v4-use-kernel-restore
```

v4 继续复用 ZipCache mixed precision 参数：

```text
--zipcache-unimportant-ratio
--zipcache-k-important-bit
--zipcache-k-unimportant-bit
--zipcache-v-important-bit
--zipcache-v-unimportant-bit
--zipcache-protect-recent-tokens
```

compressed pool 内部 q4/q2/scale/ids 比例沿用 v3 的配置思想，也提供 v4 版本：

```text
--zipcache-v4-q4-pool-ratio
--zipcache-v4-q2-pool-ratio
--zipcache-v4-scale-pool-ratio
--zipcache-v4-ids-pool-ratio
```

### 4.2 ZipCache manager

新增：

```text
ZipCacheV4Manager
```

它继承 v3 的大部分 radix/cache 生命周期设计：

```text
1. compressed pool 仍是固定大小 GPU pool；
2. radix node demote 后仍指向 compressed entry；
3. compressed hit 后仍分配 normal working pages；
4. request 结束后释放临时 restored pages；
5. compressed entry 默认保留，供后续继续命中。
```

区别是：

```text
v3: _restore_entry_to_indices() 使用 PyTorch unpack/dequant/assignment。
v4: _restore_entry_to_indices() 使用 ZipCache CUDA restore kernel。
```

### 4.3 新增 CUDA kernel

新增文件：

```text
python/minisgl/kernel/zipcache.py
python/minisgl/kernel/csrc/jit/zipcache.cu
```

Python wrapper：

```text
zipcache_quantize_part(
    src_cache,
    local_ids,
    q_packed,
    min_val,
    step,
    bit,
    storage_bit,
)

zipcache_dequantize_part(
    out_cache,
    dst_indices,
    local_ids,
    q_packed,
    min_val,
    step,
    storage_bit,
)
```

CUDA kernel 做：

```text
compress:
  for each selected token/head:
      min = amin(src[token, head, :])
      step = max((amax - min) / ((1 << bit) - 1), 1e-6)
  for each packed byte:
      q = round((x - min) / step)
      write q into 2bit / 4bit packed slot

restore:
for each quantized element:
    local_token_id = local_ids[token_group]
    dst_token_id = dst_indices[local_token_id]
    q = unpack(q_packed, storage_bit)
    x = q * step[token_group, head] + min[token_group, head]
    out_cache[dst_token_id, head, dim] = x
```

其中：

```text
storage_bit = 4: 每个 uint8 存 2 个量化值；
storage_bit = 2: 每个 uint8 存 4 个量化值；
min/step: 当前实现为 fp16，shape = [num_selected_tokens, num_heads, 1]；
out_cache: normal KV pool 的某一层 K 或 V，flatten 后 shape = [num_tokens, num_heads * head_dim]。
```

### 4.4 attention backend

不新增或修改 FlashAttention 计算逻辑。

现有路径保持：

```text
AttentionLayer.forward()
  -> backend.forward()
     -> kvcache.store_kv()
     -> ZipCache manager before_attention()
     -> FlashAttention / FlashInfer
     -> ZipCache manager after_attention()
```

v4 只改变 prefix cache 命中时的 materialize/restore 实现。attention backend 仍看到普通
fp16/bf16 KV pool。

## 5. v4 与“直接让 attention 读 compressed KV”的区别

更激进的方案是让 attention kernel 直接读 compressed pool：

```text
page_table entry -> normal page or compressed entry
attention tile load -> 判断类型 -> unpack/dequant -> QK/softmax/V
```

这个方案理论上能减少 normal working page 写回，但代价很高：

```text
1. 必须修改 FlashAttention / FlashInfer 的 KV load path；
2. page table 需要支持 normal/compressed 混合地址；
3. compressed entry 中重要/不重要 token 的 ids 映射会让 attention tile load 变复杂；
4. 对每个 token/head/dim 做随机 metadata 读取，可能破坏 attention kernel 的访存合并；
5. 正确性和性能调试难度大。
```

v4 暂不做该方案。可以把它作为 v5。

## 6. v4 预期收益

相比 v3，v4 预期减少：

```text
1. demote 路径的 PyTorch amin / amax / round / pack 调度开销；
2. restore 路径的 PyTorch unpack kernel launch；
3. restore 路径的 PyTorch dequant kernel launch；
4. restore 路径的 torch.empty 临时输出 tensor；
5. flat_k[indices] = tensor 这种 advanced indexing scatter；
6. Python 层循环中的同步/调度开销。
```

v4 压缩端当前使用两个 kernel：一个计算 min/step，一个 packed 写 q。恢复端每个
K/V part 使用一个 kernel，把 unpack、dequant 和 scatter 合并在一起。

对于 shared-prefix 命中较多、compressed restore 成功较多的 workload，应重点观察：

```text
num_compressed_hits
num_restore_success
num_kernel_restore_calls
kernel_restore_tokens
kernel_restore_elements
TTFT
E2E
TPOT
```

## 7. v4 仍然不能解决的问题

```text
1. attention kernel 仍读取 normal pool，因此命中 compressed prefix 时仍需要 working pages；
2. restore 和 attention 仍是两个 kernel 阶段，不能完全消除 restore 开销；
3. demote/compress 路径仍需要为 K/V、important/unimportant 分别 launch kernel；
4. 如果 workload 很少命中 compressed prefix，v4 对端到端性能帮助有限；
5. 如果 normal pool 太小，请求并发太高，restore working pages 仍可能不足。
```

## 8. 后续 v4.1 / v4.2 优化方向

### v4.1 compression kernel 继续优化

当前 v4 已经把 demote 路径中的：

```text
amin / amax
round quantize
2bit / 4bit pack
写入 compressed pool
```

放入 CUDA kernel。后续可以进一步把 min/step 和 pack 合成单 kernel，或者减少
K/V、important/unimportant 之间的 launch 次数。

### v4.2 asynchronous restore

在 scheduler 已经确定 compressed hit 后，尽早在独立 stream 发起 restore kernel。
attention backend 准备 metadata 和 token_pool copy 可以与 restore 部分重叠。

### v4.3 restore cache

短时间内多请求命中同一个 compressed entry 时，可以复用刚 restore 到 normal pool 的
working pages，避免重复 restore。

### v5 direct compressed attention

真正修改 attention kernel 的 KV load path，让 attention tile 直接从 compressed pool
读取并在线 dequant。该方向工程风险高，建议等 v4 restore kernel 稳定后再做。

## 9. 启动方式

```bash
PYTHONPATH=python python -m minisgl \
  --model-path /path/to/model \
  --host 0.0.0.0 \
  --port 30001 \
  --cache-type radix \
  --max-running-requests 16 \
  --max-prefill-length 4096 \
  --enable-zipcache-v4 \
  --zipcache-v4-normal-pool-pages 32768 \
  --zipcache-v4-compressed-pool-mb 40960 \
  --zipcache-v4-use-kernel-compress \
  --zipcache-v4-use-kernel-restore \
  --zipcache-unimportant-ratio 0.4 \
  --zipcache-k-important-bit 4 \
  --zipcache-k-unimportant-bit 2 \
  --zipcache-v-important-bit 4 \
  --zipcache-v-unimportant-bit 2 \
  --zipcache-stats-interval 10 \
  2>&1 | tee zipcache_v4_server.log
```

如果 v4 kernel compress 或 restore 在某个环境下编译失败，可临时关闭：

```bash
--no-zipcache-v4-use-kernel-compress
--no-zipcache-v4-use-kernel-restore
```

关闭后 v4 会退回 v3 的 PyTorch 压缩/恢复路径，便于定位问题。

## 10. 当前实现修改文件

```text
python/minisgl/engine/config.py
  增加 ZipCache v4 配置字段。

python/minisgl/server/args.py
  增加 --enable-zipcache-v4 和 v4 pool / restore 参数。

python/minisgl/engine/engine.py
  根据 feature flag 创建 ZipCacheV4Manager；支持 v4 normal pool page override；
  ZipCache 开启时继续关闭 CUDA Graph。

python/minisgl/scheduler/cache.py
  请求结束后按 v4 demote flag 接入 radix node demotion。

python/minisgl/zipcache/manager.py
  新增 ZipCacheV4Manager；复用 v3 的 compressed radix 生命周期；
  用 CUDA compress/restore kernel 替换 v3 的 PyTorch 压缩/恢复；
  失败时 fallback 到 v3 的 PyTorch 路径。

python/minisgl/kernel/zipcache.py
  新增 compression / restore Python 侧 JIT wrapper。

python/minisgl/kernel/csrc/jit/zipcache.cu
  新增 min/max + quantize + pack CUDA kernel；
  新增 fused unpack + dequant + scatter CUDA kernel。

experiment/parse_zipcache_log.py
  支持解析 [ZipCacheV4] stats。
```

## 11. 实验对比

建议对比三组：

```text
main:
  原始 miniSGLang。

ZipCache v3:
  Python/PyTorch restore。

ZipCache v4:
  CUDA kernel restore。
```

重点指标：

```text
1. correctness:
   GSM8K accuracy / correctness dataset accuracy。

2. cache behavior:
   num_demotions
   num_compressed_entries
   num_compressed_hits
   num_restore_success
   active_storage_compression_ratio

3. v4 kernel behavior:
   num_kernel_restore_calls
   num_kernel_restore_fallback
   kernel_restore_tokens
   kernel_restore_elements

4. serving performance:
   TTFT
   E2E
   TPOT
   chunks/s
   RPS

5. memory/capacity:
   compressed_pool_utilization
   estimated_capacity_gain_vs_normal_pool
   max supported shared-prefix workload size
```

预期：

```text
1. v4 的正确率应接近 v3；
2. v4 的显存/压缩率应接近 v3；
3. compressed hit 较多时，v4 的 TTFT/E2E 应优于 v3；
4. 如果没有 compressed hit，v4 和 v3 差异不明显。
```

## 12. 当前公开数据集测试入口

当前一键测试脚本默认使用公开数据集派生 workload：

```text
experiment/workloads/
```

这些 workload 由以下数据构造：

```text
GSM8K
CMMLU
LongBench
RULER SQuAD helper data
generated-shared-prefix synthetic load
```

正常测试时不需要重新生成数据，直接使用仓库中已经提交的
`experiment/workloads/*.jsonl`。如果本地数据被删除或需要重新生成，可以执行：

```bash
python experiment/prepare_public_workloads.py \
  --root experiment \
  --output-dir experiment/workloads
```

### 12.1 v4 完整一键测试

先启动 v4 服务：

```bash
PYTHONPATH=python python -m minisgl \
  --model-path /root/autodl-tmp/modelscope-cache/models/Qwen/Qwen3-0___6B \
  --host 0.0.0.0 \
  --port 30001 \
  --cache-type radix \
  --max-running-requests 16 \
  --max-prefill-length 4096 \
  --enable-zipcache-v4 \
  --zipcache-v4-normal-pool-pages 32768 \
  --zipcache-v4-compressed-pool-mb 40960 \
  --zipcache-v4-use-kernel-compress \
  --zipcache-v4-use-kernel-restore \
  --zipcache-unimportant-ratio 0.4 \
  --zipcache-k-important-bit 4 \
  --zipcache-k-unimportant-bit 2 \
  --zipcache-v-important-bit 4 \
  --zipcache-v-unimportant-bit 2 \
  --zipcache-stats-interval 10 \
  2>&1 | tee zipcache_v4_server.log
```

另开一个终端执行：

```bash
python experiment/run_all_experiments.py \
  --mode zipcache_v4 \
  --base-url http://127.0.0.1:30001 \
  --server-log zipcache_v4_server.log \
  --log-root experiment/logs \
  --gpu-sample-interval 0.5
```

测试结果会保存到：

```text
experiment/logs/<时间>_zipcache_v4/
```

其中 `report.md` 汇总每个 workload 的吞吐、延迟、显存和 ZipCache 统计。

### 12.2 只跑长上下文与 shared-prefix 压测

该测试主要用于观察 v4 是否能在 shared-prefix / long-context 场景下产生
compressed hit，并验证 CUDA restore kernel 是否降低 v3 中 PyTorch restore 带来的开销。

```bash
python experiment/run_all_experiments.py \
  --mode zipcache_v4_pressure \
  --base-url http://127.0.0.1:30001 \
  --server-log zipcache_v4_server.log \
  --only longbench_long_context_pressure,public_shared_prefix,public_shared_prefix_serial,synthetic_shared_prefix \
  --gpu-sample-interval 0.5
```

重点观察：

```text
num_compressed_hits
num_restore_attempts
num_restore_success
num_kernel_restore_calls
num_kernel_restore_fallback
kernel_restore_tokens
kernel_restore_elements
compressed_pool_utilization
estimated_capacity_gain_vs_normal_pool
TTFT / E2E / TPOT
```

### 12.3 只跑正确性测试

该测试用于确认 v4 的压缩 / 解压没有明显破坏生成结果。

```bash
python experiment/run_all_experiments.py \
  --mode zipcache_v4_correctness \
  --base-url http://127.0.0.1:30001 \
  --server-log zipcache_v4_server.log \
  --only gsm8k_public_correctness,cmmlu_public_correctness,longbench_public_qa,ruler_squad_qa
```

正确性结果在对应日志目录下的：

```text
gsm8k_public_correctness_eval.json
cmmlu_public_correctness_eval.json
longbench_public_qa_eval.json
ruler_squad_qa_eval.json
```

正确性 workload 会自动关闭 `ignore_eos` 并提高生成上限：

```text
GSM8K: max_tokens=1024
CMMLU: max_tokens=128
LongBench QA: max_tokens=512
RULER SQuAD: max_tokens=256
```

如果 `report.md` 中正确性实验的 `maxed` 列不为 0，说明仍有请求达到
`max_tokens` 上限，需要继续提高对应 workload 的生成长度。

### 12.4 main / v3 / v4 对比方式

建议在同一台机器、同一个模型、同一套 workload 下分别测试三组：

```text
main:
  不带 ZipCache 参数，作为原始 miniSGLang 基线。

v3:
  --enable-zipcache-v3，压缩 / 解压主要走 PyTorch 路径。

v4:
  --enable-zipcache-v4，压缩 / 解压优先走专用 CUDA kernel。
```

main 一键测试：

```bash
python experiment/run_all_experiments.py \
  --mode main \
  --base-url http://127.0.0.1:30000 \
  --log-root experiment/logs \
  --gpu-sample-interval 0.5
```

v3 一键测试：

```bash
python experiment/run_all_experiments.py \
  --mode zipcache_v3 \
  --base-url http://127.0.0.1:30001 \
  --server-log zipcache_v3_server.log \
  --log-root experiment/logs \
  --gpu-sample-interval 0.5
```

v4 一键测试：

```bash
python experiment/run_all_experiments.py \
  --mode zipcache_v4 \
  --base-url http://127.0.0.1:30001 \
  --server-log zipcache_v4_server.log \
  --log-root experiment/logs \
  --gpu-sample-interval 0.5
```

对比时不要只看 `nvidia-smi` 的进程总显存。v3/v4 会把显存从 normal KV pool
重新分配到 compressed pool，因此更重要的是比较：

```text
1. 相同显存预算下能缓存多少历史 KV：
   estimated_effective_kv_capacity_bytes
   estimated_capacity_gain_vs_normal_pool
   compressed_pool_utilization

2. compressed cache 是否真的被命中和恢复：
   num_compressed_hits
   num_restore_attempts
   num_restore_success
   num_restore_fallback

3. v4 是否比 v3 降低 restore 开销：
   num_kernel_restore_calls
   num_kernel_restore_fallback
   kernel_restore_tokens
   kernel_restore_elements
   TTFT
   E2E
   TPOT

4. 正确性是否接近 main：
   GSM8K accuracy
   CMMLU accuracy
   LongBench / RULER text_contains accuracy
```

### 12.5 解析 ZipCacheV4 日志

服务运行后可以单独解析 v4 stats：

```bash
python experiment/parse_zipcache_log.py \
  --log zipcache_v4_server.log
```

如果输出中 `num_stats` 为 0，说明服务日志里没有 `[ZipCacheV4] stats`。此时优先检查：

```text
1. 服务是否真的带了 --enable-zipcache-v4；
2. --zipcache-stats-interval 是否过大；
3. 测试是否运行时间太短，尚未触发周期性 stats；
4. server log 路径是否传错。
```

如果 `num_compressed_hits` 长期为 0，说明测试负载没有命中 compressed radix entry。
应优先跑 `public_shared_prefix_serial` 或 `synthetic_shared_prefix`，并确认请求结束后
v4 demote 路径有 `[ZipCacheV4] demoted` 日志。
