# miniSGLang ZipCache v3 方案设计

本文档给出 ZipCache v3 的工程设计。v3 的目标不是简单让 `nvidia-smi` 立即下降，而是：

```text
在相同 GPU KV cache 显存预算下，用 ZipCache 保存更多历史 KV cache，
从而支持更长上下文、更多 batch、更多可复用 prefix。
```

v2 已经验证：

- KV 可以在 GPU 上压缩保存；
- 当前 4bit packed 方案实际存储压缩比约 `3.7x`；
- normal KV page 被 demote 后能释放回 normal pool；
- 但 normal pool 和 compressed pool 都是预分配，进程总显存不一定下降；
- 当前 restore hit 仍需要继续完善，否则只有 demote，没有复用收益。

v3 要解决两个核心问题：

1. **显存预算结构问题**

   v2 仍保留较大的 normal KV pool，再额外分配 compressed pool，容易导致总显存看起来没有优势。v3 应该把大部分 KV 预算分给 compressed pool。

2. **缓存索引语义问题**

   radix tree 命中后，不能只返回 normal KV pool index。v3 需要让 radix node 直接指向 compressed pool entry，命中后由 ZipCache manager restore 到 normal pool，再供 decode 使用。

## 1. v3 总体结论

v3 推荐采用：

```text
Compressed-Primary KV Cache
```

也就是：

```text
normal pool:
  小容量 fp16/bf16 工作区
  保存当前活跃请求尚未压缩的 KV
  保存 compressed hit 后临时 restore 出来的 KV
  保存当前 decode 必须直接读取的热 KV

compressed pool:
  大容量 GPU 压缩 KV archive
  保存所有已经被压缩的历史 KV
  作为 radix prefix cache 的主要长期存储后端

radix tree:
  仍然按 token prefix 做命中
  命中节点可以直接指向 compressed pool entry
  不再假设命中结果一定是 normal pool indices

page_table:
  保持原 miniSGLang 语义
  只能存 normal pool indices

attention kernel:
  不修改
  仍然只读取 normal fp16/bf16 KV
```

最关键的边界：

```text
compressed pool 地址不能直接写入 page_table。
page_table 只能写 normal pool 的物理 token index。
```

所以 v3 的读取路径必须是：

```text
radix match compressed entry
-> 从 compressed pool 读 q/scale/ids
-> GPU 解压到 normal pool
-> page_table 指向 restored normal indices
-> 原 attention kernel 正常 decode
```

## 2. 为什么 v3 能体现显存优势

原 miniSGLang：

```text
KV cache 长期以 fp16/bf16 保存在 normal pool
单位显存只能保存 1x 的 KV 内容
```

ZipCache v3：

```text
历史 KV 长期以 4bit/2bit compressed format 保存在 compressed pool
normal pool 只保存当前计算需要的工作集
```

假设总 KV 预算为 `B`，原 miniSGLang 的有效 KV 容量约为：

```text
effective_capacity_main = B
```

v3 若采用：

```text
normal_pool = 0.20B
compressed_pool = 0.80B
```

且 compressed pool 的实际存储压缩比约为 `3.7x`，则：

```text
effective_capacity_v3
  = normal_pool
    + compressed_pool * compression_ratio
  = 0.20B + 0.80B * 3.7
  = 3.16B
```

也就是说：

```text
同等 KV 显存预算下，v3 可保存约 3x 历史 KV。
```

这才是 v3 相比原 miniSGLang 的主要优势。

因此实验表达应改成：

```text
不是证明 ZipCache v3 的 nvidia-smi 一定更低；
而是证明同等 nvidia-smi / 同等 KV 显存预算下，
ZipCache v3 能缓存更多 prefix、更长上下文或更多 batch。
```

## 3. normal pool 与 compressed pool 的职责

### 3.1 normal pool 存什么

normal pool 存：

```text
1. 当前请求还没有被压缩的新 KV；
2. 当前 decode 正在频繁访问的热 KV；
3. compressed pool 命中后 restore 出来的临时 KV；
4. attention kernel 即将读取的 fp16/bf16 KV。
```

normal pool 不再承担：

```text
长期保存所有历史 prefix cache
```

请求结束或 prefix 变冷后：

```text
normal KV -> compress -> compressed pool
normal pages -> free list
```

### 3.2 compressed pool 存什么

compressed pool 存：

```text
1. 已经 demote 的 prefix KV；
2. finished request 的可复用 KV；
3. 被 radix tree 索引的历史 shared prefix；
4. 未来可能 restore 的长期 KV archive。
```

compressed pool 是 v3 的主缓存。

### 3.3 normal pool 需要多大

normal pool 不需要保存所有历史 KV，但必须能保存当前 attention 需要读取的工作集。

如果不修改 attention kernel，当前 kernel 仍然要求一次 attention 能看到完整上下文 KV。因此 normal pool 至少需要覆盖：

```text
当前 batch 中正在参与 attention 的 KV token 总数
```

粗略公式：

```text
normal_pool_pages >=
  max_running_requests
  * max_active_context_len_per_request
  + restore_margin
  + decode_growth_margin
```

如果限制：

```text
max_running_requests = 1
max_context_len = 4096
```

则 normal pool 可以接近：

```text
4096 + margin
```

如果：

```text
max_running_requests = 8
max_context_len = 4096
```

则 normal pool 不能只有 4096，更不能只有 200。

### 3.4 “normal pool 只保留 200 个位置”如何理解

`200 pages` 可以作为短上下文极限实验：

```text
max_running_requests = 1
max_prefill_length <= 200
```

用于验证：

```text
compressed pool 长期保存
restore 到 normal pool
attention 正确读取
用完释放 normal pages
```

但它不能代表真实长上下文 serving。对于长上下文，normal pool 至少要能容纳当前活跃 batch 的工作集。

## 4. radix tree 如何索引 compressed pool

### 4.1 当前 v2 的限制

当前 radix tree 的命中句柄最终会返回：

```text
RadixCacheHandle.get_matched_indices()
```

它默认认为命中路径上的每个 node 都有：

```text
node.value = normal KV pool indices
```

这适合原 miniSGLang，但不适合 v3。v3 中大多数历史 prefix 已经在 compressed pool 中，没有可直接读取的 normal indices。

### 4.2 v3 的 node 引用模型

v3 radix node 应该从“只保存 normal indices”升级为“保存 cache reference”。

推荐结构：

```python
class CacheRef:
    kind: Literal["normal", "compressed"]
    length: int

    # kind == "normal"
    normal_indices: torch.Tensor | None

    # kind == "compressed"
    compressed_id: int | None
```

radix node：

```python
class RadixTreeNode:
    key: torch.Tensor
    cache_ref: CacheRef
```

也可以兼容当前实现：

```python
node.value_kind = "normal" | "compressed"
node.value = normal_indices only when normal
node.compressed_id = compressed entry id when compressed
```

### 4.3 compressed entry 的物理地址

compressed pool 不能只用一个整数地址表达。一个 compressed entry 至少需要：

```text
entry_id
num_tokens
token_ids
created_time
last_access_time
hit_count
per-layer K/V compressed handles
```

每一层：

```text
layer_id
K important handle
K unimportant handle
V important handle
V unimportant handle
shape
dtype
```

每个 handle：

```text
pool_kind: q4 | q2 | scale | ids
offset
length
logical_numel
q_shape
bit_width
```

因此 radix tree 的 compressed 命中只保存：

```text
compressed_id
```

真实物理位置由：

```text
ZipCacheV3Manager.entries[compressed_id]
```

解析。

### 4.4 page_table 保持不变

这一点非常重要：

```text
page_table 不能写 compressed_id
page_table 不能写 compressed pool offset
page_table 只能写 normal KV pool token index
```

因为 attention backend 当前只根据 page_table 去 normal `k_cache/v_cache` 读 fp16/bf16。

## 5. radix 命中后的读取流程

新请求进入：

```text
input_ids = prefix + suffix
```

radix tree match：

```text
matched_nodes = path(root -> matched_node)
```

v3 materialize：

```text
for node in matched_nodes:
    if node.kind == NORMAL:
        use node.normal_indices

    if node.kind == COMPRESSED:
        entry = compressed_entries[node.compressed_id]
        restore_indices = normal_pool.allocate(entry.num_tokens)
        dequantize entry into restore_indices
        use restore_indices for this request
```

最后：

```text
matched_indices = concat(all normal/restored indices)
page_table[req.table_idx, :matched_len] = matched_indices
req.cached_len = matched_len
```

然后 scheduler 只对 suffix 分配新 pages 并 prefill：

```text
prefill suffix
decode
```

请求结束：

```text
new normal KV -> compress -> compressed pool
temporary restored normal pages -> free
compressed entry remains
```

### 5.1 v3 与 v2 restore 的关键区别

v2：

```text
compressed hit
-> restore 到 normal pool
-> radix node 改回 normal
-> compressed entry 被释放
```

v3：

```text
compressed hit
-> restore 到 normal pool
-> page_table 使用 restored indices
-> radix node 仍然保持 compressed
-> compressed entry 继续保留
-> 请求结束释放 restored normal pages
```

这保证 compressed pool 是长期主缓存，normal pool 只是工作区。

### 5.2 mixed normal/compressed path

一个 radix path 可能是：

```text
root
 -> node A: COMPRESSED
 -> node B: NORMAL
 -> node C: COMPRESSED
```

v3 materialize 需要拼接：

```text
restore(A) + B.normal_indices + restore(C)
```

并写入 page_table。

注意：

```text
如果其中任一 compressed node restore 失败，
必须截断到最后一个安全 materialized prefix，
剩余部分 fallback prefill/recompute。
```

## 6. compressed pool 设计

v3 建议 compressed pool 成为主存储后端，因此需要比 v2 更细的 pool 设计。

### 6.1 单一 4bit pool 方案

优点：

```text
实现简单
restore 逻辑简单
allocator 简单
碎片管理容易
```

缺点：

```text
2bit unimportant token 也占 4bit slot
显存容量浪费
无法充分体现 ZipCache 论文中 mixed precision 的优势
```

这个方案适合功能先跑通，但 v3 要突出显存优势，建议升级到双 pool。

### 6.2 4bit + 2bit 双 q pool 方案

ZipCache 论文和当前 demo 默认设计是：

```text
important / salient token: 4bit
unimportant token: 2bit
unimportant_ratio: 默认约 0.4
important_ratio: 约 0.6
```

因此 v3 推荐：

```text
q4_pool:
  保存 important K/V 的 4bit packed q

q2_pool:
  保存 unimportant K/V 的 2bit packed q

scale_pool:
  保存 min/step

ids_pool:
  保存 important/unimportant token ids

meta_pool:
  保存 entry/layer/shape/offset 等 metadata
```

### 6.3 双 pool 容量比例

设：

```text
u = unimportant_ratio = 0.4
i = important_ratio = 0.6
```

量化数据本体 bit 数：

```text
q4_bits = i * 4 = 0.6 * 4 = 2.4
q2_bits = u * 2 = 0.4 * 2 = 0.8
total_q_bits = 3.2
```

因此 q payload 中：

```text
q4_pool 占比 = 2.4 / 3.2 = 75%
q2_pool 占比 = 0.8 / 3.2 = 25%
```

但 compressed pool 总容量还要包含 scale、ids、metadata。建议 v3 初始总 compressed budget 划分：

```text
q4_pool:    45%
q2_pool:    15%
scale_pool: 25%
ids_pool:   10%
meta_pool:   5%
```

解释：

- q4/q2 合计 60%，内部按 75%/25% 对应 4bit/2bit payload；
- scale_pool 保留 25%，因为当前 min/step 是按 token/head 保存，开销不小；
- ids_pool 保留 10%，保存 important/unimportant token ids；
- meta_pool 保留 5%，保存 entry metadata 和 allocator 对齐余量。

如果实验发现 scale/ids 开销小，可以调整为：

```text
q4_pool:    55%
q2_pool:    18%
scale_pool: 17%
ids_pool:    7%
meta_pool:   3%
```

### 6.4 预期压缩率

如果只看 q payload：

```text
fp16 = 16 bits
mixed q = 3.2 bits
raw_q_compression = 16 / 3.2 = 5x
```

考虑 min/step/ids/metadata 后，实际压缩率可能是：

```text
3.5x ~ 4.5x
```

这比 v2 统一 4bit 存储更能体现 ZipCache 的显存优势。

## 7. normal/compressed 显存预算

v3 推荐支持两种模式。

### 7.1 同等显存预算容量优势模式

用于证明：

```text
同样 nvidia-smi 显存占用下，ZipCache v3 能保存更多 KV。
```

示例：

```text
main:
  KV budget = 50 GiB normal fp16 pool
  effective KV capacity = 50 GiB

ZipCache v3:
  normal pool = 10 GiB
  compressed pool = 40 GiB
  effective KV capacity ~= 10 + 40 * 3.7 = 158 GiB equivalent
```

这种模式下，`nvidia-smi` 不一定下降，但优势体现在：

```text
effective_kv_capacity_gain ~= 158 / 50 = 3.16x
```

### 7.2 低显存模式

用于证明：

```text
更小 GPU 显存下支持接近 main 的上下文和 batch。
```

示例：

```text
main:
  normal pool = 50 GiB

ZipCache v3:
  normal pool = 8 GiB
  compressed pool = 20 GiB
  total KV budget = 28 GiB
  effective capacity ~= 8 + 20 * 3.7 = 82 GiB equivalent
```

这种模式可能真正看到 `nvidia-smi` 下降。

## 8. 性能控制策略

压缩/解压一定会带来开销。v3 需要避免性能下降过多。

### 8.1 热 KV 不压缩

活跃 decode 请求的最近 KV 不应频繁压缩/解压。

建议：

```text
recent_window_tokens 保留 normal
finished prefix 或冷 prefix 才 demote
```

参数：

```bash
--zipcache-v3-protect-recent-tokens 128
```

### 8.2 restore 结果在请求生命周期内保留

一次 compressed hit restore 后，在该请求 decode 期间不要每步重新 restore。

流程：

```text
restore once before prefill/decode
decode 多步复用 restored normal pages
request finished 后释放 normal pages
```

### 8.3 小 prefix 不 restore

短 prefix 重新 prefill 可能比 restore 更快。

参数：

```bash
--zipcache-v3-min-restore-tokens 256
```

策略：

```text
if compressed_entry.num_tokens < min_restore_tokens:
    fallback recompute
```

### 8.4 normal pool 低水位

如果 normal pool 空闲 page 低于阈值：

```text
不要接收更多 restore
优先等待请求结束
或 fallback recompute
```

参数：

```bash
--zipcache-v3-min-normal-free-pages 512
```

### 8.5 compressed pool eviction

compressed pool 满时，应淘汰：

```text
低 hit_count
短 prefix
最近很久未访问
restore cost 高于 recompute cost
```

优先保留：

```text
长 prefix
重复命中 prefix
shared system prompt
RAG 长文档 prefix
```

## 9. v3 需要修改的模块

### 9.1 参数与配置

文件：

```text
python/minisgl/server/args.py
python/minisgl/engine/config.py
```

新增：

```bash
--enable-zipcache-v3
--zipcache-v3-normal-pool-pages
--zipcache-v3-compressed-pool-mb
--zipcache-v3-q4-pool-ratio
--zipcache-v3-q2-pool-ratio
--zipcache-v3-scale-pool-ratio
--zipcache-v3-ids-pool-ratio
--zipcache-v3-keep-compressed-after-restore
--zipcache-v3-min-restore-tokens
--zipcache-v3-protect-recent-tokens
```

### 9.2 Engine

文件：

```text
python/minisgl/engine/engine.py
```

v3 启用时：

```text
normal KV pool pages = zipcache_v3_normal_pool_pages
compressed pool size = zipcache_v3_compressed_pool_mb
```

并打印：

```text
[ZipCacheV3] normal_pool_pages=...
[ZipCacheV3] normal_pool_bytes=...
[ZipCacheV3] compressed_pool_bytes=...
[ZipCacheV3] estimated_effective_kv_capacity=...
[ZipCacheV3] estimated_capacity_gain=...
```

### 9.3 Radix cache

文件：

```text
python/minisgl/kvcache/radix_cache.py
```

需要支持：

```text
node.kind = normal/compressed
node.compressed_id
path_nodes(handle)
match compressed node 不截断
get_matched_indices 不再直接用于 mixed path
```

新增接口：

```python
def get_match_refs(handle) -> List[CacheRef]:
    ...
```

### 9.4 CacheManager

文件：

```text
python/minisgl/scheduler/cache.py
```

`match_req()` 中：

```text
radix match
-> ZipCacheV3Manager.materialize_match_for_request()
-> 返回当前请求可写入 page_table 的 normal indices
```

`cache_req()` 中：

```text
finished or cold prefix
-> demote normal pages to compressed pool
-> radix node mark compressed
-> free normal pages
```

同时要避免：

```text
temporary restored pages 被长期插入 radix tree normal node
```

### 9.5 ZipCache manager

文件：

```text
python/minisgl/zipcache/manager.py
```

新增：

```python
class ZipCacheV3Manager:
    def demote_node(...)
    def materialize_match_for_request(...)
    def restore_entry_to_normal(...)
    def release_request_restores(...)
    def maybe_evict_compressed(...)
    def stats(...)
```

v3 可以复用 v2 的：

```text
GPU quant/dequant
4bit packing
stats 框架
compressed entry metadata
```

但需要修改：

```text
restore 后不 free compressed entry
restore 后不把 radix node 永久 mark_restored
请求结束时释放 temporary restored normal pages
```

## 10. 关键伪代码

### 10.1 compressed 命中 materialize

```python
def materialize_match_for_request(req, handle, cache_manager):
    refs = radix_cache.get_match_refs(handle)
    out_indices = []

    for ref in refs:
        if ref.kind == "normal":
            out_indices.append(ref.normal_indices)
            continue

        entry = compressed_entries[ref.compressed_id]

        if entry.num_tokens < min_restore_tokens:
            break  # fallback recompute remaining suffix

        indices = cache_manager.allocate_token_indices(entry.num_tokens)
        restore_entry_to_normal(entry, indices)

        request_restore_table[req.uid].append(indices)
        out_indices.append(indices)

    matched_indices = torch.cat(out_indices)
    page_table[req.table_idx, : len(matched_indices)] = matched_indices
    return matched_indices
```

### 10.2 请求结束释放

```python
def free_request(req):
    for indices in request_restore_table.pop(req.uid, []):
        cache_manager.free(indices)

    if req.finished_prefix_should_cache:
        demote_node(req.radix_node)
```

### 10.3 demote

```python
def demote_node(node):
    if node.kind == "compressed":
        return

    normal_indices = node.value
    entry = quantize_to_compressed_pool(normal_indices)
    node.kind = "compressed"
    node.compressed_id = entry.entry_id
    node.value = empty_or_debug_only
    cache_manager.free(normal_indices)
```

## 11. 统计指标

v3 必须新增统计：

```text
normal_pool_pages
normal_pool_used_pages
normal_pool_free_pages
compressed_pool_capacity_bytes
compressed_pool_used_bytes
compressed_pool_utilization
compressed_q4_used_bytes
compressed_q2_used_bytes
compressed_scale_used_bytes
compressed_ids_used_bytes
effective_original_kv_bytes
effective_compressed_kv_bytes
effective_capacity_gain
num_demotions
num_compressed_hits
num_restore_attempts
num_restore_success
num_restore_fallback
num_temporary_restore_pages
num_restore_pages_released
num_redemotions
num_compressed_evictions
```

核心实验指标：

```text
effective_capacity_gain > 1
num_compressed_hits > 0
num_restore_success > 0
active_storage_compression_ratio >= 3
GSM8K accuracy 接近 main
repeated-prefix TTFT 不明显劣化，最好下降
```

## 12. 启动示例

同等显存预算容量优势模式：

```bash
PYTHONPATH=python python -m minisgl \
  --model-path /root/autodl-tmp/modelscope-cache/models/Qwen/Qwen3-0___6B \
  --host 0.0.0.0 \
  --port 30001 \
  --cache-type radix \
  --max-running-requests 4 \
  --max-prefill-length 4096 \
  --enable-zipcache-v3 \
  --zipcache-v3-normal-pool-pages 32768 \
  --zipcache-v3-compressed-pool-mb 40960 \
  --zipcache-unimportant-ratio 0.4 \
  --zipcache-k-important-bit 4 \
  --zipcache-k-unimportant-bit 2 \
  --zipcache-v-important-bit 4 \
  --zipcache-v-unimportant-bit 2 \
  --zipcache-v3-keep-compressed-after-restore \
  --zipcache-v3-min-restore-tokens 256 \
  --zipcache-stats-interval 30 \
  2>&1 | tee zipcache_v3_server.log
```

低显存模式：

```bash
PYTHONPATH=python python -m minisgl \
  --model-path /root/autodl-tmp/modelscope-cache/models/Qwen/Qwen3-0___6B \
  --host 0.0.0.0 \
  --port 30001 \
  --cache-type radix \
  --max-running-requests 2 \
  --max-prefill-length 4096 \
  --enable-zipcache-v3 \
  --zipcache-v3-normal-pool-pages 16384 \
  --zipcache-v3-compressed-pool-mb 20480 \
  --zipcache-unimportant-ratio 0.4 \
  --zipcache-k-important-bit 4 \
  --zipcache-k-unimportant-bit 2 \
  --zipcache-v-important-bit 4 \
  --zipcache-v-unimportant-bit 2 \
  --zipcache-v3-keep-compressed-after-restore \
  --zipcache-stats-interval 30 \
  2>&1 | tee zipcache_v3_server.log
```

## 13. 实验设计

### 13.1 对比一：同等显存预算下缓存更多 KV

main：

```text
normal pool = 50 GiB
compressed pool = 0
effective capacity = 50 GiB
```

v3：

```text
normal pool = 10 GiB
compressed pool = 40 GiB
effective capacity ~= 10 + 40 * 3.7 = 158 GiB
```

观察：

```text
effective_capacity_gain
compressed_pool_utilization
num_demotions
num_compressed_entries
```

### 13.2 对比二：重复长前缀

运行：

```text
zipcache_restore_pressure
realistic_long_context
```

观察：

```text
num_compressed_hits
num_restore_attempts
num_restore_success
TTFT
E2E
GSM8K accuracy
```

### 13.3 对比三：更长上下文 / 更多 batch

main 在固定显存预算下可能出现：

```text
KV allocation 不足
prefix cache evict 频繁
TTFT 上升
```

v3 预期：

```text
相同预算下保存更多 prefix
重复前缀 restore 命中
支持更长历史缓存
```

## 14. 风险与限制

### 14.1 compressed hit 必须先解决

如果仍然出现：

```text
num_demotions > 0
num_compressed_hits = 0
```

说明 radix compressed node 没有被正确命中。v3 必须先解决这个问题，否则 compressed pool 再大也只是 archive，不会产生复用收益。

### 14.2 normal pool 不能过小

不改 attention kernel 时，normal pool 必须能容纳当前 batch attention 的 KV 工作集。否则只能 fallback recompute 或限制 batch/context。

### 14.3 restore 可能带来性能下降

restore 包括：

```text
read q4/q2
read scale/ids
unpack
dequantize
write normal pool
```

短 prefix 不值得 restore。v3 必须通过 `min_restore_tokens` 和 cost policy 控制。

### 14.4 双 pool 增加 allocator 复杂度

4bit/2bit 分开后，需要更多 allocator 和 metadata。收益是压缩率更接近论文 mixed precision 设计，代价是实现复杂度更高。

## 15. 实现优先级

### Step 1：修复 v2/v3 compressed hit

目标：

```text
zipcache_restore_pressure:
  num_compressed_hits > 0
  num_restore_success > 0
```

### Step 2：v3 keep-compressed-after-restore

目标：

```text
restore 后 compressed entry 保留
normal restored pages 只属于当前请求
请求结束释放 restored pages
```

### Step 3：normal pool 缩小，compressed pool 扩大

目标：

```text
同等 GPU KV 预算下 effective_capacity_gain > 1
```

### Step 4：双 q pool

目标：

```text
important token 用 q4_pool
unimportant token 用 q2_pool
实际压缩率优于统一 4bit
```

### Step 5：性能策略

加入：

```text
min_restore_tokens
recent token protect
compressed eviction
restore cost policy
```

## 16. v3 成功标准

v3 版本成功不应只看 `nvidia-smi`。

必须同时满足：

```text
1. 同等显存预算下 effective_capacity_gain 明显大于 1；
2. active_storage_compression_ratio >= 3；
3. num_compressed_hits > 0；
4. num_restore_success > 0；
5. restore_fallback 比例可控；
6. 支持比 main 更多历史 prefix 或更长上下文；
7. GSM8K correctness accuracy 接近 main；
8. TTFT / TPOT / E2E 不出现不可接受下降。
```

最终论文/实验表述可以写成：

```text
ZipCache v3 does not simply reduce the CUDA process memory immediately.
Instead, under the same preallocated GPU KV-cache budget, it converts most
historical KV cache from fp16/bf16 normal pages into compressed GPU archive
entries. This increases the effective KV-cache capacity and enables longer
contexts or more reusable prefixes without modifying the attention kernel.
```

## 17. 当前 v3 代码实现状态

当前 `ZipCache` 分支已经在 v2 基础上加入 v3 原型，重点实现如下：

```text
1. 新增 --enable-zipcache-v3；
2. 新增 --zipcache-v3-normal-pool-pages，用于把 normal KV pool 缩小成工作区；
3. 新增 --zipcache-v3-compressed-pool-mb / --zipcache-v3-compressed-pool-ratio；
4. compressed pool 使用固定大小 GPU tensor；
5. q4/q2 分开存储：
   - salient / important token 默认进入 q4 buffer；
   - unimportant token 在 bit <= 2 时进入 q2 buffer；
6. radix node 被 demote 后继续保持 compressed 状态；
7. compressed 命中时临时 restore 到 normal pool；
8. restore 后 compressed entry 默认保留，后续请求仍可再次命中；
9. 请求结束时释放临时 restored normal pages；
10. attention kernel 和 attention 数学逻辑不修改；
11. 默认仍关闭 CUDA Graph；需要对比 decode CUDA Graph 收益时，可显式加
    --enable-zipcache-cuda-graph。
```

推荐启动方式：

```bash
PYTHONPATH=python python -m minisgl \
  --model-path /root/autodl-tmp/modelscope-cache/models/Qwen/Qwen3-0___6B \
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
  --zipcache-stats-interval 30 \
  2>&1 | tee zipcache_v3_server.log
```

默认启动方式会关闭 CUDA Graph，这样可以避免 graph capture 额外占用显存，
也便于和早期 v3 实验结果直接对齐。如果要测试 v3 在开启 decode CUDA Graph 后
的吞吐变化，可以在上面的命令中额外加入：

```bash
  --enable-zipcache-cuda-graph \
  --cuda-graph-max-bs 16 \
```

含义如下：

```text
1. --enable-zipcache-cuda-graph：
   允许 ZipCache v3 使用 main 中已有的 CUDA Graph decode replay 路径。
2. --cuda-graph-max-bs 16：
   只 capture batch size 不超过 16 的 decode graph。这个值越大，decode launch
   开销越低，但 graph 静态 buffer 占用也越大。
3. ZipCache 的 demote、compressed hit 检查、temporary restore 仍然发生在
   scheduler/cache 阶段，不会被 capture 到 CUDA Graph 中。
```

建议实验时至少跑三组：

```text
main:
  不开启 ZipCache，记录原始性能。

zipcache_v3:
  开启 ZipCache v3，不加 --enable-zipcache-cuda-graph。

zipcache_v3_cuda_graph:
  开启 ZipCache v3，并加 --enable-zipcache-cuda-graph --cuda-graph-max-bs 16。
```

如果开启 CUDA Graph 后启动阶段 OOM，优先降低：

```text
1. --cuda-graph-max-bs
2. --zipcache-v3-compressed-pool-mb
3. --zipcache-v3-normal-pool-pages
```

如果显存不足，优先降低：

```text
1. --zipcache-v3-compressed-pool-mb
2. --zipcache-v3-normal-pool-pages
3. --max-running-requests
4. --max-prefill-length
```

如果只想验证 v3 启动和普通推理，不想在请求结束时压缩 radix cache 节点，可以加：

```bash
--no-zipcache-v3-demote-on-finish
```

v3 日志中需要重点观察：

```text
[ZipCacheV3] demoted
[ZipCacheV3] restored temporary
[ZipCacheV3] stats
```

关键统计项：

```text
num_demotions
num_compressed_entries
num_compressed_hits
num_restore_attempts
num_restore_success
num_restore_fallback
num_temporary_restore_pages
num_restore_pages_released
active_storage_compression_ratio
compressed_pool_q4_used_bytes
compressed_pool_q2_used_bytes
compressed_pool_utilization
normal_pool_capacity_bytes
estimated_effective_kv_capacity_bytes
estimated_capacity_gain_vs_normal_pool
```

和 main 对比时，不应只比较 `nvidia-smi` 的进程总显存。v3 的核心优势是：

```text
在相同或可控的预分配显存预算下，compressed pool 能保存更多历史 KV。
因此应重点比较 effective KV capacity、compressed hits、restore success、
可承载的 shared-prefix 数量、长上下文压力测试是否更不容易触发 normal pool 不足。
```

## 15. 当前公开数据集测试入口

当前实验测试默认使用公开数据集派生 workload：

```text
experiment/workloads/
```

这些 workload 由以下公开数据构造：

```text
GSM8K
CMMLU
LongBench
RULER SQuAD helper data
generated-shared-prefix synthetic load
```

重新生成命令：

```bash
python experiment/prepare_public_workloads.py \
  --root experiment \
  --output-dir experiment/workloads
```

### 15.1 main 基线启动与测试

严格做 main / v3 对比时，推荐在同一台云服务器、同一个模型、同一套
`experiment/workloads/` 下分别切换分支测试。main 基线应使用 `main` 分支，并且
启动服务时不要带任何 ZipCache 参数。

切换到 main 分支：

```bash
git fetch origin
git checkout main
git pull origin main
```

启动 main 服务：

```bash
PYTHONPATH=python python -m minisgl \
  --model-path /root/autodl-tmp/modelscope-cache/models/Qwen/Qwen3-0___6B \
  --host 0.0.0.0 \
  --port 30000 \
  --cache-type radix \
  --max-running-requests 16 \
  --max-prefill-length 4096 \
  --cuda-graph-max-bs 0 \
  2>&1 | tee main_server.log
```

另开一个终端执行 main 一键测试：

```bash
python experiment/run_all_experiments.py \
  --mode main \
  --base-url http://127.0.0.1:30000 \
  --log-root experiment/logs \
  --gpu-sample-interval 0.5
```

只跑 main 正确性测试：

```bash
python experiment/run_all_experiments.py \
  --mode main_correctness \
  --base-url http://127.0.0.1:30000 \
  --only gsm8k_public_correctness,cmmlu_public_correctness,longbench_public_qa,ruler_squad_qa
```

只跑 main 长上下文与 shared-prefix 压测：

```bash
python experiment/run_all_experiments.py \
  --mode main_pressure \
  --base-url http://127.0.0.1:30000 \
  --only longbench_long_context_pressure,public_shared_prefix,public_shared_prefix_serial,synthetic_shared_prefix \
  --gpu-sample-interval 0.5
```

如果只是想在 ZipCache 分支上快速验证 feature flag 关闭时行为，可以不带
`--enable-zipcache-v3` 启动；但正式论文式对比建议仍使用真正的 `main` 分支，
避免分支中其他实验代码影响基线。

### 15.2 v3 推荐测试命令

启动 v3 服务后执行：

```bash
python experiment/run_all_experiments.py \
  --mode zipcache_v3 \
  --base-url http://127.0.0.1:30001 \
  --server-log zipcache_v3_server.log \
  --log-root experiment/logs \
  --gpu-sample-interval 0.5
```

如果 v3 服务启动时加入了 `--enable-zipcache-cuda-graph`，建议把测试模式名也改成
`zipcache_v3_cuda_graph`，便于日志目录和 `report.md` 区分：

```bash
python experiment/run_all_experiments.py \
  --mode zipcache_v3_cuda_graph \
  --base-url http://127.0.0.1:30001 \
  --server-log zipcache_v3_server.log \
  --log-root experiment/logs \
  --gpu-sample-interval 0.5
```

公平对比时需要保持 CUDA Graph 口径一致：

```text
1. 对比 ZipCache 本身开销：
   main 使用 --cuda-graph-max-bs 0，v3 不加 --enable-zipcache-cuda-graph。
2. 对比开启 decode graph 后的服务性能：
   main 使用 --cuda-graph-max-bs 16，v3 使用
   --enable-zipcache-cuda-graph --cuda-graph-max-bs 16。
```

只跑长上下文与 shared-prefix 压测：

```bash
python experiment/run_all_experiments.py \
  --mode zipcache_v3_pressure \
  --base-url http://127.0.0.1:30001 \
  --server-log zipcache_v3_server.log \
  --only longbench_long_context_pressure,public_shared_prefix,public_shared_prefix_serial,synthetic_shared_prefix \
  --gpu-sample-interval 0.5
```

如果 v3 速度较慢，优先使用轻量测试。该 preset 只跑 3 个实验：

```text
1. gsm8k_public_correctness：正确性；
2. longbench_long_context_pressure：长上下文显存占用；
3. public_shared_prefix_serial：前缀复用 / compressed hit。
```

推荐命令：

```bash
python experiment/run_all_experiments.py \
  --mode zipcache_v3_quick \
  --base-url http://127.0.0.1:30001 \
  --server-log zipcache_v3_server.log \
  --preset quick \
  --log-root experiment/logs \
  --gpu-sample-interval 0.5
```

如果仍然太慢，可以进一步限制每个 workload 的输入条数：

```bash
python experiment/run_all_experiments.py \
  --mode zipcache_v3_quick_16 \
  --base-url http://127.0.0.1:30001 \
  --server-log zipcache_v3_server.log \
  --preset quick \
  --max-samples 16
```

只跑正确性：

```bash
python experiment/run_all_experiments.py \
  --mode zipcache_v3_correctness \
  --base-url http://127.0.0.1:30001 \
  --server-log zipcache_v3_server.log \
  --only gsm8k_public_correctness,cmmlu_public_correctness,longbench_public_qa,ruler_squad_qa
```

正确性 workload 会自动关闭 `ignore_eos` 并提高生成上限：

```text
GSM8K: max_tokens=2048
CMMLU: max_tokens=256
LongBench QA: max_tokens=768
RULER SQuAD: max_tokens=512
```

如果 `report.md` 中正确性实验的 `maxed` 列不为 0，说明仍有请求达到
`max_tokens` 上限，需要继续提高对应 workload 的生成长度。

### 15.3 v3 日志对性能的影响

main 版本默认只会打印 HTTP 请求日志和少量 scheduler idle 日志。v3 额外维护
compressed pool、demote、restore 和容量统计，因此如果把每次 demote / restore 都以
INFO 级别打印，会引入额外开销：

```text
Python 日志格式化
stdout 写入
tee 写入 server log 文件
多请求高频 restore 时的日志锁竞争
```

这类事件级日志会影响 RPS、TTFT、E2E、TPOT，尤其是
`public_shared_prefix_serial` / `synthetic_shared_prefix` 这种容易触发 repeated
compressed hit 的 workload。

当前代码已经把以下高频事件日志降为 DEBUG，默认 INFO 级别不会打印：

```text
[ZipCacheV3] demoted
[ZipCacheV3] restored temporary
[ZipCacheV3] restored permanent
[ZipCacheV3] restore skipped
```

周期性 stats 仍保留 INFO 级别，用于实验后解析：

```text
[ZipCacheV3] stats: {...}
```

如果做“带 ZipCache 统计信息”的实验，建议保留较低频率 stats，例如：

```bash
--zipcache-stats-interval 30
```

如果做“纯性能极限对比”，可以关闭周期性 stats：

```bash
--zipcache-stats-interval 0
```

关闭后 `experiment/parse_zipcache_log.py` 将无法从 server log 中解析 ZipCache stats，
因此报告里不会有 compressed pool / restore 统计。建议正式实验至少跑两轮：

```text
1. stats 打开：确认 compression ratio、compressed hits、restore success；
2. stats 关闭：对比更干净的 serving 性能。
```

### 15.4 v3 与 main 的实验表述

v3 的对比重点不是“进程总显存立刻下降”，而是：

```text
相同或更低 KV cache 预算下，compressed pool 能保存更多历史 KV；
public_shared_prefix / synthetic_shared_prefix 能产生更多 compressed hit；
longbench_long_context_pressure 下 normal pool 不足和 OOM 风险降低；
GSM8K / CMMLU / LongBench / RULER 正确性接近 main。
```

推荐报告中同时列出：

```text
main report.md
zipcache_v3 report.md
zipcache_v3_server.log 中的 [ZipCacheV3] stats
compressed_pool_utilization
estimated_effective_kv_capacity_bytes
estimated_capacity_gain_vs_normal_pool
num_compressed_hits
num_restore_success
```
