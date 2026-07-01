# miniSGLang ZipCache v2 方案设计

本文档是 ZipCache v2 的工程设计说明。

v2 目标：

1. KV 压缩和解压都在 GPU 中完成；
2. 压缩后的 KV cache 保存在 GPU 中；
3. 不修改 CUDA attention kernel；
4. 不大改当前 paged KV cache 框架；
5. attention 计算前将需要的压缩 KV 解压恢复到普通 fp16/bf16 KV 布局；
6. 后续 prefix/radix cache 命中时，能够找到压缩 KV 并恢复使用；
7. 相比 v1，避免 CPU offload 和 CPU-GPU 往返拷贝。

## 0. 当前 v2 代码实现状态

当前仓库中的 v2 已实现一个 GPU-only prefix demotion 原型，目标是先验证正确链路：

```text
radix prefix normal KV
  -> 请求结束后 demote
  -> GPU 上量化保存 compressed entry
  -> 原 normal fp16/bf16 page 释放回 CacheManager.free_slots
  -> 后续 radix token 命中
  -> GPU 上反量化恢复到新分配的 normal KV page
  -> 原 attention kernel 继续读取 normal KV page
```

当前实现对应的主要文件：

| 文件 | 修改内容 |
| --- | --- |
| `python/minisgl/engine/config.py` | 增加 `enable_zipcache_v2`、v2 demote 配置、compressed pool 容量配置 |
| `python/minisgl/server/args.py` | 增加 `--enable-zipcache-v2`、compressed pool 参数，禁止 v1/v2 同时开启 |
| `python/minisgl/engine/engine.py` | 根据 feature flag 创建 `ZipCacheV2Manager`，并关闭 CUDA Graph |
| `python/minisgl/scheduler/scheduler.py` | 把 `zipcache_manager` 传给 `CacheManager` |
| `python/minisgl/scheduler/cache.py` | 在 `match_req()` 后 materialize compressed hit；在 finished cache 后 demote radix node |
| `python/minisgl/kvcache/radix_cache.py` | radix node 增加 `fp16/compressed` 状态、restore/demote 标记接口 |
| `python/minisgl/zipcache/manager.py` | 新增 `ZipCacheV2Manager`，实现 GPU 量化、GPU 反量化、统计日志 |

当前实现的边界：

1. **不修改 CUDA kernel**。attention 仍然只读取原 `MHAKVCache` 中的 fp16/bf16 tensor。
2. **压缩和解压都在 GPU tensor 上完成**，不再像 v1 一样把 compressed KV 保存到 CPU。
3. **restore 不是写入独立 restore workspace**，而是分配新的 normal KV page，并把 radix node 重新标记为 fp16 normal node。这样实现更保守，便于保证 page table 和原 attention backend 正确运行。
4. **当前 compressed archive 已改为固定大小 GPU compressed pool**。启动时预分配 `q_buffer`、`scale_buffer`、`ids_buffer` 三个大 tensor，entry 只持有其中的 slice。
5. **当前 q 数据已统一使用 4bit packed 存储**。important token 逻辑上按 4bit 量化，unimportant token 可按 2bit 量化，但物理上也放入 4bit slot，换取简单稳定的 GPU restore 路径。
6. **当前 demote 时机是请求 finished 后的 radix node**，不压缩活跃 decode 请求。
7. compressed node 如果发生 partial match，当前保守回退到父节点，不切分 compressed entry。

启动示例：

```bash
PYTHONPATH=python python -m minisgl \
  --model-path /path/to/model \
  --host 0.0.0.0 \
  --port 30001 \
  --cache-type radix \
  --enable-zipcache-v2 \
  --zipcache-unimportant-ratio 0.4 \
  --zipcache-k-important-bit 4 \
  --zipcache-k-unimportant-bit 2 \
  --zipcache-v-important-bit 4 \
  --zipcache-v-unimportant-bit 2 \
  --zipcache-v2-compressed-pool-ratio 0.35 \
  --zipcache-stats-interval 10
```

预期日志：

```text
[ZipCacheV2] enabled: GPU prefix demotion ...
[ZipCacheV2] demoted: entry_id=... node=... tokens=... original=... estimated_4bit=... gpu_storage=...
[ZipCacheV2] restored: entry_id=... node=... tokens=...
[ZipCacheV2] stats: {...}
```

### 0.1 为什么不用 Python dict + 独立 GPU tensor 动态 archive

最早的 v2 原型使用：

```text
entries: dict[entry_id, CompressedEntry]
CompressedEntry 内部保存很多独立 GPU tensor:
  k_important_q
  k_important_min
  k_important_step
  ...
```

这个方式的优点是：

1. 实现快，适合先验证 demote/restore 正确性；
2. 每个 entry 大小天然可变，不需要 allocator；
3. 失败回滚逻辑简单，删除 dict entry 后 tensor 由 PyTorch 释放。

缺点更关键：

1. **显存上限不可控**：只要请求不断 demote，GPU tensor 会继续增长，直到 PyTorch allocator/OOM。
2. **实验对比不稳定**：`nvidia-smi` 看到的是 PyTorch reserved memory，很多小 tensor 的缓存行为会影响结果。
3. **碎片更多**：每个 layer、K/V、important/unimportant、q/min/step/ids 都是小 tensor，频繁创建释放容易造成 allocator 碎片。
4. **不能表达真实系统预算**：ZipCache 的目标是把一部分 KV cache 预算从 normal fp16 pool 切给 compressed pool，而不是让 compressed archive 无限扩张。
5. **pool full 行为不可控**：动态 archive 只能等 OOM 或 PyTorch 分配失败，不能提前通过统计决定是否 demote。

因此对当前实验目标来说，固定大小 compressed pool 更优。

### 0.2 当前固定大小 compressed pool 的实现

当前 v2 使用：

```python
class _V2CompressedPool:
    q_buffer: torch.uint8        # 保存两个 4bit value 打包后的 byte
    scale_buffer: torch.float16
    ids_buffer: torch.long

    q_allocator: _SegmentAllocator
    scale_allocator: _SegmentAllocator
    ids_allocator: _SegmentAllocator
```

每个 compressed entry 不再拥有独立 GPU tensor，而是保存 `TensorSlice`：

```text
entry
  -> layer
     -> k.important.q   指向 q_buffer 的一段 slice
     -> k.important.min 指向 scale_buffer 的一段 slice
     -> k.important.ids 指向 ids_buffer 的一段 slice
     -> ...
```

pool 容量配置：

```text
--zipcache-v2-compressed-pool-mb 4096
```

如果不指定 MB，则按原 KV pool 估算：

```text
--zipcache-v2-compressed-pool-ratio 0.35
```

当前代码默认使用 `0.35`。原因是 q 数据已经统一 4bit packed，q 本体约是 fp16/bf16 的 25%，再加上 scale/id/allocator metadata 后，`0.30 ~ 0.35` 更适合稳定实验。

日志会同时输出：

```text
compressed_pool_capacity_bytes
compressed_pool_used_bytes
compressed_pool_utilization
active_compressed_estimated_bytes_4bit
active_compressed_storage_bytes
```

当 pool 空间不足时，demote 会失败并保留原 normal fp16 KV，不影响正确性：

```text
[ZipCacheV2] demote failed: ... compressed pool is full ...
```

这比动态 archive 更适合实验，因为显存上限、pool 使用率、demote rejected 次数都可以直接统计。

### 0.3 compressed pool 的 packed 存储策略

当前代码中的 `q_buffer` 已经使用 `torch.uint8` 作为 packed byte buffer，而不是每个量化值占一个 uint8。两个 4bit value 会打包进一个 uint8：

```text
fp16/bf16 原始值: 16 bit
旧 uint8 q:       8 bit
当前 4bit packed: 4 bit
目标 2bit q:      2 bit
```

因此，当前 q 数据本体相比 fp16/bf16 已经可以省约 75%。但 unimportant token 物理上仍占 4bit，还没有达到严格 2bit/4bit mixed precision 的最优压缩率。

有两种可选设计。

#### 方案 A：严格 4bit/2bit mixed packed

important token 使用 4bit packed：

```text
2 个 4bit value -> 1 个 uint8
```

unimportant token 使用 2bit packed：

```text
4 个 2bit value -> 1 个 uint8
```

优点：

1. 最接近 ZipCache 论文设计；
2. unimportant token 的真实存储开销最低；
3. 在 unimportant ratio 较高时压缩率最好。

缺点：

1. compressed entry 需要保存更多 metadata：

   ```text
   bit_width
   original_shape
   logical_numel
   packed_numel
   pack_axis / flatten_order
   ids slice
   q slice
   min/step slice
   ```

2. restore 时必须按 `bit_width` 选择不同 unpack 路径；
3. 2bit 和 4bit 的 packed 长度不同，allocator 碎片会更明显；
4. 如果后续要支持 partial restore / node split，需要同时切分 packed q 和 ids，复杂度较高；
5. radix cache 命中后不再只拿到 compressed handle，还必须通过 handle 中的每个 `TensorSlice` metadata 才能正确 unpack。

也就是说，方案 A 的压缩率最好，但工程复杂度和错误风险也最高。

#### 方案 B：全部 q 数据统一 4bit packed

无论 important 还是 unimportant，真实存储都使用 4bit packed：

```text
important token:   4bit packed
unimportant token: 4bit packed
```

逻辑上仍然可以记录：

```text
requested_bit_width = 2 or 4
storage_bit_width = 4
```

如果 token 原本应按 2bit 量化，有两种选择：

1. 仍按 2bit 量化，q 值范围是 `[0, 3]`，但用 4bit slot 保存；
2. 直接按 4bit 量化，降低精度损失，但与论文 2bit unimportant 不完全一致。

v2 推荐优先采用第 1 种：**量化语义仍保留 2bit/4bit，物理存储统一 4bit packed**。

优点：

1. 相比旧版 uint8 storage，q 数据真实显存减半；
2. pack/unpack 只有一种 4bit 格式，restore 路径简单；
3. `TensorSlice` 只需要记录逻辑 bit 和 storage bit，不需要 2bit/4bit 两套复杂 unpack；
4. compressed pool allocator 统一按 packed byte 分配，碎片更少；
5. radix cache 命中路径几乎不变，只是 restore 时根据 handle metadata 从 4bit packed q 中解包。

缺点：

1. unimportant token 如果逻辑上是 2bit，会浪费一半 q 存储；
2. 压缩率低于严格 4bit/2bit mixed packed；
3. 与论文最优压缩率有差距。

综合当前 miniSGLang v2 的目标：先稳定在 GPU 上运行、便于比较显存、避免大改 kernel 和 cache 框架，推荐路线是：

```text
当前版本: 统一 4bit packed q storage
后续优化: important 4bit + unimportant 2bit mixed packed
```

统一 4bit packed 后，compressed pool 默认容量建议约为：

```text
compressed_pool_ratio = 0.30 ~ 0.35
```

其中 `0.25` 是纯 4bit q 数据本体，额外 `0.05 ~ 0.10` 留给 `min/step/ids/allocator` metadata。

## 1. 当前 v1 的问题

当前 `ZipCacheV1Manager` 的路径是：

```text
attention 前:
  CPU compressed entry -> GPU fp16/bf16 tensor -> 写回原 MHAKVCache

attention 后:
  从 GPU MHAKVCache gather K/V
  GPU 上量化
  压缩结果 .to("cpu")
```

所以 v1 有三个主要问题：

1. 压缩 KV 保存在 CPU，解压时有 CPU -> GPU 传输。
2. 原始 GPU KV pool 没有释放，`nvidia-smi` 显存不明显下降。
3. 每层每步都做 Python 层 gather/scatter 和 CPU tensor 管理，性能明显下降。

v2 要解决前两个问题：让压缩数据保存在 GPU，并让原始 fp16/bf16 KV page 被释放或复用。

## 2. 可行性结论

在“不修改 attention kernel”的约束下，可行方案不是让 kernel 直接读取 int2/int4 KV，而是：

```text
压缩后:
  原 fp16/bf16 KV page 不再作为长期存储
  低 bit KV 存入 GPU compressed pool
  原 fp16/bf16 page 归还给 free page pool

attention 前:
  如果请求命中 compressed KV
  从 GPU compressed pool 解压到一小块 fp16/bf16 restore workspace
  临时 page_table 指向 restore workspace
  原 attention kernel 仍读取 fp16/bf16 KV

attention 后:
  新生成的 KV 正常写入 fp16/bf16 active page
  到达压缩时机后再压缩进 GPU compressed pool
```

这个方案可行，因为：

- attention kernel 看到的仍是普通 fp16/bf16 `k_cache/v_cache`；
- page_table 仍然表示“逻辑 token -> 物理 token index”；
- radix prefix cache 仍然可以用 token prefix 做匹配；
- 压缩数据不离开 GPU；
- 原始 KV page 可以释放，理论上能降低长期 KV 显存占用。

关键代价：

- 需要新增 GPU compressed pool；
- 需要新增 restore workspace；
- 需要让 `CacheManager` 知道某些 prefix cache entry 已经从 fp16 page demote 到 compressed pool；
- 需要在 attention metadata 生成前完成 restore，并让本次 batch 的 page_table 指向 restored fp16 workspace。

## 3. 为什么不能直接把压缩 KV 存进现有 KV cache pool

当前 `MHAKVCache` 分配的是单一 dtype tensor：

```text
_kv_buffer:
[2, num_layers, num_pages, page_size, local_kv_heads, head_dim]
dtype = fp16/bf16
```

`k_cache(layer_id)` 和 `v_cache(layer_id)` 返回的也是 fp16/bf16 tensor。

当前 FlashAttention / FlashInfer / TensorRT-LLM backend 都假设：

```text
k_cache/v_cache 是 fp16/bf16
page_table 指向 fp16/bf16 KV page
```

如果把 int2/int4/int8 压缩数据直接塞进 `_kv_buffer`：

1. dtype 不匹配；
2. shape 不匹配；
3. scale/min metadata 无处存；
4. kernel 会把压缩 bit 当 fp16/bf16 读，结果错误；
5. 不修改 kernel 就无法直接消费压缩 KV。

所以 v2 需要“旁路压缩池 + 恢复 workspace”，而不是把压缩数据硬塞进原 KV pool。

## 4. v2 总体架构

建议新增：

```text
python/minisgl/zipcache/
  gpu_quant.py
  gpu_pool.py
  v2_manager.py
```

核心对象：

```text
ZipCacheV2Manager
ZipCacheCompressedPool
ZipCacheRestoreWorkspace
CompressedKVHandle
```

整体路径：

```text
prefill/decode 正常计算
  -> store_kv 写入 fp16 active KV page
  -> v2 after_attention 统计 saliency
  -> 满足压缩条件时 demote:
       fp16 active KV page -> GPU compressed pool
       原 fp16 page 释放给 CacheManager.free_slots

下一次请求或后续 decode 命中:
  -> CacheManager / prefix cache 发现 entry 是 compressed
  -> v2 restore:
       GPU compressed pool -> fp16 restore workspace
       当前 batch page_table 临时指向 restored indices
  -> 原 attention backend 正常执行
```

## 5. v2 的存储设计

### 5.1 原始 fp16 KV pool

仍然保留 `MHAKVCache`，但角色改变：

1. 当前活跃请求的新 token KV 仍写入这里；
2. attention kernel 仍从这里或同布局 workspace 读取；
3. 不再要求所有历史 prefix 长期都以 fp16 形式保留。

### 5.2 GPU compressed pool

新增 `ZipCacheCompressedPool`。

v2 设计中，`compressed pool` 不建议使用完全动态的 `torch.empty()` 小 tensor 反复分配。原因是：

1. 频繁动态分配 GPU tensor 会带来 allocator 开销；
2. 长时间运行容易出现显存碎片；
3. 压缩 entry 生命周期和 prefix cache eviction 相关，大小不固定；
4. 如果压缩池不可控增长，可能抵消释放原 fp16 KV page 带来的显存收益。

因此 v2 推荐采用“固定预算 + 池内动态分配”的方式：

```text
启动时或首次启用时，预留一块 compressed pool 显存预算；
运行时在这块 pool 内按 entry 动态切分、分配、释放。
```

也就是说，它不像原 `MHAKVCache` 那样每个 token 都有固定 fp16 page 位置，但也不是无限制地动态创建很多 GPU tensor。

推荐设计：

```text
外层: 固定大小 GPU compressed pool
内层: free list / segment allocator 管理不同 compressed entry
```

建议按 layer 管理，每层分别保存 K/V 的压缩数据：

```python
class ZipCacheCompressedPool:
    def __init__(
        self,
        num_layers: int,
        capacity_tokens: int,
        local_kv_heads: int,
        head_dim: int,
        device: torch.device,
        k_important_bit: int,
        k_unimportant_bit: int,
        v_important_bit: int,
        v_unimportant_bit: int,
    ):
        ...
```

### 5.2.1 compressed pool 是固定大小还是动态大小

推荐 v2 初版使用固定大小 pool。

配置项示例：

```text
--zipcache-v2-compressed-pool-mb 4096
--zipcache-v2-compressed-token-capacity 65536
```

两种配置方式二选一：

1. 按显存大小配置：`compressed_pool_mb`；
2. 按可压缩 token 数配置：`compressed_token_capacity`。

如果两者都提供，优先使用显存大小。初始化时根据模型结构估算一个 compressed token 的平均开销，再反推可容纳多少 token。

不推荐完全动态分配的原因：

```text
压缩 entry 大小不固定
频繁 torch.empty / torch.cat 容易造成显存碎片
碎片会导致明明总空闲足够但无法分配连续大 tensor
实验结果难以复现
```

固定 pool 的优点：

- 显存上限可控；
- 不会无限增长；
- 便于统计压缩池使用率；
- 便于和原 fp16 KV pool 对比；
- 便于后续替换为 Triton/CUDA pack kernel。

固定 pool 的缺点：

- pool 太小会导致 demote 失败；
- pool 太大会提前占用显存，降低收益；
- 需要 eviction/淘汰策略管理 compressed entries。

因此 v2 的 compressed pool 应输出这些统计：

```text
compressed_pool_capacity_bytes
compressed_pool_used_bytes
compressed_pool_free_bytes
compressed_pool_utilization
num_compressed_entries
num_demote_rejected_pool_full
num_compressed_evictions
```

### 5.2.2 compressed pool 与原 KV pool 的容量关系

结论：`compressed pool` 不应该设计得和原始 fp16/bf16 KV pool 一样大。它保存的是低 bit 压缩后的冷 KV，容量应该按“可被 demote 的 KV 数量 × 压缩后单 token 开销”估算。

原始 KV pool 的主要显存开销可以近似写成：

```text
original_kv_bytes =
  2
  * num_layers
  * num_tokens_or_pages
  * local_kv_heads
  * head_dim
  * dtype_bytes
```

其中：

- 第一个 `2` 表示同时存 Key 和 Value；
- `dtype_bytes = 2` 表示 fp16/bf16；
- `num_tokens_or_pages` 实际实现中可能以 token、page、slot 为单位估算，本质都是“可存多少 KV 位置”。

如果把全部冷 KV 都压缩成 4bit，并且先忽略 metadata，那么压缩数据本体只需要原始 fp16/bf16 KV 的 1/4：

```text
4bit_payload_bytes = original_kv_bytes * 4 / 16
                   = original_kv_bytes * 0.25
```

但是 ZipCache 不是只保存量化后的 `q`，还需要保存 `min/step/scale`、important/unimportant token id、entry 索引、allocator 对齐碎片等额外信息。因此不能直接把 compressed pool 配成原 KV pool 的 25%。更保守的 4bit 保底设计建议使用：

```text
compressed_pool_bytes_budget = ceil(original_kv_bytes * 0.30)
```

也可以写成更通用的形式：

```text
compressed_pool_bytes_budget =
  ceil(original_kv_bytes * 0.25 * metadata_factor)

metadata_factor 建议取 1.1 ~ 1.3
```

对 Llama 常见的 `head_dim = 128`，如果每个 token-head 为 K 和 V 都保存一组 fp16 的 `min + step`，则单 token-head 的比例大约是：

```text
原始 K+V:
  2 * head_dim * 2 bytes = 4 * head_dim bytes

4bit q 数据:
  2 * head_dim * 0.5 bytes = head_dim bytes

min/step metadata:
  K 的 min+step 4 bytes + V 的 min+step 4 bytes = 8 bytes

compressed / original:
  (head_dim + 8) / (4 * head_dim)

head_dim = 128 时:
  (128 + 8) / 512 = 0.266
```

也就是说，真实 4bit packed 存储加上基础 metadata 后，大约是原 fp16/bf16 KV 的 26.6%。考虑 allocator 对齐、entry 元数据和实现余量，按 30% 作为“全部被压缩为 4bit”的保底预算比较合理。

更重要的是，`compressed pool` 不应该额外叠加在一个完整大小的原 KV pool 后面，否则启动时显存可能变成：

```text
总显存 = 原完整 fp16 KV pool + compressed pool + restore workspace
```

这会削弱甚至抵消压缩收益。v2 更合理的预算关系应该是：

```text
total_cache_budget_bytes = 原 miniSGLang 计划分给 KV cache 的总预算

compressed_pool_bytes = total_cache_budget_bytes * zipcache_v2_compressed_pool_ratio
restore_workspace_bytes = 一小块临时恢复空间
normal_fp16_pool_bytes =
  total_cache_budget_bytes
  - compressed_pool_bytes
  - restore_workspace_bytes
```

当前统一 4bit packed 实现推荐默认值：

```text
--zipcache-v2-compressed-pool-ratio 0.35
```

原因是当前 q 数据本体已经是 4bit packed，约为 fp16/bf16 的 25%；再加上 min/step/ids/allocator metadata，`0.30 ~ 0.35` 比较稳。更激进的配置可以尝试 `0.30`，但 pool full 时 demote 会更容易失败。

如果实验目标不是“让全部 KV 都可被压缩保存”，而是只保存一部分冷 prefix，可以进一步按 demote 比例估算：

```text
compressed_pool_bytes =
  original_kv_bytes
  * expected_demote_fraction
  * compressed_ratio_with_metadata

compressed_ratio_with_metadata 建议先取 0.30
```

例如只希望最多保存相当于原 KV pool 50% 的冷 KV，则：

```text
compressed_pool_bytes = original_kv_bytes * 0.50 * 0.30
                      = original_kv_bytes * 0.15
```

旧版 v2-a 如果用 `torch.uint8` 或 `torch.int8` 存放每个量化值，但不做真正 4bit packing，那么量化数据本体是 8bit，不是 4bit。此时保守容量才需要按约 55% ~ 60% 的原 KV pool 估算：

```text
uint8_payload_bytes = original_kv_bytes * 0.50
uint8_with_metadata_budget = original_kv_bytes * 0.55 ~ 0.60
```

因此文档中的 30% 是“真实 packed 4bit”的目标设计；如果第一版实现还没有 bit packing，配置和统计中必须明确标记当前是 `uint8 storage`，不能把它解释为真实 4bit 显存压缩。

### 5.2.3 compressed pool 内部如何管理元素

压缩后的元素不能只用一个 tensor 表示，因为 Key/Value、important/unimportant、quantized data、scale/min metadata 的大小都不同。

推荐把一个 compressed entry 拆成多个 pool allocation：

```text
entry
├── layer 0
│   ├── k_important_q
│   ├── k_important_min
│   ├── k_important_step
│   ├── k_unimportant_q
│   ├── k_unimportant_min
│   ├── k_unimportant_step
│   ├── v_important_q
│   ├── v_important_min
│   ├── v_important_step
│   ├── v_unimportant_q
│   ├── v_unimportant_min
│   └── v_unimportant_step
├── layer 1
│   └── ...
└── layer N
    └── ...
```

每个 allocation 由一个 `TensorSlice` 描述：

```python
@dataclass
class TensorSlice:
    buffer_name: str              # 例如 "q_packed_u8", "scale_fp16", "ids_i64"
    offset: int                   # 在 flat buffer 中的起始元素下标
    length: int                   # buffer 中的元素数量；对 q 来说是 packed uint8 数量
    shape: tuple[int, ...]        # 解包后恢复原始 tensor view 时使用
    dtype: torch.dtype

    # packed q 需要的额外信息。
    logical_numel: int = 0        # 解包后有多少个量化值
    packed_numel: int = 0         # 实际占用多少个 uint8
    requested_bit_width: int = 4  # 算法希望使用的 bit，例如 important=4, unimportant=2
    storage_bit_width: int = 4    # 实际 packed 存储 bit。当前 v2 统一使用 4
    pack_order: str = "little"    # 低位优先，例如 q0 放低 4bit，q1 放高 4bit
```

compressed pool 内部维护若干大 buffer：

```python
class ZipCacheCompressedPool:
    q_packed_u8_buffer: torch.Tensor # uint8，保存 packed 4bit 或后续 mixed 2/4bit data
    scale_fp16_buffer: torch.Tensor  # fp16，保存 min/step/scale
    ids_i64_buffer: torch.Tensor     # int64，保存 important/unimportant token ids

    q_allocator: SegmentAllocator
    scale_allocator: SegmentAllocator
    ids_allocator: SegmentAllocator
```

为什么分多个 buffer，而不是一个 byte buffer：

- PyTorch 对不同 dtype 的 view/对齐处理更简单；
- `q`、`scale`、`ids` 生命周期一致但 dtype 不同；
- 便于统计每类数据真实占用；
- 实现 packed 4bit 时只需要替换 `q_packed_u8_buffer` 的写入/读取逻辑；
- 如果后续扩展 strict 2bit/4bit mixed packed，只需要让每个 `TensorSlice.storage_bit_width` 支持 2 或 4。

radix cache 命中 compressed prefix 后，restore 流程不应该只拿 `q` 的 offset，还必须读取 `TensorSlice` 中的 packed metadata：

```text
TensorSlice.offset / length
TensorSlice.logical_numel
TensorSlice.shape
TensorSlice.requested_bit_width
TensorSlice.storage_bit_width
TensorSlice.pack_order
```

否则无法知道：

1. 从 `q_packed_u8_buffer` 中读多少 byte；
2. 解包后应该恢复多少个 quantized value；
3. 解包后的 tensor 应该 reshape 成什么形状；
4. 反量化时应该按 2bit 语义还是 4bit 语义解释 q 值。

### 5.2.4 pool allocator 设计

推荐使用 segment allocator，而不是每个 entry 一个 tensor。

最小 allocator：

```python
class SegmentAllocator:
    free_segments: list[tuple[int, int]]  # (offset, length)

    def allocate(length: int) -> int | None:
        ...

    def free(offset: int, length: int) -> None:
        ...

    def merge_adjacent_free_segments() -> None:
        ...
```

分配策略：

1. first-fit：找到第一个长度足够的 free segment；
2. 分配后切分 segment；
3. 释放后合并相邻 segment；
4. 如果碎片太多，可触发 compressed entry eviction，而不是移动已存在 entry。

不建议 v2 初版做 compaction。因为 compaction 需要移动 GPU buffer 中的数据，并更新所有 `TensorSlice.offset`，实现复杂且容易引入一致性 bug。

### 5.2.5 compressed entry 如何索引和查找

compressed pool 只负责“存 bytes/tensor slice”，不负责 prefix 匹配。

prefix 匹配仍然由 radix cache / CacheManager 根据 token ids 完成。ZipCacheV2Manager 维护逻辑索引：

```python
class ZipCacheV2Manager:
    compressed_by_entry_id: dict[int, CompressedKVHandle]
    compressed_by_token_hash: dict[str, int]
    compressed_by_radix_node: dict[int, int]
```

查找路径：

```text
新请求 input_ids
  -> radix cache match token prefix
  -> 得到 radix node / token_hash
  -> ZipCacheV2Manager 查 compressed handle
  -> handle 记录 compressed pool 中的 TensorSlice
  -> restore 时根据 TensorSlice 从 pool 读 q/min/step/ids
```

也就是说：

```text
radix cache 负责“是否命中这个前缀”
compressed pool 负责“这个前缀的压缩 KV 存在哪里”
```

两者通过 `entry_id`、`token_hash` 或 `radix_node_id` 关联。

#### 原 miniSGLang 的缓存命中路径

从当前代码看，原 miniSGLang 的 prefix cache 命中路径是：

```text
PrefillAdder._try_allocate_one()
  -> CacheManager.match_req(req)
     -> prefix_cache.match_prefix(req.input_ids[:input_len - 1])
        -> RadixPrefixCache._tree_walk(input_ids)
        -> 返回 RadixCacheHandle(cached_len, node)
  -> cached_len = handle.cached_len
  -> extend_len = req.input_len - cached_len
  -> CacheManager.lock(handle)
  -> page_table[table_idx, :cached_len].copy_(handle.get_matched_indices())
```

其中 `RadixCacheHandle.get_matched_indices()` 会从命中 node 一路回溯到 root，把每个 radix node 的 `value` 拼起来：

```text
radix node.value = normal fp16/bf16 KV physical indices
```

所以原 miniSGLang 的“命中”同时包含两个含义：

1. token prefix 命中：当前请求的 token 前缀在 radix tree 中存在；
2. KV 数据命中：radix node.value 指向的 fp16/bf16 KV page 仍在原 KV pool 中，attention kernel 可以直接读取。

这两个条件在原实现里天然绑定，因为 radix node 被 evict 时会从树中删除，并把对应 fp16 page 释放。只要 radix tree 能匹配到 node，就意味着这些 `indices` 仍然有效。

#### ZipCache v2 的缓存命中路径

ZipCache v2 不能继续假设“radix 命中 == fp16 KV 可直接读”。原因是 demote 后：

```text
token prefix 仍然有复用价值
但原 fp16/bf16 page 已经释放
真实 KV 数据在 compressed pool 中
radix node.value 中的旧 indices 不能再写入 page_table
```

因此 v2 把命中拆成两层：

```text
第一层：prefix/radix token 命中
第二层：KV materialization 命中
```

完整路径建议设计为：

```text
PrefillAdder._try_allocate_one()
  -> CacheManager.match_req(req)
     -> radix_match = prefix_cache.match_prefix(req.input_ids[:input_len - 1])
     -> 如果未启用 ZipCache v2:
          直接返回 radix_match
     -> 如果启用 ZipCache v2:
          zipcache_manager.materialize_match(req.input_ids, radix_match.cuda_handle)
          -> 如果命中的是 normal fp16 node:
               返回原 RadixCacheHandle
          -> 如果命中的是 compressed node:
               从 compressed handle 读取 TensorSlice metadata
               从 compressed pool 读取 packed q/min/step/ids
               根据 storage_bit_width/logical_numel/shape unpack
               GPU dequantize 到 restore workspace 或重新分配的 normal page
               返回 RestoredCacheHandle(cached_len, restored_indices)
          -> 如果 restore 失败:
               返回 fallback handle，cached_len 只能覆盖仍然安全可读的 fp16 prefix
               compressed suffix 走原 prefill/recompute
  -> PrefillAdder 后续仍按 handle.cached_len 计算 extend_len
  -> page_table 写入 handle.get_matched_indices()
```

也就是说，v2 对 scheduler 暴露的接口尽量保持一致：

```python
handle.cached_len
handle.get_matched_indices()
```

但 `handle` 的来源可能不同：

| handle 类型 | indices 指向哪里 | 是否能长期插入 radix | 用途 |
| --- | --- | --- | --- |
| `RadixCacheHandle` | 原 normal fp16/bf16 KV pool | 可以 | 原始 prefix cache 命中 |
| `RestoredCacheHandle` | restore workspace 或重新 materialize 的 normal page | workspace 不可以；normal page 可以 | compressed 命中后的临时可读 KV |
| `EmptyCacheHandle` / fallback handle | 空或安全 fp16 前缀 | 可以按原逻辑处理 | restore 失败后回退 recompute |

#### v2 与原 miniSGLang 的一致点

v2 保持以下部分与原 miniSGLang 一致：

1. radix tree 仍然用 token ids 做 prefix 匹配；
2. `PrefillAdder` 仍然通过 `cached_len` 计算需要 prefill 的长度；
3. attention backend 仍然只看 page table 和 fp16/bf16 KV tensor；
4. 不修改 attention kernel；
5. restore 成功后，scheduler 看到的仍然是一组可读的 KV indices。

#### v2 与原 miniSGLang 的区别

v2 新增的区别是：

1. radix node 的 token 命中不再保证 `node.value` 是有效 fp16 page；
2. `CacheManager.match_req()` 必须在返回给 `PrefillAdder` 前完成 compressed 状态检查；
3. compressed 命中必须先 materialize，不能把旧 indices 直接写入 page table；
4. restore 失败时必须降低 `cached_len`，让未恢复部分重新 prefill；
5. restored workspace indices 不能作为长期 prefix cache value 再插入 radix。

需要注意：radix cache 只负责 token prefix 是否命中，不负责解释 packed bit 存储。`storage_bit_width`、`logical_numel`、`packed_numel`、`shape` 等信息必须保存在 compressed handle / `TensorSlice` 中，由 `ZipCacheV2Manager.materialize_match()` 在 restore 时使用。

这个拆分是 v2 正确性的核心。否则会出现严重错误：radix tree 命中了一个已经 demote 的 prefix，但 scheduler 仍把旧 page indices 写进新请求的 page table，而这些 page 可能已经被其他请求复用，attention 会读到错误 KV。

### 5.2.6 compressed entry 的释放

compressed entry 释放发生在三种情况：

1. compressed pool 空间不足，需要淘汰冷 entry；
2. radix prefix 被彻底删除，不再有复用价值；
3. 请求/实验结束时清理 manager。

释放流程：

```text
CompressedKVHandle
  -> 遍历所有 LayerCompressedSlot
  -> 对每个 TensorSlice 调 pool.free(offset, length)
  -> 删除 compressed_by_entry_id / token_hash / radix_node 映射
  -> 更新统计
```

注意：释放 compressed entry 不等价于释放 restore workspace。二者生命周期不同：

```text
compressed entry: 长期保存，等待未来命中
restore workspace: 临时恢复，attention 后可复用
```

### 5.2.7 pool 满时如何处理

当 compressed pool 空间不足时，不能影响推理正确性。策略：

1. 先尝试淘汰 ref_count == 0、最近最少访问的 compressed entry；
2. 淘汰后再次分配；
3. 如果仍然失败，则本次 demote 放弃；
4. 保留原 fp16 page，不释放；
5. 打日志和统计：

```text
[ZipCacheV2] demote skipped: compressed pool full
```

也就是说：

```text
压缩失败不能导致原 KV 丢失
只有压缩完整成功后才能释放原 fp16 page
```

这是 v2 正确性的硬性要求。

PyTorch 原生 tensor 不支持 int2/int4 dtype，所以低 bit KV 必须放在 `uint8` buffer 中手动 pack。建议分三阶段：

#### v2-a：GPU int8 affine quantization

先把 important/unimportant 都存成 `torch.int8` 或 `torch.uint8`，但分别按 bit width 统计压缩率。

优点：

- 全 GPU 可运行；
- 实现简单；
- 可验证 restore workspace 和 cache 命中链路。

缺点：

- 真实 GPU 显存不一定达到 2-bit/4-bit 估算；
- 但已经避免 CPU offload。

#### v2-b：统一 4bit packed uint8 storage

这是当前 v2 代码已经采用的实现。

无论 important 还是 unimportant，物理存储都用 4bit packed：

```text
4-bit: 2 个 value pack 到 1 个 uint8
```

unimportant token 如果算法配置为 2bit，则：

```text
requested_bit_width = 2
storage_bit_width = 4
```

也就是说，q 值仍然只取 `[0, 3]`，但存入 4bit slot。这样浪费一半 unimportant q 空间，但工程上有三个好处：

1. restore 只有一种 4bit unpack 路径；
2. q allocator 只按 packed uint8 管理，不需要区分 2bit/4bit pool；
3. radix 命中后 handle metadata 简单，不容易把 bit width、shape、offset 对错。

pack 示例：

```python
packed = (q0 & 0xF) | ((q1 & 0xF) << 4)
```

如果元素数是奇数，最后一个 uint8 的高 4bit 填 0，并通过 `logical_numel` 记录真实元素数量。

#### v2-c：严格 2bit/4bit mixed packed

实现 2-bit/4-bit packing：

```text
4-bit: 2 个 value pack 到 1 个 uint8
2-bit: 4 个 value pack 到 1 个 uint8
```

4bit：

```python
packed = (q0 & 0xF) | ((q1 & 0xF) << 4)
```

2-bit：

```python
packed = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)
```

严格 mixed packed 的额外要求：

1. 每个 q slice 必须保存 `storage_bit_width`；
2. 2bit 和 4bit 的 packed length 计算不同；
3. restore 时根据 `storage_bit_width` 选择 unpack 函数；
4. 如果 compressed node 将来要支持 split，需要按 bit packed 边界重新切分。

因此 v2-c 作为压缩率优化阶段，不作为当前优先实现。

### 5.3 scale/min metadata

每个 token/head/channel 需要保存 quantization metadata。

当前 v1 是：

```text
min:  [tokens, heads, 1]
step: [tokens, heads, 1]
q:    [tokens, heads, head_dim]
```

v2 可保留这个结构，但都放 GPU。

建议 metadata dtype：

```text
min/step: fp16
ids: int32
q: uint8 packed or int8
```

### 5.4 compressed entry handle

每个压缩 prefix/cache entry 记录：

```python
@dataclass
class CompressedKVHandle:
    entry_id: int
    token_hash: str
    token_ids: torch.Tensor          # CPU 或 GPU int32，供 prefix 匹配/调试
    seq_len: int
    num_layers: int
    page_size: int
    original_indices: torch.Tensor   # 原 fp16 page indices，仅用于调试/释放后不再可读
    compressed_slots: list[LayerCompressedSlot]
    state: Literal["compressed_gpu", "restored_workspace", "active_fp"]
    ref_count: int
    last_access_time: float
```

每层 slot：

```python
@dataclass
class LayerCompressedSlot:
    layer_id: int
    k_important_slot: TensorSlice
    k_unimportant_slot: TensorSlice
    v_important_slot: TensorSlice
    v_unimportant_slot: TensorSlice
    important_ids: torch.Tensor
    unimportant_ids: torch.Tensor
    shape: tuple[int, int, int]
    dtype: torch.dtype
```

## 6. restore workspace 设计

因为不改 attention kernel，kernel 必须读 fp16/bf16 KV。

因此新增：

```python
class ZipCacheRestoreWorkspace:
    k_buffer: torch.Tensor
    v_buffer: torch.Tensor
```

shape 建议和 `MHAKVCache` 单层 cache 一致：

```text
[num_restore_pages, page_size, local_kv_heads, head_dim]
```

或者 flatten：

```text
[num_restore_tokens, local_kv_heads, head_dim]
```

为了让现有 backend 少改，推荐把 workspace 包装成类似 `MHAKVCache` 的接口：

```python
class RestoredKVView:
    def k_cache(layer_id): ...
    def v_cache(layer_id): ...
```

但注意当前 backend 每次只处理一个 `layer_id`，所以 workspace 可以按 layer 复用：

```text
before layer attention:
  decompress layer_id 的 compressed KV 到 layer workspace
  backend.forward 使用 workspace + page_table
after layer attention:
  workspace 可复用给下一层
```

### 6.1 page_table 如何指向 workspace

当前 page_table 是全局物理 token index，指向 `MHAKVCache`。

v2 有两种方案。

#### 方案 A：把 workspace 作为 MHAKVCache 的高编号页面

在 `MHAKVCache` 初始化时额外分配 restore pages：

```text
normal pages:  [0, num_pages)
restore pages: [num_pages, num_pages + restore_pages)
```

restore 时把解压后的 KV 写入这些高编号 pages。

本次 batch 的 `page_table` 对已压缩 token 临时写成 restore page indices。

优点：

- backend 基本不用改；
- kernel 仍读同一个 `k_cache/v_cache` tensor；
- page_table 语义不变。

缺点：

- 需要在 `MHAKVCache` 预留 restore pages；
- restore pages 仍占 GPU fp16 显存；
- workspace 容量必须覆盖最大 batch restore token 数。

这是 v2 最推荐方案，因为对当前框架改动最小。

#### 方案 B：backend 支持外部 KV cache view

attention backend 根据当前 batch 选择：

```text
k_cache = normal kvcache 或 restored view
```

优点：

- restore workspace 可独立于 normal pool；
- 逻辑更清楚。

缺点：

- `fa.py`、`fi.py`、`trtllm.py` 都要改；
- HybridBackend / CUDA Graph 处理复杂；
- page_table 和 cache view 必须保持一致。

v2 第一版不建议。

## 7. KV cache 压缩后的释放和命中

### 7.1 demote 时机

为了不大改 scheduler，v2 第一版建议只在 prefix cache 可复用区域 demote：

1. 请求 prefill 完成后，`CacheManager.cache_req(req, finished=False)` 插入 prefix cache；
2. 当前 req 进入 decode；
3. prefix cache 中 `ref_count == 0` 的旧 prefix 才可以 demote；
4. demote 成功后，原 fp16 page 归还 `free_slots`。

更激进的“每个活跃请求历史 KV 都压缩”会影响 decode 热路径，需要频繁 restore，不建议 v2 初版做。

### 7.2 radix cache entry 状态

当前 radix node 只保存：

```text
key: token ids
value: fp16 KV physical indices
```

v2 需要让 radix node 或 handle 支持：

```text
value_kind = "fp16" or "compressed"
value = fp16_indices or compressed_handle
```

最小改动方案：

- 不直接改 `RadixTreeNode.value` 类型；
- 新增 `ZipCacheV2Manager.compressed_by_token_hash`；
- radix node 仍保存 token ids 和旧 indices；
- demote 后旧 indices 仅作为“逻辑 token 长度/调试信息”，真实 KV 在 manager 的 compressed handle 中；
- `CacheManager.match_req()` 命中 radix 后，调用 manager 检查该 prefix 是否已 compressed。

缺点：

- 旧 indices 已释放，不能再直接给 page_table；
- restore 成功后必须生成新的 restore indices 并返回一个临时 handle。

更干净方案：

- 修改 `RadixCacheHandle.get_matched_indices()`，如果节点 compressed，则触发 restore 或返回 restored indices。

但这会把 runtime 操作放进 kvcache 数据结构，不够清晰。

建议 v2 采用：

```python
CacheManager.match_req()
  -> radix match handle
  -> zipcache_manager.materialize_match(input_ids, handle)
  -> 返回 fp16 handle 或 restored handle
```

这里建议把函数命名为 `materialize_match()`，而不是简单的 `maybe_restore_match()`。原因是它做的不只是“尝试 restore”，还要保证返回给 scheduler 的 handle 一定满足：

```text
handle.cached_len 对应的每个 token 都有当前可读的 KV indices
handle.get_matched_indices() 返回的 indices 不能包含已经释放的旧 page
```

如果 radix 命中的路径中有多段 node，v2 需要逐段判断：

```text
root -> node A(fp16) -> node B(compressed) -> node C(compressed/fp16)
```

处理原则：

1. normal fp16 node：直接复用原 `node.value`；
2. compressed node：尝试 restore；
3. restore 成功：把 restored indices 接在结果后面；
4. restore 失败：命中长度截断到失败 node 之前，后面的 token 重新 prefill；
5. 不允许返回包含无效旧 indices 的 handle。

因此 `materialize_match()` 的结果可能比 radix token match 更短：

```text
radix_token_matched_len >= materialized_cached_len
```

这和原 miniSGLang 不同。原实现中两者永远相等。

### 7.3 restored handle

新增：

```python
class RestoredCacheHandle(BaseCacheHandle):
    cached_len: int
    restored_indices: torch.Tensor

    def get_matched_indices(self):
        return self.restored_indices
```

`PrefillAdder._try_allocate_one()` 后续逻辑不用大改：

```python
page_entry.copy_(handle.get_matched_indices())
```

它会把 restored workspace indices 写入当前请求的 page table。

如果只恢复了部分 prefix，`RestoredCacheHandle.cached_len` 必须等于实际恢复成功的长度，而不是 radix tree 原本匹配到的长度。这样 `PrefillAdder` 会把剩余 token 自动放入 `extend_len`，走原 prefill 路径，保证正确性。

## 8. v2 数据流

### 8.1 prefill 新请求，无 prefix 命中

```text
PrefillManager schedule
  -> CacheManager.allocate_paged 分配 normal fp16 pages
  -> attention store_kv 写入 normal fp16 pages
  -> after_attention 计算 saliency
  -> 请求 prefill 完成后 cache_req 插入 radix
```

这时可以选择不立即压缩，因为当前请求马上要 decode，需要热 KV。

### 8.2 prefix cache 冷却后 demote

```text
某 prefix node ref_count 变为 0
  -> ZipCacheV2Manager.demote_prefix(node)
  -> gather normal fp16 pages
  -> GPU quantize
  -> 写入 GPU compressed pool
  -> 原 normal fp16 pages 返回 free_slots
  -> compressed_by_token_hash[token_hash] = handle
```

注意：

- demote 不能阻止释放原 fp16 page；
- demote 失败则保持原 fp16 prefix cache；
- 只有 ref_count == 0 的 prefix 可以 demote。

### 8.3 新请求命中 compressed prefix

```text
PrefillAdder._try_allocate_one()
  -> CacheManager.match_req()
  -> radix match 命中 token prefix
  -> manager.materialize_match()
       如果 prefix compressed:
          从 restore workspace 分配临时 restore pages
          GPU dequantize compressed KV 到 restore pages
          返回 RestoredCacheHandle
       否则:
          返回原 radix handle
  -> token_pool 写入命中 token ids
  -> page_table 写入 restored indices
```

然后 attention backend 正常执行：

```text
prepare_metadata(batch)
  -> 根据 page_table 生成 backend metadata
forward()
  -> kernel 读取 restored fp16 pages
```

v2 的命中判断需要区分三种情况：

```text
1. normal hit
   radix 命中，且命中 path 上所有 node 都仍然指向 normal fp16 page。
   行为与原 miniSGLang 完全一致。

2. compressed hit + restore success
   radix 命中，某些 node 已 demote 到 compressed pool。
   materialize_match() 先把压缩 KV 恢复成 fp16/bf16 可读 indices，
   再把 RestoredCacheHandle 返回给 PrefillAdder。

3. compressed hit + restore fallback
   radix token 命中，但 compressed entry 缺失、pool handle 失效、workspace 不足、
   或 restore 过程失败。
   materialize_match() 不能返回原 cached_len，而要返回较短的安全 handle。
   未恢复的 suffix 由原 prefill 重算。
```

这保证 feature flag 开启后，即使压缩缓存命中路径失败，也只是性能退化为 recompute，不影响生成正确性。

### 8.4 请求结束

```text
Scheduler._free_req_resources()
  -> 如果使用了 restore workspace，释放 workspace allocation
  -> cache_req(req, finished=True)
  -> 不应把 workspace pages 当 normal pages 插入 radix
```

这是关键风险点。

v2 必须区分：

```text
normal fp16 page: 可以插入 radix / 可以长期存在
restore workspace page: 只供本次 batch/请求临时使用，不应长期进入 prefix cache
```

因此 `CacheManager.cache_req()` 需要知道哪些 indices 是 restore workspace。

最小方案：

- `ZipCacheV2Manager` 提供 `is_restore_index(indices)`；
- `cache_req()` 插入 prefix cache 前，把 restore indices 对应部分排除或重新 materialize 到 normal pages。

推荐第一版策略：

- compressed prefix restore 只用于 attention 读取；
- 不把 restored workspace 再插入 radix；
- 请求若继续 decode，新生成 token 使用 normal pages；
- 请求结束时 workspace 直接释放。

## 9. 需要修改的模块

### 9.1 `python/minisgl/kvcache/mha_pool.py`

新增 restore pages 支持：

```python
normal_num_pages
restore_num_pages
total_num_pages = normal_num_pages + restore_num_pages
```

需要提供：

```python
def flatten_k_cache(layer_id) -> torch.Tensor
def flatten_v_cache(layer_id) -> torch.Tensor
def normal_token_capacity() -> int
def restore_token_range() -> tuple[int, int]
```

注意：

- `Engine.num_pages` 应仍表示 normal pages，供 `CacheManager` 管理；
- restore pages 不进入 `CacheManager.free_slots`；
- dummy page 也要和 restore pages 区分。

### 9.2 `python/minisgl/engine/engine.py`

创建 `ZipCacheV2Manager`：

```python
self.zipcache_manager = ZipCacheV2Manager(
    config=config,
    kv_pool=self.kv_cache,
    page_table=self.page_table,
    cache_manager=...
)
```

但当前 `CacheManager` 是 Scheduler 中创建的，Engine 不持有它。

建议：

- Engine 只创建 manager，并传入 kv_pool/page_table；
- Scheduler 创建 CacheManager 后，再调用：

```python
self.engine.zipcache_manager.bind_cache_manager(self.cache_manager)
```

### 9.3 `python/minisgl/scheduler/cache.py`

新增：

```python
zipcache_manager: ZipCacheV2Manager | None
```

修改点：

1. `match_req()` 中先做 radix token match，再调用 `materialize_match()`，确保返回给 `PrefillAdder` 的 handle 只包含当前可读 KV indices；
2. `_allocate()` evict 前或 evict 后触发 demote；
3. `_free()` 不允许把 restore workspace indices 放回 normal free_slots；
4. `cache_req()` 插入 prefix 时跳过/处理 restore indices。

### 9.4 `python/minisgl/kvcache/radix_cache.py`

建议新增只读接口，便于 demote 时拿到 token ids 和 indices：

```python
def evict_with_entries(size) -> tuple[torch.Tensor, list[EvictedPrefixEntry]]
```

其中：

```python
@dataclass
class EvictedPrefixEntry:
    token_ids: torch.Tensor
    indices: torch.Tensor
    cached_len: int
```

v2 初版也可以不从 eviction 路径开始，而从 `cache_req(finished=True)` 或 idle demote 开始。

### 9.5 `python/minisgl/scheduler/prefill.py`

如果 `CacheManager.match_req()` 已经返回 restored handle，则 `PrefillAdder` 基本不用改。

如果 restore 需要额外资源检查，则 `_try_allocate_one()` 要加：

```text
restore_workspace_available >= materialized_cached_len
```

检查顺序建议放在 `CacheManager.match_req()` 或 `materialize_match()` 内部完成。这样 `_try_allocate_one()` 看到的 `cached_len` 已经是最终安全可用长度，不会因为 compressed prefix token 命中但 restore 失败而错误减少 `extend_len`。

```python
handle = self.cache_manager.match_req(req).cuda_handle
handle = self.cache_manager.try_restore_zipcache(req, handle)
```

### 9.6 `python/minisgl/attention/fa.py` / `fi.py` / `trtllm.py`

v2 推荐方案 A 下，backend 基本不用知道 compressed pool。

只需要：

- `prepare_metadata()` 之前 page_table 已经指向 restored workspace；
- `k_cache/v_cache` 仍然是 `MHAKVCache` 的总 buffer，包括 normal pages + restore pages。

因此 attention backend 可少改或不改。

## 10. v2 GPU 量化实现

### 10.1 PyTorch GPU quantize

第一版可继续使用 PyTorch GPU tensor：

```python
selected = x[ids].float()
min_val = selected.amin(dim=-1, keepdim=True)
max_val = selected.amax(dim=-1, keepdim=True)
step = ((max_val - min_val) / qmax).clamp_min(1e-6)
q = torch.round((selected - min_val) / step).clamp(0, qmax).to(torch.uint8)
```

关键是不 `.cpu()`。

### 10.2 GPU dequantize

```python
out[ids] = (q.float() * step + min_val).to(dtype)
```

仍然是 GPU tensor。

### 10.3 packed bit 存储

v2-a 可以先不 pack，只做 GPU-resident uint8。

v2-b 再 pack：

```text
4-bit: [N] -> [ceil(N/2)]
2-bit: [N] -> [ceil(N/4)]
```

需要记录：

```text
original_shape
packed_shape
bit_width
padding
```

## 11. 显存预算

当前原始 KV 显存：

```text
2 * num_layers * num_pages * page_size * local_kv_heads * head_dim * dtype_bytes
```

v2 后显存组成：

```text
normal active fp16 pages
+ restore workspace fp16 pages
+ compressed GPU pool
+ quant metadata
+ temporary tensors
```

只有当：

```text
compressed pool + restore workspace < 被释放的 fp16 KV pages
```

GPU 显存才会下降。

因此 restore workspace 不能太大。

建议配置：

```text
--zipcache-v2-restore-pages
--zipcache-v2-compressed-token-capacity
--zipcache-v2-demote-threshold-tokens
```

## 12. v2 初版推荐边界

为了方案可落地，v2 初版建议：

1. 只支持 `--cache-type radix`；
2. 只支持 FlashAttention backend，FlashInfer/TRTLLM 后续验证；
3. 暂时禁用 CUDA Graph；
4. 压缩数据保存在 GPU，但先不做 2-bit/4-bit packing；
5. 先实现 prefix cache 冷数据 demote，不压缩活跃 decode 请求；
6. restore workspace 使用 `MHAKVCache` 预留高编号 pages；
7. restore 失败 fallback recompute；
8. 统计真实 GPU tensor allocated bytes 和估算 bit-pack bytes 分开输出。

## 13. v2 风险点

### 13.1 page_table 污染

restore workspace indices 不能长期写入 radix cache。否则后续 workspace 被复用后，旧 prefix 会读到错误 KV。

解决：

- handle 标记 restored；
- `cache_req()` 识别 restored indices；
- 不把 restored workspace 插入 prefix cache。

### 13.2 restore workspace 容量不足

如果一个 batch 命中很长 compressed prefix，workspace 不够。

策略：

1. restore 失败；
2. fallback recompute；
3. 或只 restore 部分 prefix。

第一版推荐 fallback recompute。

### 13.3 TP 并行

每个 TP rank 只保存本 rank KV heads。

compressed handle 必须包含：

```text
tp_rank
tp_size
local_kv_heads
```

每个 rank 独立压缩/恢复自己的 KV。

### 13.4 radix ref_count

只允许 demote ref_count == 0 的 prefix。正在被请求使用的 prefix 不能释放 fp16 pages。

### 13.5 显存碎片和容量管理

GPU compressed pool 如果采用完全动态的小 tensor 分配，会产生碎片，并且显存上限不可控。

建议：

- v2 初版就使用固定大小 compressed pool；
- pool 内部使用 segment allocator / free list 动态管理 entry；
- pool 满时优先淘汰冷 compressed entry；
- 淘汰后仍不够，则本次 demote 失败并保留原 fp16 page；
- 不做在线 compaction，避免移动数据后更新大量 handle offset。

判断 pool 设计是否合理的关键指标：

```text
compressed_pool_capacity_bytes
compressed_pool_used_bytes
compressed_pool_utilization
num_demote_rejected_pool_full
num_compressed_evictions
```

## 14. 推荐实现顺序

1. 新增 v2 feature flag：

```text
--enable-zipcache-v2
```

2. 修改 `MHAKVCache`，预留 restore pages，但默认关闭时行为不变。
3. 新增固定预算 `ZipCacheCompressedPool`，实现 `q_u8/scale_fp16/ids_i32` buffer 和 segment allocator。
4. 新增 `ZipCacheV2Manager`，实现 GPU-only quant/dequant，不 `.cpu()`。
5. 新增 `RestoredCacheHandle`。
6. 修改 `CacheManager.match_req()`，支持 compressed hit restore。
7. 修改 `CacheManager.cache_req()`，避免 restore workspace 进入 radix。
8. 增加 idle/demote 路径，把冷 prefix 从 fp16 page 压缩到 GPU compressed pool。
9. restore 成功后，让 page_table 指向 restore pages。
10. 跑 shared-prefix workload 验证：
   - 输出不崩；
   - `num_gpu_compressions > 0`；
   - `num_gpu_restores > 0`；
   - `num_restore_failures == 0`；
   - `compressed_pool_utilization` 合理；
   - `num_demote_rejected_pool_full` 不应持续增长；
   - GPU 显存峰值相比 v1 不增加；
   - 在 packed 版本中显存有下降趋势。

## 15. v2 与原 ZipCache 论文的关系

原 ZipCache 更接近：

```text
GPU 上保存压缩 KV
GPU 上解压/恢复
保持 FlashAttention 兼容
减少 GPU KV cache memory
```

本文 v2 方案也是这个方向。

但由于当前 miniSGLang 的 attention kernel 仍要求 fp16/bf16 KV，所以 v2 采用：

```text
GPU compressed pool + GPU restore workspace + 原 kernel
```

它不是 kernel 直接读 int2/int4 KV，属于“不改 kernel”的可行折中。

## 16. 最终判断

v2 在当前约束下可行，但必须明确：

1. 压缩 KV 不能直接存进现有 fp16 `MHAKVCache._kv_buffer`；
2. 需要独立且固定预算的 GPU compressed pool；
3. 需要 fp16/bf16 restore workspace；
4. attention 前必须把 compressed KV 恢复到 workspace；
5. page_table 必须临时指向 restored workspace；
6. restored workspace 不能进入 radix cache 的长期 prefix entry；
7. 真正显存下降取决于释放的 normal fp16 pages 是否大于 compressed pool + workspace + metadata。

推荐 v2 初版先实现 GPU-only 固定预算 compressed pool、pool 内 segment allocator 和 restore workspace，验证功能正确后，再做 packed 2-bit/4-bit 和更精细的 cache eviction/demotion 策略。

## 17. v2 相比原 miniSGLang 的优势：为什么要压缩 KV cache

原 miniSGLang 的 KV cache 设计简单直接：所有已经计算过的历史 token KV 都长期以 fp16/bf16 形式保存在 GPU `MHAKVCache` 中，prefix/radix cache 命中时直接复用这些 fp16/bf16 page。

这种设计的优点是读取快、实现简单、attention kernel 可以直接使用；缺点是 KV cache 显存随上下文长度、batch size、并发请求数线性增长。

### 17.1 原 miniSGLang 的主要瓶颈

LLM 推理中，模型权重通常是固定显存开销，而 KV cache 是随请求动态增长的显存开销。

对每个 token，每层都要保存 Key 和 Value：

```text
KV bytes ≈ 2 * num_layers * num_kv_heads * head_dim * dtype_bytes * num_tokens
```

因此，当出现以下场景时，原 miniSGLang 很容易被 KV cache 限制：

1. 长上下文输入，例如 8K、16K、32K prompt；
2. 多请求并发；
3. shared-prefix workload 中大量 prefix cache 常驻；
4. decode 请求持续生成较长输出；
5. 小模型权重不大，但 KV cache 反而成为主要显存占用。

如果 KV cache 占满显存，系统只能：

- 降低并发；
- 降低最大上下文长度；
- 频繁 evict prefix cache；
- 重新 prefill，增加 TTFT；
- 甚至 OOM。

### 17.2 为什么压缩 KV cache

KV cache 中并不是所有 token 对后续 attention 都同等重要。ZipCache 的核心观察是：一部分 salient token 对输出更关键，另一部分 token 的 attention 贡献较小，可以用更低 bit 保存。

因此，压缩 KV cache 的目标是：

```text
用更少显存保存历史 KV，同时尽量保持生成质量。
```

具体到 v2，压缩的意义包括：

1. **降低长期 KV 显存占用**

   被 demote 的 fp16/bf16 KV page 不再长期占用原始 KV pool，而是转换成 GPU compressed pool 中的低 bit 表示。

2. **提高可承载上下文长度**

   同样显存下，可以保留更多历史 token 的 KV cache，减少因为空间不足导致的 prefix cache eviction。

3. **提高并发能力**

   当单请求 KV 占用降低后，同一张 GPU 可以容纳更多同时运行的请求。

4. **减少 recompute / prefill 开销**

   如果 prefix cache 因显存不足被直接 evict，后续相同前缀请求只能重新 prefill。v2 希望把这部分 prefix 以压缩形式保留下来，后续命中后 restore，而不是完全重算。

5. **让 radix prefix cache 更适合长前缀场景**

   原 radix cache 命中依赖 fp16 page 常驻。v2 将其扩展为：

   ```text
   prefix 命中 -> 判断 entry 状态 -> fp16 直接读 或 compressed restore
   ```

   这样 prefix cache 不再只能在“保留完整 fp16 KV”和“彻底丢弃”之间二选一。

### 17.3 v2 相比原 miniSGLang 的预期优势

| 维度 | 原 miniSGLang | ZipCache v2 预期 |
| --- | --- | --- |
| KV 存储格式 | 长期 fp16/bf16 | 热 KV 用 fp16/bf16，冷 KV 用 GPU compressed pool |
| prefix cache 淘汰 | 空间不足时释放 fp16 page，后续需 recompute | demote 成压缩 KV，后续可 restore |
| 长上下文能力 | 受 fp16 KV 显存限制 | 同显存下可保留更多历史 KV |
| 并发能力 | KV cache 随并发线性增长 | 冷 KV 压缩后可缓解显存压力 |
| GPU 显存 | KV page 全量 fp16/bf16 常驻 | 取决于 compressed pool + restore workspace，小于原 fp16 page 时下降 |
| attention kernel | 直接读 fp16/bf16 | 仍直接读 fp16/bf16 restore workspace，不改 kernel |
| 实现复杂度 | 低 | 中等，需要 compressed entry 状态和 restore workspace |

### 17.4 v2 的收益来自哪里

v2 的收益不是来自“每次 attention 都更快”。事实上，如果每步都频繁压缩/解压，性能可能下降。

v2 真正的收益来自：

```text
把不常访问的冷 KV 从昂贵的 fp16/bf16 page 变成低 bit GPU 压缩存储。
```

也就是说，v2 更适合以下访问模式：

- 长 prefix 被多个请求复用；
- prefix cache 中有大量 ref_count 为 0 的冷节点；
- 显存压力导致原系统会 evict prefix；
- 压缩后的 prefix 未来有较高概率再次命中；
- restore 成本低于重新 prefill 成本。

### 17.5 v2 不应追求压缩所有 KV

v2 不应该把所有 KV 都立刻压缩。热 KV 如果每步都要用，频繁压缩/解压会抵消收益。

推荐策略：

```text
热 KV: 保持 fp16/bf16，保证 decode 性能
冷 prefix KV: demote 到 compressed GPU pool
再次命中: restore 到 workspace
长期高频命中: 可 promote 回 fp16 page
```

这种分层策略相比原 miniSGLang 更灵活：

```text
原 miniSGLang:
  fp16 常驻 或 evict 丢弃

ZipCache v2:
  fp16 热缓存 / compressed 冷缓存 / evict 丢弃
```

### 17.6 v2 成功的判断标准

v2 相比原 miniSGLang 是否有优势，不能只看压缩率，还要综合看：

1. `gpu_memory_used_mb_max` 是否下降；
2. 可支持的 `max_running_requests` 是否增加；
3. 可支持的长上下文长度是否增加；
4. prefix cache eviction 是否减少；
5. shared-prefix workload 下 TTFT 是否降低或少恶化；
6. restore 成功率是否高；
7. 输出正确性是否保持；
8. 压缩/restore 开销是否小于 recompute/prefill 开销。

如果只是压缩率高，但：

```text
原 fp16 page 没释放
restore workspace 过大
频繁压缩热 KV
restore 失败后频繁 recompute
```

那么 v2 不会比原 miniSGLang 更好，甚至可能更慢、更占显存。

因此，v2 的核心不是“为了压缩而压缩”，而是：

```text
在显存有限时，用低 bit GPU 压缩缓存保留更多有复用价值的历史 KV，
从而提升长上下文和多并发场景下的有效缓存容量。
```

## 18. 如何启动 v2 并与 main 版本对比测试

本节说明如何在云服务器上启动原始 main 版本和 ZipCache v2 版本，并使用 `experiment/` 目录下的一键脚本做对比。

注意：本节中早期示例若仍出现 `experiment/data/*.jsonl`，仅代表历史手工数据路径。当前默认测试已经切换到公开数据集派生的：

```text
experiment/workloads/
```

最新命令以本文档第 19 节和 `experiment/README.md` 为准。

### 18.1 测试前准备

建议 main 和 ZipCache v2 使用完全相同的条件：

```text
同一个模型路径
同一个 GPU
同一个 dtype
同一个 cache-type
同一个 max-running-requests
同一个 max-prefill-length / max-extend-length
同一个 page-size
同一份 experiment/workloads 数据集
同一套 benchmark 参数
```

如果你只有一个工作目录，可以按顺序测试：

```bash
git switch main
# 启动 main 服务并跑测试

git switch ZipCache
# 启动 ZipCache v2 服务并跑测试
```

如果你有两个工作目录，也可以分别 clone 两份仓库，一个停在 `main` 分支，一个停在 `ZipCache` 分支，同时启动两个服务：

```text
main:        http://127.0.0.1:30000
ZipCache v2: http://127.0.0.1:30001
```

### 18.2 启动 main 版本

在 main 分支目录中启动服务：

```bash
git switch main

PYTHONPATH=python python -m minisgl \
  --model-path /root/autodl-tmp/modelscope-cache/models/Qwen/Qwen3-0___6B \
  --num-pages 370000 \
  --host 0.0.0.0 \
  --port 30000 \
  --cache-type radix \
  --max-running-requests 16 \
  --max-prefill-length 4096 \
  2>&1 | tee main_server.log
```

注意：

- `/path/to/model` 必须替换成真实模型目录，目录下应包含 `config.json`。
- main 版本不要带任何 `--enable-zipcache-*` 参数。
- main 服务日志中不应该出现 `[ZipCacheV2]`。

### 18.3 启动 ZipCache v2 版本

在 ZipCache 分支目录中启动服务：

```bash
git switch ZipCache

PYTHONPATH=python python -m minisgl \
  --model-path /root/autodl-tmp/modelscope-cache/models/Qwen/Qwen3-0___6B \
  --host 0.0.0.0 \
  --port 30001 \
  --cache-type radix \
  --num-pages 370000 \
  --max-running-requests 16 \
  --max-prefill-length 4096 \
  --enable-zipcache-v2 \
  --zipcache-unimportant-ratio 0.4 \
  --zipcache-k-important-bit 4 \
  --zipcache-k-unimportant-bit 2 \
  --zipcache-v-important-bit 4 \
  --zipcache-v-unimportant-bit 2 \
  --zipcache-v2-compressed-pool-ratio 0.35 \
  --zipcache-stats-interval 10 \
  2>&1 | tee zipcache_v2_server.log
```

这里建议显式设置 `--num-pages`，用它限制原始 fp16/bf16 normal KV pool 的大小，为 ZipCache v2 的 compressed pool 留出显存。

例如云服务器日志中出现：

```text
Allocating 652296 tokens for KV cache, K + V = 69.67 GiB
```

如果 `page_size=1`，可以近似认为：

```text
每个 page 显存 ~= 69.67 GiB / 652296 ~= 0.0001068 GiB
40 GiB normal KV pool ~= 40 / 0.0001068 ~= 374500 pages
```

因此可以先设置：

```bash
--num-pages 370000
```

这样 normal KV pool 约为 39.5 GiB，剩余显存可以留给模型权重、compressed pool、临时 tensor 和 CUDA/PyTorch 运行时开销。如果仍然 OOM，就继续降低到 `350000`、`320000` 或更小；如果显存余量充足，再逐步调大。

注意：`--num-pages` 控制的是 miniSGLang 原本的 normal KV cache page 数，不是 compressed pool 的大小。compressed pool 仍由下面两个 ZipCache 参数控制。

如果你想手动指定 compressed pool 大小，而不是按原 KV pool 比例估算，可以使用：

```bash
--zipcache-v2-compressed-pool-mb 4096
```

此时会优先使用固定 MB 值，忽略 `--zipcache-v2-compressed-pool-ratio`。

ZipCache v2 启动成功后，应能在服务端日志中看到：

```text
[ZipCacheV2] enabled: GPU prefix demotion ...
[ZipCacheV2] compressed pool initialized: capacity=...
```

当前 v2 不自动改变 compressed pool 的分配方式，也不强行接管 normal KV pool 的预算。推荐的显存分配方式是手动控制：

```text
总显存 ~= 模型权重 + normal_fp16_kv_pool(--num-pages) + compressed_pool + 临时运行开销
```

因此如果启动时看到 normal KV pool 已经吃掉大部分显存，例如 `K + V = 69.67 GiB`，再额外分配 compressed pool 就很容易 OOM。此时应优先降低 `--num-pages`，而不是继续增大 `--zipcache-v2-compressed-pool-ratio`。

当发生 demote / restore 时，应能看到类似日志：

```text
[ZipCacheV2] demoted: entry_id=... node=... tokens=... original=... estimated_4bit=... gpu_storage=...
[ZipCacheV2] restored: entry_id=... node=... tokens=...
[ZipCacheV2] stats: {...}
```

### 18.4 一键运行 main 测试

main 服务启动后，在另一个终端执行：

```bash
python experiment/run_all_experiments.py \
  --mode main \
  --base-url http://127.0.0.1:30000 \
  --log-root experiment/logs \
  --gpu-sample-interval 0.5 \
  --timeout 600
```

结果会保存到：

```text
experiment/logs/<时间>_main/
```

重点文件：

```text
report.md
all_results_summary.json
shared_prefix_summary.json
realistic_long_context_summary.json
zipcache_restore_probe_summary.json
zipcache_restore_pressure_summary.json
mixed_length_summary.json
correctness_summary.json
gsm8k_correctness_summary.json
gsm8k_correctness_eval.json
```

### 18.5 一键运行 ZipCache v2 测试

ZipCache v2 服务启动后，在另一个终端执行：

```bash
python experiment/run_all_experiments.py \
  --mode zipcache_v2 \
  --base-url http://127.0.0.1:30001 \
  --log-root experiment/logs \
  --server-log zipcache_v2_server.log \
  --gpu-sample-interval 0.5 \
  --timeout 1200
```

结果会保存到：

```text
experiment/logs/<时间>_zipcache_v2/
```

说明：

- `report.md` 会记录本次运行模式和每组实验结果。
- `--server-log` 用来保留服务端 ZipCache 日志路径，便于后续检查 `[ZipCacheV2] stats`。
- 当前 `experiment/parse_zipcache_log.py` 已支持 `[ZipCacheV1] stats` 和 `[ZipCacheV2] stats`。如果 report 中没有解析到 v2 stats，可以先用 `grep` 手动确认服务端日志中是否存在 v2 输出：

```bash
grep "\[ZipCacheV2\]" zipcache_v2_server.log
grep "\[ZipCacheV2\] stats" zipcache_v2_server.log
```

### 18.5.1 更实际的长上下文 workload

当前一键脚本已经不只跑短 prompt。为了让云服务器上能观察到更明显的 KV cache 显存压力，新增了：

```text
realistic_long_context
```

对应数据文件：

```text
experiment/data/realistic_long_context.jsonl
```

默认配置：

```text
rows = 12
prompt_chars ~= 3000
concurrency = 8
repeat = 3
max_tokens = 512
```

这组 workload 会产生 36 个长上下文请求，并发度为 8，输出长度也从原来的 96/128 提高到 512。它更适合观察：

```text
gpu_memory_used_mb_max
gpu_memory_used_mb_avg
TTFT / TPOT / E2E
compressed_pool_used_bytes
active_storage_compression_ratio
num_demotions
```

单独运行方式：

```bash
python experiment/bench_openai_stream.py \
  --base-url http://127.0.0.1:30001 \
  --dataset experiment/data/realistic_long_context.jsonl \
  --output experiment/results/zipcache_realistic_long_context.jsonl \
  --summary experiment/results/zipcache_realistic_long_context_summary.json \
  --concurrency 8 \
  --repeat 3 \
  --max-tokens 512 \
  --gpu-sample-interval 0.5 \
  --timeout 1200
```

main 分支也应使用同一数据集和同一参数运行，只把 `--base-url` 改成 main 服务地址。

### 18.5.2 更强 restore workload：zipcache_restore_probe / zipcache_restore_pressure

当前一键脚本会额外运行：

```text
zipcache_restore_probe
zipcache_restore_pressure
```

对应数据文件：

```text
experiment/data/zipcache_restore_probe.jsonl
experiment/data/zipcache_restore_pressure.jsonl
```

这组 workload 的目标不是测最高吞吐，而是专门验证 ZipCache v2 的 compressed KV restore 链路。它的设计特点是：

- `zipcache_restore_probe`：1 条固定长 prompt，`concurrency=1`，`repeat=8`，`max_tokens=256`；
- `zipcache_restore_pressure`：4 条共享超长前缀 prompt，平均约 4000 字符，`concurrency=1`，`repeat=6`，`max_tokens=384`；
- 两组都采用顺序发送，优先制造“第一次请求结束 demote，后续请求再次访问相同或高度相同前缀”的条件。

这样设计的原因是：ZipCache v2 当前在请求结束或 radix entry 变冷后执行 demote。要触发 compressed hit，必须让“后续请求”再次访问已经被 demote 的相同 prefix。如果使用高并发同时发请求，很多请求会在第一个请求 demote 之前就已经进入 prefill，反而不容易验证 restore。

理想情况下，服务端日志应出现：

```text
[ZipCacheV2] demoted: ...
[ZipCacheV2] restored: ...
[ZipCacheV2] stats: {...}
```

并且 stats 中应看到：

```text
num_demotions > 0
num_compressed_hits > 0
num_restore_attempts > 0
num_restore_success > 0
```

如果仍然出现：

```text
num_demotions > 0
num_compressed_hits = 0
num_restore_success = 0
```

说明当前 v2 已经完成压缩保存，但后续请求仍未命中 compressed radix entry。此时优先排查：

- 服务是否带了 `--cache-type radix`；
- 请求 prompt 是否完全相同；
- demote 是否发生在请求结束后；
- radix cache 是否因为 split / lock / eviction 逻辑没有返回 compressed node；
- `CacheManager.match_req()` 是否调用了 `ZipCacheV2Manager.materialize_match()`。

这组 workload 可以单独运行：

```bash
python experiment/bench_openai_stream.py \
  --base-url http://127.0.0.1:30001 \
  --dataset experiment/data/zipcache_restore_probe.jsonl \
  --output experiment/results/zipcache_restore_probe.jsonl \
  --summary experiment/results/zipcache_restore_probe_summary.json \
  --concurrency 1 \
  --repeat 8 \
  --max-tokens 256 \
  --gpu-sample-interval 0.5
```

更强的 restore 压力测试：

```bash
python experiment/bench_openai_stream.py \
  --base-url http://127.0.0.1:30001 \
  --dataset experiment/data/zipcache_restore_pressure.jsonl \
  --output experiment/results/zipcache_restore_pressure.jsonl \
  --summary experiment/results/zipcache_restore_pressure_summary.json \
  --concurrency 1 \
  --repeat 6 \
  --max-tokens 384 \
  --gpu-sample-interval 0.5
```

运行后检查：

```bash
grep "\[ZipCacheV2\] demoted" zipcache_v2_server.log
grep "\[ZipCacheV2\] restored" zipcache_v2_server.log
grep "\[ZipCacheV2\] stats" zipcache_v2_server.log
```

### 18.6 对比 main 和 ZipCache v2 的结果

假设两次测试结果目录分别是：

```text
experiment/logs/2026xxxx_xxxxxx_main/
experiment/logs/2026xxxx_xxxxxx_zipcache_v2/
```

可以先直接阅读两个 `report.md`：

```bash
cat experiment/logs/2026xxxx_xxxxxx_main/report.md
cat experiment/logs/2026xxxx_xxxxxx_zipcache_v2/report.md
```

也可以用 `compare_results.py` 对单个 workload 的 jsonl 结果做对比。

shared-prefix 对比：

```bash
python experiment/compare_results.py \
  --baseline experiment/logs/2026xxxx_xxxxxx_main/shared_prefix.jsonl \
  --candidate experiment/logs/2026xxxx_xxxxxx_zipcache_v2/shared_prefix.jsonl
```

ZipCache v2 restore 强命中测试对比：

```bash
python experiment/compare_results.py \
  --baseline experiment/logs/2026xxxx_xxxxxx_main/zipcache_restore_probe.jsonl \
  --candidate experiment/logs/2026xxxx_xxxxxx_zipcache_v2/zipcache_restore_probe.jsonl
```

这组对比主要用于确认输出没有明显异常，以及观察 ZipCache v2 服务端 stats 中是否出现 `num_compressed_hits`、`num_restore_attempts` 和 `num_restore_success`。由于它是 `concurrency=1` 的顺序重复请求，不应该把它当成高并发吞吐压力测试。

真实长上下文压力测试对比：

```bash
python experiment/compare_results.py \
  --baseline experiment/logs/2026xxxx_xxxxxx_main/realistic_long_context.jsonl \
  --candidate experiment/logs/2026xxxx_xxxxxx_zipcache_v2/realistic_long_context.jsonl
```

restore 压力测试对比：

```bash
python experiment/compare_results.py \
  --baseline experiment/logs/2026xxxx_xxxxxx_main/zipcache_restore_pressure.jsonl \
  --candidate experiment/logs/2026xxxx_xxxxxx_zipcache_v2/zipcache_restore_pressure.jsonl
```

mixed-length 对比：

```bash
python experiment/compare_results.py \
  --baseline experiment/logs/2026xxxx_xxxxxx_main/mixed_length.jsonl \
  --candidate experiment/logs/2026xxxx_xxxxxx_zipcache_v2/mixed_length.jsonl
```

正确性输出对比：

```bash
python experiment/compare_results.py \
  --baseline experiment/logs/2026xxxx_xxxxxx_main/correctness.jsonl \
  --candidate experiment/logs/2026xxxx_xxxxxx_zipcache_v2/correctness.jsonl \
  --show-text
```

GSM8K 数字答案正确率对比：

```bash
cat experiment/logs/2026xxxx_xxxxxx_main/gsm8k_correctness_eval.json
cat experiment/logs/2026xxxx_xxxxxx_zipcache_v2/gsm8k_correctness_eval.json
```

也可以单独对某次输出重新判分：

```bash
python experiment/evaluate_correctness.py \
  --dataset experiment/data/gsm8k_correctness.jsonl \
  --results experiment/logs/2026xxxx_xxxxxx_zipcache_v2/gsm8k_correctness.jsonl \
  --output experiment/logs/2026xxxx_xxxxxx_zipcache_v2/gsm8k_correctness_eval.json
```

### 18.7 主要对比指标

#### 性能指标

从 `report.md` 和 `*_summary.json` 中重点看：

```text
request_throughput_rps
output_chunks_per_s
ttft_avg_s
ttft_p50_s
ttft_p90_s
e2e_avg_s
e2e_p50_s
e2e_p90_s
tpot_avg_s
```

含义：

- `request_throughput_rps` 越高越好；
- `output_chunks_per_s` 越高越好；
- `ttft` 越低越好，表示首 token 延迟；
- `e2e` 越低越好，表示端到端请求耗时；
- `tpot` 越低越好，表示每个输出 token 平均耗时。

#### GPU 显存指标

重点看：

```text
gpu_memory_used_mb_min
gpu_memory_used_mb_max
gpu_memory_used_mb_avg
```

ZipCache v2 的目标是：

```text
在 shared-prefix / 长上下文 / 高并发场景下，gpu_memory_used_mb_max 低于 main。
```

但如果 compressed pool 预留太大，或者 demote 的冷 prefix 很少，显存可能不会明显下降。

#### ZipCache v2 内部指标

从 `zipcache_v2_server.log` 中查看 `[ZipCacheV2] stats`，重点看：

```text
num_demotions
num_compressed_entries
num_compressed_hits
num_restore_attempts
num_restore_success
num_restore_fallback
num_demote_rejected_pool_full
active_estimated_compression_ratio
active_storage_compression_ratio
compressed_pool_capacity_bytes
compressed_pool_used_bytes
compressed_pool_utilization
active_original_estimated_bytes
active_compressed_storage_bytes
```

判断方式：

- `num_demotions > 0`：说明确实有 prefix 被压缩保存；
- `num_compressed_hits > 0`：说明后续请求命中过 compressed prefix；
- `num_restore_success` 越高越好；
- `num_restore_fallback` 越低越好；
- `num_demote_rejected_pool_full` 很高，说明 compressed pool 太小；
- `active_storage_compression_ratio > 1` 才说明压缩存储比原 fp16/bf16 更省；
- `compressed_pool_utilization` 接近 1 时，说明 pool 接近满，需要考虑增大 pool 或优化 eviction。

### 18.8 正确性判断

ZipCache v2 是有损 KV 量化，所以输出不一定逐 token 完全等同 main。正确性建议分三层看：

1. **服务正确性**

   ZipCache v2 服务不能崩溃，不能 OOM，不能出现 restore 后 page table 错乱。

2. **输出质量**

   `correctness` workload 下，用 `--show-text` 对比 main 和 ZipCache v2 输出是否语义一致。

3. **GSM8K 数字答案正确率**

   `gsm8k_correctness` workload 来自：

   ```text
   ZipCache/ZipCache/asset/gsm8k_sample.txt
   ```

   生成后的数据文件是：

   ```text
   experiment/data/gsm8k_correctness.jsonl
   ```

   每条数据包含 `prompt` 和标准数字答案 `answer`。一键脚本会自动调用 `experiment/evaluate_correctness.py`，从模型输出中抽取最终数字答案，并生成：

   ```text
   gsm8k_correctness_eval.json
   ```

   判分规则：

   ```text
   1. 优先匹配 "The answer is <number>"
   2. 如果没有该格式，则取输出文本中的最后一个数字
   3. 与数据集 answer 字段精确比较
   4. 输出 num_judged、num_correct、accuracy
   ```

   对比 main 和 ZipCache v2 时，重点看两者的 `accuracy` 差值。如果 ZipCache v2 的 `num_restore_success > 0`，但 GSM8K accuracy 明显低于 main，说明压缩/恢复可能影响了模型推理质量，需要进一步降低量化误差或限制 demote 策略。

如果输出差异明显，可以尝试：

```bash
--zipcache-unimportant-ratio 0.2
--zipcache-k-unimportant-bit 4
--zipcache-v-unimportant-bit 4
```

这样会减少低 bit token 或直接让所有 token 逻辑上按 4bit 量化，用于区分“量化误差”与“cache/restore 实现错误”。

### 18.9 实验结论应如何写

建议最终报告至少包含：

```text
模型名称和路径
GPU 型号
分支和 commit id
启动参数
workload 名称
main 的 report.md
ZipCache v2 的 report.md
ZipCache v2 stats
显存峰值对比
TTFT / E2E / RPS 对比
输出正确性观察
```

如果 ZipCache v2 出现以下现象：

```text
gpu_memory_used_mb_max 没下降
RPS 下降明显
TTFT/E2E 增大
num_compressed_hits 很少
compressed_pool_utilization 很低
```

通常说明 workload 没有产生足够 shared-prefix 复用，或者 demote/restore 的 Python 开销大于节省的 prefill/recompute 开销。

## 19. 当前公开数据集测试入口

当前 `experiment/` 测试套件已从旧的 `experiment/data/*.jsonl` 切换到公开数据集派生 workload：

```text
experiment/workloads/
```

默认一键脚本会运行：

```text
gsm8k_public_correctness
cmmlu_public_correctness
longbench_public_qa
longbench_long_context_pressure
public_shared_prefix
public_shared_prefix_serial
ruler_squad_qa
synthetic_shared_prefix
```

重新生成 workload：

```bash
python experiment/prepare_public_workloads.py \
  --root experiment \
  --output-dir experiment/workloads
```

### 19.1 v2 推荐测试命令

启动 ZipCache v2 后执行：

```bash
python experiment/run_all_experiments.py \
  --mode zipcache_v2 \
  --base-url http://127.0.0.1:30001 \
  --server-log zipcache_v2_server.log \
  --log-root experiment/logs \
  --gpu-sample-interval 0.5
```

如果只想观察 restore 命中：

```bash
python experiment/run_all_experiments.py \
  --mode zipcache_v2_restore \
  --base-url http://127.0.0.1:30001 \
  --server-log zipcache_v2_server.log \
  --only public_shared_prefix_serial,synthetic_shared_prefix \
  --gpu-sample-interval 0.5
```

### 19.2 v2 应重点比较的指标

正确性：

```text
gsm8k_public_correctness_eval.json
cmmlu_public_correctness_eval.json
longbench_public_qa_eval.json
ruler_squad_qa_eval.json
```

性能：

```text
TTFT
E2E
TPOT
RPS
chunks/s
gpu_memory_used_mb_max
```

ZipCache 内部统计：

```text
num_demotions
num_compressed_entries
num_compressed_hits
num_restore_attempts
num_restore_success
num_restore_fallback
compressed_pool_utilization
active_storage_compression_ratio
```

v2 如果 `num_compressed_hits == 0`，说明 workload 只触发了 demote，没有真正复用压缩 cache。此时应优先看 `public_shared_prefix_serial` 和 `synthetic_shared_prefix`，并确认服务端日志中存在 `[ZipCacheV2] demoted` 和 `[ZipCacheV2] restored`。
