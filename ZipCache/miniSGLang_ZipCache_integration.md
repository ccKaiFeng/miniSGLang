# 将 ZipCache 部署到 miniSGLang 的工程实现说明

本文档说明如何根据本目录下 ZipCache 论文和代码，把 ZipCache 的 KV cache 混合精度压缩思想迁移到 `miniSGLang`。重点不是复述论文，而是回答工程实现中最关键的问题：

- ZipCache 原代码做了什么；
- miniSGLang 当前 KV cache 数据通路是什么；
- 两者结构差异在哪里；
- 需要新增或修改 miniSGLang 的哪些模块、哪些文件；
- 第一版如何做成可运行原型；
- 第二版如何逐步接近论文效果；
- 哪些地方不能直接照搬 ZipCache 代码。

## 1. 总体结论

ZipCache 原仓库是基于 HuggingFace `generate()` / `past_key_values` 机制实现的。它在每层 attention 内部把历史 KV 从普通 tensor 换成压缩对象：

```text
普通 HuggingFace cache:
past_key_values[layer] = (key_tensor, value_tensor)

ZipCache cache:
past_key_values[layer] = (key_compress_union, value_compress_union)
```

但是 miniSGLang 不是这种结构。miniSGLang 使用运行时统一分配的 paged KV cache pool：

```text
MHAKVCache._kv_buffer:
[K/V, layer, page, page_size, local_kv_heads, head_dim]

page_table[req.table_idx, token_pos] = 物理 KV token index
```

attention backend 每层 forward 时会：

1. 从模型层得到本轮新 token 的 `q/k/v`；
2. 调用 `kv_cache.store_kv(k, v, batch.out_loc, layer_id)` 写入全局 KV pool；
3. 把 `k_cache(layer_id)`、`v_cache(layer_id)`、`page_table` 交给 FlashAttention / FlashInfer / TensorRT-LLM kernel；
4. kernel 根据 page table 读取历史 KV。

所以，ZipCache 不能直接把 `MyLlamaForCausalLM` / `MixedLlamaAttention` 搬进 miniSGLang。正确的工程方向是：

1. 复用 ZipCache 的“重要 token 识别”和“量化/解压工具”；
2. 在 miniSGLang 的 paged KV cache 层增加一个压缩管理模块；
3. 在 attention backend 调用 kernel 前，保证被 kernel 读取的 GPU KV pool 中仍然是普通 fp16/bf16 KV；
4. 若历史 KV 已经被压缩，需要先解压回 GPU KV pool，再走原始 attention 路径；
5. 第一版建议先做“压缩归档 + 命中统计 + 可选恢复前回填”，不要改 CUDA kernel 和 attention 数学逻辑。

## 2. ZipCache 原代码核心

### 2.1 主要文件

| 文件 | 作用 |
| --- | --- |
| `zipcache/models/modeling_llama.py` | 改写 HuggingFace LLaMA。核心类是 `MixedLlamaAttention`、`MyLlamaModel`、`MyLlamaForCausalLM`。 |
| `zipcache/models/modeling_mistral.py` | 改写 HuggingFace Mistral，思路与 LLaMA 类似。 |
| `zipcache/models/CompressUtils/compress_class.py` | 定义压缩 cache 对象：`CompressUnion`、`MixedPrecisionCompressUnion`。 |
| `zipcache/models/CompressUtils/compress_function.py` | 底层量化/反量化函数，包括 token-wise、channel-wise、mixed precision、GEAR、outlier 等。 |
| `zipcache_generation_demo.py` | 演示如何构造 `compress_config` 并加载 `MyLlamaForCausalLM`。 |

### 2.2 ZipCache 在 HuggingFace 中的数据流

以 `MixedLlamaAttention.forward()` 为核心：

1. `hidden_states` 经过 `q_proj/k_proj/v_proj` 得到 Q/K/V。
2. 如果 `past_key_value` 不为空：
   - 普通模式：历史 KV 是 tensor，直接 concat；
   - ZipCache 模式：历史 KV 是 `MixedPrecisionCompressUnion`，先 `decompress()` 再 concat。
3. prefill 阶段 `q_len > 1`：
   - 抽取最近 5% token + 随机 5% token 作为 probe query；
   - 计算 probe attention；
   - 对每个历史 token 的 attention 权重求和并归一化；
   - attention 较小的 token 记为 `unimportant_ids_k` / `unimportant_ids_v`。
4. decode 阶段 `q_len == 1`：
   - 每隔 `streaming_gap` 个 token 更新一次 unimportant token 集合。
5. 真实 attention 输出仍由 FlashAttention 计算。
6. 如果 `use_cache=True`：
   - Key 用 `mixed_channelwiseQ` 等方式压缩；
   - Value 可用 `channel_separate_mixed_tokenwiseQ` 等方式压缩；
   - 返回新的压缩 `past_key_value`。

### 2.3 可复用部分

可以复用：

- `compress_function.py` 中部分纯 PyTorch 量化/解压函数；
- `MixedPrecisionCompressUnion` 的配置思想；
- prefill probe 识别 salient/unimportant token 的思路；
- key/value 使用不同压缩模式和 bit 数的配置方式。

不建议直接复用：

- `MyLlamaForCausalLM`、`MyLlamaModel`、`MixedLlamaAttention` 的整体模型类；
- HuggingFace `past_key_values` 数据结构；
- 每步都 `decompress + concat + compress` 的实现方式。

原因是 miniSGLang 已经有自己的模型、scheduler、page table、KV pool 和 attention backend。

## 3. miniSGLang 当前 KV cache 数据通路

### 3.1 核心运行路径

miniSGLang 一次请求大致走：

```text
server/tokenizer
  -> Scheduler._process_one_msg()
  -> PrefillManager.add_one_req()
  -> PrefillManager.schedule_next_batch()
  -> Scheduler._prepare_batch()
  -> Engine.forward_batch()
  -> model.forward()
  -> AttentionLayer.forward()
  -> attn_backend.forward()
  -> kv_cache.store_kv()
  -> attention kernel reads k_cache/v_cache by page_table
  -> sampler
  -> Scheduler._process_last_data()
```

### 3.2 KV cache 相关文件

| 文件 | 关键类/函数 | 作用 |
| --- | --- | --- |
| `python/minisgl/kvcache/base.py` | `BaseKVCachePool`、`BasePrefixCache` | 定义 KV pool 和 prefix cache 抽象接口。 |
| `python/minisgl/kvcache/mha_pool.py` | `MHAKVCache` | 真正分配大块 GPU KV tensor，提供 `k_cache()`、`v_cache()`、`store_kv()`。 |
| `python/minisgl/scheduler/cache.py` | `CacheManager` | 管理 free pages、prefix cache、page_table 写入、cache 插入和释放。 |
| `python/minisgl/kvcache/radix_cache.py` | `RadixPrefixCache` | radix tree prefix cache，保存 token 前缀到物理 KV indices 的映射。 |
| `python/minisgl/kvcache/naive_cache.py` | `NaivePrefixCache` | 不做 prefix 命中的 cache 实现。 |
| `python/minisgl/scheduler/prefill.py` | `PrefillAdder`、`PrefillManager` | 新请求进入 prefill 前会调用 `cache_manager.match_req()` 查询 prefix cache。 |
| `python/minisgl/scheduler/scheduler.py` | `Scheduler._prepare_batch()`、`_process_last_data()`、`_free_req_resources()` | 分配 KV page、准备 batch、请求结束后释放/插入 cache。 |

### 3.3 Attention backend 相关文件

| 文件 | 关键函数 | 作用 |
| --- | --- | --- |
| `python/minisgl/layers/attention.py` | `AttentionLayer.forward()` | 拆 Q/K/V，做 RoPE，然后调用 `ctx.attn_backend.forward()`。 |
| `python/minisgl/attention/fa.py` | `FlashAttentionBackend.forward()` | 写入 KV pool 后调用 `flash_attn_with_kvcache()`。 |
| `python/minisgl/attention/fi.py` | `FlashInferBackend.forward()` | 写入 KV pool 后调用 FlashInfer paged KV wrapper。 |
| `python/minisgl/attention/trtllm.py` | `TensorRTLLMBackend.forward()` | 写入 KV pool 后调用 TensorRT-LLM/FlashInfer 接口。 |

三个 backend 都假设 `k_cache(layer_id)` / `v_cache(layer_id)` 是普通 fp16/bf16 GPU tensor。因此第一版不要让 kernel 直接读取 int2/int4/int8 压缩 KV。

## 4. 两个项目的关键结构差异

### 4.1 HuggingFace cache 是“每请求每层返回对象”

ZipCache 原实现中，cache 随 `forward()` 返回：

```text
forward(..., past_key_values=old_cache)
  -> return logits, new_past_key_values
```

每个请求可以持有自己的 `past_key_values`。

### 4.2 miniSGLang cache 是“全局 paged GPU pool”

miniSGLang 中，请求对象 `Req` 不保存每层 KV tensor。它只保存：

- `table_idx`：在 `page_table/token_pool` 中的请求槽位；
- `cached_len`：已经可复用的 token 数；
- `cache_handle`：prefix cache 命中的 handle。

真实 KV 在 `MHAKVCache._kv_buffer` 里，所有请求共享。请求通过 `page_table` 记录“逻辑 token 位置 -> 物理 KV 位置”。

### 4.3 迁移时必须遵守的原则

1. 不改 CUDA kernel。
2. 不改 attention 数学公式。
3. 不让 kernel 读压缩格式。
4. 压缩后的 KV 要复用，必须在 kernel 执行前解压回原始 GPU KV pool 布局。
5. feature flag 关闭时原行为完全不变。
6. restore 失败必须 fallback 到 recompute/prefill。

## 5. 推荐实现路线

建议分三阶段实现。

### 阶段 A：可运行原型，压缩归档但不改变推理结果

目标：

- 增加 ZipCache feature flag；
- 在 prefix cache 驱逐或请求结束释放 KV 前，把对应 KV metadata 记录下来；
- 可选把 KV tensor 拷到 CPU 并压缩保存；
- 命中 compressed archive 时只打印日志，fallback recompute；
- 不改变 attention backend 和 kernel。

优点：

- 工程风险最低；
- 可以验证驱逐路径、命中路径、统计日志；
- feature flag 关闭时很容易保证完全不变。

缺点：

- 还不能真正减少 GPU KV cache 峰值之外的 recompute 开销；
- ZipCache 论文中的 mixed precision 解压复用还没有进入 attention 路径。

### 阶段 B：压缩后可恢复到 GPU KV pool

目标：

- 在 prefix cache 命中 compressed entry 后，分配新的 GPU KV pages；
- 解压 archived KV；
- 把恢复出的 K/V 写回 `MHAKVCache._kv_buffer` 对应物理位置；
- 更新 `page_table`，让后续 attention kernel 仍然从普通 GPU KV pool 读取；
- restore 失败时 fallback 到 recompute。

这是最适合 miniSGLang 的 ZipCache 迁移方式。它仍然不改 attention kernel。

### 阶段 C：真正按 ZipCache 论文做 mixed precision salient token

目标：

- 在 prefill 阶段统计 attention 分数；
- 按 token 重要性选择不同 bit 数；
- 对 K/V 使用不同压缩模式；
- 支持 streaming decode 中周期性更新 token 重要性。

这一阶段涉及 attention score 获取。miniSGLang 当前 backend 调用的高性能 kernel 默认不返回完整 attention weights。因此有两种实现选择：

1. 第一版只用启发式重要性，例如最近 token 保高精度、较早 token 低精度；
2. 在 prefill 时额外做一次小规模 probe attention，仅用于统计重要性，不影响真实 attention 输出。

第二种更接近论文，但需要额外算子和显存，需要单独评估开销。

## 6. 需要新增的模块

### 6.1 新增 `python/minisgl/zipcache/`

建议新增目录：

```text
python/minisgl/zipcache/
  __init__.py
  config.py
  manager.py
  archive.py
  quant.py
  saliency.py
```

也可以先做成一个文件 `python/minisgl/zipcache/manager.py`，后续再拆分。

### 6.2 `config.py`

定义 ZipCache 配置数据结构，例如：

```python
@dataclass(frozen=True)
class ZipCacheConfig:
    enabled: bool = False
    archive_dir: str = "/root/autodl-tmp/zipcache_archive"
    codec: str = "mock"
    max_size_mb: int = 4096
    restore_policy: str = "cost"
    k_compress_mode: str = "mixed_channelwiseQ"
    v_compress_mode: str = "channel_separate_mixed_tokenwiseQ"
    k_quantize_bit_important: int = 4
    k_quantize_bit_unimportant: int = 2
    v_quantize_bit_important: int = 4
    v_quantize_bit_unimportant: int = 2
    k_unimportant_ratio: float = 0.4
    v_unimportant_ratio: float = 0.4
    streaming_gap: int = 100
```

### 6.3 `archive.py`

负责 compressed entry 的 metadata 和磁盘文件：

```python
@dataclass
class ZipCacheEntryMeta:
    entry_id: str
    token_ids_hash: str
    token_len: int
    page_indices: List[int]
    num_layers: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    dtype: str
    codec: str
    state: str
    created_time: float
    last_access_time: float
    hit_count: int
    original_estimated_bytes: int
    compressed_estimated_bytes: int
    metadata_path: str
    tensor_path: str | None
```

第一版可以每个 entry 保存一个 JSON：

```text
archive_dir/
  entry_xxx.json
  entry_xxx.pt      # codec 非 mock 时可选
```

### 6.4 `quant.py`

迁移或包装 ZipCache 的量化函数。

第一版建议不要完整复制 `compress_function.py` 的所有函数，而是先实现最小可验证 codec：

- `mock`：只保存 metadata；
- `int8_cpu`：对选中的 K/V tensor 做简单对称 int8 量化；
- 后续再接入 ZipCache 原始 `mixed_channelwiseQ`、`channel_separate_mixed_tokenwiseQ`。

示例接口：

```python
def quantize_int8_cpu(x: torch.Tensor) -> dict:
    scale = x.abs().amax().clamp_min(1e-8) / 127
    q = torch.round(x / scale).clamp(-128, 127).to(torch.int8).cpu()
    return {
        "q": q,
        "scale": scale.cpu(),
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
    }

def dequantize_int8_cpu(obj: dict, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return (obj["q"].to(device).float() * obj["scale"].to(device)).to(dtype)
```

### 6.5 `saliency.py`

负责 token 重要性识别。

第一版可以先不依赖 attention weights，使用启发式：

- 最近 `N` 个 token 视为 important；
- 其余 token 按比例视为 unimportant；
- 或者 mock 模式只记录 `unimportant_ids=None`。

后续如果要更接近论文，再实现 probe attention：

```text
probe query = 最近 5% token + 随机 5% token
score = softmax(Q_probe K^T) 后按 token 求和
score 小的 token -> unimportant
```

注意：在 miniSGLang 中做 probe attention 时，要避免改变真实 attention 输出。probe 只用于统计，不参与生成。

### 6.6 `manager.py`

核心管理类，建议命名：

```python
class ZipCacheManager:
    def __init__(self, config, kv_pool, page_table, logger):
        ...

    def enabled(self) -> bool:
        ...

    def demote(self, *, token_ids, indices, req_uid=None, source="evict") -> bool:
        ...

    def maybe_restore(self, *, token_ids, table_idx, cache_manager) -> RestoreResult:
        ...

    def restore_to_pages(self, entry_meta, allocated_indices) -> bool:
        ...

    def should_restore(self, meta, recompute_tokens: int) -> bool:
        ...

    def stats(self) -> dict:
        ...
```

`demote()` 的职责：

1. 根据 `token_ids` 计算 hash；
2. 根据 `indices` 找到原 GPU KV pool 中的物理 token；
3. 记录 metadata；
4. codec 为 `mock` 时只写 JSON；
5. codec 为 `int8_cpu` 或 ZipCache codec 时，从每层 `k_cache/v_cache` gather 对应 token，压缩后保存到 CPU/disk；
6. demote 成功后仍然让原来的 GPU page 被释放。

`maybe_restore()` 的职责：

1. 新请求 prefill 前，根据输入 token 前缀查 compressed archive；
2. 如果没命中，返回 miss；
3. 如果命中，根据 `restore_policy` 判断是否值得恢复；
4. 如果第一版是 mock，打印 hit 日志后返回 fallback；
5. 如果实现了真实恢复，分配新的 pages，调用 `restore_to_pages()` 写回 GPU KV pool，并更新 page table。

## 7. 需要修改的 miniSGLang 文件

### 7.1 `python/minisgl/engine/config.py`

新增配置字段。建议字段：

```python
zipcache_enabled: bool = False
zipcache_archive_dir: str = "/root/autodl-tmp/zipcache_archive"
zipcache_codec: str = "mock"
zipcache_max_size_mb: int = 4096
zipcache_restore_policy: str = "cost"
zipcache_k_quantize_bit_important: int = 4
zipcache_k_quantize_bit_unimportant: int = 2
zipcache_v_quantize_bit_important: int = 4
zipcache_v_quantize_bit_unimportant: int = 2
zipcache_k_unimportant_ratio: float = 0.4
zipcache_v_unimportant_ratio: float = 0.4
zipcache_streaming_gap: int = 100
```

影响：

- 这些字段会被 `ServerArgs` 继承；
- `Engine`、`Scheduler` 都能拿到配置。

### 7.2 `python/minisgl/server/args.py`

新增命令行参数：

```text
--enable-zipcache
--zipcache-archive-dir
--zipcache-codec {mock,int8_cpu,zipcache_mixed}
--zipcache-max-size-mb
--zipcache-restore-policy {never,cost,always}
--zipcache-k-important-bit
--zipcache-k-unimportant-bit
--zipcache-v-important-bit
--zipcache-v-unimportant-bit
--zipcache-k-unimportant-ratio
--zipcache-v-unimportant-ratio
--zipcache-streaming-gap
```

第一版建议默认关闭：

```text
zipcache_enabled = False
zipcache_codec = "mock"
zipcache_restore_policy = "never"
```

这样 feature flag 关闭时，miniSGLang 原行为完全不变。

### 7.3 `python/minisgl/engine/engine.py`

需要在 `Engine.__init__()` 中创建 ZipCache manager 或把必要资源暴露给 Scheduler。

当前 Engine 初始化顺序中，KV pool 创建位置是：

```python
self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(...)
self.ctx.page_table = self.page_table = torch.zeros(...)
```

建议修改：

```python
if config.zipcache_enabled:
    from minisgl.zipcache import ZipCacheManager
    self.zipcache_manager = ZipCacheManager(
        config=config,
        kv_pool=self.kv_cache,
        page_table=self.page_table,
        logger=logger,
    )
else:
    self.zipcache_manager = None
```

注意初始化顺序：`page_table` 创建后才能传给 manager。如果 manager 只需要 `kv_pool`，也可以提前创建。

影响：

- Engine 持有真实 KV pool，所以真实压缩/解压最好通过 Engine 的 manager 操作；
- Scheduler 负责知道什么时候释放/驱逐，因此 Scheduler 需要能访问 `self.engine.zipcache_manager`。

### 7.4 `python/minisgl/scheduler/cache.py`

这是最重要的接入点之一。

当前 `CacheManager` 负责：

- `match_req()`：prefix cache 查询；
- `allocate_paged()`：分配 page；
- `cache_req()`：把请求插入 prefix cache 并释放多余 page；
- `_allocate()`：free page 不够时调用 `prefix_cache.evict()`；
- `_free()`：释放 page。

建议修改 `CacheManager.__init__()`：

```python
def __init__(..., zipcache_manager=None):
    self.zipcache_manager = zipcache_manager
```

#### 7.4.1 在 `_allocate()` 中接入 eviction demote

当前逻辑：

```python
if needed_pages > len(self.free_slots):
    evicted = self.prefix_cache.evict(...)
    self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
```

建议改成：

```python
if needed_pages > free_pages:
    evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
    if self.zipcache_manager is not None and self.zipcache_manager.enabled():
        self.zipcache_manager.demote_evicted_indices(evicted)
    self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
```

但这里有一个问题：`evict()` 当前只返回 `indices`，没有返回 token ids。ZipCache archive 要做 prefix 命中，必须知道被驱逐 indices 对应的 token 前缀。解决办法见 7.5。

#### 7.4.2 在 `cache_req()` 中接入 request-level demote

`cache_req(req, finished=True)` 是请求结束时释放尾部 page 的路径：

```python
if finished:
    self._free(page_indices[new_handle.cached_len :])
```

可在 `_free()` 前调用：

```python
if finished and self.zipcache_manager.enabled():
    self.zipcache_manager.demote_request_tail(
        req_uid=req.uid,
        token_ids=req.input_ids[new_handle.cached_len:req.cached_len],
        indices=page_indices[new_handle.cached_len:req.cached_len],
    )
```

第一版可以只在请求结束时 demote 整个 prompt 或尾部，便于验证，不必马上覆盖 radix evict 路径。

#### 7.4.3 在 `match_req()` 中接入 compressed archive 查询

当前：

```python
return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])
```

建议：

```python
match = self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])
if self.zipcache_manager.enabled():
    self.zipcache_manager.note_lookup(req.input_ids[: input_len - 1], match)
return match
```

第一版只打印 compressed hit，不改变 `match.cached_len`。

第二版真实 restore 时，需要在 `match_req()` 或 `PrefillAdder._try_allocate_one()` 之间插入：

1. radix 没命中或命中较短；
2. compressed archive 命中更长前缀；
3. 分配 pages；
4. 解压写回 KV pool；
5. 返回一个新的 `BaseCacheHandle`，让后续流程认为 prefix cache 已命中。

这一步较复杂，建议新建一种 `RestoredCacheHandle` 或把 restored entry 插入 radix cache 后返回正常 handle。

### 7.5 `python/minisgl/kvcache/radix_cache.py`

如果要在 prefix cache eviction 时知道“被驱逐的是哪个 token 前缀”，当前 `evict()` 返回值不够。

当前：

```python
def evict(self, size: int) -> torch.Tensor:
    ...
    evicted_indices.append(node.value)
    ...
    return torch.cat(evicted_indices)
```

建议新增非破坏性接口：

```python
@dataclass
class EvictedPrefixEntry:
    token_ids: torch.Tensor
    indices: torch.Tensor
    node_uuid: int

def evict_with_entries(self, size: int) -> tuple[torch.Tensor, list[EvictedPrefixEntry]]:
    ...
```

或者修改 `evict()` 返回更丰富结构，但这会影响 `CacheManager._allocate()`，风险更大。

推荐做法：

- 保留原 `evict()` 接口；
- 新增 `evict_with_entries()`；
- `CacheManager` 在 ZipCache 开启时调用新接口，关闭时仍调用旧接口。

为了得到完整 token 前缀，需要从 radix node 回溯到 root 拼接 key：

```python
def collect_node_token_ids(node):
    keys = []
    while not node.is_root():
        keys.append(node._key)
        node = node.parent
    keys.reverse()
    return torch.cat(keys)
```

注意：

- node 的 `value` 只保存本节点这段 indices；
- 如果 demote 整个前缀，需要沿父节点拼接所有 value；
- 如果只 demote 被 evict 的叶子段，则 token hash 只能匹配这段，不一定能作为完整 prefix restore。

工程上更推荐 demote 完整路径，因为 restore 时更自然。

### 7.6 `python/minisgl/scheduler/prefill.py`

`PrefillAdder._try_allocate_one()` 是新请求进入 prefill 前的 prefix cache 命中点：

```python
handle = self.cache_manager.match_req(req).cuda_handle
cached_len = handle.cached_len
```

第二版 restore 应在这里完成或由 `match_req()` 完成。

推荐接口：

```python
handle = self.cache_manager.match_req(req).cuda_handle
if self.cache_manager.zipcache_manager.enabled():
    handle = self.cache_manager.try_restore_compressed_prefix(req, handle)
cached_len = handle.cached_len
```

`try_restore_compressed_prefix()` 要保证：

- 如果 restore 失败，返回原 handle；
- 如果 restore 成功，返回 restored handle；
- 不改变后续 `PrefillAdder` 的控制流。

如果 restore 成功，还需要像普通 prefix hit 一样复制：

```python
device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
page_entry.copy_(handle.get_matched_indices())
```

所以 restored handle 必须能返回 `get_matched_indices()`。

### 7.7 `python/minisgl/kvcache/base.py`

可能需要新增抽象或 dataclass：

```python
class RestoredCacheHandle(BaseCacheHandle):
    indices: torch.Tensor
    def get_matched_indices(self) -> torch.Tensor:
        return self.indices
```

但不建议放在 `base.py` 里实现复杂逻辑。可以在 `minisgl/zipcache/manager.py` 中定义，继承 `BaseCacheHandle`。

### 7.8 `python/minisgl/kvcache/mha_pool.py`

第一版 mock 不需要改。

如果实现真实压缩/恢复，需要增加便捷方法：

```python
def gather_kv(self, indices: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    ...

def scatter_kv(self, indices: torch.Tensor, keys: list[torch.Tensor], values: list[torch.Tensor]) -> None:
    ...
```

当前 `k_cache(layer_id)` 返回形状：

```text
[num_pages, page_size, local_kv_heads, head_dim]
```

如果要按 token index gather，需要先 view：

```python
flat_k = self.k_cache(layer_id).view(num_pages * page_size, local_kv_heads, head_dim)
selected_k = flat_k[indices]
```

恢复写回：

```python
flat_k[indices] = restored_k
flat_v[indices] = restored_v
```

风险：

- `indices` 必须是物理 token index，不是 page id；
- `indices` 必须和 page_table 使用同一套单位；
- TP 场景下每个 rank 只保存本 rank 的 KV heads，各 rank 都要各自 demote/restore；
- restore 必须在 attention backend prepare/use metadata 前完成。

### 7.9 `python/minisgl/attention/fa.py`

第一版不建议修改。

第二版如果 restore 在 scheduler 阶段完成，也不需要修改 `fa.py`。只要 kernel 调用前 GPU KV pool 已经恢复，FlashAttention backend 可以保持原样。

如果要在 backend 内部处理 compressed page，风险很高，不推荐第一阶段做。

### 7.10 `python/minisgl/attention/fi.py`

第一版不建议修改。

FlashInfer backend 对 page size 和 page table 格式更敏感：

- 当前 `FIMetadata.page_size` 标注为 `Literal[1]`；
- `indices=torch.cat([page_table[req.table_idx, : req.device_len] ...])`；
- kernel 接收 flatten 后的 paged KV cache。

如果 restore 回普通 GPU KV pool，FlashInfer 可以继续不改。

### 7.11 `python/minisgl/attention/trtllm.py`

同 `fa.py`，第一版和第二版都建议不改。只要 restore 发生在 kernel 前，TensorRT-LLM backend 仍读取普通 KV pool。

### 7.12 `python/minisgl/layers/attention.py`

如果阶段 C 要实现 probe attention，可以考虑在这里或 `RopeAttn` 后面拿到当前层 Q/K/V。

当前 `AttentionLayer.forward()`：

```python
q, k, v = qkv.split(...)
q, k = self.rotary.forward(ctx.batch.positions, q, k)
q = q.view(...)
o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
```

这里是最容易拿到新 token Q/K/V 的位置。但要注意：

- 历史 KV 在 `ctx.kv_cache` 中，不在本函数局部变量里；
- probe attention 需要从 KV pool gather 历史 K；
- 不能影响真实 `q/k/v`；
- 额外统计会带来显存和时间开销。

建议阶段 C 新增：

```python
if ctx.zipcache_manager and ctx.zipcache_manager.should_collect_saliency(batch):
    ctx.zipcache_manager.collect_layer_saliency(...)
```

这需要 `Context` 增加 `zipcache_manager` 字段。

### 7.13 `python/minisgl/core.py`

可选修改 `Context`：

```python
zipcache_manager: ZipCacheManager | None = None
```

如果 ZipCache 只在 Scheduler/CacheManager 层工作，不需要改 `Context`。

如果阶段 C 要在 layer attention 内收集 saliency，则需要把 manager 放进 Context。

### 7.14 `python/minisgl/scheduler/scheduler.py`

需要把 `engine.zipcache_manager` 传给 `CacheManager`：

当前：

```python
self.cache_manager = CacheManager(
    self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
)
```

建议：

```python
self.cache_manager = CacheManager(
    self.engine.num_pages,
    config.page_size,
    self.engine.page_table,
    config.cache_type,
    zipcache_manager=self.engine.zipcache_manager,
)
```

请求结束释放路径：

```python
def _free_req_resources(self, req: Req) -> None:
    self.table_manager.free(req.table_idx)
    self.cache_manager.cache_req(req, finished=True)
```

ZipCache demote 最好放在 `CacheManager.cache_req()` 内，而不是这里。原因是 `CacheManager` 更清楚哪些 page 会释放、哪些 page 已插入 prefix cache。

### 7.15 `python/minisgl/engine/graph.py`

如果启用真实 restore，需要关注 CUDA Graph。

风险：

- CUDA Graph replay 假设 metadata buffer 和 page table shape 固定；
- 但 page_table 内容可以变，正常 decode 本来就会变；
- restore 如果只是在 scheduler prepare 阶段写 KV pool 和 page_table，不破坏 graph shape，理论上可兼容。

第一版建议：

- mock 模式不影响 CUDA Graph；
- 真实 restore 初期可以要求 `--cuda-graph-max-bs 0` 或禁用 graph 验证；
- 稳定后再打开 graph 测试。

## 8. 第一版详细实现方案

第一版目标：`--enable-zipcache` 开启后，能在 KV 释放/驱逐时保存 metadata，命中相同 prefix 时打印 hit，并且不影响推理结果。

### 8.1 新增命令行

修改：

- `python/minisgl/engine/config.py`
- `python/minisgl/server/args.py`

新增参数：

```text
--enable-zipcache
--zipcache-archive-dir /root/autodl-tmp/zipcache_archive
--zipcache-codec mock
--zipcache-restore-policy never
```

### 8.2 新增 manager

新增：

```text
python/minisgl/zipcache/__init__.py
python/minisgl/zipcache/manager.py
```

第一版 manager 需要维护统计：

```python
num_evictions
num_demotions
num_mock_demotions
num_int8_demotions
num_compressed_entries
compressed_bytes
original_estimated_bytes
num_compressed_hits
num_restore_attempts
num_restore_success
num_restore_fallback
saved_prefill_tokens_estimated
```

metadata 至少包含：

```json
{
  "entry_id": "...",
  "request_id": "...",
  "token_ids_hash": "...",
  "num_tokens": 128,
  "num_layers": 24,
  "dtype": "torch.float16",
  "codec": "mock",
  "state": "COLD",
  "created_time": 123.0,
  "last_access_time": 123.0,
  "hit_count": 0,
  "original_estimated_bytes": 123456,
  "compressed_estimated_bytes": 1024,
  "storage_path": "..."
}
```

### 8.3 request-level demote

先不要改 radix eviction，优先接请求结束路径。

修改 `CacheManager.cache_req()`：

```python
if finished:
    tail_indices = page_indices[new_handle.cached_len:]
    tail_token_ids = req.input_ids[new_handle.cached_len:req.cached_len]
    if self.zipcache_manager and self.zipcache_manager.enabled():
        self.zipcache_manager.demote(
            token_ids=tail_token_ids,
            indices=tail_indices,
            req_uid=req.uid,
            source="request_finished_tail",
        )
    self._free(tail_indices)
```

如果想保存完整请求前缀，也可以：

```python
full_indices = page_indices[:req.cached_len]
full_token_ids = req.input_ids[:req.cached_len]
```

但要小心：前缀中一部分可能已被 radix cache 共享，不能因为 demote 就改变其释放逻辑。demote 只读 KV，不阻止原释放/缓存流程。

### 8.4 compressed hit mock

修改 `CacheManager.match_req()`：

```python
match = self.prefix_cache.match_prefix(...)
if self.zipcache_manager and self.zipcache_manager.enabled():
    self.zipcache_manager.lookup(req.input_ids[: input_len - 1])
return match
```

`lookup()` 用最长前缀 hash 或完整 token hash 查 archive。第一版可以只匹配完整 `input_ids[:input_len-1]`，简单可靠。

命中时日志：

```text
[ZipCache] compressed hit: entry_id=..., num_tokens=...
[ZipCache] restore fallback to recompute: entry_id=..., reason=mock codec
```

### 8.5 退出或空闲时打印 stats

可以在：

- `Scheduler.run_when_idle()`；
- `Scheduler.shutdown()`；
- `ZipCacheManager.demote()` 每 N 次；

打印：

```text
[ZipCache] stats: {...}
```

第一版建议在 `run_when_idle()` 打印，便于观察。

## 9. 第二版真实恢复方案

第二版目标：compressed archive 命中后，能解压回 GPU KV pool，然后让请求复用恢复后的 KV。

### 9.1 真实恢复的关键步骤

输入：

- 新请求 `PendingReq.input_ids`；
- compressed entry metadata；
- archive 中保存的压缩 K/V；
- `CacheManager`、`TableManager`、`MHAKVCache`。

流程：

1. 在 `PrefillAdder._try_allocate_one()` 中，先查 radix cache。
2. 如果 radix 命中不足，再查 ZipCache archive。
3. 如果 ZipCache 命中长度更长，并且 `should_restore()` 返回 True：
   - 为命中长度分配 KV pages；
   - 解压每层 K/V；
   - 写回 `MHAKVCache` 对应 physical indices；
   - 构造 `RestoredCacheHandle(cached_len, indices)`；
   - 后续流程把 token ids 和 page table 复制到 table slot。
4. 如果任一步失败：
   - 删除半成品 page 或放回 free list；
   - 返回 radix 原 handle；
   - 正常 recompute。

### 9.2 恢复写回 KV pool

`MHAKVCache` 可新增：

```python
def scatter_layer_kv(self, layer_id, indices, k, v):
    flat_k = self.k_cache(layer_id).view(-1, k.shape[-2], k.shape[-1])
    flat_v = self.v_cache(layer_id).view(-1, v.shape[-2], v.shape[-1])
    flat_k[indices] = k
    flat_v[indices] = v
```

注意 shape：

```text
flat_k[indices]:
[num_tokens, local_kv_heads, head_dim]
```

如果 ZipCache 原量化函数使用 `[B, H, L, C]`，则要转换：

```text
[L, H, C] <-> [1, H, L, C]
```

### 9.3 restore 后是否插入 radix cache

有两种方案。

方案一：只返回 `RestoredCacheHandle`

- 实现简单；
- 当前请求可以用；
- 请求结束后 `cache_req()` 会尝试插入 radix cache。

方案二：restore 成功后立刻插入 radix cache

- 后续请求马上可以通过 radix 命中；
- 但要处理 ref_count、lock/unlock，更容易出错。

建议先用方案一。

## 10. 第三版 salient token 识别方案

ZipCache 论文效果的关键是 salient token identification。miniSGLang 中实现有三种层级。

### 10.1 启发式，不取 attention score

最简单：

- 最近 token 保高精度；
- 早期 token 低精度；
- 或按固定比例采样 unimportant。

优点：容易实现，不影响 kernel。

缺点：不等价于论文方法，效果可能一般。

### 10.2 额外 probe attention

在 `AttentionLayer.forward()` 中，当前层可以拿到新 token 的 Q/K/V。历史 K 在 KV pool 中。

prefill 时：

1. 根据 `batch.positions` 找 probe token；
2. 从当前层 K 和历史 K 组成可统计的 K；
3. 算 `Q_probe @ K^T`；
4. softmax 后得到 token attention score；
5. 选出 unimportant ids；
6. 把 ids 交给 `ZipCacheManager`。

风险：

- prefill 长度大时额外矩阵乘会有明显开销；
- 需要处理 batch 中多个请求的 ragged sequence；
- 要区分本轮新 K 和历史 cached K；
- 需要保证统计不改变真实输出。

### 10.3 修改或扩展 attention backend 返回统计

不建议第一阶段做。当前 FlashAttention/FlashInfer backend 的设计只返回 attention output，不返回 attention weights。强行让 kernel 返回 weights 会违背“不改 kernel”的原则，也会显著增加显存。

## 11. codec 选择建议

### 11.1 `mock`

只写 metadata。

用途：

- 验证释放/驱逐路径；
- 验证 archive 命中；
- 验证 feature flag；
- 不影响推理结果。

### 11.2 `int8_cpu`

把 selected KV 从 GPU 拷到 CPU，做简单 int8 量化。

优点：

- 工程可控；
- 易于恢复；
- 不依赖 ZipCache 原始复杂打包函数。

缺点：

- 不是论文的 2-bit/4-bit mixed precision；
- CPU 拷贝和磁盘 IO 会慢；
- 主要适合作为原型。

### 11.3 `zipcache_mixed`

复用 ZipCache 的 `MixedPrecisionCompressUnion`。

注意事项：

- 原实现假设输入 shape 多为 `[B, H, L, C]`；
- miniSGLang pool gather 出来通常是 `[L, H, C]`；
- 需要在压缩前 `permute/unsqueeze`；
- 解压后再转回 `[L, H, C]`；
- 原实现很多函数默认 CUDA tensor，保存到磁盘/CPU 时要处理 device。

建议在 `int8_cpu` 跑通后再做。

## 12. 日志要求

建议日志前缀统一用 `[ZipCache]`。

释放/驱逐：

```text
[ZipCache] eviction detected: source=radix, entry_id=..., num_tokens=..., estimated_bytes=...
```

demote 成功：

```text
[ZipCache] demoted: entry_id=..., codec=mock, original_bytes=..., compressed_bytes=...
```

命中：

```text
[ZipCache] compressed hit: entry_id=..., num_tokens=...
```

restore：

```text
[ZipCache] restore attempt: entry_id=..., policy=cost
[ZipCache] restore success: entry_id=..., restored_tokens=...
[ZipCache] restore fallback to recompute: entry_id=..., reason=...
```

统计：

```text
[ZipCache] stats: {...}
```

## 13. 启动和验证方法

### 13.1 mock 模式启动

示例：

```bash
python -m minisgl \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 30000 \
  --cache-type radix \
  --enable-zipcache \
  --zipcache-archive-dir /root/autodl-tmp/zipcache_archive \
  --zipcache-codec mock \
  --zipcache-restore-policy never
```

### 13.2 构造 shared-prefix 请求

发送多个相同长 system prompt 或相同长前缀的请求：

```text
system prompt: 这是一段很长且多次复用的背景资料 ...
user prompt A: 问题 1
user prompt B: 问题 2
```

观察：

- radix prefix cache 是否命中；
- ZipCache archive 是否出现 compressed hit；
- mock 模式下是否 fallback recompute。

### 13.3 强制触发 eviction

可以调小：

```text
--num-pages
--memory-ratio
--max-running-requests
--max-prefill-length
```

或构造多个长 prompt，让 radix cache 可驱逐 page 不够。

### 13.4 feature flag 关闭验证

不带 `--enable-zipcache` 启动。

期望：

- 日志中不出现 `[ZipCache]`；
- `CacheManager._allocate()`、`cache_req()` 行为与原始代码一致；
- 推理输出和性能不应出现可归因于 ZipCache 的变化。

## 14. 测试建议

### 14.1 单元测试

新增：

```text
python/tests/test_zipcache_manager.py
```

建议测试：

1. `mock` demote 会生成 JSON；
2. 相同 token ids hash 能 lookup 命中；
3. feature flag off 时 manager 不执行；
4. `int8_cpu` quant/dequant shape 和 dtype 正确；
5. restore 失败会返回 fallback，不抛出影响调度的异常。

### 14.2 集成测试

用 dummy weight 启动：

```bash
python -m minisgl \
  --model-path <small-model-or-local-config> \
  --dummy-weight \
  --num-pages 64 \
  --cache-type radix \
  --enable-zipcache \
  --zipcache-codec mock
```

如果本地缺少 CUDA/torch/sgl_kernel，则至少运行：

```bash
python -m compileall -q python/minisgl
```

### 14.3 正确性测试

同一请求分别用：

1. feature flag off；
2. feature flag on + mock；
3. feature flag on + int8_cpu 但 restore_policy=never；

输出应一致或只受采样随机性影响。

真实 restore 开启后，输出可能因量化误差略有变化，需要用 greedy decode 做对比。

## 15. 主要风险点

### 15.1 page index 和 token index 混淆

miniSGLang 的全局 `page_table` 存的是物理 token index，不是 page id。

但是某些 backend metadata 会把它转换成 page id：

```python
new_page_table.div_(self.page_size, rounding_mode="floor")
```

压缩/恢复时应使用物理 token index。

### 15.2 TP rank 下 KV head 切分

`MHAKVCache` 中每个 TP rank 只保存本 rank 的 `local_kv_heads`。因此：

- 每个 rank 都要独立压缩自己的 KV；
- archive 文件名需要包含 rank；
- restore 时每个 rank 恢复自己的 shard；
- rank0 命中不代表其他 rank 可以跳过 restore。

metadata 建议加入：

```json
"tp_rank": 0,
"tp_size": 2,
"local_kv_heads": 4
```

### 15.3 prefix cache handle 生命周期

radix cache 通过 `lock_handle()` / `unlock()` 管理 ref_count。

restore handle 如果不插入 radix cache，不应参与 radix ref_count。

如果 restore 后插入 radix cache，必须保证：

- 插入后 lock；
- 请求结束后 unlock；
- 被驱逐时不会驱逐正在使用的节点。

### 15.4 CUDA Graph

真实 restore 可能在 graph replay 前写 KV pool。只要不改变 graph shape，理论上可行。但初期建议关闭 CUDA Graph 验证，避免问题混在一起。

### 15.5 性能风险

CPU 压缩/磁盘保存会造成同步和 IO 开销。第一版应尽量：

- 只在 eviction/request finish 路径做；
- 避免在 hot decode path 每 token 压缩；
- 可加后台线程或异步队列，但第一版不建议复杂化。

## 16. 推荐修改顺序

1. 新增 `ZipCacheConfig` 字段和命令行参数。
2. 新增 `ZipCacheManager`，先支持 `mock`。
3. 在 `Engine` 中创建 manager。
4. 在 `Scheduler` 创建 `CacheManager` 时传入 manager。
5. 在 `CacheManager.cache_req(finished=True)` 中接 request-level demote。
6. 在 `CacheManager.match_req()` 中接 compressed lookup，mock hit 后 fallback。
7. 在 `Scheduler.run_when_idle()` 或 `shutdown()` 打印 stats。
8. 验证 feature flag off 完全无日志、无行为变化。
9. 增加 `int8_cpu` codec，只实现 compress-only。
10. 增加 restore 到 GPU KV pool。
11. 再考虑 radix eviction demote 和 ZipCache mixed precision codec。
12. 最后考虑 probe attention salient token 识别。

## 17. 最小改动文件清单

第一版 mock 原型最少需要改：

```text
python/minisgl/engine/config.py
python/minisgl/server/args.py
python/minisgl/engine/engine.py
python/minisgl/scheduler/scheduler.py
python/minisgl/scheduler/cache.py
python/minisgl/zipcache/__init__.py
python/minisgl/zipcache/manager.py
```

第二版真实恢复还需要改：

```text
python/minisgl/kvcache/mha_pool.py
python/minisgl/scheduler/prefill.py
python/minisgl/kvcache/base.py      # 可选，若 RestoredCacheHandle 放这里
```

第三版 radix eviction 和 salient token 识别还需要改：

```text
python/minisgl/kvcache/radix_cache.py
python/minisgl/layers/attention.py  # 可选，若做 probe attention
python/minisgl/core.py              # 可选，若 manager 放入 Context
```

## 18. 建议不要第一版修改的地方

第一版不要修改：

- `python/minisgl/kernel/csrc/` 下 CUDA/C++ kernel；
- `python/minisgl/attention/fa.py` 的 kernel 调用参数；
- `python/minisgl/attention/fi.py` 的 FlashInfer wrapper 逻辑；
- `python/minisgl/attention/trtllm.py` 的 TensorRT-LLM 调用；
- 模型权重加载逻辑；
- tokenizer/detokenizer 协议。

这样能保证 ZipCache feature flag 关闭时原系统不受影响。

## 19. 一句话工程方案

在 miniSGLang 中部署 ZipCache，不应替换模型类，也不应让 attention kernel 直接读压缩 KV；应新增一个运行时 `ZipCacheManager`，在 KV page 被释放/驱逐前把普通 GPU KV 归档压缩，在未来 prefix 命中时先解压写回原 paged KV pool，然后继续走 miniSGLang 原来的 attention backend。第一版先做 mock archive 和日志统计，第二版再做 int8/ZipCache mixed precision 的真实恢复。
