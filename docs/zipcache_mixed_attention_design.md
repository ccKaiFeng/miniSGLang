# ZipCache 混合 KV Attention 算子设计

本文是设计方案，不是已实现功能或性能报告。基于本仓库提交 `37baefd5a9ec43c6d1248e5f7823d62d9d245a76` 的源码核对，日期为 2026-09-06。

实现状态：仓库现在包含本文输入协议、correctness-first CUDA kernel、FI/FA
混合分派、GPU 正确性测试和微基准。Tensor Core 分块、split-KV、CUDA Graph
以及 Scheduler 绕过 `materialize_match()` 尚未实现。当前开发机没有 CUDA 环境，
新增 CUDA 代码必须在 NVIDIA GPU 服务器上完成 JIT 编译和数值验证后才能视为可用。

## 1. 目标与范围

目标是在一次 Attention 运算中同时读取 normal pool 的 FP16/BF16 KV，以及 compressed pool 的 packed 2bit/4bit KV。压缩数据在 GPU 片上分块反量化后立即参与计算，不先恢复完整 KV 到 normal pool。

```text
normal pool 的 FP16/BF16 KV ----------> normal tile loader ----+
                                                            |
compressed pool 的 packed KV --> 解包、反量化 tile loader ----+--> QK、统一 softmax、PV --> O
```

这里的“直接”是不写出完整的反量化 KV 中间张量，并不意味着直接拿 uint8 字节做普通浮点 Attention。存储精度与计算精度不同：2bit/4bit 是存储格式，矩阵计算仍使用 FP16/BF16，累加主要使用 FP32。

本阶段只涉及算子、算子输入准备、`fi.py`/`fa.py` 的分派和独立测试。不修改 radix 匹配、请求调度、归档策略或 pool 分配算法。

必须明确一个输入前提：调用方提供每个请求实际使用的 normal 段、compressed entry 引用和逻辑位置，并保证 GPU 执行完成前这些数据有效。当前 runtime 会先 `materialize_match()`，因此仅修改两个 backend 文件不能自动让真实请求绕过 restore。第一阶段由测试代码构造这份输入；完成算子验证不等于完成服务接入。

## 2. 现有源码约束

| 位置 | 当前行为 | 对新算子的约束 |
|---|---|---|
| [manager.py](../python/minisgl/zipcache/manager.py) 的 `_QuantizedPart` | `q` 为 packed uint8，`min/step` 为 FP16，`ids` 为 int64 | 使用现有编码，不重新量化或转换为 FP4 |
| `demote_node()` | 每层独立选择 token 分组，该层 K/V 使用同一组 token ID | 同层可共同遍历 K/V，但不能跨层复用一份分组表 |
| `allocate_part()` | K/V 的 q、min、step 分别分配 slice | 各字段必须独立记录 offset，不能推测它们相邻 |
| [mha_pool.py](../python/minisgl/kvcache/mha_pool.py) | 单层 normal K/V 为 `[P,S,Hkv,D]` | 可零拷贝 view 为 `[P*S,Hkv,D]` |
| [fi.py](../python/minisgl/attention/fi.py) | 整条请求 KV 索引送给 FlashInfer，逻辑 page size 为 1 | 混合路径不能把 compressed offset 塞入原 `indices` |
| [fa.py](../python/minisgl/attention/fa.py) | 原 page table 仅寻址 normal pool | 混合路径不能把 compressed buffer 伪装成 paged FP16/BF16 tensor |
| [layers/attention.py](../python/minisgl/layers/attention.py) | 调用 backend 前已对 Q/K 应用 RoPE | 不得在新算子里重复旋转 Q/K |

重要细节：K/V 的 important bit 和 unimportant bit 分别可配置。虽然默认 important=4bit、unimportant=2bit，不能在内核里假设 K/V 位宽一定相同。实际解包必须看 `storage_bit`，不能只看算法 `bit`。

## 3. 推荐路线

推荐维护一个共享的 `mixed_paged_attention` CUDA 算子，由 `fi.py` 和 `fa.py` 共同调用。没有 compressed 数据时保留各自原路径，不强迫所有请求使用新内核。

```text
fi.py: 普通输入 -> 原 FlashInfer
       混合输入 -> mixed_paged_attention

fa.py: 普通输入 -> 原 sgl-kernel FlashAttention
       混合输入 -> 同一个 mixed_paged_attention
```

这个方案意味着混合路径使用的是新内核，而不是声称“原 FA3 已原生支持 ZipCache”。避免同时维护 FlashInfer 和 FA3 两套低比特实现，可以把验证重点放在一种输入协议、一套量化语义和两种执行模式上。

CUDA 实现优先复用或基于 FlashInfer 的 paged Attention 计算组织修改 KV loader、tile 迭代器和 mask。维护独立入口与模板实例，不覆盖依赖包原算子。复用上游代码时保留许可证与来源，固定上游版本。

另外保留一个“normal Attention + compressed Attention + 状态合并”的对照实现。它便于逐段验证，但不是最终单个混合主内核。长上下文的最终实现也可能使用 split-KV 主内核加归并内核；不把“只能启动一次 CUDA kernel”作为优化目标。

## 4. 输入协议与维度

### 4.1 维度定义

| 符号 | 含义 |
|---|---|
| `B` | 本次计算的请求数 |
| `Tq` | 所有请求本轮 query token 数之和；普通 Decode 时为 B |
| `Hq/Hkv` | 当前 TP rank 的 query/KV head 数 |
| `D` | 每个 head 的通道数 |
| `P/S` | normal pool 页数、每页 token 数 |
| `L` | 模型层数；一次算子调用只计算其中一层 |
| `G_l` | 第 l 层 batch 内所有 KV 段的描述符数量 |
| `M` | 一个描述符覆盖的 token 数，不一定等于 node.length |
| `Nnormal` | batch 所有 normal 段的 token 数之和，包含本步新 KV |

### 4.2 每次调用的张量

| 输入 | shape / dtype | 作用 |
|---|---|---|
| `q` | `[Tq,Hq,D]`，FP16/BF16 | 已经经过 RoPE 的 query |
| `normal_k/normal_v` | `[P,S,Hkv,D]`，与 q 对应的浮点 dtype | 当前层 normal KV；不复制全池 |
| `q4_buffer/q2_buffer` | 一维 uint8 | pool 的原始 packed 数据 |
| `scale_buffer` | 一维 FP16 | pool 的 min/step |
| `ids_buffer` | 一维 int64 | pool 的 node 内 token 位置 |
| `qo_indptr` | `[B+1]`，GPU int32 | 第 b 个请求的 q 范围为 `[ptr[b],ptr[b+1])` |
| `q_positions` | `[Tq]`，GPU int32/int64 | 请求内部的绝对逻辑位置，优先使用 `batch.positions` |
| `kv_lens` | `[B]`，GPU int32 | 每个请求本轮允许读取的总逻辑 KV 长度，不是 normal token 数 |
| `normal_indices` | `[Nnormal]`，GPU int32 | 只保存仍然有效的 normal 物理 token index |
| `segment_indptr_l` | `[B+1]`，GPU int32 | 第 l 层每个请求使用的描述符范围 |
| `segment_meta_l` | `[G_l,5]`，GPU int32 | `kind,count,position_base,k_storage_bit,v_storage_bit` |
| `segment_offsets_l` | `[G_l,8]`，GPU int64 | 下文定义的八个独立 offset |

输出 `o` 为 `[Tq,Hq,D]`，dtype 与 q 相同。调试或分段归并时可额外输出 `[Tq,Hq]` 的 FP32 LSE，即 softmax 分母的对数。

所有数据位于同一 GPU。本阶段要求标准 MHA/GQA，`Hq % Hkv == 0`；一个 query head 使用 `hkv = hq // (Hq/Hkv)` 对应的 KV head。初版不扩展 MLA、dropout 或训练 backward。

### 4.3 描述符的具体含义

`kind` 初版只有 `NORMAL` 与 `QUANTIZED` 两种。一个 normal 描述符表示逻辑位置连续的一段 token；物理地址可以不连续。若 normal token 被 compressed 段隔开，就分成多个 normal 描述符。

八个 offset 按固定顺序为：

```text
k_q_offset, k_min_offset, k_step_offset,
v_q_offset, v_min_offset, v_step_offset,
ids_offset, normal_index_offset
```

normal 描述符只使用最后一个 offset；quantized 描述符使用前七个。没有使用的字段填零，由 kind 决定是否访问。

offset 的单位沿用 `_PoolSlice`：q offset 是 uint8 元素数，也就是 byte；scale offset 是 FP16 元素数；ids offset 是 int64 元素数；normal_index_offset 是 int32 索引数组的元素数。不要统一按 byte 相加。

使用 int64 offset 的原因是当前 compressed pool 可以是几十 GiB，一个 q buffer 就可能超过 2 GiB。CUDA 中 offset、`r*Hkv*D` 和最终地址加法也必须在足够宽的整数类型中计算，不能最后才强转。

一个 compressed node 在一层通常生成两个描述符：important 与 unimportant；空组不生成描述符。分别从 `layer_entry.k/v` 读取 q/min/step 的 slice offset。当前 K/V 分组相同，可共享读取 K 侧的 ids，但应在测试或描述符构建校验中确认两侧 token 顺序一致；将来若分组不同，需要另行映射，不能直接套用。

CPU 只读取已有 Python 元数据，如 slice offset、shape、bit 和 node 的位置起点；不对 GPU `ids/q/min/step` 做 `.cpu()`。描述符按 batch 构建或更新，稳定的各层 entry 描述应缓存；不要在每层 `forward()` 中遍历字典并分配 CUDA tensor。

### 4.4 位置示例

假设一个 compressed node 对应请求位置 `[100,106)`，某层分组为：

```text
important.ids   = [0,2,5]   -> 实际逻辑位置 [100,102,105]
unimportant.ids = [1,3,4]   -> 实际逻辑位置 [101,103,104]
normal 后缀                  -> 实际逻辑位置 [106,107]
```

算子可以先读 important，再读 unimportant，最后读 normal。这改变了 KV 遍历顺序，但只要 K/V 始终配对、每个 token 恰好出现一次、mask 按真实逻辑位置判断，就保持同一 Attention 数学结果，浮点归约顺序可能造成小差异。

不需要为了“按 token 顺序”把 q4/q2 解包、scatter 成完整 `[N,Hkv,D]`。`ids` 在这里主要用于逻辑位置计算与可见性判断，而不是用于恢复物理 KV 地址。

## 5. CUDA 数据读取

### 5.1 normal loader

对 normal 描述符内第 r 个 token：

```text
idx = normal_indices[normal_index_offset + r]
page = idx // S
slot = idx % S
K = normal_k[page,slot,hkv,d]
V = normal_v[page,slot,hkv,d]
kv_position = position_base + r
```

这是现有 pool 的读取逻辑。可使用零拷贝 flat view 直接通过 idx 访问，避免重复做除法。

### 5.2 compressed loader

对 quantized 描述符内第 r 个 token、第 h 个 KV head、第 d 个通道，以 K 为例：

```text
b = k_storage_bit                         # 2 或 4
values_per_byte = 8 // b
e = (r * Hkv + h) * D + d                 # 在该 QuantizedPart 内的扁平元素下标
byte_offset = k_q_offset + e // values_per_byte
shift = (e % values_per_byte) * b
packed = q2_buffer[byte_offset] 或 q4_buffer[byte_offset]
quant = (packed >> shift) & ((1 << b) - 1)
minimum = scale_buffer[k_min_offset + r * Hkv + h]
step = scale_buffer[k_step_offset + r * Hkv + h]
K_hat = cast_to_original_kv_dtype(float32(quant) * float32(step) + float32(minimum))
kv_position = position_base + ids_buffer[ids_offset + r]
```

V 独立使用自己的位宽和三个 offset，但采用同一个 r、h 和 token 位置。

这个解包顺序与现有 `_pack_lowbit()` 一致：一个 byte 的低位保存更靠前的量化值。不能误用高位优先格式，也不能假设每个 head 的起点都独立做过 byte padding；现有代码是在整个 part 展平后打包，只在末尾补齐。

反量化使用 pool 中已经舍入到 FP16 的 min/step，不使用量化前 FP32 参数。为对齐现有 restore，反量化后还要转换到原 KV dtype，再参与 Attention。CUDA FMA 融合等可能改变末位结果，先与现有 `_dequantize_part_gpu_into()` 单独比对，不把逐 bit 一致作为默认承诺。

### 5.3 片上执行与边界

tile 指每次处理的一小块 token/head 数据，例如 `[BN,D]`。压缩 loader 读取 packed bytes 与少量参数，在寄存器里解包，再按矩阵乘法要求写入共享内存；FP16/BF16 tile 使用后复用这块片上空间。

不分配随完整上下文长度增长的 FP16/BF16 KV workspace，也不写回 normal pool。允许存在按 tile、query 或 split 数计费的临时空间，但必须统计；寄存器溢出到 local memory 也可能产生显存流量，需要 profiler 核对。

一个 tile 尽量只包含一个描述符中的 token，避免 warp 内不同线程走 2bit、4bit、normal 三套分支。遍历当前格式不要求重排 pool，但很多短 node 会造成 tile 利用率低和描述符开销，这必须实测。

原 pool 子分配不保证每段都有 16-byte 对齐。先支持非对齐与尾部受保护读取，再对满足条件的片段启用向量化。不能为了凑满向量越过 slice 读取其他 entry 的数据。初版优先 D=64/128 的专用模板，其他 D 明确拒绝或走受验证的慢路径。

## 6. Attention 主循环

### 6.1 统一 softmax

对一个 query/head，normal、important、unimportant token 都属于同一个 softmax 分母。不能分别算三个归一化输出后直接相加或平均。

对当前 query tile，维护 FP32 状态：最大分数 `m[BM]`、指数和 `ell[BM]`、未归一化加权和 `acc[BM,D]`。每读取一块 KV：

```text
scores = Q_tile @ K_tile.T / sqrt(D)      # [BM,BN]
scores[不可见位置] = -inf
m_new = maximum(m, row_max(scores))
alpha = exp(m - m_new)
p = exp(scores - m_new[:,None])
ell = alpha * ell + row_sum(p)
acc = alpha[:,None] * acc + p @ V_tile
m = m_new
```

遍历所有 normal/quantized 描述符后返回 `acc / ell[:,None]`。公式使用自然指数，实际 CUDA 可转换为 exp2 实现；两者的 score scale 与 LSE 约定必须一致。分块在线 softmax 与矩阵乘法的基础结构可参考 [Triton 官方 Fused Attention 示例](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)，本方案扩展的是 KV 来源和位置语义。

初始状态为 `m=-inf, ell=0, acc=0`。全被 mask 的 tile 不能直接计算 `-inf - -inf`；需要有效行判断，空 tile 保持状态不变。整行无可见 KV 时约定输出零、LSE=-inf，padding 请求不得读取无效地址。

### 6.2 Prefill 与 Decode

两种模式共享 loader、描述符和 softmax 语义，但使用不同调度模板。

Decode 优先优化显存带宽与并行度。可以按 `(request, KV head, KV split)` 分配 CTA，并在 CTA 内处理共享这个 KV head 的多个 query head，复用解包后的 K/V。小 GQA ratio 下也要测试 CUDA-core 路径，不能假设 Tensor Core 始终更快。

长上下文且 batch 较小时使用 split-KV：不同 CTA 处理互不重叠的 KV 子集，输出局部 O/LSE 后再归并。split 工作量应结合 normal/2bit/4bit 的成本调优，而不只看 token 数。不要为每个 node 从 Python 单独启动 kernel。

Prefill 使用 `[BM,D]` 的 query tile 和 `[BN,D]` 的 KV tile，矩阵乘法优先复用成熟的 Tensor Core 实现。可从 BM=16/32/64、BN=32/64/128 搜索配置；这些是候选值，不是已验证的最优参数。限制 split workspace，避免长 Prefill 时部分输出存储反过来主导显存占用。

### 6.3 因果位置

可见性依据是 `kv_position <= q_positions[q_idx]`，还要检查 request 边界、token 数和 padding。chunked Prefill 的 query 并非从位置 0 开始，不能用 tile 内行号代替绝对位置。

现有已命中的 compressed prefix 通常早于本轮所有 query，可以在确认这一条件后走免逐元素 causal mask 的快速路径；不能对任意 compressed token 都作此假设。

重新排列 token 后，上游基于 `kv_len-qo_len` 与连续 `kv_idx` 的剪枝可能跳过本应可见的 tile。修改 CUDA 时既要换 mask，也要换或禁用这些依赖逻辑顺序的循环边界优化，不能仅修改最后一个 mask 表达式。

初版支持已应用 RoPE、标准 causal、`softmax_scale=D**-0.5`，不支持的 sliding window、ALiBi、softcap、特殊 mask 必须显式报错；后续支持时两类 KV 应使用同一位置和分数变换规则。

## 7. fi.py / fa.py 数据准备改动

### 7.1 共享元数据

建议新增 `MixedAttnMetadata(BaseAttnMetadata)`，实现 `get_last_indices(bs)`，仍返回 `qo_indptr[1:bs+1]-1`。这与现有 [BaseAttnMetadata](../python/minisgl/attention/base.py) 的接口兼容，不必把新格式硬塞进 `FIMetadata` 或 `FAMetadata`。

新增输入准备 helper，例如 `build_mixed_metadata(batch, request_sources)`。`request_sources` 是测试提供的每请求 KV 段列表，不通过旧 page table 反向猜测 compressed 身份。helper 只构造描述符与正常索引；不执行 `_dequantize_mixed_gpu()`、不重排完整 KV、也不重复量化。

`qo_indptr`、query position、normal indices 可以跨层共享；每层的 compressed 描述符不同。首次构建后，后续 Decode 若 prefix 不变，只更新 query 位置、长度、normal 尾部和发生变化的描述符内容。

### 7.2 fi.py

普通路径保留现有 `prepare_metadata()` 和 `wrapper.plan()`。混合路径由 helper 生成 `MixedAttnMetadata`，在 `forward()` 开头识别，并在原 `FIMetadata` 断言和 `_initialize_metadata_once()` 之前分派。

```python
# 设计伪代码；这些新接口尚未实现。
metadata = batch.attn_metadata
if isinstance(metadata, MixedAttnMetadata):
    self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
    return mixed_paged_attention(
        q=q,
        normal_k=self.kvcache.k_cache(layer_id),
        normal_v=self.kvcache.v_cache(layer_id),
        compressed_buffers=metadata.compressed_buffers,
        layer_metadata=metadata.for_layer(layer_id),
    )
# 原来的 FIMetadata 检查、plan、store_kv、wrapper.run 路径不变。
```

新 K/V 仍只写入 normal pool，一轮中只能写一次。normal 描述符必须包含这些新 token，而且与 compressed 描述符不重复。

混合数据不传给原 FlashInfer `plan()`。原 `seq_lens` 和 `paged_kv_indices` 是单一 KV 表示；把总长度与仅 normal 的 indices 混用可能越界。全 normal 的 batch 则继续使用原实现。含两类请求的 batch 初版可以整体走新内核，每个请求使用自己的描述符。

### 7.3 fa.py

同样在 `FAMetadata` 断言之前识别共享 metadata，先 `store_kv()`，再调用相同的新算子。普通输入仍进入 `_fa_sgl_impl()`。

混合路径不再依赖旧 `page_table`、`cu_seqlens_k` 来定位 compressed 数据，而是使用描述符和 `kv_lens`。不能对 compressed offset 执行 `div_(page_size)`，它不是 normal page ID。

因此第一阶段可以不修改 sgl-kernel FA3 CUDA 源码：`fa.py` 是接入位置，混合 kernel 是另一个实现。如果以后要求保留 FA3 的 Hopper 主循环，再实现相同协议对应的 FA3 loader，这应作为独立的硬件优化阶段。

### 7.4 CUDA Graph

第一阶段混合路径使用 eager 执行。必须显式阻止混合 batch 落入原 FI/FA graph replay，不能只修改 eager `forward()` 就声称支持 graph。

后续为 qo_indptr、position、indices、各层描述符和 split workspace 分配固定容量 buffer；capture/replay 保持地址、shape、kernel 拓扑与编译专用参数稳定，只更新内容和有效计数。按容量分桶，超容量时 eager 回退或重新 capture。压缩 pool 地址固定有帮助，但它本身不保证 entry 的 slice 在执行期间不会失效。

## 8. 分段合并对照版本

对同一个 query 分别计算 normal 子集和 compressed 子集，得到 `(O_n,L_n)`、`(O_c,L_c)`。两者覆盖不重叠的 KV，且 mask/scale 相同时：

```text
a = max(L_n,L_c)
w_n = exp(L_n-a)
w_c = exp(L_c-a)
O = (w_n*O_n + w_c*O_c) / (w_n+w_c)
```

LSE 为 `[Tq,Hq]`，对 O 的最后一个 D 维广播。用稳定形式避免指数溢出；空子集为 O=0、LSE=-inf，两个子集同时为空时单独处理。

FlashInfer v0.5.3 的 [Prefill wrapper](https://github.com/flashinfer-ai/flashinfer/blob/v0.5.3/flashinfer/prefill.py) 和 [Decode wrapper](https://github.com/flashinfer-ai/flashinfer/blob/v0.5.3/flashinfer/decode.py) 都提供 `run(..., return_lse=True)`。[merge_state](https://docs.flashinfer.ai/generated/flashinfer.cascade.merge_state.html) 接收两个子集的 O/LSE 并合并，适合构造对照路径。

注意 LSE 的对数底数。核对的 FlashInfer v0.5.3 [state.cuh](https://github.com/flashinfer-ai/flashinfer/blob/v0.5.3/include/flashinfer/attention/state.cuh) 使用 `m+log2(d)`，其 [variants.cuh](https://github.com/flashinfer-ai/flashinfer/blob/v0.5.3/include/flashinfer/attention/variants.cuh) 把 score 缩放到 exp2 域。与自然对数协议交换时需要 `L_e=L_2*ln(2)`。不要凭变量名为 lse 就直接跨库合并；应对固定版本和具体 backend 做实测校验。

第一版分段对照限定“compressed 历史前缀 + 连续 normal 后缀”，normal 子调用使用自己的有效长度。若任意过滤 KV 或引入位置相关 mask，就必须验证重新排列后的可见性，不能照搬原总长度与 causal 对齐。FA 的 LSE 返回形状和底数也要由适配器校验，不假设与 FlashInfer 一致。

该路径不物化 compressed KV，但会多读 Q、写出部分 O/LSE，并增加 kernel launch；它与统一内核应分别计时。

## 9. 具体 CUDA 修改位置

推荐以 FlashInfer v0.5.3 为可核对的起点。仓库依赖目前是 `flashinfer-python>=0.5.3`，这不代表服务器装的就是 0.5.3；实际实施前记录 wheel 版本、源码 tag/commit、CUDA 与 GPU 型号。

| 上游位置 | 可以复用的部分 | 需要修改的部分 |
|---|---|---|
| [attention/decode.cuh](https://github.com/flashinfer-ai/flashinfer/blob/v0.5.3/include/flashinfer/attention/decode.cuh) | QK、局部状态更新、split-KV 组织 | paged 地址生成、K/V 的 cp_async 读取、真实位置传递 |
| [attention/prefill.cuh](https://github.com/flashinfer-ai/flashinfer/blob/v0.5.3/include/flashinfer/attention/prefill.cuh) | 分块矩阵乘法、在线 softmax | `page_produce_kv`、描述符遍历、mask 与依赖连续位置的剪枝 |

普通异步拷贝不能顺带执行任意位解包和 affine 反量化。压缩路径需要先装载 packed bytes/参数，再由线程反量化到计算所需的片上布局，并建立正确的 producer/consumer 同步。

高层 logits 变换接口处理的是已经计算出的分数，不是完整替换低比特 K/V 读取路径的接口。还需要扩展 C++/FFI 参数结构及 launch/plan，使两个 pool 指针和描述符真正到达 device kernel。

若后续优化 FA3，参考上游 [Hopper mainloop](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp) 和 [launch template](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_launch_template.h)，处理 packed load、共享内存格式、流水线 barrier 和类型分派。这里链接是 Dao-AILab 当前 main 的参考，不是声称与本项目所装 sgl-kernel 逐文件一致。

## 10. 建议文件范围与实施顺序

第一阶段已经新增或修改以下文件：

| 文件 | 职责 |
|---|---|
| `python/minisgl/attention/mixed.py` | 输入描述符、元数据构建与校验 |
| `python/minisgl/kernel/mixed_attention.py` | Python/FFI 入口、专用参数与 workspace 管理 |
| `python/minisgl/kernel/csrc/include/minisgl/mixed_kv.cuh` | 描述符定义、normal/quantized loader |
| `python/minisgl/kernel/csrc/jit/mixed_attention.cu` | correctness-first Prefill/Decode 共用入口与 dispatch |
| `python/minisgl/attention/fi.py`、`fa.py` | 混合分派、原路径保留、graph 能力检查 |
| `tests/kernel/test_mixed_attention.py` | 解包、寻址、mask、Attention 与 backend 接入测试 |
| `experiment/bench_mixed_attention.py` | 算子与数据准备微基准 |

CUDA 扩展优先沿用项目 [kernel/utils.py](../python/minisgl/kernel/utils.py) 的 TVM FFI/JIT 机制。调用使用当前 PyTorch CUDA stream；不依赖默认 stream 恰好同步。

1. 输入协议与参考输出：用真实 `_V3CompressedPool.allocate_part()` 生成数据；先核对每个 token 的位置、bit 和地址。
2. 独立 CUDA loader：验证 2bit/4bit 解包与现有反量化结果，覆盖非零 offset、尾部和 K/V 不同 bit。
3. 单请求 Decode：normal + 一个 compressed node，再扩展多 node、batch、GQA 与 split-KV。
4. Prefill：加入多 query、chunked Prefill、位置 mask，保留 normal-only 的上游对照。
5. `fi.py`/`fa.py` 接入测试：由测试直接构建混合 metadata，检查新 KV 写入和两条 backend 的输出，不修改 Scheduler。
6. 性能优化与 CUDA Graph：向量化、GQA 复用、流水线、容量分桶；只有测试通过的组合才启用。

只完成 Decode 时还不能说已解决 prefix-hit Prefill 的 restore 问题。该阶段可以独立交付算子结果，但必须标明支持范围。

## 11. 正确性与性能验收

### 11.1 正确性

主参考必须使用“同一份 packed 数据经现有函数反量化后的 KV + normal KV”，再按真实逻辑位置计算 Attention；不能只拿量化前的原始 KV 比，否则量化误差和新 kernel 误差混在一起。

至少覆盖：全 normal、全 compressed 历史加新 normal token、空组、全 4bit/全 2bit、混合比例、多 compressed node、normal/compressed 交错、不同层分组、K/V 位宽不同、非连续物理页、非零 byte offset、非对齐读取和 tile 尾部。

batch/GQA/Prefill 覆盖不同请求长度、D=64/128、Hq/Hkv=1/2/4/8、chunked Prefill、padding、全 mask 子集、split 归并和共用 prefix。额外做因果不变性测试：修改 query 未来位置的 K/V，不应影响该 query 输出。

记录最大绝对误差、相对 L2 误差、O/LSE 和 NaN/Inf。容差按 dtype、长度以及相同数据上的成熟 backend 数值误差确定，不预先承诺逐 bit 一致。用 Compute Sanitizer 检查越界及同步问题；GPU 测试的跳过不能计为通过。

### 11.2 基准

| 版本 | 目的 |
|---|---|
| 原始 FP16/BF16 KV + 原 FI/FA | 量化前质量和成熟内核性能参考 |
| 当前 V3 restore + 原 FI/FA | 当前真实处理方式的对照 |
| 同量化数据预先 restore，单测原 FI/FA | 分离 Attention 本身与 restore 成本 |
| 融合 CUDA restore 到 normal + 原 FI/FA | 排除“只是原 PyTorch restore 慢”这一因素 |
| normal/压缩分段计算再合并 | 检查统一内核是否值得增加复杂度 |
| 统一混合内核 | 最终候选 |

扫描 batch=1/4/16/32，KV 长度=1K/4K/16K/32K，query 长度=1/16/128/512，compressed 占比=0/25/50/75/100% 的历史 KV，以及 node 数=1/8/64 和不同 2bit/4bit 比例。容量不足的组合记录为不支持，不硬跑 OOM。

分别报告 kernel-only 延迟、metadata 准备延迟、含新 KV 写入的 backend 延迟、restore 加首轮 Attention、完整 N 步 Decode 总时间与 workspace 峰值。CUDA Event 测 GPU 延迟，CPU 计时明确同步边界；JIT/预热与稳态分开，graph/eager 使用相同条件。

特别注意：V3 对一次请求的 compressed prefix 通常只 restore 一次，后续 Decode 重用 normal KV。真实基线不能在每一步 Decode 都重复计入 restore。

用 Nsight 核对 DRAM 读写、带宽、寄存器占用、spill、共享内存、Tensor Core 利用率及 kernel 数量。不仅检查代码中有没有 `torch.empty`，也检查是否真的没有完整 KV 的显存写回。

### 11.3 收益与风险

潜在收益是减少 active prefix 的 normal 副本、减少历史 KV 读取字节数，并省去命中时完整 restore 的显存写回。它不是减少需要关注的 token 数，也没有减少基本的 QK/PV 数学运算规模。

以 K/V 都有 60% token 为 4bit、40% 为 2bit 为例，平均存储位宽为 3.2bit。每层每 token 的原 KV 为 `4*Hkv*D` byte；当前压缩格式约为 `0.8*Hkv*D + 8*Hkv + 16` byte，后三项分别反映 packed K/V、两侧 min/step 和重复保存的两侧 ids，另有尾部及 entry 元数据。算子共享读取一侧 ids 时，实际读取量与归档存储量又有所不同。

上述字节比不等于延迟加速比。反量化每个 Decode step 都要执行；短 KV、小 compressed 比例、很多碎片 node、大量 query 的 Prefill 或低效 tile/GQA 复用都可能让新路径更慢。最终必须给出适用区间，而不是预设一定优于原 FI/FA。

本阶段验收结论应表述为“在指定输入、GPU 与算子形状下，混合 KV Attention 正确并达到某项微基准结果”。服务吞吐、prefix 命中收益和实际 normal pool 容量收益，需要后续 runtime 接入后另行验证。
