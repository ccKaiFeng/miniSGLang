# modeling_llama.py 说明文档

`modeling_llama.py` 是 ZipCache 中最重要的模型实现文件之一。它基于 HuggingFace Transformers 的 LLaMA 代码改写，在保持 LLaMA 模型结构和 `generate()` 使用方式基本不变的前提下，把 KV cache 的保存方式从普通 FP16/BF16 tensor 改成 ZipCache 的压缩对象。

## 这个文件在项目中的作用

在整个 ZipCache 项目中，`modeling_llama.py` 起到“把压缩算法接入真实 LLaMA 推理流程”的作用。

项目中不同文件的分工是：

- `zipcache_generation_demo.py`：演示如何加载模型、设置 `compress_config`、调用 `generate()`。
- `CompressUtils/compress_function.py`：提供底层量化/解量化数学函数。
- `CompressUtils/compress_class.py`：把量化函数包装成能保存状态的压缩对象。
- `modeling_llama.py`：把这些压缩对象接到 LLaMA 的 attention 和 `past_key_values` 流程中。

如果只看压缩算法，读 `CompressUtils`；如果想知道 ZipCache 如何真正影响模型推理，就读 `modeling_llama.py`。

## 原始 LLaMA 推理流程

普通 LLaMA 解码时，每层 attention 都会保存历史 Key/Value：

```python
past_key_value = (key_states, value_states)
```

下一次生成 token 时，模型不再重复计算全部历史 token，而是直接把历史 KV 与当前 token 的 KV 拼接：

```python
key_states = torch.cat([past_key_value[0], key_states], dim=2)
value_states = torch.cat([past_key_value[1], value_states], dim=2)
```

这样可以节省计算，但历史 KV 会一直占显存。上下文越长，KV cache 越大。

## ZipCache 改了什么

ZipCache 不改变 attention 公式，也不改变模型权重。它主要改变 KV cache 的保存方式。

普通模式：

```python
past_key_value = (key_states, value_states)
```

ZipCache 模式：

```python
past_key_value = (past_key_union, past_value_union)
```

其中 `past_key_union` 和 `past_value_union` 是 `MixedPrecisionCompressUnion` 对象，内部保存：

- 压缩后的 Key/Value 数据；
- important token 的位置；
- unimportant token 的位置；
- 量化需要的 `min`、`step`、`scale`；
- 原始 shape、dtype、序列长度；
- streaming 模式下还没有重新压缩的 buffer。

下一轮 attention 前会调用：

```python
prev_keys = past_key_value[0].decompress()
prev_values = past_key_value[1].decompress()
```

也就是说，attention 看到的仍然是普通 tensor，只是存储时被压缩了。

## 文件中的主要类

### `LlamaRMSNorm`

LLaMA 使用的归一化层。它和 LayerNorm 类似，但只根据均方根做归一化。

### `LlamaRotaryEmbedding`

实现 RoPE 位置编码。LLaMA 不使用传统绝对位置 embedding，而是把位置信息旋转到 Q/K 向量里。

### `LlamaMLP`

Transformer block 里的前馈网络，包含 gate/up/down 三个线性层。

### `LlamaAttention`

普通 attention 路径，保留 HuggingFace 风格，同时支持 `CompressUnion` 这种简单压缩对象。

### `MixedLlamaAttention`

ZipCache 的核心类。它完成：

1. 计算当前输入的 Q/K/V；
2. 如果有历史 `past_key_value`，先解压历史 KV；
3. prefill 阶段通过 probe attention 估计 token 重要性；
4. 把注意力较低的 token 标记为 unimportant；
5. 使用 FlashAttention 计算真实 attention 输出；
6. 把新的 KV cache 压缩成 `MixedPrecisionCompressUnion`；
7. 返回新的 `past_key_value` 给下一轮 decode。

### `MyLlamaDecoderLayer`

ZipCache 版本的 decoder layer。它把原始 self-attention 替换成 `MixedLlamaAttention`。

### `MyLlamaModel`

ZipCache 版本的 LLaMA 主体。它把每一层都构造成 `MyLlamaDecoderLayer`，并负责把每层的 `past_key_values` 传下去。

### `MyLlamaForCausalLM`

对外使用的语言模型类。demo 中加载的就是它：

```python
model = MyLlamaForCausalLM.from_pretrained(..., compress_config=compress_config)
```

它负责：

- 调用 `MyLlamaModel` 得到 hidden states；
- 通过 `lm_head` 转成 logits；
- 支持 HuggingFace `generate()`；
- 在 `prepare_inputs_for_generation()` 中处理 `past_key_values`。

## 关键数据流

### 1. prefill 阶段

输入是一整段 prompt，`q_len > 1`。

`MixedLlamaAttention.forward()` 会：

1. 计算整段 prompt 的 Q/K/V；
2. 抽取一部分 probe token；
3. 用 probe token 计算注意力；
4. 统计每个历史 token 被关注的程度；
5. 找出注意力最低的一部分 token，保存为 `unimportant_ids_k` 和 `unimportant_ids_v`；
6. 真实 attention 输出仍然用 FlashAttention 计算；
7. 把 KV cache 压缩后返回。

### 2. decode 阶段

输入通常只有一个新 token，`q_len == 1`。

流程是：

1. 从 `past_key_value` 解压历史 KV；
2. 和当前 token 的 K/V 拼接；
3. 计算当前 token 对所有历史 token 的 attention；
4. 如果到达 `streaming_gap`，重新更新最近一段 token 的重要性；
5. 把更新后的 KV 再次压缩保存。

## 重要变量解释

- `hidden_states`：当前层输入 hidden vector。
- `query_states`：Q 矩阵，用来查询应该关注哪些历史 token。
- `key_states`：K 矩阵，被 Q 匹配。
- `value_states`：V 矩阵，attention 权重最终加权求和的内容。
- `past_key_value`：历史 KV cache。普通模式是 tensor tuple，ZipCache 模式是压缩对象 tuple。
- `kv_seq_len`：当前 attention 能看到的总 KV 长度。
- `q_len`：当前 forward 的 query token 数。大于 1 通常是 prefill，等于 1 通常是 decode。
- `unimportant_ids_k`：Key cache 中低重要性 token 的索引。
- `unimportant_ids_v`：Value cache 中低重要性 token 的索引。
- `compress_config`：压缩配置，从 demo 传入，每层 attention 都会使用。
- `use_cache`：是否返回新的 KV cache。生成时必须为 True，ZipCache 才会生效。

## 为什么需要 `prepare_inputs_for_generation()`

HuggingFace 的 `generate()` 会循环调用模型。第一次输入完整 prompt，后面只需要输入最新 token。

`prepare_inputs_for_generation()` 的作用是：

1. 如果已经有 `past_key_values`，说明历史 token 已经缓存；
2. 从 `input_ids` 中裁掉已经处理过的前缀；
3. 只把新 token 送进模型；
4. 同时把 `past_key_values` 继续传给模型。

ZipCache 修改了这里：当 `past_key_values[0][0]` 不是 tensor，而是压缩对象时，要用它的 `seq_length` 获取历史长度。

## 阅读建议

建议按这个顺序阅读：

1. `MyLlamaForCausalLM`
2. `MyLlamaModel`
3. `MyLlamaDecoderLayer`
4. `MixedLlamaAttention.forward()`
5. `CompressUtils/compress_class.py`
6. `CompressUtils/compress_function.py`

其中最核心的是 `MixedLlamaAttention.forward()`。如果你能理解它如何解压旧 KV、识别不重要 token、压缩新 KV，就基本理解了 ZipCache 如何接入 LLaMA。
