# MyLlamaModel 类说明

`MyLlamaModel` 位于 `zipcache/models/modeling_llama.py`，是 ZipCache 版本的 LLaMA 主干网络。它对应 HuggingFace 原版的 `LlamaModel`，但内部 decoder layer 被替换成 `MyLlamaDecoderLayer`，从而让每层 attention 都可以使用 ZipCache 的 KV cache 压缩逻辑。

## 在项目中的作用

整个 ZipCache LLaMA 推理链路大致是：

```text
zipcache_generation_demo.py
  -> MyLlamaForCausalLM
    -> MyLlamaModel
      -> MyLlamaDecoderLayer
        -> MixedLlamaAttention
          -> MixedPrecisionCompressUnion
          -> compress_function.py
```

`MyLlamaModel` 处在中间位置。它本身不直接做采样，也不直接输出最终 token，而是负责把输入 token 经过多层 Transformer decoder，得到最后的 hidden states，并把每层新的 `past_key_values` 收集起来返回给外层。

外层 `MyLlamaForCausalLM` 会把 `MyLlamaModel` 输出的 hidden states 送入 `lm_head`，得到词表 logits，然后 HuggingFace `generate()` 再根据 logits 选择下一个 token。

## 类初始化 `__init__`

```python
def __init__(self, config: LlamaConfig, compress_config=None)
```

这个函数创建模型结构。

主要步骤：

1. 保存 padding token id 和词表大小；
2. 创建 token embedding 表；
3. 创建 `config.num_hidden_layers` 个 `MyLlamaDecoderLayer`；
4. 创建最终的 `LlamaRMSNorm`；
5. 调用 `post_init()` 执行 HuggingFace 模型初始化收尾；
6. 保存 `compress_config`。

关键成员：

- `self.embed_tokens`：把 token id 转成 embedding 向量。
- `self.layers`：多层 decoder block。
- `self.norm`：最后的 RMSNorm。
- `self.compress_config`：ZipCache 压缩配置。
- `self.gradient_checkpointing`：训练省显存选项，推理时通常为 False。

## `get_input_embeddings`

```python
def get_input_embeddings(self)
```

返回输入 embedding 层。

这是 HuggingFace 统一接口。外部工具可能通过它访问或替换 embedding，例如扩展词表时调用 `resize_token_embeddings()`。

## `set_input_embeddings`

```python
def set_input_embeddings(self, value)
```

替换输入 embedding 层。

常见场景：

- tokenizer 词表变化；
- 实验中替换 embedding；
- 加载某些特殊权重。

## `forward`

```python
def forward(
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
)
```

这是 `MyLlamaModel` 最重要的函数，负责执行一次模型主体前向传播。

### 输入参数

- `input_ids`：token id，形状 `[batch, seq_len]`。
- `attention_mask`：标记哪些 token 是有效 token，哪些是 padding。
- `position_ids`：每个 token 的位置编号；如果不传，函数内部自动生成。
- `past_key_values`：历史 KV cache。
- `inputs_embeds`：已经查好 embedding 的输入向量。它和 `input_ids` 二选一。
- `use_cache`：是否返回新的 KV cache。生成时通常为 True。
- `output_attentions`：是否返回每层 attention 权重。
- `output_hidden_states`：是否返回每层 hidden states。
- `return_dict`：是否返回 HuggingFace dataclass 格式。

### `past_key_values` 在普通 LLaMA 和 ZipCache 中的区别

普通 LLaMA：

```python
past_key_values[layer_id] = (key_tensor, value_tensor)
```

其中 `key_tensor.shape[2]` 是历史 token 数。

ZipCache：

```python
past_key_values[layer_id] = (key_compress_union, value_compress_union)
```

压缩对象没有 `shape[2]`，所以 `MyLlamaModel.forward()` 用：

```python
past_key_values[0][0].seq_length
```

来获得历史 token 数。

### forward 的执行流程

1. 解析输出控制参数。

如果调用者没有传 `output_attentions`、`output_hidden_states`、`use_cache`、`return_dict`，就使用模型 config 中的默认值。

2. 检查输入形式。

`input_ids` 和 `inputs_embeds` 只能传一个。如果传 `input_ids`，函数内部会查 embedding；如果传 `inputs_embeds`，说明外部已经查好了。

3. 计算历史 cache 长度。

如果 `past_key_values` 不为空：

- 普通 tensor cache 用 `shape[2]`；
- ZipCache 压缩对象用 `seq_length`。

4. 生成 `position_ids`。

如果当前是 decode 阶段，历史 cache 已经有很多 token，新 token 的 position 必须接在历史后面，否则 RoPE 位置编码会错。

5. 得到 `inputs_embeds`。

如果输入是 token id，就通过 `self.embed_tokens(input_ids)` 转成向量。

6. 准备 attention mask。

普通 attention 路径使用 4D causal mask，FlashAttention 路径通常使用 2D mask 或 None。

7. 逐层执行 decoder layer。

每层都会接收：

- 当前 `hidden_states`；
- `attention_mask`；
- `position_ids`；
- 当前层自己的 `past_key_value`；
- `use_cache` 等控制参数。

如果开启 ZipCache，`MyLlamaDecoderLayer` 内部的 `MixedLlamaAttention` 会完成：

- 解压历史 KV；
- 计算 attention；
- 识别 unimportant token；
- 压缩新的 KV；
- 返回新的 `past_key_value`。

8. 收集输出。

每层输出的 cache 会被追加到 `next_decoder_cache` 中，最后作为新的 `past_key_values` 返回。

9. 最终 RMSNorm。

所有 decoder layer 结束后，对最后的 hidden states 做一次 RMSNorm。

10. 返回结果。

如果 `return_dict=True`，返回：

```python
BaseModelOutputWithPast(
    last_hidden_state=hidden_states,
    past_key_values=next_cache,
    hidden_states=all_hidden_states,
    attentions=all_self_attns,
)
```

如果 `return_dict=False`，返回 tuple。

## prefill 和 decode 中的行为

### prefill

第一次输入完整 prompt，`past_key_values=None`，`seq_len` 通常大于 1。

此时 `MyLlamaModel` 会让每层处理完整 prompt，并返回每层压缩后的 KV cache。

### decode

后续生成时，`past_key_values` 已经存在。HuggingFace `generate()` 通常只传最新 token。

此时 `MyLlamaModel` 会：

1. 根据历史 cache 长度生成新 token 的 position；
2. 把每层历史压缩 cache 传给对应 decoder layer；
3. 收集每层更新后的压缩 cache；
4. 返回给下一轮 decode。

## 为什么 MyLlamaModel 不直接输出文本

`MyLlamaModel` 只输出 hidden states。文本生成还需要：

1. `MyLlamaForCausalLM.lm_head` 把 hidden states 转成 logits；
2. 采样或 greedy decoding 选择 token；
3. tokenizer 把 token id 解码成字符串。

所以 `MyLlamaModel` 是“模型主体”，不是完整的语言模型接口。

## 新手阅读建议

阅读这个类时，可以抓住三个问题：

1. 输入 token 是怎么变成 hidden states 的？
2. 每层 decoder 是怎么接收和返回 `past_key_values` 的？
3. ZipCache 为什么需要用 `seq_length` 替代普通 tensor 的 `shape[2]`？

理解这三个问题，就能明白 `MyLlamaModel` 在 ZipCache 中的核心作用。
