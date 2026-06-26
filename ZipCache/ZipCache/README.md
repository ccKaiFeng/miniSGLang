# ZipCache 代码阅读说明

本仓库是论文 **ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification** 的实现。核心目标是降低大语言模型推理时 KV cache 的显存占用：先用注意力分布识别“重要 token”和“不重要 token”，再对两类 token 使用不同 bit 数的量化。重要 token 用较高精度保留，不重要 token 用较低精度压缩。

当前代码主要基于 HuggingFace Transformers 的 LLaMA/Mistral 模型代码改写，压缩逻辑集中在 attention 的 `past_key_value` 读写路径中。

论文链接：https://arxiv.org/abs/2405.14256

## 论文要点

论文指出，长上下文推理中 KV cache 会占用大量显存，直接低 bit 量化虽然能省显存，但如果重要 token 被过度压缩，模型准确率会明显下降。ZipCache 的核心是更准确地识别 salient tokens，并对不同 token 做自适应混合精度量化。

论文中的几个关键点和代码对应关系：

- Channel-separable tokenwise quantization：代码中对应 `channel_separate_mixed_tokenwiseQ`，主要用于 Value cache。
- Normalized attention score：代码中用 attention sum 再按 token 可见范围做归一化，避免靠前 token 因为被更多后续 token 看到而天然分数更高。
- Salient token identification：代码中用 `topk(..., largest=False)` 找低注意力 token，作为 unimportant ids。
- FlashAttention compatibility：代码中先用 probe token 近似估计 token 重要性，再用 FlashAttention 计算真实 attention 输出。
- Mixed precision KV cache：important token 使用 4-bit，unimportant token 使用 2-bit，是 demo 默认配置。

## 核心思想

LLM 生成时会把历史 token 的 Key/Value 保存在 KV cache 里。普通实现保存 FP16/BF16 tensor，显存开销随上下文长度线性增长。

ZipCache 的处理流程是：

1. prefill 阶段计算 prompt 内部注意力分布；
2. 根据每个历史 token 得到的注意力重要性，找出低注意力的 unimportant tokens；
3. 对重要 token 使用较高 bit，例如 4-bit；
4. 对不重要 token 使用较低 bit，例如 2-bit；
5. decode 阶段需要读历史 KV 时，先解压回近似 FP16/BF16 tensor，再走原 attention 计算。

注意：这份代码不是修改 CUDA attention kernel，而是在 Python / PyTorch 层把 `past_key_value` 从普通 tensor 换成压缩对象。

## 目录结构

```text
ZipCache/
├── README.md
├── setup.py
├── requirements.txt
├── zipcache_generation_demo.py
├── asset/
│   └── gsm8k_sample.txt
└── zipcache/
    ├── __init__.py
    └── models/
        ├── modeling_llama.py
        ├── modeling_mistral.py
        └── CompressUtils/
            ├── __init__.py
            ├── compress_class.py
            └── compress_function.py
```

## 主要文件作用

### `zipcache_generation_demo.py`

最小推理示例。它完成：

- 构造 `compress_config`；
- 加载 tokenizer；
- 读取 `asset/gsm8k_sample.txt` 作为 prompt；
- 加载 `MyLlamaForCausalLM`；
- 调用 `model.generate()`；
- 打印生成文本。

这里的 `compress_config` 是理解 ZipCache 的入口：

- `compress_mode`：Key cache 的压缩方式；
- `quantize_bit_important`：重要 Key token 使用的 bit 数；
- `quantize_bit_unimportant`：不重要 Key token 使用的 bit 数；
- `k_unimportant_ratio`：Key 中低重要性 token 的比例；
- `v_compress_mode`：Value cache 的压缩方式；
- `v_quantize_bit_important`：重要 Value token 使用的 bit 数；
- `v_quantize_bit_unimportant`：不重要 Value token 使用的 bit 数；
- `v_unimportant_ratio`：Value 中低重要性 token 的比例；
- `stream`：是否启用流式压缩；
- `streaming_gap`：每生成多少个 token 重新做一次压缩/重要性更新。

### `zipcache/models/CompressUtils/compress_function.py`

底层压缩和解压函数集合。它不保存状态，只负责把 tensor 量化/解量化。

主要功能：

- 2-bit / 4-bit 打包和解包；
- channel-wise 量化；
- token-wise 量化；
- group-wise 量化；
- important/unimportant 混合精度量化；
- GEAR 风格的低秩残差补偿；
- outlier 保留。

输入 KV tensor 通常是 `[batch, head, seq_len, head_dim]`。很多函数会临时 reshape 或 permute，以便按 token、channel 或 group 维度统计 min/max/scale。

### `zipcache/models/CompressUtils/compress_class.py`

压缩状态对象。它把 `compress_function.py` 中的无状态函数包装成可以放进 `past_key_value` 的对象。

主要类：

- `CompressUnion`：普通压缩对象，用于单一精度或非 mixed precision 模式。
- `MixedPrecisionCompressUnion`：混合精度压缩对象，保存 important/unimportant token 的索引和各自量化结果。

模型 attention 中的 `past_key_value` 原本是：

```python
(key_states, value_states)
```

开启 ZipCache 后会变成：

```python
(past_key_union, past_value_union)
```

其中 `past_key_union.decompress()` 会在下一步 attention 前把压缩 cache 解回 tensor。

### `zipcache/models/modeling_llama.py`

基于 HuggingFace LLaMA 模型代码改写。主要新增/改动点：

- `MixedLlamaAttention`：ZipCache 的主要 LLaMA attention 实现；
- `MyLlamaDecoderLayer`：使用 `MixedLlamaAttention` 的 decoder layer；
- `MyLlamaModel`：把每一层替换为 ZipCache 版本；
- `MyLlamaForCausalLM`：对外加载和生成使用的 CausalLM 类。

关键路径：

1. forward 中计算 Q/K/V；
2. 如果 `past_key_value` 已存在，先调用 `decompress()` 取回历史 KV；
3. prefill 阶段用 probe token 的 attention 分布识别不重要 token；
4. 使用 FlashAttention 计算真实 attention 输出；
5. 如果 `use_cache=True`，把新的 KV 压缩成 `MixedPrecisionCompressUnion` 并返回。

### `zipcache/models/modeling_mistral.py`

基于 HuggingFace Mistral 模型代码改写，结构和 LLaMA 文件类似。

主要类：

- `MixedMistralAttention`：带混合精度 KV cache 压缩的 Mistral attention；
- `MyMistralAttention`：另一版手写 attention 路径；
- `MyMistralDecoderLayer`；
- `MyMistralModel`；
- `MyMistralForCausalLM`。

注意：当前仓库文件中 `modeling_mistral.py` 引用了 `zipcache.utils.globalvar`，但仓库里没有对应文件。如果要运行 Mistral 路径，需要先补齐这个模块或移除相关统计代码。

### `zipcache/__init__.py`

包级导出文件。安装后可以直接：

```python
from zipcache import MyLlamaForCausalLM
```

### `setup.py`

Python 包安装脚本。`pip install -e .` 会把 `zipcache` 注册到当前环境中，方便 demo 或其他脚本 import。

## 推理数据流

以 LLaMA 为例：

```text
zipcache_generation_demo.py
  -> MyLlamaForCausalLM.from_pretrained(..., compress_config=...)
    -> MyLlamaModel
      -> MyLlamaDecoderLayer
        -> MixedLlamaAttention.forward()
          -> 计算当前 token 的 Q/K/V
          -> 解压历史 past_key_value
          -> 计算注意力输出
          -> 根据注意力识别 unimportant token
          -> 压缩新的 KV cache
          -> 返回新的 past_key_value
```

## 新手阅读建议

1. 先读 `zipcache_generation_demo.py`，理解如何配置和启动。
2. 再读 `compress_class.py`，理解压缩对象如何替代普通 KV tensor。
3. 再读 `compress_function.py`，理解 2-bit/4-bit 打包和 mixed precision 量化。
4. 最后读 `modeling_llama.py` 的 `MixedLlamaAttention.forward()`，这是 ZipCache 接入模型的核心。

## 运行方式

安装依赖：

```bash
pip install packaging ninja
pip install flash-attn --no-build-isolation
pip install -e .
```

修改 `zipcache_generation_demo.py` 中的 `MODEL_PATH`，然后运行：

```bash
python3 zipcache_generation_demo.py
```

## 当前实现限制

- demo 默认只处理 LLaMA 路径。
- Mistral 路径引用了缺失的 `zipcache.utils.globalvar`。
- 代码依赖 `flash_attn` 和特定版本 Transformers API。
- 压缩/解压是在 Python/PyTorch 层实现，重点验证算法效果，不等价于高度工程化的推理 runtime。
