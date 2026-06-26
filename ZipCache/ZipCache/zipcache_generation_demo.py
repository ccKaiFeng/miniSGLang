import torch
from transformers import AutoTokenizer

from zipcache import MyLlamaForCausalLM

# 这个脚本演示如何加载 ZipCache 改写后的 LLaMA，并用一段 GSM8K prompt 做生成。
# 对新手来说，重点看 compress_config：它决定 KV cache 怎么区分重要/不重要
# token，以及分别用多少 bit 保存。

compress_config = {}

## Key compress config
# Key cache 的压缩策略。mixed_channelwiseQ 表示：
# - 先按注意力重要性把 token 分成 important / unimportant；
# - 两类 token 分别做 channel-wise 量化；
# - important 用较高 bit，unimportant 用较低 bit。
compress_config["compress_mode"] = "mixed_channelwiseQ"
compress_config["quantize_bit_important"] = 4
compress_config["quantize_bit_unimportant"] = 2
compress_config["k_unimportant_ratio"] = 0.4

## Value compress config
# Value cache 可以使用和 Key 不同的压缩方式。这里采用
# channel_separate_mixed_tokenwiseQ：先区分重要 token，再做带 channel scale 的
# token-wise 量化。
compress_config["v_compress_mode"] = "channel_separate_mixed_tokenwiseQ"
compress_config["v_quantize_bit_important"] = 4
compress_config["v_quantize_bit_unimportant"] = 2
compress_config["v_unimportant_ratio"] = 0.4
compress_config["stream"] = True # 开启流式压缩：decode 中不是每步都完整重压缩。
compress_config["streaming_gap"] = 100 # 每生成 N 个 token 重新压缩/更新一次。

MODEL_PATH='/data/models--meta-llama--Meta-Llama-3-8B/snapshots/1460c22666392e470910ce3d44ffeb2ab7dbd4df/' ## your llama path here

# tokenizer 负责把文本 prompt 转成 token id。local_files_only=True 表示只读本地
# 模型目录，不自动从 HuggingFace 下载。
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, use_fast=True, cache_dir=MODEL_PATH, local_files_only=True
)
with open('asset/gsm8k_sample.txt', 'r') as file:
    prompt_text = file.read()
input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').input_ids.cuda()

if 'Llama' in MODEL_PATH:
    # MyLlamaForCausalLM 是仓库改写后的模型类。compress_config 会一路传到每层
    # attention，使 past_key_value 使用压缩对象保存。
    model = MyLlamaForCausalLM.from_pretrained(
        MODEL_PATH,
        cache_dir=MODEL_PATH,
        compress_config=compress_config,
        torch_dtype=torch.float16,
        local_files_only=True
    )
else:
    raise NotImplementedError

model.half().eval().cuda()

# generate_kwargs 是 HuggingFace generate() 的常规参数。use_cache=True 很关键：
# 只有开启 KV cache，ZipCache 的压缩 past_key_value 路径才会被使用。
generate_kwargs = dict(
    return_dict_in_generate=False,
    max_new_tokens=128,
    output_scores=False,
    pad_token_id=tokenizer.eos_token_id,
    use_cache=True,
)

generate_kwargs["do_sample"] = False
generate_kwargs["temperature"] = None
generate_kwargs["top_k"] = None
generate_kwargs["top_p"] = None

# generate() 会循环调用模型 forward。第一次长 prompt 是 prefill，后续逐 token
# decode。ZipCache 在这两个阶段分别做重要 token 识别和压缩缓存复用。
generate_ids = model.generate(input_ids, **generate_kwargs)
result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("################## Generated Context with Our Cache ###################")
print(result)
