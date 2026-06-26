# zipcache 包的对外入口。
#
# 安装本仓库后，用户通常只需要从这里导入改写后的模型类：
#   from zipcache import MyLlamaForCausalLM
#
# MyLlamaForCausalLM 继承 HuggingFace PreTrainedModel 的用法，但内部 decoder
# layer 使用 ZipCache 改写后的 attention，可以接收 compress_config。

from .models.modeling_llama import MyLlamaForCausalLM

__all__ = ["MyLlamaForCausalLM"]
