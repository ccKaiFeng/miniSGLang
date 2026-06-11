import functools
import json
import os

# 这个文件封装 HuggingFace 相关操作：加载 tokenizer/config、下载权重。

from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.asyncio import tqdm
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

class DisabledTqdm(tqdm):
    """禁用 tqdm 进度条的 helper，用于安静下载。"""

    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    """加载模型对应的 tokenizer。

    如果 tokenizer 自身没有 chat_template，会尝试额外下载 chat_template.json。
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Some Mistral models store chat_template in a separate JSON file
    if not getattr(tokenizer, "chat_template", None):
        try:
            path = hf_hub_download(repo_id=model_path, filename="chat_template.json")
            with open(path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = json.load(f)["chat_template"]
        except Exception:
            pass
    return tokenizer


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    """加载 HuggingFace config，并用 cache 避免重复网络/磁盘读取。"""

    return AutoConfig.from_pretrained(model_path)


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    """返回一份新的 config 对象，避免调用方修改缓存对象本体。"""

    config = _load_hf_config(model_path)
    return type(config)(**config.to_dict())


def download_hf_weight(model_path: str) -> str:
    """如果 model_path 是本地目录则直接返回，否则从 HuggingFace 下载 safetensors。"""

    if os.path.isdir(model_path):
        return model_path
    try:
        return snapshot_download(
            model_path,
            allow_patterns=["*.safetensors"],
            tqdm_class=DisabledTqdm,
        )
    except Exception as e:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )
