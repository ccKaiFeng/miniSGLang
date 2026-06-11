from __future__ import annotations

# 这个文件负责“文本 -> token id”。
#
# 大模型不能直接处理字符串，它只能处理数字 token。Tokenizer 的工作就是：
#   用户输入的 prompt/messages  ->  一串整数 token id
#
# 本文件只做前处理，不跑模型。

from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    """封装 HuggingFace tokenizer 的 tokenize 逻辑。"""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """保存具体模型对应的 tokenizer。

        不同模型可能使用不同的分词规则和 chat template，所以这里不自己实现
        分词算法，而是复用 HuggingFace tokenizer。
        """

        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        """把一批 TokenizeMsg 转成一批 token id tensor。

        输入：
        - msgs：多个用户请求，每个请求里有 uid、text、sampling_params。

        输出：
        - List[torch.Tensor]：每个请求对应一个 CPU 侧一维 int32 tensor。
        """

        results: List[torch.Tensor] = []

        # TODO: batch tokenization
        # 当前实现是逐条请求 tokenize。TODO 表示以后可以把多条 prompt
        # 一次性送给 tokenizer，提高吞吐。
        for msg in msgs:
            if isinstance(msg.text, list):
                # OpenAI chat-completions 传入的是 messages 列表，例如：
                # [{"role": "user", "content": "..."}]
                # apply_chat_template 会把它拼成模型训练时习惯的对话 prompt。
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                assert isinstance(prompt, str)
            else:
                # 普通字符串 prompt 直接使用。
                prompt = msg.text

            # encode(..., return_tensors="pt") 会返回形状类似 [1, seq_len] 的 tensor。
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="pt")
            )

            # 后端希望拿到一维 int32 token id，所以这里 view(-1) 展平成 [seq_len]。
            results.append(input_ids.view(-1).to(torch.int32))
        return results
