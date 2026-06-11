from dataclasses import dataclass
from typing import Dict, List

from minisgl.message import DetokenizeMsg
from transformers import PreTrainedTokenizerBase

# 这个文件负责“token id -> 文本”。
#
# 模型每次通常只生成一个 token id。为了流式返回给用户，系统需要不断把这些
# token id 解码成字符串片段。但这里有一个细节：有些 token 可能只表示半个词、
# 半个 unicode 字符，或者暂时还不能安全打印。因此本文件会维护每个请求的
# 解码状态，只把“可以显示”的增量文本发给前端。
#
# 部分逻辑借鉴自 sglang / transformers streamer。


def _is_chinese_char(cp: int):
    """判断一个 Unicode codepoint 是否是 CJK 汉字。"""

    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def find_printable_text(text: str):
    """找出当前可以安全打印的最长文本前缀。

    英文流式输出时，一个词可能被拆成多个 token。为了避免前端先显示半个词，
    这里通常只打印到最后一个空格为止。

    中文没有空格分词，所以 CJK 字符可以更积极地输出。
    """

    # Borrowed from https://github.com/huggingface/transformers/blob/061580c82c2db1de9139528243e105953793f7a2/src/transformers/generation/streamers.py#L99

    # After the symbol for a new line, we flush the cache.
    if text.endswith("\n"):
        return text
    # If the last token is a CJK character, we print the characters.
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # Otherwise if the penultimate token is a CJK character, we print the characters except for the last one.
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
    # which may change with the subsequent token -- there are probably smarter ways to do this!)
    else:
        return text[: text.rfind(" ") + 1]


@dataclass
class DecodeStatus:
    """单个用户请求的 detokenize 状态。

    同一个 uid 会持续收到多个 DetokenizeMsg。为了知道这次应该返回哪一段新文本，
    需要保存这个 uid 之前已经解码、已经发送到前端的位置。

    字段含义：
    - decoded_ids：目前累计收到的 token id；
    - decoded_str：目前已经确认可用的完整字符串；
    - read_offset：已经完整 decode 过的 token 数；
    - surr_offset：surrounding ids 的起点，用于处理 tokenizer 解码边界；
    - sent_offset：已经发送给前端的字符串长度。
    """

    decoded_ids: List[int]
    decoded_str: str
    read_offset: int  # length of read ids
    surr_offset: int  # length of surr ids
    sent_offset: int  # length of sent out string


class DetokenizeManager:
    """管理多个用户请求的增量 detokenize。"""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # uid -> DecodeStatus
        # 每个正在生成的请求都有一份 DecodeStatus；请求 finished 后删除。
        self.decode_map: Dict[int, DecodeStatus] = {}
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id

    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
        """把一批新 token 解码成一批增量文本。

        输入：
        - msgs：每条消息包含 uid、next_token、finished。

        输出：
        - List[str]：每个 uid 本次新增、可以发给前端显示的文本片段。
        """

        read_ids: List[List[int]] = []
        surr_ids: List[List[int]] = []
        for msg in msgs:
            if msg.uid not in self.decode_map:
                # 第一次看到这个 uid，创建解码状态。
                self.decode_map[msg.uid] = DecodeStatus(
                    decoded_ids=[],
                    decoded_str="",
                    read_offset=0,
                    surr_offset=0,
                    sent_offset=0,
                )
            s = self.decode_map[msg.uid]

            # 如果 finished 且 token 是 EOS，就不把 EOS 本身加入输出文本。
            # EOS 是“结束标记”，不是用户应该看到的文字。
            if not (msg.finished and msg.next_token == self.eos_token_id):
                s.decoded_ids.append(msg.next_token)

            # read_ids：从上次 surrounding offset 到当前末尾，表示需要重新 decode 的 token。
            read_ids.append(s.decoded_ids[s.surr_offset :])

            # surr_ids：上一次已经读过的一小段上下文，用来对齐新旧 decode 字符串。
            surr_ids.append(s.decoded_ids[s.surr_offset : s.read_offset])

        # batch_decode 一次性解码一批 token id list。
        read_texts = self.tokenizer.batch_decode(read_ids)
        surr_texts = self.tokenizer.batch_decode(surr_ids)

        incremental_strs: List[str] = []
        for msg, read_str, surr_str in zip(msgs, read_texts, surr_texts, strict=True):
            s = self.decode_map[msg.uid]

            # read_str 包含旧上下文 + 新 token 解出来的文本；
            # surr_str 是旧上下文解出来的文本；
            # 两者相减后就是这次新增的文本。
            new_text = read_str[len(surr_str) :]

            # Streaming chunk: update the decode status
            if len(new_text) > 0 and not new_text.endswith("�"):
                # 如果新文本不是损坏/不完整 unicode，就认为可以更新完整解码状态。
                output_str = s.decoded_str + new_text
                s.decoded_str = output_str
                s.surr_offset = s.read_offset
                s.read_offset = len(s.decoded_ids)
            else:
                # 如果末尾看起来不完整，只取可以安全显示的部分。
                new_text = find_printable_text(new_text)
                output_str = s.decoded_str + new_text

            # 只返回“还没有发给前端”的新增部分。
            incremental_output = output_str[s.sent_offset :]
            s.sent_offset = len(output_str)
            incremental_strs.append(incremental_output)

            if msg.finished:
                # 这个请求已经结束，释放 uid 对应的 detokenize 状态。
                del self.decode_map[msg.uid]

        return incremental_strs
