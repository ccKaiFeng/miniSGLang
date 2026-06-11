from __future__ import annotations

# 这个文件提供离线 Python API：LLM。
#
# 它继承 Scheduler，但不走 HTTP/ZMQ 多进程链路，而是在同一进程里直接提交
# prompts、调度、推理并返回结果。benchmark/offline 会使用这个接口。

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    UserMsg,
)
from minisgl.scheduler import Scheduler, SchedulerConfig


class RequestAllFinished(Exception):
    """离线模式下表示所有请求都处理完。"""

    pass


@dataclass
class RequestStatus:
    """离线请求的输入/输出 token 状态。"""

    uid: int
    input_ids: List[int]
    output_ids: List[int]


class LLM(Scheduler):
    """可直接在 Python 中调用的离线推理接口。"""

    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        """创建单进程、单 TP rank 的 Scheduler/Engine。"""

        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),
            dtype=dtype,
            offline_mode=True,
            **kwargs,
        )
        super().__init__(config)
        self.pending_requests: List[Tuple[List[int] | str, SamplingParams]] = []
        self.status_map: Dict[int, RequestStatus] = {}
        self.counter = 0

    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        """把字符串 prompt 或用户已给定的 token id list 转成 int32 tensor。"""

        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        else:
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """离线模式下模拟“从 tokenizer 收到 UserMsg”。"""

        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()
        results: List[BaseBackendMsg] = []
        added, sum_input_len = 0, 0
        for tokens_or_prompt, sampling_params in self.pending_requests:
            if sum_input_len >= self.prefill_budget:
                break
            input_ids = self._tokenize_one(tokens_or_prompt)
            sum_input_len += len(input_ids)
            uid, added = self.counter + added, added + 1
            results.append(UserMsg(uid=uid, input_ids=input_ids, sampling_params=sampling_params))
            self.status_map[uid] = RequestStatus(
                uid=uid,
                input_ids=(
                    input_ids.tolist() if isinstance(tokens_or_prompt, str) else tokens_or_prompt
                ),
                output_ids=[],
            )
        self.counter += added
        self.pending_requests = self.pending_requests[added:]
        return results

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        """离线模式下接收 scheduler 生成的 token，并记录到 status_map。"""

        for msg in reply:
            status = self.status_map[msg.uid]
            if not (msg.finished and msg.next_token == self.eos_token_id):
                status.output_ids.append(msg.next_token)

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List[SamplingParams] | SamplingParams,
    ) -> List[Dict[str, str | List[int]]]:
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.pending_requests.append((prompt, sp))
        try:
            self.run_forever()
        except RequestAllFinished:
            pass
        results: List[Dict[str, str | List[int]]] = []
        for i in range(len(prompts)):
            status = self.status_map[i]
            output_text = self.tokenizer.decode(status.output_ids)
            results.append({"text": output_text, "token_ids": status.output_ids})
        return results
