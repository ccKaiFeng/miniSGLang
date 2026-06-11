from __future__ import annotations

# 这个文件负责“从 logits 选出下一个 token”。
#
# 模型 forward 的输出是 logits，可以理解为每个词的未归一化分数。
# Sampler 会根据用户的 SamplingParams，把 logits 转成下一个 token id。

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    """一个 batch 的采样参数张量。

    如果 temperatures=None，表示整个 batch 都是 greedy 解码，直接 argmax。
    否则 top_k/top_p/temperatures 会放到 GPU 上供 flashinfer sampling kernel 使用。
    """

    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """把 Python list 转成 GPU tensor。

    pin_memory=True 让 CPU 到 GPU 的异步拷贝更高效。
    """

    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    """实际采样实现。

    这里使用 flashinfer.sampling：
    1. softmax(logits, temperature) 得到概率；
    2. 根据 top_k/top_p 是否启用，选择对应采样 kernel。
    """

    import flashinfer.sampling as sampling

    # temperature 会影响概率分布的尖锐程度。
    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        # 不限制 top-k/top-p，直接从完整概率分布采样。
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        # 只从概率最高的 k 个 token 里采样。
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        # 只从累计概率达到 p 的 token 集合里采样。
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    # 同时启用 top-k 和 top-p。
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    """封装 batch 级采样流程。"""

    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        """把每个请求的 SamplingParams 整理成 GPU tensor。

        batch 中每个请求可以有不同 temperature/top_k/top_p。为了让采样 kernel
        一次处理整个 batch，需要把这些 Python 参数收集成 tensor。
        """

        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            # 全部请求都是 greedy，后续直接 argmax，不需要准备采样 tensor。
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6

        # 避免 temperature/top_p 变成 0 导致 kernel 中数值问题。
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]

        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None

        # 如果所有请求都等价于不限制 top_k，就不传 top_k tensor。
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)

        # 如果所有请求 top_p 都是 1.0，就不传 top_p tensor。
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        """根据 logits 和采样参数输出 next_token。"""

        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:  # greedy sampling
                # greedy：选择 logits 最大的 token。
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
