from __future__ import annotations

"""ZipCache normal/2bit/4bit KV 的融合 Attention CUDA wrapper。"""

import functools
from typing import TYPE_CHECKING

import torch

from .utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from minisgl.attention.mixed import MixedAttnMetadata
    from tvm_ffi import Module


@functools.cache
def _jit_mixed_attention_module(head_dim: int, num_threads: int = 128) -> Module:
    """按 head_dim JIT 编译并缓存 correctness-first mixed Attention kernel。"""

    args = make_cpp_args(head_dim, num_threads)
    return load_jit(
        "mixed_attention",
        *args,
        cuda_files=["mixed_attention.cu"],
        cuda_wrappers=[("launch", f"MixedAttentionKernel<{args}>::run")],
    )


def mixed_paged_attention(
    q: torch.Tensor,
    normal_k: torch.Tensor,
    normal_v: torch.Tensor,
    metadata: MixedAttnMetadata,
    layer_id: int,
    *,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """直接读取 normal/compressed KV 并执行统一 causal Attention。

    输入：
    - q shape ``[Tq,Hq,D]``，FP16/BF16，已经应用 RoPE；
    - normal_k/normal_v shape ``[P,S,Hkv,D]``；
    - metadata 保存 normal indices、compressed pool 和当前层 segment descriptor；
    - layer_id 指定本次 transformer layer。

    输出 shape ``[Tq,Hq,D]``。return_lse=True 时额外返回自然对数域的
    FP32 LSE，shape ``[Tq,Hq]``。算子不会写 normal/compressed KV。
    """

    _validate_inputs(q, normal_k, normal_v, metadata)
    q = q.contiguous()
    head_dim = q.shape[2]
    num_kv_heads = normal_k.shape[2]
    flat_k = normal_k.view(-1, num_kv_heads, head_dim)
    flat_v = normal_v.view(-1, num_kv_heads, head_dim)
    output = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    if q.numel() != 0:
        layer = metadata.for_layer(layer_id)
        buffers = metadata.compressed_buffers
        module = _jit_mixed_attention_module(head_dim)
        module.launch(
            q,
            flat_k,
            flat_v,
            buffers.q4,
            buffers.q2,
            buffers.scale,
            buffers.ids,
            metadata.qo_indptr,
            metadata.q_positions,
            metadata.kv_lens,
            metadata.normal_indices,
            layer.segment_indptr,
            layer.segment_meta,
            layer.segment_offsets,
            output,
            lse,
        )
    return (output, lse) if return_lse else output


def _validate_inputs(
    q: torch.Tensor,
    normal_k: torch.Tensor,
    normal_v: torch.Tensor,
    metadata: MixedAttnMetadata,
) -> None:
    """在进入 JIT/FFI 前给出易理解的 shape、dtype 和 device 错误。"""

    if q.dim() != 3:
        raise ValueError(f"q must have shape [Tq,Hq,D], got {tuple(q.shape)}")
    if normal_k.shape != normal_v.shape or normal_k.dim() != 4:
        raise ValueError("normal_k/normal_v must have the same [P,S,Hkv,D] shape")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("mixed Attention supports FP16/BF16 q only")
    if normal_k.dtype != q.dtype or normal_v.dtype != q.dtype:
        raise TypeError("q and normal KV must use the same dtype")
    if normal_k.shape[-1] != q.shape[-1]:
        raise ValueError("q and normal KV head_dim must match")
    if q.shape[-1] not in (64, 128):
        raise ValueError("correctness-first mixed Attention currently supports D=64 or D=128")
    if normal_k.shape[2] <= 0 or q.shape[1] % normal_k.shape[2] != 0:
        raise ValueError("Hq must be divisible by Hkv for MHA/GQA")
    if metadata.has_compressed and (
        metadata.compressed_num_kv_heads != normal_k.shape[2]
        or metadata.compressed_head_dim != q.shape[2]
    ):
        raise ValueError(
            "compressed [Hkv,D] does not match normal KV: "
            f"compressed={[metadata.compressed_num_kv_heads, metadata.compressed_head_dim]}, "
            f"normal={[normal_k.shape[2], q.shape[2]]}"
        )
    tensors = (q, normal_k, normal_v)
    if any(t.device != metadata.compressed_buffers.device for t in tensors):
        raise ValueError("q, normal KV and compressed buffers must be on the same CUDA device")
    if not normal_k.is_contiguous() or not normal_v.is_contiguous():
        raise ValueError("normal KV cache must be contiguous; copying the full pool is not allowed")
    if metadata.q_positions.numel() != q.shape[0]:
        raise ValueError("q_positions length must equal Tq")
    if metadata.qo_indptr.numel() != metadata.kv_lens.numel() + 1:
        raise ValueError("qo_indptr must have shape [B+1] and kv_lens shape [B]")


__all__ = ["mixed_paged_attention"]
