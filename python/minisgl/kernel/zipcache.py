from __future__ import annotations

# ZipCache 专用 CUDA kernel wrapper。
#
# v4 不修改 FlashAttention / FlashInfer attention kernel，而是在 miniSGLang
# 自己的 JIT kernel 中完成 demote 侧压缩，以及 restore 侧恢复到 normal KV pool。

import functools
from typing import TYPE_CHECKING

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_ZIPCACHE_KERNEL_CONFIG = KernelConfig(num_threads=256, max_occupancy=1, use_pdl=False)


@functools.cache
def _jit_zipcache_quant_module(
    bit: int,
    storage_bit: int,
    *,
    config: KernelConfig = DEFAULT_ZIPCACHE_KERNEL_CONFIG,
) -> Module:
    """按量化 bit 和 packed bit 宽 JIT 编译并缓存 ZipCache quant kernel。"""

    if bit < 1 or bit > 4:
        raise ValueError(f"Unsupported ZipCache quant bit: {bit}")
    if storage_bit not in (2, 4) or bit > storage_bit:
        raise ValueError(f"Unsupported ZipCache storage bit: {storage_bit} for bit={bit}")
    args = make_cpp_args(bit, storage_bit, *config)
    return load_jit(
        "zipcache_quant",
        *args,
        cuda_files=["zipcache.cu"],
        cuda_wrappers=[("quantize", f"ZipCacheQuantizeKernel<{args}>::run")],
    )


@functools.cache
def _jit_zipcache_dequant_module(
    storage_bit: int,
    *,
    config: KernelConfig = DEFAULT_ZIPCACHE_KERNEL_CONFIG,
) -> Module:
    """按 packed bit 宽 JIT 编译并缓存 ZipCache dequant kernel。"""

    if storage_bit not in (2, 4):
        raise ValueError(f"Unsupported ZipCache storage bit: {storage_bit}")
    args = make_cpp_args(storage_bit, *config)
    return load_jit(
        "zipcache_dequant",
        *args,
        cuda_files=["zipcache.cu"],
        cuda_wrappers=[("dequantize", f"ZipCacheDequantizeKernel<{args}>::run")],
    )


def zipcache_quantize_part(
    src_cache: torch.Tensor,
    local_ids: torch.Tensor,
    q_packed: torch.Tensor,
    min_val: torch.Tensor,
    step: torch.Tensor,
    bit: int,
    storage_bit: int,
) -> None:
    """把一个 important/unimportant token group 量化并 packed 写入 compressed pool。

    Args:
        src_cache: 待压缩的单层 K 或 V，形状 [num_tokens, num_heads, head_dim]。
        local_ids: 当前 part 包含的 local token id。
        q_packed: compressed pool 中预先分配好的 uint8 packed 输出。
        min_val: compressed pool 中预先分配好的 fp16 min 输出。
        step: compressed pool 中预先分配好的 fp16 step 输出。
        bit: 实际量化 bit，例如 4 或 2。
        storage_bit: packed slot bit，支持 2 或 4。
    """

    if local_ids.numel() == 0:
        return
    module = _jit_zipcache_quant_module(int(bit), int(storage_bit))
    module.quantize(
        src_cache.view(src_cache.shape[0], -1),
        local_ids,
        q_packed,
        min_val,
        step,
    )


def zipcache_dequantize_part(
    out_cache: torch.Tensor,
    dst_indices: torch.Tensor,
    local_ids: torch.Tensor,
    q_packed: torch.Tensor,
    min_val: torch.Tensor,
    step: torch.Tensor,
    storage_bit: int,
) -> None:
    """把一个 important/unimportant quantized part 解压并写入 normal KV pool。

    Args:
        out_cache: 单层 K 或 V cache，形状可 view 成 [num_tokens, num_heads, head_dim]。
        dst_indices: 当前 compressed entry 每个 local token 对应的 normal KV 物理位置。
        local_ids: 当前 part 包含的 local token id，例如 important token ids。
        q_packed: 2bit/4bit packed 量化值。
        min_val: fp16 min，形状 [num_selected_tokens, num_heads, 1]。
        step: fp16 step，形状同 min_val。
        storage_bit: packed bit width，支持 2 或 4。
    """

    if local_ids.numel() == 0:
        return
    num_tokens = out_cache.shape[0]
    module = _jit_zipcache_dequant_module(int(storage_bit))
    module.dequantize(
        out_cache.view(num_tokens, -1),
        dst_indices,
        local_ids,
        q_packed,
        min_val,
        step,
    )
