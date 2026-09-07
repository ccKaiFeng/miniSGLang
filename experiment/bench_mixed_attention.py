from __future__ import annotations

"""Mixed KV Attention 的 correctness-first GPU 微基准。

用法示例：
    python experiment/bench_mixed_attention.py --kv-len 4096 --compressed-ratio 0.75

脚本只测算子，不代表完整服务吞吐。baseline 每次显式反量化 compressed KV，
目的是观察省去完整浮点中间张量后的量级差异。
"""

import argparse
from types import SimpleNamespace

import torch
from minisgl.attention.mixed import (
    CompressedKVSegment,
    MixedCompressedBuffers,
    NormalKVSegment,
    build_mixed_metadata,
)
from minisgl.benchmark.perf import perf_cuda
from minisgl.kernel import mixed_paged_attention
from minisgl.zipcache.manager import (
    _V3CompressedPool,
    _dequantize_mixed_gpu,
    _quantize_mixed_gpu,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-len", type=int, default=4096)
    parser.add_argument("--compressed-ratio", type=float, default=0.75)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--qo-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--repetitions", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if args.kv_len <= 0 or not 0.0 < args.compressed_ratio < 1.0:
        raise ValueError("kv-len must be positive and compressed-ratio must be in (0,1)")
    if args.qo_heads % args.kv_heads != 0:
        raise ValueError("qo-heads must be divisible by kv-heads")

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    compressed_len = max(1, min(args.kv_len - 1, int(args.kv_len * args.compressed_ratio)))
    normal_len = args.kv_len - compressed_len
    pool_bytes = max(
        16 * 1024 * 1024,
        compressed_len * args.kv_heads * args.head_dim * 4,
    )
    pool = _V3CompressedPool(
        total_bytes=pool_bytes,
        device=device,
        q4_ratio=0.4,
        q2_ratio=0.2,
        scale_ratio=0.25,
        ids_ratio=0.15,
    )

    torch.manual_seed(0)
    source_k = torch.randn(compressed_len, args.kv_heads, args.head_dim, device=device, dtype=dtype)
    source_v = torch.randn_like(source_k)
    unimportant_count = int(compressed_len * 0.4)
    unimportant = torch.arange(unimportant_count, device=device, dtype=torch.int64)
    quantized_k = _quantize_mixed_gpu(
        source_k, unimportant, important_bit=4, unimportant_bit=2, pool=pool
    )
    quantized_v = _quantize_mixed_gpu(
        source_v, unimportant, important_bit=4, unimportant_bit=2, pool=pool
    )
    entry = SimpleNamespace(
        length=compressed_len,
        layers={0: SimpleNamespace(k=quantized_k, v=quantized_v)},
    )

    normal_k = torch.randn(normal_len, 1, args.kv_heads, args.head_dim, device=device, dtype=dtype)
    normal_v = torch.randn_like(normal_k)
    normal_indices = torch.arange(normal_len, dtype=torch.int32, device=device)
    q = torch.randn(1, args.qo_heads, args.head_dim, device=device, dtype=dtype)
    position = torch.tensor([args.kv_len - 1], dtype=torch.int32, device=device)
    batch = SimpleNamespace(
        padded_reqs=[SimpleNamespace(extend_len=1, device_len=args.kv_len)],
        positions=position,
    )
    metadata = build_mixed_metadata(
        batch,
        [
            [
                CompressedKVSegment(0, entry),
                NormalKVSegment(compressed_len, normal_indices),
            ]
        ],
        compressed_buffers=MixedCompressedBuffers.from_pool(pool),
        num_layers=1,
    )

    group_size = args.qo_heads // args.kv_heads

    def baseline() -> torch.Tensor:
        restored_k = _dequantize_mixed_gpu(quantized_k, device)
        restored_v = _dequantize_mixed_gpu(quantized_v, device)
        full_k = torch.cat([restored_k, normal_k[:, 0]], dim=0).repeat_interleave(group_size, dim=1)
        full_v = torch.cat([restored_v, normal_v[:, 0]], dim=0).repeat_interleave(group_size, dim=1)
        scores = torch.einsum("qhd,khd->qhk", q.float(), full_k.float())
        probs = torch.softmax(scores * (args.head_dim**-0.5), dim=-1)
        return torch.einsum("qhk,khd->qhd", probs, full_v.float()).to(dtype)

    def mixed() -> torch.Tensor:
        return mixed_paged_attention(q, normal_k, normal_v, metadata, 0)

    expected = baseline()
    actual = mixed()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0.12, rtol=0.08)
    baseline_ms = perf_cuda(baseline, repetitions=args.repetitions, cuda_graph_repetitions=None)
    mixed_ms = perf_cuda(mixed, repetitions=args.repetitions, cuda_graph_repetitions=None)
    print(
        f"kv={args.kv_len} compressed={compressed_len} D={args.head_dim} "
        f"Hq/Hkv={args.qo_heads}/{args.kv_heads} dtype={args.dtype}"
    )
    print(f"materialize+torch baseline: {baseline_ms:.3f} ms")
    print(f"direct mixed kernel:        {mixed_ms:.3f} ms")


if __name__ == "__main__":
    main()
