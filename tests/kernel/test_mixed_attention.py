from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.mixed import (
    CompressedKVSegment,
    MixedCompressedBuffers,
    NormalKVSegment,
    build_mixed_metadata,
)
from minisgl.kernel import mixed_paged_attention
from minisgl.zipcache.manager import (
    _V3CompressedPool,
    _dequantize_mixed_gpu,
    _quantize_mixed_gpu,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _make_compressed_pool() -> _V3CompressedPool:
    """创建足够覆盖非零 slice offset 的小型测试 pool。"""

    return _V3CompressedPool(
        total_bytes=1024 * 1024,
        device=torch.device("cuda"),
        q4_ratio=0.35,
        q2_ratio=0.25,
        scale_ratio=0.25,
        ids_ratio=0.15,
    )


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """用显式 FP32 softmax 计算与 mixed kernel 相同的 causal 参考结果。"""

    group_size = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group_size, dim=1).float()
    v = v.repeat_interleave(group_size, dim=1).float()
    scores = torch.einsum("qhd,khd->qhk", q.float(), k) * (q.shape[-1] ** -0.5)
    kv_positions = torch.arange(k.shape[0], device=q.device)
    mask = kv_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
    scores = scores.masked_fill(~mask.unsqueeze(1), -torch.inf)
    lse = torch.logsumexp(scores, dim=-1)
    output = torch.einsum("qhk,khd->qhd", torch.softmax(scores, dim=-1), v)
    return output.to(q.dtype), lse


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_mixed_attention_reads_normal_and_compressed_pool(
    dtype: torch.dtype, head_dim: int
) -> None:
    """验证 mixed bit、GQA、因果位置和非连续 normal index 的共同结果。"""

    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("This CUDA device does not support BF16")
    torch.manual_seed(7)
    device = torch.device("cuda")
    num_kv_heads = 2
    num_qo_heads = 4
    compressed_len = 3
    normal_len = 2
    pool = _make_compressed_pool()

    compressed_k = torch.randn(compressed_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    compressed_v = torch.randn_like(compressed_k)
    unimportant_ids = torch.tensor([1], dtype=torch.int64, device=device)
    quantized_k = _quantize_mixed_gpu(
        compressed_k,
        unimportant_ids,
        important_bit=4,
        unimportant_bit=2,
        pool=pool,
    )
    # K/V 的位宽配置故意不同，防止 kernel 错误地共用 storage_bit。
    quantized_v = _quantize_mixed_gpu(
        compressed_v,
        unimportant_ids,
        important_bit=2,
        unimportant_bit=4,
        pool=pool,
    )
    layer_entry = SimpleNamespace(k=quantized_k, v=quantized_v)
    entry = SimpleNamespace(length=compressed_len, layers={0: layer_entry})

    normal_k = torch.randn(8, 1, num_kv_heads, head_dim, device=device, dtype=dtype)
    normal_v = torch.randn_like(normal_k)
    normal_indices = torch.tensor([5, 2], dtype=torch.int32, device=device)
    suffix_k = torch.randn(normal_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    suffix_v = torch.randn_like(suffix_k)
    normal_k.view(-1, num_kv_heads, head_dim)[normal_indices.long()] = suffix_k
    normal_v.view(-1, num_kv_heads, head_dim)[normal_indices.long()] = suffix_v

    q_positions = torch.tensor([3, 4], dtype=torch.int32, device=device)
    q = torch.randn(2, num_qo_heads, head_dim, device=device, dtype=dtype)
    request = SimpleNamespace(extend_len=2, device_len=compressed_len + normal_len)
    batch = SimpleNamespace(padded_reqs=[request], positions=q_positions)
    sources = [
        [
            CompressedKVSegment(position_base=0, entry=entry),
            NormalKVSegment(position_base=compressed_len, indices=normal_indices),
        ]
    ]
    metadata = build_mixed_metadata(
        batch,
        sources,
        compressed_buffers=MixedCompressedBuffers.from_pool(pool),
        num_layers=1,
    )

    normal_k_before = normal_k.clone()
    normal_v_before = normal_v.clone()
    output, lse = mixed_paged_attention(
        q,
        normal_k,
        normal_v,
        metadata,
        layer_id=0,
        return_lse=True,
    )
    restored_k = _dequantize_mixed_gpu(quantized_k, device)
    restored_v = _dequantize_mixed_gpu(quantized_v, device)
    expected, expected_lse = _reference_attention(
        q,
        torch.cat([restored_k, suffix_k]),
        torch.cat([restored_v, suffix_v]),
        q_positions,
    )

    atol = 0.035 if dtype == torch.float16 else 0.12
    rtol = 0.02 if dtype == torch.float16 else 0.08
    torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(lse, expected_lse, atol=atol, rtol=rtol)
    # 独立 kernel 只读两个 pool；完整 KV 不应被恢复或写回 normal pool。
    torch.testing.assert_close(normal_k, normal_k_before, atol=0, rtol=0)
    torch.testing.assert_close(normal_v, normal_v_before, atol=0, rtol=0)


def test_mixed_attention_maps_multiple_requests() -> None:
    """验证 qo/segment indptr 能把扁平 query 映射到各自请求。"""

    from minisgl.attention.mixed import attach_mixed_kv_sources, maybe_prepare_mixed_metadata

    torch.manual_seed(11)
    device = torch.device("cuda")
    head_dim = 64
    num_kv_heads = 1
    num_qo_heads = 2
    pool = _make_compressed_pool()
    normal_k = torch.randn(10, 1, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    normal_v = torch.randn_like(normal_k)
    req0_indices = torch.tensor([6, 2], dtype=torch.int32, device=device)
    req1_indices = torch.tensor([7, 1, 8], dtype=torch.int32, device=device)
    q = torch.randn(2, num_qo_heads, head_dim, device=device, dtype=torch.float16)
    q_positions = torch.tensor([1, 2], dtype=torch.int32, device=device)
    batch = SimpleNamespace(
        padded_reqs=[
            SimpleNamespace(extend_len=1, device_len=2),
            SimpleNamespace(extend_len=1, device_len=3),
        ],
        positions=q_positions,
    )
    request_sources = [
        [NormalKVSegment(0, req0_indices)],
        [NormalKVSegment(0, req1_indices)],
    ]
    attach_mixed_kv_sources(
        batch,
        request_sources,
        compressed_pool=pool,
        num_layers=1,
    )
    assert maybe_prepare_mixed_metadata(batch)

    output = mixed_paged_attention(q, normal_k, normal_v, batch.attn_metadata, 0)
    flat_k = normal_k[:, 0]
    flat_v = normal_v[:, 0]
    expected0, _ = _reference_attention(
        q[:1], flat_k[req0_indices.long()], flat_v[req0_indices.long()], q_positions[:1]
    )
    expected1, _ = _reference_attention(
        q[1:], flat_k[req1_indices.long()], flat_v[req1_indices.long()], q_positions[1:]
    )
    torch.testing.assert_close(output, torch.cat([expected0, expected1]), atol=0.035, rtol=0.02)


def test_mixed_metadata_rejects_missing_or_overlapping_ranges() -> None:
    """防止 descriptor 重复读取或遗漏请求中的 KV token。"""

    pool = _make_compressed_pool()
    device = torch.device("cuda")
    request = SimpleNamespace(extend_len=1, device_len=4)
    batch = SimpleNamespace(
        padded_reqs=[request],
        positions=torch.tensor([3], dtype=torch.int32, device=device),
    )
    sources = [
        [
            NormalKVSegment(
                position_base=0,
                indices=torch.tensor([0, 1], dtype=torch.int32, device=device),
            ),
            NormalKVSegment(
                position_base=1,
                indices=torch.tensor([2, 3], dtype=torch.int32, device=device),
            ),
        ]
    ]
    with pytest.raises(ValueError, match="must partition"):
        build_mixed_metadata(
            batch,
            sources,
            compressed_buffers=MixedCompressedBuffers.from_pool(pool),
            num_layers=1,
        )
