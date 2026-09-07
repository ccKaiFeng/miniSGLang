#include <minisgl/mixed_kv.cuh>
#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

struct MixedAttentionParams {
  // q/output: [Tq,Hq,D]。normal K/V: [normal_capacity,Hkv,D]。
  const void *__restrict__ q;
  const void *__restrict__ normal_k;
  const void *__restrict__ normal_v;
  const std::uint8_t *__restrict__ q4;
  const std::uint8_t *__restrict__ q2;
  // scale 保存所有 QuantizedPart 的 FP16 min/step；ids 保存 node 内相对位置。
  const __half *__restrict__ scale;
  const std::int64_t *__restrict__ ids;
  const std::int32_t *__restrict__ qo_indptr;
  const std::int32_t *__restrict__ q_positions;
  const std::int32_t *__restrict__ kv_lens;
  const std::int32_t *__restrict__ normal_indices;
  // segment_indptr: [B+1]；segment_meta: [G,5]；segment_offsets: [G,8]。
  const std::int32_t *__restrict__ segment_indptr;
  const std::int32_t *__restrict__ segment_meta;
  const std::int64_t *__restrict__ segment_offsets;
  void *__restrict__ output;
  float *__restrict__ lse;
  std::int32_t total_queries;
  std::int32_t num_qo_heads;
  std::int32_t num_kv_heads;
  std::int32_t num_requests;
  std::int64_t normal_capacity;
};

template <typename DType, int kHeadDim, int kNumThreads>
__global__ __launch_bounds__(kNumThreads) void mixed_attention_kernel(
    const __grid_constant__ MixedAttentionParams params) {
  static_assert(kHeadDim > 0 && kHeadDim <= 256);
  static_assert(kNumThreads == 64 || kNumThreads == 128 || kNumThreads == 256);

  // 一个 CTA 负责一个 (query token, query head)。这是正确性优先的实现，
  // 后续可以在保持 descriptor 协议不变的前提下换成 Tensor Core tile。
  const int query_idx = static_cast<int>(blockIdx.x);
  const int qo_head_idx = static_cast<int>(blockIdx.y);
  const int tid = static_cast<int>(threadIdx.x);
  if (query_idx >= params.total_queries || qo_head_idx >= params.num_qo_heads) {
    return;
  }

  __shared__ float reduction[kNumThreads];
  __shared__ float shared_alpha;
  __shared__ float shared_beta;
  __shared__ float shared_inv_l;
  __shared__ int shared_request;
  __shared__ int shared_segment_meta[5];
  __shared__ std::int64_t shared_segment_offsets[8];

  if (tid == 0) {
    // q 已按请求连续拼接。通过 qo_indptr 二分查找当前 query 属于哪个请求；
    // 这样 kernel 不需要额外的 [Tq] query_to_request buffer。
    int low = 0;
    int high = params.num_requests;
    while (low + 1 < high) {
      const int mid = (low + high) / 2;
      if (query_idx < params.qo_indptr[mid]) {
        high = mid;
      } else {
        low = mid;
      }
    }
    shared_request = low;
  }
  __syncthreads();

  const int request_idx = shared_request;
  const int q_position = params.q_positions[query_idx];
  const int kv_len = params.kv_lens[request_idx];
  const int group_size = params.num_qo_heads / params.num_kv_heads;
  const int kv_head_idx = qo_head_idx / group_size;
  const auto *q = static_cast<const DType *>(params.q);
  auto *output = static_cast<DType *>(params.output);
  const auto *normal_k = static_cast<const DType *>(params.normal_k);
  const auto *normal_v = static_cast<const DType *>(params.normal_v);

  constexpr int kValuesPerThread =
      (kHeadDim + kNumThreads - 1) / kNumThreads;
  // Q 只读取一次并留在寄存器。每个线程持有 D 维中的若干元素，同时持有
  // 对应的 FP32 output accumulator。
  float q_fragment[kValuesPerThread];
  float accumulator[kValuesPerThread];
#pragma unroll
  for (int item = 0; item < kValuesPerThread; ++item) {
    const int dim = tid + item * kNumThreads;
    if (dim < kHeadDim) {
      const std::int64_t offset =
          (std::int64_t(query_idx) * params.num_qo_heads + qo_head_idx) *
              kHeadDim +
          dim;
      q_fragment[item] = mixed_kv::to_float(q[offset]);
    } else {
      q_fragment[item] = 0.0f;
    }
    accumulator[item] = 0.0f;
  }

  float row_max = -CUDART_INF_F;
  float row_sum = 0.0f;
  const int segment_begin = params.segment_indptr[request_idx];
  const int segment_end = params.segment_indptr[request_idx + 1];

  for (int segment_idx = segment_begin; segment_idx < segment_end;
       ++segment_idx) {
    // descriptor 对 CTA 内所有线程相同，先由前几个线程搬入 shared memory。
    if (tid < 5) {
      shared_segment_meta[tid] = params.segment_meta[segment_idx * 5 + tid];
    }
    if (tid < 8) {
      shared_segment_offsets[tid] =
          params.segment_offsets[segment_idx * 8 + tid];
    }
    __syncthreads();

    const int kind = shared_segment_meta[0];
    const int count = shared_segment_meta[1];
    const int position_base = shared_segment_meta[2];
    const int k_storage_bit = shared_segment_meta[3];
    const int v_storage_bit = shared_segment_meta[4];
    const std::int64_t k_q_offset =
        shared_segment_offsets[mixed_kv::kKQOffset];
    const std::int64_t k_min_offset =
        shared_segment_offsets[mixed_kv::kKMinOffset];
    const std::int64_t k_step_offset =
        shared_segment_offsets[mixed_kv::kKStepOffset];
    const std::int64_t v_q_offset =
        shared_segment_offsets[mixed_kv::kVQOffset];
    const std::int64_t v_min_offset =
        shared_segment_offsets[mixed_kv::kVMinOffset];
    const std::int64_t v_step_offset =
        shared_segment_offsets[mixed_kv::kVStepOffset];
    const std::int64_t ids_offset =
        shared_segment_offsets[mixed_kv::kIdsOffset];
    const std::int64_t normal_index_offset =
        shared_segment_offsets[mixed_kv::kNormalIndexOffset];

    for (int token_idx = 0; token_idx < count; ++token_idx) {
      const std::int64_t physical_idx =
          kind == mixed_kv::kNormalSegment
              ? params.normal_indices[normal_index_offset + token_idx]
              : 0;
      const bool normal_address_valid =
          kind != mixed_kv::kNormalSegment ||
          (physical_idx >= 0 && physical_idx < params.normal_capacity);
      const std::int64_t token_head_base =
          (std::int64_t(token_idx) * params.num_kv_heads + kv_head_idx) *
          kHeadDim;
      const std::int64_t normal_head_base =
          (physical_idx * params.num_kv_heads + kv_head_idx) * kHeadDim;

      float partial_score = 0.0f;
#pragma unroll
      for (int item = 0; item < kValuesPerThread; ++item) {
        const int dim = tid + item * kNumThreads;
        if (dim < kHeadDim) {
          const float key =
              kind == mixed_kv::kNormalSegment
                  ? (normal_address_valid
                         ? mixed_kv::to_float(normal_k[normal_head_base + dim])
                         : 0.0f)
                  : mixed_kv::to_float(mixed_kv::from_float<DType>(
                        mixed_kv::load_quantized(
                            params.q4, params.q2, params.scale, k_q_offset,
                            k_min_offset, k_step_offset, k_storage_bit,
                            token_head_base + dim,
                            std::int64_t(token_idx) * params.num_kv_heads +
                                kv_head_idx)));
          partial_score += q_fragment[item] * key;
        }
      }
      reduction[tid] = partial_score;
      __syncthreads();
      // 将各线程负责的 D 维点积归约成一个 QK score。
      for (int stride = kNumThreads / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
          reduction[tid] += reduction[tid + stride];
        }
        __syncthreads();
      }

      if (tid == 0) {
        const std::int64_t kv_position =
            kind == mixed_kv::kNormalSegment
                ? std::int64_t(position_base) + token_idx
                : std::int64_t(position_base) +
                      params.ids[ids_offset + token_idx];
        const bool visible = normal_address_valid && kv_position >= 0 &&
                             kv_position < kv_len && kv_position <= q_position;
        if (visible) {
          // normal、4bit important、2bit unimportant 共享同一组在线 softmax
          // 状态。这里不能分别归一化后再直接相加。
          const float score = reduction[0] * rsqrtf(float(kHeadDim));
          const float next_max = fmaxf(row_max, score);
          shared_alpha = isfinite(row_max) ? expf(row_max - next_max) : 0.0f;
          shared_beta = expf(score - next_max);
          row_sum = row_sum * shared_alpha + shared_beta;
          row_max = next_max;
        } else {
          shared_alpha = 1.0f;
          shared_beta = 0.0f;
        }
      }
      __syncthreads();

      const float alpha = shared_alpha;
      const float beta = shared_beta;
#pragma unroll
      for (int item = 0; item < kValuesPerThread; ++item) {
        const int dim = tid + item * kNumThreads;
        if (dim < kHeadDim) {
          float value = 0.0f;
          if (beta != 0.0f) {
            value =
                kind == mixed_kv::kNormalSegment
                    ? (normal_address_valid
                           ? mixed_kv::to_float(normal_v[normal_head_base + dim])
                           : 0.0f)
                    : mixed_kv::to_float(mixed_kv::from_float<DType>(
                          mixed_kv::load_quantized(
                              params.q4, params.q2, params.scale, v_q_offset,
                              v_min_offset, v_step_offset, v_storage_bit,
                              token_head_base + dim,
                              std::int64_t(token_idx) * params.num_kv_heads +
                                  kv_head_idx)));
          }
          accumulator[item] = accumulator[item] * alpha + beta * value;
        }
      }
      __syncthreads();
    }
  }

  if (tid == 0) {
    if (row_sum > 0.0f) {
      shared_inv_l = 1.0f / row_sum;
      params.lse[std::int64_t(query_idx) * params.num_qo_heads + qo_head_idx] =
          row_max + logf(row_sum);
    } else {
      shared_inv_l = 0.0f;
      params.lse[std::int64_t(query_idx) * params.num_qo_heads + qo_head_idx] =
          -CUDART_INF_F;
    }
  }
  __syncthreads();

#pragma unroll
  for (int item = 0; item < kValuesPerThread; ++item) {
    const int dim = tid + item * kNumThreads;
    if (dim < kHeadDim) {
      const std::int64_t offset =
          (std::int64_t(query_idx) * params.num_qo_heads + qo_head_idx) *
              kHeadDim +
          dim;
      output[offset] =
          mixed_kv::from_float<DType>(accumulator[item] * shared_inv_l);
    }
  }
}

template <int head_dim, int num_threads = 128> struct MixedAttentionKernel {
  static void run(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView normal_k,
      const tvm::ffi::TensorView normal_v,
      const tvm::ffi::TensorView q4,
      const tvm::ffi::TensorView q2,
      const tvm::ffi::TensorView scale,
      const tvm::ffi::TensorView ids,
      const tvm::ffi::TensorView qo_indptr,
      const tvm::ffi::TensorView q_positions,
      const tvm::ffi::TensorView kv_lens,
      const tvm::ffi::TensorView normal_indices,
      const tvm::ffi::TensorView segment_indptr,
      const tvm::ffi::TensorView segment_meta,
      const tvm::ffi::TensorView segment_offsets,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView lse) {
    using namespace host;
    auto Tq = SymbolicSize{"Tq"};
    auto Hq = SymbolicSize{"Hq"};
    auto Hkv = SymbolicSize{"Hkv"};
    auto NnormalPool = SymbolicSize{"NnormalPool"};
    auto NnormalIndices = SymbolicSize{"NnormalIndices"};
    auto B = SymbolicSize{"B"};
    auto BplusOne = SymbolicSize{"BplusOne"};
    auto G = SymbolicSize{"G"};
    auto dtype = SymbolicDType{};
    auto scale_dtype = SymbolicDType{};
    auto device = SymbolicDevice{};

    TensorMatcher({Tq, Hq, head_dim})
        .with_dtype(dtype)
        .with_device<kDLCUDA>(device)
        .verify(q)
        .verify(output);
    TensorMatcher({NnormalPool, Hkv, head_dim})
        .with_dtype(dtype)
        .with_device<kDLCUDA>(device)
        .verify(normal_k)
        .verify(normal_v);
    TensorMatcher({-1})
        .with_dtype<std::uint8_t>()
        .with_device<kDLCUDA>(device)
        .verify(q4)
        .verify(q2);
    TensorMatcher({-1})
        .with_dtype(scale_dtype)
        .with_device<kDLCUDA>(device)
        .verify(scale);
    TensorMatcher({-1})
        .with_dtype<std::int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(ids);
    TensorMatcher({BplusOne})
        .with_dtype<std::int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(qo_indptr)
        .verify(segment_indptr);
    TensorMatcher({Tq})
        .with_dtype<std::int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(q_positions);
    TensorMatcher({B})
        .with_dtype<std::int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(kv_lens);
    TensorMatcher({NnormalIndices})
        .with_dtype<std::int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(normal_indices);
    TensorMatcher({G, 5})
        .with_dtype<std::int32_t>()
        .with_device<kDLCUDA>(device)
        .verify(segment_meta);
    TensorMatcher({G, 8})
        .with_dtype<std::int64_t>()
        .with_device<kDLCUDA>(device)
        .verify(segment_offsets);
    TensorMatcher({Tq, Hq})
        .with_dtype<float>()
        .with_device<kDLCUDA>(device)
        .verify(lse);

    const auto value_dtype = dtype.unwrap();
    const bool is_fp16 = value_dtype.code == DLDataTypeCode::kDLFloat &&
                         value_dtype.bits == 16 && value_dtype.lanes == 1;
    const bool is_bf16 = value_dtype.code == DLDataTypeCode::kDLBfloat &&
                         value_dtype.bits == 16 && value_dtype.lanes == 1;
    RuntimeCheck(is_fp16 || is_bf16,
                 "MixedAttentionKernel supports FP16/BF16 only");
    const auto scales_dtype = scale_dtype.unwrap();
    RuntimeCheck(scales_dtype.code == DLDataTypeCode::kDLFloat &&
                     scales_dtype.bits == 16 && scales_dtype.lanes == 1,
                 "MixedAttentionKernel scale must be FP16");
    RuntimeCheck(BplusOne.unwrap() == B.unwrap() + 1,
                 "qo/segment indptr must have shape [B+1]");
    RuntimeCheck(Hkv.unwrap() > 0 && Hq.unwrap() % Hkv.unwrap() == 0,
                 "Hq must be divisible by Hkv");
    RuntimeCheck(Tq.unwrap() <= std::numeric_limits<std::int32_t>::max(),
                 "Tq exceeds int32 range");
    RuntimeCheck(Hq.unwrap() <= 65535, "Hq exceeds CUDA grid.y limit");

    if (Tq.unwrap() == 0) {
      return;
    }
    const auto params = MixedAttentionParams{
        .q = q.data_ptr(),
        .normal_k = normal_k.data_ptr(),
        .normal_v = normal_v.data_ptr(),
        .q4 = static_cast<const std::uint8_t *>(q4.data_ptr()),
        .q2 = static_cast<const std::uint8_t *>(q2.data_ptr()),
        .scale = static_cast<const __half *>(scale.data_ptr()),
        .ids = static_cast<const std::int64_t *>(ids.data_ptr()),
        .qo_indptr = static_cast<const std::int32_t *>(qo_indptr.data_ptr()),
        .q_positions =
            static_cast<const std::int32_t *>(q_positions.data_ptr()),
        .kv_lens = static_cast<const std::int32_t *>(kv_lens.data_ptr()),
        .normal_indices =
            static_cast<const std::int32_t *>(normal_indices.data_ptr()),
        .segment_indptr =
            static_cast<const std::int32_t *>(segment_indptr.data_ptr()),
        .segment_meta =
            static_cast<const std::int32_t *>(segment_meta.data_ptr()),
        .segment_offsets =
            static_cast<const std::int64_t *>(segment_offsets.data_ptr()),
        .output = output.data_ptr(),
        .lse = static_cast<float *>(lse.data_ptr()),
        .total_queries = static_cast<std::int32_t>(Tq.unwrap()),
        .num_qo_heads = static_cast<std::int32_t>(Hq.unwrap()),
        .num_kv_heads = static_cast<std::int32_t>(Hkv.unwrap()),
        .num_requests = static_cast<std::int32_t>(B.unwrap()),
        .normal_capacity = NnormalPool.unwrap(),
    };
    const dim3 grid(static_cast<unsigned>(Tq.unwrap()),
                    static_cast<unsigned>(Hq.unwrap()));
    if (is_fp16) {
      LaunchKernel(grid, num_threads, device.unwrap())(
          mixed_attention_kernel<__half, head_dim, num_threads>, params);
    } else {
      LaunchKernel(grid, num_threads, device.unwrap())(
          mixed_attention_kernel<__nv_bfloat16, head_dim, num_threads>, params);
    }
  }
};

} // namespace
