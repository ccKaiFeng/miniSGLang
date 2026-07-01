#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

struct ZipCacheQuantizeParams {
  const void *__restrict__ src_cache;
  const void *__restrict__ local_ids;
  std::uint8_t *__restrict__ q_packed;
  __half *__restrict__ min_val;
  __half *__restrict__ step;
  std::int64_t src_stride;
  std::int64_t src_len;
  std::int64_t num_ids;
  std::int64_t hidden_size;
  std::int64_t num_heads;
  std::int64_t head_dim;
  std::int64_t total_elements;
  std::int64_t total_packed;
};

struct ZipCacheDequantizeParams {
  void *__restrict__ out_cache;
  const void *__restrict__ dst_indices;
  const void *__restrict__ local_ids;
  const std::uint8_t *__restrict__ q_packed;
  const __half *__restrict__ min_val;
  const __half *__restrict__ step;
  std::int64_t out_stride;
  std::int64_t dst_len;
  std::int64_t num_ids;
  std::int64_t hidden_size;
  std::int64_t num_heads;
  std::int64_t head_dim;
  std::int64_t total_elements;
};

template <typename T> __device__ __forceinline__ float cast_to_float(T value);

template <> __device__ __forceinline__ float cast_to_float<__half>(__half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float cast_to_float<__nv_bfloat16>(
    __nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T> __device__ __forceinline__ T cast_from_float(float value);

template <> __device__ __forceinline__ __half cast_from_float<__half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 cast_from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <std::size_t kBit, typename InT, typename IdT>
__global__ void zipcache_quant_scale_kernel(
    const __grid_constant__ ZipCacheQuantizeParams params) {
  const auto idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto total_scale = params.num_ids * params.num_heads;
  if (idx >= total_scale) {
    return;
  }

  const auto token_group = idx / params.num_heads;
  const auto head_id = idx - token_group * params.num_heads;
  const auto local_token = static_cast<const IdT *>(params.local_ids)[token_group];
  if (local_token < 0 || local_token >= params.src_len) {
    return;
  }

  const auto base = local_token * params.src_stride + head_id * params.head_dim;
  const auto *src = static_cast<const InT *>(params.src_cache);
  auto min_f = INFINITY;
  auto max_f = -INFINITY;
  for (std::int64_t d = 0; d < params.head_dim; ++d) {
    const auto value = cast_to_float<InT>(src[base + d]);
    min_f = fminf(min_f, value);
    max_f = fmaxf(max_f, value);
  }

  constexpr auto qmax = static_cast<float>((1 << kBit) - 1);
  const auto step_f = fmaxf((max_f - min_f) / qmax, 1.0e-6f);
  const auto scale_offset = token_group * params.num_heads + head_id;
  params.min_val[scale_offset] = __float2half_rn(min_f);
  params.step[scale_offset] = __float2half_rn(step_f);
}

template <std::size_t kBit, std::size_t kStorageBit, typename InT, typename IdT>
__global__ void zipcache_quant_pack_kernel(
    const __grid_constant__ ZipCacheQuantizeParams params) {
  const auto byte_idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (byte_idx >= params.total_packed) {
    return;
  }

  constexpr auto kValuesPerByte = 8 / kStorageBit;
  constexpr auto kMask = (1 << kStorageBit) - 1;
  constexpr auto kQMax = (1 << kBit) - 1;
  const auto *src = static_cast<const InT *>(params.src_cache);
  std::uint8_t packed = 0;

  for (std::int64_t lane = 0; lane < static_cast<std::int64_t>(kValuesPerByte);
       ++lane) {
    const auto logical_idx = byte_idx * kValuesPerByte + lane;
    if (logical_idx >= params.total_elements) {
      continue;
    }

    const auto token_group = logical_idx / params.hidden_size;
    const auto hidden_offset = logical_idx - token_group * params.hidden_size;
    const auto head_id = hidden_offset / params.head_dim;
    const auto local_token = static_cast<const IdT *>(params.local_ids)[token_group];
    if (local_token < 0 || local_token >= params.src_len) {
      continue;
    }

    const auto scale_offset = token_group * params.num_heads + head_id;
    const auto min_f = __half2float(params.min_val[scale_offset]);
    const auto step_f = __half2float(params.step[scale_offset]);
    const auto value =
        cast_to_float<InT>(src[local_token * params.src_stride + hidden_offset]);
    auto q = static_cast<int>(roundf((value - min_f) / step_f));
    q = q < 0 ? 0 : (q > kQMax ? kQMax : q);
    packed |= static_cast<std::uint8_t>((q & kMask) << (lane * kStorageBit));
  }

  params.q_packed[byte_idx] = packed;
}

template <std::size_t kStorageBit>
__device__ __forceinline__ std::uint8_t unpack_lowbit(const std::uint8_t *q,
                                                      std::int64_t logical_idx) {
  constexpr auto kValuesPerByte = 8 / kStorageBit;
  constexpr auto kMask = (1 << kStorageBit) - 1;
  const auto packed = q[logical_idx / kValuesPerByte];
  const auto shift = static_cast<unsigned>((logical_idx % kValuesPerByte) * kStorageBit);
  return static_cast<std::uint8_t>((packed >> shift) & kMask);
}

template <std::size_t kStorageBit, typename OutT, typename DstIndexT, typename IdT>
__global__ void zipcache_dequantize_kernel(
    const __grid_constant__ ZipCacheDequantizeParams params) {
  const auto idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= params.total_elements) {
    return;
  }

  const auto token_group = idx / params.hidden_size;
  const auto hidden_offset = idx - token_group * params.hidden_size;
  const auto head_id = hidden_offset / params.head_dim;
  const auto local_token = static_cast<const IdT *>(params.local_ids)[token_group];

  if (local_token < 0 || local_token >= params.dst_len) {
    return;
  }

  const auto dst_token = static_cast<const DstIndexT *>(params.dst_indices)[local_token];
  const auto scale_offset = token_group * params.num_heads + head_id;
  const auto q = static_cast<float>(unpack_lowbit<kStorageBit>(params.q_packed, idx));
  const auto min_value = __half2float(params.min_val[scale_offset]);
  const auto step_value = __half2float(params.step[scale_offset]);
  const auto value = q * step_value + min_value;

  auto *out = static_cast<OutT *>(params.out_cache);
  out[dst_token * params.out_stride + hidden_offset] = cast_from_float<OutT>(value);
}

template <std::size_t bit,         // actual quantization bit, <= storage_bit
          std::size_t storage_bit, // 2 or 4
          std::size_t num_threads = 256, std::size_t max_concurrency = 1,
          bool use_pdl = false>
struct ZipCacheQuantizeKernel {
  static void run(const tvm::ffi::TensorView src_cache,
                  const tvm::ffi::TensorView local_ids,
                  const tvm::ffi::TensorView q_packed,
                  const tvm::ffi::TensorView min_val,
                  const tvm::ffi::TensorView step) {
    using namespace host;
    static_assert(bit >= 1 && bit <= 4);
    static_assert(storage_bit == 2 || storage_bit == 4);
    static_assert(bit <= storage_bit);

    auto S = SymbolicSize{"S"};
    auto X = SymbolicSize{"X"};
    auto L = SymbolicSize{"L"};
    auto Q = SymbolicSize{"Q"};
    auto H = SymbolicSize{"H"};
    auto one = SymbolicSize{"1"};
    one.set_value(1);
    auto src_stride = SymbolicSize{"src_stride"};
    auto src_dtype = SymbolicDType{};
    auto id_dtype = SymbolicDType{};
    auto q_dtype = SymbolicDType{};
    auto scale_dtype = SymbolicDType{};
    auto device = SymbolicDevice{};

    TensorMatcher({S, X})
        .with_strides({src_stride, 1})
        .with_device<kDLCUDA>(device)
        .with_dtype(src_dtype)
        .verify(src_cache);
    TensorMatcher({L})
        .with_device<kDLCUDA>(device)
        .with_dtype<int32_t, int64_t>(id_dtype)
        .verify(local_ids);
    TensorMatcher({Q})
        .with_device<kDLCUDA>(device)
        .with_dtype<std::uint8_t>(q_dtype)
        .verify(q_packed);
    TensorMatcher({L, H, one})
        .with_strides({H, 1, 1})
        .with_device<kDLCUDA>(device)
        .with_dtype(scale_dtype)
        .verify(min_val)
        .verify(step);

    const auto src_dt = src_dtype.unwrap();
    const auto scale_dt = scale_dtype.unwrap();
    constexpr auto kDLBfloatCode = static_cast<DLDataTypeCode>(4);
    const auto is_bf16 = src_dt.code == kDLBfloatCode && src_dt.bits == 16;
    const auto is_fp16 = src_dt.code == DLDataTypeCode::kDLFloat && src_dt.bits == 16;
    RuntimeCheck(is_fp16 || is_bf16, "ZipCache v4 source must be fp16 or bf16");
    RuntimeCheck(scale_dt.code == DLDataTypeCode::kDLFloat && scale_dt.bits == 16,
                 "ZipCache v4 min/step tensors must be fp16");

    const auto hidden_size = X.unwrap();
    const auto num_heads = H.unwrap();
    RuntimeCheck(hidden_size % num_heads == 0,
                 "hidden_size must be divisible by num_heads");
    const auto head_dim = hidden_size / num_heads;
    const auto num_ids = L.unwrap();
    const auto total_elements = num_ids * hidden_size;
    constexpr auto kValuesPerByte = 8 / storage_bit;
    const auto expected_q = div_ceil(total_elements, kValuesPerByte);
    RuntimeCheck(Q.unwrap() >= expected_q, "q_packed is shorter than expected");

    const auto params = ZipCacheQuantizeParams{
        .src_cache = src_cache.data_ptr(),
        .local_ids = local_ids.data_ptr(),
        .q_packed = static_cast<std::uint8_t *>(q_packed.data_ptr()),
        .min_val = static_cast<__half *>(min_val.data_ptr()),
        .step = static_cast<__half *>(step.data_ptr()),
        .src_stride = src_stride.unwrap(),
        .src_len = S.unwrap(),
        .num_ids = num_ids,
        .hidden_size = hidden_size,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .total_elements = total_elements,
        .total_packed = expected_q,
    };

    if (total_elements == 0) {
      return;
    }

    const auto device_ = device.unwrap();
    const auto ids_int32 = id_dtype.unwrap().bits == 32;
    const auto scale_blocks =
        div_ceil(num_ids * num_heads, static_cast<std::int64_t>(num_threads));
    const auto pack_blocks = div_ceil(expected_q, static_cast<std::int64_t>(num_threads));

#define LAUNCH_ZIPCACHE_QUANT(IN_T, ID_T)                                             \
  LaunchKernel(scale_blocks, num_threads, device_)                                    \
      .with_attr(use_pdl)(zipcache_quant_scale_kernel<bit, IN_T, ID_T>, params);      \
  LaunchKernel(pack_blocks, num_threads, device_)                                     \
      .with_attr(use_pdl)(zipcache_quant_pack_kernel<bit, storage_bit, IN_T, ID_T>,   \
                          params)

    if (is_fp16) {
      if (ids_int32) {
        LAUNCH_ZIPCACHE_QUANT(__half, int32_t);
      } else {
        LAUNCH_ZIPCACHE_QUANT(__half, int64_t);
      }
    } else {
      if (ids_int32) {
        LAUNCH_ZIPCACHE_QUANT(__nv_bfloat16, int32_t);
      } else {
        LAUNCH_ZIPCACHE_QUANT(__nv_bfloat16, int64_t);
      }
    }
#undef LAUNCH_ZIPCACHE_QUANT
  }
};

template <std::size_t storage_bit, // 2 or 4
          std::size_t num_threads = 256, std::size_t max_concurrency = 1,
          bool use_pdl = false>
struct ZipCacheDequantizeKernel {
  static void run(const tvm::ffi::TensorView out_cache,
                  const tvm::ffi::TensorView dst_indices,
                  const tvm::ffi::TensorView local_ids,
                  const tvm::ffi::TensorView q_packed,
                  const tvm::ffi::TensorView min_val,
                  const tvm::ffi::TensorView step) {
    using namespace host;
    static_assert(storage_bit == 2 || storage_bit == 4);

    auto N = SymbolicSize{"N"};
    auto X = SymbolicSize{"X"};
    auto S = SymbolicSize{"S"};
    auto L = SymbolicSize{"L"};
    auto Q = SymbolicSize{"Q"};
    auto H = SymbolicSize{"H"};
    auto one = SymbolicSize{"1"};
    one.set_value(1);
    auto out_stride = SymbolicSize{"out_stride"};
    auto out_dtype = SymbolicDType{};
    auto index_dtype = SymbolicDType{};
    auto id_dtype = SymbolicDType{};
    auto q_dtype = SymbolicDType{};
    auto scale_dtype = SymbolicDType{};
    auto device = SymbolicDevice{};

    TensorMatcher({N, X})
        .with_strides({out_stride, 1})
        .with_device<kDLCUDA>(device)
        .with_dtype(out_dtype)
        .verify(out_cache);
    TensorMatcher({S})
        .with_device<kDLCUDA>(device)
        .with_dtype<int32_t, int64_t>(index_dtype)
        .verify(dst_indices);
    TensorMatcher({L})
        .with_device<kDLCUDA>(device)
        .with_dtype<int32_t, int64_t>(id_dtype)
        .verify(local_ids);
    TensorMatcher({Q})
        .with_device<kDLCUDA>(device)
        .with_dtype<std::uint8_t>(q_dtype)
        .verify(q_packed);
    TensorMatcher({L, H, one})
        .with_strides({H, 1, 1})
        .with_device<kDLCUDA>(device)
        .with_dtype(scale_dtype)
        .verify(min_val)
        .verify(step);

    const auto out_dt = out_dtype.unwrap();
    const auto scale_dt = scale_dtype.unwrap();
    RuntimeCheck(out_dt.bits == 16, "ZipCache v4 only supports fp16/bf16 output");
    RuntimeCheck(scale_dt.code == DLDataTypeCode::kDLFloat && scale_dt.bits == 16,
                 "ZipCache v4 min/step tensors must be fp16");

    const auto hidden_size = X.unwrap();
    const auto num_heads = H.unwrap();
    RuntimeCheck(hidden_size % num_heads == 0,
                 "hidden_size must be divisible by num_heads");
    const auto head_dim = hidden_size / num_heads;
    const auto num_ids = L.unwrap();
    const auto total_elements = num_ids * hidden_size;
    constexpr auto kValuesPerByte = 8 / storage_bit;
    const auto expected_q = div_ceil(total_elements, kValuesPerByte);
    RuntimeCheck(Q.unwrap() >= expected_q, "q_packed is shorter than expected");

    const auto params = ZipCacheDequantizeParams{
        .out_cache = out_cache.data_ptr(),
        .dst_indices = dst_indices.data_ptr(),
        .local_ids = local_ids.data_ptr(),
        .q_packed = static_cast<const std::uint8_t *>(q_packed.data_ptr()),
        .min_val = static_cast<const __half *>(min_val.data_ptr()),
        .step = static_cast<const __half *>(step.data_ptr()),
        .out_stride = out_stride.unwrap(),
        .dst_len = S.unwrap(),
        .num_ids = num_ids,
        .hidden_size = hidden_size,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .total_elements = total_elements,
    };

    const auto blocks = div_ceil(total_elements, static_cast<std::int64_t>(num_threads));
    if (total_elements == 0) {
      return;
    }

    const auto device_ = device.unwrap();
    const auto dst_int32 = index_dtype.unwrap().bits == 32;
    const auto ids_int32 = id_dtype.unwrap().bits == 32;
    constexpr auto kDLBfloatCode = static_cast<DLDataTypeCode>(4);
    const auto is_bf16 = out_dt.code == kDLBfloatCode && out_dt.bits == 16;
    const auto is_fp16 = out_dt.code == DLDataTypeCode::kDLFloat && out_dt.bits == 16;
    RuntimeCheck(is_fp16 || is_bf16, "ZipCache v4 output must be fp16 or bf16");

#define LAUNCH_ZIPCACHE_KERNEL(OUT_T, DST_T, ID_T)                                      \
  LaunchKernel(blocks, num_threads, device_)                                            \
      .with_attr(use_pdl)(zipcache_dequantize_kernel<storage_bit, OUT_T, DST_T, ID_T>,  \
                          params)

    if (is_fp16) {
      if (dst_int32 && ids_int32) {
        LAUNCH_ZIPCACHE_KERNEL(__half, int32_t, int32_t);
      } else if (dst_int32) {
        LAUNCH_ZIPCACHE_KERNEL(__half, int32_t, int64_t);
      } else if (ids_int32) {
        LAUNCH_ZIPCACHE_KERNEL(__half, int64_t, int32_t);
      } else {
        LAUNCH_ZIPCACHE_KERNEL(__half, int64_t, int64_t);
      }
    } else {
      if (dst_int32 && ids_int32) {
        LAUNCH_ZIPCACHE_KERNEL(__nv_bfloat16, int32_t, int32_t);
      } else if (dst_int32) {
        LAUNCH_ZIPCACHE_KERNEL(__nv_bfloat16, int32_t, int64_t);
      } else if (ids_int32) {
        LAUNCH_ZIPCACHE_KERNEL(__nv_bfloat16, int64_t, int32_t);
      } else {
        LAUNCH_ZIPCACHE_KERNEL(__nv_bfloat16, int64_t, int64_t);
      }
    }
#undef LAUNCH_ZIPCACHE_KERNEL
  }
};

} // namespace
