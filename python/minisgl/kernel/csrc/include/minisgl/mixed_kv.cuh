#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace mixed_kv {

// segment_meta[:, 0] 使用的来源类型。normal segment 通过 int32 物理
// token index 寻址；quantized segment 通过下面的 pool offset 寻址。
inline constexpr int kNormalSegment = 0;
inline constexpr int kQuantizedSegment = 1;

// segment_offsets 每行八列的固定布局。所有 offset 都以对应 buffer 的
// 元素为单位，而不是统一的 byte offset。
inline constexpr int kKQOffset = 0;
inline constexpr int kKMinOffset = 1;
inline constexpr int kKStepOffset = 2;
inline constexpr int kVQOffset = 3;
inline constexpr int kVMinOffset = 4;
inline constexpr int kVStepOffset = 5;
inline constexpr int kIdsOffset = 6;
inline constexpr int kNormalIndexOffset = 7;

template <typename T>
__device__ __forceinline__ float to_float(T value) {
  if constexpr (std::is_same_v<T, __half>) {
    return __half2float(value);
  } else {
    return __bfloat162float(value);
  }
}

template <typename T>
__device__ __forceinline__ T from_float(float value) {
  if constexpr (std::is_same_v<T, __half>) {
    return __float2half_rn(value);
  } else {
    return __float2bfloat16_rn(value);
  }
}

__device__ __forceinline__ float load_quantized(
    const std::uint8_t *__restrict__ q4,
    const std::uint8_t *__restrict__ q2,
    const __half *__restrict__ scale,
    std::int64_t q_offset,
    std::int64_t min_offset,
    std::int64_t step_offset,
    int storage_bit,
    std::int64_t logical_index,
    std::int64_t scale_index) {
  // _pack_lowbit() 采用低位优先：2bit 时一个 byte 保存四个值，4bit 时
  // 保存两个值。logical_index 是 QuantizedPart 展平后的 [M,H,D] 下标。
  const int values_per_byte = 8 / storage_bit;
  const std::uint8_t *packed = storage_bit == 2 ? q2 : q4;
  const auto byte = packed[q_offset + logical_index / values_per_byte];
  const int shift = int(logical_index % values_per_byte) * storage_bit;
  const int quant = (byte >> shift) & ((1 << storage_bit) - 1);
  const float minimum = __half2float(scale[min_offset + scale_index]);
  const float step = __half2float(scale[step_offset + scale_index]);
  // min/step 是每个 [token, KV head] 一组，因此 scale_index 不含 D。
  return float(quant) * step + minimum;
}

} // namespace mixed_kv
