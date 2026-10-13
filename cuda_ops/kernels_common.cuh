/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
  
#pragma once
#include <cuda_fp16.h>
#include <stdint.h>
#include "include/nve_types.hpp"

namespace nve {

inline uint32_t DivRoundUp(uint32_t n, uint32_t m)
{
    uint32_t res = (n + (m - 1))/m;
    return res;
}

typedef struct __align__(16) {
  __half a, b, c, d, e, f, g, h;
}
half8;

typedef struct __align__(8) {
  __half x, y, z, w;
}
half4;

inline uint32_t nextPow2(uint32_t x) 
{
    return x == 1 ? 1 : 1<<(32-__builtin_clz(x) - 1); 
}

template<typename DataType>
struct VecWidthHelper
{
};

template<>
struct VecWidthHelper<float>
{
    using Vec4 = float4;
    using Vec2 = float2;
    using Vec1 = float;
};

template<>
struct VecWidthHelper<__half>
{
    using Vec4 = half4;
    using Vec2 = half2;
    using Vec1 = __half;
};

template<DataType_t DataType>
struct QuantizationHelper
{
};

template<>
struct QuantizationHelper<DataType_t::QInt8RowwiseF32>
{
    using Vec4 = char4;
    using Vec2 = char2;
    using Vec1 = signed char; // on aarch64 char is by default unsigned
    using ParamType = float;
    constexpr static const bool has_scale = true;
    constexpr static const bool has_offset = false;
    constexpr static const size_t element_size = sizeof(char);
};

template<>
struct QuantizationHelper<DataType_t::Float32>
{
    using Vec4 = float4;
    using Vec2 = float2;
    using Vec1 = float;
    using ParamType = float;
    constexpr static const bool has_scale = false;
    constexpr static const bool has_offset = false;
    constexpr static const size_t element_size = sizeof(float);
};

template<>
struct QuantizationHelper<DataType_t::Float16>
{
    using Vec4 = half4;
    using Vec2 = half2;
    using Vec1 = __half;
    using ParamType = __half;
    constexpr static const bool has_scale = false;
    constexpr static const bool has_offset = false;
    constexpr static const size_t element_size = sizeof(__half);
};

template<>
struct QuantizationHelper<DataType_t::QUint8RowwiseF32>
{
    using Vec4 = uchar4;
    using Vec2 = uchar2;
    using Vec1 = unsigned char;
    using ParamType = float;
    constexpr static const bool has_scale = true;
    constexpr static const bool has_offset = true;
    constexpr static const size_t element_size = sizeof(uint8_t);
};

template<>
struct QuantizationHelper<DataType_t::QInt8RowwiseF16>
{
    using Vec4 = char4;
    using Vec2 = char2;
    using Vec1 = signed char;
    using ParamType = __half;
    constexpr static const bool has_scale = true;
    constexpr static const bool has_offset = false;
    constexpr static const size_t element_size = sizeof(int8_t);
};

template<>
struct QuantizationHelper<DataType_t::QUint8RowwiseF16>
{
    using Vec4 = uchar4;
    using Vec2 = uchar2;
    using Vec1 = unsigned char;
    using ParamType = __half;
    constexpr static const bool has_scale = true;
    constexpr static const bool has_offset = true;
    constexpr static const size_t element_size = sizeof(uint8_t);
};

// Byte size of one stored row: elements followed by any per-row quantization parameters.
template<DataType_t DTYPE>
inline constexpr uint32_t get_row_size_in_bytes(uint32_t row_size_in_elements)
{
    uint32_t bytes = row_size_in_elements * QuantizationHelper<DTYPE>::element_size;
    if constexpr (QuantizationHelper<DTYPE>::has_scale)
    {
        bytes += sizeof(typename QuantizationHelper<DTYPE>::ParamType);
    }
    if constexpr (QuantizationHelper<DTYPE>::has_offset)
    {
        bytes += sizeof(typename QuantizationHelper<DTYPE>::ParamType);
    }
    return bytes;
}

// Size in bytes of the dequantized value type (QuantizationHelper::ParamType) for a runtime data
// type. Used to convert an output row stride (in bytes) into an element count. Returns 0 for
// unsupported types (callers should only pass value/output types: *F32 -> float, *F16 -> __half).
inline size_t value_size_in_bytes(DataType_t dtype) {
    if (dtype == DataType_t::Float32 || dtype == DataType_t::Float16) {
        return static_cast<size_t>(dtype_size(dtype));
    }
    if (is_quant_rowwise(dtype)) {
        return static_cast<size_t>(dtype_size(quant_rowwise_output_dtype(dtype)));
    }
    NVE_THROW_("Unsupported value type for value_size_in_bytes(): ", static_cast<uint32_t>(dtype));
    return 0;
}

template<typename QUANTIZE_TYPE, typename ELEMENT_VEC_TYPE, typename INPUT_VEC_TYPE>
inline ELEMENT_VEC_TYPE __device__ dequantize(const INPUT_VEC_TYPE& el_quantized, const typename QUANTIZE_TYPE::ParamType& scale, const typename QUANTIZE_TYPE::ParamType& offset);

template<>
inline float4 __device__ dequantize<QuantizationHelper<DataType_t::QInt8RowwiseF32>, float4, char4>(const char4& el_quantized, const float& scale, const float& offset) {
  float4 tmp;
  tmp.x = float(el_quantized.x) * scale + offset;
  tmp.y = float(el_quantized.y) * scale + offset;
  tmp.z = float(el_quantized.z) * scale + offset;
  tmp.w = float(el_quantized.w) * scale + offset;
  return tmp;
}

template<>
inline float2 __device__ dequantize<QuantizationHelper<DataType_t::QInt8RowwiseF32>, float2, char2>(const char2& el_quantized, const float& scale, const float& offset) {
  float2 tmp;
  tmp.x = float(el_quantized.x) * scale + offset;
  tmp.y = float(el_quantized.y) * scale + offset;
  return tmp;
}

template<>
inline float __device__ dequantize<QuantizationHelper<DataType_t::QInt8RowwiseF32>, float, signed char>(const signed char& el_quantized, const float& scale, const float& offset) {
  return float(el_quantized) * scale + offset;
}

template<>
inline float4 __device__ dequantize<QuantizationHelper<DataType_t::Float32>, float4, float4>(const float4& el_quantized, const float& scale, const float& offset) {
  return el_quantized;
}

template<>
inline float2 __device__ dequantize<QuantizationHelper<DataType_t::Float32>, float2, float2>(const float2& el_quantized, const float& scale, const float& offset) {
  return el_quantized;
}

template<>
inline float __device__ dequantize<QuantizationHelper<DataType_t::Float32>, float, float>(const float& el_quantized, const float& scale, const float& offset) {
  return el_quantized;
}

template<>
inline half4 __device__ dequantize<QuantizationHelper<DataType_t::Float16>, half4, half4>(const half4& el_quantized, const __half& scale, const __half& offset) {
  return el_quantized;
}

template<>
inline half2 __device__ dequantize<QuantizationHelper<DataType_t::Float16>, half2, half2>(const half2& el_quantized, const __half& scale, const __half& offset) {
  return el_quantized;
}

template<>
inline __half __device__ dequantize<QuantizationHelper<DataType_t::Float16>, __half, __half>(const __half& el_quantized, const __half& scale, const __half& offset) {
  return el_quantized;
}

template<>
inline float4 __device__ dequantize<QuantizationHelper<DataType_t::QUint8RowwiseF32>, float4, uchar4>(const uchar4& el_quantized, const float& scale, const float& offset) {
  float4 tmp;
  tmp.x = float(el_quantized.x) * scale + offset;
  tmp.y = float(el_quantized.y) * scale + offset;
  tmp.z = float(el_quantized.z) * scale + offset;
  tmp.w = float(el_quantized.w) * scale + offset;
  return tmp;
}

template<>
inline float2 __device__ dequantize<QuantizationHelper<DataType_t::QUint8RowwiseF32>, float2, uchar2>(const uchar2& el_quantized, const float& scale, const float& offset) {
  float2 tmp;
  tmp.x = float(el_quantized.x) * scale + offset;
  tmp.y = float(el_quantized.y) * scale + offset;
  return tmp;
}

template<>
inline float __device__ dequantize<QuantizationHelper<DataType_t::QUint8RowwiseF32>, float, unsigned char>(const unsigned char& el_quantized, const float& scale, const float& offset) {
  return float(el_quantized) * scale + offset;
}

template<>
inline half4 __device__ dequantize<QuantizationHelper<DataType_t::QInt8RowwiseF16>, half4, char4>(const char4& el_quantized, const __half& scale, const __half& offset) {
  half4 tmp;
  tmp.x = __float2half(float(el_quantized.x) * __half2float(scale));
  tmp.y = __float2half(float(el_quantized.y) * __half2float(scale));
  tmp.z = __float2half(float(el_quantized.z) * __half2float(scale));
  tmp.w = __float2half(float(el_quantized.w) * __half2float(scale));
  return tmp;
}

template<>
inline half2 __device__ dequantize<QuantizationHelper<DataType_t::QInt8RowwiseF16>, half2, char2>(const char2& el_quantized, const __half& scale, const __half& offset) {
  half2 tmp;
  tmp.x = __float2half(float(el_quantized.x) * __half2float(scale));
  tmp.y = __float2half(float(el_quantized.y) * __half2float(scale));
  return tmp;
}

template<>
inline __half __device__ dequantize<QuantizationHelper<DataType_t::QInt8RowwiseF16>, __half, signed char>(const signed char& el_quantized, const __half& scale, const __half& offset) {
  return __float2half(float(el_quantized) * __half2float(scale));
}

template<>
inline half4 __device__ dequantize<QuantizationHelper<DataType_t::QUint8RowwiseF16>, half4, uchar4>(const uchar4& el_quantized, const __half& scale, const __half& offset) {
  half4 tmp;
  tmp.x = __float2half(float(el_quantized.x) * __half2float(scale) + __half2float(offset));
  tmp.y = __float2half(float(el_quantized.y) * __half2float(scale) + __half2float(offset));
  tmp.z = __float2half(float(el_quantized.z) * __half2float(scale) + __half2float(offset));
  tmp.w = __float2half(float(el_quantized.w) * __half2float(scale) + __half2float(offset));
  return tmp;
}

template<>
inline half2 __device__ dequantize<QuantizationHelper<DataType_t::QUint8RowwiseF16>, half2, uchar2>(const uchar2& el_quantized, const __half& scale, const __half& offset) {
  half2 tmp;
  tmp.x = __float2half(float(el_quantized.x) * __half2float(scale) + __half2float(offset));
  tmp.y = __float2half(float(el_quantized.y) * __half2float(scale) + __half2float(offset));
  return tmp;
}

template<>
inline __half __device__ dequantize<QuantizationHelper<DataType_t::QUint8RowwiseF16>, __half, uint8_t>(const uint8_t& el_quantized, const __half& scale, const __half& offset) {
  return __float2half(float(el_quantized) * __half2float(scale) + __half2float(offset));
}

template<typename DataType>
inline void __device__ InitAcc(DataType& acc);

template<>
inline void __device__ InitAcc(float& acc) {
  acc = 0;
}

template<>
inline void __device__ InitAcc(__half& acc) {
  acc = 0;
}

template<>
inline void __device__ InitAcc(float2& acc) {
  acc.x = acc.y = 0;
}

template<>
inline void __device__ InitAcc(half2& acc) {
  acc.x = acc.y = 0;
}

template<>
inline void __device__ InitAcc(float4& acc) {
  acc.x = acc.y = acc.z = acc.w = 0;
}

template<>
inline void __device__ InitAcc(half4& acc) {
  acc.x = acc.y = acc.z = acc.w = 0;
}

template<typename DataType>
inline void __device__ Accumulate(DataType& acc, const DataType& d);

template<>
inline void __device__ Accumulate(float& acc, const float& d) {
  acc += d;
}

template<>
inline void __device__ Accumulate(__half& acc, const __half& d) {
  acc += d;
}

template<>
inline void __device__ Accumulate(float2& acc, const float2& d) {
  acc.x += d.x;
  acc.y += d.y;
}

template<>
inline void __device__ Accumulate(half2& acc, const half2& d) {
  acc.x += d.x;
  acc.y += d.y;
}

template<>
inline void __device__ Accumulate(float4& acc, const float4& d) {
  acc.x += d.x;
  acc.y += d.y;
  acc.z += d.z;
  acc.w += d.w;
}

template<>
inline void __device__ Accumulate(half4& acc, const half4& d) {
  acc.x += d.x;
  acc.y += d.y;
  acc.z += d.z;
  acc.w += d.w;
}

template<typename DataType>
inline void __device__ MulAccumulate(DataType& acc, const DataType& el, const DataType& weight);

template<>
inline void __device__ MulAccumulate(float& acc, const float& el, const float& weight) {
  acc += el * weight;
}

template<>
inline void __device__ MulAccumulate(__half& acc, const __half& el, const __half& weight) {
  acc = __hfma(el, weight, acc);
}

template<typename DataType, typename WeightType>
inline void __device__ MulAccumulate(DataType& acc, const DataType& el, const WeightType& weight);

template<>
inline void __device__ MulAccumulate(float2& acc, const float2& el, const float& weight) {
  acc.x += el.x * weight;
  acc.y += el.y * weight;
}

template<>
inline void __device__ MulAccumulate(float4& acc, const float4& el, const float& weight) {
  acc.x += el.x * weight;
  acc.y += el.y * weight;
  acc.z += el.z * weight;
  acc.w += el.w * weight;
}

template<>
inline void __device__ MulAccumulate(__half2& acc, const __half2& el, const __half& weight) {
  acc.x = __hfma(el.x, weight, acc.x);
  acc.y = __hfma(el.y, weight, acc.y);
}

template<>
inline void __device__ MulAccumulate(half4& acc, const half4& el, const __half& weight) {
  acc.x = __hfma(el.x, weight, acc.x);
  acc.y = __hfma(el.y, weight, acc.y);
  acc.w = __hfma(el.w, weight, acc.w);
  acc.z = __hfma(el.z, weight, acc.z);
}

template<typename DataType, typename WeightType>
inline void __device__ Div(DataType& acc, const WeightType& weight);

template<>
inline void __device__ Div(float& acc, const float& weight) {
  acc /= weight;
}

template<>
inline void __device__ Div(float2& acc, const float& weight) {
  acc.x /= weight;
  acc.y /= weight;
}

template<>
inline void __device__ Div(float4& acc, const float& weight) {
  acc.x /= weight;
  acc.y /= weight;
  acc.z /= weight;
  acc.w /= weight;
}

template<>
inline void __device__ Div(__half& acc, const __half& weight) {
  acc /= weight;
}

template<>
inline void __device__ Div(__half2& acc, const __half& weight) {
  acc.x /= weight;
  acc.y /= weight;
}

template<>
inline void __device__ Div(half4& acc, const __half& weight) {
  acc.x /= weight;
  acc.y /= weight;
  acc.z /= weight;
  acc.w /= weight;
}

// Zero test for the Mean/WeightedMean averaging denominator, so a zero denominator (empty bag, or
// weights summing to zero) can zero-fill the output row instead of dividing by zero.
inline bool __device__ IsZero(const float& weight) { return weight == 0.f; }

inline bool __device__ IsZero(const __half& weight) { return __half2float(weight) == 0.f; }

template<typename FromType, typename ToType>
inline ToType __device__ Cast(const FromType& d) {
  return d;
}

template<>
inline float __device__ Cast(const __half& d) {
  return __half2float(d);
}

template<>
inline float2 __device__ Cast(const __half2& d) {
  float2 tmp;
  // TODO: change to builtin
  tmp.x = __half2float(d.x);
  tmp.y = __half2float(d.y);
  return tmp;
}

template<>
inline float4 __device__ Cast(const half4& d) {
  float4 tmp;
  tmp.x = __half2float(d.x);
  tmp.y = __half2float(d.y);
  tmp.z = __half2float(d.z);
  tmp.w = __half2float(d.w);
  return tmp;
}

template<>
inline __half __device__ Cast(const float& d) {
  return __float2half(d);
}

template<>
inline __half2 __device__ Cast(const float2& d) {
  __half2 tmp;
  tmp.x = __float2half(d.x);
  tmp.y = __float2half(d.y);
  return tmp;
}

template<>
inline half4 __device__ Cast(const float4& d) {
  half4 tmp;
  tmp.x = __float2half(d.x);
  tmp.y = __float2half(d.y);
  tmp.z = __float2half(d.z);
  tmp.w = __float2half(d.w);
  return tmp;
}

template<typename DataType>
inline void __device__ AtomicAccumulate(DataType* src, DataType d)
{
  atomicAdd(src, d);
}

template<>
inline void __device__ AtomicAccumulate(float4* src, float4 d) {
  atomicAdd((float*)src, d.x);
  atomicAdd((float*)src+1, d.y);
  atomicAdd((float*)src+2, d.z);
  atomicAdd((float*)src+3, d.w);
}

template<>
inline void __device__ AtomicAccumulate(float2* src, float2 d) {
  atomicAdd((float*)src, d.x);
  atomicAdd((float*)src+1, d.y);
}

template<>
inline void __device__ AtomicAccumulate(half4* src, half4 d) {
  half2 d1;
  d1.x = d.x;
  d1.y = d.y;
  atomicAdd((half2*)src, d1);
  half2 d2;
  d2.x = d.z;
  d2.y = d.w;
  atomicAdd((half2*)(src) + 1, d2);
}

}  // namespace nve
