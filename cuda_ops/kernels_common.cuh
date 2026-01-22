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
