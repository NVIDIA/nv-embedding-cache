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

template<typename T>
__device__ inline T Add(T a, T b)
{
  return a + b;
}

template<>
__device__ inline float4 Add(float4 a, float4 b)
{
  float4 ret;
  ret.x = a.x + b.x;
  ret.y = a.y + b.y;
  ret.z = a.z + b.z;
  ret.w = a.w + b.w;
  return ret;
}

template<typename IndexT>
constexpr __device__ __host__ inline IndexT GetInvalidIndex()
{
  return static_cast<IndexT>(-1);
}

} // namespace nve