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
#include <stdint.h>
#include <cuda_runtime.h>
#include "kernels_common.cuh"
#include "cuda_ops/cuda_common.h"

namespace nve {

template<uint32_t SubwarpWidth, typename KeyType, typename DataType>
__global__ void UpdateTableKernel
    (const int8_t* __restrict__ src,
    const KeyType* __restrict__ indices,
    int8_t* __restrict__ embedding_table,
    const uint32_t embed_width_in_bytes,
    const uint32_t embed_src_stride_in_bytes,
    const uint32_t embed_dst_stride_in_bytes,
    const int32_t num_indices)
{
    const int id = blockIdx.x * blockDim.y + threadIdx.y;

    if (id >= num_indices) {
      return;
    }

    KeyType key = indices[id];

    const DataType* embed_src = reinterpret_cast<const DataType*>(src + id * embed_src_stride_in_bytes);
    DataType* embed_dst = reinterpret_cast<DataType*>(embedding_table + key * embed_dst_stride_in_bytes);

    auto num_elements = embed_width_in_bytes / sizeof(DataType);

    for (int el = threadIdx.x; el < num_elements; el += SubwarpWidth) {
      embed_dst[el] = embed_src[el];
    }
}

template<uint32_t SubwarpWidth, typename KeyType, typename DataType>
__global__ void UpdateAccumulateTableKernel(
      const DataType* __restrict__ src,
      const KeyType* __restrict__ indices,
      DataType* __restrict__ embedding_table,
      const uint32_t embed_width,
      const uint32_t embed_src_stride,
      const uint32_t embed_dst_stride,
      const int32_t num_indices)
{
    const int id = blockIdx.x * blockDim.y + threadIdx.y;

    if (id >= num_indices) {
      return;
    }

    KeyType key = indices[id];

    const DataType* embed_src = src + id * embed_src_stride;
    DataType* embed_dst = embedding_table + key * embed_dst_stride;

    for (int el = threadIdx.x; el < embed_width; el += SubwarpWidth) {
      atomicAdd(embed_dst + el, embed_src[el]);
    }
}

template<uint32_t SubwarpWidth, typename KeyType, typename DataType>
void CallUpdateKernelVecTypeSubwarp(
                              const int8_t* __restrict__ src,
                              const KeyType* __restrict__ indices,
                              int8_t* __restrict__ embedding_table,
                              const uint32_t embed_width_in_bytes,
                              const uint32_t embed_src_stride_in_bytes,
                              const uint32_t embed_dst_stride_in_bytes,
                              const int32_t num_indices,
                              const cudaStream_t stream = 0)
{
    uint32_t indices_per_warp = 32 / SubwarpWidth;
    dim3 grid_size ((num_indices + indices_per_warp - 1) / indices_per_warp, 1);
    dim3 block_size (SubwarpWidth, indices_per_warp);
    UpdateTableKernel<SubwarpWidth, KeyType, DataType><<<grid_size, block_size, 0, stream>>>(
        src, indices, embedding_table, embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, num_indices);
    NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
}

template<typename KeyType, typename DataType>
void CallUpdatKernelVecType(const int8_t* __restrict__ src,
                            const KeyType* __restrict__ indices,
                            int8_t* __restrict__ embedding_table,
                            const uint32_t embed_width_in_bytes,
                            const uint32_t embed_src_stride_in_bytes,
                            const uint32_t embed_dst_stride_in_bytes,
                            const int32_t num_indices,
                            const cudaStream_t stream = 0)
{
    uint32_t subgroupWidth = std::min(nextPow2(embed_width_in_bytes / sizeof(DataType)), 32u);
    switch (subgroupWidth)
    {
    case 32:
        CallUpdateKernelVecTypeSubwarp<32, KeyType, DataType>(src, indices, embedding_table, embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, num_indices, stream);
        break;
    case 16:
        CallUpdateKernelVecTypeSubwarp<16, KeyType, DataType>(src, indices, embedding_table, embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, num_indices, stream);
        break;
    case 8:
        CallUpdateKernelVecTypeSubwarp<8, KeyType, DataType>(src, indices, embedding_table, embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, num_indices, stream);
        break;
    case 4:
        CallUpdateKernelVecTypeSubwarp<4, KeyType, DataType>(src, indices, embedding_table, embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, num_indices, stream);
        break;
    case 2:
        CallUpdateKernelVecTypeSubwarp<2, KeyType, DataType>(src, indices, embedding_table, embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, num_indices, stream);
        break;
    case 1:
        CallUpdateKernelVecTypeSubwarp<1, KeyType, DataType>(src, indices, embedding_table, embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, num_indices, stream);
        break;
    default:
        assert(0);
    }
}

template<typename KeyType>
void UpdateTable(const void* src,
                 const KeyType* indices,
                 void* embedding_table,
                 const uint32_t embed_width_in_bytes,
                 const uint32_t embed_src_stride_in_bytes,
                 const uint32_t embed_dst_stride_in_bytes,
                 const int32_t num_indices,
                 const cudaStream_t stream)
{
    if ((embed_width_in_bytes % 16) == 0) {
      using Vec4 = typename VecWidthHelper<float>::Vec4;
      CallUpdatKernelVecType<KeyType, Vec4>(reinterpret_cast<const int8_t*>(src), indices, reinterpret_cast<int8_t*>(embedding_table),
                                            embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, num_indices, stream);
    } else if ((embed_width_in_bytes % 8) == 0) {
      using Vec2 = typename VecWidthHelper<float>::Vec2;
      CallUpdatKernelVecType<KeyType, Vec2>(reinterpret_cast<const int8_t*>(src), indices, reinterpret_cast<int8_t*>(embedding_table),
                                            embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, num_indices, stream);
    } else if ((embed_width_in_bytes % 4) == 0) {
      using Vec1 = typename VecWidthHelper<float>::Vec1;
      CallUpdatKernelVecType<KeyType, Vec1>(reinterpret_cast<const int8_t*>(src), indices, reinterpret_cast<int8_t*>(embedding_table),
                                            embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, num_indices, stream);
    } else if ((embed_width_in_bytes % 2) == 0) {
      using Vec1 = typename VecWidthHelper<__half>::Vec1;
      CallUpdatKernelVecType<KeyType, Vec1>(reinterpret_cast<const int8_t*>(src), indices, reinterpret_cast<int8_t*>(embedding_table),
                                            embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, num_indices, stream);
    } 
}

template<uint32_t SubwarpWidth, typename KeyType, typename DataType>
void CallUpdateAccumulateKernelSubwarp(
                              const DataType* __restrict__ src,
                              const KeyType* __restrict__ indices,
                              DataType* __restrict__ embedding_table,
                              const uint32_t embed_width,
                              const uint32_t embed_src_stride,
                              const uint32_t embed_dst_stride,
                              const int32_t num_indices,
                              const cudaStream_t stream = 0)
{
    uint32_t indices_per_warp = 32 / SubwarpWidth;
    dim3 grid_size ((num_indices + indices_per_warp - 1) / indices_per_warp, 1);
    dim3 block_size (SubwarpWidth, indices_per_warp);
    UpdateAccumulateTableKernel<SubwarpWidth, KeyType, DataType><<<grid_size, block_size, 0, stream>>>(
        src, indices, embedding_table, embed_width, embed_src_stride, embed_dst_stride, num_indices);
    NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
}

template<typename KeyType, typename DataType>
void UpdateAccumulateTable(const DataType* src,
                           const KeyType* indices,
                           DataType* embedding_table,
                           const uint32_t embed_width,
                           const uint32_t embed_src_stride,
                           const uint32_t embed_dst_stride,
                           const int32_t num_indices,
                           const cudaStream_t stream)
{
    uint32_t subgroupWidth = std::min(nextPow2(embed_width), 32u);
    switch (subgroupWidth)
    {
      case 32:
          CallUpdateAccumulateKernelSubwarp<32, KeyType, DataType>(
              src, indices, embedding_table,
              embed_width, embed_src_stride, embed_dst_stride, num_indices, stream);
          break;
      case 16:
          CallUpdateAccumulateKernelSubwarp<16, KeyType, DataType>(
              src, indices, embedding_table,
              embed_width, embed_src_stride, embed_dst_stride, num_indices, stream);
          break;
      case 8:
          CallUpdateAccumulateKernelSubwarp<8, KeyType, DataType>(
              src, indices, embedding_table,
              embed_width, embed_src_stride, embed_dst_stride, num_indices, stream);
          break;
      case 4:
          CallUpdateAccumulateKernelSubwarp<4, KeyType, DataType>(
              src, indices, embedding_table,
              embed_width, embed_src_stride, embed_dst_stride, num_indices, stream);
          break;
      case 2:
          CallUpdateAccumulateKernelSubwarp<2, KeyType, DataType>(
              src, indices, embedding_table,
              embed_width, embed_src_stride, embed_dst_stride, num_indices, stream);
          break;
      case 1:
          CallUpdateAccumulateKernelSubwarp<1, KeyType, DataType>(
              src, indices, embedding_table,
              embed_width, embed_src_stride, embed_dst_stride, num_indices, stream);
          break;
      default:
          assert(0);
    }
}

}  // namespace nve
