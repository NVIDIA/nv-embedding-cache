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
#include <ecache/embedding_cache_combined.cuh>

namespace nve {

template<uint32_t SubwarpWidth, typename DataType>
__global__ void EmbedScatter( const int8_t* __restrict__ src,
                              int8_t* __restrict__ dst,
                              const uint32_t embed_width_in_bytes,
                              const uint32_t embed_src_stride_in_bytes,
                              const uint32_t embed_dst_stride_in_bytes,
                              const uint64_t* __restrict__ hit_mask,
                              const int32_t num_indices)
{
    const int embed = blockIdx.x * blockDim.y + threadIdx.y;

    if (embed >= num_indices) {
      return;
    }

    const int mask_entry = embed / 64;
    const int mask_bit = embed % 64;
    uint64_t mask = 1;
    mask <<= mask_bit;

    if (hit_mask[mask_entry] & mask) {
      return;
    }

    const int8_t* embed_src = src + embed * embed_src_stride_in_bytes;
    int8_t* embed_dst = dst + embed * embed_dst_stride_in_bytes;

    nve::memcpy_warp<SubwarpWidth, DataType>(embed_dst, embed_src, embed_width_in_bytes);
}

template<uint32_t SubwarpWidth, typename DataType>
void CallScatterKernelVecTypeSubwarp(
                              const int8_t* __restrict__ src,
                              int8_t* __restrict__ dst,
                              const uint32_t embed_width_in_bytes,
                              const uint32_t embed_src_stride_in_bytes,
                              const uint32_t embed_dst_stride_in_bytes,
                              const uint64_t* __restrict__ hit_mask,
                              const int32_t num_indices,
                              const cudaStream_t stream = 0)
{
    uint32_t indices_per_warp = 64 / SubwarpWidth;
    dim3 grid_size ((num_indices + indices_per_warp - 1) / indices_per_warp, 1);
    dim3 block_size (SubwarpWidth, indices_per_warp);
    EmbedScatter<SubwarpWidth, DataType><<<grid_size, block_size, 0, stream>>>(
        src, dst, embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices);
    NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
}

template<typename DataType>
void CallScatterKernelVecType(
                              const int8_t* __restrict__ src,
                              int8_t* __restrict__ dst,
                              const uint32_t embed_width_in_bytes,
                              const uint32_t embed_src_stride_in_bytes,
                              const uint32_t embed_dst_stride_in_bytes,
                              const uint64_t* __restrict__ hit_mask,
                              const int32_t num_indices,
                              const cudaStream_t stream = 0)
{
    uint32_t subgroupWidth = std::min(nextPow2(embed_width_in_bytes / sizeof(DataType)), 32u);
    switch (subgroupWidth)
    {
    case 32:
        CallScatterKernelVecTypeSubwarp<32, DataType>(src, dst, embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices, stream);
        break;
    case 16:
        CallScatterKernelVecTypeSubwarp<16, DataType>(src, dst, embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices, stream);
        break;
    case 8:
        CallScatterKernelVecTypeSubwarp<8, DataType>(src, dst, embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices, stream);
        break;
    case 4:
        CallScatterKernelVecTypeSubwarp<4, DataType>(src, dst, embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices, stream);
        break;
    case 2:
        CallScatterKernelVecTypeSubwarp<2, DataType>(src, dst, embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices, stream);
        break;
    case 1:
        CallScatterKernelVecTypeSubwarp<1, DataType>(src, dst, embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices, stream);
        break;
    default:
        assert(0);
    }
}

inline void EmbeddingForwardScatter(const void* src,
                             void* dst,
                             const uint32_t embed_width_in_bytes,
                             const uint32_t embed_src_stride_in_bytes,
                             const uint32_t embed_dst_stride_in_bytes,
                             const uint64_t* hit_mask,
                             const int32_t num_indices,
                             const cudaStream_t stream)
{
    if ((embed_width_in_bytes % 16) == 0) {
      using Vec4 = typename VecWidthHelper<float>::Vec4;
      CallScatterKernelVecType<Vec4>(reinterpret_cast<const int8_t*>(src), reinterpret_cast<int8_t*>(dst), embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices, stream);
    } else if ((embed_width_in_bytes % 8) == 0) {
      using Vec2 = typename VecWidthHelper<float>::Vec2;
      CallScatterKernelVecType<Vec2>(reinterpret_cast<const int8_t*>(src), reinterpret_cast<int8_t*>(dst), embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices, stream);
    } else if ((embed_width_in_bytes % 4) == 0) {
      using Vec1 = typename VecWidthHelper<float>::Vec1;
      CallScatterKernelVecType<Vec1>(reinterpret_cast<const int8_t*>(src), reinterpret_cast<int8_t*>(dst), embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices, stream);
    } else if ((embed_width_in_bytes % 2) == 0) {
      using Vec1 = typename VecWidthHelper<__half>::Vec1;
      CallScatterKernelVecType<Vec1>(reinterpret_cast<const int8_t*>(src), reinterpret_cast<int8_t*>(dst), embed_width_in_bytes, embed_src_stride_in_bytes, embed_dst_stride_in_bytes, hit_mask, num_indices, stream);
    } 
}

}  // namespace nve

