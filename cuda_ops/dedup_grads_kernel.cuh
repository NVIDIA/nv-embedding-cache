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
#include "cuda_ops/cuda_common.h"
#include "cuda_ops/kernels_common.cuh"
#include "include/nve_types.hpp"

#define LOOP(COUNT, STEP) \
    { \
      for (uint32_t i = 0; i < COUNT; i+= STEP) { \
        const IndexType thread_idx = i + threadIdx.x; \
        const IndexType thread_offset = inverse_buffer[offset + thread_idx]; \
        for (uint32_t j = 0; j < STEP; j++) \
        {  \
            const IndexType current_offset = __shfl_sync(0xffffffff, thread_offset, j); \
            const DataType* embed_src = src + current_offset * embed_src_stride + el + threadIdx.x; \
            if ((el + threadIdx.x) < num_elements) { \
                nve::Accumulate(acc, *embed_src); \
            } \
        }\
      }\
    }

#define LOOP_SHORT(COUNT, STEP) \
    { \
      IndexType thread_offset = 0; \
      for (uint32_t i = 0; i < COUNT; i+= STEP) { \
        const IndexType thread_idx = i + threadIdx.x; \
        if (thread_idx < COUNT) thread_offset = inverse_buffer[offset + thread_idx]; \
        for (uint32_t j = 0; j < STEP; j++) \
        {  \
            const IndexType current_offset = __shfl_sync(0xffffffff, thread_offset, j); \
            const DataType* embed_src = src + current_offset * embed_src_stride + el + threadIdx.x;\
            if ((el + threadIdx.x) < num_elements) { \
                nve::Accumulate(acc, *embed_src); \
            } \
        }\
      }\
    }

template<uint32_t SubwarpWidth, typename DataType, typename IndexType>
__global__ void GradientDedup(const DataType* __restrict__ src,
                              DataType* __restrict__ dst,
                              const IndexType* __restrict__ unique_keys,
                              const IndexType* __restrict__ inverse_buffer,
                              const IndexType* dst_loc_map,
                              const IndexType* key_offsets,
                              const uint64_t num_unique_keys,                                                        
                              const uint32_t num_elements,
                              const uint32_t embed_src_stride,
                              const uint32_t embed_dst_stride)
{
    const int key_id = blockIdx.x * blockDim.y + threadIdx.y;

    if (key_id >= num_unique_keys) {
      return;
    }

    const IndexType count = key_offsets[key_id + 1] - key_offsets[key_id];
    const IndexType offset = key_offsets[key_id];
    const IndexType dst_id = (dst_loc_map == nullptr) ? key_id : dst_loc_map[key_id];

    for (uint32_t el = 0; el < num_elements; el += SubwarpWidth) {
      DataType acc;
      nve::InitAcc(acc);

      uint32_t main_loop_cnt, rem_loop_cnt;
      if (count < 8) {
        LOOP_SHORT(count, 1);
        rem_loop_cnt = 0;
      } else if (count < 32) {
        main_loop_cnt = (count / 8) * 8;
        rem_loop_cnt = count - main_loop_cnt;
        LOOP_SHORT(main_loop_cnt, 8);
      } else {
        main_loop_cnt = (count / 32) * 32;
        LOOP(main_loop_cnt, 32);
        rem_loop_cnt = count - main_loop_cnt;
      }

      if ((el + threadIdx.x) < num_elements) {
        for (uint32_t i = 0; i < rem_loop_cnt; i++) {
          const IndexType current_offset = inverse_buffer[offset + main_loop_cnt + i];
          const DataType* embed_src = src + current_offset * embed_src_stride + el + threadIdx.x;
           nve::Accumulate(acc, *embed_src);    
        }
      }

      if ((el + threadIdx.x) < num_elements) {
        DataType* embed_dst = dst + dst_id * embed_dst_stride + el + threadIdx.x;
        nve::AtomicAccumulate(embed_dst, acc);
      }
    }
}

template<uint32_t SubwarpWidth, typename DataType, typename IndexType>
void CallGradientDedupKernelVecTypeSubwarp(
                              const DataType* __restrict__ src,
                              DataType* __restrict__ dst,
                              const IndexType* __restrict__ unique_keys,
                              const IndexType* __restrict__ inverse_buffer,
                              const IndexType* dst_loc_map,
                              const IndexType* key_offsets,
                              const uint64_t num_unique_keys,                                                        
                              const uint32_t num_elements,
                              const uint32_t embed_src_stride,
                              const uint32_t embed_dst_stride,
                              const cudaStream_t stream = 0)
{
    const uint32_t WARPS_PER_SM = 4;
    uint32_t keys_per_warp = 32 / SubwarpWidth;
    uint32_t keys_per_sm = keys_per_warp * WARPS_PER_SM;
    dim3 grid_size (static_cast<uint32_t>((num_unique_keys + keys_per_sm - 1) / keys_per_sm), 1);
    dim3 block_size (SubwarpWidth, keys_per_sm);
    GradientDedup<SubwarpWidth, DataType, IndexType><<<grid_size, block_size, 0, stream>>>(
        src, dst, unique_keys, inverse_buffer, dst_loc_map, key_offsets,
        num_unique_keys, num_elements, embed_src_stride, embed_dst_stride);
    NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
}

template<typename IndexType>
void CallDedupGradients(const void* __restrict__ src,
                        void* __restrict__ dst,
                        const IndexType* __restrict__ unique_keys,
                        const IndexType* __restrict__ inverse_buffer,
                        const IndexType* dst_loc_map,
                        const IndexType* key_offsets,
                        nve::DataType_t element_format,
                        const IndexType* num_unique_keys,                                                        
                        const uint32_t embed_width_in_bytes,
                        const uint32_t embed_src_stride_in_bytes,
                        const uint32_t embed_dst_stride_in_bytes,
                        const cudaStream_t stream = 0)
{
    assert((element_format == nve::DataType_t::Float16) || (element_format == nve::DataType_t::Float32));
    const uint32_t num_elements = (element_format == nve::DataType_t::Float16) ? (embed_width_in_bytes / 2) : (embed_width_in_bytes / 4);
    auto num_unique_grads = num_unique_keys[0];
    auto num_unique_runs = num_unique_keys[1];

    NVE_CHECK_(cudaMemsetAsync(dst, 0, embed_width_in_bytes * num_unique_grads, stream));

    if ((num_elements % 4) == 0) {
      if (element_format == nve::DataType_t::Float16) {
        using Vec4 = typename nve::VecWidthHelper<__half>::Vec4;
        CallGradientDedupKernelVecTypeSubwarp<32, Vec4, IndexType>(
            reinterpret_cast<const Vec4*>(src), reinterpret_cast<Vec4*>(dst), 
            unique_keys, inverse_buffer, dst_loc_map, key_offsets, num_unique_runs,
            num_elements / 4, embed_src_stride_in_bytes / (4 * sizeof(__half)), 
            embed_dst_stride_in_bytes / (4 * sizeof(__half)), stream);
      } else {
        using Vec4 = typename nve::VecWidthHelper<float>::Vec4;
        CallGradientDedupKernelVecTypeSubwarp<32, Vec4, IndexType>(
            reinterpret_cast<const Vec4*>(src), reinterpret_cast<Vec4*>(dst), 
            unique_keys, inverse_buffer, dst_loc_map, key_offsets, num_unique_runs,
            num_elements / 4, embed_src_stride_in_bytes / (4 * sizeof(float)), 
            embed_dst_stride_in_bytes / (4 * sizeof(float)), stream);
      }
    } else if ((num_elements % 2) == 0) {
      if (element_format == nve::DataType_t::Float16) {
        using Vec2 = typename nve::VecWidthHelper<__half>::Vec2;
        CallGradientDedupKernelVecTypeSubwarp<32, Vec2, IndexType>(
            reinterpret_cast<const Vec2*>(src), reinterpret_cast<Vec2*>(dst), 
            unique_keys, inverse_buffer, dst_loc_map, key_offsets, num_unique_runs,
            num_elements / 2, embed_src_stride_in_bytes / (2 * sizeof(__half)), 
            embed_dst_stride_in_bytes / (2 * sizeof(__half)), stream);
      } else {
        using Vec2 = typename nve::VecWidthHelper<float>::Vec2;
        CallGradientDedupKernelVecTypeSubwarp<32, Vec2, IndexType>(
            reinterpret_cast<const Vec2*>(src), reinterpret_cast<Vec2*>(dst), 
            unique_keys, inverse_buffer, dst_loc_map, key_offsets, num_unique_runs,
            num_elements / 2, embed_src_stride_in_bytes / (2 * sizeof(float)), 
            embed_dst_stride_in_bytes / (2 * sizeof(float)), stream);
      }
    } else {
      if (element_format == nve::DataType_t::Float16) {
        using Vec1 = typename nve::VecWidthHelper<__half>::Vec1;
        CallGradientDedupKernelVecTypeSubwarp<32, Vec1, IndexType>(
            reinterpret_cast<const Vec1*>(src), reinterpret_cast<Vec1*>(dst), 
            unique_keys, inverse_buffer, dst_loc_map, key_offsets, num_unique_runs,
            num_elements, embed_src_stride_in_bytes / sizeof(__half), 
            embed_dst_stride_in_bytes / sizeof(__half), stream);
      } else {
        using Vec1 = typename nve::VecWidthHelper<float>::Vec1;
        CallGradientDedupKernelVecTypeSubwarp<32, Vec1, IndexType>(
            reinterpret_cast<const Vec1*>(src), reinterpret_cast<Vec1*>(dst), 
            unique_keys, inverse_buffer, dst_loc_map, key_offsets, num_unique_runs,
            num_elements, embed_src_stride_in_bytes / sizeof(float), 
            embed_dst_stride_in_bytes / sizeof(float), stream);
      }
    }
}

template<typename IndexT>
void DedupGradients(const void* d_gradients, void* d_unique_gradients, nve::DataType_t element_format,
    const IndexT* d_unique_keys, const IndexT* d_counts, const IndexT* d_offsets, const IndexT* d_inverse_buffer,
    uint64_t embedding_width_in_bytes, IndexT* num_unique_keys, cudaStream_t stream)
{
    CallDedupGradients(d_gradients, d_unique_gradients, d_unique_keys, d_inverse_buffer, d_counts, d_offsets,
        element_format, num_unique_keys, static_cast<uint32_t>(embedding_width_in_bytes), 
        static_cast<uint32_t>(embedding_width_in_bytes), static_cast<uint32_t>(embedding_width_in_bytes), stream);
}

template<typename IndexType, typename DataType>
void CallComputeGradients(const DataType* __restrict__ src,
                          DataType* __restrict__ dst,
                          const IndexType* __restrict__ unique_keys,
                          const IndexType* __restrict__ inverse_buffer,
                          const IndexType* dst_loc_map,
                          const IndexType* key_offsets,
                          const IndexType* num_unique_keys,                                                        
                          const uint32_t embed_width,
                          const cudaStream_t stream = 0)
{
    auto num_unique_grads = num_unique_keys[0];
    auto num_unique_runs = num_unique_keys[1];

    NVE_CHECK_(cudaMemsetAsync(dst, 0, embed_width * sizeof(DataType) * num_unique_grads, stream));

    if ((embed_width % 4) == 0) {
      using Vec4 = typename nve::VecWidthHelper<DataType>::Vec4;
      CallGradientDedupKernelVecTypeSubwarp<32, Vec4, IndexType>(
          reinterpret_cast<const Vec4*>(src), reinterpret_cast<Vec4*>(dst), 
          unique_keys, inverse_buffer, dst_loc_map, key_offsets, num_unique_runs,
          embed_width / 4, embed_width / 4, embed_width / 4, stream);
    } else if ((embed_width % 2) == 0) {
      using Vec2 = typename nve::VecWidthHelper<DataType>::Vec2;
      CallGradientDedupKernelVecTypeSubwarp<32, Vec2, IndexType>(
          reinterpret_cast<const Vec2*>(src), reinterpret_cast<Vec2*>(dst), 
          unique_keys, inverse_buffer, dst_loc_map, key_offsets, num_unique_runs,
          embed_width / 2, embed_width / 2, embed_width / 2, stream);
    } else {
      using Vec1 = typename nve::VecWidthHelper<DataType>::Vec1;
        CallGradientDedupKernelVecTypeSubwarp<32, Vec1, IndexType>(
            reinterpret_cast<const Vec1*>(src), reinterpret_cast<Vec1*>(dst), 
            unique_keys, inverse_buffer, dst_loc_map, key_offsets, num_unique_runs,
            embed_width, embed_width, embed_width, stream);
    }
}
