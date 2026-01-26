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

#include "include/thread_pool.hpp"
#include "include/ecache/ec_set_associative.cuh"
#include "include/execution_context.hpp"
#include "include/common.hpp"
#include <vector>
#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"
#include "cuda_ops/scatter.cuh"
#include <cub/cub.cuh>

namespace nve {

struct GatherKernelPipelineParams {
    uint64_t task_size;
    uint64_t num_aux_streams;
};

template<uint32_t SubwarpWidth, typename DataType>
__global__ void EmbedPipelineScatter(const FindOutput* out_buff, size_t num_indices, int8_t* buff, size_t row_size_in_bytes)
{
    const int sample_in_warp = threadIdx.x / SubwarpWidth;
    const int SAMPLES_PER_WARP = 32 / SubwarpWidth;
    const int embed = (blockIdx.x * blockDim.y + threadIdx.y) * SAMPLES_PER_WARP + sample_in_warp;

    if (embed >= num_indices) {
      return;
    }

    const int8_t* embed_src = buff + embed * row_size_in_bytes;
    int8_t* embed_dst = (int8_t*)(out_buff[embed].dst_ptr);

    nve::memcpy_warp<SubwarpWidth, DataType>(embed_dst, embed_src, row_size_in_bytes);
}

template<uint32_t SubwarpWidth, typename DataType>
inline void CallPipelineScatterInner(const FindOutput* out_buff, 
                                size_t num_indices, 
                                int8_t* buff, 
                                size_t row_size_in_bytes,
                                const cudaStream_t stream)
{
    dim3 block_size(32, 4);
    auto indices_per_block = (32 / SubwarpWidth) * block_size.y;
    dim3 grid_size ((static_cast<uint32_t>(num_indices) + indices_per_block - 1) / indices_per_block, 1);
    EmbedPipelineScatter<SubwarpWidth, DataType><<<grid_size, block_size, 0, stream>>>(
            out_buff, num_indices, buff, row_size_in_bytes);
}

inline void CallPipelineScatter(const FindOutput* out_buff, 
                                size_t num_indices, 
                                int8_t* buff, 
                                size_t row_size_in_bytes,
                              const cudaStream_t stream = 0)
{
    if (row_size_in_bytes % 16 == 0) {
        if (sizeof(uint4) * 32 <= row_size_in_bytes) {
            CallPipelineScatterInner<32, uint4>(out_buff, num_indices, buff, row_size_in_bytes, stream);
        }
        else if (sizeof(uint4) * 16 <= row_size_in_bytes) {
            CallPipelineScatterInner<16, uint4>(out_buff, num_indices, buff, row_size_in_bytes, stream);
        }
        else if (sizeof(uint4) * 8 <= row_size_in_bytes) {
            CallPipelineScatterInner<8, uint4>(out_buff, num_indices, buff, row_size_in_bytes, stream);
        }
        else if (sizeof(uint4) * 4 <= row_size_in_bytes) {
            CallPipelineScatterInner<4, uint4>(out_buff, num_indices, buff, row_size_in_bytes, stream);
        }
        else if (sizeof(uint4) * 2 <= row_size_in_bytes) {
            CallPipelineScatterInner<2, uint4>(out_buff, num_indices, buff, row_size_in_bytes, stream);
        }
        else if (sizeof(uint4) * 1 <= row_size_in_bytes) {
            CallPipelineScatterInner<1, uint4>(out_buff, num_indices, buff, row_size_in_bytes, stream);
        }
    } else if (row_size_in_bytes % 4 == 0) {
        CallPipelineScatterInner<32, uint32_t>(out_buff, num_indices, buff, row_size_in_bytes, stream);
    } else {
        CallPipelineScatterInner<32, int8_t>(out_buff, num_indices, buff, row_size_in_bytes, stream);
    }
    NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
}

template<typename IndexT>
cudaError_t launchGpuGatherKernel(const FindOutput* d_sorted_find_output, 
                                  uint64_t num_keys_for_gather, 
                                  size_t row_size_in_bytes, 
                                  uint32_t num_keys_per_y,
                                  cudaStream_t gather_stream)
{
    constexpr auto block_y = 4;
    dim3 gather_block_dims(32, block_y);
    const auto num_keys_per_block_gather = num_keys_per_y*block_y;
    constexpr auto unroll = 8;
    dim3 gather_grid_dims(static_cast<uint32_t>((num_keys_for_gather + num_keys_per_block_gather - 1)/num_keys_per_block_gather));
    
    if (row_size_in_bytes % 16 == 0)
    {
        nve::gather<IndexT, uint4, block_y, unroll, 32><<<gather_grid_dims, gather_block_dims, 0, gather_stream>>>(d_sorted_find_output, num_keys_for_gather, num_keys_per_y, static_cast<uint32_t>(row_size_in_bytes));
        return cudaGetLastError();
    } else if (row_size_in_bytes % 4 == 0)
    {
        nve::gather<IndexT, uint32_t, block_y, unroll, 32><<<gather_grid_dims, gather_block_dims, 0, gather_stream>>>(d_sorted_find_output, num_keys_for_gather, num_keys_per_y, static_cast<uint32_t>(row_size_in_bytes));
        return cudaGetLastError();
    } else
    {
        nve::gather<IndexT, uint8_t, block_y, unroll, 32><<<gather_grid_dims, gather_block_dims, 0, gather_stream>>>(d_sorted_find_output, num_keys_for_gather, num_keys_per_y, static_cast<uint32_t>(row_size_in_bytes));
        return cudaGetLastError();
    }
}

template<uint32_t CHUNK_COPY_SIZE, uint32_t vec_size>
void processGatherChunk(const FindOutput* h_sorted_find_output,
                       int8_t* h_values,
                       size_t value_stride,
                       size_t row_size_in_bytes,
                       uint64_t base_key,
                       uint64_t start_row,
                       uint64_t end_row)
{   
    for (uint64_t i = start_row; i < end_row; i += vec_size) {
        int8_t* src_ptr_arr[vec_size];
        int8_t* dst_ptr_arr[vec_size];
        for (uint64_t j = 0; j < vec_size; j++) {
            
            FindOutput key = reinterpret_cast<const FindOutput*>(h_sorted_find_output)[(base_key + i + j)];
            src_ptr_arr[j] = reinterpret_cast<int8_t*>(key.src_ptr);
            dst_ptr_arr[j] = reinterpret_cast<int8_t*>(reinterpret_cast<int8_t*>(h_values) + (base_key + i + j) * value_stride);
        }

        for (uint64_t j = 0; j < row_size_in_bytes; j += CHUNK_COPY_SIZE) {
            for (uint64_t k = 0; k < vec_size; k++) {
                std::memcpy(dst_ptr_arr[k] + j, src_ptr_arr[k] + j, CHUNK_COPY_SIZE);
            }
        }        
    }
}

template<uint32_t CHUNK_COPY_SIZE, uint32_t vec_size>
void gather_task_impl(uint64_t n,
                       const FindOutput* h_sorted_find_output,
                       int8_t* h_values,
                       size_t value_stride,
                       size_t row_size_in_bytes,
                       size_t idx,
                       const FindOutput* d_sorted_find_output,
                       uint64_t keys_per_task,
                       std::vector<cudaStream_t>& aux_streams,
                       bool bDoScatterInThread,
                       GatherKernelPipelineParams params)
{
    const auto base_key = (idx) * keys_per_task;
    uint64_t main_loop_size = keys_per_task;
    uint64_t rem = 0;
    if (base_key >= n) {
        main_loop_size = 0;
        rem = 0;
    } else if (base_key + keys_per_task >= n) {
        main_loop_size = ((n - base_key) / vec_size) * vec_size;
        assert(main_loop_size <= keys_per_task);
        assert(main_loop_size % vec_size == 0);
        assert(base_key + main_loop_size <= n);
        rem  = n - (base_key + main_loop_size);
    }
    // process main loop with unroll of vec_size
    processGatherChunk<CHUNK_COPY_SIZE, vec_size>(h_sorted_find_output, h_values, value_stride, row_size_in_bytes, base_key, 0, main_loop_size);
    // process rem with unroll of 1
    if (rem > 0) {
        processGatherChunk<CHUNK_COPY_SIZE, 1>(h_sorted_find_output, h_values, value_stride, row_size_in_bytes, base_key, main_loop_size, main_loop_size + rem);
    }
    
    
    if (bDoScatterInThread)
    {
        const FindOutput* out_buff = (d_sorted_find_output)+(base_key);
        int8_t* buff = reinterpret_cast<int8_t*>(h_values) + (base_key) * value_stride;
        auto stream = aux_streams[(idx % (params.num_aux_streams))];
        CallPipelineScatter(out_buff, main_loop_size + rem, buff, row_size_in_bytes, stream);
    }
}

template<typename IndexT, typename TagT>
void executeCpuGatherPhase(std::shared_ptr<nve::ExecutionContext> ctx,
                          const FindOutput* h_sorted_find_output,
                          int8_t* h_values,
                          size_t value_stride,
                          size_t row_size_in_bytes,
                          uint64_t n,
                          const FindOutput* d_sorted_find_output,
                          cudaEvent_t event_copy_find_output,
                          std::vector<cudaStream_t>& aux_streams,
                          GatherKernelPipelineParams params,
                          int device_id)
{
    if (n == 0) {
        return;
    }
    constexpr auto vec_size = 1;
    const uint64_t num_threads = ctx->get_thread_pool()->num_workers();
    uint64_t min_task_size = ((n + num_threads - 1) / num_threads);
    min_task_size = ((min_task_size + vec_size - 1) / vec_size) * vec_size; // round up to a multiple of vec_size
    auto keys_per_task = (params.task_size == 0) ? min_task_size : std::min(params.task_size, min_task_size);
    uint64_t num_tasks = (n + keys_per_task - 1) / keys_per_task;
    const bool bDoScatterInThread = (params.num_aux_streams != 0);
    const auto gather_task = [=, &aux_streams] (const size_t idx) {
        ScopedDevice scope_device(device_id);
        #define CALL_GATHER_TASK_IMPL(CHUNK_COPY_SIZE) \
        gather_task_impl<CHUNK_COPY_SIZE, vec_size>(n, \
                h_sorted_find_output, \
                    h_values, \
                    value_stride, \
                    row_size_in_bytes, \
                    idx, \
                    d_sorted_find_output, \
                    keys_per_task, \
                    aux_streams, \
                    bDoScatterInThread, \
                    params);
        if (row_size_in_bytes % 256 == 0) {
            CALL_GATHER_TASK_IMPL(256);
        } else if (row_size_in_bytes % 128 == 0) {
            CALL_GATHER_TASK_IMPL(128);
        } else if (row_size_in_bytes % 64 == 0) {
            CALL_GATHER_TASK_IMPL(64);
        } else if (row_size_in_bytes % 32 == 0) {
            CALL_GATHER_TASK_IMPL(32);
        } else if (row_size_in_bytes % 16 == 0) {
            CALL_GATHER_TASK_IMPL(16);
        } else if (row_size_in_bytes % 8 == 0) {
            CALL_GATHER_TASK_IMPL(8);
        } else {
            CALL_GATHER_TASK_IMPL(1);
        }
    #undef CALL_GATHER_TASK_IMPL
    };

    auto thread_pool = ctx->get_thread_pool();
    NVE_CHECK_(cudaEventSynchronize(event_copy_find_output));
    
    thread_pool->execute_n(0, num_tasks, gather_task);
    
    if (!bDoScatterInThread)
    {
        CallPipelineScatter(d_sorted_find_output, n, h_values, row_size_in_bytes, aux_streams[0]);
    }
    else
    {
        for (uint64_t i = 0; i < params.num_aux_streams; i++) {
            NVE_CHECK_(cudaStreamSynchronize(aux_streams[i]));
        }
    }
}

template<typename IndexT, typename TagT>
void gather_flow_pipeline(std::shared_ptr<nve::ExecutionContext> ctx,
              std::shared_ptr<nve::EmbedCacheSA<IndexT, TagT>> cache_ptr,
              LookupContextHandle lookup_handle,
              uint64_t n,
              const IndexT* d_keys, 
              void* d_values, 
              size_t value_stride,               
              const int8_t* uvm_table_ptr,
              size_t row_size_in_bytes,
              cudaStream_t stream,
              int device_id,
              GatherKernelPipelineParams params)
{
    auto cache_data = cache_ptr->get_cache_data(lookup_handle); 
    auto aux_streams = ctx->get_aux_streams("gather_kernel_aux_streams", std::max(params.num_aux_streams, 1lu));
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    // setup
    using SortKeyType = IndexT;

    SortKeyType* d_sort_key_buf = nullptr;
    SortKeyType* d_sort_key_sorted_buf = nullptr;
    FindOutput* d_find_output = nullptr;
    FindOutput* d_sorted_find_output = nullptr;
    FindOutput* h_sorted_find_output = nullptr;
    int8_t* d_cub_aux_buf = nullptr;
    
    int8_t* h_values = nullptr;

    size_t find_bytes = sizeof(FindOutput)*n;
    size_t sortKeyBytes = sizeof(SortKeyType)*n;
    size_t findSortedBytes = find_bytes;
    size_t sortKeySortedBytes = sortKeyBytes;
    size_t cub_aux_bytes = 0;
    NVE_CHECK_(cub::DeviceRadixSort::SortPairs(d_cub_aux_buf, cub_aux_bytes,
            d_sort_key_buf, d_sort_key_sorted_buf, d_find_output, d_sorted_find_output, n, 0, sizeof(SortKeyType)*8, stream));

    std::vector<cudaEvent_t> events;

    d_sort_key_buf = (SortKeyType*)ctx->get_buffer("d_gather_kernel_sort_key_buf", sortKeyBytes, false);
    d_sort_key_sorted_buf = (SortKeyType*)ctx->get_buffer("d_gather_kernel_sorted_key_buf", sortKeySortedBytes, false);
    d_find_output = (FindOutput*)ctx->get_buffer("d_gather_kernel_find_output", find_bytes, false);
    d_sorted_find_output = (FindOutput*)ctx->get_buffer("d_gather_kernel_sorted_find_output", findSortedBytes, false);
    h_sorted_find_output = (FindOutput*)ctx->get_buffer("h_gather_kernel_sorted_find_output", findSortedBytes, true);
    d_cub_aux_buf = (int8_t*)ctx->get_buffer("d_gather_kernel_cub_aux_buf", cub_aux_bytes, false);
    
    h_values = (int8_t*)ctx->get_buffer("h_values", n * row_size_in_bytes, true);
    
    cudaEvent_t event_misses;
    cudaEvent_t event_sort;
    cudaEvent_t event_copy_find_output;
    cudaEvent_t event_gpu_gather_done;
    NVE_CHECK_(cudaEventCreate(&event_misses));
    NVE_CHECK_(cudaEventCreate(&event_sort));
    NVE_CHECK_(cudaEventCreate(&event_copy_find_output));
    NVE_CHECK_(cudaEventCreate(&event_gpu_gather_done));
    events.push_back(event_misses);
    events.push_back(event_sort);
    events.push_back(event_copy_find_output);
    events.push_back(event_gpu_gather_done);
    auto thread_pool = ctx->get_thread_pool();

    //////////////////////////////////////////////////////////////////////////////////////////////
    // find phase (calculate misses)
    int64_t prev_misses = 0;
    int64_t curr_misses = 0;
    cudaMemcpyAsync(&prev_misses, cache_data.misses, sizeof(int64_t), cudaMemcpyDefault, stream);
    constexpr auto num_keys_per_block = 32*2;
    dim3 blockDims(32, 2);
    dim3 gridDims(static_cast<uint32_t>((n + num_keys_per_block - 1)/num_keys_per_block));
    cache_ptr->start_custom_flow();
    find<IndexT, TagT, 2, SortKeyType><<<gridDims, blockDims, 0, stream>>>(uvm_table_ptr,
                            reinterpret_cast<int8_t*>(d_values),
                            d_sort_key_buf,
                            d_find_output,
                            d_keys,
                            n,
                            cache_data);
    NVE_CHECK_(cudaGetLastError());
    NVE_CHECK_(cudaMemcpyAsync(&curr_misses, cache_data.misses, sizeof(int64_t), cudaMemcpyDefault, stream));
    NVE_CHECK_(cudaEventRecord(event_misses, stream));

    //////////////////////////////////////////////////////////////////////////////////////////////
    // sort phase
    NVE_CHECK_(cub::DeviceRadixSort::SortPairs(d_cub_aux_buf, cub_aux_bytes,
            d_sort_key_buf, d_sort_key_sorted_buf, d_find_output, d_sorted_find_output, n, 0, sizeof(SortKeyType)*8, stream));

    NVE_CHECK_(cudaEventRecord(event_sort, stream));
    
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    // get misses from find phase
    NVE_CHECK_(cudaEventSynchronize(event_misses));
    int64_t misses = curr_misses - prev_misses;

    //////////////////////////////////////////////////////////////////////////////////////////////
    // copy sorted values to host
    // might not need this stream
    
    NVE_CHECK_(cudaMemcpyAsync(h_sorted_find_output, d_sorted_find_output, misses * sizeof(FindOutput), cudaMemcpyDefault, stream));
    NVE_CHECK_(cudaEventRecord(event_copy_find_output, stream));
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    // gpu gather phase in parrllel with cpu gather phase
    auto gather_stream = aux_streams[0];
    auto num_keys_for_gather = n - misses;
    if (num_keys_for_gather > 0) {
        auto constexpr num_keys_per_y = 128;
        NVE_CHECK_(cudaStreamWaitEvent(gather_stream, event_sort, 0));
        NVE_CHECK_(launchGpuGatherKernel<IndexT>(d_sorted_find_output + misses, num_keys_for_gather, row_size_in_bytes, num_keys_per_y, gather_stream));
        NVE_CHECK_(cudaEventRecord(event_gpu_gather_done, gather_stream));
        NVE_CHECK_(cudaStreamWaitEvent(stream, event_gpu_gather_done, 0));
    }
    cache_ptr->end_custom_flow();
    
    //////////////////////////////////////////////////////////////////////////////////////////////
    // cpu gather phase
    executeCpuGatherPhase<IndexT, TagT>(ctx, h_sorted_find_output, h_values, value_stride, row_size_in_bytes, misses, d_sorted_find_output, event_copy_find_output, aux_streams, params, device_id);

    for (auto event : events) {
        NVE_CHECK_(cudaEventDestroy(event));
    }
}

}
