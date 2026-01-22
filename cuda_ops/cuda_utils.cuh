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
#include <cub/cub.cuh>


template<typename IndexType, int SPLIT_SIZE = 1024>
__global__ void ComputeSplitCounts(const IndexType* __restrict__ key_counts,
                                   IndexType* __restrict__ num_split_chunks,
                                   const uint64_t num_unique_keys)
{
    const int key_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (key_id < num_unique_keys) {
        num_split_chunks[key_id] = (key_counts[key_id] + SPLIT_SIZE - 1) / SPLIT_SIZE;
    }
}

template<typename IndexT, int SPLIT_SIZE = 1024>
void CallComputeSplitCounts(const IndexT* key_counts, IndexT* num_split_chunks, uint64_t num_counts, cudaStream_t stream)
{
    const uint32_t THREADS_PER_SM = 128;
    dim3 grid_size (static_cast<uint32_t>((num_counts + THREADS_PER_SM - 1) / THREADS_PER_SM), 1);
    dim3 block_size (THREADS_PER_SM, 1);
    ComputeSplitCounts<IndexT, SPLIT_SIZE><<<grid_size, block_size, 0, stream>>>(key_counts, num_split_chunks, num_counts);
    NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
}

template<typename IndexType, int SPLIT_SIZE = 1024>
__global__ void ComputeLocationMapping(const IndexType* __restrict__ chunk_counts,
                                       const IndexType* __restrict__ chunk_offsets,
                                       const IndexType* __restrict__ key_offsets,
                                       IndexType* __restrict__ output_location_mapping,
                                       IndexType* __restrict__ split_key_offsets,
                                       const uint64_t num_unique_keys)
{
    const int key_id = blockIdx.x * blockDim.y + threadIdx.y;
    if (key_id < num_unique_keys) {
        IndexType count = chunk_counts[key_id];
        auto location_map_p = output_location_mapping + chunk_offsets[key_id];
        auto split_offsets_p = split_key_offsets + chunk_offsets[key_id];
        for (IndexType i = threadIdx.x; i < count; i += warpSize) {
            location_map_p[i] = key_id;
            split_offsets_p[i] = key_offsets[key_id] + i*SPLIT_SIZE;
        }
        if ((key_id + 1) == num_unique_keys) {
            split_offsets_p[count] = key_offsets[key_id + 1];
        }
    }
}

template<typename IndexT, int SPLIT_SIZE = 1024>
void CallComputeLocationMapping(const IndexT* chunk_counts,
                                const IndexT* chunk_offsets,
                                const IndexT* __restrict__ key_offsets,
                                IndexT* output_location_mapping,
                                IndexT* __restrict__ split_key_offsets,
                                uint64_t num_counts, cudaStream_t stream)
{
    const uint32_t WARPS_PER_SM = 4;
    dim3 grid_size (static_cast<uint32_t>((num_counts + WARPS_PER_SM - 1) / WARPS_PER_SM), 1);
    dim3 block_size (32, WARPS_PER_SM);
    ComputeLocationMapping<IndexT, SPLIT_SIZE><<<grid_size, block_size, 0, stream>>>(
        chunk_counts, chunk_offsets, key_offsets, output_location_mapping, split_key_offsets, num_counts);
    NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
}

template<typename IndexT, int32_t MAX_RUN_SIZE = -1>
class Deduper
{
public:
    Deduper() {
        NVE_CHECK_(cudaMalloc(&m_d_num_runs_out, sizeof(IndexT)));
    }

    ~Deduper()
    {
        NVE_CHECK_(cudaFree(m_d_num_runs_out));
    }

    // input d_keys, an array of size num_keys with keys to de duplicate
    // return - 
    //  1. d_unique_out - device buffer of unique keys,
    //  2. d_counts_out - device buffer of number of duplicates per key
    //  3. d_loc_map_out - device buffer required when maximal run size is limited
    //     for each sub-run this buffer will hold the unique key id it belongs to 
    //  4. h_num_runs_out - size of array for d_unique_out (and d_counts_out)
    //  5. d_inverse_buffer - device buffer to map unique keys to their original location
    //  6. d_offsets buffer - device buffer to record the where each inverse of each key starts
    //     offsets can be used to compute key counters, as d_counts_out[i] = d_offsets[i+1] - d_offsets[i]
    //  inverse example: find all locations of d_unique_out[i]:
    //      inverse_locations[d_counts[i]]
    //      for j < d_counts[i]
    //          inverse_locations[j] = d_inverse_buffer[d_offsets[i]+j]
    void Dedup(const IndexT* d_keys, 
        const uint64_t num_keys, 
        IndexT* d_unique_out, 
        IndexT* d_counts_out,
        IndexT* d_loc_map_out, 
        IndexT* h_num_runs_out, 
        IndexT* d_inverse_buffer, 
        IndexT* d_offsets, 
        cudaStream_t stream)
    {
        // Deduping using CUB operands
        // 1. sort enumrate(d_keys) 
        // 2. Perform run length encoding on the sorted array 
        // 3. perform exclusive sum on the length array to get the offsets for each key inverse mapping 

        // Run sorting operation
        NVE_CHECK_(cudaMemcpyAsync(m_d_location_buffer, m_h_location_buffer, sizeof(IndexT)*num_keys, cudaMemcpyDefault, stream));
        
        NVE_CHECK_(cub::DeviceRadixSort::SortPairs(m_d_temp_storage_sort, m_temp_storage_bytes_sort,
            d_keys, m_d_sorted_buffer, m_d_location_buffer, d_inverse_buffer, num_keys, 0, sizeof(IndexT)*8, stream));

        // set d_counts_out to zeros and call ExclusiveSum on + 1, to make sure prefix sum 
        // returns sum of all counts in last element
        NVE_CHECK_(cudaMemsetAsync(d_counts_out, 0, num_keys * sizeof(IndexT), stream));

        // Run encoding
        NVE_CHECK_(cub::DeviceRunLengthEncode::Encode(
            m_d_temp_storage_encode, m_temp_storage_bytes_encode,
            m_d_sorted_buffer, d_unique_out, d_counts_out, m_d_num_runs_out, static_cast<int>(num_keys), stream));
        
        NVE_CHECK_(cudaMemcpyAsync(h_num_runs_out, m_d_num_runs_out, sizeof(IndexT), cudaMemcpyDefault, stream));
        NVE_CHECK_(cudaStreamSynchronize(stream));

        IndexT* d_output_offset_buffer = (MAX_RUN_SIZE != -1) ? m_d_tmp_offset_buffer : d_offsets;
        auto num_unique_out = static_cast<int>(*h_num_runs_out);
        
        // Run exclusive prefix sum
        NVE_CHECK_(cub::DeviceScan::ExclusiveSum(
            m_d_temp_storage_ex_sum, m_temp_storage_bytes_ex_sum,
            d_counts_out, d_output_offset_buffer, num_unique_out + 1, stream));

        // split runs to chunks of limited length, if required
        if (MAX_RUN_SIZE != -1) {
             // Run split count compute
             CallComputeSplitCounts<IndexT, MAX_RUN_SIZE>(d_counts_out, m_d_split_count_buffer, num_unique_out, stream);

            // Run exclusive prefix sum on split data
            NVE_CHECK_(cub::DeviceScan::ExclusiveSum(
                m_d_temp_storage_ex_sum, m_temp_storage_bytes_ex_sum,
                m_d_split_count_buffer, m_d_split_offset_buffer, num_unique_out + 1, stream));

            // The number of chunks is the last element in ExclusiveSum output
            // Return both numbers, because both are needed for backward path
            NVE_CHECK_(cudaMemcpyAsync(h_num_runs_out + 1, m_d_split_offset_buffer + num_unique_out, sizeof(IndexT), cudaMemcpyDefault, stream));

            // Compute final input/output offsets compute
            CallComputeLocationMapping<IndexT, MAX_RUN_SIZE>(
                m_d_split_count_buffer,
                m_d_split_offset_buffer,
                m_d_tmp_offset_buffer,
                d_loc_map_out,
                d_offsets,
                num_unique_out,
                stream);
        } else {
            h_num_runs_out[1] = h_num_runs_out[0];
        }
        NVE_CHECK_(cudaStreamSynchronize(stream));
    }

    IndexT* GetSorted() const
    {
        return m_d_sorted_buffer;
    }

    void GetAllocRequirements(uint64_t num_keys, size_t& tmp_mem_size_device, size_t& tmp_mem_size_host) {
        tmp_mem_size_host = 0;
        tmp_mem_size_device = 0;

        IndexT* p = nullptr;
        {
            // determing and allocate temporary storage for sort
            NVE_CHECK_(cub::DeviceRadixSort::SortPairs(p, m_temp_storage_bytes_sort,
                    p, p, p, p, num_keys, 0, sizeof(IndexT)*8));
            // align everyrhing to 512B
            m_temp_storage_bytes_sort = ((m_temp_storage_bytes_sort + 511) >> 9) << 9;
            tmp_mem_size_device += m_temp_storage_bytes_sort;
        }
        {
            // determing and allocate temporary storage for encode
            NVE_CHECK_(cub::DeviceRunLengthEncode::Encode(
                    p, m_temp_storage_bytes_encode,
                    p, p, p, p, static_cast<int>(num_keys)));

            m_temp_storage_bytes_encode = ((m_temp_storage_bytes_encode + 511) >> 9) << 9;
            tmp_mem_size_device += m_temp_storage_bytes_encode;
        }
        {
            // determing and allocate temporary storage for ex sum
            // ExclusiveSum will be called on one extra input (0)
            // in order to get last offset equal to number of input keys
            int num_keys_ = static_cast<int>(num_keys) + 1;

            NVE_CHECK_(cub::DeviceScan::ExclusiveSum(
                    p, m_temp_storage_bytes_ex_sum,
                    p, p, num_keys_));
            m_temp_storage_bytes_ex_sum = ((m_temp_storage_bytes_ex_sum + 511) >> 9) << 9;
            tmp_mem_size_device += m_temp_storage_bytes_ex_sum;
        }
        size_t index_buffer_size = sizeof(IndexT) * (num_keys + 1);
        index_buffer_size = ((index_buffer_size + 1023) >> 10) << 10;

        tmp_mem_size_device += index_buffer_size; //m_d_sorted_buffer
        tmp_mem_size_device += index_buffer_size; //m_d_location_buffer

        if (MAX_RUN_SIZE != -1) {
            tmp_mem_size_device += index_buffer_size; // m_d_split_count_buffer
            tmp_mem_size_device += index_buffer_size; //m_d_tmp_offset_buffer
            tmp_mem_size_device += index_buffer_size; //m_d_split_offset_buffer
        }
        tmp_mem_size_host += index_buffer_size; //m_h_location_buffer
    }

    void SetAndInitBuffers(uint64_t num_keys, char* tmp_mem_device, char* tmp_mem_host) {
        char* curr_device_ptr = tmp_mem_device;
        size_t index_buffer_size = sizeof(IndexT) * (num_keys + 1);
        index_buffer_size = ((index_buffer_size + 511) >> 9) << 9;

        m_d_temp_storage_sort = curr_device_ptr;
        curr_device_ptr += m_temp_storage_bytes_sort;
        m_d_temp_storage_encode = curr_device_ptr;
        curr_device_ptr += m_temp_storage_bytes_encode;
        m_d_temp_storage_ex_sum = curr_device_ptr;
        curr_device_ptr += m_temp_storage_bytes_ex_sum;
        m_d_sorted_buffer = reinterpret_cast<IndexT*>(curr_device_ptr);
        curr_device_ptr += index_buffer_size;
        m_d_location_buffer = reinterpret_cast<IndexT*>(curr_device_ptr);
        curr_device_ptr += index_buffer_size;

        if (MAX_RUN_SIZE != -1) {
            m_d_split_count_buffer = reinterpret_cast<IndexT*>(curr_device_ptr);
            curr_device_ptr += index_buffer_size;
            m_d_tmp_offset_buffer = reinterpret_cast<IndexT*>(curr_device_ptr);
            curr_device_ptr += index_buffer_size;
            m_d_split_offset_buffer = reinterpret_cast<IndexT*>(curr_device_ptr);
            curr_device_ptr += index_buffer_size;
        }

        m_h_location_buffer = reinterpret_cast<IndexT*>(tmp_mem_host);
        for (IndexT i = 0; i < static_cast<IndexT>(num_keys); i++) {
            m_h_location_buffer[i] = i;
        }
    }

private:
    size_t   m_temp_storage_bytes_sort = 0;
    size_t   m_temp_storage_bytes_encode = 0;
    size_t   m_temp_storage_bytes_ex_sum = 0;

    void     *m_d_temp_storage_sort = nullptr;
    void     *m_d_temp_storage_encode = nullptr;
    void     *m_d_temp_storage_ex_sum = nullptr;

    IndexT   *m_d_location_buffer = nullptr; // [1, 2, ..., ] helper buffer to build the reverse map
    IndexT   *m_h_location_buffer = nullptr;
    IndexT   *m_d_sorted_buffer = nullptr; // holds the sorted output
    IndexT   *m_d_num_runs_out = nullptr;    // e.g., [ ]
    IndexT   *m_d_split_count_buffer = nullptr;
    IndexT   *m_d_tmp_offset_buffer = nullptr;
    IndexT   *m_d_split_offset_buffer = nullptr;
};

template<typename IndexT, uint32_t SUBWARP_WIDTH, typename DataType>
__global__ void UnpackDedupKernel(const IndexT* d_inverse_buffer, const IndexT* d_offsets, const IndexT* d_counts_out, const int8_t* uniq_buffer, int8_t* out_buffer, uint64_t rowSizeInBytes)
{
    DataType localBuffer[4];
    const uint32_t rl_idx = blockIdx.x; 
    auto intra_warp_idx = threadIdx.x;
    constexpr auto ELEMENT_SIZE = sizeof(DataType);
    // load input
    auto src_ptr = uniq_buffer + rl_idx * rowSizeInBytes;
    for (uint32_t k = 0, h = 0; k < rowSizeInBytes; k += ELEMENT_SIZE*SUBWARP_WIDTH, h++)
    {
        uint32_t offset = k + intra_warp_idx * ELEMENT_SIZE;
        if (offset < rowSizeInBytes)
        {
            DataType d = *(DataType*)(src_ptr + offset);
            localBuffer[h] = d;
        }
    }

    auto num_outputs = d_counts_out[rl_idx];
    for (uint32_t i = 0; i < num_outputs; i++)
    {
        for (uint32_t k = 0, h = 0; k < rowSizeInBytes; k += ELEMENT_SIZE*SUBWARP_WIDTH, h++)
        {
            uint32_t offset = k + intra_warp_idx * ELEMENT_SIZE;
            auto outputIdx = d_inverse_buffer[d_offsets[rl_idx] + i];
            if (offset < rowSizeInBytes)
            {
                DataType* dst_ptr = (DataType*)(out_buffer + outputIdx * rowSizeInBytes + offset);
                *dst_ptr = localBuffer[h];
            }
        }
    }
}

template<typename IndexT>
void UnpackDedup(const IndexT* d_inverse_buffer, const IndexT* d_offsets, const IndexT* d_counts_out, const int8_t* d_uniq_buffer, int8_t* d_out_buffer, uint64_t num_counts, uint64_t rowSizeInBytes, cudaStream_t stream)
{
    const uint32_t blockX = 32;
    const uint32_t blockY = 1;
    const uint32_t nBlock = static_cast<uint32_t>(num_counts);
    dim3 gridDims(nBlock);
    dim3 blockDims(blockX, blockY);
    if (rowSizeInBytes % sizeof(uint4) == 0)
    {
        UnpackDedupKernel<IndexT, 32, uint4><<<gridDims, blockDims, 0, stream>>>(d_inverse_buffer, d_offsets, d_counts_out, d_uniq_buffer, d_out_buffer, rowSizeInBytes);
    }
    else if (rowSizeInBytes % sizeof(uint32_t) == 0)
    {
        UnpackDedupKernel<IndexT, 32, uint32_t><<<gridDims, blockDims, 0, stream>>>(d_inverse_buffer, d_offsets, d_counts_out, d_uniq_buffer, d_out_buffer, rowSizeInBytes);
    }
    else
    {
        UnpackDedupKernel<IndexT, 32, uint8_t><<<gridDims, blockDims, 0, stream>>>(d_inverse_buffer, d_offsets, d_counts_out, d_uniq_buffer, d_out_buffer, rowSizeInBytes);
    }
    NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
}

template<uint32_t SubwarpWidth, typename DataType>
__global__ void EmbedPooling(const DataType* __restrict__ src,
                             DataType* __restrict__ dst,
                             const uint32_t num_elements,
                             const uint32_t embed_src_stride,
                             const uint32_t embed_dst_stride,
                             const uint32_t hotness,
                             const uint32_t num_bags)
{
    const int bag = blockIdx.x * blockDim.y + threadIdx.y;

    if (bag >= num_bags) {
      return;
    }

    for (uint32_t el = threadIdx.x; el < num_elements; el += SubwarpWidth) {
      DataType acc;
      InitAcc(acc);
      for (uint32_t i = 0; i < hotness; i++) {
        const DataType* embed_src = src + (bag * hotness + i) * embed_src_stride + el;
        nve::Accumulate(acc, *embed_src);
      }
      DataType* embed_dst = dst + bag * embed_dst_stride + el;
      *embed_dst = acc;
    }
}

template<uint32_t SubwarpWidth, typename DataType>
void CallPoolingKernelVecTypeSubwarp(
                              const DataType* __restrict__ src,
                              DataType* __restrict__ dst,
                              const uint32_t num_elements,
                              const uint32_t embed_src_stride,
                              const uint32_t embed_dst_stride,
                              const uint32_t hotness,
                              const uint32_t num_bags,
                              const cudaStream_t stream = 0)
{
    uint32_t bags_per_warp = 32 / SubwarpWidth;
    uint32_t bags_per_sm = bags_per_warp * 4;
    dim3 grid_size ((num_bags + bags_per_sm - 1) / bags_per_sm, 1);
    dim3 block_size (SubwarpWidth, bags_per_sm);
    EmbedPooling<SubwarpWidth, DataType><<<grid_size, block_size, 0, stream>>>(
        src, dst, num_elements, embed_src_stride, embed_dst_stride, hotness, num_bags);
    NVE_CHECK_(cudaGetLastError()); // Check kernel launch didn't generate an error
}
