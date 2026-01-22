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
#include <cuda_support.hpp>

namespace nve {

template<typename KeyType>
__global__ void GatherKeysAndDataPtrs(
      int8_t* data,
      KeyType* __restrict__ mapping,
      int* __restrict__ priorities,
      const KeyType* __restrict__ keys,
      float norm_factor,
      int64_t num_unique_keys,
      int64_t embed_width_in_bytes,
      KeyType* __restrict__ unique_keys,
      int8_t** __restrict__ data_ptrs)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= num_unique_keys) {
      return;
    }

    float* priority_out = reinterpret_cast<float*>(priorities);
    KeyType entry = mapping[id];

    data_ptrs[id] = data + entry * embed_width_in_bytes;
    unique_keys[id] = keys[entry];

    priority_out[id] = float(priorities[id]) * norm_factor;
}

template<typename KeyType>
__global__ void GatherLocations(
      int64_t num_unique_keys, int* offsets,
      KeyType* idx_mapping_all, KeyType* idx_mapping_unique) {
    
    const int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= num_unique_keys) {
      return;
    }
    idx_mapping_unique[id] =  idx_mapping_all[offsets[id]];
}

template<typename KeyType>
inline void CallGatherKeysAndDataPtrs(
      const int8_t* __restrict__ data,
      KeyType* __restrict__ mapping,
      int* __restrict__ priorities,
      const KeyType* __restrict__ keys,
      float norm_factor,
      int64_t num_unique_keys,
      int64_t embed_width_in_bytes,
      KeyType* __restrict__ unique_keys,
      int8_t** __restrict__ data_ptrs,
      const cudaStream_t stream = 0)
{
    constexpr uint32_t warp_size = 32;
    constexpr uint32_t warps_per_block = 8;
    constexpr uint32_t indices_per_block = warp_size * warps_per_block;

    dim3 grid_size ((static_cast<uint32_t>(num_unique_keys) + indices_per_block - 1) / indices_per_block, 1);
    dim3 block_size (indices_per_block);
    GatherKeysAndDataPtrs<KeyType><<<grid_size, block_size, 0, stream>>>(
        const_cast<int8_t*>(data), mapping, priorities, keys,
        norm_factor, num_unique_keys, embed_width_in_bytes, unique_keys, data_ptrs);
    NVE_CHECK_(cudaGetLastError());
}

template<typename KeyType>
void CallGatherLocations(
      int64_t num_unique_keys, int* offsets,
      KeyType* idx_mapping_all, KeyType* idx_mapping_unique,
      const cudaStream_t stream = 0) {
    constexpr uint32_t warp_size = 32;
    constexpr uint32_t warps_per_block = 8;
    constexpr uint32_t indices_per_block = warp_size * warps_per_block;

    dim3 grid_size ((static_cast<uint32_t>(num_unique_keys) + indices_per_block - 1) / indices_per_block, 1);
    dim3 block_size (indices_per_block);
    GatherLocations<KeyType><<<grid_size, block_size, 0, stream>>>(num_unique_keys, offsets, idx_mapping_all, idx_mapping_unique);
    NVE_CHECK_(cudaGetLastError());
}

template<typename KeyType>
class DefaultGPUHistogram
{
public:
    DefaultGPUHistogram(int64_t num_keys)
        : m_numKeys(num_keys) {
        computeAllocSize();
    }

    void computeHistogram(const KeyType* keys, int64_t num_keys, const int8_t* data, size_t strideInBytes, void* tmpStorage, cudaStream_t histStream = 0) {
        NVE_CHECK_(num_keys <= m_numKeys, "histogram object was allocated for fewer keys than provided!");
        
        // Allocate internal and temporary buffers
        int8_t* curr_ptr = reinterpret_cast<int8_t*>(tmpStorage);
        m_pDataPtrs = reinterpret_cast<int8_t**>(curr_ptr);
        curr_ptr += num_keys * sizeof(int8_t*);

        KeyType* idx_mapping = reinterpret_cast<KeyType*>(curr_ptr);
        curr_ptr += num_keys * sizeof(KeyType);
        KeyType* idx_mapping_sorted = reinterpret_cast<KeyType*>(curr_ptr);
        curr_ptr += num_keys * sizeof(KeyType);
        m_pUniqueKeys = reinterpret_cast<KeyType*>(curr_ptr);
        curr_ptr += num_keys * sizeof(KeyType);
        KeyType* tmp_storage = reinterpret_cast<KeyType*>(curr_ptr);
        curr_ptr += m_tmpAllocSize;
        m_pPriority = reinterpret_cast<float*>(curr_ptr);
        curr_ptr += num_keys * sizeof(float);
        int* counters = reinterpret_cast<int*>(curr_ptr);
        curr_ptr += num_keys * sizeof(int);
        int* num_runs_out = reinterpret_cast<int*>(curr_ptr);

        // reused buffers
        KeyType* sorted_keys = reinterpret_cast<KeyType*>(m_pDataPtrs);
        int* offsets = reinterpret_cast<int*>(m_pUniqueKeys);

        // Create index
        std::vector<KeyType> index(num_keys);
        for (KeyType i = 0; i < num_keys; ++i)  index[i] = i;

        NVE_CHECK_(cudaMemcpyAsync(idx_mapping, &index[0], num_keys * sizeof(KeyType), cudaMemcpyDefault, histStream));

        // Compute priorities
        NVE_CHECK_(cub::DeviceRadixSort::SortPairs(
                tmp_storage, m_tmpAllocSize, keys,
                sorted_keys, idx_mapping, idx_mapping_sorted, num_keys, 0, sizeof(KeyType)*8, histStream),
            "Failed to call cub::DeviceRadixSort::SortKeys");            

        NVE_CHECK_(cub::DeviceRunLengthEncode::Encode(
                tmp_storage, m_tmpAllocSize,
                sorted_keys, m_pUniqueKeys,
                counters, num_runs_out,
                static_cast<int>(num_keys), histStream),
            "Failed to call cub::DeviceRunLengthEncode::Encode"); 

        // Copy num_runs_out
        NVE_CHECK_(cudaMemcpyAsync(&m_numUniqueKeys, num_runs_out, sizeof(int), cudaMemcpyDeviceToHost, histStream));
        NVE_CHECK_(cudaStreamSynchronize(histStream));

        // Compute offsets and get mapping of unique keys
        NVE_CHECK_(cub::DeviceScan::ExclusiveSum(
                reinterpret_cast<int*>(tmp_storage), m_tmpAllocSize,
                counters, offsets, static_cast<int>(m_numUniqueKeys), histStream), "Failed to call DeviceScan::ExclusiveSum");

        CallGatherLocations<KeyType>(m_numUniqueKeys, offsets, idx_mapping_sorted, idx_mapping, histStream);

        NVE_CHECK_(cub::DeviceRadixSort::SortPairsDescending(
                tmp_storage, m_tmpAllocSize,
                counters, reinterpret_cast<int*>(m_pPriority),
                idx_mapping,
                idx_mapping_sorted,
                m_numUniqueKeys, 0, sizeof(int)*8, histStream),
            "Failed to call cub::DeviceRadixSort::SortPairs");  

        // Gather unique keys and data ptrs
        CallGatherKeysAndDataPtrs<KeyType>(
            reinterpret_cast<const int8_t*>(data), idx_mapping_sorted, 
            reinterpret_cast<int*>(m_pPriority), keys,
            1.0f / float(num_keys), m_numUniqueKeys, strideInBytes,
            m_pUniqueKeys, m_pDataPtrs, histStream);
    }

    size_t getAllocSize() const {
        return m_tmpAllocSize + m_numKeys * sizeof(int8_t*) + 3 * m_numKeys * sizeof(KeyType)
                              + m_numKeys * sizeof(float) + m_numKeys * sizeof(int) + sizeof(int);
    }

    int64_t getNumBins() const
    {
        return m_numUniqueKeys;
    }

    float* getPriority() 
    {
        return m_pPriority;
    }

    KeyType* getKeys() 
    {
        return m_pUniqueKeys;
    }

    const int8_t* const* getData()
    {
        return m_pDataPtrs;
    }

private:

    void computeAllocSize() {
        KeyType* p = nullptr;
        int* pi = nullptr;
        size_t alloc_size;

        // determing and allocate temporary storage for cub kernels
        NVE_CHECK_(cub::DeviceRadixSort::SortPairs(p, m_tmpAllocSize,
                p, p, p, p, m_numKeys, 0, sizeof(KeyType)*8), "Failed to call cub::DeviceRadixSort::SortPairs");            

        NVE_CHECK_(cub::DeviceRunLengthEncode::Encode(
                p, alloc_size,
                p, p, p, p, static_cast<int>(m_numKeys)), "Failed to call cub::DeviceRunLengthEncode::Encode"); 
        if (alloc_size > m_tmpAllocSize) {
            m_tmpAllocSize = alloc_size;
        }

        NVE_CHECK_(cub::DeviceScan::ExclusiveSum(
                pi, alloc_size,
                pi, pi, static_cast<int>(m_numKeys)), "Failed to call DeviceScan::ExclusiveSum");
        if (alloc_size > m_tmpAllocSize) {
            m_tmpAllocSize = alloc_size;
        }

        // make it cache line aligned
        m_tmpAllocSize = ((m_tmpAllocSize + 127) >> 7) << 7;
    }

    size_t    m_tmpAllocSize;
    int8_t**  m_pDataPtrs;
    KeyType*  m_pUniqueKeys;
    float*    m_pPriority;
    int64_t   m_numKeys;
    int       m_numUniqueKeys;
};
}  // namespace nve

