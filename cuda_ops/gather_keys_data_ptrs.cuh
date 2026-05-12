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
        : num_keys_(num_keys) {
        compute_alloc_size();
    }

    void compute_histogram(const KeyType* keys, int64_t num_keys, const int8_t* data, size_t strideInBytes, void* tmpStorage, cudaStream_t histStream = 0) {
        NVE_CHECK_(num_keys <= num_keys_, "histogram object was allocated for fewer keys than provided!");
        
        // Allocate internal and temporary buffers
        int8_t* curr_ptr = reinterpret_cast<int8_t*>(tmpStorage);
        data_ptrs_ = reinterpret_cast<int8_t**>(curr_ptr);
        curr_ptr += num_keys * sizeof(int8_t*);

        KeyType* idx_mapping = reinterpret_cast<KeyType*>(curr_ptr);
        curr_ptr += num_keys * sizeof(KeyType);
        KeyType* idx_mapping_sorted = reinterpret_cast<KeyType*>(curr_ptr);
        curr_ptr += num_keys * sizeof(KeyType);
        unique_keys_ = reinterpret_cast<KeyType*>(curr_ptr);
        curr_ptr += num_keys * sizeof(KeyType);
        KeyType* tmp_storage = reinterpret_cast<KeyType*>(curr_ptr);
        curr_ptr += tmp_alloc_size_;
        priority_ = reinterpret_cast<float*>(curr_ptr);
        curr_ptr += num_keys * sizeof(float);
        int* counters = reinterpret_cast<int*>(curr_ptr);
        curr_ptr += num_keys * sizeof(int);
        int* num_runs_out = reinterpret_cast<int*>(curr_ptr);

        // reused buffers
        KeyType* sorted_keys = reinterpret_cast<KeyType*>(data_ptrs_);
        int* offsets = reinterpret_cast<int*>(unique_keys_);

        // Create index
        std::vector<KeyType> index(num_keys);
        for (KeyType i = 0; i < num_keys; ++i)  index[i] = i;

        NVE_CHECK_(cudaMemcpyAsync(idx_mapping, &index[0], num_keys * sizeof(KeyType), cudaMemcpyDefault, histStream));

        // Compute priorities
        NVE_CHECK_(cub::DeviceRadixSort::SortPairs(
                tmp_storage, tmp_alloc_size_, keys,
                sorted_keys, idx_mapping, idx_mapping_sorted, num_keys, 0, sizeof(KeyType)*8, histStream),
            "Failed to call cub::DeviceRadixSort::SortKeys");            

        NVE_CHECK_(cub::DeviceRunLengthEncode::Encode(
                tmp_storage, tmp_alloc_size_,
                sorted_keys, unique_keys_,
                counters, num_runs_out,
                static_cast<int>(num_keys), histStream),
            "Failed to call cub::DeviceRunLengthEncode::Encode"); 

        // Copy num_runs_out
        NVE_CHECK_(cudaMemcpyAsync(&num_unique_keys_, num_runs_out, sizeof(int), cudaMemcpyDeviceToHost, histStream));
        NVE_CHECK_(cudaStreamSynchronize(histStream));

        // Compute offsets and get mapping of unique keys
        NVE_CHECK_(cub::DeviceScan::ExclusiveSum(
                reinterpret_cast<int*>(tmp_storage), tmp_alloc_size_,
                counters, offsets, static_cast<int>(num_unique_keys_), histStream), "Failed to call DeviceScan::ExclusiveSum");

        CallGatherLocations<KeyType>(num_unique_keys_, offsets, idx_mapping_sorted, idx_mapping, histStream);

        NVE_CHECK_(cub::DeviceRadixSort::SortPairsDescending(
                tmp_storage, tmp_alloc_size_,
                counters, reinterpret_cast<int*>(priority_),
                idx_mapping,
                idx_mapping_sorted,
                num_unique_keys_, 0, sizeof(int)*8, histStream),
            "Failed to call cub::DeviceRadixSort::SortPairs");  

        // Gather unique keys and data ptrs
        CallGatherKeysAndDataPtrs<KeyType>(
            reinterpret_cast<const int8_t*>(data), idx_mapping_sorted, 
            reinterpret_cast<int*>(priority_), keys,
            1.0f / float(num_keys), num_unique_keys_, strideInBytes,
            unique_keys_, data_ptrs_, histStream);
    }

    size_t get_alloc_size() const {
        return tmp_alloc_size_ + num_keys_ * sizeof(int8_t*) + 3 * num_keys_ * sizeof(KeyType)
                              + num_keys_ * sizeof(float) + num_keys_ * sizeof(int) + sizeof(int);
    }

    int64_t get_num_bins() const
    {
        return num_unique_keys_;
    }

    float* get_priority() 
    {
        return priority_;
    }

    KeyType* get_keys() 
    {
        return unique_keys_;
    }

    const int8_t* const* get_data()
    {
        return data_ptrs_;
    }

private:

    void compute_alloc_size() {
        KeyType* p = nullptr;
        int* pi = nullptr;
        size_t alloc_size;

        // determing and allocate temporary storage for cub kernels
        NVE_CHECK_(cub::DeviceRadixSort::SortPairs(p, tmp_alloc_size_,
                p, p, p, p, num_keys_, 0, sizeof(KeyType)*8), "Failed to call cub::DeviceRadixSort::SortPairs");            

        NVE_CHECK_(cub::DeviceRunLengthEncode::Encode(
                p, alloc_size,
                p, p, p, p, static_cast<int>(num_keys_)), "Failed to call cub::DeviceRunLengthEncode::Encode"); 
        if (alloc_size > tmp_alloc_size_) {
            tmp_alloc_size_ = alloc_size;
        }

        NVE_CHECK_(cub::DeviceScan::ExclusiveSum(
                pi, alloc_size,
                pi, pi, static_cast<int>(num_keys_)), "Failed to call DeviceScan::ExclusiveSum");
        if (alloc_size > tmp_alloc_size_) {
            tmp_alloc_size_ = alloc_size;
        }

        // make it cache line aligned
        tmp_alloc_size_ = ((tmp_alloc_size_ + 127) >> 7) << 7;
    }

    size_t    tmp_alloc_size_ = 0;
    int8_t**  data_ptrs_ = nullptr;
    KeyType*  unique_keys_ = nullptr;
    float*    priority_ = nullptr;
    int64_t   num_keys_ = 0;
    int       num_unique_keys_ = 0;
};
}  // namespace nve

