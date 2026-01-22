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
#include <cub/cub.cuh>

namespace nve {

template<typename KeyType, typename CounterType, uint32_t NUM_WAYS>
__global__ void ComputeSetKernel(uint32_t num_keys, uint32_t num_sets,
                                const KeyType* __restrict__ keys, uint32_t* __restrict__ sets, CounterType* __restrict__ counters,
                                float decay_rate) {
    const uint32_t offset = blockIdx.x * warpSize * blockDim.y + threadIdx.y * warpSize;
    if ((offset + threadIdx.x) < num_keys) {
        sets[offset + threadIdx.x] = keys[offset + threadIdx.x] % num_sets;
    }
    const uint32_t sets_per_warp = warpSize / NUM_WAYS;
    const uint32_t set = blockIdx.x * sets_per_warp * blockDim.y + threadIdx.y * sets_per_warp + (threadIdx.x / NUM_WAYS);
    const uint32_t way = threadIdx.x % NUM_WAYS;
    if (set < num_sets) {
        CounterType* set_counters_ptr = counters + set * NUM_WAYS;

        // update set counters
        set_counters_ptr[way] *= decay_rate;
    }
}

template<typename CounterType>
__device__ __forceinline__ void SortBlock(CounterType& count, uint32_t& pos, const uint32_t num_ways) {
    uint32_t my_idx = threadIdx.x;
    uint32_t my_slice = threadIdx.y;
    uint32_t mask = __activemask(); //(1<<NUM_WAYS) - 1;

    for (uint32_t k = 2; k <= num_ways; k *= 2) { // k is doubled every iteration
        for (uint32_t j = k/2; j > 0; j /= 2) { // j is halved at every iteration, with truncation of fractional parts
            uint32_t pair_idx = (threadIdx.x ^ j);
            uint32_t pair_pos = __shfl_sync(mask, pos, my_slice * num_ways + pair_idx);
            CounterType pair_count = __shfl_sync(mask, count, my_slice * num_ways + pair_idx);

            if (count != pair_count) {
                bool in_order_pos = my_idx > pair_idx;
                bool in_order_count = count > pair_count;

                if ((my_idx & k) != 0) {
                    // ascending order here
                    if (in_order_pos != in_order_count) {
                        // swap
                        pos = pair_pos;
                        count = pair_count;
                    }                
                } else {
                    // descending order here
                    if (in_order_pos == in_order_count) {
                        // swap
                        pos = pair_pos;
                        count = pair_count;
                    }
                }
            }
        }
    }
}

template<typename PriorityType>
__device__ __forceinline__ void MergeBlock(
        PriorityType& p1, uint32_t& pos1, PriorityType& p2, uint32_t& pos2, uint32_t NUM_WAYS) {
    uint32_t my_idx = threadIdx.x;
    uint32_t my_slice = threadIdx.y;
    uint32_t mask = __activemask(); //(1<<NUM_WAYS) - 1;

    float reverse_p = __shfl_sync(mask, p2, my_slice * NUM_WAYS + (NUM_WAYS - 1 - my_idx));
    float reverse_pos = __shfl_sync(mask, pos2, my_slice * NUM_WAYS + (NUM_WAYS - 1 - my_idx));

    if (reverse_p > p1) {
        p1 = reverse_p;
        pos1 = reverse_pos;
    }

    for (uint32_t j = NUM_WAYS/2; j > 0; j /= 2) { // j is halved at every iteration, with truncation of fractional parts
        uint32_t pair_idx = (threadIdx.x ^ j);
        uint32_t pair_pos = __shfl_sync(mask, pos1, my_slice * NUM_WAYS + pair_idx);
        float pair_p = __shfl_sync(mask, p1, my_slice * NUM_WAYS + pair_idx);

        bool in_order_pos = my_idx < pair_idx;
        bool in_order_p = p1 > pair_p;

        // descending order here
        if (in_order_pos != in_order_p) {
            //printf("in merge thread %d with count %f swaps with thread %d with count %f\n", my_idx, p1, pair_idx, pair_p);
            // swap
            pos1 = pair_pos;
            p1 = pair_p;
        }
    }
}

template<typename KeyType, typename CounterType, uint32_t NUM_WAYS>
__global__ void SortKernel(uint32_t num_sets, const CounterType* __restrict__ counters, KeyType* __restrict__ res) {
    const uint32_t set = blockIdx.x * blockDim.y + threadIdx.y;
    
    if (set < num_sets) {
        const CounterType* set_counters_ptr = counters + set * NUM_WAYS;
        KeyType* result_ptr = res + set * NUM_WAYS;

        // sort the set by counters
        if (threadIdx.x < NUM_WAYS) {
            uint32_t pos = threadIdx.x;
            CounterType count = set_counters_ptr[pos];

            SortBlock(count, pos, NUM_WAYS);
            result_ptr[threadIdx.x] = pos;
        }
    }
}

template<typename KeyType, typename CounterType, uint32_t NUM_WAYS>
cudaError_t CallSortKernel(const CounterType* counters, KeyType* res, uint32_t num_sets, cudaStream_t stream = 0) {
    assert (NUM_WAYS <= 32);
    assert ((NUM_WAYS & (NUM_WAYS - 1)) == 0);

    auto sets_in_warp = 32 / NUM_WAYS;
    dim3 gridSize ((num_sets + sets_in_warp - 1) / sets_in_warp, 1);
    dim3 blockSize (NUM_WAYS, sets_in_warp);

    SortKernel<KeyType, CounterType, NUM_WAYS><<<gridSize, blockSize, 0, stream>>>(num_sets, counters, res);

    return cudaGetLastError();
}

template<typename KeyType, typename TagType, typename CounterType, uint32_t NUM_WAYS>
__global__ void SetReplaceDataKernel(
    const int8_t* const* data_ptrs,
    const KeyType* __restrict__ keys,
    const uint32_t* __restrict__ represented_sets,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ inverse_map,
    uint32_t num_keys,
    const float* __restrict__ priority,
    const TagType* __restrict__ tags,
    CounterType* __restrict__ counters,
    int8_t* cache_ptr,
    uint32_t embed_width_in_bytes,
    uint32_t num_sets,
    uint32_t num_represented_sets,
    uint32_t max_update_size,
    void* __restrict__ replace_entries) {

    using ModifyEntry = typename EmbedCacheSA<KeyType, TagType>::ModifyEntry;
    using ModifyList = typename EmbedCacheSA<KeyType, TagType>::ModifyList;
    ModifyList* result = reinterpret_cast<ModifyList*>(replace_entries);
    ModifyEntry* entries = reinterpret_cast<ModifyEntry*>(result->pEntries);
    uint32_t* num_replace_entries_to_update = &result->nEntries;

    const uint32_t setID = blockIdx.x * blockDim.y + threadIdx.y;

    if (setID < num_represented_sets) {
        const uint32_t set = represented_sets[setID];
        uint32_t count = offsets[setID + 1] - offsets[setID];

        const TagType* set_ways_ptr = tags + set * NUM_WAYS;
        CounterType* set_counters_ptr = counters + set * NUM_WAYS;

        // find NUM_WAYS best candidates
        // process NUM_WAYS candidates each time
        // sort 
        // merge to the previous result
        uint32_t num_replacements = std::min(NUM_WAYS, count);

        uint32_t processed = 0;
        uint32_t curr_offset = threadIdx.x;
        uint32_t curr_loc = offsets[setID] + curr_offset;
        uint32_t src_pos = 0;
        float highest_priority = 0;

        bool hit = false;
        if (threadIdx.x < count) {
            uint32_t location = inverse_map[offsets[setID] + threadIdx.x];
            KeyType key = keys[location];

            // find this key and update counter
            for (int w = 0; w < NUM_WAYS; w++) {
                TagType tag = set_ways_ptr[w];
                KeyType key_in_cache = static_cast<KeyType>(tag * num_sets + set);
                if (key_in_cache == key) {
                    set_counters_ptr[w] += priority[location];
                    hit = true;
                }
            }

            // hits are filtered out by all threads, only new keys are checked vs. cache content
            if (hit == false) {
                src_pos = location;
                highest_priority = priority[src_pos];
            }
        }
        __syncwarp();

        // sort first chunk of values
        SortBlock(highest_priority, src_pos, NUM_WAYS);

        processed += NUM_WAYS;
        while (processed < count) {
            curr_loc += NUM_WAYS;

            uint32_t src_pos2 = 0;
            float highest_priority2 = 0;

            if ((processed + threadIdx.x) < count) {
                uint32_t location = inverse_map[offsets[setID] + processed + threadIdx.x];
                KeyType key = keys[location];

                hit = false;
                // find this key and update counter
                for (int w = 0; w < NUM_WAYS; w++) {
                    TagType tag = set_ways_ptr[w];
                    KeyType key_in_cache = static_cast<KeyType>(tag) * num_sets + set;
                    if (key_in_cache == key) {
                        set_counters_ptr[w] += priority[location];
                        hit = true;
                    }
                }
                if (hit == false) {
                    src_pos2 = location;
                    highest_priority2 = priority[src_pos2];
                }
            }
            __syncwarp();

            // sort next chunk of values
            SortBlock(highest_priority2, src_pos2, NUM_WAYS);

            // merge to best so far
            MergeBlock(highest_priority, src_pos, highest_priority2, src_pos2, NUM_WAYS);

            processed += NUM_WAYS;
        }

        // sort the set by counters
        uint32_t dst_pos = threadIdx.x;
        CounterType counter = set_counters_ptr[dst_pos];
        TagType tag = set_ways_ptr[dst_pos];
        if (tag == TagType(INVALID_IDX)) {
            counter = 0;
        }
        SortBlock(counter, dst_pos, NUM_WAYS);

        // reverse best candidate index
        uint32_t mask = __activemask(); //(1<<NUM_WAYS) - 1;
        dst_pos = __shfl_sync(mask, dst_pos, threadIdx.y * NUM_WAYS + NUM_WAYS - threadIdx.x - 1);
        counter = __shfl_sync(mask, counter, threadIdx.y * NUM_WAYS + NUM_WAYS - threadIdx.x - 1);

        KeyType index = keys[src_pos];
        tag = index / num_sets;
        uint32_t final_dst_pos = dst_pos;

        bool need_to_write = (threadIdx.x < num_replacements) && (highest_priority > counter);

        if (need_to_write) {
            // create replacement record
            uint32_t curr_num_replace_entries = atomicAdd(num_replace_entries_to_update, 1);

            // this condition is redundant because we limit the number of keys to be at most max_update_size
            // not sure how much perf it costs, to be checked
            if (curr_num_replace_entries < max_update_size) {
                // create a record
                KeyType index = keys[src_pos];

                uint64_t dst_offset = set * NUM_WAYS + final_dst_pos;
                dst_offset *= static_cast<uint64_t>(embed_width_in_bytes);
                int8_t* dst_ptr = cache_ptr + dst_offset;
                entries[curr_num_replace_entries].pSrc = data_ptrs[src_pos];
                entries[curr_num_replace_entries].pDst = dst_ptr;
                entries[curr_num_replace_entries].set = set;
                entries[curr_num_replace_entries].way = final_dst_pos;
                entries[curr_num_replace_entries].tag = TagType(index / num_sets);

                // set counter for the new row
                // it should happen in update kernel and not here, but since we don't support
                // multiple modifies we can safely change here and save perf and avoid launching 
                // another kernel to set priorities
                set_counters_ptr[final_dst_pos] = highest_priority;
            }
        }
    }
}

static cudaError_t GetSetReplaceDataAllocRequirements(uint64_t num_keys, uint64_t num_sets, size_t& size) {
    uint32_t* p = nullptr;
    size_t allocSize;
    size_t finalAllocSize = 0;
    cudaError_t err = cudaSuccess;

    // determing and allocate temporary storage for sort
    err = cub::DeviceRadixSort::SortPairs(p, allocSize,
               p, p, p, p, num_keys, 0, sizeof(uint32_t)*8);
    if (err != cudaSuccess) {
        return err;
    }              
    finalAllocSize = allocSize;

    // determing and allocate temporary storage for encode
    err = cub::DeviceRunLengthEncode::Encode(
               p, allocSize,
               p, p, p, p, static_cast<int>(num_keys));
    if (err != cudaSuccess) {
        return err;
    }
    if (allocSize > finalAllocSize) {
        finalAllocSize = allocSize;
    }

    // determing and allocate temporary storage for ex sum
    // ExclusiveSum will be called on one extra input (0)
    // in order to get last offset equal to number of input keys
    int num_sets_ = static_cast<int>(num_sets) + 1;

    err = cub::DeviceScan::ExclusiveSum(
               p, allocSize,
               p, p, num_sets_);
    if (err != cudaSuccess) {
        return err;
    }
    if (allocSize > finalAllocSize) {
        finalAllocSize = allocSize;
    }

    // align tmp storage
    finalAllocSize = ((finalAllocSize + 511) >> 9) << 9;

    finalAllocSize += 4 * num_keys * sizeof(uint32_t); // sets, sorted sets, locs and inverse map
    finalAllocSize += (num_sets + 1) * sizeof(uint32_t); // offsets
    finalAllocSize += sizeof(uint32_t); // numSetsDevice

    size = finalAllocSize;

    return err;
}

template<typename KeyType, typename TagType, typename CounterType, uint32_t NUM_WAYS, bool inputsOnDevice>
cudaError_t ComputeSetReplaceData(
    const int8_t* const* data_ptrs,
    const KeyType* keys,
    int8_t* tmpStorage,
    size_t& tmpStorageSize,
    uint64_t num_keys,
    const float* priority,
    const TagType* tags,
    CounterType* counters,
    int8_t* cache_ptr,
    uint64_t embed_width_in_bytes,
    float decay_rate,
    uint64_t num_sets,
    uint32_t max_update_size,
    void* replace_entries,
    cudaStream_t stream) {
        assert (NUM_WAYS <= 32);
        assert ((NUM_WAYS & (NUM_WAYS - 1)) == 0);

        cudaError_t err = cudaSuccess;

        if (tmpStorage == nullptr) {
            err = GetSetReplaceDataAllocRequirements(num_keys, num_sets, tmpStorageSize);
            if (err != cudaSuccess) {
                return err;
            }
            if (inputsOnDevice == false) {
                tmpStorageSize += num_keys * sizeof(KeyType);
                tmpStorageSize += num_keys * sizeof(float);
                tmpStorageSize += num_keys * sizeof(uint64_t);
            }
            return cudaGetLastError();
        }

        int8_t* curr_ptr = tmpStorage;
        size_t storageSize = tmpStorageSize;

        if (inputsOnDevice == false) {
            err = cudaMemcpyAsync(curr_ptr, data_ptrs, num_keys * sizeof(uint64_t), cudaMemcpyDefault, stream);
            if (err != cudaSuccess) {
                return err;
            }
            data_ptrs = reinterpret_cast<const int8_t* const*>(curr_ptr);
            curr_ptr += num_keys * sizeof(uint64_t);

            err = cudaMemcpyAsync(curr_ptr, keys, num_keys * sizeof(KeyType), cudaMemcpyDefault, stream);
            if (err != cudaSuccess) {
                return err;
            }
            keys = reinterpret_cast<KeyType*>(curr_ptr);
            curr_ptr += num_keys * sizeof(KeyType);
            
            err = cudaMemcpyAsync(curr_ptr, priority, num_keys * sizeof(float), cudaMemcpyDefault, stream);
            if (err != cudaSuccess) {
                return err;
            }            
            priority = reinterpret_cast<float*>(curr_ptr);
            curr_ptr += num_keys * sizeof(float);

            storageSize -= num_keys * (sizeof(KeyType) + sizeof(float) + sizeof(uint64_t));
        }

        uint32_t* sets = reinterpret_cast<uint32_t*>(curr_ptr);
        uint32_t* counts_out = sets; // reuse
        curr_ptr += num_keys * sizeof(uint32_t);
        uint32_t* sets_sorted = reinterpret_cast<uint32_t*>(curr_ptr);
        curr_ptr += num_keys * sizeof(uint32_t);
        uint32_t* locations_device = reinterpret_cast<uint32_t*>(curr_ptr);
        uint32_t* unique_sets = locations_device; // reuse
        curr_ptr += num_keys * sizeof(uint32_t);
        uint32_t* inverse_map = reinterpret_cast<uint32_t*>(curr_ptr);
        curr_ptr += num_keys * sizeof(uint32_t);
        uint32_t* offsets = reinterpret_cast<uint32_t*>(curr_ptr);
        curr_ptr += (num_sets + 1) * sizeof(uint32_t);
        uint32_t* numSetsDevice = reinterpret_cast<uint32_t*>(curr_ptr);
        curr_ptr += sizeof(uint32_t);

        storageSize -= (4 * num_keys + num_sets + 1 + 1) * sizeof(uint32_t);

        const uint32_t num_warps = 1;
        auto sets_in_block = num_warps * (32 / NUM_WAYS);
        auto keys_in_block = num_warps * 32;

        const uint32_t grid_size_keys = (static_cast<uint32_t>(num_keys) + keys_in_block - 1) / keys_in_block;
        const uint32_t grid_size_sets = (static_cast<uint32_t>(num_sets) + sets_in_block - 1) / sets_in_block;

        dim3 gridSizeComputeSets (std::max(grid_size_keys, grid_size_sets), 1);
        dim3 blockSizeComputeSets (32, num_warps);

        ComputeSetKernel<KeyType, CounterType, NUM_WAYS><<<gridSizeComputeSets, blockSizeComputeSets, 0, stream>>>(
            static_cast<uint32_t>(num_keys), static_cast<uint32_t>(num_sets), 
            keys, sets, counters, decay_rate);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }
        std::vector<uint32_t> locations(num_keys);
        for (uint32_t i = 0; i < num_keys; i++) {
            locations[i] = i;
        }
        // Run sorting operation
        err = cudaMemcpyAsync(locations_device, locations.data(), sizeof(uint32_t)*num_keys, cudaMemcpyDefault, stream);
        if (err != cudaSuccess) {
            return err;
        }
        err = cub::DeviceRadixSort::SortPairs(curr_ptr, storageSize,
            sets, sets_sorted, locations_device, inverse_map, num_keys, 0, sizeof(uint32_t)*8, stream);
        if (err != cudaSuccess) {
            return err;
        }            
        // Run encoding
        err = cub::DeviceRunLengthEncode::Encode(
            curr_ptr, storageSize,
            sets_sorted, unique_sets, counts_out, numSetsDevice, static_cast<int>(num_keys), stream);
        if (err != cudaSuccess) {
            return err;
        }        
        uint32_t num_represented_sets;
        err = cudaMemcpyAsync(&num_represented_sets, numSetsDevice, sizeof(uint32_t), cudaMemcpyDefault, stream);
        if (err != cudaSuccess) {
            return err;
        }
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            return err;
        }
        // Run exclusive prefix sum
        err = cub::DeviceScan::ExclusiveSum(
            curr_ptr, storageSize,
            counts_out, offsets, num_represented_sets + 1, stream);
        if (err != cudaSuccess) {
            return err;
        }
        auto sets_in_warp = 32 / NUM_WAYS;
        dim3 gridSize ((num_represented_sets + sets_in_warp - 1) / sets_in_warp, 1);
        dim3 blockSize (NUM_WAYS, sets_in_warp);

        SetReplaceDataKernel<KeyType, TagType, CounterType, NUM_WAYS><<<gridSize, blockSize, 0, stream>>>(
            data_ptrs, keys, unique_sets, offsets, inverse_map,
            static_cast<uint32_t>(num_keys), priority, tags, counters,
            cache_ptr, static_cast<uint32_t>(embed_width_in_bytes),
            static_cast<uint32_t>(num_sets), num_represented_sets,
            max_update_size, replace_entries);

        return cudaGetLastError();
    }

template<typename KeyType, typename TagType, uint32_t NUM_WAYS, bool invalidate_only = false>
__global__ void SetUpdateDataKernel(
    int8_t* cache_ptr,
    const int8_t* values,
    const KeyType* __restrict__ keys,
    uint32_t num_keys,
    uint32_t num_sets,
    uint32_t value_stride,
    uint32_t embed_width_in_bytes,
    const TagType* __restrict__ tags,
    uint32_t max_update_size,
    void* __restrict__ replace_entries) {
        
    using ModifyEntry = typename EmbedCacheSA<KeyType, TagType>::ModifyEntry;
    using ModifyList = typename EmbedCacheSA<KeyType, TagType>::ModifyList;
    ModifyList* result = reinterpret_cast<ModifyList*>(replace_entries);
    ModifyEntry* entries = reinterpret_cast<ModifyEntry*>(result->pEntries);
    uint32_t* num_replace_entries_to_update = &result->nEntries;
        
    const uint32_t keyID = blockIdx.x * blockDim.x + threadIdx.x;
    if (keyID < num_keys) {
        KeyType key = keys[keyID];
        uint32_t set = key % num_sets;

        const TagType* set_ways_ptr = tags + set * NUM_WAYS;
    
        for (int w = 0; w < NUM_WAYS; w++) {
            TagType tag = set_ways_ptr[w];
            KeyType key_in_cache = static_cast<KeyType>(tag) * num_sets + set;
            if (key_in_cache == key) {
                uint32_t curr_num_replace_entries = atomicAdd(num_replace_entries_to_update, 1);
                if (curr_num_replace_entries < max_update_size) {
                    // create a record
                    if (invalidate_only) {
                        entries[curr_num_replace_entries].pSrc = nullptr;
                        entries[curr_num_replace_entries].pDst = nullptr;
                        entries[curr_num_replace_entries].tag = static_cast<TagType>(-1);
                    } else {
                        int8_t* dst_ptr = cache_ptr + ( set * NUM_WAYS + w ) * embed_width_in_bytes;
                        entries[curr_num_replace_entries].pSrc = values + keyID * value_stride;
                        entries[curr_num_replace_entries].pDst = dst_ptr;
                        entries[curr_num_replace_entries].tag = static_cast<TagType>(key / num_sets);
                    }
                    entries[curr_num_replace_entries].set = set;
                    entries[curr_num_replace_entries].way = w;
                    
                }
            }
        }
    }
}

template<typename KeyType, typename TagType, uint32_t NUM_WAYS>
cudaError_t ComputeSetInvalidateData(
    const KeyType* __restrict__ keys,
    uint64_t num_keys,
    uint64_t num_sets,
    const TagType* __restrict__ tags,
    uint32_t max_update_size,
    void* __restrict__ replace_entries,
    cudaStream_t stream = 0) {

    assert (NUM_WAYS <= 32);
    assert ((NUM_WAYS & (NUM_WAYS - 1)) == 0);

    dim3 gridSize (static_cast<uint32_t>(num_keys + 32 - 1) / 32, 1);
    dim3 blockSize (32, 1);

    SetUpdateDataKernel<KeyType, TagType, NUM_WAYS, true><<<gridSize, blockSize, 0, stream>>>(
        nullptr, nullptr, keys, static_cast<uint32_t>(num_keys),static_cast<uint32_t>( num_sets), 0, 0,
        tags, max_update_size, replace_entries);

    return cudaGetLastError();
}

template<typename KeyType, typename TagType, uint32_t NUM_WAYS>
cudaError_t ComputeSetUpdateData(
    int8_t* cache_ptr,
    const int8_t* values,
    const KeyType* __restrict__ keys,
    uint64_t num_keys,
    uint64_t num_sets,
    uint64_t value_stride,
    uint64_t embed_width_in_bytes,
    const TagType* __restrict__ tags,
    uint32_t max_update_size,
    void* __restrict__ replace_entries,
    cudaStream_t stream = 0) {

    assert (NUM_WAYS <= 32);
    assert ((NUM_WAYS & (NUM_WAYS - 1)) == 0);

    dim3 gridSize (static_cast<uint32_t>(num_keys + 32 - 1) / 32, 1);
    dim3 blockSize (32, 1);

    SetUpdateDataKernel<KeyType, TagType, NUM_WAYS><<<gridSize, blockSize, 0, stream>>>(
        cache_ptr, values, keys, static_cast<uint32_t>(num_keys),
        static_cast<uint32_t>(num_sets),
        static_cast<uint32_t>(value_stride), 
        static_cast<uint32_t>(embed_width_in_bytes),
        tags, max_update_size, replace_entries);
    
    return cudaGetLastError();
}
}