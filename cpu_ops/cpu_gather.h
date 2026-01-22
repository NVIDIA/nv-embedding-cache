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

#include "include/common.hpp"
#include "include/thread_pool.hpp"

namespace nve {

template<typename IndexT>
int cpu_kernel_gather(thread_pool_ptr_t thread_pool,
              uint64_t n,
              const IndexT* keys, 
              max_bitmask_repr_t* hit_mask,
              size_t value_stride, 
              void* values, 
              int8_t* uvm_table_ptr,
              size_t row_size_in_bytes,
              uint64_t num_threads)
{
    constexpr uint64_t num_bits_in_hit_mask = sizeof(max_bitmask_repr_t) * 8;
    auto keys_per_task = (n + num_threads - 1)/ num_threads;
    keys_per_task = ((keys_per_task + num_bits_in_hit_mask - 1) / num_bits_in_hit_mask)*num_bits_in_hit_mask; // align to 64

    const auto gather_task = [=] (const size_t idx) {
        const auto base_key = (idx) * keys_per_task;
        const auto base_mask_idx = base_key / num_bits_in_hit_mask;
        
        for (uint64_t i = 0; i < keys_per_task; i++) {
            if (base_key + i >= n) {
                break;
            }
            auto mask_idx = i / num_bits_in_hit_mask;
            auto bit_idx = i % num_bits_in_hit_mask;
            if ((hit_mask[base_mask_idx + mask_idx] & (1ULL << bit_idx)) == 0) {
                IndexT key = reinterpret_cast<const IndexT*>(keys)[base_key + i];
                int8_t* src_ptr = uvm_table_ptr + (static_cast<size_t>(key) * row_size_in_bytes);
                int8_t* dst_ptr = reinterpret_cast<int8_t*>(values) + (base_key + i) * value_stride;
                memcpy(dst_ptr, src_ptr, row_size_in_bytes);
            }
        }
    };

    thread_pool->execute_n(0, static_cast<int64_t>(num_threads), gather_task);
    // we resolve all misses so set everything to 1
    // first set all up to the last qword to 1
    memset(hit_mask, 0xff, static_cast<size_t>(n / num_bits_in_hit_mask) * sizeof(max_bitmask_repr_t));
    // then set the remaining bits to 1
    uint64_t rem = n - ((n / num_bits_in_hit_mask) * num_bits_in_hit_mask); 
    hit_mask[n / num_bits_in_hit_mask] = ((1ULL << (rem)) - 1);
    
    return 0;
}

template<typename IndexT>
int cpu_kernel_gather_dispatch(thread_pool_ptr_t thread_pool,
              uint64_t n,
              const IndexT* keys, 
              max_bitmask_repr_t* hit_mask,
              size_t value_stride, 
              void* values, 
              int8_t* uvm_table_ptr,
              size_t row_size_in_bytes,
              uint64_t num_threads)
{
    return cpu_kernel_gather<IndexT>(thread_pool, n, keys, hit_mask, value_stride, values, uvm_table_ptr, row_size_in_bytes, num_threads);
}

} // namespace nve

