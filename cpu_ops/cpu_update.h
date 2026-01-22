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
#include "include/nve_types.hpp"

namespace nve {
    
template<typename IndexT>
int cpu_kernel_update(thread_pool_ptr_t thread_pool,
              uint64_t n,
              const IndexT* keys, 
              size_t value_stride, 
              const void* values, 
              int8_t* uvm_table_ptr,
              size_t row_size_in_bytes,
              uint64_t num_threads)
{
    auto keys_per_task = (n + num_threads - 1)/ num_threads;

    const auto update_task = [=] (const size_t idx) {
        const auto base_key = (idx) * keys_per_task;
        
        for (uint64_t i = 0; i < keys_per_task; i++) {
            if (base_key + i >= n) {
                break;
            }
            IndexT key = reinterpret_cast<const IndexT*>(keys)[base_key + i];
            int8_t* dst_ptr = uvm_table_ptr + (static_cast<size_t>(key) * row_size_in_bytes);
            const int8_t* src_ptr = reinterpret_cast<const int8_t*>(values) + (base_key + i) * value_stride;
            memcpy(dst_ptr, src_ptr, row_size_in_bytes);
        }
    };

    thread_pool->execute_n(0, static_cast<int64_t>(num_threads), update_task);
    return 0;
}

template<typename TypeA, typename TypeB>
TypeA cpu_kernel_add(TypeA a, TypeB b) {
    return a + static_cast<TypeA>(b);
}

template<typename IndexT, typename ValueT, typename UpdateT>
int cpu_kernel_update_accumulate(thread_pool_ptr_t thread_pool,
              uint64_t n,
              const IndexT* keys, 
              size_t value_stride, 
              const void* updates, 
              int8_t* uvm_table_ptr,
              size_t row_size_in_bytes,
              uint64_t num_threads)
{
    auto keys_per_task = (n + num_threads - 1)/ num_threads;
    const size_t num_values = row_size_in_bytes / sizeof(ValueT);

    const auto update_task = [=] (const size_t idx) {
        const auto base_key = (idx) * keys_per_task;
        
        for (uint64_t i = 0; i < keys_per_task; i++) {
            if (base_key + i >= n) {
                break;
            }
            IndexT key = reinterpret_cast<const IndexT*>(keys)[base_key + i];
            ValueT* dst_ptr = reinterpret_cast<ValueT*>(uvm_table_ptr + (static_cast<size_t>(key) * row_size_in_bytes));
            const UpdateT* src_ptr = reinterpret_cast<const UpdateT*>(reinterpret_cast<const int8_t*>(updates) + (base_key + i) * value_stride);
            
            // Accumulate each value in the row
            for (size_t j = 0; j < num_values; j++) {
                dst_ptr[j] = cpu_kernel_add(dst_ptr[j], static_cast<ValueT>(src_ptr[j]));
            }
        }
    };

    thread_pool->execute_n(0, static_cast<int64_t>(num_threads), update_task);
    return 0;
}

template<typename IndexT>
void cpu_kernel_update_accumulate_dispatch(thread_pool_ptr_t thread_pool,
              uint64_t n,
              const IndexT* keys, 
              size_t value_stride, 
              const void* updates, 
              int8_t* uvm_table_ptr,
              size_t row_size_in_bytes,
              DataType_t update_dtype,
              uint64_t num_threads)
{
    // Dispatch based on update_dtype
    switch (update_dtype) {
        case DataType_t::Float32:
        {   
            int res = cpu_kernel_update_accumulate<IndexT, float, float>(
                thread_pool, n, keys, value_stride, updates, uvm_table_ptr, row_size_in_bytes, num_threads);
            NVE_CHECK_(res == 0);
            break;
        }
        default:
            NVE_THROW_("Unsupported data type ", update_dtype);
    }
}

}
