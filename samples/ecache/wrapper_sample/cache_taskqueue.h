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
#include <thread>
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <deque>
#include <iostream>
#include <cuda_support.hpp>
#include "cache_wrapper.h"
#include "cache_metrics.h"
#include "work_queue.h"
#include "mock_cache.hpp"


template<typename IndexT, int32_t SPLIT_SIZE>
class Deduper;

using namespace nve;

template<typename IndexT>
class CacheTaskQueue {
public:
    CacheTaskQueue(
        std::shared_ptr<ECacheWrapper<IndexT>> cache,
        int device_id,
        uint64_t insert_key_buffer_size,
        uint64_t insert_val_buffer_size,
        uint64_t max_keys,
        std::shared_ptr<MockHostCache<IndexT>> host_cache,
        std::shared_ptr<MockParameterServer<IndexT>> ps);
    ~CacheTaskQueue()
    {
        auto allocator = m_cache->GetAllocator();
        NVE_CHECK_(allocator->hostFree(m_insert_h_keys));
        NVE_CHECK_(allocator->deviceFree(m_insert_d_values));
        NVE_CHECK_(allocator->deviceFree(m_d_unique_buffer));
        NVE_CHECK_(allocator->hostFree(m_priority));

    }
    void Drain() {
        m_main_queue.Submit([](){}).wait();
        m_insert_queue.Submit([](){}).wait();
    }

    std::future<void> Lookup(
        IndexT* d_keys,
        IndexT* h_keys,
        const uint64_t batch_size,
        const uint64_t hotness,
        int8_t* d_values,
        int8_t* h_values,
        uint64_t* d_hit_mask,
        uint64_t* h_hit_mask,
        int8_t* d_pooling_output,
        bool host_input,
        CacheIterationMetric* metric);

    
    std::future<void> LookupDeduped(
        IndexT* d_keys,
        IndexT* h_keys,
        uint64_t hotness,
        uint64_t batch_size,
        IndexT* d_unique_keys, 
        IndexT* h_unique_keys,
        IndexT* d_counts, 
        IndexT* h_counts,
        IndexT* d_inverse_buffer, 
        IndexT* d_offsets,
        IndexT* h_num_runs, 
        int8_t* d_values,
        int8_t* h_values,
        uint64_t* d_hit_mask,
        uint64_t* h_hit_mask,
        int8_t* d_pooling_values,
        bool host_input,
        CacheIterationMetric* metric);

    std::future<void> GradientsDedup(
        const uint64_t num_unique_keys,
        const int8_t* d_gradients,
        int8_t* d_accumulate,
        const IndexT* d_grads_unique_keys,
        const IndexT* d_grads_counts,
        const IndexT* d_grads_offsets,
        const IndexT* m_d_grads_inverse_buffer,
        CacheIterationMetric* metric);

    std::future<void> Accumulate(
        const IndexT* d_keys,
        const uint64_t num_keys,
        int8_t* d_values,
        nve::DataTypeFormat value_format,
        const IndexT* h_keys,
        int8_t* h_values,
        CacheIterationMetric* metric);

    std::shared_ptr<ECacheWrapper<IndexT>> GetCache() const { return m_cache; }
private:
    // Cache wrapper
    std::shared_ptr<ECacheWrapper<IndexT>> m_cache;

    // Main work queue for lookup/accumulate tasks
    WorkQueue m_main_queue;

    // Secondary work queue for offloading "cache Insert()" and auxiliary resources
    WorkQueue m_insert_queue;
    IndexT* m_insert_h_keys {nullptr};
    int8_t* m_insert_d_values {nullptr};
    std::atomic_flag m_insert_lock = ATOMIC_FLAG_INIT;

    int8_t* m_d_unique_buffer {nullptr};
    float* m_priority {nullptr};
    std::shared_ptr<Deduper<IndexT, -1>> m_deduper;

    // Mock host cache
    std::shared_ptr<MockHostCache<IndexT>> m_host_cache;
    // Mock param sertver
    std::shared_ptr<MockParameterServer<IndexT>> m_ps;
};