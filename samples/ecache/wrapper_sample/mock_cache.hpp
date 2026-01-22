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
#include <bits/stdint-uintn.h>
#include <unordered_map>
#include <stdexcept>
#include <random>
#include <cassert>
#include <cstring>
#include <thread>
#include "work_queue.h"
#include <cuda_fp16.h>
#include <chrono>
#include <cuda_runtime.h>
#include "cache_metrics.h"
#include <embedding_cache_combined.h>

/**
 * Mock host cache
 * Supporting Lookup/Accumulate from a DRAM bandwidth standpoint.
 * Not modelling correctness.
 * Not modelling locking/synchronization.
*/
template<typename IndexT>
class MockHostCache {
public:
    MockHostCache(
        uint64_t num_rows,
        uint64_t row_size,
        size_t num_threads,
        float target_hitrate
    ) : m_num_rows(num_rows), m_row_size(row_size), m_target_hitrate(target_hitrate), m_work_queue(0, num_threads) {
        m_data = new int8_t[num_rows * row_size];
        if (!m_data) {
            throw std::runtime_error("Failed to allocate host cache!");
        }
    }
    ~MockHostCache() {
        delete[] m_data;
    }

    void Lookup(
        IndexT* h_keys,
        const uint64_t num_keys,
        int8_t* h_values,
        uint64_t stride,
        uint64_t* h_hit_mask,
        CacheIterationMetric* metric)
    {
        if (metric) {
            metric->host_cache_lookup.h_start = metric->Now();
        }
        std::vector<std::future<void>> futures;
        constexpr uint64_t hitmask_elem_bits = sizeof(h_hit_mask) * 8;
        for (uint64_t k=0 ; k<num_keys ; k+=hitmask_elem_bits) {
            const uint64_t keys = std::min(hitmask_elem_bits, num_keys - k);
            futures.emplace_back(
                Lookp64(
                    h_keys + k,
                    keys,
                    h_values + (k * stride),
                    stride,
                    h_hit_mask + (k / hitmask_elem_bits)
                )
            );
        }
        for (auto& f : futures) {
            f.wait();
        }
        if (metric) {
            metric->host_cache_lookup.h_end = metric->Now();
        }
    }

    void Accumulate(
        const IndexT* h_keys,
        const uint64_t num_keys,
        int8_t* h_values,
        nve::DataTypeFormat value_format,
        CacheIterationMetric* metric)
    {
        if (metric) {
            metric->host_cache_accumulate.h_start = metric->Now();
        }
        std::vector<std::future<void>> futures;
        uint64_t constexpr group_size(64);
        for (uint64_t k=0 ; k<num_keys ; k+=group_size) {
            const uint64_t keys = std::min(group_size, num_keys - k);
            futures.emplace_back(
                AccumulateGroup(
                    h_keys + k,
                    keys,
                    h_values + (k * m_row_size),
                    value_format
                )
            );
        }
        for (auto& f : futures) {
            f.wait();
        }
        if (metric) {
            metric->host_cache_accumulate.h_end = metric->Now();
        }
    }

private:
    // Lookup (upto) 64 rows
    // The 64 matches the size of the hit mask element (in bits).
    std::future<void> Lookp64(
        IndexT* h_keys,
        const uint64_t num_keys,
        int8_t* h_values,
        uint64_t stride,
        uint64_t* h_hit_mask) {
        return m_work_queue.Submit([=]() {
            assert(num_keys <= (sizeof(h_hit_mask) * 8));
            for (uint64_t k=0; k<num_keys; k++) {
                const IndexT key = h_keys[k];
                const uint64_t key_mask = (1ul << k);
                if (!(*h_hit_mask & key_mask))
                {
                    static thread_local std::mt19937 gen(std::hash<std::thread::id>{}(std::this_thread::get_id()));
                    std::uniform_real_distribution<> dis(0.0, 1.0);
                    if (dis(gen) <= m_target_hitrate) {
                        // read data
                        std::memcpy(
                            h_values + (k * stride),
                            m_data + (m_row_size * (key % m_num_rows)),
                            m_row_size);
                        // update hitmask
                        *h_hit_mask |= key_mask;
                    }
                }
            }
        });
    }

    template<typename T>
    static inline void AccumulateRow(int8_t* __restrict dst, const int8_t* __restrict src, const uint64_t row_size) {
        const uint64_t num_vals = row_size / sizeof(T);
        const T* src_ = reinterpret_cast<const T*>(src);
        T* dst_ = reinterpret_cast<T*>(dst);
        for (uint64_t i=0 ; i<num_vals ; i++) {
            dst_[i] =  static_cast<T>(static_cast<float>(dst_[i]) + static_cast<float>(src_[i]));
        }
    }

    std::future<void> AccumulateGroup(
        const IndexT* h_keys,
        const uint64_t num_keys,
        const int8_t* h_values,
        nve::DataTypeFormat value_format) {
        return m_work_queue.Submit([=]() {
            for (uint64_t k=0; k<num_keys; k++) {
                const IndexT key = h_keys[k];
                static thread_local std::mt19937 gen(std::hash<std::thread::id>{}(std::this_thread::get_id()));
                std::uniform_real_distribution<> dis(0.0, 1.0);
                if (dis(gen) <= m_target_hitrate) {
                    switch (value_format)
                    {
                    case nve::DATATYPE_FP16:
                        AccumulateRow<half>(
                            m_data + ((key % m_num_rows) * m_row_size),
                            h_values + (k * m_row_size),
                            m_row_size);
                        break;
                    case nve::DATATYPE_FP32:
                        AccumulateRow<float>(
                            m_data + ((key % m_num_rows) * m_row_size),
                            h_values + (k * m_row_size),
                            m_row_size);
                        break;
                    default:
                        assert(0); // unhandled datatype
                        break;
                    }
                }
            }
        });
    }

    const uint64_t m_num_rows;
    const uint64_t m_row_size;
    const float m_target_hitrate;

    int8_t* m_data;
    WorkQueue m_work_queue;
};

/**
 * Mock parameter server
 * Not modelling correctness.
 * Not modelling locking/synchronization.
*/
template<typename IndexT>
class MockParameterServer {
public:
    MockParameterServer(uint64_t row_size, size_t num_threads, uint64_t min_latency_ms) 
    : m_row_size(row_size), m_min_latency_ms(min_latency_ms), m_work_queue(0, num_threads) {}
    ~MockParameterServer() = default;

    void Lookup(
        IndexT* h_keys,
        const uint64_t num_keys,
        int8_t* h_values,
        uint64_t stride,
        uint64_t* h_hit_mask,
        CacheIterationMetric* metric)
    {
        auto start_time = CacheIterationMetric::Now();
        if (metric) {
            metric->ps_lookup.h_start = start_time;
        }
        std::vector<std::future<void>> futures;
        constexpr uint64_t hitmask_elem_bits = sizeof(h_hit_mask) * 8;
        for (uint64_t k=0 ; k<num_keys ; k+=hitmask_elem_bits) {
            const uint64_t keys = std::min(hitmask_elem_bits, num_keys - k);
            futures.emplace_back(
                Lookp64(
                    h_keys + k,
                    keys,
                    h_values + (k * stride),
                    stride,
                    h_hit_mask + (k / hitmask_elem_bits)
                )
            );
        }
        for (auto& f : futures) {
            f.wait();
        }
        const auto lookup_duration = std::chrono::duration<float, std::micro>(CacheIterationMetric::Now() - start_time).count();
        const uint64_t sleep_duration = static_cast<uint64_t>((1000.f * static_cast<float>(m_min_latency_ms)) - lookup_duration);
        if (sleep_duration > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_duration));
        }
        if (metric) {
            metric->ps_lookup.h_end = metric->Now();
        }
    }
    void Accumulate(
        const IndexT* h_keys,
        const uint64_t num_keys,
        int8_t* h_values,
        nve::DataTypeFormat value_format,
        CacheIterationMetric* metric)
    {
        if (metric) {
            metric->ps_accumulate.h_start = metric->Now();
        }
        std::vector<std::future<void>> futures;
        uint64_t constexpr group_size(64);
        for (uint64_t k=0 ; k<num_keys ; k+=group_size) {
            const uint64_t keys = std::min(group_size, num_keys - k);
            futures.emplace_back(
                AccumulateGroup(
                    h_keys + k,
                    keys,
                    h_values + (k * m_row_size),
                    value_format
                )
            );
        }

        if (metric) {
            metric->ps_accumulate.h_end = metric->Now();
        }
    }
private:
    // Lookup (upto) 64 rows
    // The 64 matches the size of the hit mask element (in bits).
    std::future<void> Lookp64(
        IndexT* h_keys,
        const uint64_t num_keys,
        int8_t* h_values,
        uint64_t stride,
        uint64_t* h_hit_mask) {
        return m_work_queue.Submit([=]() {
            assert(num_keys <= (sizeof(h_hit_mask) * 8));
            for (uint64_t k=0; k<num_keys; k++) {
                const IndexT key = h_keys[k];
                const uint64_t key_mask = (1ul << k);
                if (!(*h_hit_mask & key_mask))
                {
                    // write result data
                    std::memset(
                        h_values + (k * stride),
                        0xff,
                        m_row_size);
                }
            }
            *h_hit_mask = uint64_t(-1);
        });
    }

    std::future<void> AccumulateGroup(
        const IndexT* h_keys,
        const uint64_t num_keys,
        const int8_t* h_values,
        nve::DataTypeFormat value_format) {
        return m_work_queue.Submit([=]() {
            for (uint64_t k=0; k<num_keys; k++) {
                const IndexT key = h_keys[k];
                switch (value_format)
                {
                case nve::DATATYPE_FP16:
                    ReadRow<int16_t>( // using int16_t instead of half to accomodate the volatile in ReadRow()
                        h_values + (k * m_row_size),
                        m_row_size);
                    break;
                case nve::DATATYPE_FP32:
                    ReadRow<float>(
                        h_values + (k * m_row_size),
                        m_row_size);
                    break;
                default:
                    assert(0); // unhandled datatype
                    break;
                }
            }
        });
    }

    // Only simulating read of the gradients
    template<typename T>
    static inline void ReadRow(const int8_t* __restrict src, const uint64_t row_size) {
        const uint64_t num_vals = row_size / sizeof(T);
        const T* src_ = reinterpret_cast<const T*>(src);
        volatile T mock_val; // volatile to prevent the compiler from optimizing the reads below to NOP
        for (uint64_t i=0 ; i<num_vals ; i++) {
            mock_val = src_[i];
        }
        mock_val;
    }

    const uint64_t m_row_size;
    const uint64_t m_min_latency_ms;
    WorkQueue m_work_queue;
};

class MockGPUWork {
public:
    MockGPUWork(uint64_t latency_ms) : m_latency_ms(latency_ms) {
        // Query device properties
        int device;
        cudaDeviceProp deviceProp;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&deviceProp, device);

        m_blocks_per_grid = deviceProp.multiProcessorCount;
        m_threads_per_block = deviceProp.maxThreadsPerBlock;
    }
    ~MockGPUWork() = default;
    void Run(cudaStream_t stream);
    const uint64_t m_latency_ms;
private:
    int m_blocks_per_grid;
    int m_threads_per_block;
};
