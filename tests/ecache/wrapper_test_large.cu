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

#include "gtest/gtest.h"
#include "samples/ecache/wrapper_sample/cache_wrapper.h"
#include "samples/ecache/wrapper_sample/training_wave.h"
#include <unordered_map>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstring>
#include <memory>
#include <unordered_set>
#include <random>
#include <type_traits>
#include "../common/check_error.h"

// Generate reference datavector for an index (used for correctness checking)
template <typename ValType>
void PushValue(std::vector<unsigned char>& vec, ValType value) {
    constexpr auto num_bytes = sizeof(ValType);
    for (uint64_t i=0 ; i<num_bytes ; i++) {
        vec.push_back(value % 256);
        value = (value >> 8);
    };
}

template <typename IndexType>
std::vector<unsigned char> GenerateRefVector(IndexType index, int64_t size_in_bytes, uint64_t version = 0) {
  static_assert(sizeof(unsigned char) == 1);
  std::vector<unsigned char> result;
  
  // Start with index
  PushValue<uint64_t>(result, index);
  // Next push version
  PushValue<uint64_t>(result, version);

  // Then pad with psuedo-random
  while (static_cast<int64_t>(result.size()) < size_in_bytes) {
    auto pos = result.size();
    uint64_t mask = 0x31337beefa941548;
    uint64_t x = mask ^ (static_cast<uint64_t>(pos + index * pos) * mask);
    if (!x) x = mask;
    PushValue<uint64_t>(result, x);
  }

  // Finally, trim to requested size
  while (static_cast<int64_t>(result.size()) > size_in_bytes) {
    result.pop_back();
  }

  return result;
}

// Reference Implementation of the cache wrapper class for testing.
// Important: this class is host side only (no GPU usage), so all device pointers (d_*) should be in system memory!
template<typename IndexT>
class WrapperRef
{
public:

    WrapperRef([[maybe_unused]] allocator_ptr_t allocator = {}, [[maybe_unused]] std::shared_ptr<nve::Logger> logger = {}) {
        // Input args unused but aligned wih ECacheWrapper API
    }
    ~WrapperRef() {}
    void Init(
        uint64_t /*cache_size*/,
        uint64_t row_elements,
        nve::DataTypeFormat element_format,
        const std::vector<cudaStream_t>& /*lookup_streams*/,
        cudaStream_t /*modify_stream*/,
        uint64_t /*max_modify_size*/)
    {
        m_element_format = element_format;

        m_row_bytes = row_elements;
        switch (m_element_format)
        {
            case nve::DATATYPE_FP16:
                m_row_bytes *= sizeof(__half);
                break;
            case nve::DATATYPE_FP32:
                m_row_bytes *= sizeof(float);
                break;
            default:
                FAIL() << "Invalid element type.";
        }

        if (m_row_bytes < (sizeof(uint64_t)*2)) {
            FAIL() << "Cache row size is too small (minimum is 2*uint64_t for index and version).";
        }
    }
    const std::vector<cudaStream_t>& GetLookupStreams() const {
        static std::vector<cudaStream_t> lookup_streams{0};
        return lookup_streams;
    }
    cudaStream_t GetModifyStream() const { return 0; }
    nve::allocator_ptr_t GetAllocator() const { return {}; }
    void Lookup(uint64_t /*stream_index*/, const IndexT* d_keys, const uint64_t num_keys, int8_t* d_values, uint64_t stride, uint64_t* d_hit_mask)
    {
        const uint64_t mask_elements = ((num_keys + 63) / 64); 
        std::memset(d_hit_mask, 0, sizeof(uint64_t) * mask_elements);
        for (uint64_t i=0 ; i<num_keys ; i++)
        {
            const IndexT idx = d_keys[i];
            auto iter = m_kv.find(idx);
            const uint64_t bit = uint64_t(1) << (i % 64);
            if (iter == m_kv.end()) {
                // miss
                d_hit_mask[i/64] &= ~bit;
            } else {
                // hit
                std::memcpy(d_values + (i*stride), iter->second.data() , m_row_bytes);
                d_hit_mask[i/64] |= bit;
            }
        }
    }
    void Insert(const IndexT* h_keys, const uint64_t num_keys, const int8_t* d_values, uint64_t stride)
    {
        // Assume no eviction logic
        // Use the following cache APIs to setup properly
        //  GetKeysStoredInCache
        //  GetMaxNumEmbeddingVectorsInCache
        for (uint64_t i=0 ; i<num_keys ; i++)
        {
            auto vec_start = d_values + (i * stride);
            auto vec_end = vec_start + m_row_bytes;
            m_kv.insert_or_assign(h_keys[i], std::vector<unsigned char>(vec_start, vec_end));
        }
    }
    void Update(const IndexT* h_keys, const uint64_t num_keys, const int8_t* d_values, uint64_t stride)
    {
        for (uint64_t i=0 ; i<num_keys ; i++)
        {
            const IndexT idx = h_keys[i];
            auto iter = m_kv.find(idx);
            if (iter != m_kv.end()) { // hit
                std::memcpy(iter->second.data(), d_values + (i * stride) , m_row_bytes);
            }
        }
    }
    void Accumulate(const IndexT* h_keys, const uint64_t num_keys, const int8_t* /*d_values*/, uint64_t /*stride*/, nve::DataTypeFormat /*value_format*/)
    {
        // Special handling for first 16 bytes (index,version)
        // For this ref usage we only increment the version (no real accumulation)
        for (uint64_t i=0 ; i<num_keys ; i++)
        {
            const IndexT idx = h_keys[i];
            auto iter = m_kv.find(idx);
            if (iter != m_kv.end()) { // hit
                uint64_t *vals = reinterpret_cast<uint64_t*>(iter->second.data());
                vals[1]++;
            }
        }
    }
    void AccumulateNoSync(const IndexT* d_keys, const uint64_t num_keys, const int8_t* d_values, uint64_t stride, nve::DataTypeFormat value_format)
    {
        Accumulate(d_keys, num_keys, d_values, stride, value_format);
    }

    // For testing purposes
    uint64_t GetRowBytes() const { return m_row_bytes; }
    size_t Size() const { return m_kv.size(); }
private:
    uint64_t m_row_bytes{0};
    nve::DataTypeFormat m_element_format{nve::DataTypeFormat::NUM_DATA_TYPES_FORMATS};
    std::unordered_map<IndexT, std::vector<unsigned char>> m_kv;
};
template class WrapperRef<int32_t>;
template class WrapperRef<int64_t>;

template<typename IndexT>
class WrapperTestLarge : public testing::Test
{
public:
    ~WrapperTestLarge()
    {
        if (m_allocator) {
            CHECK_EC(m_allocator->deviceFree(m_d_values));
            CHECK_EC(m_allocator->hostFree(m_h_values));
            CHECK_EC(m_allocator->deviceFree(m_d_input));
            CHECK_EC(m_allocator->hostFree(m_h_hitmask));
            CHECK_EC(m_allocator->deviceFree(m_d_hitmask));
            CHECK_EC(m_allocator->hostFree(m_h_ref_values));
            CHECK_EC(m_allocator->hostFree(m_h_ref_hitmask));
        }
    }
    void InitCache(
        uint64_t cache_size,
        uint64_t row_elements,
        nve::DataTypeFormat element_format,
        const std::vector<cudaStream_t>& lookup_streams,
        cudaStream_t modify_stream,
        uint64_t max_modify_size = (1u<<20))
    {
        // init cache and ref
        m_max_modify_size = max_modify_size;
        m_cache = std::make_shared<ECacheWrapper<IndexT>>();
        m_cache->Init(cache_size, row_elements, element_format, lookup_streams, modify_stream, m_max_modify_size);
        m_ref = std::make_shared<WrapperRef<IndexT>>();
        m_ref->Init(cache_size, row_elements, element_format, lookup_streams, modify_stream, m_max_modify_size);
        m_row_bytes = m_ref->GetRowBytes();
        m_allocator = m_cache->GetAllocator();
    }

    void InitInputs(
        uint64_t num_inputs,
        uint64_t num_rows,
        uint64_t hotness,
        uint64_t batch_size,
        float alpha)
    {
        // generate a bunch kv pairs
        m_data = std::make_shared<typename TrainingWave<IndexT>::TableData>(m_cache, num_inputs, num_rows, hotness, batch_size, alpha, 0, 0);
        // insert kv to cache
        const uint64_t num_keys = batch_size * hotness;
        const uint64_t input_size = num_keys * sizeof(IndexT);
        const uint64_t hitmask_size = ((num_keys + 63) / 64) * sizeof(uint64_t);
        const uint64_t value_buffer_size = num_keys * m_row_bytes;
        //allocate d_values and populate with ref (GenerateRefVector)
        CHECK_EC(m_allocator->deviceAllocate((void**)(&m_d_values), value_buffer_size));
        CHECK_EC(m_allocator->hostAllocate((void**)(&m_h_values), value_buffer_size));
        CHECK_EC(m_allocator->hostAllocate((void**)(&m_h_ref_values), value_buffer_size));
        for (auto& input : m_data->m_host_inputs) {
            const auto num_indices = input.size();
            for (uint64_t i=0 ; i<num_indices ; i++) {
                auto ref = GenerateRefVector<IndexT>(input[i], m_row_bytes);
                std::memcpy(m_h_values + (i*m_row_bytes), ref.data(), ref.size());
            }
            CHECK_CUDA_ERROR(cudaMemcpy(m_d_values, m_h_values, num_indices * m_row_bytes, cudaMemcpyDefault));
            m_cache->Insert(input.data(), num_indices, m_d_values, m_row_bytes);
            CHECK_CUDA_ERROR(cudaStreamSynchronize(m_cache->GetModifyStream()));
        }
        NVE_CHECK_(m_allocator->deviceAllocate((void**)(&m_d_input), input_size));
        NVE_CHECK_(m_allocator->hostAllocate((void**)(&m_h_hitmask), hitmask_size));
        NVE_CHECK_(m_allocator->hostAllocate((void**)(&m_h_ref_hitmask), hitmask_size));
        NVE_CHECK_(m_allocator->deviceAllocate((void**)(&m_d_hitmask), hitmask_size));

        // query residency (fail if too low)
        auto ec = m_cache->GetCache();
        auto max_keys = ec->GetMaxNumEmbeddingVectorsInCache();
        std::vector<IndexT> resident_indices(max_keys);
        size_t num_resident_keys(0);
        nve::LookupContextHandle ctx;
        CHECK_EC(ec->LookupContextCreate(ctx, nullptr, 0));
        CHECK_EC(ec->GetKeysStoredInCache(ctx, resident_indices.data(), num_resident_keys));
        CHECK_EC(ec->LookupContextDestroy(ctx));
        resident_indices.resize(num_resident_keys);

        // check if residency is too low
        std::unordered_set<IndexT> unique_indices;
        for (auto& input : m_data->m_host_inputs) {
            for (auto& idx : input) {
                unique_indices.insert(idx);
            }
        }

        constexpr double min_cache_ratio = 0.9;
        const auto min_residency = min_cache_ratio * static_cast<double>(std::min(max_keys, unique_indices.size()));
        EXPECT_GE(num_resident_keys, min_residency);

        // replicate residency on ref
        auto insert_size = (value_buffer_size / m_row_bytes);
        
        for (uint64_t start=0 ; start<num_resident_keys ; start+=insert_size) {
            uint64_t end = std::min((start + insert_size), num_resident_keys);
            // populate m_h_values with ref vectors
            for(uint64_t i=start ; i<end ; i++){
                auto ref = GenerateRefVector<IndexT>(resident_indices[i], m_row_bytes);
                std::memcpy(m_h_values + ((i-start) * m_row_bytes), ref.data(), m_row_bytes);
            }
            // insert to ref cache
            m_ref->Insert(resident_indices.data() + start, (end - start), m_h_values, m_row_bytes);
        }

        EXPECT_EQ(num_resident_keys, m_ref->Size());
    }

    void Lookup(bool allow_older_version = false) {
        for (auto& input : m_data->m_host_inputs) {
            // copy host input to device buffer
            CHECK_CUDA_ERROR(cudaMemcpy(m_d_input, input.data(), input.size() * sizeof(IndexT), cudaMemcpyDefault));
            // call cache lookupcd
            CHECK_CUDA_ERROR(cudaMemset(m_d_hitmask, 0, ((input.size() + 63) / 64) * sizeof(uint64_t))); 
            m_cache->Lookup(0, m_d_input, input.size(), m_d_values, m_row_bytes, m_d_hitmask);
            CHECK_CUDA_ERROR(cudaStreamSynchronize(m_cache->GetLookupStreams().at(0)));
            // copy out values and hitmask
            const uint64_t output_size = input.size() * m_row_bytes;
            CHECK_CUDA_ERROR(cudaMemcpy(m_h_values, m_d_values, output_size, cudaMemcpyDefault));
            const uint64_t hitmask_size = ((input.size() + 63) / 64) * sizeof(uint64_t);
            CHECK_CUDA_ERROR(cudaMemcpy(m_h_hitmask, m_d_hitmask, hitmask_size, cudaMemcpyDefault));
            // run lookup on ref
            m_ref->Lookup(0, input.data(), input.size(), m_h_ref_values, m_row_bytes, m_h_ref_hitmask);

            // compare cache and ref hitmask
            const uint64_t hitmask_elements = ((input.size() + 63) / 64);
            for (uint64_t i=0 ; i<hitmask_elements ; i++)
            {
                ASSERT_EQ(m_h_hitmask[i], m_h_ref_hitmask[i]);
            }

            // compare cache and ref output
            uint64_t* cache_output = reinterpret_cast<uint64_t*>(m_h_values);
            uint64_t* ref_output = reinterpret_cast<uint64_t*>(m_h_ref_values);
            EXPECT_EQ(m_row_bytes % sizeof(uint64_t), 0);
            const auto num_rows = output_size / m_row_bytes;
            const auto elements_in_row = m_row_bytes / sizeof(uint64_t);
            
            for (uint64_t i=0 ; i<num_rows ; i++)
            {
                const auto mask = m_h_hitmask[i/64];
                const auto bit = uint64_t(1) << (i % 64);
                if (mask & bit)
                {
                    for (uint64_t j=0 ; j<elements_in_row ; j++)
                    {
                        const auto offset = (i*elements_in_row) + j;
                        if(allow_older_version && (j == 1)) {
                            // comparing version allowing for older versions in cache
                            ASSERT_LE(cache_output[offset], ref_output[offset]);
                        } else {
                            ASSERT_EQ(cache_output[offset], ref_output[offset]);
                        }
                    }
                }
            }
        }
    }

    void Update(uint64_t num_updates = 1) {
        const auto num_inputs = m_data->m_host_inputs.size();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<unsigned long long> dist(
            std::numeric_limits<std::uint64_t>::min(),
            std::numeric_limits<std::uint64_t>::max()
        );

        for (uint64_t u=0 ; u<num_updates ; u++) {
            for (uint64_t i=0 ; i<num_inputs ; i++) {
                auto& unique_input = m_data->m_host_inputs_unique.at(i % num_inputs);
                const auto unique_size = unique_input.size();
                auto random_uint64 = dist(gen);
                uint64_t update_size = random_uint64 % unique_size;
                for (uint64_t j=0 ; j<update_size ; j++) {
                    auto ref = GenerateRefVector<IndexT>(unique_input[i], m_row_bytes, u + 1);
                    // use m_h_output for the update values as we're not using it atm and it's large enough
                    std::memcpy(m_data->m_h_output + (j * m_row_bytes), ref.data(), m_row_bytes);
                }
                // copy update data (reuse m_d_accmulate as it's large enough and unused at this point)
                CHECK_CUDA_ERROR(cudaMemcpy(m_data->m_d_accumulate, m_data->m_h_output, update_size * m_row_bytes, cudaMemcpyDefault));
                // Update cache
                m_cache->Update(unique_input.data(), update_size, m_data->m_d_accumulate, m_row_bytes);
                // Update ref
                m_ref->Update(unique_input.data(), update_size, m_data->m_h_output, m_row_bytes);
            }
        }
        CHECK_CUDA_ERROR(cudaStreamSynchronize(m_cache->GetModifyStream()));
    }

    std::shared_ptr<ECacheWrapper<IndexT>> m_cache;
    std::shared_ptr<WrapperRef<IndexT>> m_ref;
    std::shared_ptr<typename TrainingWave<IndexT>::TableData> m_data;
    uint64_t m_row_bytes;
    uint64_t m_max_modify_size;
    int8_t* m_h_values{nullptr};
    int8_t* m_d_values{nullptr};
    IndexT* m_d_input{nullptr};
    uint64_t* m_h_hitmask{nullptr};
    uint64_t* m_d_hitmask{nullptr};

    int8_t* m_h_ref_values{nullptr};
    uint64_t* m_h_ref_hitmask{nullptr};

    nve::allocator_ptr_t m_allocator;
};

using WrapperTestLarge64 = WrapperTestLarge<int64_t>;
using WrapperTestLarge32 = WrapperTestLarge<int32_t>;

TEST_F(WrapperTestLarge64, SingleStream)
{
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    std::vector<cudaStream_t> lookup_streams {stream};

    InitCache(1<<30, 256, nve::DATATYPE_FP16, lookup_streams, lookup_streams.at(0));
    InitInputs(10, 1<<30, 512, 512, 1.05f);
    Lookup();
    Update(10);
    Lookup();
}
