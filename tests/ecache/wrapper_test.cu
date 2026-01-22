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
#include "../common/check_error.h"

struct LookupInsertUpdateTestParams {
  uint64_t cache_size;
  uint64_t row_elements;
  uint64_t num_keys;
  uint64_t num_embeddings;
  nve::DataTypeFormat update_data_type;
};

template<typename KeyT_>
class WrapperTest : public testing::TestWithParam<LookupInsertUpdateTestParams>
{
protected:
    using KeyT = KeyT_;
    WrapperTest() : m_d_values(nullptr), m_h_values(nullptr), d_keys(nullptr), m_d_hit_mask(nullptr), m_h_hit_mask(nullptr)
    {
        constexpr auto max_streams = 16;
        for (uint32_t i = 0; i < max_streams; i++)
        {
            cudaStream_t stream;
            CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
            m_streams.push_back(stream);
        }
    }

    std::vector<float> GenerateVector(KeyT key, uint64_t version, uint64_t row_elements)
    {
        std::vector<float> ret(row_elements);
        ret[0] = static_cast<float>(version);
        for (uint32_t i = 1; i < row_elements; i++)
        {
            ret[i] = static_cast<float>(key);
        }
        return ret;
    }

    std::vector<float> UpdateVersion(uint64_t row_elements)
    {
        std::vector<float> ret(row_elements);
        ret[0] = 1.f;
        for (uint32_t i = 1; i < row_elements; i++)
        {
            ret[i] = 0.f;
        }
        return ret;
        
    }

    template<typename DataT, typename ScaleT>
    void Quantize(const std::vector<float>& data, DataT* dst) {
        std::vector<DataT> data_quantized;
	ScaleT scale = 0.015625;
	for (size_t i=0; i<data.size(); i++) {
	    DataT val = static_cast<DataT>(data[i] / scale);
	    data_quantized.push_back(val);
        }
        memcpy(dst, data_quantized.data(), data.size());
        memcpy(dst + data.size(), &scale, sizeof(ScaleT));
    }

    void AllocateBuffers(uint64_t max_keys, uint64_t max_embeddings, uint64_t embedding_width_in_bytes)
    {
        hit_mask_sz_in_bytes = ((max_embeddings+63)/64)*sizeof(uint64_t);
        CHECK_CUDA_ERROR(cudaMalloc(&d_keys, sizeof(KeyT)*max_keys));
        CHECK_CUDA_ERROR(cudaMalloc(&m_d_values, embedding_width_in_bytes * max_embeddings));
        CHECK_CUDA_ERROR(cudaMallocHost(&m_h_values, embedding_width_in_bytes * max_embeddings));
        CHECK_CUDA_ERROR(cudaMalloc(&m_d_hit_mask, hit_mask_sz_in_bytes));
        CHECK_CUDA_ERROR(cudaMallocHost(&m_h_hit_mask, hit_mask_sz_in_bytes));
    }
    int8_t* m_d_values{nullptr};
    int8_t* m_h_values{nullptr};
    uint64_t* m_d_hit_mask{nullptr};
    uint64_t* m_h_hit_mask{nullptr};
    size_t hit_mask_sz_in_bytes{0};
    KeyT* d_keys{nullptr};
    std::vector<cudaStream_t> m_streams;
    ECacheWrapper<KeyT> m_cache_wrapper;
};

using WrapperTest64 = WrapperTest<int64_t>;

TEST_F(WrapperTest64, Init)
{
    constexpr auto num_lookup_streams = 4;
    m_cache_wrapper.Init(64*1024, 8, nve::DATATYPE_FP16, std::vector<cudaStream_t>(m_streams.begin(), m_streams.begin()+num_lookup_streams), m_streams[num_lookup_streams]);
}

TEST_F(WrapperTest64, EmptyLookup)
{
    constexpr auto num_lookup_streams = 4;
    constexpr uint64_t cache_size = 16*1024;
    constexpr uint64_t row_elements = 8;
    constexpr uint64_t num_keys = 1;
    constexpr uint64_t num_embeddings = 16;
    constexpr auto data_type = nve::DATATYPE_FP32;
    constexpr uint64_t embedding_width_in_bytes = row_elements * 4;
    AllocateBuffers(num_keys, num_embeddings, embedding_width_in_bytes);
    m_cache_wrapper.Init(cache_size, row_elements, data_type, std::vector<cudaStream_t>(m_streams.begin(), m_streams.begin()+num_lookup_streams), m_streams[num_lookup_streams]);

    std::vector<KeyT> h_keys(num_keys);
    h_keys.push_back(1);

    CHECK_CUDA_ERROR(cudaMemsetAsync(m_d_hit_mask, 0, hit_mask_sz_in_bytes, (m_cache_wrapper.GetLookupStreams()[0])));
    CHECK_CUDA_ERROR(cudaMemcpy(d_keys, h_keys.data(), sizeof(KeyT)*num_keys, cudaMemcpyDefault));
    m_cache_wrapper.Lookup(0, d_keys, 1, m_d_values, embedding_width_in_bytes, m_d_hit_mask);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_h_hit_mask, m_d_hit_mask, hit_mask_sz_in_bytes, cudaMemcpyDefault, m_cache_wrapper.GetLookupStreams()[0]));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_cache_wrapper.GetLookupStreams()[0]));
    EXPECT_EQ(m_h_hit_mask[0], 0);
}

TEST_F(WrapperTest64, InsertLookup)
{
    constexpr auto num_lookup_streams = 4;
    constexpr uint64_t cache_size = 1024;
    constexpr uint64_t row_elements = 8;
    constexpr uint64_t num_keys = 1;
    constexpr uint64_t num_embeddings = 16;
    constexpr auto data_type = nve::DATATYPE_FP32;
    constexpr uint64_t embedding_width_in_bytes = row_elements * 4;
    AllocateBuffers(num_keys, num_embeddings, embedding_width_in_bytes);
    m_cache_wrapper.Init(cache_size, row_elements, data_type, std::vector<cudaStream_t>(m_streams.begin(), m_streams.begin()+num_lookup_streams), m_streams[num_lookup_streams]);

    std::vector<KeyT> h_keys(num_keys);
    h_keys[0] = 1;

    CHECK_CUDA_ERROR(cudaMemsetAsync(m_d_hit_mask, 0, hit_mask_sz_in_bytes, m_cache_wrapper.GetLookupStreams()[0]));
    CHECK_CUDA_ERROR(cudaMemcpy(d_keys, h_keys.data(), sizeof(KeyT)*num_keys, cudaMemcpyDefault));
    m_cache_wrapper.Lookup(0, d_keys, 1, m_d_values, embedding_width_in_bytes, m_d_hit_mask);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_h_hit_mask, m_d_hit_mask, hit_mask_sz_in_bytes, cudaMemcpyDefault, m_cache_wrapper.GetLookupStreams()[0]));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_cache_wrapper.GetLookupStreams()[0]));
    EXPECT_EQ(m_h_hit_mask[0], 0);

    for (uint32_t i = 0; i < num_keys; i++)
    {
        auto key = h_keys[i];
        uint64_t mask_entry = i / 64;
        uint64_t bit_entry = i % 64;
        bool cache_hit = ((m_h_hit_mask[mask_entry] & (1llu << bit_entry)) > 0);
        if (!cache_hit)
        {
            auto vec = GenerateVector(key, 0, row_elements);
            memcpy(m_h_values + i * embedding_width_in_bytes, vec.data(), embedding_width_in_bytes);
        }
    }
    nve::DefaultHistogram histogram(h_keys.data(), num_keys, m_h_values, embedding_width_in_bytes, false);
    m_cache_wrapper.Insert(histogram.GetKeys(), histogram.GetNumBins(), histogram.GetPriority(), histogram.GetData());
    CHECK_CUDA_ERROR(cudaMemsetAsync(m_d_hit_mask, 0, hit_mask_sz_in_bytes, m_cache_wrapper.GetLookupStreams()[0]));
    m_cache_wrapper.Lookup(0, d_keys, 1, m_d_values, embedding_width_in_bytes, m_d_hit_mask);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_h_hit_mask, m_d_hit_mask, hit_mask_sz_in_bytes, cudaMemcpyDefault, m_cache_wrapper.GetLookupStreams()[0]));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_cache_wrapper.GetLookupStreams()[0]));
    EXPECT_EQ(m_h_hit_mask[0], 1);

}

TEST_P(WrapperTest64, LookupInsertUpdate)
{
    auto params = GetParam();
    constexpr auto num_lookup_streams = 4;
    uint64_t cache_size = params.cache_size;
    uint64_t row_elements = params.row_elements;
    uint64_t num_keys = params.num_keys;
    uint64_t num_embeddings = params.num_embeddings;
    auto update_data_type = params.update_data_type;
    constexpr auto data_type = nve::DATATYPE_FP32;
    uint64_t embedding_width_in_bytes = row_elements * 4;

    AllocateBuffers(num_keys, num_embeddings, embedding_width_in_bytes);
    m_cache_wrapper.Init(cache_size, row_elements, data_type, std::vector<cudaStream_t>(m_streams.begin(), m_streams.begin()+num_lookup_streams), m_streams[num_lookup_streams]);

    std::vector<KeyT> h_keys(num_keys);
    for (uint32_t i = 0; i < num_keys; i++)
    {
        h_keys[i] = i + 1;
    }

    CHECK_CUDA_ERROR(cudaMemcpy(d_keys, h_keys.data(), sizeof(KeyT)*num_keys, cudaMemcpyDefault));    
    CHECK_CUDA_ERROR(cudaMemsetAsync(m_d_hit_mask, 0, hit_mask_sz_in_bytes, m_cache_wrapper.GetLookupStreams()[0]));
    m_cache_wrapper.Lookup(0, d_keys, num_keys, m_d_values, embedding_width_in_bytes, m_d_hit_mask);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_h_hit_mask, m_d_hit_mask, hit_mask_sz_in_bytes, cudaMemcpyDefault, m_cache_wrapper.GetLookupStreams()[0]));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_cache_wrapper.GetLookupStreams()[0]));
    EXPECT_EQ(m_h_hit_mask[0], 0);
    uint64_t updateVersion = 0;
    for (uint32_t i = 0; i < num_keys; i++)
    {
        auto key = h_keys[i];
        uint64_t mask_entry = i / 64;
        uint64_t bit_entry = i % 64;
        bool cache_hit = ((m_h_hit_mask[mask_entry] & (1llu << bit_entry)) > 0);
        if (!cache_hit)
        {
            auto vec = GenerateVector(key, updateVersion, row_elements);
            memcpy(m_h_values + i * embedding_width_in_bytes, vec.data(), embedding_width_in_bytes);
        }
    }
    nve::DefaultHistogram histogram(h_keys.data(), num_keys, m_h_values, embedding_width_in_bytes, false);
    m_cache_wrapper.Insert(histogram.GetKeys(), histogram.GetNumBins(), histogram.GetPriority(), histogram.GetData());
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_cache_wrapper.GetModifyStream()));
    CHECK_CUDA_ERROR(cudaMemsetAsync(m_d_hit_mask, 0, hit_mask_sz_in_bytes, m_cache_wrapper.GetLookupStreams()[0]));
    m_cache_wrapper.Lookup(0, d_keys, num_keys, m_d_values, embedding_width_in_bytes, m_d_hit_mask);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_h_hit_mask, m_d_hit_mask, hit_mask_sz_in_bytes, cudaMemcpyDefault, m_cache_wrapper.GetLookupStreams()[0]));
    memset(m_h_values, 0, embedding_width_in_bytes*num_keys);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_h_values, m_d_values, embedding_width_in_bytes*num_keys, cudaMemcpyDefault, m_cache_wrapper.GetLookupStreams()[0]));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_cache_wrapper.GetLookupStreams()[0]));

    EXPECT_EQ(m_h_hit_mask[0], (1 << num_keys) - 1);

    for (uint32_t i = 0; i < num_keys; i++)
    {
        std::vector<float> test((float*)m_h_values+i*row_elements, (float*)m_h_values+(i+1)*row_elements);
        std::vector<float> ref = GenerateVector(h_keys[i], updateVersion, row_elements);
	    EXPECT_EQ(test, ref);
    }
    // reset h values
    memset(m_h_values, 0, num_embeddings*embedding_width_in_bytes);
    // update
    updateVersion++;
    for (uint32_t i = 0; i < num_keys; i++)
    {
        auto vec = UpdateVersion(row_elements);
        if (update_data_type == nve::DATATYPE_INT8_SCALED) {
            int8_t* dst = reinterpret_cast<int8_t*>(m_h_values) + i * (row_elements + 4);
            Quantize<int8_t, float>(vec, dst);
        } else {
            memcpy(m_h_values + i * embedding_width_in_bytes, vec.data(), embedding_width_in_bytes);
        }
    }
    auto stride = (update_data_type == nve::DATATYPE_INT8_SCALED) ? (row_elements + 4) : embedding_width_in_bytes;
    m_cache_wrapper.Accumulate(h_keys.data(), num_keys, m_h_values, stride, update_data_type);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_cache_wrapper.GetModifyStream()));
    CHECK_CUDA_ERROR(cudaMemsetAsync(m_d_hit_mask, 0, hit_mask_sz_in_bytes, m_cache_wrapper.GetLookupStreams()[0]));
    m_cache_wrapper.Lookup(0, d_keys, num_keys, m_d_values, embedding_width_in_bytes, m_d_hit_mask);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_h_hit_mask, m_d_hit_mask, hit_mask_sz_in_bytes, cudaMemcpyDefault, m_cache_wrapper.GetLookupStreams()[0]));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(m_h_values, m_d_values, embedding_width_in_bytes*num_keys, cudaMemcpyDefault, m_cache_wrapper.GetLookupStreams()[0]));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(m_cache_wrapper.GetLookupStreams()[0]));
    EXPECT_EQ(m_h_hit_mask[0], (1 << num_keys) - 1);
    for (uint32_t i = 0; i < num_keys; i++)
    {
        std::vector<float> test((float*)m_h_values+i*row_elements, (float*)m_h_values+(i+1)*row_elements);
        std::vector<float> ref = GenerateVector(h_keys[i], updateVersion, row_elements);
        EXPECT_EQ(test, ref);
    }   
}

auto lookup_test_values = testing::Values(
    LookupInsertUpdateTestParams({1024,16,2,16,nve::DATATYPE_INT8_SCALED}),
    LookupInsertUpdateTestParams({1024,16,7,32,nve::DATATYPE_INT8_SCALED}),
    LookupInsertUpdateTestParams({1024,16,2,16,nve::DATATYPE_FP32}),
    LookupInsertUpdateTestParams({1024,16,7,32,nve::DATATYPE_FP32}));

INSTANTIATE_TEST_SUITE_P(LookupInsertUpdate, WrapperTest64, lookup_test_values);
