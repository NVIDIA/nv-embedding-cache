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
#include "samples/wrapper_sample/cuda_utils.cuh"
#include "samples/wrapper_sample/communicator.h"
#include <datagen.h>
#include <memory>
#include <algorithm>
#include <thread>
#include <numeric>
#include "../common/buffer.h"

template<typename KeyT_>
class DedupTest : public testing::Test
{
public:
    using KeyT = KeyT_;
    DedupTest()
    {
        CHECK_CUDA_ERROR(cudaStreamCreate(&m_stream));
    }

    void AllocateDedupBuffers(size_t num_keys)
    {
        m_num_keys = num_keys;
        m_keys = std::make_shared<Buffer<KeyT>>(num_keys*sizeof(KeyT));
        m_unique_out = std::make_shared<Buffer<KeyT>>(num_keys*sizeof(KeyT));
        m_counts_out = std::make_shared<Buffer<KeyT>>(num_keys*sizeof(KeyT));
        m_num_runs_out = std::make_shared<Buffer<KeyT>>(sizeof(KeyT));
        m_inverse_buffer = std::make_shared<Buffer<KeyT>>(num_keys*sizeof(KeyT));
        m_offsets = std::make_shared<Buffer<KeyT>>(num_keys*sizeof(KeyT));
        m_deduper = std::make_shared<Deduper<KeyT>>(num_keys);
    }

    void AllocateExpandBuffers(size_t row_size_in_bytes, size_t num_rows, size_t num_keys)
    {
        m_dense_buffer = std::make_shared<Buffer<int8_t>>(num_rows * row_size_in_bytes);
        m_expand_buffer = std::make_shared<Buffer<int8_t>>(num_keys * row_size_in_bytes);
    }

    void AllocateAccumulateBuffers(size_t row_size_in_bytes, size_t num_keys, size_t num_rows)
    {
        m_dense_buffer = std::make_shared<Buffer<int8_t>>(num_rows * row_size_in_bytes);
        m_accumulate_buffer = std::make_shared<Buffer<int8_t>>(num_keys * row_size_in_bytes);
    }

    void GenerateInputs(size_t batch, size_t hotness, const size_t N, const float alpha)
    {
        size_t num_keys = batch * hotness;
        ASSERT_LE(num_keys , m_num_keys);
        // generate inputs
        auto sg = getSampleGenerator<KeyT>(alpha, N, hotness, 283982);
        for (auto b = 0; b < batch; b++)
        {
            auto sample = sg->getCategoryIndices();
            std::copy(sample.begin(), sample.end(), m_keys->ph + b*hotness);
        }

        CHECK_CUDA_ERROR(cudaMemcpyAsync(m_keys->pd, m_keys->ph, sizeof(KeyT)*num_keys, cudaMemcpyDefault, m_stream));
    }

    void CheckDedup(size_t num_keys)
    {
        //copyout
        m_unique_out->DtoH(m_stream);
        m_counts_out->DtoH(m_stream);
        m_inverse_buffer->DtoH(m_stream);
        m_offsets->DtoH(m_stream);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(m_stream));
        std::vector<KeyT> test_keys(num_keys);
        auto num_runs = *(m_num_runs_out->ph);
        EXPECT_EQ(m_offsets->ph[num_runs - 1] + m_counts_out->ph[num_runs - 1], num_keys);
        for (size_t i = 0; i < num_runs; i++)
        {
            auto key = m_unique_out->ph[i];
            auto key_count = m_counts_out->ph[i];
            for (size_t j = 0; j < key_count; j++)
            {
                test_keys[m_inverse_buffer->ph[m_offsets->ph[i] + j]] = key;
            }
        }

        EXPECT_TRUE(std::equal(test_keys.begin(), test_keys.end(), m_keys->ph));
    }

    void CheckUnpack(size_t num_keys, size_t row_size_in_bytes)
    {
        m_unique_out->DtoH(m_stream);
        m_counts_out->DtoH(m_stream);
        m_inverse_buffer->DtoH(m_stream);
        m_offsets->DtoH(m_stream);
        m_expand_buffer->DtoH(m_stream);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(m_stream));
        uint64_t keys_expanded = 0;
        auto num_runs = m_num_runs_out->ph[0];
        for (size_t i = 0; i < num_runs; i++)
        {
            auto key_count = m_counts_out->ph[i];
            for (size_t j = 0; j < key_count; j++)
            {
                auto inv_loc = m_inverse_buffer->ph[m_offsets->ph[i]+j];
                EXPECT_TRUE(std::equal(m_expand_buffer->ph + inv_loc * row_size_in_bytes, m_expand_buffer->ph + (inv_loc + 1) * row_size_in_bytes, m_dense_buffer->ph + i * row_size_in_bytes));
                keys_expanded++;
            }
        }
        EXPECT_EQ(keys_expanded, num_keys);

    }

    template <typename DataType>
    void CheckAccumulate(size_t num_keys, size_t row_size_in_bytes)
    {
        m_unique_out->DtoH(m_stream);
        m_counts_out->DtoH(m_stream);
        m_inverse_buffer->DtoH(m_stream);
        m_offsets->DtoH(m_stream);
        m_accumulate_buffer->DtoH(m_stream);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(m_stream));
        uint64_t keys_expanded = 0;
        auto num_runs = m_num_runs_out->ph[0];
        auto row_elements = row_size_in_bytes / sizeof(DataType);

        for (size_t i = 0; i < num_runs; i++)
        {
            std::vector<DataType> acc_res(row_elements, 0.0);
            auto key_count = m_counts_out->ph[i];
            for (size_t j = 0; j < key_count; j++)
            {
                auto inv_loc = m_inverse_buffer->ph[m_offsets->ph[i]+j];
                auto src_row = reinterpret_cast<DataType*>(m_dense_buffer->ph + inv_loc * row_size_in_bytes);
                for (size_t k = 0; k < row_elements; k++) {
                    acc_res[k] += src_row[k];
                }

                keys_expanded++;
            }
            DataType* ref = &acc_res[0];
            DataType* d_res = reinterpret_cast<DataType*>(m_accumulate_buffer->ph + i * row_size_in_bytes);
            EXPECT_TRUE(std::equal(ref, ref + row_elements, d_res)); 
        }
        EXPECT_EQ(keys_expanded, num_keys);

    }

    std::vector<uint32_t> GenerateVector(uint64_t seed, uint64_t row_size_in_bytes)
    {
        auto row_elements = row_size_in_bytes / sizeof(uint32_t);
        std::vector<uint32_t> ret(row_elements);
        for (uint32_t i = 0; i < row_elements; i++)
        {
            ret[i] = seed;
        }
        return ret;
    }

    void FillDenseBuffer(uint64_t row_size_in_bytes, uint64_t num_rows)
    {
        for (uint64_t i = 0; i < num_rows; i++)
        {
            auto vec = GenerateVector(i, row_size_in_bytes);
            std::copy(vec.begin(), vec.end(), m_dense_buffer->ph + i * row_size_in_bytes);
        }

        m_dense_buffer->HtoD(m_stream);
    }

    template<typename DataType>
    void GenerateGradients(uint64_t row_size_in_bytes, uint64_t num_rows)
    {
        auto row_elements = row_size_in_bytes / sizeof(DataType);
        for (uint64_t i = 0; i < num_rows; i++)
        {
            DataType* dense_row = reinterpret_cast<DataType*>(m_dense_buffer->ph + i * row_size_in_bytes);
            for (uint64_t j = 0; j < row_elements; j++)
            {
                int32_t nom = rand() % 100;
                int32_t denom = 1 + rand() % 100;
                DataType val = DataType(nom) / DataType(denom);
                dense_row[j] = ((rand() % 2) == 0) ? val : -val;
            }
        }

        m_dense_buffer->HtoD(m_stream);
    }

    std::shared_ptr<Buffer<KeyT>> m_keys = nullptr;
    std::shared_ptr<Buffer<KeyT>> m_unique_out = nullptr;
    std::shared_ptr<Buffer<KeyT>> m_counts_out = nullptr;
    std::shared_ptr<Buffer<KeyT>> m_num_runs_out = nullptr;
    std::shared_ptr<Buffer<KeyT>> m_inverse_buffer = nullptr;
    std::shared_ptr<Buffer<KeyT>> m_offsets = nullptr;
    std::shared_ptr<Buffer<int8_t>> m_expand_buffer = nullptr;
    std::shared_ptr<Buffer<int8_t>> m_dense_buffer = nullptr;
    std::shared_ptr<Buffer<int8_t>> m_accumulate_buffer = nullptr;

    std::shared_ptr<Deduper<KeyT>> m_deduper = nullptr;
    cudaStream_t m_stream;
    size_t m_num_keys;
};

using TestDedup64 = DedupTest<uint64_t>;
TEST_F(TestDedup64, Basic)
{
    constexpr float alpha = 1.05;
    constexpr uint64_t N = (2llu << 30);
    constexpr uint64_t batch = 64*1024;
    constexpr uint64_t hot = 4;
    constexpr uint64_t num_keys = batch * hot;
    AllocateDedupBuffers(num_keys);
    GenerateInputs(batch, hot, N, alpha);
    m_deduper->Dedup(m_keys->pd, num_keys, m_unique_out->pd, m_counts_out->pd, m_num_runs_out->ph, m_inverse_buffer->pd, m_offsets->pd, m_stream);
    CheckDedup(num_keys);  
}

TEST_F(TestDedup64, Expand)
{
    constexpr float alpha = 1.05;
    constexpr uint64_t N = (2llu << 30);
    constexpr uint64_t batch = 64*1024;
    constexpr uint64_t hot = 4;
    constexpr uint64_t num_keys = batch * hot;
    constexpr uint64_t row_size_in_bytes = 512;
    AllocateDedupBuffers(num_keys);
    
    GenerateInputs(batch, hot, N, alpha);
    m_deduper->Dedup(m_keys->pd, num_keys, m_unique_out->pd, m_counts_out->pd, m_num_runs_out->ph, m_inverse_buffer->pd, m_offsets->pd, m_stream);
    AllocateExpandBuffers(row_size_in_bytes, m_num_runs_out->ph[0], num_keys);
    FillDenseBuffer(row_size_in_bytes, *(m_num_runs_out->ph));
    UnpackDedup<KeyT>(m_inverse_buffer->pd, m_offsets->pd, m_counts_out->pd, m_dense_buffer->pd, m_expand_buffer->pd, m_num_runs_out->ph[0], row_size_in_bytes, m_stream);
    CheckUnpack(num_keys, row_size_in_bytes);
}

TEST_F(TestDedup64, DedupAccumulate)
{
    constexpr float alpha = 1.05;
    constexpr uint64_t N = 2*1024*1024*1024llu;
    constexpr uint64_t batch = 64*1024;
    constexpr uint64_t hot = 4;
    constexpr uint64_t num_keys = batch * hot;
    constexpr uint64_t row_size_in_bytes = 512;
    AllocateDedupBuffers(num_keys);
    
    GenerateInputs(batch, hot, N, alpha);
    m_deduper->Dedup(m_keys->pd, num_keys, m_unique_out->pd, m_counts_out->pd, m_num_runs_out->ph, m_inverse_buffer->pd, m_offsets->pd, m_stream);
    
    AllocateAccumulateBuffers(row_size_in_bytes, m_num_runs_out->ph[0], num_keys);

    GenerateGradients<float>(row_size_in_bytes, num_keys);
    CallGradientDedupKernelVecTypeSubwarp<32, float, KeyT>(
        reinterpret_cast<const float*>(m_dense_buffer->pd), 
        reinterpret_cast<float*>(m_accumulate_buffer->pd),
        reinterpret_cast<const KeyT*>(m_unique_out->pd),
        reinterpret_cast<const KeyT*>(m_inverse_buffer->pd),
        reinterpret_cast<const KeyT*>(m_counts_out->pd),
        reinterpret_cast<const KeyT*>(m_offsets->pd),
        m_num_runs_out->ph[0],                                                        
        row_size_in_bytes / sizeof(float),
        row_size_in_bytes / sizeof(float),
        row_size_in_bytes / sizeof(float),
        m_stream);

    CheckAccumulate<float>(num_keys, row_size_in_bytes);

    GenerateGradients<__half>(row_size_in_bytes, num_keys);
    CallGradientDedupKernelVecTypeSubwarp<32, __half, KeyT>(
        reinterpret_cast<const __half*>(m_dense_buffer->pd), 
        reinterpret_cast<__half*>(m_accumulate_buffer->pd),
        reinterpret_cast<const KeyT*>(m_unique_out->pd),
        reinterpret_cast<const KeyT*>(m_inverse_buffer->pd),
        reinterpret_cast<const KeyT*>(m_counts_out->pd),
        reinterpret_cast<const KeyT*>(m_offsets->pd),
        m_num_runs_out->ph[0],                                                        
        row_size_in_bytes / sizeof(__half),
        row_size_in_bytes / sizeof(__half),
        row_size_in_bytes / sizeof(__half),
        m_stream);

    CheckAccumulate<__half>(num_keys, row_size_in_bytes);
}

TEST(CommunicatorTest, basic)
{
    constexpr uint32_t data_offset = 1;
    int device_count = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
    if (device_count == 1)
    {
	    GTEST_SKIP();
    }
    std::vector<std::shared_ptr<std::thread>> threads;
    std::vector<cudaStream_t> streams;
    std::vector<Buffer<uint64_t>*> combined_buffs;
    std::vector<Buffer<uint64_t>*> out_buffs;
    for (uint32_t dev = 0; dev < device_count; dev++)
    {
        CHECK_CUDA_ERROR(cudaSetDevice(dev));
        cudaStream_t stream;
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
        Buffer<uint64_t>* combined_buff = new Buffer<uint64_t>(device_count*sizeof(uint64_t));
        memset(combined_buff->ph, 0, combined_buff->m_size);
        combined_buff->HtoD(stream);
        Buffer<uint64_t>* out_buff = new Buffer<uint64_t>(sizeof(uint64_t));
        out_buff->ph[0] = dev + data_offset;
        out_buff->HtoD(stream);
        threads.push_back(std::make_shared<std::thread>([=] {
            CHECK_CUDA_ERROR(cudaSetDevice(dev));
            Communicator comm(device_count, dev, "__temp__", dev);
            comm.AllToAll((const int8_t*)out_buff->pd, (int8_t*)combined_buff->pd, sizeof(uint64_t), stream);
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        }));
        streams.push_back(stream);
        combined_buffs.push_back(combined_buff);
        out_buffs.push_back(out_buff);
    }
    for (auto t : threads)
    {
        t->join();
    }

    std::vector<uint64_t> ref(device_count);
    // fill reference buffer with data_offset , ... , data_offset + device_count
    std::iota(ref.begin(), ref.end(), data_offset);
    for (uint32_t dev = 0; dev < device_count; dev++)
    {
        CHECK_CUDA_ERROR(cudaSetDevice(dev));
        combined_buffs[dev]->DtoH(streams[dev]);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[dev]));
        EXPECT_TRUE(std::equal(ref.begin(), ref.end(), combined_buffs[dev]->ph));
        delete combined_buffs[dev];
        delete out_buffs[dev];
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[dev]));
    }
}
