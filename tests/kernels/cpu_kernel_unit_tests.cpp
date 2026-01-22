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
#include "cpu_ops/cpu_gather.h"
#include "cpu_ops/cpu_update.h"
#include "include/thread_pool.hpp"
#include "include/nve_types.hpp"

#include <random>
#include <cstdlib>

namespace nve {

enum KeyType {  
    INT32,
    INT64
};

struct GatherTestParams {
    GatherTestParams(int64_t n_, size_t row_size_in_bytes_, KeyType key_type_, uint64_t num_threads_, uint32_t seed_) :
        n(n_), row_size_in_bytes(row_size_in_bytes_), key_type(key_type_), num_threads(num_threads_), seed(seed_) {}
    int64_t n;
    size_t row_size_in_bytes;
    KeyType key_type;
    uint64_t num_threads;
    uint32_t seed;
};

class CpuKernelGatherUvmTest : public ::testing::TestWithParam<GatherTestParams> {
protected:
    void SetUp() override {
        thread_pool_ = default_thread_pool();         
    }

    template<typename IndexT>
    void run_test() {
        const auto& params = GetParam();
        const uint64_t n = static_cast<uint64_t>(params.n);
        const auto row_size_in_bytes = params.row_size_in_bytes;
    
        // Allocate memory for test data
        std::vector<IndexT> keys(n);
        std::vector<max_bitmask_repr_t> hit_mask(((n + 63) / 64), 0);
        std::vector<int8_t> values(n * row_size_in_bytes, 0);
        std::vector<int8_t> uvm_table(n * row_size_in_bytes);
    
        // Initialize test data
        std::mt19937 rng(params.seed);
        std::uniform_int_distribution<int> dist(0, 1);  // For 50/50 probability
        for (uint64_t i = 0; i < n; ++i) {
            keys[i] = static_cast<IndexT>(i);  // Use sequential keys
            // Initialize UVM table with test data
            for (uint64_t j = 0; j < row_size_in_bytes; ++j) {
                uvm_table[i * row_size_in_bytes + j] = static_cast<int8_t>((i + j) % 128);
            }
            // Randomly set hit mask bits with 50% probability
            if (dist(rng)) {
                hit_mask[i / 64] |= (1ULL << (i % 64));
            }
        }
        // Call the gather function
        // make a copy of the hit mask so we can check the rows that were gathered
        std::vector<max_bitmask_repr_t> hit_mask_copy(hit_mask);

        NVE_CHECK_(cpu_kernel_gather<IndexT>(
            thread_pool_,
            n,
            keys.data(),
            hit_mask.data(),
            row_size_in_bytes,
            values.data(),
            uvm_table.data(),
            row_size_in_bytes,
            params.num_threads
        ) == 0);
    
        // check that hit mask is all 1s
        for (uint64_t i = 0; i < (n / 64); ++i) {
            EXPECT_EQ(hit_mask[i], 0xffffffffffffffffULL);
        }

        uint64_t rem = n - ((n / 64) * 64); 
        if (rem > 0) {
            EXPECT_EQ(hit_mask[n / 64], ((1ULL << (rem)) - 1));
        }

        // Verify results
        for (uint64_t i = 0; i < n; ++i) {
            // we only filled the previous table misses
            if ((hit_mask_copy[i / 64] & (1ULL << (i % 64))) == 0) {
                for (uint64_t j = 0; j < row_size_in_bytes; ++j) {
                    EXPECT_EQ(values[i * row_size_in_bytes + j], 
                            uvm_table[static_cast<size_t>(keys[i]) * row_size_in_bytes + j])
                    << "Mismatch at position (" << i << ", " << j << "), key: " << keys[i] << ")";
                }
            }
        }
    }

    std::shared_ptr<nve::ThreadPool> thread_pool_;
    
};

TEST_P(CpuKernelGatherUvmTest, GatherTest) {
    auto params = GetParam();
    switch (params.key_type) {
        case KeyType::INT32:
            run_test<int32_t>();
            break;
        case KeyType::INT64:
            run_test<int64_t>();
            break;
    }
}

static std::vector<int64_t> cases_n = {1, 1024, 1000, 9758};
static std::vector<size_t> cases_row_size_in_bytes = {1, 17, 32, 52, 128};
static std::vector<KeyType> cases_key_type = {KeyType::INT32, KeyType::INT64};
static std::vector<uint64_t> cases_num_threads = {1, 3, 4, 7, 12, 64};

static std::vector<GatherTestParams> genCase(const std::vector<int64_t>& _n, const std::vector<size_t>& _row_size_in_bytes, const std::vector<KeyType>& _key_type, const std::vector<uint64_t>& _num_threads) {
    std::vector<GatherTestParams> ret;
    
    for (uint32_t n = 0; n < _n.size(); n++)
    {
        for (uint32_t row_size_in_bytes = 0; row_size_in_bytes < _row_size_in_bytes.size(); row_size_in_bytes++)
        {
            for (uint32_t key_type = 0; key_type < _key_type.size(); key_type++)
            {
                for (uint32_t num_threads = 0; num_threads < _num_threads.size(); num_threads++)
                {
                    ret.emplace_back(_n[n], _row_size_in_bytes[row_size_in_bytes], _key_type[key_type], _num_threads[num_threads], n * 1024 + row_size_in_bytes * 234 + num_threads * 2);
                }
            }
        }
    }
    return ret;
}

INSTANTIATE_TEST_SUITE_P(
    GatherTests,
    CpuKernelGatherUvmTest,
    ::testing::ValuesIn(genCase(cases_n, cases_row_size_in_bytes, cases_key_type, cases_num_threads))
    );

class CpuKernelUpdateUvmTest : public ::testing::TestWithParam<GatherTestParams> {
protected:
    void SetUp() override {
        thread_pool_ = default_thread_pool();         
    }

    template<typename IndexT>
    void run_test() {
        const auto& params = GetParam();
        
        const uint64_t n = static_cast<uint64_t>(params.n);
        const auto row_size_in_bytes = params.row_size_in_bytes;
    
        // Allocate memory for test data
        std::vector<IndexT> keys(n);
        std::vector<int8_t> values(n * row_size_in_bytes, 0);
        std::vector<int8_t> uvm_table(n * row_size_in_bytes);
    
        // Initialize test data
        for (uint64_t i = 0; i < n; ++i) {
            keys[i] = static_cast<IndexT>(i);  // Use sequential keys
            // Initialize UVM table with test data
            for (uint64_t j = 0; j < row_size_in_bytes; ++j) {
                uvm_table[i * row_size_in_bytes + j] = static_cast<int8_t>((i + j) % 128);
                values[i * row_size_in_bytes + j] = static_cast<int8_t>((i * j) % 128); // Different pattern for updates
            }
        }

        // Call the update function
        NVE_CHECK_(cpu_kernel_update<IndexT>(
            thread_pool_,
            n,
            keys.data(),
            row_size_in_bytes,
            values.data(),
            uvm_table.data(),
            row_size_in_bytes,
            params.num_threads
        ) == 0);
    
        // Verify results
        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t j = 0; j < row_size_in_bytes; ++j) {
                EXPECT_EQ(uvm_table[static_cast<size_t>(keys[i]) * row_size_in_bytes + j], 
                        values[i * row_size_in_bytes + j])
                    << "Mismatch at position (" << i << ", " << j << "), key: " << keys[i] << ")";
            }
        }
    }

    std::shared_ptr<nve::ThreadPool> thread_pool_;
};

TEST_P(CpuKernelUpdateUvmTest, UpdateTest) {
    auto params = GetParam();
    switch (params.key_type) {
        case KeyType::INT32:
            run_test<int32_t>();
            break;
        case KeyType::INT64:
            run_test<int64_t>();
            break;
    }
}

INSTANTIATE_TEST_SUITE_P(
    UpdateTests,
    CpuKernelUpdateUvmTest,
    ::testing::ValuesIn(genCase(cases_n, cases_row_size_in_bytes, cases_key_type, cases_num_threads))
);

class CpuKernelUpdateAccumulateUvmTest : public ::testing::TestWithParam<GatherTestParams> {
protected:
    void SetUp() override {
        thread_pool_ = default_thread_pool();         
    }

    template<typename IndexT>
    void run_test() {
        using ValueT = float;
        const auto& params = GetParam();
        if (params.row_size_in_bytes % sizeof(ValueT) != 0) {
            GTEST_SKIP() << "Row size is not a multiple of the value type size";
            return;
        }
        const uint64_t n = static_cast<uint64_t>(params.n);
        const auto row_size_in_bytes = params.row_size_in_bytes;
    
        // Allocate memory for test data
        std::vector<IndexT> keys(n);
        std::vector<ValueT> values(n * (row_size_in_bytes / sizeof(ValueT)), 0);
        std::vector<ValueT> uvm_table(n * (row_size_in_bytes / sizeof(ValueT)));
    
        // Initialize test data
        std::mt19937 rng(params.seed);
        std::uniform_real_distribution<ValueT> dist(-1.0f, 1.0f);
        for (uint64_t i = 0; i < n; ++i) {
            keys[i] = static_cast<IndexT>(i);  // Use sequential keys
            // Initialize UVM table and values with random floats
            for (uint64_t j = 0; j < row_size_in_bytes / sizeof(ValueT); ++j) {
                uvm_table[i * (row_size_in_bytes / sizeof(ValueT)) + j] = dist(rng);
                values[i * (row_size_in_bytes / sizeof(ValueT)) + j] = dist(rng);
            }
        }

        // Make a copy of the UVM table for verification
        std::vector<float> uvm_table_copy = uvm_table;

        // Call the update_accumulate function
        cpu_kernel_update_accumulate_dispatch<IndexT>(
            thread_pool_,
            n,
            keys.data(),
            row_size_in_bytes,
            values.data(),
            reinterpret_cast<int8_t*>(uvm_table.data()),
            row_size_in_bytes,
            nve::DataType_t::Float32,
            params.num_threads
        );
    
        // Verify results
        for (uint64_t i = 0; i < n; ++i) {
            for (uint64_t j = 0; j < row_size_in_bytes / sizeof(ValueT); ++j) {
                ValueT expected = uvm_table_copy[static_cast<size_t>(keys[i]) * (row_size_in_bytes / sizeof(ValueT)) + j] + 
                               values[i * (row_size_in_bytes / sizeof(ValueT)) + j];
                EXPECT_FLOAT_EQ(uvm_table[static_cast<size_t>(keys[i]) * (row_size_in_bytes / sizeof(ValueT)) + j], 
                              expected)
                    << "Mismatch at position (" << i << ", " << j << "), key: " << keys[i] << ")";
            }
        }
    }

    std::shared_ptr<nve::ThreadPool> thread_pool_;
};

TEST_P(CpuKernelUpdateAccumulateUvmTest, UpdateAccumulateTest) {
    auto params = GetParam();
    switch (params.key_type) {
        case KeyType::INT32:
            run_test<int32_t>();
            break;
        case KeyType::INT64:
            run_test<int64_t>();
            break;
    }
}

INSTANTIATE_TEST_SUITE_P(
    UpdateAccumulateTests,
    CpuKernelUpdateAccumulateUvmTest,
    ::testing::ValuesIn(genCase(cases_n, cases_row_size_in_bytes, cases_key_type, cases_num_threads))
);

} // namespace nve