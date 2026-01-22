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

#include <gtest/gtest.h>
#include <common.hpp>
#include <cstring>
#include <linear_host_table.hpp>
#include <default_allocator.hpp>
#include <vector>

namespace nve {

#define TEST_KEY (1337)
#define TEST_ROW_SIZE (128)

struct LinearHostTableTestParams {
    int64_t row_size_bytes;
    int64_t num_rows;
    DataType_t value_dtype;
};

template <typename KeyType>
class LinearHostTableTest : public testing::TestWithParam<LinearHostTableTestParams> {
public:
    using TableType = LinearHostTable<KeyType>;
    using DataType = float;

    LinearHostTableTest() : h_table_(nullptr) {
        Init();
    }

    ~LinearHostTableTest() {
        if (h_table_) {
            NVE_CHECK_(cudaFreeHost(h_table_));
        }
    }

    void test_find() {
        const auto& params = GetParam();
        const size_t data_elements = static_cast<size_t>(params.row_size_bytes) / sizeof(DataType);
        std::vector<DataType> h_output(data_elements, static_cast<DataType>(0.0f));
        KeyType h_keys[] = {TEST_KEY};
        find(h_keys, 1, h_output.data(), params.row_size_bytes);

        // validate results
        for (size_t i = 0; i < data_elements; i++) {
            EXPECT_FLOAT_EQ(h_table_[static_cast<size_t>(TEST_KEY) * data_elements + i], h_output[i]);
        }
    }

    void test_update() {
        const auto& params = GetParam();
        const size_t data_elements = static_cast<size_t>(params.row_size_bytes) / sizeof(DataType);
        std::vector<DataType> h_data(data_elements);
        for (size_t i = 0; i < data_elements; i++) {
            h_data[i] = static_cast<DataType>(-i);
        }
        update(TEST_KEY, h_data.data(), params.row_size_bytes);

        std::vector<DataType> h_output(data_elements, static_cast<DataType>(0.0f));
        KeyType h_keys[] = {TEST_KEY};
        find(h_keys, 1, h_output.data(), params.row_size_bytes);

        // validate results
        for (size_t i = 0; i < data_elements; i++) {
            EXPECT_FLOAT_EQ(h_data[i], h_output[i]);
        }
    }

    void test_update_accumulate() {
        const auto& params = GetParam();
        const size_t data_elements = static_cast<size_t>(params.row_size_bytes) / sizeof(DataType);
        std::vector<DataType> h_data(data_elements);
        for (size_t i = 0; i < data_elements; i++) {
            h_data[i] = static_cast<DataType>(i);
        }
        // make a copy of original data
        std::vector<DataType> h_data_copy(h_table_ + TEST_KEY * data_elements, h_table_ + (TEST_KEY + 1) *  data_elements);
        update_accumulate(TEST_KEY, h_data.data(), params.row_size_bytes, params.value_dtype);

        std::vector<DataType> h_output(data_elements, static_cast<DataType>(0.0f));
        KeyType h_keys[] = {TEST_KEY};
        find(h_keys, 1, h_output.data(), params.row_size_bytes);

        for (size_t i = 0; i < data_elements; i++) {
            EXPECT_FLOAT_EQ(h_data_copy[i] + h_data[i], h_output[i]); // Original value + accumulated value
        }
    }

    table_ptr_t tb_;
    context_ptr_t ctx_;
    DataType* h_table_;

private:
    void Init() {
        const auto& params = GetParam();
        LinearHostTableConfig cfg;
        cfg.value_dtype = params.value_dtype;
        cfg.max_threads = 64;
        cfg.max_value_size = params.row_size_bytes;

        const size_t table_size = static_cast<size_t>(params.row_size_bytes * params.num_rows);
        NVE_CHECK_(cudaMallocHost(&h_table_, table_size));
        const int64_t table_elements = table_size / sizeof(DataType);
        for (int64_t i = 0; i < table_elements; i++) {
            h_table_[i] = static_cast<DataType>(10000 + i);
        }
        cfg.emb_table = h_table_;

        tb_ = std::make_shared<TableType>(cfg);
        ctx_ = tb_->create_execution_context(0, 0, nullptr, nullptr);
    }

    void find(KeyType* h_keys, int64_t num_keys, DataType* h_data, int64_t row_size) {
        std::vector<max_bitmask_repr_t> hit_mask(((static_cast<size_t>(num_keys) + 63) / 64), 0);
        tb_->find(ctx_, num_keys, h_keys, hit_mask.data(), row_size, h_data, nullptr);
    }

    void update(KeyType h_key, const DataType* h_data, int64_t row_size) {
        tb_->update(ctx_, 1, &h_key, row_size, row_size, h_data);
    }

    void update_accumulate(KeyType h_key, const DataType* h_data, int64_t row_size, DataType_t dtype) {
        tb_->update_accumulate(ctx_, 1, &h_key, row_size, row_size, h_data, dtype);
    }
};

// Define the test parameters
const LinearHostTableTestParams test_params[] = {
    // row_size_bytes, num_rows, value_dtype
    { 128, 10000, DataType_t::Float32}, 
};

using LHTFixture_INT32_T = LinearHostTableTest<int32_t>;
using LHTFixture_INT64_T = LinearHostTableTest<int64_t>;

#define TEST_FORMAT(test_name, test_func) \
TEST_P(LHTFixture_INT32_T, test_name)     \
{                                         \
    test_func();                          \
}                                         \
TEST_P(LHTFixture_INT64_T, test_name)     \
{                                         \
    test_func();                          \
}                                         \

TEST_FORMAT(find, test_find);
TEST_FORMAT(update, test_update);
TEST_FORMAT(update_accumulate, test_update_accumulate);
#undef TEST_FORMAT

// Instantiate the tests with both type and value parameters
INSTANTIATE_TEST_SUITE_P(
    LinearHostTableTestInt32,
    LHTFixture_INT32_T,
    testing::ValuesIn(test_params));

INSTANTIATE_TEST_SUITE_P(
    LinearHostTableTestInt64,
    LHTFixture_INT64_T,
    testing::ValuesIn(test_params));

} // namespace nve 
