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

#include <buffer_wrapper.hpp>
#include <cmath>
#include <common.hpp>
#include <algorithm>
#include <cstring>
#include <thread>
#include <cuda_support.hpp>
#include <gpu_table.hpp>
#include <default_allocator.hpp>
#include <random>
#include <tuple>
#include <vector>

namespace nve {

#define GT_TEST_DEVICE (0)
#define GT_TEST_KEY (1337)
#define GT_TEST_SORT_GATHER_THRESHOLD (4096)
#define GT_TEST_KERNEL_MODE_INDEX_THERSHOLD (0)

struct GpuTableTestParams {
    size_t cache_size_bytes;
    int64_t row_size_bytes;
    int64_t num_rows;
    int device_id;
    int64_t max_keys;
    bool allocate_uvm_table;
    bool data_storage_on_host;
    bool modify_on_gpu;
};

template <typename KeyType,                 // Type used for keys/indices
          typename OffsetType = KeyType,    // Type used for Offset during lookup of CSR
          typename ValueType = float,       // Type used for the data vectors
          typename OutputType = ValueType,  // Type used for output data vectors (can differ from
                                            // ValueType only when combining multiple rows)
          typename WeightType =
              float>  // Type used for weights used by some combiner types (e.g. weighted sum)
class GpuTableTest : public testing::TestWithParam<GpuTableTestParams> {
 public:
  using TableType = GpuTable<KeyType>;
  GpuTableTest()
      : d_keys_(nullptr), d_data_(nullptr), d_hitmask_(nullptr) {
    Init();
  }

  ~GpuTableTest() {
    if (h_table_) {
      NVE_CHECK_(cudaFreeHost(h_table_));
    }
    auto allocator = GetDefaultAllocator();
    allocator->device_free(d_keys_);
    allocator->device_free(d_data_);
    allocator->device_free(d_hitmask_);
  }

  void test_insert() {
    const auto& params = GetParam();
    const size_t data_elements = static_cast<size_t>(params.row_size_bytes) / sizeof(float);
    std::vector<float> h_data;
    for (size_t i = 0; i < data_elements; i++) {
      h_data.push_back(float(i));
    }
    insert(GT_TEST_KEY, &(h_data[0]), params.row_size_bytes);
    std::vector<float> h_output(data_elements, 0.0f);
    KeyType h_keys[] = {GT_TEST_KEY};
    auto found = find(h_keys, 1, h_output.data(), params.row_size_bytes);

    // validate results
    EXPECT_TRUE(found);
    for (size_t i = 0; i < data_elements; i++) {
      EXPECT_FLOAT_EQ(h_data[i], h_output[i]);
    }
  }

  void test_find_uvm() {
    const auto& params = GetParam();
    if (!params.allocate_uvm_table) {
      return;
    }
    const size_t data_elements = static_cast<size_t>(params.row_size_bytes) / sizeof(float);
    std::vector<float> h_output;
    h_output.resize(data_elements * static_cast<size_t>(params.max_keys));

    KeyType TEST_KEY = static_cast<KeyType>(params.num_rows / 2);
    std::vector<KeyType> h_keys(static_cast<size_t>(params.max_keys), TEST_KEY);
    
    find(h_keys.data(), params.max_keys, h_output.data(), params.row_size_bytes);

    for (size_t i = 0; i < h_keys.size(); i++) {
      for (size_t j = 0; j < data_elements; j++) {
        KeyType key = h_keys[i];
        EXPECT_FLOAT_EQ(h_output[i * data_elements + j], h_table_[static_cast<size_t>(key) * data_elements + j]);
      }
    }
  }

  void test_update() {
    const auto& params = GetParam();
    const size_t data_elements = static_cast<size_t>(params.row_size_bytes) / sizeof(float);
    std::vector<float> h_data;
    std::vector<float> h_data2;
    for (size_t i = 0; i < data_elements; i++) {
      h_data.push_back(static_cast<float>(i));
      h_data2.push_back(static_cast<float>(-i));
    }
    insert(GT_TEST_KEY, &(h_data[0]), params.row_size_bytes);
    update(GT_TEST_KEY, &(h_data2[0]), params.row_size_bytes);

    std::vector<float> h_output(data_elements, 0.0f);
    KeyType h_keys[] = {GT_TEST_KEY};
    auto found = find(h_keys, 1, h_output.data(), params.row_size_bytes);

    // validate results
    EXPECT_TRUE(found);
    for (size_t i = 0; i < data_elements; i++) {
      EXPECT_FLOAT_EQ(h_data2[i], h_output[i]);
    }
  }

  void test_missing_key() {
    const auto& params = GetParam();
    const size_t data_elements = static_cast<size_t>(params.row_size_bytes) / sizeof(float);
    std::vector<float> h_output(data_elements, 0.0f);
    KeyType h_keys[] = {GT_TEST_KEY};
    auto found = find(h_keys, 1, h_output.data(), params.row_size_bytes);

    // validate results
    EXPECT_FALSE(found);
  }

  // Negative keys must not crash insert/find. -1 collides with the INVALID
  // tag sentinel, so its hit/miss is undefined; we only check that no CUDA
  // error escapes.
  void test_negative_key() {
    const auto& params = GetParam();
    const size_t data_elements = static_cast<size_t>(params.row_size_bytes) / sizeof(float);
    std::vector<float> h_data;
    for (size_t i = 0; i < data_elements; i++) {
      h_data.push_back(float(i));
    }

    KeyType neg_key = static_cast<KeyType>(-1);
    insert(neg_key, &(h_data[0]), params.row_size_bytes);

    std::vector<float> h_output(data_elements, 0.0f);
    KeyType h_keys[] = {neg_key};
    (void) find(h_keys, 1, h_output.data(), params.row_size_bytes);

    NVE_CHECK_(cudaPeekAtLastError());
  }

  // Non-sentinel negative keys must round-trip correctly through insert +
  // non-UVM find. GpuTable<KeyType> uses EmbedCacheSA<KeyType, KeyType> so
  // TagT == KeyT and the unsigned-cast hash + reconstruction recover the
  // original key bit-for-bit.
  void test_negative_key_hits() {
    const auto& params = GetParam();
    if (params.allocate_uvm_table) {
      return;  // UVM lookup falls back to table[key] on miss; OOB for negatives.
    }
    const size_t data_elements = static_cast<size_t>(params.row_size_bytes) / sizeof(float);
    std::vector<float> h_data;
    for (size_t i = 0; i < data_elements; i++) {
      h_data.push_back(float(i));
    }

    KeyType neg_key = static_cast<KeyType>(-100);
    insert(neg_key, &(h_data[0]), params.row_size_bytes);

    std::vector<float> h_output(data_elements, 0.0f);
    KeyType h_keys[] = {neg_key};
    bool found = find(h_keys, 1, h_output.data(), params.row_size_bytes);

    EXPECT_TRUE(found);
    for (size_t i = 0; i < data_elements; i++) {
      EXPECT_FLOAT_EQ(h_data[i], h_output[i]);
    }
  }

  void test_erase() {
    const auto& params = GetParam();
    const size_t data_elements = static_cast<size_t>(params.row_size_bytes) / sizeof(float);
    std::vector<float> h_data;
    for (size_t i = 0; i < data_elements; i++) {
      h_data.push_back(float(i));
    }
    insert(GT_TEST_KEY, &(h_data[0]), params.row_size_bytes);
    erase(GT_TEST_KEY);

    std::vector<float> h_output(data_elements, 0.0f);
    KeyType h_keys[] = {GT_TEST_KEY};
    auto found = find(h_keys, 1, h_output.data(), params.row_size_bytes);

    // validate results
    EXPECT_FALSE(found);
  }
  void test_combine() {
    const auto& params = GetParam();
    if (!params.allocate_uvm_table) {
      GTEST_SKIP() << "find_and_combine requires a UVM table";
    }
    const size_t row_elements = static_cast<size_t>(params.row_size_bytes) / sizeof(float);
    insert(0, h_table_, params.row_size_bytes);
    KeyType h_keys[] = {0, 1, 2, 3};
    const size_t num_keys = sizeof(h_keys) / sizeof(h_keys[0]);
    std::vector<float> h_data(row_elements, 0.0f);
    single_combine(h_keys, num_keys, h_data.data(), params.row_size_bytes);

    // validate output
    for (size_t i = 0; i < row_elements; i++) {
      float ref = 0.;
      for (size_t j = 0; j < num_keys; j++) {
        ref += h_table_[(j * row_elements) + i];
      }
      EXPECT_FLOAT_EQ(ref, h_data[i]);
    }
  }

  std::shared_ptr<TableType> tb_;
  context_ptr_t ctx_;
  void* d_keys_;
  void* d_data_;
  void* d_hitmask_;
  float* h_table_;
  bool modify_on_gpu_;
  size_t hit_mask_size_in_bytes_;
 private:
  void Init() {
    const auto& params = GetParam();
    typename nve::GPUTableConfig cfg;
    cfg.device_id = params.device_id;
    cfg.cache_size = params.cache_size_bytes;
    cfg.row_size_in_bytes = params.row_size_bytes;
    cfg.value_dtype = data_type<ValueType>();  // stored value type; find_and_combine dispatches on it
    cfg.uvm_table = nullptr;
    cfg.data_storage_on_host = params.data_storage_on_host;
    modify_on_gpu_ = cfg.modify_on_gpu = params.modify_on_gpu;
    cfg.kernel_mode_value = GT_TEST_SORT_GATHER_THRESHOLD;
    cfg.kernel_mode_type = GT_TEST_KERNEL_MODE_INDEX_THERSHOLD;
    if (params.allocate_uvm_table) {
      const size_t table_size = static_cast<size_t>(params.row_size_bytes * params.num_rows);
      // const int64_t row_elements = params.row_size_bytes / sizeof(float);
      NVE_CHECK_(cudaMallocHost(&h_table_, table_size));
      const int64_t table_elements = table_size / sizeof(float);
      for (int64_t i = 0; i < table_elements; i++) {
        h_table_[i] = float(10000 + i);
      }
      cfg.uvm_table = h_table_;
      cfg.uvm_num_rows = params.num_rows;
    } else {
      h_table_ = nullptr;
    }

    tb_ = std::make_shared<TableType>(cfg);
    ctx_ = tb_->create_execution_context(0, 0, nullptr, nullptr);
    auto allocator = GetDefaultAllocator();
    NVE_CHECK_(allocator->device_allocate(&d_keys_, sizeof(KeyType) * static_cast<size_t>(params.max_keys)));
    NVE_CHECK_(allocator->device_allocate(&d_data_, static_cast<size_t>(params.row_size_bytes) * static_cast<size_t>(params.max_keys)));
    hit_mask_size_in_bytes_ = to_uint(ceil_div(params.max_keys, bitmask64::num_bits)) * sizeof(bitmask64_t);
    NVE_CHECK_(allocator->device_allocate(&d_hitmask_, hit_mask_size_in_bytes_));

    ASSERT_TRUE(d_keys_);
    ASSERT_TRUE(d_data_);
    ASSERT_TRUE(d_hitmask_);
  }

  void insert(KeyType h_key, const ValueType* h_data, int64_t row_size) {
    // copy key and data vector to gpu buffer
    NVE_CHECK_(cudaMemcpy(d_data_, h_data, static_cast<size_t>(row_size), cudaMemcpyDefault));

    auto values_bw = std::make_shared<BufferWrapper<const void>>(
      ctx_, "values", d_data_, static_cast<size_t>(row_size));
    // launch insert
    if (modify_on_gpu_) {
      NVE_CHECK_(cudaMemcpy(d_keys_, &h_key, sizeof(KeyType), cudaMemcpyDefault));
      auto keys_bw = std::make_shared<BufferWrapper<const void>>(
        ctx_, "keys", d_keys_, sizeof(KeyType));
      tb_->insert(ctx_, 1, std::move(keys_bw), row_size, row_size, std::move(values_bw));
    } else {
      auto keys_bw = std::make_shared<BufferWrapper<const void>>(
        ctx_, "keys", &h_key, sizeof(KeyType));
      tb_->insert(ctx_, 1, std::move(keys_bw), row_size, row_size, std::move(values_bw));
    }
    NVE_CHECK_(cudaDeviceSynchronize());
  }

  void update(KeyType h_key, const ValueType* h_data, int64_t row_size) {
    // copy key and data vector to gpu buffer
    NVE_CHECK_(cudaMemcpy(d_data_, h_data, static_cast<size_t>(row_size), cudaMemcpyDefault));

    auto values_bw = std::make_shared<BufferWrapper<const void>>(
      ctx_, "values", d_data_, static_cast<size_t>(row_size));
    // launch update
    if (modify_on_gpu_) {
      NVE_CHECK_(cudaMemcpy(d_keys_, &h_key, sizeof(KeyType), cudaMemcpyDefault));
      auto keys_bw = std::make_shared<BufferWrapper<const void>>(
        ctx_, "keys", d_keys_, sizeof(KeyType));
      tb_->update(ctx_, 1, std::move(keys_bw), row_size, row_size, std::move(values_bw));
    } else {
      auto keys_bw = std::make_shared<BufferWrapper<const void>>(
        ctx_, "keys", &h_key, sizeof(KeyType));
      tb_->update(ctx_, 1, std::move(keys_bw), row_size, row_size, std::move(values_bw));
    }
    NVE_CHECK_(cudaDeviceSynchronize());
  }

  bool find(KeyType* h_keys, int64_t num_keys, ValueType* h_data, int64_t row_size) {
    // return true if all keys were found
    
    const auto& params = GetParam();

    //  copy key to gpu
    NVE_CHECK_(cudaMemcpy(d_keys_, h_keys, sizeof(KeyType) * static_cast<size_t>(num_keys), cudaMemcpyDefault));
    NVE_CHECK_(cudaMemset(d_data_, 0, static_cast<size_t>(row_size) * static_cast<size_t>(num_keys)));
    NVE_CHECK_(cudaMemset(d_hitmask_, 0, hit_mask_size_in_bytes_));
    NVE_CHECK_(cudaDeviceSynchronize());

    // find the inserted key-data
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(
      ctx_, "keys", d_keys_, sizeof(KeyType) * static_cast<size_t>(num_keys));
    auto hit_mask_bw = std::make_shared<BufferWrapper<bitmask64_t>>(
      ctx_, "hit_mask", static_cast<bitmask64_t*>(d_hitmask_), hit_mask_size_in_bytes_);
    auto values_bw = std::make_shared<BufferWrapper<void>>(
      ctx_, "values", d_data_, static_cast<size_t>(row_size) * static_cast<size_t>(num_keys));
    tb_->find(ctx_, num_keys, std::move(keys_bw), std::move(hit_mask_bw), row_size,
                 std::move(values_bw), nullptr);
    NVE_CHECK_(cudaDeviceSynchronize());

    std::vector<bitmask64_t> h_hitmask(hit_mask_size_in_bytes_/sizeof(bitmask64_t) , 0);
    
    NVE_CHECK_(cudaMemcpy(h_data, d_data_, static_cast<size_t>(row_size) * static_cast<size_t>(num_keys), cudaMemcpyDefault));
    NVE_CHECK_(cudaMemcpy(h_hitmask.data(), d_hitmask_, hit_mask_size_in_bytes_, cudaMemcpyDefault));
    int64_t hit_count = 0;
    NVE_CHECK_(cudaDeviceSynchronize());
    if (!params.allocate_uvm_table) {
      // verify results
      for (size_t i = 0; i < hit_mask_size_in_bytes_/sizeof(bitmask64_t); i++) {
        hit_count += static_cast<int64_t>(bitmask64::count(h_hitmask[i]));
      }
      return hit_count == num_keys;
    } else {
      tb_->get_lookup_counter(ctx_, &hit_count);
      // hit mask is ignored in case of uvm
      return (tb_->lookup_counter_hits() ? hit_count == num_keys : hit_count == 0);
    }
  }

  void single_combine(KeyType* h_keys, int64_t num_keys, ValueType* h_data, int64_t row_size) {
    const auto& params = GetParam();
    EXPECT_LE(num_keys, params.max_keys);
    //  copy keys to gpu
    NVE_CHECK_(cudaMemcpy(d_keys_, h_keys, sizeof(KeyType) * static_cast<size_t>(num_keys),
                          cudaMemcpyDefault));
    // zero out d_data_
    NVE_CHECK_(cudaMemset(d_data_, 0, static_cast<size_t>(row_size)));
    NVE_CHECK_(cudaDeviceSynchronize());

    auto keys_bw = std::make_shared<BufferWrapper<const void>>(
        ctx_, "keys", d_keys_, sizeof(KeyType) * static_cast<size_t>(num_keys));
    auto values_bw = std::make_shared<BufferWrapper<void>>(
        ctx_, "values", d_data_, static_cast<size_t>(row_size));
    // Element type comes from the table config (value_dtype); pass the output/weight types.
    tb_->find_and_pool(
        ctx_, num_keys, std::move(keys_bw), SparseType_t::Fixed, 0 /*num_offsets*/,
        nullptr /*offsets*/, num_keys /* fixed_hotness*/, PoolingType_t::Sum,
        nullptr /*weights*/, data_type<OutputType>(), data_type<WeightType>(),
        row_size, std::move(values_bw));
    NVE_CHECK_(cudaDeviceSynchronize());

    // copy outputs to host
    NVE_CHECK_(cudaMemcpy(h_data, d_data_, static_cast<size_t>(row_size), cudaMemcpyDefault));
    NVE_CHECK_(cudaDeviceSynchronize());
  }

  void erase(KeyType h_key) {
    if (modify_on_gpu_) {
        NVE_CHECK_(cudaMemcpy(d_keys_, &h_key, sizeof(KeyType), cudaMemcpyDefault));
        NVE_CHECK_(cudaDeviceSynchronize());
        auto keys_bw = std::make_shared<BufferWrapper<const void>>(
          ctx_, "keys", d_keys_, sizeof(KeyType));
        tb_->erase(ctx_, 1, std::move(keys_bw));
    } else {
        auto keys_bw = std::make_shared<BufferWrapper<const void>>(
          ctx_, "keys", &h_key, sizeof(KeyType));
        tb_->erase(ctx_, 1, std::move(keys_bw));
    }
  }

  void clear() { tb_->clear(ctx_); }
};

// TRTREC-81: add test for clear (missing api in emb. cache)

// Define the test parameters
const GpuTableTestParams test_params[] = {
    //  cache size bytes, row size bytes, num rows, device id, max keys, allocate uvm table, cache data on host, modify on gpu
    {1l << 20, 1l << 7, 128, 0, 4, false, false, false},  // Use 1mb caches for sanity tests
    {1l << 20, 1l << 7, 128, 0, 4, false, true, false},
    {1l << 20, 1l << 7, 128, 0, 4, false, false, true},
    {1l << 20, 1l << 7, 128, 0, 4, false, true, true},
    {1l << 20, 1l << 7, GT_TEST_KEY+1, 0, GT_TEST_SORT_GATHER_THRESHOLD, true, false, false}, // case to go to the sort gather path
    {1l << 20, 1l << 7, GT_TEST_KEY+1, 0, (GT_TEST_SORT_GATHER_THRESHOLD*2), true, false, false}, // case to go to the sort gather path
};

using GTFixture_INT32_T = GpuTableTest<int32_t>;
using GTFixture_INT64_T = GpuTableTest<int64_t>;

#define TEST_FORMAT(test_name, test_func) \
TEST_P(GTFixture_INT32_T, test_name)      \
{                                         \
    test_func();                          \
}                                         \
TEST_P(GTFixture_INT64_T, test_name)      \
{                                         \
    test_func();                          \
}                                         \

TEST_FORMAT(insert, test_insert);
TEST_FORMAT(update, test_update);
TEST_FORMAT(missing_key, test_missing_key);
TEST_FORMAT(negative_key, test_negative_key);
TEST_FORMAT(negative_key_hits, test_negative_key_hits);
TEST_FORMAT(erase, test_erase);
TEST_FORMAT(find, test_find_uvm);
TEST_FORMAT(combine, test_combine);
#undef TEST_FORMAT

// Instantiate the tests with both type and value parameters
INSTANTIATE_TEST_SUITE_P(
    GpuTableTestInt32,
    GTFixture_INT32_T,
    testing::ValuesIn(test_params));

INSTANTIATE_TEST_SUITE_P(
    GpuTableTestInt64,
    GTFixture_INT64_T,
    testing::ValuesIn(test_params));

TEST(GpuTableInvalidKey, GetInvalidKeyCustom) {
  GPUTableConfig cfg;
  cfg.device_id = GT_TEST_DEVICE;
  cfg.cache_size = 1l << 20;
  cfg.row_size_in_bytes = 1l << 7;
  cfg.uvm_table = nullptr;
  cfg.invalid_key = 12345;
  GpuTable<int64_t> tab(cfg);
  EXPECT_EQ(int64_t{12345}, tab.get_invalid_key());
}

// A batch containing the configured invalid_key sentinel must not crash insert/find,
// and other (valid) keys in the same batch must still round-trip correctly.
TEST(GpuTableInvalidKey, ValidKeysSurviveBatchWithSentinel) {
  using KeyType = int64_t;
  constexpr int64_t row_size = 1l << 7;  // 128 bytes = 32 floats
  constexpr int64_t batch_size = 4;
  constexpr KeyType invalid_key = static_cast<KeyType>(0x7fffffffffffffffLL);
  constexpr int64_t hit_mask_bytes = sizeof(bitmask64_t);

  GPUTableConfig cfg;
  cfg.device_id = GT_TEST_DEVICE;
  cfg.cache_size = 1l << 20;
  cfg.row_size_in_bytes = row_size;
  cfg.uvm_table = nullptr;
  cfg.invalid_key = static_cast<int64_t>(invalid_key);
  GpuTable<KeyType> tab(cfg);
  auto ctx = tab.create_execution_context(0, 0, nullptr, nullptr);

  // Build a batch: three valid keys + one sentinel.
  std::vector<KeyType> keys{static_cast<KeyType>(101),
                            static_cast<KeyType>(202),
                            invalid_key,
                            static_cast<KeyType>(404)};
  ASSERT_EQ(static_cast<size_t>(batch_size), keys.size());

  const size_t row_floats = static_cast<size_t>(row_size) / sizeof(float);
  std::vector<float> values(static_cast<size_t>(batch_size) * row_floats, 0.0f);
  for (int64_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < row_floats; ++j) {
      values[static_cast<size_t>(i) * row_floats + j] = static_cast<float>(i * 1000 + static_cast<int64_t>(j));
    }
  }

  auto allocator = GetDefaultAllocator();
  void* d_keys = nullptr;
  void* d_values = nullptr;
  void* d_hitmask = nullptr;
  NVE_CHECK_(allocator->device_allocate(&d_keys, sizeof(KeyType) * batch_size));
  NVE_CHECK_(allocator->device_allocate(&d_values, static_cast<size_t>(row_size) * batch_size));
  NVE_CHECK_(allocator->device_allocate(&d_hitmask, hit_mask_bytes));

  NVE_CHECK_(cudaMemcpy(d_values, values.data(),
                        static_cast<size_t>(row_size) * batch_size, cudaMemcpyDefault));
  NVE_CHECK_(cudaMemcpy(d_keys, keys.data(), sizeof(KeyType) * batch_size, cudaMemcpyDefault));

  {
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(
      ctx, "keys", d_keys, sizeof(KeyType) * batch_size);
    auto values_bw = std::make_shared<BufferWrapper<const void>>(
      ctx, "values", d_values, static_cast<size_t>(row_size) * batch_size);
    tab.insert(ctx, batch_size, std::move(keys_bw), row_size, row_size, std::move(values_bw));
  }
  NVE_CHECK_(cudaDeviceSynchronize());
  NVE_CHECK_(cudaPeekAtLastError());

  // Find back: keys/values/hitmask all device-side.
  NVE_CHECK_(cudaMemcpy(d_keys, keys.data(), sizeof(KeyType) * batch_size, cudaMemcpyDefault));
  NVE_CHECK_(cudaMemset(d_values, 0, static_cast<size_t>(row_size) * batch_size));
  NVE_CHECK_(cudaMemset(d_hitmask, 0, hit_mask_bytes));
  {
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(
      ctx, "keys", d_keys, sizeof(KeyType) * batch_size);
    auto hit_mask_bw = std::make_shared<BufferWrapper<bitmask64_t>>(
      ctx, "hit_mask", static_cast<bitmask64_t*>(d_hitmask), hit_mask_bytes);
    auto values_bw = std::make_shared<BufferWrapper<void>>(
      ctx, "values", d_values, static_cast<size_t>(row_size) * batch_size);
    tab.find(ctx, batch_size, std::move(keys_bw), std::move(hit_mask_bw), row_size,
                std::move(values_bw), nullptr);
  }
  NVE_CHECK_(cudaDeviceSynchronize());
  NVE_CHECK_(cudaPeekAtLastError());

  std::vector<float> out(static_cast<size_t>(batch_size) * row_floats, 0.0f);
  bitmask64_t hitmask_host = 0;
  NVE_CHECK_(cudaMemcpy(out.data(), d_values,
                        static_cast<size_t>(row_size) * batch_size, cudaMemcpyDefault));
  NVE_CHECK_(cudaMemcpy(&hitmask_host, d_hitmask, hit_mask_bytes, cudaMemcpyDefault));

  // Valid keys (indices 0, 1, 3) must hit and round-trip exactly. Sentinel slot is undefined.
  for (int64_t i : {int64_t{0}, int64_t{1}, int64_t{3}}) {
    const bool hit = (hitmask_host >> i) & 0x1u;
    EXPECT_TRUE(hit) << "valid key at slot " << i << " should hit";
    for (size_t j = 0; j < row_floats; ++j) {
      EXPECT_FLOAT_EQ(values[static_cast<size_t>(i) * row_floats + j],
                      out[static_cast<size_t>(i) * row_floats + j])
          << "slot " << i << " float " << j;
    }
  }

  allocator->device_free(d_keys);
  allocator->device_free(d_values);
  allocator->device_free(d_hitmask);
}

// With a non-default sentinel, -1 becomes a regular key and must round-trip.
TEST(GpuTableInvalidKey, MinusOneRoundTripsWhenSentinelIsCustom) {
  using KeyType = int64_t;
  constexpr int64_t row_size = 1l << 7;  // 128 bytes = 32 floats
  constexpr int64_t batch_size = 3;
  constexpr KeyType custom_sentinel = static_cast<KeyType>(0x7fffffffffffffffLL);
  constexpr KeyType target_key = static_cast<KeyType>(-1);
  constexpr int64_t hit_mask_bytes = sizeof(bitmask64_t);

  GPUTableConfig cfg;
  cfg.device_id = GT_TEST_DEVICE;
  cfg.cache_size = 1l << 20;
  cfg.row_size_in_bytes = row_size;
  cfg.uvm_table = nullptr;
  cfg.invalid_key = static_cast<int64_t>(custom_sentinel);
  GpuTable<KeyType> tab(cfg);
  auto ctx = tab.create_execution_context(0, 0, nullptr, nullptr);

  std::vector<KeyType> keys{static_cast<KeyType>(101), target_key, static_cast<KeyType>(303)};
  ASSERT_EQ(static_cast<size_t>(batch_size), keys.size());

  const size_t row_floats = static_cast<size_t>(row_size) / sizeof(float);
  std::vector<float> values(static_cast<size_t>(batch_size) * row_floats, 0.0f);
  for (int64_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < row_floats; ++j) {
      values[static_cast<size_t>(i) * row_floats + j] = static_cast<float>(i * 1000 + static_cast<int64_t>(j));
    }
  }

  auto allocator = GetDefaultAllocator();
  void* d_keys = nullptr;
  void* d_values = nullptr;
  void* d_hitmask = nullptr;
  NVE_CHECK_(allocator->device_allocate(&d_keys, sizeof(KeyType) * batch_size));
  NVE_CHECK_(allocator->device_allocate(&d_values, static_cast<size_t>(row_size) * batch_size));
  NVE_CHECK_(allocator->device_allocate(&d_hitmask, hit_mask_bytes));

  NVE_CHECK_(cudaMemcpy(d_values, values.data(),
                        static_cast<size_t>(row_size) * batch_size, cudaMemcpyDefault));
  NVE_CHECK_(cudaMemcpy(d_keys, keys.data(), sizeof(KeyType) * batch_size, cudaMemcpyDefault));

  {
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(
      ctx, "keys", d_keys, sizeof(KeyType) * batch_size);
    auto values_bw = std::make_shared<BufferWrapper<const void>>(
      ctx, "values", d_values, static_cast<size_t>(row_size) * batch_size);
    tab.insert(ctx, batch_size, std::move(keys_bw), row_size, row_size, std::move(values_bw));
  }
  NVE_CHECK_(cudaDeviceSynchronize());
  NVE_CHECK_(cudaPeekAtLastError());

  NVE_CHECK_(cudaMemcpy(d_keys, keys.data(), sizeof(KeyType) * batch_size, cudaMemcpyDefault));
  NVE_CHECK_(cudaMemset(d_values, 0, static_cast<size_t>(row_size) * batch_size));
  NVE_CHECK_(cudaMemset(d_hitmask, 0, hit_mask_bytes));
  {
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(
      ctx, "keys", d_keys, sizeof(KeyType) * batch_size);
    auto hit_mask_bw = std::make_shared<BufferWrapper<bitmask64_t>>(
      ctx, "hit_mask", static_cast<bitmask64_t*>(d_hitmask), hit_mask_bytes);
    auto values_bw = std::make_shared<BufferWrapper<void>>(
      ctx, "values", d_values, static_cast<size_t>(row_size) * batch_size);
    tab.find(ctx, batch_size, std::move(keys_bw), std::move(hit_mask_bw), row_size,
                std::move(values_bw), nullptr);
  }
  NVE_CHECK_(cudaDeviceSynchronize());
  NVE_CHECK_(cudaPeekAtLastError());

  std::vector<float> out(static_cast<size_t>(batch_size) * row_floats, 0.0f);
  bitmask64_t hitmask_host = 0;
  NVE_CHECK_(cudaMemcpy(out.data(), d_values,
                        static_cast<size_t>(row_size) * batch_size, cudaMemcpyDefault));
  NVE_CHECK_(cudaMemcpy(&hitmask_host, d_hitmask, hit_mask_bytes, cudaMemcpyDefault));

  // All three keys (including -1) must hit and round-trip exactly.
  for (int64_t i = 0; i < batch_size; ++i) {
    const bool hit = (hitmask_host >> i) & 0x1u;
    EXPECT_TRUE(hit) << "key " << keys[static_cast<size_t>(i)] << " at slot " << i << " should hit";
    for (size_t j = 0; j < row_floats; ++j) {
      EXPECT_FLOAT_EQ(values[static_cast<size_t>(i) * row_floats + j],
                      out[static_cast<size_t>(i) * row_floats + j])
          << "slot " << i << " float " << j;
    }
  }

  allocator->device_free(d_keys);
  allocator->device_free(d_values);
  allocator->device_free(d_hitmask);
}

// Update/accumulate must ignore keys outside [0, uvm_num_rows) rather than write past the table.
// The UVM buffer holds `guard_rows` extra rows past uvm_num_rows: out of range keys are aimed at
// them, so an unchecked write shows up as a modified guard row.
template <typename KeyType>
static void RunUvmOutOfRangeKeysTest(bool accumulate, bool uvm_cpu_accumulate, int64_t num_keys) {
  const int64_t num_rows = num_keys;  // one key per row keeps the in-range keys unique
  constexpr int64_t guard_rows = 8;
  constexpr int64_t row_size = 128;  // bytes == 32 floats
  const size_t row_floats = static_cast<size_t>(row_size) / sizeof(float);
  const size_t total_floats = static_cast<size_t>(num_rows + guard_rows) * row_floats;

  float* h_table = nullptr;
  NVE_CHECK_(cudaMallocHost(&h_table, static_cast<size_t>(num_rows + guard_rows) * row_size));
  for (size_t i = 0; i < total_floats; ++i) {
    h_table[i] = static_cast<float>(i);
  }
  std::vector<float> ref_table(h_table, h_table + total_floats);

  GPUTableConfig cfg;
  cfg.device_id = GT_TEST_DEVICE;
  cfg.cache_size = 1l << 20;
  cfg.row_size_in_bytes = row_size;
  cfg.value_dtype = DataType_t::Float32;
  cfg.uvm_table = h_table;
  cfg.uvm_num_rows = num_rows;
  cfg.uvm_cpu_accumulate = uvm_cpu_accumulate;
  GpuTable<KeyType> tab(cfg);
  auto ctx = tab.create_execution_context(0, 0, nullptr, nullptr);

  // Every 7th key is out of range: negative, or aimed at a guard row past the end of the table.
  std::vector<KeyType> keys(static_cast<size_t>(num_keys));
  for (int64_t i = 0; i < num_keys; ++i) {
    if ((i % 7) == 1) {
      keys[static_cast<size_t>(i)] = static_cast<KeyType>(-(i + 1));
    } else if ((i % 7) == 3) {
      keys[static_cast<size_t>(i)] = static_cast<KeyType>(num_rows + (i % guard_rows));
    } else {
      keys[static_cast<size_t>(i)] = static_cast<KeyType>(i);
    }
  }

  std::vector<float> updates(static_cast<size_t>(num_keys) * row_floats);
  for (int64_t i = 0; i < num_keys; ++i) {
    for (size_t j = 0; j < row_floats; ++j) {
      updates[static_cast<size_t>(i) * row_floats + j] = static_cast<float>(1000 + i) + static_cast<float>(j);
    }
  }

  // Reference: apply the in-range keys only
  for (int64_t i = 0; i < num_keys; ++i) {
    const KeyType key = keys[static_cast<size_t>(i)];
    if (key < 0 || static_cast<int64_t>(key) >= num_rows) {
      continue;
    }
    for (size_t j = 0; j < row_floats; ++j) {
      float& dst = ref_table[static_cast<size_t>(key) * row_floats + j];
      const float src = updates[static_cast<size_t>(i) * row_floats + j];
      dst = accumulate ? (dst + src) : src;
    }
  }

  const size_t keys_bytes = sizeof(KeyType) * static_cast<size_t>(num_keys);
  const size_t updates_bytes = static_cast<size_t>(row_size) * static_cast<size_t>(num_keys);
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys.data(), keys_bytes);
  auto updates_bw = std::make_shared<BufferWrapper<const void>>(ctx, "updates", updates.data(), updates_bytes);
  if (accumulate) {
    tab.update_accumulate(ctx, num_keys, std::move(keys_bw), row_size, row_size,
                          std::move(updates_bw), DataType_t::Float32);
  } else {
    tab.update(ctx, num_keys, std::move(keys_bw), row_size, row_size, std::move(updates_bw));
  }
  NVE_CHECK_(cudaDeviceSynchronize());
  NVE_CHECK_(cudaPeekAtLastError());

  for (int64_t r = 0; r < num_rows + guard_rows; ++r) {
    for (size_t j = 0; j < row_floats; ++j) {
      const size_t idx = static_cast<size_t>(r) * row_floats + j;
      ASSERT_FLOAT_EQ(ref_table[idx], h_table[idx])
          << (r >= num_rows ? "guard row " : "row ") << r << " float " << j;
    }
  }

  NVE_CHECK_(cudaFreeHost(h_table));
}

// Batch large enough to take update_accumulate's multi-copy CPU path
// (it triggers at num_workers * 512 / 2 keys).
static int64_t large_accumulate_batch() {
  return static_cast<int64_t>(std::max(8u, std::thread::hardware_concurrency())) * 512;
}

TEST(GpuTableOutOfRangeKeys, UpdateInt64) {
  RunUvmOutOfRangeKeysTest<int64_t>(false /*accumulate*/, true /*uvm_cpu_accumulate*/, 64);
}

TEST(GpuTableOutOfRangeKeys, UpdateInt32) {
  RunUvmOutOfRangeKeysTest<int32_t>(false /*accumulate*/, true /*uvm_cpu_accumulate*/, 64);
}

TEST(GpuTableOutOfRangeKeys, AccumulateOnCpu) {
  RunUvmOutOfRangeKeysTest<int64_t>(true /*accumulate*/, true /*uvm_cpu_accumulate*/, 64);
}

TEST(GpuTableOutOfRangeKeys, AccumulateOnCpuLargeBatch) {
  RunUvmOutOfRangeKeysTest<int64_t>(true /*accumulate*/, true /*uvm_cpu_accumulate*/,
                                    large_accumulate_batch());
}

TEST(GpuTableOutOfRangeKeys, AccumulateOnGpu) {
  RunUvmOutOfRangeKeysTest<int64_t>(true /*accumulate*/, false /*uvm_cpu_accumulate*/, 64);
}

// A UVM table without a row count would make every key look out of range, so it is rejected.
TEST(GpuTableOutOfRangeKeys, RejectsUvmTableWithoutNumRows) {
  std::vector<float> table(64 * 32);
  GPUTableConfig cfg;
  cfg.device_id = GT_TEST_DEVICE;
  cfg.cache_size = 1l << 20;
  cfg.row_size_in_bytes = 128;
  cfg.uvm_table = table.data();
  EXPECT_THROW(std::make_shared<GpuTable<int64_t>>(cfg), nve::Exception);
}

// ---------------------------------------------------------------------------
// Rowwise-quantized find_and_dequant / find_and_combine tests (float-scale variants).
//
// Builds a host UVM table of per-row quantized rows ([ q[N] ][ scale ][ offset? ]) and verifies
// that GpuTable::find_and_dequant (one dequantized row per key, no pooling) and the quantized
// find_and_combine sum path reconstruct the same float values a CPU dequant reference does. Keys
// are never inserted into the GPU cache, so every lookup resolves through the UVM table fallback.
// ---------------------------------------------------------------------------
namespace {

constexpr int64_t QUANT_NUM_ELEMENTS = 32;  // % 4 == 0 -> exercises the char4 (Vec4) dequant path

// Byte size of one quantized row for a float-scale variant, padded to a 4-byte stride so each row
// base stays char4-aligned (and divisible by 2 as GpuTable requires).
int64_t quant_row_bytes(bool has_offset) {
  int64_t bytes = QUANT_NUM_ELEMENTS + (has_offset ? 2 : 1) * static_cast<int64_t>(sizeof(float));
  return (bytes + 3) & ~static_cast<int64_t>(3);
}

// Quantize one row of floats into a quantized row, mirroring the kernel/test convention:
//   QInt8RowwiseF32  (signed, symmetric): scale = max|v| / 127, q = round(v / scale), no offset.
//   QUint8RowwiseF32 (unsigned, affine):  scale = (max - min) / 255, offset = min,
//                                         q = round((v - offset) / scale).
void quantize_row_f32(const std::vector<float>& v, bool has_offset, int8_t* row) {
  float scale = 1.0f, offset = 0.0f;
  if (!has_offset) {
    float amax = 0.0f;
    for (float x : v) amax = std::max(amax, std::fabs(x));
    scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    for (size_t j = 0; j < v.size(); ++j) {
      long q = std::lround(v[j] / scale);
      q = std::min<long>(127, std::max<long>(-127, q));
      reinterpret_cast<int8_t*>(row)[j] = static_cast<int8_t>(q);
    }
  } else {
    float lo = v[0], hi = v[0];
    for (float x : v) { lo = std::min(lo, x); hi = std::max(hi, x); }
    scale = hi > lo ? (hi - lo) / 255.0f : 1.0f;
    offset = lo;
    for (size_t j = 0; j < v.size(); ++j) {
      long q = std::lround((v[j] - offset) / scale);
      q = std::min<long>(255, std::max<long>(0, q));
      reinterpret_cast<uint8_t*>(row)[j] = static_cast<uint8_t>(q);
    }
  }
  int8_t* meta = row + QUANT_NUM_ELEMENTS;  // element_size == 1 byte
  std::memcpy(meta, &scale, sizeof(float));
  if (has_offset) std::memcpy(meta + sizeof(float), &offset, sizeof(float));
}

// CPU reference dequant of element e: float(q[e]) * scale [+ offset], mirroring the kernel.
float dequant_row_element(const int8_t* row, int64_t e, bool has_offset, bool is_signed) {
  const float base = is_signed ? static_cast<float>(reinterpret_cast<const int8_t*>(row)[e])
                               : static_cast<float>(reinterpret_cast<const uint8_t*>(row)[e]);
  const int8_t* meta = row + QUANT_NUM_ELEMENTS;
  float scale = 1.0f, offset = 0.0f;
  std::memcpy(&scale, meta, sizeof(float));
  if (has_offset) std::memcpy(&offset, meta + sizeof(float), sizeof(float));
  return base * scale + offset;
}

// Build a host UVM table of `num_rows` quantized rows (returns the pinned-host pointer).
int8_t* make_quant_uvm_table(int64_t num_rows, bool has_offset, int64_t row_bytes, size_t seed) {
  int8_t* h_table = nullptr;
  NVE_CHECK_(cudaMallocHost(&h_table, static_cast<size_t>(num_rows * row_bytes)));
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(QUANT_NUM_ELEMENTS);
  for (int64_t r = 0; r < num_rows; ++r) {
    for (auto& x : v) x = dist(gen);
    quantize_row_f32(v, has_offset, h_table + r * row_bytes);
  }
  return h_table;
}

}  // namespace

template <typename KeyType>
static void RunFindAndDequantQuant(DataType_t dtype) {
  const bool has_offset = (dtype == DataType_t::QUint8RowwiseF32);
  const bool is_signed = (dtype == DataType_t::QInt8RowwiseF32);
  const int64_t row_bytes = quant_row_bytes(has_offset);
  constexpr int64_t num_rows = 512;
  int8_t* h_table = make_quant_uvm_table(num_rows, has_offset, row_bytes, 0xC0FFEE);

  GPUTableConfig cfg;
  cfg.device_id = GT_TEST_DEVICE;
  cfg.cache_size = 1l << 20;
  cfg.row_size_in_bytes = row_bytes;
  cfg.value_dtype = dtype;
  cfg.uvm_table = h_table;
  cfg.uvm_num_rows = num_rows;
  GpuTable<KeyType> tab(cfg);
  auto ctx = tab.create_execution_context(0, 0, nullptr, nullptr);

  std::vector<KeyType> keys{3, 7, 42, 100, 0, 255, 511};
  const int64_t num_keys = static_cast<int64_t>(keys.size());
  const int64_t out_stride = QUANT_NUM_ELEMENTS * static_cast<int64_t>(sizeof(float));

  auto allocator = GetDefaultAllocator();
  void* d_keys = nullptr;
  void* d_out = nullptr;
  NVE_CHECK_(allocator->device_allocate(&d_keys, sizeof(KeyType) * static_cast<size_t>(num_keys)));
  NVE_CHECK_(allocator->device_allocate(&d_out, static_cast<size_t>(out_stride * num_keys)));
  NVE_CHECK_(cudaMemcpy(d_keys, keys.data(), sizeof(KeyType) * static_cast<size_t>(num_keys), cudaMemcpyDefault));
  NVE_CHECK_(cudaMemset(d_out, 0, static_cast<size_t>(out_stride * num_keys)));

  {
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(
        ctx, "keys", d_keys, sizeof(KeyType) * static_cast<size_t>(num_keys));
    auto out_bw = std::make_shared<BufferWrapper<void>>(
        ctx, "values", d_out, static_cast<size_t>(out_stride * num_keys));
    // Concatenate routes find_and_pool to the dequant path (one dequantized row per key).
    tab.find_and_pool(ctx, num_keys, std::move(keys_bw), SparseType_t::Fixed, 0 /*num_offsets*/,
                      nullptr /*offsets*/, 1 /*fixed_hotness*/, PoolingType_t::Concatenate,
                      nullptr /*weights*/, DataType_t::Float32 /*output*/, DataType_t::Float32 /*weight*/,
                      out_stride, std::move(out_bw));
  }
  NVE_CHECK_(cudaDeviceSynchronize());

  std::vector<float> out(static_cast<size_t>(num_keys * QUANT_NUM_ELEMENTS), 0.0f);
  NVE_CHECK_(cudaMemcpy(out.data(), d_out, static_cast<size_t>(out_stride * num_keys), cudaMemcpyDefault));

  for (int64_t i = 0; i < num_keys; ++i) {
    const int8_t* row = h_table + static_cast<int64_t>(keys[static_cast<size_t>(i)]) * row_bytes;
    for (int64_t e = 0; e < QUANT_NUM_ELEMENTS; ++e) {
      EXPECT_NEAR(out[static_cast<size_t>(i * QUANT_NUM_ELEMENTS + e)],
                  dequant_row_element(row, e, has_offset, is_signed), 1e-3f)
          << "key slot " << i << " element " << e;
    }
  }

  allocator->device_free(d_keys);
  allocator->device_free(d_out);
  NVE_CHECK_(cudaFreeHost(h_table));
}

template <typename KeyType>
static void RunFindAndCombineQuant(DataType_t dtype) {
  const bool has_offset = (dtype == DataType_t::QUint8RowwiseF32);
  const bool is_signed = (dtype == DataType_t::QInt8RowwiseF32);
  const int64_t row_bytes = quant_row_bytes(has_offset);
  constexpr int64_t num_rows = 512;
  int8_t* h_table = make_quant_uvm_table(num_rows, has_offset, row_bytes, 0xBEEF);

  GPUTableConfig cfg;
  cfg.device_id = GT_TEST_DEVICE;
  cfg.cache_size = 1l << 20;
  cfg.row_size_in_bytes = row_bytes;
  cfg.value_dtype = dtype;
  cfg.uvm_table = h_table;
  cfg.uvm_num_rows = num_rows;
  GpuTable<KeyType> tab(cfg);
  auto ctx = tab.create_execution_context(0, 0, nullptr, nullptr);

  // Single fixed-hotness bag summing num_keys dequantized rows into one output row.
  std::vector<KeyType> keys{1, 5, 13, 200};
  const int64_t num_keys = static_cast<int64_t>(keys.size());
  const int64_t out_stride = QUANT_NUM_ELEMENTS * static_cast<int64_t>(sizeof(float));

  auto allocator = GetDefaultAllocator();
  void* d_keys = nullptr;
  void* d_out = nullptr;
  NVE_CHECK_(allocator->device_allocate(&d_keys, sizeof(KeyType) * static_cast<size_t>(num_keys)));
  NVE_CHECK_(allocator->device_allocate(&d_out, static_cast<size_t>(out_stride)));
  NVE_CHECK_(cudaMemcpy(d_keys, keys.data(), sizeof(KeyType) * static_cast<size_t>(num_keys), cudaMemcpyDefault));
  NVE_CHECK_(cudaMemset(d_out, 0, static_cast<size_t>(out_stride)));

  {
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(
        ctx, "keys", d_keys, sizeof(KeyType) * static_cast<size_t>(num_keys));
    auto out_bw = std::make_shared<BufferWrapper<void>>(ctx, "values", d_out, static_cast<size_t>(out_stride));
    tab.find_and_pool(ctx, num_keys, std::move(keys_bw), SparseType_t::Fixed, 0 /*num_offsets*/,
                      nullptr /*offsets*/, num_keys /*fixed_hotness*/, PoolingType_t::Sum,
                      nullptr /*weights*/, DataType_t::Float32 /*output*/, DataType_t::Float32 /*weight*/,
                      out_stride, std::move(out_bw));
  }
  NVE_CHECK_(cudaDeviceSynchronize());

  std::vector<float> out(static_cast<size_t>(QUANT_NUM_ELEMENTS), 0.0f);
  NVE_CHECK_(cudaMemcpy(out.data(), d_out, static_cast<size_t>(out_stride), cudaMemcpyDefault));

  for (int64_t e = 0; e < QUANT_NUM_ELEMENTS; ++e) {
    float ref = 0.0f;
    for (int64_t i = 0; i < num_keys; ++i) {
      const int8_t* row = h_table + static_cast<int64_t>(keys[static_cast<size_t>(i)]) * row_bytes;
      ref += dequant_row_element(row, e, has_offset, is_signed);
    }
    EXPECT_NEAR(out[static_cast<size_t>(e)], ref, 1e-3f) << "element " << e;
  }

  allocator->device_free(d_keys);
  allocator->device_free(d_out);
  NVE_CHECK_(cudaFreeHost(h_table));
}

TEST(GpuTableQuant, FindAndDequantQInt8F32_Int64) {
  RunFindAndDequantQuant<int64_t>(DataType_t::QInt8RowwiseF32);
}
TEST(GpuTableQuant, FindAndDequantQUint8F32_Int64) {
  RunFindAndDequantQuant<int64_t>(DataType_t::QUint8RowwiseF32);
}
TEST(GpuTableQuant, FindAndDequantQInt8F32_Int32) {
  RunFindAndDequantQuant<int32_t>(DataType_t::QInt8RowwiseF32);
}
TEST(GpuTableQuant, FindAndCombineQInt8F32_Int64) {
  RunFindAndCombineQuant<int64_t>(DataType_t::QInt8RowwiseF32);
}
TEST(GpuTableQuant, FindAndCombineQUint8F32_Int64) {
  RunFindAndCombineQuant<int64_t>(DataType_t::QUint8RowwiseF32);
}

}  // namespace nve
