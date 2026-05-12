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
#include <cuda_support.hpp>
#include <gpu_table.hpp>
#include <default_allocator.hpp>
#include <tuple>
#include <vector>

namespace nve {

#define GT_TEST_DEVICE (0)
#define GT_TEST_KEY (1337)
#define GT_TEST_HITMASK_SIZE_BYTES (sizeof(max_bitmask_repr_t))
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
          typename OffsetType = KeyType,    // Type used for Offset during lookup of COO/CSR
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
    } else {
      h_table_ = nullptr;
    }

    tb_ = std::make_shared<TableType>(cfg);
    ctx_ = tb_->create_execution_context(0, 0, nullptr, nullptr);
    auto allocator = GetDefaultAllocator();
    NVE_CHECK_(allocator->device_allocate(&d_keys_, sizeof(KeyType) * static_cast<size_t>(params.max_keys)));
    NVE_CHECK_(allocator->device_allocate(&d_data_, static_cast<size_t>(params.row_size_bytes) * static_cast<size_t>(params.max_keys)));
    hit_mask_size_in_bytes_ = ((static_cast<size_t>(params.max_keys) + 63) / 64) * sizeof(int64_t);
    NVE_CHECK_(allocator->device_allocate(&d_hitmask_, hit_mask_size_in_bytes_));

    ASSERT_TRUE(d_keys_);
    ASSERT_TRUE(d_data_);
    ASSERT_TRUE(d_hitmask_);
  }

  void insert(KeyType h_key, const ValueType* h_data, int64_t row_size) {
    // copy key and data vector to gpu buffer
    NVE_CHECK_(cudaMemcpy(d_data_, h_data, static_cast<size_t>(row_size), cudaMemcpyDefault));

    // launch insert
    if (modify_on_gpu_) {
      NVE_CHECK_(cudaMemcpy(d_keys_, &h_key, sizeof(KeyType), cudaMemcpyDefault));
      tb_->insert(ctx_, 1, d_keys_, row_size, row_size, d_data_);
    } else {
      tb_->insert(ctx_, 1, &h_key, row_size, row_size, d_data_);
    }
    NVE_CHECK_(cudaDeviceSynchronize());
  }

  void update(KeyType h_key, const ValueType* h_data, int64_t row_size) {
    // copy key and data vector to gpu buffer
    NVE_CHECK_(cudaMemcpy(d_data_, h_data, static_cast<size_t>(row_size), cudaMemcpyDefault));

    // launch update
    if (modify_on_gpu_) {
      NVE_CHECK_(cudaMemcpy(d_keys_, &h_key, sizeof(KeyType), cudaMemcpyDefault));
      tb_->update(ctx_, 1, d_keys_, row_size, row_size, d_data_);
    } else {
      tb_->update(ctx_, 1, &h_key, row_size, row_size, d_data_);
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
    tb_->find(ctx_, num_keys, static_cast<const KeyType*>(d_keys_),
              static_cast<max_bitmask_repr_t*>(d_hitmask_), row_size,
              static_cast<OutputType*>(d_data_), nullptr);
    NVE_CHECK_(cudaDeviceSynchronize());

    std::vector<uint64_t> h_hitmask(hit_mask_size_in_bytes_/8 , 0);
    
    NVE_CHECK_(cudaMemcpy(h_data, d_data_, static_cast<size_t>(row_size) * static_cast<size_t>(num_keys), cudaMemcpyDefault));
    NVE_CHECK_(cudaMemcpy(h_hitmask.data(), d_hitmask_, hit_mask_size_in_bytes_, cudaMemcpyDefault));
    int64_t hit_count = 0;
    NVE_CHECK_(cudaDeviceSynchronize());
    if (!params.allocate_uvm_table) {
      // verify results
      for (size_t i = 0; i < hit_mask_size_in_bytes_/8; i++) {
        hit_count += static_cast<int64_t>(__builtin_popcountll(h_hitmask[i]));
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

    tb_->find_and_combine(ctx_, num_keys, d_keys_, SparseType_t::Fixed, 0 /*num_offsets*/,
                          static_cast<OffsetType*>(nullptr) /*offsets*/,
                          num_keys /* fixed_hotness*/, PoolingType_t::Sum,
                          static_cast<WeightType*>(nullptr), row_size,
                          static_cast<OutputType*>(d_data_));
    NVE_CHECK_(cudaDeviceSynchronize());

    // copy outputs to host
    NVE_CHECK_(cudaMemcpy(h_data, d_data_, static_cast<size_t>(row_size), cudaMemcpyDefault));
    NVE_CHECK_(cudaDeviceSynchronize());
  }

  void erase(KeyType h_key) {
    if (modify_on_gpu_) {
        NVE_CHECK_(cudaMemcpy(d_keys_, &h_key, sizeof(KeyType), cudaMemcpyDefault));
        NVE_CHECK_(cudaDeviceSynchronize());
        tb_->erase(ctx_, 1, d_keys_);
    } else {
        tb_->erase(ctx_, 1, &h_key);
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
TEST_FORMAT(DISABLED_combine, test_combine); // Disabled - pooling path not implemented yet
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
  constexpr int64_t hit_mask_bytes = sizeof(max_bitmask_repr_t);

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

  tab.insert(ctx, batch_size, d_keys, row_size, row_size, d_values);
  NVE_CHECK_(cudaDeviceSynchronize());
  NVE_CHECK_(cudaPeekAtLastError());

  // Find back: keys/values/hitmask all device-side.
  NVE_CHECK_(cudaMemcpy(d_keys, keys.data(), sizeof(KeyType) * batch_size, cudaMemcpyDefault));
  NVE_CHECK_(cudaMemset(d_values, 0, static_cast<size_t>(row_size) * batch_size));
  NVE_CHECK_(cudaMemset(d_hitmask, 0, hit_mask_bytes));
  tab.find(ctx, batch_size, d_keys, static_cast<max_bitmask_repr_t*>(d_hitmask),
           row_size, d_values, nullptr);
  NVE_CHECK_(cudaDeviceSynchronize());
  NVE_CHECK_(cudaPeekAtLastError());

  std::vector<float> out(static_cast<size_t>(batch_size) * row_floats, 0.0f);
  max_bitmask_repr_t hitmask_host = 0;
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
  constexpr int64_t hit_mask_bytes = sizeof(max_bitmask_repr_t);

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

  tab.insert(ctx, batch_size, d_keys, row_size, row_size, d_values);
  NVE_CHECK_(cudaDeviceSynchronize());
  NVE_CHECK_(cudaPeekAtLastError());

  NVE_CHECK_(cudaMemcpy(d_keys, keys.data(), sizeof(KeyType) * batch_size, cudaMemcpyDefault));
  NVE_CHECK_(cudaMemset(d_values, 0, static_cast<size_t>(row_size) * batch_size));
  NVE_CHECK_(cudaMemset(d_hitmask, 0, hit_mask_bytes));
  tab.find(ctx, batch_size, d_keys, static_cast<max_bitmask_repr_t*>(d_hitmask),
           row_size, d_values, nullptr);
  NVE_CHECK_(cudaDeviceSynchronize());
  NVE_CHECK_(cudaPeekAtLastError());

  std::vector<float> out(static_cast<size_t>(batch_size) * row_floats, 0.0f);
  max_bitmask_repr_t hitmask_host = 0;
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

}  // namespace nve
