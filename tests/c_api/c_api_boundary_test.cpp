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

// Boundary / semantic edge-case tests for the C API's table functions
//
// Uses a host table (in-tree stl-map plugin) rather than a GPU table for most
// cases: host tables take plain host pointers for keys/values (no CUDA buffers
// needed), which keeps these edge-case tests simple, and the behavior under test
// (argument validation, counter semantics, n==0 handling) is table-implementation
// agnostic -- it's all in the shared nve_c_api_tables.cpp wrapper layer.

#include <gtest/gtest.h>
#include <nve_c_api.h>

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "c_api_ptr.hpp"
#include "cuda_ptr.hpp"
#include "test_utils.hpp"

namespace {

using nve::ContextPtr;
using nve::TableFactoryPtr;
using nve::TablePtr;
using nve::ThreadPoolPtr;

class HostTableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // A dedicated thread pool rather than the process-wide default.
    ASSERT_EQ(NVE_SUCCESS, nve_thread_pool_create(&pool_, "{\"name\": \"simple\", \"num_workers\": 1}"));
    ASSERT_EQ(NVE_SUCCESS,
              nve_create_table_factory(&factory_, "libnve-plugin-stl-map.so", "{}"));
    ASSERT_EQ(NVE_SUCCESS, nve_table_factory_produce(factory_, 0, "{}", &table_));
    ASSERT_EQ(NVE_SUCCESS,
              nve_table_create_execution_context(table_, &ctx_, nullptr, nullptr, pool_, nullptr));
  }

  void TearDown() override {
    if (ctx_) nve_context_destroy(ctx_);
    if (table_) nve_table_destroy(table_);
    if (factory_) nve_table_factory_destroy(factory_);
    if (pool_) nve_thread_pool_destroy(pool_);
  }

  nve_thread_pool_t pool_ = nullptr;
  nve_table_factory_t factory_ = nullptr;
  nve_table_t table_ = nullptr;
  nve_context_t ctx_ = nullptr;
};

}  // namespace

// ---- Exception -> status mapping ----

// A CUDA runtime failure must surface as NVE_ERROR_CUDA, not as a generic runtime error.
TEST(StatusMapping, CudaFailureIsReportedAsCudaError) {
  nve_gpu_table_config_t cfg;
  ASSERT_EQ(NVE_SUCCESS, nve_gpu_table_config_init(&cfg, sizeof(cfg)));
  cfg.device_id = 1000;  // no such device: the first cudaSetDevice on it fails
  cfg.cache_size = 1 << 20;
  cfg.row_size_in_bytes = 64;
  cfg.value_dtype = NVE_DTYPE_FLOAT32;
  nve_table_t table = nullptr;
  nve_context_t ctx = nullptr;
  nve_status_t status = nve_gpu_table_create(&table, NVE_KEY_INT64, &cfg, nullptr);
  if (status == NVE_SUCCESS) {
    status = nve_table_create_execution_context(table, &ctx, nullptr, nullptr, nullptr, nullptr);
  }
  if (status == NVE_SUCCESS) {
    status = nve_table_clear(table, ctx);
  }
  EXPECT_EQ(NVE_ERROR_CUDA, status);
  if (ctx) nve_context_destroy(ctx);
  if (table) nve_table_destroy(table);
  cudaGetLastError();  // clear any sticky error for later tests
}

// An unimplemented code path must surface as NVE_ERROR_NOT_IMPLEMENTED.
TEST(StatusMapping, NotImplementedIsReportedAsNotImplemented) {
  constexpr int64_t row_size = 64;
  constexpr int64_t rows = 16;
  void* h_uvm = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMallocHost(&h_uvm, static_cast<size_t>(rows * row_size)));
  nve::HostPtr<void> uvm_owner(h_uvm);
  nve_gpu_table_config_t cfg;
  ASSERT_EQ(NVE_SUCCESS, nve_gpu_table_config_init(&cfg, sizeof(cfg)));
  cfg.device_id = 0;
  cfg.cache_size = 1 << 20;
  cfg.row_size_in_bytes = row_size;
  cfg.value_dtype = NVE_DTYPE_FLOAT32;
  cfg.uvm_table = h_uvm;
  cfg.uvm_num_rows = rows;
  cfg.kernel_mode_type = 99;  // no such lookup kernel mode: the lookup hits NVE_THROW_NOT_IMPLEMENTED_
  nve_table_t table = nullptr;
  ASSERT_EQ(NVE_SUCCESS, nve_gpu_table_create(&table, NVE_KEY_INT64, &cfg, nullptr));
  TablePtr table_owner(table);
  nve_context_t ctx = nullptr;
  ASSERT_EQ(NVE_SUCCESS, nve_table_create_execution_context(table, &ctx, nullptr, nullptr, nullptr, nullptr));
  ContextPtr ctx_owner(ctx);
  void* d_keys = nullptr;
  void* d_out = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_keys, sizeof(int64_t)));
  nve::DevicePtr<void> keys_owner(d_keys);
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_out, static_cast<size_t>(row_size)));
  nve::DevicePtr<void> out_owner(d_out);
  ASSERT_EQ(cudaSuccess, cudaMemset(d_keys, 0, sizeof(int64_t)));
  EXPECT_EQ(NVE_ERROR_NOT_IMPLEMENTED,
            nve_table_find(table, ctx, 1, d_keys, nullptr, row_size, d_out, nullptr));
  cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------------
// n == 0 for find/insert/update/erase: buffer sizes become 0; confirm no crash and
// a correct no-op success.
// ---------------------------------------------------------------------------------
TEST_F(HostTableTest, ZeroKeysIsANoOpForFind) {
  EXPECT_EQ(NVE_SUCCESS,
            nve_table_find(table_, ctx_, 0, nullptr, nullptr, sizeof(int64_t), nullptr, nullptr));
}
TEST_F(HostTableTest, ZeroKeysIsANoOpForInsert) {
  EXPECT_EQ(NVE_SUCCESS, nve_table_insert(table_, ctx_, 0, nullptr, sizeof(int64_t),
                                          sizeof(int64_t), nullptr));
}
TEST_F(HostTableTest, ZeroKeysIsANoOpForUpdate) {
  EXPECT_EQ(NVE_SUCCESS, nve_table_update(table_, ctx_, 0, nullptr, sizeof(int64_t),
                                          sizeof(int64_t), nullptr));
}
TEST_F(HostTableTest, ZeroKeysIsANoOpForErase) {
  EXPECT_EQ(NVE_SUCCESS, nve_table_erase(table_, ctx_, 0, nullptr));
}

// ---------------------------------------------------------------------------------
// NULL keys with n > 0 must be rejected with NVE_ERROR_INVALID_ARGUMENT rather than
// dereferenced (keys is the one non-optional buffer of these operations).
// ---------------------------------------------------------------------------------
TEST_F(HostTableTest, NullKeysWithPositiveCountIsRejectedForFind) {
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT,
            nve_table_find(table_, ctx_, 1, nullptr, nullptr, sizeof(int64_t), nullptr, nullptr));
}
TEST_F(HostTableTest, NullKeysWithPositiveCountIsRejectedForInsert) {
  const int64_t value = 1;
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT,
            nve_table_insert(table_, ctx_, 1, nullptr, sizeof(int64_t), sizeof(int64_t), &value));
}
TEST_F(HostTableTest, NullKeysWithPositiveCountIsRejectedForUpdate) {
  const int64_t value = 1;
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT,
            nve_table_update(table_, ctx_, 1, nullptr, sizeof(int64_t), sizeof(int64_t), &value));
}
TEST_F(HostTableTest, NullKeysWithPositiveCountIsRejectedForUpdateAccumulate) {
  const double update = 1.0;
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT,
            nve_table_update_accumulate(table_, ctx_, 1, nullptr, sizeof(double), sizeof(double),
                                        &update, NVE_DTYPE_FLOAT64));
}
TEST_F(HostTableTest, NullKeysWithPositiveCountIsRejectedForErase) {
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_table_erase(table_, ctx_, 1, nullptr));
}

// ---------------------------------------------------------------------------------
// nve_table_find with hit_mask, values, or value_sizes individually NULL (each is
// documented as independently optional): every combination must work standalone.
// ---------------------------------------------------------------------------------
TEST_F(HostTableTest, FindWithOnlyHitMaskNull) {
  const int64_t key = 7;
  const int64_t value = 700;
  ASSERT_EQ(NVE_SUCCESS,
            nve_table_insert(table_, ctx_, 1, &key, sizeof(int64_t), sizeof(int64_t), &value));
  ASSERT_EQ(NVE_SUCCESS, nve_context_wait(ctx_));

  int64_t out_value = -1;
  int64_t value_size = -1;
  EXPECT_EQ(NVE_SUCCESS, nve_table_find(table_, ctx_, 1, &key, /*hit_mask=*/nullptr,
                                        sizeof(int64_t), &out_value, &value_size));
  EXPECT_EQ(value, out_value);
}
TEST_F(HostTableTest, FindWithOnlyValuesNull) {
  const int64_t key = 8;
  const int64_t value = 800;
  ASSERT_EQ(NVE_SUCCESS,
            nve_table_insert(table_, ctx_, 1, &key, sizeof(int64_t), sizeof(int64_t), &value));
  ASSERT_EQ(NVE_SUCCESS, nve_context_wait(ctx_));

  uint64_t hit_mask = 0;
  EXPECT_EQ(NVE_SUCCESS, nve_table_find(table_, ctx_, 1, &key, &hit_mask, sizeof(int64_t),
                                        /*values=*/nullptr, nullptr));
  EXPECT_EQ(1u, hit_mask & 0x1);
}
TEST_F(HostTableTest, FindWithOnlyValueSizesNull) {
  const int64_t key = 9;
  const int64_t value = 900;
  ASSERT_EQ(NVE_SUCCESS,
            nve_table_insert(table_, ctx_, 1, &key, sizeof(int64_t), sizeof(int64_t), &value));
  ASSERT_EQ(NVE_SUCCESS, nve_context_wait(ctx_));

  uint64_t hit_mask = 0;
  int64_t out_value = -1;
  EXPECT_EQ(NVE_SUCCESS, nve_table_find(table_, ctx_, 1, &key, &hit_mask, sizeof(int64_t),
                                        &out_value, /*value_sizes=*/nullptr));
  EXPECT_EQ(value, out_value);
  EXPECT_EQ(1u, hit_mask & 0x1);
}
TEST_F(HostTableTest, FindWithAllThreeOptionalArgsNull) {
  const int64_t key = 10;
  const int64_t value = 1000;
  ASSERT_EQ(NVE_SUCCESS,
            nve_table_insert(table_, ctx_, 1, &key, sizeof(int64_t), sizeof(int64_t), &value));
  ASSERT_EQ(NVE_SUCCESS, nve_context_wait(ctx_));

  EXPECT_EQ(NVE_SUCCESS, nve_table_find(table_, ctx_, 1, &key, nullptr, sizeof(int64_t), nullptr,
                                        nullptr));
}

// Device outputs for a host table: results are copied back and slots this call skips or misses
// keep the caller's contents.
TEST_F(HostTableTest, FindIntoDeviceBuffersKeepsUnresolvedSlots) {
  const std::vector<int64_t> stored_keys{1, 2};
  const std::vector<int64_t> stored_values{100, 200};
  ASSERT_EQ(NVE_SUCCESS, nve_table_insert(table_, ctx_, 2, stored_keys.data(), sizeof(int64_t),
                                          sizeof(int64_t), stored_values.data()));
  ASSERT_EQ(NVE_SUCCESS, nve_context_wait(ctx_));

  // Key 1 is already resolved by the caller, key 2 is found, key 3 is missing.
  const std::vector<int64_t> keys{1, 2, 3};
  uint64_t hit_mask = 0x1;
  std::vector<int64_t> values{555, -1, -1};
  void* d_hit_mask = nullptr;
  void* d_values = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_hit_mask, sizeof(uint64_t)));
  nve::DevicePtr<void> hit_mask_owner(d_hit_mask);
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_values, values.size() * sizeof(int64_t)));
  nve::DevicePtr<void> values_owner(d_values);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_hit_mask, &hit_mask, sizeof(uint64_t), cudaMemcpyDefault));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(d_values, values.data(), values.size() * sizeof(int64_t), cudaMemcpyDefault));

  EXPECT_EQ(NVE_SUCCESS, nve_table_find(table_, ctx_, 3, keys.data(), static_cast<uint64_t*>(d_hit_mask),
                                        sizeof(int64_t), d_values, nullptr));
  ASSERT_EQ(NVE_SUCCESS, nve_context_wait(ctx_));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(&hit_mask, d_hit_mask, sizeof(uint64_t), cudaMemcpyDefault));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(values.data(), d_values, values.size() * sizeof(int64_t), cudaMemcpyDefault));
  EXPECT_EQ(0x2u, hit_mask & 0x6);
  EXPECT_EQ((std::vector<int64_t>{555, 200, -1}), values);
}

// ---------------------------------------------------------------------------------
// nve_table_get_device_id/get_max_row_size/get_key_size/get_lookup_counter each
// called with out == NULL.
// ---------------------------------------------------------------------------------
TEST_F(HostTableTest, GetDeviceIdNullOut) {
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_table_get_device_id(table_, nullptr));
}
TEST_F(HostTableTest, GetMaxRowSizeNullOut) {
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_table_get_max_row_size(table_, nullptr));
}
TEST_F(HostTableTest, GetKeySizeNullOut) {
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_table_get_key_size(table_, nullptr));
}
TEST_F(HostTableTest, GetLookupCounterNullOut) {
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_table_get_lookup_counter(table_, ctx_, nullptr));
}

// ---------------------------------------------------------------------------------
// reset_lookup_counter / get_lookup_counter sequencing: reset reads back as 0; host
// tables count hits, so finding present keys increments the
// counter by the hit count, and finding absent keys leaves it unchanged.
// ---------------------------------------------------------------------------------
TEST_F(HostTableTest, LookupCounterTracksHitsAfterReset) {
  const int64_t keys[3] = {1, 2, 3};
  const int64_t values[3] = {10, 20, 30};
  ASSERT_EQ(NVE_SUCCESS,
            nve_table_insert(table_, ctx_, 3, keys, sizeof(int64_t), sizeof(int64_t), values));
  ASSERT_EQ(NVE_SUCCESS, nve_context_wait(ctx_));

  ASSERT_EQ(NVE_SUCCESS, nve_table_reset_lookup_counter(table_, ctx_));
  int64_t counter = -1;
  ASSERT_EQ(NVE_SUCCESS, nve_table_get_lookup_counter(table_, ctx_, &counter));
  EXPECT_EQ(0, counter);

  int64_t out[3];
  uint64_t hit_mask = 0;
  ASSERT_EQ(NVE_SUCCESS,
            nve_table_find(table_, ctx_, 3, keys, &hit_mask, sizeof(int64_t), out, nullptr));
  ASSERT_EQ(NVE_SUCCESS, nve_table_get_lookup_counter(table_, ctx_, &counter));
  EXPECT_EQ(3, counter) << "all 3 present keys should count as hits";
  EXPECT_EQ(0x7u, hit_mask & 0x7u);

  const int64_t missing_keys[2] = {100, 200};
  int64_t out2[2];
  hit_mask = 0;
  ASSERT_EQ(NVE_SUCCESS, nve_table_find(table_, ctx_, 2, missing_keys, &hit_mask, sizeof(int64_t),
                                        out2, nullptr));
  ASSERT_EQ(NVE_SUCCESS, nve_table_get_lookup_counter(table_, ctx_, &counter));
  EXPECT_EQ(3, counter) << "misses must not change the hit counter";
  EXPECT_EQ(0u, hit_mask & 0x3u);
}

// ---------------------------------------------------------------------------------
// nve_table_size / nve_table_get_keys: argument validation, then the size tracks inserts
// and get_keys fills at most `capacity` keys and reports how many it wrote.
// ---------------------------------------------------------------------------------
TEST_F(HostTableTest, SizeAndGetKeysNullArgs) {
  int64_t out = 0;
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_table_size(table_, ctx_, nullptr));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_table_size(nullptr, ctx_, &out));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_table_get_keys(table_, ctx_, 1, nullptr, &out));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_table_get_keys(table_, ctx_, 1, &out, nullptr));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_table_get_keys(table_, ctx_, -1, &out, &out));
  // Zero capacity needs no buffer.
  EXPECT_EQ(NVE_SUCCESS, nve_table_get_keys(table_, ctx_, 0, nullptr, &out));
  EXPECT_EQ(0, out);
}

// Keys are only written to host memory.
TEST_F(HostTableTest, GetKeysRejectsDeviceBuffer) {
  void* d_keys = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&d_keys, sizeof(int64_t)));
  nve::DevicePtr<void> owner(d_keys);
  int64_t out = 0;
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_table_get_keys(table_, ctx_, 1, d_keys, &out));
}

TEST_F(HostTableTest, SizeAndGetKeysTrackContents) {
  int64_t size = -1;
  ASSERT_EQ(NVE_SUCCESS, nve_table_size(table_, ctx_, &size));
  EXPECT_EQ(0, size);

  const int64_t keys[3] = {1, 2, 3};
  const int64_t values[3] = {10, 20, 30};
  ASSERT_EQ(NVE_SUCCESS,
            nve_table_insert(table_, ctx_, 3, keys, sizeof(int64_t), sizeof(int64_t), values));
  ASSERT_EQ(NVE_SUCCESS, nve_table_size(table_, ctx_, &size));
  EXPECT_EQ(3, size);

  int64_t out[4] = {-1, -1, -1, -1};
  int64_t written = -1;
  ASSERT_EQ(NVE_SUCCESS, nve_table_get_keys(table_, ctx_, 4, out, &written));
  EXPECT_EQ(3, written);
  std::sort(out, out + 3);
  EXPECT_EQ(1, out[0]);
  EXPECT_EQ(2, out[1]);
  EXPECT_EQ(3, out[2]);
  EXPECT_EQ(-1, out[3]) << "slots past the written count must stay untouched";

  // A smaller buffer is filled to capacity.
  ASSERT_EQ(NVE_SUCCESS, nve_table_get_keys(table_, ctx_, 2, out, &written));
  EXPECT_EQ(2, written);
  EXPECT_NE(out[0], out[1]);
}

// ---------------------------------------------------------------------------------
// nve_table_update_accumulate with an update_dtype that doesn't match the table's storage dtype.
// A float32 host table seeded with one row (key 1 == 10.0f); owning handles release
// everything in reverse order even when an ASSERT_* returns early.
// ---------------------------------------------------------------------------------
class UpdateAccumulateDtype : public ::testing::Test {
 protected:
  void SetUp() override {
    nve_thread_pool_t pool = nullptr;
    ASSERT_EQ(NVE_SUCCESS, nve_thread_pool_create(&pool, "{\"name\": \"simple\", \"num_workers\": 1}"));
    pool_.reset(pool);
    nve_table_factory_t factory = nullptr;
    ASSERT_EQ(NVE_SUCCESS,
              nve_create_table_factory(&factory, "libnve-plugin-stl-map.so", "{}"));
    factory_.reset(factory);
    nve_table_t table = nullptr;
    ASSERT_EQ(NVE_SUCCESS,
              nve_table_factory_produce(
                  factory, 0, "{\"value_dtype\": \"float32\", \"max_value_size\": 4}", &table));
    table_.reset(table);
    nve_context_t ctx = nullptr;
    ASSERT_EQ(NVE_SUCCESS,
              nve_table_create_execution_context(table, &ctx, nullptr, nullptr, pool, nullptr));
    ctx_.reset(ctx);

    float value = 10.0f;
    ASSERT_EQ(NVE_SUCCESS,
              nve_table_insert(table, ctx, 1, &key_, sizeof(float), sizeof(float), &value));
    ASSERT_EQ(NVE_SUCCESS, nve_context_wait(ctx));
  }

  const int64_t key_ = 1;
  ThreadPoolPtr pool_;
  TableFactoryPtr factory_;
  TablePtr table_;
  ContextPtr ctx_;
};

TEST_F(UpdateAccumulateDtype, MismatchedButSupportedDtypeConvertsCorrectly) {
  // fp16 bit pattern for 1.0. The table stores float32; the accumulate call declares
  // the update as float16 -- a supported, real dtype conversion, not a bug.
  const uint16_t half_one = 0x3C00;
  ASSERT_EQ(NVE_SUCCESS,
            nve_table_update_accumulate(table_.get(), ctx_.get(), 1, &key_, sizeof(uint16_t),
                                        sizeof(uint16_t), &half_one, NVE_DTYPE_FLOAT16));
  ASSERT_EQ(NVE_SUCCESS, nve_context_wait(ctx_.get()));

  float out_value = -1.0f;
  uint64_t hit_mask = 0;
  ASSERT_EQ(NVE_SUCCESS,
            nve_table_find(table_.get(), ctx_.get(), 1, &key_, &hit_mask, sizeof(float), &out_value,
                           nullptr));
  EXPECT_FLOAT_EQ(11.0f, out_value) << "10.0f (storage) + 1.0 (fp16 update) via correct conversion";
}

TEST_F(UpdateAccumulateDtype, UpdateSizeNotMultipleOfDtypeSizeIsRejected) {
  // 3 bytes is not a multiple of dtype_size(Float32) == 4.
  const uint8_t bogus_update[3] = {1, 2, 3};
  EXPECT_NE(NVE_SUCCESS, nve_table_update_accumulate(table_.get(), ctx_.get(), 1, &key_, 3, 3,
                                                     bogus_update, NVE_DTYPE_FLOAT32));
}
