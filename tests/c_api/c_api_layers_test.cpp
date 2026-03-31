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
#include <nve_c_api.h>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <cmath>

/* ============================================================================
 * Helpers
 * ============================================================================ */

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err = (call);                                               \
    ASSERT_EQ(cudaSuccess, err) << "CUDA error: " << cudaGetErrorString(err); \
  } while (0)

#define NVE_CHECK(call)                              \
  do {                                               \
    nve_status_t st = (call);                        \
    if (st != NVE_SUCCESS) {                         \
      const char* msg = nullptr;                     \
      nve_get_last_error(&msg);                      \
      FAIL() << "NVE error " << st << ": " << msg;  \
    }                                                \
  } while (0)

static constexpr int DEVICE_ID = 0;
static constexpr int64_t ROW_SIZE = 128;  // bytes per embedding row (32 floats)
static constexpr int64_t NUM_FLOATS = ROW_SIZE / sizeof(float);

// Fill host memory with a pattern: row i gets value (base + i) in each float
static void fill_host_data(float* data, int64_t num_rows, int64_t floats_per_row, float base) {
  for (int64_t r = 0; r < num_rows; ++r) {
    for (int64_t c = 0; c < floats_per_row; ++c) {
      data[r * floats_per_row + c] = base + static_cast<float>(r);
    }
  }
}

/* ============================================================================
 * GPU Embedding Layer Tests
 * ============================================================================ */

class GpuEmbeddingLayerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDA_CHECK(cudaSetDevice(DEVICE_ID));

    num_embeddings_ = 1024;
    const size_t table_bytes = static_cast<size_t>(num_embeddings_ * ROW_SIZE);

    // Allocate and fill host table
    CUDA_CHECK(cudaMallocHost(&h_table_, table_bytes));
    fill_host_data(static_cast<float*>(h_table_), num_embeddings_, NUM_FLOATS, 1.0f);

    // Copy to GPU
    CUDA_CHECK(cudaMalloc(&d_table_, table_bytes));
    CUDA_CHECK(cudaMemcpy(d_table_, h_table_, table_bytes, cudaMemcpyHostToDevice));

    // Create layer
    auto cfg = nve_gpu_embedding_layer_config_default();
    cfg.device_id = DEVICE_ID;
    cfg.layer_name = "test_gpu_layer";
    cfg.embedding_table = d_table_;
    cfg.num_embeddings = num_embeddings_;
    cfg.embedding_width_in_bytes = ROW_SIZE;
    cfg.value_dtype = NVE_DTYPE_FLOAT32;

    NVE_CHECK(nve_gpu_embedding_layer_create(&layer_, NVE_KEY_INT64, &cfg, nullptr));
    NVE_CHECK(nve_layer_create_execution_context(layer_, &ctx_, nullptr, nullptr, nullptr, nullptr));
  }

  void TearDown() override {
    if (ctx_) {
      nve_context_wait(ctx_);
      nve_context_destroy(ctx_);
    }
    if (layer_) nve_layer_destroy(layer_);
    if (d_table_) cudaFree(d_table_);
    if (h_table_) cudaFreeHost(h_table_);
  }

  int64_t num_embeddings_ = 0;
  void* h_table_ = nullptr;
  void* d_table_ = nullptr;
  nve_layer_t layer_ = nullptr;
  nve_context_t ctx_ = nullptr;
};

TEST_F(GpuEmbeddingLayerTest, LookupSingleKey) {
  const int64_t num_keys = 1;
  int64_t h_key = 5;
  void* d_keys = nullptr;
  void* d_output = nullptr;

  CUDA_CHECK(cudaMalloc(&d_keys, sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_output, ROW_SIZE));
  CUDA_CHECK(cudaMemcpy(d_keys, &h_key, sizeof(int64_t), cudaMemcpyHostToDevice));

  float hitrate = 0.0f;
  NVE_CHECK(nve_layer_lookup(layer_, ctx_, num_keys, d_keys, d_output, ROW_SIZE, nullptr, &hitrate));
  NVE_CHECK(nve_context_wait(ctx_));

  // Verify output
  std::vector<float> h_output(NUM_FLOATS);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, ROW_SIZE, cudaMemcpyDeviceToHost));

  // Row 5 should have value 1.0 + 5 = 6.0 in each float
  for (uint64_t i = 0; i < NUM_FLOATS; ++i) {
    EXPECT_FLOAT_EQ(6.0f, h_output[i]) << "Mismatch at float index " << i;
  }

  EXPECT_FLOAT_EQ(1.0f, hitrate);

  cudaFree(d_keys);
  cudaFree(d_output);
}

TEST_F(GpuEmbeddingLayerTest, LookupMultipleKeys) {
  const int64_t num_keys = 4;
  std::vector<int64_t> h_keys = {0, 10, 100, 500};
  void* d_keys = nullptr;
  void* d_output = nullptr;

  CUDA_CHECK(cudaMalloc(&d_keys, num_keys * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_output, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(int64_t), cudaMemcpyHostToDevice));

  NVE_CHECK(nve_layer_lookup(layer_, ctx_, num_keys, d_keys, d_output, ROW_SIZE, nullptr, nullptr));
  NVE_CHECK(nve_context_wait(ctx_));

  std::vector<float> h_output(num_keys * NUM_FLOATS);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_keys * ROW_SIZE, cudaMemcpyDeviceToHost));

  for (uint64_t k = 0; k < num_keys; ++k) {
    float expected = 1.0f + static_cast<float>(h_keys[k]);
    EXPECT_FLOAT_EQ(expected, h_output[k * NUM_FLOATS]) << "Key " << h_keys[k];
  }

  cudaFree(d_keys);
  cudaFree(d_output);
}

TEST_F(GpuEmbeddingLayerTest, UpdateAndLookup) {
  const int64_t num_keys = 2;
  std::vector<int64_t> h_keys = {3, 7};
  void* d_keys = nullptr;
  void* d_values = nullptr;
  void* d_output = nullptr;

  CUDA_CHECK(cudaMalloc(&d_keys, num_keys * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_values, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMalloc(&d_output, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(int64_t), cudaMemcpyHostToDevice));

  // Write new values: 99.0 for all
  std::vector<float> new_values(num_keys * NUM_FLOATS, 99.0f);
  CUDA_CHECK(cudaMemcpy(d_values, new_values.data(), num_keys * ROW_SIZE, cudaMemcpyHostToDevice));

  NVE_CHECK(nve_layer_update(layer_, ctx_, num_keys, d_keys, ROW_SIZE, ROW_SIZE, d_values));
  NVE_CHECK(nve_context_wait(ctx_));

  // Lookup the updated keys
  NVE_CHECK(nve_layer_lookup(layer_, ctx_, num_keys, d_keys, d_output, ROW_SIZE, nullptr, nullptr));
  NVE_CHECK(nve_context_wait(ctx_));

  std::vector<float> h_output(num_keys * NUM_FLOATS);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_keys * ROW_SIZE, cudaMemcpyDeviceToHost));

  for (uint64_t i = 0; i < num_keys * NUM_FLOATS; ++i) {
    EXPECT_FLOAT_EQ(99.0f, h_output[i]) << "Mismatch at index " << i;
  }

  cudaFree(d_keys);
  cudaFree(d_values);
  cudaFree(d_output);
}

TEST_F(GpuEmbeddingLayerTest, Clear) {
  // Just verify clear doesn't crash -- the layer has a linear backing table
  // so clear is a valid operation
  NVE_CHECK(nve_layer_clear(layer_, ctx_));
  NVE_CHECK(nve_context_wait(ctx_));
}

/* ============================================================================
 * GPU Table Tests
 * ============================================================================ */

class GpuTableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDA_CHECK(cudaSetDevice(DEVICE_ID));

    const int64_t num_rows = 256;
    const size_t uvm_bytes = static_cast<size_t>(num_rows * ROW_SIZE);
    CUDA_CHECK(cudaMallocHost(&h_uvm_table_, uvm_bytes));
    memset(h_uvm_table_, 0, uvm_bytes);

    auto cfg = nve_gpu_table_config_default();
    cfg.device_id = DEVICE_ID;
    cfg.cache_size = 1 << 20;  // 1MB cache
    cfg.row_size_in_bytes = ROW_SIZE;
    cfg.uvm_table = h_uvm_table_;
    cfg.count_misses = 1;
    cfg.value_dtype = NVE_DTYPE_FLOAT32;

    NVE_CHECK(nve_gpu_table_create(&table_, NVE_KEY_INT64, &cfg, nullptr));
    NVE_CHECK(nve_table_create_execution_context(table_, &ctx_, nullptr, nullptr, nullptr, nullptr));
  }

  void TearDown() override {
    if (ctx_) {
      nve_context_wait(ctx_);
      nve_context_destroy(ctx_);
    }
    if (table_) nve_table_destroy(table_);
    if (h_uvm_table_) cudaFreeHost(h_uvm_table_);
  }

  void* h_uvm_table_ = nullptr;
  nve_table_t table_ = nullptr;
  nve_context_t ctx_ = nullptr;
};

TEST_F(GpuTableTest, Properties) {
  int32_t device_id = -1;
  NVE_CHECK(nve_table_get_device_id(table_, &device_id));
  EXPECT_EQ(DEVICE_ID, device_id);

  int64_t max_row = 0;
  NVE_CHECK(nve_table_get_max_row_size(table_, &max_row));
  EXPECT_EQ(ROW_SIZE, max_row);
}

TEST_F(GpuTableTest, InsertFindErase) {
  const uint64_t num_keys = 8;

  std::vector<int64_t> h_keys(num_keys);
  std::iota(h_keys.begin(), h_keys.end(), 100);

  // Values on device
  std::vector<float> h_values(num_keys * NUM_FLOATS);
  for (uint64_t r = 0; r < num_keys; ++r) {
    for (uint64_t c = 0; c < NUM_FLOATS; ++c) {
      h_values[r * NUM_FLOATS + c] = static_cast<float>(r + 1) * 10.0f;
    }
  }

  // All buffers on device (modify_on_gpu=true by default)
  void* d_keys = nullptr;
  void* d_values = nullptr;
  void* d_output = nullptr;
  void* d_hitmask = nullptr;
  const size_t hitmask_size = ((num_keys + 63) / 64) * sizeof(uint64_t);

  CUDA_CHECK(cudaMalloc(&d_keys, num_keys * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_values, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMalloc(&d_output, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMalloc(&d_hitmask, hitmask_size));

  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), num_keys * ROW_SIZE, cudaMemcpyHostToDevice));

  // Insert (keys on device for modify_on_gpu=true)
  NVE_CHECK(nve_table_insert(table_, ctx_, num_keys, d_keys, ROW_SIZE, ROW_SIZE, d_values));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Reset lookup counter, then find
  NVE_CHECK(nve_table_reset_lookup_counter(table_, ctx_));
  CUDA_CHECK(cudaMemset(d_hitmask, 0, hitmask_size));
  CUDA_CHECK(cudaMemset(d_output, 0, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaDeviceSynchronize());

  NVE_CHECK(nve_table_find(table_, ctx_, num_keys, d_keys,
                           static_cast<uint64_t*>(d_hitmask), ROW_SIZE,
                           d_output, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());

  // With UVM backing, hitmask is not reliable -- use lookup counter instead
  // GPU table counts misses, so 0 misses = all found
  int64_t miss_count = -1;
  NVE_CHECK(nve_table_get_lookup_counter(table_, ctx_, &miss_count));
  CUDA_CHECK(cudaDeviceSynchronize());
  EXPECT_EQ(0, miss_count) << "Expected 0 misses after insert";

  // Verify output values
  std::vector<float> h_output(num_keys * NUM_FLOATS);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_keys * ROW_SIZE, cudaMemcpyDeviceToHost));
  for (uint64_t r = 0; r < num_keys; ++r) {
    EXPECT_FLOAT_EQ(static_cast<float>(r + 1) * 10.0f, h_output[r * NUM_FLOATS])
        << "Row " << r;
  }

  // Erase (keys on device for modify_on_gpu=true)
  NVE_CHECK(nve_table_erase(table_, ctx_, num_keys, d_keys));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Find again after erase -- with UVM backing, keys will still resolve from UVM
  // so we verify via the lookup counter (misses should increase since cache was erased)
  NVE_CHECK(nve_table_reset_lookup_counter(table_, ctx_));
  CUDA_CHECK(cudaMemset(d_hitmask, 0, hitmask_size));
  CUDA_CHECK(cudaDeviceSynchronize());

  NVE_CHECK(nve_table_find(table_, ctx_, num_keys, d_keys,
                           static_cast<uint64_t*>(d_hitmask), ROW_SIZE,
                           d_output, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());

  int64_t miss_count_after_erase = 0;
  NVE_CHECK(nve_table_get_lookup_counter(table_, ctx_, &miss_count_after_erase));
  CUDA_CHECK(cudaDeviceSynchronize());
  // After erase, keys should miss in cache (resolved from UVM but counted as cache misses)
  EXPECT_NE(0, miss_count_after_erase) << "Expected cache misses after erase";

  cudaFree(d_keys);
  cudaFree(d_values);
  cudaFree(d_output);
  cudaFree(d_hitmask);
}

TEST_F(GpuTableTest, InsertUpdateFind) {
  const int64_t num_keys = 4;
  std::vector<int64_t> h_keys = {200, 201, 202, 203};

  void* d_keys = nullptr;
  void* d_values = nullptr;
  void* d_output = nullptr;
  void* d_hitmask = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, num_keys * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_values, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMalloc(&d_output, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMalloc(&d_hitmask, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(int64_t), cudaMemcpyHostToDevice));

  // Initial values
  std::vector<float> h_values(num_keys * NUM_FLOATS, 1.0f);
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), num_keys * ROW_SIZE, cudaMemcpyHostToDevice));

  NVE_CHECK(nve_table_insert(table_, ctx_, num_keys, d_keys, ROW_SIZE, ROW_SIZE, d_values));
  NVE_CHECK(nve_context_wait(ctx_));

  // Update with new values
  std::vector<float> h_updates(num_keys * NUM_FLOATS, 42.0f);
  CUDA_CHECK(cudaMemcpy(d_values, h_updates.data(), num_keys * ROW_SIZE, cudaMemcpyHostToDevice));

  NVE_CHECK(nve_table_update(table_, ctx_, num_keys, d_keys, ROW_SIZE, ROW_SIZE, d_values));
  NVE_CHECK(nve_context_wait(ctx_));

  // Find and verify
  CUDA_CHECK(cudaMemset(d_hitmask, 0, sizeof(uint64_t)));
  NVE_CHECK(nve_table_find(table_, ctx_, num_keys, d_keys,
                           static_cast<uint64_t*>(d_hitmask), ROW_SIZE,
                           d_output, nullptr));
  NVE_CHECK(nve_context_wait(ctx_));

  std::vector<float> h_output(num_keys * NUM_FLOATS);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_keys * ROW_SIZE, cudaMemcpyDeviceToHost));

  for (uint64_t i = 0; i < num_keys * NUM_FLOATS; ++i) {
    EXPECT_FLOAT_EQ(42.0f, h_output[i]) << "Mismatch at index " << i;
  }

  cudaFree(d_keys);
  cudaFree(d_values);
  cudaFree(d_output);
  cudaFree(d_hitmask);
}

TEST_F(GpuTableTest, ClearTable) {
  const int64_t num_keys = 4;
  std::vector<int64_t> h_keys = {300, 301, 302, 303};

  void* d_keys = nullptr;
  void* d_values = nullptr;
  void* d_hitmask = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, num_keys * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_values, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMalloc(&d_hitmask, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(int64_t), cudaMemcpyHostToDevice));

  std::vector<float> h_values(num_keys * NUM_FLOATS, 5.0f);
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), num_keys * ROW_SIZE, cudaMemcpyHostToDevice));

  NVE_CHECK(nve_table_insert(table_, ctx_, num_keys, d_keys, ROW_SIZE, ROW_SIZE, d_values));
  NVE_CHECK(nve_context_wait(ctx_));

  // Clear
  NVE_CHECK(nve_table_clear(table_, ctx_));
  NVE_CHECK(nve_context_wait(ctx_));

  // Find should miss
  CUDA_CHECK(cudaMemset(d_hitmask, 0, sizeof(uint64_t)));
  NVE_CHECK(nve_table_find(table_, ctx_, num_keys, d_keys,
                           static_cast<uint64_t*>(d_hitmask), ROW_SIZE,
                           d_values, nullptr));
  NVE_CHECK(nve_context_wait(ctx_));

  uint64_t h_hitmask = 0;
  CUDA_CHECK(cudaMemcpy(&h_hitmask, d_hitmask, sizeof(uint64_t), cudaMemcpyDeviceToHost));
  EXPECT_EQ(0ULL, h_hitmask);

  cudaFree(d_keys);
  cudaFree(d_values);
  cudaFree(d_hitmask);
}

/* ============================================================================
 * GPU Table with int32 keys
 * ============================================================================ */

TEST(GpuTableInt32, InsertAndFind) {
  CUDA_CHECK(cudaSetDevice(DEVICE_ID));

  void* h_uvm = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_uvm, 256 * ROW_SIZE));
  memset(h_uvm, 0, 256 * ROW_SIZE);

  auto cfg = nve_gpu_table_config_default();
  cfg.device_id = DEVICE_ID;
  cfg.cache_size = 1 << 20;
  cfg.row_size_in_bytes = ROW_SIZE;
  cfg.uvm_table = h_uvm;
  cfg.value_dtype = NVE_DTYPE_FLOAT32;

  nve_table_t table = nullptr;
  NVE_CHECK(nve_gpu_table_create(&table, NVE_KEY_INT32, &cfg, nullptr));

  nve_context_t ctx = nullptr;
  NVE_CHECK(nve_table_create_execution_context(table, &ctx, nullptr, nullptr, nullptr, nullptr));

  const int64_t num_keys = 4;
  std::vector<int32_t> h_keys = {10, 20, 30, 40};
  std::vector<float> h_values(num_keys * NUM_FLOATS, 7.0f);

  void* d_keys = nullptr;
  void* d_values = nullptr;
  void* d_output = nullptr;
  void* d_hitmask = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, num_keys * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_values, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMalloc(&d_output, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMalloc(&d_hitmask, sizeof(uint64_t)));

  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(int32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), num_keys * ROW_SIZE, cudaMemcpyHostToDevice));

  // Insert (keys on device for modify_on_gpu=true)
  NVE_CHECK(nve_table_insert(table, ctx, num_keys, d_keys, ROW_SIZE, ROW_SIZE, d_values));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Find
  NVE_CHECK(nve_table_reset_lookup_counter(table, ctx));
  CUDA_CHECK(cudaMemset(d_hitmask, 0, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(d_output, 0, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaDeviceSynchronize());

  NVE_CHECK(nve_table_find(table, ctx, num_keys, d_keys,
                           static_cast<uint64_t*>(d_hitmask), ROW_SIZE,
                           d_output, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());

  // With UVM backing, use lookup counter: GPU table counts misses, 0 = all found
  int64_t miss_count = -1;
  NVE_CHECK(nve_table_get_lookup_counter(table, ctx, &miss_count));
  CUDA_CHECK(cudaDeviceSynchronize());
  EXPECT_EQ(0, miss_count) << "Expected 0 misses after insert";

  std::vector<float> h_output(num_keys * NUM_FLOATS);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_keys * ROW_SIZE, cudaMemcpyDeviceToHost));
  for (uint64_t i = 0; i < num_keys * NUM_FLOATS; ++i) {
    EXPECT_FLOAT_EQ(7.0f, h_output[i]);
  }

  nve_context_wait(ctx);
  nve_context_destroy(ctx);
  nve_table_destroy(table);
  cudaFree(d_keys);
  cudaFree(d_values);
  cudaFree(d_output);
  cudaFree(d_hitmask);
  cudaFreeHost(h_uvm);
}

/* ============================================================================
 * Linear UVM Embedding Layer Tests
 * ============================================================================ */

class LinearUvmLayerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDA_CHECK(cudaSetDevice(DEVICE_ID));

    num_rows_ = 1024;
    const size_t table_bytes = static_cast<size_t>(num_rows_ * ROW_SIZE);

    // Allocate pinned host memory as UVM backing store
    CUDA_CHECK(cudaMallocHost(&h_uvm_table_, table_bytes));
    fill_host_data(static_cast<float*>(h_uvm_table_), num_rows_, NUM_FLOATS, 0.0f);

    // Create GPU table with UVM backing
    auto gpu_cfg = nve_gpu_table_config_default();
    gpu_cfg.device_id = DEVICE_ID;
    gpu_cfg.cache_size = 1 << 20;  // 1MB GPU cache
    gpu_cfg.row_size_in_bytes = ROW_SIZE;
    gpu_cfg.uvm_table = h_uvm_table_;
    gpu_cfg.count_misses = 1;
    gpu_cfg.value_dtype = NVE_DTYPE_FLOAT32;

    NVE_CHECK(nve_gpu_table_create(&gpu_table_, NVE_KEY_INT64, &gpu_cfg, nullptr));

    // Create Linear UVM layer wrapping the GPU table
    auto layer_cfg = nve_linear_uvm_layer_config_default();
    layer_cfg.layer_name = "test_uvm_layer";

    NVE_CHECK(nve_linear_uvm_layer_create(&layer_, NVE_KEY_INT64, &layer_cfg, gpu_table_, nullptr));
    NVE_CHECK(nve_layer_create_execution_context(layer_, &ctx_, nullptr, nullptr, nullptr, nullptr));
  }

  void TearDown() override {
    if (ctx_) {
      nve_context_wait(ctx_);
      nve_context_destroy(ctx_);
    }
    if (layer_) nve_layer_destroy(layer_);
    if (gpu_table_) nve_table_destroy(gpu_table_);
    if (h_uvm_table_) cudaFreeHost(h_uvm_table_);
  }

  int64_t num_rows_ = 0;
  void* h_uvm_table_ = nullptr;
  nve_table_t gpu_table_ = nullptr;
  nve_layer_t layer_ = nullptr;
  nve_context_t ctx_ = nullptr;
};

TEST_F(LinearUvmLayerTest, LookupFromUvm) {
  // Keys that aren't in GPU cache should be resolved from UVM backing
  const int64_t num_keys = 4;
  std::vector<int64_t> h_keys = {0, 1, 2, 3};

  void* d_keys = nullptr;
  void* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, num_keys * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_output, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(int64_t), cudaMemcpyHostToDevice));

  float hitrates[2] = {0.0f, 0.0f};
  NVE_CHECK(nve_layer_lookup(layer_, ctx_, num_keys, d_keys, d_output, ROW_SIZE, nullptr, hitrates));
  NVE_CHECK(nve_context_wait(ctx_));

  // Verify output matches UVM backing store
  std::vector<float> h_output(num_keys * NUM_FLOATS);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_keys * ROW_SIZE, cudaMemcpyDeviceToHost));

  for (uint64_t k = 0; k < num_keys; ++k) {
    float expected = static_cast<float>(h_keys[k]);  // base=0.0 + row
    EXPECT_FLOAT_EQ(expected, h_output[k * NUM_FLOATS]) << "Key " << h_keys[k];
  }

  cudaFree(d_keys);
  cudaFree(d_output);
}

TEST_F(LinearUvmLayerTest, InsertAndLookup) {
  const int64_t num_keys = 4;
  std::vector<int64_t> h_keys = {500, 501, 502, 503};

  // Insert with specific values into the GPU cache (table_id=0)
  std::vector<float> h_values(num_keys * NUM_FLOATS, 77.0f);
  void* d_keys = nullptr;
  void* d_values = nullptr;
  void* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, num_keys * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_values, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMalloc(&d_output, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), num_keys * ROW_SIZE, cudaMemcpyHostToDevice));

  NVE_CHECK(nve_layer_insert(layer_, ctx_, num_keys, d_keys, ROW_SIZE, ROW_SIZE, d_values, 0));
  NVE_CHECK(nve_context_wait(ctx_));

  // Lookup should find them in GPU cache
  NVE_CHECK(nve_layer_lookup(layer_, ctx_, num_keys, d_keys, d_output, ROW_SIZE, nullptr, nullptr));
  NVE_CHECK(nve_context_wait(ctx_));

  std::vector<float> h_output(num_keys * NUM_FLOATS);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_keys * ROW_SIZE, cudaMemcpyDeviceToHost));

  for (uint64_t i = 0; i < num_keys * NUM_FLOATS; ++i) {
    EXPECT_FLOAT_EQ(77.0f, h_output[i]);
  }

  cudaFree(d_keys);
  cudaFree(d_values);
  cudaFree(d_output);
}

TEST_F(LinearUvmLayerTest, EraseAndClear) {
  const int64_t num_keys = 2;
  std::vector<int64_t> h_keys = {600, 601};

  void* d_keys = nullptr;
  std::vector<float> h_values(num_keys * NUM_FLOATS, 55.0f);
  void* d_values = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, num_keys * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_values, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), num_keys * ROW_SIZE, cudaMemcpyHostToDevice));

  // Insert then erase
  NVE_CHECK(nve_layer_insert(layer_, ctx_, num_keys, d_keys, ROW_SIZE, ROW_SIZE, d_values, 0));
  NVE_CHECK(nve_context_wait(ctx_));

  NVE_CHECK(nve_layer_erase(layer_, ctx_, num_keys, d_keys, 0));
  NVE_CHECK(nve_context_wait(ctx_));

  // Clear all
  NVE_CHECK(nve_layer_clear(layer_, ctx_));
  NVE_CHECK(nve_context_wait(ctx_));

  cudaFree(d_keys);
  cudaFree(d_values);
}

TEST_F(LinearUvmLayerTest, UpdateAccumulate) {
  const int64_t num_keys = 2;
  std::vector<int64_t> h_keys = {10, 11};

  void* d_keys = nullptr;
  void* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, num_keys * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_output, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(int64_t), cudaMemcpyHostToDevice));

  // First lookup to populate cache from UVM
  NVE_CHECK(nve_layer_lookup(layer_, ctx_, num_keys, d_keys, d_output, ROW_SIZE, nullptr, nullptr));
  NVE_CHECK(nve_context_wait(ctx_));

  // Accumulate gradients
  std::vector<float> h_grads(num_keys * NUM_FLOATS, 1.0f);
  void* d_grads = nullptr;
  CUDA_CHECK(cudaMalloc(&d_grads, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMemcpy(d_grads, h_grads.data(), num_keys * ROW_SIZE, cudaMemcpyHostToDevice));

  NVE_CHECK(nve_layer_accumulate(layer_, ctx_, num_keys, d_keys,
                                 ROW_SIZE, ROW_SIZE, d_grads, NVE_DTYPE_FLOAT32));
  NVE_CHECK(nve_context_wait(ctx_));

  // Lookup again to verify accumulation
  NVE_CHECK(nve_layer_lookup(layer_, ctx_, num_keys, d_keys, d_output, ROW_SIZE, nullptr, nullptr));
  NVE_CHECK(nve_context_wait(ctx_));

  std::vector<float> h_output(num_keys * NUM_FLOATS);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_keys * ROW_SIZE, cudaMemcpyDeviceToHost));

  // Values should be original + gradient
  for (uint64_t k = 0; k < num_keys; ++k) {
    float original = static_cast<float>(h_keys[k]);  // from UVM fill_host_data(base=0)
    float expected = original + 1.0f;
    EXPECT_FLOAT_EQ(expected, h_output[k * NUM_FLOATS])
        << "Key " << h_keys[k];
  }

  cudaFree(d_keys);
  cudaFree(d_output);
  cudaFree(d_grads);
}

/* ============================================================================
 * Hierarchical Embedding Layer Tests (GPU-only, 1 table)
 * ============================================================================ */

class HierarchicalLayerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDA_CHECK(cudaSetDevice(DEVICE_ID));

    const size_t uvm_bytes = static_cast<size_t>(512 * ROW_SIZE);
    CUDA_CHECK(cudaMallocHost(&h_uvm_table_, uvm_bytes));
    fill_host_data(static_cast<float*>(h_uvm_table_), 512, NUM_FLOATS, 100.0f);

    auto gpu_cfg = nve_gpu_table_config_default();
    gpu_cfg.device_id = DEVICE_ID;
    gpu_cfg.cache_size = 1 << 20;
    gpu_cfg.row_size_in_bytes = ROW_SIZE;
    gpu_cfg.uvm_table = h_uvm_table_;
    gpu_cfg.count_misses = 1;
    gpu_cfg.value_dtype = NVE_DTYPE_FLOAT32;

    NVE_CHECK(nve_gpu_table_create(&gpu_table_, NVE_KEY_INT64, &gpu_cfg, nullptr));

    // Create hierarchical layer with single GPU table
    auto hier_cfg = nve_hierarchical_layer_config_default();
    hier_cfg.layer_name = "test_hier_layer";

    nve_table_t tables[] = {gpu_table_};
    NVE_CHECK(nve_hierarchical_layer_create(&layer_, NVE_KEY_INT64, &hier_cfg, tables, 1, nullptr));
    NVE_CHECK(nve_layer_create_execution_context(layer_, &ctx_, nullptr, nullptr, nullptr, nullptr));
  }

  void TearDown() override {
    if (ctx_) {
      nve_context_wait(ctx_);
      nve_context_destroy(ctx_);
    }
    if (layer_) nve_layer_destroy(layer_);
    if (gpu_table_) nve_table_destroy(gpu_table_);
    if (h_uvm_table_) cudaFreeHost(h_uvm_table_);
  }

  void* h_uvm_table_ = nullptr;
  nve_table_t gpu_table_ = nullptr;
  nve_layer_t layer_ = nullptr;
  nve_context_t ctx_ = nullptr;
};

TEST_F(HierarchicalLayerTest, LookupInsertUpdateClearErase) {
  const int64_t num_keys = 4;
  std::vector<int64_t> h_keys = {0, 1, 2, 3};

  void* d_keys = nullptr;
  void* d_output = nullptr;
  void* d_values = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, num_keys * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_output, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMalloc(&d_values, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(int64_t), cudaMemcpyHostToDevice));

  // Lookup -- should resolve from UVM backing store through GPU table
  float hitrates[1] = {0.0f};
  NVE_CHECK(nve_layer_lookup(layer_, ctx_, num_keys, d_keys, d_output, ROW_SIZE, nullptr, hitrates));
  NVE_CHECK(nve_context_wait(ctx_));

  std::vector<float> h_output(num_keys * NUM_FLOATS);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_keys * ROW_SIZE, cudaMemcpyDeviceToHost));

  // UVM was filled with base=100.0, so row 0 = 100.0, row 1 = 101.0, ...
  for (u_char k = 0; k < num_keys; ++k) {
    float expected = 100.0f + static_cast<float>(h_keys[k]);
    EXPECT_FLOAT_EQ(expected, h_output[k * NUM_FLOATS]) << "Key " << h_keys[k];
  }

  // Insert new keys
  std::vector<float> h_new_vals(num_keys * NUM_FLOATS, 200.0f);
  CUDA_CHECK(cudaMemcpy(d_values, h_new_vals.data(), num_keys * ROW_SIZE, cudaMemcpyHostToDevice));
  NVE_CHECK(nve_layer_insert(layer_, ctx_, num_keys, d_keys, ROW_SIZE, ROW_SIZE, d_values, 0));
  NVE_CHECK(nve_context_wait(ctx_));

  // Update
  std::vector<float> h_upd_vals(num_keys * NUM_FLOATS, 300.0f);
  CUDA_CHECK(cudaMemcpy(d_values, h_upd_vals.data(), num_keys * ROW_SIZE, cudaMemcpyHostToDevice));
  NVE_CHECK(nve_layer_update(layer_, ctx_, num_keys, d_keys, ROW_SIZE, ROW_SIZE, d_values));
  NVE_CHECK(nve_context_wait(ctx_));

  // Verify update took effect
  NVE_CHECK(nve_layer_lookup(layer_, ctx_, num_keys, d_keys, d_output, ROW_SIZE, nullptr, nullptr));
  NVE_CHECK(nve_context_wait(ctx_));

  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_keys * ROW_SIZE, cudaMemcpyDeviceToHost));
  for (uint64_t i = 0; i < num_keys * NUM_FLOATS; ++i) {
    EXPECT_FLOAT_EQ(300.0f, h_output[i]);
  }

  // Erase
  NVE_CHECK(nve_layer_erase(layer_, ctx_, num_keys, d_keys, 0));
  NVE_CHECK(nve_context_wait(ctx_));

  // Clear
  NVE_CHECK(nve_layer_clear(layer_, ctx_));
  NVE_CHECK(nve_context_wait(ctx_));

  cudaFree(d_keys);
  cudaFree(d_output);
  cudaFree(d_values);
}

TEST_F(HierarchicalLayerTest, Accumulate) {
  const int64_t num_keys = 2;
  std::vector<int64_t> h_keys = {10, 11};

  void* d_keys = nullptr;
  void* d_output = nullptr;
  void* d_grads = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, num_keys * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_output, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMalloc(&d_grads, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(int64_t), cudaMemcpyHostToDevice));

  // Insert values first
  std::vector<float> h_vals(num_keys * NUM_FLOATS, 50.0f);
  void* d_vals = nullptr;
  CUDA_CHECK(cudaMalloc(&d_vals, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMemcpy(d_vals, h_vals.data(), num_keys * ROW_SIZE, cudaMemcpyHostToDevice));

  NVE_CHECK(nve_layer_insert(layer_, ctx_, num_keys, d_keys, ROW_SIZE, ROW_SIZE, d_vals, 0));
  NVE_CHECK(nve_context_wait(ctx_));

  // Accumulate gradient of 10.0
  std::vector<float> h_grads_data(num_keys * NUM_FLOATS, 10.0f);
  CUDA_CHECK(cudaMemcpy(d_grads, h_grads_data.data(), num_keys * ROW_SIZE, cudaMemcpyHostToDevice));

  NVE_CHECK(nve_layer_accumulate(layer_, ctx_, num_keys, d_keys,
                                 ROW_SIZE, ROW_SIZE, d_grads, NVE_DTYPE_FLOAT32));
  NVE_CHECK(nve_context_wait(ctx_));

  // Verify: should be 50.0 + 10.0 = 60.0
  NVE_CHECK(nve_layer_lookup(layer_, ctx_, num_keys, d_keys, d_output, ROW_SIZE, nullptr, nullptr));
  NVE_CHECK(nve_context_wait(ctx_));

  std::vector<float> h_output(num_keys * NUM_FLOATS);
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_keys * ROW_SIZE, cudaMemcpyDeviceToHost));

  for (uint64_t i = 0; i < num_keys * NUM_FLOATS; ++i) {
    EXPECT_FLOAT_EQ(60.0f, h_output[i]);
  }

  cudaFree(d_keys);
  cudaFree(d_output);
  cudaFree(d_grads);
  cudaFree(d_vals);
}

/* ============================================================================
 * Layer with heuristic
 * ============================================================================ */

TEST(LayerWithHeuristic, LinearUvmWithDefaultHeuristic) {
  CUDA_CHECK(cudaSetDevice(DEVICE_ID));

  void* h_uvm = nullptr;
  const int64_t num_rows = 256;
  CUDA_CHECK(cudaMallocHost(&h_uvm, num_rows * ROW_SIZE));
  fill_host_data(static_cast<float*>(h_uvm), num_rows, NUM_FLOATS, 0.0f);

  auto gpu_cfg = nve_gpu_table_config_default();
  gpu_cfg.device_id = DEVICE_ID;
  gpu_cfg.cache_size = 1 << 20;
  gpu_cfg.row_size_in_bytes = ROW_SIZE;
  gpu_cfg.uvm_table = h_uvm;
  gpu_cfg.count_misses = 1;
  gpu_cfg.value_dtype = NVE_DTYPE_FLOAT32;

  nve_table_t gpu_table = nullptr;
  NVE_CHECK(nve_gpu_table_create(&gpu_table, NVE_KEY_INT64, &gpu_cfg, nullptr));

  // Create heuristic
  float thresholds[] = {0.5f};
  nve_heuristic_t heuristic = nullptr;
  NVE_CHECK(nve_heuristic_create_default(&heuristic, thresholds, 1));

  auto layer_cfg = nve_linear_uvm_layer_config_default();
  layer_cfg.layer_name = "uvm_with_heuristic";
  layer_cfg.insert_heuristic = heuristic;
  layer_cfg.min_insert_size_gpu = 2;

  nve_layer_t layer = nullptr;
  NVE_CHECK(nve_linear_uvm_layer_create(&layer, NVE_KEY_INT64, &layer_cfg, gpu_table, nullptr));

  nve_context_t ctx = nullptr;
  NVE_CHECK(nve_layer_create_execution_context(layer, &ctx, nullptr, nullptr, nullptr, nullptr));

  // Perform a lookup
  const int64_t num_keys = 4;
  std::vector<int64_t> h_keys = {0, 1, 2, 3};
  void* d_keys = nullptr;
  void* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_keys, num_keys * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_output, num_keys * ROW_SIZE));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), num_keys * sizeof(int64_t), cudaMemcpyHostToDevice));

  NVE_CHECK(nve_layer_lookup(layer, ctx, num_keys, d_keys, d_output, ROW_SIZE, nullptr, nullptr));
  NVE_CHECK(nve_context_wait(ctx));

  // Cleanup
  cudaFree(d_keys);
  cudaFree(d_output);
  nve_context_wait(ctx);
  nve_context_destroy(ctx);
  nve_layer_destroy(layer);
  nve_heuristic_destroy(heuristic);
  nve_table_destroy(gpu_table);
  cudaFreeHost(h_uvm);
}

/* ============================================================================
 * Hierarchical Layer with GPU + NVHM host table
 * ============================================================================ */

class HierarchicalNvhmLayerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDA_CHECK(cudaSetDevice(DEVICE_ID));

    // Load NVHM plugin
    nve_status_t st = nve_load_host_table_plugin("nvhm");
    if (st != NVE_SUCCESS) {
      GTEST_SKIP() << "NVHM plugin not available";
    }

    // GPU table without UVM backing — misses go to host table
    auto gpu_cfg = nve_gpu_table_config_default();
    gpu_cfg.device_id = DEVICE_ID;
    gpu_cfg.cache_size = 1 << 20;
    gpu_cfg.row_size_in_bytes = ROW_SIZE;
    gpu_cfg.uvm_table = nullptr;
    gpu_cfg.count_misses = 1;
    gpu_cfg.value_dtype = NVE_DTYPE_FLOAT32;
    NVE_CHECK(nve_gpu_table_create(&gpu_table_, NVE_KEY_INT64, &gpu_cfg, nullptr));

    // NVHM host table via factory + produce
    NVE_CHECK(nve_create_host_table_factory(&nvhm_factory_, R"({"implementation": "nvhm_map"})"));

    const char* table_config = R"({
      "mask_size": 8,
      "key_size": 8,
      "max_value_size": 128,
      "value_dtype": "float32",
      "num_partitions": 2,
      "initial_capacity": 1024,
      "value_alignment": 32
    })";
    NVE_CHECK(nve_host_factory_produce(nvhm_factory_, 0, table_config, &nvhm_table_));

    // Hierarchical layer: GPU cache + NVHM host table
    auto hier_cfg = nve_hierarchical_layer_config_default();
    hier_cfg.layer_name = "test_hier_nvhm";

    nve_table_t tables[] = {gpu_table_, nvhm_table_};
    NVE_CHECK(nve_hierarchical_layer_create(&layer_, NVE_KEY_INT64, &hier_cfg, tables, 2, nullptr));
    NVE_CHECK(nve_layer_create_execution_context(layer_, &ctx_, nullptr, nullptr, nullptr, nullptr));

    // Clear tables
    NVE_CHECK(nve_layer_clear(layer_, ctx_));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void TearDown() override {
    if (ctx_) { nve_context_wait(ctx_); nve_context_destroy(ctx_); }
    if (layer_) nve_layer_destroy(layer_);
    if (gpu_table_) nve_table_destroy(gpu_table_);
    if (nvhm_table_) nve_table_destroy(nvhm_table_);
    if (nvhm_factory_) nve_host_factory_destroy(nvhm_factory_);
  }

  nve_table_t gpu_table_ = nullptr;
  nve_host_factory_t nvhm_factory_ = nullptr;
  nve_table_t nvhm_table_ = nullptr;
  nve_layer_t layer_ = nullptr;
  nve_context_t ctx_ = nullptr;
};

TEST_F(HierarchicalNvhmLayerTest, InsertAndLookup) {
  const uint64_t num_keys = 4;
  std::vector<int64_t> h_keys = {0, 1, 2, 3};

  // Insert values into the layer (goes to both GPU cache and host table)
  std::vector<float> h_vals(num_keys * NUM_FLOATS, 500.0f);
  // Set distinct values per row so we can verify correctness
  for (uint64_t r = 0; r < num_keys; ++r) {
    for (uint64_t c = 0; c < NUM_FLOATS; ++c) {
      h_vals[r * NUM_FLOATS + c] = 500.0f + static_cast<float>(r);
    }
  }

  void* p_output = nullptr;
  CUDA_CHECK(cudaMallocHost(&p_output, num_keys * ROW_SIZE));

  NVE_CHECK(nve_layer_insert(layer_, ctx_, num_keys, h_keys.data(), ROW_SIZE, ROW_SIZE, h_vals.data(), 0));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Lookup inserted values
  NVE_CHECK(nve_layer_lookup(layer_, ctx_, num_keys, h_keys.data(), p_output, ROW_SIZE, nullptr, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());

  auto* out = static_cast<float*>(p_output);
  for (uint64_t k = 0; k < num_keys; ++k) {
    float expected = 500.0f + static_cast<float>(k);
    EXPECT_FLOAT_EQ(expected, out[k * NUM_FLOATS]) << "Key " << h_keys[k];
  }

  cudaFreeHost(p_output);
}

TEST_F(HierarchicalNvhmLayerTest, UpdateAndAccumulate) {
  const int64_t num_keys = 2;
  std::vector<int64_t> h_keys = {10, 11};

  void* p_output = nullptr;
  CUDA_CHECK(cudaMallocHost(&p_output, num_keys * ROW_SIZE));

  // Insert known values
  std::vector<float> h_vals(num_keys * NUM_FLOATS, 25.0f);
  NVE_CHECK(nve_layer_insert(layer_, ctx_, num_keys, h_keys.data(), ROW_SIZE, ROW_SIZE, h_vals.data(), 0));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Update to new values
  std::vector<float> h_upd(num_keys * NUM_FLOATS, 50.0f);
  NVE_CHECK(nve_layer_update(layer_, ctx_, num_keys, h_keys.data(), ROW_SIZE, ROW_SIZE, h_upd.data()));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Accumulate gradient of 5.0
  std::vector<float> h_grads(num_keys * NUM_FLOATS, 5.0f);
  NVE_CHECK(nve_layer_accumulate(layer_, ctx_, num_keys, h_keys.data(),
                                 ROW_SIZE, ROW_SIZE, h_grads.data(), NVE_DTYPE_FLOAT32));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Verify: 50.0 + 5.0 = 55.0
  NVE_CHECK(nve_layer_lookup(layer_, ctx_, num_keys, h_keys.data(), p_output, ROW_SIZE, nullptr, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());

  auto* out = static_cast<float*>(p_output);
  for (int64_t i = 0; i < num_keys * NUM_FLOATS; ++i) {
    EXPECT_FLOAT_EQ(55.0f, out[i]);
  }

  cudaFreeHost(p_output);
}

TEST_F(HierarchicalNvhmLayerTest, EraseAndClear) {
  const int64_t num_keys = 4;
  std::vector<int64_t> h_keys = {20, 21, 22, 23};

  // Insert values
  std::vector<float> h_vals(num_keys * NUM_FLOATS, 100.0f);
  NVE_CHECK(nve_layer_insert(layer_, ctx_, num_keys, h_keys.data(), ROW_SIZE, ROW_SIZE, h_vals.data(), 0));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Erase subset
  NVE_CHECK(nve_layer_erase(layer_, ctx_, 2, h_keys.data(), 0));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Clear all
  NVE_CHECK(nve_layer_clear(layer_, ctx_));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Lookup should return misses (zeros)
  void* p_output = nullptr;
  CUDA_CHECK(cudaMallocHost(&p_output, num_keys * ROW_SIZE));
  memset(p_output, 0xFF, num_keys * ROW_SIZE);  // fill with non-zero to detect writes

  uint64_t hitmask = 0xFFFFFFFFFFFFFFFF;
  NVE_CHECK(nve_layer_lookup(layer_, ctx_, num_keys, h_keys.data(), p_output, ROW_SIZE, &hitmask, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());

  // All should be misses after clear
  EXPECT_EQ(0u, hitmask & 0xF);

  cudaFreeHost(p_output);
}

/* ============================================================================
 * Hierarchical Layer with GPU + Redis table
 * ============================================================================ */

static bool is_redis_ready() {
  FILE* pipe = popen("/usr/bin/redis-cli -p 7000 ping", "r");
  if (!pipe) return false;
  char out[16] = {0};
  if (fgets(out, 16, pipe) == nullptr) { pclose(pipe); return false; }
  int ret = pclose(pipe);
  return ret == 0 && strncmp(out, "PONG", 4) == 0;
}

class HierarchicalRedisLayerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!is_redis_ready()) {
      GTEST_SKIP() << "Redis cluster not available on localhost:7000";
    }

    CUDA_CHECK(cudaSetDevice(DEVICE_ID));

    nve_status_t st = nve_load_host_table_plugin("redis");
    if (st != NVE_SUCCESS) {
      GTEST_SKIP() << "Redis plugin not available";
    }

    // GPU table without UVM backing — misses go to host table
    auto gpu_cfg = nve_gpu_table_config_default();
    gpu_cfg.device_id = DEVICE_ID;
    gpu_cfg.cache_size = 1 << 20;
    gpu_cfg.row_size_in_bytes = ROW_SIZE;
    gpu_cfg.uvm_table = nullptr;
    gpu_cfg.count_misses = 1;
    gpu_cfg.value_dtype = NVE_DTYPE_FLOAT32;
    NVE_CHECK(nve_gpu_table_create(&gpu_table_, NVE_KEY_INT64, &gpu_cfg, nullptr));

    // Redis host table via factory + produce
    NVE_CHECK(nve_create_host_table_factory(
        &redis_factory_, R"({"implementation": "redis_cluster", "address": "localhost:7000"})"));

    const char* table_config = R"({
      "mask_size": 8,
      "key_size": 8,
      "max_value_size": 128,
      "value_dtype": "float32"
    })";
    NVE_CHECK(nve_host_factory_produce(redis_factory_, 0, table_config, &redis_table_));

    // Hierarchical layer: GPU + Redis
    auto hier_cfg = nve_hierarchical_layer_config_default();
    hier_cfg.layer_name = "test_hier_redis";

    nve_table_t tables[] = {gpu_table_, redis_table_};
    NVE_CHECK(nve_hierarchical_layer_create(&layer_, NVE_KEY_INT64, &hier_cfg, tables, 2, nullptr));
    NVE_CHECK(nve_layer_create_execution_context(layer_, &ctx_, nullptr, nullptr, nullptr, nullptr));

    // Clear Redis state from previous runs
    NVE_CHECK(nve_layer_clear(layer_, ctx_));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void TearDown() override {
    if (layer_ && ctx_) {
      nve_layer_clear(layer_, ctx_);
      cudaDeviceSynchronize();
    }
    if (ctx_) { nve_context_wait(ctx_); nve_context_destroy(ctx_); }
    if (layer_) nve_layer_destroy(layer_);
    if (gpu_table_) nve_table_destroy(gpu_table_);
    if (redis_table_) nve_table_destroy(redis_table_);
    if (redis_factory_) nve_host_factory_destroy(redis_factory_);
  }

  nve_table_t gpu_table_ = nullptr;
  nve_host_factory_t redis_factory_ = nullptr;
  nve_table_t redis_table_ = nullptr;
  nve_layer_t layer_ = nullptr;
  nve_context_t ctx_ = nullptr;
};

TEST_F(HierarchicalRedisLayerTest, InsertLookupAndUpdate) {
  const int64_t num_keys = 4;
  std::vector<int64_t> h_keys = {100, 101, 102, 103};

  void* p_output = nullptr;
  CUDA_CHECK(cudaMallocHost(&p_output, num_keys * ROW_SIZE));

  // Insert values into the layer
  std::vector<float> h_vals(num_keys * NUM_FLOATS, 777.0f);
  NVE_CHECK(nve_layer_insert(layer_, ctx_, num_keys, h_keys.data(), ROW_SIZE, ROW_SIZE, h_vals.data(), 0));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Lookup and verify
  NVE_CHECK(nve_layer_lookup(layer_, ctx_, num_keys, h_keys.data(), p_output, ROW_SIZE, nullptr, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());

  auto* out = static_cast<float*>(p_output);
  for (int64_t i = 0; i < num_keys * NUM_FLOATS; ++i) {
    EXPECT_FLOAT_EQ(777.0f, out[i]);
  }

  // Update values
  std::vector<float> h_upd(num_keys * NUM_FLOATS, 888.0f);
  NVE_CHECK(nve_layer_update(layer_, ctx_, num_keys, h_keys.data(), ROW_SIZE, ROW_SIZE, h_upd.data()));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Verify update
  NVE_CHECK(nve_layer_lookup(layer_, ctx_, num_keys, h_keys.data(), p_output, ROW_SIZE, nullptr, nullptr));
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int64_t i = 0; i < num_keys * NUM_FLOATS; ++i) {
    EXPECT_FLOAT_EQ(888.0f, out[i]);
  }

  cudaFreeHost(p_output);
}
