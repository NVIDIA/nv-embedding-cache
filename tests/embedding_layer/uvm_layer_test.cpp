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

#include "emb_layer_utils.hpp"
#include <embedding_layer.hpp>
#include <linear_embedding_layer.hpp>
#include <execution_context.hpp>
#include <gpu_table.hpp>
#include <host_table.hpp>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <cuda_support.hpp>

#include "mock_host_table.hpp"
#include "cuda_ops/cuda_common.h"

namespace nve {

enum class PrivateStreamMode : uint64_t {
  None,
  Optimized,
  NonOptimized,
};

// Fixture for functional tests
struct UVMTestCase {
  int64_t uvm_table_size;
  int64_t gpu_table_size;
  int64_t row_size;
  size_t test_keys;
  DataType_t data_type; // for now using the same type for table and accumulate
  PrivateStreamMode private_stream_mode;
  bool modify_on_gpu;
};
class UVM : public ::testing::TestWithParam<UVMTestCase> {};

static bool IsCpu(const std::string cpu_name) {
  FILE* pipe = popen("lscpu|grep 'Model name:'|cut -d':' -f2|sed -e 's/^[[:space:]]*//'", "r");
  if (!pipe) {
    throw std::runtime_error("Failed to check CPU model (popen)");
  }
  char out[1024] = {0};
  if (fgets(out, 1024, pipe) == NULL) {
    throw std::runtime_error("Failed to check CPU model (fgets)");
  }
  auto ret = pclose(pipe);
  if (ret != 0) {
    throw std::runtime_error("Failed to check CPU model (pclose)");
  }

  std::string cpu_str(out);
  return (cpu_str.find(cpu_name) != std::string::npos);
}

static bool IsLargeTestAnd7xxx(const nve::UVMTestCase& tc) {
  static const bool is7xxx = IsCpu("AMD EPYC 7313P") || IsCpu("AMD EPYC 7232P"); // static so we only check this once instead of every test
  return (is7xxx && (tc.uvm_table_size > int64_t(1) << 30));
}

template <typename IndexT>
class UVMLayerTest {
 public:
  using layer_type = LinearUVMEmbeddingLayer<IndexT>;
  static constexpr int64_t MAX_MODIFY_SIZE = (1l << 20);

  UVMLayerTest( int64_t row_size, int64_t gpu_table_size, int64_t uvm_table_size, DataType_t data_type,
                PrivateStreamMode private_stream_mode, bool modify_on_gpu, int device_id, bool insert_heuristic = false,
                std::function<void(GPUTableConfig&)> gpu_cfg_overwrite = nullptr,
                size_t seed = 31337)
    : m_row_size(row_size), m_max_rows(uvm_table_size / row_size), m_device_id(device_id) {
    ScopedDevice dev(device_id);
    // init linear table
    {
      NVE_CHECK_(cudaMallocHost(&m_linear_table, static_cast<size_t>(m_row_size * m_max_rows)));
      NVE_CHECK_(static_cast<size_t>(m_row_size) % sizeof(float) == 0); // cannot use ASSERT_EQ in c'tor

      // Init table values with multiple threads to save on test time
      std::vector<std::shared_ptr<std::thread>> input_gen_threads;
      const int64_t num_threads = static_cast<int64_t>(std::thread::hardware_concurrency());
      for (int64_t t=0 ; t<num_threads ; t++) {
        auto start_row = t * m_max_rows / num_threads;
        auto end_row = std::min<int64_t>((t + 1) * m_max_rows / num_threads, m_max_rows);

        input_gen_threads.push_back(std::make_shared<std::thread>(
          InitTableRows, m_linear_table, m_row_size, start_row, end_row, data_type, seed + static_cast<size_t>(t)
        ));
      }
      for (auto& t : input_gen_threads) {
          t->join();
      }
    }
    // init private stream
    if (private_stream_mode != PrivateStreamMode::None) {
      NVE_CHECK_(cudaStreamCreate(&private_stream));
    }

    // init gpu table
    {
      GPUTableConfig cfg;
      cfg.device_id = device_id;
      cfg.cache_size = static_cast<size_t>(gpu_table_size);
      cfg.max_modify_size = MAX_MODIFY_SIZE;
      cfg.row_size_in_bytes = row_size;
      cfg.uvm_table = m_linear_table;
      cfg.count_misses = true;
      cfg.value_dtype = data_type;
      cfg.private_stream = private_stream;
      cfg.modify_on_gpu = modify_on_gpu;
      if (gpu_cfg_overwrite != nullptr) {
        gpu_cfg_overwrite(cfg);
      }
      m_gpu_tab = std::make_shared<GpuTable<IndexT>>(cfg);
    }

    // init layer
    {
      typename LinearUVMEmbeddingLayer<IndexT>::Config cfg;
      if (insert_heuristic) {
        cfg.insert_heuristic = std::make_shared<TestInsertHeuristic>();
        cfg.min_insert_freq_gpu = 0;
        cfg.min_insert_size_gpu = 1024;
      } else {
        cfg.insert_heuristic = std::make_shared<NeverInsertHeuristic>();
      }
      m_layer = std::make_shared<LinearUVMEmbeddingLayer<IndexT>>(cfg, m_gpu_tab);
    }
    // init context
    cudaStream_t ctx_stream = (private_stream_mode == PrivateStreamMode::Optimized) ? private_stream : 0;
    m_ctx = m_layer->create_execution_context(ctx_stream, ctx_stream, nullptr, nullptr);

    // create ref (using single mock table)
    {
      HostTableConfig mock_cfg;
      mock_cfg.value_dtype = data_type;
      m_ref_tab = std::make_shared<MockHostTable<IndexT>>(mock_cfg, true /*functional_ref*/);
    }
  }

  UVMLayerTest(UVMTestCase tc, int device_id = 0) : UVMLayerTest<IndexT>(tc.row_size, tc.gpu_table_size, tc.uvm_table_size, tc.data_type, tc.private_stream_mode, tc.modify_on_gpu, device_id) {}
  ~UVMLayerTest() {
    ScopedDevice dev(m_device_id);
    m_ctx->wait();
    m_ctx.reset();  // explicitly destroy the context before the stream since the context synchronizes on the stream during d'tor
                    // (which may be the private stream about to be destroyed)
    if (private_stream) {
      NVE_CHECK_(cudaStreamSynchronize(private_stream)); // Make sure all work for the stream is complete before destruction
      NVE_CHECK_(cudaStreamDestroy(private_stream));
    }
    NVE_CHECK_(cudaFreeHost(m_linear_table));
  }

  void LookupAndCheck(std::vector<IndexT>& keys, uint64_t start_key = 0,
                      uint64_t end_key = uint64_t(-1), float hit_tol = 0.01f, float* hitrate = nullptr) {
    if (keys.empty()) {
      return;
    }
    SetupKeys setup(keys, start_key, end_key);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;

    auto output_size = static_cast<size_t>(num_keys * m_row_size);
    int8_t* output{nullptr};
    NVE_CHECK_(cudaMallocHost(&output, output_size));
    NVE_CHECK_(output != 0);
    std::vector<int8_t> ref_output(output_size);

    int64_t mask_bits_per_elem = static_cast<int64_t>(sizeof(max_bitmask_repr_t) * 8);
    auto hitmask_size = static_cast<size_t>((num_keys + mask_bits_per_elem - 1) / mask_bits_per_elem);
    std::vector<max_bitmask_repr_t> ref_hitmask(hitmask_size);

    std::vector<float> hitrates(3);

    m_layer->lookup(m_ctx, num_keys, keys_buffer, output, m_row_size, nullptr /*hitmask*/,
                    nullptr /*pool_params*/, hitrates.data());
    NVE_CHECK_(cudaDeviceSynchronize());
    int64_t ref_hits;
    m_ref_tab->reset_lookup_counter(m_ctx);
    m_ref_tab->find(m_ctx, num_keys, keys_buffer, ref_hitmask.data(), m_row_size, ref_output.data(), nullptr /*value_sizes*/);
    m_ref_tab->get_lookup_counter(m_ctx, &ref_hits);


    // compare hitrates
    ASSERT_NEAR(hitrates[0], float(ref_hits) / float(num_keys), hit_tol);

    // compare outputs for hits
    for (int64_t i = 0; i < num_keys; i++) {
      bool ref_hit = ref_hitmask.at(static_cast<size_t>(i / mask_bits_per_elem)) & (1ll << (i % mask_bits_per_elem));
      const int8_t* ref_vec = ref_hit ? ref_output.data() + (i * m_row_size) : m_linear_table + (keys.at(static_cast<size_t>(i)) * m_row_size);
      for (int64_t j = 0; j < m_row_size; j++) {
        ASSERT_EQ(output[(i * m_row_size) + j], *(ref_vec + j));
      }
    }
    NVE_CHECK_(cudaFreeHost(output));
    if (hitrate != nullptr) {
      *hitrate = hitrates[0];
    }
  }

  void Insert(std::vector<IndexT>& keys, std::vector<uint8_t>& datavectors, uint64_t start_key = 0, uint64_t end_key = uint64_t(-1)) {
    if (keys.empty()) {
      return;
    }
    SetupKeys setup(keys, start_key, end_key, datavectors, m_row_size);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;
    auto data_buffer = setup.data_buffer;

    // "insert" to the linear table
    for (int64_t i=0 ; i<num_keys ; i++) {
      auto pos = start_key + static_cast<size_t>(i);
      auto key = keys[pos];
      auto row_size = static_cast<size_t>(m_row_size);
      ASSERT_LE(key, m_max_rows); // Not handling keys too big for the linear table
      
      std::memcpy(m_linear_table + (key * m_row_size), datavectors.data() + (pos * row_size), row_size);
    }

    m_layer->insert(m_ctx, num_keys, keys_buffer, m_row_size, m_row_size, data_buffer, 0);
    bool ref_insert = (m_gpu_tab != nullptr);
    if (ref_insert) {
      m_ref_tab->insert(m_ctx, num_keys, keys_buffer, m_row_size, m_row_size, data_buffer);
    }
  }

  void Update(std::vector<IndexT>& keys, std::vector<uint8_t>& datavectors, uint64_t start_key = 0,
              uint64_t end_key = uint64_t(-1)) {
    if (keys.empty()) {
      return;
    }
    SetupKeys setup(keys, start_key, end_key, datavectors, m_row_size);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;
    auto data_buffer = setup.data_buffer;

    m_layer->update(m_ctx, num_keys, keys_buffer, m_row_size, m_row_size, data_buffer);
    m_ref_tab->update(m_ctx, num_keys, keys_buffer, m_row_size, m_row_size, data_buffer);
  }

  void Accumulate(std::vector<IndexT>& keys, std::vector<uint8_t>& datavectors,
                  DataType_t value_type, uint64_t start_key = 0, uint64_t end_key = uint64_t(-1)) {
    if (keys.empty()) {
      return;
    }
    SetupKeys setup(keys, start_key, end_key, datavectors, m_row_size);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;
    auto data_buffer = setup.data_buffer;

    m_layer->accumulate(m_ctx, num_keys, keys_buffer, m_row_size, m_row_size, data_buffer, value_type);
    m_ref_tab->update_accumulate(m_ctx, num_keys, keys_buffer, m_row_size, m_row_size, data_buffer, value_type);
  }

  void Erase(std::vector<IndexT>& keys, uint64_t start_key = 0, uint64_t end_key = uint64_t(-1)) {
    if (keys.empty()) {
      return;
    }
    SetupKeys setup(keys, start_key, end_key);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;
    m_layer->erase(m_ctx, num_keys, keys_buffer, 0);
    m_ref_tab->erase(m_ctx, num_keys, keys_buffer);
  }

  void Clear(bool clear_ref = true) {
    m_layer->clear(m_ctx);
    if (clear_ref) {
      m_ref_tab->clear(m_ctx);
    }
  }

  void Wait() { m_ctx->wait(); }

  typename LinearUVMEmbeddingLayer<IndexT>::Config Config() const {
    return m_layer->get_config();
  }

 public:
  const int64_t m_row_size;
  const int64_t m_max_rows;
 private:
  typename layer_type::gpu_table_ptr_t m_gpu_tab{nullptr};
  host_table_ptr_t m_ref_tab;
  int8_t* m_linear_table{nullptr};
  std::shared_ptr<layer_type> m_layer;
  context_ptr_t m_ctx;
  cudaStream_t private_stream{0};
  int m_device_id;
};

// [Sanity] Init the layer
TEST_P(UVM, Init) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  if (IsLargeTestAnd7xxx(tc)) {
    GTEST_SKIP() << "Skipping large UVM test";
    return;
  }
  UVMLayerTest<int64_t> ult(tc);
}

// [Sanity] insert 1key, lookup 1key
TEST_P(UVM, SingleLookup) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  if (IsLargeTestAnd7xxx(tc)) {
    GTEST_SKIP() << "Skipping large UVM test";
    return;
  }
  UVMLayerTest<int64_t> ult(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, 1, tc.row_size, 0, ult.m_max_rows, tc.data_type);
  ult.Insert(keys, data);
  ult.LookupAndCheck(keys);
}

// [Lookup] insert k, lookup 2k, get k hits, k misses (per table, k small enough to avoid partial
// inserts), clear, lookup
TEST_P(UVM, Lookup) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  if (IsLargeTestAnd7xxx(tc)) {
    GTEST_SKIP() << "Skipping large UVM test";
    return;
  }
  UVMLayerTest<int64_t> ult(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.m_max_rows, tc.data_type);
  ult.Insert(keys, data, 0, tc.test_keys);
  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);
  ult.Clear();
  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);
}

// [Update] insert k, update 2k, lookup 2k,...
TEST_P(UVM, Update) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  if (IsLargeTestAnd7xxx(tc)) {
    GTEST_SKIP() << "Skipping large UVM test";
    return;
  }
  UVMLayerTest<int64_t> ult(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.m_max_rows, tc.data_type);
  ult.Insert(keys, data, 0, tc.test_keys / 2); // insert half so the rest will be read from UVM
  NVE_CHECK_(cudaDeviceSynchronize());

  // change some data so there's something to update
  const auto num_rows = static_cast<int64_t>(data.size()) / tc.row_size;
  for (int64_t i = 0; i < num_rows; i+=2) {  // only updating half the rows
    for (int64_t j = 0; j < tc.row_size; j++) {
      data.at(static_cast<size_t>((i * tc.row_size) + j)) *= 3;
    }
  }

  ult.Update(keys, data);
  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);
}

// [Insert] insert k, lookup k+k`, insert k`, lookup k+k`
TEST_P(UVM, Insert) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  if (IsLargeTestAnd7xxx(tc)) {
    GTEST_SKIP() << "Skipping large UVM test";
    return;
  }
  UVMLayerTest<int64_t> ult(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.m_max_rows, tc.data_type);
  ult.Insert(keys, data, 0, tc.test_keys / 6);
  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);

  ult.Insert(keys, data, tc.test_keys * 3 / 6, tc.test_keys * 4 / 6);
  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);
}

// [Accumulate] insert k+k`, accumulate k+k``
TEST_P(UVM, Accumulate) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  if (IsLargeTestAnd7xxx(tc)) {
    GTEST_SKIP() << "Skipping large UVM test";
    return;
  }
  UVMLayerTest<int64_t> ult(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.m_max_rows, tc.data_type);
  ult.Insert(keys, data, 0, tc.test_keys / 2);
  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);

  // Accumulate some keys that were inserted and some that weren't (reusing original data as
  // gradients)
  ult.Accumulate(keys, data, tc.data_type, tc.test_keys / 16, tc.test_keys * 3 / 16);
  ult.Accumulate(keys, data, tc.data_type, tc.test_keys * 5 / 16, tc.test_keys * 12 / 16);

  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);
}

//  [Erase] inserk k+k`, lookup k+k`+k``, erase k, lookup k+k`+k``
TEST_P(UVM, Erase) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  if (IsLargeTestAnd7xxx(tc)) {
    GTEST_SKIP() << "Skipping large UVM test";
    return;
  }
  UVMLayerTest<int64_t> ult(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.m_max_rows, tc.data_type);
  ult.Insert(keys, data, 0, tc.test_keys / 2);
  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);

  ult.Erase(keys, tc.test_keys * 2 / 6, tc.test_keys * 3 / 6);
  ult.Erase(keys, tc.test_keys * 4 / 6, tc.test_keys * 5 / 6);

  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);
}

TEST_P(UVM, LookupNonDefaultDevice) {
  cudaGetLastError();  // Clear potential errors left by previous tests.

  // check we have at least 2 devices
  int num_devices = 0;
  NVE_CHECK_(cudaGetDeviceCount(&num_devices));
  if (num_devices < 2) {
    GTEST_SKIP() << "Skipping multi-GPU test since only " << num_devices << " device(s) found";
    return;
  }
  const auto tc = GetParam();
  if (IsLargeTestAnd7xxx(tc)) {
    GTEST_SKIP() << "Skipping large UVM test";
    return;
  }
  UVMLayerTest<int64_t> ult(tc, 1 /*device_id*/);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.m_max_rows, tc.data_type);
  ult.Insert(keys, data, 0, tc.test_keys / 2);
  NVE_CHECK_(cudaDeviceSynchronize());
  ult.LookupAndCheck(keys);
}

TEST_P(UVM, LookupMultiDevice) {
  cudaGetLastError();  // Clear potential errors left by previous tests.

  // check we have at least 2 devices
  int num_devices = 0;
  NVE_CHECK_(cudaGetDeviceCount(&num_devices));
  if (num_devices < 2) {
    GTEST_SKIP() << "Skipping multi-GPU test since only " << num_devices << " device(s) found";
    return;
  }

  const auto tc = GetParam();
  if (IsLargeTestAnd7xxx(tc)) {
    GTEST_SKIP() << "Skipping large UVM test";
    return;
  }
  UVMLayerTest<int64_t> ult_0(tc, 0 /*device_id*/);
  UVMLayerTest<int64_t> ult_1(tc, 1 /*device_id*/);
  std::thread thread_0([&ult_0, tc]() {
      ScopedDevice dev_0(0);
      for (int i = 0; i < 10; i++) {
        std::vector<int64_t> keys;
        std::vector<uint8_t> data;
        GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult_0.m_max_rows, tc.data_type);
        ult_0.Insert(keys, data, 0, tc.test_keys / 2);
        NVE_CHECK_(cudaDeviceSynchronize());
        ult_0.LookupAndCheck(keys);
      }
    });
  std::thread thread_1([&ult_1, tc]() {
      ScopedDevice dev_1(1);    
      for (int i = 0; i < 10; i++) {
        std::vector<int64_t> keys;
        std::vector<uint8_t> data;
        GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult_1.m_max_rows, tc.data_type);
        ult_1.Insert(keys, data, 0, tc.test_keys / 2);
        NVE_CHECK_(cudaDeviceSynchronize());
        ult_1.LookupAndCheck(keys);
      }
    });
  thread_0.join();
  thread_1.join();
}

INSTANTIATE_TEST_SUITE_P(
    EmbLayer,
    UVM,
    ::testing::Values(
        //  TestCase: uvm_table_size,   gpu_table_Size,   row_size,         test_keys         data_type            private_stream                   modify_on_gpu
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 20, int64_t(1) << 10, int64_t(1) << 0,  DataType_t::Float16, PrivateStreamMode::None,         false}),  // Single key
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 20, int64_t(1) << 10, int64_t(1) << 0,  DataType_t::Float32, PrivateStreamMode::None,         false}),  // Single key
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16, PrivateStreamMode::None,         false}), // Medium
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32, PrivateStreamMode::None,         false}), // Medium
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16, PrivateStreamMode::NonOptimized, false}), // Medium, private stream non optimized
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32, PrivateStreamMode::NonOptimized, false}), // Medium, private stream non optimized
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float16, PrivateStreamMode::Optimized,    false}), // Medium, private stream optimized
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32, PrivateStreamMode::Optimized,    false}), // Medium, private stream optimized
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 20, int64_t(1) << 10, int64_t(1) << 0,  DataType_t::Float16, PrivateStreamMode::None,         true}),  // Single key
        // Reduced set for modify_on_gpu
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 20, int64_t(1) << 10, int64_t(1) << 0,  DataType_t::Float16, PrivateStreamMode::None,         true}),  // Single key
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 20, int64_t(1) << 10, int64_t(1) << 0,  DataType_t::Float32, PrivateStreamMode::None,         true}),  // Single key
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32, PrivateStreamMode::None,         true}), // Medium
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 11, DataType_t::Float32, PrivateStreamMode::Optimized,    true}) // Medium, private stream optimized
      ));

INSTANTIATE_TEST_SUITE_P(
    EmbLayer_Large,
    UVM,
    ::testing::Values(
        //  TestCase: uvm_table_size,   gpu_table_Size,   row_size,         test_keys         data_Type            private_stream           modify_on_gpu
        UVMTestCase({ int64_t(1) << 32, int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 16, DataType_t::Float16, PrivateStreamMode::None, false}),  // Large 
        UVMTestCase({ int64_t(1) << 32, int64_t(1) << 30, int64_t(1) << 10, int64_t(1) << 16, DataType_t::Float32, PrivateStreamMode::None, false})  // Large 
      ));

class UVMSpecialConfig : public ::testing::TestWithParam<UVMTestCase> {};

TEST_P(UVMSpecialConfig, InflightInsert) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  UVMLayerTest<int64_t> ult(tc.row_size, tc.gpu_table_size, tc.uvm_table_size, tc.data_type, tc.private_stream_mode, false, 0 /*device_id*/,
                            true /*insert_heuristic*/); // override insert_heuristic
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.m_max_rows, tc.data_type);

  // Insert keys to ref and linear (insert to all, the remove from gpu table)
  ult.Insert(keys, data);
  ult.Clear(false /* clear_ref */); // remove from gpu_table, data still left in linear table and ref

  auto first_insert_keys = static_cast<uint64_t>(ult.Config().min_insert_size_gpu / 2);
  // Test relies on having enough keys to trigger auto insert
  ASSERT_LE(first_insert_keys, tc.test_keys); 

  // First lookup: hitrate should be 0 since we cleared the gpu table, too few keys to trigger auto-insert
  float hitrate{0.f};
  ult.LookupAndCheck(keys, 0, first_insert_keys, 1.f, &hitrate);
  ASSERT_EQ(hitrate, 0.f);
  ult.Wait(); // Wait in case auto insert was triggereed prematurely

  // Second lookup: hitrate should still be 0, but auto insert should be triggered
  ult.LookupAndCheck(keys, first_insert_keys, uint64_t(-1), 1.f, &hitrate);
  ASSERT_EQ(hitrate, 0.f);
  ult.Wait(); // Wait for auto insert to finish

  // Third lookup: hitrate should be just for auto insert size, and trigger second auto-insert for all keys
  ult.LookupAndCheck(keys, 0, uint64_t(-1), 1.f, &hitrate);
  ASSERT_NEAR(hitrate, float(ult.Config().min_insert_size_gpu) / float(keys.size()), 1e-2);
  ult.Wait(); // Wait for auto insert to finish

  // Last lookup - now hitrate should be (almost) 100%
  ult.LookupAndCheck(keys);
}

TEST_P(UVMSpecialConfig, LargeModify) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  UVMLayerTest<int64_t> ult(tc.row_size, tc.gpu_table_size, tc.uvm_table_size, tc.data_type,
                            tc.private_stream_mode, tc.modify_on_gpu, 0 /*device_id*/, false /*insert_heuristic*/,
                            [&tc](GPUTableConfig& cfg){ cfg.max_modify_size = tc.test_keys / 3; }); // override config to reduce max_modify_size
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;
  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.m_max_rows, tc.data_type);

  // Test insert in parts
  ult.Insert(keys, data);
  ult.Wait();

  // since insert can only process 1/3 of the keys - we expect 0.33 hitrate vs. the ref with 1.0 (ref doesn't model modify size)  
  ult.LookupAndCheck(keys, 0, uint64_t(-1), 0.7f);

  // Now do smaller inserts to make sure the baseline state is the same compared to the ref.
  const auto num_keys = keys.size();
  for (uint64_t k_start = 0; k_start < num_keys ; k_start += (num_keys/4)) {
    uint64_t k_end = std::min(num_keys, k_start + (num_keys/4));
    ult.Insert(keys, data, k_start, k_end);
    ult.Wait();
  }
  ult.LookupAndCheck(keys);

  // Test update in parts
  // Change some data so there's something to update
  const auto num_rows = static_cast<int64_t>(data.size()) / tc.row_size;
  for (int64_t i = 0; i < num_rows; i+=2) {  // only updating half the rows
    for (int64_t j = 0; j < tc.row_size; j++) {
      data.at(static_cast<size_t>((i * tc.row_size) + j)) *= 3;
    }
  }
  ult.Update(keys, data);
  ult.Wait();
  ult.LookupAndCheck(keys);

  // Test accumulate in parts
  ult.Accumulate(keys, data, tc.data_type);
  ult.Wait();
  ult.LookupAndCheck(keys);
}

TEST_P(UVMSpecialConfig, UVMUpdate) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  UVMLayerTest<int64_t> ult(tc);

  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.m_max_rows, tc.data_type);

  // Insert only half the keys
  ult.Insert(keys, data, tc.test_keys / 4 , tc.test_keys * 3 / 4);

  // Update all keys (half will update in the cache, and all will update the table)
  ult.Update(keys, data);
  ult.Wait(); // Wait for auto insert to finish
  
  // Lookup and compare
  ult.LookupAndCheck(keys);
}

TEST_P(UVMSpecialConfig, UVMCPUAccumulate) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  UVMLayerTest<int64_t> ult(tc);

  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.m_max_rows, tc.data_type);
  // Insert only a small set of keys to guarantee hit rate with a small cache
  ult.Insert(keys, data, 0, 100);

  // Accumulate all keys (a few will update in the cache, and all will update the table)
  ult.Accumulate(keys, data, tc.data_type);
  ult.Wait(); // Wait for auto insert to finish
  
  // Lookup and compare
  ult.LookupAndCheck(keys);
}

TEST_P(UVMSpecialConfig, UVMGPUAccumulate) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  UVMLayerTest<int64_t> ult(tc.row_size, tc.gpu_table_size, tc.uvm_table_size, tc.data_type, tc.private_stream_mode, tc.modify_on_gpu, 0 /*device_id*/,
                            false /*insert_heuristic*/,
                            [](GPUTableConfig& cfg){ cfg.uvm_cpu_accumulate = false; }); // override config to disable cpu uvm update

  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, ult.m_max_rows, tc.data_type);
  // Insert only a small set of keys to guarantee hit rate with a small cache
  ult.Insert(keys, data, 0, 100);

  // Accumulate all keys (a few will update in the cache, and all will update the table)
  ult.Accumulate(keys, data, tc.data_type);
  ult.Wait(); // Wait for auto insert to finish
  
  // Lookup and compare
  ult.LookupAndCheck(keys);
}

INSTANTIATE_TEST_SUITE_P(
    EmbLayer,
    UVMSpecialConfig,
    ::testing::Values(
        //  TestCase: uvm_table_size,   gpu_table_Size,   row_size,         test_keys         data_type            private_stream                modify_on_gpu
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 15, DataType_t::Float16, PrivateStreamMode::None,      false}),
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 15, DataType_t::Float32, PrivateStreamMode::None,      false}),
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 15, DataType_t::Float16, PrivateStreamMode::Optimized, false}),
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 15, DataType_t::Float32, PrivateStreamMode::Optimized, false}),
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 15, DataType_t::Float32, PrivateStreamMode::None,      true}),
        UVMTestCase({ int64_t(1) << 30, int64_t(1) << 26, int64_t(1) << 10, int64_t(1) << 15, DataType_t::Float32, PrivateStreamMode::Optimized, true})

    ));

}  // namespace nve
