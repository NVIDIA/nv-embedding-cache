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
#include <gtest/gtest-spi.h>

#include <stdio.h>
#include "emb_layer_utils.hpp"
#include "test_utils.hpp"
#include <buffer_wrapper.hpp>
#include <cuda_support.hpp>
#include <embedding_layer.hpp>
#include <hierarchical_embedding_layer.hpp>
#include <gpu_table.hpp>
#include <host_table.hpp>
#include <linear_host_table.hpp>
#include <memory>
#include <string>
#include <vector>
#include <execution_context.hpp>
#include <thread_pool.hpp>
#include "mock_host_table.hpp"

namespace nve {

enum class HostTableType : uint64_t {
  None = 0,
  NVHashMap,
  Redis,
  Abseil,
  Phmap,
  Redis_String,
};

enum class TableType : uint64_t {
  Device,
  Host,
  Remote,
  Ref,
};

static bool is_host_table_type_available(HostTableType ht) {
  switch (ht) {
    case HostTableType::None:
      return true;
    case HostTableType::NVHashMap:
#ifdef NVE_FEATURE_NVHM_PLUGIN
      return true;
#else
      return false;
#endif
    case HostTableType::Redis:
    case HostTableType::Redis_String:
#ifdef NVE_FEATURE_REDIS_PLUGIN
      return true;
#else
      return false;
#endif
    case HostTableType::Abseil:
#ifdef NVE_FEATURE_ABSEIL_PLUGIN
      return true;
#else
      return false;
#endif
    case HostTableType::Phmap:
#ifdef NVE_FEATURE_PHMAP_PLUGIN
      return true;
#else
      return false;
#endif
    default:
      return false;
  }
}

static const char* host_table_type_name(HostTableType ht) {
  switch (ht) {
    case HostTableType::None: return "None";
    case HostTableType::NVHashMap: return "NVHashMap";
    case HostTableType::Redis: return "Redis";
    case HostTableType::Abseil: return "Abseil";
    case HostTableType::Phmap: return "Phmap";
    case HostTableType::Redis_String: return "Redis_String";
    default: return "Unknown";
  }
}

#define SKIP_IF_HOST_TABLE_UNAVAILABLE(tc) \
  do { \
    if (!is_host_table_type_available((tc).host_cache)) { \
      GTEST_SKIP() << "Host table type " << host_table_type_name((tc).host_cache) << " is not available"; \
    } \
  } while (0)

// Fixture for functional tests
struct HierTestCase {
  bool gpu_cache;
  HostTableType host_cache;
  bool remote_ps;
  int64_t row_size;
  int64_t table_size;
  size_t test_keys;
  bool insert_heuristic;
  DataType_t data_type; // for now using the same type for table and accumulate
};
class Hierachical : public ::testing::TestWithParam<HierTestCase> {};

template <typename IndexT>
class HierLayerTest {
 public:
  using layer_type = HierarchicalEmbeddingLayer<IndexT>;
  static constexpr int DEVICE_ID = 0;
  static constexpr int64_t MAX_MODIFY_SIZE = (1l << 20);
  static constexpr int64_t MIN_INSERT_HEURISTIC_SIZE = (1l << 10);

  HierLayerTest(bool gpu_table, HostTableType host_table, bool remote_table, int64_t row_size,
                int64_t table_size, bool insert_heuristic, DataType_t data_type)
      : m_row_size(row_size) {
    // init gpu table
    if (gpu_table) {
      GPUTableConfig cfg;
      cfg.device_id = DEVICE_ID;
      cfg.cache_size = static_cast<size_t>(table_size);
      cfg.max_modify_size = MAX_MODIFY_SIZE;
      cfg.row_size_in_bytes = row_size;
      cfg.uvm_table = nullptr;
      cfg.value_dtype = data_type;
      m_gpu_tab = std::make_shared<GpuTable<IndexT>>(cfg);
    }
    // init host table
    switch (host_table) {
      case HostTableType::None:
        break;// do nothing
      case HostTableType::NVHashMap: {
        std::vector<std::string> plugin_names{nve_test::plugin_full_path("nvhm")};
        load_host_table_plugins(plugin_names.begin(), plugin_names.end());
        nlohmann::json nvhm_conf = {{"mask_size", 8},
                                  {"key_size", sizeof(IndexT)},
                                  {"max_value_size", row_size},
                                  {"value_dtype", to_string(data_type)},
                                  {"num_partitions", 2},
                                  {"initial_capacity", 1024},
                                  {"value_alignment", 32},
                                  {"minimize_psl", true},
                                  {"auto_shrink", true},
                                  {"overflow_policy",
                                  {{"overflow_margin", (table_size / row_size)},
                                    {"handler", "evict_lru"},
                                    {"resolution_margin", 0.5}}}};
        host_table_factory_ptr_t nvhm_fac{
            create_host_table_factory(R"({"implementation": "nvhm_map"})"_json)};
        m_host_tab = nvhm_fac->produce(4711, nvhm_conf);
        break;
      }
      case HostTableType::Redis: {
        check_redis_ready(7000);
        std::vector<std::string> plugin_names{nve_test::plugin_full_path("redis")};
        load_host_table_plugins(plugin_names.begin(), plugin_names.end());
        nlohmann::json redis_conf = {
                                      {"num_partitions", 1},
                                      {"mask_size", 8},
                                      {"key_size", sizeof(IndexT)},
                                      {"max_value_size", row_size},
                                      {"value_dtype", to_string(data_type)},
                                    };
        host_table_factory_ptr_t redis_fac{
            create_host_table_factory(R"(
            {
              "address": "localhost:7000",
              "implementation": "redis_cluster"
            })"_json)};
        m_host_tab = redis_fac->produce(4712, redis_conf);
        break;
      }
      case HostTableType::Redis_String: {
        check_redis_ready(6379);
        std::vector<std::string> plugin_names{nve_test::plugin_full_path("redis")};
        load_host_table_plugins(plugin_names.begin(), plugin_names.end());
        nlohmann::json redis_conf = {
                                      {"num_partitions", 0},
                                      {"mask_size", 8},
                                      {"key_size", sizeof(IndexT)},
                                      {"max_value_size", row_size},
                                      {"value_dtype", to_string(data_type)},
                                    };
        host_table_factory_ptr_t redis_fac{
            create_host_table_factory(R"(
            {
              "address": "localhost:6379",
              "single_node": true,
              "implementation": "redis_cluster"
            })"_json)};
        m_host_tab = redis_fac->produce(4715, redis_conf);
        break;
      }
      case HostTableType::Abseil: {
        std::vector<std::string> plugin_names{nve_test::plugin_full_path("abseil")};
        load_host_table_plugins(plugin_names.begin(), plugin_names.end());
        nlohmann::json abseil_conf = {{"mask_size", 8},
                                  {"key_size", sizeof(IndexT)},
                                  {"max_value_size", row_size},
                                  {"value_dtype", to_string(data_type)},
                                  {"num_partitions", 2},
                                  {"initial_capacity", 1024},
                                  {"value_alignment", 32},
                                  {"overflow_policy",
                                  {{"overflow_margin", (table_size / row_size)},
                                    {"handler", "evict_lru"},
                                    {"resolution_margin", 0.5}}}};
        host_table_factory_ptr_t abseil_fac{
            create_host_table_factory(R"({"implementation": "abseil_flat_map"})"_json)};
        m_host_tab = abseil_fac->produce(4713, abseil_conf);
        break;
      }
      case HostTableType::Phmap: {
        std::vector<std::string> plugin_names{nve_test::plugin_full_path("phmap")};
        load_host_table_plugins(plugin_names.begin(), plugin_names.end());
        nlohmann::json phm_conf = {{"mask_size", 8},
                                  {"key_size", sizeof(IndexT)},
                                  {"max_value_size", row_size},
                                  {"value_dtype", to_string(data_type)},
                                  {"num_partitions", 2},
                                  {"initial_capacity", 1024},
                                  {"value_alignment", 32},
                                  {"overflow_policy",
                                  {{"overflow_margin", (table_size / row_size)},
                                    {"handler", "evict_lru"},
                                    {"resolution_margin", 0.5}}}};
        host_table_factory_ptr_t phm_fac{
            create_host_table_factory(R"({"implementation": "phmap_flat_map"})"_json)};
        m_host_tab = phm_fac->produce(4714, phm_conf);
        break;
      }
      default:
        throw std::runtime_error("Invalid host table type");
    }

    // init remote table (mock)
    if (remote_table) {
      HostTableConfig remote_cfg;
      remote_cfg.value_dtype = data_type;
      remote_cfg.max_value_size = row_size;
      m_remote_tab = std::make_shared<MockHostTable<IndexT>>(remote_cfg, true /*functional_ref*/);
    }
    // init layer
    typename HierarchicalEmbeddingLayer<IndexT>::Config cfg;
    std::vector<std::shared_ptr<nve::Table>> tables;
    std::vector<bool> heuristic_results;
    if (m_gpu_tab) {
      tables.push_back(m_gpu_tab);
      heuristic_results.push_back(true);
    }
    if (m_host_tab) {
      tables.push_back(m_host_tab);
      heuristic_results.push_back(true);
    }
    if (m_remote_tab) {
      tables.push_back(m_remote_tab);
      heuristic_results.push_back(false); // Don't test auto-insert for remote tables
    }
    if (insert_heuristic) {
      cfg.insert_heuristic = std::make_shared<TestInsertHeuristic>(heuristic_results);
      cfg.min_insert_freq_gpu = 0;
      cfg.min_insert_freq_host = 0;
      cfg.min_insert_size_gpu =  MIN_INSERT_HEURISTIC_SIZE;
      cfg.min_insert_size_host = MIN_INSERT_HEURISTIC_SIZE;
    } else {
      cfg.insert_heuristic = std::make_shared<NeverInsertHeuristic>();
    }
    m_layer = std::make_shared<HierarchicalEmbeddingLayer<IndexT>>(cfg, tables);

    // init context
    m_ctx = m_layer->create_execution_context(0, 0, nullptr, nullptr);
    // clear the layer tables (mostly needed for persistent Redis cluster used in testing)
    m_layer->clear(m_ctx);

    // create ref (using single mock table)
    HostTableConfig mock_cfg;
    mock_cfg.value_dtype = data_type;
    m_ref_tab = std::make_shared<MockHostTable<IndexT>>(mock_cfg, true /*functional_ref*/);
  }
  HierLayerTest(HierTestCase tc) : HierLayerTest<IndexT>(tc.gpu_cache, tc.host_cache, tc.remote_ps, tc.row_size, tc.table_size, tc.insert_heuristic, tc.data_type) {}
  ~HierLayerTest() {
    m_layer->clear(m_ctx); // clearing potentially persistent tables (e.g. redis)
    Wait();
  }

  // lookup on layer and ref, compare results (when hit_tol >= 0)
  void LookupAndCheck(std::vector<IndexT>& keys, uint64_t start_key = 0,
                      uint64_t end_key = uint64_t(-1), float hit_tol = 0.01f) {
    if (keys.empty()) {
      return;
    }

    SetupKeys setup(keys, start_key, end_key);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;

    size_t output_size = static_cast<size_t>(num_keys * m_row_size);
    int8_t* output{nullptr};
    NVE_CHECK_(cudaMallocHost(&output, output_size));
    NVE_CHECK_(output != 0);
    std::vector<int8_t> ref_output(output_size);

    auto mask_bits_per_elem = sizeof(max_bitmask_repr_t) * 8;
    uint64_t hitmask_size = (static_cast<uint64_t>(num_keys) + mask_bits_per_elem - 1) / mask_bits_per_elem;
    std::vector<max_bitmask_repr_t> hitmask(hitmask_size);
    std::vector<max_bitmask_repr_t> ref_hitmask(hitmask_size);

    std::vector<float> hitrates(3);

    m_layer->lookup(m_ctx, num_keys, keys_buffer, output, m_row_size, hitmask.data(),
                    nullptr /*pool_params*/, hitrates.data());
    NVE_CHECK_(cudaDeviceSynchronize());
    int64_t ref_hits;
    m_ref_tab->reset_lookup_counter(m_ctx);
    {
      auto keys_bw = std::make_shared<BufferWrapper<const void>>(
          m_ctx, "keys", keys_buffer, static_cast<size_t>(num_keys) * sizeof(IndexT));
      auto hit_mask_bw = std::make_shared<BufferWrapper<max_bitmask_repr_t>>(
          m_ctx, "hit_mask", ref_hitmask.data(), hitmask_size * sizeof(max_bitmask_repr_t));
      auto values_bw = std::make_shared<BufferWrapper<void>>(
          m_ctx, "values", ref_output.data(),
          static_cast<size_t>(num_keys) * static_cast<size_t>(m_row_size));
      m_ref_tab->find(m_ctx, num_keys, std::move(keys_bw), std::move(hit_mask_bw), m_row_size,
                         std::move(values_bw), nullptr /*value_sizes*/);
    }
    m_ref_tab->get_lookup_counter(m_ctx, &ref_hits);

    if (hit_tol >= 0.f) {
      // compare hitrates unless tolerance is negative
      ASSERT_NEAR(std::accumulate(hitrates.begin(), hitrates.end(), float(0)),
                  float(ref_hits) / float(num_keys), hit_tol);
      // compare outputs for hits
      const uint64_t row_size = static_cast<uint64_t>(m_row_size);
      for (uint64_t i = 0; i < static_cast<uint64_t>(num_keys); i++) {
        bool layer_hit = hitmask.at(i / mask_bits_per_elem) & (1llu << (i % mask_bits_per_elem));
        bool ref_hit = ref_hitmask.at(i / mask_bits_per_elem) & (1llu << (i % mask_bits_per_elem));
        if (layer_hit && ref_hit) {
          for (uint64_t j = 0; j < row_size; j++) {
            ASSERT_EQ(output[(i * row_size) + j], ref_output.at((i * row_size) + j));
          }
        }
      }
    }
    NVE_CHECK_(cudaFreeHost(output));
  }
  void Insert(std::vector<IndexT>& keys, std::vector<uint8_t>& datavectors, TableType db_type,
              uint64_t start_key = 0, uint64_t end_key = uint64_t(-1)) {
    if (keys.empty()) {
      return;
    }
    SetupKeys setup(keys, start_key, end_key, datavectors, m_row_size);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;
    auto data_buffer = setup.data_buffer;

    if (db_type != TableType::Ref) {
      const int64_t table_id = table_id_from_type(db_type);
      if (table_id >= 0) {
        m_layer->insert(m_ctx, num_keys, keys_buffer, m_row_size, m_row_size, data_buffer, table_id);
      }
    }

    bool ref_insert = (db_type == TableType::Device && m_gpu_tab) ||
                      (db_type == TableType::Host && m_host_tab) ||
                      (db_type == TableType::Remote && m_remote_tab) ||
                      (db_type == TableType::Ref);

    if (ref_insert) {
      auto keys_bw = std::make_shared<BufferWrapper<const void>>(
          m_ctx, "keys", keys_buffer, static_cast<size_t>(num_keys) * sizeof(IndexT));
      auto values_bw = std::make_shared<BufferWrapper<const void>>(
          m_ctx, "values", data_buffer,
          static_cast<size_t>(num_keys) * static_cast<size_t>(m_row_size));
      m_ref_tab->insert(m_ctx, num_keys, std::move(keys_bw), m_row_size, m_row_size,
                           std::move(values_bw));
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

    m_layer->update(m_ctx, num_keys, keys_buffer, m_row_size, m_row_size, data_buffer, -1);
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(
        m_ctx, "keys", keys_buffer, static_cast<size_t>(num_keys) * sizeof(IndexT));
    auto values_bw = std::make_shared<BufferWrapper<const void>>(
        m_ctx, "values", data_buffer,
        static_cast<size_t>(num_keys) * static_cast<size_t>(m_row_size));
    m_ref_tab->update(m_ctx, num_keys, std::move(keys_bw), m_row_size, m_row_size,
                         std::move(values_bw));
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

    m_layer->accumulate(m_ctx, num_keys, keys_buffer, m_row_size, m_row_size, data_buffer,
                        value_type, -1);
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(
        m_ctx, "keys", keys_buffer, static_cast<size_t>(num_keys) * sizeof(IndexT));
    auto updates_bw = std::make_shared<BufferWrapper<const void>>(
        m_ctx, "updates", data_buffer,
        static_cast<size_t>(num_keys) * static_cast<size_t>(m_row_size));
    m_ref_tab->update_accumulate(m_ctx, num_keys, std::move(keys_bw), m_row_size, m_row_size,
                                    std::move(updates_bw), value_type);
  }
  void Erase(std::vector<IndexT>& keys, uint64_t start_key = 0, uint64_t end_key = uint64_t(-1)) {
    if (keys.empty()) {
      return;
    }
    SetupKeys setup(keys, start_key, end_key);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;

    // only handling erase for all layers, otherwise ref becomes more complex (i.e. lookup on all
    // layers before erase and if any of them hit, don't erase from ref)
    m_layer->erase(m_ctx, num_keys, keys_buffer, -1);

    auto keys_bw = std::make_shared<BufferWrapper<const void>>(
        m_ctx, "keys", keys_buffer, static_cast<size_t>(num_keys) * sizeof(IndexT));
    m_ref_tab->erase(m_ctx, num_keys, std::move(keys_bw));
  }
  void Clear() {
    m_layer->clear(m_ctx);
    m_ref_tab->clear(m_ctx);
  }

  void Clear(TableType db_type) {
    switch (db_type) {
      case TableType::Device:
        if (m_gpu_tab) m_gpu_tab->clear(m_ctx);
        break;
      case TableType::Host:
        if (m_host_tab) m_host_tab->clear(m_ctx);
        break;
      case TableType::Remote:
        if (m_remote_tab) m_remote_tab->clear(m_ctx);
        break;
      case TableType::Ref:
        if (m_ref_tab) m_ref_tab->clear(m_ctx);
        break;
      default:
        throw std::runtime_error("Invalid database type in test!");
    }
  }

  void Wait() { m_ctx->wait(); }

 private:
  const int64_t m_row_size;
  table_ptr_t m_gpu_tab;
  host_table_ptr_t m_host_tab;
  host_table_ptr_t m_remote_tab;
  host_table_ptr_t m_ref_tab;
  std::shared_ptr<layer_type> m_layer{nullptr};
  context_ptr_t m_ctx;

  void check_redis_ready(int port) {
    const std::string cmd{"/usr/bin/redis-cli -p " + std::to_string(port) + " ping"};
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
      throw std::runtime_error("Failed to check Redis cluster! (popen)");
    }
    char out[16] = {0};
    if (fgets(out, 16, pipe) == NULL) {
      throw std::runtime_error("Failed to check Redis cluster! (fgets)");
    }
    auto ret = pclose(pipe);
    if (ret != 0 || strncmp (out, "PONG", 4)) {
      throw std::runtime_error("Redis cluster not ready!");
    }
    // All good!
  }
  int64_t table_id_from_type(TableType tt) {
    switch (tt) {
      case TableType::Device: return m_gpu_tab ? 0 : -1;
      case TableType::Host: return m_host_tab ? (m_gpu_tab ? 1 : 0) : -1;
      case TableType::Remote: return m_remote_tab ? (m_gpu_tab ? 1 : 0) + (m_host_tab ? 1 : 0) : -1;
      case TableType::Ref: 
        throw std::runtime_error("Unexpected type!");
        return -2;
    }
    throw std::runtime_error("Invalid table type!");
    return -3;
  }
};

// [Sanity] Init the layer
TEST_P(Hierachical, Init) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  SKIP_IF_HOST_TABLE_UNAVAILABLE(tc);
  HierLayerTest<int64_t> hlt(tc);
}

// [Sanity] insert 1key, lookup 1key
TEST_P(Hierachical, SingleLookup) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  SKIP_IF_HOST_TABLE_UNAVAILABLE(tc);
  HierLayerTest<int64_t> hlt(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, 1, tc.row_size, 0, 1ul << 30, tc.data_type);
  hlt.Insert(keys, data, TableType::Device);
  hlt.Insert(keys, data, TableType::Host);
  hlt.Insert(keys, data, TableType::Remote);
  NVE_CHECK_(cudaDeviceSynchronize());
  hlt.LookupAndCheck(keys);
}

// [Lookup] insert k, lookup 2k, get k hits, k misses (per table, k small enough to avoid partial
// inserts), clear, lookup
TEST_P(Hierachical, Lookup) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  SKIP_IF_HOST_TABLE_UNAVAILABLE(tc);
  HierLayerTest<int64_t> hlt(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, 1ul << 30, tc.data_type);
  hlt.Insert(keys, data, TableType::Device, 0, tc.test_keys / 2);
  hlt.Insert(keys, data, TableType::Host, tc.test_keys / 4, tc.test_keys * 3 / 4);
  hlt.Insert(keys, data, TableType::Remote, tc.test_keys / 2, tc.test_keys);
  NVE_CHECK_(cudaDeviceSynchronize());
  hlt.LookupAndCheck(keys);
  hlt.Clear();
  NVE_CHECK_(cudaDeviceSynchronize());
  hlt.LookupAndCheck(keys);
}

// [Update] insert k, update 2k, lookup 2k,...
TEST_P(Hierachical, Update) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  SKIP_IF_HOST_TABLE_UNAVAILABLE(tc);
  HierLayerTest<int64_t> hlt(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, 1ul << 30, tc.data_type);
  hlt.Insert(keys, data, TableType::Device, 0, tc.test_keys / 2);
  hlt.Insert(keys, data, TableType::Host, tc.test_keys / 4, tc.test_keys * 3 / 4);
  hlt.Insert(keys, data, TableType::Remote, tc.test_keys / 2, tc.test_keys);

  // change some data so there's something to update
  const auto num_rows = data.size() / static_cast<size_t>(tc.row_size);
  const uint64_t row_size = static_cast<uint64_t>(tc.row_size);
  for (uint64_t i = 0; i < num_rows; i+=2) {  // only updating half the rows
    for (uint64_t j = 0; j < row_size; j++) {
      data.at((i * row_size) + j) *= 3;
    }
  }

  NVE_CHECK_(cudaDeviceSynchronize());
  hlt.Update(keys, data);
  NVE_CHECK_(cudaDeviceSynchronize());
  hlt.LookupAndCheck(keys);
}

// [Insert] insert k, lookup k+k`, insert k`, lookup k+k`
TEST_P(Hierachical, Insert) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  SKIP_IF_HOST_TABLE_UNAVAILABLE(tc);
  HierLayerTest<int64_t> hlt(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, 1ul << 30, tc.data_type);
  hlt.Insert(keys, data, TableType::Device, 0, tc.test_keys / 6);
  hlt.Insert(keys, data, TableType::Host, tc.test_keys / 6, tc.test_keys * 2 / 6);
  hlt.Insert(keys, data, TableType::Remote, tc.test_keys * 2 / 6, tc.test_keys * 3 / 6);
  NVE_CHECK_(cudaDeviceSynchronize());
  hlt.LookupAndCheck(keys);

  hlt.Insert(keys, data, TableType::Device, tc.test_keys * 3 / 6, tc.test_keys * 4 / 6);
  hlt.Insert(keys, data, TableType::Host, tc.test_keys * 4 / 6, tc.test_keys * 5 / 6);
  hlt.Insert(keys, data, TableType::Remote, tc.test_keys * 5 / 6, tc.test_keys);

  NVE_CHECK_(cudaDeviceSynchronize());
  hlt.LookupAndCheck(keys);
}

// [Accumulate] insert k+k`, accumulate k+k``
TEST_P(Hierachical, Accumulate) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  SKIP_IF_HOST_TABLE_UNAVAILABLE(tc);
  HierLayerTest<int64_t> hlt(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, 1ul << 30, tc.data_type);
  hlt.Insert(keys, data, TableType::Device, 0, tc.test_keys / 4);
  hlt.Insert(keys, data, TableType::Host, tc.test_keys / 8, tc.test_keys * 3 / 8);
  hlt.Insert(keys, data, TableType::Remote, tc.test_keys / 4, tc.test_keys / 2);
  NVE_CHECK_(cudaDeviceSynchronize());
  hlt.LookupAndCheck(keys);

  // Accumulate some keys that were inserted and some that weren't (reusing original data as
  // gradients)
  hlt.Accumulate(keys, data, tc.data_type, tc.test_keys / 16, tc.test_keys * 3 / 16);
  hlt.Accumulate(keys, data, tc.data_type, tc.test_keys * 5 / 16, tc.test_keys * 9 / 16);

  NVE_CHECK_(cudaDeviceSynchronize());
  hlt.LookupAndCheck(keys);
}

//  [Erase] inserk k+k`, lookup k+k`+k``, erase k, lookup k+k`+k``
TEST_P(Hierachical, Erase) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  SKIP_IF_HOST_TABLE_UNAVAILABLE(tc);
  HierLayerTest<int64_t> hlt(tc);
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, 1ul << 30, tc.data_type);
  hlt.Insert(keys, data, TableType::Device, 0, tc.test_keys / 2);
  hlt.Insert(keys, data, TableType::Host, tc.test_keys / 4, tc.test_keys * 3 / 4);
  hlt.Insert(keys, data, TableType::Remote, tc.test_keys / 2, tc.test_keys);
  NVE_CHECK_(cudaDeviceSynchronize());
  hlt.LookupAndCheck(keys);

  hlt.Erase(keys, tc.test_keys * 2 / 6, tc.test_keys * 3 / 6);
  hlt.Erase(keys, tc.test_keys * 4 / 6, tc.test_keys * 5 / 6);

  NVE_CHECK_(cudaDeviceSynchronize());
  hlt.LookupAndCheck(keys);
}

//  * [overflow] small size table, insert k, lookup k, insert k`, lookup k+k`
//  * [large]
//  * [multiple execution contexts at once]
//  * [insert heuristics] default and custom

INSTANTIATE_TEST_SUITE_P(
    EmbLayer_SingleKey,
    Hierachical,
    ::testing::Values(
        //  TestCase: gpu_cache, host_cache, remote_ps, row_size, table_size, test_keys, insert_heuristic, data_type
        HierTestCase({true,  HostTableType::None,      false, int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float32}),
        HierTestCase({false, HostTableType::NVHashMap, false, int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float32}),
        HierTestCase({false, HostTableType::None,      true,  int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float32}),
        HierTestCase({true,  HostTableType::NVHashMap, true,  int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float32})
        ));

INSTANTIATE_TEST_SUITE_P(
    EmbLayer,
    Hierachical,
    ::testing::Values(
        //  TestCase: gpu_cache, host_cache, remote_ps, row_size, table_size, test_keys, insert_heuristic, data_type
        HierTestCase({true,  HostTableType::None,      false, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({false, HostTableType::NVHashMap, false, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({false, HostTableType::Abseil,    false, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({false, HostTableType::Phmap,     false, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({false, HostTableType::None,      true,  int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({true,  HostTableType::NVHashMap, true,  int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({true,  HostTableType::Abseil,    true,  int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({true,  HostTableType::Phmap,     true,  int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({true,  HostTableType::None,      false, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({false, HostTableType::NVHashMap, false, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({false, HostTableType::Abseil,    false, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({false, HostTableType::Phmap,     false, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({false, HostTableType::None,      true,  int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({true,  HostTableType::NVHashMap, true,  int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({true,  HostTableType::Abseil,    true,  int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({true,  HostTableType::Phmap,     true,  int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32})));

INSTANTIATE_TEST_SUITE_P(
    EmbLayer_WithRedis,
    Hierachical,
    ::testing::Values(
        //  TestCase: gpu_cache, host_cache, remote_ps, row_size, table_size, test_keys, insert_heuristic, data_Type
        HierTestCase({false, HostTableType::Redis, false, int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float16}),
        HierTestCase({true,  HostTableType::Redis, true,  int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float16}),
        HierTestCase({false, HostTableType::Redis, false, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({true,  HostTableType::Redis, true,  int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({false, HostTableType::Redis, false, int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float32}),
        HierTestCase({true,  HostTableType::Redis, true,  int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float32}),
        HierTestCase({false, HostTableType::Redis, false, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({true,  HostTableType::Redis, true,  int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),

        // Same coverage as above but against a standalone Redis in string mode (num_partitions == 0).
        HierTestCase({false, HostTableType::Redis_String, false, int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float16}),
        HierTestCase({true,  HostTableType::Redis_String, true,  int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float16}),
        HierTestCase({false, HostTableType::Redis_String, false, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({true,  HostTableType::Redis_String, true,  int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({false, HostTableType::Redis_String, false, int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float32}),
        HierTestCase({true,  HostTableType::Redis_String, true,  int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float32}),
        HierTestCase({false, HostTableType::Redis_String, false, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({true,  HostTableType::Redis_String, true,  int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32})));

INSTANTIATE_TEST_SUITE_P(
    EmbLayer_Large,
    Hierachical,
    ::testing::Values(
        //  TestCase: gpu_cache, host_cache, remote_ps, row_size, table_size, test_keys, insert_heuristic, data_type
        HierTestCase({true,  HostTableType::None,      false, int64_t(1) << 12, int64_t(1) << 30, int64_t(1) << 16, false, DataType_t::Float32}),
        HierTestCase({false, HostTableType::NVHashMap, false, int64_t(1) << 12, int64_t(1) << 30, int64_t(1) << 16, false, DataType_t::Float32}),
        HierTestCase({false, HostTableType::Abseil,    false, int64_t(1) << 12, int64_t(1) << 30, int64_t(1) << 16, false, DataType_t::Float32}),
        HierTestCase({false, HostTableType::Phmap,     false, int64_t(1) << 12, int64_t(1) << 30, int64_t(1) << 16, false, DataType_t::Float32}),
        HierTestCase({false, HostTableType::None,      true,  int64_t(1) << 12, int64_t(1) << 30, int64_t(1) << 16, false, DataType_t::Float32}),
        HierTestCase({true,  HostTableType::NVHashMap, true,  int64_t(1) << 12, int64_t(1) << 30, int64_t(1) << 16, false, DataType_t::Float32}),
        HierTestCase({true,  HostTableType::Abseil,    true,  int64_t(1) << 12, int64_t(1) << 30, int64_t(1) << 16, false, DataType_t::Float32}),
        HierTestCase({true,  HostTableType::Phmap,     true,  int64_t(1) << 12, int64_t(1) << 30, int64_t(1) << 16, false, DataType_t::Float32})));

class HierachicalSpecialConfig : public ::testing::TestWithParam<HierTestCase> {};

TEST_P(HierachicalSpecialConfig, InflightInsert) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  SKIP_IF_HOST_TABLE_UNAVAILABLE(tc);
  HierLayerTest<int64_t> hlt(tc);
  ASSERT_EQ(tc.insert_heuristic, true); // The mock remote is needed to properly enable the auto insert
  ASSERT_GE(tc.test_keys, hlt.MIN_INSERT_HEURISTIC_SIZE); // Must have at enough keys to trigger auto insert (instead of accumulation)
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, 1ul << 30, tc.data_type);

  // Insert only to mock remote table so that the output can be resolved properly (needed for the auto insert)
  // This will also trigger an insert to the ref
  hlt.Insert(keys, data, TableType::Remote);

  // This lookup will trigger auto insert without comparison to ref
  hlt.LookupAndCheck(keys, 0, uint64_t(-1), -1.f);
  hlt.Wait(); // Wait for auto insert to finish

  // Now remove everything from the remote table to make sure hits we get in the next lookup will only from gpu/host table
  hlt.Clear(TableType::Remote);
  NVE_CHECK_(cudaDeviceSynchronize()); // Wait for potential poending clear kernel on the GPU table

  // Now compare after the auto insert and expect hit rate to match the ref (ref implicitly had an insert when we called insert on the remote table)
  hlt.LookupAndCheck(keys);
}

// This helper is needed to use the EXPECT_FATAL_FAILURE macro, which disallows using local variables
static struct {
  HierLayerTest<int64_t>* hlt;
  std::vector<int64_t>* keys;
  uint64_t start_key;
  uint64_t end_key;
  float hit_tol;
} FailHelperParams;

static void ExpectedFailLookupHelper() {
  FailHelperParams.hlt->LookupAndCheck(
    *(FailHelperParams.keys),
    FailHelperParams.start_key,
    FailHelperParams.end_key,
    FailHelperParams.hit_tol);
}

TEST_P(HierachicalSpecialConfig, InflightInsertAccumulation) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  SKIP_IF_HOST_TABLE_UNAVAILABLE(tc);
  HierLayerTest<int64_t> hlt(tc);
  ASSERT_EQ(tc.insert_heuristic, true); // The mock remote is needed to properly enable the auto insert
  ASSERT_GE(tc.test_keys, hlt.MIN_INSERT_HEURISTIC_SIZE); // Must have at enough keys to trigger auto insert (instead of accumulation)
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, 1ul << 30, tc.data_type);

  // Insert only to mock remote table so that the output can be resolved properly (needed for the auto insert)
  // This will also trigger an insert to the ref
  hlt.Insert(keys, data, TableType::Remote, 0, hlt.MIN_INSERT_HEURISTIC_SIZE);
  hlt.Wait(); // Wait for insert to finish

  // This lookup will trigger auto insert accumulation
  hlt.LookupAndCheck(keys, 0, hlt.MIN_INSERT_HEURISTIC_SIZE / 2);

  // This lookup will trigger auto insert (accumulation has enough keys to run)
  hlt.LookupAndCheck(keys, hlt.MIN_INSERT_HEURISTIC_SIZE / 2, hlt.MIN_INSERT_HEURISTIC_SIZE);
  hlt.Wait(); // Wait for auto insert to finish

  // Now remove everything from the remote table to make sure hits we get in the next lookup will only from gpu/host table
  hlt.Clear(TableType::Remote);

  // Now check results match ref
  hlt.LookupAndCheck(keys);
  hlt.Wait(); // Wait for auto insert to finish before tear down
  NVE_CHECK_(cudaDeviceSynchronize());
}

// This test checks insert accumulation doesn't trigger prematurely
TEST_P(HierachicalSpecialConfig, InflightInsertAccumulationPremature) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  SKIP_IF_HOST_TABLE_UNAVAILABLE(tc);
  HierLayerTest<int64_t> hlt(tc);
  ASSERT_EQ(tc.insert_heuristic, true); // The mock remote is needed to properly enable the auto insert
  ASSERT_GE(tc.test_keys, hlt.MIN_INSERT_HEURISTIC_SIZE); // Must have at enough keys to trigger auto insert (instead of accumulation)
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, 1ul << 30, tc.data_type);

  // Insert only to mock remote table so that the output can be resolved properly (needed for the auto insert)
  // This will also trigger an insert to the ref
  hlt.Insert(keys, data, TableType::Remote, 0, hlt.MIN_INSERT_HEURISTIC_SIZE);
  hlt.Wait(); // Wait for insert to finish

  // This lookup will trigger auto insert accumulation
  hlt.LookupAndCheck(keys, 0, hlt.MIN_INSERT_HEURISTIC_SIZE / 2);
  hlt.Wait(); // Wait in case auto insert was erroneously triggered

  // Now remove everything from the remote table to make sure hits we get in the next lookup will only from gpu/host table
  hlt.Clear(TableType::Remote);
  hlt.Wait(); // Wait for auto insert to finish
  NVE_CHECK_(cudaDeviceSynchronize());

  // This lookup should fail since we didn't accumulate enough to trigger an insert and there's no fallback in the remote
  FailHelperParams.hlt = &hlt;
  FailHelperParams.keys = &keys;
  FailHelperParams.start_key = 0;
  FailHelperParams.end_key = hlt.MIN_INSERT_HEURISTIC_SIZE / 4;
  FailHelperParams.hit_tol = 1e-3f;
  EXPECT_FATAL_FAILURE(ExpectedFailLookupHelper(), "exceeds hit_tol");
  hlt.Wait(); // Wait for auto insert to finish
  NVE_CHECK_(cudaDeviceSynchronize());
}

// This test checks that modify ops (update in this case) during insert accumulation, flush the accumulated keys
TEST_P(HierachicalSpecialConfig, InflightInsertAccumulationFlush) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto tc = GetParam();
  SKIP_IF_HOST_TABLE_UNAVAILABLE(tc);
  HierLayerTest<int64_t> hlt(tc);
  ASSERT_EQ(tc.insert_heuristic, true); // The mock remote is needed to properly enable the auto insert
  ASSERT_GE(tc.test_keys, hlt.MIN_INSERT_HEURISTIC_SIZE); // Must have at enough keys to trigger auto insert (instead of accumulation)
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;

  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, 1ul << 30, tc.data_type);

  // Insert only to mock remote table so that the output can be resolved properly (needed for the auto insert)
  // This will also trigger an insert to the ref
  hlt.Insert(keys, data, TableType::Remote, 0, hlt.MIN_INSERT_HEURISTIC_SIZE);

  // This lookup will trigger auto insert accumulation
  hlt.LookupAndCheck(keys, 0, hlt.MIN_INSERT_HEURISTIC_SIZE / 2);
  
  // The Update should flush the accumulated keys for insert
  hlt.Update(keys, data);

  // This lookup should trigger auto insert accumulation but not trigger an actual insert since the previous accumulation was flushed
  hlt.LookupAndCheck(keys, hlt.MIN_INSERT_HEURISTIC_SIZE / 2, hlt.MIN_INSERT_HEURISTIC_SIZE);
  hlt.Wait(); // Wait in case auto insert was erroneously triggered

  // Now remove everything from the remote table to make sure hits we get in the next lookup will only from gpu/host table
  hlt.Clear(TableType::Remote);

  // This lookup should fail since no actual insert was performed on the gpu/host table
  FailHelperParams.hlt = &hlt;
  FailHelperParams.keys = &keys;
  FailHelperParams.start_key = 0;
  FailHelperParams.end_key = hlt.MIN_INSERT_HEURISTIC_SIZE / 4;
  FailHelperParams.hit_tol = 1e-3f;
  EXPECT_FATAL_FAILURE(ExpectedFailLookupHelper(), "exceeds hit_tol");
  hlt.Wait(); // Wait for auto insert to finish
  NVE_CHECK_(cudaDeviceSynchronize());
}

// Test that keys missing from all tables get filled with the configured default embedding,
// while keys that hit retain their stored values.
TEST(HierachicalDefaultEmbedding, FillsMissesFromConfig) {
  cudaGetLastError();
  using IndexT = int64_t;
  constexpr uint64_t row_size = 64;
  constexpr uint64_t num_keys = 32;
  constexpr uint64_t num_inserted = 16;
  constexpr uint8_t default_byte = 0xAB;
  constexpr uint8_t inserted_byte = 0x42;

  HostTableConfig host_cfg;
  host_cfg.value_dtype = DataType_t::Float32;
  host_cfg.max_value_size = static_cast<int64_t>(row_size);
  auto host_tab = std::make_shared<MockHostTable<IndexT>>(host_cfg, true /*functional_ref*/);

  std::vector<uint8_t> default_emb(row_size, default_byte);

  typename HierarchicalEmbeddingLayer<IndexT>::Config layer_cfg;
  layer_cfg.insert_heuristic = std::make_shared<NeverInsertHeuristic>();
  layer_cfg.default_embedding = default_emb;
  std::vector<table_ptr_t> tables{host_tab};
  auto layer = std::make_shared<HierarchicalEmbeddingLayer<IndexT>>(layer_cfg, tables);
  auto ctx = layer->create_execution_context(0, 0, nullptr, nullptr);
  layer->clear(ctx);

  std::vector<IndexT> keys(num_keys);
  for (uint64_t i = 0; i < num_keys; i++) keys[i] = static_cast<IndexT>(i);
  std::vector<uint8_t> insert_data(num_inserted * row_size, inserted_byte);
  layer->insert(ctx, static_cast<int64_t>(num_inserted), keys.data(),
                static_cast<int64_t>(row_size), static_cast<int64_t>(row_size),
                insert_data.data(), 0 /*table_id*/);
  ctx->wait();

  std::vector<uint8_t> output(num_keys * row_size, 0x00);
  constexpr uint64_t mask_bits = sizeof(max_bitmask_repr_t) * 8;
  std::vector<max_bitmask_repr_t> hitmask((num_keys + mask_bits - 1) / mask_bits, 0);
  layer->lookup(ctx, static_cast<int64_t>(num_keys), keys.data(), output.data(),
                static_cast<int64_t>(row_size), hitmask.data(),
                nullptr /*pool_params*/, nullptr /*hitrates*/);
  ctx->wait();

  for (uint64_t i = 0; i < num_keys; i++) {
    const bool hit = (hitmask[i / mask_bits] >> (i % mask_bits)) & 0x1u;
    if (i < num_inserted) {
      ASSERT_TRUE(hit) << "expected hit for key " << i;
      for (uint64_t j = 0; j < row_size; j++) {
        ASSERT_EQ(inserted_byte, output[i * row_size + j])
            << "key " << i << " byte " << j;
      }
    } else {
      ASSERT_FALSE(hit) << "expected miss for key " << i;
      for (uint64_t j = 0; j < row_size; j++) {
        ASSERT_EQ(default_byte, output[i * row_size + j])
            << "key " << i << " byte " << j;
      }
    }
  }
}

// Sanity: when default_embedding is unset, miss outputs are not overwritten.
TEST(HierachicalDefaultEmbedding, NoDefaultLeavesMissesUntouched) {
  cudaGetLastError();
  using IndexT = int64_t;
  constexpr uint64_t row_size = 32;
  constexpr uint64_t num_keys = 8;
  constexpr uint8_t sentinel = 0x77;

  HostTableConfig host_cfg;
  host_cfg.value_dtype = DataType_t::Float32;
  host_cfg.max_value_size = static_cast<int64_t>(row_size);
  auto host_tab = std::make_shared<MockHostTable<IndexT>>(host_cfg, true /*functional_ref*/);

  typename HierarchicalEmbeddingLayer<IndexT>::Config layer_cfg;
  layer_cfg.insert_heuristic = std::make_shared<NeverInsertHeuristic>();
  // default_embedding intentionally left empty
  std::vector<table_ptr_t> tables{host_tab};
  auto layer = std::make_shared<HierarchicalEmbeddingLayer<IndexT>>(layer_cfg, tables);
  auto ctx = layer->create_execution_context(0, 0, nullptr, nullptr);
  layer->clear(ctx);

  std::vector<IndexT> keys(num_keys);
  for (uint64_t i = 0; i < num_keys; i++) keys[i] = static_cast<IndexT>(1000 + i);  // none inserted
  std::vector<uint8_t> output(num_keys * row_size, sentinel);
  constexpr uint64_t mask_bits = sizeof(max_bitmask_repr_t) * 8;
  std::vector<max_bitmask_repr_t> hitmask((num_keys + mask_bits - 1) / mask_bits, 0);
  layer->lookup(ctx, static_cast<int64_t>(num_keys), keys.data(), output.data(),
                static_cast<int64_t>(row_size), hitmask.data(),
                nullptr /*pool_params*/, nullptr /*hitrates*/);
  ctx->wait();

  for (uint64_t i = 0; i < num_keys; i++) {
    const bool hit = (hitmask[i / mask_bits] >> (i % mask_bits)) & 0x1u;
    ASSERT_FALSE(hit) << "expected miss for key " << i;
    for (uint64_t j = 0; j < row_size; j++) {
      ASSERT_EQ(sentinel, output[i * row_size + j])
          << "key " << i << " byte " << j;
    }
  }
}

// Exercises HierarchicalEmbeddingLayer wrapping a single LinearHostTable — the
// configuration that backs the Python HostLayer binding.
TEST(HierachicalLinearHost, LookupReadsHostBuffer) {
  cudaGetLastError();
  using IndexT = int64_t;
  using DataT = float;
  constexpr uint64_t row_elements = 8;
  constexpr uint64_t row_size = row_elements * sizeof(DataT);
  constexpr uint64_t num_rows = 64;
  constexpr uint64_t num_keys = 16;

  // Pre-populate the host buffer: row i = [i*10, i*10+1, ...].
  DataT* h_table = nullptr;
  NVE_CHECK_(cudaMallocHost(&h_table, num_rows * row_size));
  for (uint64_t i = 0; i < num_rows; ++i) {
    for (uint64_t j = 0; j < row_elements; ++j) {
      h_table[i * row_elements + j] = static_cast<DataT>(i * 10 + j);
    }
  }

  LinearHostTableConfig table_cfg;
  table_cfg.value_dtype = DataType_t::Float32;
  table_cfg.key_size = sizeof(IndexT);
  table_cfg.max_value_size = static_cast<int64_t>(row_size);
  table_cfg.emb_table = h_table;
  auto host_tab = std::make_shared<LinearHostTable<IndexT>>(table_cfg);

  typename HierarchicalEmbeddingLayer<IndexT>::Config layer_cfg;
  layer_cfg.layer_name = "host_layer";
  layer_cfg.insert_heuristic = std::make_shared<NeverInsertHeuristic>();
  std::vector<table_ptr_t> tables{host_tab};
  auto layer = std::make_shared<HierarchicalEmbeddingLayer<IndexT>>(layer_cfg, tables);
  auto ctx = layer->create_execution_context(0, 0, nullptr, nullptr);

  std::vector<IndexT> keys(num_keys);
  for (uint64_t i = 0; i < num_keys; ++i) keys[i] = static_cast<IndexT>(i * 3);

  std::vector<DataT> output(num_keys * row_elements, DataT{0});
  constexpr uint64_t mask_bits = sizeof(max_bitmask_repr_t) * 8;
  std::vector<max_bitmask_repr_t> hitmask((num_keys + mask_bits - 1) / mask_bits, 0);

  layer->lookup(ctx, static_cast<int64_t>(num_keys), keys.data(), output.data(),
                static_cast<int64_t>(row_size), hitmask.data(),
                nullptr /*pool_params*/, nullptr /*hitrates*/);
  ctx->wait();

  for (uint64_t i = 0; i < num_keys; ++i) {
    const bool hit = (hitmask[i / mask_bits] >> (i % mask_bits)) & 0x1u;
    ASSERT_TRUE(hit) << "expected hit for key " << keys[i];
    for (uint64_t j = 0; j < row_elements; ++j) {
      EXPECT_FLOAT_EQ(h_table[static_cast<uint64_t>(keys[i]) * row_elements + j],
                      output[i * row_elements + j])
          << "key=" << keys[i] << " col=" << j;
    }
  }

  NVE_CHECK_(cudaFreeHost(h_table));
}

// Update through the layer should be visible on the next lookup.
TEST(HierachicalLinearHost, UpdateThenLookup) {
  cudaGetLastError();
  using IndexT = int64_t;
  using DataT = float;
  constexpr uint64_t row_elements = 4;
  constexpr uint64_t row_size = row_elements * sizeof(DataT);
  constexpr uint64_t num_rows = 32;
  constexpr uint64_t num_keys = 4;

  DataT* h_table = nullptr;
  NVE_CHECK_(cudaMallocHost(&h_table, num_rows * row_size));
  std::fill(h_table, h_table + num_rows * row_elements, DataT{0});

  LinearHostTableConfig table_cfg;
  table_cfg.value_dtype = DataType_t::Float32;
  table_cfg.key_size = sizeof(IndexT);
  table_cfg.max_value_size = static_cast<int64_t>(row_size);
  table_cfg.emb_table = h_table;
  auto host_tab = std::make_shared<LinearHostTable<IndexT>>(table_cfg);

  typename HierarchicalEmbeddingLayer<IndexT>::Config layer_cfg;
  layer_cfg.layer_name = "host_layer";
  layer_cfg.insert_heuristic = std::make_shared<NeverInsertHeuristic>();
  std::vector<table_ptr_t> tables{host_tab};
  auto layer = std::make_shared<HierarchicalEmbeddingLayer<IndexT>>(layer_cfg, tables);
  auto ctx = layer->create_execution_context(0, 0, nullptr, nullptr);

  std::vector<IndexT> keys{1, 5, 9, 13};
  std::vector<DataT> updates(num_keys * row_elements);
  for (uint64_t i = 0; i < num_keys; ++i) {
    for (uint64_t j = 0; j < row_elements; ++j) {
      updates[i * row_elements + j] = static_cast<DataT>(keys[i] * 100 + static_cast<IndexT>(j));
    }
  }

  layer->update(ctx, static_cast<int64_t>(num_keys), keys.data(),
                static_cast<int64_t>(row_size), static_cast<int64_t>(row_size),
                updates.data(), /*table_id=*/-1);
  ctx->wait();

  std::vector<DataT> output(num_keys * row_elements, DataT{0});
  constexpr uint64_t mask_bits = sizeof(max_bitmask_repr_t) * 8;
  std::vector<max_bitmask_repr_t> hitmask((num_keys + mask_bits - 1) / mask_bits, 0);
  layer->lookup(ctx, static_cast<int64_t>(num_keys), keys.data(), output.data(),
                static_cast<int64_t>(row_size), hitmask.data(),
                nullptr /*pool_params*/, nullptr /*hitrates*/);
  ctx->wait();

  for (uint64_t i = 0; i < num_keys; ++i) {
    const bool hit = (hitmask[i / mask_bits] >> (i % mask_bits)) & 0x1u;
    ASSERT_TRUE(hit) << "expected hit for key " << keys[i];
    for (uint64_t j = 0; j < row_elements; ++j) {
      EXPECT_FLOAT_EQ(updates[i * row_elements + j],
                      output[i * row_elements + j])
          << "key=" << keys[i] << " col=" << j;
    }
  }

  NVE_CHECK_(cudaFreeHost(h_table));
}

// These cases must have remote PS for the test flow to work correctly
INSTANTIATE_TEST_SUITE_P(
    EmbLayer,
    HierachicalSpecialConfig,
    ::testing::Values(
        //  TestCase: gpu_cache, host_cache, remote_ps, row_size, table_size, test_keys, insert_heuristic, data_Type
        HierTestCase({true,  HostTableType::None,      true, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, true, DataType_t::Float32}),
        HierTestCase({false, HostTableType::NVHashMap, true, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, true, DataType_t::Float32}),
        HierTestCase({false, HostTableType::Abseil,    true, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, true, DataType_t::Float32}),
        HierTestCase({false, HostTableType::Phmap,     true, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, true, DataType_t::Float32}),
        HierTestCase({true,  HostTableType::NVHashMap, true, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, true, DataType_t::Float32}),
        HierTestCase({true,  HostTableType::Abseil,    true, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, true, DataType_t::Float32}),
        HierTestCase({true,  HostTableType::Phmap,     true, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, true, DataType_t::Float32})));

}  // namespace nve
