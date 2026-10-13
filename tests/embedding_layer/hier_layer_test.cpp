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
#include <unistd.h>
#include "emb_layer_utils.hpp"
#include "test_utils.hpp"
#include <buffer_wrapper.hpp>
#include <cuda_support.hpp>
#include <embedding_layer.hpp>
#include <hierarchical_embedding_layer.hpp>
#include <gpu_table.hpp>
#include <host_table.hpp>
#include <plugin/plugin_loader.hpp>
#include <linear_host_table.hpp>
#include <cmath>
#include <filesystem>
#include <memory>
#include <optional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <execution_context.hpp>
#include <thread_pool.hpp>
#include <nve_types.hpp>
#include "mock_host_table.hpp"

namespace nve {

// Tiers a HierTestCase can combine: Gpu and Mock are always available in-process, the rest are
// pluggable host-table backends that may be compiled out.
enum class TestTableType : uint64_t {
  None = 0,
  Gpu,
  Mock,
  NVHashMap,
  Redis,
  Abseil,
  Phmap,
  Redis_String,
  Sph,
  RocksDB,
};

enum class TableType : uint64_t {
  Device,
  Host,
  Remote,
  Ref,
};

static bool is_table_type_available(TestTableType ht) {
  switch (ht) {
    case TestTableType::None:
    case TestTableType::Gpu:
    case TestTableType::Mock:
      return true;
    case TestTableType::NVHashMap:
#ifdef NVE_FEATURE_NVHM_PLUGIN
      return true;
#else
      return false;
#endif
    case TestTableType::Redis:
    case TestTableType::Redis_String:
#ifdef NVE_FEATURE_REDIS_PLUGIN
      return true;
#else
      return false;
#endif
    case TestTableType::Abseil:
#ifdef NVE_FEATURE_ABSEIL_PLUGIN
      return true;
#else
      return false;
#endif
    case TestTableType::Phmap:
#ifdef NVE_FEATURE_PHMAP_PLUGIN
      return true;
#else
      return false;
#endif
    case TestTableType::Sph:
#ifdef NVE_FEATURE_SPH_PLUGIN
      return true;
#else
      return false;
#endif
    case TestTableType::RocksDB:
#ifdef NVE_FEATURE_ROCKSDB_PLUGIN
      return true;
#else
      return false;
#endif
    default:
      return false;
  }
}

// SPH implements neither erase nor update_accumulate — both throw — so the two
// tests that exercise them skip for it. Every other type runs unchanged.
static bool host_table_supports_erase(TestTableType ht) { return ht != TestTableType::Sph; }
static bool host_table_supports_accumulate(TestTableType ht) { return ht != TestTableType::Sph; }

static const char* test_table_type_name(TestTableType ht) {
  switch (ht) {
    case TestTableType::None: return "None";
    case TestTableType::Gpu: return "Gpu";
    case TestTableType::Mock: return "Mock";
    case TestTableType::NVHashMap: return "NVHashMap";
    case TestTableType::Redis: return "Redis";
    case TestTableType::Abseil: return "Abseil";
    case TestTableType::Phmap: return "Phmap";
    case TestTableType::Redis_String: return "Redis_String";
    case TestTableType::Sph: return "Sph";
    case TestTableType::RocksDB: return "RocksDB";
    default: return "Unknown";
  }
}

#define SKIP_IF_HOST_TABLE_UNAVAILABLE(tc) \
  do { \
    for (auto tt : (tc).tables) { \
      if (!is_table_type_available(tt)) { \
        GTEST_SKIP() << "Table type " << test_table_type_name(tt) << " is not available"; \
      } \
    } \
  } while (0)

#define SKIP_IF_ERASE_UNSUPPORTED(tc) \
  do { \
    for (auto tt : (tc).tables) { \
      if (!host_table_supports_erase(tt)) { \
        GTEST_SKIP() << test_table_type_name(tt) << " does not implement erase"; \
      } \
    } \
  } while (0)

#define SKIP_IF_ACCUMULATE_UNSUPPORTED(tc) \
  do { \
    for (auto tt : (tc).tables) { \
      if (!host_table_supports_accumulate(tt)) { \
        GTEST_SKIP() << test_table_type_name(tt) << " does not implement accumulate"; \
      } \
    } \
  } while (0)

// Fixture for functional tests. `tables` lists the tiers to build, in order: at most one Gpu, at
// most two pluggable host backends, at most one Mock. The remote PS slot is taken by Mock or by
// the second pluggable backend (e.g. {Gpu, NVHashMap, RocksDB}), never both.
struct HierTestCase {
  std::vector<TestTableType> tables;
  int64_t row_size;
  int64_t table_size;
  size_t test_keys;
  bool insert_heuristic;
  DataType_t data_type; // for now using the same type for table and accumulate
};
class Hierachical : public ::testing::TestWithParam<HierTestCase> {};

template <typename IndexT>
class HierLayerTest {
  // Declared first so it is destroyed last, i.e. after the tables below have closed the DB.
  std::optional<nve_test::TempPathCleanup> rocksdb_cleanup_;

 public:
  using layer_type = HierarchicalEmbeddingLayer<IndexT>;
  static constexpr int DEVICE_ID = 0;
  static constexpr int64_t MAX_MODIFY_SIZE = (1l << 20);
  static constexpr int64_t MIN_INSERT_HEURISTIC_SIZE = (1l << 10);

  HierLayerTest(const std::vector<TestTableType>& tables_cfg, int64_t row_size,
                int64_t table_size, bool insert_heuristic, DataType_t data_type)
      : row_size_(row_size) {
    for (TestTableType tt : tables_cfg) {
      switch (tt) {
        case TestTableType::None:
          break;  // do nothing
        case TestTableType::Gpu:
          if (gpu_tab_) {
            NVE_THROW_("Invalid HierTestCase: more than one Gpu tier requested");
          }
          init_gpu_table(row_size, table_size, data_type);
          break;
        case TestTableType::Mock:
          if (remote_tab_) {
            NVE_THROW_("Invalid HierTestCase: more than one Mock tier requested");
          }
          init_remote_table(row_size, data_type);
          break;
        default:
          if (!host_tab_) {
            host_tab_ = make_host_table(tt, row_size, table_size, data_type);
          } else if (!remote_tab_) {
            remote_tab_ = make_host_table(tt, row_size, table_size, data_type);
          } else {
            NVE_THROW_("Invalid HierTestCase: more than two host table tiers requested");
          }
          break;
      }
    }
    // init layer. Tier order here is always Gpu -> Host -> Remote regardless of tables_cfg's
    // order above -- tables_cfg only controls which tiers get built, not the layer's tier order.
    typename HierarchicalEmbeddingLayer<IndexT>::Config cfg;
    std::vector<std::shared_ptr<nve::Table>> tables;
    std::vector<bool> heuristic_results;
    if (gpu_tab_) {
      tables.push_back(gpu_tab_);
      heuristic_results.push_back(true);
    }
    if (host_tab_) {
      tables.push_back(host_tab_);
      heuristic_results.push_back(true);
    }
    if (remote_tab_) {
      tables.push_back(remote_tab_);
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
    layer_ = std::make_shared<HierarchicalEmbeddingLayer<IndexT>>(cfg, tables);

    // init context
    ctx_ = layer_->create_execution_context(0, 0, nullptr, nullptr);
    // clear the layer tables (mostly needed for persistent Redis cluster used in testing)
    layer_->clear(ctx_);

    // create ref (using single mock table)
    HostTableConfig mock_cfg;
    mock_cfg.value_dtype = data_type;
    mock_cfg.max_value_size = row_size; // MockHostTable::combine() uses this as the input row stride
    ref_tab_ = std::make_shared<MockHostTable<IndexT>>(mock_cfg, true /*functional_ref*/);
  }
  HierLayerTest(HierTestCase tc)
      : HierLayerTest<IndexT>(tc.tables, tc.row_size, tc.table_size, tc.insert_heuristic,
                               tc.data_type) {}
  ~HierLayerTest() {
    layer_->clear(ctx_); // clearing potentially persistent tables (e.g. redis)
    Wait();
  }

 private:
  void init_gpu_table(int64_t row_size, int64_t table_size, DataType_t data_type) {
    GPUTableConfig cfg;
    cfg.device_id = DEVICE_ID;
    cfg.cache_size = static_cast<size_t>(table_size);
    cfg.max_modify_size = MAX_MODIFY_SIZE;
    cfg.row_size_in_bytes = row_size;
    cfg.uvm_table = nullptr;
    cfg.value_dtype = data_type;
    gpu_tab_ = std::make_shared<GpuTable<IndexT>>(cfg);
  }

  void init_remote_table(int64_t row_size, DataType_t data_type) {
    HostTableConfig remote_cfg;
    remote_cfg.value_dtype = data_type;
    remote_cfg.max_value_size = row_size;
    remote_tab_ = std::make_shared<MockHostTable<IndexT>>(remote_cfg, true /*functional_ref*/);
  }

  table_ptr_t make_host_table(TestTableType host_table, int64_t row_size, int64_t table_size,
                              DataType_t data_type) {
    switch (host_table) {
      case TestTableType::NVHashMap: {
        nlohmann::json nvhm_conf = {
                                  {"key_size", sizeof(IndexT)},
                                  {"max_value_size", row_size},
                                  {"value_dtype", to_string(data_type)},
                                  {"num_partitions", 2},
                                  {"initial_capacity", 1024},
                                  {"value_alignment", 32},
                                  {"auto_shrink", true},
                                  {"overflow_policy",
                                  {{"overflow_margin", (table_size / row_size)},
                                    {"handler", "evict_lru"},
                                    {"resolution_margin", 0.5}}}};
        table_factory_ptr_t nvhm_fac{nve_test::plugin_factory("nvhm")};
        return nvhm_fac->produce(4711, nvhm_conf);
      }
      case TestTableType::Redis: {
        check_redis_ready(7000);
        nlohmann::json redis_conf = {
                                      {"num_partitions", 1},
                                      {"key_size", sizeof(IndexT)},
                                      {"max_value_size", row_size},
                                      {"value_dtype", to_string(data_type)},
                                    };
        table_factory_ptr_t redis_fac{nve_test::plugin_factory("redis", R"(
            {
              "address": "localhost:7000"
            })"_json)};
        return redis_fac->produce(4712, redis_conf);
      }
      case TestTableType::Redis_String: {
        check_redis_ready(6379);
        nlohmann::json redis_conf = {
                                      {"num_partitions", 0},
                                      {"key_size", sizeof(IndexT)},
                                      {"max_value_size", row_size},
                                      {"value_dtype", to_string(data_type)},
                                    };
        table_factory_ptr_t redis_fac{nve_test::plugin_factory("redis", R"(
            {
              "address": "localhost:6379",
              "single_node": true
            })"_json)};
        return redis_fac->produce(4715, redis_conf);
      }
      case TestTableType::Abseil: {
        nlohmann::json abseil_conf = {
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
        table_factory_ptr_t abseil_fac{nve_test::plugin_factory("abseil")};
        return abseil_fac->produce(4713, abseil_conf);
      }
      case TestTableType::Phmap: {
        nlohmann::json phm_conf = {
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
        table_factory_ptr_t phm_fac{nve_test::plugin_factory("phmap")};
        return phm_fac->produce(4714, phm_conf);
      }
      case TestTableType::Sph: {
        // GPU-resident, so it takes device_id and sizes its capacity from the
        // same byte budget the GPU table gets. No overflow policy: SPH treats
        // capacity as a target and drops keys past it rather than evicting.
        nlohmann::json sph_conf = {
                                  {"device_id", DEVICE_ID},
                                  {"cache_size", table_size},
                                  {"row_size_in_bytes", row_size},
                                  {"value_dtype", to_string(data_type)}};
        table_factory_ptr_t sph_fac{nve_test::plugin_factory("sph-experimental")};
        return sph_fac->produce(4716, sph_conf);
      }
      case TestTableType::RocksDB: {
        // Persistent on-disk store, so it gets a per-process scratch directory that is removed
        // once the table has closed the DB. No overflow policy: capacity is bounded by disk.
        // A small max_batch_size makes the multi-key tests span several MultiGet/WriteBatch rounds.
        const auto db_path{std::filesystem::temp_directory_path() /
                           ("nve_hier_layer_test_rocksdb_" + std::to_string(getpid()))};
        rocksdb_cleanup_.emplace(db_path);
        nlohmann::json rocksdb_conf = {
                                  {"key_size", sizeof(IndexT)},
                                  {"max_value_size", row_size},
                                  {"value_dtype", to_string(data_type)},
                                  {"max_batch_size", 1024}};
        table_factory_ptr_t rocksdb_fac{nve_test::plugin_factory("rocksdb", {{"path", db_path.string()}})};
        return rocksdb_fac->produce(4717, rocksdb_conf);
      }
      default:
        throw std::runtime_error("Invalid host table type");
    }
  }

 public:
  // lookup on layer and ref, compare results (when hit_tol >= 0)
  void LookupAndCheck(std::vector<IndexT>& keys, uint64_t start_key = 0,
                      uint64_t end_key = uint64_t(-1), float hit_tol = 0.01f) {
    if (keys.empty()) {
      return;
    }

    SetupKeys setup(keys, start_key, end_key);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;

    size_t output_size = static_cast<size_t>(num_keys * row_size_);
    int8_t* output{nullptr};
    NVE_CHECK_(cudaMallocHost(&output, output_size));
    NVE_CHECK_(output != 0);
    std::vector<int8_t> ref_output(output_size);

    std::vector<bitmask64_t> hitmask(to_uint(ceil_div(num_keys, bitmask64::num_bits)));
    std::vector<bitmask64_t> ref_hitmask(hitmask.size());

    std::vector<float> hitrates(3);

    layer_->lookup(ctx_, num_keys, keys_buffer, output, row_size_, hitmask.data(),
                    nullptr /*pool_params*/, hitrates.data());
    NVE_CHECK_(cudaDeviceSynchronize());
    int64_t ref_hits;
    ref_tab_->reset_lookup_counter(ctx_);
    {
      auto keys_bw = std::make_shared<BufferWrapper<const void>>(
          ctx_, "keys", keys_buffer, static_cast<size_t>(num_keys) * sizeof(IndexT));
      auto hit_mask_bw = std::make_shared<BufferWrapper<bitmask64_t>>(
          ctx_, "hit_mask", ref_hitmask.data(), hitmask.size() * sizeof(bitmask64_t));
      auto values_bw = std::make_shared<BufferWrapper<void>>(
          ctx_, "values", ref_output.data(),
          static_cast<size_t>(num_keys) * static_cast<size_t>(row_size_));
      ref_tab_->find(ctx_, num_keys, std::move(keys_bw), std::move(hit_mask_bw), row_size_,
                         std::move(values_bw), nullptr /*value_sizes*/);
    }
    ref_tab_->get_lookup_counter(ctx_, &ref_hits);

    if (hit_tol >= 0.f) {
      // compare hitrates unless tolerance is negative
      ASSERT_NEAR(std::accumulate(hitrates.begin(), hitrates.end(), float(0)),
                  float(ref_hits) / float(num_keys), hit_tol);
      // compare outputs for hits
      for (int64_t i{}; i < num_keys; i++) {
        bool layer_hit{bitmask64::get(hitmask.at(to_uint(i / bitmask64::num_bits)), i % bitmask64::num_bits)};
        bool ref_hit{bitmask64::get(ref_hitmask.at(to_uint(i / bitmask64::num_bits)), i % bitmask64::num_bits)};
        if (layer_hit && ref_hit) {
          for (int64_t j{}; j < row_size_; j++) {
            ASSERT_EQ(output[i * row_size_ + j], ref_output.at(to_uint(i * row_size_ + j)));
          }
        }
      }
    }
    NVE_CHECK_(cudaFreeHost(output));
  }

  // Lookup with no caller-supplied hitmask.
  void LookupNoHitmask(std::vector<IndexT>& keys, void* out, float* hitrates,
                       uint64_t start_key = 0, uint64_t end_key = uint64_t(-1)) {
    if (keys.empty()) {
      return;
    }
    SetupKeys setup(keys, start_key, end_key);
    layer_->lookup(ctx_, setup.num_keys, setup.keys_buffer, out, row_size_,
                    nullptr /*hitmask*/, nullptr /*pool_params*/, hitrates);
  }

  // Lookup with pooling on the layer and compare against the mock reference (find to gather raw
  // stored rows, then combine). All keys must already be resolvable in the tiers so the gathered
  // bags match the reference. value_dtype is the tier storage dtype; out_dtype is the expected
  // output type. Same-type quantized Concatenate preserves the complete raw row.
  void LookupAndCheckPooling(std::vector<IndexT>& keys, DataType_t value_dtype, DataType_t out_dtype,
                             PoolingType_t pooling_type, SparseType_t sparse_type, int64_t hotness,
                             bool weighted) {
    const int64_t num_keys = static_cast<int64_t>(keys.size());
    if (num_keys == 0) return;

    // value elements per row (a quant row carries trailing scale[+offset] metadata).
    const bool quant = is_quant_rowwise(value_dtype);
    const bool raw_concat =
        pooling_type == PoolingType_t::Concatenate && out_dtype == value_dtype;
    const int64_t value_count =
        quant ? (row_size_ - quant_rowwise_meta_bytes(value_dtype)) : (row_size_ / dtype_size(value_dtype));
    const int64_t out_stride = raw_concat ? row_size_ : value_count * dtype_size(out_dtype);

    EmbeddingLayerBase::PoolingParams pp;
    pp.pooling_type = pooling_type;
    pp.sparse_type = sparse_type;
    pp.output_type = out_dtype;

    SetupCSROffsets<IndexT> offsets_setup(
        sparse_type == SparseType_t::CSR ? num_keys : 1, std::max<int64_t>(hotness, 1));
    int64_t output_bags = num_keys;
    if (sparse_type == SparseType_t::CSR) {
      pp.csr_offsets = offsets_setup.offsets_buffer;
      pp.num_csr_offsets = static_cast<int64_t>(offsets_setup.num_offsets);
      output_bags = static_cast<int64_t>(offsets_setup.num_offsets) - 1;
    } else {
      pp.fixed_hotness = hotness;
      if (pooling_type != PoolingType_t::Concatenate) output_bags = num_keys / hotness;
    }

    std::vector<int8_t> weights;
    if (weighted) {
      GenerateWeights(weights, static_cast<uint64_t>(num_keys), DataType_t::Float32);
      pp.weights = weights.data();
      pp.weight_type = DataType_t::Float32;
    }

    int8_t* output = nullptr;
    NVE_CHECK_(cudaMallocHost(&output, static_cast<size_t>(num_keys * out_stride)));
    std::memset(output, 0, static_cast<size_t>(num_keys * out_stride));

    std::vector<float> hitrates(static_cast<size_t>(layer_->get_num_tables()));
    layer_->lookup(ctx_, num_keys, keys.data(), output, out_stride, nullptr /*hitmask*/, &pp,
                    hitrates.data());

    // Reference: gather raw rows from the mock, then combine into the dequantized/pooled output.
    std::vector<int8_t> find_output(static_cast<size_t>(num_keys * row_size_), 0);
    {
      auto keys_bw = std::make_shared<BufferWrapper<const void>>(
          ctx_, "keys", keys.data(), static_cast<size_t>(num_keys) * sizeof(IndexT));
      auto values_bw = std::make_shared<BufferWrapper<void>>(
          ctx_, "values", find_output.data(), static_cast<size_t>(num_keys * row_size_));
      ref_tab_->find(ctx_, num_keys, std::move(keys_bw), nullptr /*hitmask*/, row_size_,
                      std::move(values_bw), nullptr /*value_sizes*/);
    }
    std::vector<int8_t> ref_output;
    if (raw_concat) {
      ref_output = std::move(find_output);
    } else {
      ref_output.resize(static_cast<size_t>(output_bags * out_stride), 0);
      auto* mock_ref = static_cast<MockHostTable<IndexT>*>(ref_tab_.get());
      mock_ref->combine(find_output.data(), num_keys, pp.pooling_type, pp.sparse_type,
                        pp.csr_offsets, pp.num_csr_offsets, pp.fixed_hotness, pp.weights,
                        pp.weight_type, ref_output.data(), out_dtype);
    }
    NVE_CHECK_(cudaDeviceSynchronize());

    if (raw_concat) {
      for (size_t i = 0; i < ref_output.size(); ++i) {
        ASSERT_EQ(output[i], ref_output[i]) << "raw byte " << i;
      }
    } else {
      const bool mean =
          pooling_type == PoolingType_t::Mean || pooling_type == PoolingType_t::WeightedMean;
      float tol;
      if (out_dtype == DataType_t::Float16) {
        tol = mean ? 5e-2f : 5e-2f * static_cast<float>(std::max<int64_t>(hotness, 1));
      } else {
        tol = quant ? 1e-3f : 1e-4f;
      }
      const int64_t output_elements = output_bags * value_count;
      for (int64_t i = 0; i < output_elements; i++) {
        ASSERT_NEAR(load_as_float(output, i, out_dtype),
                    load_as_float(ref_output.data(), i, out_dtype), tol)
            << "elem " << i << " value_dtype " << static_cast<int>(value_dtype) << " pooling "
            << static_cast<int>(pooling_type);
      }
    }
    NVE_CHECK_(cudaFreeHost(output));
  }

  // Minimal pooling lookup with no result checking; used to assert host-only configs reject pooling.
  void LookupPoolingNoCheck(std::vector<IndexT>& keys, PoolingType_t pooling) {
    const int64_t num_keys = static_cast<int64_t>(keys.size());
    EmbeddingLayerBase::PoolingParams pp;
    pp.pooling_type = pooling;
    pp.sparse_type = SparseType_t::Fixed;
    IndexT hotness = 1;
    pp.fixed_hotness = hotness;
    pp.output_type = DataType_t::Float32;
    std::vector<int8_t> out(static_cast<size_t>(num_keys * row_size_), 0);
    layer_->lookup(ctx_, num_keys, keys.data(), out.data(), row_size_, nullptr /*hitmask*/, &pp,
                    nullptr /*hitrates*/);
  }
  void Insert(std::vector<IndexT>& keys, std::vector<uint8_t>& datavectors, TableType db_type,
              uint64_t start_key = 0, uint64_t end_key = uint64_t(-1)) {
    if (keys.empty()) {
      return;
    }
    SetupKeys setup(keys, start_key, end_key, datavectors, row_size_);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;
    auto data_buffer = setup.data_buffer;

    if (db_type != TableType::Ref) {
      const int64_t table_id = table_id_from_type(db_type);
      if (table_id >= 0) {
        layer_->insert(ctx_, num_keys, keys_buffer, row_size_, row_size_, data_buffer, table_id);
      }
    }

    bool ref_insert = (db_type == TableType::Device && gpu_tab_) ||
                      (db_type == TableType::Host && host_tab_) ||
                      (db_type == TableType::Remote && remote_tab_) ||
                      (db_type == TableType::Ref);

    if (ref_insert) {
      auto keys_bw = std::make_shared<BufferWrapper<const void>>(
          ctx_, "keys", keys_buffer, static_cast<size_t>(num_keys) * sizeof(IndexT));
      auto values_bw = std::make_shared<BufferWrapper<const void>>(
          ctx_, "values", data_buffer,
          static_cast<size_t>(num_keys) * static_cast<size_t>(row_size_));
      ref_tab_->insert(ctx_, num_keys, std::move(keys_bw), row_size_, row_size_,
                           std::move(values_bw));
    }
  }
  void Update(std::vector<IndexT>& keys, std::vector<uint8_t>& datavectors, uint64_t start_key = 0,
              uint64_t end_key = uint64_t(-1)) {
    if (keys.empty()) {
      return;
    }
    SetupKeys setup(keys, start_key, end_key, datavectors, row_size_);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;
    auto data_buffer = setup.data_buffer;

    layer_->update(ctx_, num_keys, keys_buffer, row_size_, row_size_, data_buffer, -1);
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(
        ctx_, "keys", keys_buffer, static_cast<size_t>(num_keys) * sizeof(IndexT));
    auto values_bw = std::make_shared<BufferWrapper<const void>>(
        ctx_, "values", data_buffer,
        static_cast<size_t>(num_keys) * static_cast<size_t>(row_size_));
    ref_tab_->update(ctx_, num_keys, std::move(keys_bw), row_size_, row_size_,
                         std::move(values_bw));
  }
  void Accumulate(std::vector<IndexT>& keys, std::vector<uint8_t>& datavectors,
                  DataType_t value_type, uint64_t start_key = 0, uint64_t end_key = uint64_t(-1)) {
    if (keys.empty()) {
      return;
    }
    SetupKeys setup(keys, start_key, end_key, datavectors, row_size_);
    auto num_keys = setup.num_keys;
    auto keys_buffer = setup.keys_buffer;
    auto data_buffer = setup.data_buffer;

    layer_->accumulate(ctx_, num_keys, keys_buffer, row_size_, row_size_, data_buffer,
                        value_type, -1);
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(
        ctx_, "keys", keys_buffer, static_cast<size_t>(num_keys) * sizeof(IndexT));
    auto updates_bw = std::make_shared<BufferWrapper<const void>>(
        ctx_, "updates", data_buffer,
        static_cast<size_t>(num_keys) * static_cast<size_t>(row_size_));
    ref_tab_->update_accumulate(ctx_, num_keys, std::move(keys_bw), row_size_, row_size_,
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
    layer_->erase(ctx_, num_keys, keys_buffer, -1);

    auto keys_bw = std::make_shared<BufferWrapper<const void>>(
        ctx_, "keys", keys_buffer, static_cast<size_t>(num_keys) * sizeof(IndexT));
    ref_tab_->erase(ctx_, num_keys, std::move(keys_bw));
  }
  void Clear() {
    layer_->clear(ctx_);
    ref_tab_->clear(ctx_);
  }

  void Clear(TableType db_type) {
    switch (db_type) {
      case TableType::Device:
        if (gpu_tab_) gpu_tab_->clear(ctx_);
        break;
      case TableType::Host:
        if (host_tab_) host_tab_->clear(ctx_);
        break;
      case TableType::Remote:
        if (remote_tab_) remote_tab_->clear(ctx_);
        break;
      case TableType::Ref:
        if (ref_tab_) ref_tab_->clear(ctx_);
        break;
      default:
        throw std::runtime_error("Invalid database type in test!");
    }
  }

  void Wait() { ctx_->wait(); }

  // Raw layer erase, bypassing the reference table, for argument validation checks.
  void EraseRaw(int64_t num_keys, const void* keys, int64_t table_id) {
    layer_->erase(ctx_, num_keys, keys, table_id);
  }

  int64_t NumTables() const { return layer_->get_num_tables(); }

 private:
  const int64_t row_size_;
  table_ptr_t gpu_tab_;
  table_ptr_t host_tab_;
  table_ptr_t remote_tab_;
  host_table_ptr_t ref_tab_;
  std::shared_ptr<layer_type> layer_{nullptr};
  context_ptr_t ctx_;

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
      case TableType::Device: return gpu_tab_ ? 0 : -1;
      case TableType::Host: return host_tab_ ? (gpu_tab_ ? 1 : 0) : -1;
      case TableType::Remote: return remote_tab_ ? (gpu_tab_ ? 1 : 0) + (host_tab_ ? 1 : 0) : -1;
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
  const auto& tc = GetParam();
  SKIP_IF_HOST_TABLE_UNAVAILABLE(tc);
  HierLayerTest<int64_t> hlt(tc);
}

// [Sanity] insert 1key, lookup 1key
TEST_P(Hierachical, SingleLookup) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
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
  const auto& tc = GetParam();
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
  const auto& tc = GetParam();
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
  const auto& tc = GetParam();
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
  const auto& tc = GetParam();
  SKIP_IF_HOST_TABLE_UNAVAILABLE(tc);
  SKIP_IF_ACCUMULATE_UNSUPPORTED(tc);
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
  const auto& tc = GetParam();
  SKIP_IF_HOST_TABLE_UNAVAILABLE(tc);
  SKIP_IF_ERASE_UNSUPPORTED(tc);
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

  // An out-of-range table_id is rejected even for an empty erase, since table_id is validated
  // before the empty-input shortcut, as in the host and UVM layers.
  const int64_t invalid_table_id = hlt.NumTables();  // valid: negative (all) or < NumTables()
  EXPECT_THROW(hlt.EraseRaw(0, nullptr, invalid_table_id), Exception);

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
        //  TestCase: tables, row_size, table_size, test_keys, insert_heuristic, data_type
        HierTestCase({{TestTableType::Gpu}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float32}),
        HierTestCase({{TestTableType::NVHashMap}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Sph}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float32}),
        HierTestCase({{TestTableType::RocksDB}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::NVHashMap, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float32})
        ));

INSTANTIATE_TEST_SUITE_P(
    EmbLayer,
    Hierachical,
    ::testing::Values(
        //  TestCase: tables, row_size, table_size, test_keys, insert_heuristic, data_type
        HierTestCase({{TestTableType::Gpu}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({{TestTableType::NVHashMap}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({{TestTableType::Abseil}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({{TestTableType::Phmap}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({{TestTableType::Sph}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({{TestTableType::RocksDB}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({{TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({{TestTableType::Gpu, TestTableType::NVHashMap, TestTableType::RocksDB}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({{TestTableType::Gpu, TestTableType::NVHashMap, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Abseil, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Phmap, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Sph}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Sph, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({{TestTableType::Gpu, TestTableType::RocksDB, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({{TestTableType::Gpu}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::NVHashMap}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Abseil}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Phmap}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Sph}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::RocksDB}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::NVHashMap, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::NVHashMap, TestTableType::RocksDB}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Abseil, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Phmap, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Sph}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Sph, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::RocksDB, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32})));

INSTANTIATE_TEST_SUITE_P(
    EmbLayer_WithRedis,
    Hierachical,
    ::testing::Values(
        //  TestCase: tables, row_size, table_size, test_keys, insert_heuristic, data_Type
        HierTestCase({{TestTableType::Redis}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float16}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Redis, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float16}),
        HierTestCase({{TestTableType::Redis}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Redis, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({{TestTableType::Redis}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Redis, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Redis}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Redis, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),

        // Same coverage as above but against a standalone Redis in string mode (num_partitions == 0).
        HierTestCase({{TestTableType::Redis_String}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float16}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Redis_String, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float16}),
        HierTestCase({{TestTableType::Redis_String}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Redis_String, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float16}),
        HierTestCase({{TestTableType::Redis_String}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Redis_String, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1), false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Redis_String}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Redis_String, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32})));

INSTANTIATE_TEST_SUITE_P(
    EmbLayer_Large,
    Hierachical,
    ::testing::Values(
        //  TestCase: tables, row_size, table_size, test_keys, insert_heuristic, data_type
        HierTestCase({{TestTableType::Gpu}, int64_t(1) << 12, int64_t(1) << 30, int64_t(1) << 16, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::NVHashMap}, int64_t(1) << 12, int64_t(1) << 30, int64_t(1) << 16, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Abseil}, int64_t(1) << 12, int64_t(1) << 30, int64_t(1) << 16, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Phmap}, int64_t(1) << 12, int64_t(1) << 30, int64_t(1) << 16, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Sph}, int64_t(1) << 12, int64_t(1) << 30, int64_t(1) << 16, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::RocksDB}, int64_t(1) << 12, int64_t(1) << 30, int64_t(1) << 16, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Mock}, int64_t(1) << 12, int64_t(1) << 30, int64_t(1) << 16, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::NVHashMap, TestTableType::Mock}, int64_t(1) << 12, int64_t(1) << 30, int64_t(1) << 16, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Abseil, TestTableType::Mock}, int64_t(1) << 12, int64_t(1) << 30, int64_t(1) << 16, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Phmap, TestTableType::Mock}, int64_t(1) << 12, int64_t(1) << 30, int64_t(1) << 16, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Sph}, int64_t(1) << 12, int64_t(1) << 30, int64_t(1) << 16, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Sph, TestTableType::Mock}, int64_t(1) << 12, int64_t(1) << 30, int64_t(1) << 16, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::RocksDB, TestTableType::Mock}, int64_t(1) << 12, int64_t(1) << 30, int64_t(1) << 16, false, DataType_t::Float32})));

// Test context/buffer reuse so that no stale data is consumed from temporary buffers
class HierachicalContextReuse : public ::testing::TestWithParam<HierTestCase> {};
TEST_P(HierachicalContextReuse, BackToBackLookup) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
  SKIP_IF_HOST_TABLE_UNAVAILABLE(tc);
  HierLayerTest<int64_t> hlt(tc);

  std::vector<int64_t> keys;
  std::vector<uint8_t> data;
  GenerateData<int64_t>(keys, data, tc.test_keys, tc.row_size, 0, 1ul << 30, tc.data_type);
  hlt.Insert(keys, data, TableType::Device);
  hlt.Insert(keys, data, TableType::Host);
  hlt.Insert(keys, data, TableType::Remote);
  NVE_CHECK_(cudaDeviceSynchronize());

  // Deliberately not a multiple of 64, so the second lookup ends mid-word in the hitmask -- where
  // the trailing-bit handling of the word walk is easiest to get wrong.
  constexpr size_t second_keys{1000};
  ASSERT_LT(second_keys, tc.test_keys) << "the second lookup must be smaller, otherwise the "
                                          "scratch buffer is reallocated instead of reused";
  constexpr uint8_t kUnwritten{0xa5};
  const size_t row_bytes{to_uint(tc.row_size)};

  // Prime the context: every key resolves, so the scratch hitmask comes back fully set.
  std::vector<float> hitrates(3, 0.f);
  std::vector<uint8_t> out(keys.size() * row_bytes, kUnwritten);
  hlt.LookupNoHitmask(keys, out.data(), hitrates.data());
  NVE_CHECK_(cudaDeviceSynchronize());
  ASSERT_NEAR(std::accumulate(hitrates.begin(), hitrates.end(), 0.f), 1.f, 0.01f)
      << "the priming lookup did not resolve every key";

  // Same context, fewer keys: every scratch buffer still holds the priming call's bytes.
  std::fill(out.begin(), out.end(), kUnwritten);
  std::fill(hitrates.begin(), hitrates.end(), 0.f);
  hlt.LookupNoHitmask(keys, out.data(), hitrates.data(), 0, second_keys);
  NVE_CHECK_(cudaDeviceSynchronize());

  ASSERT_NEAR(std::accumulate(hitrates.begin(), hitrates.end(), 0.f), 1.f, 0.01f)
      << "keys were skipped on the second lookup -- stale context scratch";
  for (size_t i{}; i < second_keys; i++) {
    const size_t off{i * row_bytes};
    ASSERT_EQ(std::memcmp(&out[off], &data[off], row_bytes), 0)
        << "row " << i << " (key " << keys.at(i) << ") was not written by the second lookup";
  }
}

INSTANTIATE_TEST_SUITE_P(
    EmbLayer,
    HierachicalContextReuse,
    ::testing::Values(
        //  TestCase: tables, row_size, table_size, test_keys, insert_heuristic, data_type
        HierTestCase({{TestTableType::Gpu}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::NVHashMap}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Abseil}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Phmap}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::RocksDB}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::NVHashMap, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, false, DataType_t::Float32})));

class HierachicalSpecialConfig : public ::testing::TestWithParam<HierTestCase> {};

TEST_P(HierachicalSpecialConfig, InflightInsert) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const auto& tc = GetParam();
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
  const auto& tc = GetParam();
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
  const auto& tc = GetParam();
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
  const auto& tc = GetParam();
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
  layer_cfg.default_embedding = std::move(default_emb);
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
  std::vector<bitmask64_t> hitmask(ceil_div(num_keys, to_uint(bitmask64::num_bits)), 0);
  layer->lookup(ctx, static_cast<int64_t>(num_keys), keys.data(), output.data(),
                static_cast<int64_t>(row_size), hitmask.data(),
                nullptr /*pool_params*/, nullptr /*hitrates*/);
  ctx->wait();

  for (uint64_t i = 0; i < num_keys; i++) {
    const auto mask{hitmask[i / to_uint(bitmask64::num_bits)]};
    const bool hit{bitmask64::get(mask, static_cast<int64_t>(i) % bitmask64::num_bits)};
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
  std::vector<bitmask64_t> hitmask(ceil_div(num_keys, to_uint(bitmask64::num_bits)), 0);
  layer->lookup(ctx, static_cast<int64_t>(num_keys), keys.data(), output.data(),
                static_cast<int64_t>(row_size), hitmask.data(),
                nullptr /*pool_params*/, nullptr /*hitrates*/);
  ctx->wait();

  for (uint64_t i = 0; i < num_keys; i++) {
    const auto mask{hitmask[i / to_uint(bitmask64::num_bits)]};
    const bool hit{bitmask64::get(mask, static_cast<int64_t>(i) % bitmask64::num_bits)};
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
  table_cfg.num_rows = static_cast<int64_t>(num_rows);
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
  std::vector<bitmask64_t> hitmask(ceil_div(num_keys, to_uint(bitmask64::num_bits)), 0);

  layer->lookup(ctx, static_cast<int64_t>(num_keys), keys.data(), output.data(),
                static_cast<int64_t>(row_size), hitmask.data(),
                nullptr /*pool_params*/, nullptr /*hitrates*/);
  ctx->wait();

  for (uint64_t i = 0; i < num_keys; ++i) {
    const auto mask{hitmask[i / to_uint(bitmask64::num_bits)]};
    const bool hit{bitmask64::get(mask, static_cast<int64_t>(i) % bitmask64::num_bits)};
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
  table_cfg.num_rows = static_cast<int64_t>(num_rows);
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
  std::vector<bitmask64_t> hitmask(ceil_div(num_keys, to_uint(bitmask64::num_bits)), 0);
  layer->lookup(ctx, static_cast<int64_t>(num_keys), keys.data(), output.data(),
                static_cast<int64_t>(row_size), hitmask.data(),
                nullptr /*pool_params*/, nullptr /*hitrates*/);
  ctx->wait();

  for (uint64_t i = 0; i < num_keys; ++i) {
    const auto mask{hitmask[i / to_uint(bitmask64::num_bits)]};
    const bool hit{bitmask64::get(mask, static_cast<int64_t>(i) % bitmask64::num_bits)};
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
        //  TestCase: tables, row_size, table_size, test_keys, insert_heuristic, data_Type
        HierTestCase({{TestTableType::Gpu, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, true, DataType_t::Float32}),
        HierTestCase({{TestTableType::NVHashMap, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, true, DataType_t::Float32}),
        HierTestCase({{TestTableType::Abseil, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, true, DataType_t::Float32}),
        HierTestCase({{TestTableType::Phmap, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, true, DataType_t::Float32}),
        HierTestCase({{TestTableType::RocksDB, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, true, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::NVHashMap, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, true, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Abseil, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, true, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::Phmap, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, true, DataType_t::Float32}),
        HierTestCase({{TestTableType::Gpu, TestTableType::RocksDB, TestTableType::Mock}, int64_t(1) << 10, int64_t(1) << 30, int64_t(1) << 11, true, DataType_t::Float32})));

// ---------------------------------------------------------------------------
// Pooling / rowwise-quant dequant through HierarchicalEmbeddingLayer.
//
// For reductions or dequantization, the layer gathers raw stored rows across tiers into a device
// buffer, then runs the dequant/combine kernel over it (ad-hoc ECNoCache functor,
// load_indices=false). Same-type Concatenate instead gathers directly into the final output. All
// keys are inserted up front so every bag fully resolves; the result is checked against the mock
// reference (find to gather, then combine). insert_tier == Host with a gpu+host config exercises the
// host->device scatter feeding the kernel.
// ---------------------------------------------------------------------------
template <typename IndexT>
static void RunHierPooling(bool gpu, TestTableType host, TableType insert_tier, int64_t value_count,
                           DataType_t value_dtype, DataType_t out_dtype, PoolingType_t pooling,
                           SparseType_t sparse, int64_t hotness, bool weighted) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  const bool quant = is_quant_rowwise(value_dtype);
  const int64_t row_size =
      quant ? (value_count + quant_rowwise_meta_bytes(value_dtype)) : (value_count * dtype_size(value_dtype));

  std::vector<TestTableType> tables;
  if (gpu) tables.push_back(TestTableType::Gpu);
  if (host != TestTableType::None) tables.push_back(host);
  HierLayerTest<IndexT> hlt(tables, row_size, int64_t(1) << 30,
                            false /*insert_heuristic*/, value_dtype);

  const int64_t num_keys = (pooling == PoolingType_t::Concatenate) ? 64 : (hotness * 32);
  std::vector<IndexT> keys(static_cast<size_t>(num_keys));
  std::vector<uint8_t> data;
  if (quant) {
    for (int64_t i = 0; i < num_keys; i++) keys[static_cast<size_t>(i)] = static_cast<IndexT>(i + 1);
    data.resize(static_cast<size_t>(num_keys * row_size));
    InitTableRowsQuant(reinterpret_cast<int8_t*>(data.data()), row_size, value_count, 0,
                       static_cast<uint64_t>(num_keys), value_dtype, 4242);
  } else {
    // GenerateData fills random unique keys and matching float/half row data.
    GenerateData<IndexT>(keys, data, static_cast<size_t>(num_keys), row_size, 1, int64_t(1) << 20,
                         value_dtype, true /*unique*/, 4242);
  }

  hlt.Insert(keys, data, insert_tier);
  NVE_CHECK_(cudaDeviceSynchronize());
  hlt.LookupAndCheckPooling(keys, value_dtype, out_dtype, pooling, sparse, hotness, weighted);
}

// Float32 / Float16, all pooling types, both sparse layouts. GPU-only (gather straight on device).
TEST(HierachicalPooling, F32_Sum_Fixed)        { RunHierPooling<int64_t>(true, TestTableType::None, TableType::Device, 64, DataType_t::Float32, DataType_t::Float32, PoolingType_t::Sum,          SparseType_t::Fixed, 8, false); }
TEST(HierachicalPooling, F32_Mean_CSR)         { RunHierPooling<int64_t>(true, TestTableType::None, TableType::Device, 64, DataType_t::Float32, DataType_t::Float32, PoolingType_t::Mean,         SparseType_t::CSR,   8, false); }
TEST(HierachicalPooling, F32_WeightedSum_Fixed){ RunHierPooling<int64_t>(true, TestTableType::None, TableType::Device, 64, DataType_t::Float32, DataType_t::Float32, PoolingType_t::WeightedSum,  SparseType_t::Fixed, 8, true);  }
TEST(HierachicalPooling, F32_Concatenate)      { RunHierPooling<int64_t>(true, TestTableType::None, TableType::Device, 64, DataType_t::Float32, DataType_t::Float32, PoolingType_t::Concatenate,  SparseType_t::Fixed, 1, false); }
TEST(HierachicalPooling, F16_Sum_Fixed)        { RunHierPooling<int64_t>(true, TestTableType::None, TableType::Device, 64, DataType_t::Float16, DataType_t::Float16, PoolingType_t::Sum,          SparseType_t::Fixed, 8, false); }
TEST(HierachicalPooling, F16_WeightedMean_CSR) { RunHierPooling<int64_t>(true, TestTableType::None, TableType::Device, 64, DataType_t::Float16, DataType_t::Float16, PoolingType_t::WeightedMean, SparseType_t::CSR,   8, true);  }

// Rowwise-quant FP16 and FP32 storage -> dequantized FP32/FP16 output.
TEST(HierachicalPooling, QInt8F16_Sum_Fixed)   { RunHierPooling<int64_t>(true, TestTableType::None, TableType::Device, 62, DataType_t::QInt8RowwiseF16,  DataType_t::Float16, PoolingType_t::Sum,         SparseType_t::Fixed, 8, false); }
TEST(HierachicalPooling, QUint8F16_Mean_CSR)   { RunHierPooling<int64_t>(true, TestTableType::None, TableType::Device, 64, DataType_t::QUint8RowwiseF16, DataType_t::Float16, PoolingType_t::Mean,        SparseType_t::CSR,   8, false); }
TEST(HierachicalPooling, QInt8F16_Concatenate) { RunHierPooling<int64_t>(true, TestTableType::None, TableType::Device, 62, DataType_t::QInt8RowwiseF16,  DataType_t::Float16, PoolingType_t::Concatenate, SparseType_t::Fixed, 1, false); }
TEST(HierachicalPooling, QUint8F16_Concatenate){ RunHierPooling<int64_t>(true, TestTableType::None, TableType::Device, 64, DataType_t::QUint8RowwiseF16, DataType_t::Float16, PoolingType_t::Concatenate, SparseType_t::Fixed, 1, false); }
TEST(HierachicalPooling, QInt8F32_WeightedSum) { RunHierPooling<int64_t>(true, TestTableType::None, TableType::Device, 64, DataType_t::QInt8RowwiseF32,  DataType_t::Float32, PoolingType_t::WeightedSum, SparseType_t::Fixed, 8, true);  }
TEST(HierachicalPooling, QUint8F32_Sum_CSR)    { RunHierPooling<int64_t>(true, TestTableType::None, TableType::Device, 64, DataType_t::QUint8RowwiseF32, DataType_t::Float32, PoolingType_t::Sum,         SparseType_t::CSR,   8, false); }
TEST(HierachicalPooling, QUint8F32_Concatenate){ RunHierPooling<int64_t>(true, TestTableType::None, TableType::Device, 64, DataType_t::QUint8RowwiseF32, DataType_t::Float32, PoolingType_t::Concatenate, SparseType_t::Fixed, 1, false); }

// gpu+host: keys resolve in the host tier, so the gathered rows are scattered down to device before
// the kernel runs (exercises the cross-tier gather path).
TEST(HierachicalPooling, F32_Sum_GpuHost_HostHit)   { RunHierPooling<int64_t>(true, TestTableType::NVHashMap, TableType::Host, 64, DataType_t::Float32, DataType_t::Float32, PoolingType_t::Sum, SparseType_t::Fixed, 8, false); }
TEST(HierachicalPooling, QUint8F16_GpuHost_HostHit) { RunHierPooling<int64_t>(true, TestTableType::NVHashMap, TableType::Host, 64, DataType_t::QUint8RowwiseF16, DataType_t::Float16, PoolingType_t::Sum, SparseType_t::Fixed, 8, false); }

// A pooling lookup still auto-inserts the gathered raw rows into faster tiers (gather_bw holds one
// raw stored row per key). Populate only the host tier, pool-lookup (which should auto-insert into
// the GPU tier), then drop the host tier: the keys must now resolve from the GPU tier alone.
TEST(HierachicalPooling, AutoInsertFromGather) {
  cudaGetLastError();
  const int64_t value_count = 64;
  const int64_t row_size = value_count * dtype_size(DataType_t::Float32);
  HierLayerTest<int64_t> hlt({TestTableType::Gpu, TestTableType::NVHashMap}, row_size,
                             int64_t(1) << 30, true /*insert_heuristic*/, DataType_t::Float32);
  const int64_t hotness = 8;
  const int64_t num_keys = 4096;  // > min_insert_size_gpu (1<<10) so GPU auto-insert triggers
  std::vector<int64_t> keys;
  std::vector<uint8_t> data;
  GenerateData<int64_t>(keys, data, static_cast<size_t>(num_keys), row_size, 1, int64_t(1) << 20,
                        DataType_t::Float32, true /*unique*/, 4242);

  // Populate only the host tier (+ ref); the GPU tier starts empty.
  hlt.Insert(keys, data, TableType::Host);
  NVE_CHECK_(cudaDeviceSynchronize());
  // Pooling lookup: resolves from host, gathers raw rows, and auto-inserts them into the GPU tier.
  hlt.LookupAndCheckPooling(keys, DataType_t::Float32, DataType_t::Float32, PoolingType_t::Sum,
                            SparseType_t::Fixed, hotness, false);
  hlt.Wait();  // wait for the async auto-insert to finish

  // Drop the host tier; the keys must now be served from the GPU tier that auto-insert populated.
  hlt.Clear(TableType::Host);
  NVE_CHECK_(cudaDeviceSynchronize());
  hlt.LookupAndCheckPooling(keys, DataType_t::Float32, DataType_t::Float32, PoolingType_t::Sum,
                            SparseType_t::Fixed, hotness, false);
}

// Host-only configs (no GPU tier) pool/dequant on the CPU path (pool_gathered_host), reusing the
// host layer's kernels. Verify correctness across pooling modes, sparse layouts, weighting, and
// quantized dequant -- mirroring the GPU-tier cases above but with gpu=false.
TEST(HierachicalPooling, HostOnly_F32_Sum_Fixed)       { RunHierPooling<int64_t>(false, TestTableType::NVHashMap, TableType::Host, 64, DataType_t::Float32, DataType_t::Float32, PoolingType_t::Sum,          SparseType_t::Fixed, 8, false); }
TEST(HierachicalPooling, HostOnly_F32_Mean_CSR)        { RunHierPooling<int64_t>(false, TestTableType::NVHashMap, TableType::Host, 64, DataType_t::Float32, DataType_t::Float32, PoolingType_t::Mean,         SparseType_t::CSR,   8, false); }
TEST(HierachicalPooling, HostOnly_F32_WeightedSum)     { RunHierPooling<int64_t>(false, TestTableType::NVHashMap, TableType::Host, 64, DataType_t::Float32, DataType_t::Float32, PoolingType_t::WeightedSum,  SparseType_t::Fixed, 8, true);  }
TEST(HierachicalPooling, HostOnly_F32_Concatenate)     { RunHierPooling<int64_t>(false, TestTableType::NVHashMap, TableType::Host, 64, DataType_t::Float32, DataType_t::Float32, PoolingType_t::Concatenate,  SparseType_t::Fixed, 1, false); }
TEST(HierachicalPooling, HostOnly_QUint8F16_Mean_CSR)  { RunHierPooling<int64_t>(false, TestTableType::NVHashMap, TableType::Host, 64, DataType_t::QUint8RowwiseF16, DataType_t::Float16, PoolingType_t::Mean,        SparseType_t::CSR,   8, false); }
TEST(HierachicalPooling, HostOnly_QInt8F32_WeightedSum){ RunHierPooling<int64_t>(false, TestTableType::NVHashMap, TableType::Host, 64, DataType_t::QInt8RowwiseF32,  DataType_t::Float32, PoolingType_t::WeightedSum, SparseType_t::Fixed, 8, true);  }
TEST(HierachicalPooling, HostOnly_QInt8F32_RawConcatenate) { RunHierPooling<int64_t>(false, TestTableType::NVHashMap, TableType::Host, 64, DataType_t::QInt8RowwiseF32, DataType_t::QInt8RowwiseF32, PoolingType_t::Concatenate, SparseType_t::Fixed, 1, false); }
// Host-only dequant honors cross-precision output (a fp32-scale quant type -> fp16 output), which
// the GPU dequant kernels cannot do -- the host CPU path has no such restriction.
TEST(HierachicalPooling, HostOnly_QInt8F32_outF16_Sum) { RunHierPooling<int64_t>(false, TestTableType::NVHashMap, TableType::Host, 64, DataType_t::QInt8RowwiseF32,  DataType_t::Float16, PoolingType_t::Sum,        SparseType_t::Fixed, 8, false); }

// Host tier whose insert always fails; everything else forwards to an empty MockHostTable, so every
// lookup misses and a heuristic that always says "insert" launches a failing async auto-insert.
template <typename IndexT>
class FailingInsertTable final : public HostTable<HostTableConfig> {
 public:
  using base_type = HostTable<HostTableConfig>;
  explicit FailingInsertTable(const HostTableConfig& cfg)
      : base_type(0, cfg), inner_(cfg, true /*functional_ref*/) {}

  int64_t size(context_ptr_t& ctx, bool exact) const override { return inner_.size(ctx, exact); }
  void clear(context_ptr_t& ctx) override { inner_.clear(ctx); }
  void erase(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys) override {
    inner_.erase(ctx, n, std::move(keys));
  }
  void find(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, buffer_ptr<bitmask64_t> hit_mask,
            int64_t value_stride, buffer_ptr<void> values, buffer_ptr<int64_t> value_sizes) const override {
    inner_.find(ctx, n, std::move(keys), std::move(hit_mask), value_stride, std::move(values),
                std::move(value_sizes));
  }
  void insert(context_ptr_t&, int64_t, buffer_ptr<const void>, int64_t, int64_t,
              buffer_ptr<const void>) override {
    throw std::runtime_error("FailingInsertTable: insert fails on purpose");
  }
  void update(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, int64_t update_stride,
              int64_t update_size, buffer_ptr<const void> updates) override {
    inner_.update(ctx, n, std::move(keys), update_stride, update_size, std::move(updates));
  }
  void update_accumulate(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, int64_t update_stride,
                         int64_t update_size, buffer_ptr<const void> updates,
                         DataType_t update_dtype) override {
    inner_.update_accumulate(ctx, n, std::move(keys), update_stride, update_size, std::move(updates),
                             update_dtype);
  }
  void reset_lookup_counter(context_ptr_t& ctx) override { inner_.reset_lookup_counter(ctx); }
  void get_lookup_counter(context_ptr_t& ctx, int64_t* counter) const override {
    inner_.get_lookup_counter(ctx, counter);
  }

 private:
  MockHostTable<IndexT> inner_;
};

// A failed async auto-insert is reported by an explicit wait(), but must neither surface as an error
// of an unrelated later lookup nor escape the context destructor (which would std::terminate).
TEST(HierachicalAutoInsert, FailedInsertDoesNotEscapeLookupOrTeardown) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  constexpr int64_t row_size = 64;
  constexpr int64_t num_keys = 16;
  HostTableConfig tcfg;
  tcfg.value_dtype = DataType_t::Float32;
  tcfg.max_value_size = row_size;
  auto table = std::make_shared<FailingInsertTable<int64_t>>(tcfg);

  HierarchicalEmbeddingLayer<int64_t>::Config cfg;
  cfg.insert_heuristic = std::make_shared<TestInsertHeuristic>(std::vector<bool>{true});
  cfg.min_insert_freq_host = 0;
  cfg.min_insert_size_host = 1;  // every lookup launches an insert right away
  auto layer = std::make_shared<HierarchicalEmbeddingLayer<int64_t>>(cfg, std::vector<table_ptr_t>{table});
  auto ctx = layer->create_execution_context(0, 0, nullptr, nullptr);

  std::vector<int64_t> keys(num_keys);
  std::iota(keys.begin(), keys.end(), 0);
  std::vector<uint8_t> out(static_cast<size_t>(num_keys * row_size));
  std::vector<uint8_t> row(static_cast<size_t>(row_size), 0);
  const auto lookup = [&]() {
    layer->lookup(ctx, num_keys, keys.data(), out.data(), row_size, nullptr /*hitmask*/,
                  nullptr /*pool_params*/, nullptr /*hitrates*/);
  };
  // update() takes the auto-insert modify lock, so it returns only once the in-flight insert task
  // has finished: a deterministic barrier without touching the layer internals.
  const auto drain_insert = [&]() {
    layer->update(ctx, 1, keys.data(), row_size, row_size, row.data(), -1 /*all tables*/);
  };

  // The first lookup schedules an insert that fails asynchronously.
  ASSERT_NO_THROW(lookup());
  ASSERT_NO_THROW(drain_insert());
  // The next lookup replaces the failed task; the old failure must not be rethrown from it.
  EXPECT_NO_THROW(lookup());
  ASSERT_NO_THROW(drain_insert());
  // An explicit wait() is where the failure is reported.
  EXPECT_THROW(ctx->wait(), std::exception);
  // Leave one more failed insert behind and make sure teardown swallows it.
  ASSERT_NO_THROW(lookup());
  ASSERT_NO_THROW(drain_insert());
  EXPECT_NO_THROW(ctx.reset());
  layer.reset();
}

// Records the hit rate every tier reports to the insert heuristic; never asks for an insert.
class RecordingInsertHeuristic final : public InsertHeuristic {
 public:
  bool insert_needed(const float hitrate, const size_t table_id) override {
    hitrates.resize(std::max(hitrates.size(), table_id + 1), -1.f);
    hitrates.at(table_id) = hitrate;
    return false;
  }
  std::vector<float> hitrates;
};

// When the first tier resolves every key, the tiers behind it are asked to resolve nothing. Their
// hit rate must be reported as 1.0 (nothing missed), not 0/0 = NaN, which used to reach the insert
// heuristic.
TEST(HierachicalHitrate, TierWithNoCandidatesReportsOneNotNaN) {
  cudaGetLastError();  // Clear potential errors left by previous tests.
  constexpr int64_t row_size = 64;
  constexpr size_t num_keys = 100;
  HostTableConfig tcfg;
  tcfg.value_dtype = DataType_t::Float32;
  tcfg.max_value_size = row_size;
  auto first = std::make_shared<MockHostTable<int64_t>>(tcfg, true /*functional_ref*/);
  auto second = std::make_shared<MockHostTable<int64_t>>(tcfg, true /*functional_ref*/);
  auto heuristic = std::make_shared<RecordingInsertHeuristic>();

  HierarchicalEmbeddingLayer<int64_t>::Config cfg;
  cfg.insert_heuristic = heuristic;
  cfg.min_insert_freq_host = 0;  // consult the heuristic on every lookup
  auto layer = std::make_shared<HierarchicalEmbeddingLayer<int64_t>>(
      cfg, std::vector<table_ptr_t>{first, second});
  auto ctx = layer->create_execution_context(0, 0, nullptr, nullptr);

  std::vector<int64_t> keys;
  std::vector<uint8_t> data;
  GenerateData<int64_t>(keys, data, num_keys, row_size, 0, 1ul << 30, DataType_t::Float32);
  layer->insert(ctx, static_cast<int64_t>(num_keys), keys.data(), row_size, row_size, data.data(),
                0 /*first tier only*/);

  std::vector<uint8_t> out(num_keys * row_size);
  std::vector<float> hitrates(2, -1.f);
  layer->lookup(ctx, static_cast<int64_t>(num_keys), keys.data(), out.data(), row_size,
                nullptr /*hitmask*/, nullptr /*pool_params*/, hitrates.data());

  ASSERT_EQ(heuristic->hitrates.size(), 2u);
  EXPECT_FLOAT_EQ(heuristic->hitrates.at(0), 1.f);
  EXPECT_FALSE(std::isnan(heuristic->hitrates.at(1)));
  EXPECT_FLOAT_EQ(heuristic->hitrates.at(1), 1.f);
  // The caller-facing hit rates are relative to num_keys and unchanged: all hits in tier 0.
  EXPECT_FLOAT_EQ(hitrates.at(0), 1.f);
  EXPECT_FLOAT_EQ(hitrates.at(1), 0.f);
  ctx.reset();
}

}  // namespace nve
