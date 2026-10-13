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

#include <unistd.h>
#include "emb_layer_utils.hpp"
#include "test_utils.hpp"
#include <gpu_table.hpp>
#include <hierarchical_embedding_layer.hpp>
#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace nve {

// GPU cache in front of a LinearFileTable loaded from the plugin. The file tier is dense (every
// in-range key hits; insert/erase/clear are no-ops), so it can't share the Hierachical suite's
// MockHostTable reference.
class HierachicalLinearFile : public ::testing::Test {
  // Declared first so it is destroyed last, i.e. after the file table has closed the file.
  std::optional<nve_test::TempPathCleanup> file_cleanup_;

 protected:
  using IndexT = int64_t;
  static constexpr int64_t kRowElems{32};
  static constexpr int64_t kRowSize{kRowElems * static_cast<int64_t>(sizeof(float))};
  static constexpr int64_t kNumRows{4096};

  static float pattern(IndexT key, int64_t elem) { return static_cast<float>(key * kRowElems + elem); }

  struct LookupResult {
    std::vector<float> rows;
    std::vector<bitmask64_t> hitmask;
    std::array<float, 2> hitrates{};  // [gpu, file]

    bool hit(size_t i) const {
      return bitmask64::get(hitmask[i / to_uint(bitmask64::num_bits)],
                            static_cast<int64_t>(i) % bitmask64::num_bits);
    }
  };

  void SetUp() override {
    SKIP_IF_LINEAR_FILE_UNAVAILABLE();
    cudaGetLastError();  // Clear potential errors left by previous tests.

    const auto path{std::filesystem::temp_directory_path() /
                    ("nve_hier_layer_test_linear_file_" + std::to_string(getpid()))};
    file_cleanup_.emplace(path);
    std::vector<IndexT> all_keys(kNumRows);
    std::iota(all_keys.begin(), all_keys.end(), IndexT{0});
    const auto rows{MakeRows(all_keys, pattern)};
    std::ofstream file{path, std::ios::binary};
    file.write(reinterpret_cast<const char*>(rows.data()),
               static_cast<std::streamsize>(rows.size() * sizeof(float)));
    file.close();
    ASSERT_TRUE(file) << "could not write " << path;

    GPUTableConfig gpu_cfg;
    gpu_cfg.device_id = 0;
    gpu_cfg.cache_size = 1 << 24;  // holds every row, so cached keys are never evicted
    gpu_cfg.row_size_in_bytes = kRowSize;
    gpu_cfg.uvm_table = nullptr;
    gpu_cfg.value_dtype = DataType_t::Float32;
    auto gpu_tab{std::make_shared<GpuTable<IndexT>>(gpu_cfg)};

    const nlohmann::json file_conf = {{"key_size", sizeof(IndexT)},
                                      {"max_value_size", kRowSize},
                                      {"value_dtype", to_string(DataType_t::Float32)},
                                      {"path", path.string()},
                                      {"num_rows", kNumRows}};
    table_factory_ptr_t file_fac{nve_test::plugin_factory("linear-file")};
    auto file_tab{file_fac->produce(4718, file_conf)};

    typename HierarchicalEmbeddingLayer<IndexT>::Config layer_cfg;
    layer_cfg.layer_name = "gpu_linear_file";
    // Auto-insert every GPU miss on the first lookup; the file tier ignores inserts.
    layer_cfg.insert_heuristic = std::make_shared<TestInsertHeuristic>(std::vector<bool>{true, false});
    layer_cfg.min_insert_size_gpu = 1;
    layer_ = std::make_shared<HierarchicalEmbeddingLayer<IndexT>>(
        layer_cfg, std::vector<table_ptr_t>{gpu_tab, file_tab});
    ctx_ = layer_->create_execution_context(0, 0, nullptr, nullptr);
  }

  // n distinct keys spread over the file (7 is coprime with kNumRows).
  static std::vector<IndexT> SpreadKeys(size_t n) {
    std::vector<IndexT> keys(n);
    for (size_t i{}; i < n; ++i) keys[i] = static_cast<IndexT>(i * 7) % kNumRows;
    return keys;
  }

  // Row i holds value(keys[i], elem).
  template <typename Fn>
  static std::vector<float> MakeRows(const std::vector<IndexT>& keys, Fn value) {
    std::vector<float> rows(keys.size() * to_uint(kRowElems));
    for (size_t i{}; i < keys.size(); ++i) {
      for (int64_t e{}; e < kRowElems; ++e) rows[i * to_uint(kRowElems) + to_uint(e)] = value(keys[i], e);
    }
    return rows;
  }

  // Looks up through the layer and waits, including for the auto-insert it triggers.
  LookupResult Lookup(const std::vector<IndexT>& keys, float fill = 0.f) {
    LookupResult r;
    r.rows.assign(keys.size() * to_uint(kRowElems), fill);
    r.hitmask.assign(ceil_div(keys.size(), to_uint(bitmask64::num_bits)), 0);
    layer_->lookup(ctx_, static_cast<int64_t>(keys.size()), keys.data(), r.rows.data(), kRowSize,
                   r.hitmask.data(), nullptr /*pool_params*/, r.hitrates.data());
    ctx_->wait();
    return r;
  }

  // Every key hits and row i holds expected(keys[i], elem).
  template <typename Fn>
  static void ExpectRows(const std::vector<IndexT>& keys, const LookupResult& r, Fn expected) {
    for (size_t i{}; i < keys.size(); ++i) {
      ASSERT_TRUE(r.hit(i)) << "expected hit for key " << keys[i];
      for (int64_t e{}; e < kRowElems; ++e) {
        ASSERT_FLOAT_EQ(expected(keys[i], e), r.rows[i * to_uint(kRowElems) + to_uint(e)])
            << "key=" << keys[i] << " col=" << e;
      }
    }
  }

  std::shared_ptr<HierarchicalEmbeddingLayer<IndexT>> layer_;
  context_ptr_t ctx_;
};

TEST_F(HierachicalLinearFile, GpuMissReadsFileThenCaches) {
  const auto keys{SpreadKeys(512)};

  const auto first{Lookup(keys)};
  EXPECT_FLOAT_EQ(first.hitrates[0], 0.f);
  EXPECT_FLOAT_EQ(first.hitrates[1], 1.f);
  ExpectRows(keys, first, pattern);

  // The misses were auto-inserted, so the same keys now resolve on the GPU.
  const auto second{Lookup(keys)};
  EXPECT_GT(second.hitrates[0], 0.99f);
  ExpectRows(keys, second, pattern);
}

// The update reaches both tiers: GPU-cached rows must not come back stale.
TEST_F(HierachicalLinearFile, UpdateThenLookup) {
  const auto keys{SpreadKeys(512)};
  Lookup({keys.begin(), keys.begin() + 256});  // caches the first half on the GPU

  const auto updated = [](IndexT key, int64_t e) { return -1.f - pattern(key, e); };
  const auto values{MakeRows(keys, updated)};
  layer_->update(ctx_, static_cast<int64_t>(keys.size()), keys.data(), kRowSize, kRowSize,
                 values.data(), -1 /*table_id*/);
  ctx_->wait();

  const auto after{Lookup(keys)};
  EXPECT_NEAR(after.hitrates[0], 0.5f, 0.01f);  // half from the GPU, half from the file
  ExpectRows(keys, after, updated);
}

// Keys past the end of the file miss in every tier and leave their output untouched.
TEST_F(HierachicalLinearFile, OutOfRangeKeysMiss) {
  constexpr float sentinel{-7.f};
  const std::vector<IndexT> keys{5, kNumRows, 100, kNumRows + 123, kNumRows - 1};

  const auto r{Lookup(keys, sentinel)};
  for (size_t i{}; i < keys.size(); ++i) {
    const bool in_range{keys[i] < kNumRows};
    ASSERT_EQ(in_range, r.hit(i)) << "key " << keys[i];
    for (int64_t e{}; e < kRowElems; ++e) {
      ASSERT_FLOAT_EQ(in_range ? pattern(keys[i], e) : sentinel, r.rows[i * to_uint(kRowElems) + to_uint(e)])
          << "key=" << keys[i] << " col=" << e;
    }
  }
}

}  // namespace nve
