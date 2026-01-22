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

#include <algorithm>
#include <execution_context.hpp>
#include <host_table.hpp>
#include <random>
#include "test_utils.hpp"

using namespace nve;
using namespace nlohmann::literals;

std::vector<int64_t> make_uniform_keys(int64_t n) {
  std::vector<int64_t> keys(static_cast<size_t>(n));

  std::default_random_engine rng(random_device());
  std::uniform_int_distribution dist(0, 1'000'000);
  std::generate(keys.begin(), keys.end(), [&dist, &rng]() { return dist(rng); });

  return keys;
}

void replace(std::string& x, const std::string_view& a, const std::string_view& b) {
  x.replace(x.find(a), a.size(), b);
}

void find_insert_find(const nlohmann::json& fac_json) {
  try {
    using key_type = int64_t;

    host_table_factory_ptr_t fac{create_host_table_factory(fac_json)};
    std::string table_conf{R"({
      "max_value_size": %max_value_size
    })"};
    const int64_t max_value_size{20};

    replace(table_conf, "%max_value_size", std::to_string(max_value_size));
    host_table_ptr_t tab{fac->produce(4711, nlohmann::json::parse(table_conf))};

    auto ctx = tab->create_execution_context(0, 0, nullptr, nullptr);
    tab->clear(ctx);

    // Find -> nothing -> insert -> find.
    int64_t n{63999};
    std::vector<key_type> keys{make_uniform_keys(n)};
    std::vector<max_bitmask_repr_t> hit_mask(static_cast<uint64_t>(max_bitmask_t::mask_size(n)));
    const int64_t value_stride{max_value_size + 1};

    int64_t cnt;

    std::fill(hit_mask.begin(), hit_mask.end(), 0);
    tab->reset_lookup_counter(ctx);
    tab->find(ctx, n, keys.data(), hit_mask.data(), max_value_size, nullptr, nullptr);
    tab->get_lookup_counter(ctx, &cnt);
    ASSERT_EQ(cnt, 0);
    tab->insert(ctx, n, keys.data(), value_stride, 0, nullptr);
    tab->reset_lookup_counter(ctx);
    tab->find(ctx, n, keys.data(), hit_mask.data(), max_value_size, nullptr, nullptr);
    tab->get_lookup_counter(ctx, &cnt);
    ASSERT_EQ(cnt, n);

    std::fill(hit_mask.begin(), hit_mask.end(), max_bitmask_t::full());
    tab->reset_lookup_counter(ctx);
    tab->find(ctx, n, keys.data(), hit_mask.data(), max_value_size, nullptr, nullptr);
    tab->get_lookup_counter(ctx, &cnt);
    ASSERT_EQ(cnt, 0);

    std::set<key_type> keys_set(keys.begin(), keys.end());
    std::vector<key_type> keys2{make_uniform_keys(n)};
    int64_t act_cnt{std::count_if(keys2.begin(), keys2.end(), [&keys_set](const key_type key) {
      return keys_set.find(key) != keys_set.end();
    })};
    std::fill(hit_mask.begin(), hit_mask.end(), 0);
    tab->reset_lookup_counter(ctx);
    tab->find(ctx, n, keys2.data(), hit_mask.data(), max_value_size, nullptr, nullptr);
    tab->get_lookup_counter(ctx, &cnt);
    ASSERT_EQ(cnt, act_cnt);

    // TODO: Add values check.
    // std::vector<float> values(value_stride)

  } catch (const Exception& e) {
    NVE_LOG_CRITICAL_(e);
    FAIL();
  }
}

TEST(find_insert_find, stl_map_table) { find_insert_find(R"({"implementation": "umap"})"_json); }

TEST(find_insert_find, nvhm_table) {
  SKIP_IF_NVHM_UNAVAILABLE();
  load_host_table_plugin("nvhm");
  find_insert_find(R"({"implementation": "nvhm_map"})"_json);
}

TEST(find_insert_find, abseil_flat_map_table) {
  SKIP_IF_ABSEIL_UNAVAILABLE();
  load_host_table_plugin("abseil");
  find_insert_find(R"({"implementation": "abseil_flat_map"})"_json);
}

TEST(find_insert_find, phmap_flat_map_table) {
  SKIP_IF_PHMAP_UNAVAILABLE();
  load_host_table_plugin("phmap");
  find_insert_find(R"({"implementation": "phmap_flat_map"})"_json);
}

TEST(find_insert_find, rocksdb_table) {
  SKIP_IF_ROCKSDB_UNAVAILABLE();
  load_host_table_plugin("rocksdb");
  find_insert_find(R"({"implementation": "rocksdb"})"_json);
}
