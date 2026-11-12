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
#include <limits>
#include <buffer_wrapper.hpp>
#include <numeric>
#include <set>
#include <type_traits>
#include <execution_context.hpp>
#include <host_table.hpp>
#include <plugin/plugin_loader.hpp>
#include <random>
#include "test_utils.hpp"

using namespace nve;
using namespace nlohmann::literals;

// Key type matching the build's default host table key size.
using default_ht_key_type = std::conditional_t<
    default_ht_key_size == sizeof(int64_t), int64_t,
    std::conditional_t<default_ht_key_size == sizeof(int32_t), int32_t,
                       std::conditional_t<default_ht_key_size == sizeof(int16_t), int16_t, int8_t>>>;

// Uniform keys in [0, 1'000'000], narrowed to what `KeyType` can hold.
template <typename KeyType>
std::vector<KeyType> make_uniform_keys(int64_t n) {
  std::vector<KeyType> keys(to_uint(n));

  std::default_random_engine rng(random_seed());
  std::uniform_int_distribution<int64_t> dist(0, std::min<int64_t>(1'000'000, std::numeric_limits<KeyType>::max()));
  std::generate(keys.begin(), keys.end(), [&dist, &rng]() { return static_cast<KeyType>(dist(rng)); });

  return keys;
}

void replace(std::string& x, const std::string_view& a, const std::string_view& b) {
  x.replace(x.find(a), a.size(), b);
}

void find_insert_find(const std::string& plugin_path, const nlohmann::json& fac_json) {
  try {
    using key_type = default_ht_key_type;

    Plugin plugin{plugin_path};
    table_factory_ptr_t fac{plugin.create_table_factory(fac_json)};
    std::string table_conf{R"({
      "key_size": %key_size,
      "max_value_size": %max_value_size
    })"};
    const int64_t max_value_size{20};

    replace(table_conf, "%key_size", std::to_string(sizeof(key_type)));
    replace(table_conf, "%max_value_size", std::to_string(max_value_size));
    table_ptr_t tab{fac->produce(4711, nlohmann::json::parse(table_conf))};

    auto ctx = tab->create_execution_context(0, 0, nullptr, nullptr);
    tab->clear(ctx);

    // Find -> nothing -> insert -> find.
    int64_t n{63999};
    std::vector<key_type> keys{make_uniform_keys<key_type>(n)};
    std::vector<bitmask64_t> hit_mask(to_uint(ceil_div(n, bitmask64::num_bits)));
    const int64_t value_stride{max_value_size + 1};

    int64_t cnt;

    auto make_keys_bw = [&](const key_type* p) {
      return std::make_shared<BufferWrapper<const void>>(ctx, "keys", p, to_uint(n) * sizeof(key_type));
    };
    auto make_hit_mask_bw = [&]() {
      return std::make_shared<BufferWrapper<bitmask64_t>>(
          ctx, "hit_mask", hit_mask.data(), hit_mask.size() * sizeof(bitmask64_t));
    };

    std::fill(hit_mask.begin(), hit_mask.end(), 0);
    tab->reset_lookup_counter(ctx);
    tab->find(ctx, n, make_keys_bw(keys.data()), make_hit_mask_bw(), max_value_size, nullptr, nullptr);
    tab->get_lookup_counter(ctx, &cnt);
    ASSERT_EQ(cnt, 0);
    tab->insert(ctx, n, make_keys_bw(keys.data()), value_stride, 0, nullptr);
    tab->reset_lookup_counter(ctx);
    tab->find(ctx, n, make_keys_bw(keys.data()), make_hit_mask_bw(), max_value_size, nullptr, nullptr);
    tab->get_lookup_counter(ctx, &cnt);
    ASSERT_EQ(cnt, n);

    std::fill(hit_mask.begin(), hit_mask.end(), bitmask64::full);
    tab->reset_lookup_counter(ctx);
    tab->find(ctx, n, make_keys_bw(keys.data()), make_hit_mask_bw(), max_value_size, nullptr, nullptr);
    tab->get_lookup_counter(ctx, &cnt);
    ASSERT_EQ(cnt, 0);

    std::set<key_type> keys_set(keys.begin(), keys.end());
    std::vector<key_type> keys2{make_uniform_keys<key_type>(n)};
    int64_t act_cnt{std::count_if(keys2.begin(), keys2.end(), [&keys_set](const key_type key) {
      return keys_set.find(key) != keys_set.end();
    })};
    std::fill(hit_mask.begin(), hit_mask.end(), 0);
    tab->reset_lookup_counter(ctx);
    tab->find(ctx, n, make_keys_bw(keys2.data()), make_hit_mask_bw(), max_value_size, nullptr, nullptr);
    tab->get_lookup_counter(ctx, &cnt);
    ASSERT_EQ(cnt, act_cnt);

    // TODO: Add values check.
    // std::vector<float> values(value_stride)

  } catch (const Exception& e) {
    NVE_LOG_CRITICAL_(e);
    FAIL();
  }
}

// size()/get_keys() track insert, erase and clear; get_keys() clips to the buffer capacity.
void size_and_get_keys(const std::string& plugin_path, const nlohmann::json& fac_json,
                       const nlohmann::json& extra_tab_json = nlohmann::json::object()) {
  try {
    using key_type = default_ht_key_type;

    Plugin plugin{plugin_path};
    table_factory_ptr_t fac{plugin.create_table_factory(fac_json)};
    const int64_t max_value_size{16};
    nlohmann::json tab_json{{"key_size", sizeof(key_type)}, {"max_value_size", max_value_size}};
    tab_json.update(extra_tab_json);
    table_ptr_t tab{fac->produce(4712, tab_json)};

    auto ctx = tab->create_execution_context(0, 0, nullptr, nullptr);
    tab->clear(ctx);
    ASSERT_EQ(tab->size(ctx), 0);

    // Distinct positive keys; fewer than 5000 only for 8-bit keys.
    const int64_t n{std::min<int64_t>(5000, std::numeric_limits<key_type>::max())};
    std::vector<key_type> keys(to_uint(n));
    std::iota(keys.begin(), keys.end(), key_type{1});
    std::vector<char> values(to_uint(n * max_value_size), 1);
    tab->insert(ctx, n,
                std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys.data(), to_uint(n) * sizeof(key_type)),
                max_value_size, max_value_size,
                std::make_shared<BufferWrapper<const void>>(ctx, "values", values.data(), values.size()));
    ASSERT_EQ(tab->size(ctx), n);

    auto fetch = [&](const int64_t capacity) {
      std::vector<key_type> out(to_uint(capacity), key_type{-1});
      const int64_t written{tab->get_keys(ctx, capacity, out.data())};
      out.resize(to_uint(written));
      return out;
    };

    // Full capacity returns exactly the inserted set.
    std::vector<key_type> all{fetch(n + 10)};
    ASSERT_EQ(static_cast<int64_t>(all.size()), n);
    std::sort(all.begin(), all.end());
    ASSERT_EQ(all, keys);

    // A smaller buffer is filled to capacity with stored keys only.
    const std::set<key_type> key_set(keys.begin(), keys.end());
    std::vector<key_type> some{fetch(n / 3)};
    ASSERT_EQ(static_cast<int64_t>(some.size()), n / 3);
    ASSERT_TRUE(std::all_of(some.begin(), some.end(), [&](key_type k) { return key_set.count(k) == 1; }));
    ASSERT_EQ(std::set<key_type>(some.begin(), some.end()).size(), some.size()) << "duplicate keys returned";
    ASSERT_EQ(fetch(0).size(), 0u);
    EXPECT_THROW(tab->get_keys(ctx, -1, nullptr), nve::Exception);

    // Erase half: only the remaining keys are reported.
    const int64_t half{n / 2};
    tab->erase(ctx, half,
               std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys.data(), to_uint(half) * sizeof(key_type)));
    ASSERT_EQ(tab->size(ctx), n - half);
    std::vector<key_type> rest{fetch(n)};
    std::sort(rest.begin(), rest.end());
    ASSERT_EQ(rest, std::vector<key_type>(keys.begin() + half, keys.end()));

    tab->clear(ctx);
    ASSERT_EQ(tab->size(ctx), 0);
    ASSERT_EQ(fetch(n).size(), 0u);
  } catch (const Exception& e) {
    NVE_LOG_CRITICAL_(e);
    FAIL();
  }
}

TEST(size_and_get_keys, stl_map_table) {
  size_and_get_keys(nve_test::plugin_full_path("stl-map"), nlohmann::json::object());
}

TEST(size_and_get_keys, nvhm_table) {
  SKIP_IF_NVHM_UNAVAILABLE();
  size_and_get_keys(nve_test::plugin_full_path("nvhm"), nlohmann::json::object());
}

// Partial capacities skip some partitions and stop one partway.
TEST(size_and_get_keys, nvhm_table_partitioned) {
  SKIP_IF_NVHM_UNAVAILABLE();
  size_and_get_keys(nve_test::plugin_full_path("nvhm"), nlohmann::json::object(), {{"num_partitions", 8}});
}

TEST(size_and_get_keys, abseil_flat_map_table) {
  SKIP_IF_ABSEIL_UNAVAILABLE();
  size_and_get_keys(nve_test::plugin_full_path("abseil"), nlohmann::json::object());
}

TEST(size_and_get_keys, phmap_flat_map_table) {
  SKIP_IF_PHMAP_UNAVAILABLE();
  size_and_get_keys(nve_test::plugin_full_path("phmap"), nlohmann::json::object());
}

TEST(size_and_get_keys, rocksdb_table) {
  SKIP_IF_ROCKSDB_UNAVAILABLE();
  size_and_get_keys(nve_test::plugin_full_path("rocksdb"), nlohmann::json::object());
}

TEST(find_insert_find, stl_map_table) {
  find_insert_find(nve_test::plugin_full_path("stl-map"), nlohmann::json::object());
}

TEST(find_insert_find, nvhm_table) {
  SKIP_IF_NVHM_UNAVAILABLE();
  find_insert_find(nve_test::plugin_full_path("nvhm"), nlohmann::json::object());
}

TEST(find_insert_find, abseil_flat_map_table) {
  SKIP_IF_ABSEIL_UNAVAILABLE();
  find_insert_find(nve_test::plugin_full_path("abseil"), nlohmann::json::object());
}

TEST(find_insert_find, phmap_flat_map_table) {
  SKIP_IF_PHMAP_UNAVAILABLE();
  find_insert_find(nve_test::plugin_full_path("phmap"), nlohmann::json::object());
}

TEST(find_insert_find, rocksdb_table) {
  SKIP_IF_ROCKSDB_UNAVAILABLE();
  find_insert_find(nve_test::plugin_full_path("rocksdb"), nlohmann::json::object());
}
