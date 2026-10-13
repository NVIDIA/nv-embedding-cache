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
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
#include <unistd.h>

#include <buffer_wrapper.hpp>
#include <execution_context.hpp>
#include <host_table.hpp>
#include <plugin/plugin_loader.hpp>
#include "test_utils.hpp"

using namespace nve;
using namespace nlohmann::literals;

TEST(general_tests, load_dlls) {
  try {
    std::vector<std::string> plugin_names{};
#ifdef NVE_FEATURE_NVHM_PLUGIN
    plugin_names.push_back(nve_test::plugin_full_path("nvhm"));
#endif
#ifdef NVE_FEATURE_ABSEIL_PLUGIN
    plugin_names.push_back(nve_test::plugin_full_path("abseil"));
#endif
#ifdef NVE_FEATURE_PHMAP_PLUGIN
    plugin_names.push_back(nve_test::plugin_full_path("phmap"));
#endif
#ifdef NVE_FEATURE_REDIS_PLUGIN
    plugin_names.push_back(nve_test::plugin_full_path("redis"));
#endif
#ifdef NVE_FEATURE_ROCKSDB_PLUGIN
    plugin_names.push_back(nve_test::plugin_full_path("rocksdb"));
#endif
    std::for_each(plugin_names.begin(), plugin_names.end(),
                  [](const std::string& name) { Plugin plugin{name}; });
  } catch (const Exception& e) {
    NVE_LOG_CRITICAL_(e);
    FAIL();
  }
}

TEST(general_tests, shorthand_plugin_names_are_not_supported) {
  EXPECT_THROW(Plugin{"abseil"}, Exception);
}

void create_and_clear(const std::string& plugin_path, const nlohmann::json& fac_json) {
  try {
    Plugin plugin{plugin_path};
    table_factory_ptr_t fac{plugin.create_table_factory(fac_json)};
    table_ptr_t tab{fac->produce(4711, R"({})"_json)};
    auto ctx = tab->create_execution_context(0, 0, nullptr, nullptr);
    tab->clear(ctx);
  } catch (const Exception& e) {
    NVE_LOG_CRITICAL_(e);
    FAIL();
  }
}

TEST(create_and_clear, stl_map_table) {
  create_and_clear(nve_test::plugin_full_path("stl-map"), nlohmann::json::object());
}

TEST(create_and_clear, nvhm_table) {
  SKIP_IF_NVHM_UNAVAILABLE();
  create_and_clear(nve_test::plugin_full_path("nvhm"), nlohmann::json::object());
}

TEST(create_and_clear, abseil_flat_map_table) {
  SKIP_IF_ABSEIL_UNAVAILABLE();
  create_and_clear(nve_test::plugin_full_path("abseil"), nlohmann::json::object());
}

TEST(create_and_clear, phmap_flat_map_table) {
  SKIP_IF_PHMAP_UNAVAILABLE();
  create_and_clear(nve_test::plugin_full_path("phmap"), nlohmann::json::object());
}

TEST(create_and_clear, rocksdb_table) {
  SKIP_IF_ROCKSDB_UNAVAILABLE();
  create_and_clear(nve_test::plugin_full_path("rocksdb"), nlohmann::json::object());
}

TEST(host_table_invalid_key, custom_value_round_trips) {
  table_factory_ptr_t fac{nve_test::plugin_factory("stl-map")};
  table_ptr_t tab{fac->produce(4712, R"({"max_value_size": 16, "invalid_key": 999})"_json)};
  EXPECT_EQ(int64_t{999}, tab->get_invalid_key());
}

// A batch containing the configured invalid_key sentinel must not crash insert/find,
// and other (valid) keys in the same batch must still be retrievable.
TEST(host_table_invalid_key, valid_keys_survive_batch_with_sentinel) {
  using key_type = int64_t;
  constexpr int64_t max_value_size = 16;
  constexpr key_type sentinel = static_cast<key_type>(7777777);

  table_factory_ptr_t fac{nve_test::plugin_factory("stl-map")};
  nlohmann::json table_conf = nlohmann::json::object();
  table_conf["max_value_size"] = max_value_size;
  table_conf["invalid_key"] = static_cast<int64_t>(sentinel);
  table_ptr_t tab{fac->produce(4713, table_conf)};
  ASSERT_EQ(static_cast<int64_t>(sentinel), tab->get_invalid_key());

  auto ctx = tab->create_execution_context(0, 0, nullptr, nullptr);
  tab->clear(ctx);

  // Mixed batch: three valid keys + the configured sentinel at slot 2.
  std::vector<key_type> keys{static_cast<key_type>(1),
                             static_cast<key_type>(2),
                             sentinel,
                             static_cast<key_type>(4)};
  const int64_t n = static_cast<int64_t>(keys.size());

  // Insert with values=null (only registers presence; find_insert_tests.cpp does the same).
  {
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(
        ctx, "keys", keys.data(), to_uint(n) * sizeof(key_type));
    tab->insert(ctx, n, std::move(keys_bw), max_value_size + 1, 0, nullptr);
  }

  std::vector<bitmask64_t> hit_mask(to_uint(ceil_div(n, bitmask64::num_bits)), 0);
  tab->reset_lookup_counter(ctx);
  {
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(
        ctx, "keys", keys.data(), to_uint(n) * sizeof(key_type));
    auto hit_mask_bw = std::make_shared<BufferWrapper<bitmask64_t>>(
        ctx, "hit_mask", hit_mask.data(), hit_mask.size() * sizeof(bitmask64_t));
    tab->find(ctx, n, std::move(keys_bw), std::move(hit_mask_bw), max_value_size, nullptr, nullptr);
  }

  // Look up the valid keys on their own — they must hit.
  std::vector<key_type> valid_keys{static_cast<key_type>(1),
                                   static_cast<key_type>(2),
                                   static_cast<key_type>(4)};
  const int64_t valid_n{static_cast<int64_t>(valid_keys.size())};
  std::vector<bitmask64_t> valid_hit_mask(to_uint(ceil_div(valid_n, bitmask64::num_bits)), 0);
  int64_t valid_cnt = 0;
  tab->reset_lookup_counter(ctx);
  {
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(
        ctx, "keys", valid_keys.data(), valid_keys.size() * sizeof(key_type));
    auto hit_mask_bw = std::make_shared<BufferWrapper<bitmask64_t>>(
        ctx, "hit_mask", valid_hit_mask.data(), valid_hit_mask.size() * sizeof(bitmask64_t));
    tab->find(ctx, valid_n, std::move(keys_bw), std::move(hit_mask_bw),
                 max_value_size, nullptr, nullptr);
  }
  tab->get_lookup_counter(ctx, &valid_cnt);
  EXPECT_EQ(static_cast<int64_t>(valid_keys.size()), valid_cnt);
}

TEST(create_and_clear, arbitrary_so_path) {
  SKIP_IF_ABSEIL_UNAVAILABLE();
  const std::filesystem::path source_path{nve_test::plugin_full_path("abseil")};
  if (!std::filesystem::exists(source_path)) {
    GTEST_SKIP() << "abseil plugin is not available at " << source_path;
  }

  const std::filesystem::path target_path{
      std::filesystem::temp_directory_path() /
      ("nve-host-table-plugin-any-name-" + std::to_string(getpid()) + ".so")};
  std::filesystem::copy_file(source_path, target_path,
                             std::filesystem::copy_options::overwrite_existing);
  const nve_test::TempPathCleanup cleanup{target_path};
  Plugin plugin{target_path.string()};

  table_factory_ptr_t fac{plugin.create_table_factory(nlohmann::json::object())};
  EXPECT_NE(fac, nullptr);
}
