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

#include <execution_context.hpp>
#include <host_table.hpp>
#include "test_utils.hpp"

using namespace nve;
using namespace nlohmann::literals;

TEST(general_tests, load_dlls) {
  try {
    std::vector<std::string> plugin_names{};
#ifdef NVE_FEATURE_NVHM_PLUGIN
    plugin_names.push_back("nvhm");
#endif
#ifdef NVE_FEATURE_ABSEIL_PLUGIN
    plugin_names.push_back("abseil");
#endif
#ifdef NVE_FEATURE_PHMAP_PLUGIN
    plugin_names.push_back("phmap");
#endif
#ifdef NVE_FEATURE_REDIS_PLUGIN
    plugin_names.push_back("redis");
#endif
#ifdef NVE_FEATURE_ROCKSDB_PLUGIN
    plugin_names.push_back("rocksdb");
#endif
    load_host_table_plugins(plugin_names.begin(), plugin_names.end());
  } catch (const Exception& e) {
    NVE_LOG_CRITICAL_(e);
    FAIL();
  }
}

void create_and_clear(const nlohmann::json& fac_json) {
  try {
    host_table_factory_ptr_t fac{create_host_table_factory(fac_json)};
    host_table_ptr_t tab{fac->produce(4711, R"({})"_json)};
    auto ctx = tab->create_execution_context(0, 0, nullptr, nullptr);
    tab->clear(ctx);
  } catch (const Exception& e) {
    NVE_LOG_CRITICAL_(e);
    FAIL();
  }
}

TEST(create_and_clear, stl_map_table) { create_and_clear(R"({"implementation": "umap"})"_json); }

TEST(create_and_clear, nvhm_table) {
  SKIP_IF_NVHM_UNAVAILABLE();
  create_and_clear(R"({"implementation": "nvhm_map"})"_json);
}

TEST(create_and_clear, abseil_flat_map_table) {
  SKIP_IF_ABSEIL_UNAVAILABLE();
  create_and_clear(R"({"implementation": "abseil_flat_map"})"_json);
}

TEST(create_and_clear, phmap_flat_map_table) {
  SKIP_IF_PHMAP_UNAVAILABLE();
  create_and_clear(R"({"implementation": "phmap_flat_map"})"_json);
}

TEST(create_and_clear, rocksdb_table) {
  SKIP_IF_ROCKSDB_UNAVAILABLE();
  create_and_clear(R"({"implementation": "rocksdb"})"_json);
}
