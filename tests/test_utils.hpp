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

#pragma once

#include <gtest/gtest.h>

#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <host_table.hpp>

namespace nve_test {

inline std::string plugin_so(const std::string& plugin_name) {
  return "libnve-plugin-" + plugin_name + ".so";
}

inline std::filesystem::path nve_library_dir() {
  Dl_info dli;
  if (dladdr(reinterpret_cast<const void*>(nve::load_host_table_plugin), &dli) == 0) {
    throw std::runtime_error("Could not locate nve-common shared library");
  }
  return std::filesystem::path{dli.dli_fname}.parent_path();
}

inline std::string plugin_full_path(const std::string& plugin_name) {
  return (nve_library_dir() / plugin_so(plugin_name)).string();
}

}  // namespace nve_test

// Macros to skip tests if specific plugins are not available.
// These use compile-time checks based on NVE_FEATURE_* preprocessor definitions.

#ifndef NVE_FEATURE_NVHM_PLUGIN
#define SKIP_IF_NVHM_UNAVAILABLE() \
  GTEST_SKIP() << "nvhm plugin is not available (NVE_FEATURE_NVHM_PLUGIN not defined)"
#else
#define SKIP_IF_NVHM_UNAVAILABLE() ((void)0)
#endif

#ifndef NVE_FEATURE_ABSEIL_PLUGIN
#define SKIP_IF_ABSEIL_UNAVAILABLE() \
  GTEST_SKIP() << "abseil plugin is not available (NVE_FEATURE_ABSEIL_PLUGIN not defined)"
#else
#define SKIP_IF_ABSEIL_UNAVAILABLE() ((void)0)
#endif

#ifndef NVE_FEATURE_PHMAP_PLUGIN
#define SKIP_IF_PHMAP_UNAVAILABLE() \
  GTEST_SKIP() << "phmap plugin is not available (NVE_FEATURE_PHMAP_PLUGIN not defined)"
#else
#define SKIP_IF_PHMAP_UNAVAILABLE() ((void)0)
#endif

#ifndef NVE_FEATURE_REDIS_PLUGIN
#define SKIP_IF_REDIS_UNAVAILABLE() \
  GTEST_SKIP() << "redis plugin is not available (NVE_FEATURE_REDIS_PLUGIN not defined)"
#else
#define SKIP_IF_REDIS_UNAVAILABLE() ((void)0)
#endif

#ifndef NVE_FEATURE_ROCKSDB_PLUGIN
#define SKIP_IF_ROCKSDB_UNAVAILABLE() \
  GTEST_SKIP() << "rocksdb plugin is not available (NVE_FEATURE_ROCKSDB_PLUGIN not defined)"
#else
#define SKIP_IF_ROCKSDB_UNAVAILABLE() ((void)0)
#endif
