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

#include <plugin.hpp>
#include <redis_cluster_table.hpp>

extern "C" const char* plugin_ident() noexcept { return "NVIDIA plugin for Redis"; }

extern "C" const char* plugin_developer() noexcept { return "NVIDIA Corporation"; }

extern "C" void enum_host_table_implementations(void* dll,
                                                void (*callback)(void*, const char*)) noexcept {
  callback(dll, "redis_cluster");
}

using namespace nve;
using namespace nve::plugin;

#pragma GCC diagnostic push
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wreturn-type-c-linkage"
#endif
extern "C" host_table_factory_ptr_t create_redis_cluster_table_factory(const nlohmann::json& json) {
#pragma GCC diagnostic pop
  return {new RedisClusterTableFactory(static_cast<RedisClusterTableFactoryConfig>(json)),
          [](RedisClusterTableFactory* const p) { delete p; }};
}
