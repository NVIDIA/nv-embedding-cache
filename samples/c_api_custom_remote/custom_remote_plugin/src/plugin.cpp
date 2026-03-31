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

#include <custom_remote_table.hpp>
#include <plugin.hpp>

extern "C" const char* plugin_ident() noexcept {
  return "std::map-backed remote table plugin (sample)";
}

extern "C" const char* plugin_developer() noexcept { return "NVIDIA Corporation"; }

extern "C" void enum_host_table_implementations(void* dll,
                                                void (*callback)(void*, const char*)) noexcept {
  callback(dll, "custom_remote");
}

using namespace nve;

#pragma GCC diagnostic push
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wreturn-type-c-linkage"
#endif
extern "C" host_table_factory_ptr_t create_custom_remote_table_factory(
    const nlohmann::json& /*json*/) {
#pragma GCC diagnostic pop
  return {new CustomRemoteTableFactory(),
          [](CustomRemoteTableFactory* const p) { delete p; }};
}
