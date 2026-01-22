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

#include <dlfcn.h>

#include <bit_ops.hpp>
#include <filesystem>
#include <host_table.hpp>
#include <json_support.hpp>
#include <limits>
#include <plugin.hpp>
#include <stl_map_backed_table.hpp>
#include <unordered_map>

namespace nve {

static host_table_factory_ptr_t create_stl_map_table_factory(const nlohmann::json& json) {
  STLMapTableFactoryConfig config{json};
  return std::make_shared<STLMapTableFactory>(config);
}

NVE_DEFINE_JSON_ENUM_CONVERSION_(Partitioner_t, Partitioner_t::AlwaysZero,
                                 Partitioner_t::FowlerNollVo, Partitioner_t::Murmur3,
                                 Partitioner_t::Rrxmrrxmsx0, Partitioner_t::StdHash)

NVE_DEFINE_JSON_ENUM_CONVERSION_(OverflowHandler_t, OverflowHandler_t::EvictRandom,
                                 OverflowHandler_t::EvictLRU, OverflowHandler_t::EvictLFU)

void OverflowPolicyConfig::check() const {
  NVE_CHECK_(overflow_margin >= 0);
  NVE_CHECK_(resolution_margin >= 0.0 && resolution_margin <= 1.0);
}

void from_json(const nlohmann::json& json, OverflowPolicyConfig& conf) {
  NVE_READ_JSON_FIELD_(overflow_margin);
  NVE_READ_JSON_FIELD_(handler);
  NVE_READ_JSON_FIELD_(resolution_margin);
}

void to_json(nlohmann::json& json, const OverflowPolicyConfig& conf) {
  json = json.object();

  NVE_WRITE_JSON_FIELD_(overflow_margin);
  NVE_WRITE_JSON_FIELD_(handler);
  NVE_WRITE_JSON_FIELD_(resolution_margin);
}

void HostTableConfig::check() const {
  static const auto mask_sizes{
      to_array<int64_t>(bitmask8_t::size, bitmask16_t::size, bitmask32_t::size, bitmask64_t::size)};
  NVE_CHECK_(std::find(mask_sizes.begin(), mask_sizes.end(), mask_size) != mask_sizes.end());

  static const auto key_sizes{
      to_array<int64_t>(sizeof(int8_t), sizeof(int16_t), sizeof(int32_t), sizeof(int64_t))};
  NVE_CHECK_(std::find(key_sizes.begin(), key_sizes.end(), key_size) != key_sizes.end());

  NVE_CHECK_(max_value_size > 0 &&
             max_value_size <=
                 static_cast<int64_t>(std::numeric_limits<int32_t>::max() - sizeof(int32_t)));
  NVE_CHECK_(max_value_size % value_dtype_size() == 0);
}

void from_json(const nlohmann::json& json, HostTableConfig& conf) {
  NVE_READ_JSON_FIELD_(mask_size);
  NVE_READ_JSON_FIELD_(key_size);
  NVE_READ_JSON_FIELD_(max_value_size);
  NVE_READ_JSON_FIELD_(value_dtype);
}

void to_json(nlohmann::json& json, const HostTableConfig& conf) {
  json = json.object();

  NVE_WRITE_JSON_FIELD_(mask_size);
  NVE_WRITE_JSON_FIELD_(key_size);
  NVE_WRITE_JSON_FIELD_(max_value_size);
  NVE_WRITE_JSON_FIELD_(value_dtype);
}

void HostTableFactoryConfig::check() const {}

void from_json(const nlohmann::json&, HostTableFactoryConfig&) {}

void to_json(nlohmann::json& json, const HostTableFactoryConfig&) { json = json.object(); }

HostTableLike::HostTableLike(const table_id_t id) : id{id} {}

static std::unordered_map<std::string, create_host_table_factory_t> htf_impls{
    {"stl_map", create_stl_map_table_factory},
    {"map", create_stl_map_table_factory},
    {"stl_unordered_map", create_stl_map_table_factory},
    {"unordered_map", create_stl_map_table_factory},
    {"stl_umap", create_stl_map_table_factory},
    {"umap", create_stl_map_table_factory}};

static void register_implementation(void* dll, const char* htf_name) {
  if (htf_impls.find(htf_name) != htf_impls.end()) {
    NVE_LOG_INFO_("HostTableFactory implementation '", htf_name, "' is already registered.");
    return;
  }

  const std::string create_htf_name{std::string{"create_"} + htf_name + "_table_factory"};
  create_host_table_factory_t create_htf;
  NVE_CHECK_((create_htf = reinterpret_cast<create_host_table_factory_t>(
                  dlsym(dll, create_htf_name.c_str()))) != nullptr,
             dlerror());
  htf_impls.emplace(htf_name, create_htf);

  NVE_LOG_INFO_("Registered HostTableFactory implementation '", htf_name, "'.");
}

void load_host_table_plugin(const std::string_view& plugin_name) {
  const std::string dll_name{to_string("libnve-plugin-", plugin_name, ".so")};
  NVE_LOG_INFO_("Attempting to load host table plugin '", dll_name, "\'.");

  // Try to load adjacent DLLs first, and if that fails use `LD_LIBRARY_PATH`.
  Dl_info dli;
  NVE_CHECK_(dladdr(reinterpret_cast<const void*>(load_host_table_plugin), &dli) != 0);
  void* dll{dlopen((std::filesystem::path{dli.dli_fname}.parent_path() / dll_name).c_str(),
                   RTLD_NOW | RTLD_GLOBAL)};
  if (dll == nullptr) {
    NVE_LOG_CRITICAL_("error: ", dlerror());
    NVE_CHECK_((dll = dlopen(dll_name.c_str(), RTLD_NOW | RTLD_GLOBAL)) != nullptr, dlerror());
  }

  plugin_info_t plugin_ident;
  NVE_CHECK_(
      (plugin_ident = reinterpret_cast<plugin_info_t>(dlsym(dll, "plugin_ident"))) != nullptr,
      dlerror());
  plugin_info_t plugin_dev;
  NVE_CHECK_(
      (plugin_dev = reinterpret_cast<plugin_info_t>(dlsym(dll, "plugin_developer"))) != nullptr,
      dlerror());
  NVE_LOG_INFO_("Plugin = '", plugin_ident(), "\', developer = '", plugin_dev(), "'.");

  // Register all implementations present in this plugin.
  enum_implementations_t enum_ht_implementations;
  NVE_CHECK_((enum_ht_implementations = reinterpret_cast<enum_implementations_t>(
                  dlsym(dll, "enum_host_table_implementations"))) != nullptr,
             dlerror());
  enum_ht_implementations(dll, register_implementation);
}

create_host_table_factory_t resolve_host_table_implementation(const std::string& impl_name) {
  // Try to find HTF.
  auto it{htf_impls.find(impl_name)};
  if (it != htf_impls.end()) {
    NVE_LOG_INFO_("Plugin '", impl_name, "' already loaded!");
    return it->second;
  }

  // Try to load plugin DLL with the same name and try again.
  load_host_table_plugin(impl_name);
  return htf_impls.at(impl_name);
}

host_table_factory_ptr_t create_host_table_factory(const nlohmann::json& json) {
  // Determine what implementation should be used.
  std::string impl_name{"umap"};
  auto it{json.find("implementation")};
  if (it != json.end()) {
    it->get_to(impl_name);
  }
  create_host_table_factory_t create_htf{resolve_host_table_implementation(impl_name)};

  NVE_LOG_INFO_("Creating table factory via implementation '", impl_name, "\'.");
  return create_htf(json);
}

std::map<table_id_t, host_table_ptr_t> build_host_database(const nlohmann::json& json_db) {
  auto it{json_db.find("plugins")};
  if (it != json_db.end()) {
    const nlohmann::json json_plugs{*it};
    for (auto it{json_plugs.begin()}; it != json_plugs.end(); ++it) {
      load_host_table_plugin(it->get<std::string>());
    }
  }

  std::unordered_map<std::string, host_table_factory_ptr_t> factories;
  it = json_db.find("table_factories");
  if (it != json_db.end()) {
    const nlohmann::json json_facs{*it};
    for (auto it{json_facs.begin()}; it != json_facs.end(); ++it) {
      const nlohmann::json json{it.value()};
      host_table_factory_ptr_t factory{create_host_table_factory(json)};
      factories.emplace(it.key(), factory);
    }
  }

  std::map<table_id_t, host_table_ptr_t> tables;
  it = json_db.find("tables");
  if (it != json_db.end()) {
    const nlohmann::json json_tabs{*it};
    for (auto it{json_tabs.begin()}; it != json_tabs.end(); ++it) {
      const table_id_t table_id{std::stoll(it.key())};
      const nlohmann::json json{it.value()};
      host_table_factory_ptr_t factory{factories.at(json.get<std::string>())};
      tables.emplace(table_id, factory->produce(table_id, json));
    }
  }
  return tables;
}

}  // namespace nve
