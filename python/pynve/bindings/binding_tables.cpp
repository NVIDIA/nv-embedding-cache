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

#include "python/pynve/bindings/binding_tables.hpp"

#include "include/buffer_wrapper.hpp"
#include "include/common.hpp"
#include "include/serialization.hpp"
#include "include/table_utils.hpp"

namespace nve {

ParameterServerTable::ParameterServerTable(
    uint64_t num_rows,
    uint64_t row_elements,
    nve::DataType_t data_type,
    uint64_t initial_size,
    ParameterServerTable::PSType_t ps_type,
    const std::string& extra_params)
  : PyNVETable({}),
    num_rows_(num_rows),
    row_elements_(row_elements),
    row_bytes_(row_elements * static_cast<uint64_t>(dtype_size(data_type))),
    data_type_(data_type)
{
    table_id_t table_id =
        resolve_ps_type(ps_type, initial_size, extra_params);
    init_from_plugin(table_id);
}

table_id_t ParameterServerTable::resolve_ps_type(
    ParameterServerTable::PSType_t ps_type,
    uint64_t initial_size,
    const std::string& extra_params)
{
    constexpr int64_t num_partitions = 1; // Single partition is better for inference
    const int64_t keys_per_partition = static_cast<int64_t>(num_rows_) / num_partitions;
    initial_size = std::max<uint64_t>(initial_size, 1024);

    table_config_ = {
        {"key_size", sizeof(KeyType)},
        {"initial_capacity", initial_size / static_cast<uint64_t>(num_partitions)},
        {"max_value_size", row_bytes_},
        {"num_partitions", num_partitions},
        {"value_dtype", to_string(data_type_)},
        {"value_alignment", 32},
        {"overflow_policy",
        {
            // Using random eviction, assuming gpu cache handles all host keys. Otherwise, replace with "evict_lru".
            {"handler", "evict_random"},
            // (num_rows_ == 0) implies unlimited size (grow until OOM). Use erase_keys() to remove data manually.
            {"overflow_margin", (num_rows_ > 0) ? keys_per_partition : INT64_MAX},
            {"resolution_margin", 0.95}
        }
        },
        {"max_num_keys_per_task", 4*1024},
    };
    if (num_partitions == 1){
        table_config_["partitioner"] = "always_zero";
    }

    // An empty json, instead of a null json (the default c'tor)
    // Since merge_patch() with a null json results with a null json, regardless of the other json
    const auto empty_json = nlohmann::json(nlohmann::json::value_t::object);
    auto json_extra_params = (extra_params.length() > 0) ? nlohmann::json::parse(extra_params) : empty_json;
    nlohmann::json extra_params_plugin(empty_json);
    nlohmann::json extra_params_table(empty_json);
    auto constexpr PLUGIN_CFG = "plugin";
    auto constexpr TABLE_CFG = "table";

    if (json_extra_params.contains(PLUGIN_CFG)) {
        extra_params_plugin = json_extra_params[PLUGIN_CFG];
    }
    if (json_extra_params.contains(TABLE_CFG)) {
        extra_params_table = json_extra_params[TABLE_CFG];
    }
    table_config_.merge_patch(extra_params_table);

    switch (ps_type) {
        default:
            NVE_LOG_WARNING_("Invalid ps_type, defaulting to NVHashMap");
            [[fallthrough]];
        case ParameterServerTable::PSType_t::NVHashMap:
            plugin_name_    = "libnve-plugin-nvhm.so";
            factory_config_ = R"({"implementation": "nvhm_map"})"_json;
            return 100;
        case ParameterServerTable::PSType_t::Abseil:
            plugin_name_    = "libnve-plugin-abseil.so";
            factory_config_ = R"({"implementation": "abseil_flat_map"})"_json;
            return 200;
        case ParameterServerTable::PSType_t::ParallelHash:
            plugin_name_    = "libnve-plugin-phmap.so";
            factory_config_ = R"({"implementation": "phmap_flat_map"})"_json;
            return 300;
        case ParameterServerTable::PSType_t::Redis:
            plugin_name_    = "libnve-plugin-redis.so";
            factory_config_ = R"({"address": "localhost:7000", "implementation": "redis_cluster"})"_json;
            factory_config_.merge_patch(extra_params_plugin);
            // Redis uses a different table-config shape than the in-memory plugins.
            table_config_ = {
                {"mask_size", sizeof(uint64_t)},
                {"key_size", sizeof(KeyType)},
                {"max_value_size", row_bytes_},
                {"value_dtype", to_string(data_type_)},
            };
            table_config_.merge_patch(extra_params_table);
            return 400;
    }
}

ParameterServerTable::ParameterServerTable(
    uint64_t row_elements,
    nve::DataType_t data_type,
    std::string plugin_name,
    nlohmann::json factory_config,
    nlohmann::json table_config,
    uint64_t num_rows,
    table_id_t table_id)
  : PyNVETable({}),
    num_rows_(num_rows),
    row_elements_(row_elements),
    row_bytes_(row_elements * static_cast<uint64_t>(dtype_size(data_type))),
    data_type_(data_type),
    plugin_name_(std::move(plugin_name)),
    factory_config_(std::move(factory_config)),
    table_config_(std::move(table_config))
{
    init_from_plugin(table_id);
}

void ParameterServerTable::init_from_plugin(table_id_t table_id)
{
    set_table(create_table_from_plugin(
        plugin_name_, factory_config_, table_config_, table_id));
    // initialize an execution context for the long-lived insert/erase ops
    ctx_ = create_execution_context(0, 0, nullptr, nullptr);
}

nlohmann::json ParameterServerTable::export_config() const
{
    return {
        {"remote_ps_type", "plugin"},
        {"plugin_name",    plugin_name_},
        {"factory_config", factory_config_},
        {"table_config",   table_config_},
        {"row_elements",   row_elements_},
        {"num_rows",       num_rows_},
        {"data_type",      to_string(data_type_)},
    };
}

ParameterServerTable::~ParameterServerTable() {
    ctx_.reset();
}

void ParameterServerTable::insert_keys(size_t num_keys, uintptr_t keys, uintptr_t values) {
    NVE_CHECK_(keys != 0, "Invalid Keys tensor");
    NVE_CHECK_(values != 0, "Invalid Values tensor");
    const auto keys_buffer_size = num_keys * sizeof(KeyType);
    const auto values_buffer_size = num_keys * row_bytes_;
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx_, "keys", reinterpret_cast<const void*>(keys), keys_buffer_size);
    auto values_bw = std::make_shared<BufferWrapper<const void>>(ctx_, "values", reinterpret_cast<const void*>(values), values_buffer_size);
    insert(  ctx_,
                static_cast<int64_t>(num_keys),
                std::move(keys_bw),
                static_cast<int64_t>(row_bytes_),
                static_cast<int64_t>(row_bytes_),
                std::move(values_bw));
}

void ParameterServerTable::insert_keys_from_tensor_file(std::shared_ptr<TensorFileFormatBase> keys_file_reader,
                                                        std::shared_ptr<TensorFileFormatBase> values_file_reader,
                                                        uint64_t batch_size) {
    nve::insert_keys_from_tensor_file(inner_table(), ctx_,
                                      std::move(keys_file_reader),
                                      std::move(values_file_reader),
                                      row_bytes_, batch_size);
}

uint64_t ParameterServerTable::get_row_size_in_bytes() const {
    return row_bytes_;
}

void ParameterServerTable::insert_keys_from_filepath(const std::string& keys_path,
                                                     const std::string& values_path,
                                                     uint64_t batch_size) {
    nve::insert_keys_from_filepath(inner_table(), ctx_,
                                   keys_path, values_path,
                                   row_bytes_, batch_size);
}

void ParameterServerTable::erase_keys(size_t num_keys, uintptr_t keys) {
    NVE_CHECK_(keys != 0, "Invalid Keys tensor");
    const auto keys_buffer_size = num_keys * sizeof(KeyType);
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx_, "keys", reinterpret_cast<const void*>(keys), keys_buffer_size);
    erase(ctx_, static_cast<int64_t>(num_keys), std::move(keys_bw));
}

void ParameterServerTable::clear_keys() {
    clear(ctx_);
}

} // namespace nve
