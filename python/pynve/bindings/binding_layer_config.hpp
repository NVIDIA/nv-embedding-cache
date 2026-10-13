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

#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "include/common.hpp"

namespace nve {

struct EmbedLayerConfig {
    int64_t logging_interval = -1;
    int64_t kernel_mode = 0;
    int64_t kernel_mode_value_1 = 0;
    int64_t kernel_mode_value_2 = 0;
    int64_t max_modify_size = 0; // means tables default
    int64_t default_row_index = -1; // Row returned for keys outside the table, negative disables the check
    std::vector<uint8_t> default_embedding = {}; // Raw row bytes for missing keys (Host/Hierarchical).
    bool disable_threadpool = false; // HostLayer only: run lookups on the caller thread.
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(
    EmbedLayerConfig,
    logging_interval, kernel_mode, kernel_mode_value_1, kernel_mode_value_2, max_modify_size,
    default_row_index, default_embedding, disable_threadpool)

// Partial patches in metadata format. Precedence: export < defaults < per_layer.
struct LayerConfigOverrides {
    nlohmann::json defaults = nlohmann::json::object();
    // Exact exported module_path; an empty path addresses a root embedding.
    std::map<std::string, nlohmann::json> per_layer;
};

namespace detail {

inline nlohmann::json layer_config_schema(const std::string& layer_type,
                                         const nlohmann::json& schema) {
    nlohmann::json result = {{"optimize_for_training", schema.at("optimize_for_training")}};
    std::vector<std::string> config_fields = {"logging_interval"};
    if (layer_type == "LinearUVM") {
        result["gpu_cache_size"] = schema.at("gpu_cache_size");
        config_fields.insert(config_fields.end(), {"kernel_mode", "kernel_mode_value_1",
                             "kernel_mode_value_2", "max_modify_size", "default_row_index"});
    } else if (layer_type == "Hierarchical") {
        result["gpu_cache_size"] = schema.at("gpu_cache_size");
        result["host_cache_size"] = schema.at("host_cache_size");
        config_fields.insert(config_fields.end(), {"max_modify_size", "default_embedding"});
    } else if (layer_type == "GPULayer") {
        config_fields.push_back("default_row_index");
    } else if (layer_type == "HostLayer") {
        config_fields.insert(config_fields.end(), {"default_embedding", "disable_threadpool"});
    } else {
        NVE_THROW_("Unsupported layer_type: '" + layer_type + "'");
    }
    for (const auto& field : config_fields) {
        result["config"][field] = schema.at("config").at(field);
    }
    return result;
}

inline void validate_layer_config_patch(const nlohmann::json& patch,
                                       const nlohmann::json& schema,
                                       const std::string& path) {
    NVE_CHECK_(patch.is_object(), path + " must be an object");
    for (const auto& [key, value] : patch.items()) {
        const std::string field = path + "." + key;
        auto it = schema.find(key);
        NVE_CHECK_(it != schema.end(), "Unsupported layer override: " + field);
        const auto& expected = *it;
        if (expected.is_object()) {
            validate_layer_config_patch(value, expected, field);
        } else if (expected.is_boolean()) {
            NVE_CHECK_(value.is_boolean(), field + " must be a boolean");
        } else if (expected.is_number_unsigned()) {
            NVE_CHECK_(value.is_number_integer() &&
                       (value.is_number_unsigned() || value.get<int64_t>() >= 0),
                       field + " must be a nonnegative integer");
            NVE_CHECK_(value.get<uint64_t>() <= std::numeric_limits<size_t>::max(),
                       field + " exceeds size_t range");
        } else if (expected.is_number_integer()) {
            NVE_CHECK_(value.is_number_integer(), field + " must be an integer");
            NVE_CHECK_(!value.is_number_unsigned() ||
                       value.get<uint64_t>() <= uint64_t{std::numeric_limits<int64_t>::max()},
                       field + " exceeds int64_t range");
        } else if (expected.is_array()) {
            // EmbedLayerConfig's array field holds raw row bytes.
            NVE_CHECK_(value.is_array(), field + " must be an array of bytes");
            for (const auto& byte : value) {
                NVE_CHECK_(byte.is_number_integer() &&
                           (byte.is_number_unsigned() || byte.get<int64_t>() >= 0) &&
                           byte.get<uint64_t>() <= 255,
                           field + " must contain integers in [0, 255]");
            }
        } else {
            NVE_THROW_("Unsupported layer config type: " + field);
        }
    }
}

inline void merge_layer_config_patch(nlohmann::json& entry, const nlohmann::json& patch) {
    for (const auto& [key, value] : patch.items()) {
        if (key == "config") {
            if (!entry.contains(key)) entry[key] = nlohmann::json::object();
            NVE_CHECK_(entry[key].is_object(),
                       "Layer '" + entry.at("module_path").get<std::string>() +
                       "' config must be an object");
            entry[key].update(value);
        } else {
            entry[key] = value;
        }
    }
}

}  // namespace detail

// Shared by the C++ and Python loaders; validates all patches before loading resources.
// Copies metadata so patches never change the export or the caller's objects.
inline nlohmann::json apply_layer_config_overrides(
        const nlohmann::json& layers, const LayerConfigOverrides& overrides) {
    const nlohmann::json schema = {
        {"gpu_cache_size", size_t{0}},
        {"host_cache_size", size_t{0}},
        {"optimize_for_training", false},
        {"config", EmbedLayerConfig{}}
    };
    detail::validate_layer_config_patch(overrides.defaults, schema, "overrides.defaults");
    NVE_CHECK_(layers.is_array(), "Layer metadata must be an array");
    std::set<std::string> module_paths;
    for (const auto& entry : layers) {
        module_paths.insert(entry.at("module_path").get<std::string>());
    }
    for (const auto& [module_path, patch] : overrides.per_layer) {
        NVE_CHECK_(module_paths.count(module_path) > 0,
                   "Unknown layer override module_path: '" + module_path + "'");
        detail::validate_layer_config_patch(patch, schema,
                                           "overrides.per_layer['" + module_path + "']");
    }

    for (const auto& entry : layers) {
        const auto module_path = entry.at("module_path").get<std::string>();
        auto it = overrides.per_layer.find(module_path);
        if (overrides.defaults.empty() && it == overrides.per_layer.end()) continue;
        const auto layer_type = entry.at("layer_type").get<std::string>();
        const auto layer_schema = detail::layer_config_schema(layer_type, schema);
        const auto target = " for layer '" + module_path + "' (" + layer_type + ")";
        detail::validate_layer_config_patch(overrides.defaults, layer_schema,
                                           "overrides.defaults" + target);
        if (it != overrides.per_layer.end()) {
            detail::validate_layer_config_patch(it->second, layer_schema,
                                               "overrides.per_layer['" + module_path + "']" + target);
        }
    }

    auto result = layers;
    for (auto& entry : result) {
        detail::merge_layer_config_patch(entry, overrides.defaults);
        auto it = overrides.per_layer.find(entry.at("module_path").get<std::string>());
        if (it != overrides.per_layer.end()) {
            detail::merge_layer_config_patch(entry, it->second);
        }
    }
    return result;
}

}  // namespace nve
