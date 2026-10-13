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

#include "python/pynve/bindings/binding_layer_config.hpp"

namespace nve {
namespace {

using json = nlohmann::json;

json layer(const std::string& module_path, const std::string& layer_type = "Hierarchical") {
    return {{"module_path", module_path}, {"layer_type", layer_type}, {"gpu_cache_size", 1024},
            {"host_cache_size", 2048}, {"optimize_for_training", true},
            {"config", {{"logging_interval", 10}, {"kernel_mode", 2},
                        {"default_embedding", {1, 2}}}}};
}

TEST(LayerConfigOverridesTest, PerLayerPrecedenceAndPreservation) {
    const auto metadata = json::array({layer("user"), layer("item"), layer("other")});
    const auto original = metadata;
    LayerConfigOverrides overrides;
    overrides.defaults = {{"gpu_cache_size", 4096},
                          {"config", {{"logging_interval", 100}}}};
    overrides.per_layer = {
        {"user", {{"gpu_cache_size", 0}, {"optimize_for_training", false},
                  {"config", {{"max_modify_size", 0}, {"default_embedding", json::array()}}}}},
        {"item", {{"host_cache_size", 8192},
                  {"config", {{"logging_interval", -1}}}}}
    };
    const auto original_defaults = overrides.defaults;
    const auto original_per_layer = overrides.per_layer;
    const auto result = apply_layer_config_overrides(metadata, overrides);

    EXPECT_EQ(result[0]["gpu_cache_size"], 0);
    EXPECT_EQ(result[0]["optimize_for_training"], false);
    EXPECT_EQ(result[0]["config"]["logging_interval"], 100);
    EXPECT_EQ(result[0]["config"]["kernel_mode"], 2);
    EXPECT_EQ(result[0]["config"]["max_modify_size"], 0);
    EXPECT_EQ(result[0]["config"]["default_embedding"], json::array());
    EXPECT_EQ(result[1]["gpu_cache_size"], 4096);
    EXPECT_EQ(result[1]["host_cache_size"], 8192);
    EXPECT_EQ(result[1]["config"]["logging_interval"], -1);
    EXPECT_EQ(result[1]["config"]["default_embedding"], json::array({1, 2}));
    EXPECT_EQ(result[2]["gpu_cache_size"], 4096);
    EXPECT_EQ(result[2]["host_cache_size"], 2048);
    EXPECT_EQ(result[2]["config"]["logging_interval"], 100);
    EXPECT_EQ(metadata, original);
    EXPECT_EQ(overrides.defaults, original_defaults);
    EXPECT_EQ(overrides.per_layer, original_per_layer);
}

TEST(LayerConfigOverridesTest, LegacyRootLayerWithoutConfig) {
    auto root = layer("", "LinearUVM");
    root.erase("config");
    LayerConfigOverrides overrides;
    overrides.per_layer[""] = {{"config", {{"default_row_index", 3}}}};
    const auto result = apply_layer_config_overrides(json::array({root}), overrides);
    const auto cfg = result[0]["config"].get<EmbedLayerConfig>();
    EXPECT_EQ(cfg.default_row_index, 3);
    EXPECT_EQ(cfg.logging_interval, EmbedLayerConfig{}.logging_interval);
    EXPECT_EQ(cfg.kernel_mode, EmbedLayerConfig{}.kernel_mode);
    EXPECT_FALSE(cfg.disable_threadpool);
}

TEST(LayerConfigOverridesTest, EmptyOverridesPreserveMetadata) {
    const auto metadata = json::array({layer("user"), layer("item")});
    EXPECT_EQ(apply_layer_config_overrides(metadata, {}), metadata);
}

TEST(LayerConfigOverridesTest, RejectNonObjectMetadataConfig) {
    const std::vector<json> invalid_configs = {nullptr, json::array(), "invalid", 1, true};
    const json patch = {{"config", {{"logging_interval", 100}}}};
    for (const auto& config : invalid_configs) {
        SCOPED_TRACE(config.dump());
        auto entry = layer("user");
        entry["config"] = config;
        const auto metadata = json::array({entry});
        for (const bool use_defaults : {true, false}) {
            SCOPED_TRACE(use_defaults);
            LayerConfigOverrides overrides;
            if (use_defaults) {
                overrides.defaults = patch;
            } else {
                overrides.per_layer["user"] = patch;
            }
            try {
                apply_layer_config_overrides(metadata, overrides);
                FAIL() << "Expected malformed config to be rejected";
            } catch (const nve::Exception& error) {
                EXPECT_NE(std::string(error.what()).find("Layer 'user' config must be an object"),
                          std::string::npos) << error.what();
            }
            EXPECT_EQ(metadata[0]["config"], config);
        }
    }
}

TEST(LayerConfigOverridesTest, RejectUnknownPaths) {
    LayerConfigOverrides overrides;
    overrides.per_layer["typo"] = {{"gpu_cache_size", 1024}};
    EXPECT_THROW(apply_layer_config_overrides(json::array({layer("user")}), overrides),
                 nve::Exception);
}

TEST(LayerConfigOverridesTest, RejectInvalidPatches) {
    const std::vector<json> patches = {
        nullptr, json::array(),
        {{"gpu_cache_sze", 1024}}, {{"dtype", "float16"}},
        {{"id", 2}}, {{"layer_type", "HostLayer"}}, {{"storage_ref", "mb-1"}},
        {{"embedding_size", 16}}, {{"num_embeddings", 10}},
        {{"gpu_cache_size", nullptr}}, {{"gpu_cache_size", -1}},
        {{"gpu_cache_size", 1.5}}, {{"gpu_cache_size", true}},
        {{"host_cache_size", "2048"}}, {{"optimize_for_training", 0}},
        {{"config", nullptr}}, {{"config", json::array()}},
        {{"config", {{"kernel_mod", 1}}}}, {{"config", {{"kernel_mode", 1.5}}}},
        {{"config", {{"logging_interval", false}}}},
        {{"config", {{"disable_threadpool", 1}}}},
        {{"config", {{"logging_interval", std::numeric_limits<uint64_t>::max()}}}},
        {{"config", {{"default_embedding", nullptr}}}},
        {{"config", {{"default_embedding", {-1}}}}},
        {{"config", {{"default_embedding", {256}}}}},
        {{"config", {{"default_embedding", {1.5}}}}},
        {{"config", {{"default_embedding", {true}}}}}
    };
    const auto metadata = json::array({layer("user")});
    for (const auto& patch : patches) {
        SCOPED_TRACE(patch.dump());
        LayerConfigOverrides overrides;
        overrides.defaults = patch;
        EXPECT_THROW(apply_layer_config_overrides(metadata, overrides), nve::Exception);
        overrides.defaults = json::object();
        overrides.per_layer["user"] = patch;
        EXPECT_THROW(apply_layer_config_overrides(metadata, overrides), nve::Exception);
    }
}

TEST(LayerConfigOverridesTest, KnobsAreValidatedAgainstLayerType) {
    const std::vector<std::string> layer_types = {"LinearUVM", "Hierarchical", "GPULayer", "HostLayer"};
    const std::map<std::string, std::set<std::string>> supported_types = {
        {"gpu_cache_size", {"LinearUVM", "Hierarchical"}},
        {"host_cache_size", {"Hierarchical"}},
        {"optimize_for_training", {"LinearUVM", "Hierarchical", "GPULayer", "HostLayer"}},
        {"logging_interval", {"LinearUVM", "Hierarchical", "GPULayer", "HostLayer"}},
        {"kernel_mode", {"LinearUVM"}},
        {"kernel_mode_value_1", {"LinearUVM"}},
        {"kernel_mode_value_2", {"LinearUVM"}},
        {"max_modify_size", {"LinearUVM", "Hierarchical"}},
        {"default_row_index", {"LinearUVM", "GPULayer"}},
        {"default_embedding", {"Hierarchical", "HostLayer"}},
        {"disable_threadpool", {"HostLayer"}}
    };
    json patches = EmbedLayerConfig{};
    patches["gpu_cache_size"] = 0;
    patches["host_cache_size"] = 0;
    patches["optimize_for_training"] = false;
    for (const auto& layer_type : layer_types) {
        const auto metadata = json::array({layer("user", layer_type)});
        for (const auto& [field, value] : patches.items()) {
            const bool top_level = field == "gpu_cache_size" || field == "host_cache_size" ||
                                   field == "optimize_for_training";
            const json patch = top_level ? json{{field, value}} : json{{"config", {{field, value}}}};
            for (const bool use_defaults : {true, false}) {
                SCOPED_TRACE(layer_type + ": " + patch.dump());
                SCOPED_TRACE(use_defaults);
                LayerConfigOverrides overrides;
                if (use_defaults) overrides.defaults = patch;
                else overrides.per_layer["user"] = patch;
                if (supported_types.at(field).count(layer_type)) {
                    const auto result = apply_layer_config_overrides(metadata, overrides);
                    EXPECT_EQ(top_level ? result[0][field] : result[0]["config"][field], value);
                } else {
                    try {
                        apply_layer_config_overrides(metadata, overrides);
                        FAIL() << "Expected unsupported knob to be rejected";
                    } catch (const nve::Exception& error) {
                        const std::string message = error.what();
                        EXPECT_NE(message.find("Unsupported layer override"), std::string::npos);
                        EXPECT_NE(message.find("layer 'user' (" + layer_type + ")"), std::string::npos);
                        EXPECT_NE(message.find(field), std::string::npos);
                    }
                }
            }
        }
    }
}

TEST(LayerConfigOverridesTest, MixedLayerDefaultsRequireSupportFromEveryLayer) {
    const auto metadata = json::array({layer("cache"), layer("host", "HostLayer")});
    LayerConfigOverrides overrides;
    overrides.defaults = {{"host_cache_size", 8192}};
    EXPECT_THROW(apply_layer_config_overrides(metadata, overrides), nve::Exception);
    overrides.defaults = {{"config", {{"logging_interval", 100}}}};
    overrides.per_layer["cache"] = {{"host_cache_size", 8192}};
    overrides.per_layer["host"] = {{"config", {{"disable_threadpool", true}}}};
    const auto result = apply_layer_config_overrides(metadata, overrides);
    EXPECT_EQ(result[0]["host_cache_size"], 8192);
    EXPECT_EQ(result[1]["config"]["disable_threadpool"], true);
    EXPECT_EQ(result[0]["config"]["logging_interval"], 100);
    EXPECT_EQ(result[1]["config"]["logging_interval"], 100);
}

TEST(LayerConfigOverridesTest, HostThreadpoolOverridePrecedence) {
    const auto metadata = json::array({layer("user", "HostLayer"), layer("item", "HostLayer")});
    LayerConfigOverrides overrides;
    overrides.defaults = {{"config", {{"disable_threadpool", true}}}};
    overrides.per_layer["item"] = {{"config", {{"disable_threadpool", false}}}};
    const auto result = apply_layer_config_overrides(metadata, overrides);
    EXPECT_TRUE(result[0]["config"].get<EmbedLayerConfig>().disable_threadpool);
    EXPECT_FALSE(result[1]["config"].get<EmbedLayerConfig>().disable_threadpool);
}

}  // namespace
}  // namespace nve
