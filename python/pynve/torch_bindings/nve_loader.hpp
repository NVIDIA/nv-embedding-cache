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

// C++ loader for NVE exported models.
//
// Reads metadata.json + per-module .nve weight files from an export directory,
// creates LinearUVMEmbedding layers, loads weights, and registers them in
// the NVELayerRegistry so torch custom ops can find them by layer_id.

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <algorithm>

#include <nlohmann/json.hpp>

#include "python/pynve/bindings/binding_layers.hpp"
#include "python/pynve/torch_bindings/nve_registry.hpp"
#include "include/execution_context.hpp"
#include "include/memblock.hpp"
#include "include/serialization.hpp"
#include "include/table_utils.hpp"

namespace nve {

struct LoadedLayer {
    int64_t layer_id;
    std::string module_path;
    std::shared_ptr<NVEmbedBinding<int64_t>> binding;
};

inline DataType_t parse_dtype(const std::string& s) {
    if (s.find("float32") != std::string::npos) return DataType_t::Float32;
    if (s.find("float16") != std::string::npos) return DataType_t::Float16;
    throw std::runtime_error("nve_loader: unsupported dtype: " + s);
}

// Build a MemBlock from an exported metadata.json memblock_type tag.
// device_resident=true makes Linear/User allocate on device_index instead
// of host (the only mode that can satisfy a NoCache layer).
inline std::shared_ptr<MemBlock> create_memblock(
    const std::string& memblock_type,
    size_t embedding_size, size_t num_embeddings,
    DataType_t dtype, int device_index,
    bool device_resident = false)
{
    if (memblock_type.find("Managed") != std::string::npos) {
        return std::make_shared<ManagedMemBlock>(
            embedding_size, num_embeddings, dtype,
            std::vector<int>{device_index});
    }
    if (memblock_type.find("Linear") != std::string::npos) {
        const int linear_device_id = device_resident ? device_index : -1;
        return std::make_shared<LinearMemBlock>(
            embedding_size, num_embeddings, dtype, linear_device_id);
    }
    if (memblock_type.find("User") != std::string::npos) {
        // "User" is an export-only marker for a caller-provided buffer. Only
        // the device-resident path can fabricate replacement storage; 
        NVE_CHECK_(device_resident,
                   "nve_loader: 'User' memblock_type can only be loaded for "
                   "device-resident layers (e.g. NoCache); got host-backed "
                   "request for memblock_type=" + memblock_type);
        return std::make_shared<LinearMemBlock>(
            embedding_size, num_embeddings, dtype, /*device_id=*/device_index);
    }
    if (memblock_type.find("NVL") != std::string::npos) {
        return std::make_shared<NVLMemBlock>(
            embedding_size, num_embeddings, dtype,
            std::vector<int>{device_index});
    }
    NVE_CHECK_(false, "nve_loader: unsupported memblock_type: " + memblock_type);
    return nullptr;
}

/// RAII loader for NVE exported models.
///
/// The constructor reads metadata.json, creates embedding layers,
/// loads weights from weights/<module_path>.nve, and registers each
/// layer in the NVELayerRegistry.
///
/// The destructor unregisters all layers from the registry.
///
/// Usage:
///   nve::LayerDirectory dir("save_dir/", device_index);
///   // layers are registered — run the AOT model
///   auto layer = dir.get_layer(0);  // lookup by layer_id
///   // destructor unregisters when dir goes out of scope
class LayerDirectory {
public:
    LayerDirectory(const std::string& save_dir,
                   int device_index = 0,
                   size_t gpu_cache_size_override = 0)
        : save_dir_(save_dir)
    {
        std::ifstream meta_file(save_dir + "/metadata.json");
        NVE_CHECK_(meta_file.is_open(),
                  "nve_loader: cannot open " + save_dir + "/metadata.json");
        auto metadata = nlohmann::json::parse(meta_file);

        EmbedLayerConfig cfg{};

        for (const auto& entry : metadata) {
            int64_t layer_id       = entry["id"];
            std::string module_path = entry["module_path"];
            size_t num_emb         = entry["num_embeddings"];
            size_t emb_size        = entry["embedding_size"];
            std::string dtype_str  = entry["dtype"];
            std::string cache_type = entry["cache_type"];
            size_t gpu_cache       = gpu_cache_size_override > 0
                                         ? gpu_cache_size_override
                                         : entry["gpu_cache_size"].get<size_t>();
            bool training          = entry["optimize_for_training"];
            DataType_t nve_dtype   = parse_dtype(dtype_str);

            std::shared_ptr<NVEmbedBinding<int64_t>> layer;

            if (cache_type == "LinearUVM" || cache_type == "NoCache") {
                std::string mb_type = entry.value("memblock_type", "MemBlockType.Managed");
                const bool device_resident = (cache_type == "NoCache");
                auto mem_block = create_memblock(
                    mb_type, emb_size, num_emb, nve_dtype, device_index, device_resident);

                std::string weight_name = module_path;
                std::replace(weight_name.begin(), weight_name.end(), '.', '_');
                std::string weight_path = save_dir + "/weights/" + weight_name + ".nve";
                InputFileStreamWrapper weight_stream(weight_path);

                if (cache_type == "LinearUVM") {
                    auto uvm_layer = std::make_shared<LinearUVMEmbedding<int64_t>>(
                        emb_size, num_emb, nve_dtype, mem_block,
                        gpu_cache, training, device_index, cfg);
                    uvm_layer->load_tensor_from_stream(weight_stream, layer_id);
                    layer = uvm_layer;
                } else {  // NoCache
                    auto gpu_layer = std::make_shared<GPUEmbedding<int64_t>>(
                        emb_size, num_emb, nve_dtype, mem_block,
                        device_index, cfg);
                    gpu_layer->load_tensor_from_stream(weight_stream, layer_id);
                    layer = gpu_layer;
                }
            } else if (cache_type == "Hierarchical") {
                auto ps_it = entry.find("remote_ps_config");
                NVE_CHECK_(ps_it != entry.end(),
                           "nve_loader: Hierarchical layer is missing 'remote_ps_config'");
                const nlohmann::json& ps_cfg = *ps_it;
                NVE_CHECK_(ps_cfg.value("remote_ps_type", "") == "plugin",
                           "nve_loader: only remote_ps_type=='plugin' is supported");

                host_table_ptr_t host_remote = create_table_from_plugin(
                    ps_cfg.at("plugin_name").get<std::string>(),
                    ps_cfg.value("factory_config", nlohmann::json::object()),
                    ps_cfg.value("table_config",   nlohmann::json::object()));

                // Validate that the produced PS dtype matches the layer's dtype.
                // The plugin's factory honors table_config["value_dtype"] when producing,
                // so any mismatch here is an export-time bug (or a hand-edited
                // metadata.json) — fail loudly before silently corrupting data.
                const DataType_t ps_dtype = host_remote->config().value_dtype;
                NVE_CHECK_(ps_dtype == nve_dtype,
                           "nve_loader: layer '" + module_path +
                           "' has data_type=" + dtype_str +
                           " but remote PS data_type=" + to_string(ps_dtype) +
                           " — they must match");

                // Optional pre-populated PS data (npy / dyn files written by export).
                auto data_it = entry.find("remote_ps_data");
                if (data_it != entry.end() && data_it->is_object()) {
                    const uint64_t row_bytes =
                        ps_cfg.at("row_elements").get<uint64_t>() *
                        static_cast<uint64_t>(dtype_size(ps_dtype));
                    // One-shot ctx for the populate. Async insert ops run on its
                    // stream — wait() before the ctx is destroyed or kernels
                    // outlive the context (UB).
                    auto ctx = host_remote->create_execution_context(0, 0, nullptr, nullptr);
                    insert_keys_from_filepath(
                        host_remote, ctx,
                        data_it->at("keys").get<std::string>(),
                        data_it->at("values").get<std::string>(),
                        row_bytes,
                        /*batch_size=*/1ull << 20);
                    ctx->wait();
                }

                size_t host_cache = entry.value("host_cache_size", size_t{0});
                const uint64_t ps_num_rows = ps_cfg.value("num_rows", uint64_t{0});
                layer = std::make_shared<HierarchicalEmbedding<int64_t>>(
                    emb_size, nve_dtype, gpu_cache, host_cache,
                    host_remote, ps_num_rows,
                    /*use_private_stream=*/training,
                    device_index, cfg);
            } else {
                NVE_CHECK_(false,
                           "nve_loader: unsupported cache_type: " + cache_type);
            }

            // Register in the global registry
            NVELayerRegistry::instance().register_binding(layer_id, layer);
            layers_[layer_id] = {layer_id, std::move(module_path), layer};
        }
    }

    ~LayerDirectory() {
        for (const auto& [id, _] : layers_) {
            NVELayerRegistry::instance().unregister_binding(id);
        }
    }

    // Non-copyable, movable
    LayerDirectory(const LayerDirectory&) = delete;
    LayerDirectory& operator=(const LayerDirectory&) = delete;
    LayerDirectory(LayerDirectory&&) = default;
    LayerDirectory& operator=(LayerDirectory&&) = default;

    const LoadedLayer& get_layer(int64_t layer_id) const {
        auto it = layers_.find(layer_id);
        NVE_CHECK_(it != layers_.end(),
                  "nve_loader: no layer with id=" + std::to_string(layer_id));
        return it->second;
    }

    size_t size() const { return layers_.size(); }
private:
    std::string save_dir_;
    std::map<int64_t, LoadedLayer> layers_;
};

} // namespace nve