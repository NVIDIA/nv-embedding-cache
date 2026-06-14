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
#include <set>
#include <memory>
#include <algorithm>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

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
    // Per-layer identity marker. The LayerDirectory owns it; the AOTI container
    // holds only a non-owning (user_managed) handle, and the NVELayerRegistry is
    // keyed by its data_ptr(). Must outlive the loader.
    at::Tensor marker;
};

// Fully-qualified name of a layer's marker buffer in the exported graph.
inline std::string marker_fqn(const std::string& module_path) {
    return module_path.empty() ? std::string("marker_tensor")
                               : module_path + ".marker_tensor";
}

inline DataType_t parse_dtype(const std::string& s) {
    if (s.find("float32") != std::string::npos) return DataType_t::Float32;
    if (s.find("float16") != std::string::npos) return DataType_t::Float16;
    throw std::runtime_error("nve_loader: unsupported dtype: " + s);
}

// Build the per-layer EmbedLayerConfig from a metadata layer entry. Uses the
// from_json defined alongside the struct (single source of truth). A missing
// "config" block (legacy v1 exports) yields the struct defaults.
inline EmbedLayerConfig parse_embed_config(const nlohmann::json& entry) {
    auto it = entry.find("config");
    if (it == entry.end()) return EmbedLayerConfig{};
    return it->get<EmbedLayerConfig>();
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
    if (memblock_type.find("Host") != std::string::npos) {
        // Owning host block backed by plain malloc — no CUDA, driverless-safe.
        return std::make_shared<HostMemBlock>(embedding_size, num_embeddings, dtype);
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

static constexpr int METADATA_SCHEMA_VERSION = 2;

// Resource id as used in metadata.json (e.g. "mb-<id>", "ps-<id>").
using resource_id_type = std::string;

// Load-time device topology: maps a resource id to the concrete devices it
// should occupy. For an NVL memblock this is the full span of GPUs; 
using TopologyMap = std::map<resource_id_type, std::vector<int>>;

// Registry of already-constructed storage objects, shared across LayerDirectory
// instances to preserve shared storage across models.
//
// Keyed by the resource id string from metadata.json (e.g. "mb-<id>",
// "ps-<id>").  LayerDirectory mutates the registry in-place on miss and reuses
// existing objects on hit.  The remap table redirects a key found in the file
// to a different key in the registry — useful when two exports from different
// processes should share the same storage.
//
// ResourceDirectory is NOT thread-safe; do not share across concurrent
// LayerDirectory constructions.
class ResourceDirectory {
public:
    // Apply remap and look up a memblock. Returns nullptr on miss.
    std::shared_ptr<MemBlock> find_memblock(const std::string& key) const {
        auto it = memblocks_.find(remap(key));
        return it != memblocks_.end() ? it->second : nullptr;
    }

    // Apply remap and look up a PS table. Returns nullptr on miss.
    host_table_ptr_t find_ps(const std::string& key) const {
        auto it = ps_tables_.find(remap(key));
        return it != ps_tables_.end() ? it->second : nullptr;
    }

    void insert_memblock(const std::string& key, std::shared_ptr<MemBlock> mb) {
        memblocks_[remap(key)] = std::move(mb);
    }

    void insert_ps(const std::string& key, host_table_ptr_t ps) {
        ps_tables_[remap(key)] = std::move(ps);
    }

    void add_remap(const std::string& from, const std::string& to) {
        remap_[from] = to;
    }

    std::string remap(const std::string& key) const {
        auto it = remap_.find(key);
        return it != remap_.end() ? it->second : key;
    }

private:
    std::unordered_map<std::string, std::shared_ptr<MemBlock>> memblocks_;
    std::unordered_map<std::string, host_table_ptr_t>          ps_tables_;
    std::unordered_map<std::string, std::string>               remap_;
};

// RAII loader for NVE exported (AOTInductor) models.
//
// The constructor reads metadata.json, creates embedding layers, loads weights
// from weights/<module_path>.nve, allocates a per-layer marker tensor, points
// the AOTI container's marker constant at it (user_managed, both buffers), and
// registers each layer in the NVELayerRegistry keyed by the marker's data_ptr.
//
// The destructor unregisters all layers from the registry. The LayerDirectory
// owns the marker tensors and MUST outlive the loader (the container holds only
// non-owning handles).
//
// Usage:
//   torch::inductor::AOTIModelPackageLoader loader(save_dir + "/model.pt2");
//   nve::LayerDirectory dir(save_dir, loader, device_index);
//   auto outputs = loader.run({keys});
//   // destructor unregisters when dir goes out of scope
class LayerDirectory {
public:
    LayerDirectory(const std::string& save_dir,
                   torch::inductor::AOTIModelPackageLoader& loader,
                   int device_index,
                   std::shared_ptr<ResourceDirectory> resource_dir,
                   size_t gpu_cache_size_override = 0,
                   const TopologyMap& topology = {})
        : save_dir_(save_dir), resource_dir_(std::move(resource_dir))
    {
        std::ifstream meta_file(save_dir + "/metadata.json");
        NVE_CHECK_(meta_file.is_open(),
                  "nve_loader: cannot open " + save_dir + "/metadata.json");
        auto raw = nlohmann::json::parse(meta_file);

        // The marker scheme requires each layer's marker to be a runtime-updatable
        // constant. load_constants silently ignores unknown names, so if a marker
        // was baked inline (e.g. exported without runtime constant folding) the
        // user_managed write below would no-op and only surface as a confusing
        // "no binding registered" error at run time. Validate per-layer below
        // against this set.
        const auto fqn_vec = loader.get_constant_fqns();
        const std::set<std::string> pkg_fqns(fqn_vec.begin(), fqn_vec.end());

        // Detect schema version: array = v1 legacy; object = v2.
        if (raw.is_array()) {
            load_layers_v1(raw, loader, pkg_fqns, device_index,
                            gpu_cache_size_override);
        } else {
            const int version = raw.value("version", 1);
            NVE_CHECK_(version <= METADATA_SCHEMA_VERSION,
                       "nve_loader: metadata.json was exported with schema version " +
                       std::to_string(version) + " but this NVE runtime only supports "
                       "up to version " + std::to_string(METADATA_SCHEMA_VERSION) +
                       ". Upgrade the NVE runtime.");
            load_layers_v2(raw, loader, pkg_fqns, device_index,
                            gpu_cache_size_override, topology);
        }
    }

    ~LayerDirectory() {
        for (const auto& [id, ll] : layers_) {
            NVELayerRegistry::instance().unregister_binding(ll.marker.data_ptr());
        }
    }

    // Re-point every layer's marker constant at its marker tensor. Call after a
    // host-side swap_constant_buffer that was preceded by an inactive-buffer
    // rebuild (e.g. dense-weight hot-swap), which re-copies the marker as a
    // fresh non-user-managed tensor and breaks dispatch. No re-registration is
    // needed (the marker tensor / its data_ptr is unchanged).
    void rebind_markers(torch::inductor::AOTIModelPackageLoader& loader) {
        for (auto& [id, ll] : layers_) {
            write_marker(loader, marker_fqn(ll.module_path), ll.marker);
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
    // Point the container's marker constant at `marker` (no copy) in BOTH
    // buffers, so the dispatch ptr survives a bare swap_constant_buffer.
    static void write_marker(torch::inductor::AOTIModelPackageLoader& loader,
                             const std::string& fqn, at::Tensor marker) {
        std::unordered_map<std::string, at::Tensor> mm{{fqn, marker}};
        loader.load_constants(mm, /*use_inactive=*/false,
                              /*check_full_update=*/false, /*user_managed=*/true);
        loader.load_constants(mm, /*use_inactive=*/true,
                              /*check_full_update=*/false, /*user_managed=*/true);
    }

    void register_layer(std::shared_ptr<NVEmbedBinding<int64_t>> layer,
                         int64_t layer_id, std::string module_path,
                         torch::inductor::AOTIModelPackageLoader& loader,
                         const std::set<std::string>& pkg_fqns,
                         int device_index) {
        const std::string fqn = marker_fqn(module_path);
        NVE_CHECK_(pkg_fqns.count(fqn) > 0,
                   "nve_loader: marker constant '" + fqn + "' is not an "
                   "updatable constant in the AOTI package. It was likely "
                   "baked inline; re-export with export_aot (it sets "
                   "aot_inductor.use_runtime_constant_folding=True).");
        auto opts = at::TensorOptions().dtype(at::kLong);
        opts = (device_index >= 0) ? opts.device(at::kCUDA, device_index)
                                   : opts.device(at::kCPU);
        at::Tensor marker = at::tensor(std::vector<int64_t>{layer_id}, opts);
        write_marker(loader, fqn, marker);
        NVELayerRegistry::instance().register_binding(marker.data_ptr(), layer);
        layers_[layer_id] = {layer_id, std::move(module_path),
                             std::move(layer), marker};
    }

    // Build a PS from the resources section, consulting/updating resource_dir_.
    host_table_ptr_t get_or_build_ps(
            const std::string& key,
            const nlohmann::json& ps_resources,
            const std::string& layer_name,
            DataType_t nve_dtype,
            const std::string& dtype_str) {
        const std::string eff = resource_dir_->remap(key);
        auto existing = resource_dir_->find_ps(eff);
        if (existing) return existing;

        auto ps_it = ps_resources.find(eff);
        NVE_CHECK_(ps_it != ps_resources.end(),
                   "nve_loader: storage_ref '" + eff +
                   "' not found in resources.remote_ps");
        const nlohmann::json& ps_cfg = *ps_it;
        NVE_CHECK_(ps_cfg.value("remote_ps_type", "") == "plugin",
                   "nve_loader: only remote_ps_type=='plugin' is supported");

        host_table_ptr_t host_remote = create_table_from_plugin(
            ps_cfg.at("plugin_name").get<std::string>(),
            ps_cfg.value("factory_config", nlohmann::json::object()),
            ps_cfg.value("table_config",   nlohmann::json::object()));

        const DataType_t ps_dtype = host_remote->config().value_dtype;
        NVE_CHECK_(ps_dtype == nve_dtype,
                   "nve_loader: layer '" + layer_name +
                   "' has data_type=" + dtype_str +
                   " but remote PS data_type=" + to_string(ps_dtype) +
                   " — they must match");

        auto data_it = ps_cfg.find("remote_ps_data");
        if (data_it != ps_cfg.end() && data_it->is_object()) {
            const uint64_t row_bytes =
                ps_cfg.at("row_elements").get<uint64_t>() *
                static_cast<uint64_t>(dtype_size(ps_dtype));
            auto ctx = host_remote->create_execution_context(0, 0, nullptr, nullptr);
            insert_keys_from_filepath(
                host_remote, ctx,
                data_it->at("keys").get<std::string>(),
                data_it->at("values").get<std::string>(),
                row_bytes, /*batch_size=*/1ull << 20);
            ctx->wait();
        }

        resource_dir_->insert_ps(eff, host_remote);
        return host_remote;
    }

    // Build a memblock from the resources section, consulting/updating resource_dir_.
    // Returns {memblock, needs_weight_load}. needs_weight_load is false on a
    // registry hit — weights are already present in the shared memblock.
    std::pair<std::shared_ptr<MemBlock>, bool> get_or_build_memblock(
            const std::string& key,
            const nlohmann::json& mb_resources,
            size_t emb_size, size_t num_emb,
            DataType_t nve_dtype, int device_index,
            bool device_resident,
            const TopologyMap& topology) {
        const std::string eff = resource_dir_->remap(key);
        auto existing = resource_dir_->find_memblock(eff);
        if (existing) return {existing, false};

        auto mb_it = mb_resources.find(eff);
        NVE_CHECK_(mb_it != mb_resources.end(),
                   "nve_loader: storage_ref '" + eff +
                   "' not found in resources.memblocks");
        const nlohmann::json& mb_cfg = *mb_it;

        // Geometry assertion
        NVE_CHECK_(mb_cfg.at("row_elements").get<size_t>() == emb_size,
                   "nve_loader: resource '" + eff + "' row_elements mismatch");
        NVE_CHECK_(mb_cfg.at("num_rows").get<size_t>() == num_emb,
                   "nve_loader: resource '" + eff + "' num_rows mismatch");

        // Topology override for this resource, if the caller supplied one.
        std::vector<int> override;
        auto topo_it = topology.find(eff);
        if (topo_it != topology.end()) override = topo_it->second;

        const std::string mb_type = mb_cfg.value("type", "MemBlockType.Managed");
        std::shared_ptr<MemBlock> mb;
        if (mb_type.find("NVL") != std::string::npos) {
            // Not in topology → span all devices [0, deviceCount-1].
            auto gpu_ids = resolve_memblock_devices(MemBlockType::NVL, /*def_index=*/0, override);
            mb = std::make_shared<NVLMemBlock>(emb_size, num_emb, nve_dtype, gpu_ids);
        } else {
            // Not in topology → the supplied device_index.
            int dev = override.empty() ? device_index : override.front();
            mb = create_memblock(mb_type, emb_size, num_emb, nve_dtype,
                                 dev, device_resident);
        }

        resource_dir_->insert_memblock(eff, mb);
        return {mb, true};
    }

    void load_layers_v1(const nlohmann::json& metadata,
                         torch::inductor::AOTIModelPackageLoader& loader,
                         const std::set<std::string>& pkg_fqns,
                         int device_index,
                         size_t gpu_cache_size_override) {
        std::set<std::string> weights_loaded;
        for (const auto& entry : metadata) {
            int64_t layer_id        = entry["id"];
            std::string module_path = entry["module_path"];
            size_t num_emb          = entry["num_embeddings"];
            size_t emb_size         = entry["embedding_size"];
            std::string dtype_str   = entry["dtype"];
            std::string layer_type  = entry["layer_type"];
            size_t gpu_cache        = gpu_cache_size_override > 0
                                          ? gpu_cache_size_override
                                          : entry["gpu_cache_size"].get<size_t>();
            bool training           = entry["optimize_for_training"];
            DataType_t nve_dtype    = parse_dtype(dtype_str);
            EmbedLayerConfig cfg    = parse_embed_config(entry);

            std::shared_ptr<NVEmbedBinding<int64_t>> layer;

            if (layer_type == "LinearUVM" || layer_type == "GPULayer" ||
                layer_type == "HostLayer") {
                std::string mb_type = entry.value("memblock_type", "MemBlockType.Managed");
                const bool device_resident = (layer_type == "GPULayer");
                if (layer_type == "HostLayer") mb_type = "MemBlockType.Host";
                auto mem_block = create_memblock(
                    mb_type, emb_size, num_emb, nve_dtype, device_index, device_resident);

                std::string weight_name = module_path;
                std::replace(weight_name.begin(), weight_name.end(), '.', '_');
                std::string weight_path = save_dir_ + "/weights/" + weight_name + ".nve";
                InputFileStreamWrapper weight_stream(weight_path);

                if (layer_type == "LinearUVM") {
                    auto l = std::make_shared<LinearUVMEmbedding<int64_t>>(
                        emb_size, num_emb, nve_dtype, mem_block,
                        gpu_cache, training, device_index, cfg);
                    l->load_tensor_from_stream(weight_stream, layer_id);
                    layer = l;
                } else if (layer_type == "GPULayer") {
                    auto l = std::make_shared<GPUEmbedding<int64_t>>(
                        emb_size, num_emb, nve_dtype, mem_block, device_index, cfg);
                    l->load_tensor_from_stream(weight_stream, layer_id);
                    layer = l;
                } else {
                    auto l = std::make_shared<HostEmbedding<int64_t>>(
                        emb_size, num_emb, nve_dtype, mem_block, device_index, cfg);
                    l->load_tensor_from_stream(weight_stream, layer_id);
                    layer = l;
                }
            } else if (layer_type == "Hierarchical") {
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

                const DataType_t ps_dtype = host_remote->config().value_dtype;
                NVE_CHECK_(ps_dtype == nve_dtype,
                           "nve_loader: layer '" + module_path +
                           "' has data_type=" + dtype_str +
                           " but remote PS data_type=" + to_string(ps_dtype) +
                           " — they must match");

                auto data_it = entry.find("remote_ps_data");
                if (data_it != entry.end() && data_it->is_object()) {
                    const uint64_t row_bytes =
                        ps_cfg.at("row_elements").get<uint64_t>() *
                        static_cast<uint64_t>(dtype_size(ps_dtype));
                    auto ctx = host_remote->create_execution_context(0, 0, nullptr, nullptr);
                    insert_keys_from_filepath(
                        host_remote, ctx,
                        data_it->at("keys").get<std::string>(),
                        data_it->at("values").get<std::string>(),
                        row_bytes, /*batch_size=*/1ull << 20);
                    ctx->wait();
                }

                size_t host_cache     = entry.value("host_cache_size", size_t{0});
                uint64_t ps_num_rows  = ps_cfg.value("num_rows", uint64_t{0});
                layer = std::make_shared<HierarchicalEmbedding<int64_t>>(
                    emb_size, nve_dtype, gpu_cache, host_cache,
                    host_remote, ps_num_rows,
                    /*use_private_stream=*/training, device_index, cfg);
            } else {
                NVE_CHECK_(false, "nve_loader: unsupported layer_type: " + layer_type);
            }

            register_layer(std::move(layer), layer_id, std::move(module_path),
                            loader, pkg_fqns, device_index);
        }
    }

    void load_layers_v2(const nlohmann::json& doc,
                         torch::inductor::AOTIModelPackageLoader& loader,
                         const std::set<std::string>& pkg_fqns,
                         int device_index,
                         size_t gpu_cache_size_override,
                         const TopologyMap& topology) {
        const auto& resources    = doc.value("resources", nlohmann::json::object());
        const auto& ps_resources = resources.value("remote_ps", nlohmann::json::object());
        const auto& mb_resources = resources.value("memblocks", nlohmann::json::object());

        std::set<std::string> weights_loaded;

        for (const auto& entry : doc["layers"]) {
            int64_t layer_id        = entry["id"];
            std::string module_path = entry["module_path"];
            size_t num_emb          = entry["num_embeddings"];
            size_t emb_size         = entry["embedding_size"];
            std::string dtype_str   = entry["dtype"];
            std::string layer_type  = entry["layer_type"];
            size_t gpu_cache        = gpu_cache_size_override > 0
                                          ? gpu_cache_size_override
                                          : entry["gpu_cache_size"].get<size_t>();
            size_t host_cache       = entry.value("host_cache_size", size_t{0});
            bool training           = entry["optimize_for_training"];
            DataType_t nve_dtype    = parse_dtype(dtype_str);
            std::string storage_ref = entry.value("storage_ref", std::string{});
            EmbedLayerConfig cfg    = parse_embed_config(entry);

            std::shared_ptr<NVEmbedBinding<int64_t>> layer;

            if (layer_type == "LinearUVM" || layer_type == "GPULayer" ||
                layer_type == "HostLayer") {
                const bool device_resident = (layer_type == "GPULayer");

                std::shared_ptr<MemBlock> mem_block;
                bool needs_weight_load;
                if (!storage_ref.empty()) {
                    std::tie(mem_block, needs_weight_load) =
                        get_or_build_memblock(storage_ref, mb_resources,
                                               emb_size, num_emb, nve_dtype,
                                               device_index, device_resident, topology);
                } else {
                    std::string mb_type = (layer_type == "HostLayer")
                        ? "MemBlockType.Host" : "MemBlockType.Managed";
                    mem_block = create_memblock(mb_type, emb_size, num_emb, nve_dtype,
                                               device_index, device_resident);
                    needs_weight_load = true;
                }

                // Weight file is always named after the original (pre-remap) storage_ref.
                std::string weight_key = storage_ref.empty()
                    ? [&]{ auto s = module_path;
                           std::replace(s.begin(), s.end(), '.', '_');
                           return s; }()
                    : storage_ref;

                if (layer_type == "LinearUVM") {
                    auto l = std::make_shared<LinearUVMEmbedding<int64_t>>(
                        emb_size, num_emb, nve_dtype, mem_block,
                        gpu_cache, training, device_index, cfg);
                    if (needs_weight_load && !weights_loaded.count(weight_key)) {
                        InputFileStreamWrapper ws(save_dir_ + "/weights/" + weight_key + ".nve");
                        l->load_tensor_from_stream(ws, layer_id);
                        weights_loaded.insert(weight_key);
                    }
                    layer = l;
                } else if (layer_type == "GPULayer") {
                    auto l = std::make_shared<GPUEmbedding<int64_t>>(
                        emb_size, num_emb, nve_dtype, mem_block, device_index, cfg);
                    if (needs_weight_load && !weights_loaded.count(weight_key)) {
                        InputFileStreamWrapper ws(save_dir_ + "/weights/" + weight_key + ".nve");
                        l->load_tensor_from_stream(ws, layer_id);
                        weights_loaded.insert(weight_key);
                    }
                    layer = l;
                } else {
                    auto l = std::make_shared<HostEmbedding<int64_t>>(
                        emb_size, num_emb, nve_dtype, mem_block, device_index, cfg);
                    if (needs_weight_load && !weights_loaded.count(weight_key)) {
                        InputFileStreamWrapper ws(save_dir_ + "/weights/" + weight_key + ".nve");
                        l->load_tensor_from_stream(ws, layer_id);
                        weights_loaded.insert(weight_key);
                    }
                    layer = l;
                }
            } else if (layer_type == "Hierarchical") {
                NVE_CHECK_(!storage_ref.empty(),
                           "nve_loader: Hierarchical layer '" + module_path +
                           "' is missing storage_ref");
                host_table_ptr_t host_remote = get_or_build_ps(
                    storage_ref, ps_resources, module_path, nve_dtype, dtype_str);
                uint64_t ps_num_rows = 0;
                {
                    const std::string eff = resource_dir_->remap(storage_ref);
                    auto ps_it = ps_resources.find(eff);
                    if (ps_it != ps_resources.end())
                        ps_num_rows = ps_it->value("num_rows", uint64_t{0});
                }
                layer = std::make_shared<HierarchicalEmbedding<int64_t>>(
                    emb_size, nve_dtype, gpu_cache, host_cache,
                    host_remote, ps_num_rows,
                    /*use_private_stream=*/training, device_index, cfg);
            } else {
                NVE_CHECK_(false, "nve_loader: unsupported layer_type: " + layer_type);
            }

            register_layer(std::move(layer), layer_id, std::move(module_path),
                            loader, pkg_fqns, device_index);
        }
    }

    std::string save_dir_;
    std::shared_ptr<ResourceDirectory> resource_dir_;
    std::map<int64_t, LoadedLayer> layers_;
};

} // namespace nve