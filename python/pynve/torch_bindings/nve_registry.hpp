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

// NVEmbedBinding is forward-declared so this header stays free of CUDA-only
// includes (binding_layers.hpp transitively pulls in <cub/cub.cuh>). Pure-host
// translation units (e.g. nve_torch_ops_cpu.cpp) can include this header and
// route through the free-function helpers below; full definitions live in
// nve_registry.cu where binding_layers.hpp is visible.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace nve {

template <typename IndexT> class NVEmbedBinding;

/**
 * Global registry mapping per-layer marker pointers to NVEmbedBinding instances.
 *
 * Used by torch custom ops (nve_ops::embedding_lookup, nve_ops::embedding_lookup_with_pooling)
 * to locate the C++ embedding layer during forward execution.
 *
 * The key is the data_ptr() of a per-layer "marker" tensor (a small buffer threaded
 * into every op call). 
 *
 * Currently only supports int64_t key type.
 */
class __attribute__((visibility("default"))) NVELayerRegistry {
public:
    static NVELayerRegistry& instance();

    void register_binding(const void* marker_ptr, std::shared_ptr<NVEmbedBinding<int64_t>> binding);
    std::shared_ptr<NVEmbedBinding<int64_t>> get_binding(const void* marker_ptr);
    void unregister_binding(const void* marker_ptr);

private:
    NVELayerRegistry() = default;
    ~NVELayerRegistry() = default;
    NVELayerRegistry(const NVELayerRegistry&) = delete;
    NVELayerRegistry& operator=(const NVELayerRegistry&) = delete;

    std::unordered_map<const void*, std::shared_ptr<NVEmbedBinding<int64_t>>> entries_;
    // Reader-writer lock: get_binding (per-op hot path) takes a shared lock so
    // concurrent dispatches don't serialize; register/unregister (load/teardown)
    // take an exclusive lock.
    std::shared_mutex mutex_;
};

// ---------------------------------------------------------------------------
// Free-function accessors over the binding.
//
// Defined in nve_registry.cu — that's the only TU that needs the full
// NVEmbedBinding template definition. Callers (CPU .cpp / CUDA .cu) can
// invoke these without including binding_layers.hpp.
// ---------------------------------------------------------------------------

// Normalized dtype tags used by the helpers to avoid leaking DataType_t into
// host-only translation units. The .cu helpers translate to/from DataType_t.
enum BindingDtype : int {
    kBindingDtypeUnknown = 0,
    kBindingDtypeFloat32 = 1,
    kBindingDtypeFloat16 = 2,
};

int64_t binding_embedding_dim(const std::shared_ptr<NVEmbedBinding<int64_t>>& b);

// Returns a BindingDtype tag for the binding's storage dtype.
int binding_data_type_int(const std::shared_ptr<NVEmbedBinding<int64_t>>& b);

void binding_lookup(const std::shared_ptr<NVEmbedBinding<int64_t>>& b,
                    std::size_t num_keys,
                    std::uintptr_t keys,
                    std::uintptr_t output,
                    std::uint64_t stream);

// weight_dtype is a BindingDtype tag.
void binding_lookup_with_pooling(const std::shared_ptr<NVEmbedBinding<int64_t>>& b,
                                 std::size_t num_keys,
                                 std::uintptr_t keys,
                                 std::uintptr_t output,
                                 std::uint32_t pooling_type,
                                 std::size_t num_offsets,
                                 std::uintptr_t offsets,
                                 int weight_dtype,
                                 std::uintptr_t weights,
                                 std::uint64_t stream);

} // namespace nve
