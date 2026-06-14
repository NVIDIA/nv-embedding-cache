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

#include "nve_registry.hpp"
#include "python/pynve/bindings/binding_layers.hpp"
#include <cstdio>
#include <string>

namespace nve {

NVELayerRegistry& NVELayerRegistry::instance() {
    static NVELayerRegistry inst;
    return inst;
}

void NVELayerRegistry::register_binding(const void* marker_ptr, std::shared_ptr<NVEmbedBinding<int64_t>> binding) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    entries_[marker_ptr] = std::move(binding);
}

std::shared_ptr<NVEmbedBinding<int64_t>> NVELayerRegistry::get_binding(const void* marker_ptr) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = entries_.find(marker_ptr);
    if (it == entries_.end()) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%p", marker_ptr);
        NVE_THROW_(std::string("NVELayerRegistry: no binding registered for marker_ptr=") + buf);
    }
    return it->second;
}

void NVELayerRegistry::unregister_binding(const void* marker_ptr) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    entries_.erase(marker_ptr);
}

// ---------------------------------------------------------------------------
// Free-function accessors. Bridge the forward-declared NVEmbedBinding in
// nve_registry.hpp to the full template defined in binding_layers.hpp.
// ---------------------------------------------------------------------------

static int dtype_to_tag(DataType_t dt) {
    switch (dt) {
        case DataType_t::Float32: return kBindingDtypeFloat32;
        case DataType_t::Float16: return kBindingDtypeFloat16;
        default: return kBindingDtypeUnknown;
    }
}

static DataType_t tag_to_dtype(int tag) {
    switch (tag) {
        case kBindingDtypeFloat32: return DataType_t::Float32;
        case kBindingDtypeFloat16: return DataType_t::Float16;
        default: return DataType_t::Unknown;
    }
}

int64_t binding_embedding_dim(const std::shared_ptr<NVEmbedBinding<int64_t>>& b) {
    return b->get_embedding_dim();
}

int binding_data_type_int(const std::shared_ptr<NVEmbedBinding<int64_t>>& b) {
    return dtype_to_tag(b->get_data_type());
}

void binding_lookup(const std::shared_ptr<NVEmbedBinding<int64_t>>& b,
                    std::size_t num_keys,
                    std::uintptr_t keys,
                    std::uintptr_t output,
                    std::uint64_t stream) {
    b->lookup(num_keys, keys, output, stream);
}

void binding_lookup_with_pooling(const std::shared_ptr<NVEmbedBinding<int64_t>>& b,
                                 std::size_t num_keys,
                                 std::uintptr_t keys,
                                 std::uintptr_t output,
                                 std::uint32_t pooling_type,
                                 std::size_t num_offsets,
                                 std::uintptr_t offsets,
                                 int weight_dtype,
                                 std::uintptr_t weights,
                                 std::uint64_t stream) {
    b->lookup_with_pooling(num_keys, keys, output, pooling_type, num_offsets, offsets,
                           static_cast<std::uint64_t>(tag_to_dtype(weight_dtype)),
                           weights, stream);
}

} // namespace nve
