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

#include "python/pynve/bindings/binding_layers.hpp"
#include <unordered_map>
#include <mutex>
#include <memory>
#include <cstdint>

namespace nve {

/**
 * Global registry mapping integer layer IDs to NVEmbedBinding instances.
 *
 * Used by torch custom ops (nve_ops::embedding_lookup, nve_ops::embedding_lookup_with_pooling)
 * to locate the C++ embedding layer during forward execution and meta-shape inference
 * (for torch.export).
 *
 * Currently only supports int64_t key type.
 */
class __attribute__((visibility("default"))) NVELayerRegistry {
public:
    static NVELayerRegistry& instance();

    void register_binding(int64_t id, std::shared_ptr<NVEmbedBinding<int64_t>> binding);
    std::shared_ptr<NVEmbedBinding<int64_t>> get_binding(int64_t id);
    void unregister_binding(int64_t id);

private:
    NVELayerRegistry() = default;
    ~NVELayerRegistry() = default;
    NVELayerRegistry(const NVELayerRegistry&) = delete;
    NVELayerRegistry& operator=(const NVELayerRegistry&) = delete;

    std::unordered_map<int64_t, std::shared_ptr<NVEmbedBinding<int64_t>>> entries_;
    std::mutex mutex_;
};

} // namespace nve
