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
#include <stdexcept>

namespace nve {

NVELayerRegistry& NVELayerRegistry::instance() {
    static NVELayerRegistry inst;
    return inst;
}

void NVELayerRegistry::register_binding(int64_t id, std::shared_ptr<NVEmbedBinding<int64_t>> binding) {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_[id] = std::move(binding);
}

std::shared_ptr<NVEmbedBinding<int64_t>> NVELayerRegistry::get_binding(int64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(id);
    if (it == entries_.end()) {
        throw std::runtime_error("NVELayerRegistry: no binding registered for layer_id=" + std::to_string(id));
    }
    return it->second;
}

void NVELayerRegistry::unregister_binding(int64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.erase(id);
}

} // namespace nve
