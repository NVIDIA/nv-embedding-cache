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

// nve_torch_ops_cpu.cpp — CPU dispatch implementations for nve_ops custom ops.
//
// Mirrors nve_torch_ops.cu but never calls the CUDA stream shim and creates
// outputs on a CPU device. The underlying NVE binding receives stream=0; the
// host layer's execution context gates all CUDA ops on driver availability, so
// the path is safe on a driverless system.
//
// This TU is intentionally CUDA-free: it talks to the binding via free
// functions declared in nve_registry.hpp (forward-declared NVEmbedBinding).

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <array>
#include <stdexcept>

#include "nve_registry.hpp"

namespace ts = torch::stable;

static AtenTensorHandle to_shared_handle(const ts::Tensor& t) {
    AtenTensorHandle h = nullptr;
    aoti_torch_new_tensor_handle(t.get(), &h);
    return h;
}

static ts::ScalarType dtype_tag_to_stable(int tag) {
    if (tag == nve::kBindingDtypeFloat32) return ts::ScalarType::Float;
    if (tag == nve::kBindingDtypeFloat16) return ts::ScalarType::Half;
    throw std::runtime_error("nve-torch-ops-cpu: unsupported BindingDtype tag");
}

extern "C" AtenTensorHandle nve_embedding_lookup_cpu(
    AtenTensorHandle marker_handle, AtenTensorHandle keys_handle)
{
    ts::Tensor marker(marker_handle);
    ts::Tensor keys(keys_handle);
    auto binding = nve::NVELayerRegistry::instance().get_binding(marker.data_ptr());

    int64_t num_keys = keys.numel();
    int64_t emb_dim = nve::binding_embedding_dim(binding);

    std::array<int64_t, 2> out_sizes = {num_keys, emb_dim};
    ts::Tensor output = ts::empty(
        ts::IntHeaderOnlyArrayRef(out_sizes.data(), 2),
        dtype_tag_to_stable(nve::binding_data_type_int(binding)),
        std::nullopt,
        ts::Device(ts::DeviceType::CPU));

    // stream=0 — sentinel for host-only execution context cache key.
    nve::binding_lookup(
        binding,
        static_cast<std::size_t>(num_keys),
        reinterpret_cast<std::uintptr_t>(keys.data_ptr()),
        reinterpret_cast<std::uintptr_t>(output.data_ptr()),
        /*stream=*/0);

    return to_shared_handle(output);
}

extern "C" AtenTensorHandle nve_embedding_lookup_with_pooling_cpu(
    AtenTensorHandle marker_handle, AtenTensorHandle keys_handle,
    AtenTensorHandle offsets_handle,
    AtenTensorHandle weights_handle,
    int64_t pooling_type)
{
    ts::Tensor marker(marker_handle);
    ts::Tensor keys(keys_handle);
    ts::Tensor offsets(offsets_handle);
    auto binding = nve::NVELayerRegistry::instance().get_binding(marker.data_ptr());

    int64_t num_bags = offsets.numel() - 1;
    int64_t emb_dim = nve::binding_embedding_dim(binding);

    std::array<int64_t, 2> out_sizes = {num_bags, emb_dim};
    ts::Tensor output = ts::empty(
        ts::IntHeaderOnlyArrayRef(out_sizes.data(), 2),
        dtype_tag_to_stable(nve::binding_data_type_int(binding)),
        std::nullopt,
        ts::Device(ts::DeviceType::CPU));

    int weight_dtype = nve::kBindingDtypeUnknown;
    uintptr_t weight_ptr = 0;
    if (weights_handle != nullptr) {
        ts::Tensor weights(weights_handle);
        auto st = weights.scalar_type();
        weight_dtype = (st == ts::ScalarType::Float) ? nve::kBindingDtypeFloat32
                                                     : nve::kBindingDtypeFloat16;
        weight_ptr = reinterpret_cast<uintptr_t>(weights.data_ptr());
    }

    nve::binding_lookup_with_pooling(
        binding,
        static_cast<std::size_t>(keys.numel()),
        reinterpret_cast<std::uintptr_t>(keys.data_ptr()),
        reinterpret_cast<std::uintptr_t>(output.data_ptr()),
        static_cast<std::uint32_t>(pooling_type),
        static_cast<std::size_t>(num_bags),
        reinterpret_cast<std::uintptr_t>(offsets.data_ptr()),
        weight_dtype,
        weight_ptr,
        /*stream=*/0);

    return to_shared_handle(output);
}
