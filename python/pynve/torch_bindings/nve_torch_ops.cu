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

// nve_torch_ops.cu — CUDA kernel implementations for nve_ops custom ops.
//
// Uses the PyTorch Stable ABI (torch::stable::Tensor, torch::stable::empty,
// aoti_torch_* C shim) instead of ATen C++ API for version independence.
//
// No TORCH_LIBRARY / STABLE_TORCH_LIBRARY macros here (those live in
// torch_binding.cpp). Functions are exposed via extern "C" using
// AtenTensorHandle. Talks to NVEmbedBinding through the free-function helpers
// in nve_registry.hpp — same surface used by the CPU dispatch TU.

// Required for aoti_torch_get_current_cuda_stream in shim.h (guarded by #ifdef USE_CUDA)
#ifndef USE_CUDA
#define USE_CUDA
#endif
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <array>
#include <stdexcept>

#include "nve_registry.hpp"
#include "nve_torch_ops_checks.hpp"

namespace ts = torch::stable;

static AtenTensorHandle to_shared_handle(const ts::Tensor& t) {
    AtenTensorHandle h = nullptr;
    aoti_torch_new_tensor_handle(t.get(), &h);
    return h;
}

static ts::ScalarType dtype_tag_to_stable(int tag) {
    if (tag == nve::kBindingDtypeFloat32) return ts::ScalarType::Float;
    if (tag == nve::kBindingDtypeFloat16) return ts::ScalarType::Half;
    throw std::runtime_error("nve-torch-ops: unsupported BindingDtype tag");
}

// Get raw cudaStream_t via C shim (not part of stable Tensor API)
static void* get_cuda_stream(int32_t device_index) {
    void* stream = nullptr;
    aoti_torch_get_current_cuda_stream(device_index, &stream);
    return stream;
}

// ---------------------------------------------------------------------------
// CUDA implementations
// ---------------------------------------------------------------------------

extern "C" AtenTensorHandle nve_embedding_lookup_cuda(
    AtenTensorHandle marker_handle, AtenTensorHandle keys_handle)
{
    ts::Tensor marker(marker_handle);
    ts::Tensor keys(keys_handle);
    nve::check_index_tensor(keys, "keys");
    auto binding = nve::NVELayerRegistry::instance().get_binding(marker.data_ptr());

    int64_t num_keys = keys.numel();
    int64_t emb_dim = nve::binding_embedding_dim(binding);
    int32_t device_index = keys.get_device();
    void* stream = get_cuda_stream(device_index);

    std::array<int64_t, 2> out_sizes = {num_keys, emb_dim};
    ts::Tensor output = ts::empty(
        ts::IntHeaderOnlyArrayRef(out_sizes.data(), 2),
        dtype_tag_to_stable(nve::binding_output_data_type_int(binding)),
        std::nullopt,
        ts::Device(ts::DeviceType::CUDA, device_index));

    nve::binding_lookup(
        binding,
        static_cast<std::size_t>(num_keys),
        reinterpret_cast<std::uintptr_t>(keys.data_ptr()),
        reinterpret_cast<std::uintptr_t>(output.data_ptr()),
        reinterpret_cast<std::uint64_t>(stream));

    return to_shared_handle(output);
}

extern "C" AtenTensorHandle nve_embedding_lookup_with_pooling_cuda(
    AtenTensorHandle marker_handle, AtenTensorHandle keys_handle,
    AtenTensorHandle offsets_handle,
    AtenTensorHandle weights_handle,
    int64_t pooling_type)
{
    ts::Tensor marker(marker_handle);
    ts::Tensor keys(keys_handle);
    ts::Tensor offsets(offsets_handle);
    nve::check_index_tensor(keys, "keys");
    nve::check_bag_offsets(offsets);
    auto binding = nve::NVELayerRegistry::instance().get_binding(marker.data_ptr());

    int32_t device_index = keys.get_device();
    void* stream = get_cuda_stream(device_index);
    int64_t num_bags = offsets.numel() - 1;
    int64_t emb_dim = nve::binding_embedding_dim(binding);

    std::array<int64_t, 2> out_sizes = {num_bags, emb_dim};
    ts::Tensor output = ts::empty(
        ts::IntHeaderOnlyArrayRef(out_sizes.data(), 2),
        dtype_tag_to_stable(nve::binding_output_data_type_int(binding)),
        std::nullopt,
        ts::Device(ts::DeviceType::CUDA, device_index));

    int weight_dtype = nve::kBindingDtypeUnknown;
    std::uintptr_t weight_ptr = 0;
    if (weights_handle != nullptr) {
        ts::Tensor weights(weights_handle);
        weight_dtype = nve::check_pooling_weights(weights, keys.numel());
        weight_ptr = reinterpret_cast<std::uintptr_t>(weights.data_ptr());
    }

    nve::binding_lookup_with_pooling(
        binding,
        static_cast<std::size_t>(keys.numel()),
        reinterpret_cast<std::uintptr_t>(keys.data_ptr()),
        reinterpret_cast<std::uintptr_t>(output.data_ptr()),
        static_cast<std::uint32_t>(pooling_type),
        static_cast<std::size_t>(offsets.numel()),
        reinterpret_cast<std::uintptr_t>(offsets.data_ptr()),
        weight_dtype,
        weight_ptr,
        reinterpret_cast<std::uint64_t>(stream));

    return to_shared_handle(output);
}

// Meta (shape-inference) impls intentionally omitted — see note in
// torch_binding.cpp. Python-side torch.library.register_fake handles it.
