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
// AtenTensorHandle.

// Required for aoti_torch_get_current_cuda_stream in shim.h (guarded by #ifdef USE_CUDA)
#ifndef USE_CUDA
#define USE_CUDA
#endif
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include "nve_registry.hpp"

namespace ts = torch::stable;

// Create a new owning AtenTensorHandle from a stable::Tensor.
// The returned handle must be freed by the caller (the dispatcher does this).
// We can't use tensor.get() directly because the stable::Tensor's shared_ptr
// would delete the handle when it goes out of scope.
static AtenTensorHandle to_owning_handle(const ts::Tensor& t) {
    AtenTensorHandle clone = nullptr;
    aoti_torch_clone(t.get(), &clone);
    return clone;
}

static ts::ScalarType nve_to_stable_dtype(nve::DataType_t dt) {
    if (dt == nve::DataType_t::Float32) return ts::ScalarType::Float;
    if (dt == nve::DataType_t::Float16) return ts::ScalarType::Half;
    throw std::runtime_error("nve-torch-ops: unsupported DataType_t");
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
    AtenTensorHandle keys_handle, int64_t layer_id)
{
    ts::Tensor keys(keys_handle);
    auto binding = nve::NVELayerRegistry::instance().get_binding(layer_id);

    int64_t num_keys = keys.numel();
    int64_t emb_dim = static_cast<int64_t>(binding->get_embedding_dim());
    int32_t device_index = keys.get_device();
    void* stream = get_cuda_stream(device_index);

    std::array<int64_t, 2> out_sizes = {num_keys, emb_dim};
    ts::Tensor output = ts::empty(
        ts::IntHeaderOnlyArrayRef(out_sizes.data(), 2),
        nve_to_stable_dtype(binding->get_data_type()),
        std::nullopt,
        ts::Device(ts::DeviceType::CUDA, device_index));

    binding->lookup(
        static_cast<size_t>(num_keys),
        reinterpret_cast<uintptr_t>(keys.data_ptr()),
        reinterpret_cast<uintptr_t>(output.data_ptr()),
        reinterpret_cast<uint64_t>(stream));

    return to_owning_handle(output);
}

extern "C" AtenTensorHandle nve_embedding_lookup_with_pooling_cuda(
    AtenTensorHandle keys_handle, AtenTensorHandle offsets_handle,
    AtenTensorHandle weights_handle,
    int64_t pooling_type, int64_t layer_id)
{
    ts::Tensor keys(keys_handle);
    ts::Tensor offsets(offsets_handle);
    auto binding = nve::NVELayerRegistry::instance().get_binding(layer_id);

    int32_t device_index = keys.get_device();
    void* stream = get_cuda_stream(device_index);
    int64_t num_bags = offsets.numel() - 1;
    int64_t emb_dim = static_cast<int64_t>(binding->get_embedding_dim());

    std::array<int64_t, 2> out_sizes = {num_bags, emb_dim};
    ts::Tensor output = ts::empty(
        ts::IntHeaderOnlyArrayRef(out_sizes.data(), 2),
        nve_to_stable_dtype(binding->get_data_type()),
        std::nullopt,
        ts::Device(ts::DeviceType::CUDA, device_index));

    uint64_t weight_dtype = static_cast<uint64_t>(nve::DataType_t::Unknown);
    uintptr_t weight_ptr = 0;
    if (weights_handle != nullptr) {
        ts::Tensor weights(weights_handle);
        auto st = weights.scalar_type();
        if (st == ts::ScalarType::Float) {
            weight_dtype = static_cast<uint64_t>(nve::DataType_t::Float32);
        } else {
            weight_dtype = static_cast<uint64_t>(nve::DataType_t::Float16);
        }
        weight_ptr = reinterpret_cast<uintptr_t>(weights.data_ptr());
    }

    binding->lookup_with_pooling(
        static_cast<size_t>(keys.numel()),
        reinterpret_cast<uintptr_t>(keys.data_ptr()),
        reinterpret_cast<uintptr_t>(output.data_ptr()),
        static_cast<uint32_t>(pooling_type),
        static_cast<size_t>(num_bags),
        reinterpret_cast<uintptr_t>(offsets.data_ptr()),
        weight_dtype,
        weight_ptr,
        reinterpret_cast<uint64_t>(stream));

    return to_owning_handle(output);
}

// ---------------------------------------------------------------------------
// Meta implementations (shape inference for torch.export, no CUDA)
// ---------------------------------------------------------------------------

extern "C" AtenTensorHandle nve_embedding_lookup_meta(
    AtenTensorHandle keys_handle, int64_t layer_id)
{
    ts::Tensor keys(keys_handle);
    auto binding = nve::NVELayerRegistry::instance().get_binding(layer_id);

    int64_t num_keys = keys.numel();
    int64_t emb_dim = static_cast<int64_t>(binding->get_embedding_dim());

    std::array<int64_t, 2> out_sizes = {num_keys, emb_dim};
    ts::Tensor output = ts::empty(
        ts::IntHeaderOnlyArrayRef(out_sizes.data(), 2),
        nve_to_stable_dtype(binding->get_data_type()),
        std::nullopt,
        ts::Device(ts::DeviceType::Meta));

    return to_owning_handle(output);
}

extern "C" AtenTensorHandle nve_embedding_lookup_with_pooling_meta(
    AtenTensorHandle keys_handle, AtenTensorHandle offsets_handle,
    AtenTensorHandle /*weights_handle*/,
    int64_t /*pooling_type*/, int64_t layer_id)
{
    ts::Tensor keys(keys_handle);
    ts::Tensor offsets(offsets_handle);
    auto binding = nve::NVELayerRegistry::instance().get_binding(layer_id);

    int64_t num_bags = offsets.numel() - 1;
    int64_t emb_dim = static_cast<int64_t>(binding->get_embedding_dim());

    std::array<int64_t, 2> out_sizes = {num_bags, emb_dim};
    ts::Tensor output = ts::empty(
        ts::IntHeaderOnlyArrayRef(out_sizes.data(), 2),
        nve_to_stable_dtype(binding->get_data_type()),
        std::nullopt,
        ts::Device(ts::DeviceType::Meta));

    return to_owning_handle(output);
}
