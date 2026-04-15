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

// torch_binding.cpp — Stable ABI registration for nve_ops custom ops.
//
// Actual kernel implementations live in nve_torch_ops.cu (regular ATen API).
// This file only contains boxed wrappers that shuttle StableIValues between
// the dispatcher and the kernels via AtenTensorHandle.

#include <torch/csrc/stable/library.h>
#include <optional>

// Suppress deprecation warnings for to<T>/from<T> — newer torch (25.06+)
// moved them to torch::stable::detail:: but the old API still works.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

// ---------------------------------------------------------------------------
// Forward declarations of kernel functions (implemented in nve_torch_ops.cu).
// AtenTensorHandle is an opaque pointer (at::Tensor* underneath).
// Input handles are BORROWED; output handles are OWNING (new at::Tensor*).
// ---------------------------------------------------------------------------
extern "C" {
AtenTensorHandle nve_embedding_lookup_cuda(
    AtenTensorHandle keys, int64_t layer_id);

AtenTensorHandle nve_embedding_lookup_with_pooling_cuda(
    AtenTensorHandle keys, AtenTensorHandle offsets,
    AtenTensorHandle weights,  // nullptr when optional is empty
    int64_t pooling_type, int64_t layer_id);

AtenTensorHandle nve_embedding_lookup_meta(
    AtenTensorHandle keys, int64_t layer_id);

AtenTensorHandle nve_embedding_lookup_with_pooling_meta(
    AtenTensorHandle keys, AtenTensorHandle offsets,
    AtenTensorHandle weights,  // nullptr when optional is empty
    int64_t pooling_type, int64_t layer_id);
}

// ---------------------------------------------------------------------------
// Boxed kernel wrappers
//
// The dispatcher passes inputs on a StableIValue* stack (arg0 at index 0).
// We read inputs with to<T>(), call the kernel, and write the output
// tensor handle back to stack[0] with from().
// ---------------------------------------------------------------------------

// embedding_lookup(Tensor keys, int layer_id) -> Tensor
static void embedding_lookup_cuda_boxed(
    StableIValue* stack, uint64_t /*num_inputs*/, uint64_t /*num_outputs*/)
{
    AtenTensorHandle keys = to<AtenTensorHandle>(stack[0]);
    int64_t layer_id = to<int64_t>(stack[1]);
    AtenTensorHandle result = nve_embedding_lookup_cuda(keys, layer_id);
    stack[0] = from(result);
}

// embedding_lookup_with_pooling(Tensor keys, Tensor offsets,
//     Tensor? weights, int pooling_type, int layer_id) -> Tensor
static void embedding_lookup_with_pooling_cuda_boxed(
    StableIValue* stack, uint64_t /*num_inputs*/, uint64_t /*num_outputs*/)
{
    AtenTensorHandle keys    = to<AtenTensorHandle>(stack[0]);
    AtenTensorHandle offsets = to<AtenTensorHandle>(stack[1]);
    auto weights_opt = to<std::optional<AtenTensorHandle>>(stack[2]);
    int64_t pooling_type = to<int64_t>(stack[3]);
    int64_t layer_id     = to<int64_t>(stack[4]);
    AtenTensorHandle weights_handle =
        weights_opt.has_value() ? weights_opt.value() : nullptr;
    AtenTensorHandle result = nve_embedding_lookup_with_pooling_cuda(
        keys, offsets, weights_handle, pooling_type, layer_id);
    stack[0] = from(result);
}

// Meta (shape-only) variants for torch.export
static void embedding_lookup_meta_boxed(
    StableIValue* stack, uint64_t /*num_inputs*/, uint64_t /*num_outputs*/)
{
    AtenTensorHandle keys = to<AtenTensorHandle>(stack[0]);
    int64_t layer_id = to<int64_t>(stack[1]);
    AtenTensorHandle result = nve_embedding_lookup_meta(keys, layer_id);
    stack[0] = from(result);
}

static void embedding_lookup_with_pooling_meta_boxed(
    StableIValue* stack, uint64_t /*num_inputs*/, uint64_t /*num_outputs*/)
{
    AtenTensorHandle keys    = to<AtenTensorHandle>(stack[0]);
    AtenTensorHandle offsets = to<AtenTensorHandle>(stack[1]);
    auto weights_opt = to<std::optional<AtenTensorHandle>>(stack[2]);
    int64_t pooling_type = to<int64_t>(stack[3]);
    int64_t layer_id     = to<int64_t>(stack[4]);
    AtenTensorHandle weights_handle =
        weights_opt.has_value() ? weights_opt.value() : nullptr;
    AtenTensorHandle result = nve_embedding_lookup_with_pooling_meta(
        keys, offsets, weights_handle, pooling_type, layer_id);
    stack[0] = from(result);
}

// ---------------------------------------------------------------------------
// Schema definitions (STABLE_TORCH_LIBRARY uses C shim, no std::string)
// ---------------------------------------------------------------------------
STABLE_TORCH_LIBRARY(nve_ops, m) {
    m.def("embedding_lookup(Tensor keys, int layer_id) -> Tensor");
    m.def("embedding_lookup_with_pooling(Tensor keys, Tensor offsets, "
          "Tensor? weights, int pooling_type, int layer_id) -> Tensor");
}

// ---------------------------------------------------------------------------
// CUDA dispatch
// ---------------------------------------------------------------------------
STABLE_TORCH_LIBRARY_IMPL(nve_ops, CUDA, m) {
    m.impl("embedding_lookup", embedding_lookup_cuda_boxed);
    m.impl("embedding_lookup_with_pooling",
           embedding_lookup_with_pooling_cuda_boxed);
}

// ---------------------------------------------------------------------------
// Meta dispatch (shape inference for torch.export)
// ---------------------------------------------------------------------------
STABLE_TORCH_LIBRARY_IMPL(nve_ops, Meta, m) {
    m.impl("embedding_lookup", embedding_lookup_meta_boxed);
    m.impl("embedding_lookup_with_pooling",
           embedding_lookup_with_pooling_meta_boxed);
}

#pragma GCC diagnostic pop
