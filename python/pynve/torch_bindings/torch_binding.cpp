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
// The layer is located via the per-layer marker tensor's data_ptr (see
// NVELayerRegistry). embedding_size / dtype are baked into the op only for the
// Python meta/fake impl; the real kernels read shape+dtype from the binding, so
// those scalars are not forwarded here.
extern "C" {
AtenTensorHandle nve_embedding_lookup_cuda(
    AtenTensorHandle marker, AtenTensorHandle keys);

AtenTensorHandle nve_embedding_lookup_with_pooling_cuda(
    AtenTensorHandle marker, AtenTensorHandle keys, AtenTensorHandle offsets,
    AtenTensorHandle weights,  // nullptr when optional is empty
    int64_t pooling_type);

AtenTensorHandle nve_embedding_lookup_cpu(
    AtenTensorHandle marker, AtenTensorHandle keys);

AtenTensorHandle nve_embedding_lookup_with_pooling_cpu(
    AtenTensorHandle marker, AtenTensorHandle keys, AtenTensorHandle offsets,
    AtenTensorHandle weights,  // nullptr when optional is empty
    int64_t pooling_type);
}

// ---------------------------------------------------------------------------
// Boxed kernel wrappers
//
// The dispatcher passes inputs on a StableIValue* stack (arg0 at index 0).
// We read inputs with to<T>(), call the kernel, and write the output
// tensor handle back to stack[0] with from().
// ---------------------------------------------------------------------------

// embedding_lookup(Tensor marker, Tensor keys, int embedding_size, int dtype) -> Tensor
// embedding_size/dtype (stack[2]/stack[3]) are consumed by the Python fake only.
static void embedding_lookup_cuda_boxed(
    StableIValue* stack, uint64_t /*num_inputs*/, uint64_t /*num_outputs*/)
{
    AtenTensorHandle marker = to<AtenTensorHandle>(stack[0]);
    AtenTensorHandle keys   = to<AtenTensorHandle>(stack[1]);
    AtenTensorHandle result = nve_embedding_lookup_cuda(marker, keys);
    stack[0] = from(result);
}

// embedding_lookup_with_pooling(Tensor marker, Tensor keys, Tensor offsets,
//     Tensor? weights, int pooling_type, int embedding_size, int dtype) -> Tensor
static void embedding_lookup_with_pooling_cuda_boxed(
    StableIValue* stack, uint64_t /*num_inputs*/, uint64_t /*num_outputs*/)
{
    AtenTensorHandle marker  = to<AtenTensorHandle>(stack[0]);
    AtenTensorHandle keys    = to<AtenTensorHandle>(stack[1]);
    AtenTensorHandle offsets = to<AtenTensorHandle>(stack[2]);
    auto weights_opt = to<std::optional<AtenTensorHandle>>(stack[3]);
    int64_t pooling_type = to<int64_t>(stack[4]);
    AtenTensorHandle weights_handle =
        weights_opt.has_value() ? weights_opt.value() : nullptr;
    AtenTensorHandle result = nve_embedding_lookup_with_pooling_cuda(
        marker, keys, offsets, weights_handle, pooling_type);
    stack[0] = from(result);
}

// CPU dispatch — mirrors the CUDA boxed wrappers, just routes to the *_cpu
// kernel which never touches the CUDA runtime.
static void embedding_lookup_cpu_boxed(
    StableIValue* stack, uint64_t /*num_inputs*/, uint64_t /*num_outputs*/)
{
    AtenTensorHandle marker = to<AtenTensorHandle>(stack[0]);
    AtenTensorHandle keys   = to<AtenTensorHandle>(stack[1]);
    AtenTensorHandle result = nve_embedding_lookup_cpu(marker, keys);
    stack[0] = from(result);
}

static void embedding_lookup_with_pooling_cpu_boxed(
    StableIValue* stack, uint64_t /*num_inputs*/, uint64_t /*num_outputs*/)
{
    AtenTensorHandle marker  = to<AtenTensorHandle>(stack[0]);
    AtenTensorHandle keys    = to<AtenTensorHandle>(stack[1]);
    AtenTensorHandle offsets = to<AtenTensorHandle>(stack[2]);
    auto weights_opt = to<std::optional<AtenTensorHandle>>(stack[3]);
    int64_t pooling_type = to<int64_t>(stack[4]);
    AtenTensorHandle weights_handle =
        weights_opt.has_value() ? weights_opt.value() : nullptr;
    AtenTensorHandle result = nve_embedding_lookup_with_pooling_cpu(
        marker, keys, offsets, weights_handle, pooling_type);
    stack[0] = from(result);
}

// ---------------------------------------------------------------------------
// Schema definitions (STABLE_TORCH_LIBRARY uses C shim, no std::string)
// ---------------------------------------------------------------------------
STABLE_TORCH_LIBRARY(nve_ops, m) {
    m.def("embedding_lookup(Tensor marker, Tensor keys, int embedding_size, int dtype) -> Tensor");
    m.def("embedding_lookup_with_pooling(Tensor marker, Tensor keys, Tensor offsets, "
          "Tensor? weights, int pooling_type, int embedding_size, int dtype) -> Tensor");
}

// ---------------------------------------------------------------------------
// CUDA dispatch
//
// Meta (shape-inference) dispatch is intentionally *not* registered here —
// the stable C ABI can't resolve SymInts through aoti_torch_get_numel, so
// Meta is handled by torch.library.register_fake on the Python side
// (see pynve/torch/__init__.py).
// ---------------------------------------------------------------------------
STABLE_TORCH_LIBRARY_IMPL(nve_ops, CUDA, m) {
    m.impl("embedding_lookup", embedding_lookup_cuda_boxed);
    m.impl("embedding_lookup_with_pooling",
           embedding_lookup_with_pooling_cuda_boxed);
}

// ---------------------------------------------------------------------------
// CPU dispatch — used by LayerType.HostLayer with device='cpu'.
// ---------------------------------------------------------------------------
STABLE_TORCH_LIBRARY_IMPL(nve_ops, CPU, m) {
    m.impl("embedding_lookup", embedding_lookup_cpu_boxed);
    m.impl("embedding_lookup_with_pooling",
           embedding_lookup_with_pooling_cpu_boxed);
}

#pragma GCC diagnostic pop
