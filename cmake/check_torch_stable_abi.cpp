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

// Minimal test to verify the torch stable ABI is available and functional.
// Used by CMake try_compile to detect whether libnve-torch-ops.so can be built.
// Tests: STABLE_TORCH_LIBRARY, torch::stable::Tensor, torch::stable::empty.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>

STABLE_TORCH_LIBRARY(nve_abi_check, m) {
    m.def("dummy(int x) -> int");
}

int main() {
    // Verify stable Tensor and empty() are available
    (void)sizeof(torch::stable::Tensor);
    return 0;
}
