# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# libnve-torch-ops.so — PyTorch custom op registration + CUDA kernels + registry.

if(NOT TORCH_FOUND)
    message(FATAL_ERROR "Torch package was not found!")
endif()

# CPU/HostLayer serving runtime: nve_torch_ops.cu calls
# aoti_torch_get_current_cuda_stream, a symbol that lives ONLY in
# libtorch_cuda.so (the GPU edition of libtorch). Linked against a CPU-only
# libtorch (e.g. @libtorch_znver3, which ships no libtorch_cuda.so) that symbol
# is undefined, so the combined .so cannot load on a driverless host. When
# NVE_TORCH_OPS_CPU_ONLY=ON we drop the CUDA dispatch TU and its CUDA
# registration (guarded in torch_binding.cpp) and keep only the CPU kernel — the
# HostLayer forward path dispatches through torch's CPU key. nve_registry.cu
# STAYS: it is nvcc-compiled (it instantiates NVEmbedBinding<int64_t> from
# binding_layers.hpp) but references no libtorch_cuda symbol — only NVE +
# libcudart, which is bundled and stays dormant on a HostLayer (driverless) load.
option(NVE_TORCH_OPS_CPU_ONLY
       "Build libnve-torch-ops.so without the CUDA dispatch TU (CPU/HostLayer serving)" OFF)

set(NVE_TORCH_OPS_SRCS
    nve_registry.cu
    torch_binding.cpp
    nve_torch_ops_cpu.cpp
)
if(NOT NVE_TORCH_OPS_CPU_ONLY)
    list(APPEND NVE_TORCH_OPS_SRCS nve_torch_ops.cu)
endif()

add_library(nve-torch-ops SHARED ${NVE_TORCH_OPS_SRCS})

if(NVE_TORCH_OPS_CPU_ONLY)
    target_compile_definitions(nve-torch-ops PRIVATE NVE_TORCH_OPS_CPU_ONLY=1)
    message(STATUS "nve-torch-ops: CPU-only build (no CUDA dispatch TU)")
endif()

# Target the minimum torch stable ABI version (set in top-level CMakeLists.txt).
# This avoids using APIs added in newer versions, so code built against e.g.
# torch 2.11 headers runs on torch 2.10 runtime.
# TORCH_TARGET_VERSION format: (major << 56) | (minor << 48)
math(EXPR NVE_TORCH_TARGET "(${NVE_MIN_TORCH_MAJOR} << 56) | (${NVE_MIN_TORCH_MINOR} << 48)")
target_compile_definitions(nve-torch-ops PRIVATE
    TORCH_TARGET_VERSION=${NVE_TORCH_TARGET}
)

target_include_directories(nve-torch-ops PRIVATE
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(nve-torch-ops
    PUBLIC  nve-common
    PRIVATE ${TORCH_LIBRARIES}
)
