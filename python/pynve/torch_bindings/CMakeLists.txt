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

if (NOT TORCH_FOUND)
    message(FATAL_ERROR "Torch package was not found!")
endif()

# nve_torch_ops.cu (the CUDA op dispatch) calls aoti_torch_get_current_cuda_stream, which is
# defined only in libtorch_cuda.so — a hard NEEDED that prevents loading on driverless hosts. In a
# driverless build we drop it (and its torch_binding_cuda.cpp registration) and register only the
# CPU dispatch, so the library carries no libtorch_cuda dependency. torch_binding.cpp holds the
# shared op schema + CPU dispatch and is always compiled; the CPU TU uses a sentinel stream and
# never touches libtorch_cuda.
set(nve_torch_ops_src
    nve_registry.cu
    torch_binding.cpp
    nve_torch_ops_cpu.cpp
)
if (NOT NVE_DRIVERLESS_BUILD)
    list(APPEND nve_torch_ops_src nve_torch_ops.cu torch_binding_cuda.cpp)
endif()

add_library(nve-torch-ops SHARED ${nve_torch_ops_src})

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

# The `torch` umbrella target (in ${TORCH_LIBRARIES}) forces libtorch_cuda.so NEEDED via
# -Wl,--no-as-needed even when no CUDA symbol is used (its interface is
# "torch_cpu_library;torch_cuda_library"). For a driverless build, link only the CPU torch
# interface so libnve-torch-ops.so carries no libtorch_cuda dependency.
if (NVE_DRIVERLESS_BUILD)
    set(nve_torch_link_libs torch_cpu_library c10)
else()
    set(nve_torch_link_libs ${TORCH_LIBRARIES})
endif()

# Torch headers call dlsym (libdl before glibc 2.34); PUBLIC so executables that
# include the same headers (e.g. nve_inference_driverless) link it as well.
target_link_libraries(nve-torch-ops
    PUBLIC  nve-common ${CMAKE_DL_LIBS}
    PRIVATE ${nve_torch_link_libs}
)
