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

add_library(nve-torch-ops SHARED
    nve_registry.cu
    torch_binding.cpp
    nve_torch_ops.cu
)

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
