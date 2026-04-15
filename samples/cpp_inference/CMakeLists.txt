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

# C++ inference sample using AOTInductor + NVE.
# Built as part of the main NVE build when samples are enabled.
# No Python or pybind11 dependency at runtime.

if(NOT TORCH_FOUND)
    message(FATAL_ERROR "Torch package was not found!")
endif()

add_executable(nve_inference inference.cu)

target_include_directories(nve_inference PRIVATE
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(nve_inference
    ${TORCH_LIBRARIES}
    nve-torch-ops
    nve-common
)
