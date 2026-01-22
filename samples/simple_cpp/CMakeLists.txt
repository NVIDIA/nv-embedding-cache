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

add_executable(simple_cpp simple_cpp.cu)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(simple_cpp PUBLIC -lcuda)
target_link_libraries(simple_cpp PUBLIC nve-common)
target_compile_options(simple_cpp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${NVCC_BASE_FLAGS} >)
target_compile_options(simple_cpp PRIVATE $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>: ${NVCC_DEBUG_FLAGS} >)
target_compile_options(simple_cpp PRIVATE $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>: ${NVCC_RELEASE_FLAGS} >)
