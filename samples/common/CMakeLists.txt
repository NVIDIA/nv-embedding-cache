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

file(GLOB SOURCE_FILES
     *.h
     *.cpp
     *.cu
     *.cuh
)

add_library(nve-samples-common STATIC ${SOURCE_FILES})
target_link_libraries(nve-samples-common PUBLIC -lnvinfer -lcuda)
target_compile_options(nve-samples-common PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${NVCC_BASE_FLAGS} >)
target_compile_options(nve-samples-common PRIVATE $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>: ${NVCC_DEBUG_FLAGS} >)
target_compile_options(nve-samples-common PRIVATE $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>: ${NVCC_RELEASE_FLAGS} >)
