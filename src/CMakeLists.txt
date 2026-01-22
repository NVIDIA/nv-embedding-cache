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

cmake_minimum_required(VERSION 3.18)

find_package(MPI REQUIRED)

file(GLOB_RECURSE common_src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.cu)
add_library(nve-common SHARED ${common_src})
target_compile_features(nve-common PUBLIC cxx_std_17)

target_include_directories(nve-common PRIVATE
  ${MPI_INCLUDE_PATH}
  ${PROJECT_SOURCE_DIR}/third_party/cuembed
)
target_link_libraries(nve-common PUBLIC
  CUDA::cudart
  CUDA::cuda_driver
  MPI::MPI_CXX
)

if (${NVE_WITH_LIBHWLOC})
  target_include_directories(nve-common PUBLIC ${hwloc_INCLUDE_DIRS})
  target_link_directories(nve-common PUBLIC ${hwloc_LIBRARIES})
  target_link_libraries(nve-common PUBLIC hwloc)
endif()
if (${NVE_WITH_LIBNUMA})
  target_include_directories(nve-common PUBLIC ${numa_INCLUDE_DIRS})
  target_link_directories(nve-common PUBLIC ${numa_LIBRARIES})
  target_link_libraries(nve-common PUBLIC numa)
endif()
