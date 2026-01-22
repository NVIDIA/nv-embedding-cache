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

message("NV Embedding Cache Plugin: nvHashMap")

include_directories(
  ${NVE_THIRD_PARTY_DIR}/nvhashmap/include
  ${CMAKE_BINARY_DIR}/include  # Needed for nvhashmap_conf.hpp, which is created during CMake.
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

file(GLOB nve_plugin_nvhm_src "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(nve-plugin-nvhm SHARED ${nve_plugin_nvhm_src})
target_link_libraries(nve-plugin-nvhm PUBLIC nve-common)

if(NOT DEFINED NVE_DISABLE_TESTS_AND_SAMPLES)
  add_subdirectory(tests)
endif()
