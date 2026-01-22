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

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include/ecache)
include_directories(${PROJECT_SOURCE_DIR}/samples/common)

add_executable(ecache_test api_test.cu)
target_link_libraries(ecache_test PUBLIC gtest_main nve-common)

add_executable(wrapper_test 
  wrapper_test.cu
  wrapper_test_large.cu
  ${PROJECT_SOURCE_DIR}/samples/ecache/wrapper_sample/cache_wrapper.cu
  ${PROJECT_SOURCE_DIR}/samples/ecache/wrapper_sample/communicator.cu)
target_link_libraries(wrapper_test PUBLIC gtest_main nccl nve-common)
