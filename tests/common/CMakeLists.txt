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
     *.hpp
     *.cpp
     *.cu
     *.cuh
)

include_directories(${PROJECT_SOURCE_DIR}/googletest/googletest/include)

add_executable(logger_test logger_test.cpp)
target_link_libraries(logger_test PUBLIC nve-common)
target_link_libraries(logger_test PUBLIC gtest_main)

add_executable(thread_pool_test thread_pool_test.cpp)
target_link_libraries(thread_pool_test PUBLIC gtest_main nve-common)

add_executable(insert_heuristic_test insert_heuristic_test.cpp)
target_link_libraries(insert_heuristic_test PUBLIC gtest_main nve-common)

add_executable(allocator_test allocator_test.cpp)
target_link_libraries(allocator_test PUBLIC gtest_main nve-common)

find_package(MPI REQUIRED)
add_executable(mpi_buffer_test
  mpi_buffer_test.cpp
)
target_include_directories(mpi_buffer_test PRIVATE
     ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
     ${MPI_INCLUDE_PATH}
)
target_link_libraries(mpi_buffer_test
     gtest_main
     nve-common
     CUDA::cuda_driver
)
