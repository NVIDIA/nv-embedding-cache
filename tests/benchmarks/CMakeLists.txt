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

if (TARGET nve-plugin-linear-file)
  add_executable(single_layer_bench_sanity_test single_layer_bench_sanity_test.cpp)
  target_link_libraries(single_layer_bench_sanity_test PRIVATE gtest_main nve-common CUDA::cudart)
  target_compile_definitions(single_layer_bench_sanity_test PRIVATE
    NVE_SINGLE_LAYER_BENCH_PATH="$<TARGET_FILE:single_layer_bench>")
  add_dependencies(single_layer_bench_sanity_test single_layer_bench nve-plugin-linear-file)
  if (TARGET nve-plugin-nvhm)
    target_compile_definitions(single_layer_bench_sanity_test PRIVATE NVE_HAVE_NVHM)
    add_dependencies(single_layer_bench_sanity_test nve-plugin-nvhm)
  endif()
  if (TARGET nve-plugin-rocksdb)
    target_compile_definitions(single_layer_bench_sanity_test PRIVATE NVE_HAVE_ROCKSDB)
    add_dependencies(single_layer_bench_sanity_test nve-plugin-rocksdb)
  endif()
endif()
