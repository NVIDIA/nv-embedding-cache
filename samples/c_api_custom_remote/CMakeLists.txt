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

# The custom_remote plugin (libnve-plugin-custom_remote.so) is built in
# samples/common/custom_remote_plugin (added to the build before this
# directory in samples/CMakeLists.txt) and shared with the pytorch and
# cpp_inference_custom_ps samples. The sample loads it at runtime via
# nve_load_host_table_plugin("libnve-plugin-custom_remote.so"), so no explicit link step is
# needed here.

# Sample executable
add_executable(c_api_custom_remote_sample c_api_custom_remote_sample.cpp)
target_include_directories(c_api_custom_remote_sample PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(c_api_custom_remote_sample PUBLIC nve-common)
add_dependencies(c_api_custom_remote_sample nve-plugin-custom_remote)
