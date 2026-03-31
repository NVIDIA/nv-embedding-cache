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

# Plugin: custom_remote (sample remote table backed by std::map)
add_subdirectory(custom_remote_plugin)

# Sample executable
add_executable(c_api_custom_remote_sample c_api_custom_remote_sample.cpp)
target_include_directories(c_api_custom_remote_sample PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(c_api_custom_remote_sample PUBLIC nve-common)
