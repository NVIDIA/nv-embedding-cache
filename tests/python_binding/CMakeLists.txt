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

if (NOT NVE_DRIVERLESS_BUILD AND NOT NVE_DISABLE_TORCH_BINDINGS AND TORCH_FOUND)
    add_executable(export_inference_test
        export_inference_test.cpp
        layer_config_overrides_test.cpp
    )
    target_link_libraries(export_inference_test PRIVATE gtest_main nve-common)
    target_compile_definitions(export_inference_test PRIVATE
        NVE_TEST_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}"
        NVE_TEST_INFERENCE_EXECUTABLE="$<TARGET_FILE:nve_inference>"
        NVE_TEST_CUSTOM_REMOTE_PLUGIN="$<TARGET_FILE:nve-plugin-custom_remote>"
    )
    add_dependencies(export_inference_test
        nve_inference
        nve-plugin-custom_remote
    )
endif()
