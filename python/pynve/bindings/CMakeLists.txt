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

# nve.so — pybind11 Python extension module for NVE embedding layers.
# When NVE_DISABLE_TORCH_BINDINGS is OFF (default), links against libnve-torch-ops.so
# to share the NVELayerRegistry singleton and expose register/unregister.

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

file(GLOB SOURCE_FILES *.h *.hpp *.cpp *.cu *.cuh)

add_library(nve MODULE ${SOURCE_FILES})

execute_process(
    COMMAND ${Python_EXECUTABLE} -c
        "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE Python_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set_target_properties(nve PROPERTIES
    PREFIX ""
    SUFFIX "${Python_EXT_SUFFIX}"
)

target_include_directories(nve PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/pybind11/include
    ${Python_INCLUDE_DIRS}
)

target_link_libraries(nve
    PUBLIC  nve-common
    PRIVATE Python::Module
)

if(NOT NVE_DISABLE_TORCH_BINDINGS)
    target_compile_definitions(nve PRIVATE NVE_WITH_TORCH_BINDINGS)
    target_link_libraries(nve PRIVATE -Wl,--no-as-needed nve-torch-ops)
endif()
