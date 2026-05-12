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

# nve_configure_torch()
#
# Validates that PyTorch is installed and its stable ABI is compatible with
# the minimum version requirements (NVE_MIN_TORCH_MAJOR / NVE_MIN_TORCH_MINOR).
#
# Always sets in the caller's scope:
#   NVE_MIN_TORCH_MAJOR / NVE_MIN_TORCH_MINOR — minimum PyTorch version for stable ABI
#   NVE_DISABLE_TORCH_BINDINGS — set to ON if torch was not found or ABI check failed
#   TORCH_ARCH_STR             — space-separated arch list in "M.m" format (e.g. "8.0 8.6")
#                                derived from CMAKE_CUDA_ARCHITECTURES, for use with
#                                TORCH_CUDA_ARCH_LIST (empty string on failure)
#
# On success also sets CMAKE_PREFIX_PATH in the caller's scope so that the
# caller's find_package(Torch REQUIRED) can locate torch. find_package must be
# called by the caller (not inside this function) so that TORCH_FOUND,
# TORCH_LIBRARIES, TORCH_INCLUDE_DIRS, etc. land in the global scope and are
# visible to subdirectories.

function(nve_configure_torch)
    # Minimum PyTorch version required for a stable ABI
    set(NVE_MIN_TORCH_MAJOR 2  PARENT_SCOPE)
    set(NVE_MIN_TORCH_MINOR 10 PARENT_SCOPE)

    # Initialize outputs
    set(TORCH_ARCH_STR "" PARENT_SCOPE)

    if(NVE_DISABLE_TORCH_BINDINGS)
        return()
    endif()

    # Auto-detect torch cmake prefix
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE _torch_prefix
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _torch_prefix_result
    )

    if(NOT (_torch_prefix_result EQUAL 0 AND _torch_prefix))
        message(WARNING "Could not find torch — disabling torch bindings.\n"
                        "Install PyTorch or set NVE_DISABLE_TORCH_BINDINGS=ON to suppress this warning.")
        set(NVE_DISABLE_TORCH_BINDINGS ON PARENT_SCOPE)
        return()
    endif()

    # Verify torch stable ABI is available by compiling a minimal test program.
    # _torch_prefix is e.g. /usr/local/.../torch/share/cmake → torch root is ../../
    get_filename_component(_torch_include "${_torch_prefix}/../../include" ABSOLUTE)
    get_filename_component(_torch_lib     "${_torch_prefix}/../../lib"     ABSOLUTE)
    try_compile(TORCH_STABLE_ABI_OK
        ${CMAKE_BINARY_DIR}/torch_abi_check
        SOURCES ${CMAKE_SOURCE_DIR}/cmake/check_torch_stable_abi.cpp
        CMAKE_FLAGS
            "-DCMAKE_PREFIX_PATH=${_torch_prefix}"
            "-DCMAKE_CXX_STANDARD=17"
            "-DINCLUDE_DIRECTORIES=${_torch_include}"
            "-DLINK_DIRECTORIES=${_torch_lib}"
        LINK_LIBRARIES torch torch_cpu c10
        OUTPUT_VARIABLE _torch_abi_output
    )

    if(NOT TORCH_STABLE_ABI_OK)
        message(WARNING "Torch stable ABI check failed — disabling torch bindings.\n"
                        "Set NVE_DISABLE_TORCH_BINDINGS=ON to suppress this warning.\n"
                        "Output: ${_torch_abi_output}")
        set(NVE_DISABLE_TORCH_BINDINGS ON PARENT_SCOPE)
        return()
    endif()

    message(STATUS "Torch stable ABI check passed — building libnve-torch-ops.so")

    # Propagate CMAKE_PREFIX_PATH so the caller's find_package(Torch REQUIRED) can find torch.
    if(NOT CMAKE_PREFIX_PATH MATCHES "torch")
        set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${_torch_prefix}" PARENT_SCOPE)
        message(STATUS "Auto-detected torch cmake prefix: ${_torch_prefix}")
    endif()

    # Build TORCH_ARCH_STR from CMAKE_CUDA_ARCHITECTURES.
    # TORCH_CUDA_ARCH_LIST expects "7.5;8.0;8.6" format; CMAKE_CUDA_ARCHITECTURES is "75;80;86".
    set(_arch_list "")
    foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
        string(LENGTH "${arch}" _arch_len)
        if(_arch_len EQUAL 2)
            string(SUBSTRING "${arch}" 0 1 _major)
            string(SUBSTRING "${arch}" 1 1 _minor)
        elseif(_arch_len EQUAL 3)
            string(SUBSTRING "${arch}" 0 2 _major)
            string(SUBSTRING "${arch}" 2 1 _minor)
        else()
            continue()
        endif()
        list(APPEND _arch_list "${_major}.${_minor}")
    endforeach()
    string(REPLACE ";" " " _arch_str "${_arch_list}")
    set(TORCH_ARCH_STR "${_arch_str}" PARENT_SCOPE)
endfunction()
