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

cmake_minimum_required(VERSION 3.18) # Need 3.18 for nvcc target compute arch

# For CMake Debugging
function(dump_all_vars)
  # todo: use "string REGEX MATCH" to print only vars matching regex
  set(VAR_FILE "cmake.vars")
  file(WRITE ${VAR_FILE} "")
  get_cmake_property(_variableNames VARIABLES)
  list (SORT _variableNames)
  foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
    file(APPEND ${VAR_FILE} "${_variableName}=${${_variableName}}\n")
  endforeach()
endfunction(dump_all_vars)

# Read version from version.txt
file(READ "${CMAKE_SOURCE_DIR}/version.txt" NVE_VERSION)
string(STRIP "${NVE_VERSION}" NVE_VERSION)

project(NV_Embedding_Cache LANGUAGES C CXX CUDA VERSION ${NVE_VERSION}) # Version format: Year.Month.Patch

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Supported optional features.
set(NVE_AVAILABLE_FEATURES "")
list(APPEND NVE_AVAILABLE_FEATURES
  "abseil_plugin"
  "nvhm_plugin"
  "phmap_plugin"
  "redis_plugin"
  "rocksdb_plugin"
)
list(APPEND NVE_AVAILABLE_FEATURES "ht_mask_8" "ht_mask_16" "ht_mask_32" "ht_mask_64")
list(APPEND NVE_AVAILABLE_FEATURES "ht_key_8" "ht_key_16" "ht_key_32" "ht_key_64")
list(APPEND NVE_AVAILABLE_FEATURES "ht_kernel_8" "ht_kernel_16" "ht_kernel_32" "ht_kernel_64" "ht_kernel_128" "ht_kernel_256" "ht_kernel_512" "ht_kernel_1024")
list(APPEND NVE_AVAILABLE_FEATURES "ht_part_fnv1a" "ht_part_murmur3" "ht_part_rrxmrrxmsx0" "ht_part_std_hash")

set(NVE_MINIMAL_FEATURES "")
list(APPEND NVE_MINIMAL_FEATURES "ht_mask_64")
list(APPEND NVE_MINIMAL_FEATURES "ht_key_64")
list(APPEND NVE_MINIMAL_FEATURES "ht_part_fnv1a")
list(APPEND NVE_MINIMAL_FEATURES "ht_kernel_64")

set(NVE_DEFAULT_FEATURES ${NVE_MINIMAL_FEATURES})
list(APPEND NVE_DEFAULT_FEATURES
  "abseil_plugin"
  "nvhm_plugin"
  "phmap_plugin"
  "redis_plugin"
  "rocksdb_plugin"
)
list(APPEND NVE_DEFAULT_FEATURES "ht_key_32")
list(APPEND NVE_DEFAULT_FEATURES "ht_kernel_128")

#set(NVE_DEFAULT_FEATURES ${NVE_MINIMAL_FEATURES})  # Default = minimal set of features = fast compile, small binary size.
#set(NVE_DEFAULT_FEATURES ${NVE_AVAILABLE_FEATURES})  # Default = enable all features = slow compile, large binary size.

# Enabled features.
if(DEFINED NVE_FEATURES)
  string(REPLACE " " ";" NVE_FEATURES "${NVE_FEATURES}")
else()
  if(NVE_DISABLE_PLUGINS)
    set(NVE_FEATURES "${NVE_MINIMAL_FEATURES}" CACHE STRING "Enabled minimal features")
  else()
    set(NVE_FEATURES "${NVE_DEFAULT_FEATURES}" CACHE STRING "Enabled standard features")
  endif()
endif()
list(REMOVE_DUPLICATES NVE_FEATURES)

# Option to disable nvhm plugin specifically
option(NVE_DISABLE_NVHM_PLUGIN "Disable the nvhm plugin" OFF)
if(NVE_DISABLE_NVHM_PLUGIN)
  list(REMOVE_ITEM NVE_FEATURES "nvhm_plugin")
  message(STATUS "nvhm_plugin disabled via NVE_DISABLE_NVHM_PLUGIN")
endif()

string(TOLOWER "${NVE_FEATURES}" NVE_FEATURES)

message(STATUS "Enabled optional NVE features:")
foreach(NVE_FEATURE ${NVE_FEATURES})
  message(STATUS "  ${NVE_FEATURE}")
  list(FIND NVE_AVAILABLE_FEATURES "${NVE_FEATURE}" NVE_FEATURE_ENABLED)
  if(${NVE_FEATURE_ENABLED} EQUAL -1)
    message(STATUS "Feature \"${NVE_FEATURE}\" is unknown. Available NVE features are:")
    foreach(NVE_FEATURE ${NVE_AVAILABLE_FEATURES})
      message(STATUS "  ${NVE_FEATURE}")
    endforeach()
    message(FATAL_ERROR "Unsupported feature \"${NVE_FEATURE}\"!")
  endif()
  unset(NVE_FEATURE_ENABLED)
endforeach()
unset(NVE_FEATURE)

# Determine CPU cache line size. (need to set before third_party/nvhashmap)
if(NOT NVE_CACHE_LINE_SIZE)
  if (WIN32)
    set(NVE_CACHE_LINE_SIZE "64")
  else()
    execute_process(
      COMMAND getconf LEVEL1_DCACHE_LINESIZE
      OUTPUT_VARIABLE NVE_CACHE_LINE_SIZE
      COMMAND_ERROR_IS_FATAL ANY
    )
  endif()
endif()
string(STRIP "${NVE_CACHE_LINE_SIZE}" NVE_CACHE_LINE_SIZE)
message(STATUS "Cache line size = ${NVE_CACHE_LINE_SIZE} bytes")

# nve_configure_avx() sets the following variables in this scope:
#   AVX_SUPPORTED / AVX2_SUPPORTED / AVX512F_SUPPORTED /
#   AVX512BW_SUPPORTED / AVX512VL_SUPPORTED / AVX512VBMI_SUPPORTED
#   NVE_AVX_LEVEL        — selected level: AVX512 | AVX512F | AVX2 | AVX | NONE
#   NVE_AVX_FLAGS        — compiler flags for the selected level (e.g. "-mavx2")
#   CMAKE_C_FLAGS / CMAKE_CXX_FLAGS / CMAKE_CUDA_FLAGS (base, _RELEASE, _DEBUG)
#                        — extended with NVE_AVX_FLAGS and -DNVE_AVX_LEVEL_<level>=1
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/nve_avx.cmake)
nve_configure_avx()

# set Release build by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message(STATUS "BUILD TYPE = " ${CMAKE_BUILD_TYPE})

# Export compile commands for clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS "1" CACHE STRING "" FORCE)

# Disable AVX-512 instructions (WA for some CI machines)
if(NVE_DISABLE_AVX512)
  message(STATUS "Disabling AVX-512 instructions")
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -mno-avx512f")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -mno-avx512f")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -mno-avx512f")
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE) # Make sure fPIC is set before third_party so it affects those too

# Compile third party stuff first before we taint the compiler flags to our preference (except build_type, avx512 and fPIC)
set(NVE_THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party CACHE INTERNAL "" FORCE)
add_subdirectory(third_party)

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Compiler = " ${CMAKE_CUDA_COMPILER})
message(STATUS "CUDA Host Compiler = " ${CMAKE_CUDA_HOST_COMPILER})

# Enable warnings.
set(CMAKE_C_FLAGS_RELEASE    "${CMAKE_C_FLAGS_RELEASE}    -Werror -Wall -Wextra -Wconversion")
set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE}  -Werror -Wall -Wextra -Wconversion")

# NVCC additional flags
if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -Wall -Xcompiler -Werror -Xcompiler -Wextra -Xcompiler -Wconversion")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math")
elseif (CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Wall -Werror -Wextra -Wconversion")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unknown-cuda-version -ffast-math -Wno-nan-infinity-disabled")
endif()

# Debug specific flags
set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -O0 -g")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -O0 -g")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -g -G")

set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DNVE_CACHE_LINE_SIZE=${NVE_CACHE_LINE_SIZE}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DNVE_CACHE_LINE_SIZE=${NVE_CACHE_LINE_SIZE}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNVE_CACHE_LINE_SIZE=${NVE_CACHE_LINE_SIZE}")

find_path(hwloc_INCLUDE_DIRS hwloc.h)
find_library(hwloc_LIBRARIES hwloc)
if(NOT ${hwloc_INCLUDE_DIRS} STREQUAL "hwloc_INCLUDE_DIRS-NOTFOUND" AND
   NOT ${hwloc_LIBRARIES} STREQUAL "hwloc_LIBRARIES-NOTFOUND")
  message(STATUS "libhwloc: found @ ${hwloc_LIBRARIES} : ${hwloc_INCLUDE_DIRS}")
  option(NVE_WITH_LIBHWLOC "Enable hwloc NUMA control" ON)
else()
  message(STATUS "libhwloc: not found")
  option(NVE_WITH_LIBHWLOC "Enable hwloc NUMA control" OFF)
endif()
message(STATUS "NVE_WITH_LIBHWLOC: ${NVE_WITH_LIBHWLOC}")
if (${NVE_WITH_LIBHWLOC})
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -DNVE_WITH_LIBHWLOC=1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNVE_WITH_LIBHWLOC=1")
endif()

find_path(numa_INCLUDE_DIRS numa.h)
find_library(numa_LIBRARIES numa)
if(NOT ${numa_INCLUDE_DIRS} STREQUAL "numa_INCLUDE_DIRS-NOTFOUND" AND
   NOT ${numa_LIBRARIES} STREQUAL "numa_LIBRARIES-NOTFOUND")
  message(STATUS "libnuma: found @ ${numa_LIBRARIES}, ${numa_INCLUDE_DIRS}")
  option(NVE_WITH_LIBNUMA "Enable libnuma NUMA control" ON)
else()
  message(STATUS "libnuma: not found")
  option(NVE_WITH_LIBNUMA "Enable libnuma NUMA control" OFF)
endif()
message(STATUS "NVE_WITH_LIBNUMA: ${NVE_WITH_LIBNUMA}")
if (${NVE_WITH_LIBNUMA})
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -DNVE_WITH_LIBNUMA=1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNVE_WITH_LIBNUMA=1")
endif()

if(NOT DEFINED NVE_SM_VERSION)
    # Get target compute capability from current platform
    message(STATUS "NVE_SM_VERSION not defined")
    execute_process (
        COMMAND bash -c "which nvidia-smi"
        RESULT_VARIABLE SMI_FOUND
        OUTPUT_QUIET
    )
    if(SMI_FOUND EQUAL "0")
        execute_process (
            COMMAND bash -c "nvidia-smi --query-gpu=compute_cap --id=0 --format=csv"
            OUTPUT_VARIABLE SMI_COMPUTE
        )
        if(SMI_COMPUTE MATCHES "^No devices were found.*")
          set(NVE_SM_VERSION "80;86;89;90;100")
        else()
          string(REPLACE "compute_cap" "" SMI_COMPUTE ${SMI_COMPUTE})
          string(REPLACE "\n" "" SMI_COMPUTE ${SMI_COMPUTE})
          string(REPLACE "." "" NVE_SM_VERSION ${SMI_COMPUTE})
          message(STATUS "Using NVE_SM_VERSION=" ${NVE_SM_VERSION} " from nvidia-smi")
        endif()
    else()
        set(NVE_SM_VERSION 80 86 90 100)
        message(STATUS "Could not find nvidia-smi! Using default SM versions: " ${NVE_SM_VERSION})
    endif()
endif()

set(CMAKE_CUDA_ARCHITECTURES ${NVE_SM_VERSION} CACHE STRING "CUDA architectures" FORCE)
message(STATUS "CMAKE_CUDA_ARCHITECTURES=" ${CMAKE_CUDA_ARCHITECTURES})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enabling NVE_ORIGIN_RPATH will allow relocated .so files (e.g. during pip install) to start
# the search for .so dependencies in the local directory
if(NVE_ORIGIN_RPATH)
  set(CMAKE_INSTALL_RPATH "\$ORIGIN:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
endif()

include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/third_party/json/single_include
)

# Set compiler switches for optional features.
foreach(NVE_FEATURE ${NVE_FEATURES})
  string(TOUPPER "${NVE_FEATURE}" NVE_FEATURE)
  message(STATUS "${NVE_FEATURE}")
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DNVE_FEATURE_${NVE_FEATURE}=1")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DNVE_FEATURE_${NVE_FEATURE}=1")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNVE_FEATURE_${NVE_FEATURE}=1")
endforeach()
unset(NVE_FEATURE)

if(NVE_ENABLE_NVTX)
  message(STATUS "NVTX enabled")
  add_compile_definitions(NVE_ENABLE_NVTX)
else()
  message(STATUS "NVTX disabled")
endif()

add_subdirectory(src)
add_subdirectory(plugins)

option(NVE_DISABLE_TORCH_BINDINGS
       "Disable building libnve-torch-ops.so with PyTorch custom op bindings" OFF)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/nve_torch.cmake)
# nve_configure_torch() sets the following variables in this scope:
#   NVE_MIN_TORCH_MAJOR      — minimum required PyTorch major version
#   NVE_MIN_TORCH_MINOR      — minimum required PyTorch minor version
#   NVE_DISABLE_TORCH_BINDINGS — set to ON if torch is missing or ABI check fails
#   TORCH_ARCH_STR           — space-separated arch list (e.g. "8.0 8.6") for TORCH_CUDA_ARCH_LIST
#   CMAKE_PREFIX_PATH        — extended with the torch cmake prefix (on success only)
nve_configure_torch()
if(NOT NVE_DISABLE_TORCH_BINDINGS)
    find_package(Torch REQUIRED)
    add_subdirectory(python/pynve/torch_bindings)
endif()

add_subdirectory(python/pynve/bindings)
if(NOT DEFINED NVE_DISABLE_TESTS_AND_SAMPLES)
  message(STATUS "Tests and samples are enabled")
  add_subdirectory(samples)
  add_subdirectory(tests)
else()
  message(STATUS "Tests and samples are disabled (NVE_DISABLE_TESTS_AND_SAMPLES = ${NVE_DISABLE_TESTS_AND_SAMPLES})")
endif()
