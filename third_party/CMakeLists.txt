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

# Disable warnings in third party code
set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -Wno-nontrivial-memcall") # Needed for RocksDB

add_subdirectory(googletest)

set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
add_subdirectory(json)
add_subdirectory(pybind11)

# Adding plugin dependencies in function to avoid contaminating the CMake scope with internal variables

function(add_nvhashmap)
  set(NVHM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(NVHM_BUILD_BENCH OFF CACHE BOOL "" FORCE)
  set(NVHM_WANT_CACHE_LINE_SIZE ${NVE_CACHE_LINE_SIZE} CACHE STRING "" FORCE)
  set(NVHM_WANT_SSE ON CACHE BOOL "" FORCE)
  set(NVHM_WANT_SSE_PREFETCH OFF CACHE BOOL "" FORCE)
  set(NVHM_WANT_SSE2 ON CACHE BOOL "" FORCE)
  set(NVHM_WANT_SSE3 ON CACHE BOOL "" FORCE)
  set(NVHM_WANT_SSE4 ON CACHE BOOL "" FORCE)
  set(NVHM_WANT_AVX2 ON CACHE BOOL "" FORCE)
  if(${DISABLE_AVX512})
    set(NVHM_WANT_AVX512 OFF CACHE BOOL "" FORCE)
    set(NVHM_WANT_AVX_FVL OFF CACHE BOOL "" FORCE)
    set(NVHM_WANT_AVX_BWVL OFF CACHE BOOL "" FORCE)
    set(NVHM_WANT_AVX_VBMI OFF CACHE BOOL "" FORCE)
  else()
    if(${AVX512F_SUPPORTED} AND ${AVX512BW_SUPPORTED})
      set(NVHM_WANT_AVX_FVL ON CACHE BOOL "" FORCE)
    else()
      set(NVHM_WANT_AVX_FVL OFF CACHE BOOL "" FORCE)
    endif()
    if(${AVX512BW_SUPPORTED} AND ${AVX512VL_SUPPORTED})
      set(NVHM_WANT_AVX_BWVL ON CACHE BOOL "" FORCE)
    else()
      set(NVHM_WANT_AVX_BWVL OFF CACHE BOOL "" FORCE)
    endif()
    if(${AVX512VBMI_SUPPORTED})
      set(NVHM_WANT_AVX_VBMI ON CACHE BOOL "" FORCE)
    else()
      set(NVHM_WANT_AVX_VBMI OFF CACHE BOOL "" FORCE)
    endif()
  endif()
  set(NVHM_WANT_NEON ON CACHE BOOL "" FORCE)
  set(NVHM_WANT_SVE ON CACHE BOOL "" FORCE)
  add_subdirectory(nvhashmap)
endfunction(add_nvhashmap)

function(add_abseil)
  set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(ABSL_BUILD_TEST_HELPERS OFF CACHE BOOL "" FORCE)
  add_subdirectory(abseil-cpp)
endfunction(add_abseil)

function(add_phmap)
  set(PHMAP_MASTER_PROJECT OFF CACHE BOOL "" FORCE)
  set(PHMAP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(PHMAP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  add_subdirectory(parallel-hashmap)
endfunction(add_phmap)

function(add_rocksdb)
  SET(ROCKSDB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
  SET(WITH_BENCHMARK_TOOLS OFF CACHE BOOL "" FORCE)
  SET(WITH_CORE_TOOLS OFF CACHE BOOL "" FORCE)
  SET(WITH_GFLAGS OFF CACHE BOOL "" FORCE)
  SET(WITH_JNI OFF CACHE BOOL "" FORCE)
  SET(WITH_TESTS OFF CACHE BOOL "" FORCE)
  SET(WITH_TOOLS OFF CACHE BOOL "" FORCE)
  SET(WITH_TRACE_TOOLS OFF CACHE BOOL "" FORCE)
  add_subdirectory(rocksdb)
endfunction(add_rocksdb)

function(add_hiredis)
  SET(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  SET(DISABLE_TESTS ON CACHE BOOL "" FORCE)
  SET(ENABLE_SSL ON CACHE BOOL "" FORCE)
  SET(ENABLE_SSL_TESTS OFF CACHE BOOL "" FORCE)
  SET(ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  SET(ENABLE_ASYNC_TESTS OFF CACHE BOOL "" FORCE)
  add_subdirectory(hiredis)
endfunction(add_hiredis)

function(add_redis_plus_plus)
  SET(REDIS_PLUS_PLUS_CXX_STANDARD ${CMAKE_CXX_STANDARD} CACHE STRING "" FORCE)
  SET(REDIS_PLUS_PLUS_USE_TLS ON CACHE BOOL "" FORCE)
  SET(REDIS_PLUS_PLUS_BUILD_STATIC ON CACHE BOOL "" FORCE)
  SET(REDIS_PLUS_PLUS_BUILD_SHARED OFF CACHE BOOL "" FORCE)
  SET(REDIS_PLUS_PLUS_BUILD_TEST OFF CACHE BOOL "" FORCE)
  SET(HIREDIS_HEADER ${NVE_THIRD_PARTY_DIR} CACHE STRING "" FORCE)
  add_subdirectory(redis++)
endfunction(add_redis_plus_plus)

# Add plugin dependencies by selected features

list(FIND NVE_FEATURES "nvhm_plugin" NVE_FEATURE_ENABLED)
if(${NVE_FEATURE_ENABLED} GREATER -1)
  add_nvhashmap()
endif()
unset(NVE_FEATURE_ENABLED)

list(FIND NVE_FEATURES "abseil_plugin" NVE_FEATURE_ENABLED)
if(${NVE_FEATURE_ENABLED} GREATER -1)
  add_abseil()
endif()
unset(NVE_FEATURE_ENABLED)

list(FIND NVE_FEATURES "phmap_plugin" NVE_FEATURE_ENABLED)
if(${NVE_FEATURE_ENABLED} GREATER -1)
  add_phmap()
endif()
unset(NVE_FEATURE_ENABLED)

list(FIND NVE_FEATURES "rocksdb_plugin" NVE_FEATURE_ENABLED)
if(${NVE_FEATURE_ENABLED} GREATER -1)
  add_rocksdb()
endif()
unset(NVE_FEATURE_ENABLED)

list(FIND NVE_FEATURES "redis_plugin" NVE_FEATURE_ENABLED)
if(${NVE_FEATURE_ENABLED} GREATER -1)
  add_hiredis()
  add_redis_plus_plus()
endif()
unset(NVE_FEATURE_ENABLED)
