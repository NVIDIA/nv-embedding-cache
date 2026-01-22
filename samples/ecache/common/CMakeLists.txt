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

find_package(CUDAToolkit REQUIRED)

# Create separate lib for the cupti sample code so we can disable warnings specifically for it
# CUPTI dependencies
if(DEFINED CUPTI_ROOT)
    message(STATUS "CUPTI_ROOT defined as " ${CUPTI_ROOT})
    set(CUPTI_INC "${CUPTI_ROOT}/include" "${CUPTI_ROOT}/samples/common" "${CUPTI_ROOT}/samples/extensions/include/c_util/" "${CUPTI_ROOT}/samples/extensions/include/profilerhost_util/")
    set(CUPTI_LIB "${CUPTI_ROOT}/lib64")
set(CUPTI_SRC "${CUPTI_ROOT}/samples/extensions/src/profilerhost_util/")
else()
    # Get target compute capability from current platform
    message(STATUS "CUPTI_ROOT not defined, using files from CUDA toolkit")
    if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 12 AND CUDAToolkit_VERSION_MINOR GREATER_EQUAL 5)
        set(CUPTI_INC "${CUDAToolkit_LIBRARY_ROOT}/targets/${CMAKE_HOST_SYSTEM_PROCESSOR}-linux/include" "${CUDAToolkit_INCLUDE_DIR}" "${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/samples/common/" "${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/samples/extensions/include/c_util/" "${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/samples/extensions/include/profilerhost_util/")
        set(CUPTI_LIB "${CUDAToolkit_LIBRARY_ROOT}/targets/${CMAKE_HOST_SYSTEM_PROCESSOR}-linux/lib")
    else()
        set(CUPTI_INC "${CUDAToolkit_INCLUDE_DIR}" "${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/samples/common/" "${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/samples/extensions/include/c_util/" "${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/samples/extensions/include/profilerhost_util/")
        set(CUPTI_LIB "${CUDAToolkit_LIBRARY_ROOT}/lib64")
    endif()
set(CUPTI_SRC "${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/samples/extensions/src/profilerhost_util/")
endif()

file(GLOB CUPTI_EXAMPLE_SRC
    ${CUPTI_SRC}/*.cpp
)
add_library(samples_common_cupti STATIC ${CUPTI_EXAMPLE_SRC})
message(STATUS "CUPTI_INC = " "${CUPTI_INC}")
message(STATUS "CUPTI_SRC = " "${CUPTI_SRC}")
message(STATUS "CUPTI_LIB = " "${CUPTI_LIB}")
target_include_directories(samples_common_cupti BEFORE PUBLIC "${CUPTI_INC}")
target_include_directories(samples_common_cupti PUBLIC ${CUDAToolkit_INCLUDE_DIR})
target_link_directories(samples_common_cupti BEFORE PUBLIC ${CUPTI_LIB})
target_link_libraries(samples_common_cupti PUBLIC -lcupti -lnvperf_host -lnvperf_target)
target_compile_options(samples_common_cupti PRIVATE -w)


file(GLOB SOURCE_FILES
     *.h
     *.cpp
     *.cu
     *.cuh
)

add_library(ec_samples_common STATIC ${SOURCE_FILES})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # Needed for .cpp files using cuda

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)

target_link_libraries(ec_samples_common PUBLIC -lcuda -lnvinfer)
target_link_libraries(ec_samples_common PUBLIC samples_common_cupti)
target_compile_options(ec_samples_common PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: ${NVCC_BASE_FLAGS} >)
target_compile_options(ec_samples_common PRIVATE $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>: ${NVCC_DEBUG_FLAGS} >)
target_compile_options(ec_samples_common PRIVATE $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>: ${NVCC_RELEASE_FLAGS} >)

find_library(NVINFER NAMES "nvinfer" "/usr" PATH_SUFFIXES lib)
