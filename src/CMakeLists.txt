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

cmake_minimum_required(VERSION 3.18)

# Generate the split FindAndCombine kernel instantiations. FAC_GEN_DIR is set in the top-level
# CMakeLists.txt (and added to the global include path there).
set(FAC_GEN_SCRIPT ${PROJECT_SOURCE_DIR}/cmake/gen_find_and_combine_instantiations.py)
# Number of generated .cu translation units the instantiation matrix is split across. Each TU pays a
# fixed header-parse cost, so a few large TUs build fastest; the default keeps each generated TU off
# the nve-common critical path. Defaults to the constants at the head of the generator script.
set(FAC_NUM_TUS "" CACHE STRING "Override number of FindAndCombine generated .cu files (blank = script default)")
if (FAC_NUM_TUS)
  set(_fac_gen_args --num-tus ${FAC_NUM_TUS})
endif()
if (NOT NVE_OPTIMIZE_BIN_SIZE)
  list(APPEND _fac_gen_args --specialize-flags)
endif()

# Configure-time generation so the generated source list exists for this configure run.
execute_process(
  COMMAND ${Python3_EXECUTABLE} ${FAC_GEN_SCRIPT} --out-dir ${FAC_GEN_DIR} ${_fac_gen_args}
  RESULT_VARIABLE _fac_rc)
if (NOT _fac_rc EQUAL 0)
  message(FATAL_ERROR "FindAndCombine instantiation generation failed (${_fac_rc})")
endif()
include(${FAC_GEN_DIR}/find_and_combine_generated.cmake)  # -> FAC_GENERATED_SOURCES

# Regenerate (and re-read FAC_GENERATED_SOURCES) by re-running CMake configure whenever the
# generator or the kernel template header changes. The execute_process above does the actual
# generation; listing the inputs here triggers a reconfigure when they change, so generation runs
# exactly once per relevant change. This is also what makes a signature change safe: the generator
# prunes stale inst_*.cu files and rewrites the source list, and the refreshed list replaces the one
# snapshotted above instead of referencing a now-deleted file. (No build-time add_custom_command:
# it would re-run the generator redundantly with this reconfigure, and a multi-output custom command
# under the Makefiles generator can fire more than once per build.)
set_property(DIRECTORY ${CMAKE_SOURCE_DIR} APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${FAC_GEN_SCRIPT} ${PROJECT_SOURCE_DIR}/cuda_ops/find_and_combine_kernel.cuh)

# The generated sources already exist on disk (written by execute_process above), so they are
# deliberately NOT marked GENERATED: that property makes the Ninja generator emit a no-op build
# edge per file, which `ninja -t clean` would delete without being able to regenerate.

file(GLOB common_src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.cu)
add_library(nve-common SHARED ${common_src} ${FAC_GENERATED_SOURCES})
target_compile_features(nve-common PUBLIC cxx_std_17)

# In a driverless build the driver-API translation units (memblock_nvl.cpp, distributed.cpp)
# compile to throwing stubs guarded by this define, and CUDA::cuda_driver is dropped below.
if (NVE_DRIVERLESS_BUILD)
  target_compile_definitions(nve-common PRIVATE NVE_DRIVERLESS_BUILD)
endif()

target_compile_definitions(nve-common PRIVATE
  NVE_VERSION_MAJOR=${NVE_VERSION_MAJOR}
  NVE_VERSION_MINOR=${NVE_VERSION_MINOR}
  NVE_VERSION_PATCH=${NVE_VERSION_PATCH}
)

target_include_directories(nve-common PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/cuembed
)
target_link_libraries(nve-common PUBLIC
  CUDA::cudart
)

# MPI Support
if (NVE_DISABLE_MPI)
  target_compile_definitions(nve-common PRIVATE NVE_DISABLE_MPI)
else()
  find_package(MPI REQUIRED)
  target_include_directories(nve-common PRIVATE ${MPI_INCLUDE_PATH})
  target_link_libraries(nve-common PUBLIC MPI::MPI_CXX)
endif()

# link to cuda driver only on non driverless build
if (NOT NVE_DRIVERLESS_BUILD)
  target_link_libraries(nve-common PUBLIC CUDA::cuda_driver)
endif()

if (${NVE_WITH_LIBHWLOC})
  target_include_directories(nve-common PUBLIC ${hwloc_INCLUDE_DIRS})
  target_link_directories(nve-common PUBLIC ${hwloc_LIBRARIES})
  target_link_libraries(nve-common PUBLIC hwloc)
endif()
if (${NVE_WITH_LIBNUMA})
  target_include_directories(nve-common PUBLIC ${numa_INCLUDE_DIRS})
  target_link_directories(nve-common PUBLIC ${numa_LIBRARIES})
  target_link_libraries(nve-common PUBLIC numa)
endif()
