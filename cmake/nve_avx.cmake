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

# nve_configure_avx()
#
# Detects AVX support on the current host and configures compiler flags
# accordingly. Respects NVE_DISABLE_AVX512 to skip AVX-512 even when available.
#
# Sets the following variables in the caller's scope:
#   AVX_SUPPORTED        — whether the host supports AVX
#   AVX2_SUPPORTED       — whether the host supports AVX2
#   AVX512F_SUPPORTED    — whether the host supports AVX-512 F
#   AVX512BW_SUPPORTED   — whether the host supports AVX-512 BW
#   AVX512VL_SUPPORTED   — whether the host supports AVX-512 VL
#   AVX512VBMI_SUPPORTED — whether the host supports AVX-512 VBMI
#   NVE_AVX_LEVEL        — selected level: AVX512 | AVX512F | AVX2 | AVX | NONE
#   NVE_AVX_FLAGS        — compiler flags for the selected level (e.g. "-mavx2")
#   CMAKE_C_FLAGS        — extended with NVE_AVX_FLAGS and -DNVE_AVX_LEVEL_<level>=1
#   CMAKE_CXX_FLAGS      — extended with NVE_AVX_FLAGS and -DNVE_AVX_LEVEL_<level>=1
#   CMAKE_CUDA_FLAGS     — extended with NVE_AVX_FLAGS and -DNVE_AVX_LEVEL_<level>=1
#   CMAKE_C_FLAGS_RELEASE   / CMAKE_CXX_FLAGS_RELEASE   / CMAKE_CUDA_FLAGS_RELEASE
#   CMAKE_C_FLAGS_DEBUG     / CMAKE_CXX_FLAGS_DEBUG     / CMAKE_CUDA_FLAGS_DEBUG

function(nve_configure_avx)
    include(CheckCXXSourceCompiles)

    # CMAKE_REQUIRED_FLAGS is local to this function scope — no backup/restore needed.
    set(CMAKE_REQUIRED_FLAGS "-march=native")

    check_cxx_source_compiles("
      #if __AVX__
      int main() { return 0; }
      #else
      #error AVX is not available!
      #endif" AVX_SUPPORTED
    )
    check_cxx_source_compiles("
      #if __AVX2__
      int main() { return 0; }
      #else
      #error AVX2 is not available!
      #endif" AVX2_SUPPORTED
    )
    check_cxx_source_compiles("
      #if __AVX512F__
      int main() { return 0; }
      #else
      #error AVX-512 F is not available!
      #endif" AVX512F_SUPPORTED
    )
    check_cxx_source_compiles("
      #if __AVX512BW__
      int main() { return 0; }
      #else
      #error AVX-512 BW is not available!
      #endif" AVX512BW_SUPPORTED
    )
    check_cxx_source_compiles("
      #if __AVX512VL__
      int main() { return 0; }
      #else
      #error AVX-512 VL is not available!
      #endif" AVX512VL_SUPPORTED
    )
    check_cxx_source_compiles("
      #if __AVX512VBMI__
      int main() { return 0; }
      #else
      #error AVX-512 VBMI is not available!
      #endif" AVX512VBMI_SUPPORTED
    )

    message(STATUS "AVX_SUPPORTED:        ${AVX_SUPPORTED}")
    message(STATUS "AVX2_SUPPORTED:       ${AVX2_SUPPORTED}")
    message(STATUS "AVX512F_SUPPORTED:    ${AVX512F_SUPPORTED}")
    message(STATUS "AVX512BW_SUPPORTED:   ${AVX512BW_SUPPORTED}")
    message(STATUS "AVX512VL_SUPPORTED:   ${AVX512VL_SUPPORTED}")
    message(STATUS "AVX512VBMI_SUPPORTED: ${AVX512VBMI_SUPPORTED}")

    # Propagate detection results for use by third-party subdirectories
    set(AVX_SUPPORTED        ${AVX_SUPPORTED}        PARENT_SCOPE)
    set(AVX2_SUPPORTED       ${AVX2_SUPPORTED}       PARENT_SCOPE)
    set(AVX512F_SUPPORTED    ${AVX512F_SUPPORTED}    PARENT_SCOPE)
    set(AVX512BW_SUPPORTED   ${AVX512BW_SUPPORTED}   PARENT_SCOPE)
    set(AVX512VL_SUPPORTED   ${AVX512VL_SUPPORTED}   PARENT_SCOPE)
    set(AVX512VBMI_SUPPORTED ${AVX512VBMI_SUPPORTED} PARENT_SCOPE)

    # Determine the best AVX level to use
    if(AVX512BW_SUPPORTED AND AVX512VL_SUPPORTED AND NOT NVE_DISABLE_AVX512)
        set(_avx_level "AVX512")
        if(AVX512VBMI_SUPPORTED)
            set(_avx_flags "-mavx512f -mavx512bw -mavx512vl -mavx512vbmi")
            message(STATUS "Using AVX-512 with BW, VL and VBMI extensions")
        else()
            set(_avx_flags "-mavx512f -mavx512bw -mavx512vl")
            message(STATUS "Using AVX-512 with BW and VL extensions, but without VBMI")
        endif()
    elseif(AVX512F_SUPPORTED AND NOT NVE_DISABLE_AVX512)
        set(_avx_level "AVX512F")
        set(_avx_flags "-mavx512f")
        message(STATUS "Using AVX-512F")
    elseif(AVX2_SUPPORTED)
        set(_avx_level "AVX2")
        set(_avx_flags "-mavx2")
        message(STATUS "Using AVX2")
    elseif(AVX_SUPPORTED)
        set(_avx_level "AVX")
        set(_avx_flags "-mavx")
        message(STATUS "Using AVX")
    else()
        set(_avx_level "NONE")
        set(_avx_flags "")
        message(STATUS "No AVX support detected, using standard instructions")
    endif()

    set(NVE_AVX_LEVEL ${_avx_level} PARENT_SCOPE)
    set(NVE_AVX_FLAGS ${_avx_flags} PARENT_SCOPE)

    # Append AVX flags and preprocessor macro to all compiler flag sets
    if(_avx_flags)
        set(CMAKE_C_FLAGS            "${CMAKE_C_FLAGS} ${_avx_flags}"                       PARENT_SCOPE)
        set(CMAKE_CXX_FLAGS          "${CMAKE_CXX_FLAGS} ${_avx_flags}"                     PARENT_SCOPE)
        set(CMAKE_CUDA_FLAGS         "${CMAKE_CUDA_FLAGS} ${_avx_flags}"                    PARENT_SCOPE)
        set(CMAKE_C_FLAGS_RELEASE    "${CMAKE_C_FLAGS_RELEASE} ${_avx_flags}"               PARENT_SCOPE)
        set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} ${_avx_flags}"             PARENT_SCOPE)
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} ${_avx_flags}"            PARENT_SCOPE)
        set(CMAKE_C_FLAGS_DEBUG      "${CMAKE_C_FLAGS_DEBUG} ${_avx_flags}"                 PARENT_SCOPE)
        set(CMAKE_CXX_FLAGS_DEBUG    "${CMAKE_CXX_FLAGS_DEBUG} ${_avx_flags}"               PARENT_SCOPE)
        set(CMAKE_CUDA_FLAGS_DEBUG   "${CMAKE_CUDA_FLAGS_DEBUG} ${_avx_flags}"              PARENT_SCOPE)
    endif()

    message(STATUS "AVX Configuration Summary:")
    message(STATUS "  Detected AVX Level: ${_avx_level}")
    message(STATUS "  AVX Flags: ${_avx_flags}")
endfunction()
