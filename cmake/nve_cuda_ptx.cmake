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

# nve_drop_unused_ptx(<archs_var>)
#
# Rewrites the CUDA architecture list in <archs_var> (CMAKE_CUDA_ARCHITECTURES syntax) so PTX is
# only embedded where a GPU could use it. A GPU runs SASS built for its own major architecture at an
# equal or lower minor version, and otherwise JIT-compiles the newest embedded PTX that is not above
# its own architecture. PTX for an architecture is therefore never used when the next higher
# plain-number architecture in the list (suffixed entries are ignored when finding it) is in the
# same major family, or is the first one of the next family (X.0): every GPU in between runs this
# architecture's SASS. Those entries become "<arch>-real" (SASS only). PTX is kept before gaps
# (e.g. 100 when the next is 120, for 11.x GPUs) and for the newest architecture, except in a
# single-architecture list (usually the build machine's own GPU), which gets SASS only.
# Entries that are not plain numbers (e.g. 90a, 89-real, native) are left as given.
function(nve_drop_unused_ptx archs_var)
    set(archs ${${archs_var}})
    set(numeric ${archs})
    list(FILTER numeric INCLUDE REGEX "^[0-9]+$")
    set(result "")
    foreach(arch IN LISTS archs)
        set(next "")
        if (arch MATCHES "^[0-9]+$")
            foreach(other IN LISTS numeric)
                if (other GREATER arch AND (next STREQUAL "" OR other LESS next))
                    set(next ${other})
                endif()
            endforeach()
        endif()
        set(entry ${arch})
        if (NOT next STREQUAL "")
            math(EXPR major "${arch} / 10")
            math(EXPR next_family "${major} + 1")
            math(EXPR next_major "${next} / 10")
            math(EXPR next_minor "${next} % 10")
            if (next_major EQUAL major OR (next_major EQUAL next_family AND next_minor EQUAL 0))
                set(entry ${arch}-real)
            endif()
        endif()
        list(APPEND result ${entry})
    endforeach()
    list(LENGTH archs count)
    if (count EQUAL 1 AND archs MATCHES "^[0-9]+$")
        set(result ${archs}-real)
    endif()
    set(${archs_var} ${result} PARENT_SCOPE)
endfunction()
