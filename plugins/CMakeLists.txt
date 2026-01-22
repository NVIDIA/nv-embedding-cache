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

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)

list(FIND NVE_FEATURES "nvhm_plugin" NVE_FEATURE_ENABLED)
if(${NVE_FEATURE_ENABLED} GREATER -1)
  add_subdirectory(nvhm)
endif()
unset(NVE_FEATURE_ENABLED)

list(FIND NVE_FEATURES "abseil_plugin" NVE_FEATURE_ENABLED)
if(${NVE_FEATURE_ENABLED} GREATER -1)
  add_subdirectory(abseil)
endif()
unset(NVE_FEATURE_ENABLED)

list(FIND NVE_FEATURES "phmap_plugin" NVE_FEATURE_ENABLED)
if(${NVE_FEATURE_ENABLED} GREATER -1)
  add_subdirectory(phmap)
endif()
unset(NVE_FEATURE_ENABLED)

list(FIND NVE_FEATURES "redis_plugin" NVE_FEATURE_ENABLED)
if(${NVE_FEATURE_ENABLED} GREATER -1)
  add_subdirectory(redis)
endif()
unset(NVE_FEATURE_ENABLED)

list(FIND NVE_FEATURES "rocksdb_plugin" NVE_FEATURE_ENABLED)
if(${NVE_FEATURE_ENABLED} GREATER -1)
  add_subdirectory(rocksdb)
endif()
unset(NVE_FEATURE_ENABLED)
