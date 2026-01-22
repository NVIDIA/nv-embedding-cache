/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <common.hpp>
#include <vector>

namespace nve {

enum class NumaLibrary_t : uint64_t {
  None,
  LibNuma,
  LibHwloc,
};

static constexpr const char* to_string(const NumaLibrary_t nl) {
  switch (nl) {
    case NumaLibrary_t::None:
      return "none";
    case NumaLibrary_t::LibNuma:
      return "lib_numa";
    case NumaLibrary_t::LibHwloc:
      return "lib_hwloc";
  }
  NVE_THROW_("Unknown NUMA library!");
}

static inline std::ostream& operator<<(std::ostream& o, const NumaLibrary_t nl) {
  return o << to_string(nl);
}

NumaLibrary_t numa_library() noexcept;

int64_t num_numa_nodes();

int64_t numa_node_logical_cores(int64_t node_idx);

int64_t numa_node_memory_size(int64_t node_idx);

int64_t num_cpu_sockets();

int64_t cpu_socket_num_numa_nodes(int64_t socket_idx);

std::vector<int64_t> cpu_socket_numa_nodes(int64_t socket_idx);

void bind_thread_to_numa_node(int64_t node_idx);

}  // namespace nve
