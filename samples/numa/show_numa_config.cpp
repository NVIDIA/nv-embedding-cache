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

#include <bit_ops.hpp>
#include <numa_support.hpp>
#include <iostream>
#include <iomanip>

using namespace nve;

int main() {
  const int64_t numa_nodes{num_numa_nodes()};
  std::cout << "NUMA nodes: " << numa_nodes << std::endl;
  
  for (int64_t i{}; i < numa_nodes; ++i) {
    std::cout << "  " << std::setw(2) << i << ": "
      << std::setw(4) << numa_node_logical_cores(i) << " cores, "
      << std::setw(4) << numa_node_memory_size(i) / (INT64_C(1) << 30) << " GiB\n";
  }
  std::cout << '\n';

  const int64_t cpu_sockets{num_cpu_sockets()};
  std::cout << "CPU sockets: " << cpu_sockets << std::endl;
  for (int64_t i{}; i < cpu_sockets; ++i) {
    const std::vector<int64_t> numa_nodes{cpu_socket_numa_nodes(i)};
    std::cout << "  " << std::setw(2) << i << ": numa nodes [ ";
    for (size_t j{}; j < numa_nodes.size(); ++j) {
      std::cout << (j ? ", " : "") << std::setw(2) << numa_nodes[j];
    }
    std::cout << " ]\n";
  }

  return 0;
}
 