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

#include <numa_support.hpp>

#if NVE_WITH_LIBHWLOC
#include <algorithm>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <hwloc.h>
#pragma GCC diagnostic pop

#define NVE_CHECK_HWLOC_(_expr_) \
  NVE_CHECK_((_expr_), "hwloc error #", errno, " = ", std::strerror(errno));

constexpr int32_t numa_locality{
#if HWLOC_VERSION_MAJOR * 1'000'000 + HWLOC_VERSION_MINOR * 1'000 + HWLOC_VERSION_RELEASE >= \
    2'012'001
    HWLOC_LOCAL_NUMANODE_FLAG_INTERSECT_LOCALITY
#else
    HWLOC_LOCAL_NUMANODE_FLAG_SMALLER_LOCALITY
#endif
};

#elif NVE_WITH_LIBNUMA
#include <numa.h>

#include <iterator>
#include <set>

#define NVE_CHECK_LIBNUMA_(_expr_) \
  NVE_CHECK_((_expr_), "libnuma error #", errno, " = ", std::strerror(errno));

#else
#include <sys/sysinfo.h>

#include <thread>
#endif

namespace nve {

NumaLibrary_t numa_library() noexcept {
#if NVE_WITH_LIBHWLOC
  return NumaLibrary_t::LibHwloc;
#elif NVE_WITH_LIBNUMA
  return NumaLibrary_t::LibNuma;
#else
  return NumaLibrary_t::None;
#endif
}

#if NVE_WITH_LIBHWLOC
static std::unique_ptr<hwloc_topology, void (*)(hwloc_topology_t)> topo{
    []() -> hwloc_topology_t {
      hwloc_topology_t t;
      NVE_CHECK_HWLOC_(hwloc_topology_init(&t) == 0);
      NVE_CHECK_HWLOC_(hwloc_topology_load(t) == 0);
      return t;
    }(),
    hwloc_topology_destroy};

#elif NVE_WITH_LIBNUMA
using numa_bitmap_ptr = std::unique_ptr<bitmask, decltype(&numa_free_cpumask)>;
#endif

int64_t num_numa_nodes() {
#if NVE_WITH_LIBHWLOC
  const int64_t n{hwloc_get_nbobjs_by_type(topo.get(), HWLOC_OBJ_NUMANODE)};
  NVE_CHECK_HWLOC_(n > 0);
  return n;

#elif NVE_WITH_LIBNUMA
  NVE_CHECK_LIBNUMA_(numa_available() >= 0);
  return numa_num_configured_nodes();

#else
  return 1;
#endif
}

int64_t numa_node_logical_cores(const int64_t node_idx) {
  NVE_CHECK_(node_idx >= 0 && node_idx < num_numa_nodes());

#if NVE_WITH_LIBHWLOC
  const hwloc_obj_t obj{
      hwloc_get_obj_by_type(topo.get(), HWLOC_OBJ_NUMANODE, static_cast<uint32_t>(node_idx))};
  NVE_CHECK_HWLOC_(obj != nullptr);

  return std::max(hwloc_bitmap_weight(obj->cpuset), 0);

#elif NVE_WITH_LIBNUMA
  numa_bitmap_ptr cpus{numa_allocate_cpumask(), numa_free_cpumask};
  NVE_CHECK_LIBNUMA_(numa_node_to_cpus(static_cast<int32_t>(node_idx), cpus.get()) == 0);

  int64_t n{};
  for (uint32_t i{}; i < cpus->size; ++i) {
    n += numa_bitmask_isbitset(cpus.get(), i);
  }
  return n;

#else
  return std::thread::hardware_concurrency();
#endif
}

int64_t numa_node_memory_size(const int64_t node_idx) {
  NVE_CHECK_(node_idx >= 0 && node_idx < num_numa_nodes());

#if NVE_WITH_LIBHWLOC
  const hwloc_obj_t obj{
      hwloc_get_obj_by_type(topo.get(), HWLOC_OBJ_NUMANODE, static_cast<uint32_t>(node_idx))};
  NVE_CHECK_HWLOC_(obj != nullptr);

  return static_cast<int64_t>(obj->total_memory);

#elif NVE_WITH_LIBNUMA
  return std::max(numa_node_size64(static_cast<int>(node_idx), nullptr), {});

#else
  struct sysinfo info;
  if (sysinfo(&info) == 0) {
    return static_cast<int64_t>(info.totalram) * info.mem_unit;
  } else {
    return std::numeric_limits<int64_t>::max();
  }
#endif
}

int64_t num_cpu_sockets() {
#if NVE_WITH_LIBHWLOC
  const int64_t n{hwloc_get_nbobjs_by_type(topo.get(), HWLOC_OBJ_PACKAGE)};
  NVE_CHECK_HWLOC_(n > 0);
  return n;

#elif NVE_WITH_LIBNUMA
  std::set<int64_t> cpus;
  for (int32_t i{}; i < numa_num_configured_cpus(); ++i) {
    const int32_t node_idx{numa_node_of_cpu(i)};
    NVE_CHECK_LIBNUMA_(node_idx >= 0);
    cpus.emplace(node_idx);
  }
  return static_cast<int64_t>(cpus.size());

#else
  return num_numa_nodes();
#endif
}

int64_t cpu_socket_num_numa_nodes(int64_t socket_idx) {
  NVE_CHECK_(socket_idx >= 0 && socket_idx < num_cpu_sockets());

#if NVE_WITH_LIBHWLOC
  const hwloc_obj_t obj{
      hwloc_get_obj_by_type(topo.get(), HWLOC_OBJ_PACKAGE, static_cast<uint32_t>(socket_idx))};
  NVE_CHECK_HWLOC_(obj != nullptr);

  hwloc_location loc;
  loc.type = HWLOC_LOCATION_TYPE_OBJECT;
  loc.location.object = obj;

  uint32_t n;
  NVE_CHECK_HWLOC_(hwloc_get_local_numanode_objs(topo.get(), &loc, &n, nullptr, numa_locality) ==
                   0);
  return static_cast<int64_t>(n);

#else
  return 1;
#endif
}

std::vector<int64_t> cpu_socket_numa_nodes(const int64_t socket_idx) {
  NVE_CHECK_(socket_idx >= 0 && socket_idx < num_cpu_sockets());

#if NVE_WITH_LIBHWLOC
  const hwloc_obj_t obj{
      hwloc_get_obj_by_type(topo.get(), HWLOC_OBJ_PACKAGE, static_cast<uint32_t>(socket_idx))};
  NVE_CHECK_HWLOC_(obj != nullptr);

  hwloc_location loc;
  loc.type = HWLOC_LOCATION_TYPE_OBJECT;
  loc.location.object = obj;

  uint32_t n{};
  NVE_CHECK_HWLOC_(hwloc_get_local_numanode_objs(topo.get(), &loc, &n, nullptr, numa_locality) ==
                   0);

  std::vector<hwloc_obj_t> nodes(n);
  NVE_CHECK_HWLOC_(
      hwloc_get_local_numanode_objs(topo.get(), &loc, &n, nodes.data(), numa_locality) == 0);

  std::vector<int64_t> indices(n);
  std::transform(nodes.begin(), nodes.end(), indices.begin(),
                 [](const hwloc_obj_t obj) { return static_cast<int64_t>(obj->logical_index); });
  return indices;

#elif NVE_WITH_LIBNUMA
  std::set<int64_t> cpus;
  for (int32_t i{}; i < numa_num_configured_cpus(); ++i) {
    const int32_t node_idx{numa_node_of_cpu(i)};
    NVE_CHECK_LIBNUMA_(node_idx >= 0);
    cpus.emplace(node_idx);
  }
  auto it{cpus.begin()};
  std::advance(it, socket_idx);
  return {*it};

#else
  return {socket_idx};
#endif
}

void bind_thread_to_numa_node(const int64_t node_idx) {
  NVE_CHECK_(node_idx >= 0 && node_idx < num_numa_nodes());
  NVE_LOG_INFO_("Attempting to bind thread '", this_thread_name(), "' to NUMA node #", node_idx);

#if NVE_WITH_LIBHWLOC
  const hwloc_obj_t obj{
      hwloc_get_obj_by_type(topo.get(), HWLOC_OBJ_NUMANODE, static_cast<unsigned>(node_idx))};
  NVE_CHECK_HWLOC_(obj != nullptr);

  NVE_CHECK_HWLOC_(hwloc_set_cpubind(topo.get(), obj->cpuset, HWLOC_CPUBIND_THREAD) == 0);
  NVE_CHECK_HWLOC_(hwloc_set_membind(topo.get(), obj->nodeset, HWLOC_MEMBIND_DEFAULT,
                                     HWLOC_MEMBIND_THREAD | HWLOC_MEMBIND_BYNODESET) == 0);

#elif NVE_WITH_LIBNUMA
  NVE_CHECK_LIBNUMA_(numa_run_on_node(static_cast<int>(node_idx)) == 0);
  numa_set_localalloc();

#else
  (void)node_idx;
#endif
}

}  // namespace nve
