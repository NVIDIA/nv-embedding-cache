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

#include <bit_ops.hpp>
#include <host_table.hpp>
#include <nvhashmap/map.hpp>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace nve {
namespace plugin {

struct NvhmMapTableConfig final : public HostTableConfig {
  using base_type = HostTableConfig;

  int64_t num_partitions{1};  // Number of partitions create (Must be a power of 2).
  Partitioner_t partitioner{default_partitioner};  // Partitioner to use.
  std::vector<int64_t> workgroups{0};  // Workgroup to use per partition (thread pool feature). Will wrap around.
  int64_t max_find_task_size{128};  // Maximum number of masks to parse per "find" task (Must be a power of 2).

  int64_t kernel_size{nvhm::default_kernel_t::size};  // Kernel size to use. Must be in a
                                                                 // power of 2 between [1, 1024].
  int64_t initial_capacity{4096};          // Initial capacity of each map. Must be >= 0.
  int64_t value_alignment{16};  // Alignment of stored values in memory (Must be > 0. For
                                           // performance reasons we require a power of 2).

  int64_t key_fetch_queue_length{8};  // Key prefetching mechanism queue length. Must be in [0, 1, 2, 4, 8].
  bool prefetch_values{true};  // Issue software prefetches on values.

  bool minimize_psl{false};  // Shorten probe search length, if certain conditions apply.
  bool auto_shrink{false};   // Automatically, shrink map to save memory after massive evictions.

  OverflowPolicyConfig overflow_policy;  // Overflow detection / handling parameters.

  void check() const;
};

void from_json(const nlohmann::json& json, NvhmMapTableConfig& conf);

void to_json(nlohmann::json& json, const NvhmMapTableConfig& conf);

template <typename MaskType, typename MapType, typename PartitionerType>
class NvhmMapTable final : public HostTable<NvhmMapTableConfig> {
 public:
  using base_type = HostTable<NvhmMapTableConfig>;
  using mask_type = MaskType;
  using mask_repr_type = typename mask_type::repr_type;
  using map_type = MapType;
  using key_type = typename map_type::key_type;
  using meta_type = std::conditional_t<map_type::has_values, typename map_type::value_type, void>;
  using prefetch_type = typename map_type::prefetch_type;
  using read_pos_type = typename map_type::read_pos_type;
  using write_pos_type = typename map_type::write_pos_type;

  static constexpr PartitionerType partitioner{};
  static constexpr bool minimize_psl{map_type::minimize_psl};
  static constexpr bool auto_shrink{map_type::auto_shrink};

  NVE_PREVENT_COPY_AND_MOVE_(NvhmMapTable);

  NvhmMapTable() = delete;

  NvhmMapTable(const table_id_t id, const NvhmMapTableConfig& config);

  ~NvhmMapTable() override = default;

  void clear(context_ptr_t& ctx) override;

  void erase(context_ptr_t& ctx, int64_t n, const void* keys) override;

  void find(context_ptr_t& ctx, int64_t n, const void* keys, max_bitmask_repr_t* hit_mask,
            int64_t value_stride, void* values, int64_t* value_sizes) const override;

  void insert(context_ptr_t& ctx, int64_t n, const void* keys, int64_t value_stride,
              int64_t value_size, const void* values) override;

  int64_t size(context_ptr_t& ctx, bool exact) const override;

  void update(context_ptr_t& ctx, int64_t n, const void* keys, int64_t value_stride,
              int64_t value_size, const void* values) override;

  void update_accumulate(context_ptr_t& ctx, int64_t n, const void* keys, int64_t update_stride,
                         int64_t update_size, const void* updates,
                         DataType_t update_dtype) override;

 private:
  template <size_t KeyFetchQueueLength, bool PrefetchValues>
  int64_t find_(context_ptr_t& ctx, int64_t n, const key_type* keys, mask_repr_type* hit_mask,
                int64_t value_stride, char* values, int64_t* value_sizes) const;

  template <size_t KeyFetchQueueLength, bool PrefetchValues, bool WithValues, bool WithValueSizes>
  int64_t find_(context_ptr_t& ctx, int64_t n, const key_type* keys, mask_repr_type* hit_mask,
                int64_t value_stride, char* values, int64_t* value_sizes) const;

 private:
  struct Partition final {
    NVE_PREVENT_COPY_AND_MOVE_(Partition);

    Partition() = default;

    mutable std::shared_mutex read_write alignas(cpu_cache_line_size);
    map_type map alignas(cpu_cache_line_size);
  };

  std::vector<Partition> parts_;
};

struct NvhmMapTableFactoryConfig final : public HostTableFactoryConfig {
  using base_type = HostTableFactoryConfig;

  void check() const;
};

void from_json(const nlohmann::json& json, NvhmMapTableFactoryConfig& conf);

void to_json(nlohmann::json& json, const NvhmMapTableFactoryConfig& conf);

class NvhmMapTableFactory final : public HostTableFactory<NvhmMapTableFactoryConfig, NvhmMapTableConfig> {
 public:
  using base_type = HostTableFactory<NvhmMapTableFactoryConfig, NvhmMapTableConfig>;

  NVE_PREVENT_COPY_AND_MOVE_(NvhmMapTableFactory);

  NvhmMapTableFactory() = delete;

  NvhmMapTableFactory(const config_type& config);

  ~NvhmMapTableFactory() override = default;

  host_table_ptr_t produce(table_id_t id, const NvhmMapTableConfig& config) override;
};

}  // namespace plugin
}  // namespace nve
