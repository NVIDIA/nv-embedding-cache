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

#include <host_table.hpp>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace nve {

struct STLContainerTableConfig : public HostTableConfig {
  using base_type = HostTableConfig;

  int64_t num_partitions{1};  // Number of partitions to create (Must be a power of 2).
  Partitioner_t partitioner{default_partitioner};  // Partitioner to use.
  std::vector<int64_t> workgroups{0};  // Workgroup to use per partition (thread pool feature). Will wrap around.
  int64_t max_find_task_size{128};  // Maximum number of masks to parse per "find" task (Must be a power of 2.

  int64_t value_alignment{
      sizeof(float)};  // Alignment of stored value/meta tuples in memory. Must be a power of 2.
  int64_t allocation_rate{INT64_C(64) * 1024 *
                          1024};  // Rate at which we allocate more memory for storing
                                  // values. Must be > `value_size + meta_size()`.

  bool prefetch_values{true};  // Issue software prefetches on values.

  OverflowPolicyConfig overflow_policy;  // Overflow detection / handling parameters.

  void check() const;

  inline int64_t meta_size() const noexcept { return overflow_policy.meta_size(); }

  inline int64_t slot_size() const noexcept { return this->max_value_size + meta_size(); }

  inline int64_t slot_stride() const noexcept { return next_aligned(slot_size(), value_alignment); }
};

void from_json(const nlohmann::json& json, STLContainerTableConfig& conf);

void to_json(nlohmann::json& json, const STLContainerTableConfig& conf);

/**
 * Dummy implementation that is tuned for readability / clarity, rather than performance. Useful as
 * a last resort, if plugins are not available, and used to sanity check more advanced
 * implementations，
 */
template <typename ConfigType, typename MaskType, typename KeyType, typename MetaType,
          typename PartitionerType>
class STLContainerTable : public HostTable<ConfigType> {
 public:
  using base_type = HostTable<ConfigType>;
  using config_type = typename base_type::config_type;
  using map_type = typename config_type::template map_type<KeyType>;
  using mask_type = MaskType;
  using mask_repr_type = typename mask_type::repr_type;
  using key_type = typename map_type::key_type;
  using meta_type = MetaType;
  static constexpr PartitionerType partitioner{};

  NVE_PREVENT_COPY_AND_MOVE_(STLContainerTable);

  STLContainerTable() = delete;

  STLContainerTable(const table_id_t id, const config_type& config);

  ~STLContainerTable() override = default;

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
  template <bool PrefetchValues, bool WithValues, bool WithValueSizes>
  int64_t find_(context_ptr_t& ctx, int64_t n, const key_type* keys, mask_repr_type* hit_mask,
                int64_t value_stride, char* values, int64_t* value_sizes) const;

 protected:
  struct Partition final {
    NVE_PREVENT_COPY_AND_MOVE_(Partition);

    Partition() = default;

    mutable std::shared_mutex read_write alignas(cpu_cache_line_size);
    map_type slot_map alignas(cpu_cache_line_size);
    std::vector<char*> available_slots;
    // TODO: Switch to aligned host memory allocator.
    std::vector<std::vector<char>> slot_buffers;
  };

  std::vector<Partition> parts_;
};

struct STLContainerTableFactoryConfig : public HostTableFactoryConfig {
  using base_type = HostTableFactoryConfig;

  void check() const;
};

void from_json(const nlohmann::json& json, STLContainerTableFactoryConfig& conf);

void to_json(nlohmann::json& json, const STLContainerTableFactoryConfig& conf);

template <typename ConfigType, typename TableConfigType>
using STLContainerTableFactory = HostTableFactory<ConfigType, TableConfigType>;

struct STLMapTableConfig : public STLContainerTableConfig {
  using base_type = STLContainerTableConfig;

  template <typename KeyType>
  using map_type = std::unordered_map<KeyType, char*>;

  void check() const;
};

void from_json(const nlohmann::json& json, STLMapTableConfig& conf);

void to_json(nlohmann::json& json, const STLMapTableConfig& conf);

template <typename MaskType, typename KeyType, typename MetaType, typename PartitionerType>
using STLMapTable = STLContainerTable<STLMapTableConfig, MaskType, KeyType, MetaType, PartitionerType>;

struct STLMapTableFactoryConfig : public STLContainerTableFactoryConfig {
  using base_type = STLContainerTableFactoryConfig;

  void check() const;
};

void from_json(const nlohmann::json& json, STLMapTableFactoryConfig& conf);

void to_json(nlohmann::json& json, const STLMapTableFactoryConfig& conf);

class STLMapTableFactory : public STLContainerTableFactory<STLMapTableFactoryConfig, STLMapTableConfig> {
 public:
  using base_type = STLContainerTableFactory<STLMapTableFactoryConfig, STLMapTableConfig>;

  NVE_PREVENT_COPY_AND_MOVE_(STLMapTableFactory);

  STLMapTableFactory() = delete;

  STLMapTableFactory(const config_type& config);

  virtual ~STLMapTableFactory() = default;

  virtual host_table_ptr_t produce(table_id_t id, const table_config_type& config) override;
};

}  // namespace nve
