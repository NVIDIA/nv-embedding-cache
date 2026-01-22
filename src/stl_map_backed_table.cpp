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

#include <execution_context.hpp>
#include <host_table_detail.hpp>
#include <stl_map_backed_table_detail.hpp>

namespace nve {

void STLContainerTableConfig::check() const {
  base_type::check();

  NVE_CHECK_(num_partitions >= 0 && has_single_bit(static_cast<uint64_t>(num_partitions)));
  NVE_CHECK_(!workgroups.empty());
  NVE_CHECK_(max_find_task_size > 0);

  NVE_CHECK_(value_alignment >= 0 && has_single_bit(static_cast<uint64_t>(value_alignment)));
  NVE_CHECK_(allocation_rate >= max_value_size);

  overflow_policy.check();
}

void from_json(const nlohmann::json& json, STLContainerTableConfig& conf) {
  using base_type = STLContainerTableConfig::base_type;
  from_json(json, static_cast<base_type&>(conf));

  NVE_READ_JSON_FIELD_(num_partitions);
  NVE_READ_JSON_FIELD_(partitioner);
  NVE_READ_JSON_FIELD_(workgroups);
  NVE_READ_JSON_FIELD_(max_find_task_size);

  NVE_READ_JSON_FIELD_(value_alignment);
  NVE_READ_JSON_FIELD_(allocation_rate);

  NVE_READ_JSON_FIELD_(prefetch_values);

  NVE_READ_JSON_FIELD_(overflow_policy);
}

void to_json(nlohmann::json& json, const STLContainerTableConfig& conf) {
  using base_type = STLContainerTableConfig::base_type;
  to_json(json, static_cast<const base_type&>(conf));

  NVE_WRITE_JSON_FIELD_(num_partitions);
  NVE_WRITE_JSON_FIELD_(partitioner);
  NVE_WRITE_JSON_FIELD_(workgroups);
  NVE_WRITE_JSON_FIELD_(max_find_task_size);

  NVE_WRITE_JSON_FIELD_(value_alignment);
  NVE_WRITE_JSON_FIELD_(allocation_rate);

  NVE_WRITE_JSON_FIELD_(prefetch_values);

  NVE_WRITE_JSON_FIELD_(overflow_policy);
}

void STLContainerTableFactoryConfig::check() const { base_type::check(); }

void from_json(const nlohmann::json& json, STLContainerTableFactoryConfig& conf) {
  using base_type = typename STLContainerTableFactoryConfig::base_type;
  from_json(json, static_cast<base_type&>(conf));
}

void to_json(nlohmann::json& json, const STLContainerTableFactoryConfig& conf) {
  using base_type = typename STLContainerTableFactoryConfig::base_type;
  to_json(json, static_cast<const base_type&>(conf));
}

void STLMapTableConfig::check() const { base_type::check(); }

void from_json(const nlohmann::json& json, STLMapTableConfig& conf) {
  using base_type = STLMapTableConfig::base_type;
  from_json(json, static_cast<base_type&>(conf));
}

void to_json(nlohmann::json& json, const STLMapTableConfig& conf) {
  using base_type = STLMapTableConfig::base_type;
  to_json(json, static_cast<const base_type&>(conf));
}

void STLMapTableFactoryConfig::check() const { base_type::check(); }

void from_json(const nlohmann::json& json, STLMapTableFactoryConfig& conf) {
  using base_type = STLMapTableFactoryConfig::base_type;
  from_json(json, static_cast<base_type&>(conf));
}

void to_json(nlohmann::json& json, const STLMapTableFactoryConfig& conf) {
  using base_type = STLMapTableFactoryConfig::base_type;
  to_json(json, static_cast<const base_type&>(conf));
}

STLMapTableFactory::STLMapTableFactory(const config_type& config) : base_type(config) {}

template <typename MaskType, typename KeyType, typename MetaType>
inline static host_table_ptr_t make_map_table_3(const table_id_t id, const STLMapTableConfig& config) {
  if (config.num_partitions == 1) {
    if (config.partitioner != Partitioner_t::AlwaysZero) {
      NVE_LOG_VERBOSE_("Selected ", config.partitioner, " partitioner was disabled because table has only 1 partition.");
    }
    return std::make_shared<STLMapTable<MaskType, KeyType, MetaType, AlwaysZeroPartitioner>>(id, config);
  }

  switch (config.partitioner) {
#if defined(NVE_FEATURE_HT_PART_FNV1A)
    case Partitioner_t::FowlerNollVo:
      return std::make_shared<STLMapTable<MaskType, KeyType, MetaType, FowlerNollVoPartitioner>>(id, config);
#endif
#if defined(NVE_FEATURE_HT_PART_MURMUR)
    case Partitioner_t::Murmur3:
      return std::make_shared<STLMapTable<MaskType, KeyType, MetaType, Murmur3Partitioner>>(id, config);
#endif
#if defined(NVE_FEATURE_HT_PART_RRXMRRXMSX0)
    case Partitioner_t::Rrxmrrxmsx0:
      return std::make_shared<
          STLMapTable<MaskType, KeyType, MetaType, Rrxmrrxmsx0Partitioner>>(id, config);
#endif
#if defined(NVE_FEATURE_HT_PART_STD_HASH)
    case Partitioner_t::StdHash:
      return std::make_shared<STLMapTable<MaskType, KeyType, MetaType, StdHashPartitioner>>(id, config);
#endif
    default:
      NVE_THROW_("`config.partitioner` (", config.partitioner, ") is out of bounds!");
  }
}

template <typename MaskType, typename KeyType>
static host_table_ptr_t make_map_table_2(const table_id_t id, const STLMapTableConfig& config) {
  switch (config.overflow_policy.handler) {
    case OverflowHandler_t::EvictRandom:
      return make_map_table_3<MaskType, KeyType, void>(id, config);
    case OverflowHandler_t::EvictLRU:
      return make_map_table_3<MaskType, KeyType, std::chrono::system_clock::time_point>(id, config);
    case OverflowHandler_t::EvictLFU:
      return make_map_table_3<MaskType, KeyType, int64_t>(id, config);
  }
  NVE_THROW_("`config.overflow_policy.handler` (", config.overflow_policy.handler,
             ") is out of bounds!");
}

template <typename MaskType>
static host_table_ptr_t make_map_table_1(const table_id_t id, const STLMapTableConfig& config) {
  switch (config.key_size) {
#if defined(NVE_FEATURE_HT_KEY_8)
    case sizeof(int8_t):
      return make_map_table_2<MaskType, int8_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KEY_16)
    case sizeof(int16_t):
      return make_map_table_2<MaskType, int16_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KEY_32)
    case sizeof(int32_t):
      return make_map_table_2<MaskType, int32_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KEY_64)
    case sizeof(int64_t):
      return make_map_table_2<MaskType, int64_t>(id, config);
#endif
  }
  NVE_THROW_("`config.key_size` (", config.key_size, ") is out of bounds!");
}

host_table_ptr_t STLMapTableFactory::produce(const table_id_t id, const STLMapTableConfig& config) {
  switch (config.mask_size) {
#if defined(NVE_FEATURE_HT_MASK_8)
    case bitmask8_t::size:
      return make_map_table_1<bitmask8_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_MASK_16)
    case bitmask16_t::size:
      return make_map_table_1<bitmask16_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_MASK_32)
    case bitmask32_t::size:
      return make_map_table_1<bitmask32_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_MASK_64)
    case bitmask64_t::size:
      return make_map_table_1<bitmask64_t>(id, config);
#endif
  }
  NVE_THROW_("`config.mask_size` (", config.mask_size, ") is out of bounds!");
}

}  // namespace nve
