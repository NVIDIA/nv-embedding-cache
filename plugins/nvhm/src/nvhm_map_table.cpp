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
#include <nvhm_map_table.hpp>
#include <buffer_wrapper.hpp>
#include <thread_pool.hpp>

// NOTE: The NvhmMapTable template member definitions live in nvhm_map_table_impl.hpp and are
// compiled by the generated nvhm_map_table_inst_*.cpp TUs (explicit instantiations). This TU must
// NOT include the impl header: keeping the member bodies invisible here is what prevents the
// factory chain below from re-instantiating the whole matrix in a single TU. Keep the combination
// space in cmake/gen_nvhm_map_table_instantiations.py in sync with the dispatch chain below.

namespace nve {
namespace plugin {

void NvhmMapTableConfig::check() const {
  base_type::check();

  NVE_CHECK_(num_partitions >= 0 && has_single_bit(to_uint(num_partitions)));
  NVE_CHECK_(!workgroups.empty());
  NVE_CHECK_(max_find_task_size > 0);

  NVE_CHECK_(kernel_size > 0 && kernel_size <= 512 && has_single_bit(to_uint(kernel_size)));

  NVE_CHECK_(key_fetch_queue_length == 0 ||
    (key_fetch_queue_length > 0 && key_fetch_queue_length <= 8 && has_single_bit(to_uint(key_fetch_queue_length))));
  NVE_CHECK_(insert_fetch_queue_length == 0 ||
    (insert_fetch_queue_length > 0 && insert_fetch_queue_length <= 8 && has_single_bit(to_uint(insert_fetch_queue_length))));

  NVE_CHECK_(initial_capacity >= 0);
  NVE_CHECK_(value_alignment > 0 && has_single_bit(to_uint(value_alignment)));

  overflow_policy.check();
}

void from_json(const nlohmann::json& json, NvhmMapTableConfig& conf) {
  using base_type = NvhmMapTableConfig::base_type;
  from_json(json, static_cast<base_type&>(conf));

  NVE_READ_JSON_FIELD_(num_partitions);
  NVE_READ_JSON_FIELD_(partitioner);
  NVE_READ_JSON_FIELD_(workgroups);
  NVE_READ_JSON_FIELD_(max_find_task_size);

  NVE_READ_JSON_FIELD_(kernel_size);
  NVE_READ_JSON_FIELD_(initial_capacity);
  NVE_READ_JSON_FIELD_(value_alignment);

  NVE_READ_JSON_FIELD_(key_fetch_queue_length);
  NVE_READ_JSON_FIELD_(insert_fetch_queue_length);
  NVE_READ_JSON_FIELD_(prefetch_values);

  NVE_READ_JSON_FIELD_(auto_shrink);

  NVE_READ_JSON_FIELD_(overflow_policy);
}

void to_json(nlohmann::json& json, const NvhmMapTableConfig& conf) {
  using base_type = NvhmMapTableConfig::base_type;
  to_json(json, static_cast<const base_type&>(conf));

  NVE_WRITE_JSON_FIELD_(num_partitions);
  NVE_WRITE_JSON_FIELD_(partitioner);
  NVE_WRITE_JSON_FIELD_(workgroups);
  NVE_WRITE_JSON_FIELD_(max_find_task_size);

  NVE_WRITE_JSON_FIELD_(kernel_size);
  NVE_WRITE_JSON_FIELD_(initial_capacity);
  NVE_WRITE_JSON_FIELD_(value_alignment);

  NVE_WRITE_JSON_FIELD_(key_fetch_queue_length);
  NVE_WRITE_JSON_FIELD_(insert_fetch_queue_length);
  NVE_WRITE_JSON_FIELD_(prefetch_values);

  NVE_WRITE_JSON_FIELD_(auto_shrink);

  NVE_WRITE_JSON_FIELD_(overflow_policy);
}

// (Template member definitions moved to nvhm_map_table_impl.hpp; see note at the top.)

void NvhmMapTableFactoryConfig::check() const { base_type::check(); }

void from_json(const nlohmann::json& json, NvhmMapTableFactoryConfig& conf) {
  using base_type = NvhmMapTableFactoryConfig::base_type;
  from_json(json, static_cast<base_type&>(conf));
}

void to_json(nlohmann::json& json, const NvhmMapTableFactoryConfig& conf) {
  using base_type = NvhmMapTableFactoryConfig::base_type;
  to_json(json, static_cast<const base_type&>(conf));
}

NvhmMapTableFactory::NvhmMapTableFactory(const config_type& config) : base_type(config) {}

template <typename MapType>
inline static host_table_ptr_t make_nvhm_map_table_6(const table_id_t id,
                                                  const NvhmMapTableConfig& config) {
  if (config.num_partitions == 1) {
    if (config.partitioner != Partitioner_t::AlwaysZero) {
      NVE_LOG_VERBOSE_("Selected ", config.partitioner, " partitioner was disabled because table has only 1 partition.");
    }
    return std::make_shared<NvhmMapTable<MapType, AlwaysZeroPartitioner>>(id, config);
  }

  switch (config.partitioner) {
#if defined(NVE_FEATURE_HT_PART_FNV1A)
    case Partitioner_t::FowlerNollVo:
      return std::make_shared<NvhmMapTable<MapType, FowlerNollVoPartitioner>>(id, config);
#endif
#if defined(NVE_FEATURE_HT_PART_MURMUR3)
    case Partitioner_t::Murmur3:
      return std::make_shared<NvhmMapTable<MapType, Murmur3Partitioner>>(id, config);
#endif
#if defined(NVE_FEATURE_HT_PART_RRXMRRXMSX0)
    case Partitioner_t::Rrxmrrxmsx0:
      return std::make_shared<NvhmMapTable<MapType, Rrxmrrxmsx0Partitioner>>(id, config);
#endif
#if defined(NVE_FEATURE_HT_PART_STD_HASH)
    case Partitioner_t::StdHash:
      return std::make_shared<NvhmMapTable<MapType, StdHashPartitioner>>(id, config);
#endif
    default:
      NVE_THROW_("`config.partitioner` (", config.partitioner, ") is out of bounds!");
  }
}

template <typename KeyType, typename MetaType, nvhm::flags_t Flags, typename KernelType>
static host_table_ptr_t make_nvhm_map_table_5(const table_id_t id, const NvhmMapTableConfig& config) {
  using map_t = nvhm::map<KeyType, MetaType, Flags, KernelType>;
  return make_nvhm_map_table_6<map_t>(id, config);
}

template <typename KeyType, typename MetaType, nvhm::flags_t Flags>
static host_table_ptr_t make_nvhm_map_table_4(const table_id_t id, const NvhmMapTableConfig& config) {
  switch (config.kernel_size) {
#if defined(NVE_FEATURE_HT_KERNEL_1)
    case 1:
      return make_nvhm_map_table_5<KeyType, MetaType, Flags,
                                        nvhm::default_kernel1_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KERNEL_2)
    case 2:
      return make_nvhm_map_table_5<KeyType, MetaType, Flags,
                                        nvhm::default_kernel2_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KERNEL_4)
    case 4:
      return make_nvhm_map_table_5<KeyType, MetaType, Flags,
                                        nvhm::default_kernel4_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KERNEL_8)
    case 8:
      return make_nvhm_map_table_5<KeyType, MetaType, Flags,
                                        nvhm::default_kernel8_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KERNEL_16)
    case 16:
      return make_nvhm_map_table_5<KeyType, MetaType, Flags,
                                        nvhm::default_kernel16_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KERNEL_32)
    case 32:
      return make_nvhm_map_table_5<KeyType, MetaType, Flags,
                                        nvhm::default_kernel32_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KERNEL_64)
    case 64:
      return make_nvhm_map_table_5<KeyType, MetaType, Flags,
                                        nvhm::default_kernel64_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KERNEL_128)
    case 128:
      return make_nvhm_map_table_5<KeyType, MetaType, Flags,
                                        nvhm::default_kernel128_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KERNEL_256)
    case 256:
      return make_nvhm_map_table_5<KeyType, MetaType, Flags,
                                        nvhm::default_kernel256_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KERNEL_512)
    case 512:
      return make_nvhm_map_table_5<KeyType, MetaType, Flags,
                                        nvhm::default_kernel512_t>(id, config);
#endif
  }
  NVE_THROW_("`config.kernel_size` (", config.kernel_size, ") is out of bounds!");
}

template <typename KeyType, typename MetaType>
static host_table_ptr_t make_nvhm_map_table_3(const table_id_t id, const NvhmMapTableConfig& config) {
  constexpr nvhm::flags_t flags{nvhm::flags_t::blobs | nvhm::flags_t::aggressive_prefetch};
  if (config.auto_shrink) {
    return make_nvhm_map_table_4<KeyType, MetaType, flags | nvhm::flags_t::auto_shrink>(id, config);
  } else {
    return make_nvhm_map_table_4<KeyType, MetaType, flags>(id, config);
  }
}

template <typename KeyType>
static host_table_ptr_t make_nvhm_map_table_2(const table_id_t id, const NvhmMapTableConfig& config) {
  switch (config.overflow_policy.handler) {
    case OverflowHandler_t::EvictRandom:
      return make_nvhm_map_table_3<KeyType, no_meta_t>(id, config);
    case OverflowHandler_t::EvictLRU:
      return make_nvhm_map_table_3<KeyType, lru_meta_t>(id, config);
    case OverflowHandler_t::EvictLFU:
      return make_nvhm_map_table_3<KeyType, lfu_meta_t>(id, config);
  }
  NVE_THROW_("`config.overflow_policy.handler` (", config.overflow_policy.handler,
             ") is out of bounds!");
}

static host_table_ptr_t make_nvhm_map_table_1(const table_id_t id, const NvhmMapTableConfig& config) {
  switch (config.key_size) {
#if defined(NVE_FEATURE_HT_KEY_8)
    case sizeof(int8_t):
      return make_nvhm_map_table_2<int8_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KEY_16)
    case sizeof(int16_t):
      return make_nvhm_map_table_2<int16_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KEY_32)
    case sizeof(int32_t):
      return make_nvhm_map_table_2<int32_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KEY_64)
    case sizeof(int64_t):
      return make_nvhm_map_table_2<int64_t>(id, config);
#endif
  }
  NVE_THROW_("`config.key_size` (", config.key_size, ") is out of bounds!");
}

host_table_ptr_t NvhmMapTableFactory::produce(const table_id_t id, const NvhmMapTableConfig& config) {
  return make_nvhm_map_table_1(id, config);
}

}  // namespace plugin
}  // namespace nve
