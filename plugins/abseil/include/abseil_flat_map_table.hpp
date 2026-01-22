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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <absl/container/flat_hash_map.h>
#pragma GCC diagnostic pop

#include <stl_map_backed_table.hpp>

namespace nve {
namespace plugin {

struct AbseilFlatMapTableConfig final : public STLContainerTableConfig {
  using base_type = STLContainerTableConfig;

  template <typename KeyType>
  using map_type = absl::flat_hash_map<KeyType, char*>;

  int64_t initial_capacity{};  // Initial capacity of each map. Must be >= 0.

  void check() const;
};

void from_json(const nlohmann::json& json, AbseilFlatMapTableConfig& conf);

void to_json(nlohmann::json& json, const AbseilFlatMapTableConfig& conf);

template <typename MaskType, typename KeyType, typename MetaType, typename PartitionerType>
class AbseilFlatMapTable final : public STLContainerTable<AbseilFlatMapTableConfig, MaskType,
                                                          KeyType, MetaType, PartitionerType> {
 public:
  using base_type =
      STLContainerTable<AbseilFlatMapTableConfig, MaskType, KeyType, MetaType, PartitionerType>;
  using config_type = typename base_type::config_type;
  using mask_type = typename base_type::mask_type;
  using mask_repr_type = typename base_type::mask_repr_type;
  using key_type = typename base_type::key_type;
  using meta_type = typename base_type::meta_type;

  NVE_PREVENT_COPY_AND_MOVE_(AbseilFlatMapTable);

  AbseilFlatMapTable() = delete;

  AbseilFlatMapTable(const table_id_t id, const AbseilFlatMapTableConfig& config);

  ~AbseilFlatMapTable() override = default;
};

struct AbseilFlatMapTableFactoryConfig final : public STLContainerTableFactoryConfig {
  using base_type = STLContainerTableFactoryConfig;

  void check() const;
};

void from_json(const nlohmann::json& json, AbseilFlatMapTableFactoryConfig& conf);

void to_json(nlohmann::json& json, const AbseilFlatMapTableFactoryConfig& conf);

class AbseilFlatMapTableFactory final
    : public STLContainerTableFactory<AbseilFlatMapTableFactoryConfig, AbseilFlatMapTableConfig> {
 public:
  using base_type = STLContainerTableFactory<AbseilFlatMapTableFactoryConfig, AbseilFlatMapTableConfig>;

  NVE_PREVENT_COPY_AND_MOVE_(AbseilFlatMapTableFactory);

  AbseilFlatMapTableFactory() = delete;

  AbseilFlatMapTableFactory(const config_type& config);

  ~AbseilFlatMapTableFactory() override = default;

  host_table_ptr_t produce(table_id_t id, const AbseilFlatMapTableConfig& config) override;
};

}  // namespace plugin
}  // namespace nve
