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

#include <gtest/gtest.h>

#include <execution_context.hpp>
#include <stl_map_backed_table.hpp>

using namespace nve;
using namespace nlohmann::literals;

TEST(stl_map_backed_table, parse_table_conf) {
  const STLMapTableConfig conf{{{4, sizeof(int32_t), 16, DataType_t::BFloat},
                                2, Partitioner_t::AlwaysZero, {1}, 128,
                                32, 4096,
                                false,
                                {1024, OverflowHandler_t::EvictLRU, 0.5}}};
  std::cout << static_cast<nlohmann::json>(conf) << '\n';

  nlohmann::json json(R"({
    "mask_size": 4,
    "key_size": 4,
    "max_value_size": 16,
    "value_dtype": "bfloat",
    
    "num_partitions": 2,
    "partitioner": "always_zero",
    "workgroups": [1],
    "max_find_task_size": 128,
    
    "value_alignment": 32,
    "allocation_rate": 4096,
    
    "prefetch_values": false,
    
    "overflow_policy": {
      "overflow_margin": 1024,
      "handler": "evict_lru",
      "resolution_margin": 0.5
    }
  })"_json);
  ASSERT_EQ(json, static_cast<nlohmann::json>(conf));

  STLMapTableConfig parsed_conf(json);
  ASSERT_EQ(static_cast<nlohmann::json>(parsed_conf), static_cast<nlohmann::json>(conf));
}

TEST(stl_map_backed_table, parse_factory_conf) {
  const STLMapTableFactoryConfig conf{};
  std::cout << static_cast<nlohmann::json>(conf) << '\n';

  nlohmann::json json(R"({})"_json);
  ASSERT_EQ(json, static_cast<nlohmann::json>(conf));

  STLMapTableFactoryConfig parsed_conf(json);
  ASSERT_EQ(static_cast<nlohmann::json>(parsed_conf), static_cast<nlohmann::json>(conf));
}
