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
#include <nvhm_map_table.hpp>

using namespace nve;
using namespace nve::plugin;
using namespace nlohmann::literals;

TEST(nvhm_plugin, parse_table_conf) {
  const NvhmMapTableConfig conf{{4, sizeof(int32_t), 16, DataType_t::BFloat},
                                     2, Partitioner_t::AlwaysZero, {1}, 128,
                                     16, 1337, 32,
                                     4, false,
                                     true, true,
                                     {1024, OverflowHandler_t::EvictLRU, 0.5}};
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
    
    "kernel_size": 16,
    "initial_capacity": 1337,
    "value_alignment": 32,

    "key_fetch_queue_length": 4,
    "prefetch_values": false,

    "minimize_psl": true,
    "auto_shrink": true,
    
    "overflow_policy": {
      "overflow_margin": 1024,
      "handler": "evict_lru",
      "resolution_margin": 0.5
    }
  })"_json);
  ASSERT_EQ(json, static_cast<nlohmann::json>(conf));

  NvhmMapTableConfig parsed_conf(json);
  ASSERT_EQ(static_cast<nlohmann::json>(parsed_conf), static_cast<nlohmann::json>(conf));
}

TEST(nvhm_plugin, parse_factory_conf) {
  const NvhmMapTableFactoryConfig conf{};
  std::cout << static_cast<nlohmann::json>(conf) << '\n';

  nlohmann::json json(R"({})"_json);
  ASSERT_EQ(json, static_cast<nlohmann::json>(conf));

  NvhmMapTableFactoryConfig parsed_conf(json);
  ASSERT_EQ(static_cast<nlohmann::json>(parsed_conf), static_cast<nlohmann::json>(conf));
}
