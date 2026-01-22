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
#include <redis_cluster_table.hpp>

using namespace nve;
using namespace nve::plugin;
using namespace nlohmann::literals;

TEST(redis_plugin, parse_table_conf) {
  const RedisClusterTableConfig conf{{4, sizeof(int32_t), 16, DataType_t::BFloat},
                                     1024,
                                     2,
                                     Partitioner_t::AlwaysZero,
                                     {1},
                                     "whatever",
                                     {1024, OverflowHandler_t::EvictLRU, 0.5}};
  std::cout << static_cast<nlohmann::json>(conf) << '\n';

  nlohmann::json json(R"(
    {
      "mask_size": 4,
      "key_size": 4,
      "max_value_size": 16,
      "value_dtype": "bfloat",

      "max_batch_size": 1024,
      
      "num_partitions": 2,
      "partitioner": "always_zero",
      "workgroups": [1],
      "hash_key": "whatever",
      
      "overflow_policy": {
        "overflow_margin": 1024,
        "handler": "evict_lru",
        "resolution_margin": 0.5
      }
    }
  )"_json);
  ASSERT_EQ(json, static_cast<nlohmann::json>(conf));

  RedisClusterTableConfig parsed_conf(json);
  ASSERT_EQ(static_cast<nlohmann::json>(parsed_conf), static_cast<nlohmann::json>(conf));
}

TEST(redis_plugin, parse_factory_conf) {
  const RedisClusterTableFactoryConfig conf{
      {},   "localhost:7000", "admin",       "12345",      false,   3,
      true, "my_bundle.crt",  "my_cert.pem", "my_key.pem", "my_sni"};
  std::cout << static_cast<nlohmann::json>(conf) << '\n';

  nlohmann::json json(R"({
    "address": "localhost:7000",
    "user_name": "admin",
    "password": "12345",
    
    "keep_alive": false,
    "connections_per_node": 3,
    
    "use_tls": true,
    "ca_certificate": "my_bundle.crt",
    "client_certificate": "my_cert.pem",
    "client_key": "my_key.pem",
    "server_name_identification": "my_sni"
  })"_json);
  ASSERT_EQ(json, static_cast<nlohmann::json>(conf));

  RedisClusterTableFactoryConfig parsed_conf(json);
  ASSERT_EQ(static_cast<nlohmann::json>(parsed_conf), static_cast<nlohmann::json>(conf));
}
