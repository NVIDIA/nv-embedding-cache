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
#include <rocksdb_table.hpp>

using namespace nve;
using namespace nve::plugin;
using namespace nlohmann::literals;

TEST(rocksdb_plugin, parse_table_conf) {
  const RocksDBTableConfig conf{
      {4, sizeof(int32_t), 16, DataType_t::BFloat}, 1024, "some_column_name", false};
  std::cout << static_cast<nlohmann::json>(conf) << '\n';

  nlohmann::json json(R"(
    {
      "mask_size": 4,
      "key_size": 4,
      "max_value_size": 16,
      "value_dtype": "bfloat",
      
      "max_batch_size": 1024,
      
      "column_family": "some_column_name",
      "verify_checksums": false
    }
  )"_json);
  ASSERT_EQ(json, static_cast<nlohmann::json>(conf));

  RocksDBTableConfig parsed_conf(json);
  ASSERT_EQ(static_cast<nlohmann::json>(parsed_conf), static_cast<nlohmann::json>(conf));
}

TEST(rocksdb_plugin, parse_factory_conf) {
  const RocksDBTableFactoryConfig conf{{}, "/tmp/another_db", true, 8};
  std::cout << static_cast<nlohmann::json>(conf) << '\n';

  nlohmann::json json(R"({
    "path": "/tmp/another_db",
    "read_only": true,
    "num_threads": 8
  })"_json);
  ASSERT_EQ(json, static_cast<nlohmann::json>(conf));

  RocksDBTableFactoryConfig parsed_conf(json);
  ASSERT_EQ(static_cast<nlohmann::json>(parsed_conf), static_cast<nlohmann::json>(conf));
}
