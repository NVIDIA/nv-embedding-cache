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

// Malformed-input tests for the C API

#include <gtest/gtest.h>
#include <nve_c_api.h>

#include <cstring>

#include "test_utils.hpp"

// ---------------------------------------------------------------------------------
// Syntactically invalid JSON: nlohmann::json::parse throws, which must be
// caught by NVE_C_CATCH's generic std::exception handler
// ---------------------------------------------------------------------------------
TEST(MalformedInput, CreateTableFactoryInvalidJsonDoesNotThrow) {
  nve_table_factory_t factory = nullptr;
  nve_status_t status = NVE_SUCCESS;
  ASSERT_NO_THROW({
    status = nve_create_table_factory(&factory, "libnve-plugin-stl-map.so", "{not valid json");
  });
  EXPECT_NE(NVE_SUCCESS, status);
  EXPECT_EQ(nullptr, factory);
}

TEST(MalformedInput, TableFactoryProduceInvalidJsonDoesNotThrow) {
  nve_table_factory_t factory = nullptr;
  ASSERT_EQ(NVE_SUCCESS,
            nve_create_table_factory(&factory, "libnve-plugin-stl-map.so", "{}"));

  nve_table_t table = nullptr;
  nve_status_t status = NVE_SUCCESS;
  ASSERT_NO_THROW({ status = nve_table_factory_produce(factory, 0, "{not valid json", &table); });
  EXPECT_NE(NVE_SUCCESS, status);
  EXPECT_EQ(nullptr, table);

  nve_table_factory_destroy(factory);
}

// ---------------------------------------------------------------------------------
// Syntactically valid JSON that is not an object (an array or a bare number): the
// explicit is_object() check must reject it, not merely leave the config uninitialized.
// ---------------------------------------------------------------------------------
TEST(MalformedInput, CreateTableFactoryRejectsJsonArray) {
  nve_table_factory_t factory = nullptr;
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT,
            nve_create_table_factory(&factory, "libnve-plugin-stl-map.so", "[1, 2, 3]"));
  EXPECT_EQ(nullptr, factory);
}
TEST(MalformedInput, CreateTableFactoryRejectsJsonNumber) {
  nve_table_factory_t factory = nullptr;
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT,
            nve_create_table_factory(&factory, "libnve-plugin-stl-map.so", "42"));
  EXPECT_EQ(nullptr, factory);
}
TEST(MalformedInput, TableFactoryProduceRejectsJsonArray) {
  nve_table_factory_t factory = nullptr;
  ASSERT_EQ(NVE_SUCCESS,
            nve_create_table_factory(&factory, "libnve-plugin-stl-map.so", "{}"));

  nve_table_t table = nullptr;
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_table_factory_produce(factory, 0, "[1, 2, 3]", &table));
  EXPECT_EQ(nullptr, table);

  nve_table_factory_destroy(factory);
}
TEST(MalformedInput, TableFactoryProduceRejectsJsonNumber) {
  nve_table_factory_t factory = nullptr;
  ASSERT_EQ(NVE_SUCCESS,
            nve_create_table_factory(&factory, "libnve-plugin-stl-map.so", "{}"));

  nve_table_t table = nullptr;
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_table_factory_produce(factory, 0, "42", &table));
  EXPECT_EQ(nullptr, table);

  nve_table_factory_destroy(factory);
}

// ---------------------------------------------------------------------------------
// nve_gpu_table_create with an unsupported nve_key_type_t value must return NVE_ERROR_INVALID_ARGUMENT
// ---------------------------------------------------------------------------------
TEST(MalformedInput, GpuTableCreateRejectsUnsupportedKeyType) {
  nve_gpu_table_config_t cfg;
  ASSERT_EQ(NVE_SUCCESS, nve_gpu_table_config_init(&cfg, sizeof(cfg)));
  cfg.device_id = 0;
  cfg.cache_size = 1 << 20;
  cfg.row_size_in_bytes = 128;

  nve_table_t table = nullptr;
  // Bit-copy rather than static_cast<nve_key_type_t>(99) to avoid compiler warning
  nve_key_type_t bogus_key_type;
  const int32_t raw_key_type = 99;
  static_assert(sizeof(bogus_key_type) == sizeof(raw_key_type), "unexpected enum size");
  std::memcpy(&bogus_key_type, &raw_key_type, sizeof(raw_key_type));
  EXPECT_EQ(NVE_ERROR_INVALID_ARGUMENT, nve_gpu_table_create(&table, bogus_key_type, &cfg, nullptr));
  EXPECT_EQ(nullptr, table);
}
