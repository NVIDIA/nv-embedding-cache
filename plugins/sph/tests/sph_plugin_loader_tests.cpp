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

#include <cuda_runtime.h>

#include <buffer_wrapper.hpp>
#include <json_support.hpp>
#include <plugin/plugin_loader.hpp>

#include <cstdint>
#include <vector>

#include "cuda_ptr.hpp"

using namespace nve;

TEST(sph_plugin_loader, table_retains_library_lease) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  constexpr int64_t num_rows = 32;
  constexpr int64_t row_size = 16;
  table_ptr_t table;
  {
    Plugin plugin{NVE_SPH_PLUGIN_PATH};
    EXPECT_EQ(PluginMode::Internal, plugin.mode());
    auto factory = plugin.create_table_factory(nlohmann::json::object());
    table = factory->produce(
        1, nlohmann::json{{"device_id", 0}, {"initial_capacity", 1024},
                          {"row_size_in_bytes", row_size},
                          {"value_dtype", "float32"}});
  }
  ASSERT_NE(nullptr, table);

  auto context = table->create_execution_context(0, 0, nullptr, nullptr);
  std::vector<int64_t> keys(static_cast<size_t>(num_rows));
  std::vector<unsigned char> values(static_cast<size_t>(num_rows * row_size));
  for (int64_t index = 0; index < num_rows; ++index) {
    keys[static_cast<size_t>(index)] = index + 1;
    values[static_cast<size_t>(index * row_size)] = static_cast<unsigned char>(index);
  }

  int64_t* device_keys = nullptr;
  unsigned char* device_values = nullptr;
  unsigned char* device_output = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&device_keys, keys.size() * sizeof(int64_t)));
  DevicePtr<int64_t> keys_owner(device_keys);
  ASSERT_EQ(cudaSuccess, cudaMalloc(&device_values, values.size()));
  DevicePtr<unsigned char> values_owner(device_values);
  ASSERT_EQ(cudaSuccess, cudaMalloc(&device_output, values.size()));
  DevicePtr<unsigned char> output_owner(device_output);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(device_keys, keys.data(), keys.size() * sizeof(int64_t),
                                    cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(device_values, values.data(), values.size(),
                                    cudaMemcpyHostToDevice));

  auto key_buffer = std::make_shared<BufferWrapper<const void>>(
      context, "keys", device_keys, keys.size() * sizeof(int64_t));
  auto value_buffer = std::make_shared<BufferWrapper<const void>>(
      context, "values", device_values, values.size());
  table->insert(context, num_rows, std::move(key_buffer), row_size, row_size,
                std::move(value_buffer));

  key_buffer = std::make_shared<BufferWrapper<const void>>(
      context, "keys_find", device_keys, keys.size() * sizeof(int64_t));
  auto output_buffer = std::make_shared<BufferWrapper<void>>(
      context, "output", device_output, values.size());
  table->find(context, num_rows, std::move(key_buffer), nullptr, row_size,
              std::move(output_buffer), nullptr);
  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

  std::vector<unsigned char> output(values.size());
  ASSERT_EQ(cudaSuccess, cudaMemcpy(output.data(), device_output, output.size(),
                                    cudaMemcpyDeviceToHost));
  EXPECT_EQ(values, output);

  EXPECT_EQ(cudaSuccess, cudaFree(keys_owner.release()));
  EXPECT_EQ(cudaSuccess, cudaFree(values_owner.release()));
  EXPECT_EQ(cudaSuccess, cudaFree(output_owner.release()));
}
