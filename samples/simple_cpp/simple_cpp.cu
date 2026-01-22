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

#include <cuda.h>
#include <memory>
#include <gpu_table.hpp>
#include <linear_embedding_layer.hpp>
#include <iostream>

int main(int, char*[]) {
  constexpr int64_t row_size = int64_t(1)<<10; // 1KB
  constexpr int64_t cache_size = int64_t(1)<<22; // 4MB
  constexpr int64_t linear_table_size = int64_t(1<<26); // 64MB
  using key_type = int64_t; // embedding keys will be int64 values
  using table_type = nve::GpuTable<key_type>;
  using layer_type = nve::LinearUVMEmbeddingLayer<key_type>;

  // Allocate a 2GB linear table buffer in host memory
  // We need to use either cudaMallocHost or malloc, then cudaHostRegister to make the buffer accessible by the GPU.
  std::cout << "Allocating linear table in host memory" << std::endl;
  void* linear_table = nullptr;
  NVE_CHECK_(cudaMallocHost(&linear_table, linear_table_size));
  // Here would be a good place to initialize the data in the table (e.g. loading from file)

  // Create a GPU table for the layer with a 1GB cache in GPU memory
  std::cout << "Creating a GPU table with cache" << std::endl;
  table_type::config_type tab_cfg;
  tab_cfg.device_id = 0;
  tab_cfg.cache_size = cache_size;
  tab_cfg.row_size_in_bytes = row_size;
  tab_cfg.uvm_table = linear_table;
  auto gpu_tab = std::make_shared<table_type>(tab_cfg); // The table is using int64 indices (template arg)
  
  // Create the linear layer
  std::cout << "Creating a linear Layer" << std::endl;
  auto layer = std::make_shared<layer_type>(layer_type::Config(), gpu_tab);

  // Create an execution context with defaults
  std::cout << "Creating an execution context" << std::endl;
  auto ctx = layer->create_execution_context(0, 0, nullptr, nullptr);

  // Insert a key
  std::cout << "Inserting a key-row to the GPU table" << std::endl;
    // First setup the buffers for the keys (in this case a single key) and rows
  const int64_t key = 12345;
  void* input_row;
  NVE_CHECK_(cudaMalloc(&input_row, row_size));
    // Here would be a good place to set the embedding row values
  layer->insert(ctx, 1, &key, row_size, row_size, input_row, 0); // the last parameter indicates we want to insert the key-row pair to the first table
    // At this point the operation is inflight (async), we can wait for it's completion on the context (there are also other ways)
  ctx->wait();

  // Lookup a key
  std::cout << "Looking up a key in the layer" << std::endl;
    // First setup the buffer for the output (we can reuse the key from above as the keys buffer)
  void* output_row;
  NVE_CHECK_(cudaMalloc(&output_row, row_size));
  layer->lookup(ctx, 1, &key, output_row, row_size, nullptr, nullptr, nullptr);

  // Teardown
  std::cout << "Tearing down allocated objects" << std::endl;
  ctx->wait(); // wait for all pending work to end
  ctx.reset();
  layer.reset();
  gpu_tab.reset();
  NVE_CHECK_(cudaFreeHost(linear_table));
  NVE_CHECK_(cudaFree(input_row));
  NVE_CHECK_(cudaFree(output_row));

  return 0;
}