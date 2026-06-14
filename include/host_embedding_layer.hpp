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
#include "include/embedding_layer.hpp"
#include <vector>
#include <string>

namespace nve {

/**
 * Host-only embedding layer wrapping a single host-resident table (e.g. LinearHostTable).
 * Performs lookup / update entirely on the CPU — never enters the CUDA runtime, never creates
 * an execution context that holds CUDA streams. Intended for inference on systems without a
 * GPU or CUDA driver. The associated execution context is built directly by this layer; the
 * underlying table's create_execution_context is not invoked.
 */
template <typename KeyType>
class HostEmbeddingLayer : public EmbeddingLayerBase {
 public:
  struct Config {
    std::string layer_name;
    std::vector<uint8_t> default_embedding = {}; // optional default vector for keys missing from the table
  };

  NVE_PREVENT_COPY_AND_MOVE_(HostEmbeddingLayer);
  using key_type = KeyType;

  /**
   * Create a host-only embedding layer.
   * @param cfg Layer configuration parameters.
   * @param table A host-resident table (must report get_device_id() < 0).
   * @param allocator Allocator for internal resources not bound to a specific context.
   *                  nullptr implies using GetDefaultAllocator().
   */
  HostEmbeddingLayer(const Config& cfg, table_ptr_t table, allocator_ptr_t allocator = {});

  ~HostEmbeddingLayer() override;

  void lookup(context_ptr_t& ctx, const int64_t num_keys, const void* keys, void* output,
              const int64_t output_stride, max_bitmask_repr_t* hitmask,
              const PoolingParams* pool_params, float* hitrates) override;
  void insert(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
              const int64_t value_stride, const int64_t value_size, const void* values,
              int64_t table_id) override;
  void update(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
              const int64_t value_stride, const int64_t value_size,
              const void* values, int64_t table_id) override;
  void accumulate(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
                  const int64_t value_stride, const int64_t value_size, const void* values,
                  DataType_t value_type, int64_t table_id) override;
  void clear(context_ptr_t& ctx) override;
  void erase(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
             int64_t table_id) override;
  int64_t get_num_tables() const override { return 1; }
  context_ptr_t create_execution_context(
    cudaStream_t lookup_stream, cudaStream_t modify_stream,
    thread_pool_ptr_t thread_pool, allocator_ptr_t allocator) override;

  inline const Config& get_config() const { return config_; }

 private:
  const Config config_;
  allocator_ptr_t allocator_;
  table_ptr_t table_;
};

}  // namespace nve
