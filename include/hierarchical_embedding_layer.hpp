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
#include <embedding_layer.hpp>
#include <vector>
#include "layer_utils.hpp"

namespace nve {
 
template <typename KeyType> class GpuTable;
class InsertHeuristic;

/**
 * An embedding layer with a hierarchy of tables (caches/databases).
 * Lookups will be resolved by processing tables in order, such that indices missing in one table will be forwarded to the next.
 */
template <typename KeyType>
class HierarchicalEmbeddingLayer : public EmbeddingLayerBase {
 public:
  struct Config {
    std::string layer_name;
    std::shared_ptr<InsertHeuristic> insert_heuristic = nullptr;
    int64_t min_insert_freq_gpu = 0; // increase this to throttle down auto inserts
    int64_t min_insert_freq_host = 0;
    int64_t min_insert_size_gpu = 1 << 16;
    int64_t min_insert_size_host = 0;
  };

  NVE_PREVENT_COPY_AND_MOVE_(HierarchicalEmbeddingLayer);
  using key_type = KeyType;

  /**
   * Create a Hierarchical embedding layer, using multiple tables.
   * 
   * @param cfg Layer configuration parameters.
   * @param tables Vector of valid table shared_ptrs to use.
   *               Tables will be handled in order during lookup.
   *               i.e. table performance is expected to decrease along the table vector
   * @param allocator Allocator to use for internal resources not bound to a specific context
   *                  nullptr implies using the default allocator
   */
  HierarchicalEmbeddingLayer(const Config& cfg, const std::vector<table_ptr_t>& tables,
                             allocator_ptr_t allocator = {});

  ~HierarchicalEmbeddingLayer() override;

  void lookup(context_ptr_t& ctx, const int64_t num_keys, const void* keys, void* output,
              const int64_t output_stride, max_bitmask_repr_t* hitmask,
              const PoolingParams* pool_params, float* hitrates) override;
  void insert(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
              const int64_t value_stride, const int64_t value_size, const void* values,
              int64_t table_id) override;
  void update(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
              const int64_t value_stride, const int64_t value_size,
              const void* values) override;
  void accumulate(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
                  const int64_t value_stride, const int64_t value_size, const void* values,
                  DataType_t value_type) override;
  void clear(context_ptr_t& ctx) override;
  void erase(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
             int64_t table_id) override;
  context_ptr_t create_execution_context(
    cudaStream_t lookup_stream, cudaStream_t modify_stream, thread_pool_ptr_t thread_pool, allocator_ptr_t allocator) override;

  inline const Config& get_config() const { return config_; }

 private:
  const Config config_;
  allocator_ptr_t allocator_;
  std::vector<table_ptr_t> tables_;
  int32_t gpu_device_; // gpu device id or -1

  std::vector<std::shared_ptr<AutoInsertHandler>> auto_insert_handlers_;
};

}  // namespace nve
