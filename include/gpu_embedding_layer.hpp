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
#include <mutex>

namespace nve {

/**
 * An embedding layer with a linear table on the GPU, without cache.
 * This allows the GPU kernel to resolve all indices without returning to the host during lookup.
 */

struct GPUEmbeddingLayerConfig {
  std::string layer_name;
  int device_id{0};                  // Device id of the GPU used
  void* embedding_table;             // Pointer to linear table in GPU memory.
  int64_t num_embeddings{0};         // Number of rows in the table.
  int64_t embedding_width_in_bytes;  // Stored bytes per row. Rowwise-quantized rows must have an
                                     // even width.
  DataType_t value_dtype{
      DataType_t::Unknown};          // Float16/Float32 or QInt8/QUint8RowwiseF16/F32 storage.
  int64_t default_row_index{-1};     // Index of a table row holding the default embedding, used by
                                     // lookups of keys outside [0, num_embeddings).
                                     // Negative disables the check, user is responsible for all keys being valid.
};

void from_json(const nlohmann::json& json, GPUEmbeddingLayerConfig& conf);
void to_json(nlohmann::json& json, const GPUEmbeddingLayerConfig& conf);

template <typename KeyType>
class GPUEmbeddingLayer : public EmbeddingLayerBase {
 public:
  NVE_PREVENT_COPY_AND_MOVE_(GPUEmbeddingLayer);

  using key_type = KeyType;

  GPUEmbeddingLayer(const GPUEmbeddingLayerConfig& config,
                    allocator_ptr_t allocator = {});

  ~GPUEmbeddingLayer() override;

  void lookup(context_ptr_t& ctx, const int64_t num_keys, const void* keys, void* output,
              const int64_t output_stride, bitmask64_t* hitmask,
              const PoolingParams* pool_params, float* hitrates) override;
  void insert(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
              const int64_t value_stride, const int64_t value_size, const void* values,
              const int64_t table_id) override;
  void update(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
              const int64_t value_stride, const int64_t value_size,
              const void* values, const int64_t table_id) override;
  void accumulate(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
                  const int64_t value_stride, const int64_t value_size, const void* values,
                  DataType_t value_type, const int64_t table_id) override;
  void clear(context_ptr_t& ctx) override;
  void erase(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
             const int64_t table_id) override;
  context_ptr_t create_execution_context(
    cudaStream_t lookup_stream, cudaStream_t modify_stream, thread_pool_ptr_t thread_pool, allocator_ptr_t allocator) override;
  int64_t get_num_tables() const override { return 1; }

 private:
  GPUEmbeddingLayerConfig config_;
  allocator_ptr_t allocator_;

  std::mutex kernel_launch_mutex_;
  std::shared_ptr<ContextRegistry> contexts_;
  cudaEvent_t modify_in_progress_;
  cudaStream_t private_modify_stream_;
};

}  // namespace nve
