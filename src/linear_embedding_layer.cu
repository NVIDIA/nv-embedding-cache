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

#include <linear_embedding_layer.hpp>
#include <layer_utils.hpp>
#include <default_allocator.hpp>
#include <gpu_table.hpp>
#include <insert_heuristic.hpp>
#include "cuda_ops/cuda_common.h"
#include <buffer_wrapper.hpp>

namespace nve {

template <typename KeyType>
context_ptr_t LinearUVMEmbeddingLayer<KeyType>::create_execution_context(
  cudaStream_t lookup_stream,
  cudaStream_t modify_stream,
  thread_pool_ptr_t thread_pool,
  allocator_ptr_t allocator) {
  allocator_ptr_t actual_allocator = allocator ? allocator : allocator_;
  auto gpu_ctx = gpu_table_ ? gpu_table_->create_execution_context(lookup_stream, modify_stream, thread_pool, actual_allocator) : nullptr;
  std::vector<context_ptr_t> contexts{gpu_ctx};
  return std::make_shared<LayerExecutionContext>(
    lookup_stream,
    modify_stream,
    thread_pool,
    actual_allocator,
    contexts);
}

template <typename KeyType>
LinearUVMEmbeddingLayer<KeyType>::LinearUVMEmbeddingLayer(const Config& cfg, gpu_table_ptr_t gpu_table, allocator_ptr_t allocator)
  : config_(cfg), gpu_table_(std::move(gpu_table)) {
    allocator_ = allocator ? allocator : GetDefaultAllocator();
    NVE_CHECK_(allocator_ != nullptr, "Failed to get default allocator");
    NVE_CHECK_(gpu_table_ != nullptr, "Invalid GPU table");
    NVE_CHECK_(gpu_table_->config().uvm_table != nullptr, "GPU Table must be backed by a UVM buffer");

    auto heuristic = config_.insert_heuristic;
    if (!heuristic) {
      heuristic = std::make_shared<DefaultInsertHeuristic>(std::vector<float>{DefaultInsertHeuristic::DEFAULT_THRESHOLD});
    }
    auto_insert_handler_ = std::make_shared<AutoInsertHandler>(
      heuristic,
      gpu_table_,
      0 /* table_id */,
      allocator_,
      config_.min_insert_freq_gpu,
      config_.min_insert_size_gpu,
      sizeof(KeyType),
      gpu_table_->get_device_id()
    );
}

template <typename KeyType>
LinearUVMEmbeddingLayer<KeyType>::~LinearUVMEmbeddingLayer() {
}

template <typename KeyType>
void LinearUVMEmbeddingLayer<KeyType>::lookup(context_ptr_t& ctx, const int64_t num_keys, const void* keys, void* output,
                    const int64_t output_stride, max_bitmask_repr_t* hitmask,
                    const PoolingParams* pool_params, float* hitrates) {
  NVE_NVTX_SCOPED_FUNCTION_COL1_();
  ScopedDevice scope_device(gpu_table_->config().device_id);
  NVE_CHECK_(num_keys >= 0, "Invalid num_keys");
  NVE_CHECK_(keys != nullptr, "Invalid keys buffer");
  NVE_CHECK_(output != nullptr, "Invalid output buffer");
  NVE_CHECK_(hitmask == nullptr, "Hitmask is not supported for the UVM layer");

  auto layer_ctx = std::dynamic_pointer_cast<LayerExecutionContext>(ctx);
  NVE_CHECK_(layer_ctx != nullptr, "Invalid layer context");
  const cudaStream_t lookup_stream = layer_ctx->get_lookup_stream();
  const auto output_buffer_size = num_keys * output_stride;
  const auto key_buffer_size = sizeof(KeyType) * num_keys;

  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, key_buffer_size);
  auto output_bw = std::make_shared<BufferWrapper<void>>(ctx, "output", output, output_buffer_size);

  const bool collect_misses = (hitrates || !std::dynamic_pointer_cast<NeverInsertHeuristic>(config_.insert_heuristic));
  if (collect_misses) {
    NVE_CHECK_(gpu_table_->config().count_misses, "GPU table was not configured to report hit counts");
    gpu_table_->reset_lookup_counter(layer_ctx->table_contexts_.at(0));
  }

  // Lookup 
  if (pool_params) {
    auto offsets_buffer_size = pool_params->num_key_indices * sizeof(KeyType);
    
    // This cast from int64_t* to KeyType* is not good (need the pool params to be templated on keytype or just force offsets to int64 always)
    // right now just preserving the current state (this "bug" also existed before this refactor)
    // TODO: fix it
    auto offsets_bw = std::make_shared<BufferWrapper<const KeyType>>(ctx, "offsets", reinterpret_cast<const KeyType*>(pool_params->key_indices), offsets_buffer_size);

    int64_t hotness;
    switch (pool_params->sparse_type)
    {
    case SparseType_t::CSR:
      hotness = 0;
      break;
    case SparseType_t::Fixed:
      // TODO: can only copy the first element if offsets was on GPU (access_buffer for host can copy the entire array if buffer was only accessed on GPU)
      hotness = offsets_bw->access_buffer(cudaMemoryTypeHost, true /*copy_content*/, lookup_stream)[0];
    break;
    default:
      NVE_THROW_("Unsupported pooling sparse type");
    }

    // TODO: add support for more data type combinations if needed
    if (pool_params->sparse_weights != nullptr) {
      NVE_CHECK_(pool_params->weight_type == gpu_table_->config().value_dtype, "Pooling weight type differs from table data type");
    }
    switch(gpu_table_->config().value_dtype) {
      case DataType_t::Float32:
      {
        auto weights_buffer_size = num_keys * sizeof(float);
        std::shared_ptr<BufferWrapper<const float>> weights_bw = 
          pool_params->sparse_weights ?
          std::make_shared<BufferWrapper<const float>>(ctx, "weights", reinterpret_cast<const float*>(pool_params->sparse_weights), weights_buffer_size) :
          nullptr;
        gpu_table_->template find_and_combine_bw<KeyType, float>(
            layer_ctx->table_contexts_.at(0), num_keys, keys_bw,
            pool_params->sparse_type, pool_params->num_key_indices - 1,
            offsets_bw, hotness,
            pool_params->pooling_type, weights_bw,
            output_stride, output_bw);
        break;
      }
      case DataType_t::Float16:
      {
        auto weights_buffer_size = num_keys * sizeof(__half);
        std::shared_ptr<BufferWrapper<const __half>> weights_bw = 
          pool_params->sparse_weights ?
          std::make_shared<BufferWrapper<const __half>>(ctx, "weights", reinterpret_cast<const __half*>(pool_params->sparse_weights), weights_buffer_size) :
          nullptr;
        gpu_table_->template find_and_combine_bw<KeyType, __half>(
            layer_ctx->table_contexts_.at(0), num_keys, keys_bw,
            pool_params->sparse_type, pool_params->num_key_indices - 1,
            offsets_bw, hotness,
            pool_params->pooling_type, weights_bw,
            output_stride, output_bw);
        break;
      }
      default:
        NVE_THROW_("Unsupported data type");
    }
  } else {
    gpu_table_->find_bw(layer_ctx->table_contexts_.at(0), num_keys, keys_bw, nullptr, output_stride, output_bw, nullptr);
  }

  auto d_output = output_bw->get_buffer(cudaMemoryTypeDevice);
  if (output != d_output) {
    NVE_CHECK_(cudaMemcpyAsync(output, d_output, output_buffer_size, cudaMemcpyDefault, lookup_stream));
  }

  // Handle hitrates
  if (collect_misses) {
    int64_t misses = 0;
    gpu_table_->get_lookup_counter(layer_ctx->table_contexts_.at(0), &misses);
    NVE_CHECK_(cudaStreamSynchronize(lookup_stream)); // Need to wait for lookup to end before checking misses
    const float hitrate = 1.f - (static_cast<float>(misses) / static_cast<float>(num_keys));
    if (hitrates) {
      *hitrates = hitrate;
    }
    // Handle automatic inserts
    if (auto_insert_handler_) {
      auto_insert_handler_->auto_insert(layer_ctx, keys_bw, output_bw, hitrate, num_keys, output_stride);
    }
  }
}

template <typename KeyType>
void LinearUVMEmbeddingLayer<KeyType>::insert(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
                    const int64_t value_stride, const int64_t value_size, const void* values,
                    const int64_t table_id) {
  NVE_NVTX_SCOPED_FUNCTION_COL2_();
  if (table_id != 0) {
    NVE_LOG_INFO_("Insert called with invalid table_id - ignored call");
    return;
  }

  ScopedDevice scope_device(gpu_table_->config().device_id);
  NVE_CHECK_(num_keys >= 0, "Invalid num_keys");
  NVE_CHECK_(keys != nullptr, "Invalid keys buffer");
  NVE_CHECK_(values != nullptr, "Invalid values buffer");
  NVE_CHECK_(value_size == gpu_table_->config().row_size_in_bytes, "Invalid value size");

  auto layer_ctx = std::dynamic_pointer_cast<LayerExecutionContext>(ctx);
  NVE_CHECK_(layer_ctx != nullptr, "Invalid layer context");

  const auto keys_buffer_size = sizeof(KeyType) * num_keys;
  const auto values_buffer_size = num_keys * value_stride;
 
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, keys_buffer_size);
  auto values_bw = std::make_shared<BufferWrapper<const void>>(ctx, "values", values, values_buffer_size);

  gpu_table_->insert_bw(layer_ctx->table_contexts_.at(0), num_keys, keys_bw, value_stride, value_stride, values_bw);
}

template <typename KeyType>
void LinearUVMEmbeddingLayer<KeyType>::update(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
                    const int64_t value_stride, const int64_t value_size,
                    const void* values) {
  NVE_NVTX_SCOPED_FUNCTION_COL3_();
  ScopedDevice scope_device(gpu_table_->config().device_id);
  NVE_CHECK_(num_keys >= 0, "Invalid num_keys");
  NVE_CHECK_(keys != nullptr, "Invalid keys buffer");
  NVE_CHECK_(values != nullptr, "Invalid values buffer");
  NVE_CHECK_(value_size == gpu_table_->config().row_size_in_bytes, "Invalid value size");

  auto layer_ctx = std::dynamic_pointer_cast<LayerExecutionContext>(ctx);
  NVE_CHECK_(layer_ctx != nullptr, "Invalid layer context");

  const auto keys_buffer_size = sizeof(KeyType) * num_keys;
  const auto values_buffer_size = num_keys * value_stride;
 
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, keys_buffer_size);
  auto values_bw = std::make_shared<BufferWrapper<const void>>(ctx, "values", values, values_buffer_size);

  if (auto_insert_handler_) {
    auto_insert_handler_->lock_modify();
  }
  gpu_table_->update_bw(layer_ctx->table_contexts_.at(0), num_keys, keys_bw, value_stride, value_stride, values_bw);
  if (auto_insert_handler_) {
    auto_insert_handler_->unlock_modify();
  }
}

template <typename KeyType>
void LinearUVMEmbeddingLayer<KeyType>::accumulate(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
                        const int64_t value_stride, const int64_t value_size, const void* values,
                        DataType_t value_type) {
  NVE_NVTX_SCOPED_FUNCTION_COL4_();
  ScopedDevice scope_device(gpu_table_->config().device_id);
  NVE_CHECK_(num_keys >= 0, "Invalid num_keys");
  NVE_CHECK_(keys != nullptr, "Invalid keys buffer");
  NVE_CHECK_(values != nullptr, "Invalid values buffer");
  NVE_CHECK_(value_size == gpu_table_->config().row_size_in_bytes, "Invalid value size");

  auto layer_ctx = std::dynamic_pointer_cast<LayerExecutionContext>(ctx);
  NVE_CHECK_(layer_ctx != nullptr, "Invalid layer context");

  const auto keys_buffer_size = sizeof(KeyType) * num_keys;
  const auto values_buffer_size = num_keys * value_stride;
 
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, keys_buffer_size);
  auto values_bw = std::make_shared<BufferWrapper<const void>>(ctx, "values", values, values_buffer_size);

  if (auto_insert_handler_) {
    auto_insert_handler_->lock_modify();
  }
  gpu_table_->update_accumulate_bw(layer_ctx->table_contexts_.at(0), num_keys, keys_bw, value_stride, value_stride, values_bw, value_type);
  if (auto_insert_handler_) {
    auto_insert_handler_->unlock_modify();
  }
}

template <typename KeyType>
void LinearUVMEmbeddingLayer<KeyType>::clear(context_ptr_t& ctx) { 
  NVE_NVTX_SCOPED_FUNCTION_COL5_();
  ScopedDevice scope_device(gpu_table_->config().device_id);
  auto layer_ctx = std::dynamic_pointer_cast<LayerExecutionContext>(ctx);
  NVE_CHECK_(layer_ctx != nullptr, "Invalid layer context");

  if (auto_insert_handler_) {
    auto_insert_handler_->lock_modify();
  }
  gpu_table_->clear(layer_ctx->table_contexts_.at(0));
  if (auto_insert_handler_) {
    auto_insert_handler_->unlock_modify();
  }
}

template <typename KeyType>
void LinearUVMEmbeddingLayer<KeyType>::erase(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
                    const int64_t table_id) {
  NVE_NVTX_SCOPED_FUNCTION_COL6_();
  if (table_id != 0) {
    NVE_LOG_INFO_("Erase called with invalid table_id - ignored call");
    return;
  }
  ScopedDevice scope_device(gpu_table_->config().device_id);

  NVE_CHECK_(num_keys >= 0, "Invalid num_keys");
  NVE_CHECK_(keys != nullptr, "Invalid keys buffer");

  auto layer_ctx = std::dynamic_pointer_cast<LayerExecutionContext>(ctx);
  NVE_CHECK_(layer_ctx != nullptr, "Invalid layer context");

  auto keys_buffer_size = sizeof(KeyType) * num_keys;
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, keys_buffer_size);

  if (auto_insert_handler_) {
    auto_insert_handler_->lock_modify();
  }
  gpu_table_->erase_bw(layer_ctx->table_contexts_.at(0), num_keys, keys_bw);
  if (auto_insert_handler_) {
    auto_insert_handler_->unlock_modify();
  }
}

template class LinearUVMEmbeddingLayer<int32_t>;
template class LinearUVMEmbeddingLayer<int64_t>;

}  // namespace nve
