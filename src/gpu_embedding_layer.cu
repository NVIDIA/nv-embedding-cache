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

#include <gpu_embedding_layer.hpp>
#include <layer_utils.hpp>
#include <limits>

// Disable warnings for cuEmbed
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include <cuembed/include/embedding_lookup.cuh>
#pragma GCC diagnostic pop

#include <default_allocator.hpp>
#include "cuda_ops/find_and_dequant.cuh"
#include "cuda_ops/pool_gathered.cuh"
#include "cuda_ops/sanitize_keys.cuh"
#include "cuda_ops/update_accumulate.cuh"
#include <ecache/ec_no_cache.cuh>
#include <ecache/embed_cache.h> // DefaultECEvent
#include "cuda_ops/cuda_common.h"
#include <buffer_wrapper.hpp>

namespace nve {

namespace {

void validate_quant_pool_types_and_stride(
    const GPUEmbeddingLayerConfig& config,
    const EmbeddingLayerBase::PoolingParams& pool_params,
    int64_t output_stride) {
  const DataType_t required_output_type = quant_rowwise_output_dtype(config.value_dtype);
  NVE_CHECK_ARG_(pool_params.output_type == required_output_type,
                 "Quantized GPU embedding output type must match the quantization metadata precision");
  const int64_t value_count = row_num_elements(config.value_dtype, config.embedding_width_in_bytes);
  NVE_CHECK_ARG_(output_stride == value_count * dtype_size(required_output_type),
                 "Invalid dequantized output stride for quantized GPU embedding table");

  if (pool_params.pooling_type == PoolingType_t::Concatenate) {
    return;
  }

  switch (pool_params.pooling_type) {
    case PoolingType_t::Sum:
    case PoolingType_t::Mean:
    case PoolingType_t::WeightedSum:
    case PoolingType_t::WeightedMean: break;
    default:
      NVE_THROW_ARG_("Unsupported pooling type ",
                     static_cast<uint32_t>(pool_params.pooling_type));
  }

  if (is_weighted_pooling(pool_params.pooling_type)) {
    NVE_CHECK_ARG_(pool_params.weight_type == DataType_t::Float32 ||
                       pool_params.weight_type == DataType_t::Float16,
                   "Quantized GPU embedding weights must be Float32 or Float16");
  }
}

}  // namespace

class GPUEmbeddingTableExecutionContext: public ExecutionContext {
 public:
  GPUEmbeddingTableExecutionContext(
    cudaStream_t lookup_stream,
    cudaStream_t modify_stream,
    thread_pool_ptr_t thread_pool,
    allocator_ptr_t allocator,
    std::shared_ptr<ContextRegistry>& context_registry) 
      : ExecutionContext(lookup_stream, modify_stream, std::move(thread_pool), std::move(allocator)),context_registry_(context_registry) {
      NVE_CHECK_(context_registry != nullptr, "Invalid context registry");
      context_registry_->add_context(this);
  }

  virtual ~GPUEmbeddingTableExecutionContext() {
    NVE_CHECK_(cudaStreamSynchronize(lookup_stream_));
    NVE_CHECK_(cudaStreamSynchronize(modify_stream_));
    context_registry_->remove_context(this);
  }
 private:
  std::shared_ptr<ContextRegistry> context_registry_;
};

template <typename KeyType>
context_ptr_t GPUEmbeddingLayer<KeyType>::create_execution_context(
  cudaStream_t lookup_stream,
  cudaStream_t modify_stream,
  thread_pool_ptr_t thread_pool,
  allocator_ptr_t allocator) {
  allocator_ptr_t actual_allocator = allocator ? allocator : allocator_;

  auto table_ctx = std::make_shared<GPUEmbeddingTableExecutionContext>(lookup_stream, modify_stream,
                                                                       thread_pool, actual_allocator, contexts_);
  std::vector<context_ptr_t> ctx_vec{table_ctx};
  return std::make_shared<LayerExecutionContext>(
    lookup_stream,
    modify_stream,
    thread_pool,
    actual_allocator,
    ctx_vec);
}

void from_json(const nlohmann::json& json, GPUEmbeddingLayerConfig& conf) {
  NVE_READ_JSON_FIELD_(layer_name);
  NVE_READ_JSON_FIELD_(device_id);
  NVE_READ_JSON_FIELD_(num_embeddings);
  NVE_READ_JSON_FIELD_(embedding_width_in_bytes);
  NVE_READ_JSON_FIELD_(value_dtype);
  NVE_READ_JSON_FIELD_(default_row_index);
}

void to_json(nlohmann::json& json, const GPUEmbeddingLayerConfig& conf) {
  NVE_WRITE_JSON_FIELD_(layer_name);
  NVE_WRITE_JSON_FIELD_(device_id);
  NVE_WRITE_JSON_FIELD_(num_embeddings);
  NVE_WRITE_JSON_FIELD_(embedding_width_in_bytes);
  NVE_WRITE_JSON_FIELD_(value_dtype);
  NVE_WRITE_JSON_FIELD_(default_row_index);
}

// cuEmbed's index and CSR-offset types follow the layer's KeyType. Quantized pooling bypasses this
// adapter after its raw gather and uses the KeyType-typed shared CUDA pooling path instead. fp16
// math is currently off, and will be added to pooling params later.
template <typename KeyType>
void call_cuembed_forward(void* embedding_table,
                          const int embed_width_in_bytes,
                          const void* indices,
                          const KeyType* offsets,
                          const void* weights,
                          const int batch_size,
                          const int num_hots,
                          const cuembed::CombineMode mode,
                          void* ret,
                          const cudaStream_t stream,
                          DataType_t value_dtype) {
  switch (value_dtype) {
      case DataType_t::Float32:
          cuembed::EmbeddingForward<float, float, KeyType, KeyType, false>(
              reinterpret_cast<const float*>(embedding_table),
              embed_width_in_bytes / sizeof(float),
              reinterpret_cast<const KeyType*>(indices), offsets,
              reinterpret_cast<const float*>(weights),
              batch_size, num_hots, 
              static_cast<cuembed::CombineMode>(mode),
              reinterpret_cast<float*>(ret), stream);
          break;
      case DataType_t::Float16:
          cuembed::EmbeddingForward<__half, __half, KeyType, KeyType, false>(
              reinterpret_cast<const __half*>(embedding_table),
              embed_width_in_bytes / sizeof(__half),
              reinterpret_cast<const KeyType*>(indices), offsets,
              reinterpret_cast<const __half*>(weights),
              batch_size, num_hots,
              static_cast<cuembed::CombineMode>(mode),
              reinterpret_cast<__half*>(ret), stream);
          break;
      default:
          NVE_LOG_ERROR_("Unsupported Data type");
          throw std::invalid_argument(std::string("Unsupported Data type"));
  };
}

template <typename KeyType>
int64_t gpu_find_raw(context_ptr_t& ctx, const GPUEmbeddingLayerConfig& config,
                     int64_t num_keys, const void* keys, void* values) {
  NVE_CHECK_(ctx != nullptr, "Invalid execution context");
  auto lookup_stream{ctx->get_lookup_stream()};

  NVE_CHECK_(num_keys <= INT_MAX, "Number of keys exceeding max int value is not supported");
  NVE_CHECK_(config.embedding_width_in_bytes <= INT_MAX, "Embedding width exceeding max int value is not supported");

  if (is_quant_rowwise(config.value_dtype)) {
    // Reuse the Float16 identity specialization as a byte-preserving raw-row gather. Quantized GPU
    // rows are required to have an even byte width, so treating the complete row (including scale
    // and offset metadata) as half elements copies every bit without interpreting the payload.
    typename ECNoCache<KeyType>::CacheData cache{};
    cache.row_size_in_bytes = static_cast<uint32_t>(config.embedding_width_in_bytes);
    const cudaError_t status = call_find_and_dequant<KeyType, typename ECNoCache<KeyType>::CacheData>(
        static_cast<const KeyType*>(keys), static_cast<size_t>(num_keys),
        static_cast<int8_t*>(values), static_cast<const int8_t*>(config.embedding_table),
        DataType_t::Float16, DataType_t::Float16,
        static_cast<uint32_t>(config.embedding_width_in_bytes / sizeof(__half)), cache,
        lookup_stream, static_cast<size_t>(config.embedding_width_in_bytes), true /*load_indices*/);
    NVE_CHECK_(status, "Raw quantized GPU lookup failed");
    return -1;
  }

  // need to optimize hotness (inner loop) value
  call_cuembed_forward<KeyType>(
      config.embedding_table,
      static_cast<int>(config.embedding_width_in_bytes),
      keys, nullptr, nullptr,
      static_cast<int>(num_keys),
      static_cast<int>(1), cuembed::CombineMode::kConcat,
      values, lookup_stream, config.value_dtype);

  return -1;
}

template <typename KeyType>
void cuembed_find_and_combine(context_ptr_t& ctx, const GPUEmbeddingLayerConfig& config,
                              int64_t num_keys, const void* keys,
                              SparseType_t hot_type, int64_t num_offsets,
                              const KeyType* offsets, int64_t fixed_hotness,
                              PoolingType_t pooling_type, const void* weights,
                              void* values) {

  if (is_weighted_pooling(pooling_type)) {
    NVE_CHECK_(weights != nullptr, "Weights should be provided for weighted sum/mean pooling");
  }
  NVE_CHECK_(ctx != nullptr, "Invalid execution context");
  
  auto lookup_stream{ctx->get_lookup_stream()};

  cuembed::CombineMode mode = cuembed::CombineMode::kConcat;
  switch (pooling_type) {
    case PoolingType_t::WeightedSum:
    case PoolingType_t::Sum:
      mode = cuembed::CombineMode::kSum;
      break;
    case PoolingType_t::Mean:
    case PoolingType_t::WeightedMean:
      mode = cuembed::CombineMode::kMean;
      break;
    case PoolingType_t::Concatenate:
      mode = cuembed::CombineMode::kConcat;
      break;
    default:
      NVE_LOG_ERROR_("Invalid Pooling type");
      throw std::invalid_argument(std::string("Invalid Pooling type"));
      break;
  }

  switch (hot_type) {
    case SparseType_t::Fixed:
    {
      if (num_keys % fixed_hotness > 0) {
        NVE_LOG_ERROR_("Number of keys doesn't divide by fixed hotness");
        throw std::invalid_argument("Invalid number of keys for fixed hotness");
      }
      int64_t batch_size = num_keys / fixed_hotness;
      NVE_CHECK_(batch_size <= INT_MAX, "Batch size exceeding max int value is not supported");
      NVE_CHECK_(fixed_hotness <= INT_MAX, "Hotness exceeding max int value is not supported");
      NVE_CHECK_(config.embedding_width_in_bytes <= INT_MAX, "Embedding width exceeding max int value is not supported");

      call_cuembed_forward<KeyType>(
          config.embedding_table,
          static_cast<int>(config.embedding_width_in_bytes),
          keys, nullptr, weights,
          static_cast<int>(batch_size),
          static_cast<int>(fixed_hotness), mode,
          values, lookup_stream, config.value_dtype);
      break;
    }
    case SparseType_t::CSR:
    {
      NVE_CHECK_(num_offsets <= INT_MAX, "Batch size exceeding max int value is not supported");
      NVE_CHECK_(config.embedding_width_in_bytes <= INT_MAX, "Embedding width exceeding max int value is not supported");

      call_cuembed_forward<KeyType>(
          config.embedding_table,
          static_cast<int>(config.embedding_width_in_bytes),
          keys, offsets, weights,
          static_cast<int>(num_offsets),
          0, mode, values, lookup_stream, config.value_dtype);
      break;
    }
    default:
      NVE_LOG_ERROR_("Invalid Hotness type");
      throw std::invalid_argument(std::string("Invalid Hotness type"));
  }
}

template <typename KeyType>
GPUEmbeddingLayer<KeyType>::GPUEmbeddingLayer(const GPUEmbeddingLayerConfig& config,
                                              allocator_ptr_t allocator)
  : config_(config) {
    NVE_CHECK_(static_cast<bool>(config_.embedding_table), "Invalid embedding table");
    NVE_CHECK_(config_.num_embeddings > 0, "Invalid number of embeddings");
    NVE_CHECK_(config_.value_dtype == DataType_t::Float32 ||
                   config_.value_dtype == DataType_t::Float16 ||
                   is_quant_rowwise(config_.value_dtype),
               "Unsupported GPU embedding table value type");
    if (is_quant_rowwise(config_.value_dtype)) {
      NVE_CHECK_(config_.embedding_width_in_bytes % sizeof(__half) == 0,
                 "Quantized GPU embedding row width must be even");
      NVE_CHECK_(config_.embedding_width_in_bytes > quant_rowwise_meta_bytes(config_.value_dtype),
                 "Quantized GPU embedding rows must contain values in addition to metadata");
    }
    NVE_CHECK_(static_cast<uint64_t>(config_.num_embeddings) <=
                   static_cast<uint64_t>(std::numeric_limits<KeyType>::max()),
               "Number of embeddings cannot be represented by the layer key type");
    NVE_CHECK_(config_.default_row_index < config_.num_embeddings,
               "Default row index must be less than num_embeddings");

    allocator_ = allocator ? allocator : GetDefaultAllocator();
    NVE_CHECK_(allocator_ != nullptr, "Failed to get default allocator");
    
    // Initialize context registry
    contexts_ = std::make_shared<ContextRegistry>();
    NVE_CHECK_(contexts_ != nullptr, "Failed to create context registry");
    
    // create modify event
    NVE_CHECK_(cudaEventCreate(&modify_in_progress_, cudaEventDisableTiming));

    // create a stream for modifies
    NVE_CHECK_(cudaStreamCreate(&private_modify_stream_));
}

template <typename KeyType>
GPUEmbeddingLayer<KeyType>::~GPUEmbeddingLayer() {
  // destroy modify event and stream (will be destroyed once the work is finished)
  NVE_CHECK_(cudaEventDestroy(modify_in_progress_));
  NVE_CHECK_(cudaStreamDestroy(private_modify_stream_));
}

template <typename KeyType>
void GPUEmbeddingLayer<KeyType>::lookup(context_ptr_t& ctx, const int64_t num_keys, const void* keys, void* output,
                    const int64_t output_stride, bitmask64_t* hitmask,
                    const PoolingParams* pool_params, float* hitrates) {
  NVE_NVTX_SCOPED_FUNCTION_COL1_();
  ScopedDevice scope_device(config_.device_id);
  NVE_CHECK_ARG_(num_keys >= 0, "Invalid num_keys");
  NVE_CHECK_ARG_(keys != nullptr, "Invalid keys buffer");
  NVE_CHECK_ARG_(output != nullptr, "Invalid output buffer");
  NVE_CHECK_ARG_(hitmask == nullptr, "Hitmask is not supported for GPU embedding layer");
  NVE_CHECK_ARG_(ctx != nullptr, "Invalid layer context");
  NVE_CHECK_ARG_(num_keys <= INT_MAX, "Number of keys exceeding max int value is not supported");
  NVE_CHECK_(config_.embedding_width_in_bytes <= INT_MAX,
             "Embedding width exceeding max int value is not supported");

  const bool quantized = is_quant_rowwise(config_.value_dtype);
  const bool concatenate =
      pool_params && pool_params->pooling_type == PoolingType_t::Concatenate;
  const bool raw_concatenate = is_pooling_raw_concat(pool_params, config_.value_dtype);
  if (pool_params) {
    validate_pool_params(*pool_params);
    if (raw_concatenate) {
      NVE_CHECK_ARG_(output_stride == config_.embedding_width_in_bytes,
                 "Raw Concatenate output stride must be the same as the stored row width");
    } else if (quantized) {
      validate_quant_pool_types_and_stride(config_, *pool_params, output_stride);
    } else {
      NVE_CHECK_ARG_(output_stride == config_.embedding_width_in_bytes,
                     "Output stride must be the same as embedding width");
      NVE_CHECK_ARG_(pool_params->output_type == config_.value_dtype,
                     "GPU embedding layer requires pooled output type to match the table value type");
      if (!concatenate && pool_params->weights != nullptr) {
        NVE_CHECK_ARG_(pool_params->weight_type == config_.value_dtype,
                       "GPU embedding layer requires weight type to match the table value type");
      }
    }
  } else {
    NVE_CHECK_ARG_(output_stride == config_.embedding_width_in_bytes,
                   "Output stride must be the same as embedding width");
  }

  const cudaStream_t lookup_stream = ctx->get_lookup_stream();

  // Make sure keys are device accessible
  const auto keys_buffer_size = static_cast<size_t>(sizeof(KeyType)) * num_keys;
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, keys_buffer_size);
  const void* d_keys = keys_bw->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, lookup_stream);

  // Keys outside the table are replaced with the default row, so the gather below stays in bounds
  // (no-op when no default row is configured).
  d_keys = sanitize_lookup_keys<KeyType>(ctx, d_keys, num_keys, config_.num_embeddings,
                                         config_.default_row_index, lookup_stream);

  if (quantized) {
    if (pool_params && !raw_concatenate) {
      const auto gather_buffer_size =
          static_cast<size_t>(num_keys) * static_cast<size_t>(config_.embedding_width_in_bytes);
      auto* gather_dev = static_cast<int8_t*>(
          ctx->get_buffer("gpu_pool_gather", gather_buffer_size, false /*host_alloc*/));
      NVE_CHECK_(gather_dev != nullptr, "Failed to allocate quantized GPU pooling gather buffer");

      std::lock_guard lock(kernel_launch_mutex_);
      NVE_CHECK_(cudaStreamWaitEvent(lookup_stream, modify_in_progress_));
      gpu_find_raw<KeyType>(ctx, config_, num_keys, d_keys, gather_dev);
      pool_gathered_device<KeyType>(ctx, *pool_params, config_.value_dtype, gather_dev,
                                    config_.embedding_width_in_bytes, num_keys, output,
                                    output_stride, lookup_stream);
    } else {
      // Plain lookup and same-type Concatenate both return the complete stored row, including
      // rowwise-quantization metadata.
      const auto output_buffer_size =
          static_cast<size_t>(num_keys) * static_cast<size_t>(output_stride);
      auto output_bw =
          std::make_shared<BufferWrapper<void>>(ctx, "output", output, output_buffer_size);
      void* d_output =
          output_bw->access_buffer(cudaMemoryTypeDevice, false /*copy_content*/, lookup_stream);

      std::lock_guard lock(kernel_launch_mutex_);
      NVE_CHECK_(cudaStreamWaitEvent(lookup_stream, modify_in_progress_));
      gpu_find_raw<KeyType>(ctx, config_, num_keys, d_keys, d_output);
      if (output != d_output) {
        NVE_CHECK_(cudaMemcpyAsync(output, d_output, output_buffer_size, cudaMemcpyDefault,
                                   lookup_stream));
      }
    }

    if (hitrates) {
      hitrates[0] = 1.0f;
    }
    return;
  }

  const int64_t output_rows = get_lookup_output_rows(num_keys, pool_params);
  const auto output_buffer_size =
      static_cast<size_t>(output_rows) * static_cast<size_t>(output_stride);
  auto output_bw =
      std::make_shared<BufferWrapper<void>>(ctx, "output", output, output_buffer_size);
  void* d_output =
      output_bw->access_buffer(cudaMemoryTypeDevice, false /*copy_content*/, lookup_stream);

  // Lookup 
  if (pool_params && !concatenate) {
    const KeyType* d_offsets = nullptr;
    int64_t hotness = pool_params->fixed_hotness;
    int64_t num_offsets = 0;
    std::shared_ptr<BufferWrapper<const KeyType>> offsets_bw;

    if (pool_params->sparse_type == SparseType_t::CSR) {
      num_offsets = pool_params->num_csr_offsets - 1;
      const auto offsets_buffer_size =
          static_cast<size_t>(pool_params->num_csr_offsets) * sizeof(KeyType);
      offsets_bw = std::make_shared<BufferWrapper<const KeyType>>(
          ctx, "offsets", static_cast<const KeyType*>(pool_params->csr_offsets),
          offsets_buffer_size);
      d_offsets = offsets_bw->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, lookup_stream);
    }

    const void* d_weights = nullptr;
    if (pool_params->weights != nullptr) {
      const auto weights_buffer_size = dtype_size(pool_params->weight_type) * num_keys;
      auto weights_bw = std::make_shared<BufferWrapper<const void>>(ctx, "weights", pool_params->weights, weights_buffer_size);
      d_weights = weights_bw->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, lookup_stream);

    }

    // Wait until no other lookup/modify is queued
    std::lock_guard lock(kernel_launch_mutex_);

    NVE_CHECK_(cudaStreamWaitEvent(lookup_stream, modify_in_progress_));

    cuembed_find_and_combine<KeyType>(ctx, config_,
                                      num_keys, d_keys, pool_params->sparse_type,
                                      num_offsets, d_offsets,
                                      hotness,
                                      pool_params->pooling_type, d_weights,
                                      d_output);
  } else {
    // Plain lookup and Concatenate both emit one stored row per key. Concatenate's sparse-layout
    // and weight fields are intentionally ignored; output_type compatibility was checked above.
    // Wait until no other lookup/modify is queued
    std::lock_guard lock(kernel_launch_mutex_);

    NVE_CHECK_(cudaStreamWaitEvent(lookup_stream, modify_in_progress_));
    gpu_find_raw<KeyType>(ctx, config_, num_keys, d_keys, d_output);
  }

  if (output != d_output) {
    // output was not on gpu, need to copy the results back to host.
    NVE_CHECK_(cudaMemcpyAsync(output, d_output, output_buffer_size, cudaMemcpyDefault, lookup_stream));
  }

  // Handle hitrates
  if (hitrates) {
    hitrates[0] = 1.0f;
  }
}

template <typename KeyType>
void GPUEmbeddingLayer<KeyType>::insert(context_ptr_t& /*ctx*/, const int64_t /*num_keys*/, const void* /*keys*/,
  const int64_t /*value_stride*/, const int64_t /*value_size*/, const void* /*values*/, const int64_t /*table_id*/) {
  NVE_NVTX_SCOPED_FUNCTION_COL2_();
  NVE_LOG_WARNING_("insert method has no effect for GPU layer, use update to change table content");
}

template <typename KeyType>
void GPUEmbeddingLayer<KeyType>::update(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
                    const int64_t value_stride, const int64_t value_size,
                    const void* values, const int64_t table_id) {
  NVE_NVTX_SCOPED_FUNCTION_COL3_();
  ScopedDevice scope_device(config_.device_id);
  NVE_CHECK_(num_keys >= 0, "Invalid num_keys");
  NVE_CHECK_(keys != nullptr, "Invalid keys buffer");
  NVE_CHECK_(values != nullptr, "Invalid values buffer");
  NVE_CHECK_(value_size == config_.embedding_width_in_bytes, "Invalid value size");
  NVE_CHECK_(table_id < get_num_tables(), "Invalid table_id");

  auto layer_ctx = std::dynamic_pointer_cast<LayerExecutionContext>(ctx);
  NVE_CHECK_(layer_ctx != nullptr, "Invalid layer context");
  const cudaStream_t modify_stream = layer_ctx->get_modify_stream();

  // Make sure keys are device accessible
  const auto keys_buffer_size = sizeof(KeyType) * num_keys;
  const auto values_buffer_size = num_keys * value_stride;
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, keys_buffer_size);
  auto values_bw = std::make_shared<BufferWrapper<const void>>(ctx, "values", values, values_buffer_size);
  const void* d_keys = keys_bw->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, modify_stream);
  const void* d_values = values_bw->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, modify_stream);

  // Lock to protect queueing of all stream sync events
  std::lock_guard lock(kernel_launch_mutex_);

  // Wait for all lookups and a modify (if any) issued before this one
  NVE_CHECK_(cudaStreamWaitEvent(private_modify_stream_, modify_in_progress_));
  StreamCoordinator sc(modify_stream, private_modify_stream_);

  std::shared_ptr<nve::DefaultECEvent> syncEvent = contexts_->create_sync_event();
  NVE_CHECK_(syncEvent->event_record());
  NVE_CHECK_(syncEvent->event_wait_stream(private_modify_stream_));

  // call update accumulate kernel, use private modify stream for execution
  UpdateTable<KeyType>(d_values, reinterpret_cast<const KeyType*>(d_keys),
                       config_.embedding_table, 
                       static_cast<uint32_t>(config_.embedding_width_in_bytes),
                       static_cast<uint32_t>(value_stride),
                       static_cast<uint32_t>(config_.embedding_width_in_bytes),
                       static_cast<int32_t>(num_keys),
                       static_cast<uint64_t>(config_.num_embeddings),
                       private_modify_stream_);

  NVE_CHECK_(cudaEventRecord(modify_in_progress_, private_modify_stream_));
  NVE_CHECK_(cudaStreamWaitEvent(modify_stream, modify_in_progress_));
}

template <typename KeyType>
void GPUEmbeddingLayer<KeyType>::accumulate(context_ptr_t& ctx, const int64_t num_keys, const void* keys,
                        const int64_t value_stride, const int64_t value_size, const void* values,
                        DataType_t value_type, const int64_t table_id) {
  NVE_NVTX_SCOPED_FUNCTION_COL4_();
  ScopedDevice scope_device(config_.device_id);
  NVE_CHECK_(num_keys >= 0, "Invalid num_keys");
  NVE_CHECK_(keys != nullptr, "Invalid keys buffer");
  NVE_CHECK_(values != nullptr, "Invalid values buffer");
  NVE_CHECK_(!is_quant_rowwise(config_.value_dtype),
             "Accumulate is not supported for quantized GPU embedding tables");
  NVE_CHECK_(value_size == config_.embedding_width_in_bytes, "Invalid value size");
  NVE_CHECK_(table_id < get_num_tables(), "Invalid table_id");

  auto layer_ctx = std::dynamic_pointer_cast<LayerExecutionContext>(ctx);
  NVE_CHECK_(layer_ctx != nullptr, "Invalid layer context");
  const cudaStream_t modify_stream = layer_ctx->get_modify_stream();

  // Make sure keys are device accessible
  const auto keys_buffer_size = sizeof(KeyType) * num_keys;
  const auto values_buffer_size = num_keys * value_stride;
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, keys_buffer_size);
  auto values_bw = std::make_shared<BufferWrapper<const void>>(ctx, "values", values, values_buffer_size);
  const void* d_keys = keys_bw->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, modify_stream);
  const void* d_values = values_bw->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, modify_stream);

  // Lock to protect queueing of all stream sync events
  std::lock_guard lock(kernel_launch_mutex_);

  // Wait for all lookups and a modify (if any) issued before this one
  NVE_CHECK_(cudaStreamWaitEvent(private_modify_stream_, modify_in_progress_));
  StreamCoordinator sc(modify_stream, private_modify_stream_);

  std::shared_ptr<nve::DefaultECEvent> syncEvent = contexts_->create_sync_event();
  NVE_CHECK_(syncEvent->event_record());
  NVE_CHECK_(syncEvent->event_wait_stream(private_modify_stream_));

  // call update accumulate kernel
  switch (value_type) {
      case DataType_t::Float32:
        {
          uint32_t embedding_width = static_cast<uint32_t>(config_.embedding_width_in_bytes / sizeof(float));
          uint32_t value_stride_elements = static_cast<uint32_t>(value_stride / sizeof(float));
          UpdateAccumulateTable<KeyType, float>(
              reinterpret_cast<const float*>(d_values), reinterpret_cast<const KeyType*>(d_keys),
              reinterpret_cast<float*>(config_.embedding_table),
              embedding_width, value_stride_elements, embedding_width,
              static_cast<int32_t>(num_keys),
              static_cast<uint64_t>(config_.num_embeddings), private_modify_stream_);
        }
        break;
      case DataType_t::Float16:
        {
          uint32_t embedding_width = static_cast<uint32_t>(config_.embedding_width_in_bytes / sizeof(__half));
          uint32_t value_stride_elements = static_cast<uint32_t>(value_stride / sizeof(__half));
          UpdateAccumulateTable<KeyType, __half>(
              reinterpret_cast<const __half*>(d_values), reinterpret_cast<const KeyType*>(d_keys),
              reinterpret_cast<__half*>(config_.embedding_table),
              embedding_width, value_stride_elements, embedding_width,
              static_cast<int32_t>(num_keys),
              static_cast<uint64_t>(config_.num_embeddings), private_modify_stream_);
        }
        break;
    default:
          NVE_LOG_ERROR_("Unsupported Data type");
          throw std::invalid_argument(std::string("Unsupported Data type"));
  }

  NVE_CHECK_(cudaEventRecord(modify_in_progress_, private_modify_stream_));
  NVE_CHECK_(cudaStreamWaitEvent(modify_stream, modify_in_progress_));
}

template <typename KeyType>
void GPUEmbeddingLayer<KeyType>::clear(context_ptr_t& /*ctx*/) { 
  NVE_NVTX_SCOPED_FUNCTION_COL5_();
  NVE_LOG_WARNING_("clear method has no effect for GPU layer");
}

template <typename KeyType>
void GPUEmbeddingLayer<KeyType>::erase(context_ptr_t& /*ctx*/, const int64_t /*num_keys*/, const void* /*keys*/,
  const int64_t /*table_id*/) {
  NVE_NVTX_SCOPED_FUNCTION_COL6_();
  NVE_LOG_WARNING_("erase method has no effect for GPU layer");
}

template class GPUEmbeddingLayer<int32_t>;
template class GPUEmbeddingLayer<int64_t>;

}  // namespace nve
