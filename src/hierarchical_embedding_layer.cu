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

#include <bitset>
#include <hierarchical_embedding_layer.hpp>
#include <table.hpp>
#include <host_table.hpp>
#include <gpu_table.hpp>
#include <default_allocator.hpp>
#include <vector>
#include <layer_utils.hpp>
#include "cuda_ops/scatter.cuh"
#include "cuda_ops/cuda_common.h"
#include "cuda_ops/pool_gathered.cuh"
#include <insert_heuristic.hpp>
#include <buffer_wrapper.hpp>
#include <cuda_support.hpp>

namespace nve {

template <typename KeyType>
context_ptr_t HierarchicalEmbeddingLayer<KeyType>::create_execution_context(
  cudaStream_t lookup_stream,
  cudaStream_t modify_stream,
  thread_pool_ptr_t thread_pool,
  allocator_ptr_t allocator) {
  allocator_ptr_t actual_allocator = allocator ? allocator : allocator_;
  std::vector<context_ptr_t> table_contexts;
  for (auto& t : tables_) {
    auto ctx = t->create_execution_context(lookup_stream, modify_stream, thread_pool, actual_allocator);
    NVE_CHECK_(ctx != nullptr, "Failed to create table execution context");
    table_contexts.push_back(std::move(ctx));
  }
  return std::make_shared<LayerExecutionContext>(
    lookup_stream,
    modify_stream,
    thread_pool,
    actual_allocator,
    table_contexts);
}

template <typename KeyType>
HierarchicalEmbeddingLayer<KeyType>::HierarchicalEmbeddingLayer(
  const Config& cfg,
  const std::vector<table_ptr_t>& tables,
  allocator_ptr_t allocator)
  : config_(cfg), tables_(tables) {
  NVE_CHECK_(tables_.size() > 0, "No tables provided to layer");
  allocator_ = allocator ? allocator : GetDefaultAllocator();
  NVE_CHECK_(allocator_ != nullptr, "Failed to get default allocator");
  gpu_device_ = -1;
  for (size_t i=0; i < tables_.size(); i++) {
    auto table = tables_.at(i);
    NVE_CHECK_(table != nullptr, "Invalid table");
    auto device = table->get_device_id();
    if (device >= 0) {
      NVE_CHECK_((i == 0) || (device == tables_.at(i-1)->get_device_id()), "GPU table must have the same device as the previous table");
      // We don't handle GPU tables of different devices or GPU tables following CPU tables (makes no perf. sense)
      // The call to scatter in the lookup relies on this (i.e. the device hitmask does not contain any previous hits from CPU tables)
      gpu_device_ = device;
    }
    NVE_CHECK_(table->get_max_row_size() == tables_.at(0)->get_max_row_size(), "All layer tables must have the same row size");
  }
  if (config_.default_embedding.size() > 0) {
    NVE_CHECK_(tables_.at(0)->get_max_row_size() == static_cast<int64_t>(config_.default_embedding.size()), "Default embedding row size must match table row size");
    NVE_CHECK_((*tables_.rbegin())->get_device_id() < 0, "Default embedding row requires last table to be on host");
  }
  auto heuristic = config_.insert_heuristic;
  if (!heuristic) {
    // Build default thresholds: DEFAULT_THRESHOLD for GPU and host tables, 0.0 for the last host table
    std::vector<float> thresholds;
    size_t last_host_idx = tables_.size(); // sentinel: no host tables
    for (size_t i = tables_.size(); i > 0; --i) {
      if (tables_.at(i - 1)->get_device_id() < 0) {
        last_host_idx = i - 1;
        break;
      }
    }
    for (size_t i = 0; i < tables_.size(); i++) {
      if (i == last_host_idx) {
        thresholds.push_back(0.0f);
      } else {
        thresholds.push_back(DefaultInsertHeuristic::DEFAULT_THRESHOLD);
      }
    }
    heuristic = std::make_shared<DefaultInsertHeuristic>(thresholds);
  }
  for (size_t i=0; i < tables_.size(); i++) {
    auto table = tables_.at(i);
    bool gpu_table = (table->get_device_id() >= 0);
    KeyType invalid_key = static_cast<KeyType>(table->get_invalid_key());
    auto_insert_handlers_.push_back(std::make_shared<AutoInsertHandler>(
      heuristic,
      table,
      i,
      allocator_,
      gpu_table ? config_.min_insert_freq_gpu : config_.min_insert_freq_host,
      gpu_table ? config_.min_insert_size_gpu : config_.min_insert_size_host,
      sizeof(KeyType),
      gpu_device_,
      &invalid_key
    ));
  }
}

template <typename KeyType>
HierarchicalEmbeddingLayer<KeyType>::~HierarchicalEmbeddingLayer() {
}

template <typename KeyType>
void HierarchicalEmbeddingLayer<KeyType>::lookup(context_ptr_t& ctx, const int64_t num_keys,
                                                 const void* keys, void* output,
                                                 const int64_t output_stride,
                                                 bitmask64_t* output_hitmask,
                                                 const PoolingParams* pool_params,
                                                 float* hitrates) {
  NVE_NVTX_SCOPED_FUNCTION_COL1_();
  std::shared_ptr<ScopedDevice> scope_device = nullptr;
  if (gpu_device_ >= 0) {
    scope_device = std::make_shared<ScopedDevice>(gpu_device_);
  }
  auto layer_ctx = std::dynamic_pointer_cast<LayerExecutionContext>(ctx);
  NVE_CHECK_(layer_ctx != nullptr, "Invalid layer context");
  const cudaStream_t lookup_stream = layer_ctx->get_lookup_stream();
  const bool pooling_requested = (pool_params != nullptr);
  if (pooling_requested) {
    validate_pool_params(*pool_params);
    for (const auto& table : tables_) {
      NVE_CHECK_(table->get_value_type() == tables_.at(0)->get_value_type(),
                 "All layer tables must have the same value type for pooling");
    }
  }

  const int64_t row_size = tables_.at(0)->get_max_row_size();
  // check for avoidable concat with no dequant, then use the lookup path to avoid a redundant copy.
  const bool avoidable_concat =
      pooling_requested && pool_params->pooling_type == PoolingType_t::Concatenate &&
      pool_params->output_type == tables_.at(0)->get_value_type();
  const bool do_pool = pooling_requested && !avoidable_concat;

  // For pooling/dequant we gather one raw stored row per key into a scratch buffer, then run the
  // post-processing kernel over it into the user output. Plain lookup and same-type Concatenate
  // gather straight into the user output. gather_stride is the raw stored row stride for pooling.
  const int64_t gather_stride = do_pool ? row_size : output_stride;

  const auto hitmask_elements = to_uint(ceil_div(num_keys, bitmask64::num_bits));
  const auto hitmask_buffer_size = hitmask_elements * sizeof(bitmask64_t);
  const auto output_buffer_size = num_keys * output_stride;
  const auto gather_buffer_size = num_keys * gather_stride;
  const auto key_buffer_size = sizeof(KeyType) * num_keys;

  // Prepare buffer wrappers
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, key_buffer_size);
  // gather_bw is the target the tiers gather raw rows into: a scratch buffer for post-processing,
  // or the user output buffer directly for plain lookup / same-type Concatenate.
  std::shared_ptr<BufferWrapper<void>> gather_bw;
  if (do_pool) {
    // With a GPU tier the dequant/combine kernels run on the device, so the scratch is
    // device-resident; without one the CPU pool path needs it host-resident.
    auto gather_buf = ctx->get_buffer("pool_gather", gather_buffer_size, gpu_device_ < 0 /*host_alloc*/);
    gather_bw = std::make_shared<BufferWrapper<void>>(ctx, "pool_gather", gather_buf, gather_buffer_size);
  } else {
    gather_bw = std::make_shared<BufferWrapper<void>>(ctx, "output", output, output_buffer_size);
  }
  std::shared_ptr<BufferWrapper<bitmask64_t>> hitmask_bw(nullptr);

  // allocate hitmask buffer if needed
  const auto first_device = (*tables_.begin())->get_device_id();
  if (output_hitmask == nullptr) {
    auto hitmask_buf = reinterpret_cast<bitmask64_t*>(
        ctx->get_buffer("hitmask", hitmask_buffer_size, first_device < 0 /*host_alloc*/));
    hitmask_bw = std::make_shared<BufferWrapper<bitmask64_t>>(ctx, "hitmask", hitmask_buf, hitmask_buffer_size);
  } else {
    hitmask_bw = std::make_shared<BufferWrapper<bitmask64_t>>(ctx, "hitmask", output_hitmask, hitmask_buffer_size);
  }
  // zero initial hitmask
  const auto hitmask_first_access = hitmask_bw->get_last_access();
  auto hitmask_buf = hitmask_bw->access_buffer(hitmask_first_access, false /*copy_content*/, lookup_stream);
  if (hitmask_first_access == cudaMemoryTypeDevice) {
    NVE_CHECK_(cudaMemsetAsync(hitmask_buf, 0, hitmask_buffer_size, lookup_stream));
  } else {
    // guaranteed to be host accessible
    std::memset(hitmask_buf, 0, hitmask_buffer_size);
  }

  // Lookup tables
  const auto num_tables = tables_.size();
  std::vector<int64_t> table_hits(num_tables);
  std::vector<float> table_hitrates(num_tables);

  for (size_t i=0; i < num_tables; i++) {
    auto table = tables_.at(i);
    auto table_ctx = layer_ctx->table_contexts_.at(i);
    NVE_CHECK_(table_ctx != nullptr, "Invalid table context");

    // Call table->find and collect counters
    table->reset_lookup_counter(table_ctx);
    std::shared_ptr<BufferWrapper<int64_t>> value_sizes{nullptr}; // for now variable value size is not supported at the layer level
    table->find(table_ctx, num_keys, keys_bw, hitmask_bw, gather_stride, gather_bw, std::move(value_sizes));
    table->get_lookup_counter(table_ctx, table_hits.data() + i);
  }

  // Handle hit counters
  const bool first_table_on_gpu = (*tables_.begin())->get_device_id() >= 0;
  const bool last_table_on_host = (*tables_.rbegin())->get_device_id() < 0;
  if (first_table_on_gpu && !last_table_on_host) {
    NVE_CHECK_(cudaStreamSynchronize(lookup_stream)); // Synchronizing for reading counters from at least one gpu table
    // Optimized out when we have both GPU and CPU tables and final output is on GPU
    // in this case we already had a sync copying the hitmask from GPU to CPU
  }
  int64_t left_keys = num_keys;
  for (size_t i=0; i<table_hits.size(); i++) {
    if (tables_.at(i)->lookup_counter_hits() == false) {
      // Table counts misses
      table_hits.at(i) = left_keys - table_hits.at(i);
    }
    const auto hits = table_hits.at(i);
    // A tier left with no candidate keys missed nothing: report 1.0 rather than 0/0 = NaN, which
    // would reach the insert heuristic and poison StatisticalInsertHeuristic's statistics.
    table_hitrates[i] = (left_keys > 0) ? static_cast<float>(hits) / static_cast<float>(left_keys) : 1.0f;
    left_keys -= hits;
  }
  // Update hitrates
  if (hitrates) {
    for (size_t i=0; i<num_tables; i++) {
      hitrates[i] = static_cast<float>(table_hits[i]) / static_cast<float>(num_keys);
    }
  }

  // Handle default embedding
  if ((left_keys > 0) && (config_.default_embedding.size() > 0)) {
    NVE_ASSERT_((*tables_.rbegin())->get_device_id() < 0); // Last table must be on host (checked during construction)
                                                           // Relying on this to have triggered a sync if needed (copying hitmask from gpu to host)

    // Get buffers (shouldn't trigger any copy since last layer was already on host)
    auto* hit_mask_buf = hitmask_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, ctx->get_lookup_stream());
    auto* output_buf = gather_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, ctx->get_lookup_stream());

    // Fill default embedding rows for keys missing from all tables, in parallel via the context's thread pool.
    auto thread_pool = layer_ctx->get_thread_pool();
    const int64_t num_workers = thread_pool->num_workers();
    const int64_t keys_per_task = std::max<int64_t>(1, ceil_div(num_keys, num_workers));
    const int64_t num_tasks = ceil_div(num_keys, keys_per_task);
    const size_t default_row_size = static_cast<size_t>(row_size);
    const uint8_t* default_emb = config_.default_embedding.data();
    auto* output_bytes = static_cast<uint8_t*>(output_buf);

    const auto fill_default_task = [=](const int64_t idx) {
      const int64_t start_key = idx * keys_per_task;
      const int64_t end_key = std::min<int64_t>(start_key + keys_per_task, num_keys);
      for (int64_t k = start_key; k < end_key; k++) {
        const bitmask64_t elem{hit_mask_buf[k / bitmask64::num_bits]};
        const bool bit{bitmask64::get(elem, k % bitmask64::num_bits)};
        if (!bit) {
          std::memcpy(output_bytes + k * gather_stride, default_emb, default_row_size);
        }
      }
    };
    thread_pool->execute_n(0, num_tasks, fill_default_task);
  }

  // Combine hits
  auto h_output = gather_bw->get_buffer(cudaMemoryTypeHost);
  bool unregistered_output = false;
  if (h_output == nullptr) {
    // output buffer may be unregistered so we can't assume host output will be cudaMemoryTypeHost
    h_output = gather_bw->get_buffer(cudaMemoryTypeUnregistered);
    unregistered_output = true;
  }
  auto d_output = gather_bw->get_buffer(cudaMemoryTypeDevice);

  if (first_table_on_gpu && last_table_on_host) {
    // Need to scatter reolved hits from host to device
    if (unregistered_output) {
      NVE_LOG_WARNING_("Output buffer isn't CUDA registered, scatter will trigger extra copies!");
      h_output = gather_bw->access_buffer(cudaMemoryTypeHost, true, lookup_stream);
    }
    EmbeddingForwardScatter(
      h_output,
      gather_bw->access_buffer(cudaMemoryTypeDevice, false /*copy_content*/, lookup_stream), // using access_buffer so last_access is updated in the wrapper
      static_cast<uint32_t>(row_size),
      static_cast<uint32_t>(gather_stride),
      static_cast<uint32_t>(gather_stride),
      reinterpret_cast<uint64_t*>(hitmask_bw->get_buffer(cudaMemoryTypeDevice)), // We use the GPU hitmask to copy everything missed by the GPU table (potentially default values too)
      static_cast<int32_t>(num_keys),
      lookup_stream);
  }

  if (do_pool) {
    if (gpu_device_ >= 0) {
      // The gathered raw rows are now device-resident; pool/dequant on the GPU.
      pool_gathered_device<KeyType>(ctx, *pool_params, tables_.at(0)->get_value_type(),
                                    static_cast<const int8_t*>(d_output), gather_stride, num_keys,
                                    output, output_stride, lookup_stream);
    } else {
      // All tables on host
      auto* gather_host = static_cast<const int8_t*>(
          gather_bw->access_buffer(cudaMemoryTypeUnregistered, /*copy_content=*/true, lookup_stream));
      pool_gathered_host<KeyType>(ctx, *pool_params, tables_.at(0)->get_value_type(), gather_host,
                                  gather_stride, row_size, num_keys, output, output_stride,
                                  ctx->get_thread_pool()->num_workers());
    }
  } else {
    // Direct gather: copy to the caller only if BufferWrapper had to use a different residency.
    void* final_output = gpu_device_ >= 0 ? d_output : h_output;
    if (final_output != output) {
      NVE_CHECK_(cudaMemcpyAsync(output, final_output, output_buffer_size, cudaMemcpyDefault, lookup_stream));
    }
  }

  // Handle copy to output hitmask if needed
  void* final_hitmask = hitmask_bw->get_buffer(hitmask_bw->get_last_access());
  if ( (output_hitmask != nullptr) && (output_hitmask != final_hitmask) ) {
    NVE_CHECK_(cudaMemcpyAsync(output_hitmask, final_hitmask, hitmask_buffer_size, cudaMemcpyDefault, lookup_stream));
  }

  // Handle automatic inserts. Use gather_bw as we need the raw unpooled values.
  if (!auto_insert_handlers_.empty()) {
    for (size_t i=0; i<table_hits.size(); i++) {
      auto_insert_handlers_.at(i)->auto_insert(
        layer_ctx, keys_bw, gather_bw, table_hitrates[i], num_keys, gather_stride,
        (left_keys > 0) ? hitmask_bw : nullptr // If there are no unbresolved keys then we can skip processing the hitmask
      );
    }
  }
}

template <typename KeyType>
void HierarchicalEmbeddingLayer<KeyType>::insert(context_ptr_t& ctx, const int64_t num_keys,
                                                 const void* keys, const int64_t value_stride,
                                                 const int64_t value_size, const void* values,
                                                 const int64_t table_id) {
  NVE_NVTX_SCOPED_FUNCTION_COL2_();
  if (num_keys < 1) {
    return;
  }
  NVE_CHECK_(keys != nullptr, "Invalid Keys buffer");
  NVE_CHECK_(table_id < get_num_tables(), "Invalid table_id");
  auto layer_ctx = std::dynamic_pointer_cast<LayerExecutionContext>(ctx);
  NVE_CHECK_(layer_ctx != nullptr, "Invalid layer context");

  const auto values_buffer_size = num_keys * value_stride;
  const auto key_buffer_size = sizeof(KeyType) * num_keys;
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, key_buffer_size);
  auto values_bw = std::make_shared<BufferWrapper<const void>>(ctx, "values", values, values_buffer_size);

  std::shared_ptr<ScopedDevice> scope_device = nullptr;
  if (gpu_device_ >= 0) { // Need scoped device for potential copies in buffer wrappers
    scope_device = std::make_shared<ScopedDevice>(gpu_device_);
  }

  const size_t t_start = table_id < 0 ? 0 : static_cast<size_t>(table_id);
  const size_t t_end = table_id < 0 ? tables_.size() : static_cast<size_t>(table_id + 1);
  for (size_t i = t_start; i < t_end; i++) {
    auto table = tables_.at(i);
    NVE_CHECK_(table != nullptr);
    auto table_ctx = layer_ctx->table_contexts_.at(i);
    NVE_CHECK_(table_ctx != nullptr, "Invalid table context");
    table->insert(table_ctx, num_keys, keys_bw, value_stride, value_size, values_bw);
  }
}

template <typename KeyType>
void HierarchicalEmbeddingLayer<KeyType>::update(context_ptr_t& ctx, const int64_t num_keys,
                                                 const void* keys, const int64_t value_stride,
                                                 const int64_t value_size, const void* values,
                                                 const int64_t table_id) {
  NVE_NVTX_SCOPED_FUNCTION_COL3_();
  auto layer_ctx = std::dynamic_pointer_cast<LayerExecutionContext>(ctx);
  NVE_CHECK_(layer_ctx != nullptr, "Invalid layer context");
  const auto values_buffer_size = num_keys * value_stride;
  const auto key_buffer_size = sizeof(KeyType) * num_keys;
  NVE_CHECK_(table_id < get_num_tables(), "Invalid table_id");

  std::shared_ptr<ScopedDevice> scope_device = nullptr;
  if (gpu_device_ >= 0) {
    scope_device = std::make_shared<ScopedDevice>(gpu_device_);
  }

  const size_t t_start = table_id < 0 ? 0 : static_cast<size_t>(table_id);
  const size_t t_end = table_id < 0 ? tables_.size() : static_cast<size_t>(table_id + 1);
  for (size_t i = t_start; i < t_end; i++) {
    auto table = tables_.at(i);
    auto table_ctx = layer_ctx->table_contexts_.at(i);
    NVE_CHECK_(table_ctx != nullptr, "Invalid table context");
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, key_buffer_size);
    auto values_bw = std::make_shared<BufferWrapper<const void>>(ctx, "values", values, values_buffer_size);

    ScopedModifyLock modify_lock{handler_at(i)};
    table->update(table_ctx, num_keys, std::move(keys_bw), value_stride, value_size, std::move(values_bw));
  }
}

template <typename KeyType>
void HierarchicalEmbeddingLayer<KeyType>::accumulate(context_ptr_t& ctx, const int64_t num_keys,
                                                     const void* keys, const int64_t value_stride,
                                                     const int64_t value_size, const void* values,
                                                     DataType_t value_type, const int64_t table_id) {
  NVE_NVTX_SCOPED_FUNCTION_COL4_();
  auto layer_ctx = std::dynamic_pointer_cast<LayerExecutionContext>(ctx);
  NVE_CHECK_(layer_ctx != nullptr, "Invalid layer context");
  const cudaStream_t modify_stream = layer_ctx->get_modify_stream();
  const auto values_buffer_size = num_keys * value_stride;
  const auto key_buffer_size = sizeof(KeyType) * num_keys;
  NVE_CHECK_(table_id < get_num_tables(), "Invalid table_id");

  std::shared_ptr<ScopedDevice> scope_device = nullptr;
  if (gpu_device_ >= 0) {
    scope_device = std::make_shared<ScopedDevice>(gpu_device_);
  }

  const size_t t_start = table_id < 0 ? 0 : static_cast<size_t>(table_id);
  const size_t t_end = table_id < 0 ? tables_.size() : static_cast<size_t>(table_id + 1);
  for (size_t i = t_start; i < t_end; i++) {
    auto table = tables_.at(i);
    auto table_ctx = layer_ctx->table_contexts_.at(i);
    NVE_CHECK_(table_ctx != nullptr, "Invalid table context");
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, key_buffer_size);
    auto values_bw = std::make_shared<BufferWrapper<const void>>(ctx, "values", values, values_buffer_size);

    ScopedModifyLock modify_lock{handler_at(i)};
    table->update_accumulate(table_ctx, num_keys, std::move(keys_bw), value_stride, value_size, std::move(values_bw), value_type);
  }
}

template <typename KeyType>
void HierarchicalEmbeddingLayer<KeyType>::clear(context_ptr_t& ctx) {
  NVE_NVTX_SCOPED_FUNCTION_COL5_();
  auto layer_ctx = std::dynamic_pointer_cast<LayerExecutionContext>(ctx);
  NVE_CHECK_(layer_ctx != nullptr, "Invalid layer context");
  for (size_t i=0; i < tables_.size(); i++) {
    auto table = tables_.at(i);
    auto table_ctx = layer_ctx->table_contexts_.at(i);
    NVE_CHECK_(table_ctx != nullptr, "Invalid table context");

    ScopedModifyLock modify_lock{handler_at(i)};
    table->clear(table_ctx);
  }
}

template <typename KeyType>
void HierarchicalEmbeddingLayer<KeyType>::erase(context_ptr_t& ctx, const int64_t num_keys,
                                                const void* keys, const int64_t table_id) {
  NVE_NVTX_SCOPED_FUNCTION_COL6_();
  // Validate the target table before the empty-input shortcut, as the other layers do.
  NVE_CHECK_(table_id < get_num_tables(), "Invalid table_id");
  if (num_keys < 1) {
    return;
  }
  NVE_CHECK_(keys != nullptr, "Invalid Keys buffer");
  auto layer_ctx = std::dynamic_pointer_cast<LayerExecutionContext>(ctx);
  NVE_CHECK_(layer_ctx != nullptr, "Invalid layer context");
  const auto keys_buffer_size = sizeof(KeyType) * num_keys;
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, keys_buffer_size);

  std::shared_ptr<ScopedDevice> scope_device = nullptr;
  if (gpu_device_ >= 0) { // Need scoped device for potential copies in buffer wrappers
    scope_device = std::make_shared<ScopedDevice>(gpu_device_);
  }

  const size_t t_start = table_id < 0 ? 0 : static_cast<size_t>(table_id);
  const size_t t_end = table_id < 0 ? tables_.size() : static_cast<size_t>(table_id + 1);
  for (size_t i = t_start; i < t_end; i++) {
    auto table = tables_.at(i);
    NVE_CHECK_(table != nullptr);
    auto table_ctx = layer_ctx->table_contexts_.at(i);
    NVE_CHECK_(table_ctx != nullptr, "Invalid table context");

    ScopedModifyLock modify_lock{handler_at(i)};
    table->erase(table_ctx, num_keys, keys_bw);
  }
}

template <typename KeyType>
int64_t HierarchicalEmbeddingLayer<KeyType>::get_num_tables() const {
  return static_cast<int64_t>(tables_.size());
}

// Instantiate versions of HierarchicalEmbeddingLayer (todo: add more)
template class HierarchicalEmbeddingLayer<int32_t>;
template class HierarchicalEmbeddingLayer<int64_t>;

}  // namespace nve
