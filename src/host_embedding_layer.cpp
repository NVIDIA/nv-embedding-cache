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

#include "include/host_embedding_layer.hpp"
#include "include/table.hpp"
#include "include/buffer_wrapper.hpp"
#include "include/default_allocator.hpp"
#include "include/thread_pool.hpp"
#include "include/bit_ops.hpp"
#include "include/layer_utils.hpp"
#include "cpu_ops/cpu_pooling.hpp"
#include <cstring>

namespace nve {

template <typename KeyType>
HostEmbeddingLayer<KeyType>::HostEmbeddingLayer(const Config& cfg, table_ptr_t table,
                                                allocator_ptr_t allocator)
    : config_(cfg), allocator_(allocator ? std::move(allocator) : GetDefaultAllocator()), table_(std::move(table)) {
  NVE_CHECK_(table_ != nullptr, "Invalid table");
  NVE_CHECK_(table_->get_device_id() < 0, "HostEmbeddingLayer requires a host table (device_id < 0)");
  NVE_CHECK_(config_.task_size_threshold > 0, "Task size threshold needs to be greater than 0.");
  if (config_.default_embedding.size() > 0) {
    NVE_CHECK_(table_->get_max_row_size() == static_cast<int64_t>(config_.default_embedding.size()),
               "Default embedding row size must match table row size");
  }
}

template <typename KeyType>
HostEmbeddingLayer<KeyType>::~HostEmbeddingLayer() = default;

template <typename KeyType>
int64_t HostEmbeddingLayer<KeyType>::num_tasks_for(const context_ptr_t& ctx,
                                                    const int64_t num_keys) const {
  return select_num_tasks(num_keys, ctx->get_thread_pool()->num_workers(),
                          config_.task_size_threshold);
}

template <typename KeyType>
context_ptr_t HostEmbeddingLayer<KeyType>::create_execution_context(
    cudaStream_t lookup_stream, cudaStream_t modify_stream,
    thread_pool_ptr_t thread_pool, allocator_ptr_t allocator) {
  if (config_.disable_threadpool) {
    thread_pool = InlineThreadPool::instance();
  }
  return table_->create_execution_context(
      lookup_stream, modify_stream, std::move(thread_pool),
      allocator ? std::move(allocator) : allocator_);
}

template <typename KeyType>
void HostEmbeddingLayer<KeyType>::lookup(context_ptr_t& ctx, const int64_t num_keys,
                                         const void* keys, void* output,
                                         const int64_t output_stride,
                                         bitmask64_t* output_hitmask,
                                         const PoolingParams* pool_params,
                                         float* hitrates) {
  NVE_NVTX_SCOPED_FUNCTION_COL1_();
  NVE_CHECK_ARG_(ctx != nullptr, "Invalid context");
  NVE_CHECK_ARG_(num_keys >= 0, "Invalid number of keys");
  NVE_CHECK_ARG_(keys != nullptr || num_keys == 0, "Invalid keys");
  const cudaStream_t lookup_stream = ctx->get_lookup_stream();

  if (pool_params) {
    validate_pool_params(*pool_params);
  }

  // Empty lookup (e.g. an all-empty-bags batch): nothing to gather, but CSR pooling
  // may still have bags — zero them, matching torch.nn.EmbeddingBag and the GPU
  // layers, which accept num_keys == 0.
  if (num_keys == 0) {
    const int64_t output_rows = get_lookup_output_rows(num_keys, pool_params);
    if (output_rows > 0) {
      NVE_CHECK_ARG_(output != nullptr, "Invalid output");
      const size_t output_buffer_size =
          static_cast<size_t>(output_rows) * static_cast<size_t>(output_stride);
      auto output_bw = std::make_shared<BufferWrapper<void>>(ctx, "output", output, output_buffer_size);
      auto* output_host = output_bw->access_buffer(cudaMemoryTypeUnregistered,
                                                   /*copy_content=*/false, lookup_stream);
      std::memset(output_host, 0, output_buffer_size);
      auto* final_output = output_bw->get_buffer(cudaMemoryTypeUnregistered);
      if (final_output != nullptr && final_output != output) {
        NVE_CHECK_(cudaMemcpyAsync(output, final_output, output_buffer_size,
                                   cudaMemcpyDefault, lookup_stream));
      }
    }
    if (hitrates) {
      hitrates[0] = 1.f;  // vacuously, no key missed
    }
    return;
  }
  NVE_CHECK_ARG_(output != nullptr, "Invalid output");

  const size_t num_keys_sz = static_cast<size_t>(num_keys);
  const size_t hitmask_elements = to_uint(ceil_div(num_keys, bitmask64::num_bits));
  const size_t hitmask_buffer_size = hitmask_elements * sizeof(bitmask64_t);
  const size_t key_buffer_size = sizeof(KeyType) * num_keys_sz;
  const size_t row_size = static_cast<size_t>(table_->get_max_row_size());

  // Same-type Concatenate is a plain per-key memcpy: no scratch buffer, type conversion, or
  // reduction is needed. Treat it as the no-pooling path by clearing pool_params so the gather
  // writes final data directly into the caller's output buffer at the correct stride. All other
  // pooling/dequant work is delegated to pool_gathered_host after the gather.
  if (pool_params && (pool_params->pooling_type == PoolingType_t::Concatenate)) {
    const DataType_t in_dtype = table_->get_value_type();
    const DataType_t out_dtype = pool_params->output_type;
    if (in_dtype == out_dtype) {
      NVE_CHECK_ARG_(output_stride >= static_cast<int64_t>(row_size),
                     "Output stride is too small for same-type Concatenate: got ", output_stride,
                     " bytes, need at least ", row_size, " bytes (full row)");
      pool_params = nullptr;
    }
  }

  // When pooling, the table gathers all num_keys raw rows into a host scratch buffer (one row per
  // key), then pool_gathered_host reduces them into the caller's output. Otherwise it gathers
  // straight into the caller's output buffer (one row per key, num_keys rows).
  const size_t output_buffer_size = num_keys_sz * static_cast<size_t>(output_stride);
  const size_t gather_buffer_size = pool_params ? num_keys_sz * row_size : output_buffer_size;

  // Build buffer wrappers — LinearHostTable accesses them as cudaMemoryTypeUnregistered.
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, key_buffer_size);

  // When pooling, the table gathers into a host scratch buffer (one row per key) and
  // pool_gathered_host owns the caller's output; otherwise it gathers straight into the caller's
  // output buffer (output_bw, which we also push back below if it relocated).
  const int64_t gather_stride = pool_params ? static_cast<int64_t>(row_size) : output_stride;
  std::shared_ptr<BufferWrapper<void>> output_bw;
  std::shared_ptr<BufferWrapper<void>> gather_bw;
  if (pool_params) {
    void* gather_scratch =
        ctx->get_buffer("pool_gather", gather_buffer_size, true /*host_alloc*/);
    gather_bw = std::make_shared<BufferWrapper<void>>(ctx, "pool_gather", gather_scratch,
                                                      gather_buffer_size);
  } else {
    output_bw = std::make_shared<BufferWrapper<void>>(ctx, "output", output, output_buffer_size);
    gather_bw = output_bw;
  }

  // Hitmask handling. We need a hitmask buffer if either the caller wants one,
  // or we have a default embedding to fill for missed keys (we need to know which
  // keys missed). When the caller doesn't supply one but we still need it, allocate
  // a host-resident scratch buffer from the context. When neither applies, pass
  // null through to the table (it skips recording hits entirely).
  const bool need_hitmask = (output_hitmask != nullptr) || (config_.default_embedding.size() > 0);
  std::shared_ptr<BufferWrapper<bitmask64_t>> hitmask_bw;
  if (need_hitmask) {
    bitmask64_t* hitmask_ptr = output_hitmask;
    if (hitmask_ptr == nullptr) {
      hitmask_ptr = reinterpret_cast<bitmask64_t*>(
          ctx->get_buffer("hitmask", hitmask_buffer_size, true /*host_alloc*/));
    }
    hitmask_bw = std::make_shared<BufferWrapper<bitmask64_t>>(ctx, "hitmask", hitmask_ptr, hitmask_buffer_size);
    auto* hitmask_host = hitmask_bw->access_buffer(cudaMemoryTypeUnregistered,
                                                   /*copy_content=*/false,
                                                   lookup_stream);
    std::memset(hitmask_host, 0, hitmask_buffer_size);
  }

  // Single host table lookup (into the gather buffer, which is the caller's output
  // when not pooling, or a host scratch buffer of raw per-key rows when pooling).
  table_->reset_lookup_counter(ctx);
  std::shared_ptr<BufferWrapper<int64_t>> value_sizes{nullptr};
  table_->find(ctx, num_keys, std::move(keys_bw), hitmask_bw, gather_stride, gather_bw, std::move(value_sizes));
  int64_t hits = 0;
  table_->get_lookup_counter(ctx, &hits);
  if (!table_->lookup_counter_hits()) {
    hits = num_keys - hits;
  }
  const int64_t misses = num_keys - hits;

  // Default-embedding fill for keys that missed the table. Operates per-key on the
  // gather buffer (raw rows), before any pooling reduction.
  if (misses > 0 && config_.default_embedding.size() > 0) {
    // need_hitmask is true whenever default_embedding is set, so hitmask_bw exists here.
    NVE_ASSERT_(hitmask_bw != nullptr);
    auto* hit_mask_buf = hitmask_bw->access_buffer(cudaMemoryTypeUnregistered,
                                                   /*copy_content=*/true,
                                                   lookup_stream);
    auto* gather_buf = gather_bw->access_buffer(cudaMemoryTypeUnregistered,
                                                /*copy_content=*/true,
                                                lookup_stream);
    auto thread_pool = ctx->get_thread_pool();
    const int64_t num_tasks = num_tasks_for(ctx, num_keys);
    const int64_t keys_per_task = ceil_div(num_keys, num_tasks);
    const uint8_t* default_emb = config_.default_embedding.data();
    auto* gather_bytes = static_cast<uint8_t*>(gather_buf);

    const auto fill_default_task = [=](const int64_t idx) {
      const int64_t start_key = idx * keys_per_task;
      const int64_t end_key = std::min<int64_t>(start_key + keys_per_task, num_keys);
      for (int64_t k = start_key; k < end_key; k++) {
        const bitmask64_t elem{hit_mask_buf[k / bitmask64::num_bits]};
        const bool bit{bitmask64::get(elem, k % bitmask64::num_bits)};
        if (!bit) {
          std::memcpy(gather_bytes + k * gather_stride, default_emb, row_size);
        }
      }
    };
    thread_pool->execute_n(0, num_tasks, fill_default_task);
  }

  // Pooling/Dequant post-process: reduce the gathered raw rows into the caller's output. Shared
  // with the hierarchical layer's no-GPU-tier path.
  if (pool_params) {
    auto* gather_host = static_cast<int8_t*>(gather_bw->access_buffer(
        cudaMemoryTypeUnregistered, /*copy_content=*/true, lookup_stream));
    pool_gathered_host<KeyType>(ctx, *pool_params, table_->get_value_type(), gather_host,
                                gather_stride, static_cast<int64_t>(row_size), num_keys, output,
                                output_stride, num_tasks_for(ctx, num_keys));
  }

  // No-pooling path: copy back to the caller's output buffer if BufferWrapper picked up a
  // different host buffer (this happens when the caller's output is GPU / managed memory — the
  // table only knows how to gather into host; we then push the host slot back across the lookup
  // stream). The pooling path's copy-back is handled inside pool_gathered_host.
  //
  // Picking the copy primitive from the buffer's original residency: when
  // BufferWrapper allocated a separate host slot, the user-provided pointer
  // is not host-resident (Device / Managed), so we must go through
  // cudaMemcpyAsync. Note that the torch CUDA shim returns nullptr for the
  // legacy default stream — that's still a valid CUDA stream argument and
  // cudaMemcpyAsync handles it correctly.
  if (output_bw != nullptr) {
    auto* final_output = output_bw->get_buffer(cudaMemoryTypeUnregistered);
    if (final_output != nullptr && final_output != output) {
      NVE_CHECK_(cudaMemcpyAsync(output, final_output, output_buffer_size,
                                 cudaMemcpyDefault, lookup_stream));
    }
  }
  // Same for hitmask, when the caller supplied one.
  if (output_hitmask != nullptr) {
    auto* final_hitmask = hitmask_bw->get_buffer(hitmask_bw->get_last_access());
    if (final_hitmask != nullptr && final_hitmask != output_hitmask) {
      NVE_CHECK_(cudaMemcpyAsync(output_hitmask, final_hitmask, hitmask_buffer_size,
                                 cudaMemcpyDefault, lookup_stream));
    }
  }

  if (hitrates) {
    hitrates[0] = static_cast<float>(hits) / static_cast<float>(num_keys);
  }
}

template <typename KeyType>
void HostEmbeddingLayer<KeyType>::insert(context_ptr_t& ctx, const int64_t num_keys,
                                         const void* keys, const int64_t value_stride,
                                         const int64_t value_size, const void* values,
                                         const int64_t table_id) {
  NVE_NVTX_SCOPED_FUNCTION_COL2_();
  // Single-table layer: a negative table_id means "all tables", 0 is the host table; anything
  // else is invalid, as in the other layers.
  NVE_CHECK_(table_id < get_num_tables(), "Invalid table_id");
  const size_t key_buffer_size = sizeof(KeyType) * static_cast<size_t>(num_keys);
  const size_t values_buffer_size = static_cast<size_t>(num_keys) * static_cast<size_t>(value_stride);
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, key_buffer_size);
  auto values_bw = std::make_shared<BufferWrapper<const void>>(ctx, "values", values, values_buffer_size);
  table_->insert(ctx, num_keys, std::move(keys_bw), value_stride, value_size, std::move(values_bw));
}

template <typename KeyType>
void HostEmbeddingLayer<KeyType>::update(context_ptr_t& ctx, const int64_t num_keys,
                                         const void* keys, const int64_t value_stride,
                                         const int64_t value_size, const void* values,
                                         const int64_t table_id) {
  NVE_NVTX_SCOPED_FUNCTION_COL3_();
  // Single-table layer: a negative table_id means "all tables", 0 is the host table; anything
  // else is invalid, as in the other layers.
  NVE_CHECK_(table_id < get_num_tables(), "Invalid table_id");
  const size_t key_buffer_size = sizeof(KeyType) * static_cast<size_t>(num_keys);
  const size_t values_buffer_size = static_cast<size_t>(num_keys) * static_cast<size_t>(value_stride);
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, key_buffer_size);
  auto values_bw = std::make_shared<BufferWrapper<const void>>(ctx, "values", values, values_buffer_size);
  table_->update(ctx, num_keys, std::move(keys_bw), value_stride, value_size, std::move(values_bw));
}

template <typename KeyType>
void HostEmbeddingLayer<KeyType>::accumulate(context_ptr_t& ctx, const int64_t num_keys,
                                             const void* keys, const int64_t value_stride,
                                             const int64_t value_size, const void* values,
                                             DataType_t value_type, const int64_t table_id) {
  NVE_NVTX_SCOPED_FUNCTION_COL4_();
  // Single-table layer: a negative table_id means "all tables", 0 is the host table; anything
  // else is invalid, as in the other layers.
  NVE_CHECK_(table_id < get_num_tables(), "Invalid table_id");
  const size_t key_buffer_size = sizeof(KeyType) * static_cast<size_t>(num_keys);
  const size_t values_buffer_size = static_cast<size_t>(num_keys) * static_cast<size_t>(value_stride);
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, key_buffer_size);
  auto values_bw = std::make_shared<BufferWrapper<const void>>(ctx, "values", values, values_buffer_size);
  table_->update_accumulate(ctx, num_keys, std::move(keys_bw), value_stride, value_size,
                            std::move(values_bw), value_type);
}

template <typename KeyType>
void HostEmbeddingLayer<KeyType>::clear(context_ptr_t& ctx) {
  NVE_NVTX_SCOPED_FUNCTION_COL5_();
  table_->clear(ctx);
}

template <typename KeyType>
void HostEmbeddingLayer<KeyType>::erase(context_ptr_t& ctx, const int64_t num_keys,
                                        const void* keys, const int64_t table_id) {
  NVE_NVTX_SCOPED_FUNCTION_COL6_();
  // Single-table layer: a negative table_id means "all tables", 0 is the host table; anything
  // else is invalid, as in the other layers.
  NVE_CHECK_(table_id < get_num_tables(), "Invalid table_id");
  if (num_keys < 1) {
    return;
  }
  NVE_CHECK_(keys != nullptr, "Invalid Keys buffer");
  const size_t keys_buffer_size = sizeof(KeyType) * static_cast<size_t>(num_keys);
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, keys_buffer_size);
  table_->erase(ctx, num_keys, std::move(keys_bw));
}

template class HostEmbeddingLayer<int32_t>;
template class HostEmbeddingLayer<int64_t>;

}  // namespace nve
