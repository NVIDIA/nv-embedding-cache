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
#include "cpu_ops/cpu_pooling.h"
#include <cstring>

namespace nve {

template <typename KeyType>
HostEmbeddingLayer<KeyType>::HostEmbeddingLayer(const Config& cfg, table_ptr_t table,
                                                allocator_ptr_t allocator)
    : config_(cfg), allocator_(allocator ? allocator : GetDefaultAllocator()), table_(std::move(table)) {
  NVE_CHECK_(table_ != nullptr, "Invalid table");
  NVE_CHECK_(table_->get_device_id() < 0, "HostEmbeddingLayer requires a host table (device_id < 0)");
  if (config_.default_embedding.size() > 0) {
    NVE_CHECK_(table_->get_max_row_size() == static_cast<int64_t>(config_.default_embedding.size()),
               "Default embedding row size must match table row size");
  }
}

template <typename KeyType>
HostEmbeddingLayer<KeyType>::~HostEmbeddingLayer() = default;

template <typename KeyType>
context_ptr_t HostEmbeddingLayer<KeyType>::create_execution_context(
    cudaStream_t lookup_stream, cudaStream_t modify_stream,
    thread_pool_ptr_t thread_pool, allocator_ptr_t allocator) {
  return table_->create_execution_context(
      lookup_stream, modify_stream, std::move(thread_pool),
      allocator ? std::move(allocator) : allocator_);
}

template <typename KeyType>
void HostEmbeddingLayer<KeyType>::lookup(context_ptr_t& ctx, const int64_t num_keys,
                                         const void* keys, void* output,
                                         const int64_t output_stride,
                                         max_bitmask_repr_t* output_hitmask,
                                         const PoolingParams* pool_params,
                                         float* hitrates) {
  NVE_NVTX_SCOPED_FUNCTION_COL1_();
  NVE_CHECK_(ctx != nullptr, "Invalid context");
  NVE_CHECK_(keys != nullptr, "Invalid keys");
  NVE_CHECK_(output != nullptr, "Invalid output");
  NVE_CHECK_(num_keys > 0, "Invalid number of keys");
  const cudaStream_t lookup_stream = ctx->get_lookup_stream();

  const size_t num_keys_sz = static_cast<size_t>(num_keys);
  constexpr size_t hitmask_elem_bits = sizeof(max_bitmask_repr_t) * 8;
  const size_t hitmask_elements = (num_keys_sz + hitmask_elem_bits - 1) / hitmask_elem_bits;
  const size_t hitmask_buffer_size = hitmask_elements * sizeof(max_bitmask_repr_t);
  const size_t key_buffer_size = sizeof(KeyType) * num_keys_sz;
  const size_t row_size = static_cast<size_t>(table_->get_max_row_size());

  // Pooling metadata. When pooling, the table gathers all num_keys raw rows into a
  // host scratch buffer, then a CPU pooling op reduces them into num_bags output rows.
  DataType_t in_dtype = DataType_t::Unknown;
  DataType_t out_dtype = DataType_t::Unknown;
  int64_t row_width = 0;
  int64_t num_bags = num_keys;
  int64_t fixed_hotness = 1;
  SparseType_t effective_sparse = SparseType_t::Fixed;
  std::shared_ptr<BufferWrapper<const KeyType>> offsets_bw;
  const KeyType* offsets_host = nullptr;
  if (pool_params) {
    // Concatenate means no arithmetic reduction: each key yields one output row,
    // only converting/dequantizing the raw gathered data to the output type. The
    // sparse layout and weights are ignored (per the PoolingParams contract).
    const bool concat = (pool_params->pooling_type == PoolingType_t::Concatenate);

    in_dtype = table_->get_value_type();
    const bool quant_in = is_quant_rowwise(in_dtype);
    // For quantized input the result is dequantized to a float type unless the caller
    // requests same-type Concatenate (passthrough). For float input with no explicit
    // output type, the output keeps the table's value type.
    out_dtype = (pool_params->output_type != DataType_t::Unknown) ? pool_params->output_type
                : quant_in                                         ? DataType_t::Float32
                                                                   : in_dtype;
    NVE_CHECK_(in_dtype == DataType_t::Float32 || in_dtype == DataType_t::Float16 || quant_in,
               "HostEmbeddingLayer pooling supports only fp32/fp16 or QInt8/QUint8Rowwise table "
               "values, got: ", in_dtype);
    // Float32/Float16 output is required for all reduction modes. For Concatenate, a
    // same-type output is also accepted (no dequantization, full row is passed through).
    NVE_CHECK_(out_dtype == DataType_t::Float32 || out_dtype == DataType_t::Float16 ||
                   (concat && out_dtype == in_dtype),
               "HostEmbeddingLayer pooling supports fp32/fp16 output; same-type output is only "
               "accepted for Concatenate mode");
    if (quant_in) {
      // row_size is the full quantized row stride: value bytes + trailing scale[/offset].
      const int64_t meta_bytes = quant_rowwise_meta_bytes(in_dtype);
      NVE_CHECK_(static_cast<int64_t>(row_size) > meta_bytes,
                 "Quantized row size smaller than its scale/offset metadata");
      row_width = static_cast<int64_t>(row_size) - meta_bytes;
    } else {
      NVE_CHECK_(row_size % static_cast<size_t>(dtype_size(in_dtype)) == 0,
                 "Table row size not a multiple of the value element size");
      row_width = static_cast<int64_t>(row_size) / dtype_size(in_dtype);
    }
    // For same-type Concatenate the output holds a full raw row including quantized
    // metadata, so the minimum stride is the full row_size. For all other modes the
    // output holds only the decoded value elements.
    const int64_t min_output_stride = (concat && out_dtype == in_dtype)
                                          ? static_cast<int64_t>(row_size)
                                          : row_width * dtype_size(out_dtype);
    NVE_CHECK_(output_stride >= min_output_stride,
               "Output stride is too small for pooling output type: got ", output_stride,
               " bytes, need at least ", min_output_stride, " bytes for ", row_width,
               " elements of ", out_dtype);

    if (concat) {
      // One output row per key: the Concatenate path in cpu_kernel_pooling_dispatch
      // handles this directly (fixed hotness=1, total_rows == num_keys).
      effective_sparse = SparseType_t::Fixed;
      fixed_hotness = 1;
      num_bags = num_keys;
    } else {
      NVE_CHECK_(pool_params->key_indices != nullptr, "Invalid pooling key_indices");
      NVE_CHECK_(pool_params->num_key_indices > 0, "Invalid pooling num_key_indices");
      NVE_CHECK_(pool_params->sparse_type == SparseType_t::Fixed ||
                     pool_params->sparse_type == SparseType_t::CSR,
                 "HostEmbeddingLayer pooling supports only Fixed and CSR sparse types");
      effective_sparse = pool_params->sparse_type;

      // key_indices is host-resident for the host layer; wrap so device pointers are
      // handled too, then read the offsets/hotness on the host.
      const size_t offsets_buffer_size =
          static_cast<size_t>(pool_params->num_key_indices) * sizeof(KeyType);
      offsets_bw = std::make_shared<BufferWrapper<const KeyType>>(
          ctx, "offsets", static_cast<const KeyType*>(pool_params->key_indices), offsets_buffer_size);
      offsets_host = offsets_bw->access_buffer(cudaMemoryTypeUnregistered,
                                               /*copy_content=*/true, lookup_stream);
      if (pool_params->sparse_type == SparseType_t::Fixed) {
        fixed_hotness = offsets_host[0];
        NVE_CHECK_(fixed_hotness > 0, "Invalid fixed hotness");
        NVE_CHECK_(num_keys % fixed_hotness == 0,
                   "Number of keys does not divide by fixed hotness");
        num_bags = num_keys / fixed_hotness;
      } else {  // CSR
        num_bags = pool_params->num_key_indices - 1;
        NVE_CHECK_(num_bags >= 0, "Invalid CSR offsets");
      }
    }
  }

  // Concatenate with identical in/out type is a plain per-key memcpy:
  // no scratch buffer, no type conversion, and no reduction are needed. Treat it as
  // the no-pooling path by clearing pool_params — the gather writes final data
  // directly into the caller's output buffer at the correct stride.
  if (pool_params && (pool_params->pooling_type == PoolingType_t::Concatenate) &&
      (in_dtype == out_dtype)) {
    pool_params = nullptr;
  }

  const size_t output_buffer_size =
      static_cast<size_t>(num_bags) * static_cast<size_t>(output_stride);
  const size_t gather_buffer_size = pool_params ? num_keys_sz * row_size : output_buffer_size;

  // Build buffer wrappers — LinearHostTable accesses them as cudaMemoryTypeUnregistered.
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys, key_buffer_size);
  auto output_bw = std::make_shared<BufferWrapper<void>>(ctx, "output", output, output_buffer_size);

  // When pooling, the table gathers into a host scratch buffer (one row per key);
  // otherwise it gathers straight into the caller's output buffer.
  const int64_t gather_stride = pool_params ? static_cast<int64_t>(row_size) : output_stride;
  std::shared_ptr<BufferWrapper<void>> gather_bw = output_bw;
  if (pool_params) {
    void* gather_scratch = ctx->get_buffer("pool_gather", gather_buffer_size, /*host_alloc=*/true);
    gather_bw = std::make_shared<BufferWrapper<void>>(ctx, "pool_gather", gather_scratch,
                                                      gather_buffer_size);
  }

  // Hitmask handling. We need a hitmask buffer if either the caller wants one,
  // or we have a default embedding to fill for missed keys (we need to know which
  // keys missed). When the caller doesn't supply one but we still need it, allocate
  // a host-resident scratch buffer from the context. When neither applies, pass
  // null through to the table (it skips recording hits entirely).
  const bool need_hitmask = (output_hitmask != nullptr) || (config_.default_embedding.size() > 0);
  std::shared_ptr<BufferWrapper<max_bitmask_repr_t>> hitmask_bw;
  if (need_hitmask) {
    max_bitmask_repr_t* hitmask_ptr = output_hitmask;
    if (hitmask_ptr == nullptr) {
      hitmask_ptr = reinterpret_cast<max_bitmask_repr_t*>(
          ctx->get_buffer("hitmask", hitmask_buffer_size, /*host_alloc=*/true));
    }
    hitmask_bw = std::make_shared<BufferWrapper<max_bitmask_repr_t>>(ctx, "hitmask", hitmask_ptr, hitmask_buffer_size);
    auto* hitmask_host = hitmask_bw->access_buffer(cudaMemoryTypeUnregistered,
                                                   /*copy_content=*/false,
                                                   lookup_stream);
    std::memset(hitmask_host, 0, hitmask_buffer_size);
  }

  // Single host table lookup (into the gather buffer, which is the caller's output
  // when not pooling, or a host scratch buffer of raw per-key rows when pooling).
  table_->reset_lookup_counter(ctx);
  std::shared_ptr<BufferWrapper<int64_t>> value_sizes{nullptr};
  table_->find(ctx, num_keys, keys_bw, hitmask_bw, gather_stride, gather_bw, std::move(value_sizes));
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
    const int64_t num_workers = thread_pool->num_workers();
    const int64_t keys_per_task = std::max<int64_t>(1, (num_keys + num_workers - 1) / num_workers);
    const int64_t num_tasks = (num_keys + keys_per_task - 1) / keys_per_task;
    const uint8_t* default_emb = config_.default_embedding.data();
    auto* gather_bytes = static_cast<uint8_t*>(gather_buf);

    const auto fill_default_task = [=](const int64_t idx) {
      const int64_t start_key = idx * keys_per_task;
      const int64_t end_key = std::min<int64_t>(start_key + keys_per_task, num_keys);
      for (int64_t k = start_key; k < end_key; k++) {
        const size_t k_sz = static_cast<size_t>(k);
        const auto elem = hit_mask_buf[k_sz / hitmask_elem_bits];
        const auto bit = (elem >> (k_sz % hitmask_elem_bits)) & static_cast<max_bitmask_repr_t>(1);
        if (bit == 0) {
          std::memcpy(gather_bytes + k * gather_stride, default_emb, row_size);
        }
      }
    };
    thread_pool->execute_n(0, num_tasks, fill_default_task);
  }

  // Pooling/Dequant post-process: reduce the gathered raw rows into the caller's output.
  if (pool_params) {
    auto* gather_host = static_cast<int8_t*>(gather_bw->access_buffer(
        cudaMemoryTypeUnregistered, /*copy_content=*/true, lookup_stream));
    auto* out_host = static_cast<int8_t*>(output_bw->access_buffer(
        cudaMemoryTypeUnregistered, /*copy_content=*/false, lookup_stream));

    const void* weights = nullptr;
    std::shared_ptr<BufferWrapper<const void>> weights_bw;
    if (pool_params->sparse_weights != nullptr) {
      NVE_CHECK_(pool_params->weight_type == DataType_t::Float32 || pool_params->weight_type == DataType_t::Float16,
                 "Pooling weight_type must be Float32 or Float16 when sparse_weights is provided");
      const size_t weights_buffer_size =
          num_keys_sz * static_cast<size_t>(dtype_size(pool_params->weight_type));
      weights_bw = std::make_shared<BufferWrapper<const void>>(
          ctx, "weights", pool_params->sparse_weights, weights_buffer_size);
      weights = weights_bw->access_buffer(cudaMemoryTypeUnregistered, /*copy_content=*/true,
                                          lookup_stream);
    }

    auto thread_pool = ctx->get_thread_pool();
    const int64_t num_workers = thread_pool->num_workers();
    cpu_kernel_pooling_dispatch<KeyType>(thread_pool, num_bags, row_width, gather_host, gather_stride,
                                         out_host, output_stride, effective_sparse, fixed_hotness,
                                         offsets_host, pool_params->pooling_type, weights,
                                         pool_params->weight_type, in_dtype, out_dtype, num_workers);
  }

  // Copy back to the caller's output buffer if BufferWrapper picked up a
  // different host buffer (this happens when the caller's output is GPU /
  // managed memory — the table only knows how to gather into host; we then
  // push the host slot back across the lookup stream).
  //
  // Picking the copy primitive from the buffer's original residency: when
  // BufferWrapper allocated a separate host slot, the user-provided pointer
  // is not host-resident (Device / Managed), so we must go through
  // cudaMemcpyAsync. Note that the torch CUDA shim returns nullptr for the
  // legacy default stream — that's still a valid CUDA stream argument and
  // cudaMemcpyAsync handles it correctly.
  auto* final_output = output_bw->get_buffer(cudaMemoryTypeUnregistered);
  if (final_output != nullptr && final_output != output) {
    NVE_CHECK_(cudaMemcpyAsync(output, final_output, output_buffer_size,
                               cudaMemcpyDefault, lookup_stream));
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
  if (table_id != 0) {
    NVE_LOG_INFO_("HostEmbeddingLayer::insert called with invalid table_id - ignored");
    return;
  }
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
  // Single-table layer: table_id 0 (or negative == all) targets the host table.
  if (table_id > 0) {
    NVE_LOG_INFO_("HostEmbeddingLayer::update called with invalid table_id - ignored");
    return;
  }
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
  // Single-table layer: table_id 0 (or negative == all) targets the host table.
  if (table_id > 0) {
    NVE_LOG_INFO_("HostEmbeddingLayer::accumulate called with invalid table_id - ignored");
    return;
  }
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
  if (num_keys < 1) {
    return;
  }
  if (table_id != 0) {
    NVE_LOG_INFO_("HostEmbeddingLayer::erase called with invalid table_id - ignored");
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
