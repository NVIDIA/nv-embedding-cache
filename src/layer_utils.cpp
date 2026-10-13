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

#include <execution_context.hpp>
#include <layer_utils.hpp>
#include <ecache/embed_cache.h>
#include <insert_heuristic.hpp>
#include <buffer_wrapper.hpp>
#include <table.hpp>
#include <resizeable_buffer.hpp>
#include "cpu_ops/cpu_pooling.hpp"
#include <algorithm>
#include <cstring>

namespace nve {

void ContextRegistry::add_context(const ExecutionContext* ctx) {
  std::lock_guard lock(mutex_);
  NVE_ASSERT_(ctx != nullptr);
  contexts_.insert(ctx);
  update_streams();
}

void ContextRegistry::remove_context(const ExecutionContext* ctx) {
  std::lock_guard lock(mutex_);
  NVE_ASSERT_(ctx != nullptr);
  contexts_.erase(ctx);
  update_streams();
}

bool ContextRegistry::empty() const {
  std::lock_guard lock(mutex_);
  return contexts_.empty();
}

std::shared_ptr<nve::DefaultECEvent> ContextRegistry::create_sync_event() {
  std::lock_guard lock(mutex_);
  return std::make_shared<nve::DefaultECEvent>(lookup_streams_);
}

void ContextRegistry::update_streams() {
  lookup_streams_.clear();
  for (auto& ctx : contexts_) {
    lookup_streams_.insert(ctx->get_lookup_stream());
  }
}

AutoInsertHandler::AutoInsertHandler(
  std::shared_ptr<InsertHeuristic> heuristic,
  table_ptr_t table,
  const int64_t table_id,
  allocator_ptr_t allocator,
  const int64_t min_insert_wait,
  const int64_t min_insert_size,
  const int64_t key_size,
  const int32_t layer_gpu_device,
  const void* invalid_key_bytes,
  uvm_insert_fn_t uvm_insert_fn) :
    heuristic_(std::move(heuristic)), table_(std::move(table)), table_id_(table_id), allocator_(std::move(allocator)),
    min_insert_wait_(min_insert_wait), min_insert_size_(min_insert_size), key_size_(key_size),
    layer_gpu_device_(layer_gpu_device), uvm_insert_fn_(std::move(uvm_insert_fn))
{
  // For now keeping key's duplicate on host, TODO: consider switching to GPU (depends on GPU insert)
  insert_keys_ = std::make_unique<ResizeableBuffer>(allocator_, true /*host_alloc*/);

  // If we we have a GPU table, then the lookup data is going to be combined on GPU
  insert_data_ = std::make_unique<ResizeableBuffer>(allocator_, (layer_gpu_device_ < 0) /*host_alloc*/);

  NVE_CHECK_(invalid_key_bytes != nullptr);
  invalid_key_bytes_.resize(static_cast<size_t>(key_size_));
  std::memcpy(invalid_key_bytes_.data(), invalid_key_bytes, static_cast<size_t>(key_size_));
}

AutoInsertHandler::~AutoInsertHandler() {
  // if insert is in progress, wait for completion
  insert_lock_.lock();
  insert_lock_.unlock();
}

void AutoInsertHandler::auto_insert(
  std::shared_ptr<LayerExecutionContext> layer_ctx,
  std::shared_ptr<BufferWrapper<const void>>& keys_bw,
  std::shared_ptr<BufferWrapper<void>>& output_bw,
  const float hitrate,
  const int64_t num_keys,
  const int64_t output_stride,
  std::shared_ptr<BufferWrapper<bitmask64_t>> hitmask_bw,
  const bool insert_from_uvm
) {
  if ((heuristic_ == nullptr) || !insert_lock_.try_lock()) {
    // no heuristic or another auto_insert in progress -> nothing to do
    return;
  }
  // Adopt the lock so it is released on exception before launch_insert transfers ownership to the lambda.
  std::unique_lock<TestableMutex> lock(insert_lock_, std::adopt_lock);
  NVE_CHECK_(!insert_from_uvm || uvm_insert_fn_, "insert_from_uvm requested but no uvm insert callback was provided");

  auto lookup_stream = layer_ctx->get_lookup_stream();

  if (collected_keys_ > 0) {
    // Keep collecting. A pooled lookup joining a non-pooled collection switches it to UVM mode (sticky);
    // from then on we collect keys only and the eventual insert promotes them from UVM.
    collection_is_uvm_ = collection_is_uvm_ || insert_from_uvm;
    // In UVM mode the pooling output_stride legitimately differs from the row-size stride a non-pooled
    // collection started with - the collected per-key data is discarded so the stride is irrelevant.
    NVE_CHECK_(collection_is_uvm_ || (output_stride == collected_output_stride_), "Output stride must be fixed during accumulation");

    // collect keys (and data, unless in UVM mode)
    collect_keys_and_data(keys_bw, output_bw, hitmask_bw, lookup_stream, num_keys);
  } else if (--insert_freq_cnt_ < 0  && heuristic_->insert_needed(hitrate, static_cast<size_t>(table_id_))) {
    // start collecting
    collection_is_uvm_ = insert_from_uvm;
    collected_output_stride_ = output_stride;
    collected_keys_ = 0;

    // collect keys (and data, unless in UVM mode)
    collect_keys_and_data(keys_bw, output_bw, hitmask_bw, lookup_stream, num_keys);
  }
  if (insert_freq_cnt_ < 0) {
    insert_freq_cnt_ = 0; // prevent looping from largest negative to positive then preventing inserts
  }

  // Check if insert can be launched
  if ((collected_keys_ > 0) && (collected_keys_ >= min_insert_size_)) {
    lock.release(); // transfer lock ownership to the async lambda in launch_insert
    launch_insert(std::move(layer_ctx), output_stride);
  }
  // else: lock destructs here and calls insert_lock_.unlock()
}

void AutoInsertHandler::collect_keys_and_data(
  std::shared_ptr<BufferWrapper<const void>>& keys_bw,
  std::shared_ptr<BufferWrapper<void>>& output_bw,
  std::shared_ptr<BufferWrapper<bitmask64_t>>& hitmask_bw,
  cudaStream_t lookup_stream,
  const int64_t num_keys) {

  const int64_t collection_total_size =
    (collected_keys_ == 0) ?
    std::max(num_keys, min_insert_size_) : // Starting collection - We want at least min_insert_size_ keys
    min_insert_size_; // Collection in progress
                      // We collect until min_insert_size_ and won't exceed, since that can trigger a realloc
                      // of the buffers and invalidate previous collection parts.
  const int64_t collection_part_size = std::min(num_keys, collection_total_size - collected_keys_);

  auto keys_host_buf = keys_bw->get_buffer(cudaMemoryTypeHost);
  uint8_t* dst_keys = reinterpret_cast<uint8_t*>(insert_keys_->get_ptr(static_cast<size_t>(collection_total_size * key_size_)))
                      + (collected_keys_ * key_size_);
  if (keys_host_buf) {
    std::memcpy(
      dst_keys,
      keys_host_buf,
      static_cast<size_t>(collection_part_size * key_size_));
  } else {
    NVE_CHECK_(cudaMemcpyAsync(
      dst_keys,
      keys_bw->get_buffer(keys_bw->get_last_access()),
      static_cast<size_t>(collection_part_size * key_size_),
      cudaMemcpyDefault,
      lookup_stream));
  }

  // Make the hitmask host-visible so we can rewrite missed keys to the sentinel after the sync below.
  // access_buffer queues a D2H copy on lookup_stream when the last access was on device, so it's covered
  // by the existing stream sync.
  bitmask64_t* h_hitmask = nullptr;
  if (hitmask_bw) {
    h_hitmask = hitmask_bw->access_buffer(cudaMemoryTypeHost, true /*copy_content*/, lookup_stream);
  }

  // copy partial data - skipped in UVM mode, where the insert reads each key's row directly from the
  // backing UVM table (no per-key row data is collected here).
  if (!collection_is_uvm_) {
    NVE_CHECK_(cudaMemcpyAsync(
      reinterpret_cast<uint8_t*>(insert_data_->get_ptr(static_cast<size_t>(collection_total_size * collected_output_stride_))) + (collected_keys_ * collected_output_stride_),
      output_bw->get_buffer(output_bw->get_last_access()),
      static_cast<size_t>(collection_part_size * collected_output_stride_),
      cudaMemcpyDefault,
      lookup_stream));
  }

  // synchronize since other threads can copy on other lookup streams, and when we launch we won't know who to wait on
  // also input key/data buffers can change once we return
  NVE_CHECK_(cudaStreamSynchronize(lookup_stream));

  // Rewrite missed keys (hitmask bit == 0) to the configured sentinel so the destination table is not
  // contaminated with garbage values from unresolved slots. Per-call hitmask indices [0, collection_part_size)
  // map to insert_keys_ slots [collected_keys_, collected_keys_ + collection_part_size).
  if (h_hitmask) {
    for (int64_t j = 0; j < collection_part_size; ++j) {
      const bitmask64_t word{h_hitmask[j / bitmask64::num_bits]};
      const bool bit{bitmask64::get(word, j % bitmask64::num_bits)};
      if (!bit) {
        std::memcpy(dst_keys + j * key_size_, invalid_key_bytes_.data(), static_cast<size_t>(key_size_));
      }
    }
  }

  collected_keys_ += collection_part_size;
}

void AutoInsertHandler::lock_modify() {
  if (heuristic_) {
    insert_lock_.lock();
  }
}

void AutoInsertHandler::unlock_modify() {
  if (heuristic_) {
    // When we lock for modify op, we invalidate any pending insert accumulation (collected keys/data are no longer valid)
    collected_keys_ = 0;
    collection_is_uvm_ = false;
    insert_lock_.unlock();
  }
}

void AutoInsertHandler::launch_insert(
  std::shared_ptr<LayerExecutionContext> layer_ctx,
  const int64_t output_stride) {
  NVE_LOG_PERF_("Auto-insert launched for table ", table_id_);
  NVE_NVTX_MARK_("Auto-insert launched");
  auto ctx = std::dynamic_pointer_cast<ExecutionContext>(layer_ctx);
  // insert_lock_ is held until the task completes, so collection_is_uvm_ is stable - capture the decision.
  const bool from_uvm = collection_is_uvm_;
  const std::string buffer_suffix = "_" + std::to_string(table_id_);
  auto insert_keys_bw = std::make_shared<BufferWrapper<const void>>(
    ctx, "insert_keys" + buffer_suffix, insert_keys_->get_ptr(0), insert_keys_->get_size());
  // In UVM mode no per-key data was collected; the insert reads rows from the backing UVM table.
  std::shared_ptr<BufferWrapper<const void>> insert_output_bw;
  if (!from_uvm) {
    insert_output_bw = std::make_shared<BufferWrapper<const void>>(
      ctx, "insert_data" + buffer_suffix, insert_data_->get_ptr(0), insert_data_->get_size());
  }

  // Now schedule the insert on the threapool
  layer_ctx->submit_parallel_task(
    [=] () mutable {
      auto cleanup = [&]() noexcept {
        insert_freq_cnt_ = min_insert_wait_;
        collected_keys_ = 0;
        collection_is_uvm_ = false;
        layer_ctx.reset();
        insert_lock_.unlock();
      };
      try {
        auto table_ctx = layer_ctx->table_contexts_.at(static_cast<size_t>(table_id_));
        NVE_CHECK_(table_ctx != nullptr, "Invalid table context");
        // Coverity complains about insert_keys_bw copied to the lambda function [=], leaving as intentional
        // coverity[COPY_INSTEAD_OF_MOVE]
        if (from_uvm) {
          uvm_insert_fn_(table_ctx, collected_keys_, std::move(insert_keys_bw));
        } else {
          table_->insert(table_ctx, collected_keys_, std::move(insert_keys_bw), output_stride, table_->get_max_row_size(), std::move(insert_output_bw));
        }
      } catch (...) {
        cleanup();
        throw;
      }
      cleanup();
    }, table_id_);
}

void validate_pool_params(const EmbeddingLayerBase::PoolingParams& pool_params) {
  NVE_CHECK_ARG_(pool_params.output_type != DataType_t::Unknown, "Unknown output type");

  // Concatenate is a per-key copy/dequant operation. It has no bags or weights, so every field
  // other than pooling_type and output_type is intentionally ignored.
  if (pool_params.pooling_type == PoolingType_t::Concatenate) {
    return;
  }

  if (is_weighted_pooling(pool_params.pooling_type)) {
    if (pool_params.weights == nullptr) {
      NVE_THROW_ARG_("Sparse weights are required for weighted pooling");
    }
  }


  switch (pool_params.sparse_type) {
    case SparseType_t::Fixed:
      NVE_CHECK_ARG_(pool_params.fixed_hotness > 0, "Invalid fixed hotness");
      break;
    case SparseType_t::CSR:
      NVE_CHECK_ARG_(pool_params.csr_offsets != nullptr, "CSR offsets are required for pooling");
      NVE_CHECK_ARG_(pool_params.num_csr_offsets >= 2,
                     "CSR pooling requires at least two offsets");
      break;
    default:
      NVE_THROW_ARG_("Unsupported sparse type ", static_cast<uint32_t>(pool_params.sparse_type));
  }
}

int64_t get_lookup_output_rows(
    const int64_t num_keys, const EmbeddingLayerBase::PoolingParams* pool_params) {
  NVE_CHECK_ARG_(num_keys >= 0, "Invalid number of keys");
  if (pool_params == nullptr || pool_params->pooling_type == PoolingType_t::Concatenate) {
    return num_keys;
  }

  switch (pool_params->sparse_type) {
    case SparseType_t::Fixed:
      NVE_CHECK_ARG_(pool_params->fixed_hotness > 0, "Invalid fixed hotness");
      NVE_CHECK_ARG_(num_keys % pool_params->fixed_hotness == 0,
                     "Number of keys does not divide by fixed hotness");
      return num_keys / pool_params->fixed_hotness;
    case SparseType_t::CSR:
      NVE_CHECK_ARG_(pool_params->csr_offsets != nullptr, "CSR offsets are required for pooling");
      NVE_CHECK_ARG_(pool_params->num_csr_offsets >= 2,
                     "CSR pooling requires at least two offsets");
      return pool_params->num_csr_offsets - 1;
    default:
      NVE_THROW_ARG_("Unsupported sparse type ", static_cast<uint32_t>(pool_params->sparse_type));
  }
}

template <typename KeyType>
void pool_gathered_host(context_ptr_t& ctx,
                        const EmbeddingLayerBase::PoolingParams& pool_params,
                        DataType_t value_dtype,
                        const int8_t* gather_host, int64_t gather_stride,
                        int64_t row_size, int64_t num_keys,
                        void* output, int64_t output_stride, int64_t max_tasks) {
  const cudaStream_t lookup_stream = ctx->get_lookup_stream();
  NVE_CHECK_ARG_(max_tasks > 0, "Invalid number of tasks");

  // Concatenate means no arithmetic reduction: each key yields one output row, only
  // converting/dequantizing the raw gathered data to the output type. The sparse layout and
  // weights are ignored (per the PoolingParams contract).
  const bool concat = (pool_params.pooling_type == PoolingType_t::Concatenate);

  const DataType_t in_dtype = value_dtype;
  const bool quant_in = is_quant_rowwise(in_dtype);
  // For quantized input the result is dequantized to a float type unless the caller requests
  // same-type Concatenate (passthrough).
  const DataType_t out_dtype = pool_params.output_type;
  NVE_CHECK_ARG_(out_dtype != DataType_t::Unknown, "Unknown output type");
  NVE_CHECK_(in_dtype == DataType_t::Float32 || in_dtype == DataType_t::Float16 || quant_in,
             "pool_gathered_host supports only fp32/fp16 or QInt8/QUint8Rowwise table values, got: ",
             in_dtype);
  // Float32/Float16 output is required for all reduction modes. For Concatenate, a same-type
  // output is also accepted (no dequantization, full row is passed through).
  NVE_CHECK_ARG_(out_dtype == DataType_t::Float32 || out_dtype == DataType_t::Float16 ||
                     (concat && out_dtype == in_dtype),
                 "pool_gathered_host supports fp32/fp16 output; same-type output is only accepted "
                 "for Concatenate mode");

  int64_t row_width = 0;
  if (quant_in) {
    // row_size is the full quantized row stride: value bytes + trailing scale[/offset].
    const int64_t meta_bytes = quant_rowwise_meta_bytes(in_dtype);
    NVE_CHECK_(row_size > meta_bytes, "Quantized row size smaller than its scale/offset metadata");
    row_width = row_size - meta_bytes;
  } else {
    NVE_CHECK_(row_size % static_cast<int64_t>(dtype_size(in_dtype)) == 0,
               "Table row size not a multiple of the value element size");
    row_width = row_size / dtype_size(in_dtype);
  }
  // For same-type Concatenate the output holds a full raw row including quantized metadata, so the
  // minimum stride is the full row_size. For all other modes the output holds only the decoded
  // value elements.
  const int64_t min_output_stride =
      (concat && out_dtype == in_dtype) ? row_size : row_width * dtype_size(out_dtype);
  NVE_CHECK_ARG_(output_stride >= min_output_stride,
                 "Output stride is too small for pooling output type: got ", output_stride,
                 " bytes, need at least ", min_output_stride, " bytes for ", row_width,
                 " elements of ", out_dtype);

  // Identical input/output Concatenate is a raw gather: preserve the complete stored row,
  // including scale/offset metadata for rowwise-quantized values. The typed CPU concatenate
  // kernels only produce Float32/Float16, so this also handles quantized passthrough without
  // routing a quantized output dtype through their conversion dispatch.
  if (concat && out_dtype == in_dtype) {
    const size_t output_buffer_size =
        static_cast<size_t>(num_keys) * static_cast<size_t>(output_stride);
    auto output_bw =
        std::make_shared<BufferWrapper<void>>(ctx, "pool_output", output, output_buffer_size);
    auto* output_host = static_cast<int8_t*>(output_bw->access_buffer(
        cudaMemoryTypeUnregistered, /*copy_content=*/false, lookup_stream));

    auto thread_pool = ctx->get_thread_pool();
    const int64_t rows_per_task = std::max<int64_t>(1, ceil_div(num_keys, max_tasks));
    const int64_t num_tasks = ceil_div(num_keys, rows_per_task);
    const auto copy_rows = [=](const int64_t task_idx) {
      const int64_t row_start = task_idx * rows_per_task;
      const int64_t row_end = std::min<int64_t>(row_start + rows_per_task, num_keys);
      for (int64_t row = row_start; row < row_end; ++row) {
        std::memcpy(output_host + row * output_stride, gather_host + row * gather_stride,
                    static_cast<size_t>(row_size));
      }
    };
    thread_pool->execute_n(0, num_tasks, copy_rows);

    auto* final_output = output_bw->get_buffer(cudaMemoryTypeUnregistered);
    if (final_output != nullptr && final_output != output) {
      NVE_CHECK_(cudaMemcpyAsync(output, final_output, output_buffer_size, cudaMemcpyDefault,
                                 lookup_stream));
    }
    return;
  }

  // Conversion/dequantization Concatenate also has one independent output row per key. Dispatch
  // directly to the concatenate kernel so none of the bag-layout or weight fields are inspected.
  if (concat) {
    const size_t output_buffer_size =
        static_cast<size_t>(num_keys) * static_cast<size_t>(output_stride);
    auto output_bw =
        std::make_shared<BufferWrapper<void>>(ctx, "pool_output", output, output_buffer_size);
    auto* output_host = static_cast<int8_t*>(output_bw->access_buffer(
        cudaMemoryTypeUnregistered, /*copy_content=*/false, lookup_stream));

    cpu_kernel_concatenate_dispatch(ctx->get_thread_pool(), num_keys, row_width, gather_host,
                                    gather_stride, output_host, output_stride, in_dtype, out_dtype,
                                    max_tasks);

    auto* final_output = output_bw->get_buffer(cudaMemoryTypeUnregistered);
    if (final_output != nullptr && final_output != output) {
      NVE_CHECK_(cudaMemcpyAsync(output, final_output, output_buffer_size, cudaMemcpyDefault,
                                 lookup_stream));
    }
    return;
  }

  SparseType_t effective_sparse = pool_params.sparse_type;
  int64_t fixed_hotness = pool_params.fixed_hotness;
  int64_t num_bags = 0;
  std::shared_ptr<BufferWrapper<const KeyType>> offsets_bw;
  const KeyType* offsets_host = nullptr;
  NVE_CHECK_ARG_(pool_params.sparse_type == SparseType_t::Fixed ||
                     pool_params.sparse_type == SparseType_t::CSR,
                 "pool_gathered_host supports only Fixed and CSR sparse types");

  if (pool_params.sparse_type == SparseType_t::Fixed) {
    NVE_CHECK_ARG_(fixed_hotness > 0, "Invalid fixed hotness");
    NVE_CHECK_ARG_(num_keys % fixed_hotness == 0,
                   "Number of keys does not divide by fixed hotness");
    num_bags = num_keys / fixed_hotness;
  } else {  // CSR
    // CSR offsets may be host- or device-resident; wrap so device pointers are handled too.
    const size_t offsets_buffer_size =
        static_cast<size_t>(pool_params.num_csr_offsets) * sizeof(KeyType);
    offsets_bw = std::make_shared<BufferWrapper<const KeyType>>(
        ctx, "offsets", static_cast<const KeyType*>(pool_params.csr_offsets),
        offsets_buffer_size);
    offsets_host = offsets_bw->access_buffer(cudaMemoryTypeUnregistered,
                                             /*copy_content=*/true, lookup_stream);
    num_bags = pool_params.num_csr_offsets - 1;
    NVE_CHECK_(num_bags >= 0, "Invalid CSR offsets");
    // Enforce the CSR contract before the pooling kernel indexes the gathered rows:
    // offsets must start at 0, be non-decreasing, and end with a sentinel equal to
    // num_keys (torch's include_last_offset=True convention). A bad span would
    // otherwise read outside the num_keys-row gather buffer.
    NVE_CHECK_ARG_(offsets_host[0] == 0, "CSR offsets must start at 0, got ", offsets_host[0]);
    for (int64_t bag = 0; bag < num_bags; bag++) {
      NVE_CHECK_ARG_(offsets_host[bag] <= offsets_host[bag + 1],
                     "CSR offsets must be non-decreasing: offsets[", bag + 1, "] = ",
                     offsets_host[bag + 1], " < offsets[", bag, "] = ", offsets_host[bag]);
    }
    NVE_CHECK_ARG_(static_cast<int64_t>(offsets_host[num_bags]) == num_keys,
                   "CSR offsets must end with a trailing sentinel equal to the number of keys "
                   "(include_last_offset=True convention): got ", offsets_host[num_bags],
                   ", expected ", num_keys);
  }

  const void* weights = nullptr;
  std::shared_ptr<BufferWrapper<const void>> weights_bw;
  if (pool_params.weights != nullptr) {
    NVE_CHECK_ARG_(pool_params.weight_type == DataType_t::Float32 ||
                       pool_params.weight_type == DataType_t::Float16,
                   "Pooling weight_type must be Float32 or Float16 when weights is provided");
    const size_t weights_buffer_size =
        static_cast<size_t>(num_keys) * static_cast<size_t>(dtype_size(pool_params.weight_type));
    weights_bw = std::make_shared<BufferWrapper<const void>>(
        ctx, "weights", pool_params.weights, weights_buffer_size);
    weights = weights_bw->access_buffer(cudaMemoryTypeUnregistered, /*copy_content=*/true,
                                        lookup_stream);
  }

  // Wrap the user output so a device/managed buffer is written via a host scratch slot; the CPU
  // pooling kernels require host-resident output. Copy back below if BufferWrapper relocated it.
  const size_t output_buffer_size =
      static_cast<size_t>(num_bags) * static_cast<size_t>(output_stride);
  auto output_bw = std::make_shared<BufferWrapper<void>>(ctx, "pool_output", output, output_buffer_size);
  auto* output_host = static_cast<int8_t*>(
      output_bw->access_buffer(cudaMemoryTypeUnregistered, /*copy_content=*/false, lookup_stream));

  cpu_kernel_pooling_dispatch<KeyType>(ctx->get_thread_pool(), num_bags, row_width, gather_host, gather_stride,
                                       output_host, output_stride, effective_sparse, fixed_hotness,
                                       offsets_host, pool_params.pooling_type, weights,
                                       pool_params.weight_type, in_dtype, out_dtype, max_tasks);

  // Push the host result back to the user buffer if it lives on device/managed memory.
  auto* final_output = output_bw->get_buffer(cudaMemoryTypeUnregistered);
  if (final_output != nullptr && final_output != output) {
    NVE_CHECK_(cudaMemcpyAsync(output, final_output, output_buffer_size, cudaMemcpyDefault,
                               lookup_stream));
  }
}

template void pool_gathered_host<int32_t>(context_ptr_t&,
                                          const EmbeddingLayerBase::PoolingParams&, DataType_t,
                                          const int8_t*, int64_t, int64_t, int64_t, void*,
                                          int64_t, int64_t);
template void pool_gathered_host<int64_t>(context_ptr_t&,
                                          const EmbeddingLayerBase::PoolingParams&, DataType_t,
                                          const int8_t*, int64_t, int64_t, int64_t, void*,
                                          int64_t, int64_t);

bool is_pooling_raw_concat(const EmbeddingLayerBase::PoolingParams* pool_params, DataType_t value_dtype) {
  return (pool_params &&
          pool_params->pooling_type == PoolingType_t::Concatenate &&
          pool_params->output_type == value_dtype);
}

}  // namespace nve
