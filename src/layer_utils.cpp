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
  const int32_t layer_gpu_device) : 
    heuristic_(heuristic), table_(table), table_id_(table_id), allocator_(allocator),
    min_insert_wait_(min_insert_wait), min_insert_size_(min_insert_size), key_size_(key_size),
    layer_gpu_device_(layer_gpu_device)
{
  // For now keeping key's duplicate on host, TODO: consider switching to GPU (depends on GPU insert)
  insert_keys_ = std::make_unique<ResizeableBuffer>(allocator_, true /*host_alloc*/);
  
  // If we we have a GPU table, then the lookup data is going to be combined on GPU
  insert_data_ = std::make_unique<ResizeableBuffer>(allocator_, (layer_gpu_device_ < 0) /*host_alloc*/);
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
  const int64_t output_stride
) {
  if ((heuristic_ == nullptr) || !insert_lock_.try_lock()) {
    // no heuristic or another auto_insert in progress -> nothing to do
    return;
  }

  auto lookup_stream = layer_ctx->get_lookup_stream();

  if (collected_keys_ > 0) {
    // Keep collecting
    NVE_CHECK_(output_stride == collected_output_stride_, "Output stride must be fixed during accumulation");

    // collect keys and data
    collect_keys_and_data(keys_bw, output_bw, lookup_stream, num_keys);
  } else if (--insert_freq_cnt_ < 0  && heuristic_->InsertNeeded(hitrate, static_cast<size_t>(table_id_))) {
    // start collecting
    collected_output_stride_ = output_stride;
    collected_keys_ = 0;

    // collect keys and data
    collect_keys_and_data(keys_bw, output_bw, lookup_stream, num_keys);
  }
  if (insert_freq_cnt_ < 0) {
    insert_freq_cnt_ = 0; // prevent looping from largest negative to positive then preventing inserts
  }

  // Check if insert can be launched
  if ((collected_keys_ > 0) && (collected_keys_ >= min_insert_size_)) {
    launch_insert(layer_ctx, output_stride);
  } else {
    insert_lock_.unlock(); // if we launched an insert, it will release the lock.
  }
}

void AutoInsertHandler::collect_keys_and_data(
  std::shared_ptr<BufferWrapper<const void>>& keys_bw,
  std::shared_ptr<BufferWrapper<void>>& output_bw,
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
  if (keys_host_buf) {
    std::memcpy(
      reinterpret_cast<uint8_t*>(insert_keys_->get_ptr(static_cast<size_t>(collection_total_size * key_size_))) + (collected_keys_ * key_size_),
      keys_host_buf,
      static_cast<size_t>(collection_part_size * key_size_));
  } else {
    NVE_CHECK_(cudaMemcpyAsync(
      reinterpret_cast<uint8_t*>(insert_keys_->get_ptr(static_cast<size_t>(collection_total_size * key_size_))) + (collected_keys_ * key_size_),
      keys_bw->get_buffer(keys_bw->get_last_access()),
      static_cast<size_t>(collection_part_size * key_size_),
      cudaMemcpyDefault,
      lookup_stream));
  }

  // copy partial data
  NVE_CHECK_(cudaMemcpyAsync(
    reinterpret_cast<uint8_t*>(insert_data_->get_ptr(static_cast<size_t>(collection_total_size * collected_output_stride_))) + (collected_keys_ * collected_output_stride_),
    output_bw->get_buffer(keys_bw->get_last_access()),
    static_cast<size_t>(collection_part_size * collected_output_stride_),
    cudaMemcpyDefault,
    lookup_stream));
  
  // synchronize since other threads can copy on other lookup streams, and when we launch we won't know who to wait on
  // also input key/data buffers can change once we return
  NVE_CHECK_(cudaStreamSynchronize(lookup_stream));
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
    insert_lock_.unlock();
  }
}

void AutoInsertHandler::launch_insert(
  std::shared_ptr<LayerExecutionContext> layer_ctx,
  const int64_t output_stride) {
  NVE_NVTX_MARK_("Auto-insert launched");
  auto ctx = std::dynamic_pointer_cast<ExecutionContext>(layer_ctx);
  auto insert_keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "insert_keys", insert_keys_->get_ptr(0), insert_keys_->get_size());
  auto insert_output_bw = std::make_shared<BufferWrapper<const void>>(ctx, "insert_data", insert_data_->get_ptr(0), insert_data_->get_size());

  // Now schedule the insert on the threapool
  layer_ctx->submit_parallel_task(
    [=] () {
      auto table_ctx = layer_ctx->table_contexts_.at(static_cast<size_t>(table_id_));
      NVE_CHECK_(table_ctx != nullptr, "Invalid table context");
      table_->insert_bw(table_ctx, collected_keys_, insert_keys_bw, output_stride, table_->get_max_row_size(), insert_output_bw);
      insert_freq_cnt_ = min_insert_wait_; // reset freq counters
      collected_keys_ = 0; // clear collected keys
      insert_lock_.unlock();
    }, table_id_);
}

}  // namespace nve
