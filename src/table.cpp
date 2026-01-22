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
#include <json_support.hpp>
#include <table.hpp>
#include <unordered_map>
#include <buffer_wrapper.hpp>
namespace nve {

class TableExecutionContext: public ExecutionContext {
 public:
  using base_type = ExecutionContext;

  NVE_PREVENT_COPY_AND_MOVE_(TableExecutionContext);

  TableExecutionContext() = delete;

  TableExecutionContext(
    cudaStream_t lookup_stream,
    cudaStream_t modify_stream,
    thread_pool_ptr_t thread_pool,
    allocator_ptr_t allocator)
    : base_type(lookup_stream, modify_stream, thread_pool, allocator) {}

  ~TableExecutionContext() override = default;
};


NVE_DEFINE_JSON_ENUM_CONVERSION_(DataType_t, DataType_t::Unknown, DataType_t::Float32,
                                 DataType_t::Float16, DataType_t::BFloat, DataType_t::E4M3,
                                 DataType_t::E5M2, DataType_t::Float64)

Table::Table() { NVE_LOG_VERBOSE_("Constructing table #", this); }

Table::~Table() { NVE_LOG_VERBOSE_("Destructing table #", this); }

context_ptr_t Table::create_execution_context(
  cudaStream_t lookup_stream,
  cudaStream_t modify_stream,
  thread_pool_ptr_t thread_pool,
  allocator_ptr_t allocator) {
  return std::make_shared<TableExecutionContext>(lookup_stream, modify_stream, thread_pool, allocator);
}


// Default fallback from BufferWrapper to raw pointers, always using host buffers for funcitonality (perf out the window)
void Table::erase_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys) {
  auto keys_buf = keys->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, ctx->get_modify_stream());
  erase(ctx, n, keys_buf);
}

void Table::find_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, buffer_ptr<max_bitmask_repr_t> hit_mask,
                      int64_t value_stride, buffer_ptr<void> values, buffer_ptr<int64_t> /*value_sizes*/) const {
  auto keys_buf = keys->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, ctx->get_lookup_stream());
  auto hit_mask_buf = hit_mask->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, ctx->get_lookup_stream());
  auto values_buf = values->access_buffer(cudaMemoryTypeUnregistered, false /*copy_content*/, ctx->get_lookup_stream());
  find(ctx, n, keys_buf, hit_mask_buf, value_stride, values_buf, nullptr);
}

void Table::insert_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, int64_t value_stride,
                    int64_t value_size, buffer_ptr<const void> values) {
  auto keys_buf = keys->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, ctx->get_modify_stream());
  auto values_buf = values->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, ctx->get_modify_stream());
  insert(ctx, n, keys_buf, value_stride, value_size, values_buf);
}

void Table::update_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, int64_t value_stride,
                        int64_t value_size, buffer_ptr<const void> values) {
  auto keys_buf = keys->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, ctx->get_modify_stream());
  auto values_buf = values->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, ctx->get_modify_stream());
  update(ctx, n, keys_buf, value_stride, value_size, values_buf);
}

void Table::update_accumulate_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys,
                                int64_t update_stride, int64_t update_size, buffer_ptr<const void> updates,
                                DataType_t update_dtype) {
  auto keys_buf = keys->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, ctx->get_modify_stream());
  auto updates_buf = updates->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, ctx->get_modify_stream());
  update_accumulate(ctx, n, keys_buf, update_stride, update_size, updates_buf, update_dtype);
}

}  // namespace nve
