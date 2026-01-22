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
#include <common.hpp>
#include <default_allocator.hpp>
#include <thread_pool.hpp>
#include <resizeable_buffer.hpp>

namespace nve {

// Get a named scratch buffer
// If a buffer already exists with the same name it will be reused if the size is large enough (otherwise will be reallocated)
// Calling this function potentially invalidates previously returned buffers for the same name.
void* ExecutionContext::get_buffer(const std::string& name, size_t size, bool host_alloc) {
    const std::string key = internal_name(name, host_alloc);
    auto kv = buffer_storage_.find(key);
    if (kv == buffer_storage_.end()) {
        auto res = buffer_storage_.emplace(
            std::make_pair(key, std::make_unique<ResizeableBuffer>(allocator_, host_alloc)));
        NVE_ASSERT_(res.second);
        kv = res.first;
    }
    auto buffer = kv->second;
    return buffer->get_ptr(size);
}

std::vector<cudaStream_t> ExecutionContext::get_aux_streams(const std::string& name, size_t num_streams) {
    auto kv = aux_streams_storage_.find(name);
    if (kv == aux_streams_storage_.end()) {
        std::vector<cudaStream_t> streams(num_streams);
        for (size_t i = 0; i < num_streams; i++) {
            NVE_CHECK_(cudaStreamCreate(&streams[i]));
        }
        auto res = aux_streams_storage_.emplace(std::make_pair(name, streams));
        NVE_ASSERT_(res.second);
        kv = res.first;
    } else if (kv->second.size() < num_streams) {
        for (size_t i = kv->second.size(); i < num_streams; i++) {
            cudaStream_t stream;
            NVE_CHECK_(cudaStreamCreate(&stream));
            kv->second.push_back(stream);
        }
    }
    return kv->second;
}

bool ExecutionContext::is_owned(const void* ptr, const std::string& name, bool host_alloc) {
    const std::string key = internal_name(name, host_alloc);
    auto kv = buffer_storage_.find(key);
    return (kv != buffer_storage_.end()) && (kv->second->get_ptr(0) == ptr);
}

std::string ExecutionContext::internal_name(const std::string& name, bool host_alloc) {
  return (host_alloc ? std::string("H__") : std::string("D__")) + name;
}

ExecutionContext::ExecutionContext(
    cudaStream_t lookup_stream,
    cudaStream_t modify_stream,
    thread_pool_ptr_t thread_pool,
    allocator_ptr_t allocator) : 
    lookup_stream_(lookup_stream), modify_stream_(modify_stream), thread_pool_(thread_pool)
{
  if (!thread_pool_) {
    thread_pool_ = default_thread_pool();
    NVE_CHECK_(thread_pool_ != nullptr);
  }
  allocator_ = allocator ? allocator : GetDefaultAllocator();
  NVE_CHECK_(allocator_ != nullptr, "Failed to get default allocator");
}

ExecutionContext::~ExecutionContext() {
  wait();
  for (auto& kv : aux_streams_storage_) {
    for (auto& stream : kv.second) {
      NVE_CHECK_(cudaStreamDestroy(stream));
    }
  }
}

}  // namespace nve
