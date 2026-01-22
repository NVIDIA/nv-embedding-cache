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

#include <common.hpp>
#include <nve_types.hpp>
#include <cuda_support.hpp>
#include <unordered_map>
#include <memory>
#include <vector>

namespace nve {

class ResizeableBuffer;

// Internal class to share functionality across different execution contexts
// This is not part of the external API, applications should use Layer/Table::create_context() instead
class ExecutionContext {
 public:
  NVE_PREVENT_COPY_AND_MOVE_(ExecutionContext);
  virtual ~ExecutionContext();

  // Getters
  inline cudaStream_t get_lookup_stream() const { return lookup_stream_; };
  inline cudaStream_t get_modify_stream() const { return modify_stream_; };
  inline thread_pool_ptr_t get_thread_pool() const { return thread_pool_; }
  inline allocator_ptr_t get_allocator() const { return allocator_; }

  // get temporary buffer
  void* get_buffer(const std::string& name, size_t size, bool host_alloc);

  // check if a buffer is owned by the context
  bool is_owned(const void* ptr, const std::string& name, bool host_alloc);

  // get aux streams
  std::vector<cudaStream_t> get_aux_streams(const std::string& name, size_t num_streams);

  // Wait until pending work is complete
  // Note that this may include additional tasks offloaded to other threads
  virtual void wait() {
    NVE_CHECK_(cudaStreamSynchronize(lookup_stream_));
    NVE_CHECK_(cudaStreamSynchronize(modify_stream_));
    for (auto& kv : aux_streams_storage_) {
      for (auto& stream : kv.second) {
        NVE_CHECK_(cudaStreamSynchronize(stream));
      }
    }
  }

 protected:
  // using nullptr for threadpool/allocator implies use the default one.
  ExecutionContext(
    cudaStream_t lookup_stream,
    cudaStream_t modify_stream,
    thread_pool_ptr_t thread_pool,
    allocator_ptr_t allocator);

  cudaStream_t lookup_stream_;
  cudaStream_t modify_stream_;
  thread_pool_ptr_t thread_pool_;
  allocator_ptr_t allocator_;
  std::unordered_map<std::string, std::shared_ptr<ResizeableBuffer>> buffer_storage_;
  std::unordered_map<std::string, std::vector<cudaStream_t>> aux_streams_storage_;
  static std::string internal_name(const std::string& name, bool host_alloc);
};

}  // namespace nve
