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
#include <cuda_support.hpp>
#include <memory>
#include <string>
#include <cstring>
#include <execution_context.hpp>
#include <unordered_map>

namespace nve {

template<typename T>
class BufferWrapper {
 public:
  NVE_PREVENT_COPY_AND_MOVE_(BufferWrapper);

  BufferWrapper(context_ptr_t& ctx, const std::string name, T* buffer, const size_t size_in_bytes)
    : ctx_(ctx), name_(std::move(name)), size_(size_in_bytes) {
    NVE_CHECK_(ctx_ != nullptr, "Invalid context");
    NVE_CHECK_(buffer != nullptr, "Invalid buffer");
    last_access_ = BufferType(buffer);
    buffers_[last_access_] = buffer;
  }

  ~BufferWrapper() = default; // No buffer deallocation here, allocated buffers are owned by the context

  template<typename CLONE_T = T>
  std::shared_ptr<BufferWrapper<CLONE_T>> clone(const std::string name, cudaStream_t stream) {
    NVE_CHECK_(name != name_, "Clone buffer wrapper name must be different than source");
    auto last_buffer = buffers_.at(last_access_);
    const bool clone_host_buffer = (last_access_ != cudaMemoryTypeDevice);
    auto clone_buffer = ctx_->get_buffer(name_, size_, clone_host_buffer);
    if (clone_host_buffer) {
        std::memcpy(clone_buffer, last_buffer, size_);
    } else {
        NVE_CHECK_(cudaMemcpyAsync(clone_buffer, last_buffer, size_, cudaMemcpyDefault, stream));
    }
    return std::make_shared<BufferWrapper<CLONE_T>>(ctx_, name, clone_buffer, size_);
  }

  T* access_buffer(cudaMemoryType mem_type, bool copy_content, cudaStream_t stream) {
    // if buffer doesn't exist - allocate
    bool new_buffer{false};
    if (buffers_.count(mem_type) == 0) {
        NVE_CHECK_(!std::is_const_v<T> || copy_content, "Const wrapper must access with copy_content, otherwise data will never initialize");
        switch (mem_type) {
            case cudaMemoryTypeUnregistered: {
                // alloc host and reuse for unregistered
                buffers_[cudaMemoryTypeUnregistered] = access_buffer(cudaMemoryTypeHost, copy_content, stream);
            }
            break;
            case cudaMemoryTypeHost: {
                buffers_[cudaMemoryTypeHost] = reinterpret_cast<T*>(ctx_->get_buffer(name_, size_, true /*host_alloc*/));
                NVE_CHECK_(buffers_.at(cudaMemoryTypeHost) != nullptr, "Failed to allocated temporary buffer ", name_);
            }
            break;
            case cudaMemoryTypeDevice: {
                buffers_[cudaMemoryTypeDevice] = reinterpret_cast<T*>(ctx_->get_buffer(name_, size_, false /*host_alloc*/));
                NVE_CHECK_(buffers_.at(cudaMemoryTypeDevice) != nullptr, "Failed to allocated temporary buffer ", name_);
            }
            break;
            case cudaMemoryTypeManaged:
                NVE_THROW_("Buffer wrapper cannot allocate managed buffers (access_buffer() only valid for wrappers initialiazed with managed)");
            default:
                NVE_THROW_("Invalid cudaMemoryType");
        }
        new_buffer = true;
    }

    // when accessing const buffer the copy should only happen with new allocations (the data is constant, so later accesses don't need to copy)
    if (copy_content && (!std::is_const_v<T> || new_buffer)) {
        auto src = buffers_.at(last_access_);
        void* dst = const_cast<void*>(reinterpret_cast<const void*>(buffers_.at(mem_type))); // reinterpret cast then strip const for the copies handling const buffers
        if (src != dst) {
            NVE_CHECK_(cudaMemcpyAsync(dst, src, size_, cudaMemcpyDefault, stream));
            if (mem_type != cudaMemoryTypeDevice) {
                NVE_CHECK_(cudaStreamSynchronize(stream)); // Synchronizing since next access can be on host.
            }
        }
    }

    last_access_ = mem_type;
    return buffers_.at(mem_type);
  }

  cudaMemoryType get_last_access() const {
    return last_access_;
  }

  /**
   * get_buffer will only return a pointer to the raw buffer if one exists (nullptr otherwise)
   * There will be no allocations or copies made, nor will the last_access be changed.
   */
  T* get_buffer(cudaMemoryType mem_type) const {
    auto it = buffers_.find(mem_type);
    if (it == buffers_.end()) {
      return nullptr;
    } else {
      return it->second;
    }
  }

 protected:
  context_ptr_t ctx_;
  const std::string name_;
  const size_t size_{0};
  cudaMemoryType last_access_{cudaMemoryTypeUnregistered};
  std::unordered_map<cudaMemoryType, T*> buffers_;

  cudaMemoryType BufferType(const void* ptr) {
    cudaPointerAttributes attr;
    auto err = cudaPointerGetAttributes(&attr, ptr);
    return (err == cudaSuccess) ? attr.type : cudaMemoryTypeUnregistered;
  }
};

}  // namespace nve
