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

#include "sph_check.h"
#include <cuda_runtime.h>
#include <cstddef>

// Standalone replacement for the parent repo's nve::sph_support::ResizeableBuffer.
// Drops the allocator_ptr_t indirection — the only two backends we care
// about are plain `cudaMalloc` (device) and `cudaMallocHost` (pinned host),
// so we call them directly.
//
// Behaviour preserved: get_ptr(N) returns a device-or-host buffer of at
// least N bytes; if the caller asks for more than the current capacity,
// the old buffer is freed and a new one allocated (old data is lost).
namespace nve {
namespace sph_support {

class ResizeableBuffer {
 public:
  explicit ResizeableBuffer(bool host_alloc)
      : host_alloc_(host_alloc), buffer_(nullptr), size_(0) {}

  ~ResizeableBuffer() {
    if (buffer_) {
      if (host_alloc_) {
        SPH_TEARDOWN(cudaFreeHost(buffer_));
      } else {
        SPH_TEARDOWN(cudaFree(buffer_));
      }
    }
    size_ = 0;
  }

  ResizeableBuffer(const ResizeableBuffer&)            = delete;
  ResizeableBuffer& operator=(const ResizeableBuffer&) = delete;

  // If requested size is larger than existing size, a new buffer is allocated
  // (old data is lost).
  void* get_ptr(std::size_t buffer_size) {
    if (buffer_size > size_) {
      if (buffer_) {
        if (host_alloc_) {
          SPH_CHECK(cudaFreeHost(buffer_));
        } else {
          SPH_CHECK(cudaFree(buffer_));
        }
        buffer_ = nullptr;
      }
      if (host_alloc_) {
        SPH_CHECK(cudaMallocHost(&buffer_, buffer_size));
      } else {
        SPH_CHECK(cudaMalloc(&buffer_, buffer_size));
      }
      size_ = buffer_size;
    }
    return buffer_;
  }

  std::size_t get_size() const { return size_; }

 private:
  bool        host_alloc_;
  void*       buffer_;
  std::size_t size_;
};

}  // namespace sph_support
}  // namespace nve
