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
#include <cuda_support.hpp>
#include <memory>
#include <allocator.hpp>

namespace nve {

// Internal class for managing scratch buffers
class ResizeableBuffer {
 public:
  ResizeableBuffer(allocator_ptr_t& allocator, bool host_alloc)
      : host_alloc_(host_alloc), buffer_(nullptr), size_(0), allocator_(allocator) {
    NVE_ASSERT_(allocator);
  }
  ~ResizeableBuffer() {
    if (host_alloc_) {
      NVE_CHECK_(allocator_->hostFree(buffer_));
    } else {
      NVE_CHECK_(allocator_->deviceFree(buffer_));
    }
    size_ = 0;
  }
  // If requested size is larger than existing size, a new buffer will be allocated (old data is lost)
  void* get_ptr(size_t buffer_size) {
    if (buffer_size > size_) {
      if (host_alloc_) {
        if (buffer_) {
          NVE_CHECK_(allocator_->hostFree(buffer_));
        }
        NVE_CHECK_(allocator_->hostAllocate(&buffer_, buffer_size));
      } else {
        if (buffer_) {
          NVE_CHECK_(allocator_->deviceFree(buffer_));
        }
        NVE_CHECK_(allocator_->deviceAllocate(&buffer_, buffer_size));
      }
      size_ = buffer_size;
    }
    return buffer_;
  }
  size_t get_size() const { return size_; }
 private:
  bool host_alloc_;
  void* buffer_;
  size_t size_;
  allocator_ptr_t allocator_;
};

}  // namespace nve
