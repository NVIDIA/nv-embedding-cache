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

#include <cuda_runtime_api.h>
#include <cstdlib>
#include <memory>
#include <nve_types.hpp>
#include <unordered_map>
#include <sys/mman.h>
#include <common.hpp>
#include <allocator.hpp>

namespace nve {
  /**
 * Round up a value using a given base.
 */
template<typename T>
inline T round_up(T val, T base) {
    return ((val + base - 1) / base) * base;
}

/**
 * Get largest available huge page bits for a given size. This value is needed when calling mmap()
 * @param alloc_size Size of the allocation needed in bytes
 * @returns Number of bits in the huge page enum or 0 if there aren't enough huge pages.
 *          E.g. if there are enough 2MB pages for the desired size the return value will be 21 (1<<21 is 2MB for the page size)
 */
size_t get_largest_hugepage_bits(size_t alloc_size);

/**
 * Default allocator for large memory allocation.
 */
allocator_ptr_t GetDefaultAllocator();

class DefaultAllocator : public Allocator {
public:
  static constexpr size_t DEFAULT_HOST_ALLOC_THRESHOLD = 1ULL << 30; // 1GB

  DefaultAllocator(size_t host_alloc_threshold_bytes);
  virtual ~DefaultAllocator() override = default;
  virtual cudaError_t device_allocate(void** ptr, size_t sz, int device_id = -1) noexcept override;
  virtual cudaError_t device_free(void* ptr, int device_id = -1) noexcept override;
  virtual cudaError_t device_allocate_async(void** ptr, size_t sz, cudaStream_t stream, int device_id = -1) noexcept override;
  virtual cudaError_t device_free_async(void* ptr, cudaStream_t stream, int device_id = -1) noexcept override;
  virtual cudaError_t host_allocate(void** ptr, size_t sz) noexcept override;
  virtual cudaError_t host_free(void* ptr) noexcept override;

private:
  std::unordered_map<int,cudaMemPool_t> mem_pools_;
  cudaMemPool_t getMemPool(int device) noexcept;
  const size_t host_alloc_threshold_;
  std::unordered_map<void*, size_t> host_mmap_allocations_;
};

}  // namespace nve
