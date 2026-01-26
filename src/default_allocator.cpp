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

#include <cuda_support.hpp>
#include <bit_ops.hpp>
#include "include/default_allocator.hpp"
#include <filesystem>
#include <sstream>
#include <fstream>

// Define debug print macro before CUDA error macro
#ifdef DEBUG_DEFAULT_ALLOCATOR
#include <cstdio>
  #define ALLOC_DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
  #define ALLOC_DEBUG_PRINT(...)
#endif

#ifdef RETURN_IF_CUDA_ERROR_
#error RETURN_IF_CUDA_ERROR_ was already defined.
#endif
#define RETURN_IF_CUDA_ERROR_(_expr_)   \
do {                                    \
  auto res = _expr_;                    \
  if (res != cudaSuccess) {             \
    return res;                         \
  }                                     \
} while (false)

namespace nve {

allocator_ptr_t GetDefaultAllocator() {
    static allocator_ptr_t global_default_allocator;
    if (!global_default_allocator) {
        global_default_allocator = std::make_shared<DefaultAllocator>(DefaultAllocator::DEFAULT_HOST_ALLOC_THRESHOLD);
        NVE_CHECK_(global_default_allocator != nullptr, "Failed to construct the global allocator.");
    }
    return global_default_allocator;
}

size_t get_largest_hugepage_bits(size_t alloc_size) {
    const std::filesystem::path hugepage_dir{"/sys/devices/system/node/node0/hugepages"};

    size_t largest_bits = 0;

    for (auto const& entry : std::filesystem::directory_iterator{hugepage_dir}) {
        std::string path_str(entry.path().c_str());

        // Get page size in kB
        std::string start_token("-");
        std::string end_token("kB");
        auto start = path_str.find(start_token);
        auto end = path_str.find(end_token);
        if ((start == std::string::npos) || (end == std::string::npos)) {
            continue;
        }
        start += start_token.size(); // advance start to remove the start token
        size_t page_size_kb;
        std::istringstream(path_str.substr(start, end - start)) >> page_size_kb;
        NVE_ASSERT_(std::popcount(page_size_kb) == 1);
        const size_t page_size = page_size_kb << 10;

        // Check free pages
        size_t needed_hugepages = ceil_div(alloc_size, page_size);
        size_t free_hugepages = 0;
        std::ifstream free_hugepages_stream(std::string(path_str) + "/free_hugepages");
        if (free_hugepages_stream.is_open()) {
            free_hugepages_stream >> free_hugepages;
        }

        if (free_hugepages >= needed_hugepages) {
            auto page_size_bits = countr_zero(page_size);// convert from 1024kB to 20 etc.
            largest_bits = std::max(largest_bits, static_cast<size_t>(page_size_bits));
        }
    }
    return largest_bits;
}

DefaultAllocator::DefaultAllocator(size_t host_alloc_threshold_bytes) : host_alloc_threshold_(host_alloc_threshold_bytes) {}

cudaError_t DefaultAllocator::device_allocate(void** ptr, size_t sz, int device_id) noexcept {
    // Figure out if we need to change current device
    int curr_device;
    RETURN_IF_CUDA_ERROR_(cudaGetDevice(&curr_device));
    const bool swap_device = (device_id >= 0) && (device_id != curr_device);
    if (swap_device) {
        RETURN_IF_CUDA_ERROR_(cudaSetDevice(device_id));
    }

    // coverity[error_interface]
    auto res = cudaMalloc(ptr,sz); // not returning on error before changing back to original device
    // Error will be returned to caller, thereforce the coverity error is disabled as intentional

    // Swap back to original device
    if (swap_device) {
        RETURN_IF_CUDA_ERROR_(cudaSetDevice(curr_device));
    }
    ALLOC_DEBUG_PRINT("[%s] ptr=%p, sz=%ld, device=%d\n", __FUNCTION__, *ptr, sz, device_id);
    return res;
}

cudaError_t DefaultAllocator::device_free(void* ptr, int device_id) noexcept {
    ALLOC_DEBUG_PRINT("[%s] ptr=%p, device=%d\n", __FUNCTION__, ptr, device_id);
    int curr_device;
    RETURN_IF_CUDA_ERROR_(cudaGetDevice(&curr_device));
    const bool swap_device = (device_id >= 0) && (device_id != curr_device);
    if (swap_device) {
        RETURN_IF_CUDA_ERROR_(cudaSetDevice(device_id));
    }

    // coverity[error_interface]
    auto res = cudaFree(ptr); // not returning on error before changing back to original device
    // Error will be returned to caller, thereforce the coverity error is disabled as intentional

    if (swap_device) {
        RETURN_IF_CUDA_ERROR_(cudaSetDevice(curr_device));
    }
    return res;
}

cudaError_t DefaultAllocator::device_allocate_async(void** ptr, size_t sz, cudaStream_t stream, int device_id) noexcept {
    // Figure out if we need to change current device
    int curr_device;
    RETURN_IF_CUDA_ERROR_(cudaGetDevice(&curr_device));
    const bool swap_device = (device_id >= 0) && (device_id != curr_device);

    // Get memory pool if available
    cudaMemPool_t mem_pool;
    if (swap_device) {
        RETURN_IF_CUDA_ERROR_(cudaSetDevice(device_id));
        mem_pool = getMemPool(device_id);
    } else {
        mem_pool = getMemPool(curr_device);
    }

    // Allocate
    // coverity[error_interface]
    cudaError_t res = mem_pool ? cudaMallocFromPoolAsync(ptr, sz, mem_pool, stream) : cudaMalloc(ptr,sz);
    // not returning on error before changing back to original device
    // Error will be returned to caller, thereforce the coverity error is disabled as intentional

    // Swap back to original device
    if (swap_device) {
        RETURN_IF_CUDA_ERROR_(cudaSetDevice(curr_device));
    }
    ALLOC_DEBUG_PRINT("[%s] ptr=%p, sz=%ld, stream=%p, device=%d\n", __FUNCTION__, *ptr, sz, stream, device_id);
    return res;
}

cudaError_t DefaultAllocator::device_free_async(void* ptr, cudaStream_t stream, int device_id) noexcept {
    ALLOC_DEBUG_PRINT("[%s] ptr=%p, stream=%p, device=%d\n", __FUNCTION__, ptr, stream, device_id);
    int curr_device;
    RETURN_IF_CUDA_ERROR_(cudaGetDevice(&curr_device));
    const bool swap_device = (device_id >= 0) && (device_id != curr_device);
    if (swap_device) {
        RETURN_IF_CUDA_ERROR_(cudaSetDevice(device_id));
    }

    // coverity[error_interface]
    auto res = cudaFreeAsync(ptr, stream); // not returning on error before changing back to original device
    // Error will be returned to caller, thereforce the coverity error is disabled as intentional

    if (swap_device) {
        RETURN_IF_CUDA_ERROR_(cudaSetDevice(curr_device));
    }
    return res;
}

cudaError_t DefaultAllocator::host_allocate(void** ptr, size_t sz) noexcept {
    size_t hugepage_bits = 0;
    if(sz >= host_alloc_threshold_) {
      hugepage_bits = get_largest_hugepage_bits(sz);
      if (hugepage_bits == 0) {
          NVE_LOG_WARNING_("Not enough huge pages available to allocate, performance will be impacted.");
      }
    } 
    if(hugepage_bits == 0) {
      // Allocate using cudaMallocHost
      auto res = cudaMallocHost(ptr, sz);
      ALLOC_DEBUG_PRINT("[%s] cudaMallocHost allocation: ptr=%p, sz=%ld\n", __FUNCTION__, *ptr, sz);
      return res;
    }
    else {
      // Allocate huge pages
      size_t totalBytes = round_up(sz, size_t(1) << static_cast<size_t>(hugepage_bits));
      *ptr = mmap(NULL, totalBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_HUGETLB | (static_cast<int>(hugepage_bits) << MAP_HUGE_SHIFT), -1, 0);
      if (*ptr == MAP_FAILED) {
          ALLOC_DEBUG_PRINT("[%s] Huge pages allocation failed: ptr=%p, sz=%ld\n", __FUNCTION__, *ptr, sz);
          return cudaErrorMemoryAllocation;
      }
      host_mmap_allocations_[*ptr] = totalBytes;
      ALLOC_DEBUG_PRINT("[%s] Huge pages allocation: ptr=%p, requested=%ld, allocated=%ld\n", __FUNCTION__, *ptr, sz, totalBytes);
      return cudaHostRegister(*ptr, totalBytes, cudaHostRegisterDefault);
    }
}

cudaError_t DefaultAllocator::host_free(void* ptr) noexcept {
    auto ptr_alloc_data = host_mmap_allocations_.find(ptr);
    if (ptr_alloc_data != host_mmap_allocations_.end()) {
      auto res = cudaHostUnregister(ptr);
      if (res != cudaSuccess) {
          return res;
      }
      if (munmap(ptr, ptr_alloc_data->second) != 0) {
          ALLOC_DEBUG_PRINT("[%s] Failed to free: ptr=%p, sz=%ld\n", __FUNCTION__, ptr, ptr_alloc_data->second);
          return cudaErrorInvalidValue;
      }
      ALLOC_DEBUG_PRINT("[%s] ptr=%p\n", __FUNCTION__, ptr);
      host_mmap_allocations_.erase(ptr_alloc_data);
      return cudaSuccess;
    }
    else {
      ALLOC_DEBUG_PRINT("[%s] ptr=%p\n", __FUNCTION__, ptr);
      return cudaFreeHost(ptr);
    }
}

cudaMemPool_t DefaultAllocator::getMemPool(int device) noexcept {
    if (mem_pools_.find(device) == mem_pools_.end()) {
        mem_pools_[device] = nullptr;
        int mem_pool_supported;
        if (cudaDeviceGetAttribute(&mem_pool_supported, cudaDevAttrMemoryPoolsSupported, device)) {
            // Silently defaulting to no mem pool
            mem_pool_supported = 0;
        }
        if (mem_pool_supported) {
            cudaMemPool_t mem_pool;
            size_t free, total;
            if ( ((cudaDeviceGetDefaultMemPool(&mem_pool, device)) == cudaSuccess) && 
                 ((cudaMemGetInfo(&free, &total)) == cudaSuccess) ) {
              uint64_t threshold = total;
              if (cudaMemPoolSetAttribute(mem_pool, cudaMemPoolAttrReleaseThreshold, &threshold) == cudaSuccess) {
                mem_pools_[device] = mem_pool;
              }
            }
            // Implciti else - silently defaulting to no mem pool
        }
    }
    return mem_pools_.at(device);
}

#undef RETURN_IF_CUDA_ERROR_
#undef ALLOC_DEBUG_PRINT

}  // namespace nve
