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

#include "include/memblock.hpp"
#include "include/common.hpp"
#include "include/bit_ops.hpp"

#ifndef NVE_DRIVERLESS_BUILD

#include "include/cuda_support.hpp"
#include "include/cuda_driver_support.hpp"
#include <cstring>
#include <numeric>

namespace nve {

NVLMemBlock::NVLMemBlock(size_t row_size, size_t num_embeddings, nve::DataType_t dtype, const std::vector<int>& gpu_ids)
  : NVLMemBlock(memblock_size_in_bytes(row_size, num_embeddings, dtype), gpu_ids) {}

struct GpuAllocationInfo
{
  size_t sz;
  int id;
};

NVLMemBlock::NVLMemBlock(size_t size_to_alloc, const std::vector<int>& gpu_ids)
  : MemBlock(MemBlockType::NVL) {
  static constexpr size_t kLargePageSizeBytes = size_t(2) * 1024 * 1024;

  int cur_device = 0;
  auto num_devices = gpu_ids.size();
  int num_gpus_in_system = 0;
  NVE_CHECK_(cudaGetDevice(&cur_device));
  NVE_CHECK_(cudaGetDeviceCount(&num_gpus_in_system));
  NVE_CHECK_(num_devices <= static_cast<size_t>(num_gpus_in_system));

  size_t byte_per_gpu = ceil_div(size_to_alloc, num_devices); // first check per device how many bytes are needed
  byte_per_gpu = round_up(byte_per_gpu, kLargePageSizeBytes); // align to large page size`
  NVE_CHECK_((byte_per_gpu % kLargePageSizeBytes) == 0);

  size_t remined_total_bytes = size_to_alloc;
  std::vector<GpuAllocationInfo> allocation_info;
  for (int peer_device : gpu_ids) {
    GpuAllocationInfo info;
    if (remined_total_bytes > byte_per_gpu) {
      info.sz = byte_per_gpu;
      info.id = peer_device;
      remined_total_bytes -= byte_per_gpu;
    } else {
      info.sz = round_up(remined_total_bytes, kLargePageSizeBytes);
      info.id = peer_device;
      remined_total_bytes = 0;
    }
    NVE_CHECK_(cudaSetDevice(peer_device));
    size_t avail_phy_vidmem = 0, total_phy_vidmem = 0;
    NVE_CHECK_(cudaMemGetInfo(&avail_phy_vidmem, &total_phy_vidmem));
    NVE_CHECK_(info.sz <= avail_phy_vidmem, "Not enough available GPU memory, consider using: torch.cuda.empty_cache()");
    allocation_info.push_back(info);
    if (remined_total_bytes == 0) {
      break;
    }
  }

  handles_.resize(allocation_info.size());
  for (size_t i = 0; i < allocation_info.size(); ++i) {
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof(CUmemAllocationProp));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = allocation_info[i].id;

    // Create context on each GPU
    NVE_CHECK_(cudaSetDevice(allocation_info[i].id));
    NVE_CHECK_(cudaFree(0));
    NVE_CHECK_(cudaSetDevice(0));

    NVE_CHECK_((cuMemCreate(&handles_[i], allocation_info[i].sz, &prop, 0) == 0));
  }

  total_bytes_ = std::accumulate(allocation_info.begin(), allocation_info.end(), size_t(0), [](size_t a, const GpuAllocationInfo& b) {
    return a + b.sz;
  });
  // Reserve VA space for the total size of allocation
  NVE_CHECK_((cuMemAddressReserve(&ptr_, total_bytes_, 0, 0, 0) == 0));

  // Map each GPU's physical allocation to corresponding portion of VA space
  size_t offset = 0;
  for (size_t i = 0; i < handles_.size(); ++i) {
    NVE_CHECK_(cuMemMap(ptr_ + offset, allocation_info[i].sz, 0, handles_[i], 0) == 0);
    offset = offset + allocation_info[i].sz;
  }

  // Make allocation visible to all local GPUs
  std::vector<CUmemAccessDesc> access_descriptors(static_cast<size_t>(num_gpus_in_system));
  for (size_t i = 0; i < access_descriptors.size(); ++i) {
    access_descriptors[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_descriptors[i].location.id = static_cast<int>(i);
    access_descriptors[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  }
  NVE_CHECK_(cuMemSetAccess(ptr_, total_bytes_, access_descriptors.data(), access_descriptors.size()) == 0);

  NVE_CHECK_(cudaDeviceSynchronize());
  NVE_CHECK_(cudaMemset(get_ptr(), 0, total_bytes_));
  NVE_CHECK_(cudaSetDevice(cur_device));
}

NVLMemBlock::~NVLMemBlock() {
  NVE_CHECK_(cuMemUnmap(ptr_, total_bytes_) == 0);
  for (size_t i = 0; i < handles_.size(); ++i) {
    NVE_CHECK_(cuMemRelease(handles_[i]) == 0);
  }
  NVE_CHECK_(cuMemAddressFree(ptr_, total_bytes_) == 0);
}

void* NVLMemBlock::get_ptr() const {
  return reinterpret_cast<void*>(ptr_);
}

}  // namespace nve

#else  // NVE_DRIVERLESS_BUILD

// Driverless stub: same symbols, throwing constructors, no libcuda dependency.
namespace nve {

namespace {
constexpr char kNoDriver[] =
  "NVLMemBlock requires CUDA driver support; this build was compiled with "
  "NVE_DRIVERLESS_BUILD=ON (no CUDA driver support).";
}

NVLMemBlock::NVLMemBlock(size_t, size_t, nve::DataType_t, const std::vector<int>&)
  : MemBlock(MemBlockType::NVL) {
  NVE_THROW_(kNoDriver);
}

NVLMemBlock::NVLMemBlock(size_t, const std::vector<int>&)
  : MemBlock(MemBlockType::NVL) {
  NVE_THROW_(kNoDriver);
}

NVLMemBlock::~NVLMemBlock() {}

void* NVLMemBlock::get_ptr() const {
  return nullptr;
}

}  // namespace nve

#endif  // NVE_DRIVERLESS_BUILD
