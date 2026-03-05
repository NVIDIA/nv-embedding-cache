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

#include <distributed.hpp>
#include <common.hpp>
#include <cuda_support.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <sys/syscall.h>
#include <filesystem>

#define ERRNO_CHECK(_expr_) \
do { \
  auto res = (_expr_); \
  if (res != 0) { \
    std::ostringstream oss; \
    oss << "Error (" << std::strerror(res) << ") at " << __FILE__ << ":" << __LINE__ << " :: " #_expr_ << std::endl; \
    throw std::runtime_error(oss.str());\
  } \
} while (0);

#define ROUND_UP(n, multiple) \
    (((n) + ((multiple)-1)) - (((n) + ((multiple)-1)) % (multiple)))


// Wrapper for pidfd_open syscall
static int pidfd_open(pid_t pid, unsigned int flags) {
    return static_cast<int>(syscall(SYS_pidfd_open, pid, flags));
}

// Wrapper for pidfd_getfd syscall
static int pidfd_getfd(int pidfd, int targetfd, unsigned int flags) {
    return static_cast<int>(syscall(SYS_pidfd_getfd, pidfd, targetfd, flags));
}

namespace nve {

CUDADistributedBuffer::CUDADistributedBuffer(uint64_t size, std::shared_ptr<DistributedEnv> dist_env, BufferLocation location) : env_(dist_env) {
  NVE_CHECK_(dist_env != nullptr);
  single_host_ = env_->single_host();
  if (single_host_) {
    init_single_host(size, location);
  } else {
    NVE_CHECK_(location != BufferLocation::ALLOCATION_SYS_MEM, "Distributed buffer on host is not supported for multi-host deployments");
    init_multi_host(size);
  }
  env_->barrier();
  NVE_IF_DEBUG_(
    std::cout << __FUNCTION__
      << " size " << size
      << ", rank " << env_->rank()
      << ", local_device " << env_->local_device()
      << ", world_size " << env_->world_size()
      << ", num_shards_ " << num_shards_
      << ", shard_size_ " << shard_size_
      << ", total_size_ " << total_size_
      << std::endl
  );
}

CUDADistributedBuffer::~CUDADistributedBuffer() {
  // Make sure all processes got here
  env_->barrier();

  // Unmap each allocation
  for (size_t i=0 ; i<num_shards_ ; i++) {
    const CUdeviceptr buf_start = reinterpret_cast<CUdeviceptr>(buffer_ + (i * shard_size_));
    NVE_CHECK_(cuMemUnmap(buf_start, shard_size_));
  }

  // Release all allocations

  for (size_t i=0 ; i<all_alloc_handles_.size() ; i++) {
    NVE_CHECK_(cuMemRelease(all_alloc_handles_[i]));
  }

  if (!single_host_ && (all_devices_[env_->rank()] >= 0)) {
    NVE_CHECK_(cuMemRelease(alloc_handle_));
  }

  // Make sure all processes got here
  env_->barrier();

  // Release buffer reservation
  NVE_CHECK_(cuMemAddressFree(reinterpret_cast<CUdeviceptr>(buffer_), total_size_));
}

size_t CUDADistributedBuffer::get_device_granularity(CUmemAllocationProp prop) {
  size_t granularity = 0;
  prop.location.id = 0;
  NVE_CHECK_(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

  // Verify all devices use the same granularity
  for (size_t i=1 ; i<env_->device_count() ; i++) {
    size_t dev_granularity;
    prop.location.id = static_cast<int>(i);
    NVE_CHECK_(cuMemGetAllocationGranularity(&dev_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    NVE_CHECK_(granularity == dev_granularity);
  }

  return granularity;
}

void CUDADistributedBuffer::init_single_host(uint64_t size, BufferLocation location) {
  const bool root_proc = (env_->rank() == 0);
  NVE_CHECK_(env_->single_host());
  num_shards_ = collect_devices(all_devices_);
  const size_t world_size = env_->world_size();

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = (location == BufferLocation::ALLOCATION_SYS_MEM) ? CU_MEM_LOCATION_TYPE_HOST_NUMA : CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  size_t granularity = get_device_granularity(prop); // we assume all GPUs will have the same granularity requirements

  // Round shard to multiple of num_shards_ * granularity (need every shard to be aligned to granularity)
  total_size_ = ROUND_UP(size, num_shards_ * granularity);
  shard_size_ = total_size_ / num_shards_;

  // Rank 0 should do all cuMemCreate and export
  std::vector<int> shareable_handles(world_size, -1);
  if (root_proc) {
    for (size_t i=0 ; i<world_size ; i++) {
      if (all_devices_[i] >= 0) {
        // Allocate & export handle
        CUmemGenericAllocationHandle handle;
        prop.location.id = (location == BufferLocation::ALLOCATION_SYS_MEM) ? 0 : all_devices_[i];
        NVE_CHECK_(cuMemCreate(&handle, shard_size_, &prop, 0 /*flags*/));
        NVE_CHECK_(cuMemExportToShareableHandle(&shareable_handles.at(i), handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0 /*flags*/));
        all_alloc_handles_.push_back(handle);
      }
    }
  }

  // Broadcast the shareable handles to all procs
  env_->broadcast(reinterpret_cast<uintptr_t>(shareable_handles.data()), sizeof(shareable_handles[0]) * shareable_handles.size(), 0);

  // Broadcast root pid so we can map FDs
  pid_t root_pid = root_proc ? getpid() : -1;
  env_->broadcast(reinterpret_cast<uintptr_t>(&root_pid), sizeof(root_pid), 0);

  // Map FD's in all non root procs
  if (!root_proc) {
    int root_pidfd = pidfd_open(root_pid, 0);
    NVE_CHECK_(root_pidfd != -1);
    for (size_t i=0 ; i<world_size; i++) {
      if (all_devices_[i] >= 0) {
        int local_fd = pidfd_getfd(root_pidfd, shareable_handles[i], 0);
        NVE_CHECK_(local_fd != -1, "pidfd_getfd() failed, make sure SYS_PTRACE is available!");
        shareable_handles[i] = local_fd;
      }
    }
    // Now close the root pidfd - it's no longer used
    ERRNO_CHECK(close(root_pidfd));
  }

  // Now all procs can reserve, import and map
  // Reserve virtual address range for the unified buffer
  NVE_CHECK_(cuMemAddressReserve((CUdeviceptr *) &buffer_, total_size_, 0, 0 /*baseVA*/, 0 /*flags*/));

  for (size_t i=0, shard_id=0 ; i<world_size ; i++) {
    if (all_devices_[i] >= 0) {
      // Import
      CUmemGenericAllocationHandle handle;
      NVE_CHECK_(cuMemImportFromShareableHandle(&handle, reinterpret_cast<void*>(shareable_handles.at(i)), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

      // Map
      const CUdeviceptr buf_start = reinterpret_cast<CUdeviceptr>(buffer_ + (shard_id * shard_size_));
      NVE_CHECK_(cuMemMap(buf_start, shard_size_, 0 /*offset*/, handle, 0 /*flags*/));

      // Release the handled we imported (not needed anymore)
      NVE_CHECK_(cuMemRelease(handle));
      shard_id++;
    }
  }

  // Set access
  std::vector<CUmemAccessDesc> access_descs;
  for (size_t i=0 ; i<world_size ; i++) {
    if (all_devices_[i] >= 0) {
      CUmemAccessDesc desc;
      desc.location.id = all_devices_[i];
      desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      access_descs.push_back(desc);
    }
  }
  NVE_CHECK_(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(buffer_), total_size_, access_descs.data(), access_descs.size()));

  // Now we can close all shareable handles
  env_->barrier();
  for (size_t i=0 ; i<world_size; i++) {
    if (all_devices_[i] >= 0) {
      ERRNO_CHECK(close(shareable_handles.at(i)));
    }
  }
}

bool CUDADistributedBuffer::check_imex() {
  const std::string imex_root("/dev/nvidia-caps-imex-channels");
  std::filesystem::path fs_path(imex_root);
  bool found_imex = false;
  try {
    for ([[maybe_unused]] const auto& it : std::filesystem::directory_iterator(fs_path)) {
      NVE_IF_DEBUG_(std::cout << "Found IMEX channel: " << it.path().string() << std::endl);
      found_imex = true;
    }
  } catch (const std::filesystem::filesystem_error& e) {
    return false;
  }
  return found_imex;
}

void CUDADistributedBuffer::init_multi_host(uint64_t size) {
  // Check for IMEX channels
  NVE_CHECK_(check_imex(), "Failed to locate IMEX channel");

  const auto local_device = env_->local_device();
  num_shards_ = collect_devices(all_devices_);
  const size_t world_size = env_->world_size();

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<int>(std::max(local_device, 0)); // Assuming required granularity of all devices is the same (so using local device or 0 to query)
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

  size_t granularity = get_device_granularity(prop);

  // Round shard to multiple of num_shards_ * granularity (need every shard to be aligned to granularity)
  total_size_ = ROUND_UP(size, num_shards_ * granularity);
  shard_size_ = total_size_ / num_shards_;

  // Allocate & export handle
  CUmemFabricHandle fabric_handle = {};
  if (local_device >= 0) {
    NVE_CHECK_(cuMemCreate(&alloc_handle_, shard_size_, &prop, 0 /*flags*/));
    NVE_CHECK_(cuMemExportToShareableHandle(&fabric_handle, alloc_handle_, CU_MEM_HANDLE_TYPE_FABRIC, 0 /*flags*/));
  }

  // Gather all exported handles
  std::vector<CUmemFabricHandle> all_fabric_handles(world_size);
  env_->all_gather(reinterpret_cast<uintptr_t>(&fabric_handle), reinterpret_cast<uintptr_t>(&all_fabric_handles[0]), sizeof(fabric_handle));

  // Import all exported handles
  for (size_t i=0 ; i<world_size ; i++) {
    if (all_devices_[i] >= 0) {
      CUmemGenericAllocationHandle handle = {};
      NVE_CHECK_(cuMemImportFromShareableHandle(&handle, reinterpret_cast<void*>(&(all_fabric_handles.at(i))), CU_MEM_HANDLE_TYPE_FABRIC));
      all_alloc_handles_.push_back(handle);
    }
  }
  NVE_CHECK_(all_alloc_handles_.size() == num_shards_);

  // Reserve virtual address range for the unified buffer
  NVE_CHECK_(cuMemAddressReserve(reinterpret_cast<CUdeviceptr*>(&buffer_), total_size_, 0, 0 /*baseVA*/, 0 /*flags*/));

  // Finally map each shared handle to its part of the reserved bufer
  for (size_t i=0 ; i<num_shards_ ; i++) {
    const CUdeviceptr buf_start = reinterpret_cast<CUdeviceptr>(buffer_ + (i * shard_size_));
    NVE_CHECK_(cuMemMap(buf_start, shard_size_, 0 /*offset*/, all_alloc_handles_.at(i), 0 /*flags*/));
  }

  // And set the access for the buffer
  CUmemAccessDesc desc = {};
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = static_cast<int>(std::max(local_device, 0));
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  NVE_CHECK_(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(buffer_), total_size_, &desc, 1 /*count*/));
}

uint64_t CUDADistributedBuffer::collect_devices(std::vector<int>& all_devices) {
  int local_device = env_->local_device();
  const auto world_size = env_->world_size();

  all_devices.resize(world_size);
  env_->all_gather(reinterpret_cast<uintptr_t>(&local_device), reinterpret_cast<uintptr_t>(&all_devices[0]), sizeof(local_device));

  uint64_t num_devices = 0;
  for (auto& d : all_devices) {
    if (d >= 0) {
      num_devices++;
    }
  }
  return num_devices;
}

} // namespace nve
