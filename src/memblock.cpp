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
#include "include/bit_ops.hpp"
#include "include/common.hpp"
#include "include/cuda_support.hpp"
#include "include/default_allocator.hpp"
#include "include/distributed.hpp"
#include <cstring>

namespace nve {

size_t memblock_size_in_bytes(size_t row_size, size_t num_embeddings, nve::DataType_t dtype) {
  // row_size counts values; rowwise-quant rows also carry trailing metadata.
  NVE_CHECK_(row_size <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
             "MemBlock row size computation overflows result type");
  const auto row_bytes = static_cast<size_t>(row_size_in_bytes(dtype, static_cast<int64_t>(row_size)));
  return checked_mul(row_bytes, num_embeddings, "MemBlock size");
}

LinearMemBlock::LinearMemBlock(size_t row_size, size_t num_embeddings, nve::DataType_t dtype, int device_id)
  : LinearMemBlock(memblock_size_in_bytes(row_size, num_embeddings, dtype), device_id) {}

LinearMemBlock::LinearMemBlock(size_t size_to_alloc, int device_id)
  : MemBlock(MemBlockType::LINEAR),
  size_(size_to_alloc),
  allocator_(GetDefaultAllocator()),
  device_id_(device_id) {
  NVE_CHECK_(allocator_ != nullptr);
  if (device_id_ < 0) {
    NVE_CHECK_((allocator_->host_allocate(&ptr_, size_to_alloc)));
  } else {
    NVE_CHECK_((allocator_->device_allocate(&ptr_, size_to_alloc, device_id_)));
  }
  NVE_CHECK_(ptr_ != nullptr);
}

LinearMemBlock::~LinearMemBlock() {
  if (device_id_ < 0) {
    allocator_->host_free(ptr_);
  } else {
    allocator_->device_free(ptr_, device_id_);
  }
}

void* LinearMemBlock::get_ptr() const {
  return ptr_;
}

DistMemBlock::DistMemBlock(std::shared_ptr<DistributedEnv> env, size_t row_size, size_t num_embeddings, nve::DataType_t dtype) 
  : MemBlock(MemBlockType::MPI) {
  auto size_to_alloc = memblock_size_in_bytes(row_size, num_embeddings, dtype);
  dist_buffer_ = std::make_shared<nve::CUDADistributedBuffer>(size_to_alloc, env, nve::BufferLocation::ALLOCATION_GPU_MEM);
  NVE_CHECK_(dist_buffer_->ptr() != nullptr);
}

void* DistMemBlock::get_ptr() const {
  return dist_buffer_->ptr();
}

size_t DistMemBlock::get_size_in_bytes() const {
  return dist_buffer_->total_size();
}

DistHostMemBlock::DistHostMemBlock(std::shared_ptr<DistributedEnv> env, size_t row_size, size_t num_embeddings, nve::DataType_t dtype) 
  : MemBlock(MemBlockType::MPI) {
  auto size_to_alloc = memblock_size_in_bytes(row_size, num_embeddings, dtype);
  dist_buffer_ = std::make_shared<nve::CUDADistributedBuffer>(size_to_alloc, env, nve::BufferLocation::ALLOCATION_SYS_MEM);
  NVE_CHECK_(dist_buffer_->ptr() != nullptr);
}

void* DistHostMemBlock::get_ptr() const {
  return dist_buffer_->ptr();
}

size_t DistHostMemBlock::get_size_in_bytes() const {
  return dist_buffer_->total_size();
}

UserMemBlock::UserMemBlock(uint64_t ptr, size_t size)
  : MemBlock(MemBlockType::USER), ptr_(reinterpret_cast<void*>(ptr)), size_(size) {}

void* UserMemBlock::get_ptr() const {
  return ptr_;
}

HostMemBlock::HostMemBlock(size_t row_size, size_t num_embeddings, nve::DataType_t dtype)
  : HostMemBlock(memblock_size_in_bytes(row_size, num_embeddings, dtype)) {}

HostMemBlock::HostMemBlock(size_t size_to_alloc)
  : MemBlock(MemBlockType::HOST), ptr_(nullptr), size_(size_to_alloc), allocator_(GetDefaultAllocator()) {
  NVE_CHECK_(allocator_ != nullptr);
  NVE_CHECK_((allocator_->host_allocate(&ptr_, size_to_alloc)));
  NVE_CHECK_(ptr_ != nullptr);
}

HostMemBlock::~HostMemBlock() {
  allocator_->host_free(ptr_);
}

void* HostMemBlock::get_ptr() const {
  return ptr_;
}

ManagedMemBlock::ManagedMemBlock(size_t row_size, size_t num_embeddings, nve::DataType_t dtype, const std::vector<int>& gpu_ids) 
  : ManagedMemBlock(memblock_size_in_bytes(row_size, num_embeddings, dtype), gpu_ids) {}

ManagedMemBlock::ManagedMemBlock(size_t size_to_alloc, const std::vector<int>& gpu_ids)
  : MemBlock(MemBlockType::MANAGED), size_(size_to_alloc) {
  NVE_CHECK_(cudaMallocManaged(&ptr_, size_to_alloc));
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
  cudaMemLocation loc;
  loc.id = 0;
  loc.type = cudaMemLocationTypeHost;
  NVE_CHECK_(cudaMemAdvise(ptr_, size_to_alloc, cudaMemAdviseSetPreferredLocation, loc));
  for (int gpu_id : gpu_ids) {
    loc.type = cudaMemLocationTypeDevice;
    loc.id = gpu_id;
    NVE_CHECK_(cudaMemAdvise(ptr_, size_to_alloc, cudaMemAdviseSetAccessedBy, loc));
  }
#else
  NVE_CHECK_(cudaMemAdvise(ptr_, size_to_alloc, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
  for (int gpu_id : gpu_ids) {
    NVE_CHECK_(cudaMemAdvise(ptr_, size_to_alloc, cudaMemAdviseSetAccessedBy, gpu_id));
  }
#endif
  NVE_CHECK_(cudaDeviceSynchronize());
}

void* ManagedMemBlock::get_ptr() const {
  return ptr_;
}

ManagedMemBlock::~ManagedMemBlock() {
  NVE_CHECK_(cudaFree(ptr_));
}

std::vector<int> resolve_memblock_devices(
  MemBlockType type, int def_index,
  const std::vector<int>& override)
{
  if (!override.empty()) {
    return override;
  }
  if (type == MemBlockType::NVL) {
    int num_gpus = 0;
    NVE_CHECK_(cudaGetDeviceCount(&num_gpus));
    NVE_CHECK_(def_index >= 0 && def_index < num_gpus,
           "resolve_memblock_devices: def_index=" + std::to_string(def_index) +
           " out of range, system has " + std::to_string(num_gpus) + " GPUs");
    std::vector<int> ids;
    ids.reserve(static_cast<size_t>(num_gpus - def_index));
    for (int i = def_index; i < num_gpus; ++i)
      ids.push_back(i);
    return ids;
  }
  return {def_index};
}

}  // namespace nve
