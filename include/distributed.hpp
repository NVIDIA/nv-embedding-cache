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
#include <string>
#include <vector>
#include <memory>
#include <cuda.h>

namespace nve {

/**
 * This class abstracts operations in a distributed setting (multi process, multi node, ...).
 * Failed ops (barrier, broadcast, all_gather) are expected to throw std::runtime_error
 */
class DistributedEnv {
public:
  virtual ~DistributedEnv() = default;
  virtual size_t rank() const = 0;
  virtual size_t world_size() const = 0;
  virtual size_t device_count() const = 0;
  virtual int local_device() const = 0;
  virtual bool single_host() const = 0;
  virtual void barrier() = 0;
  virtual void broadcast(uintptr_t buffer, size_t size, int root) = 0;
  virtual void all_gather(uintptr_t send_buffer, uintptr_t recv_buffer, size_t size) = 0;
};

class CUDADistributedBuffer{
public:
  CUDADistributedBuffer(uint64_t size, std::shared_ptr<DistributedEnv> dist_env);
  ~CUDADistributedBuffer();

  std::byte* ptr() const { return buffer_; }
  uint64_t total_size() const { return total_size_; }
  uint64_t shard_size() const { return shard_size_; }
  uint64_t num_shards() const { return num_shards_; }
  
private:
  std::shared_ptr<DistributedEnv> env_ = nullptr;
  uint64_t total_size_ = 0;
  uint64_t shard_size_ = 0;
  uint64_t num_shards_ = 0;
  std::byte* buffer_ = {};
  bool single_host_;
  CUmemGenericAllocationHandle alloc_handle_ = {};
  std::vector<CUmemGenericAllocationHandle> all_alloc_handles_;
  std::vector<int> all_devices_;
  
  size_t get_device_granularity(CUmemAllocationProp prop);
  void init_single_host(uint64_t size);
  void init_multi_host(uint64_t size);
  bool check_imex();
  uint64_t collect_devices(std::vector<int>& all_devices);
};

} // namespace nve
