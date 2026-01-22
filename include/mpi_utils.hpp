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
#include <ostream>
#include <cuda.h>
#include <distributed.hpp>
#include <mpi.h>

namespace nve {

/**
 * Wrapper class for common MPI utilities
 * Note: if MPI_Init() was not called before constructing this object, the c'tor will call MPI_Init
 * and MPI_Finalize will be called during unload.
 */
class MPIEnv : public DistributedEnv {
public:
  MPIEnv(const std::vector<size_t> ranks = std::vector<size_t>{}, const std::vector<int> devices = std::vector<int>{});
  virtual ~MPIEnv() override;
  virtual size_t rank() const override { return rank_; }
  virtual size_t world_size() const override { return world_size_; }
  virtual size_t device_count() const override { return device_count_; }
  virtual int local_device() const override { return local_device_; }
  virtual bool single_host() const override { return single_host_; }
  virtual void barrier() override;
  virtual void broadcast(uintptr_t buffer, size_t size, int root) override;
  virtual void all_gather(uintptr_t send_buffer, uintptr_t recv_buffer, size_t size) override;
  
  const std::string& host_name() const { return host_name_; }
  const std::string& device_name() const { return device_name_; }

private:
  size_t rank_ = 0;
  size_t world_size_ = 0;
  size_t device_count_ = 0;
  int local_device_ = -1;
  std::string host_name_;
  std::string device_name_;
  bool single_host_ = true;
};

std::ostream& operator<<(std::ostream& os, const MPIEnv& env);

} // namespace nve
