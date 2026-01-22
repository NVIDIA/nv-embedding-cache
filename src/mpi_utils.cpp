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

#include <vector>
#include <memory>
#include <mpi_utils.hpp>
#include <mpi.h>
#include "cuda_ops/cuda_common.h"
#include <cuda_support.hpp>
#include <unistd.h>
#include <sys/syscall.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <stdexcept>
#include <common.hpp>
#include <cstring>

// Macro to only report the error when we can't throw an exception (e.g. destructors)
#define MAX_ERROR_STRING (1024)
#define MPI_CHECK_NO_THROW(_expr_) \
do { \
  auto res = (_expr_); \
  if (res != MPI_SUCCESS) { \
    char err_desc[MAX_ERROR_STRING]; \
    int result_len; \
    MPI_Error_string(res, err_desc, &result_len); \
    std::ostringstream oss; \
    oss << "MPI Error (" << err_desc << ") at " << __FILE__ << ":" << __LINE__ << " :: " #_expr_ << std::endl; \
    std::cerr << oss.str() << std::endl; \
  } \
} while (0);

#define MPI_CHECK(_expr_) \
do { \
  auto res = (_expr_); \
  if (res != MPI_SUCCESS) { \
    char err_desc[MAX_ERROR_STRING]; \
    int result_len; \
    MPI_Error_string(res, err_desc, &result_len); \
    std::ostringstream oss; \
    oss << "MPI Error (" << err_desc << ") at " << __FILE__ << ":" << __LINE__ << " :: " #_expr_ << std::endl; \
    throw std::runtime_error(oss.str());\
  } \
} while (0);

namespace nve {

// Class to initialize/finalize MPI when needed
class MPIInit {
public:
  MPIInit() { 
    int mpi_initialized = 0;
    MPI_CHECK(MPI_Initialized(&mpi_initialized));
    if (mpi_initialized) {
      finalize_needed_ = false;
    } else {
      finalize_needed_ = true;
      MPI_CHECK(MPI_Init(NULL, NULL)); 
    }
  }
  ~MPIInit() {
    if (finalize_needed_) {
      int finalized=0;
      MPI_CHECK_NO_THROW(MPI_Finalized(&finalized));
      if (!finalized) {
        MPI_CHECK_NO_THROW(MPI_Finalize());
      }
    }
  }
private:
  bool finalize_needed_ = false;
};
static std::unique_ptr<MPIInit> _mpi_; // Will finalize MPI during unload if init was called

MPIEnv::MPIEnv(const std::vector<size_t> ranks, const std::vector<int> devices) {
  NVE_CHECK_(ranks.size() == devices.size());
  if (!_mpi_) {
    _mpi_ = std::make_unique<MPIInit>();
  }

  // Init cuda
  int val = 0;
  NVE_CHECK_(cuInit(0));
  NVE_CHECK_(cuDeviceGetCount(&val));
  device_count_ = static_cast<size_t>(val);
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &val));
  world_size_ = static_cast<size_t>(val);
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &val));
  rank_ = static_cast<size_t>(val);

  constexpr size_t STRING_LENGTH = 1024;
  char tmp_str[STRING_LENGTH] = {0};

  if (gethostname(tmp_str, STRING_LENGTH)) {
    throw std::runtime_error("gethostname failed!");
  } else {
    host_name_ = tmp_str;
  }

  std::vector<char> all_hostnames(world_size_ * STRING_LENGTH);
  MPI_CHECK(MPI_Allgather(tmp_str, STRING_LENGTH, MPI_BYTE, &all_hostnames[0], STRING_LENGTH, MPI_BYTE, MPI_COMM_WORLD));

  // Check for single host
  single_host_ = true;
  for (size_t i=0 ; i<world_size_ ; i++) {
    if (strncmp(tmp_str, &all_hostnames[i * STRING_LENGTH], STRING_LENGTH) != 0) {
      single_host_ = false;
      break;
    }
  }

  // Determine local device (-1 means this rank doesn't own a shard)
  if (devices.size() == 0) {
    int local_rank_ = 0;
    for (size_t i=0; i<rank_; i++) {
      if (strncmp(tmp_str, &all_hostnames[i * STRING_LENGTH], STRING_LENGTH) == 0) {
        local_rank_++;
      }
    }
    local_device_ = local_rank_ % static_cast<int>(device_count_);
  } else {
    local_device_ = -1;
    for (size_t i=0; i<ranks.size(); i++) {
      if (ranks.at(i) == rank_) {
        local_device_ = devices.at(i);
        break;
      }
    }
  }

  if (local_device_ >= 0) {
    CUdevice dev;
    int busId, deviceId, domainId;
    NVE_CHECK_(cuDeviceGet(&dev, static_cast<int>(local_device_)));
    NVE_CHECK_(cuDeviceGetName(tmp_str, STRING_LENGTH, dev));
    NVE_CHECK_(cuDeviceGetAttribute(&domainId, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev));
    NVE_CHECK_(cuDeviceGetAttribute(&busId, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev));
    NVE_CHECK_(cuDeviceGetAttribute(&deviceId, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev));
    std::stringstream oss;
    oss << tmp_str << " (" <<
      std::hex << std::setw(8) << std::setfill('0') << domainId << ":" <<
      std::hex << std::setw(2) << std::setfill('0') << busId << ":" <<
      std::hex << std::setw(2) << std::setfill('0') << deviceId << ")" <<
      std::dec << std::setfill(' ') << std::setw(0);  // reset formatting
    device_name_ = oss.str();
  } else {
    device_name_ = "N/A";
  }

  // Wait for all to finish initialization
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
}

MPIEnv::~MPIEnv() {
}

void MPIEnv::barrier() {
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
}

void MPIEnv::broadcast(uintptr_t buffer, size_t size, int root) {
  MPI_CHECK(MPI_Bcast(reinterpret_cast<void*>(buffer), static_cast<int>(size), MPI_BYTE, root, MPI_COMM_WORLD));
}

void MPIEnv::all_gather(uintptr_t send_buffer, uintptr_t recv_buffer, size_t size) {
  MPI_CHECK(MPI_Allgather(
    reinterpret_cast<void*>(send_buffer), static_cast<int>(size), MPI_BYTE,
    reinterpret_cast<void*>(recv_buffer), static_cast<int>(size), MPI_BYTE,
    MPI_COMM_WORLD));
}

std::ostream& operator<<(std::ostream& os, const MPIEnv& env) {
  os << "MPIEnv" << std::endl
    << "  Rank: " << env.rank() << std::endl
    << "  World Size: " << env.world_size() << std::endl
    << "  Device Count: " << env.device_count() << std::endl
    << "  Local Device: " << env.local_device() << std::endl
    << "  Host Name: " << env.host_name() << std::endl
    << "  Device Name: " << env.device_name() << std::endl;
  return os;
}

} // namespace nve
