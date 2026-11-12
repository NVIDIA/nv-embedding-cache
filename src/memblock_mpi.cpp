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

#ifndef NVE_DISABLE_MPI

#include "include/mpi_utils.hpp"
#include "include/distributed.hpp"

namespace nve {

MPIMemBlock::MPIMemBlock(size_t row_size, size_t num_embeddings, nve::DataType_t dtype, const std::vector<size_t> ranks, const std::vector<int> devices)
  : MPIMemBlock(memblock_size_in_bytes(row_size, num_embeddings, dtype), ranks, devices) {}

MPIMemBlock::MPIMemBlock(size_t size_to_alloc, const std::vector<size_t> ranks, const std::vector<int> devices) : MemBlock(MemBlockType::MPI) {
  auto mpi_env = std::make_shared<nve::MPIEnv>(ranks, devices);
  mpi_buffer_ = std::make_shared<nve::CUDADistributedBuffer>(size_to_alloc, mpi_env, nve::BufferLocation::ALLOCATION_GPU_MEM);
  NVE_CHECK_(mpi_buffer_->ptr() != nullptr);
}

void* MPIMemBlock::get_ptr() const {
  return mpi_buffer_->ptr();
}

size_t MPIMemBlock::get_size_in_bytes() const {
  return mpi_buffer_->total_size();
}

}  // namespace nve

#else  // NVE_DISABLE_MPI

// MPI-less stub: same symbols, throwing constructors, no OpenMPI dependency.
namespace nve {

namespace {
constexpr char kNoMpi[] =
  "MPIMemBlock requires MPI support; this build was compiled with "
  "NVE_DISABLE_MPI=ON (no OpenMPI support).";
}

MPIMemBlock::MPIMemBlock(size_t, size_t, nve::DataType_t, const std::vector<size_t>, const std::vector<int>)
  : MemBlock(MemBlockType::MPI) {
  NVE_THROW_(kNoMpi);
}

MPIMemBlock::MPIMemBlock(size_t, const std::vector<size_t>, const std::vector<int>)
  : MemBlock(MemBlockType::MPI) {
  NVE_THROW_(kNoMpi);
}

void* MPIMemBlock::get_ptr() const {
  return nullptr;
}

size_t MPIMemBlock::get_size_in_bytes() const {
  return 0;
}

}  // namespace nve

#endif  // NVE_DISABLE_MPI
