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

#include <stdexcept>
#include <iostream>
#include <sstream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include "cuda_ops/cuda_common.h"
#include <unistd.h>

#include "mpi_utils.hpp"
#include <gtest/gtest.h>
#include <sys/utsname.h>

// Functions to check if the linux kernel version is advanced enough to support pidfd_getfd and pidfd_open
static bool IsKernelGE(int major, int minor) {
  // Get system info
  struct utsname system_info;
  if (uname(&system_info) == -1) {
    std::cerr << "uname() failed" << std::endl;
    return false;
  }

  // Parse kernel version
  std::string ver_str(system_info.release);
  size_t first_dot = ver_str.find(".", 0);
  size_t second_dot = ver_str.find(".", first_dot + 1);
  if (first_dot == std::string::npos || second_dot == std::string::npos) {
    std::cerr << "Kernel version parse failed: \"" << ver_str << "\"" << std::endl;
    return false;
  }
  int kernel_major = std::stoi(ver_str.substr(0,first_dot));
  int kernel_minor = std::stoi(ver_str.substr(first_dot + 1,second_dot));

  return (kernel_major > major) || ((kernel_major == major) && (kernel_minor >= minor));
}

static bool IsKernelSupportsPidfd() {
  // Kernel version must be 5.6 or higher to support both pidfd_open and pidfd_getfd
  static const bool kernel_ge_5_6 = IsKernelGE(5, 6);
  return kernel_ge_5_6;
}

// Fixture for MPI tests
struct TestCase {
  size_t table_size;
};
class MPIBufferTest : public ::testing::TestWithParam<TestCase> {};

TEST(MPI, Init) {
  auto mpi_env = std::make_shared<nve::MPIEnv>();
}

TEST_P(MPIBufferTest, InitBuffer) {
  if (!IsKernelSupportsPidfd()) {
    GTEST_SKIP() << "Skipping test using pidfd (kernel version too low)";
    return;
  }
  const auto tc = GetParam();
  auto mpi_env = std::make_shared<nve::MPIEnv>();
  auto shared_buf = std::make_shared<nve::CUDADistributedBuffer>(tc.table_size, mpi_env);
}

void TestDistBuffer(size_t table_size, std::shared_ptr<nve::MPIEnv> mpi_env) {
  auto shared_buf = std::make_shared<nve::CUDADistributedBuffer>(table_size, mpi_env);

  // Write to local shard
  const auto rank = mpi_env->rank();
  NVE_CHECK_(cudaSetDevice(static_cast<int>(std::max(mpi_env->local_device(), 0))));
  const auto table_ptr = shared_buf->ptr();
  const auto shard_size = shared_buf->shard_size();
  const auto num_shards = shared_buf->num_shards();
  if (rank < num_shards) {
    NVE_CHECK_(cudaMemset(table_ptr + rank * shard_size, static_cast<int>(rank), shard_size));
    NVE_CHECK_(cudaDeviceSynchronize());
  }
  mpi_env->barrier(); // wait for all writes

  // Read from every shard and verify value
  std::vector<uint8_t> read_data(num_shards);
  for (size_t i=0 ; i<num_shards ; i++) {
    NVE_CHECK_(cudaMemcpy(&(read_data[i]), table_ptr + i * shard_size, 1, cudaMemcpyDefault));
  }
  NVE_CHECK_(cudaDeviceSynchronize());
  for (size_t i=0 ; i<num_shards ; i++) {
    EXPECT_EQ(read_data[i], i);
  }
}

TEST_P(MPIBufferTest, ReadWrite) {
  if (!IsKernelSupportsPidfd()) {
    GTEST_SKIP() << "Skipping test using pidfd (kernel version too low)";
    return;
  }
  const auto tc = GetParam();
  auto mpi_env = std::make_shared<nve::MPIEnv>();
  TestDistBuffer(tc.table_size, mpi_env);
}

TEST_P(MPIBufferTest, ReadWrite_PartialGroup) {
  if (!IsKernelSupportsPidfd()) {
    GTEST_SKIP() << "Skipping test using pidfd (kernel version too low)";
    return;
  }
  auto mpi_env = std::make_shared<nve::MPIEnv>();

  std::vector<size_t> ranks;
  std::vector<int> devices;
  const auto world_size = mpi_env->world_size();
  auto num_devices = mpi_env->device_count();
  for (size_t i = 0; i < world_size; i++) {
    if (i%2 == 0) {
      ranks.push_back(i);
      devices.push_back(static_cast<int>(i % num_devices));
    }
  }
  auto mpi_env_partial = std::make_shared<nve::MPIEnv>(ranks, devices);

  const auto tc = GetParam();
  TestDistBuffer(tc.table_size, mpi_env_partial);
}

INSTANTIATE_TEST_SUITE_P(
  ShardedBuffer,
  MPIBufferTest,
  ::testing::Values(
    TestCase({size_t(1) << 10}),
    TestCase({size_t(1) << 20}),
    TestCase({size_t(1) << 30})
  )
);