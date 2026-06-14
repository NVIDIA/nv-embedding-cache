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

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <buffer_wrapper.hpp>
#include <execution_context.hpp>
#include <host_table.hpp>

using namespace nve;
using namespace nlohmann::literals;

namespace {

// A minimal fixture that owns an always-available umap host table just to source an ExecutionContext
// for BufferWrapper construction. BufferWrapper's behavior is independent of the table type.
class BufferWrapperTest : public ::testing::Test {
 protected:
  void SetUp() override {
    host_table_factory_ptr_t fac{create_host_table_factory(R"({"implementation": "umap"})"_json)};
    table_ = fac->produce(1, R"({})"_json);
    ctx_ = table_->create_execution_context(0, 0, nullptr, nullptr);
  }

  host_table_ptr_t table_;
  context_ptr_t ctx_;
};

}  // namespace

// ---- BufferWrapper ----

TEST_F(BufferWrapperTest, ConstructFromUnregisteredBuffer) {
  std::vector<int32_t> data(8, 42);
  BufferWrapper<int32_t> wrap(ctx_, "buf", data.data(), data.size() * sizeof(int32_t));
  EXPECT_EQ(wrap.get_last_access(), cudaMemoryTypeUnregistered);
  EXPECT_EQ(wrap.get_buffer(cudaMemoryTypeUnregistered), data.data());
  EXPECT_EQ(wrap.get_buffer(cudaMemoryTypeHost), nullptr);
  EXPECT_EQ(wrap.get_buffer(cudaMemoryTypeDevice), nullptr);
}

TEST_F(BufferWrapperTest, ConstructWithNullptrThrows) {
  EXPECT_THROW((BufferWrapper<int32_t>(ctx_, "buf", nullptr, 16)), Exception);
}

TEST_F(BufferWrapperTest, AccessSameTypeReturnsOriginalPointer) {
  std::vector<int32_t> data(8, 7);
  BufferWrapper<int32_t> wrap(ctx_, "buf", data.data(), data.size() * sizeof(int32_t));
  auto* p = wrap.access_buffer(cudaMemoryTypeUnregistered, false, ctx_->get_lookup_stream());
  EXPECT_EQ(p, data.data());
  EXPECT_EQ(wrap.get_last_access(), cudaMemoryTypeUnregistered);
}

TEST_F(BufferWrapperTest, AccessNewTypeCopiesContent) {
  std::vector<int32_t> data{1, 2, 3, 4, 5, 6, 7, 8};
  const size_t bytes = data.size() * sizeof(int32_t);
  BufferWrapper<int32_t> wrap(ctx_, "buf", data.data(), bytes);

  auto* host_ptr = wrap.access_buffer(cudaMemoryTypeHost, true, ctx_->get_lookup_stream());
  ASSERT_NE(host_ptr, nullptr);
  EXPECT_NE(host_ptr, data.data());  // distinct allocation
  EXPECT_EQ(wrap.get_last_access(), cudaMemoryTypeHost);
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(host_ptr[i], data[i]);
  }
}

TEST_F(BufferWrapperTest, AccessNewTypeWithoutCopyDoesNotCopy) {
  std::vector<int32_t> data(8, 99);
  const size_t bytes = data.size() * sizeof(int32_t);
  BufferWrapper<int32_t> wrap(ctx_, "buf", data.data(), bytes);

  auto* host_ptr = wrap.access_buffer(cudaMemoryTypeHost, false, ctx_->get_lookup_stream());
  ASSERT_NE(host_ptr, nullptr);
  EXPECT_NE(host_ptr, data.data());
  // get_buffer must report the same allocation now that it exists.
  EXPECT_EQ(wrap.get_buffer(cudaMemoryTypeHost), host_ptr);
}

TEST_F(BufferWrapperTest, AccessManagedThrows) {
  std::vector<int32_t> data(4, 1);
  BufferWrapper<int32_t> wrap(ctx_, "buf", data.data(), data.size() * sizeof(int32_t));
  EXPECT_THROW(wrap.access_buffer(cudaMemoryTypeManaged, false, ctx_->get_lookup_stream()),
               Exception);
}

TEST_F(BufferWrapperTest, GetBufferReturnsNullForMissingType) {
  std::vector<int32_t> data(4, 1);
  BufferWrapper<int32_t> wrap(ctx_, "buf", data.data(), data.size() * sizeof(int32_t));
  EXPECT_EQ(wrap.get_buffer(cudaMemoryTypeHost), nullptr);
  EXPECT_EQ(wrap.get_buffer(cudaMemoryTypeDevice), nullptr);
  EXPECT_EQ(wrap.get_buffer(cudaMemoryTypeManaged), nullptr);
}

TEST_F(BufferWrapperTest, ConstructFromPinnedHostBuffer) {
  void* pinned = nullptr;
  constexpr size_t n = 8;
  constexpr size_t bytes = n * sizeof(int32_t);
  ASSERT_EQ(cudaSuccess, cudaMallocHost(&pinned, bytes));
  auto* p = static_cast<int32_t*>(pinned);
  std::fill(p, p + n, 5);
  {
    BufferWrapper<int32_t> wrap(ctx_, "buf", p, bytes);
    EXPECT_EQ(wrap.get_last_access(), cudaMemoryTypeHost);
    EXPECT_EQ(wrap.get_buffer(cudaMemoryTypeHost), p);
    EXPECT_EQ(wrap.get_buffer(cudaMemoryTypeUnregistered), nullptr);
    EXPECT_EQ(wrap.get_buffer(cudaMemoryTypeDevice), nullptr);
  }
  cudaFreeHost(pinned);
}

TEST_F(BufferWrapperTest, ConstructFromDeviceBuffer) {
  void* dev = nullptr;
  constexpr size_t bytes = 8 * sizeof(int32_t);
  ASSERT_EQ(cudaSuccess, cudaMalloc(&dev, bytes));
  {
    BufferWrapper<int32_t> wrap(ctx_, "buf", static_cast<int32_t*>(dev), bytes);
    EXPECT_EQ(wrap.get_last_access(), cudaMemoryTypeDevice);
    EXPECT_EQ(wrap.get_buffer(cudaMemoryTypeDevice), dev);
    EXPECT_EQ(wrap.get_buffer(cudaMemoryTypeHost), nullptr);
    EXPECT_EQ(wrap.get_buffer(cudaMemoryTypeUnregistered), nullptr);
  }
  cudaFree(dev);
}

TEST_F(BufferWrapperTest, ConstructFromManagedBuffer) {
  void* managed = nullptr;
  constexpr size_t n = 8;
  constexpr size_t bytes = n * sizeof(int32_t);
  ASSERT_EQ(cudaSuccess, cudaMallocManaged(&managed, bytes));
  auto* p = static_cast<int32_t*>(managed);
  std::fill(p, p + n, 13);
  {
    BufferWrapper<int32_t> wrap(ctx_, "buf", p, bytes);
    EXPECT_EQ(wrap.get_last_access(), cudaMemoryTypeManaged);
    EXPECT_EQ(wrap.get_buffer(cudaMemoryTypeManaged), p);
    // Accessing the managed type itself is a no-op that just returns the stored pointer.
    auto* same = wrap.access_buffer(cudaMemoryTypeManaged, false, ctx_->get_lookup_stream());
    EXPECT_EQ(same, p);
  }
  cudaFree(managed);
}

// Unregistered access first allocates a pinned host buffer and aliases it under the unregistered key.
TEST_F(BufferWrapperTest, UnregisteredAccessAliasesHostAllocation) {
  std::vector<int32_t> data{1, 2, 3, 4};
  const size_t bytes = data.size() * sizeof(int32_t);
  // Start from a device buffer so unregistered is a *new* type that needs allocation.
  void* dev = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&dev, bytes));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(dev, data.data(), bytes, cudaMemcpyHostToDevice));
  {
    BufferWrapper<int32_t> wrap(ctx_, "buf", static_cast<int32_t*>(dev), bytes);
    auto* unreg_ptr = wrap.access_buffer(cudaMemoryTypeUnregistered, true, ctx_->get_lookup_stream());
    ASSERT_NE(unreg_ptr, nullptr);
    EXPECT_EQ(wrap.get_last_access(), cudaMemoryTypeUnregistered);
    // Host allocation should have been performed and aliased.
    EXPECT_EQ(wrap.get_buffer(cudaMemoryTypeHost), unreg_ptr);
    // Content copied back from device.
    for (size_t i = 0; i < data.size(); ++i) {
      EXPECT_EQ(unreg_ptr[i], data[i]);
    }
  }
  cudaFree(dev);
}

// Round-trip: host → device → host preserves content. Verifies the device-side allocation
// and copy paths in both directions.
TEST_F(BufferWrapperTest, RoundTripHostDeviceHostPreservesContent) {
  std::vector<int32_t> data{11, 22, 33, 44, 55, 66, 77, 88};
  const size_t bytes = data.size() * sizeof(int32_t);
  BufferWrapper<int32_t> wrap(ctx_, "buf", data.data(), bytes);

  auto stream = ctx_->get_lookup_stream();
  auto* dev_ptr = wrap.access_buffer(cudaMemoryTypeDevice, true, stream);
  ASSERT_NE(dev_ptr, nullptr);
  EXPECT_EQ(wrap.get_last_access(), cudaMemoryTypeDevice);

  // Mutate the device copy, then read back into a host allocation.
  std::vector<int32_t> overwrite(data.size(), -1);
  ASSERT_EQ(cudaSuccess,
            cudaMemcpyAsync(dev_ptr, overwrite.data(), bytes, cudaMemcpyHostToDevice, stream));
  ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

  auto* host_ptr = wrap.access_buffer(cudaMemoryTypeHost, true, stream);
  ASSERT_NE(host_ptr, nullptr);
  EXPECT_EQ(wrap.get_last_access(), cudaMemoryTypeHost);
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(host_ptr[i], -1);
  }
}

// Repeated access to a type that has already been allocated must not reallocate or recopy.
TEST_F(BufferWrapperTest, RepeatedAccessReturnsCachedAllocation) {
  std::vector<int32_t> data(4, 7);
  const size_t bytes = data.size() * sizeof(int32_t);
  BufferWrapper<int32_t> wrap(ctx_, "buf", data.data(), bytes);

  auto* first = wrap.access_buffer(cudaMemoryTypeHost, true, ctx_->get_lookup_stream());
  auto* second = wrap.access_buffer(cudaMemoryTypeHost, false, ctx_->get_lookup_stream());
  EXPECT_EQ(first, second);
}

// Switching back to the original type returns the original pointer and updates last_access.
TEST_F(BufferWrapperTest, SwitchBackToOriginalType) {
  std::vector<int32_t> data(4, 7);
  const size_t bytes = data.size() * sizeof(int32_t);
  BufferWrapper<int32_t> wrap(ctx_, "buf", data.data(), bytes);

  (void)wrap.access_buffer(cudaMemoryTypeHost, false, ctx_->get_lookup_stream());
  EXPECT_EQ(wrap.get_last_access(), cudaMemoryTypeHost);

  auto* back = wrap.access_buffer(cudaMemoryTypeUnregistered, false, ctx_->get_lookup_stream());
  EXPECT_EQ(back, data.data());
  EXPECT_EQ(wrap.get_last_access(), cudaMemoryTypeUnregistered);
}

// A const-typed wrapper requires copy_content=true on first access to a new type.
TEST_F(BufferWrapperTest, ConstWrapperRequiresCopyOnFirstAccess) {
  std::vector<int32_t> data(4, 9);
  const size_t bytes = data.size() * sizeof(int32_t);
  BufferWrapper<const int32_t> wrap(ctx_, "buf", data.data(), bytes);
  EXPECT_THROW(wrap.access_buffer(cudaMemoryTypeHost, false, ctx_->get_lookup_stream()),
               Exception);
  // With copy_content=true it succeeds.
  auto* p = wrap.access_buffer(cudaMemoryTypeHost, true, ctx_->get_lookup_stream());
  ASSERT_NE(p, nullptr);
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(p[i], data[i]);
  }
}
