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
#include <cuda.h>
#include <cuda_runtime.h>
#include "include/common.hpp"
#include "include/allocator.hpp"
#include "include/nve_types.hpp"
#include <memory>
#include <vector>

namespace nve {

enum class MemBlockType {
    LINEAR,
    NVL,
    MPI,
    MANAGED,
    USER,
};

class MemBlock {
public:
    MemBlock(MemBlockType type) : type_(type) {}
    MemBlockType get_type() const { return type_; }
    virtual void* get_ptr() const = 0;
    virtual ~MemBlock() = default;
    uint64_t get_handle() const { return reinterpret_cast<uint64_t>(get_ptr()); }
protected:
    MemBlockType type_;
};

class LinearMemBlock : public MemBlock {
public:
    LinearMemBlock(size_t row_size, size_t num_embeddings, nve::DataType_t dtype);
    LinearMemBlock(size_t size_to_alloc);
    ~LinearMemBlock();
    void* get_ptr() const override;

private:
    void* ptr_;
    allocator_ptr_t allocator_;
};

class ManagedMemBlock : public MemBlock {
public:
    ManagedMemBlock(size_t row_size, size_t num_embeddings, nve::DataType_t dtype, const std::vector<int>& gpu_ids);
    ManagedMemBlock(size_t size_to_alloc, const std::vector<int>& gpu_ids);
    ~ManagedMemBlock();
    void* get_ptr() const override;

private:
    void* ptr_;
};

class NVLMemBlock : public MemBlock {
public:
    NVLMemBlock(size_t row_size, size_t num_embeddings, nve::DataType_t dtype, const std::vector<int>& gpu_ids);
    NVLMemBlock(size_t size_to_alloc, const std::vector<int>& gpu_ids);
    ~NVLMemBlock();
    void* get_ptr() const override;

private:
    CUdeviceptr ptr_;
    std::vector<CUmemGenericAllocationHandle> handles_;
    size_t total_bytes_;
};

class CUDADistributedBuffer;
class MPIMemBlock : public MemBlock {
public:
    MPIMemBlock(size_t row_size, size_t num_embeddings, nve::DataType_t dtype, const std::vector<size_t> ranks, const std::vector<int> devices);
    MPIMemBlock(size_t size_to_alloc, const std::vector<size_t> ranks, const std::vector<int> devices);
    ~MPIMemBlock() = default;
    void* get_ptr() const override;

private:
    std::shared_ptr<CUDADistributedBuffer> mpi_buffer_;
};

class DistributedEnv;
class DistMemBlock : public MemBlock {
public:
    DistMemBlock(std::shared_ptr<DistributedEnv> env, size_t row_size, size_t num_embeddings, nve::DataType_t dtype);
    ~DistMemBlock() = default;
    void* get_ptr() const override;

private:
    std::shared_ptr<CUDADistributedBuffer> dist_buffer_;
};


class DistHostMemBlock : public MemBlock {
public:
    DistHostMemBlock(std::shared_ptr<DistributedEnv> env, size_t row_size, size_t num_embeddings, nve::DataType_t dtype);
    ~DistHostMemBlock() = default;
    void* get_ptr() const override;

private:
    std::shared_ptr<CUDADistributedBuffer> dist_buffer_;
};

// Memblock to allow application to provide a raw ptr that's GPU accessible
class UserMemBlock : public MemBlock {
public:
    UserMemBlock(uint64_t ptr);
    ~UserMemBlock() = default;
    void* get_ptr() const override;

private:
    void* ptr_;
};
} // namespace nve
