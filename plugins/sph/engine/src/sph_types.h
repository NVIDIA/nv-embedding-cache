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
#include <cstdint>
#include <memory>
#include <type_traits>
#include "sph_check.h"
#include "sph_deduper.cuh"
#include "sph_resizeable_buffer.hpp"
#include "sph/sph_geometry.h"
#include <iostream>
#include <cuda.h>

namespace sph {

// Ceiling division. Local copy so the engine does not depend on bit_ops.hpp.
template<typename T>
__host__ __device__ constexpr T ceil_div(T x, T n) noexcept {
    static_assert(std::is_integral_v<T>);
    return (x + n - 1) / n;
}

// ---------------------------------------------------------------------------
// Payload copy element-type selection.
//
// The payload copy kernels (find / assign / fused insert / defrag) move
// value_size bytes per slot with a vectorized element type. Following
// call_cache_query_uvm (include/ecache/embed_cache.cuh): pick the widest
// element that evenly divides value_size so any row size is supported.
//   value_size % 16 == 0 -> uint4    (vec4, 16B)
//   value_size %  4 == 0 -> uint32_t (4B)
//   otherwise            -> int8_t   (1B)
// The chosen element also divides the payload base pointer (slot * value_size),
// so vectorized loads/stores stay naturally aligned.
//
// SPH_DISPATCH_COPY_T(value_size, T, body): introduces `using T = <elem>;` in a
// fresh scope and runs `body` (which references T) for the selected element.
// ---------------------------------------------------------------------------
#define SPH_DISPATCH_COPY_T(value_size, T, ...)                                    \
    do {                                                                           \
        if      ((value_size) % (int64_t)sizeof(uint4)    == 0) { using T = uint4;    __VA_ARGS__; } \
        else if ((value_size) % (int64_t)sizeof(uint32_t) == 0) { using T = uint32_t; __VA_ARGS__; } \
        else                                                    { using T = int8_t;   __VA_ARGS__; } \
    } while (0)

struct SPHV2PDE
{
    uint64_t shard_bit : shard_bit_bits;
    uint64_t page_index : page_index_bits;
    uint64_t pilot : pilot_bits;
    uint64_t bucket_offset : pte_offset_bits;
    uint64_t bucket_size : bucket_size_bits;
};

// V3 PTE: 8 bytes, fingerprint folded alongside page_offset.
struct alignas(8) SPHV3PTE
{
    uint64_t page_offset : 8;                       // 0xff = empty slot
    uint64_t fingerprint : pte_fingerprint_bits;    // = key / num_buckets (global_seed_=1)
};
static_assert(sizeof(SPHV3PTE) == 8, "SPHV3PTE must be 8 bytes");

// Failure counter indices. FC_PILOT_FAILURE..FC_PAGE_INDEX_OVERFLOW are one
// bump per BUCKET that hit that mode; FC_DROPPED_KEYS accumulates the number of
// batch KEYS lost across all of them. Both are needed: the bucket counts say
// which limit was hit, the key count says how much data it cost, and a single
// bucket can carry anywhere from one key to thousands.
//
// Only keys from the current batch are counted as dropped. A bucket that fails
// keeps its previously committed PDE, so keys already in the table survive.
//
// FC_EXISTING_KEYS / FC_BATCH_DUPLICATE_KEYS are NOT failures. They ride in the
// same array only because it is already zeroed and copied back once per insert.
// They count keys the existing-key filter removed from the batch on purpose, and
// must stay out of InsertFailures::total() or every filtered insert would throw.
enum FailureCounter : int {
    FC_PILOT_FAILURE = 0,
    FC_MAX_BUCKET_REACHED = 1,
    FC_BUCKET_OFFSET_OVERFLOW = 2,
    FC_PAGE_INDEX_OVERFLOW = 3,
    FC_DROPPED_KEYS = 4,
    FC_EXISTING_KEYS = 5,
    FC_BATCH_DUPLICATE_KEYS = 6,
    FC_COUNT = 7,
};

// ----------------------------------------------------------------------------
// ScratchPool: per-call aux-buffer allocator over one ResizeableBuffer.
// ----------------------------------------------------------------------------
class ScratchPool
{
public:
    explicit ScratchPool(bool pinned)
        : buffer_(std::make_shared<nve::sph_support::ResizeableBuffer>(pinned)) {}

    void Begin(size_t reserve_bytes)
    {
        base_ = static_cast<char*>(buffer_->get_ptr(reserve_bytes));
        cursor_ = 0;
        capacity_ = reserve_bytes;
    }

    template <typename T>
    T* Alloc(size_t count, size_t align = 256)
    {
        size_t bytes = count * sizeof(T);
        size_t start = (cursor_ + align - 1) & ~(align - 1);
        cursor_ = start + bytes;
        // Checked in release too: overrunning the pool silently corrupts the
        // heap, and the reserve sizes are computed by hand at each call site.
        SPH_CHECK(cursor_ <= capacity_,
                  "ScratchPool: Begin(reserve) was too small, need ", cursor_,
                  " bytes but reserved ", capacity_);
        return reinterpret_cast<T*>(base_ + start);
    }

private:
    std::shared_ptr<nve::sph_support::ResizeableBuffer> buffer_;
    char* base_ = nullptr;
    size_t cursor_ = 0;
    size_t capacity_ = 0;
};

enum class VmmLocation {
    DEVICE,
    HOST
};

// Sharded VMM buffer: VA is num_shards * pages_per_shard * page_size * slot_size.
class ShardedVmmBuffer {
public:
    ShardedVmmBuffer(VmmLocation location, int64_t num_shards, int64_t pages_per_shard,
                     int64_t page_size, int64_t slot_size, int device_id = 0)
        : location_(location), device_id_(device_id), num_shards_(num_shards),
          pages_per_shard_(pages_per_shard), page_size_(page_size), slot_size_(slot_size),
          va_ptr_(0)
    {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.id = device_id_;
        prop.location.type = (location_ == VmmLocation::DEVICE) ? CU_MEM_LOCATION_TYPE_DEVICE : CU_MEM_LOCATION_TYPE_HOST;
        SPH_CHECK(cuMemGetAllocationGranularity(&granularity_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

        SPH_CHECK(num_shards_ > 0, "ShardedVmmBuffer: num_shards must be > 0, got ", num_shards_);
        SPH_CHECK(pages_per_shard_ > 0, "ShardedVmmBuffer: pages_per_shard must be > 0, got ",
                  pages_per_shard_);
        SPH_CHECK(page_size_ > 0, "ShardedVmmBuffer: page_size must be > 0, got ", page_size_);
        SPH_CHECK(slot_size_ >= 0, "ShardedVmmBuffer: slot_size must be >= 0, got ", slot_size_);

        shard_va_size_ = align_up(pages_per_shard_ * page_size_ * slot_size_);
        va_size_ = shard_va_size_ * num_shards_;

        SPH_CHECK(cuMemAddressReserve(&va_ptr_, va_size_, granularity_, 0, 0),
                  "reserving ", va_size_, " bytes of VA for ", num_shards_, " shards");

        shard_committed_pages_.resize(num_shards_, 0);
        shard_handles_.resize(num_shards_);
    }

    ~ShardedVmmBuffer() {
        for (int64_t s = 0; s < num_shards_; s++) {
            unmap_shard(s);
        }
        if (va_ptr_) {
            SPH_TEARDOWN(cuMemAddressFree(va_ptr_, va_size_), "releasing ", va_size_, " bytes of VA");
        }
    }

    void grow_shard(int64_t shard_idx, int64_t num_pages) {
        SPH_CHECK(shard_idx >= 0 && shard_idx < num_shards_,
                  "grow_shard: shard_idx ", shard_idx, " out of range [0, ", num_shards_, ")");
        SPH_CHECK(num_pages <= pages_per_shard_,
                  "grow_shard: shard ", shard_idx, " asked for ", num_pages,
                  " pages, capacity is ", pages_per_shard_);
        if (num_pages <= shard_committed_pages_[shard_idx]) return;

        int64_t old_pages = shard_committed_pages_[shard_idx];
        size_t old_bytes = align_up(old_pages * page_size_ * slot_size_);
        size_t new_bytes = align_up(num_pages * page_size_ * slot_size_);
        size_t alloc_size = new_bytes - old_bytes;
        if (alloc_size == 0) {
            shard_committed_pages_[shard_idx] = num_pages;
            return;
        }

        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.id = device_id_;
        prop.location.type = (location_ == VmmLocation::DEVICE) ? CU_MEM_LOCATION_TYPE_DEVICE : CU_MEM_LOCATION_TYPE_HOST;

        CUmemGenericAllocationHandle handle;
        SPH_CHECK(cuMemCreate(&handle, alloc_size, &prop, 0),
                  "committing ", alloc_size, " bytes for shard ", shard_idx);

        // From here on the handle is owned by this call; hand it back to the
        // driver if a later step fails, or the throw leaks physical memory.
        const CUdeviceptr map_addr = va_ptr_ + shard_idx * shard_va_size_ + old_bytes;
        try {
            SPH_CHECK(cuMemMap(map_addr, alloc_size, 0, handle, 0),
                      "mapping ", alloc_size, " bytes at shard ", shard_idx);

            CUmemAccessDesc access = {};
            access.location.id = device_id_;
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            SPH_CHECK(cuMemSetAccess(map_addr, alloc_size, &access, 1),
                      "setting access on shard ", shard_idx);
        } catch (...) {
            cuMemUnmap(map_addr, alloc_size);  // no-op if the map itself failed
            cuMemRelease(handle);
            throw;
        }

        shard_handles_[shard_idx].push_back({handle, alloc_size});
        shard_committed_pages_[shard_idx] = num_pages;
    }

    void copy_committed_pages_to_device(int64_t* d_out, cudaStream_t stream) const {
        cudaMemcpyAsync(d_out, shard_committed_pages_.data(),
            num_shards_ * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    }

    void* ptr() const { return reinterpret_cast<void*>(va_ptr_); }
    void* shard_ptr(int64_t shard_idx) const {
        return reinterpret_cast<void*>(va_ptr_ + shard_idx * shard_va_size_);
    }
    int64_t committed_pages(int64_t shard_idx) const { return shard_committed_pages_[shard_idx]; }
    int64_t num_shards() const { return num_shards_; }

    // Byte distance between consecutive shard bases in the reserved VA. This is
    // align_up(pages_per_shard * page_size * slot_size) and generally EXCEEDS the
    // packed pages_per_shard*page_size*slot_size when that product is not
    // granularity-aligned (e.g. payload slot_size not a multiple of 8 with
    // page_size=255). Payload kernels must use this stride to locate a shard's
    // base rather than the packed product, or they index into uncommitted VA.
    int64_t shard_stride_bytes() const { return (int64_t)shard_va_size_; }

    size_t total_committed_bytes() const {
        size_t total = 0;
        for (int64_t s = 0; s < num_shards_; s++) {
            total += shard_committed_pages_[s] * page_size_ * slot_size_;
        }
        return total;
    }

    void release_shard(int64_t shard_idx) { unmap_shard(shard_idx); }

    void grow_total(size_t bytes) {
        if (num_shards_ == 0) return;
        size_t bytes_per_shard = ceil_div(bytes, static_cast<size_t>(num_shards_));
        size_t bytes_per_page = (size_t)page_size_ * (size_t)slot_size_;
        int64_t pages = (int64_t)ceil_div(bytes_per_shard, bytes_per_page);
        for (int64_t s = 0; s < num_shards_; s++) grow_shard(s, pages);
    }

private:
    struct HandleEntry {
        CUmemGenericAllocationHandle handle;
        size_t size;
    };

    size_t align_up(size_t size) const {
        return ceil_div(size, granularity_) * granularity_;
    }

    // Reached from the destructor (and from release_shard() during normal
    // operation), so it must not throw: failures are logged, not raised.
    void unmap_shard(int64_t shard_idx) {
        size_t committed_bytes = align_up(shard_committed_pages_[shard_idx] * page_size_ * slot_size_);
        if (committed_bytes > 0) {
            SPH_TEARDOWN(cuMemUnmap(va_ptr_ + shard_idx * shard_va_size_, committed_bytes),
                         "unmapping shard ", shard_idx);
        }
        for (const auto& entry : shard_handles_[shard_idx]) {
            SPH_TEARDOWN(cuMemRelease(entry.handle), "releasing ", entry.size, " bytes of shard ", shard_idx);
        }
        shard_handles_[shard_idx].clear();
        shard_committed_pages_[shard_idx] = 0;
    }

    VmmLocation location_;
    int device_id_;
    int64_t num_shards_;
    int64_t pages_per_shard_;
    int64_t page_size_;
    int64_t slot_size_;
    size_t granularity_;
    size_t va_size_;
    size_t shard_va_size_;
    CUdeviceptr va_ptr_;
    std::vector<int64_t> shard_committed_pages_;
    std::vector<std::vector<HandleEntry>> shard_handles_;
};

}  // namespace sph
