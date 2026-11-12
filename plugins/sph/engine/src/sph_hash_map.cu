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

#include "sph/sph_hash_map.h"
#include "sph_common.cuh"
#include "sph_bucketize.cuh"
#include "sph_insert_kernels.cuh"
#include "sph_fused_insert.cuh"
#include "sph_find_build.cuh"
#include "sph_defrag.cuh"
#include "sph_check.h"
#include "sph_deduper.cuh"
#include "sph_resizeable_buffer.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace sph {

using HASH_TYPE = uint64_t;


// ============================================================================
// ctor / dtor / reset
// ============================================================================
ShardedPerfectHashMap::ShardedPerfectHashMap(int64_t initial_capacity, int64_t value_size, const int8_t* default_value)
    : value_size_(value_size), global_seed_(1)
{
    // reset() applies the min_num_buckets floor; only the sign is rejected here.
    SPH_CHECK(initial_capacity > 0,
              "ShardedPerfectHashMap: initial_capacity must be > 0, got ", initial_capacity);
    SPH_CHECK(value_size > 0, "ShardedPerfectHashMap: value_size must be > 0, got ", value_size);

    if (value_size_ > 0) {
        SPH_CHECK(cudaMalloc(&d_default_value_, value_size_));
        if (default_value) {
            SPH_CHECK(cudaMemcpyAsync(d_default_value_, default_value, value_size_,
                                       cudaMemcpyDefault, 0));
        } else {
            SPH_CHECK(cudaMemsetAsync(d_default_value_, 0, value_size_, 0));
        }
    }
    SPH_CHECK(cudaStreamCreateWithFlags(&aux_stream_, cudaStreamNonBlocking));
    SPH_CHECK(cudaEventCreateWithFlags(&alloc_done_event_, cudaEventDisableTiming));
    SPH_CHECK(cudaMalloc(&d_failure_counters_, FC_COUNT * sizeof(int64_t)));
    SPH_CHECK(cudaMallocHost(&h_failure_counters_, FC_COUNT * sizeof(int64_t)));

    d_pdes_                             = std::make_shared<nve::sph_support::ResizeableBuffer>(false);
    d_shard_page_counter_               = std::make_shared<nve::sph_support::ResizeableBuffer>(false);
    d_shard_page_offset_counter_        = std::make_shared<nve::sph_support::ResizeableBuffer>(false);
    d_shard_page_offset_global_counter_ = std::make_shared<nve::sph_support::ResizeableBuffer>(false);
    d_shard_locks_                      = std::make_shared<nve::sph_support::ResizeableBuffer>(false);
    d_bucket_offsets_                   = std::make_shared<nve::sph_support::ResizeableBuffer>(false);
    d_global_shard_bit_                 = std::make_shared<nve::sph_support::ResizeableBuffer>(false);
    deduper_d_tmp_device_mem_           = std::make_shared<nve::sph_support::ResizeableBuffer>(false);
    deduper_h_tmp_host_mem_             = std::make_shared<nve::sph_support::ResizeableBuffer>(true);
    deduper_ = std::make_shared<SphDeduper<KeyT>>();

    SPH_CHECK(cudaMallocHost(&h_num_active_buckets_, sizeof(int64_t)));

    reset(0, initial_capacity);
    SPH_CHECK(cudaStreamSynchronize(0));
}

ShardedPerfectHashMap::~ShardedPerfectHashMap() {
    if (aux_stream_)         SPH_TEARDOWN(cudaStreamDestroy(aux_stream_));
    if (alloc_done_event_)   SPH_TEARDOWN(cudaEventDestroy(alloc_done_event_));
    if (d_failure_counters_) SPH_TEARDOWN(cudaFree(d_failure_counters_));
    if (h_failure_counters_) SPH_TEARDOWN(cudaFreeHost(h_failure_counters_));
    if (d_default_value_)    SPH_TEARDOWN(cudaFree(d_default_value_));
}

// Out-of-line accessors: these dereference d_value_payload_ / d_bucket_offsets_,
// whose full types (ShardedVmmBuffer / ResizeableBuffer) live in sph_types.h and
// are only complete in this translation unit, not in the public header.
int64_t ShardedPerfectHashMap::get_total_payload_slots() const {
    int64_t total = 0;
    const int64_t n = num_shards_ * num_mini_shards;
    for (int64_t s = 0; s < n; s++) total += d_value_payload_->committed_pages(s) * page_size_;
    return total;
}

// The only accessor that reads DEVICE state, via a blocking copy on the default
// stream. See the freshness contract on the declaration in sph_hash_map.h: it
// reports the table as of the last COMPLETED modify, so it must not overlap one.
// The copy is checked — swallowing its status reported a partial or untouched
// h_vals as a real PTE count, which then showed up as a plausible-looking
// bits/key number rather than an error.
int64_t ShardedPerfectHashMap::get_num_ptes() const {
    const int64_t n = num_shards_ * num_mini_shards;
    std::vector<int64_t> h_vals(n, 0);
    // reset() already sized this buffer to exactly n entries, so get_ptr() only
    // hands back the existing pointer and never grows anything here.
    SPH_CHECK(cudaMemcpy(h_vals.data(),
                         d_bucket_offsets_->get_ptr(n * sizeof(int64_t)),
                         n * sizeof(int64_t), cudaMemcpyDeviceToHost),
              "get_num_ptes: reading ", n, " per-shard PTE counters");
    int64_t total = 0;
    for (int64_t v : h_vals) total += v;
    return total;
}

const int8_t* ShardedPerfectHashMap::get_value_payload_ptr() const {
    return (const int8_t*)d_value_payload_->ptr();
}

void ShardedPerfectHashMap::reset(cudaStream_t stream, int64_t capacity) {
    SPH_CHECK(capacity > 0, "reset: capacity must be > 0, got ", capacity);


    // Floor the geometry at min_num_buckets. This is a correctness bound from
    // the PTE layout, not a sizing heuristic: the fingerprint is
    // key / num_buckets stored in pte_fingerprint_bits bits, so with fewer
    // buckets the quotient truncates and find()/hash() compare against a
    // wrapped value while defrag's reconstruct_key_from_pte() rebuilds the
    // wrong key. Capacity is a target rather than a hard limit (size_
    // is allowed to exceed it) and the floor costs 32 KB, so round small
    // requests up instead of rejecting them; get_target_capacity() reports the value
    // actually used.
    target_capacity_ = std::max<int64_t>(capacity, min_num_buckets * avg_bucket_size_);
    size_ = 0;
    num_buckets_ = target_capacity_ / avg_bucket_size_;
    auto num_keys_per_shard = num_pages_per_shard_ * page_size_;
    num_shards_ = ceil_div(target_capacity_, num_keys_per_shard);  // mega-shard count
    const int64_t num_total_shards = num_shards_ * num_mini_shards;

    d_value_payload_ = std::make_shared<ShardedVmmBuffer>(
        VmmLocation::DEVICE, num_total_shards, num_pages_per_shard_, page_size_, value_size_);
    // d_ptes_ built DIRECTLY as the SPHV3PTE 8-byte-slot layout.
    d_ptes_ = std::make_shared<ShardedVmmBuffer>(
        VmmLocation::DEVICE, num_total_shards, pte_v3_pages_per_shard, pte_v3_page_slots, sizeof(SPHV3PTE));

    // Commit one page of each backing store per shard. The PTE page matters for
    // correctness, not just for warm-up: the lookup kernels form a PTE address from
    // the PDE, and while they now skip invalid PDEs (so an all-zero table is safe),
    // any shard left with zero committed PTE pages is a fault waiting for the first
    // bucket that becomes valid there. Insert's stage-3 grow already floors every
    // shard at one page, so this only moves that cost to reset() rather than adding
    // to the steady-state footprint.
    for (int64_t s = 0; s < num_total_shards; s++) {
        d_value_payload_->grow_shard(s, 1);
        d_ptes_->grow_shard(s, 1);
    }

    d_pdes_->get_ptr(num_buckets_ * sizeof(SPHV2PDE));

    deduper_->get_alloc_requirements(target_capacity_, tmp_mem_size_device_, tmp_mem_size_host_);
    char* d_tmp_device_mem = (char*)deduper_d_tmp_device_mem_->get_ptr(tmp_mem_size_device_);
    char* h_tmp_host_mem   = (char*)deduper_h_tmp_host_mem_->get_ptr(tmp_mem_size_host_);
    deduper_->set_and_init_buffers(target_capacity_, d_tmp_device_mem, h_tmp_host_mem);

    auto* p_pdes = d_pdes_->get_ptr(num_buckets_ * sizeof(SPHV2PDE));
    auto* p_shard_page_counter = d_shard_page_counter_->get_ptr(num_total_shards * sizeof(int64_t));
    auto* p_shard_page_offset  = d_shard_page_offset_counter_->get_ptr(num_total_shards * num_pages_per_shard_ * sizeof(int64_t));
    auto* p_shard_page_offset_global = d_shard_page_offset_global_counter_->get_ptr(num_total_shards * sizeof(int64_t));
    auto* p_shard_locks   = d_shard_locks_->get_ptr(num_total_shards * sizeof(int64_t));
    auto* p_bucket_offsets = d_bucket_offsets_->get_ptr(num_total_shards * sizeof(int64_t));
    auto* p_global_shard_bit = d_global_shard_bit_->get_ptr(num_shards_ * sizeof(int32_t));
    h_global_shard_bit_.assign((size_t)num_shards_, 0);

    SPH_CHECK(cudaMemsetAsync(p_pdes, 0, d_pdes_->get_size(), stream));
    SPH_CHECK(cudaMemsetAsync(p_shard_page_counter, 0, d_shard_page_counter_->get_size(), stream));
    SPH_CHECK(cudaMemsetAsync(p_shard_page_offset_global, 0, d_shard_page_offset_global_counter_->get_size(), stream));
    SPH_CHECK(cudaMemsetAsync(p_shard_page_offset, 0, d_shard_page_offset_counter_->get_size(), stream));
    SPH_CHECK(cudaMemsetAsync(p_shard_locks, 0, d_shard_locks_->get_size(), stream));
    SPH_CHECK(cudaMemsetAsync(p_bucket_offsets, 0, d_bucket_offsets_->get_size(), stream));
    SPH_CHECK(cudaMemsetAsync(p_global_shard_bit, 0, num_shards_ * sizeof(int32_t), stream));

    // 0xff == empty page_offset, matching what every other PTE-commit site writes.
    for (int64_t s = 0; s < num_total_shards; s++) {
        SPH_CHECK(cudaMemsetAsync(d_ptes_->shard_ptr(s), 0xff, pte_v3_chunk_bytes, stream));
    }
}

void ShardedPerfectHashMap::ensure_aux_pools() {
    if (!aux_dev_)  aux_dev_  = std::make_unique<ScratchPool>(/*pinned=*/false);
    if (!aux_host_) aux_host_ = std::make_unique<ScratchPool>(/*pinned=*/true);
}

void ShardedPerfectHashMap::bucketize(const int64_t* d_indices, int64_t num_indices, int64_t num_buckets,
    int64_t* d_active_buckets, int64_t* d_bucket_sizes,
    int64_t* h_num_active_buckets, int64_t* d_inverse_buffer, int64_t* d_offsets,
    int64_t* d_bucket_indices, int64_t global_seed, cudaStream_t stream)
{
    {
        call_calculate_bucket_index(d_indices, num_indices, d_bucket_indices, num_buckets, global_seed, stream);
        SPH_CHECK(cudaStreamSynchronize(stream));
    }
    {
        size_t tmp_mem_size_device, tmp_mem_size_host;
        deduper_->get_alloc_requirements(num_indices, tmp_mem_size_device, tmp_mem_size_host);
        char* tmp_host_mem   = (char*)deduper_h_tmp_host_mem_->get_ptr(tmp_mem_size_host);
        char* tmp_device_mem = (char*)deduper_d_tmp_device_mem_->get_ptr(tmp_mem_size_device);
        if (num_indices > target_capacity_) {
            deduper_->set_and_init_buffers(num_indices, tmp_device_mem, tmp_host_mem);
        }
        int sort_end_bit = 0;
        for (int64_t v = num_buckets - 1; v > 0; v >>= 1) sort_end_bit++;
        if (sort_end_bit == 0) sort_end_bit = 1;
        deduper_->dedup(d_bucket_indices, num_indices, d_active_buckets, d_bucket_sizes,
            nullptr, h_num_active_buckets, d_inverse_buffer, d_offsets, stream, sort_end_bit);
        SPH_CHECK(cudaStreamSynchronize(stream));
    }
}

// ============================================================================
// fused run — pre-pass (init_check_pilot_alpha, allocate_page) + host VMM grow
// + fused kernel + PDE commit post-pass.
// ============================================================================
template<typename HASH_TYPE, int64_t TARGET_EXPONENT, int64_t ALPHA_BUCKET_SIZE_FACTOR,
         int64_t NUM_PAGES_PER_SHARD, int64_t PAGE_SIZE, int64_t MAX_BUCKET_SIZE>
static void run_v3b_phases(
    int64_t num_buckets, int64_t num_total_buckets,
    const int64_t* d_keys, const int8_t* d_values, const int8_t* d_default_value,
    const int64_t* d_active_buckets, const int64_t* d_bucket_sizes,
    const int64_t* d_inverse_buffer, const int64_t* d_offsets,
    int64_t num_shards,
    SPHV2PDE* d_pdes,
    int64_t* d_shard_page_offsets,
    int64_t* d_shard_page_offset_global_counter,
    int64_t* d_bucket_offsets,
    BucketStateV2G* d_bucket_state,
    int64_t* h_bucket_offsets, int64_t* h_shard_page_counter,
    std::shared_ptr<ShardedVmmBuffer>& d_ptes_buf,
    std::shared_ptr<ShardedVmmBuffer>& d_value_payload_buf,
    cudaStream_t stream, cudaStream_t aux_stream, cudaEvent_t alloc_done_event,
    int64_t value_size,
    int64_t* failure_counters, int32_t* global_shard_bit, int64_t* pilot_counters)
{
    // ---- Stage 1: init_check_pilot_alpha ----
    call_kernel_v2h_init_check_pilot_alpha<HASH_TYPE, TARGET_EXPONENT, ALPHA_BUCKET_SIZE_FACTOR,
                                           NUM_PAGES_PER_SHARD, PAGE_SIZE, MAX_BUCKET_SIZE, SPHV3PTE>(
        num_buckets, num_shards,
        d_keys, d_active_buckets, d_bucket_sizes, d_offsets,
        d_inverse_buffer, d_pdes, (SPHV3PTE*)d_ptes_buf->ptr(), d_bucket_offsets,
        d_shard_page_offsets, d_shard_page_offset_global_counter,
        global_shard_bit, d_bucket_state, failure_counters, stream);

    // ---- Stage 2: allocate_page ----
    call_kernel_v2g_allocate_page<NUM_PAGES_PER_SHARD, PAGE_SIZE>(
        num_buckets, num_shards,
        d_active_buckets, d_bucket_sizes,
        d_shard_page_offset_global_counter, d_shard_page_offsets,
        d_bucket_state, failure_counters, stream);

    // ---- Stage 3: VMM grow on host ----
    // Always run: grow_shard is a no-op when the shard already has enough
    // committed pages. reserve()'s pre-commit is only an estimate; skewed
    // distributions / alpha growth / large batches can exceed it, so we must
    // reconcile against the actual per-shard page counters every insert or the
    // fused kernel writes to uncommitted pages and faults.
    {
        SPH_CHECK(cudaEventRecord(alloc_done_event, stream));
        SPH_CHECK(cudaStreamWaitEvent(aux_stream, alloc_done_event, 0));
        SPH_CHECK(cudaMemcpyAsync(h_bucket_offsets, d_bucket_offsets,
                                   num_shards * num_mini_shards * sizeof(int64_t),
                                   cudaMemcpyDeviceToHost, aux_stream));
        SPH_CHECK(cudaMemcpyAsync(h_shard_page_counter, d_shard_page_offset_global_counter,
                                   num_shards * num_mini_shards * sizeof(int64_t),
                                   cudaMemcpyDeviceToHost, aux_stream));
        SPH_CHECK(cudaStreamSynchronize(aux_stream));
        const int64_t num_total_shards = num_shards * num_mini_shards;
        for (int64_t s = 0; s < num_total_shards; s++) {
            int64_t pages_used = ceil_div(h_shard_page_counter[s], PAGE_SIZE) + 1;
            d_value_payload_buf->grow_shard(s, pages_used);
        }
        for (int64_t s = 0; s < num_total_shards; s++) {
            int64_t needed_ptes = h_bucket_offsets[s];
            int64_t needed_pages = ceil_div(needed_ptes, pte_v3_page_slots) + 1;
            if (needed_pages > d_ptes_buf->committed_pages(s)) {
                int64_t old_pages = d_ptes_buf->committed_pages(s);
                d_ptes_buf->grow_shard(s, needed_pages);
                int64_t new_pages = d_ptes_buf->committed_pages(s);
                if (new_pages > old_pages) {
                    char* shard_base = static_cast<char*>(d_ptes_buf->shard_ptr(s));
                    size_t old_bytes = old_pages * pte_v3_chunk_bytes;
                    size_t add_bytes = (new_pages - old_pages) * pte_v3_chunk_bytes;
                    SPH_CHECK(cudaMemsetAsync(shard_base + old_bytes, 0xff, add_bytes, stream));
                }
            }
        }
    }

    // ---- Stage 4: fused kernel ----
    SPHV3PTE* d_ptes          = (SPHV3PTE*)d_ptes_buf->ptr();
    int8_t*   d_value_payload = (int8_t*)d_value_payload_buf->ptr();
    const int64_t payload_shard_stride_bytes = d_value_payload_buf->shard_stride_bytes();
    call_kernel_v3b_fused<HASH_TYPE, NUM_PAGES_PER_SHARD, PAGE_SIZE, MAX_BUCKET_SIZE, SPHV3PTE>(
        num_buckets, num_total_buckets, num_shards,
        d_keys, d_values, d_default_value, d_active_buckets, d_bucket_sizes, d_offsets,
        d_inverse_buffer, d_pdes, d_ptes, d_value_payload, value_size,
        payload_shard_stride_bytes,
        d_bucket_state, pilot_counters, failure_counters, stream);

    // ---- Stage 5: PDE commit (post-pass) ----
    call_kernel_v2h_commit_pde(num_buckets, d_active_buckets, d_bucket_state, d_pdes, stream);
}

// ============================================================================
// insert
// ============================================================================
void ShardedPerfectHashMap::insert(const int64_t* d_keys, const int8_t* d_values,
                        int64_t num_keys, cudaStream_t stream,
                        OnExisting on_existing)
{
    SPH_CHECK(num_keys >= 0, "insert: num_keys must be >= 0, got ", num_keys);
    SPH_CHECK(num_keys == 0 || d_keys != nullptr, "insert: d_keys is null for ", num_keys, " keys");
    // Matches find/hash/assign, which already no-op on an empty batch. Without
    // this the whole pipeline ran on zero keys and still bumped size_.
    if (num_keys == 0) return;


    int64_t num_indices = num_keys;
    const int64_t num_buckets = target_capacity_ / avg_bucket_size_;

    ensure_aux_pools();

    constexpr size_t kSlack = 1024;
    const size_t dev_total =
          5 * ((num_indices * sizeof(int64_t) + 255) & ~size_t{255})
        +     ((num_buckets * sizeof(BucketStateV2G) + 255) & ~size_t{255})
        + kSlack;
    aux_dev_->Begin(dev_total);

    int64_t*        d_active_buckets = aux_dev_->Alloc<int64_t>(num_indices);
    int64_t*        d_bucket_sizes   = aux_dev_->Alloc<int64_t>(num_indices);
    int64_t*        d_inverse_buffer = aux_dev_->Alloc<int64_t>(num_indices);
    int64_t*        d_offsets        = aux_dev_->Alloc<int64_t>(num_indices);
    int64_t*        d_bucket_indices = aux_dev_->Alloc<int64_t>(num_indices);
    BucketStateV2G* d_bucket_state   = aux_dev_->Alloc<BucketStateV2G>(num_buckets);

    const int64_t num_total_shards = num_shards_ * num_mini_shards;
    const size_t host_total =
          ((num_total_shards * sizeof(int64_t) + 255) & ~size_t{255})
        + ((num_total_shards * sizeof(int64_t) + 255) & ~size_t{255})
        + kSlack;
    aux_host_->Begin(host_total);
    int64_t* h_bucket_offsets     = aux_host_->Alloc<int64_t>(num_total_shards);
    int64_t* h_shard_page_counter = aux_host_->Alloc<int64_t>(num_total_shards);

    SPHV2PDE* d_pdes = (SPHV2PDE*)d_pdes_->get_ptr(num_buckets * sizeof(SPHV2PDE));
    int64_t* d_shard_page_offset_counter = (int64_t*)d_shard_page_offset_counter_->get_ptr(num_total_shards * num_pages_per_shard_ * sizeof(int64_t));
    int64_t* d_bucket_offsets = (int64_t*)d_bucket_offsets_->get_ptr(num_total_shards * sizeof(int64_t));
    int64_t* d_shard_page_offset_global_counter = (int64_t*)d_shard_page_offset_global_counter_->get_ptr(num_total_shards * sizeof(int64_t));

    bucketize(d_keys, num_indices, num_buckets,
        d_active_buckets, d_bucket_sizes, h_num_active_buckets_,
        d_inverse_buffer, d_offsets, d_bucket_indices, global_seed_, stream);

    // Wire always-on failure counters + fresh-bucket steering.
    active_pilot_counters_   = nullptr;
    active_failure_counters_ = d_failure_counters_;
    SPH_CHECK(cudaMemsetAsync(active_failure_counters_, 0, FC_COUNT * sizeof(int64_t), stream));
    active_global_shard_bit_ = (int32_t*)d_global_shard_bit_->get_ptr(num_shards_ * sizeof(int32_t));

    constexpr int64_t max_bucket_size = max_alpha_bucket_size;
    const int64_t num_total_buckets = target_capacity_ / avg_bucket_size_;

    // Strip keys that are already resident, and repeats within this batch, before
    // the state machine sizes alpha or reserves payload slots. Shortens the
    // per-bucket runs in place; see sph_filter_existing.cuh. Skipped entirely
    // under Fail, so that path keeps its old behaviour and its old cost.
    if (on_existing != OnExisting::ON_EXISTING_FAIL) {
        call_kernel_v3_filter_existing<HASH_TYPE, page_size_, SPHV3PTE>(
            *h_num_active_buckets_, num_total_buckets, num_shards_,
            d_keys, d_values, d_default_value_, d_active_buckets, d_bucket_sizes, d_offsets,
            d_inverse_buffer, d_pdes, (const SPHV3PTE*)d_ptes_->ptr(),
            (int8_t*)d_value_payload_->ptr(),
            value_size_, d_value_payload_->shard_stride_bytes(),
            active_failure_counters_, on_existing == OnExisting::ON_EXISTING_UPDATE, stream);
    }

    run_v3b_phases<HASH_TYPE, target_exponent_, alpha_bucket_size_factor,
                   num_pages_per_shard_, page_size_, max_bucket_size>(
        *h_num_active_buckets_, num_total_buckets,
        d_keys, d_values, d_default_value_,
        d_active_buckets, d_bucket_sizes, d_inverse_buffer, d_offsets,
        num_shards_, d_pdes, d_shard_page_offset_counter,
        d_shard_page_offset_global_counter, d_bucket_offsets, d_bucket_state,
        h_bucket_offsets, h_shard_page_counter,
        d_ptes_, d_value_payload_,
        stream, aux_stream_, alloc_done_event_, value_size_,
        active_failure_counters_, active_global_shard_bit_, active_pilot_counters_);

    // Values are handled inside the fused kernel; no separate assign().

    SPH_CHECK(cudaMemcpyAsync(h_failure_counters_, d_failure_counters_,
                               FC_COUNT * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    SPH_CHECK(cudaStreamSynchronize(stream));

    InsertResult result;
    result.failures = InsertFailures{
        h_failure_counters_[FC_PILOT_FAILURE],
        h_failure_counters_[FC_MAX_BUCKET_REACHED],
        h_failure_counters_[FC_BUCKET_OFFSET_OVERFLOW],
        h_failure_counters_[FC_PAGE_INDEX_OVERFLOW]};
    result.keys_submitted         = num_keys;
    result.keys_existing          = h_failure_counters_[FC_EXISTING_KEYS];
    result.keys_batch_duplicates  = h_failure_counters_[FC_BATCH_DUPLICATE_KEYS];
    // Clamped defensively: a bucket can appear in more than one failure mode in
    // principle, and an over-count here would drive size_ backwards.
    //
    // The bound is the POST-FILTER batch, not num_keys: the filter shortens each
    // bucket's run before the state machine runs, so FC_DROPPED_KEYS can only
    // ever count keys that survived filtering. Clamping at num_keys would let a
    // double-count push keys_inserted() negative once filtering removed anything.
    // Read the two filter counts first so this bound is available.
    const int64_t droppable = std::max<int64_t>(
        0, num_keys - result.keys_existing - result.keys_batch_duplicates);
    result.keys_dropped = std::min<int64_t>(h_failure_counters_[FC_DROPPED_KEYS], droppable);

    // Only now, and only for the keys that actually landed. Bumping this on
    // entry meant a throw or a partial insert left it permanently overstated,
    // which silently skews get_fragmentation() and the defrag threshold.
    size_ += result.keys_inserted();

    // Throws SphInsertFailed with the full breakdown; see the InsertResult
    // overloads of sph_is_success/sph_fail in sph_hash_map.h.
    SPH_CHECK(result);
}

// ============================================================================
// find
// ============================================================================
void ShardedPerfectHashMap::find(const int64_t* keys, int64_t num_keys, int8_t* values, cudaStream_t stream,
                      uint64_t* hit_mask, int64_t value_stride)
{
    SPH_CHECK(num_keys >= 0, "find: num_keys must be >= 0, got ", num_keys);
    if (num_keys <= 0) return;
    SPH_CHECK(keys != nullptr,   "find: keys is null for ", num_keys, " keys");
    SPH_CHECK(values != nullptr, "find: values is null for ", num_keys, " keys");
    SPH_CHECK(value_stride <= 0 || value_stride >= value_size_,
              "find: value_stride ", value_stride, " is smaller than value_size ", value_size_);
    if (value_stride <= 0) value_stride = value_size_;
    constexpr int32_t KEYS_PER_THREAD = 4;
    const int64_t num_buckets = target_capacity_ / avg_bucket_size_;

    SPHV2PDE* d_pdes = (SPHV2PDE*)d_pdes_->get_ptr(0);
    SPHV3PTE* d_ptes = (SPHV3PTE*)d_ptes_->ptr();
    int8_t*   d_value_payload = (int8_t*)d_value_payload_->ptr();

    call_gather_keys_sph_v3<HASH_TYPE, KEYS_PER_THREAD>(
        keys, num_keys, d_pdes, d_ptes, num_buckets, num_shards_,
        global_seed_, num_pages_per_shard_, page_size_, d_value_payload,
        value_size_, d_value_payload_->shard_stride_bytes(), d_default_value_,
        values, value_stride, hit_mask, stream);
}

// ============================================================================
// find_locations (contains proxy)
// ============================================================================
void ShardedPerfectHashMap::find_locations(const int64_t* d_keys, int64_t num_keys, int64_t* d_locations, cudaStream_t stream)
{
    SPH_CHECK(num_keys >= 0, "find_locations: num_keys must be >= 0, got ", num_keys);
    if (num_keys <= 0) return;
    SPH_CHECK(d_keys != nullptr, "find_locations: d_keys is null for ", num_keys, " keys");
    SPH_CHECK(d_locations != nullptr,
              "find_locations: d_locations is null for ", num_keys, " keys");
    const int64_t num_buckets = target_capacity_ / avg_bucket_size_;
    SPHV2PDE* d_pdes = (SPHV2PDE*)d_pdes_->get_ptr(0);
    SPHV3PTE* d_ptes = (SPHV3PTE*)d_ptes_->ptr();

    call_hash_sph_v3<HASH_TYPE, page_size_>(
        d_keys, num_keys, d_pdes, d_ptes, num_buckets, num_shards_, global_seed_,
        d_value_payload_->shard_stride_bytes(), value_size_, d_locations, stream);
}

// ============================================================================
// assign
// ============================================================================
void ShardedPerfectHashMap::assign(const int64_t* keys, const int8_t* values, int64_t num_keys, cudaStream_t stream)
{
    SPH_CHECK(num_keys >= 0, "assign: num_keys must be >= 0, got ", num_keys);
    if (num_keys <= 0) return;
    SPH_CHECK(keys != nullptr,   "assign: keys is null for ", num_keys, " keys");
    SPH_CHECK(values != nullptr, "assign: values is null for ", num_keys, " keys");

    const int64_t num_buckets = target_capacity_ / avg_bucket_size_;

    SPHV2PDE* d_pdes          = (SPHV2PDE*)d_pdes_->get_ptr(0);
    SPHV3PTE* d_ptes          = (SPHV3PTE*)d_ptes_->ptr();
    int8_t*   d_value_payload = (int8_t*)d_value_payload_->ptr();

    constexpr int32_t KEYS_PER_THREAD = 4;
    call_assign_keys_sph<HASH_TYPE, KEYS_PER_THREAD>(
        keys, values, value_size_, num_keys, d_pdes, d_ptes, num_buckets, num_shards_,
        global_seed_, num_pages_per_shard_, page_size_, d_value_payload, value_size_,
        d_value_payload_->shard_stride_bytes(), stream);

}

// ============================================================================
// build
// ============================================================================
void ShardedPerfectHashMap::build(const int64_t* d_keys, int64_t num_keys, cudaStream_t stream, const int8_t* d_values)
{
    SPH_CHECK(num_keys >= 0, "build: num_keys must be >= 0, got ", num_keys);
    if (num_keys == 0) return;
    SPH_CHECK(d_keys != nullptr, "build: d_keys is null for ", num_keys, " keys");

    constexpr int64_t max_bucket_size = max_keys_per_bucket;
    const int64_t num_buckets = target_capacity_ / avg_bucket_size_;

    int64_t* d_buckets = nullptr;
    int64_t* d_bucket_sizes = nullptr;
    int64_t* d_page_offsets = nullptr;
    int64_t* h_bucket_offsets = nullptr;
    int64_t* h_shard_page_counter = nullptr;
    int64_t* h_shard_page_offset_global_counter = nullptr;
    {
        ensure_aux_pools();
        constexpr size_t kSlack = 1024;
        const size_t dev_total =
              ((num_buckets * max_bucket_size * sizeof(int64_t) + 255) & ~size_t{255})
            + ((num_buckets * sizeof(int64_t)                    + 255) & ~size_t{255})
            + ((num_buckets * sizeof(int64_t)                    + 255) & ~size_t{255})
            + kSlack;
        aux_dev_->Begin(dev_total);
        d_buckets       = aux_dev_->Alloc<int64_t>(num_buckets * max_bucket_size);
        d_bucket_sizes  = aux_dev_->Alloc<int64_t>(num_buckets);
        d_page_offsets  = aux_dev_->Alloc<int64_t>(num_buckets);

        const size_t host_total =
              ((num_shards_ * sizeof(int64_t) + 255) & ~size_t{255})
            + ((num_shards_ * sizeof(int64_t) + 255) & ~size_t{255})
            + ((num_shards_ * sizeof(int64_t) + 255) & ~size_t{255})
            + kSlack;
        aux_host_->Begin(host_total);
        h_bucket_offsets                   = aux_host_->Alloc<int64_t>(num_shards_);
        h_shard_page_counter               = aux_host_->Alloc<int64_t>(num_shards_);
        h_shard_page_offset_global_counter = aux_host_->Alloc<int64_t>(num_shards_);
    }

    int64_t*  d_shard_page_offset_global_counter = (int64_t*)d_shard_page_offset_global_counter_->get_ptr(num_shards_ * sizeof(int64_t));
    SPHV2PDE* d_pdes                             = (SPHV2PDE*)d_pdes_->get_ptr(num_buckets * sizeof(SPHV2PDE));
    int64_t*  d_bucket_offsets                   = (int64_t*)d_bucket_offsets_->get_ptr(num_shards_ * sizeof(int64_t));
    int64_t*  d_shard_page_counter               = (int64_t*)d_shard_page_counter_->get_ptr(num_shards_ * sizeof(int64_t));
    int64_t*  d_shard_page_offset_counter        = (int64_t*)d_shard_page_offset_counter_->get_ptr(num_shards_ * num_pages_per_shard_ * sizeof(int64_t));

    {
        SPH_CHECK(cudaMemsetAsync(d_failure_counters_, 0, FC_COUNT * sizeof(int64_t), stream));
        SPH_CHECK(cudaMemsetAsync(d_bucket_sizes, 0, num_buckets * sizeof(int64_t), stream));
        call_calculate_buckets_sph_for_build(d_keys, num_keys, num_buckets, max_bucket_size,
                                     global_seed_, d_buckets, d_bucket_sizes,
                                     d_failure_counters_, stream);
        SPH_CHECK(cudaStreamSynchronize(stream));
    }

    {
        cudaMemsetAsync(d_shard_page_offset_global_counter, 0, num_shards_ * sizeof(int64_t), stream);
        call_calculate_pilots_sph_for_build<num_pages_per_shard_, page_size_>(
            d_buckets, d_bucket_sizes, num_buckets, num_shards_, max_bucket_size,
            target_exponent_, d_pdes, d_page_offsets, d_bucket_offsets,
            d_shard_page_counter, d_shard_page_offset_counter,
            d_shard_page_offset_global_counter, d_failure_counters_, stream);
        SPH_CHECK(cudaMemcpyAsync(h_shard_page_counter, d_shard_page_counter, num_shards_ * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        SPH_CHECK(cudaMemcpyAsync(h_bucket_offsets, d_bucket_offsets, num_shards_ * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        SPH_CHECK(cudaMemcpyAsync(h_shard_page_offset_global_counter, d_shard_page_offset_global_counter, num_shards_ * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        SPH_CHECK(cudaStreamSynchronize(stream));
    }

    {
        // Fresh build: every bucket's shard_bit==0 → physical shard = mega*num_mini_shards.
        for (int64_t mega = 0; mega < num_shards_; mega++) {
            int64_t phys = mega * num_mini_shards;
            int64_t pages_used = (h_shard_page_offset_global_counter[mega] / page_size_) + 1;
            h_shard_page_counter[mega] = pages_used;
            // Unconditional, like insert's stage 3: the payload has to be backed
            // whether or not this call supplied values, because a later insert
            // will relocate these rows.
            d_value_payload_->grow_shard(phys, pages_used);

            int64_t needed_ptes = h_bucket_offsets[mega];
            int64_t needed_pages = ceil_div(needed_ptes, pte_v3_page_slots) + 1;
            int64_t old_pages = d_ptes_->committed_pages(phys);
            if (needed_pages > old_pages) {
                d_ptes_->grow_shard(phys, needed_pages);
                int64_t new_pages = d_ptes_->committed_pages(phys);
                size_t old_bytes = old_pages * pte_v3_chunk_bytes;
                size_t add_bytes = (new_pages - old_pages) * pte_v3_chunk_bytes;
                SPH_CHECK(cudaMemsetAsync((int8_t*)d_ptes_->shard_ptr(phys) + old_bytes, 0xff, add_bytes, stream));
            }
        }
        SPH_CHECK(cudaMemcpyAsync(d_shard_page_counter, h_shard_page_counter, num_shards_ * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    }

    // Remap the MEGA-indexed continuation counters into the PHYSICAL-shard
    // layout (phys = mega*num_mini_shards, shard_bit=0) a following insert reads.
    {
        const int64_t num_total = num_shards_ * num_mini_shards;
        std::vector<int64_t> bo(num_total, 0), pg(num_total, 0);
        for (int64_t mega = 0; mega < num_shards_; mega++) {
            bo[mega * num_mini_shards] = h_bucket_offsets[mega];
            pg[mega * num_mini_shards] = h_shard_page_offset_global_counter[mega];
        }
        int64_t* d_bo = (int64_t*)d_bucket_offsets_->get_ptr(num_total * sizeof(int64_t));
        int64_t* d_pg = (int64_t*)d_shard_page_offset_global_counter_->get_ptr(num_total * sizeof(int64_t));
        SPH_CHECK(cudaMemcpyAsync(d_bo, bo.data(), num_total * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        SPH_CHECK(cudaMemcpyAsync(d_pg, pg.data(), num_total * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        SPH_CHECK(cudaStreamSynchronize(stream));
    }

    SPHV3PTE* d_ptes = (SPHV3PTE*)d_ptes_->ptr();
    {
        call_build_ptes_sph_v3_for_build<HASH_TYPE, num_pages_per_shard_, page_size_>(
            num_buckets,
            d_buckets, d_bucket_sizes, d_page_offsets, d_pdes,
            num_shards_, max_bucket_size, d_ptes, stream);
        SPH_CHECK(cudaStreamSynchronize(stream));
    }

    // Always scatter: a build without values still has to initialize every row,
    // or the keys read back as whatever the freshly committed payload contained.
    // Null d_values broadcasts the default row (stride 0), the same contract the
    // insert path uses.
    {
        constexpr int32_t KEYS_PER_THREAD = 4;
        call_assign_keys_sph<HASH_TYPE, KEYS_PER_THREAD>(
            d_keys,
            d_values ? d_values : d_default_value_,
            d_values ? value_size_ : 0,
            num_keys, d_pdes, d_ptes, num_buckets, num_shards_,
            global_seed_, num_pages_per_shard_, page_size_,
            (int8_t*)d_value_payload_->ptr(), value_size_,
            d_value_payload_->shard_stride_bytes(), stream);
        SPH_CHECK(cudaStreamSynchronize(stream));
    }

    // Report dropped keys the way insert() does.
    SPH_CHECK(cudaMemcpyAsync(h_failure_counters_, d_failure_counters_,
                               FC_COUNT * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    SPH_CHECK(cudaStreamSynchronize(stream));

    InsertResult result;
    result.failures = InsertFailures{
        h_failure_counters_[FC_PILOT_FAILURE],
        h_failure_counters_[FC_MAX_BUCKET_REACHED],
        h_failure_counters_[FC_BUCKET_OFFSET_OVERFLOW],
        h_failure_counters_[FC_PAGE_INDEX_OVERFLOW]};
    result.keys_submitted = num_keys;
    // build() does no existing-key or in-batch duplicate filtering, so those stay
    // zero and keys_inserted() is just submitted - dropped.
    result.keys_dropped = std::min<int64_t>(h_failure_counters_[FC_DROPPED_KEYS], num_keys);

    // Only the keys that actually landed, and only after the work is done: a
    // throw below must not leave size_ overstating the table.
    size_ += result.keys_inserted();

    SPH_CHECK(result);
}

// ============================================================================
// reserve — pre-commit VMM for the active mini-shard of every mega.
// ============================================================================
void ShardedPerfectHashMap::reserve(int64_t num_keys, cudaStream_t stream)
{
    SPH_CHECK(num_shards_ > 0, "reserve: called on a map with no shards (capacity ", target_capacity_, ")");

    const int64_t per_shard_vals  = ceil_div(num_keys, num_shards_);
    const int64_t pay_pages       = std::min<int64_t>(
        ceil_div(per_shard_vals, page_size_), num_pages_per_shard_);
    const int64_t per_shard_ptes  = ceil_div(num_keys, num_shards_);
    const int64_t pte_pages       = std::min<int64_t>(
        ceil_div(per_shard_ptes, pte_v3_page_slots), pte_v3_pages_per_shard);

    for (int64_t mega = 0; mega < num_shards_; mega++) {
        const int64_t phys = mega * num_mini_shards;
        if (value_size_ > 0) d_value_payload_->grow_shard(phys, pay_pages);
        const int64_t old_pages = d_ptes_->committed_pages(phys);
        d_ptes_->grow_shard(phys, pte_pages);
        const int64_t new_pages = d_ptes_->committed_pages(phys);
        if (new_pages > old_pages) {
            char* shard_base = static_cast<char*>(d_ptes_->shard_ptr(phys));
            size_t old_bytes = old_pages * pte_v3_chunk_bytes;
            size_t add_bytes = (new_pages - old_pages) * pte_v3_chunk_bytes;
            SPH_CHECK(cudaMemsetAsync(shard_base + old_bytes, 0xff, add_bytes, stream));
        }
    }
    SPH_CHECK(cudaStreamSynchronize(stream));

}

// ============================================================================
// defrag (SPHV3PTE worker) + defrag_all
// ============================================================================
void ShardedPerfectHashMap::defrag(int32_t mega_shard, cudaStream_t stream)
{
    SPH_CHECK(mega_shard >= 0 && mega_shard < (int32_t)num_shards_,
              "defrag: mega_shard ", mega_shard, " out of range [0, ", num_shards_, ")");

    const int32_t num_shards     = (int32_t)num_shards_;
    const int32_t new_num_shards = num_shards;
    const int64_t num_buckets    = num_buckets_;
    const int64_t buckets_in_mega = ceil_div(num_buckets, static_cast<int64_t>(num_shards));

    const int32_t src_bit  = h_global_shard_bit_[mega_shard];
    const int32_t dst_bit  = src_bit ^ 1;
    const int32_t src_phys = mega_shard * (int32_t)num_mini_shards + src_bit;
    const int32_t dst_phys = mega_shard * (int32_t)num_mini_shards + dst_bit;

    ensure_aux_pools();
    constexpr size_t kSlack = 1024;
    const size_t dev_total =
          ((buckets_in_mega * sizeof(DefragInfo) + 255) & ~size_t{255})
        + ((sizeof(int32_t) + 255) & ~size_t{255})
        + kSlack;
    aux_dev_->Begin(dev_total);
    DefragInfo* d_defrag_info  = aux_dev_->Alloc<DefragInfo>(buckets_in_mega);
    int32_t*    d_active_count = aux_dev_->Alloc<int32_t>(1);

    int64_t* d_bucket_offsets = (int64_t*)d_bucket_offsets_->get_ptr(num_shards_ * num_mini_shards * sizeof(int64_t));
    int64_t* d_global_offset  = (int64_t*)d_shard_page_offset_global_counter_->get_ptr(num_shards_ * num_mini_shards * sizeof(int64_t));
    int64_t* d_shard_page_off = (int64_t*)d_shard_page_offset_counter_->get_ptr(num_shards_ * num_mini_shards * num_pages_per_shard_ * sizeof(int64_t));
    int64_t* d_shard_pages    = (int64_t*)d_shard_page_counter_->get_ptr(num_shards_ * num_mini_shards * sizeof(int64_t));
    int64_t* d_shard_locks    = (int64_t*)d_shard_locks_->get_ptr(num_shards_ * num_mini_shards * sizeof(int64_t));
    SPHV2PDE* d_pdes          = (SPHV2PDE*)d_pdes_->get_ptr(num_buckets_ * sizeof(SPHV2PDE));

    SPH_CHECK(cudaMemsetAsync(d_active_count, 0, sizeof(int32_t), stream));
    SPH_CHECK(cudaMemsetAsync(&d_bucket_offsets[dst_phys], 0, sizeof(int64_t), stream));
    SPH_CHECK(cudaMemsetAsync(&d_global_offset[dst_phys],  0, sizeof(int64_t), stream));
    // Zeroing the destination mini shard to allow for the publishing mechanisem
    SPH_CHECK(cudaMemsetAsync(&d_shard_page_off[dst_phys * num_pages_per_shard_], 0,
                               num_pages_per_shard_ * sizeof(int64_t), stream));

    {
        const SPHV3PTE* d_ptes_typed = (const SPHV3PTE*)d_ptes_->ptr();
        call_defrag_allocate<num_pages_per_shard_, page_size_>(
            buckets_in_mega,
            mega_shard, num_shards, new_num_shards, num_buckets,
            d_pdes, d_ptes_typed, d_global_offset, d_shard_page_off, d_bucket_offsets,
            d_defrag_info, d_active_count, stream);
    }

    int64_t h_dst_bucket_offset = 0;
    int64_t h_dst_global_offset = 0;
    int32_t h_active_count       = 0;
    {
        SPH_CHECK(cudaMemcpyAsync(&h_dst_bucket_offset, &d_bucket_offsets[dst_phys], sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        SPH_CHECK(cudaMemcpyAsync(&h_dst_global_offset, &d_global_offset[dst_phys],  sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        SPH_CHECK(cudaMemcpyAsync(&h_active_count,      d_active_count,                sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        SPH_CHECK(cudaStreamSynchronize(stream));

        if (h_active_count == 0) return;

        int64_t needed_pages = ceil_div(h_dst_bucket_offset, pte_v3_page_slots) + 1;
        int64_t old_pages = d_ptes_->committed_pages(dst_phys);
        if (needed_pages > old_pages) {
            d_ptes_->grow_shard(dst_phys, needed_pages);
            int64_t new_pages = d_ptes_->committed_pages(dst_phys);
            size_t old_bytes = old_pages * pte_v3_chunk_bytes;
            size_t add_bytes = (new_pages - old_pages) * pte_v3_chunk_bytes;
            SPH_CHECK(cudaMemsetAsync((char*)d_ptes_->shard_ptr(dst_phys) + old_bytes, 0xff, add_bytes, stream));
        }

        const int64_t pages_used = ceil_div(h_dst_global_offset, page_size_) + 1;
        d_value_payload_->grow_shard(dst_phys, pages_used);
    }

    {
        SPHV3PTE* d_ptes_typed    = (SPHV3PTE*)d_ptes_->ptr();
        int8_t*   d_value_payload = (int8_t*)d_value_payload_->ptr();
        // Keep value_size consistent with the payload buffer's shard stride
        // (both derived from value_size_); a 0-size payload copies 0 elements.
        int64_t   active_value_size = value_size_;
        call_defrag_commit<num_pages_per_shard_, page_size_>(
            h_active_count,
            num_shards, new_num_shards, active_value_size,
            d_value_payload_->shard_stride_bytes(),
            d_pdes, d_ptes_typed, d_value_payload,
            d_defrag_info, stream);
    }

    SPH_CHECK(cudaStreamSynchronize(stream));
    h_global_shard_bit_[mega_shard] = dst_bit;
    int32_t* d_global_shard_bit = (int32_t*)d_global_shard_bit_->get_ptr(num_shards_ * sizeof(int32_t));
    SPH_CHECK(cudaMemcpyAsync(&d_global_shard_bit[mega_shard],
                               &h_global_shard_bit_[mega_shard],
                               sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    d_ptes_->release_shard(src_phys);
    d_value_payload_->release_shard(src_phys);
    d_ptes_->grow_shard(src_phys, 1);
    d_value_payload_->grow_shard(src_phys, 1);
    SPH_CHECK(cudaMemsetAsync(d_ptes_->shard_ptr(src_phys), 0xff, pte_v3_chunk_bytes, stream));

    SPH_CHECK(cudaMemsetAsync(&d_bucket_offsets[src_phys], 0, sizeof(int64_t), stream));
    SPH_CHECK(cudaMemsetAsync(&d_global_offset[src_phys],  0, sizeof(int64_t), stream));
    SPH_CHECK(cudaMemsetAsync(&d_shard_pages[src_phys],    0, sizeof(int64_t), stream));
    SPH_CHECK(cudaMemsetAsync(&d_shard_locks[src_phys],    0, sizeof(int64_t), stream));
    SPH_CHECK(cudaMemsetAsync(&d_shard_page_off[src_phys * num_pages_per_shard_], 0,
                               num_pages_per_shard_ * sizeof(int64_t), stream));

}

void ShardedPerfectHashMap::defrag_all(cudaStream_t stream)
{
    for (int32_t m = 0; m < (int32_t)num_shards_; m++) defrag(m, stream);
}

}  // namespace sph
