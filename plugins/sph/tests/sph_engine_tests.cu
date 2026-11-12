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

// ============================================================================
// Functional tests for the standalone sph::ShardedPerfectHashMap engine.
//
// These drive the PUBLIC engine API directly (include/sph/sph_hash_map.h) with
// no dependency on the plugin layer:
//
//   insert(KeyT* keys, const int8_t* values, n, stream)
//   build (KeyT* keys, n, stream, const int8_t* values = nullptr)
//   find  (const int64_t* keys, n, int8_t* values_out, stream, uint64_t* mask)
//   hash  (KeyT* keys, n, int64_t* hashes_out, stream)
//   defrag_all(stream)   reset(stream, capacity)   reserve(stream)
//
// ShardedPerfectHashMap has no contains(); negative (miss-key) detection is done via
// find() into a sentinel-filled buffer — on a miss find() overwrites the row
// with the map's default value, so a surviving sentinel byte means the row was
// never written and a non-default byte means a false hit.
//
// Every test is parameterized over value_size. The sweep spans all three
// copy-element paths: multiples of 16 (uint4), of 4 but not 16 (uint32_t:
// 4, 12, 20), and odd sizes (int8_t: 1, 7, 33).
// ============================================================================
#include <gtest/gtest.h>

#include <sph/sph_hash_map.h>
#include <sph/error.h>
#include <sph_check.h>  // private support header: SPH_TEARDOWN under test

#include <cuda_runtime.h>

#include <algorithm>
#include <bit_ops.hpp>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using KeyT = sph::ShardedPerfectHashMap::KeyT;

::testing::AssertionResult CudaOk(cudaError_t err, const char* expr)
{
    if (err == cudaSuccess) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
           << expr << " failed: " << cudaGetErrorName(err) << " ("
           << cudaGetErrorString(err) << ")";
}

#define ASSERT_CUDA(expr) ASSERT_TRUE(CudaOk((expr), #expr))
#define EXPECT_CUDA(expr) EXPECT_TRUE(CudaOk((expr), #expr))

// Deterministic payload: a key/byte-index mix, so a mismatch is traceable back
// to the key that produced it.
void fill_reference(std::vector<KeyT>&         h_keys,
                    std::vector<std::uint8_t>& h_values,
                    std::int64_t               num_keys,
                    std::int64_t               value_size,
                    KeyT                       first_key = 1)
{
    h_keys.resize(static_cast<std::size_t>(num_keys));
    h_values.assign(static_cast<std::size_t>(num_keys * value_size), 0);
    for (std::int64_t i = 0; i < num_keys; ++i) {
        const KeyT key = first_key + static_cast<KeyT>(i);
        h_keys[static_cast<std::size_t>(i)] = key;
        for (std::int64_t b = 0; b < value_size; ++b) {
            h_values[static_cast<std::size_t>(i * value_size + b)] =
                static_cast<std::uint8_t>((key ^ b) & 0xff);
        }
    }
}

// Disjoint miss-keys, far from any inserted key range.
void fill_miss_keys(std::vector<KeyT>& h_miss, std::int64_t num_keys, KeyT first_key = 1)
{
    h_miss.resize(static_cast<std::size_t>(num_keys));
    const KeyT base = first_key + static_cast<KeyT>(num_keys) + 1'000'000'000;
    for (std::int64_t i = 0; i < num_keys; ++i)
        h_miss[static_cast<std::size_t>(i)] = base + static_cast<KeyT>(i);
}

// Compare a device find() result against the host reference, byte-for-byte.
// Returns the number of mismatched keys; sets first_bad to the first index.
std::int64_t count_mismatches(const std::vector<std::uint8_t>& got,
                              const std::vector<std::uint8_t>& ref,
                              std::int64_t                     num_keys,
                              std::int64_t                     value_size,
                              std::int64_t*                    first_bad = nullptr)
{
    std::int64_t bad = 0;
    if (first_bad) *first_bad = -1;
    for (std::int64_t i = 0; i < num_keys; ++i) {
        const auto off = static_cast<std::size_t>(i * value_size);
        if (std::memcmp(&got[off], &ref[off], static_cast<std::size_t>(value_size)) != 0) {
            if (bad == 0 && first_bad) *first_bad = i;
            ++bad;
        }
    }
    return bad;
}

// Count bytes that differ from the map's default row. find() overwrites the
// whole output row on a miss, so a missing key yields the default, not the
// caller's previous buffer contents.
std::int64_t count_non_default(const std::vector<std::uint8_t>& got,
                               const std::vector<std::uint8_t>& def,
                               std::int64_t                     value_size)
{
    std::int64_t bad = 0;
    for (std::size_t i = 0; i < got.size(); ++i)
        if (got[i] != def[i % static_cast<std::size_t>(value_size)]) ++bad;
    return bad;
}

// ---------------------------------------------------------------------------
// Fixture base shared by the engine and validation suites: one stream per test,
// device allocations tracked and freed wholesale in TearDown, so an ASSERT_*
// early return never leaks device memory.
// ---------------------------------------------------------------------------
template <typename GtestBase>
class SphDeviceFixture : public GtestBase {
protected:
    void SetUp() override
    {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
            GTEST_SKIP() << "CUDA device unavailable";
        ASSERT_CUDA(cudaStreamCreate(&stream_));
    }

    void TearDown() override
    {
        for (void* p : allocs_) EXPECT_CUDA(cudaFree(p));
        allocs_.clear();
        if (stream_) EXPECT_CUDA(cudaStreamDestroy(stream_));
    }

    // Returns nullptr on failure (and records a non-fatal failure); callers
    // gate on ASSERT_NE(ptr, nullptr).
    template <typename T>
    T* alloc(std::int64_t count)
    {
        void*             p   = nullptr;
        const cudaError_t err = cudaMalloc(&p, static_cast<std::size_t>(count) * sizeof(T));
        if (err != cudaSuccess) {
            ADD_FAILURE() << "cudaMalloc(" << count * static_cast<std::int64_t>(sizeof(T))
                          << " bytes) failed: " << cudaGetErrorName(err);
            return nullptr;
        }
        allocs_.push_back(p);
        return static_cast<T*>(p);
    }

    template <typename T>
    T* upload(const std::vector<T>& h)
    {
        T* d = alloc<T>(static_cast<std::int64_t>(h.size()));
        if (!d) return nullptr;
        const cudaError_t err =
            cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            ADD_FAILURE() << "cudaMemcpy H2D failed: " << cudaGetErrorName(err);
            return nullptr;
        }
        return d;
    }

    template <typename T>
    void download(std::vector<T>& h, const T* d, std::int64_t count)
    {
        h.resize(static_cast<std::size_t>(count));
        ASSERT_CUDA(cudaMemcpy(h.data(), d, static_cast<std::size_t>(count) * sizeof(T),
                               cudaMemcpyDeviceToHost));
    }

    cudaStream_t       stream_{};
    std::vector<void*> allocs_;
};

// The parameter is the row size in bytes (value_size).
class SphEngineTest : public SphDeviceFixture<::testing::TestWithParam<std::int64_t>> {
protected:
    std::int64_t value_size() const { return GetParam(); }

    // Shared bodies for the checks that are run at several capacities/scales.
    void RunInsertFindRoundtrip(std::int64_t capacity, std::int64_t num_keys);
    void RunBuildFindRoundtrip(std::int64_t capacity, std::int64_t num_keys);
};

// ---------------------------------------------------------------------------
// insert() -> find() round-trip.
//
// Insert a batch of known (key, value) pairs, verify find() returns them
// byte-for-byte, then verify disjoint keys miss (default row returned).
// ---------------------------------------------------------------------------
void SphEngineTest::RunInsertFindRoundtrip(std::int64_t capacity, std::int64_t num_keys)
{
    const std::int64_t vs = value_size();

    std::vector<KeyT>         h_keys, h_miss;
    std::vector<std::uint8_t> h_values;
    fill_reference(h_keys, h_values, num_keys, vs);
    fill_miss_keys(h_miss, num_keys);

    KeyT*         d_keys   = upload(h_keys);
    KeyT*         d_miss   = upload(h_miss);
    std::uint8_t* d_values = upload(h_values);
    std::uint8_t* d_found  = alloc<std::uint8_t>(num_keys * vs);
    ASSERT_NE(d_keys, nullptr);
    ASSERT_NE(d_miss, nullptr);
    ASSERT_NE(d_values, nullptr);
    ASSERT_NE(d_found, nullptr);

    auto m = std::make_unique<sph::ShardedPerfectHashMap>(capacity, vs);
    m->insert(d_keys, reinterpret_cast<const std::int8_t*>(d_values), num_keys, stream_);

    // Positive: every inserted key round-trips byte-for-byte.
    ASSERT_CUDA(cudaMemsetAsync(d_found, 0, static_cast<std::size_t>(num_keys * vs), stream_));
    m->find(d_keys, num_keys, reinterpret_cast<std::int8_t*>(d_found), stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    std::vector<std::uint8_t> h_found;
    ASSERT_NO_FATAL_FAILURE(download(h_found, d_found, num_keys * vs));

    std::int64_t first_bad = -1;
    EXPECT_EQ(count_mismatches(h_found, h_values, num_keys, vs, &first_bad), 0)
        << "first bad key=" << (first_bad >= 0 ? h_keys[static_cast<std::size_t>(first_bad)] : -1);

    // Negative: disjoint keys must miss -> find() overwrites the sentinel-filled
    // buffer with the default value (all-zero for this map).
    ASSERT_CUDA(cudaMemsetAsync(d_found, 0xCC, static_cast<std::size_t>(num_keys * vs), stream_));
    m->find(d_miss, num_keys, reinterpret_cast<std::int8_t*>(d_found), stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    ASSERT_NO_FATAL_FAILURE(download(h_found, d_found, num_keys * vs));

    const std::vector<std::uint8_t> h_zero_default(static_cast<std::size_t>(vs), 0);
    EXPECT_EQ(count_non_default(h_found, h_zero_default, vs), 0) << "miss-key leak bytes";
}

TEST_P(SphEngineTest, InsertFindRoundtrip1K) { RunInsertFindRoundtrip(1'024, 500); }
TEST_P(SphEngineTest, InsertFindRoundtrip100K) { RunInsertFindRoundtrip(100'000, 50'000); }
TEST_P(SphEngineTest, InsertFindRoundtrip2M) { RunInsertFindRoundtrip(2'000'000, 1'000'000); }
TEST_P(SphEngineTest, InsertFindRoundtrip20M) { RunInsertFindRoundtrip(20'000'000, 10'000'000); }

// ---------------------------------------------------------------------------
// build() -> find() round-trip.
//
// build() is the bulk single-shot construction path (distinct from the
// incremental fused insert). Verify positive round-trip + miss detection.
// ---------------------------------------------------------------------------
void SphEngineTest::RunBuildFindRoundtrip(std::int64_t capacity, std::int64_t num_keys)
{
    const std::int64_t vs = value_size();

    std::vector<KeyT>         h_keys, h_miss;
    std::vector<std::uint8_t> h_values;
    fill_reference(h_keys, h_values, num_keys, vs);
    fill_miss_keys(h_miss, num_keys);

    KeyT*         d_keys   = upload(h_keys);
    KeyT*         d_miss   = upload(h_miss);
    std::uint8_t* d_values = upload(h_values);
    std::uint8_t* d_found  = alloc<std::uint8_t>(num_keys * vs);
    ASSERT_NE(d_keys, nullptr);
    ASSERT_NE(d_miss, nullptr);
    ASSERT_NE(d_values, nullptr);
    ASSERT_NE(d_found, nullptr);

    auto m = std::make_unique<sph::ShardedPerfectHashMap>(capacity, vs);
    m->reserve(capacity, stream_);
    m->build(d_keys, num_keys, stream_, reinterpret_cast<const std::int8_t*>(d_values));

    ASSERT_CUDA(cudaMemsetAsync(d_found, 0, static_cast<std::size_t>(num_keys * vs), stream_));
    m->find(d_keys, num_keys, reinterpret_cast<std::int8_t*>(d_found), stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    std::vector<std::uint8_t> h_found;
    ASSERT_NO_FATAL_FAILURE(download(h_found, d_found, num_keys * vs));

    std::int64_t first_bad = -1;
    EXPECT_EQ(count_mismatches(h_found, h_values, num_keys, vs, &first_bad), 0)
        << "first bad key=" << (first_bad >= 0 ? h_keys[static_cast<std::size_t>(first_bad)] : -1)
        << " frag=" << m->get_fragmentation() << " bits/key=" << m->get_bits_per_key();

    ASSERT_CUDA(cudaMemsetAsync(d_found, 0xCC, static_cast<std::size_t>(num_keys * vs), stream_));
    m->find(d_miss, num_keys, reinterpret_cast<std::int8_t*>(d_found), stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    ASSERT_NO_FATAL_FAILURE(download(h_found, d_found, num_keys * vs));

    const std::vector<std::uint8_t> h_zero_default(static_cast<std::size_t>(vs), 0);
    EXPECT_EQ(count_non_default(h_found, h_zero_default, vs), 0) << "miss-key leak bytes";
}

TEST_P(SphEngineTest, BuildFindRoundtrip2M) { RunBuildFindRoundtrip(2'000'000, 1'000'000); }
TEST_P(SphEngineTest, BuildFindRoundtrip20M) { RunBuildFindRoundtrip(20'000'000, 10'000'000); }

// ---------------------------------------------------------------------------
// build() -> insert() handoff round-trip.
//
// Populate via build(), then incrementally insert() on top; verify every key
// (build + inserted) round-trips AND disjoint keys miss. Capacity is chosen
// above one shard's key span (num_pages_per_shard * page_size ~= 66.8M) so
// num_shards >= 2, exercising the per-shard continuation-counter handoff
// between build and insert — a mis-indexed handoff aliases committed ranges and
// drops keys / returns wrong-bucket values. Regression guard for that bug.
// ---------------------------------------------------------------------------
TEST_P(SphEngineTest, BuildThenInsertRoundtripMultiShard)
{
    const std::int64_t vs          = value_size();
    const std::int64_t capacity    = 80'000'000;
    const std::int64_t build_keys  = 8'000'000;
    const std::int64_t insert_keys = 1'000'000;
    const std::int64_t total       = build_keys + insert_keys;

    std::vector<KeyT>         h_keys, h_miss;
    std::vector<std::uint8_t> h_values;
    fill_reference(h_keys, h_values, total, vs);
    fill_miss_keys(h_miss, total);

    KeyT*         d_keys   = upload(h_keys);
    KeyT*         d_miss   = upload(h_miss);
    std::uint8_t* d_values = upload(h_values);
    std::uint8_t* d_found  = alloc<std::uint8_t>(total * vs);
    ASSERT_NE(d_keys, nullptr);
    ASSERT_NE(d_miss, nullptr);
    ASSERT_NE(d_values, nullptr);
    ASSERT_NE(d_found, nullptr);

    auto m = std::make_unique<sph::ShardedPerfectHashMap>(capacity, vs);
    m->reserve(capacity, stream_);
    m->build(d_keys, build_keys, stream_, reinterpret_cast<const std::int8_t*>(d_values));
    m->insert(d_keys + build_keys,
              reinterpret_cast<const std::int8_t*>(d_values + build_keys * vs), insert_keys,
              stream_);

    EXPECT_GE(m->get_num_shards(), 2) << "capacity should have produced a multi-shard map";

    ASSERT_CUDA(cudaMemsetAsync(d_found, 0, static_cast<std::size_t>(total * vs), stream_));
    m->find(d_keys, total, reinterpret_cast<std::int8_t*>(d_found), stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    std::vector<std::uint8_t> h_found;
    ASSERT_NO_FATAL_FAILURE(download(h_found, d_found, total * vs));

    std::int64_t first_bad = -1;
    EXPECT_EQ(count_mismatches(h_found, h_values, total, vs, &first_bad), 0)
        << "first bad key=" << (first_bad >= 0 ? h_keys[static_cast<std::size_t>(first_bad)] : -1)
        << " num_shards=" << m->get_num_shards();

    ASSERT_CUDA(cudaMemsetAsync(d_found, 0xCC, static_cast<std::size_t>(total * vs), stream_));
    m->find(d_miss, total, reinterpret_cast<std::int8_t*>(d_found), stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    ASSERT_NO_FATAL_FAILURE(download(h_found, d_found, total * vs));

    const std::vector<std::uint8_t> h_zero_default(static_cast<std::size_t>(vs), 0);
    EXPECT_EQ(count_non_default(h_found, h_zero_default, vs), 0) << "false-positive bytes";
}

// ---------------------------------------------------------------------------
// defrag correctness.
//
// Build a fragmented map via incremental insert batches, verify the pre-defrag
// round-trip (so a corrupt insert can't hide behind defrag), run defrag_all,
// then verify (a) fragmentation does not get worse, (b) find still returns the
// correct values byte-for-byte.
// ---------------------------------------------------------------------------
TEST_P(SphEngineTest, DefragCorrectness)
{
    const std::int64_t vs          = value_size();
    const std::int64_t capacity    = 20'000'000;
    const std::int64_t batch_size  = 1'000'000;
    const int          num_batches = 10;
    const std::int64_t total_keys  = batch_size * num_batches;

    std::vector<KeyT>         h_keys;
    std::vector<std::uint8_t> h_values;
    fill_reference(h_keys, h_values, total_keys, vs);

    KeyT*         d_keys   = upload(h_keys);
    std::uint8_t* d_values = upload(h_values);
    std::uint8_t* d_found  = alloc<std::uint8_t>(total_keys * vs);
    ASSERT_NE(d_keys, nullptr);
    ASSERT_NE(d_values, nullptr);
    ASSERT_NE(d_found, nullptr);

    auto m = std::make_unique<sph::ShardedPerfectHashMap>(capacity, vs);
    for (int b = 0; b < num_batches; ++b) {
        m->insert(d_keys + b * batch_size,
                  reinterpret_cast<const std::int8_t*>(d_values + b * batch_size * vs), batch_size,
                  stream_);
    }
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    const double pre_frag = m->get_fragmentation();

    // Pre-defrag round-trip: incremental inserts must be correct BEFORE defrag
    // relocates them (defrag only moves already-valid entries).
    ASSERT_CUDA(cudaMemsetAsync(d_found, 0, static_cast<std::size_t>(total_keys * vs), stream_));
    m->find(d_keys, total_keys, reinterpret_cast<std::int8_t*>(d_found), stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    std::vector<std::uint8_t> h_pre;
    ASSERT_NO_FATAL_FAILURE(download(h_pre, d_found, total_keys * vs));
    ASSERT_EQ(count_mismatches(h_pre, h_values, total_keys, vs), 0)
        << "incremental inserts were already wrong before defrag";

    m->defrag_all(stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    // Defrag must not make fragmentation worse (it should compact payload).
    const double post_frag = m->get_fragmentation();
    EXPECT_LE(post_frag, pre_frag + 1e-9) << "pre_frag=" << pre_frag << " post_frag=" << post_frag;

    ASSERT_CUDA(cudaMemsetAsync(d_found, 0, static_cast<std::size_t>(total_keys * vs), stream_));
    m->find(d_keys, total_keys, reinterpret_cast<std::int8_t*>(d_found), stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    std::vector<std::uint8_t> h_post;
    ASSERT_NO_FATAL_FAILURE(download(h_post, d_found, total_keys * vs));

    std::int64_t first_bad = -1;
    EXPECT_EQ(count_mismatches(h_post, h_values, total_keys, vs, &first_bad), 0)
        << "first bad key=" << (first_bad >= 0 ? h_keys[static_cast<std::size_t>(first_bad)] : -1)
        << " pre_frag=" << pre_frag << " post_frag=" << post_frag;
}

// ---------------------------------------------------------------------------
// insert after defrag.
//
// defrag_all() flips every mega-shard to its sibling mini-shard and releases
// the old one. Inserts that follow must land in (and be found through) the new
// shard, and the keys relocated by the defrag must survive them. This is the
// pattern the plugin's automatic defrag produces: fragment, compact, keep
// inserting.
// ---------------------------------------------------------------------------
TEST_P(SphEngineTest, InsertAfterDefrag)
{
    const std::int64_t vs             = value_size();
    const std::int64_t capacity       = 20'000'000;
    const std::int64_t batch_size     = 1'000'000;
    const int          batches_before = 5;
    const int          batches_after  = 5;
    const std::int64_t total_keys     = batch_size * (batches_before + batches_after);

    std::vector<KeyT>         h_keys;
    std::vector<std::uint8_t> h_values;
    fill_reference(h_keys, h_values, total_keys, vs);

    KeyT*         d_keys   = upload(h_keys);
    std::uint8_t* d_values = upload(h_values);
    std::uint8_t* d_found  = alloc<std::uint8_t>(total_keys * vs);
    ASSERT_NE(d_keys, nullptr);
    ASSERT_NE(d_values, nullptr);
    ASSERT_NE(d_found, nullptr);

    auto m = std::make_unique<sph::ShardedPerfectHashMap>(capacity, vs);
    for (int b = 0; b < batches_before; ++b) {
        m->insert(d_keys + b * batch_size,
                  reinterpret_cast<const std::int8_t*>(d_values + b * batch_size * vs), batch_size,
                  stream_);
    }
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    m->defrag_all(stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    for (int b = batches_before; b < batches_before + batches_after; ++b) {
        m->insert(d_keys + b * batch_size,
                  reinterpret_cast<const std::int8_t*>(d_values + b * batch_size * vs), batch_size,
                  stream_);
    }
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    ASSERT_CUDA(cudaMemsetAsync(d_found, 0, static_cast<std::size_t>(total_keys * vs), stream_));
    m->find(d_keys, total_keys, reinterpret_cast<std::int8_t*>(d_found), stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    std::vector<std::uint8_t> h_got;
    ASSERT_NO_FATAL_FAILURE(download(h_got, d_found, total_keys * vs));

    // Split the failures so a regression says which side lost the keys.
    const std::int64_t split = batch_size * batches_before;
    std::int64_t       bad_pre = 0, bad_post = 0;
    for (std::int64_t i = 0; i < total_keys; ++i) {
        const auto off = static_cast<std::size_t>(i * vs);
        if (std::memcmp(&h_got[off], &h_values[off], static_cast<std::size_t>(vs)) != 0)
            (i < split ? bad_pre : bad_post)++;
    }
    EXPECT_EQ(bad_pre, 0) << "keys inserted before defrag were lost";
    EXPECT_EQ(bad_post, 0) << "keys inserted after defrag were lost";
}

// ---------------------------------------------------------------------------
// hash() locate / determinism.
//
// hash() resolves each present key to its physical payload-slot location (or -1
// for a miss). It reads the PDE/PTE structure, so it requires a POPULATED map.
// After inserting a key set, verify:
//   * every inserted key resolves to a valid (>= 0) location,
//   * two hash() calls on the same keys are identical (determinism),
//   * distinct keys map to distinct slots (no payload aliasing),
//   * disjoint miss-keys all resolve to -1.
// ---------------------------------------------------------------------------
TEST_P(SphEngineTest, HashLocate)
{
    const std::int64_t vs       = value_size();
    const std::int64_t capacity = 2'000'000;
    const std::int64_t num_keys = 1'000'000;

    std::vector<KeyT>         h_keys, h_miss;
    std::vector<std::uint8_t> h_values;
    fill_reference(h_keys, h_values, num_keys, vs);
    fill_miss_keys(h_miss, num_keys);

    KeyT*          d_keys   = upload(h_keys);
    KeyT*          d_miss   = upload(h_miss);
    std::uint8_t*  d_values = upload(h_values);
    std::int64_t*  d_hash1  = alloc<std::int64_t>(num_keys);
    std::int64_t*  d_hash2  = alloc<std::int64_t>(num_keys);
    ASSERT_NE(d_keys, nullptr);
    ASSERT_NE(d_miss, nullptr);
    ASSERT_NE(d_values, nullptr);
    ASSERT_NE(d_hash1, nullptr);
    ASSERT_NE(d_hash2, nullptr);

    auto m = std::make_unique<sph::ShardedPerfectHashMap>(capacity, vs);
    m->insert(d_keys, reinterpret_cast<const std::int8_t*>(d_values), num_keys, stream_);

    // Two calls on the inserted keys -> present + deterministic.
    m->find_locations(d_keys, num_keys, d_hash1, stream_);
    m->find_locations(d_keys, num_keys, d_hash2, stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    std::vector<std::int64_t> h1, h2;
    ASSERT_NO_FATAL_FAILURE(download(h1, d_hash1, num_keys));
    ASSERT_NO_FATAL_FAILURE(download(h2, d_hash2, num_keys));

    std::int64_t nondet = 0, missing = 0;
    for (std::int64_t i = 0; i < num_keys; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        if (h1[idx] != h2[idx]) ++nondet;
        if (h1[idx] < 0) ++missing;
    }
    EXPECT_EQ(nondet, 0) << "hash() is not deterministic across calls";
    EXPECT_EQ(missing, 0) << "inserted keys resolved to -1";

    // Distinct keys must occupy distinct payload slots.
    std::vector<std::int64_t> sorted = h1;
    std::sort(sorted.begin(), sorted.end());
    const auto distinct = std::unique(sorted.begin(), sorted.end()) - sorted.begin();
    EXPECT_EQ(num_keys - static_cast<std::int64_t>(distinct), 0) << "aliased payload slots";

    // Miss-keys must all resolve to -1.
    m->find_locations(d_miss, num_keys, d_hash1, stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    ASSERT_NO_FATAL_FAILURE(download(h1, d_hash1, num_keys));
    std::int64_t false_hit = 0;
    for (std::int64_t i = 0; i < num_keys; ++i)
        if (h1[static_cast<std::size_t>(i)] != -1) ++false_hit;
    EXPECT_EQ(false_hit, 0) << "miss-keys resolved to a slot";
}

// ---------------------------------------------------------------------------
// reset() reuse.
//
// Insert one key set, reset() the map to a fresh capacity, insert a DISJOINT
// key set, and verify the second set round-trips while the first set now
// misses. Guards against reset leaving stale PTE/payload state that would leak
// old entries into the reused map.
// ---------------------------------------------------------------------------
TEST_P(SphEngineTest, ResetReuse)
{
    const std::int64_t vs       = value_size();
    const std::int64_t capacity = 2'000'000;
    const std::int64_t num_keys = 1'000'000;

    // First generation: keys [1 .. num_keys].
    std::vector<KeyT>         h_keys_a;
    std::vector<std::uint8_t> h_values_a;
    fill_reference(h_keys_a, h_values_a, num_keys, vs, /*first_key=*/1);

    // Second generation: a disjoint key range far from the first.
    const KeyT                gen_b_first = 1 + static_cast<KeyT>(num_keys) + 500'000'000;
    std::vector<KeyT>         h_keys_b;
    std::vector<std::uint8_t> h_values_b;
    fill_reference(h_keys_b, h_values_b, num_keys, vs, gen_b_first);

    KeyT*         d_keys_a   = upload(h_keys_a);
    std::uint8_t* d_values_a = upload(h_values_a);
    KeyT*         d_keys_b   = upload(h_keys_b);
    std::uint8_t* d_values_b = upload(h_values_b);
    std::uint8_t* d_found    = alloc<std::uint8_t>(num_keys * vs);
    ASSERT_NE(d_keys_a, nullptr);
    ASSERT_NE(d_values_a, nullptr);
    ASSERT_NE(d_keys_b, nullptr);
    ASSERT_NE(d_values_b, nullptr);
    ASSERT_NE(d_found, nullptr);

    auto m = std::make_unique<sph::ShardedPerfectHashMap>(capacity, vs);
    m->insert(d_keys_a, reinterpret_cast<const std::int8_t*>(d_values_a), num_keys, stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    m->reset(stream_, capacity);
    m->insert(d_keys_b, reinterpret_cast<const std::int8_t*>(d_values_b), num_keys, stream_);

    // Positive: gen-B keys round-trip.
    ASSERT_CUDA(cudaMemsetAsync(d_found, 0, static_cast<std::size_t>(num_keys * vs), stream_));
    m->find(d_keys_b, num_keys, reinterpret_cast<std::int8_t*>(d_found), stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    std::vector<std::uint8_t> h_found;
    ASSERT_NO_FATAL_FAILURE(download(h_found, d_found, num_keys * vs));

    std::int64_t first_bad = -1;
    EXPECT_EQ(count_mismatches(h_found, h_values_b, num_keys, vs, &first_bad), 0)
        << "first bad key="
        << (first_bad >= 0 ? h_keys_b[static_cast<std::size_t>(first_bad)] : -1);

    // Negative: gen-A keys must now miss -> every row is the default value.
    ASSERT_CUDA(cudaMemsetAsync(d_found, 0xCC, static_cast<std::size_t>(num_keys * vs), stream_));
    m->find(d_keys_a, num_keys, reinterpret_cast<std::int8_t*>(d_found), stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    ASSERT_NO_FATAL_FAILURE(download(h_found, d_found, num_keys * vs));

    const std::vector<std::uint8_t> h_zero_default(static_cast<std::size_t>(vs), 0);
    EXPECT_EQ(count_non_default(h_found, h_zero_default, vs), 0)
        << "stale gen-A entries leaked through reset()";
}

// ---------------------------------------------------------------------------
// reserve() then reset() then insert().
//
// Mirrors the plugin lifecycle: SphTable's ctor calls reserve() (pre-commits
// VMM), SphTable::clear() calls reset() (rebuilds the VMM buffers with a single
// committed page per shard). The next insert must not write to uncommitted VMM
// pages. The original bug skipped the host-side VMM grow whenever reserve() had
// run, so this post-reset insert faulted with an illegal memory access; the fix
// runs the grow unconditionally (a no-op when there's already room). This is the
// ONLY test that primes reserve() before reset() — ResetReuse never calls
// reserve() so it can't reproduce it.
// ---------------------------------------------------------------------------
TEST_P(SphEngineTest, ReserveResetInsert)
{
    const std::int64_t vs       = value_size();
    const std::int64_t capacity = 2'000'000;
    const std::int64_t num_keys = 1'000'000;

    std::vector<KeyT>         h_keys;
    std::vector<std::uint8_t> h_values;
    fill_reference(h_keys, h_values, num_keys, vs, /*first_key=*/1);

    KeyT*         d_keys   = upload(h_keys);
    std::uint8_t* d_values = upload(h_values);
    std::uint8_t* d_found  = alloc<std::uint8_t>(num_keys * vs);
    ASSERT_NE(d_keys, nullptr);
    ASSERT_NE(d_values, nullptr);
    ASSERT_NE(d_found, nullptr);

    auto m = std::make_unique<sph::ShardedPerfectHashMap>(capacity, vs);
    // Prime the reserve flag, as the plugin ctor does.
    m->reserve(capacity, stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    // clear()-equivalent: rebuild VMM buffers with one committed page/shard.
    m->reset(stream_, capacity);

    // The faulting call before the fix.
    m->insert(d_keys, reinterpret_cast<const std::int8_t*>(d_values), num_keys, stream_);

    ASSERT_CUDA(cudaMemsetAsync(d_found, 0, static_cast<std::size_t>(num_keys * vs), stream_));
    m->find(d_keys, num_keys, reinterpret_cast<std::int8_t*>(d_found), stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    std::vector<std::uint8_t> h_found;
    ASSERT_NO_FATAL_FAILURE(download(h_found, d_found, num_keys * vs));

    std::int64_t first_bad = -1;
    EXPECT_EQ(count_mismatches(h_found, h_values, num_keys, vs, &first_bad), 0)
        << "first bad key=" << (first_bad >= 0 ? h_keys[static_cast<std::size_t>(first_bad)] : -1);
}

// ---------------------------------------------------------------------------
// Caller-supplied default value on miss.
//
// A map constructed with an explicit default_value must return that row —
// byte-for-byte — for every key it does not hold, while hits are unaffected.
// The nullptr (all-zero) default is covered by the miss checks above.
// ---------------------------------------------------------------------------
TEST_P(SphEngineTest, DefaultValue)
{
    const std::int64_t vs       = value_size();
    const std::int64_t capacity = 2'000'000;
    const std::int64_t num_keys = 1'000'000;

    std::vector<KeyT>         h_keys, h_miss;
    std::vector<std::uint8_t> h_values;
    fill_reference(h_keys, h_values, num_keys, vs);
    fill_miss_keys(h_miss, num_keys);

    // Non-trivial default: distinct per byte so a partial/shifted copy shows up.
    std::vector<std::uint8_t> h_default(static_cast<std::size_t>(vs));
    for (std::int64_t b = 0; b < vs; ++b)
        h_default[static_cast<std::size_t>(b)] = static_cast<std::uint8_t>((0xA5 + b * 7) & 0xff);

    KeyT*         d_keys   = upload(h_keys);
    KeyT*         d_miss   = upload(h_miss);
    std::uint8_t* d_values = upload(h_values);
    std::uint8_t* d_found  = alloc<std::uint8_t>(num_keys * vs);
    ASSERT_NE(d_keys, nullptr);
    ASSERT_NE(d_miss, nullptr);
    ASSERT_NE(d_values, nullptr);
    ASSERT_NE(d_found, nullptr);

    auto m = std::make_unique<sph::ShardedPerfectHashMap>(
        capacity, vs, reinterpret_cast<const std::int8_t*>(h_default.data()));
    m->insert(d_keys, reinterpret_cast<const std::int8_t*>(d_values), num_keys, stream_);

    // Hits are unaffected by the default.
    ASSERT_CUDA(cudaMemsetAsync(d_found, 0xCC, static_cast<std::size_t>(num_keys * vs), stream_));
    m->find(d_keys, num_keys, reinterpret_cast<std::int8_t*>(d_found), stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    std::vector<std::uint8_t> h_found;
    ASSERT_NO_FATAL_FAILURE(download(h_found, d_found, num_keys * vs));

    std::int64_t first_bad = -1;
    EXPECT_EQ(count_mismatches(h_found, h_values, num_keys, vs, &first_bad), 0)
        << "first bad key=" << (first_bad >= 0 ? h_keys[static_cast<std::size_t>(first_bad)] : -1);

    // Misses get the supplied default, overwriting whatever was in the buffer.
    ASSERT_CUDA(cudaMemsetAsync(d_found, 0xCC, static_cast<std::size_t>(num_keys * vs), stream_));
    m->find(d_miss, num_keys, reinterpret_cast<std::int8_t*>(d_found), stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    ASSERT_NO_FATAL_FAILURE(download(h_found, d_found, num_keys * vs));

    EXPECT_EQ(count_non_default(h_found, h_default, vs), 0)
        << "miss rows did not get the caller-supplied default";
}

// ---------------------------------------------------------------------------
// In/out hit mask.
//
// Probe a batch that interleaves inserted keys (even indices) with absent keys
// (odd indices), having pre-set the mask bit for every third index. Expect:
//   - masked index      -> row untouched, bit still set
//   - unmasked hit      -> value returned,  bit set
//   - unmasked miss     -> default row,     bit still clear
// ---------------------------------------------------------------------------
TEST_P(SphEngineTest, HitMask)
{
    const std::int64_t vs       = value_size();
    const std::int64_t capacity = 2'000'000;
    const std::int64_t num_keys = 1'000'000;

    std::vector<KeyT>         h_keys, h_miss;
    std::vector<std::uint8_t> h_values;
    fill_reference(h_keys, h_values, num_keys, vs);
    fill_miss_keys(h_miss, num_keys);

    // Probe: even indices hit, odd indices miss.
    std::vector<KeyT> h_probe(static_cast<std::size_t>(num_keys));
    for (std::int64_t i = 0; i < num_keys; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        h_probe[idx]   = (i % 2 == 0) ? h_keys[idx] : h_miss[idx];
    }

    const std::int64_t         num_words = nve::ceil_div<std::int64_t>(num_keys, 64);
    std::vector<std::uint64_t> h_mask(static_cast<std::size_t>(num_words), 0);
    auto                       masked = [](std::int64_t i) { return (i % 3) == 0; };
    for (std::int64_t i = 0; i < num_keys; ++i)
        if (masked(i)) h_mask[static_cast<std::size_t>(i / 64)] |= 1ull << (i % 64);

    KeyT*          d_keys   = upload(h_keys);
    KeyT*          d_probe  = upload(h_probe);
    std::uint8_t*  d_values = upload(h_values);
    std::uint64_t* d_mask   = upload(h_mask);
    std::uint8_t*  d_found  = alloc<std::uint8_t>(num_keys * vs);
    ASSERT_NE(d_keys, nullptr);
    ASSERT_NE(d_probe, nullptr);
    ASSERT_NE(d_values, nullptr);
    ASSERT_NE(d_mask, nullptr);
    ASSERT_NE(d_found, nullptr);

    auto m = std::make_unique<sph::ShardedPerfectHashMap>(capacity, vs);
    m->insert(d_keys, reinterpret_cast<const std::int8_t*>(d_values), num_keys, stream_);

    ASSERT_CUDA(cudaMemsetAsync(d_found, 0xCC, static_cast<std::size_t>(num_keys * vs), stream_));
    m->find(d_probe, num_keys, reinterpret_cast<std::int8_t*>(d_found), stream_, d_mask);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    std::vector<std::uint8_t>  h_found;
    std::vector<std::uint64_t> h_out_mask;
    ASSERT_NO_FATAL_FAILURE(download(h_found, d_found, num_keys * vs));
    ASSERT_NO_FATAL_FAILURE(download(h_out_mask, d_mask, num_words));

    const std::vector<std::uint8_t> untouched(static_cast<std::size_t>(vs), 0xCC);
    const std::vector<std::uint8_t> zeros(static_cast<std::size_t>(vs), 0);
    std::int64_t bad_skipped = 0, bad_hit = 0, bad_miss = 0, bad_bit = 0;
    for (std::int64_t i = 0; i < num_keys; ++i) {
        const auto*        row = &h_found[static_cast<std::size_t>(i * vs)];
        const std::size_t  n   = static_cast<std::size_t>(vs);
        const bool         bit =
            ((h_out_mask[static_cast<std::size_t>(i / 64)] >> (i % 64)) & 1ull) != 0;
        if (masked(i)) {
            if (std::memcmp(row, untouched.data(), n) != 0) ++bad_skipped;
            if (!bit) ++bad_bit;
        } else if (i % 2 == 0) {
            if (std::memcmp(row, &h_values[static_cast<std::size_t>(i * vs)], n) != 0) ++bad_hit;
            if (!bit) ++bad_bit;
        } else {
            if (std::memcmp(row, zeros.data(), n) != 0) ++bad_miss;
            if (bit) ++bad_bit;
        }
    }
    EXPECT_EQ(bad_skipped, 0) << "pre-masked rows were overwritten";
    EXPECT_EQ(bad_hit, 0) << "unmasked hits returned the wrong value";
    EXPECT_EQ(bad_miss, 0) << "unmasked misses did not get the default row";
    EXPECT_EQ(bad_bit, 0) << "output mask bits do not match the expected hit pattern";
}

// ---------------------------------------------------------------------------
// Inserting without values must not disturb the values already stored.
//
// The fused kernel used to gate ALL payload relocation on the incoming batch
// having a non-null d_values, and the host skipped growing the payload the same
// way. So a values-free insert into a populated map rehashed buckets, moved
// their PTE page offsets, and left the old rows where they were — every
// pre-existing key in a touched bucket then read back another key's data.
//
// The map is always value-bearing, so a null d_values now means "these keys get
// the default row", which is the second half of what this checks.
// ---------------------------------------------------------------------------
TEST_P(SphEngineTest, InsertWithoutValuesPreservesStoredValues)
{
    const std::int64_t vs        = value_size();
    const std::int64_t capacity  = 2'000'000;
    const std::int64_t num_first = 200'000;
    const std::int64_t num_second = 200'000;

    // Two disjoint key ranges. The second batch is dense enough to rehash many
    // of the buckets the first batch already populated.
    std::vector<KeyT>         h_first, h_second;
    std::vector<std::uint8_t> h_first_values, h_unused;
    fill_reference(h_first, h_first_values, num_first, vs, /*first_key=*/1);
    fill_reference(h_second, h_unused, num_second, vs, /*first_key=*/num_first + 1);

    std::vector<std::uint8_t> h_default(static_cast<std::size_t>(vs));
    for (std::int64_t b = 0; b < vs; ++b)
        h_default[static_cast<std::size_t>(b)] = static_cast<std::uint8_t>((0x3B + b * 5) & 0xff);

    KeyT*         d_first        = upload(h_first);
    KeyT*         d_second       = upload(h_second);
    std::uint8_t* d_first_values = upload(h_first_values);
    std::uint8_t* d_found        = alloc<std::uint8_t>(std::max(num_first, num_second) * vs);
    ASSERT_NE(d_first, nullptr);
    ASSERT_NE(d_second, nullptr);
    ASSERT_NE(d_first_values, nullptr);
    ASSERT_NE(d_found, nullptr);

    auto m = std::make_unique<sph::ShardedPerfectHashMap>(
        capacity, vs, reinterpret_cast<const std::int8_t*>(h_default.data()));

    m->insert(d_first, reinterpret_cast<const std::int8_t*>(d_first_values), num_first, stream_);
    // The batch under test: keys only, no values.
    m->insert(d_second, nullptr, num_second, stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    // The first batch's values must have survived the rehash intact.
    ASSERT_CUDA(cudaMemsetAsync(d_found, 0xCC, static_cast<std::size_t>(num_first * vs), stream_));
    m->find(d_first, num_first, reinterpret_cast<std::int8_t*>(d_found), stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    std::vector<std::uint8_t> h_found;
    ASSERT_NO_FATAL_FAILURE(download(h_found, d_found, num_first * vs));

    std::int64_t first_bad = -1;
    EXPECT_EQ(count_mismatches(h_found, h_first_values, num_first, vs, &first_bad), 0)
        << "a values-free insert corrupted previously stored values; first bad key="
        << (first_bad >= 0 ? h_first[static_cast<std::size_t>(first_bad)] : -1);

    // And the keys inserted without values read back as the default row, rather
    // than as uninitialized payload.
    ASSERT_CUDA(cudaMemsetAsync(d_found, 0xCC, static_cast<std::size_t>(num_second * vs), stream_));
    m->find(d_second, num_second, reinterpret_cast<std::int8_t*>(d_found), stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    ASSERT_NO_FATAL_FAILURE(download(h_found, d_found, num_second * vs));

    EXPECT_EQ(count_non_default(h_found, h_default, vs), 0)
        << "keys inserted without values did not get the default row";
}

// ---------------------------------------------------------------------------
// defrag() over buckets holding more than 40 entries.
//
// defrag_commit_kernel fills a per-thread array once per active slot in the
// bucket. It was sized 40 while incremental insert admits up to
// max_keys_per_bucket (64), so a bucket that committed 41+ entries overran it
// and corrupted the thread's stack frame.
//
// Buckets are targeted directly rather than hoped for: bucket_hash is
// (global_seed * key) % num_buckets, so with global_seed == 1 every key
// congruent mod num_buckets shares a bucket. At this density pilot search does
// not succeed for every bucket, so a partial insert is expected and ignored —
// what matters is that the buckets which DO commit carry more than 40 entries
// into defrag, which the occupancy assertion below confirms before defragging.
// ---------------------------------------------------------------------------
TEST_P(SphEngineTest, DefragBucketsPastFortyKeys)
{
    const std::int64_t vs = value_size();
    // 2048 is the min_num_buckets floor: exactly 256 buckets, one mega-shard.
    const std::int64_t capacity    = 2048;
    const std::int64_t per_bucket  = 46;

    auto m = std::make_unique<sph::ShardedPerfectHashMap>(capacity, vs);
    const std::int64_t num_buckets = m->get_num_buckets();
    ASSERT_EQ(m->get_global_seed(), 1) << "this test derives bucket indices assuming seed 1";

    std::vector<KeyT> h_keys;
    h_keys.reserve(static_cast<std::size_t>(num_buckets * per_bucket));
    for (std::int64_t b = 0; b < num_buckets; ++b)
        for (std::int64_t i = 0; i < per_bucket; ++i)
            h_keys.push_back(static_cast<KeyT>(b + num_buckets * (i + 1)));
    const auto num_keys = static_cast<std::int64_t>(h_keys.size());

    std::vector<std::uint8_t> h_values(static_cast<std::size_t>(num_keys * vs));
    for (std::int64_t i = 0; i < num_keys; ++i)
        for (std::int64_t b = 0; b < vs; ++b)
            h_values[static_cast<std::size_t>(i * vs + b)] =
                static_cast<std::uint8_t>((h_keys[static_cast<std::size_t>(i)] ^ (b * 7)) & 0xff);

    KeyT*         d_keys   = upload(h_keys);
    std::uint8_t* d_values = upload(h_values);
    std::uint8_t* d_found  = alloc<std::uint8_t>(num_keys * vs);
    std::int64_t* d_hashes = alloc<std::int64_t>(num_keys);
    ASSERT_NE(d_keys, nullptr);
    ASSERT_NE(d_values, nullptr);
    ASSERT_NE(d_found, nullptr);
    ASSERT_NE(d_hashes, nullptr);

    // Partial application is expected at this density; the drop is not the point.
    try {
        m->insert(d_keys, reinterpret_cast<const std::int8_t*>(d_values), num_keys, stream_);
    } catch (const sph::SphInsertFailed&) {
    }
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    // Snapshot which keys are resident, and confirm some bucket is past 40.
    m->find_locations(d_keys, num_keys, d_hashes, stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    std::vector<std::int64_t> h_before;
    ASSERT_NO_FATAL_FAILURE(download(h_before, d_hashes, num_keys));

    std::vector<std::int64_t> occupancy(static_cast<std::size_t>(num_buckets), 0);
    for (std::int64_t i = 0; i < num_keys; ++i)
        if (h_before[static_cast<std::size_t>(i)] >= 0)
            ++occupancy[static_cast<std::size_t>(h_keys[static_cast<std::size_t>(i)] % num_buckets)];
    const std::int64_t max_occupancy = *std::max_element(occupancy.begin(), occupancy.end());
    ASSERT_GT(max_occupancy, 40)
        << "no bucket exceeded 40 entries, so this run does not exercise the overrun";

    ASSERT_CUDA(cudaMemsetAsync(d_found, 0xCC, static_cast<std::size_t>(num_keys * vs), stream_));
    m->find(d_keys, num_keys, reinterpret_cast<std::int8_t*>(d_found), stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    std::vector<std::uint8_t> h_found_before;
    ASSERT_NO_FATAL_FAILURE(download(h_found_before, d_found, num_keys * vs));

    m->defrag_all(stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    // Defrag relocates rows; it must not lose, gain or alter a single one.
    m->find_locations(d_keys, num_keys, d_hashes, stream_);
    ASSERT_CUDA(cudaMemsetAsync(d_found, 0xCC, static_cast<std::size_t>(num_keys * vs), stream_));
    m->find(d_keys, num_keys, reinterpret_cast<std::int8_t*>(d_found), stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    std::vector<std::int64_t> h_after;
    std::vector<std::uint8_t> h_found_after;
    ASSERT_NO_FATAL_FAILURE(download(h_after, d_hashes, num_keys));
    ASSERT_NO_FATAL_FAILURE(download(h_found_after, d_found, num_keys * vs));

    std::int64_t membership_changed = 0, value_changed = 0;
    for (std::int64_t i = 0; i < num_keys; ++i) {
        const bool was = h_before[static_cast<std::size_t>(i)] >= 0;
        const bool now = h_after[static_cast<std::size_t>(i)] >= 0;
        if (was != now) ++membership_changed;
        const auto off = static_cast<std::size_t>(i * vs);
        if (was && now &&
            std::memcmp(&h_found_before[off], &h_found_after[off],
                        static_cast<std::size_t>(vs)) != 0)
            ++value_changed;
    }
    EXPECT_EQ(membership_changed, 0) << "defrag changed which keys are resident";
    EXPECT_EQ(value_changed, 0) << "defrag altered stored values";
}

// ---------------------------------------------------------------------------
// assign() must ignore keys that are not in the map.
//
// The assign kernel used to derive a destination from whatever PTE happened to
// occupy the computed slot, with no valid_pde / page_offset / fingerprint check.
// For an absent key that meant writing over the resident row of whichever key
// really owns the slot, or through an empty PTE's page_offset == 255. Assigning
// a batch of pure misses must leave every stored value byte-for-byte intact.
// ---------------------------------------------------------------------------
TEST_P(SphEngineTest, AssignIgnoresMissingKeys)
{
    const std::int64_t vs       = value_size();
    const std::int64_t capacity = 2'000'000;
    const std::int64_t num_keys = 200'000;

    std::vector<KeyT>         h_keys, h_miss;
    std::vector<std::uint8_t> h_values;
    fill_reference(h_keys, h_values, num_keys, vs);
    fill_miss_keys(h_miss, num_keys);

    // Replacement rows, traceable back to the key like fill_reference's are but
    // from a different mix, so a stray write is visible as a mismatch.
    std::vector<std::uint8_t> h_new(static_cast<std::size_t>(num_keys * vs));
    for (std::int64_t i = 0; i < num_keys; ++i)
        for (std::int64_t b = 0; b < vs; ++b)
            h_new[static_cast<std::size_t>(i * vs + b)] =
                static_cast<std::uint8_t>((h_keys[static_cast<std::size_t>(i)] * 3 + b * 11 + 0x5A) & 0xff);

    KeyT*         d_keys   = upload(h_keys);
    KeyT*         d_miss   = upload(h_miss);
    std::uint8_t* d_values = upload(h_values);
    std::uint8_t* d_new    = upload(h_new);
    std::uint8_t* d_found  = alloc<std::uint8_t>(num_keys * vs);
    ASSERT_NE(d_keys, nullptr);
    ASSERT_NE(d_miss, nullptr);
    ASSERT_NE(d_values, nullptr);
    ASSERT_NE(d_new, nullptr);
    ASSERT_NE(d_found, nullptr);

    auto m = std::make_unique<sph::ShardedPerfectHashMap>(capacity, vs);
    m->insert(d_keys, reinterpret_cast<const std::int8_t*>(d_values), num_keys, stream_);

    auto find_all = [&](std::vector<std::uint8_t>& out) {
        ASSERT_CUDA(cudaMemsetAsync(d_found, 0xCC, static_cast<std::size_t>(num_keys * vs), stream_));
        m->find(d_keys, num_keys, reinterpret_cast<std::int8_t*>(d_found), stream_);
        ASSERT_CUDA(cudaStreamSynchronize(stream_));
        ASSERT_NO_FATAL_FAILURE(download(out, d_found, num_keys * vs));
    };

    // A batch of pure misses must be a no-op on the payload.
    m->assign(d_miss, reinterpret_cast<const std::int8_t*>(d_new), num_keys, stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    std::vector<std::uint8_t> h_found;
    ASSERT_NO_FATAL_FAILURE(find_all(h_found));
    std::int64_t first_bad = -1;
    EXPECT_EQ(count_mismatches(h_found, h_values, num_keys, vs, &first_bad), 0)
        << "assign() of absent keys corrupted resident rows; first bad key="
        << (first_bad >= 0 ? h_keys[static_cast<std::size_t>(first_bad)] : -1);

    // The positive path still works: assigning resident keys replaces their rows.
    m->assign(d_keys, reinterpret_cast<const std::int8_t*>(d_new), num_keys, stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    ASSERT_NO_FATAL_FAILURE(find_all(h_found));
    first_bad = -1;
    EXPECT_EQ(count_mismatches(h_found, h_new, num_keys, vs, &first_bad), 0)
        << "assign() of resident keys did not update them; first bad key="
        << (first_bad >= 0 ? h_keys[static_cast<std::size_t>(first_bad)] : -1);
}

// ---------------------------------------------------------------------------
// Lookup on a table that has never been inserted into.
//
// reset() zeroes the PDEs, so every bucket is invalid. Both lookup kernels used
// to form a PTE address and load it BEFORE testing valid_pde, which on a fresh
// table reads VMM that reset() had not committed for the PTE buffer at all — a
// fault on the very first find()/hash(). Exercised at a capacity large enough to
// span several mega-shards, and again after reset(), which is the other way to
// reach the all-invalid state.
// ---------------------------------------------------------------------------
TEST_P(SphEngineTest, LookupOnEmptyTable)
{
    const std::int64_t vs       = value_size();
    const std::int64_t capacity = 20'000'000;
    const std::int64_t num_keys = 100'000;

    std::vector<KeyT>         h_keys, h_probe;
    std::vector<std::uint8_t> h_values;
    fill_reference(h_keys, h_values, num_keys, vs);
    fill_miss_keys(h_probe, num_keys);

    // Non-trivial default, so "got the default row" is distinguishable from
    // "the buffer happened to be zero".
    std::vector<std::uint8_t> h_default(static_cast<std::size_t>(vs));
    for (std::int64_t b = 0; b < vs; ++b)
        h_default[static_cast<std::size_t>(b)] = static_cast<std::uint8_t>((0x5C + b * 3) & 0xff);

    KeyT*         d_keys   = upload(h_keys);
    KeyT*         d_probe  = upload(h_probe);
    std::uint8_t* d_values = upload(h_values);
    std::uint8_t* d_found  = alloc<std::uint8_t>(num_keys * vs);
    std::int64_t* d_hashes = alloc<std::int64_t>(num_keys);
    ASSERT_NE(d_keys, nullptr);
    ASSERT_NE(d_probe, nullptr);
    ASSERT_NE(d_values, nullptr);
    ASSERT_NE(d_found, nullptr);
    ASSERT_NE(d_hashes, nullptr);

    auto m = std::make_unique<sph::ShardedPerfectHashMap>(
        capacity, vs, reinterpret_cast<const std::int8_t*>(h_default.data()));

    // Every probe must resolve to the default row / -1, and nothing may fault.
    auto expect_all_miss = [&](const char* phase) {
        ASSERT_CUDA(cudaMemsetAsync(d_found, 0xCC, static_cast<std::size_t>(num_keys * vs), stream_));
        m->find(d_probe, num_keys, reinterpret_cast<std::int8_t*>(d_found), stream_);
        m->find_locations(d_probe, num_keys, d_hashes, stream_);
        ASSERT_CUDA(cudaStreamSynchronize(stream_));

        std::vector<std::uint8_t> h_found;
        std::vector<std::int64_t> h_hashes;
        ASSERT_NO_FATAL_FAILURE(download(h_found, d_found, num_keys * vs));
        ASSERT_NO_FATAL_FAILURE(download(h_hashes, d_hashes, num_keys));

        EXPECT_EQ(count_non_default(h_found, h_default, vs), 0)
            << phase << ": find() on an empty table did not return the default row";
        EXPECT_EQ(std::count(h_hashes.begin(), h_hashes.end(), std::int64_t{-1}), num_keys)
            << phase << ": hash() on an empty table reported a hit";
    };

    ASSERT_NO_FATAL_FAILURE(expect_all_miss("fresh"));

    // Same state reached the other way: populate, then reset back to empty.
    m->insert(d_keys, reinterpret_cast<const std::int8_t*>(d_values), num_keys, stream_);
    m->reset(stream_, capacity);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    ASSERT_NO_FATAL_FAILURE(expect_all_miss("after reset"));

    // The previously inserted keys are gone too, not merely unreachable.
    ASSERT_CUDA(cudaMemsetAsync(d_found, 0xCC, static_cast<std::size_t>(num_keys * vs), stream_));
    m->find_locations(d_keys, num_keys, d_hashes, stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    std::vector<std::int64_t> h_hashes;
    ASSERT_NO_FATAL_FAILURE(download(h_hashes, d_hashes, num_keys));
    EXPECT_EQ(std::count(h_hashes.begin(), h_hashes.end(), std::int64_t{-1}), num_keys)
        << "reset() left pre-reset keys visible";
}

INSTANTIATE_TEST_SUITE_P(
    ValueSizes, SphEngineTest,
    ::testing::Values(1, 4, 7, 12, 16, 20, 33, 64),
    [](const ::testing::TestParamInfo<std::int64_t>& info) {
        return "v" + std::to_string(info.param);
    });

// ===========================================================================
// Argument validation.
//
// Every public entry point rejects bad arguments by throwing
// sph::SphInvalidArgument, in release builds as well as debug — these
// checks replaced bare asserts that were compiled out under NDEBUG.
// ===========================================================================
class SphEngineValidationTest : public SphDeviceFixture<::testing::Test> {
protected:
    void SetUp() override
    {
        SphDeviceFixture::SetUp();
        if (IsSkipped() || HasFatalFailure()) return;
        map_ = std::make_unique<sph::ShardedPerfectHashMap>(kCapacity, kValueSize);
    }

    void TearDown() override
    {
        map_.reset();  // before the stream and buffers it may still reference
        SphDeviceFixture::TearDown();
    }

    // One buffer big enough to stand in for keys, values or hashes.
    void* scratch()
    {
        if (!d_scratch_) d_scratch_ = alloc<std::uint8_t>(4096);
        return d_scratch_;
    }

    static constexpr std::int64_t kCapacity  = 4096;
    static constexpr std::int64_t kValueSize = 16;

    std::unique_ptr<sph::ShardedPerfectHashMap> map_;
    void*                                       d_scratch_ = nullptr;
};

TEST_F(SphEngineValidationTest, ConstructorRejectsBadGeometry)
{
    EXPECT_THROW(sph::ShardedPerfectHashMap(0, kValueSize), sph::SphInvalidArgument);
    EXPECT_THROW(sph::ShardedPerfectHashMap(-1, kValueSize), sph::SphInvalidArgument);
    EXPECT_THROW(sph::ShardedPerfectHashMap(kCapacity, 0), sph::SphInvalidArgument);
    EXPECT_THROW(sph::ShardedPerfectHashMap(kCapacity, -8), sph::SphInvalidArgument);
}

// The PTE fingerprint is key / num_buckets in sph::pte_fingerprint_bits
// bits, so a table with fewer than min_num_buckets buckets would truncate it.
// Small capacities are rounded up to that floor rather than rejected.
TEST_F(SphEngineValidationTest, SmallCapacityIsRaisedToTheFingerprintFloor)
{
    constexpr std::int64_t kFloor = sph::min_num_buckets * 8;  // avg_bucket_size_

    sph::ShardedPerfectHashMap tiny(1, kValueSize);
    EXPECT_EQ(tiny.get_target_capacity(), kFloor);
    EXPECT_GE(tiny.get_num_buckets(), sph::min_num_buckets);

    // A capacity already above the floor is left alone.
    sph::ShardedPerfectHashMap big(kFloor * 4, kValueSize);
    EXPECT_EQ(big.get_target_capacity(), kFloor * 4);
}

// The fingerprint floor exists for keys near the top of the uint64 range: the
// PTE path treats a key as uint64, so a large-magnitude negative key has a
// quotient just under 2^pte_fingerprint_bits and only fits once num_buckets
// reaches min_num_buckets. Exercise exactly that at the smallest capacity the
// engine will build.
//
// -1 is excluded on purpose: it is the repo-wide invalid-key sentinel
// (SphTableConfig::invalid_key / gpu_table.hpp), not a storable key, and it
// round-trips at no capacity.
TEST_F(SphEngineValidationTest, FingerprintHoldsForTopOfRangeKeysAtMinCapacity)
{
    const std::vector<KeyT> h_keys = {-2, -3, -256, -65'537,
                                      std::numeric_limits<KeyT>::min(),
                                      std::numeric_limits<KeyT>::max(), 1, 1'000'003};
    const auto              n      = static_cast<std::int64_t>(h_keys.size());

    std::vector<std::uint8_t> h_values(static_cast<std::size_t>(n * kValueSize));
    for (std::size_t i = 0; i < h_values.size(); ++i)
        h_values[i] = static_cast<std::uint8_t>(i * 7 + 1);

    KeyT*         d_keys   = upload(h_keys);
    std::uint8_t* d_values = upload(h_values);
    std::uint8_t* d_found  = alloc<std::uint8_t>(n * kValueSize);
    ASSERT_TRUE(d_keys != nullptr && d_values != nullptr && d_found != nullptr);

    sph::ShardedPerfectHashMap m(1, kValueSize);  // rounded up to the fingerprint floor
    ASSERT_GE(m.get_num_buckets(), sph::min_num_buckets);
    m.insert(d_keys, reinterpret_cast<const std::int8_t*>(d_values), n, stream_);
    ASSERT_CUDA(cudaMemsetAsync(d_found, 0xCC, h_values.size(), stream_));
    m.find(d_keys, n, reinterpret_cast<std::int8_t*>(d_found), stream_);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    std::vector<std::uint8_t> got(h_values.size());
    ASSERT_CUDA(cudaMemcpy(got.data(), d_found, got.size(), cudaMemcpyDeviceToHost));
    for (std::int64_t i = 0; i < n; ++i) {
        const auto off = static_cast<std::size_t>(i * kValueSize);
        EXPECT_EQ(std::memcmp(&got[off], &h_values[off], static_cast<std::size_t>(kValueSize)), 0)
            << "key " << h_keys[static_cast<std::size_t>(i)] << " did not round-trip";
    }
}

TEST_F(SphEngineValidationTest, InsertRejectsBadArguments)
{
    auto* keys = static_cast<KeyT*>(scratch());
    ASSERT_NE(keys, nullptr);

    EXPECT_THROW(map_->insert(keys, nullptr, -1, stream_), sph::SphInvalidArgument);
    EXPECT_THROW(map_->insert(nullptr, nullptr, 4, stream_), sph::SphInvalidArgument);
}

TEST_F(SphEngineValidationTest, EmptyInsertIsANoOp)
{
    const std::int64_t before = map_->get_size();
    EXPECT_NO_THROW(map_->insert(nullptr, nullptr, 0, stream_));
    EXPECT_EQ(map_->get_size(), before)
        << "an empty insert must not advance the key count";
}

TEST_F(SphEngineValidationTest, FindRejectsBadArguments)
{
    auto* buf = static_cast<std::int8_t*>(scratch());
    ASSERT_NE(buf, nullptr);
    const auto* keys = reinterpret_cast<const std::int64_t*>(buf);

    EXPECT_THROW(map_->find(keys, -1, buf, stream_), sph::SphInvalidArgument);
    EXPECT_THROW(map_->find(nullptr, 4, buf, stream_), sph::SphInvalidArgument);
    EXPECT_THROW(map_->find(keys, 4, nullptr, stream_), sph::SphInvalidArgument);
    // A stride narrower than a row would make consecutive output rows overlap.
    EXPECT_THROW(map_->find(keys, 4, buf, stream_, nullptr, kValueSize - 1),
                 sph::SphInvalidArgument);
}

TEST_F(SphEngineValidationTest, AssignAndHashRejectNullBuffers)
{
    auto* buf = static_cast<std::int8_t*>(scratch());
    ASSERT_NE(buf, nullptr);
    auto* keys = reinterpret_cast<std::int64_t*>(buf);

    EXPECT_THROW(map_->assign(nullptr, buf, 4, stream_), sph::SphInvalidArgument);
    EXPECT_THROW(map_->assign(keys, nullptr, 4, stream_), sph::SphInvalidArgument);
    EXPECT_THROW(map_->find_locations(nullptr, 4, keys, stream_), sph::SphInvalidArgument);
    EXPECT_THROW(map_->find_locations(keys, 4, nullptr, stream_), sph::SphInvalidArgument);
}

TEST_F(SphEngineValidationTest, DefragRejectsOutOfRangeShard)
{
    EXPECT_THROW(map_->defrag(-1, stream_), sph::SphInvalidArgument);
    EXPECT_THROW(map_->defrag(static_cast<std::int32_t>(map_->get_num_shards()), stream_),
                 sph::SphInvalidArgument);
}

// The exception types exist so the nve C-API boundary can classify failures
// without matching on message text. NVE_C_CATCH keys off the std:: bases, so
// those relationships are what actually has to hold.
// curr_capacity used to be advanced on entry to insert(), so a rejected or
// partial insert left it permanently overstated — which skews
// get_fragmentation() and the plugin's defrag threshold.
TEST_F(SphEngineValidationTest, CurrCapacityCountsOnlyKeysThatLanded)
{
    constexpr std::int64_t kKeys = 4096;
    std::vector<KeyT>         h_keys(kKeys);
    std::vector<std::uint8_t> h_values(static_cast<std::size_t>(kKeys * kValueSize), 0);
    for (std::int64_t i = 0; i < kKeys; ++i) h_keys[static_cast<std::size_t>(i)] = i + 1;

    KeyT*         d_keys   = upload(h_keys);
    std::uint8_t* d_values = upload(h_values);
    ASSERT_TRUE(d_keys != nullptr && d_values != nullptr);

    sph::ShardedPerfectHashMap m(kKeys * 8, kValueSize);
    ASSERT_EQ(m.get_size(), 0);

    // A rejected insert must not move the counter at all.
    EXPECT_THROW(m.insert(nullptr, nullptr, 8, stream_), sph::SphInvalidArgument);
    EXPECT_EQ(m.get_size(), 0);

    // A clean insert does not throw, and every key is accounted for.
    EXPECT_NO_THROW(m.insert(d_keys, reinterpret_cast<const std::int8_t*>(d_values), kKeys, stream_));
    EXPECT_EQ(m.get_size(), kKeys);
}

// Overfilling makes the pilot search give up on most buckets, so the insert
// really does drop keys. That must surface as SphInsertFailed carrying the full
// breakdown — not as a silent partial write, and not as a process kill.
TEST_F(SphEngineValidationTest, OverfillThrowsWithDroppedKeyBreakdown)
{
    // 6x overfill: 256 buckets carrying ~48 keys each. Enough for the pilot
    // search to give up on most buckets but not all, so this exercises a
    // PARTIAL drop (non-zero keys_inserted()), which is the interesting case.
    // Kept under max_keys_per_bucket (64) so this exercises pilot failure
    // specifically; see OversizedBucketIsReportedNotTrapped for the other side.
    constexpr std::int64_t kSmallCapacity = 2048;
    constexpr std::int64_t kKeys          = 12'288;

    std::vector<KeyT> h_keys(kKeys);
    for (std::int64_t i = 0; i < kKeys; ++i) h_keys[static_cast<std::size_t>(i)] = i + 1;

    KeyT*         d_keys   = upload(h_keys);
    std::uint8_t* d_values = alloc<std::uint8_t>(kKeys * kValueSize);
    ASSERT_TRUE(d_keys != nullptr && d_values != nullptr);
    ASSERT_CUDA(cudaMemset(d_values, 0, static_cast<std::size_t>(kKeys * kValueSize)));

    sph::ShardedPerfectHashMap m(kSmallCapacity, kValueSize);

    bool threw = false;
    try {
        m.insert(d_keys, reinterpret_cast<const std::int8_t*>(d_values), kKeys, stream_);
    } catch (const sph::SphInsertFailed& error) {
        threw = true;
        const auto& r = error.result();
        EXPECT_FALSE(r.ok());
        EXPECT_GT(r.failures.total(), 0) << "a failure mode should have been recorded";
        EXPECT_GT(r.failures.pilot_failure, 0) << "expected pilot failures at this density";

        // The key-granular count is the point: bucket counts alone cannot
        // distinguish one bucket losing 8 keys from one losing 5000.
        EXPECT_GT(r.keys_dropped, 0) << "buckets failed but no keys counted as dropped";
        EXPECT_LT(r.keys_dropped, r.keys_submitted)
            << "expected a partial drop, so keys_inserted() is exercised as non-zero";
        EXPECT_GT(r.keys_inserted(), 0);
        EXPECT_GT(r.keys_dropped, r.failures.total())
            << "dropped keys should exceed the number of failing buckets";
        EXPECT_EQ(r.keys_submitted, kKeys);
        EXPECT_LE(r.keys_dropped,
                  r.keys_submitted - r.keys_existing - r.keys_batch_duplicates)
            << "keys_dropped must be bounded by the post-filter batch";
        EXPECT_EQ(m.get_size(), r.keys_inserted());
    }
    EXPECT_TRUE(threw) << "overfilling this far should have thrown";

    // The batch partially applied, so the table stays usable.
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    std::uint8_t* d_out = alloc<std::uint8_t>(kValueSize);
    ASSERT_TRUE(d_out != nullptr);
    EXPECT_NO_THROW(m.find(d_keys, 1, reinterpret_cast<std::int8_t*>(d_out), stream_));
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
}

// A bucket receiving more keys than the fused kernel's shared-memory gather can
// hold used to call __trap() and kill the process. Stage 1 now rejects it up
// front and reports it as max_bucket_reached.
TEST_F(SphEngineValidationTest, OversizedBucketIsReportedNotTrapped)
{
    // 256 buckets, ~390 keys each — far past max_keys_per_bucket (64).
    constexpr std::int64_t kSmallCapacity = 2048;
    constexpr std::int64_t kKeys          = 100'000;

    std::vector<KeyT> h_keys(kKeys);
    for (std::int64_t i = 0; i < kKeys; ++i) h_keys[static_cast<std::size_t>(i)] = i + 1;

    KeyT*         d_keys   = upload(h_keys);
    std::uint8_t* d_values = alloc<std::uint8_t>(kKeys * kValueSize);
    ASSERT_TRUE(d_keys != nullptr && d_values != nullptr);
    ASSERT_CUDA(cudaMemset(d_values, 0, static_cast<std::size_t>(kKeys * kValueSize)));

    sph::ShardedPerfectHashMap m(kSmallCapacity, kValueSize);

    bool threw = false;
    try {
        m.insert(d_keys, reinterpret_cast<const std::int8_t*>(d_values), kKeys, stream_);
    } catch (const sph::SphInsertFailed& error) {
        threw = true;
        const auto& r = error.result();
        EXPECT_GT(r.failures.max_bucket_reached, 0)
            << "oversized buckets should report max_bucket_reached";
        EXPECT_GT(r.keys_dropped, 0);
        EXPECT_EQ(r.keys_submitted, kKeys);
        // keys_dropped is clamped to the post-filter batch, so this can never go
        // negative; a negative value would silently walk curr_capacity backwards.
        EXPECT_GE(r.keys_inserted(), 0);
        EXPECT_LE(r.keys_dropped,
                  r.keys_submitted - r.keys_existing - r.keys_batch_duplicates);
    }
    EXPECT_TRUE(threw) << "an oversized bucket should throw, not trap";

    // Surviving to here at all is the assertion that matters: before the stage 1
    // gate this aborted the process with SIGABRT.
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
}

// A failed insert must not damage data that was already in the table.
//
// Re-inserting a key that is already present is guaranteed to fail under
// ON_EXISTING_FAIL: the fused kernel gathers the existing key and the new copy
// into the same bucket, and no pilot can map two equal keys to distinct slots.
// The bucket is abandoned — but the keys already committed there must survive,
// because they were never part of this batch and the PDE still points at them.
//
// FAIL is passed explicitly: the default is IGNORE, which filters the repeats
// out and never reaches the failure path this test is about.
//
// This used to lose everything: the destination PTE range was cleared BEFORE the
// pilot search, and with alpha unchanged that range is the live one, so the
// bucket was wiped on the way out. The clear now runs after the search.
TEST_F(SphEngineValidationTest, FailedInsertPreservesExistingKeys)
{
    constexpr std::int64_t kKeys = 1000;

    std::vector<KeyT> h_keys(kKeys);
    for (std::int64_t i = 0; i < kKeys; ++i) h_keys[static_cast<std::size_t>(i)] = i + 1;

    const auto    bytes   = static_cast<std::size_t>(kKeys * kValueSize);
    KeyT*         d_keys  = upload(h_keys);
    std::uint8_t* d_first = alloc<std::uint8_t>(kKeys * kValueSize);  // 0xAA payload, the data that must survive
    std::uint8_t* d_again = alloc<std::uint8_t>(kKeys * kValueSize);  // 0xBB payload, the doomed re-insert
    std::uint8_t* d_out   = alloc<std::uint8_t>(kKeys * kValueSize);
    ASSERT_TRUE(d_keys != nullptr && d_first != nullptr && d_again != nullptr && d_out != nullptr);
    ASSERT_CUDA(cudaMemset(d_first, 0xAA, bytes));
    ASSERT_CUDA(cudaMemset(d_again, 0xBB, bytes));

    sph::ShardedPerfectHashMap m(1 << 20, kValueSize);
    ASSERT_NO_THROW(m.insert(d_keys, reinterpret_cast<const std::int8_t*>(d_first), kKeys, stream_));
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    auto count_original = [&]() {
        EXPECT_CUDA(cudaMemsetAsync(d_out, 0xCC, bytes, stream_));
        m.find(d_keys, kKeys, reinterpret_cast<std::int8_t*>(d_out), stream_);
        EXPECT_CUDA(cudaStreamSynchronize(stream_));
        std::vector<std::uint8_t> got(bytes);
        EXPECT_CUDA(cudaMemcpy(got.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        std::int64_t n = 0;
        for (std::int64_t i = 0; i < kKeys; ++i)
            if (got[static_cast<std::size_t>(i * kValueSize)] == 0xAA) ++n;
        return n;
    };

    ASSERT_EQ(count_original(), kKeys) << "setup: first insert did not land";

    // Every key is a duplicate of one already stored, so every bucket fails.
    EXPECT_THROW(m.insert(d_keys, reinterpret_cast<const std::int8_t*>(d_again), kKeys, stream_,
                          sph::ShardedPerfectHashMap::OnExisting::ON_EXISTING_FAIL),
                 sph::SphInsertFailed);
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    EXPECT_EQ(count_original(), kKeys)
        << "a failed insert destroyed keys that were already committed";
}

// ===========================================================================
// OnExisting: keys already resident, and repeats within one batch.
//
// Both are filtered out of the batch before the insert state machine sizes
// alpha or reserves payload slots (kernel_v3_filter_existing). The key is
// matched exactly, not by fingerprint alone: bucket = key % num_buckets and
// fingerprint = key / num_buckets together recover the whole key.
//
// The shared shape below: insert a block of keys with one tag byte, insert
// something overlapping with a second tag, then check which tag find() returns
// and that curr_capacity() counted only the keys that actually grew the table.
// ===========================================================================
namespace {

// Bundles the map with one device buffer per role, all released by the guard's
// destructor, so the OnExisting tests read as a few named steps.
struct ExistingKeyHarness {
    static constexpr std::int64_t kValueSize = 16;

    explicit ExistingKeyHarness(std::int64_t capacity) : map(capacity, kValueSize) {}
    ~ExistingKeyHarness() { for (void* p : allocs) cudaFree(p); }

    KeyT* keys(const std::vector<KeyT>& h)
    {
        auto* d = static_cast<KeyT*>(raw(h.size() * sizeof(KeyT)));
        if (d) cudaMemcpy(d, h.data(), h.size() * sizeof(KeyT), cudaMemcpyHostToDevice);
        return d;
    }

    // n rows of kValueSize bytes, every byte set to tag.
    std::int8_t* values(std::int64_t n, std::uint8_t tag)
    {
        auto* d = static_cast<std::int8_t*>(raw(static_cast<std::size_t>(n * kValueSize)));
        if (d) cudaMemset(d, tag, static_cast<std::size_t>(n * kValueSize));
        return d;
    }

    // find() the given keys and return the first byte of each row — the tag.
    std::vector<std::uint8_t> tags_of(const KeyT* d_keys, std::int64_t n, cudaStream_t stream)
    {
        const auto bytes = static_cast<std::size_t>(n * kValueSize);
        auto* d_out = static_cast<std::int8_t*>(raw(bytes));
        std::vector<std::uint8_t> got(bytes), out(static_cast<std::size_t>(n));
        if (!d_out) return out;
        cudaMemsetAsync(d_out, 0xCC, bytes, stream);
        map.find(d_keys, n, d_out, stream);
        cudaStreamSynchronize(stream);
        cudaMemcpy(got.data(), d_out, bytes, cudaMemcpyDeviceToHost);
        for (std::int64_t i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] =
            got[static_cast<std::size_t>(i * kValueSize)];
        return out;
    }

    void* raw(std::size_t bytes)
    {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) { ADD_FAILURE() << "cudaMalloc failed"; return nullptr; }
        allocs.push_back(p);
        return p;
    }

    static bool all_tagged(const std::vector<std::uint8_t>& got, std::uint8_t tag,
                           std::int64_t from, std::int64_t to)
    {
        for (std::int64_t i = from; i < to; ++i)
            if (got[static_cast<std::size_t>(i)] != tag) return false;
        return true;
    }

    sph::ShardedPerfectHashMap map;
    std::vector<void*>   allocs;
};

constexpr std::int64_t kExistingCapacity = 1 << 20;

}  // namespace

// Ignore: the stored value wins and the incoming copy is discarded. Nothing
// throws, and the second insert must not advance the key count.
TEST_F(SphEngineValidationTest, InsertExistingKeysIgnored)
{
    constexpr std::int64_t kKeys = 1000;
    std::vector<KeyT> h_keys(kKeys);
    for (std::int64_t i = 0; i < kKeys; ++i) h_keys[static_cast<std::size_t>(i)] = i + 1;

    ExistingKeyHarness h(kExistingCapacity);
    KeyT* d_keys = h.keys(h_keys);
    ASSERT_NE(d_keys, nullptr);

    ASSERT_NO_THROW(h.map.insert(d_keys, h.values(kKeys, 0xAA), kKeys, stream_,
                                 sph::ShardedPerfectHashMap::OnExisting::ON_EXISTING_IGNORE));
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    ASSERT_EQ(h.map.get_size(), kKeys);

    EXPECT_NO_THROW(h.map.insert(d_keys, h.values(kKeys, 0xBB), kKeys, stream_,
                                 sph::ShardedPerfectHashMap::OnExisting::ON_EXISTING_IGNORE));
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    EXPECT_TRUE(ExistingKeyHarness::all_tagged(h.tags_of(d_keys, kKeys, stream_), 0xAA, 0, kKeys))
        << "Ignore must leave the stored value untouched";
    EXPECT_EQ(h.map.get_size(), kKeys)
        << "re-inserting resident keys must not grow the key count";
}

// Update: same filtering, but the incoming value overwrites the stored one in
// place. The key count still must not move — no new slot was allocated.
TEST_F(SphEngineValidationTest, InsertExistingKeysUpdated)
{
    constexpr std::int64_t kKeys = 1000;
    std::vector<KeyT> h_keys(kKeys);
    for (std::int64_t i = 0; i < kKeys; ++i) h_keys[static_cast<std::size_t>(i)] = i + 1;

    ExistingKeyHarness h(kExistingCapacity);
    KeyT* d_keys = h.keys(h_keys);
    ASSERT_NE(d_keys, nullptr);

    ASSERT_NO_THROW(h.map.insert(d_keys, h.values(kKeys, 0xAA), kKeys, stream_,
                                 sph::ShardedPerfectHashMap::OnExisting::ON_EXISTING_UPDATE));
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    EXPECT_NO_THROW(h.map.insert(d_keys, h.values(kKeys, 0xBB), kKeys, stream_,
                                 sph::ShardedPerfectHashMap::OnExisting::ON_EXISTING_UPDATE));
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    EXPECT_TRUE(ExistingKeyHarness::all_tagged(h.tags_of(d_keys, kKeys, stream_), 0xBB, 0, kKeys))
        << "Update must overwrite the stored value";
    EXPECT_EQ(h.map.get_size(), kKeys)
        << "an in-place update must not grow the key count";
}

// A batch that straddles both cases: the resident half is filtered, the fresh
// half lands. Uses Ignore so the two halves stay distinguishable by tag.
TEST_F(SphEngineValidationTest, InsertPartialOverlapCountsOnlyTheFreshKeys)
{
    constexpr std::int64_t kKeys = 1000;

    std::vector<KeyT> h_first(kKeys), h_batch(2 * kKeys);
    for (std::int64_t i = 0; i < kKeys; ++i) {
        h_first[static_cast<std::size_t>(i)] = i + 1;              // resident after step 1
        h_batch[static_cast<std::size_t>(i)] = i + 1;              // ... repeated here
        h_batch[static_cast<std::size_t>(kKeys + i)] = kKeys + i + 1;  // fresh
    }

    ExistingKeyHarness h(kExistingCapacity);
    KeyT* d_first = h.keys(h_first);
    KeyT* d_batch = h.keys(h_batch);
    ASSERT_NE(d_first, nullptr);
    ASSERT_NE(d_batch, nullptr);

    ASSERT_NO_THROW(h.map.insert(d_first, h.values(kKeys, 0xAA), kKeys, stream_,
                                 sph::ShardedPerfectHashMap::OnExisting::ON_EXISTING_IGNORE));
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    EXPECT_NO_THROW(h.map.insert(d_batch, h.values(2 * kKeys, 0xBB), 2 * kKeys, stream_,
                                 sph::ShardedPerfectHashMap::OnExisting::ON_EXISTING_IGNORE));
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    const auto tags = h.tags_of(d_batch, 2 * kKeys, stream_);
    EXPECT_TRUE(ExistingKeyHarness::all_tagged(tags, 0xAA, 0, kKeys))
        << "the resident half should have kept its original value";
    EXPECT_TRUE(ExistingKeyHarness::all_tagged(tags, 0xBB, kKeys, 2 * kKeys))
        << "the fresh half should have landed";
    EXPECT_EQ(h.map.get_size(), 2 * kKeys)
        << "only the fresh keys should have grown the key count";
}

// Duplicates *within* one batch. This is the case that fails hardest without
// the filter: two equal keys make popcount(mask) == total unsatisfiable for
// every pilot, so the bucket reports pilot_failure and takes its unrelated
// neighbours in the same bucket down with it.
//
// The last occurrence wins, so under Update the second tag is what sticks.
TEST_F(SphEngineValidationTest, InsertWithinBatchDuplicates)
{
    constexpr std::int64_t kKeys = 1000;

    // First half and second half are the same keys; the tail adds distinct keys
    // that share a bucket with the duplicated ones (bucket = key % num_buckets),
    // so a regression that drops the whole bucket is visible.
    ExistingKeyHarness h(kExistingCapacity);
    const std::int64_t num_buckets = h.map.get_num_buckets();

    std::vector<KeyT> h_batch(3 * kKeys), h_neighbours(kKeys);
    for (std::int64_t i = 0; i < kKeys; ++i) {
        h_batch[static_cast<std::size_t>(i)]             = i + 1;
        h_batch[static_cast<std::size_t>(kKeys + i)]     = i + 1;
        h_batch[static_cast<std::size_t>(2 * kKeys + i)] = i + 1 + num_buckets;
        h_neighbours[static_cast<std::size_t>(i)]        = i + 1 + num_buckets;
    }

    KeyT* d_batch      = h.keys(h_batch);
    KeyT* d_neighbours = h.keys(h_neighbours);
    ASSERT_NE(d_batch, nullptr);
    ASSERT_NE(d_neighbours, nullptr);

    // Rows 0..kKeys tagged 0xAA, kKeys..3*kKeys tagged 0xBB, so the duplicate
    // pair carries different values and "last wins" is observable.
    auto* d_values = h.values(3 * kKeys, 0xBB);
    ASSERT_NE(d_values, nullptr);
    ASSERT_CUDA(cudaMemset(d_values, 0xAA,
                           static_cast<std::size_t>(kKeys * ExistingKeyHarness::kValueSize)));

    EXPECT_NO_THROW(h.map.insert(d_batch, d_values, 3 * kKeys, stream_,
                                 sph::ShardedPerfectHashMap::OnExisting::ON_EXISTING_UPDATE));
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    const auto tags = h.tags_of(d_batch, kKeys, stream_);
    EXPECT_TRUE(ExistingKeyHarness::all_tagged(tags, 0xBB, 0, kKeys))
        << "the last occurrence of a duplicated key should win under Update";

    EXPECT_TRUE(ExistingKeyHarness::all_tagged(h.tags_of(d_neighbours, kKeys, stream_),
                                               0xBB, 0, kKeys))
        << "a duplicate must not take its bucket neighbours down with it";

    // 2*kKeys distinct keys were submitted as 3*kKeys rows.
    EXPECT_EQ(h.map.get_size(), 2 * kKeys)
        << "the duplicate copy must not be counted as an inserted key";
}

// reset() rebuilds every buffer from scratch, so it clears the accumulated
// key count.
// build() had no failure path: the bucketization kernel incremented a bucket's
// count and wrote at that index with no bound, so a bucket receiving more keys
// than d_buckets reserves per bucket wrote past the end of its row.
//
// Keys are aimed at one bucket directly: bucket_hash is
// (global_seed * key) % num_buckets, so with global_seed == 1 every key
// congruent mod num_buckets shares a bucket.
TEST_F(SphEngineValidationTest, BuildReportsOverfullBucket)
{
    sph::ShardedPerfectHashMap m(kCapacity, kValueSize);
    const std::int64_t num_buckets = m.get_num_buckets();
    ASSERT_EQ(m.get_global_seed(), 1) << "this test derives bucket indices assuming seed 1";

    // Comfortably past max_keys_per_bucket (64) for bucket 0; every other key
    // gets its own bucket so the rest of the build is healthy.
    constexpr std::int64_t kOverfull = 200;
    std::vector<KeyT> h_keys;
    for (std::int64_t i = 1; i <= kOverfull; ++i) h_keys.push_back(num_buckets * i);
    for (std::int64_t b = 1; b < num_buckets; ++b) h_keys.push_back(num_buckets + b);
    const auto num_keys = static_cast<std::int64_t>(h_keys.size());

    KeyT* d_keys = upload(h_keys);
    ASSERT_TRUE(d_keys != nullptr);

    bool threw = false;
    try {
        m.build(d_keys, num_keys, stream_);
    } catch (const sph::SphInsertFailed& error) {
        threw = true;
        const auto& r = error.result();
        EXPECT_FALSE(r.ok());
        EXPECT_GT(r.failures.max_bucket_reached, 0) << "the over-full bucket was not reported";
        EXPECT_GT(r.keys_dropped, 0) << "keys were discarded but not counted";
        EXPECT_EQ(r.keys_submitted, num_keys);
        EXPECT_LE(r.keys_dropped, r.keys_submitted);
        EXPECT_EQ(m.get_size(), r.keys_inserted())
            << "curr_capacity must count only the keys that landed";
    }
    EXPECT_TRUE(threw) << "an over-full bucket should have been reported";

    // The damage was never confined to the over-full bucket. d_buckets lays the
    // per-bucket rows out contiguously, so writing past bucket 0's row lands in
    // the rows of buckets 1, 2, 3... — their key lists were silently replaced by
    // bucket 0's overflow, and they then built PTEs for keys that do not hash to
    // them. Every one-key bucket must still hold its own key.
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    std::int64_t* d_hashes = alloc<std::int64_t>(num_keys);
    ASSERT_TRUE(d_hashes != nullptr);
    EXPECT_NO_THROW(m.find_locations(d_keys, num_keys, d_hashes, stream_));
    ASSERT_CUDA(cudaStreamSynchronize(stream_));

    std::vector<std::int64_t> h_hashes(h_keys.size());
    ASSERT_CUDA(cudaMemcpy(h_hashes.data(), d_hashes, h_hashes.size() * sizeof(std::int64_t),
                           cudaMemcpyDeviceToHost));

    std::int64_t lost = 0;
    std::int64_t first_lost = -1;
    for (std::int64_t i = kOverfull; i < num_keys; ++i) {
        if (h_hashes[static_cast<std::size_t>(i)] < 0) {
            if (lost == 0) first_lost = h_keys[static_cast<std::size_t>(i)];
            ++lost;
        }
    }
    EXPECT_EQ(lost, 0)
        << lost << " key(s) in unrelated single-key buckets were lost to the overflow;"
        << " first missing key=" << first_lost;
}

// build() does not dedup, and no pilot can separate two copies of a key, so a
// duplicated key makes its bucket's pilot search unsatisfiable. That used to
// leave pilot == 0 — the bucket reads back as invalid and every key in it is
// gone, with nothing reported.
TEST_F(SphEngineValidationTest, BuildReportsDuplicateKeys)
{
    sph::ShardedPerfectHashMap m(kCapacity, kValueSize);
    ASSERT_EQ(m.get_global_seed(), 1) << "this test derives bucket indices assuming seed 1";

    constexpr std::int64_t kKeys = 512;
    std::vector<KeyT> h_keys(kKeys);
    for (std::int64_t i = 0; i < kKeys; ++i) h_keys[static_cast<std::size_t>(i)] = i + 1;
    h_keys[7] = h_keys[6];  // one duplicated key, one doomed bucket

    KeyT* d_keys = upload(h_keys);
    ASSERT_TRUE(d_keys != nullptr);

    bool threw = false;
    try {
        m.build(d_keys, kKeys, stream_);
    } catch (const sph::SphInsertFailed& error) {
        threw = true;
        const auto& r = error.result();
        EXPECT_GT(r.failures.pilot_failure, 0) << "the unsatisfiable bucket was not reported";
        EXPECT_GT(r.keys_dropped, 0);
        EXPECT_EQ(r.keys_submitted, kKeys);
        EXPECT_EQ(m.get_size(), r.keys_inserted());
    }
    EXPECT_TRUE(threw) << "a duplicate key should have been reported";
}

TEST_F(SphEngineValidationTest, ResetClearsCounters)
{
    EXPECT_NO_THROW(map_->reset(stream_, kCapacity));
    ASSERT_CUDA(cudaStreamSynchronize(stream_));
    EXPECT_EQ(map_->get_size(), 0);
}

TEST_F(SphEngineValidationTest, ExceptionsClassifyAtTheCApiBoundary)
{
    EXPECT_THROW(map_->insert(nullptr, nullptr, 4, stream_), std::invalid_argument);
    EXPECT_THROW(map_->insert(nullptr, nullptr, 4, stream_), sph::SphException);

    static_assert(std::is_base_of_v<std::invalid_argument, sph::SphInvalidArgument>);
    static_assert(std::is_base_of_v<std::bad_alloc, sph::SphOutOfMemory>);
    static_assert(std::is_base_of_v<std::runtime_error, sph::SphCudaError>);
    static_assert(std::is_base_of_v<sph::SphException, sph::SphInvalidArgument>);
    static_assert(std::is_base_of_v<sph::SphException, sph::SphOutOfMemory>);
    static_assert(std::is_base_of_v<sph::SphException, sph::SphCudaError>);
    // Dropped keys map to NVE_ERROR_RUNTIME, not to an argument or OOM code.
    static_assert(std::is_base_of_v<sph::SphError, sph::SphInsertFailed>);
    static_assert(std::is_base_of_v<std::runtime_error, sph::SphInsertFailed>);

    // The thrown object carries the originating site, not just a message.
    try {
        map_->insert(nullptr, nullptr, 4, stream_);
        ADD_FAILURE() << "insert() with null keys did not throw";
    } catch (const sph::SphInvalidArgument& e) {
        EXPECT_NE(std::string(e.what()).find("d_keys is null"), std::string::npos) << e.what();
        EXPECT_GT(e.line(), 0);
        EXPECT_NE(e.file(), nullptr);
    }
}

// ===========================================================================
// SPH_TEARDOWN: the non-throwing check used on destructor and release paths.
// A destructor cannot throw, so the warning handler is the only signal; these
// drive the macro with runtime and driver statuses and a real release call.
// ===========================================================================

// Installs a capturing warning handler for the test's lifetime.
struct WarningCapture {
    WarningCapture()
    {
        messages().clear();
        sph::set_warning_handler(+[](const char* message) { messages().push_back(message); });
    }
    ~WarningCapture() { sph::set_warning_handler(nullptr); }

    static std::vector<std::string>& messages()
    {
        static std::vector<std::string> list;
        return list;
    }
};

// Message part with a side effect: proves parts are only formatted on failure.
int g_parts_formatted = 0;
struct CountedPart {};
std::ostream& operator<<(std::ostream& os, CountedPart)
{
    ++g_parts_formatted;
    return os << "part";
}

struct ThrowingPart {};
std::ostream& operator<<(std::ostream&, ThrowingPart) { throw std::runtime_error("formatting failed"); }

TEST(SphTeardown, RuntimeSuccessIsSilentAndSkipsFormatting)
{
    WarningCapture capture;
    g_parts_formatted = 0;
    EXPECT_TRUE(SPH_TEARDOWN(cudaSuccess, CountedPart{}));
    EXPECT_EQ(g_parts_formatted, 0);
    EXPECT_TRUE(WarningCapture::messages().empty());
}

TEST(SphTeardown, RuntimeFailureReportsComposedMessage)
{
    WarningCapture capture;
    g_parts_formatted = 0;
    const int line = __LINE__ + 1;
    EXPECT_FALSE(SPH_TEARDOWN(cudaErrorInvalidValue, "shard ", 7, CountedPart{}));
    EXPECT_EQ(g_parts_formatted, 1);
    ASSERT_EQ(WarningCapture::messages().size(), 1u);
    const std::string& m = WarningCapture::messages().front();
    EXPECT_NE(m.find("SPH_TEARDOWN(cudaErrorInvalidValue) failed"), std::string::npos) << m;
    EXPECT_NE(m.find(cudaGetErrorString(cudaErrorInvalidValue)), std::string::npos) << m;
    EXPECT_NE(m.find("shard 7part"), std::string::npos) << m;
    EXPECT_NE(m.find(":" + std::to_string(line) + "]"), std::string::npos) << m;
}

TEST(SphTeardown, RuntimeUnloadingIsSilent)
{
    WarningCapture capture;
    g_parts_formatted = 0;
    EXPECT_FALSE(SPH_TEARDOWN(cudaErrorCudartUnloading, CountedPart{}));
    EXPECT_EQ(g_parts_formatted, 0);
    EXPECT_TRUE(WarningCapture::messages().empty());
}

TEST(SphTeardown, DriverStatuses)
{
    WarningCapture capture;
    g_parts_formatted = 0;
    EXPECT_TRUE(SPH_TEARDOWN(CUDA_SUCCESS, CountedPart{}));
    EXPECT_FALSE(SPH_TEARDOWN(CUDA_ERROR_DEINITIALIZED, CountedPart{}));  // driver already torn down
    EXPECT_EQ(g_parts_formatted, 0);
    EXPECT_TRUE(WarningCapture::messages().empty());

    EXPECT_FALSE(SPH_TEARDOWN(CUDA_ERROR_INVALID_VALUE, "handle ", 3));
    ASSERT_EQ(WarningCapture::messages().size(), 1u);
    const std::string& m = WarningCapture::messages().front();
    EXPECT_NE(m.find("SPH_TEARDOWN(CUDA_ERROR_INVALID_VALUE) failed"), std::string::npos) << m;
    const char* text = nullptr;
    cuGetErrorString(CUDA_ERROR_INVALID_VALUE, &text);
    EXPECT_NE(m.find(text ? text : "(unknown driver error)"), std::string::npos) << m;
    EXPECT_NE(m.find("handle 3"), std::string::npos) << m;
}

TEST(SphTeardown, FormattingExceptionIsSwallowed)
{
    WarningCapture capture;
    EXPECT_FALSE(SPH_TEARDOWN(cudaErrorInvalidValue, ThrowingPart{}));
    EXPECT_TRUE(WarningCapture::messages().empty());
}

TEST(SphTeardown, NoHandlerDropsTheReport)
{
    sph::set_warning_handler(nullptr);
    EXPECT_FALSE(SPH_TEARDOWN(cudaErrorInvalidValue, "nobody listening"));
    sph::report_warning("also dropped");
}

// Through a real release call: freeing a host pointer fails with
// cudaErrorInvalidValue, and the message names the expression.
TEST_F(SphEngineValidationTest, TeardownReportsFailedRelease)
{
    WarningCapture capture;
    EXPECT_TRUE(SPH_TEARDOWN(cudaFree(nullptr)));  // freeing null is a documented no-op
    EXPECT_TRUE(WarningCapture::messages().empty());

    int host_object = 0;
    EXPECT_FALSE(SPH_TEARDOWN(cudaFree(&host_object), "freeing a host pointer"));
    ASSERT_EQ(WarningCapture::messages().size(), 1u);
    const std::string& m = WarningCapture::messages().front();
    EXPECT_NE(m.find("SPH_TEARDOWN(cudaFree(&host_object)) failed"), std::string::npos) << m;
    EXPECT_NE(m.find("freeing a host pointer"), std::string::npos) << m;
    (void)cudaGetLastError();  // leave no pending error for the next test
}

}  // namespace
