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
#include <cuda_runtime.h>
#include "cuda_ptr.hpp"
#include <algorithm>
#include <buffer_wrapper.hpp>
#include <cstdint>
#include <cstring>
#include <default_allocator.hpp>
#include <exception>
#include <thread>
#include <json_support.hpp>
#include <sph_table.hpp>
#include <vector>
#include <bit_ops.hpp>

using namespace nve;
using namespace nve::plugin;

// Config JSON round-trips through from_json/to_json without loss.
TEST(sph_plugin, parse_table_conf) {
  SphTableConfig conf{};
  conf.device_id = 0;
  conf.cache_size = 0;
  conf.initial_capacity = 8192;
  conf.row_size_in_bytes = 16;
  conf.value_dtype = DataType_t::Float32;
  conf.invalid_key = -1;
  conf.default_value.assign(static_cast<size_t>(conf.row_size_in_bytes), 0x5A);

  const nlohmann::json json = conf;
  const SphTableConfig parsed = json.get<SphTableConfig>();
  ASSERT_EQ(static_cast<nlohmann::json>(parsed), json);
}

// produce() -> insert -> find round-trip through the Table interface. The first
// 8 bytes of each value row encode the key, so a wrong payload is detectable.
// Parametrized over row size to cover every payload copy path: uint4 (%16),
// uint32_t (%4, not %16) and int8_t (odd sizes).
static void run_insert_find_roundtrip(int64_t row) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  constexpr int64_t N = 1024;

  SphTableConfig conf{};
  conf.initial_capacity = 8 * N;  // headroom
  conf.row_size_in_bytes = row;
  conf.value_dtype = DataType_t::Float32;

  SphTableFactory factory{SphTableFactoryConfig{}};
  table_ptr_t table = factory.produce(/*id=*/1, static_cast<nlohmann::json>(conf));
  ASSERT_NE(table, nullptr);
  EXPECT_EQ(table->get_key_size(), 8);
  EXPECT_EQ(table->get_max_row_size(), row);

  auto ctx = table->create_execution_context(/*lookup=*/0, /*modify=*/0, nullptr, nullptr);

  std::vector<int64_t> h_keys(N);
  std::vector<int8_t> h_vals(to_uint(N) * to_uint(row), 0);
  for (int64_t i = 0; i < N; ++i) {
    h_keys[to_uint(i)] = i + 1;  // avoid key 0
    // Fill the whole row deterministically from the key so a partial or
    // misaligned copy is caught by the full-row compare below.
    for (int64_t b = 0; b < row; ++b)
      h_vals[to_uint(i) * to_uint(row) + to_uint(b)] = static_cast<int8_t>((h_keys[to_uint(i)] + b * 31) & 0xff);
  }

  auto allocator = GetDefaultAllocator();
  void* d_keys = nullptr;
  void* d_vals = nullptr;
  void* d_out = nullptr;
  NVE_CHECK_(allocator->device_allocate(&d_keys, sizeof(int64_t) * to_uint(N)));
  NVE_CHECK_(allocator->device_allocate(&d_vals, to_uint(N) * to_uint(row)));
  NVE_CHECK_(allocator->device_allocate(&d_out, to_uint(N) * to_uint(row)));
  NVE_CHECK_(cudaMemcpy(d_keys, h_keys.data(), sizeof(int64_t) * to_uint(N), cudaMemcpyDefault));
  NVE_CHECK_(cudaMemcpy(d_vals, h_vals.data(), to_uint(N) * to_uint(row), cudaMemcpyDefault));

  {
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", d_keys, sizeof(int64_t) * to_uint(N));
    auto vals_bw = std::make_shared<BufferWrapper<const void>>(ctx, "values", d_vals, to_uint(N) * to_uint(row));
    table->insert(ctx, N, std::move(keys_bw), row, row, std::move(vals_bw));
  }
  NVE_CHECK_(cudaDeviceSynchronize());

  NVE_CHECK_(cudaMemset(d_out, 0, to_uint(N) * to_uint(row)));
  {
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", d_keys, sizeof(int64_t) * to_uint(N));
    auto out_bw = std::make_shared<BufferWrapper<void>>(ctx, "values_out", d_out, to_uint(N) * to_uint(row));
    table->find(ctx, N, std::move(keys_bw), nullptr /*hit_mask*/, row, std::move(out_bw), nullptr /*value_sizes*/);
  }
  NVE_CHECK_(cudaDeviceSynchronize());

  std::vector<int8_t> h_out(to_uint(N) * to_uint(row), 0);
  NVE_CHECK_(cudaMemcpy(h_out.data(), d_out, to_uint(N) * to_uint(row), cudaMemcpyDefault));

  int64_t bad = 0;
  for (int64_t i = 0; i < N; ++i) {
    if (std::memcmp(&h_out[to_uint(i) * to_uint(row)],
                    &h_vals[to_uint(i) * to_uint(row)],
                    to_uint(row)) != 0) { ++bad; }
  }
  EXPECT_EQ(bad, 0) << bad << " / " << N << " keys returned the wrong payload (row=" << row << ")";

  allocator->device_free(d_keys);
  allocator->device_free(d_vals);
  allocator->device_free(d_out);
}

// Re-inserting keys that are already in the table must not throw and must take
// the new value: SphTable::insert drives the engine with ON_EXISTING_UPDATE, so
// insert is idempotent and last-write-wins rather than a dropped batch.
static void run_insert_existing_keys(int64_t row) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  constexpr int64_t N = 1024;

  SphTableConfig conf{};
  conf.initial_capacity = 8 * N;
  conf.row_size_in_bytes = row;
  conf.value_dtype = DataType_t::Float32;

  SphTableFactory factory{SphTableFactoryConfig{}};
  table_ptr_t table = factory.produce(/*id=*/1, static_cast<nlohmann::json>(conf));
  ASSERT_NE(table, nullptr);
  auto ctx = table->create_execution_context(/*lookup=*/0, /*modify=*/0, nullptr, nullptr);

  // Same keys twice; the second pass shifts every byte so a stale row is visible.
  std::vector<int64_t> h_keys(N);
  std::vector<int8_t> h_v1(to_uint(N) * to_uint(row)), h_v2(to_uint(N) * to_uint(row));
  for (int64_t i = 0; i < N; ++i) {
    h_keys[to_uint(i)] = i + 1;  // avoid key 0
    for (int64_t b = 0; b < row; ++b) {
      const auto off = to_uint(i) * to_uint(row) + to_uint(b);
      h_v1[off] = static_cast<int8_t>((h_keys[to_uint(i)] + b * 31) & 0xff);
      h_v2[off] = static_cast<int8_t>((h_keys[to_uint(i)] + b * 31 + 77) & 0xff);
    }
  }

  auto allocator = GetDefaultAllocator();
  void* d_keys = nullptr;
  void* d_vals = nullptr;
  void* d_out = nullptr;
  NVE_CHECK_(allocator->device_allocate(&d_keys, sizeof(int64_t) * to_uint(N)));
  NVE_CHECK_(allocator->device_allocate(&d_vals, to_uint(N) * to_uint(row)));
  NVE_CHECK_(allocator->device_allocate(&d_out, to_uint(N) * to_uint(row)));
  NVE_CHECK_(cudaMemcpy(d_keys, h_keys.data(), sizeof(int64_t) * to_uint(N), cudaMemcpyDefault));

  auto insert_rows = [&](const std::vector<int8_t>& rows) {
    NVE_CHECK_(cudaMemcpy(d_vals, rows.data(), to_uint(N) * to_uint(row), cudaMemcpyDefault));
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", d_keys, sizeof(int64_t) * to_uint(N));
    auto vals_bw = std::make_shared<BufferWrapper<const void>>(ctx, "values", d_vals, to_uint(N) * to_uint(row));
    table->insert(ctx, N, std::move(keys_bw), row, row, std::move(vals_bw));
    NVE_CHECK_(cudaDeviceSynchronize());
  };

  ASSERT_NO_THROW(insert_rows(h_v1));
  ASSERT_NO_THROW(insert_rows(h_v2)) << "re-inserting resident keys must not throw";

  NVE_CHECK_(cudaMemset(d_out, 0, to_uint(N) * to_uint(row)));
  {
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", d_keys, sizeof(int64_t) * to_uint(N));
    auto out_bw = std::make_shared<BufferWrapper<void>>(ctx, "values_out", d_out, to_uint(N) * to_uint(row));
    table->find(ctx, N, std::move(keys_bw), nullptr /*hit_mask*/, row, std::move(out_bw), nullptr /*value_sizes*/);
  }
  NVE_CHECK_(cudaDeviceSynchronize());

  std::vector<int8_t> h_out(to_uint(N) * to_uint(row), 0);
  NVE_CHECK_(cudaMemcpy(h_out.data(), d_out, to_uint(N) * to_uint(row), cudaMemcpyDefault));

  int64_t bad = 0;
  for (int64_t i = 0; i < N; ++i) {
    if (std::memcmp(&h_out[to_uint(i) * to_uint(row)],
                    &h_v2[to_uint(i) * to_uint(row)],
                    to_uint(row)) != 0) { ++bad; }
  }
  EXPECT_EQ(bad, 0) << bad << " / " << N << " keys kept a stale payload (row=" << row << ")";

  allocator->device_free(d_keys);
  allocator->device_free(d_vals);
  allocator->device_free(d_out);
}

TEST(sph_plugin, insert_existing_keys)       { run_insert_existing_keys(16); }  // uint4 path
TEST(sph_plugin, insert_existing_keys_row33) { run_insert_existing_keys(33); }  // int8_t path

TEST(sph_plugin, insert_find_roundtrip)          { run_insert_find_roundtrip(16); }  // uint4 path
TEST(sph_plugin, insert_find_roundtrip_row12)    { run_insert_find_roundtrip(12); }  // uint32_t path
TEST(sph_plugin, insert_find_roundtrip_row20)    { run_insert_find_roundtrip(20); }  // uint32_t path
TEST(sph_plugin, insert_find_roundtrip_row33)    { run_insert_find_roundtrip(33); }  // int8_t path
TEST(sph_plugin, insert_find_roundtrip_row4)     { run_insert_find_roundtrip(4);  }  // uint32_t, row<8

// Insert `n` keys starting at `first_key` (row content derived from the key).
static void insert_batch(const table_ptr_t& table, context_ptr_t& ctx, int64_t first_key, int64_t n,
                         int64_t row, void* d_keys, void* d_vals) {
  std::vector<int64_t> h_keys(to_uint(n));
  std::vector<int8_t> h_vals(to_uint(n) * to_uint(row));
  for (int64_t i = 0; i < n; ++i) {
    h_keys[to_uint(i)] = first_key + i;
    for (int64_t b = 0; b < row; ++b)
      h_vals[to_uint(i) * to_uint(row) + to_uint(b)] = static_cast<int8_t>((h_keys[to_uint(i)] + b * 31) & 0xff);
  }
  NVE_CHECK_(cudaMemcpy(d_keys, h_keys.data(), sizeof(int64_t) * to_uint(n), cudaMemcpyDefault));
  NVE_CHECK_(cudaMemcpy(d_vals, h_vals.data(), to_uint(n) * to_uint(row), cudaMemcpyDefault));

  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", d_keys,
                                                             sizeof(int64_t) * to_uint(n));
  auto vals_bw = std::make_shared<BufferWrapper<const void>>(ctx, "values", d_vals,
                                                             to_uint(n) * to_uint(row));
  const_cast<table_ptr_t&>(table)->insert(ctx, n, std::move(keys_bw), row, row, std::move(vals_bw));
}

// Look every key up and count rows that don't match what insert_batch wrote.
static int64_t count_bad_rows(const table_ptr_t& table, context_ptr_t& ctx, int64_t first_key, int64_t n,
                              int64_t row, void* d_keys, void* d_out) {
  std::vector<int64_t> h_keys(to_uint(n));
  for (int64_t i = 0; i < n; ++i) h_keys[to_uint(i)] = first_key + i;
  NVE_CHECK_(cudaMemcpy(d_keys, h_keys.data(), sizeof(int64_t) * to_uint(n), cudaMemcpyDefault));
  NVE_CHECK_(cudaMemset(d_out, 0, to_uint(n) * to_uint(row)));
  {
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", d_keys,
                                                               sizeof(int64_t) * to_uint(n));
    auto out_bw = std::make_shared<BufferWrapper<void>>(ctx, "values_out", d_out,
                                                        to_uint(n) * to_uint(row));
    table->find(ctx, n, std::move(keys_bw), nullptr, row, std::move(out_bw), nullptr);
  }
  NVE_CHECK_(cudaDeviceSynchronize());

  std::vector<int8_t> h_out(to_uint(n) * to_uint(row));
  NVE_CHECK_(cudaMemcpy(h_out.data(), d_out, to_uint(n) * to_uint(row), cudaMemcpyDefault));

  int64_t bad = 0;
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t b = 0; b < row; ++b) {
      if (h_out[to_uint(i) * to_uint(row) + to_uint(b)] != static_cast<int8_t>((h_keys[to_uint(i)] + b * 31) & 0xff)) {
        ++bad;
        break;
      }
    }
  }
  return bad;
}

// find() on keys the table doesn't hold returns config.default_value, and the
// output buffer is fully overwritten (no leftovers from the caller's buffer).
// `default_value` empty means all-zero.
static void run_default_value_on_miss(int64_t row, bool explicit_default) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  constexpr int64_t N = 1024;

  // Distinct per byte so a shifted or partial copy of the default is caught.
  std::vector<int8_t> h_default(to_uint(row));
  for (int64_t b = 0; b < row; ++b)
    h_default[to_uint(b)] = explicit_default ? static_cast<int8_t>((0xA5 + b * 7) & 0xff) : 0;

  SphTableConfig conf{};
  conf.initial_capacity = 8 * N;
  conf.row_size_in_bytes = row;
  conf.value_dtype = DataType_t::Float32;
  if (explicit_default) conf.default_value.assign(h_default.begin(), h_default.end());

  SphTableFactory factory{SphTableFactoryConfig{}};
  table_ptr_t table = factory.produce(/*id=*/1, static_cast<nlohmann::json>(conf));
  auto ctx = table->create_execution_context(/*lookup=*/0, /*modify=*/0, nullptr, nullptr);

  auto allocator = GetDefaultAllocator();
  void* d_keys = nullptr;
  void* d_vals = nullptr;
  void* d_out = nullptr;
  NVE_CHECK_(allocator->device_allocate(&d_keys, sizeof(int64_t) * to_uint(N)));
  NVE_CHECK_(allocator->device_allocate(&d_vals, to_uint(N) * to_uint(row)));
  NVE_CHECK_(allocator->device_allocate(&d_out, to_uint(N) * to_uint(row)));

  insert_batch(table, ctx, /*first_key=*/1, N, row, d_keys, d_vals);
  NVE_CHECK_(cudaDeviceSynchronize());

  // Look up a disjoint key range; every row must come back as the default.
  std::vector<int64_t> h_miss(to_uint(N));
  for (int64_t i = 0; i < N; ++i) h_miss[to_uint(i)] = 1'000'000'000 + i;
  NVE_CHECK_(cudaMemcpy(d_keys, h_miss.data(), sizeof(int64_t) * to_uint(N), cudaMemcpyDefault));
  NVE_CHECK_(cudaMemset(d_out, 0xCC, to_uint(N) * to_uint(row)));
  {
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", d_keys, sizeof(int64_t) * to_uint(N));
    auto out_bw = std::make_shared<BufferWrapper<void>>(ctx, "values_out", d_out, to_uint(N) * to_uint(row));
    table->find(ctx, N, std::move(keys_bw), nullptr /*hit_mask*/, row, std::move(out_bw), nullptr /*value_sizes*/);
  }
  NVE_CHECK_(cudaDeviceSynchronize());

  std::vector<int8_t> h_out(to_uint(N) * to_uint(row), 0);
  NVE_CHECK_(cudaMemcpy(h_out.data(), d_out, to_uint(N) * to_uint(row), cudaMemcpyDefault));

  int64_t bad = 0;
  for (int64_t i = 0; i < N; ++i) {
    if (std::memcmp(&h_out[to_uint(i) * to_uint(row)], h_default.data(), to_uint(row)) != 0) ++bad;
  }
  EXPECT_EQ(bad, 0) << bad << " / " << N << " missing keys did not return the default value (row=" << row << ")";

  allocator->device_free(d_keys);
  allocator->device_free(d_vals);
  allocator->device_free(d_out);
}

TEST(sph_plugin, default_value_on_miss)        { run_default_value_on_miss(16, /*explicit_default=*/true); }
TEST(sph_plugin, default_value_on_miss_row33)  { run_default_value_on_miss(33, /*explicit_default=*/true); }
TEST(sph_plugin, zero_default_value_on_miss)   { run_default_value_on_miss(16, /*explicit_default=*/false); }

// A default_value whose length doesn't match row_size_in_bytes is rejected.
TEST(sph_plugin, mismatched_default_value_throws) {
  SphTableConfig conf{};
  conf.initial_capacity = 1024;
  conf.row_size_in_bytes = 16;
  conf.default_value.assign(8, 0);
  EXPECT_THROW(conf.check(), std::exception);
}

// Incremental inserts fragment the payload. With auto-defrag on, insert() must
// bring the table back under the threshold without losing any key.
static void run_defrag(double threshold, bool manual) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  constexpr int64_t kBatch = 64 * 1024;
  constexpr int64_t kNumBatches = 16;
  constexpr int64_t kTotal = kBatch * kNumBatches;
  constexpr int64_t row = 16;

  SphTableConfig conf{};
  conf.initial_capacity = 2 * kTotal;
  conf.row_size_in_bytes = row;
  conf.value_dtype = DataType_t::Float32;
  conf.defrag_threshold = manual ? 0.0 : threshold;

  SphTableFactory factory{SphTableFactoryConfig{}};
  table_ptr_t table = factory.produce(/*id=*/3, static_cast<nlohmann::json>(conf));
  auto* sph_table = dynamic_cast<SphTable*>(table.get());
  ASSERT_NE(sph_table, nullptr);
  auto ctx = table->create_execution_context(0, 0, nullptr, nullptr);

  auto allocator = GetDefaultAllocator();
  void* d_keys = nullptr;
  void* d_vals = nullptr;
  void* d_out = nullptr;
  NVE_CHECK_(allocator->device_allocate(&d_keys, sizeof(int64_t) * to_uint(kBatch)));
  NVE_CHECK_(allocator->device_allocate(&d_vals, to_uint(kBatch) * to_uint(row)));
  NVE_CHECK_(allocator->device_allocate(&d_out, to_uint(kBatch) * to_uint(row)));

  for (int64_t b = 0; b < kNumBatches; ++b) {
    insert_batch(table, ctx, /*first_key=*/1 + b * kBatch, kBatch, row, d_keys, d_vals);
  }
  NVE_CHECK_(cudaDeviceSynchronize());

  const double frag_before = sph_table->get_fragmentation();
  if (manual) {
    sph_table->defrag(ctx);
    EXPECT_LE(sph_table->get_fragmentation(), frag_before);
  } else {
    // Auto-defrag ran inside the last over-threshold insert().
    EXPECT_LE(sph_table->get_fragmentation(), threshold);
  }

  // Every key inserted (in any batch) must survive the compaction.
  int64_t bad = 0;
  for (int64_t b = 0; b < kNumBatches; ++b) {
    bad += count_bad_rows(table, ctx, /*first_key=*/1 + b * kBatch, kBatch, row, d_keys, d_out);
  }
  EXPECT_EQ(bad, 0) << bad << " / " << kTotal << " keys lost or corrupted by defrag";

  allocator->device_free(d_keys);
  allocator->device_free(d_vals);
  allocator->device_free(d_out);
}

TEST(sph_plugin, auto_defrag_on_insert) { run_defrag(1.05, /*manual=*/false); }
TEST(sph_plugin, manual_defrag) { run_defrag(0.0, /*manual=*/true); }

// A positive threshold below 1.0 is unreachable (fragmentation is >= 1.0).
TEST(sph_plugin, invalid_defrag_threshold_throws) {
  SphTableConfig conf{};
  conf.initial_capacity = 1024;
  conf.row_size_in_bytes = 16;
  conf.defrag_threshold = 0.5;
  EXPECT_ANY_THROW(conf.check());
}

// erase and update_accumulate are deliberately unsupported (throw).
// find() honours the in/out hit mask: keys whose bit is already set are skipped
// (their output row is left untouched for the tier that resolved them), and a bit
// is reported for every key SPH resolves. Also exercises value_stride != row.
static void run_hit_mask(int64_t row, int64_t value_stride) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  constexpr int64_t N = 1024;
  constexpr int64_t kMissKeyBase = 1'000'000;
  constexpr auto kMaskBits = static_cast<int64_t>(sizeof(bitmask64_t) * 8);
  const int64_t mask_words = nve::ceil_div(N, kMaskBits);

  SphTableConfig conf{};
  conf.initial_capacity = 8 * N;
  conf.row_size_in_bytes = row;
  conf.value_dtype = DataType_t::Float32;

  SphTableFactory factory{SphTableFactoryConfig{}};
  table_ptr_t table = factory.produce(/*id=*/1, static_cast<nlohmann::json>(conf));
  ASSERT_NE(table, nullptr);
  auto ctx = table->create_execution_context(/*lookup=*/0, /*modify=*/0, nullptr, nullptr);

  auto allocator = GetDefaultAllocator();
  void* d_keys = nullptr;
  void* d_vals = nullptr;
  void* d_probe = nullptr;
  void* d_out = nullptr;
  void* d_mask = nullptr;
  const size_t out_bytes = to_uint(N) * to_uint(value_stride);
  NVE_CHECK_(allocator->device_allocate(&d_keys, sizeof(int64_t) * to_uint(N)));
  NVE_CHECK_(allocator->device_allocate(&d_vals, to_uint(N) * to_uint(row)));
  NVE_CHECK_(allocator->device_allocate(&d_probe, sizeof(int64_t) * to_uint(N)));
  NVE_CHECK_(allocator->device_allocate(&d_out, out_bytes));
  NVE_CHECK_(allocator->device_allocate(&d_mask, sizeof(bitmask64_t) * to_uint(mask_words)));

  insert_batch(table, ctx, /*first_key=*/1, N, row, d_keys, d_vals);
  NVE_CHECK_(cudaDeviceSynchronize());

  // Probe: even indices hit, odd indices miss. Every third index is pre-resolved.
  auto masked = [](int64_t i) { return (i % 3) == 0; };
  std::vector<int64_t> h_probe(N);
  std::vector<bitmask64_t> h_mask(to_uint(mask_words), 0);
  for (int64_t i = 0; i < N; ++i) {
    h_probe[to_uint(i)] = (i % 2 == 0) ? (i + 1) : (kMissKeyBase + i);
    if (masked(i)) h_mask[to_uint(i / kMaskBits)] |= bitmask64_t{1} << (i % kMaskBits);
  }
  NVE_CHECK_(cudaMemcpy(d_probe, h_probe.data(), sizeof(int64_t) * to_uint(N), cudaMemcpyDefault));
  NVE_CHECK_(cudaMemcpy(d_mask, h_mask.data(), sizeof(bitmask64_t) * to_uint(mask_words),
                        cudaMemcpyDefault));
  NVE_CHECK_(cudaMemset(d_out, 0xCC, out_bytes));

  {
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", d_probe,
                                                              sizeof(int64_t) * to_uint(N));
    auto mask_bw = std::make_shared<BufferWrapper<bitmask64_t>>(
        ctx, "hitmask", reinterpret_cast<bitmask64_t*>(d_mask),
        sizeof(bitmask64_t) * to_uint(mask_words));
    auto out_bw = std::make_shared<BufferWrapper<void>>(ctx, "values_out", d_out, out_bytes);
    table->find(ctx, N, std::move(keys_bw), std::move(mask_bw), value_stride, std::move(out_bw),
                nullptr /*value_sizes*/);
  }
  NVE_CHECK_(cudaDeviceSynchronize());

  std::vector<int8_t> h_out(out_bytes, 0);
  std::vector<bitmask64_t> h_out_mask(to_uint(mask_words), 0);
  NVE_CHECK_(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDefault));
  NVE_CHECK_(cudaMemcpy(h_out_mask.data(), d_mask, sizeof(bitmask64_t) * to_uint(mask_words),
                        cudaMemcpyDefault));

  const std::vector<int8_t> untouched(to_uint(row), static_cast<int8_t>(0xCC));
  const std::vector<int8_t> zeros(to_uint(row), 0);
  int64_t bad_skipped = 0, bad_hit = 0, bad_miss = 0, bad_bit = 0;
  for (int64_t i = 0; i < N; ++i) {
    const int8_t* got = &h_out[to_uint(i) * to_uint(value_stride)];
    const bool bit = ((h_out_mask[to_uint(i / kMaskBits)] >> (i % kMaskBits)) & bitmask64_t{1}) != 0;
    if (masked(i)) {
      if (std::memcmp(got, untouched.data(), to_uint(row)) != 0) ++bad_skipped;
      if (!bit) ++bad_bit;
    } else if (i % 2 == 0) {
      std::vector<int8_t> want(to_uint(row));
      for (int64_t b = 0; b < row; ++b) want[to_uint(b)] = static_cast<int8_t>(((i + 1) + b * 31) & 0xff);
      if (std::memcmp(got, want.data(), to_uint(row)) != 0) ++bad_hit;
      if (!bit) ++bad_bit;
    } else {
      if (std::memcmp(got, zeros.data(), to_uint(row)) != 0) ++bad_miss;
      if (bit) ++bad_bit;
    }
  }
  EXPECT_EQ(bad_skipped, 0) << bad_skipped << " pre-resolved rows were overwritten (row=" << row << ")";
  EXPECT_EQ(bad_hit, 0) << bad_hit << " hits returned the wrong payload (row=" << row << ")";
  EXPECT_EQ(bad_miss, 0) << bad_miss << " misses did not get the default row (row=" << row << ")";
  EXPECT_EQ(bad_bit, 0) << bad_bit << " keys reported the wrong hit bit (row=" << row << ")";

  allocator->device_free(d_keys);
  allocator->device_free(d_vals);
  allocator->device_free(d_probe);
  allocator->device_free(d_out);
  allocator->device_free(d_mask);
}

TEST(sph_plugin, hit_mask)                 { run_hit_mask(16, 16); }  // uint4 path
TEST(sph_plugin, hit_mask_row12)           { run_hit_mask(12, 12); }  // uint32_t path
TEST(sph_plugin, hit_mask_row33)           { run_hit_mask(33, 33); }  // int8_t path
TEST(sph_plugin, hit_mask_strided)         { run_hit_mask(16, 24); }  // value_stride != row
TEST(sph_plugin, hit_mask_strided_row12)   { run_hit_mask(12, 32); }

TEST(sph_plugin, unsupported_ops_throw) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  SphTableConfig conf{};
  conf.initial_capacity = 1024;
  conf.row_size_in_bytes = 16;

  SphTableFactory factory{SphTableFactoryConfig{}};
  table_ptr_t table = factory.produce(/*id=*/2, static_cast<nlohmann::json>(conf));
  auto ctx = table->create_execution_context(0, 0, nullptr, nullptr);

  EXPECT_ANY_THROW(table->erase(ctx, 1, nullptr));
  EXPECT_ANY_THROW(
      table->update_accumulate(ctx, 1, nullptr, 16, 16, nullptr, DataType_t::Float32));
  EXPECT_THROW(table->size(ctx), NotImplementedError);
  int64_t key{};
  EXPECT_THROW(table->get_keys(ctx, 1, &key), NotImplementedError);
}

// The Table contract allows several execution contexts to be in use at once,
// while the engine forbids two modifies from overlapping (they share scratch
// pools and readback counters). SphTable has to serialize them: concurrent
// inserts from two contexts on disjoint key ranges must all land intact.
TEST(sph_plugin, concurrent_inserts_are_serialized) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device unavailable";
  }

  constexpr int kThreads = 4;
  constexpr int64_t kBatch = 16 * 1024;
  constexpr int64_t kBatchesPerThread = 8;
  constexpr int64_t kPerThread = kBatch * kBatchesPerThread;
  constexpr int64_t row = 16;

  SphTableConfig conf{};
  conf.initial_capacity = 2 * kThreads * kPerThread;
  conf.row_size_in_bytes = row;
  conf.value_dtype = DataType_t::Float32;
  conf.defrag_threshold = 1.05;  // let auto-defrag kick in under contention too

  SphTableFactory factory{SphTableFactoryConfig{}};
  table_ptr_t table = factory.produce(/*id=*/5, static_cast<nlohmann::json>(conf));
  auto allocator = GetDefaultAllocator();

  // Exceptions must not escape a std::thread (that is std::terminate); collect
  // them and report on the test thread after joining.
  std::vector<std::exception_ptr> worker_errors(kThreads);
  std::vector<std::thread> workers;
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&, t]() {
     try {
      // Owning handles (cuda_ptr.hpp) release on unwind. Declaration order
      // matters: buffers and ctx are torn down before the stream.
      cudaStream_t raw_stream = nullptr;
      NVE_CHECK_(cudaStreamCreateWithFlags(&raw_stream, cudaStreamNonBlocking));
      StreamPtr stream(raw_stream);
      auto ctx = table->create_execution_context(stream.get(), stream.get(), nullptr, nullptr);
      void* raw_keys = nullptr;
      void* raw_vals = nullptr;
      NVE_CHECK_(cudaMalloc(&raw_keys, sizeof(int64_t) * to_uint(kBatch)));
      DevicePtr<void> d_keys(raw_keys);
      NVE_CHECK_(cudaMalloc(&raw_vals, to_uint(kBatch) * to_uint(row)));
      DevicePtr<void> d_vals(raw_vals);
      for (int64_t b = 0; b < kBatchesPerThread; ++b) {
        insert_batch(table, ctx, /*first_key=*/1 + t * kPerThread + b * kBatch, kBatch, row, d_keys.get(),
                     d_vals.get());
      }
      NVE_CHECK_(cudaStreamSynchronize(stream.get()));
     } catch (...) {
      worker_errors[to_uint(t)] = std::current_exception();
     }
    });
  }
  for (auto& w : workers) w.join();
  for (int t = 0; t < kThreads; ++t) {
    if (!worker_errors[to_uint(t)]) continue;
    try {
      std::rethrow_exception(worker_errors[to_uint(t)]);
    } catch (const std::exception& e) {
      ADD_FAILURE() << "worker " << t << " threw: " << e.what();
    } catch (...) {
      ADD_FAILURE() << "worker " << t << " threw a non-std exception";
    }
  }
  if (::testing::Test::HasFailure()) return;

  auto ctx = table->create_execution_context(0, 0, nullptr, nullptr);
  void* d_keys = nullptr;
  void* d_out = nullptr;
  NVE_CHECK_(allocator->device_allocate(&d_keys, sizeof(int64_t) * to_uint(kBatch)));
  NVE_CHECK_(allocator->device_allocate(&d_out, to_uint(kBatch) * to_uint(row)));
  int64_t bad = 0;
  for (int t = 0; t < kThreads; ++t) {
    for (int64_t b = 0; b < kBatchesPerThread; ++b) {
      bad += count_bad_rows(table, ctx, /*first_key=*/1 + t * kPerThread + b * kBatch, kBatch, row, d_keys, d_out);
    }
  }
  EXPECT_EQ(bad, 0) << bad << " / " << kThreads * kPerThread << " keys lost or corrupted by concurrent inserts";
  allocator->device_free(d_keys);
  allocator->device_free(d_out);
}
