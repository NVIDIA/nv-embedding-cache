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
#include <fcntl.h>
#include <unistd.h>

#include <buffer_wrapper.hpp>
#include <cuda_support.hpp>
#include <gpu_table.hpp>
#include <hierarchical_embedding_layer.hpp>
#include <linear_file_table.hpp>
#include <plugin/plugin_loader.hpp>
#include <thread_pool.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <future>
#include <limits>
#include <random>
#include <thread>
#include <vector>

using namespace nve;
using namespace nve::plugin;

namespace {

using KeyType = int64_t;
constexpr int64_t kRowElems = 32;
constexpr int64_t kRowBytes = kRowElems * static_cast<int64_t>(sizeof(float));
constexpr int64_t kNumRows = 4096;

float pattern(int64_t row, int64_t elem) { return static_cast<float>(row * kRowElems + elem); }

struct TempFile {
  std::string path;
  TempFile() {
    std::string tmpl = (std::filesystem::temp_directory_path() / "nve_linear_file_XXXXXX").string();
    int fd = mkstemp(tmpl.data());
    EXPECT_GE(fd, 0);
    ::close(fd);
    path = tmpl;
  }
  ~TempFile() { ::unlink(path.c_str()); }
};

void write_pattern_file(const std::string& path, int64_t num_rows) {
  std::vector<float> rows(static_cast<size_t>(num_rows * kRowElems));
  for (int64_t r = 0; r < num_rows; ++r)
    for (int64_t e = 0; e < kRowElems; ++e) rows[static_cast<size_t>(r * kRowElems + e)] = pattern(r, e);
  int fd = ::open(path.c_str(), O_WRONLY | O_TRUNC);
  ASSERT_GE(fd, 0);
  const size_t bytes = rows.size() * sizeof(float);
  ASSERT_EQ(::write(fd, rows.data(), bytes), static_cast<ssize_t>(bytes));
  ::fsync(fd);
  ::close(fd);
}

LinearFileTableConfig make_config(const std::string& path, bool direct_io = true) {
  LinearFileTableConfig cfg;
  cfg.key_size = sizeof(KeyType);
  cfg.max_value_size = kRowBytes;
  cfg.value_dtype = DataType_t::Float32;
  cfg.path = path;
  cfg.num_rows = kNumRows;
  cfg.direct_io = direct_io;
  return cfg;
}

// Drives a table through the Table interface with host buffers.
struct Harness {
  std::shared_ptr<LinearFileTable> table;
  context_ptr_t ctx;

  explicit Harness(const LinearFileTableConfig& cfg) : Harness(std::make_shared<LinearFileTable>(0, cfg)) {}

  // Another context on an existing table; a null pool means the default one.
  explicit Harness(std::shared_ptr<LinearFileTable> t, thread_pool_ptr_t pool = nullptr)
      : table(std::move(t)), ctx(table->create_execution_context(0, 0, std::move(pool), nullptr)) {}

  int64_t find(const std::vector<KeyType>& keys, std::vector<float>& out, std::vector<bitmask64_t>* mask = nullptr,
               std::vector<int64_t>* sizes = nullptr) {
    const auto n = static_cast<int64_t>(keys.size());
    out.assign(static_cast<size_t>(n * kRowElems), 0.0f);
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys.data(), keys.size() * sizeof(KeyType));
    auto vals_bw = std::make_shared<BufferWrapper<void>>(ctx, "values", out.data(), out.size() * sizeof(float));
    std::shared_ptr<BufferWrapper<bitmask64_t>> mask_bw;
    if (mask) mask_bw = std::make_shared<BufferWrapper<bitmask64_t>>(ctx, "mask", mask->data(), mask->size() * sizeof(bitmask64_t));
    std::shared_ptr<BufferWrapper<int64_t>> sizes_bw;
    if (sizes) sizes_bw = std::make_shared<BufferWrapper<int64_t>>(ctx, "sizes", sizes->data(), sizes->size() * sizeof(int64_t));
    table->reset_lookup_counter(ctx);
    table->find(ctx, n, keys_bw, mask_bw, kRowBytes, vals_bw, sizes_bw);
    int64_t hits = 0;
    table->get_lookup_counter(ctx, &hits);
    return hits;
  }

  void update(const std::vector<KeyType>& keys, const std::vector<float>& vals) {
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "ukeys", keys.data(), keys.size() * sizeof(KeyType));
    auto vals_bw = std::make_shared<BufferWrapper<const void>>(ctx, "uvals", vals.data(), vals.size() * sizeof(float));
    table->update(ctx, static_cast<int64_t>(keys.size()), keys_bw, kRowBytes, kRowBytes, vals_bw);
  }
};

bool bit(const std::vector<bitmask64_t>& mask, int64_t i) {
  return (mask[static_cast<size_t>(i / bitmask64::num_bits)] & bitmask64::single(i % bitmask64::num_bits)) != 0;
}

std::vector<KeyType> random_keys(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<KeyType> keys(n);
  for (auto& k : keys) k = static_cast<KeyType>(rng() % kNumRows);
  return keys;
}

void expect_rows(const std::vector<KeyType>& keys, const std::vector<float>& out) {
  for (size_t i = 0; i < keys.size(); ++i)
    for (int64_t e = 0; e < kRowElems; e += 3)
      ASSERT_FLOAT_EQ(out[i * kRowElems + static_cast<size_t>(e)], pattern(keys[i], e)) << "key " << keys[i] << " elem " << e;
}

}  // namespace

TEST(LinearFileTable, ConfigJsonRoundTrip) {
  auto cfg = make_config("/some/table.bin", false);
  cfg.offset = 8192;
  cfg.read_only = true;
  cfg.staging_bytes_per_thread = 1 << 18;
  cfg.coalesce_adjacent = true;
  cfg.io_engine = "aio";
  cfg.io_threads = 7;
  const nlohmann::json json = cfg;
  const LinearFileTableConfig parsed = json;
  EXPECT_EQ(parsed.path, cfg.path);
  EXPECT_EQ(parsed.offset, cfg.offset);
  EXPECT_EQ(parsed.num_rows, cfg.num_rows);
  EXPECT_EQ(parsed.read_only, cfg.read_only);
  EXPECT_EQ(parsed.direct_io, cfg.direct_io);
  EXPECT_EQ(parsed.staging_bytes_per_thread, cfg.staging_bytes_per_thread);
  EXPECT_EQ(parsed.coalesce_adjacent, cfg.coalesce_adjacent);
  EXPECT_EQ(parsed.io_engine, cfg.io_engine);
  EXPECT_EQ(parsed.io_threads, cfg.io_threads);
  EXPECT_EQ(parsed.max_value_size, cfg.max_value_size);
  EXPECT_EQ(static_cast<nlohmann::json>(parsed), json);

  auto bad = make_config("/some/table.bin");
  bad.offset = 100;
  EXPECT_THROW(bad.check(), std::exception);
  bad = make_config("");
  EXPECT_THROW(bad.check(), std::exception);
  bad = make_config("/some/table.bin");
  bad.staging_bytes_per_thread = 100;
  EXPECT_THROW(bad.check(), std::exception);
  bad.staging_bytes_per_thread = (1 << 18) + 100;  // large enough, but not block aligned
  EXPECT_THROW(bad.check(), std::exception);
  bad = make_config("/some/table.bin");
  bad.io_engine = "mmap";
  EXPECT_THROW(bad.check(), std::exception);
  bad = make_config("/some/table.bin");
  bad.io_threads = 0;
  EXPECT_THROW(bad.check(), std::exception);
}

TEST(LinearFileTable, ConfigRejectsOverflowingStagingSizes) {
  auto cfg = make_config("/some/table.bin");
  cfg.io_threads = 4;
  cfg.staging_bytes_per_thread = int64_t{1} << 62;
  EXPECT_THROW(cfg.check(), std::exception);

  cfg = make_config("/some/table.bin");
  cfg.io_threads = std::numeric_limits<int64_t>::max();
  EXPECT_THROW(cfg.check(), std::exception);
}

TEST(LinearFileTable, ConfigRejectsOverflowingFileSize) {
  constexpr int64_t max_int = std::numeric_limits<int64_t>::max();
  auto cfg = make_config("/some/table.bin");
  cfg.max_value_size = int64_t{1} << 24;
  cfg.staging_bytes_per_thread = cfg.max_value_size + 4096;  // smallest staging that holds a row
  cfg.num_rows = int64_t{1} << 40;  // the product wraps to exactly 0
  EXPECT_THROW(cfg.check(), std::exception);
  cfg.num_rows = int64_t{1} << 39;  // the product wraps negative
  EXPECT_THROW(cfg.check(), std::exception);

  // The last row may end exactly at INT64_MAX, offset included.
  cfg.offset = 4096;
  cfg.num_rows = (max_int - cfg.offset) / cfg.max_value_size;
  EXPECT_NO_THROW(cfg.check());
  cfg.num_rows += 1;
  EXPECT_THROW(cfg.check(), std::exception);
}

TEST(LinearFileTable, ConfigRejectsUnsupportedKeySizes) {
  auto cfg = make_config("/some/table.bin");
  for (const int64_t key_size : {1, 2}) {
    cfg.key_size = key_size;
    EXPECT_THROW(cfg.check(), std::exception) << "key_size " << key_size;
  }
  for (const int64_t key_size : {4, 8}) {
    cfg.key_size = key_size;
    EXPECT_NO_THROW(cfg.check()) << "key_size " << key_size;
  }
}

TEST(LinearFileTable, ConfigStagingBoundsIncludeAlignmentPadding) {
  constexpr uint64_t block_size = 4096;
  const auto max_bytes = static_cast<uint64_t>(std::numeric_limits<size_t>::max());
  auto cfg = make_config("/some/table.bin");
  cfg.io_threads = 3;
  // The product fits size_t, but adding one alignment block wraps it to zero.
  cfg.staging_bytes_per_thread = static_cast<int64_t>((max_bytes - (block_size - 1)) / 3);
  EXPECT_THROW(cfg.check(), std::exception);

  cfg.staging_bytes_per_thread -= block_size;
  EXPECT_NO_THROW(cfg.check());  // Validate the boundary without allocating the buffers.
}

TEST(LinearFileTable, FindFillsOnlyUnresolvedInRangeKeys) {
  TempFile file;
  write_pattern_file(file.path, kNumRows);
  Harness h(make_config(file.path));

  const std::vector<KeyType> keys{0, 1, 2, kNumRows - 1, 7, kNumRows /*out of range*/, 100, 100};
  std::vector<bitmask64_t> mask(1, 0);
  mask[0] |= bitmask64::single(2);  // index 2 already resolved by an upper tier
  std::vector<float> out;
  const int64_t hits = h.find(keys, out, &mask);
  EXPECT_EQ(hits, 6);
  for (int64_t i = 0; i < static_cast<int64_t>(keys.size()); ++i) {
    if (i == 2) {
      EXPECT_FLOAT_EQ(out[static_cast<size_t>(i * kRowElems)], 0.0f);  // untouched
      EXPECT_TRUE(bit(mask, i));
    } else if (i == 5) {
      EXPECT_FALSE(bit(mask, i));
    } else {
      EXPECT_TRUE(bit(mask, i));
      for (int64_t e = 0; e < kRowElems; ++e) EXPECT_FLOAT_EQ(out[static_cast<size_t>(i * kRowElems + e)], pattern(keys[static_cast<size_t>(i)], e));
    }
  }
}

TEST(LinearFileTable, RandomKeysDirectAndBufferedAgree) {
  TempFile file;
  write_pattern_file(file.path, kNumRows);
  Harness direct(make_config(file.path, true));
  Harness buffered(make_config(file.path, false));

  const auto keys = random_keys(2000, 7);
  std::vector<float> out_direct, out_buffered;
  EXPECT_EQ(direct.find(keys, out_direct), 2000);
  EXPECT_EQ(buffered.find(keys, out_buffered), 2000);
  expect_rows(keys, out_direct);
  EXPECT_EQ(out_direct, out_buffered);
}

TEST(LinearFileTable, TruncatedFileThrowsInsteadOfReturningStaleRows) {
  for (const bool direct : {true, false}) {
    SCOPED_TRACE(direct);
    TempFile file;
    write_pattern_file(file.path, kNumRows);
    Harness h(make_config(file.path, direct));
    std::vector<float> out;
    ASSERT_EQ(h.find({kNumRows - 1}, out), 1);  // fills the staging buffer with the real row

    // Cut the file inside row kNumRows - 2.
    ASSERT_EQ(::truncate(file.path.c_str(), (kNumRows - 2) * kRowBytes + kRowBytes / 2), 0);
    EXPECT_THROW(h.find({kNumRows - 1}, out), std::exception);
    EXPECT_THROW(h.find({kNumRows - 2}, out), std::exception);

    // Rows still in the file read fine, including one whose block now ends at EOF.
    const std::vector<KeyType> intact{0, kNumRows - 3};
    EXPECT_EQ(h.find(intact, out), 2);
    expect_rows(intact, out);
  }
}

TEST(LinearFileTable, ValueSizesWithoutMaskDistinguishMisses) {
  TempFile file;
  write_pattern_file(file.path, kNumRows);
  Harness h(make_config(file.path));
  const std::vector<KeyType> keys{1, kNumRows, -1, kNumRows - 1, 1};
  std::vector<int64_t> sizes(keys.size(), -1);
  std::vector<float> out;
  EXPECT_EQ(h.find(keys, out, nullptr, &sizes), 3);
  EXPECT_EQ(sizes, (std::vector<int64_t>{kRowBytes, 0, 0, kRowBytes, kRowBytes}));

  const std::vector<KeyType> missing{kNumRows, -1};
  sizes.assign(missing.size(), -1);
  EXPECT_EQ(h.find(missing, out, nullptr, &sizes), 0);
  EXPECT_EQ(sizes, (std::vector<int64_t>{0, 0}));
}

TEST(LinearFileTable, ValueSizesPreservePreviouslyResolvedEntries) {
  TempFile file;
  write_pattern_file(file.path, kNumRows);
  Harness h(make_config(file.path));
  const std::vector<KeyType> keys{1, 2, kNumRows, -1};
  std::vector<bitmask64_t> mask{bitmask64::single(0)};
  std::vector<int64_t> sizes{16, -1, -1, -1};
  std::vector<float> out;
  EXPECT_EQ(h.find(keys, out, &mask, &sizes), 1);
  EXPECT_EQ(sizes, (std::vector<int64_t>{16, kRowBytes, 0, 0}));
  EXPECT_FLOAT_EQ(out[kRowElems], pattern(2, 0));
  EXPECT_TRUE(bit(mask, 1));
  EXPECT_FALSE(bit(mask, 2));
  EXPECT_FALSE(bit(mask, 3));

  mask[0] = bitmask64::clip(~bitmask64_t{0}, static_cast<int64_t>(keys.size()));
  sizes.assign(keys.size(), 16);
  EXPECT_EQ(h.find(keys, out, &mask, &sizes), 0);
  EXPECT_EQ(sizes, std::vector<int64_t>(keys.size(), 16));
}

TEST(LinearFileTable, ExistenceCheckReportsValueSizes) {
  TempFile file;
  write_pattern_file(file.path, kNumRows);
  Harness h(make_config(file.path));
  const std::vector<KeyType> keys{1, kNumRows, -1};
  std::vector<int64_t> sizes(keys.size(), -1);
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(h.ctx, "keys", keys.data(), keys.size() * sizeof(KeyType));
  auto sizes_bw = std::make_shared<BufferWrapper<int64_t>>(h.ctx, "sizes", sizes.data(), sizes.size() * sizeof(int64_t));
  h.table->reset_lookup_counter(h.ctx);
  h.table->find(h.ctx, static_cast<int64_t>(keys.size()), keys_bw, nullptr, kRowBytes, nullptr, sizes_bw);
  EXPECT_EQ(sizes, (std::vector<int64_t>{kRowBytes, 0, 0}));
  int64_t hits = 0;
  h.table->get_lookup_counter(h.ctx, &hits);
  EXPECT_EQ(hits, 1);
}

TEST(LinearFileTable, CoalescesAdjacentRows) {
  TempFile file;
  write_pattern_file(file.path, kNumRows);
  auto cfg = make_config(file.path);
  cfg.staging_bytes_per_thread = 8192;  // forces several read batches per thread
  cfg.io_threads = 3;
  cfg.coalesce_adjacent = true;
  Harness h(cfg);
  std::vector<KeyType> keys(1000);
  for (size_t i = 0; i < keys.size(); ++i) keys[i] = static_cast<KeyType>(i * 2);  // every other row
  std::vector<float> out;
  EXPECT_EQ(h.find(keys, out), 1000);
  expect_rows(keys, out);
}

TEST(LinearFileTable, FindRejectsInvalidStrides) {
  TempFile file;
  write_pattern_file(file.path, kNumRows);
  Harness h(make_config(file.path));
  const std::vector<KeyType> keys{5, 77};
  std::vector<float> out(keys.size() * kRowElems, -1.f);
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(h.ctx, "keys", keys.data(), keys.size() * sizeof(KeyType));
  auto vals_bw = std::make_shared<BufferWrapper<void>>(h.ctx, "values", out.data(), out.size() * sizeof(float));
  for (const int64_t stride : {int64_t{-1}, int64_t{0}, kRowBytes - 1}) {
    SCOPED_TRACE(stride);
    EXPECT_THROW(h.table->find(h.ctx, 2, keys_bw, nullptr, stride, vals_bw, nullptr), std::exception);
    EXPECT_EQ(out, std::vector<float>(out.size(), -1.f));
  }
  EXPECT_THROW(h.table->find(h.ctx, 2, keys_bw, nullptr, -1, nullptr, nullptr), std::exception);
  h.table->reset_lookup_counter(h.ctx);
  EXPECT_NO_THROW(h.table->find(h.ctx, 2, keys_bw, nullptr, 0, nullptr, nullptr));
  int64_t hits = 0;
  h.table->get_lookup_counter(h.ctx, &hits);
  EXPECT_EQ(hits, 2);
}

TEST(LinearFileTable, UpdateRejectsInvalidStridesAndInsertIsNoOp) {
  TempFile file;
  write_pattern_file(file.path, kNumRows);
  Harness h(make_config(file.path));
  const std::vector<KeyType> keys{5, 77};
  const std::vector<float> values(keys.size() * kRowElems, 0.f);
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(h.ctx, "keys", keys.data(), keys.size() * sizeof(KeyType));
  auto vals_bw = std::make_shared<BufferWrapper<const void>>(h.ctx, "values", values.data(), values.size() * sizeof(float));
  for (const int64_t stride : {int64_t{-1}, int64_t{0}, kRowBytes - 1}) {
    SCOPED_TRACE(stride);
    EXPECT_NO_THROW(h.table->insert(h.ctx, 2, keys_bw, stride, kRowBytes, vals_bw));  // ignored, rows unchanged below
    EXPECT_THROW(h.table->update(h.ctx, 2, keys_bw, stride, kRowBytes, vals_bw), std::exception);
  }
  std::vector<float> out;
  ASSERT_EQ(h.find(keys, out), 2);
  expect_rows(keys, out);
}

TEST(LinearFileTable, PaddedStridesWork) {
  TempFile file;
  write_pattern_file(file.path, kNumRows);
  Harness h(make_config(file.path));
  const std::vector<KeyType> keys{5, 77};
  constexpr int64_t stride_elems = kRowElems + 1;
  constexpr int64_t stride = stride_elems * static_cast<int64_t>(sizeof(float));
  std::vector<float> values(keys.size() * stride_elems, -1.f);
  for (size_t i = 0; i < keys.size(); ++i) {
    for (int64_t e = 0; e < kRowElems; ++e) values[i * stride_elems + static_cast<size_t>(e)] = static_cast<float>(i + 1);
  }
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(h.ctx, "keys", keys.data(), keys.size() * sizeof(KeyType));
  auto vals_bw = std::make_shared<BufferWrapper<const void>>(h.ctx, "values", values.data(), values.size() * sizeof(float));
  h.table->insert(h.ctx, 2, keys_bw, stride, kRowBytes, vals_bw);
  h.table->update(h.ctx, 2, keys_bw, stride, kRowBytes, vals_bw);

  std::vector<float> out(values.size(), -1.f);
  auto out_bw = std::make_shared<BufferWrapper<void>>(h.ctx, "output", out.data(), out.size() * sizeof(float));
  h.table->find(h.ctx, 2, keys_bw, nullptr, stride, out_bw, nullptr);
  for (size_t i = 0; i < keys.size(); ++i) {
    for (int64_t e = 0; e < kRowElems; ++e) EXPECT_FLOAT_EQ(out[i * stride_elems + static_cast<size_t>(e)], static_cast<float>(i + 1));
    EXPECT_FLOAT_EQ(out[i * stride_elems + kRowElems], -1.f);
  }
}

TEST(LinearFileTable, UpdateReachesTheFile) {
  TempFile file;
  write_pattern_file(file.path, kNumRows);
  Harness h(make_config(file.path));

  const std::vector<KeyType> keys{5, 77};
  std::vector<float> vals(keys.size() * kRowElems);
  for (size_t i = 0; i < keys.size(); ++i)
    for (int64_t e = 0; e < kRowElems; ++e) vals[i * kRowElems + static_cast<size_t>(e)] = 10.0f * static_cast<float>(i + 1) + static_cast<float>(e);
  h.update(keys, vals);

  std::vector<float> out;
  EXPECT_EQ(h.find(keys, out), 2);
  EXPECT_EQ(out, vals);

  // A second table over the same file sees the writes (direct reads are coherent with buffered writes).
  Harness other(make_config(file.path));
  EXPECT_EQ(other.find(keys, out), 2);
  EXPECT_EQ(out, vals);
}

TEST(LinearFileTable, AccumulateIsNotImplemented) {
  TempFile file;
  write_pattern_file(file.path, kNumRows);
  Harness h(make_config(file.path));
  const std::vector<KeyType> keys{5};
  const std::vector<float> updates(kRowElems, 1.f);
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(h.ctx, "keys", keys.data(), keys.size() * sizeof(KeyType));
  auto updates_bw = std::make_shared<BufferWrapper<const void>>(h.ctx, "updates", updates.data(), updates.size() * sizeof(float));
  EXPECT_THROW(h.table->update_accumulate(h.ctx, 1, keys_bw, kRowBytes, kRowBytes, updates_bw, DataType_t::Float32),
               NotImplementedError);
  std::vector<float> out;
  ASSERT_EQ(h.find(keys, out), 1);
  expect_rows(keys, out);
}

// Lookups racing updates on other contexts must see each row entirely before or after a write.
TEST(LinearFileTable, FindNeverSeesPartialUpdates) {
  constexpr int64_t row_elems = 16384;  // 64 KiB rows span many pages
  constexpr int64_t row_bytes = row_elems * static_cast<int64_t>(sizeof(float));
  const std::vector<KeyType> keys{0, 1, 2, 3};
  for (const bool direct_io : {true, false}) {
    SCOPED_TRACE(direct_io);
    TempFile file;
    auto cfg = make_config(file.path, direct_io);
    cfg.max_value_size = row_bytes;
    cfg.num_rows = static_cast<int64_t>(keys.size());
    cfg.create_if_missing = true;
    auto table = std::make_shared<LinearFileTable>(0, cfg);

    std::atomic<bool> stop{false};
    std::thread writer([&] {
      Harness w(table);
      auto keys_bw = std::make_shared<BufferWrapper<const void>>(w.ctx, "keys", keys.data(), keys.size() * sizeof(KeyType));
      std::vector<float> rows(keys.size() * row_elems);
      for (int64_t it = 1; !stop.load(); ++it) {
        std::fill(rows.begin(), rows.end(), static_cast<float>(it));
        auto vals_bw = std::make_shared<BufferWrapper<const void>>(w.ctx, "values", rows.data(), rows.size() * sizeof(float));
        w.table->update(w.ctx, static_cast<int64_t>(keys.size()), keys_bw, row_bytes, row_bytes, vals_bw);
      }
    });

    std::atomic<int64_t> torn{0};
    std::atomic<int64_t> changes{0};  // proves updates landed while the readers ran
    std::vector<std::thread> readers;
    for (int t = 0; t < 2; ++t) {
      readers.emplace_back([&] {
        Harness r(table);
        auto keys_bw = std::make_shared<BufferWrapper<const void>>(r.ctx, "keys", keys.data(), keys.size() * sizeof(KeyType));
        std::vector<float> out(keys.size() * row_elems);
        float last = 0.f;
        auto out_bw = std::make_shared<BufferWrapper<void>>(r.ctx, "output", out.data(), out.size() * sizeof(float));
        for (int it = 0; it < 300; ++it) {
          r.table->find(r.ctx, static_cast<int64_t>(keys.size()), keys_bw, nullptr, row_bytes, out_bw, nullptr);
          for (size_t k = 0; k < keys.size(); ++k) {
            const auto row = out.begin() + static_cast<std::ptrdiff_t>(k * row_elems);
            if (std::any_of(row, row + row_elems, [&](float v) { return v != *row; })) ++torn;
          }
          if (out.front() != last) ++changes;
          last = out.front();
        }
      });
    }
    for (auto& t : readers) t.join();
    stop = true;
    writer.join();
    EXPECT_EQ(torn.load(), 0);
    EXPECT_GT(changes.load(), 0);
  }
}

TEST(LinearFileTable, ReadOnlyRejectsWrites) {
  TempFile file;
  write_pattern_file(file.path, kNumRows);
  auto cfg = make_config(file.path);
  cfg.read_only = true;
  Harness h(cfg);
  std::vector<float> out;
  EXPECT_EQ(h.find({1, 2}, out), 2);
  EXPECT_THROW(h.update({1}, std::vector<float>(kRowElems, 0.f)), std::exception);
}

TEST(LinearFileTable, CreateIfMissingStartsWithZeroRows) {
  TempFile file;
  ::unlink(file.path.c_str());
  auto missing = make_config(file.path);
  EXPECT_THROW(LinearFileTable(0, missing), std::exception);

  auto cfg = make_config(file.path);
  cfg.create_if_missing = true;
  Harness h(cfg);
  std::vector<float> out;
  EXPECT_EQ(h.find({0, kNumRows - 1}, out), 2);
  for (float v : out) EXPECT_FLOAT_EQ(v, 0.0f);
  h.update({kNumRows - 1}, std::vector<float>(kRowElems, 3.0f));
  EXPECT_EQ(h.find({kNumRows - 1}, out), 1);
  EXPECT_FLOAT_EQ(out[0], 3.0f);
}

TEST(LinearFileTable, PluginFactoryProducesWorkingTable) {
  TempFile file;
  write_pattern_file(file.path, kNumRows);
  nve::Plugin plugin("libnve-plugin-linear-file.so");
  auto factory = plugin.create_table_factory(nlohmann::json::object());
  const nlohmann::json cfg = make_config(file.path);
  auto table = factory->produce(0, cfg);
  ASSERT_NE(table, nullptr);
  EXPECT_EQ(table->get_device_id(), -1);
  EXPECT_EQ(table->get_max_row_size(), kRowBytes);

  auto ctx = table->create_execution_context(0, 0, nullptr, nullptr);
  const std::vector<KeyType> keys{42, 4000};
  std::vector<float> out(keys.size() * kRowElems, 0.f);
  auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys.data(), keys.size() * sizeof(KeyType));
  auto vals_bw = std::make_shared<BufferWrapper<void>>(ctx, "values", out.data(), out.size() * sizeof(float));
  table->find(ctx, static_cast<int64_t>(keys.size()), keys_bw, nullptr, kRowBytes, vals_bw, nullptr);
  expect_rows(keys, out);
}

// GPU cache in front of the file tier: the intended deployment.
TEST(LinearFileTable, HierarchyWithGpuCache) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) GTEST_SKIP() << "needs a GPU";
  TempFile file;
  write_pattern_file(file.path, kNumRows);

  GPUTableConfig gpu_cfg;
  gpu_cfg.device_id = 0;
  gpu_cfg.cache_size = 1 << 20;
  gpu_cfg.row_size_in_bytes = kRowBytes;
  gpu_cfg.value_dtype = DataType_t::Float32;
  gpu_cfg.uvm_table = nullptr;
  auto gpu_table = std::make_shared<GpuTable<KeyType>>(gpu_cfg, nullptr);
  auto file_table = std::make_shared<LinearFileTable>(1, make_config(file.path));

  HierarchicalEmbeddingLayer<KeyType>::Config layer_cfg;
  layer_cfg.layer_name = "gpu_plus_file";
  layer_cfg.min_insert_size_gpu = 1;
  auto layer = std::make_shared<HierarchicalEmbeddingLayer<KeyType>>(layer_cfg, std::vector<table_ptr_t>{gpu_table, file_table});
  auto ctx = layer->create_execution_context(0, 0, nullptr, nullptr);

  constexpr size_t n = 3000;
  void* keys_buf = nullptr;
  void* out_buf = nullptr;
  NVE_CHECK_(cudaMallocHost(&keys_buf, n * sizeof(KeyType)));
  NVE_CHECK_(cudaMallocHost(&out_buf, n * static_cast<size_t>(kRowBytes)));
  std::vector<float> hitrates(2);
  for (uint32_t pass = 0; pass < 3; ++pass) {
    const auto keys = random_keys(n, 11 + (pass % 2));
    std::memcpy(keys_buf, keys.data(), n * sizeof(KeyType));
    std::memset(out_buf, 0, n * static_cast<size_t>(kRowBytes));
    layer->lookup(ctx, static_cast<int64_t>(n), keys_buf, out_buf, kRowBytes, nullptr, nullptr, hitrates.data());
    NVE_CHECK_(cudaDeviceSynchronize());
    const float* out = static_cast<const float*>(out_buf);
    for (size_t i = 0; i < n; ++i)
      for (int64_t e = 0; e < kRowElems; e += 5)
        ASSERT_FLOAT_EQ(out[i * kRowElems + static_cast<size_t>(e)], pattern(keys[i], e)) << "pass " << pass << " key " << keys[i];
  }

  // Updates propagate to every tier, so the file tier sees them directly.
  const std::vector<KeyType> upd_keys{9, 4000};
  std::vector<float> upd_vals(upd_keys.size() * kRowElems);
  for (size_t i = 0; i < upd_vals.size(); ++i) upd_vals[i] = 500.0f + static_cast<float>(i);
  std::memcpy(keys_buf, upd_keys.data(), upd_keys.size() * sizeof(KeyType));
  std::memcpy(out_buf, upd_vals.data(), upd_vals.size() * sizeof(float));
  layer->update(ctx, static_cast<int64_t>(upd_keys.size()), keys_buf, kRowBytes, kRowBytes, out_buf, -1);
  NVE_CHECK_(cudaDeviceSynchronize());

  Harness check(make_config(file.path));
  std::vector<float> out;
  EXPECT_EQ(check.find(upd_keys, out), 2);
  EXPECT_EQ(out, upd_vals);

  cudaFreeHost(keys_buf);
  cudaFreeHost(out_buf);
}

// --- I/O engines ----------------------------------------------------------------------------------

namespace {

// Builds the table with the requested engine, or returns nullptr when this machine lacks it.
std::unique_ptr<Harness> make_harness(const LinearFileTableConfig& cfg) {
  try {
    return std::make_unique<Harness>(cfg);
  } catch (const std::exception& e) {
    if (cfg.io_engine != "auto" && std::string(e.what()).find("unavailable") != std::string::npos) return nullptr;
    throw;
  }
}

class LinearFileTableEngine : public ::testing::TestWithParam<std::string> {};

}  // namespace

TEST(LinearFileTable, AutoEngineReadsRows) {
  TempFile file;
  write_pattern_file(file.path, kNumRows);
  Harness h(make_config(file.path));
  const std::vector<KeyType> keys{0, 77, kNumRows - 1};
  std::vector<float> out;
  ASSERT_EQ(h.find(keys, out), static_cast<int64_t>(keys.size()));
  expect_rows(keys, out);
}

TEST_P(LinearFileTableEngine, RandomKeysMatchTheFile) {
  TempFile file;
  write_pattern_file(file.path, kNumRows);
  for (const bool direct : {true, false}) {
    for (const bool coalesce : {false, true}) {
      for (const int64_t io_threads : {1, 4}) {
        for (const int64_t staging : {int64_t{8192}, int64_t{1} << 20}) {  // 2 or 256 reads per batch
          auto cfg = make_config(file.path, direct);
          cfg.io_engine = GetParam();
          cfg.coalesce_adjacent = coalesce;
          cfg.io_threads = io_threads;
          cfg.staging_bytes_per_thread = staging;
          auto h = make_harness(cfg);
          if (!h) GTEST_SKIP() << GetParam() << " is unavailable here";
          SCOPED_TRACE(testing::Message() << "direct=" << direct << " coalesce=" << coalesce << " io_threads=" << io_threads
                                          << " staging=" << staging);

          // Every third key is already resolved upstream; one key is out of range.
          auto keys = random_keys(3000, 5);
          keys[17] = kNumRows;
          std::vector<bitmask64_t> mask(ceil_div<size_t>(keys.size(), bitmask64::num_bits), 0);
          for (size_t i = 0; i < keys.size(); i += 3) mask[i / bitmask64::num_bits] |= bitmask64::single(i % bitmask64::num_bits);
          std::vector<float> out;
          EXPECT_EQ(h->find(keys, out, &mask), 1999);
          for (size_t i = 0; i < keys.size(); ++i) {
            const auto idx = static_cast<int64_t>(i);
            if (i == 17) {
              EXPECT_FALSE(bit(mask, idx));
            } else if (i % 3 == 0) {
              EXPECT_FLOAT_EQ(out[i * kRowElems], 0.0f);  // untouched
            } else {
              ASSERT_TRUE(bit(mask, idx));
              for (int64_t e = 0; e < kRowElems; e += 7) ASSERT_FLOAT_EQ(out[i * kRowElems + static_cast<size_t>(e)], pattern(keys[i], e));
            }
          }
        }
      }
    }
  }
}

// 20-byte rows straddle 4 KiB blocks and the file ends mid-block, so direct reads come back short.
TEST_P(LinearFileTableEngine, UnalignedRowsAndPartialLastBlock) {
  constexpr int64_t kElems = 5;
  constexpr int64_t kRows = 1000;  // 20000 bytes
  TempFile file;
  std::vector<float> rows(kRows * kElems);
  for (size_t i = 0; i < rows.size(); ++i) rows[i] = static_cast<float>(i);
  {
    const int fd = ::open(file.path.c_str(), O_WRONLY | O_TRUNC);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(::write(fd, rows.data(), rows.size() * sizeof(float)), static_cast<ssize_t>(rows.size() * sizeof(float)));
    ::close(fd);
  }
  for (const bool direct : {true, false}) {
    auto cfg = make_config(file.path, direct);
    cfg.max_value_size = kElems * sizeof(float);
    cfg.num_rows = kRows;
    cfg.io_engine = GetParam();
    auto table = std::unique_ptr<LinearFileTable>();
    try {
      table = std::make_unique<LinearFileTable>(0, cfg);
    } catch (const std::exception& e) {
      if (std::string(e.what()).find("unavailable") != std::string::npos) GTEST_SKIP() << GetParam() << " is unavailable here";
      throw;
    }
    auto ctx = table->create_execution_context(0, 0, nullptr, nullptr);
    std::vector<KeyType> keys(kRows);
    for (int64_t k = 0; k < kRows; ++k) keys[static_cast<size_t>(k)] = kRows - 1 - k;  // last row first
    std::vector<float> out(rows.size(), -1.0f);
    auto keys_bw = std::make_shared<BufferWrapper<const void>>(ctx, "keys", keys.data(), keys.size() * sizeof(KeyType));
    auto vals_bw = std::make_shared<BufferWrapper<void>>(ctx, "values", out.data(), out.size() * sizeof(float));
    table->find(ctx, kRows, keys_bw, nullptr, kElems * sizeof(float), vals_bw, nullptr);
    for (int64_t k = 0; k < kRows; ++k) {
      for (int64_t e = 0; e < kElems; ++e) {
        ASSERT_FLOAT_EQ(out[static_cast<size_t>(k * kElems + e)], rows[static_cast<size_t>(keys[static_cast<size_t>(k)] * kElems + e)])
            << "direct=" << direct << " key " << keys[static_cast<size_t>(k)];
      }
    }
  }
}

// Lookups from several contexts at once share the table's readers and I/O pool.
size_t open_fd_count() {
  const std::filesystem::directory_iterator fds("/proc/self/fd");
  return static_cast<size_t>(std::distance(begin(fds), end(fds)));
}

// Each io_uring reader holds a ring fd, so a dropped reader shows up as one fd fewer.
TEST(LinearFileTable, FileErrorsKeepTheReader) {
  TempFile file;
  write_pattern_file(file.path, kNumRows);
  auto cfg = make_config(file.path, false);
  cfg.io_engine = "io_uring";
  cfg.io_threads = 1;
  auto h = make_harness(cfg);
  if (!h) GTEST_SKIP() << "io_uring is unavailable here";
  std::vector<float> out;
  ASSERT_EQ(h->find({0}, out), 1);

  ASSERT_EQ(::truncate(file.path.c_str(), (kNumRows - 1) * kRowBytes), 0);
  const size_t fds = open_fd_count();
  EXPECT_THROW(h->find({kNumRows - 1}, out), std::exception);
  EXPECT_EQ(open_fd_count(), fds);
  EXPECT_EQ(h->find({0}, out), 1);
  EXPECT_EQ(open_fd_count(), fds);
}

TEST_P(LinearFileTableEngine, ConcurrentLookups) {
  TempFile file;
  write_pattern_file(file.path, kNumRows);
  auto cfg = make_config(file.path);
  cfg.io_engine = GetParam();
  cfg.io_threads = 2;
  auto h = make_harness(cfg);
  if (!h) GTEST_SKIP() << GetParam() << " is unavailable here";

  std::vector<std::thread> threads;
  std::vector<int> failures(4, 0);
  for (size_t t = 0; t < failures.size(); ++t) {
    threads.emplace_back([&, t] {
      Harness local(h->table);
      for (uint32_t iter = 0; iter < 20; ++iter) {
        const auto keys = random_keys(500, static_cast<uint32_t>(100 * t) + iter);
        std::vector<float> out;
        if (local.find(keys, out) != 500) ++failures[t];
        for (size_t i = 0; i < keys.size(); ++i) {
          if (out[i * kRowElems + 1] != pattern(keys[i], 1)) ++failures[t];
        }
      }
    });
  }
  for (auto& t : threads) t.join();
  for (size_t t = 0; t < failures.size(); ++t) EXPECT_EQ(failures[t], 0) << "thread " << t;
}

TEST(LinearFileTable, InlineThreadPoolRunsAllPartsOnTheCaller) {
  TempFile file;
  write_pattern_file(file.path, kNumRows);
  auto cfg = make_config(file.path);
  cfg.io_engine = "pread";  // one read per part, so 500 keys split into io_threads parts
  cfg.io_threads = 4;
  Harness h(std::make_shared<LinearFileTable>(0, cfg), InlineThreadPool::instance());
  const auto keys = random_keys(500, 7);
  std::vector<float> out;
  ASSERT_EQ(h.find(keys, out), 500);
  expect_rows(keys, out);
}

TEST(LinearFileTable, LookupFromAPoolWorkerDoesNotDeadlock) {
  TempFile file;
  write_pattern_file(file.path, kNumRows);
  auto cfg = make_config(file.path);
  cfg.io_engine = "pread";
  cfg.io_threads = 4;
  SimpleThreadPoolConfig pool_cfg;
  pool_cfg.name = "linear_file_test_tp";
  pool_cfg.num_workers = 1;
  const thread_pool_ptr_t pool = std::make_shared<SimpleThreadPool>(pool_cfg);
  Harness h(std::make_shared<LinearFileTable>(0, cfg), pool);
  const auto keys = random_keys(500, 11);
  std::vector<float> out;
  // The only worker runs the lookup, so its helper parts can never start: the caller must do them all.
  auto done = pool->submit([&] { ASSERT_EQ(h.find(keys, out), 500); }, 0);
  ASSERT_EQ(done.wait_for(std::chrono::seconds(60)), std::future_status::ready);
  done.get();
  expect_rows(keys, out);
}

INSTANTIATE_TEST_SUITE_P(Engines, LinearFileTableEngine, ::testing::Values("io_uring", "aio", "pread"),
                         [](const ::testing::TestParamInfo<std::string>& info) { return info.param; });
