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

#include <host_table.hpp>
#include <nve_types.hpp>

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace nve {
namespace plugin {

struct ReadSpan;    // batch_reader.hpp
class BatchReader;  // batch_reader.hpp

// How a LinearFileTable issues its reads.
enum class IoEngine {
  IoUring,  // batched async reads through io_uring (Linux 5.6+)
  Aio,      // batched async reads through Linux AIO (async only with O_DIRECT)
  Pread,    // one blocking pread per read; reads in flight <= io_threads
};

// Owns a file descriptor (-1 when empty) and closes it on destruction.
class FileDescriptor final {
 public:
  FileDescriptor() = default;
  NVE_PREVENT_COPY_AND_MOVE_(FileDescriptor);
  ~FileDescriptor() { reset(); }

  int get() const { return fd_; }

  // Closes the current descriptor and takes ownership of `fd`.
  void reset(int fd = -1);

 private:
  int fd_{-1};
};

/**
 * Configuration of a LinearFileTable: a dense, row-major embedding table stored in a file.
 * Row `k` lives at byte `offset + k * max_value_size`. Only the rows a lookup misses are read,
 * so host memory is bounded by the per-thread staging buffers, not by the file size.
 */
struct LinearFileTableConfig final : public HostTableConfig {
  using base_type = HostTableConfig;

  std::string path;                      // File holding the rows.
  int64_t offset{0};                     // Byte offset of row 0, multiple of the 4 KiB block size.
  int64_t num_rows{0};                   // Rows in the table; keys outside [0, num_rows) are ignored.
  bool read_only{false};                 // Reject update.
  bool create_if_missing{false};         // Create or extend the file to hold num_rows rows (zeros).
  bool direct_io{true};                  // Read with O_DIRECT (no page cache); falls back to buffered reads
                                         // when the filesystem refuses it.
  std::string io_engine{"auto"};         // io_uring | aio | pread, or auto: the first of these that works.
  int64_t io_threads{4};                 // Most threads doing I/O for one call, the calling thread included; the
                                         // others are taken from the context's thread pool when it has idle workers.
  int64_t staging_bytes_per_thread{1 << 20};  // Read buffer per I/O thread, a multiple of 4 KiB; also bounds
                                              // the reads a thread has in flight (one per 4 KiB block). Each
                                              // pooled reader owns one; a table keeps a reader per I/O thread
                                              // that ran at once, at least io_threads.
  bool coalesce_adjacent{false};         // Sort the keys and merge reads whose 4 KiB blocks touch. Only pays
                                         // off when missed keys cluster; random keys rarely share a block.

  void check() const;
};

void from_json(const nlohmann::json& json, LinearFileTableConfig& conf);

void to_json(nlohmann::json& json, const LinearFileTableConfig& conf);

/**
 * Host tier over a file: find() reads the missed rows from the file, update() writes rows back.
 * insert() has no effect, since every row exists; update_accumulate() is not implemented.
 * Meant to sit behind a GpuTable in a HierarchicalEmbeddingLayer.
 *
 * Writes are not batched: update() costs one blocking pwrite per row, with at most io_threads in
 * flight. The table is tuned for lookups; update-heavy workloads will see far lower throughput than find().
 * An update() blocks lookups that read the file for its duration, and vice versa. Writes through another
 * table or process are not synchronized.
 */
class LinearFileTable final : public HostTable<LinearFileTableConfig> {
 public:
  using base_type = HostTable<LinearFileTableConfig>;

  NVE_PREVENT_COPY_AND_MOVE_(LinearFileTable);

  LinearFileTable() = delete;

  LinearFileTable(table_id_t id, const config_type& config);

  ~LinearFileTable() override;

  void clear(context_ptr_t& ctx) override;

  int64_t size(context_ptr_t& ctx, bool exact) const override;

  void erase(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys) override;

  void find(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys,
            buffer_ptr<bitmask64_t> hit_mask, int64_t value_stride,
            buffer_ptr<void> values, buffer_ptr<int64_t> value_sizes) const override;

  void insert(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, int64_t value_stride,
              int64_t value_size, buffer_ptr<const void> values) override;

  void update(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, int64_t value_stride,
              int64_t value_size, buffer_ptr<const void> values) override;

  void update_accumulate(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys,
                         int64_t update_stride, int64_t update_size,
                         buffer_ptr<const void> updates, DataType_t update_dtype) override;

 protected:
  int64_t* lookup_counter_storage(context_ptr_t& ctx) const override;

 private:
  template <typename KeyT>
  int64_t find_rows(context_ptr_t& ctx, int64_t n, const KeyT* keys, bitmask64_t* hit_mask,
                    int64_t value_stride, int8_t* values, int64_t* value_sizes) const;

  template <typename KeyT>
  void write_rows(context_ptr_t& ctx, int64_t n, const KeyT* keys, int64_t value_stride, const int8_t* values);

  // Calls fn(i, key) for each in-range key under the exclusive file lock and its row write lock,
  // split across the I/O threads.
  template <typename KeyT, typename Fn>
  void modify_rows(context_ptr_t& ctx, int64_t n, const KeyT* keys, const Fn& fn);

  // Reads [file_offset, file_offset + bytes) into `dst` (block aligned when direct I/O is used).
  // Returns the bytes read, fewer than `bytes` only at EOF.
  size_t read_span(off_t file_offset, size_t bytes, void* dst) const;

  // Completes a batched read the engine left short or failed. Returns the bytes valid from span.offset.
  size_t finish_read(ReadSpan& span) const;

  // Sends later reads through the page cache after the filesystem refused an aligned direct read.
  void stop_direct_reads() const;

  // Reports a row that lies past the end of the file (e.g. truncated after open).
  [[noreturn]] void throw_truncated(off_t row_end) const;

  // Readers are single-threaded; each I/O thread takes an idle one or creates a new one.
  std::unique_ptr<BatchReader> acquire_reader() const;
  void release_reader(std::unique_ptr<BatchReader> reader) const;

  // Runs task(0 .. num_parts - 1) on the calling thread and up to num_parts - 1 helpers from the context's
  // thread pool; rethrows the first failure.
  void run_parts(context_ptr_t& ctx, int64_t num_parts, const std::function<void(int64_t)>& task) const;

  size_t staging_bytes() const { return static_cast<size_t>(config_.staging_bytes_per_thread); }

  // True iff reads use the direct-read descriptor.
  bool uses_direct_io() const { return fd_direct_.get() >= 0 && !direct_reads_refused_.load(std::memory_order_relaxed); }

  // Descriptor reads start on: direct when available, buffered otherwise.
  int read_fd() const { return uses_direct_io() ? fd_direct_.get() : fd_buffered_.get(); }

  off_t row_offset(int64_t key) const {
    return static_cast<off_t>(config_.offset + key * config_.max_value_size);
  }

  std::mutex& row_write_mutex(int64_t key) {
    return row_write_mutexes_[static_cast<size_t>(key) % row_write_mutexes_.size()];
  }

  FileDescriptor fd_buffered_;
  FileDescriptor fd_direct_;  // empty without O_DIRECT
  // Set once an aligned direct read failed with EINVAL. fd_direct_ stays open: other threads may still use it.
  mutable std::atomic<bool> direct_reads_refused_{false};
  IoEngine engine_{IoEngine::Pread};
  size_t read_depth_{1};           // reads one thread batches (and keeps in flight)
  mutable std::mutex readers_mutex_;
  mutable std::vector<std::unique_ptr<BatchReader>> idle_readers_;
  mutable std::once_flag fallback_warned_;  // pread fallback warning
  // Lookups hold it shared and updates exclusively, so find() never returns a partly written row.
  mutable std::shared_mutex file_mutex_;
  // Bound lock storage while serializing writes to each row (an update's parts run in parallel).
  static constexpr size_t kNumRowWriteMutexes{256};
  std::array<std::mutex, kNumRowWriteMutexes> row_write_mutexes_;
  // Per-context scratch buffer, named once.
  std::string pending_buffer_;
};

struct LinearFileTableFactoryConfig final : public HostTableFactoryConfig {
  using base_type = HostTableFactoryConfig;
};

void from_json(const nlohmann::json& json, LinearFileTableFactoryConfig& conf);

void to_json(nlohmann::json& json, const LinearFileTableFactoryConfig& conf);

class LinearFileTableFactory final
    : public HostTableFactory<LinearFileTableFactoryConfig, LinearFileTableConfig> {
 public:
  using base_type = HostTableFactory<LinearFileTableFactoryConfig, LinearFileTableConfig>;

  NVE_PREVENT_COPY_AND_MOVE_(LinearFileTableFactory);

  LinearFileTableFactory() = delete;

  explicit LinearFileTableFactory(const config_type& config) : base_type(config) {}

  ~LinearFileTableFactory() override = default;

  host_table_ptr_t produce(table_id_t id, const table_config_type& config) override;
};

}  // namespace plugin
}  // namespace nve
