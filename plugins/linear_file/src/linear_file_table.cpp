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

#include <linear_file_table.hpp>

#include "batch_reader.hpp"

#include <bit_ops.hpp>
#include <buffer_wrapper.hpp>
#include <cuda_support.hpp>
#include <execution_context.hpp>
#include <json_support.hpp>
#include <key_utils.hpp>
#include <thread_pool.hpp>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace nve {
namespace plugin {

namespace {

constexpr int64_t kMaxReadDepth = 4096;  // reads one thread batches (io_uring rings allow up to 32768)
constexpr int64_t kMinReadsPerThread = 32;  // batched engines wake another I/O thread only for this many reads

size_t staging_read_depth(int64_t staging_bytes) {
  return static_cast<size_t>(std::clamp<int64_t>(staging_bytes / kBlockSize, 1, kMaxReadDepth));
}

struct PendingRow {
  int64_t key;
  int64_t idx;  // position in the lookup
};

IoEngine parse_io_engine(const std::string& name) {
  if (name == "io_uring") return IoEngine::IoUring;
  if (name == "aio") return IoEngine::Aio;
  if (name == "pread") return IoEngine::Pread;
  NVE_THROW_("LinearFileTable: unknown io_engine '", name, "' (auto|io_uring|aio|pread)");
}

off_t align_down(off_t value, off_t alignment) {
  return value / alignment * alignment;
}

int8_t* align_pointer(void* ptr, size_t alignment) {
  const auto addr = reinterpret_cast<uintptr_t>(ptr);
  return reinterpret_cast<int8_t*>((addr + alignment - 1) / alignment * alignment);
}

bool block_aligned(off_t offset, size_t bytes, const void* ptr) {
  return offset % kBlockSize == 0 && bytes % kBlockSize == 0 && reinterpret_cast<uintptr_t>(ptr) % kBlockSize == 0;
}

// False if the filesystem accepted O_DIRECT at open() but refuses aligned direct reads.
bool direct_reads_work(const int fd, const off_t offset) {
  std::vector<int8_t> buffer(2 * kBlockSize);
  ssize_t got;
  do {
    got = ::pread(fd, align_pointer(buffer.data(), kBlockSize), kBlockSize, offset);
  } while (got < 0 && errno == EINTR);
  return got >= 0 || errno != EINVAL;
}

std::string errno_message(const char* what) {
  return std::string(what) + " failed: " + std::strerror(errno);
}

// Full write of `bytes` at `offset`, retrying short writes.
void pwrite_all(int fd, const void* src, size_t bytes, off_t offset) {
  size_t done = 0;
  while (done < bytes) {
    const ssize_t put = ::pwrite(fd, static_cast<const char*>(src) + done, bytes - done, offset + static_cast<off_t>(done));
    if (put < 0) {
      if (errno == EINTR) continue;
      NVE_THROW_("LinearFileTable: ", errno_message("pwrite"));
    }
    done += static_cast<size_t>(put);
  }
}

const char* io_engine_name(const IoEngine engine) {
  switch (engine) {
    case IoEngine::IoUring:
      return "io_uring";
    case IoEngine::Aio:
      return "aio";
    case IoEngine::Pread:
      return "pread";
  }
  return "unknown";
}

}  // namespace

// --- config ---------------------------------------------------------------------------------------

void LinearFileTableConfig::check() const {
  base_type::check();
  NVE_CHECK_(key_size == sizeof(int32_t) || key_size == sizeof(int64_t),
             "LinearFileTable: key_size must be 4 or 8, got ", key_size);
  NVE_CHECK_(!path.empty(), "LinearFileTable: path must not be empty");
  NVE_CHECK_(offset >= 0 && offset % kBlockSize == 0,
             "LinearFileTable: offset must be a non-negative multiple of ", kBlockSize);
  NVE_CHECK_(num_rows > 0, "LinearFileTable: num_rows must be positive");
  // Row offsets and the file size are offset + num_rows * max_value_size (max_value_size > 0 per base check).
  NVE_CHECK_(num_rows <= (std::numeric_limits<int64_t>::max() - offset) / max_value_size,
             "LinearFileTable: offset ", offset, " + ", num_rows, " rows of ", max_value_size, " bytes overflows int64");
  if (io_engine != "auto") parse_io_engine(io_engine);
  NVE_CHECK_(io_threads > 0, "LinearFileTable: io_threads must be positive");
  NVE_CHECK_(!(create_if_missing && read_only), "LinearFileTable: create_if_missing requires a writable table");
  const int64_t min_staging = round_up(max_value_size, kBlockSize) + kBlockSize;
  NVE_CHECK_(staging_bytes_per_thread >= min_staging,
             "LinearFileTable: staging_bytes_per_thread must be at least ", min_staging, " for rows of ", max_value_size, " bytes");
  // Each reader allocates its staging block aligned for O_DIRECT, which needs whole blocks.
  NVE_CHECK_(staging_bytes_per_thread % kBlockSize == 0,
             "LinearFileTable: staging_bytes_per_thread must be a multiple of ", kBlockSize);
  // Bounds the memory of the io_threads readers created up front.
  const auto max_bytes = std::numeric_limits<size_t>::max();
  const auto num_threads = static_cast<uint64_t>(io_threads);
  const auto staging = static_cast<uint64_t>(staging_bytes_per_thread);
  NVE_CHECK_(num_threads <= (max_bytes - kBlockSize) / staging,
             "LinearFileTable: staging allocation size overflows size_t");
  NVE_CHECK_(num_threads <= max_bytes / staging_read_depth(staging_bytes_per_thread) / sizeof(ReadSpan),
             "LinearFileTable: read span allocation size overflows size_t");
}

void from_json(const nlohmann::json& json, LinearFileTableConfig& conf) {
  using base_type = LinearFileTableConfig::base_type;
  from_json(json, static_cast<base_type&>(conf));
  NVE_READ_JSON_FIELD_(path);
  NVE_READ_JSON_FIELD_(offset);
  NVE_READ_JSON_FIELD_(num_rows);
  NVE_READ_JSON_FIELD_(read_only);
  NVE_READ_JSON_FIELD_(create_if_missing);
  NVE_READ_JSON_FIELD_(direct_io);
  NVE_READ_JSON_FIELD_(io_engine);
  NVE_READ_JSON_FIELD_(io_threads);
  NVE_READ_JSON_FIELD_(staging_bytes_per_thread);
  NVE_READ_JSON_FIELD_(coalesce_adjacent);
}

void to_json(nlohmann::json& json, const LinearFileTableConfig& conf) {
  using base_type = LinearFileTableConfig::base_type;
  to_json(json, static_cast<const base_type&>(conf));
  NVE_WRITE_JSON_FIELD_(path);
  NVE_WRITE_JSON_FIELD_(offset);
  NVE_WRITE_JSON_FIELD_(num_rows);
  NVE_WRITE_JSON_FIELD_(read_only);
  NVE_WRITE_JSON_FIELD_(create_if_missing);
  NVE_WRITE_JSON_FIELD_(direct_io);
  NVE_WRITE_JSON_FIELD_(io_engine);
  NVE_WRITE_JSON_FIELD_(io_threads);
  NVE_WRITE_JSON_FIELD_(staging_bytes_per_thread);
  NVE_WRITE_JSON_FIELD_(coalesce_adjacent);
}

void from_json(const nlohmann::json& json, LinearFileTableFactoryConfig& conf) {
  using base_type = LinearFileTableFactoryConfig::base_type;
  from_json(json, static_cast<base_type&>(conf));
}

void to_json(nlohmann::json& json, const LinearFileTableFactoryConfig& conf) {
  using base_type = LinearFileTableFactoryConfig::base_type;
  to_json(json, static_cast<const base_type&>(conf));
}

void FileDescriptor::reset(const int fd) {
  if (fd_ >= 0) ::close(fd_);
  fd_ = fd;
}

// --- table ----------------------------------------------------------------------------------------

LinearFileTable::LinearFileTable(const table_id_t id, const config_type& config)
    : base_type(id, config) {
  int flags = (config_.read_only ? O_RDONLY : O_RDWR) | O_CLOEXEC;
  if (config_.create_if_missing) {
    flags |= O_CREAT;
  }
  // The descriptors close themselves if anything below throws.
  fd_buffered_.reset(::open(config_.path.c_str(), flags, 0644));
  NVE_CHECK_(fd_buffered_.get() >= 0, "LinearFileTable: ", errno_message("open"), " (", config_.path, ")");

  struct stat st {};
  NVE_CHECK_(::fstat(fd_buffered_.get(), &st) == 0, "LinearFileTable: ", errno_message("fstat"));
  const int64_t needed = config_.offset + config_.num_rows * config_.max_value_size;
  if (st.st_size < needed) {
    NVE_CHECK_(config_.create_if_missing, "LinearFileTable: file ", config_.path, " has ", st.st_size, " bytes, need ",
               needed, " for ", config_.num_rows, " rows (set create_if_missing to extend it)");
    NVE_CHECK_(::ftruncate(fd_buffered_.get(), static_cast<off_t>(needed)) == 0, "LinearFileTable: ", errno_message("ftruncate"));
  }

  if (config_.direct_io) {
    fd_direct_.reset(::open(config_.path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC));
    if (fd_direct_.get() < 0) {
      NVE_LOG_WARNING_("LinearFileTable: O_DIRECT unavailable for ", config_.path, " (", std::strerror(errno),
                       "), reads go through the page cache");
    } else if (!direct_reads_work(fd_direct_.get(), static_cast<off_t>(config_.offset))) {
      // Checked before picking the engine: "auto" skips AIO without O_DIRECT.
      NVE_LOG_WARNING_("LinearFileTable: ", config_.path, " refuses O_DIRECT reads, reads go through the page cache");
      fd_direct_.reset();
    }
  }

  // "auto" takes the first engine the kernel and sandbox allow. AIO is only asynchronous with O_DIRECT.
  read_depth_ = staging_read_depth(config_.staging_bytes_per_thread);
  std::vector<IoEngine> engines{IoEngine::IoUring, IoEngine::Aio, IoEngine::Pread};
  if (config_.io_engine != "auto") {
    engines = {parse_io_engine(config_.io_engine)};
  } else if (!uses_direct_io()) {
    engines = {IoEngine::IoUring, IoEngine::Pread};
  }
  // One reader per I/O thread up front, so resource limits (e.g. RLIMIT_MEMLOCK for io_uring rings before
  // Linux 5.12) show up here, where "auto" can still move on to the next engine.
  const auto num_readers = static_cast<size_t>(config_.io_threads);
  for (const IoEngine engine : engines) {
    while (idle_readers_.size() < num_readers) {
      auto reader = make_batch_reader(engine, read_depth_, staging_bytes());
      if (!reader) break;
      idle_readers_.push_back(std::move(reader));
    }
    if (idle_readers_.size() == num_readers) {
      engine_ = engine;
      break;
    }
    idle_readers_.clear();
  }
  NVE_CHECK_(!idle_readers_.empty(), "LinearFileTable: io_engine ", config_.io_engine,
             " is unavailable (kernel support, seccomp or limits)");
  if (config_.io_engine == "auto" && engine_ == IoEngine::Pread) {
    NVE_LOG_WARNING_("LinearFileTable: no batched I/O engine available, ", config_.io_threads,
                     " blocking reads in flight at most (raise io_threads)");
  }
  const std::string prefix = "linear_file_" + std::to_string(id) + "_";
  pending_buffer_ = prefix + "pending";

  NVE_LOG_INFO_("LinearFileTable: ", config_.num_rows, " rows of ", config_.max_value_size, " bytes in ", config_.path,
                " (read_only=", config_.read_only, ", direct_io=", uses_direct_io(), ", io_engine=", io_engine_name(engine_),
                ", io_threads=", config_.io_threads, ", read batch=", read_depth_, ")");
}

// Out of line: destroying the idle readers needs the complete BatchReader.
LinearFileTable::~LinearFileTable() = default;

void LinearFileTable::clear(context_ptr_t&) {
  NVE_LOG_INFO_("Clearing a linear file table has no effect");
}

int64_t LinearFileTable::size(context_ptr_t&, bool) const {
  return config_.num_rows;
}

void LinearFileTable::erase(context_ptr_t&, int64_t, buffer_ptr<const void>) {
  NVE_LOG_INFO_("Erasing from a linear file table has no effect");
}

int64_t* LinearFileTable::lookup_counter_storage(context_ptr_t& ctx) const {
  static constexpr char buffer_name[]{"linear_file_table_key_counter"};
  NVE_CHECK_(ctx != nullptr, "Invalid context");
  void* buffer = ctx->get_buffer(buffer_name, sizeof(int64_t), true /*host_alloc*/);
  NVE_CHECK_(buffer != nullptr, "Failed to get counter buffer");
  return reinterpret_cast<int64_t*>(buffer);
}

void LinearFileTable::run_parts(context_ptr_t& ctx, const int64_t num_parts,
                                const std::function<void(int64_t)>& task) const {
  struct State {
    std::mutex mutex;
    std::condition_variable idle;
    int64_t next{0};     // next unclaimed part
    int64_t running{0};  // claimed parts that did not finish
    std::exception_ptr error;
  };
  const auto state = std::make_shared<State>();
  // Claims parts until none are left (or one failed). Helpers that start after that return without
  // touching `task`, which may be gone by then.
  const auto run = [state, num_parts, &task] {
    std::unique_lock<std::mutex> lock(state->mutex);
    while (!state->error && state->next < num_parts) {
      const int64_t part = state->next++;
      ++state->running;
      lock.unlock();
      std::exception_ptr error;
      try {
        task(part);
      } catch (...) {
        error = std::current_exception();
      }
      lock.lock();
      if (error && !state->error) state->error = error;
      if (--state->running == 0) state->idle.notify_all();
    }
  };
  // The caller always works too and waits only for parts a helper already claimed, so a busy pool
  // (or a call from one of its own workers) cannot stall it.
  const thread_pool_ptr_t pool = ctx->get_thread_pool();
  const int64_t num_helpers = pool == InlineThreadPool::instance() ? 0 : std::min(num_parts - 1, pool->num_workers());
  if (num_helpers > 0) pool->submit_n(0, num_helpers, [run](int64_t) { run(); }, nullptr, 0);
  run();
  std::unique_lock<std::mutex> lock(state->mutex);
  state->idle.wait(lock, [&state] { return state->running == 0; });
  if (state->error) std::rethrow_exception(state->error);
}

std::unique_ptr<BatchReader> LinearFileTable::acquire_reader() const {
  {
    std::lock_guard<std::mutex> lock(readers_mutex_);
    if (!idle_readers_.empty()) {
      auto reader = std::move(idle_readers_.back());
      idle_readers_.pop_back();
      return reader;
    }
  }
  // More concurrent lookups than io_threads need extra readers; if the engine refuses one, block on pread.
  if (auto reader = make_batch_reader(engine_, read_depth_, staging_bytes())) return reader;
  std::call_once(fallback_warned_, [this] {
    NVE_LOG_WARNING_("LinearFileTable: cannot create another ", io_engine_name(engine_),
                     " reader, threads beyond the pool read with pread (reported once)");
  });
  return make_batch_reader(IoEngine::Pread, read_depth_, staging_bytes());
}

void LinearFileTable::release_reader(std::unique_ptr<BatchReader> reader) const {
  // Pread fallbacks are free to recreate; pooling them would shadow the batched readers.
  if (reader->engine() != engine_) return;
  std::lock_guard<std::mutex> lock(readers_mutex_);
  idle_readers_.push_back(std::move(reader));
}

size_t LinearFileTable::finish_read(ReadSpan& span) const {
  const auto len = static_cast<int64_t>(span.len);
  if (span.result == len) return span.len;
  if (span.result >= 0) {
    // Short read, e.g. EOF inside the last block.
    const auto got = static_cast<size_t>(span.result);
    return got + read_span(span.offset + span.result, span.len - got, span.dst + span.result);
  }
  const auto err = static_cast<int>(-span.result);
  NVE_CHECK_(err == EINVAL || err == EAGAIN || err == EINTR, "LinearFileTable: read failed: ", std::strerror(err));
  return read_span(span.offset, span.len, span.dst);  // falls back to a buffered read when direct I/O refuses it
}

void LinearFileTable::throw_truncated(const off_t row_end) const {
  struct stat st {};
  const std::string size = ::fstat(fd_buffered_.get(), &st) == 0 ? std::to_string(st.st_size) : errno_message("fstat");
  NVE_THROW_("LinearFileTable: ", config_.path, " ended before byte ", row_end, " (file size now ", size,
             ", opened with at least ", config_.offset + config_.num_rows * config_.max_value_size, "); was it truncated?");
}

size_t LinearFileTable::read_span(off_t file_offset, size_t bytes, void* dst) const {
  int fd = read_fd();
  size_t done = 0;
  while (done < bytes) {
    const ssize_t got = ::pread(fd, static_cast<char*>(dst) + done, bytes - done, file_offset + static_cast<off_t>(done));
    if (got < 0) {
      if (errno == EINTR) continue;
      if (fd == fd_direct_.get() && errno == EINVAL) {
        // Unaligned direct reads (resuming a short read at EOF) just finish buffered. An aligned one means
        // the filesystem refuses direct reads, so later lookups skip them.
        if (block_aligned(file_offset + static_cast<off_t>(done), bytes - done, static_cast<char*>(dst) + done)) {
          stop_direct_reads();
        }
        fd = fd_buffered_.get();
        continue;
      }
      NVE_THROW_("LinearFileTable: ", errno_message("pread"));
    }
    if (got == 0) break;  // EOF; the caller checks the rows it needs were read
    done += static_cast<size_t>(got);
  }
  return done;
}

void LinearFileTable::stop_direct_reads() const {
  if (direct_reads_refused_.exchange(true)) return;
  NVE_LOG_WARNING_("LinearFileTable: ", config_.path, " refused an aligned O_DIRECT read, later reads go through the page cache",
                   engine_ == IoEngine::Aio ? " (AIO now submits synchronously; set io_engine=io_uring or direct_io=false)" : "");
}

// --- find -----------------------------------------------------------------------------------------

void LinearFileTable::find(context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw,
                           buffer_ptr<bitmask64_t> hit_mask_bw, const int64_t value_stride,
                           buffer_ptr<void> values_bw, buffer_ptr<int64_t> value_sizes_bw) const {
  NVE_NVTX_SCOPED_FUNCTION_COL1_();
  NVE_CHECK_(ctx != nullptr, "Invalid context");
  if (n <= 0) return;
  NVE_CHECK_(value_stride >= 0, "LinearFileTable: value_stride must be non-negative");
  NVE_CHECK_(!values_bw || value_stride >= config_.max_value_size,
             "LinearFileTable: value_stride ", value_stride, " is smaller than row size ", config_.max_value_size);
  auto lookup_stream = ctx->get_lookup_stream();
  const void* keys = keys_bw ? keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, lookup_stream) : nullptr;
  bitmask64_t* hit_mask = hit_mask_bw ? hit_mask_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, lookup_stream) : nullptr;
  int8_t* values = values_bw ? static_cast<int8_t*>(values_bw->access_buffer(cudaMemoryTypeUnregistered, false, lookup_stream)) : nullptr;
  int64_t* value_sizes = value_sizes_bw ? value_sizes_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, lookup_stream) : nullptr;
  NVE_CHECK_(keys != nullptr, "Invalid keys");

  int64_t hits = 0;
  switch (config_.key_size) {
    case sizeof(int64_t):
      hits = find_rows<int64_t>(ctx, n, static_cast<const int64_t*>(keys), hit_mask, value_stride, values, value_sizes);
      break;
    case sizeof(int32_t):
      hits = find_rows<int32_t>(ctx, n, static_cast<const int32_t*>(keys), hit_mask, value_stride, values, value_sizes);
      break;
    default:
      NVE_THROW_("LinearFileTable: unsupported key_size ", config_.key_size);
  }
  *lookup_counter_storage(ctx) += hits;
}

template <typename KeyT>
int64_t LinearFileTable::find_rows(context_ptr_t& ctx, const int64_t n, const KeyT* keys, bitmask64_t* hit_mask,
                                   const int64_t value_stride, int8_t* values, int64_t* value_sizes) const {
  // Per-context scratch, reused across calls.
  auto* pending = static_cast<PendingRow*>(ctx->get_buffer(pending_buffer_, static_cast<size_t>(n) * sizeof(PendingRow), true /*host_alloc*/));
  NVE_CHECK_(pending != nullptr, "LinearFileTable: failed to allocate the pending rows buffer");
  int64_t num_pending = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (hit_mask != nullptr && bitmask64::get(hit_mask[i / bitmask64::num_bits], i % bitmask64::num_bits)) continue;
    if (value_sizes != nullptr) value_sizes[i] = 0;
    const KeyT key = keys[i];
    if (!key_in_range(key, static_cast<uint64_t>(config_.num_rows))) continue;
    pending[num_pending++] = {static_cast<int64_t>(key), i};
  }
  if (num_pending == 0) return 0;

  // Mask words may be shared between I/O threads, hence the atomic OR.
  const auto mark_hit = [hit_mask, value_sizes, this](int64_t idx) {
    if (value_sizes != nullptr) value_sizes[idx] = config_.max_value_size;
    if (hit_mask != nullptr) {
      bitmask64::atomic_merge(&hit_mask[idx / bitmask64::num_bits], bitmask64::single(idx % bitmask64::num_bits));
    }
  };
  if (values == nullptr) {  // existence / count check only
    for (int64_t j = 0; j < num_pending; ++j) mark_hit(pending[j].idx);
    return num_pending;
  }
  const bool coalesce = config_.coalesce_adjacent;
  if (coalesce) {
    std::sort(pending, pending + num_pending, [](const PendingRow& a, const PendingRow& b) { return a.key < b.key; });
  }

  // Blocking reads parallelize only across threads; batched engines submit many reads per thread.
  const int64_t min_rows_per_part = engine_ == IoEngine::Pread ? 1 : kMinReadsPerThread;
  const int64_t num_parts = select_num_tasks(num_pending, config_.io_threads, min_rows_per_part);
  const int64_t rows_per_part = ceil_div(num_pending, num_parts);
  const size_t staging = staging_bytes();

  const int fd = read_fd();
  const int64_t row_bytes = config_.max_value_size;
  const auto block = static_cast<off_t>(kBlockSize);
  std::atomic<int64_t> hits{0};
  std::shared_lock<std::shared_mutex> file_lock(file_mutex_);
  run_parts(ctx, num_parts, [&](int64_t part) {
    const int64_t end = std::min(num_pending, (part + 1) * rows_per_part);
    // Returns the reader on every exit, file errors included; only an engine failure drops it.
    struct Lease {
      const LinearFileTable* table;
      std::unique_ptr<BatchReader> reader;
      ~Lease() {
        if (reader) table->release_reader(std::move(reader));
      }
    } lease{this, acquire_reader()};
    int8_t* buf = lease.reader->staging();
    ReadSpan* part_spans = lease.reader->spans();
    int64_t local_hits = 0;
    for (int64_t j = part * rows_per_part; j < end;) {
      // Pack spans into the staging buffer, then read them as one batch.
      size_t used = 0;
      size_t count = 0;
      while (j < end && count < read_depth_) {
        const off_t first = row_offset(pending[j].key);
        const off_t span_start = align_down(first, block);
        off_t span_end = round_up<off_t>(first + row_bytes, block);
        if (used + static_cast<size_t>(span_end - span_start) > staging) break;
        // Grow the span while the next row's blocks touch it and the staging buffer holds it.
        int64_t k = j + 1;
        for (; coalesce && k < end; ++k) {
          const off_t off = row_offset(pending[k].key);
          const off_t blk_start = align_down(off, block);
          const off_t blk_end = round_up<off_t>(off + row_bytes, block);
          if (blk_start > span_end || used + static_cast<size_t>(blk_end - span_start) > staging) break;
          span_end = std::max(span_end, blk_end);
        }
        const auto len = static_cast<size_t>(span_end - span_start);
        part_spans[count++] = {span_start, len, buf + used, j, k, 0};
        used += len;
        j = k;
      }
      try {
        lease.reader->read(fd, part_spans, count);
      } catch (...) {
        if (lease.reader->abandoned_reads()) {
          // The kernel may still write its staging: leak the reader so that memory is never reused.
          NVE_LOG_ERROR_("LinearFileTable: abandoning an I/O reader and its ", staging, " byte staging buffer");
          static_cast<void>(lease.reader.release());
        }
        lease.reader.reset();
        throw;
      }
      for (size_t s = 0; s < count; ++s) {
        ReadSpan& span = part_spans[s];
        const size_t valid = finish_read(span);
        // A short span is fine while its rows fit (the last block may extend past EOF).
        const bool short_span = valid < span.len;
        for (int64_t m = span.first; m < span.last; ++m) {
          const PendingRow& p = pending[m];
          const auto rel = static_cast<size_t>(row_offset(p.key) - span.offset);
          if (short_span && rel + static_cast<size_t>(row_bytes) > valid) throw_truncated(row_offset(p.key) + row_bytes);
          std::memcpy(values + p.idx * value_stride, span.dst + rel, static_cast<size_t>(row_bytes));
          mark_hit(p.idx);
          ++local_hits;
        }
      }
    }
    hits.fetch_add(local_hits, std::memory_order_relaxed);
  });
  return hits.load(std::memory_order_relaxed);
}

// --- insert / update ------------------------------------------------------------------------------

template <typename KeyT, typename Fn>
void LinearFileTable::modify_rows(context_ptr_t& ctx, const int64_t n, const KeyT* keys, const Fn& fn) {
  NVE_CHECK_(!config_.read_only, "LinearFileTable: table ", config_.path, " is read-only");
  std::vector<int64_t> pending;
  pending.reserve(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    if (key_in_range(keys[i], static_cast<uint64_t>(config_.num_rows))) pending.push_back(i);
  }
  if (pending.empty()) return;
  const auto num_pending = static_cast<int64_t>(pending.size());
  const int64_t num_parts = select_num_tasks(num_pending, config_.io_threads, 1);
  const int64_t rows_per_part = ceil_div(num_pending, num_parts);
  std::lock_guard<std::shared_mutex> file_lock(file_mutex_);
  run_parts(ctx, num_parts, [&](int64_t part) {
    const int64_t end = std::min(num_pending, (part + 1) * rows_per_part);
    for (int64_t m = part * rows_per_part; m < end; ++m) {
      const int64_t i = pending[static_cast<size_t>(m)];
      const auto key = static_cast<int64_t>(keys[i]);
      std::lock_guard<std::mutex> lock(row_write_mutex(key));
      fn(i, key);
    }
  });
}

template <typename KeyT>
void LinearFileTable::write_rows(context_ptr_t& ctx, const int64_t n, const KeyT* keys, const int64_t value_stride, const int8_t* values) {
  const auto row_bytes = static_cast<size_t>(config_.max_value_size);
  modify_rows(ctx, n, keys, [&](int64_t i, int64_t key) {
    pwrite_all(fd_buffered_.get(), values + i * value_stride, row_bytes, row_offset(key));
  });
}

void LinearFileTable::insert(context_ptr_t& /*ctx*/, const int64_t /*n*/, buffer_ptr<const void> /*keys*/,
                             const int64_t /*value_stride*/, const int64_t /*value_size*/, buffer_ptr<const void> /*values*/) {
  NVE_LOG_INFO_("Inserting into linear file table has no effect");
}

void LinearFileTable::update(context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw, const int64_t value_stride,
                             const int64_t value_size, buffer_ptr<const void> values_bw) {
  NVE_NVTX_SCOPED_FUNCTION_COL3_();
  NVE_CHECK_(ctx != nullptr, "Invalid context");
  if (n <= 0) return;
  NVE_CHECK_(value_size == config_.max_value_size, "LinearFileTable: value_size ", value_size, " != row size ", config_.max_value_size);
  NVE_CHECK_(value_stride >= value_size,
             "LinearFileTable: value_stride ", value_stride, " is smaller than value_size ", value_size);
  auto modify_stream = ctx->get_modify_stream();
  const void* keys = keys_bw ? keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream) : nullptr;
  const auto* values = values_bw ? static_cast<const int8_t*>(values_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream)) : nullptr;
  NVE_CHECK_(keys != nullptr, "Invalid keys");
  NVE_CHECK_(values != nullptr, "Invalid values");
  switch (config_.key_size) {
    case sizeof(int64_t):
      write_rows<int64_t>(ctx, n, static_cast<const int64_t*>(keys), value_stride, values);
      break;
    case sizeof(int32_t):
      write_rows<int32_t>(ctx, n, static_cast<const int32_t*>(keys), value_stride, values);
      break;
    default:
      NVE_THROW_("LinearFileTable: unsupported key_size ", config_.key_size);
  }
}

void LinearFileTable::update_accumulate(context_ptr_t& /*ctx*/, const int64_t /*n*/, buffer_ptr<const void> /*keys*/,
                                        const int64_t /*update_stride*/, const int64_t /*update_size*/,
                                        buffer_ptr<const void> /*updates*/, const DataType_t /*update_dtype*/) {
  NVE_THROW_NOT_IMPLEMENTED_();
}

// --- factory --------------------------------------------------------------------------------------

host_table_ptr_t LinearFileTableFactory::produce(const table_id_t id, const table_config_type& config) {
  return std::make_shared<LinearFileTable>(id, config);
}

}  // namespace plugin
}  // namespace nve
