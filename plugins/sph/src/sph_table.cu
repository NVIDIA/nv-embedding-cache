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

#include <sph_table.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <bit_ops.hpp>
#include <buffer_wrapper.hpp>
#include <common.hpp>
#include <execution_context.hpp>
#include <json_support.hpp>
#include <layer_utils.hpp>
#include <mutex>
#include <sph/error.h>
#include <sph/sph_hash_map.h>
#include <type_traits>

namespace nve {
namespace plugin {

namespace {
// Engine teardown failures (SPH_TEARDOWN) surface here; route them to the NVE log.
void forward_sph_warning(const char* message) noexcept {
  try {
    NVE_LOG_WARNING_(message);
  } catch (...) {
    // Logging must not escape a destructor path.
  }
}

// RAII device switch (the ecache ScopedDevice is not visible to plugins).
struct ScopedDevice {
  int prev_{0};
  explicit ScopedDevice(int dev) {
    NVE_CHECK_(cudaGetDevice(&prev_));
    NVE_CHECK_(cudaSetDevice(dev));
  }
  ~ScopedDevice() {
    const cudaError_t res = cudaSetDevice(prev_);
    if (res == cudaSuccess) {
      return;
    }
    // Both the unloading check and the warning format a message, which can throw;
    // nothing may escape a destructor.
    try {
      if (!cuda_runtime_unloading(res, "device restore")) {
        NVE_LOG_WARNING_("Failed to restore device ", prev_, ": ", cudaGetErrorString(res));
      }
    } catch (...) {
    }
  }
};

// Plain execution context that registers itself with the table's registry, so
// defrag can enumerate the lookup streams it has to wait for (mirrors
// GPUTableExecutionContext).
class SphExecutionContext : public ExecutionContext {
 public:
  NVE_PREVENT_COPY_AND_MOVE_(SphExecutionContext);
  SphExecutionContext() = delete;

  SphExecutionContext(cudaStream_t lookup_stream, cudaStream_t modify_stream,
                      thread_pool_ptr_t thread_pool, allocator_ptr_t allocator,
                      std::shared_ptr<ContextRegistry> context_registry)
      : ExecutionContext(lookup_stream, modify_stream, std::move(thread_pool), std::move(allocator)),
        context_registry_(std::move(context_registry)) {
    NVE_CHECK_(context_registry_ != nullptr, "Invalid context registry");
    context_registry_->add_context(this);
  }

  ~SphExecutionContext() override { context_registry_->remove_context(this); }

 private:
  std::shared_ptr<ContextRegistry> context_registry_;
};

// Per-context scratch backing the lookup counter: the key count that find() was
// given, plus a host copy of the hit mask it produced.
constexpr const char* kLookupCounterKeys = "sph_lookup_counter_keys";
constexpr const char* kLookupCounterMask = "sph_lookup_counter_mask";

constexpr size_t mask_words_for(int64_t num_keys) {
  return static_cast<size_t>(ceil_div<int64_t>(num_keys, 64));
}

}  // namespace

// ---- config json -----------------------------------------------------------

void SphTableConfig::check() const {
  NVE_CHECK_(row_size_in_bytes > 0, "SphTableConfig: row_size_in_bytes must be > 0");
  NVE_CHECK_(cache_size > 0 || initial_capacity > 0,
             "SphTableConfig: one of cache_size or initial_capacity must be set");
  // Fragmentation is payload_slots/live_keys, which never drops below 1.0, so a
  // positive threshold under 1.0 would defrag after every single insert.
  NVE_CHECK_(defrag_threshold <= 0.0 || defrag_threshold >= 1.0,
             "SphTableConfig: defrag_threshold must be >= 1.0 (or <= 0 to disable)");
  NVE_CHECK_(default_value.empty() || static_cast<int64_t>(default_value.size()) == row_size_in_bytes,
             "SphTableConfig: default_value must be empty or exactly row_size_in_bytes long");
}

void from_json(const nlohmann::json& json, SphTableConfig& conf) {
  NVE_READ_JSON_FIELD_(device_id);
  NVE_READ_JSON_FIELD_(cache_size);
  NVE_READ_JSON_FIELD_(initial_capacity);
  NVE_READ_JSON_FIELD_(row_size_in_bytes);
  NVE_READ_JSON_FIELD_(value_dtype);
  NVE_READ_JSON_FIELD_(invalid_key);
  NVE_READ_JSON_FIELD_(defrag_threshold);
  NVE_READ_JSON_FIELD_(default_value);
}

void to_json(nlohmann::json& json, const SphTableConfig& conf) {
  NVE_WRITE_JSON_FIELD_(device_id);
  NVE_WRITE_JSON_FIELD_(cache_size);
  NVE_WRITE_JSON_FIELD_(initial_capacity);
  NVE_WRITE_JSON_FIELD_(row_size_in_bytes);
  NVE_WRITE_JSON_FIELD_(value_dtype);
  NVE_WRITE_JSON_FIELD_(invalid_key);
  NVE_WRITE_JSON_FIELD_(defrag_threshold);
  NVE_WRITE_JSON_FIELD_(default_value);
}

void from_json(const nlohmann::json&, SphTableFactoryConfig&) {}

void to_json(nlohmann::json& json, const SphTableFactoryConfig&) { json = nlohmann::json::object(); }

// ---- SphTable --------------------------------------------------------------

SphTable::SphTable(table_id_t id, const SphTableConfig& config) : id_(id), config_(config) {
  config_.check();
  ScopedDevice scope_device(config_.device_id);

  capacity_ = config_.initial_capacity;
  if (capacity_ <= 0) {
    capacity_ = static_cast<int64_t>(config_.cache_size / static_cast<size_t>(config_.row_size_in_bytes));
  }
  NVE_CHECK_(capacity_ > 0, "SPH table capacity resolved to <= 0");

  const auto* default_value = config_.default_value.empty()
                                  ? nullptr
                                  : reinterpret_cast<const int8_t*>(config_.default_value.data());
  engine_ = std::make_unique<sph::ShardedPerfectHashMap>(capacity_, config_.row_size_in_bytes, default_value);
  engine_->reserve(capacity_);  // pre-commit VMM once so inserts skip the per-insert grow.

  contexts_ = std::make_shared<ContextRegistry>();
}

SphTable::~SphTable() noexcept {
  if (!contexts_->empty()) {
    NVE_LOG_ERROR_("SPH table destroyed with live execution contexts");
  }
  auto engine = std::move(engine_);
  try {
    ScopedDevice scope_device(config_.device_id);
    engine.reset();
  } catch (const std::exception& error) {
    NVE_LOG_ERROR_("SPH cleanup failed: ", error.what());
    (void)engine.release();
  } catch (...) {
    NVE_LOG_ERROR_("SPH cleanup failed with an unknown exception");
    (void)engine.release();
  }
}

context_ptr_t SphTable::create_execution_context(cudaStream_t lookup_stream, cudaStream_t modify_stream,
                                                 thread_pool_ptr_t thread_pool,
                                                 allocator_ptr_t allocator) {
  return std::make_shared<SphExecutionContext>(lookup_stream, modify_stream, std::move(thread_pool),
                                               std::move(allocator), contexts_);
}

void SphTable::clear(context_ptr_t& ctx) {
  // Exclusive: reset() rebuilds the payload/PTE buffers, so it must not overlap
  // a lookup the way an insert may.
  std::lock_guard<std::mutex> modify_lock(modify_mutex_);
  std::unique_lock<std::shared_mutex> lock(table_mutex_);
  ScopedDevice scope_device(config_.device_id);
  auto modify_stream = ctx->get_modify_stream();

  // reset() drops and re-creates the ShardedVmmBuffers on the *host*, unmapping
  // their VMM right away. An in-flight find kernel would then be reading
  // unmapped memory, so lookups have to be finished here, not merely ordered
  // behind us — hence the blocking sync rather than just the event dependency.
  drain_lookups(modify_stream);
  NVE_CHECK_(cudaStreamSynchronize(modify_stream));

  engine_->reset(modify_stream, capacity_);

  // The reset memsets are queued on modify_stream; block so a find issued on a
  // different lookup stream after clear() returns cannot outrun them.
  NVE_CHECK_(cudaStreamSynchronize(modify_stream));
}

// ---- defrag ----------------------------------------------------------------

double SphTable::get_fragmentation() const {
  // Reads host-side bookkeeping that only modifies mutate.
  std::lock_guard<std::mutex> modify_lock(modify_mutex_);
  return engine_->get_fragmentation();
}

// Make `stream` wait for the work already queued on every live context's lookup
// stream. The exclusive lock only orders host calls; this is what keeps a
// destructive op off the GPU until concurrent lookups have drained.
// Caller must hold table_mutex_ exclusively.
void SphTable::drain_lookups(cudaStream_t stream) {
  auto sync_event = contexts_->create_sync_event();
  NVE_CHECK_(sync_event->event_record(), "Failed to record SPH lookup sync event");
  NVE_CHECK_(sync_event->event_wait_stream(stream), "Failed to queue SPH lookup sync event");
}

// Caller must hold modify_mutex_ and table_mutex_ exclusively.
void SphTable::defrag_locked(context_ptr_t& ctx) {
  ScopedDevice scope_device(config_.device_id);
  auto modify_stream = ctx->get_modify_stream();

  // Lookups already queued on other contexts' streams would otherwise read the
  // payload while defrag moves it.
  drain_lookups(modify_stream);

  engine_->defrag_all(modify_stream);

  // Synchronous for now: on return the compacted layout is visible to every
  // stream, so releasing the lock cannot expose a half-moved payload. This also
  // covers defrag's host-side release_shard/grow_shard calls, which run once the
  // stream (and transitively the drained lookups) have completed.
  NVE_CHECK_(cudaStreamSynchronize(modify_stream));
}

void SphTable::defrag(context_ptr_t& ctx) {
  std::lock_guard<std::mutex> modify_lock(modify_mutex_);
  std::unique_lock<std::shared_mutex> lock(table_mutex_);
  defrag_locked(ctx);
}

// Caller must hold modify_mutex_ (so the fragmentation read cannot race an
// insert and no other modify can slip in before the exclusive table lock).
void SphTable::maybe_defrag(context_ptr_t& ctx) {
  if (config_.defrag_threshold <= 0.0) return;
  // Cheap host-side read (committed page counts); only pay for the exclusive
  // table lock when a defrag is actually due.
  if (engine_->get_fragmentation() <= config_.defrag_threshold) return;
  std::unique_lock<std::shared_mutex> lock(table_mutex_);
  defrag_locked(ctx);
}

// The counter reports MISSES (lookup_counter_hits() == false), derived from the
// hit mask by find(). It lives in per-context scratch, so concurrent finds on
// separate contexts do not share it.
void SphTable::reset_lookup_counter(context_ptr_t& ctx) {
  // Safe to write without syncing: the layer's order is reset -> find -> get,
  // and get_lookup_counter() synchronises the stream before reading, so no copy
  // from a previous round can still be in flight here.
  auto* h_keys = reinterpret_cast<int64_t*>(
      ctx->get_buffer(kLookupCounterKeys, sizeof(int64_t), true /*host_alloc*/));
  *h_keys = 0;
}

void SphTable::get_lookup_counter(context_ptr_t& ctx, int64_t* counter) const {
  NVE_CHECK_(counter != nullptr, "Invalid counter ptr");
  auto* h_keys = reinterpret_cast<int64_t*>(
      ctx->get_buffer(kLookupCounterKeys, sizeof(int64_t), true /*host_alloc*/));
  const int64_t num_keys = *h_keys;
  if (num_keys <= 0) {  // no find() with a mask since the last reset
    *counter = 0;
    return;
  }
  const size_t num_words = mask_words_for(num_keys);
  auto* h_mask = reinterpret_cast<const bitmask64_t*>(
      ctx->get_buffer(kLookupCounterMask, sizeof(bitmask64_t) * num_words, true /*host_alloc*/));
  // find() copies the mask back with an async D2H on the lookup stream.
  NVE_CHECK_(cudaStreamSynchronize(ctx->get_lookup_stream()));

  int64_t resolved = 0;
  for (size_t w = 0; w < num_words; ++w) {
    const int64_t bits = std::min<int64_t>(64, num_keys - static_cast<int64_t>(w) * 64);
    bitmask64_t word = h_mask[w];
    if (bits < 64) word &= (bitmask64_t{1} << bits) - 1;  // ignore the unused tail
    resolved += __builtin_popcountll(word);
  }
  *counter = num_keys - resolved;
}

bool SphTable::lookup_counter_hits() const { return false; }

int32_t SphTable::get_device_id() const { return config_.device_id; }
int64_t SphTable::get_max_row_size() const { return config_.row_size_in_bytes; }
int64_t SphTable::get_key_size() const { return static_cast<int64_t>(sizeof(sph::ShardedPerfectHashMap::KeyT)); }
int64_t SphTable::get_invalid_key() const { return config_.invalid_key; }
DataType_t SphTable::get_value_type() const { return config_.value_dtype; }

void SphTable::erase(context_ptr_t&, int64_t, buffer_ptr<const void>) { NVE_THROW_NOT_IMPLEMENTED_(); }

int64_t SphTable::size(context_ptr_t&) const { NVE_THROW_NOT_IMPLEMENTED_(); }

int64_t SphTable::get_keys(context_ptr_t&, int64_t, void*) const { NVE_THROW_NOT_IMPLEMENTED_(); }

void SphTable::update_accumulate(context_ptr_t&, int64_t, buffer_ptr<const void>, int64_t, int64_t,
                                 buffer_ptr<const void>, DataType_t) {
  NVE_THROW_NOT_IMPLEMENTED_();
}

void SphTable::find(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys,
                    buffer_ptr<bitmask64_t> hit_mask, int64_t value_stride,
                    buffer_ptr<void> values, buffer_ptr<int64_t> value_sizes) const {
  NVE_CHECK_(value_sizes == nullptr, "value_sizes not supported by SPH table");
  if (n <= 0 || values == nullptr) return;  // exists/count-only find (values==nullptr) is not supported.

  std::shared_lock<std::shared_mutex> lock(table_mutex_);
  ScopedDevice scope_device(config_.device_id);
  auto lookup_stream = ctx->get_lookup_stream();

  NVE_CHECK_ARG_(keys != nullptr, "keys must not be null");
  const void* keys_buf = keys->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, lookup_stream);
  void* values_buf = values->access_buffer(cudaMemoryTypeDevice, false /*copy_content*/, lookup_stream);
  // The mask is read as well as written: keys whose bit is already set are skipped.
  bitmask64_t* hit_mask_buf =
      hit_mask ? hit_mask->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, lookup_stream) : nullptr;

  engine_->find(reinterpret_cast<const int64_t*>(keys_buf), n, reinterpret_cast<int8_t*>(values_buf),
                lookup_stream, hit_mask_buf, value_stride);

  // Lookup counter. lookup_counter_hits() is false, so the contract is to report
  // the keys this table did NOT resolve; the hierarchical layer subtracts that
  // from the keys still outstanding to get our hit count. The engine returns no
  // count of its own, so derive it from the mask it just wrote. Without a mask
  // there is nothing to derive it from, and the counter is left at reset.
  if (hit_mask_buf != nullptr) {
    // The mask is n/8 bytes -- small next to the value rows already moving -- so
    // copy it back and popcount on the host in get_lookup_counter(). Doing this
    // with a kernel would need device code in the plugin itself, which does not
    // get registered with the runtime from this shared library.
    auto* h_keys = reinterpret_cast<int64_t*>(
        ctx->get_buffer(kLookupCounterKeys, sizeof(int64_t), true /*host_alloc*/));
    auto* h_mask = reinterpret_cast<bitmask64_t*>(
        ctx->get_buffer(kLookupCounterMask, sizeof(bitmask64_t) * mask_words_for(n),
                        true /*host_alloc*/));
    *h_keys = n;
    NVE_CHECK_(cudaMemcpyAsync(h_mask, hit_mask_buf, sizeof(bitmask64_t) * mask_words_for(n),
                               cudaMemcpyDeviceToHost, lookup_stream));
  }
}

void SphTable::insert(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, int64_t value_stride,
                      int64_t value_size, buffer_ptr<const void> values) {
  NVE_CHECK_(value_size == config_.row_size_in_bytes, "Unsupported value_size for SPH table");
  if (n <= 0) return;

  // Held across the insert and the possible defrag that follows it.
  std::lock_guard<std::mutex> modify_lock(modify_mutex_);
  {
    // Shared: lookups may overlap an insert, only defrag/clear exclude it.
    std::shared_lock<std::shared_mutex> lock(table_mutex_);
    ScopedDevice scope_device(config_.device_id);
    auto modify_stream = ctx->get_modify_stream();
    const int64_t row = config_.row_size_in_bytes;

    NVE_CHECK_ARG_(keys != nullptr, "keys must not be null");
    const void* keys_buf = keys->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, modify_stream);
    const void* values_buf = values->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, modify_stream);
    NVE_CHECK_(keys_buf && values_buf, "Invalid SPH insert params");

    // SPH does not mutate the key array; the const_cast matches its non-const KeyT* signature.
    auto* d_keys = const_cast<int64_t*>(reinterpret_cast<const int64_t*>(keys_buf));

    const int8_t* d_values;
    if (value_stride == value_size) {
      d_values = reinterpret_cast<const int8_t*>(values_buf);
    } else {
      // Compact strided rows into a contiguous scratch buffer (SPH assumes stride == row).
      auto* scratch = reinterpret_cast<int8_t*>(
          ctx->get_buffer("sph_insert_scratch", static_cast<size_t>(n) * row, false /*host_alloc*/));
      NVE_CHECK_(cudaMemcpy2DAsync(scratch, row, values_buf, value_stride, row, n,
                                   cudaMemcpyDeviceToDevice, modify_stream));
      d_values = scratch;
    }

    // Propagates sph::SphInsertFailed if any key was dropped (pilot search
    // giving up, a bucket/page/offset counter overflowing, a bucket exceeding
    // the fused kernel's gather capacity). It carries the full InsertResult and
    // derives from SphError -> std::runtime_error, so a caller that wants to
    // react can catch it for the breakdown, and the C API maps it to
    // NVE_ERROR_RUNTIME otherwise. Note the insert is not atomic: on failure the
    // rest of the batch has already been applied.
    engine_->insert(d_keys, d_values, n, modify_stream,
                    sph::ShardedPerfectHashMap::OnExisting::ON_EXISTING_UPDATE);
  }

  // Insert is the only op that grows the payload, so this is where the table
  // can become fragmented enough to be worth compacting.
  maybe_defrag(ctx);
}

void SphTable::update(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, int64_t value_stride,
                      int64_t value_size, buffer_ptr<const void> values) {
  NVE_CHECK_(value_size == config_.row_size_in_bytes, "Unsupported value_size for SPH table");
  if (n <= 0) return;

  std::lock_guard<std::mutex> modify_lock(modify_mutex_);
  std::shared_lock<std::shared_mutex> lock(table_mutex_);
  ScopedDevice scope_device(config_.device_id);
  auto modify_stream = ctx->get_modify_stream();
  const int64_t row = config_.row_size_in_bytes;

  NVE_CHECK_ARG_(keys != nullptr, "keys must not be null");
  const void* keys_buf = keys->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, modify_stream);
  const void* values_buf = values->access_buffer(cudaMemoryTypeDevice, true /*copy_content*/, modify_stream);
  NVE_CHECK_(keys_buf && values_buf, "Invalid SPH update params");

  const auto* d_keys = reinterpret_cast<const int64_t*>(keys_buf);

  const int8_t* d_values;
  if (value_stride == value_size) {
    d_values = reinterpret_cast<const int8_t*>(values_buf);
  } else {
    auto* scratch = reinterpret_cast<int8_t*>(
        ctx->get_buffer("sph_update_scratch", static_cast<size_t>(n) * row, false /*host_alloc*/));
    NVE_CHECK_(cudaMemcpy2DAsync(scratch, row, values_buf, value_stride, row, n,
                                 cudaMemcpyDeviceToDevice, modify_stream));
    d_values = scratch;
  }

  // assign = write-if-exists, matching update semantics.
  engine_->assign(d_keys, d_values, n, modify_stream);
}

// ---- SphTableFactory -------------------------------------------------------

SphTableFactory::SphTableFactory(const SphTableFactoryConfig& config) : config_(config) {
  config_.check();
  sph::set_warning_handler(&forward_sph_warning);
}

table_ptr_t SphTableFactory::produce(table_id_t id, const nlohmann::json& json) {
  const SphTableConfig config = json.get<SphTableConfig>();
  return std::make_shared<SphTable>(id, config);
}

}  // namespace plugin
}  // namespace nve
