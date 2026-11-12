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

#include <host_table_detail.hpp>
#include <rocksdb_table.hpp>
#include <rocksdb_utils.hpp>
#include <buffer_wrapper.hpp>
#include <cstring>

namespace nve {
namespace plugin {

void RocksDBTableConfig::check() const {
  base_type::check();

  NVE_CHECK_(max_batch_size > 0 && max_batch_size % bitmask64::num_bits == 0);

  NVE_CHECK_(!column_family.empty());
}

void from_json(const nlohmann::json& json, RocksDBTableConfig& conf) {
  using base_type = RocksDBTableConfig::base_type;
  from_json(json, static_cast<base_type&>(conf));

  NVE_READ_JSON_FIELD_(max_batch_size);

  NVE_READ_JSON_FIELD_(column_family);
  NVE_READ_JSON_FIELD_(verify_checksums);
}

void to_json(nlohmann::json& json, const RocksDBTableConfig& conf) {
  using base_type = RocksDBTableConfig::base_type;
  to_json(json, static_cast<const base_type&>(conf));

  NVE_WRITE_JSON_FIELD_(max_batch_size);

  NVE_WRITE_JSON_FIELD_(column_family);
  NVE_WRITE_JSON_FIELD_(verify_checksums);
}

RocksDBTable::RocksDBTable(const table_id_t id, const RocksDBTableConfig& config,
                           rdb_ctx_ptr_t& rdb_ctx, rocksdb::ColumnFamilyHandle* const cf)
    : base_type(id, config),
      rdb_ctx_{rdb_ctx},
      col_families_(to_uint(config.max_batch_size), cf) {
  read_opts_.verify_checksums = config.verify_checksums;
  write_opts_.sync = false;
}

// With manual_wal_flush, a write stays in process memory until the WAL buffer is flushed. Flushing
// at the end of each call makes a completed call survive a process crash (not a host crash: no fsync).
void RocksDBTable::flush_wal_() { NVE_CHECK_(rdb_ctx_->db->FlushWAL(false)); }

void RocksDBTable::clear(context_ptr_t&) {
  std::unique_ptr<rocksdb::DB>& __restrict db{rdb_ctx_->db};
  rocksdb::ColumnFamilyHandle* const __restrict cf{col_families_.front()};

  // Keys are fixed-width binary under the default bytewise comparator, so
  // `key_size + 1` bytes of 0xFF sorts strictly after every possible key.
  const std::string end_key(to_uint(config_.key_size) + 1, '\xFF');

  NVE_CHECK_(db->DeleteRange(write_opts_, cf, "", end_key));
  NVE_CHECK_(db->Flush({}, cf));
  NVE_CHECK_(db->CompactRange({}, cf, nullptr, nullptr));
}

void RocksDBTable::erase(context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw) {
  if (n <= 0) return;
  auto modify_stream{ctx->get_modify_stream()};

  const char* __restrict keys{nullptr};
  if (keys_bw) {
    auto p{keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream)};
    keys = static_cast<const char*>(p);
  }

  std::unique_ptr<rocksdb::DB>& __restrict db{rdb_ctx_->db};
  rocksdb::ColumnFamilyHandle* const __restrict cf{col_families_.front()};

  const int64_t key_size{config_.key_size};
  const int64_t max_batch_size{std::min(n, config_.max_batch_size)};

  // TODO: Prone to memory fragmentation. Use a persistent scratch buffer instead?
  rocksdb::WriteBatch batch;
  rocksdb::Slice k_view{nullptr, to_uint(key_size)};

  int64_t batch_size{};
  for (int64_t ij{}; ij < n; ++ij) {
    k_view.data_ = &keys[ij * key_size];
    NVE_CHECK_(batch.Delete(cf, k_view));
    if NVE_LIKELY_(++batch_size < max_batch_size) continue;

    NVE_CHECK_(db->Write(write_opts_, &batch));
    batch.Clear();
    batch_size = {};
  }
  if (batch_size) {
    NVE_CHECK_(db->Write(write_opts_, &batch));
  }
  flush_wal_();
}

void RocksDBTable::find(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys_bw,
                        buffer_ptr<bitmask64_t> hit_mask_bw, const int64_t value_stride,
                        buffer_ptr<void> values_bw, buffer_ptr<int64_t> value_sizes_bw) const {
  if (n <= 0) return;
  auto lookup_stream{ctx->get_lookup_stream()};

  const char* keys{nullptr};
  if (keys_bw) {
    auto p{keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, lookup_stream)};
    keys = static_cast<const char*>(p);
  }
  bitmask64_t* hit_mask;
  if (hit_mask_bw) {
    hit_mask = hit_mask_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, lookup_stream);
  } else {
    const uint64_t size{to_uint(ceil_div(n, bitmask64::num_bits)) * sizeof(bitmask64_t)};
    auto p{ctx->get_buffer("hitmask", size, true /*host_alloc*/)};
    std::memset(p, 0, size);
    hit_mask = static_cast<bitmask64_t*>(p);
  }
  char* values{nullptr};
  if (values_bw) {
    auto p{values_bw->access_buffer(cudaMemoryTypeUnregistered, false, lookup_stream)};
    values = static_cast<char*>(p);
  }
  int64_t* value_sizes{nullptr};
  if (value_sizes_bw) {
    value_sizes = value_sizes_bw->access_buffer(cudaMemoryTypeUnregistered, false, lookup_stream);
  }

  if (values) {
    if (value_sizes) {
      n = find_<true, true>(n, keys, hit_mask, value_stride, values, value_sizes);
    } else {
      n = find_<true, false>(n, keys, hit_mask, value_stride, values, value_sizes);
    }
  } else {
    if (value_sizes) {
      n = find_<false, true>(n, keys, hit_mask, value_stride, values, value_sizes);
    } else {
      n = find_<false, false>(n, keys, hit_mask, value_stride, values, value_sizes);
    }
  }

  auto counter = this->lookup_counter_storage(ctx);
  NVE_CHECK_(counter != nullptr, "Invalid key counter");
  *counter += n;
}

void RocksDBTable::insert(context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw,
                          const int64_t value_stride, const int64_t value_size,
                          buffer_ptr<const void> values_bw) {
  if (n <= 0) return;
  auto modify_stream{ctx->get_modify_stream()};

  const char* __restrict keys{nullptr};
  if (keys_bw) {
    auto p{keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream)};
    keys = static_cast<const char*>(p);
  }
  const char* __restrict values{nullptr};
  if (values_bw) {
    auto p{values_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream)};
    values = static_cast<const char*>(p);
  }

  std::unique_ptr<rocksdb::DB>& __restrict db{rdb_ctx_->db};
  rocksdb::ColumnFamilyHandle* const __restrict cf{col_families_.front()};

  const int64_t key_size{config_.key_size};
  const int64_t max_batch_size{std::min(n, config_.max_batch_size)};
  NVE_CHECK_(value_size >= 0 && value_size <= config_.max_value_size);

  // TODO: Prone to memory fragmentation. Use a persistent scratch buffer instead?
  rocksdb::WriteBatch batch;
  rocksdb::Slice k_view{nullptr, to_uint(key_size)};
  rocksdb::Slice v_view{nullptr, to_uint(value_size)};

  int64_t batch_size{};
  for (int64_t ij{}; ij < n; ++ij) {
    k_view.data_ = &keys[ij * key_size];
    v_view.data_ = &values[ij * value_stride];
    NVE_CHECK_(batch.Put(cf, k_view, v_view));
    if NVE_LIKELY_(++batch_size < max_batch_size) continue;

    NVE_CHECK_(db->Write(write_opts_, &batch));
    batch.Clear();
    batch_size = {};
  }
  if (batch_size) {
    NVE_CHECK_(db->Write(write_opts_, &batch));
  }
  flush_wal_();
}

int64_t RocksDBTable::size(context_ptr_t&) const {
  std::unique_ptr<rocksdb::DB>& __restrict db{rdb_ctx_->db};
  rocksdb::ColumnFamilyHandle* const __restrict cf{col_families_.front()};

  int64_t n{};
  std::unique_ptr<rocksdb::Iterator> it{db->NewIterator(read_opts_, cf)};
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    ++n;
  }
  NVE_CHECK_(it->status());
  return n;
}

int64_t RocksDBTable::get_keys(context_ptr_t&, const int64_t capacity, void* const keys_ptr) const {
  NVE_CHECK_(capacity >= 0, "capacity must not be negative");
  char* const __restrict keys{static_cast<char*>(keys_ptr)};

  std::unique_ptr<rocksdb::DB>& __restrict db{rdb_ctx_->db};
  rocksdb::ColumnFamilyHandle* const __restrict cf{col_families_.front()};
  const int64_t key_size{config_.key_size};

  int64_t n{};
  std::unique_ptr<rocksdb::Iterator> it{db->NewIterator(read_opts_, cf)};
  for (it->SeekToFirst(); it->Valid() && n < capacity; it->Next(), ++n) {
    const rocksdb::Slice key{it->key()};
    NVE_CHECK_(static_cast<int64_t>(key.size()) == key_size, "Unexpected key size in RocksDB table");
    std::memcpy(&keys[n * key_size], key.data(), to_uint(key_size));
  }
  NVE_CHECK_(it->status());
  return n;
}

void RocksDBTable::update(context_ptr_t& ctx, const int64_t n,
                          buffer_ptr<const void> keys_bw, const int64_t value_stride,
                          const int64_t value_size, buffer_ptr<const void> values_bw) {
  if (n <= 0) return;
  auto modify_stream{ctx->get_modify_stream()};

  const char* __restrict keys{nullptr};
  if (keys_bw) {
    auto p{keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream)};
    keys = static_cast<const char*>(p);
  }
  bitmask64_t* __restrict const hit_mask{[&](){
    const uint64_t size{to_uint(ceil_div(n, bitmask64::num_bits)) * sizeof(bitmask64_t)};
    auto p{ctx->get_buffer("hitmask", size, true /*host_alloc*/)};
    std::memset(p, 0, size);
    return static_cast<bitmask64_t*>(p);
  }()};
  const char* __restrict values{nullptr};
  if (values_bw) {
    auto p{values_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream)};
    values = static_cast<const char*>(p);
  }

  if (!find_<false, false>(n, keys, hit_mask, 0, nullptr, nullptr)) return;

  std::unique_ptr<rocksdb::DB>& __restrict db{rdb_ctx_->db};
  rocksdb::ColumnFamilyHandle* const __restrict cf{col_families_.front()};

  const int64_t key_size{config_.key_size};
  const int64_t max_batch_size{config_.max_batch_size};
  NVE_CHECK_(value_size >= 0 && value_size <= config_.max_value_size);

  // TODO: Prone to memory fragmentation. Use a persistent scratch buffer instead?
  rocksdb::WriteBatch batch;
  rocksdb::Slice k_view{nullptr, to_uint(key_size)};
  rocksdb::Slice v_view{nullptr, to_uint(value_size)};

  int64_t batch_size{};
  for (int64_t i0{}; i0 < n; i0 += bitmask64::num_bits) {
    for (auto it{hit_mask[i0 / bitmask64::num_bits]}; it; it = bitmask64::skip(it)) {
      const int64_t ij{i0 + bitmask64::next(it)};

      k_view.data_ = &keys[ij * key_size];
      v_view.data_ = &values[ij * value_stride];
      NVE_CHECK_(batch.Put(cf, k_view, v_view));
      if NVE_LIKELY_(++batch_size < max_batch_size) continue;

      NVE_CHECK_(db->Write(write_opts_, &batch));
      batch.Clear();
      batch_size = {};
    }
  }
  if (batch_size) {
    NVE_CHECK_(db->Write(write_opts_, &batch));
  }
  flush_wal_();
}

void RocksDBTable::update_accumulate(
    context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw, const int64_t update_stride,
    const int64_t update_size, buffer_ptr<const void> updates_bw, const DataType_t update_dtype) {
  if (n <= 0) return;
  auto modify_stream{ctx->get_modify_stream()};

  const char* __restrict keys{nullptr};
  if (keys_bw) {
    auto p{keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream)};
    keys = static_cast<const char*>(p);
  }
  bitmask64_t* __restrict const hit_mask{[&](){
    const uint64_t size{to_uint(ceil_div(n, bitmask64::num_bits)) * sizeof(bitmask64_t)};
    auto p{ctx->get_buffer("hitmask", size, true /*host_alloc*/)};
    std::memset(p, 0, size);
    return static_cast<bitmask64_t*>(p);
  }()};
  const char* __restrict updates{nullptr};
  if (updates_bw) {
    auto p{updates_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream)};
    updates = static_cast<const char*>(p);
  }

  std::unique_ptr<rocksdb::DB>& __restrict db{rdb_ctx_->db};
  rocksdb::ColumnFamilyHandle* const __restrict cf{col_families_.front()};

  const int64_t key_size{config_.key_size};
  const int64_t max_batch_size{config_.max_batch_size};
  const int64_t max_value_size{config_.max_value_size};
  NVE_CHECK_(update_size >= 0 && update_size % dtype_size(update_dtype) == 0);
  NVE_CHECK_(update_size / dtype_size(update_dtype) <= max_value_size / config_.value_dtype_size());
  const update_kernel_t update_kernel{pick_cpu_update_kernel(config_.value_dtype, update_dtype)};

  // TODO: Prone to memory fragmentation. Maybe should avoid running a full `find_`?
  std::vector<char> values(to_uint(n * max_value_size));
  std::vector<int64_t> value_sizes(to_uint(n));
  if (!find_<true, true>(n, keys, hit_mask, max_value_size, values.data(), value_sizes.data())) return;

  // TODO: Prone to memory fragmentation. Use a persistent scratch buffer instead?
  rocksdb::WriteBatch batch;
  rocksdb::Slice k_view{nullptr, to_uint(key_size)};

  int64_t batch_size{};
  for (int64_t i0{}; i0 < n; i0 += bitmask64::num_bits) {
    for (auto it{hit_mask[i0 / bitmask64::num_bits]}; it; it = bitmask64::skip(it)) {
      const int64_t ij{i0 + bitmask64::next(it)};

      char* value{&values[to_uint(ij * max_value_size)]};
      const int64_t value_size{value_sizes[to_uint(ij)]};
      update_kernel(value, value, value_size, &updates[ij * update_stride], update_size);

      k_view.data_ = &keys[ij * key_size];
      NVE_CHECK_(batch.Put(cf, k_view, {value, to_uint(value_size)}));
      if NVE_LIKELY_(++batch_size < max_batch_size) continue;

      NVE_CHECK_(db->Write(write_opts_, &batch));
      batch.Clear();
      batch_size = {};
    }
  }
  if (batch_size) {
    NVE_CHECK_(db->Write(write_opts_, &batch));
  }
  flush_wal_();
}

template <bool WithValues, bool WithValueSizes>
int64_t RocksDBTable::find_(int64_t n, const char* const __restrict keys, bitmask64_t* const __restrict hit_mask,
  int64_t value_stride, char* const __restrict values, int64_t* const __restrict value_sizes) const {
  std::unique_ptr<rocksdb::DB>& __restrict db{rdb_ctx_->db};
  const auto cfs{const_cast<rocksdb::ColumnFamilyHandle**>(col_families_.data())};

  const int64_t key_size{config_.key_size};
  const int64_t max_batch_size{std::min(n, config_.max_batch_size)};

  // TODO: Prone to memory fragmentation. Use a persistent scratch buffer instead?
  std::vector<rocksdb::Slice> k_views(to_uint(max_batch_size), {nullptr, to_uint(key_size)});
  std::vector<rocksdb::PinnableSlice> v_views(to_uint(max_batch_size));
  std::vector<rocksdb::Status> statuses(to_uint(max_batch_size));
  int64_t batch_size{};

  int64_t num_hits{};
  const auto process_batch{[&]() {
    db->MultiGet(read_opts_, to_uint(batch_size), cfs, k_views.data(), v_views.data(), nullptr, statuses.data());

    int64_t prev_i{-1};
    bitmask64_t mask{};

    for (int64_t k{}; k < batch_size; ++k) {
      const rocksdb::Status& __restrict status{statuses[to_uint(k)]};
      if (status.IsNotFound()) continue;
      NVE_CHECK_(status);

      // Reconstruct ij from the key view.
      const int64_t ij{(k_views[to_uint(k)].data() - keys) / key_size};
      const int64_t i{ij / bitmask64::num_bits};
      const int64_t j{ij % bitmask64::num_bits};

      if (i != prev_i) {
        if (prev_i >= 0) {
          hit_mask[prev_i] = mask;
        }
        prev_i = i;
        mask = hit_mask[i];
      }

      const rocksdb::PinnableSlice& __restrict v_view{v_views[to_uint(k)]};
      const int64_t value_size{static_cast<int64_t>(v_view.size())};

      if constexpr (WithValues) {
        NVE_CHECK_(value_size <= value_stride, "The value stored in RocksDB (=", value_size,
          ") exceeds the value stride (=", value_stride, ").");
        std::memcpy(&values[ij * value_stride], v_view.data(), to_uint(value_size));
      } else {
        NVE_ASSERT_(values == nullptr);
        (void)values;
        (void)value_stride;
      }

      if constexpr (WithValueSizes) {
        value_sizes[ij] = value_size;
      } else {
        NVE_ASSERT_(value_sizes == nullptr);
        (void)value_sizes;
      }

      mask |= bitmask64::single(j);
      ++num_hits;
    }

    if (prev_i >= 0) {
      hit_mask[prev_i] = mask;
    }
  }};
  
  for (int64_t i0{}; i0 < n; i0 += bitmask64::num_bits) {
    auto it{bitmask64::clip(~hit_mask[i0 / bitmask64::num_bits], n - i0)};

    // Run query if the batch is about to overflow.
    if (batch_size + bitmask64::count(it) > max_batch_size) {
      process_batch();
      batch_size = {};
    }

    for (; it; it = bitmask64::skip(it)) {
      const int64_t ij{i0 + bitmask64::next(it)};
      k_views[to_uint(batch_size++)].data_ = &keys[ij * key_size];
    }
  }
  if (batch_size) {
    process_batch();
  }

  return num_hits;
}

void RocksDBTableFactoryConfig::check() const {
  base_type::check();

  NVE_CHECK_(!path.empty());
  NVE_CHECK_(num_threads > 0);
}

void from_json(const nlohmann::json& json, RocksDBTableFactoryConfig& conf) {
  using base_type = RocksDBTableFactoryConfig::base_type;
  from_json(json, static_cast<base_type&>(conf));

  NVE_READ_JSON_FIELD_(path);
  NVE_READ_JSON_FIELD_(read_only);
  NVE_READ_JSON_FIELD_(num_threads);
}

void to_json(nlohmann::json& json, const RocksDBTableFactoryConfig& conf) {
  using base_type = RocksDBTableFactoryConfig::base_type;
  to_json(json, static_cast<const base_type&>(conf));

  NVE_WRITE_JSON_FIELD_(path);
  NVE_WRITE_JSON_FIELD_(read_only);
  NVE_WRITE_JSON_FIELD_(num_threads);
}

RocksDBTableFactory::RocksDBTableFactory(const RocksDBTableFactoryConfig& config)
    : base_type(config), rdb_ctx_{std::make_shared<RocksDBContext>(config.path, config.read_only, config.num_threads)} {}

host_table_ptr_t RocksDBTableFactory::produce(const table_id_t id,
                                              const table_config_type& config) {
  config.check();

  rdb_ctx_ptr_t& ctx{rdb_ctx_};
  rocksdb::ColumnFamilyHandle* cf;
  {
    std::lock_guard lock(ctx->write);
    const auto it{std::find_if(ctx->col_families.begin(), ctx->col_families.end(),
                               [&config](rocksdb::ColumnFamilyHandle* const cf) {
                                 return cf->GetName() == config.column_family;
                               })};
    if (it != ctx->col_families.end()) {
      cf = *it;
    } else if (ctx->read_only) {
      NVE_CHECK_(false, "Column family ", config.column_family, " not found. Cannot create new tables in read-only mode.");
    } else {
      NVE_CHECK_(
        ctx->db->CreateColumnFamily(ctx->col_family_opts, config.column_family, &cf));
      ctx->col_families.emplace_back(cf);
    }
  }

  return std::make_shared<RocksDBTable>(id, config, ctx, cf);
}

}  // namespace plugin
}  // namespace nve
