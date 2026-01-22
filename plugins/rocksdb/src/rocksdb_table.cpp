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

namespace nve {
namespace plugin {

void RocksDBTableConfig::check() const {
  base_type::check();

  NVE_CHECK_(max_batch_size > 0 && max_batch_size % mask_size == 0);

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

template <typename MaskType>
RocksDBTable<MaskType>::RocksDBTable(const table_id_t id, const RocksDBTableConfig& config,
                                     rdb_ctx_ptr_t& rdb_ctx,
                                     rocksdb::ColumnFamilyHandle* const cf)
    : base_type(id, config),
      rdb_ctx_{rdb_ctx},
      col_families_(static_cast<uint64_t>(config.max_batch_size), cf) {
  read_opts_.verify_checksums = config.verify_checksums;
  write_opts_.sync = false;
}

template <typename MaskType>
void RocksDBTable<MaskType>::clear(context_ptr_t&) {
  std::unique_ptr<rocksdb::DB>& __restrict db{rdb_ctx_->db};
  rocksdb::ColumnFamilyHandle* const __restrict cf{col_families_.front()};

  NVE_CHECK_(db->DeleteRange(write_opts_, cf, "", "~"));
}

template <typename MaskType>
void RocksDBTable<MaskType>::erase(context_ptr_t&, const int64_t n, const void* const keys_vptr) {
  if (n <= 0) return;

  const char* const __restrict keys{reinterpret_cast<const char*>(keys_vptr)};

  const auto& __restrict config{config_};
  std::unique_ptr<rocksdb::DB>& __restrict db{rdb_ctx_->db};
  rocksdb::ColumnFamilyHandle* const __restrict cf{col_families_.front()};
  const rocksdb::WriteOptions& __restrict write_opts{write_opts_};

  const int64_t key_size{config.key_size};
  const int64_t max_batch_size{std::min(n, config.max_batch_size)};

  // TODO: Prone to memory fragmentation. Use a persistent scratch buffer instead?
  rocksdb::WriteBatch batch;
  rocksdb::Slice k_view{nullptr, static_cast<uint64_t>(key_size)};

  int64_t batch_size{};
  for (int64_t i{}; i != n; ++i) {
    k_view.data_ = &keys[i * key_size];
    NVE_CHECK_(batch.Delete(cf, k_view));
    if NVE_LIKELY_(++batch_size < max_batch_size) continue;

    NVE_CHECK_(db->Write(write_opts, &batch));
    batch.Clear();
    batch_size = {};
  }
  if (batch_size) {
    NVE_CHECK_(db->Write(write_opts_, &batch));
  }
}

template <typename MaskType>
void RocksDBTable<MaskType>::find(context_ptr_t& ctx, int64_t n, const void* const keys_vptr,
                                  max_bitmask_repr_t* const hit_mask, const int64_t value_stride,
                                  void* const values_vptr, int64_t* const value_sizes) const {
  if (n <= 0) return;

  const char* const __restrict keys{reinterpret_cast<const char*>(keys_vptr)};
  char* const __restrict hm{reinterpret_cast<char*>(hit_mask)};
  char* const __restrict values{reinterpret_cast<char*>(values_vptr)};

  if (values) {
    if (value_sizes) {
      n = find_<true, true>(n, keys, hm, value_stride, values, value_sizes);
    } else {
      n = find_<true, false>(n, keys, hm, value_stride, values, value_sizes);
    }
  } else {
    if (value_sizes) {
      n = find_<false, true>(n, keys, hm, value_stride, values, value_sizes);
    } else {
      n = find_<false, false>(n, keys, hm, value_stride, values, value_sizes);
    }
  }

  auto counter = this->get_internal_counter(ctx);
  NVE_CHECK_(counter != nullptr, "Invalid key counter");
  *counter += n;
}

template <typename MaskType>
void RocksDBTable<MaskType>::insert(context_ptr_t&, const int64_t n, const void* const keys_vptr,
                                    const int64_t value_stride, const int64_t value_size,
                                    const void* const values_vptr) {
  if (n <= 0) return;

  const char* const __restrict keys{reinterpret_cast<const char*>(keys_vptr)};
  const char* const __restrict values{reinterpret_cast<const char*>(values_vptr)};

  const auto& __restrict config{config_};
  std::unique_ptr<rocksdb::DB>& __restrict db{rdb_ctx_->db};
  rocksdb::ColumnFamilyHandle* const __restrict cf{col_families_.front()};
  const rocksdb::WriteOptions& __restrict write_opts{write_opts_};

  const int64_t key_size{config.key_size};
  const int64_t max_batch_size{std::min(n, config.max_batch_size)};
  NVE_CHECK_(value_size >= 0 && value_size <= config.max_value_size);

  // TODO: Prone to memory fragmentation. Use a persistent scratch buffer instead?
  rocksdb::WriteBatch batch;
  rocksdb::Slice k_view{nullptr, static_cast<uint64_t>(key_size)};
  rocksdb::Slice v_view{nullptr, static_cast<uint64_t>(value_size)};

  int64_t batch_size{};
  for (int64_t i{}; i != n; ++i) {
    k_view.data_ = &keys[i * key_size];
    v_view.data_ = &values[i * value_stride];
    NVE_CHECK_(batch.Put(cf, k_view, v_view));
    if NVE_LIKELY_(++batch_size < max_batch_size) continue;

    NVE_CHECK_(db->Write(write_opts, &batch));
    batch.Clear();
    batch_size = {};
  }
  if (batch_size) {
    NVE_CHECK_(db->Write(write_opts_, &batch));
  }
}

template <typename MaskType>
int64_t RocksDBTable<MaskType>::size(context_ptr_t&, const bool exact) const {
  std::unique_ptr<rocksdb::DB>& __restrict db{rdb_ctx_->db};
  rocksdb::ColumnFamilyHandle* const __restrict cf{col_families_.front()};

  uint64_t n{};
  if (exact) {
    std::unique_ptr<rocksdb::Iterator> it{db->NewIterator(read_opts_, cf)};
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      ++n;
    }
  } else {
    NVE_CHECK_(db->GetIntProperty(cf, rocksdb::DB::Properties::kEstimateNumKeys, &n));
  }
  return static_cast<int64_t>(n);
}

template <typename MaskType>
void RocksDBTable<MaskType>::update(context_ptr_t&, const int64_t n,
                                    const void* const keys_vptr, const int64_t value_stride,
                                    const int64_t value_size, const void* const values_vptr) {
  if (n <= 0) return;

  const char* const __restrict keys{reinterpret_cast<const char*>(keys_vptr)};
  const char* const __restrict values{reinterpret_cast<const char*>(values_vptr)};

  // TODO: Prone to memory fragmentation. Also inefficient. Could we do this with MergeOperator's?
  std::vector<max_bitmask_repr_t> hit_mask(static_cast<uint64_t>(max_bitmask_t::mask_size(n)), {});
  char* const __restrict hm{reinterpret_cast<char*>(hit_mask.data())};
  const int64_t num_hits{find_<false, false>(n, keys, hm, value_stride, nullptr, nullptr)};
  if (!num_hits) return;

  const auto& __restrict config{config_};
  std::unique_ptr<rocksdb::DB>& __restrict db{rdb_ctx_->db};
  rocksdb::ColumnFamilyHandle* const __restrict cf{col_families_.front()};
  const rocksdb::WriteOptions& __restrict write_opts{write_opts_};

  const int64_t key_size{config.key_size};
  const int64_t max_batch_size{config.max_batch_size};
  NVE_CHECK_(value_size >= 0 && value_size <= config.max_value_size);

  // TODO: Prone to memory fragmentation. Use a persistent scratch buffer instead?
  rocksdb::WriteBatch batch;
  rocksdb::Slice k_view{nullptr, static_cast<uint64_t>(key_size)};
  rocksdb::Slice v_view{nullptr, static_cast<uint64_t>(value_size)};

  int64_t batch_size{};
  for (int64_t i{}; i < n; i += mask_type::num_bits) {
    for (auto it{mask_type::load(hm, i)}; mask_type::has_next(it); it = mask_type::skip(it)) {
      const int64_t ij{i + mask_type::next(it)};

      k_view.data_ = &keys[ij * key_size];
      v_view.data_ = &values[ij * value_stride];
      NVE_CHECK_(batch.Put(cf, k_view, v_view));
      if NVE_LIKELY_(++batch_size < max_batch_size) continue;

      NVE_CHECK_(db->Write(write_opts, &batch));
      batch.Clear();
      batch_size = {};
    }
  }
  if (batch_size) {
    NVE_CHECK_(db->Write(write_opts_, &batch));
  }
}

template <typename MaskType>
void RocksDBTable<MaskType>::update_accumulate(
    context_ptr_t&, const int64_t n, const void* const keys_vptr, const int64_t update_stride,
    const int64_t update_size, const void* const updates_vptr, const DataType_t update_dtype) {
  if (n <= 0) return;

  const char* const __restrict keys{reinterpret_cast<const char*>(keys_vptr)};
  const char* const __restrict updates{reinterpret_cast<const char*>(updates_vptr)};

  // TODO: Prone to memory fragmentation. Also inefficient. Could we do this with MergeOperator's?
  std::vector<max_bitmask_repr_t> hit_mask(static_cast<uint64_t>(max_bitmask_t::mask_size(n)), {});
  char* const __restrict hm{reinterpret_cast<char*>(hit_mask.data())};
  const int64_t value_stride{config_.max_value_size};
  std::vector<char> values_vec(static_cast<uint64_t>(n * value_stride));
  char* const __restrict values{values_vec.data()};
  std::vector<int64_t> value_sizes_vec(static_cast<uint64_t>(n));
  int64_t* const __restrict value_sizes{value_sizes_vec.data()};
  const int64_t num_hits{find_<true, true>(n, keys, hm, value_stride, values, value_sizes)};
  if (!num_hits) return;
  
  const auto& __restrict config{config_};
  std::unique_ptr<rocksdb::DB>& __restrict db{rdb_ctx_->db};
  rocksdb::ColumnFamilyHandle* const __restrict cf{col_families_.front()};
  const rocksdb::WriteOptions& __restrict write_opts{write_opts_};

  const int64_t key_size{config.key_size};
  const int64_t max_batch_size{config.max_batch_size};
  NVE_CHECK_(update_size >= 0 && update_size <= value_stride);
  const update_kernel_t update_kernel{pick_cpu_update_kernel(config.value_dtype, update_dtype)};

  // TODO: Prone to memory fragmentation. Use a persistent scratch buffer instead?
  rocksdb::WriteBatch batch;
  rocksdb::Slice k_view{nullptr, static_cast<uint64_t>(key_size)};

  int64_t batch_size{};
  for (int64_t i{}; i < n; i += mask_type::num_bits) {
    for (mask_repr_type it{mask_type::load(hm, i)}; mask_type::has_next(it);
         it = mask_type::skip(it)) {
      const int64_t ij{i + mask_type::next(it)};
      update_kernel(&values[ij * value_stride], &updates[ij * update_stride], update_size);

      k_view.data_ = &keys[ij * key_size];
      const int64_t value_size{std::max(value_sizes[ij], update_size)};
      NVE_CHECK_(batch.Put(cf, k_view,
                           {&values[ij * value_stride], static_cast<uint64_t>(value_size)}));
      if NVE_LIKELY_(++batch_size < max_batch_size) continue;

      NVE_CHECK_(db->Write(write_opts, &batch));
      batch.Clear();
      batch_size = {};
    }
  }
  if (batch_size) {
    NVE_CHECK_(db->Write(write_opts, &batch));
  }
}

template <typename MaskType>
template <bool HasValues, bool HasValueSizes>
int64_t RocksDBTable<MaskType>::find_(int64_t n, const char* const __restrict keys, char* const __restrict hm,
  int64_t value_stride, char* const __restrict values, int64_t* const __restrict value_sizes) const {
  const auto& __restrict config{config_};
  std::unique_ptr<rocksdb::DB>& __restrict db{rdb_ctx_->db};
  const auto cfs{const_cast<rocksdb::ColumnFamilyHandle**>(col_families_.data())};
  const rocksdb::ReadOptions& __restrict read_opts{read_opts_};

  const int64_t key_size{config.key_size};
  const int64_t max_batch_size{std::min(n, config.max_batch_size)};

  // TODO: Prone to memory fragmentation. Use a persistent scratch buffer instead?
  std::vector<rocksdb::Slice> k_views_vec(static_cast<uint64_t>(max_batch_size), {nullptr, static_cast<uint64_t>(key_size)});
  rocksdb::Slice* const __restrict k_views{k_views_vec.data()};
  std::vector<rocksdb::PinnableSlice> v_views_vec(static_cast<uint64_t>(max_batch_size));
  rocksdb::PinnableSlice* const __restrict v_views{v_views_vec.data()};
  std::vector<rocksdb::Status> statuses_vec(static_cast<uint64_t>(max_batch_size));
  rocksdb::Status* const __restrict statuses{statuses_vec.data()};
  int64_t batch_size{};

  int64_t num_hits{};
  const auto process_batch{[&]() {
    db->MultiGet(read_opts, static_cast<uint64_t>(batch_size), cfs,
                 k_views, v_views, nullptr, statuses);

    int64_t prev_i{-1};
    mask_repr_type mask;

    for (int64_t k{}; k < batch_size; ++k) {
      const rocksdb::Status& __restrict status{statuses[k]};
      if (status.IsNotFound()) continue;
      NVE_CHECK_(status);

      // Reconstruct ij from the key view.
      const int64_t ij{(k_views[k].data() - keys) / key_size};
      const int64_t i{ij & ~mask_type::num_bits_mask};
      const int64_t j{ij & mask_type::num_bits_mask};

      if (i != prev_i) {
        if (prev_i >= 0) {
          mask_type::store(hm, prev_i, mask);
        }
        prev_i = i;
        mask = mask_type::load(hm, i);
      }
#pragma GCC diagnostic push
#if !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
      mask = mask_type::insert(mask, j);
#pragma GCC diagnostic pop
      ++num_hits;

      const rocksdb::PinnableSlice& __restrict v_view{v_views[k]};
      const int64_t value_size{static_cast<int64_t>(v_view.size())};
      if constexpr (HasValues) {
        NVE_CHECK_(value_size <= value_stride);
        std::copy_n(v_view.data(), value_size, &values[ij * value_stride]);
      }
      if constexpr (HasValueSizes) {
        value_sizes[ij] = value_size;
      }
    }

    if (prev_i >= 0) {
      mask_type::store(hm, prev_i, mask);
    }
  }};
  
  for (int64_t i{}; i < n; i += mask_type::num_bits) {
    auto it{mask_type::clip(mask_type::invert(mask_type::load(hm, i)), n - i)};

    // Run query if the batch is about to overflow.
    if (batch_size + mask_type::count(it) > max_batch_size) {
      process_batch();
      batch_size = {};
    }

    for (; mask_type::has_next(it); it = mask_type::skip(it)) {
      const int64_t ij{i + mask_type::next(it)};
      k_views[batch_size++].data_ = &keys[ij * key_size];
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
    } else {
      NVE_CHECK_(
        ctx->db->CreateColumnFamily(ctx->col_family_opts, config.column_family, &cf));
        ctx->col_families.emplace_back(cf);
    }
  }

  switch (config.mask_size) {
#if defined(NVE_FEATURE_HT_MASK_8)
    case bitmask8_t::size:
      return std::make_shared<RocksDBTable<bitmask8_t>>(id, config, ctx, cf);
#endif
#if defined(NVE_FEATURE_HT_MASK_16)
    case bitmask16_t::size:
      return std::make_shared<RocksDBTable<bitmask16_t>>(id, config, ctx, cf);
#endif
#if defined(NVE_FEATURE_HT_MASK_32)
    case bitmask32_t::size:
      return std::make_shared<RocksDBTable<bitmask32_t>>(id, config, ctx, cf);
#endif
#if defined(NVE_FEATURE_HT_MASK_64)
    case bitmask64_t::size:
      return std::make_shared<RocksDBTable<bitmask64_t>>(id, config, ctx, cf);
#endif
  }
  NVE_THROW_("`config.mask_size` (", config.mask_size, ") is out of bounds!");
}

}  // namespace plugin
}  // namespace nve
