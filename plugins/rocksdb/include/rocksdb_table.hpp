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
#include <rocksdb_utils.hpp>

namespace nve {
namespace plugin {

struct RocksDBTableConfig final : public HostTableConfig {
  using base_type = HostTableConfig;

  int64_t max_batch_size{16'384};  // Maximum batch size to use for queries into the column family.
                                   // Must be a multiple of `mask_size`.

  std::string column_family{rocksdb::kDefaultColumnFamilyName};
  bool verify_checksums{true};  // Toggle to `false` turn off checksum verifications.

  void check() const;
};

void from_json(const nlohmann::json& json, RocksDBTableConfig& conf);

void to_json(nlohmann::json& json, const RocksDBTableConfig& conf);

struct RocksDBContext;
using rdb_ctx_ptr_t = std::shared_ptr<RocksDBContext>;

template <typename MaskType>
class RocksDBTable final : public HostTable<RocksDBTableConfig> {
 public:
  using base_type = HostTable<RocksDBTableConfig>;
  using mask_type = MaskType;
  using mask_repr_type = typename mask_type::repr_type;

  NVE_PREVENT_COPY_AND_MOVE_(RocksDBTable);

  RocksDBTable() = delete;

  RocksDBTable(table_id_t id, const config_type& config, rdb_ctx_ptr_t& rdb_ctx,
               rocksdb::ColumnFamilyHandle* cf);

  ~RocksDBTable() override = default;

  void clear(context_ptr_t& ctx) override;

  void erase(context_ptr_t& ctx, int64_t n, const void* keys) override;

  void find(context_ptr_t& ctx, int64_t n, const void* keys, max_bitmask_repr_t* hit_mask,
            int64_t value_stride, void* values, int64_t* value_sizes) const override;

  void insert(context_ptr_t& ctx, int64_t n, const void* keys, int64_t value_stride,
              int64_t value_size, const void* values) override;

  int64_t size(context_ptr_t& ctx, bool exact) const override;

  void update(context_ptr_t& ctx, int64_t n, const void* keys, int64_t value_stride,
              int64_t value_size, const void* values) override;

  void update_accumulate(context_ptr_t& ctx, int64_t n, const void* keys, int64_t update_stride,
                         int64_t update_size, const void* updates,
                         DataType_t update_dtype) override;

 private:
  template <bool HasValues, bool HasValueSizes>
  int64_t find_(int64_t n, const char* keys, char* hit_mask,
                int64_t value_stride, char* values, int64_t* value_sizes) const;

 private:
  rdb_ctx_ptr_t rdb_ctx_;
  std::vector<rocksdb::ColumnFamilyHandle*> col_families_;
  rocksdb::ReadOptions read_opts_;
  rocksdb::WriteOptions write_opts_;
};

struct RocksDBTableFactoryConfig final : public HostTableFactoryConfig {
  using base_type = HostTableFactoryConfig;

  std::string path{"/tmp/rocksdb"};  // File-system path to the database.
  bool read_only{false};  // If \p true will open the database in read-only mode. Obviously, all
                          // write operations will fail. But it allows you querying the same RocksDB
                          // database from multiple clients.
  int64_t num_threads{16};  // Number of threads that the RocksDB instance may use.

  void check() const;
};

void from_json(const nlohmann::json& json, RocksDBTableFactoryConfig& conf);

void to_json(nlohmann::json& json, const RocksDBTableFactoryConfig& conf);

class RocksDBTableFactory final : public HostTableFactory<RocksDBTableFactoryConfig, RocksDBTableConfig> {
 public:
  using base_type = HostTableFactory<RocksDBTableFactoryConfig, RocksDBTableConfig>;

  NVE_PREVENT_COPY_AND_MOVE_(RocksDBTableFactory);

  RocksDBTableFactory() = delete;

  RocksDBTableFactory(const config_type& config);

  ~RocksDBTableFactory() override = default;

  host_table_ptr_t produce(table_id_t id, const table_config_type& config) override;

 private:
  rdb_ctx_ptr_t rdb_ctx_;
};

}  // namespace plugin
}  // namespace nve
