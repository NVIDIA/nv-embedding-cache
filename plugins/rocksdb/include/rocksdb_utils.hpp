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

#include <common.hpp>
#include <host_table.hpp>
#include <memory>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#include <rocksdb/db.h>
#pragma GCC diagnostic pop

namespace nve {

template <>
inline bool is_success(const rocksdb::Status& status) noexcept {
  return status.ok();
}

template <>
class RuntimeError<rocksdb::Status> : public Exception {
 public:
  using base_type = Exception;

  RuntimeError() = delete;

  inline RuntimeError(const char file[], const int line, const char expr[],
                      const rocksdb::Status& status, const std::string& hint) noexcept
      : base_type(file, line, expr, hint), status_{status} {}

  inline RuntimeError(const RuntimeError& that) noexcept : base_type(that) {}

  inline RuntimeError& operator=(const RuntimeError& that) noexcept {
    base_type::operator=(that);
    return *this;
  }

  inline rocksdb::Status status() const noexcept { return status_; }

  const char* what() const noexcept override;

  std::string to_string() const override;

 private:
  rocksdb::Status status_;
};

namespace plugin {

struct RocksDBContext final {
  NVE_PREVENT_COPY_AND_MOVE_(RocksDBContext);

  RocksDBContext() = delete;

  RocksDBContext(const std::string& path, bool read_only, int64_t num_threads);

  ~RocksDBContext();

  const std::string path;
  std::unique_ptr<rocksdb::DB> db;

  mutable std::mutex write;
  rocksdb::ColumnFamilyOptions col_family_opts;
  std::vector<rocksdb::ColumnFamilyHandle*> col_families;
};

}  // namespace plugin
}  // namespace nve
