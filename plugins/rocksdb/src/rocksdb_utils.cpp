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

#include <rocksdb_utils.hpp>

namespace nve {

const char* RuntimeError<rocksdb::Status>::what() const noexcept {
  return hint().empty() ? status().getState() : hint().c_str();
}

std::string RuntimeError<rocksdb::Status>::to_string() const {
  std::ostringstream o;

  const char* const what{this->what()};
  o << "RocksDB runtime error " << static_cast<int>(status().code()) << '-'
    << static_cast<int>(status().subcode()) << " = '" << what << "' @ " << file() << ':' << line();
  const std::string& thread{thread_name()};
  if (!thread.empty()) {
    o << " in thread: '" << thread << '\'';
  }
  const char* const expr{expression()};
  if (what != expr) {
    o << ", expression: '" << expr << '\'';
  }
  const char* const sstr{status().getState()};
  if (what != sstr) {
    o << ", description: '" << sstr << '\'';
  }
  o << '.';

  return o.str();
}

namespace plugin {

RocksDBContext::RocksDBContext(const std::string& path, const bool read_only, const int64_t num_threads)
  : path{path} {
  NVE_LOG_INFO_("Connecting to RocksDB database '", path, "'.");

  // Enumerate existing column families.
  rocksdb::Options opts;
  opts.create_if_missing = true;
  opts.manual_wal_flush = true;
  opts.OptimizeForPointLookup(8);
  opts.OptimizeLevelStyleCompaction();
  NVE_CHECK_(num_threads <= 4096);
  opts.IncreaseParallelism(static_cast<int>(num_threads));

  std::vector<std::string> col_names;
  const auto status{rocksdb::DB::ListColumnFamilies(opts, path, &col_names)};
  if (!status.IsPathNotFound()) {
    NVE_CHECK_(status);
  }
  if (std::find(col_names.begin(), col_names.end(), rocksdb::kDefaultColumnFamilyName) ==
      col_names.end()) {
    col_names.emplace_back(rocksdb::kDefaultColumnFamilyName);
  }

  col_family_opts.OptimizeForPointLookup(8);
  col_family_opts.OptimizeLevelStyleCompaction();

  std::vector<rocksdb::ColumnFamilyDescriptor> col_descs;
  for (const std::string& cn : col_names) {
    NVE_LOG_INFO_("RocksDB database '", path, "'; found column family '", cn, "'.");
    col_descs.emplace_back(cn, col_family_opts);
  }

  // Connect to DB and request full access to all column families.
  col_families.reserve(col_descs.size());
  if (read_only) {
    NVE_CHECK_(rocksdb::DB::OpenForReadOnly(opts, path, col_descs, &col_families, &db));
  } else {
    NVE_CHECK_(rocksdb::DB::Open(opts, path, col_descs, &col_families, &db));
  }

  NVE_LOG_INFO_("Connection to RocksDB database '", path, "' successful.");
}

RocksDBContext::~RocksDBContext() {
  const std::lock_guard lock{write};
  NVE_LOG_INFO_("Disconnecting from RocksDB database '", path, "'.");

  for (rocksdb::ColumnFamilyHandle* const cf : col_families) {
    NVE_CHECK_(db->DestroyColumnFamilyHandle(cf));
  }
  col_families.clear();
  NVE_CHECK_(db->SyncWAL());
  NVE_CHECK_(db->Close());
  db.release();

  NVE_LOG_INFO_("Disconnection from RocksDB database '", path, "' successful.");
}

}  // namespace plugin
}  // namespace nve
