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

#include <map>
#include <vector>

namespace nve {

/* ============================================================================
 * CustomRemoteTable — A simple std::map-backed "remote" host table (sample).
 *
 * This table is intended as a readable reference implementation for custom
 * remote host tables.  A real deployment would replace the std::map with
 * network I/O to a parameter server or distributed KV store.
 * ============================================================================ */

class CustomRemoteTable final : public HostTableLike {
 public:
  CustomRemoteTable(table_id_t id, int64_t row_size);

  const HostTableConfig& config() const override { return config_; }
  int64_t size(context_ptr_t& ctx, bool exact) const override;

  void find(context_ptr_t& ctx, int64_t n, const void* keys,
            max_bitmask_repr_t* hit_mask, int64_t value_stride,
            void* values, int64_t* value_sizes) const override;

  void insert(context_ptr_t& ctx, int64_t n, const void* keys,
              int64_t value_stride, int64_t value_size,
              const void* values) override;

  void update(context_ptr_t& ctx, int64_t n, const void* keys,
              int64_t value_stride, int64_t value_size,
              const void* values) override;

  void update_accumulate(context_ptr_t& ctx, int64_t n,
                         const void* keys, int64_t update_stride,
                         int64_t update_size, const void* updates,
                         DataType_t update_dtype) override;

  void clear(context_ptr_t& ctx) override;

  void erase(context_ptr_t& ctx, int64_t n, const void* keys) override;

 private:
  HostTableConfig config_;
  int64_t row_size_;
  std::map<int64_t, std::vector<char>> store_;
};

/* ============================================================================
 * CustomRemoteTableFactory
 * ============================================================================ */

class CustomRemoteTableFactory final : public HostTableLikeFactory {
 public:
  CustomRemoteTableFactory() = default;

  host_table_ptr_t produce(table_id_t id, const nlohmann::json& json) override;
};

}  // namespace nve
