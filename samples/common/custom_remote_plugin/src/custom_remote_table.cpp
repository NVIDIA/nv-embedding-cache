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

#include <custom_remote_table.hpp>

#include <cstring>
#include <stdexcept>

namespace nve {

/* ============================================================================
 * CustomRemoteTable
 * ============================================================================ */

CustomRemoteTable::CustomRemoteTable(table_id_t id, int64_t row_size)
    : HostTableLike(id), row_size_(row_size) {
  config_.max_value_size = row_size_;
  config_.key_size       = sizeof(int64_t);
  config_.value_dtype    = DataType_t::Float32;
}

int64_t CustomRemoteTable::size(context_ptr_t& /*ctx*/, bool /*exact*/) const {
  return static_cast<int64_t>(store_.size());
}

/* -- find ------------------------------------------------------------------ */
void CustomRemoteTable::find(context_ptr_t& ctx, int64_t n, const void* keys,
                             max_bitmask_repr_t* hit_mask, int64_t value_stride,
                             void* values, int64_t* value_sizes) const {
  const auto* typed_keys = static_cast<const int64_t*>(keys);
  auto*       out        = static_cast<char*>(values);
  int64_t     hits       = 0;

  for (int64_t i = 0; i < n; ++i) {
    if (hit_mask && ((hit_mask[i / 64] >> (i % 64)) & 1)) continue;

    auto it = store_.find(typed_keys[i]);
    if (it == store_.end()) continue;

    if (out) {
      std::memcpy(out + i * value_stride, it->second.data(), static_cast<size_t>(row_size_));
      if (value_sizes) value_sizes[i] = row_size_;
    }

    if (hit_mask) hit_mask[i / 64] |= (uint64_t{1} << (i % 64));
    ++hits;
  }

  auto* counter = get_internal_counter(ctx);
  if (counter) *counter += hits;
}

/* -- insert ---------------------------------------------------------------- */
void CustomRemoteTable::insert(context_ptr_t& /*ctx*/, int64_t n,
                               const void* keys, int64_t value_stride,
                               int64_t value_size, const void* values) {
  const auto* typed_keys = static_cast<const int64_t*>(keys);
  const auto* in         = static_cast<const char*>(values);

  for (int64_t i = 0; i < n; ++i) {
    auto& slot = store_[typed_keys[i]];
    slot.assign(in + i * value_stride,
                in + i * value_stride + value_size);
  }
}

/* -- update ---------------------------------------------------------------- */
void CustomRemoteTable::update(context_ptr_t& /*ctx*/, int64_t n,
                               const void* keys, int64_t value_stride,
                               int64_t value_size, const void* values) {
  const auto* typed_keys = static_cast<const int64_t*>(keys);
  const auto* in         = static_cast<const char*>(values);

  for (int64_t i = 0; i < n; ++i) {
    auto it = store_.find(typed_keys[i]);
    if (it == store_.end()) continue;
    it->second.assign(in + i * value_stride,
                      in + i * value_stride + value_size);
  }
}

/* -- update_accumulate ----------------------------------------------------- */
void CustomRemoteTable::update_accumulate(context_ptr_t& /*ctx*/, int64_t n,
                                          const void* keys,
                                          int64_t update_stride,
                                          int64_t update_size,
                                          const void* updates,
                                          DataType_t update_dtype) {
  if (update_dtype != DataType_t::Float32) {
    throw std::invalid_argument(
        "custom_remote plugin only supports update_dtype=Float32 in update_accumulate");
  }
  const auto* typed_keys     = static_cast<const int64_t*>(keys);
  const auto* in             = static_cast<const float*>(updates);
  const int64_t floats_per_row = update_size / static_cast<int64_t>(sizeof(float));
  const int64_t float_stride   = update_stride / static_cast<int64_t>(sizeof(float));

  for (int64_t i = 0; i < n; ++i) {
    auto it = store_.find(typed_keys[i]);
    if (it == store_.end()) continue;

    auto* dst       = reinterpret_cast<float*>(it->second.data());
    const auto* src = in + i * float_stride;
    for (int64_t d = 0; d < floats_per_row; ++d) dst[d] += src[d];
  }
}

/* -- clear / erase --------------------------------------------------------- */
void CustomRemoteTable::clear(context_ptr_t& /*ctx*/) { store_.clear(); }

void CustomRemoteTable::erase(context_ptr_t& /*ctx*/, int64_t n,
                              const void* keys) {
  const auto* typed_keys = static_cast<const int64_t*>(keys);
  for (int64_t i = 0; i < n; ++i) store_.erase(typed_keys[i]);
}

/* ============================================================================
 * CustomRemoteTableFactory
 * ============================================================================ */

host_table_ptr_t CustomRemoteTableFactory::produce(table_id_t id,
                                                   const nlohmann::json& json) {
  if (json.contains("key_size")) {
    int64_t key_size = json.at("key_size").get<int64_t>();
    if (key_size != 8) {
      throw std::invalid_argument(
          "custom_remote plugin only supports key_size=8 (int64_t), got " +
          std::to_string(key_size));
    }
  }
  if (json.contains("mask_size")) {
    int64_t mask_size = json.at("mask_size").get<int64_t>();
    if (mask_size != 8) {
      throw std::invalid_argument(
          "custom_remote plugin only supports mask_size=8, got " +
          std::to_string(mask_size));
    }
  }

  int64_t max_value_size = 128;
  if (json.contains("max_value_size")) {
    max_value_size = json.at("max_value_size").get<int64_t>();
  }
  if (max_value_size <= 0) {
    throw std::invalid_argument("max_value_size must be positive");
  }
  return std::make_shared<CustomRemoteTable>(id, max_value_size);
}

}  // namespace nve
