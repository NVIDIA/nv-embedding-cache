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
#include <buffer_wrapper.hpp>

#include <cstring>
#include <stdexcept>
#include <string>

// Validates a caller-supplied argument and throws std::invalid_argument when it does not hold.
#define CHECK_ARG(_expr_, _msg_)                                                  \
  do {                                                                            \
    if (!(_expr_)) {                                                              \
      throw std::invalid_argument(std::string("custom_remote plugin: ") + (_msg_)); \
    }                                                                             \
  } while (false)

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

int64_t CustomRemoteTable::size(context_ptr_t& /*ctx*/) const {
  return static_cast<int64_t>(store_.size());
}

/* -- get_keys -------------------------------------------------------------- */
int64_t CustomRemoteTable::get_keys(context_ptr_t& /*ctx*/, int64_t capacity, void* keys) const {
  NVE_CHECK_(capacity >= 0, "capacity must not be negative");
  auto* out = static_cast<int64_t*>(keys);
  int64_t n = 0;
  for (auto it = store_.begin(); it != store_.end() && n < capacity; ++it) out[n++] = it->first;
  return n;
}

/* -- find ------------------------------------------------------------------ */
void CustomRemoteTable::find(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys,
                             buffer_ptr<bitmask64_t> hit_mask, int64_t value_stride,
                             buffer_ptr<void> values, buffer_ptr<int64_t> value_sizes) const {
  auto lookup_stream = ctx->get_lookup_stream();
  const void* keys_buf = keys ? keys->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, lookup_stream)
                              : nullptr;
  bitmask64_t* hit_mask_buf =
      hit_mask ? hit_mask->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, lookup_stream)
               : nullptr;
  void* values_buf = values ? values->access_buffer(cudaMemoryTypeUnregistered, false /*copy_content*/, lookup_stream)
                            : nullptr;
  int64_t* value_sizes_buf =
      value_sizes ? value_sizes->access_buffer(cudaMemoryTypeUnregistered, false /*copy_content*/, lookup_stream)
                  : nullptr;
  const auto* typed_keys = static_cast<const int64_t*>(keys_buf);
  auto*       out        = static_cast<char*>(values_buf);
  int64_t     hits       = 0;

  CHECK_ARG(n == 0 || typed_keys != nullptr, "find requires a non-null keys buffer");

  for (int64_t i = 0; i < n; ++i) {
    if (hit_mask_buf && ((hit_mask_buf[i / 64] >> (i % 64)) & 1)) continue;

    auto it = store_.find(typed_keys[i]);
    if (it == store_.end()) continue;

    if (out) {
      std::memcpy(out + i * value_stride, it->second.data(), static_cast<size_t>(row_size_));
      if (value_sizes_buf) value_sizes_buf[i] = row_size_;
    }

    if (hit_mask_buf) hit_mask_buf[i / 64] |= (uint64_t{1} << (i % 64));
    ++hits;
  }

  auto* counter = lookup_counter_storage(ctx);
  if (counter) *counter += hits;
}

/* -- insert ---------------------------------------------------------------- */
void CustomRemoteTable::insert(context_ptr_t& ctx, int64_t n,
                               buffer_ptr<const void> keys, int64_t value_stride,
                               int64_t value_size, buffer_ptr<const void> values) {
  auto modify_stream = ctx->get_modify_stream();
  const void* keys_buf = keys ? keys->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream)
                              : nullptr;
  const void* values_buf =
      values ? values->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream)
             : nullptr;
  const auto* typed_keys = static_cast<const int64_t*>(keys_buf);
  const auto* in         = static_cast<const char*>(values_buf);

  CHECK_ARG(n == 0 || (typed_keys != nullptr && in != nullptr),
            "insert requires non-null keys and values buffers");

  for (int64_t i = 0; i < n; ++i) {
    auto& slot = store_[typed_keys[i]];
    slot.assign(in + i * value_stride,
                in + i * value_stride + value_size);
  }
}

/* -- update ---------------------------------------------------------------- */
void CustomRemoteTable::update(context_ptr_t& ctx, int64_t n,
                               buffer_ptr<const void> keys, int64_t value_stride,
                               int64_t value_size, buffer_ptr<const void> values) {
  auto modify_stream = ctx->get_modify_stream();
  const void* keys_buf = keys ? keys->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream)
                              : nullptr;
  const void* values_buf =
      values ? values->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream)
             : nullptr;
  const auto* typed_keys = static_cast<const int64_t*>(keys_buf);
  const auto* in         = static_cast<const char*>(values_buf);

  CHECK_ARG(n == 0 || (typed_keys != nullptr && in != nullptr),
            "update requires non-null keys and values buffers");

  for (int64_t i = 0; i < n; ++i) {
    auto it = store_.find(typed_keys[i]);
    if (it == store_.end()) continue;
    it->second.assign(in + i * value_stride,
                      in + i * value_stride + value_size);
  }
}

/* -- update_accumulate ----------------------------------------------------- */
void CustomRemoteTable::update_accumulate(context_ptr_t& ctx, int64_t n,
                                          buffer_ptr<const void> keys,
                                          int64_t update_stride,
                                          int64_t update_size,
                                          buffer_ptr<const void> updates,
                                          DataType_t update_dtype) {
  CHECK_ARG(update_dtype == DataType_t::Float32,
            "only update_dtype=Float32 is supported in update_accumulate");
  auto modify_stream = ctx->get_modify_stream();
  const void* keys_buf = keys ? keys->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream)
                              : nullptr;
  const void* updates_buf =
      updates ? updates->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream)
              : nullptr;
  const auto* typed_keys     = static_cast<const int64_t*>(keys_buf);
  const auto* in             = static_cast<const float*>(updates_buf);
  const int64_t floats_per_row = update_size / static_cast<int64_t>(sizeof(float));
  const int64_t float_stride   = update_stride / static_cast<int64_t>(sizeof(float));

  CHECK_ARG(n == 0 || (typed_keys != nullptr && in != nullptr),
            "update_accumulate requires non-null keys and updates buffers");

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

void CustomRemoteTable::erase(context_ptr_t& ctx, int64_t n,
                              buffer_ptr<const void> keys) {
  const void* keys_buf = keys ? keys->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/,
                                                    ctx->get_modify_stream())
                              : nullptr;
  const auto* typed_keys = static_cast<const int64_t*>(keys_buf);
  CHECK_ARG(n == 0 || typed_keys != nullptr, "erase requires a non-null keys buffer");
  for (int64_t i = 0; i < n; ++i) store_.erase(typed_keys[i]);
}

/* ============================================================================
 * CustomRemoteTableFactory
 * ============================================================================ */

table_ptr_t CustomRemoteTableFactory::produce(table_id_t id,
                                              const nlohmann::json& json) {
  if (json.contains("key_size")) {
    int64_t key_size = json.at("key_size").get<int64_t>();
    CHECK_ARG(key_size == 8,
              "only key_size=8 (int64_t) is supported, got " + std::to_string(key_size));
  }

  int64_t max_value_size = 128;
  if (json.contains("max_value_size")) {
    max_value_size = json.at("max_value_size").get<int64_t>();
  }
  CHECK_ARG(max_value_size > 0,
            "max_value_size must be positive, got " + std::to_string(max_value_size));
  return std::make_shared<CustomRemoteTable>(id, max_value_size);
}

}  // namespace nve
