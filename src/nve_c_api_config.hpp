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

#include "nve_c_api_internal.hpp"

#include <gpu_embedding_layer.hpp>
#include <hierarchical_embedding_layer.hpp>
#include <host_embedding_layer.hpp>
#include <linear_embedding_layer.hpp>
#include <linear_host_table.hpp>

#include <cstring>
#include <iterator>
#include <type_traits>
#include <utility>

struct nve_config_field {
  size_t offset;
  size_t size;
};

// Converts to any member type, so T{nve_any_member_...} probes how many members an aggregate has.
struct nve_any_member_ {
  template <typename T>
  operator T() const;
};

template <typename T, typename Indices, typename = void>
struct nve_brace_constructible_ : std::false_type {};

template <typename T, size_t... I>
struct nve_brace_constructible_<T, std::index_sequence<I...>,
                                std::void_t<decltype(T{(void(I), nve_any_member_{})...})>>
    : std::true_type {};

// Number of direct members (a base class counts as one) of aggregate T.
template <typename T, size_t N = 0>
constexpr size_t nve_aggregate_member_count() {
  static_assert(std::is_aggregate_v<T>);
  if constexpr (nve_brace_constructible_<T, std::make_index_sequence<N + 1>>::value) {
    return nve_aggregate_member_count<T, N + 1>();
  } else {
    return N;
  }
}

// The opaque config header holds a per-type tag (upper 32 bits) written by *_config_init() and
// the caller's size (lower 32 bits), so creation rejects uninitialized or mistyped configs.
// The encoding is private to the library and may change.
inline constexpr unsigned nve_config_tag_shift = 32;
inline constexpr uint64_t nve_config_size_mask = 0xFFFFFFFFull;

// Keep the values distinct; each must be unlikely as leftover stack data.
enum : uint32_t {
  NVE_CONFIG_TAG_GPU_TABLE_ = 0x4E564301,
  NVE_CONFIG_TAG_LINEAR_HOST_TABLE_ = 0x4E564302,
  NVE_CONFIG_TAG_GPU_EMBEDDING_LAYER_ = 0x4E564303,
  NVE_CONFIG_TAG_LINEAR_UVM_LAYER_ = 0x4E564304,
  NVE_CONFIG_TAG_HIERARCHICAL_LAYER_ = 0x4E564305,
  NVE_CONFIG_TAG_HOST_EMBEDDING_LAYER_ = 0x4E564306,
};

template <typename Config>
struct nve_config_traits;

#define NVE_CONFIG_FIELD_(field) \
  {offsetof(Config, field), sizeof(Config::field)}

// Fails when a C++ config gains or loses members, so the C config cannot silently fall behind.
#define NVE_CHECK_CXX_CONFIG_MEMBERS_(CxxConfig, count)                         \
  static_assert(nve_aggregate_member_count<CxxConfig>() == (count),            \
                #CxxConfig " changed: update its C config, nve_config_traits, " \
                "the C++ config mapping, and this member count")

template <>
struct nve_config_traits<nve_gpu_table_config_t> {
  using Config = nve_gpu_table_config_t;
  static constexpr uint32_t tag = NVE_CONFIG_TAG_GPU_TABLE_;
  NVE_CHECK_CXX_CONFIG_MEMBERS_(nve::GPUTableConfig, 16);
  static Config defaults() {
    static const nve::GPUTableConfig defaults{};
    Config c{};
    c.device_id = defaults.device_id;
    c.cache_size = defaults.cache_size;
    c.row_size_in_bytes = defaults.row_size_in_bytes;
    c.uvm_table = defaults.uvm_table;
    c.uvm_num_rows = defaults.uvm_num_rows;
    c.count_misses = defaults.count_misses;
    c.max_modify_size = defaults.max_modify_size;
    c.value_dtype = nve::convert_external_dtype(defaults.value_dtype);
    c.private_stream = defaults.private_stream;
    c.disable_uvm_update = defaults.disable_uvm_update;
    c.uvm_cpu_accumulate = defaults.uvm_cpu_accumulate;
    c.data_storage_on_host = defaults.data_storage_on_host;
    c.modify_on_gpu = defaults.modify_on_gpu;
    c.kernel_mode_type = defaults.kernel_mode_type;
    c.kernel_mode_value = defaults.kernel_mode_value;
    c.invalid_key = defaults.invalid_key;
    return c;
  }
  inline static constexpr nve_config_field fields[] = {
    NVE_CONFIG_FIELD_(device_id),
    NVE_CONFIG_FIELD_(cache_size),
    NVE_CONFIG_FIELD_(row_size_in_bytes),
    NVE_CONFIG_FIELD_(uvm_table),
    NVE_CONFIG_FIELD_(count_misses),
    NVE_CONFIG_FIELD_(max_modify_size),
    NVE_CONFIG_FIELD_(value_dtype),
    NVE_CONFIG_FIELD_(private_stream),
    NVE_CONFIG_FIELD_(disable_uvm_update),
    NVE_CONFIG_FIELD_(uvm_cpu_accumulate),
    NVE_CONFIG_FIELD_(data_storage_on_host),
    NVE_CONFIG_FIELD_(modify_on_gpu),
    NVE_CONFIG_FIELD_(kernel_mode_type),
    NVE_CONFIG_FIELD_(kernel_mode_value),
    NVE_CONFIG_FIELD_(invalid_key),
    NVE_CONFIG_FIELD_(uvm_num_rows),
  };
};

template <>
struct nve_config_traits<nve_linear_host_table_config_t> {
  using Config = nve_linear_host_table_config_t;
  static constexpr uint32_t tag = NVE_CONFIG_TAG_LINEAR_HOST_TABLE_;
  NVE_CHECK_CXX_CONFIG_MEMBERS_(nve::HostTableConfig, 4);
  NVE_CHECK_CXX_CONFIG_MEMBERS_(nve::LinearHostTableConfig, 5);
  static Config defaults() {
    static const nve::LinearHostTableConfig defaults{};
    Config c{};
    c.emb_table = defaults.emb_table;
    c.num_rows = defaults.num_rows;
    c.row_size_in_bytes = defaults.max_value_size;
    c.value_dtype = nve::convert_external_dtype(defaults.value_dtype);
    c.max_threads = defaults.max_threads;
    c.task_size_threshold = defaults.task_size_threshold;
    return c;
  }
  inline static constexpr nve_config_field fields[] = {
    NVE_CONFIG_FIELD_(emb_table),
    NVE_CONFIG_FIELD_(num_rows),
    NVE_CONFIG_FIELD_(row_size_in_bytes),
    NVE_CONFIG_FIELD_(value_dtype),
    NVE_CONFIG_FIELD_(max_threads),
    NVE_CONFIG_FIELD_(task_size_threshold),
  };
};

// Static default objects keep C string and embedding-buffer pointers valid.
template <>
struct nve_config_traits<nve_gpu_embedding_layer_config_t> {
  using Config = nve_gpu_embedding_layer_config_t;
  static constexpr uint32_t tag = NVE_CONFIG_TAG_GPU_EMBEDDING_LAYER_;
  NVE_CHECK_CXX_CONFIG_MEMBERS_(nve::GPUEmbeddingLayerConfig, 7);
  static Config defaults() {
    static const nve::GPUEmbeddingLayerConfig defaults{};
    Config c{};
    c.layer_name = defaults.layer_name.c_str();
    c.device_id = defaults.device_id;
    c.embedding_table = defaults.embedding_table;
    c.num_embeddings = defaults.num_embeddings;
    c.embedding_width_in_bytes = defaults.embedding_width_in_bytes;
    c.value_dtype = nve::convert_external_dtype(defaults.value_dtype);
    c.default_row_index = defaults.default_row_index;
    return c;
  }
  inline static constexpr nve_config_field fields[] = {
    NVE_CONFIG_FIELD_(layer_name),
    NVE_CONFIG_FIELD_(device_id),
    NVE_CONFIG_FIELD_(embedding_table),
    NVE_CONFIG_FIELD_(num_embeddings),
    NVE_CONFIG_FIELD_(embedding_width_in_bytes),
    NVE_CONFIG_FIELD_(value_dtype),
    NVE_CONFIG_FIELD_(default_row_index),
  };
};

template <>
struct nve_config_traits<nve_linear_uvm_layer_config_t> {
  using Config = nve_linear_uvm_layer_config_t;
  static constexpr uint32_t tag = NVE_CONFIG_TAG_LINEAR_UVM_LAYER_;
  NVE_CHECK_CXX_CONFIG_MEMBERS_(nve::LinearUVMEmbeddingLayer<int64_t>::Config, 5);
  static Config defaults() {
    static const nve::LinearUVMEmbeddingLayer<int64_t>::Config defaults{};
    Config c{};
    c.layer_name = defaults.layer_name.c_str();
    c.insert_heuristic = nullptr;  // NULL selects the C++ config default.
    c.min_insert_freq_gpu = defaults.min_insert_freq_gpu;
    c.min_insert_size_gpu = defaults.min_insert_size_gpu;
    c.default_row_index = defaults.default_row_index;
    return c;
  }
  inline static constexpr nve_config_field fields[] = {
    NVE_CONFIG_FIELD_(layer_name),
    NVE_CONFIG_FIELD_(insert_heuristic),
    NVE_CONFIG_FIELD_(min_insert_freq_gpu),
    NVE_CONFIG_FIELD_(min_insert_size_gpu),
    NVE_CONFIG_FIELD_(default_row_index),
  };
};

template <>
struct nve_config_traits<nve_hierarchical_layer_config_t> {
  using Config = nve_hierarchical_layer_config_t;
  static constexpr uint32_t tag = NVE_CONFIG_TAG_HIERARCHICAL_LAYER_;
  NVE_CHECK_CXX_CONFIG_MEMBERS_(nve::HierarchicalEmbeddingLayer<int64_t>::Config, 7);
  static Config defaults() {
    static const nve::HierarchicalEmbeddingLayer<int64_t>::Config defaults{};
    Config c{};
    c.layer_name = defaults.layer_name.c_str();
    c.insert_heuristic = nullptr;  // NULL selects the C++ config default.
    c.min_insert_freq_gpu = defaults.min_insert_freq_gpu;
    c.min_insert_freq_host = defaults.min_insert_freq_host;
    c.min_insert_size_gpu = defaults.min_insert_size_gpu;
    c.min_insert_size_host = defaults.min_insert_size_host;
    c.default_embedding =
        defaults.default_embedding.empty() ? nullptr : defaults.default_embedding.data();
    c.default_embedding_size = static_cast<int64_t>(defaults.default_embedding.size());
    return c;
  }
  inline static constexpr nve_config_field fields[] = {
    NVE_CONFIG_FIELD_(layer_name),
    NVE_CONFIG_FIELD_(insert_heuristic),
    NVE_CONFIG_FIELD_(min_insert_freq_gpu),
    NVE_CONFIG_FIELD_(min_insert_freq_host),
    NVE_CONFIG_FIELD_(min_insert_size_gpu),
    NVE_CONFIG_FIELD_(min_insert_size_host),
    NVE_CONFIG_FIELD_(default_embedding),
    NVE_CONFIG_FIELD_(default_embedding_size),
  };
};

template <>
struct nve_config_traits<nve_host_embedding_layer_config_t> {
  using Config = nve_host_embedding_layer_config_t;
  static constexpr uint32_t tag = NVE_CONFIG_TAG_HOST_EMBEDDING_LAYER_;
  NVE_CHECK_CXX_CONFIG_MEMBERS_(nve::HostEmbeddingLayer<int64_t>::Config, 4);
  static Config defaults() {
    static const nve::HostEmbeddingLayer<int64_t>::Config defaults{};
    Config c{};
    c.layer_name = defaults.layer_name.c_str();
    c.default_embedding =
        defaults.default_embedding.empty() ? nullptr : defaults.default_embedding.data();
    c.default_embedding_size = static_cast<int64_t>(defaults.default_embedding.size());
    c.task_size_threshold = defaults.task_size_threshold;
    c.disable_threadpool = defaults.disable_threadpool;
    return c;
  }
  inline static constexpr nve_config_field fields[] = {
    NVE_CONFIG_FIELD_(layer_name),
    NVE_CONFIG_FIELD_(default_embedding),
    NVE_CONFIG_FIELD_(default_embedding_size),
    NVE_CONFIG_FIELD_(task_size_threshold),
    NVE_CONFIG_FIELD_(disable_threadpool),
  };
};

#undef NVE_CHECK_CXX_CONFIG_MEMBERS_
#undef NVE_CONFIG_FIELD_

// True when fields[] registers every C config member after the header exactly once.
template <typename Config>
constexpr bool nve_config_fields_complete() {
  const auto& fields = nve_config_traits<Config>::fields;
  for (size_t i = 0; i < std::size(fields); ++i) {
    for (size_t j = i + 1; j < std::size(fields); ++j) {
      if (fields[i].offset == fields[j].offset) {
        return false;
      }
    }
  }
  return std::size(fields) + 1 == nve_aggregate_member_count<Config>();
}

template <typename Config>
Config nve_config_defaults() {
  static_assert(std::is_standard_layout_v<Config>);
  static_assert(offsetof(Config, header) == 0);
  static_assert(nve_config_fields_complete<Config>(),
                "every C config field after header must be registered in nve_config_traits::fields");
  return nve_config_traits<Config>::defaults();
}

template <typename Config>
nve_status_t nve_validate_config_size(uint64_t size) {
  if (size < sizeof(uint64_t)) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "config size must include the config header");
  }
  if (size > sizeof(Config)) {
    return nve_set_error(NVE_ERROR_NOT_IMPLEMENTED, "config is newer than this library supports");
  }
  for (const auto& field : nve_config_traits<Config>::fields) {
    if (size > field.offset && size < field.offset + field.size) {
      return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "config size ends inside a field");
    }
  }
  return NVE_SUCCESS;
}

template <typename Config>
void nve_copy_config_fields(Config* destination, const Config* source, uint64_t size) {
  auto* dst = reinterpret_cast<unsigned char*>(destination);
  const auto* src = reinterpret_cast<const unsigned char*>(source);
  for (const auto& field : nve_config_traits<Config>::fields) {
    if (field.offset + field.size <= size) {
      std::memcpy(dst + field.offset, src + field.offset, field.size);
    }
  }
}

template <typename Config>
nve_status_t nve_init_config(Config* config, size_t size) {
  if (!config) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "config must not be NULL");
  }
  const auto status = nve_validate_config_size<Config>(size);
  if (status != NVE_SUCCESS) {
    return status;
  }
  const auto defaults = nve_config_defaults<Config>();
  // Clear padding too, but never write past the caller's prefix.
  std::memset(config, 0, size);
  nve_copy_config_fields(config, &defaults, size);
  const uint64_t encoded =
      (uint64_t{nve_config_traits<Config>::tag} << nve_config_tag_shift) | size;
  std::memcpy(config, &encoded, sizeof(encoded));
  return NVE_SUCCESS;
}

template <typename Config>
nve_status_t nve_read_config(const Config* config, Config& normalized) {
  if (!config) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "config must not be NULL");
  }
  uint64_t encoded;
  std::memcpy(&encoded, config, sizeof(encoded));
  if ((encoded >> nve_config_tag_shift) != nve_config_traits<Config>::tag) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT,
                         "config was not initialized by its matching *_config_init() function");
  }
  const uint64_t size = encoded & nve_config_size_mask;
  const auto status = nve_validate_config_size<Config>(size);
  if (status != NVE_SUCCESS) {
    return status;
  }
  normalized = nve_config_defaults<Config>();
  nve_copy_config_fields(&normalized, config, size);
  return NVE_SUCCESS;
}
