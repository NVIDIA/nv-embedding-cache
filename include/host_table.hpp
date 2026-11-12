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

#include <bit_ops.hpp>
#include <chrono>
#include <common.hpp>
#include <cstdint>
#include <json_support.hpp>
#include <limits>
#include <map>
#include <optional>
#include <plugin/plugin.hpp>
#include <string>
#include <string_view>
#include <table.hpp>
#include <execution_context.hpp>
#include <variant>
#include <vector>

namespace nve {

struct Partitioner {};

struct AlwaysZeroPartitioner final : public Partitioner {
  static constexpr char name[]{"always_zero"};

  template <typename KeyType>
  constexpr int64_t operator()(const KeyType, const int64_t) const noexcept { return {}; }
};

/**
 * Adaption of the `fnv1a_pippip_yuri` hash function for use as a numeric mixer.
 *
 * In essence this is FNV1a (same offset basis), but with a larger prime to process 64-bit at a time.
 *
 * See also:
 * https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
 * https://github.com/rurban/smhasher/blob/a75e2a9d27444d823cf60211d25d6a3564af1b20/Hashes.cpp
 */
struct FowlerNollVoPartitioner final : public Partitioner {
  static constexpr char name[]{"fowler_noll_vo"};

  template <typename KeyType>
  constexpr int64_t operator()(const KeyType key, const int64_t mask) const noexcept {
    constexpr std::hash<KeyType> hash;
    uint64_t x{hash(key)};

    x ^= UINT64_C(14'695'981'039'346'656'037);
    x *= UINT64_C(591'798'841);
    x ^= x >> 32;
    x ^= x >> 16;

    return static_cast<int64_t>(x) & mask;
  }
};

/**
 * Public domain 64-bit numeric mixer created by Austin Appleby for the murmur3 hash function.
 *
 * https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
 */
struct Murmur3Partitioner final : public Partitioner {
  static constexpr char name[]{"murmur3"};

  template <typename KeyType>
  constexpr int64_t operator()(const KeyType key, const int64_t mask) const noexcept {
    constexpr std::hash<KeyType> hash;
    uint64_t x{hash(key)};

    x ^= x >> 33;
    x *= UINT64_C(0xff51'afd7'ed55'8ccd);
    x ^= x >> 33;
    x *= UINT64_C(0xc4ce'b9fe'1a85'ec53);
    x ^= x >> 33;

    return static_cast<int64_t>(x) & mask;
  }
};

/**
 * A fairly strong but simple public domain 64-bit numeric mixer by Pelle Evensen.
 *
 * https://mostlymangling.blogspot.com/2019/01/better-stronger-mixer-and-test-procedure.html
 */
struct Rrxmrrxmsx0Partitioner final : public Partitioner {
  static constexpr char name[]{"rrxmrrxmsx0"};

  template <typename KeyType>
  constexpr int64_t operator()(const KeyType key, const int64_t mask) const noexcept {
    constexpr std::hash<KeyType> hash;
    uint64_t x{hash(key)};

    x ^= rotr(x, 25) ^ rotr(x, 50);
    x *= UINT64_C(0xa24b'aed4'963e'e407);
    x ^= rotr(x, 24) ^ rotr(x, 49);
    x *= UINT64_C(0x9fb2'1c65'1e98'df25);
    x ^= x >> 28;

    return static_cast<int64_t>(x) & mask;
  }
};

struct StdHashPartitioner final : public Partitioner {
  static constexpr char name[]{"std_hash"};

  template <typename KeyType>
  constexpr int64_t operator()(const KeyType key, const int64_t mask) const noexcept {
    constexpr std::hash<KeyType> hash;
    uint64_t x{hash(key)};

    return static_cast<int64_t>(x) & mask;
  }
};

/**
 * Set of available partitioners.
 */
enum class Partitioner_t : uint64_t {
  AlwaysZero,
  FowlerNollVo,
  Murmur3,
  Rrxmrrxmsx0,
  StdHash,
};

static constexpr Partitioner_t default_partitioner{
#if defined(NVE_FEATURE_HT_PART_FNV1A)
  Partitioner_t::FowlerNollVo
#elif defined(NVE_FEATURE_HT_PART_MURMUR3)
  Partitioner_t::Murmur3
#elif defined(NVE_FEATURE_HT_PART_RRXMRRXMSX0)
  Partitioner_t::Rrxmrrxmsx0
#elif defined(NVE_FEATURE_HT_PART_STD_HASH)
  Partitioner_t::StdHash
#elif defined(NVE_FEATURE_HT_PART_ALWAYS_ZERO)
  Partitioner_t::AlwaysZero
#else
#error At least one NVE_FEATURE_HT_PART_xxx must be enabled. See CMakeLists.txt!
#endif
};

constexpr const char* to_string(const Partitioner_t p) {
  switch (p) {
    case Partitioner_t::AlwaysZero:
      return AlwaysZeroPartitioner::name;
    case Partitioner_t::FowlerNollVo:
      return FowlerNollVoPartitioner::name;
    case Partitioner_t::Murmur3:
      return Murmur3Partitioner::name;
    case Partitioner_t::Rrxmrrxmsx0:
      return Rrxmrrxmsx0Partitioner::name;
    case Partitioner_t::StdHash:
      return StdHashPartitioner::name;
  }
  NVE_THROW_("Unknown overflow handler!");
}

inline std::ostream& operator<<(std::ostream& o, const Partitioner_t p) {
  return o << to_string(p);
}

void to_json(nlohmann::json& json, const Partitioner_t e);

void from_json(const nlohmann::json& j, Partitioner_t& e);

/**
 * Determines what action should be taken in case of an overflow condition is detected.
 */
enum class OverflowHandler_t : uint64_t {
  EvictRandom,  // [default] Evict a random key-value pairs.
  EvictLRU,     // Evict the least recently used key-value pairs.
  EvictLFU,     // Evict the least frequently used key-value pairs.
};

constexpr OverflowHandler_t default_overflow_handler{OverflowHandler_t::EvictRandom};

constexpr const char* to_string(const OverflowHandler_t oh) {
  switch (oh) {
    case OverflowHandler_t::EvictRandom:
      return "evict_random";
    case OverflowHandler_t::EvictLRU:
      return "evict_lru";
    case OverflowHandler_t::EvictLFU:
      return "evict_lfu";
  }
  NVE_THROW_("Unknown overflow handler!");
}

inline std::ostream& operator<<(std::ostream& o, const OverflowHandler_t oh) {
  return o << to_string(oh);
}

void to_json(nlohmann::json& json, const OverflowHandler_t e);

void from_json(const nlohmann::json& j, OverflowHandler_t& e);

using no_meta_t = std::monostate;
using lru_meta_t = std::chrono::system_clock::time_point;
using lfu_meta_t = int32_t;

constexpr lfu_meta_t max_lfu_meta{std::numeric_limits<lfu_meta_t>::max()};

// Just a sanity check. If any of this is violated, you need to check the codes that use these types.
static_assert(sizeof(no_meta_t) == 1 && alignof(no_meta_t) == 1);
static_assert(sizeof(lru_meta_t) <= sizeof(int64_t) && alignof(lru_meta_t) <= sizeof(int64_t));
static_assert(sizeof(lfu_meta_t) <= sizeof(int64_t) && alignof(lfu_meta_t) <= sizeof(int64_t));

template <typename Meta>
inline Meta default_meta_value() noexcept {
  if constexpr (std::is_same_v<Meta, no_meta_t>) {
    return {};
  } else if constexpr (std::is_same_v<Meta, lru_meta_t>) {
    // TODO: Assumes nodes are in sync. Add synchronized network timestamp provider?
    return std::chrono::system_clock::now();
  } else if constexpr (std::is_same_v<Meta, lfu_meta_t>) {
    return 1;
  } else {
    static_assert(dependent_false_v<Meta>, "Overflow handler not implemented.");
  }
}

template <typename Meta>
constexpr OverflowHandler_t overflow_handler_v{[]() {
  if constexpr (std::is_same_v<Meta, no_meta_t>) {
    return OverflowHandler_t::EvictRandom;
  } else if constexpr (std::is_same_v<Meta, lru_meta_t>) {
    return OverflowHandler_t::EvictLRU;
  } else if constexpr (std::is_same_v<Meta, lfu_meta_t>) {
    return OverflowHandler_t::EvictLFU;
  } else {
    static_assert(dependent_false_v<Meta>);
  }
}()};

// Largest 64-bit integer that survives `int64_t -> double -> int64_t`.
constexpr int64_t max_overflow_margin{std::numeric_limits<int64_t>::max() - 1023};

struct OverflowPolicyConfig {
  int64_t overflow_margin{max_overflow_margin};  // Margin at which overflow an condition is triggered.
                                                 // max_overflow_margin = Disable overflow checks.
  OverflowHandler_t handler{default_overflow_handler};  // How to resolve such a condition?
  double resolution_margin{0.8};  // Margin at which the overflow condition is considered resolved?

  void check() const;

  constexpr int64_t meta_alignment() const noexcept {
    switch (handler) {
      case OverflowHandler_t::EvictRandom:
        return alignof(no_meta_t);
      case OverflowHandler_t::EvictLRU:
        return alignof(lru_meta_t);
      case OverflowHandler_t::EvictLFU:
        return alignof(lfu_meta_t);
    }
    NVE_ASSERT_(false);
    return 1;
  }
  
  constexpr int64_t meta_size() const noexcept {
    switch (handler) {
      case OverflowHandler_t::EvictRandom:
        return 0;
      case OverflowHandler_t::EvictLRU:
        return sizeof(lru_meta_t);
      case OverflowHandler_t::EvictLFU:
        return sizeof(lfu_meta_t);
    }
    NVE_ASSERT_(false);
    return 0;
  }

  constexpr int64_t abs_resolution_margin() const noexcept {
    return static_cast<int64_t>(static_cast<double>(overflow_margin) * resolution_margin);
  }
};

void from_json(const nlohmann::json& json, OverflowPolicyConfig& conf);

void to_json(nlohmann::json& json, const OverflowPolicyConfig& conf);

constexpr int64_t default_ht_key_size{
#if defined(NVE_FEATURE_HT_KEY_64)
  sizeof(int64_t)
#elif defined(NVE_FEATURE_HT_KEY_32)
  sizeof(int32_t)
#elif defined(NVE_FEATURE_HT_KEY_16)
  sizeof(int16_t)
#elif defined(NVE_FEATURE_HT_KEY_8)
  sizeof(int8_t)
#else
#error At least one NVE_FEATURE_HT_KEY_xxx must be enabled. See CMakeLists.txt!
#endif
};

struct HostTableConfig {
  int64_t key_size{default_ht_key_size};    // Key size to use.
  int64_t max_value_size{8};  // Maximum size of each table value in bytes. Must be in [1,
                              // 2^31-5], should be a multiple of value_dtype.
  DataType_t value_dtype{DataType_t::Unknown};  // Storage data type of the table values. Only used
                                                // by `update_accumulate()`.
  int64_t invalid_key{-1};  // Key used to signal invalid entries (cast to int64_t)
                            // All operations on such keys (lookup, insert, ...) have undefined results
                            // Other keys in the same batch are unaffected

  void check() const;

  inline int64_t value_dtype_size() const noexcept { return dtype_size(value_dtype); }
};

void from_json(const nlohmann::json& json, HostTableConfig& conf);

void to_json(nlohmann::json& json, const HostTableConfig& conf);

class HostTableLike : public Table {
 public:
  using base_type = Table;

  const table_id_t id;

  NVE_PREVENT_COPY_AND_MOVE_(HostTableLike);

  HostTableLike() = delete;

  HostTableLike(const table_id_t id);

  ~HostTableLike() override = default;

  virtual const HostTableConfig& config() const = 0;

  int32_t get_device_id() const override final { return -1; }
  int64_t get_max_row_size() const override final { return config().max_value_size; }
  int64_t get_key_size() const override final { return config().key_size; }
  virtual int64_t get_invalid_key() const override final { return config().invalid_key; }
  virtual DataType_t get_value_type() const override final { return config().value_dtype; }

  bool lookup_counter_hits() const override final { return true; }

  void reset_lookup_counter(context_ptr_t& ctx) override {
    auto ctx_counter = lookup_counter_storage(ctx);
    NVE_CHECK_(ctx_counter != nullptr, "Invalid key counter");
    *ctx_counter = 0;
  }

  void get_lookup_counter(context_ptr_t& ctx, int64_t* counter) const override {
    auto ctx_counter = lookup_counter_storage(ctx);
    NVE_CHECK_(ctx_counter != nullptr, "Invalid key counter");
    *counter = *ctx_counter;
  }

 protected:
  virtual int64_t* lookup_counter_storage(context_ptr_t& ctx) const {
    static constexpr char buffer_name[]{"host_table_key_counter"};

    NVE_CHECK_(ctx != nullptr, "Invalid context");
    void* buffer = ctx->get_buffer(buffer_name, sizeof(int64_t), true /*host_alloc*/);
    NVE_CHECK_(buffer != nullptr, "Failed to get counter buffer");
    return reinterpret_cast<int64_t*>(buffer);
  }
};

using host_table_ptr_t = std::shared_ptr<HostTableLike>;

struct HostTableFactoryConfig {
  void check() const;
};

void from_json(const nlohmann::json& json, HostTableFactoryConfig& conf);

void to_json(nlohmann::json& json, const HostTableFactoryConfig& conf);

template <typename ConfigType>
class HostTable : public HostTableLike {
 public:
  using base_type = HostTableLike;
  using config_type = ConfigType;

  static_assert(std::is_base_of_v<HostTableConfig, config_type>);

  NVE_PREVENT_COPY_AND_MOVE_(HostTable);

  HostTable() = delete;

  inline HostTable(const table_id_t id, const config_type& config)
      : base_type(id), config_{config} {
    config_.check();
  }

  ~HostTable() override = default;

  const config_type& config() const override final { return config_; }

 protected:
  const config_type config_;
};

template <typename ConfigType, typename TableConfigType>
class HostTableFactory : public TableFactory {
 public:
  using base_type = TableFactory;
  using config_type = ConfigType;
  using table_config_type = TableConfigType;

  static_assert(std::is_base_of_v<HostTableFactoryConfig, config_type>);
  static_assert(std::is_base_of_v<HostTableConfig, table_config_type>);

  const config_type config;

  NVE_PREVENT_COPY_AND_MOVE_(HostTableFactory);

  HostTableFactory() = delete;

  inline HostTableFactory(const config_type& config) : config{config} { config.check(); }

  ~HostTableFactory() override = default;

  table_ptr_t produce(table_id_t id, const nlohmann::json& json) override final {
    return produce(id, static_cast<table_config_type>(json));
  }

  virtual host_table_ptr_t produce(table_id_t id, const table_config_type& config) = 0;
};

}  // namespace nve
