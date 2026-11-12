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

#include <memory>
#include <mutex>
#include <nlohmann/json_fwd.hpp>
#include <plugin/plugin.hpp>
#include <shared_mutex>
#include <table.hpp>
#include <vector>

// The SPH engine (sph::ShardedPerfectHashMap) is only referenced through a pimpl, so
// its CUDA-heavy header stays out of this file (which plugin.cpp/tests include).
namespace sph {
class ShardedPerfectHashMap;
}  // namespace sph

namespace nve {

class ContextRegistry;

namespace plugin {

/**
 * Configuration for a SphTable. SPH is a GPU-resident, insert/lookup hash table
 * with fixed-size opaque value rows (see engine/include/sph/sph_hash_map.h).
 */
struct SphTableConfig {
  int device_id{0};                         // GPU device the table lives on.
  size_t cache_size{0};                     // Total storage in bytes. When non-zero the key
                                            // capacity is derived as cache_size / row_size_in_bytes.
  int64_t initial_capacity{0};              // Explicit key capacity. Takes precedence over cache_size.
  int64_t row_size_in_bytes{512};           // Size in bytes of each value row (SPH `value_size`).
  DataType_t value_dtype{DataType_t::Unknown};  // Storage dtype (only used for pooling/dequant metadata).
  int64_t invalid_key{-1};                  // Key used to signal invalid entries (cast to int64_t).
  double defrag_threshold{1.5};             // Defrag once payload_slots/live_keys exceeds this ratio
                                            // (>= 1.0 by construction). <= 0 disables auto-defrag.
  std::vector<uint8_t> default_value{};     // Row returned by find() for keys not in the table. Either
                                            // empty (all-zero) or exactly row_size_in_bytes long.

  void check() const;
};

void from_json(const nlohmann::json& json, SphTableConfig& conf);
void to_json(nlohmann::json& json, const SphTableConfig& conf);

/**
 * GPU-resident Table backed by sph::ShardedPerfectHashMap. Mirrors the GpuTable
 * wrapper pattern (a Table subclass whose engine member is the hash map).
 *
 * Supported: insert, update (write-if-exists), find, clear.
 * Not supported (throw NVE_THROW_NOT_IMPLEMENTED_): erase, update_accumulate, size, get_keys.
 * find() supports the in/out hit_mask (keys whose bit is already set are skipped,
 * and a bit is reported for every key resolved here) but ignores value_sizes.
 * The lookup counter reports misses, counted from that mask; it is only produced
 * when the caller supplies a hit_mask.
 *
 */
class SphTable final : public Table {
 public:
  using config_type = SphTableConfig;

  NVE_PREVENT_COPY_AND_MOVE_(SphTable);

  SphTable() = delete;

  SphTable(table_id_t id, const SphTableConfig& config);

  ~SphTable() noexcept override;

  context_ptr_t create_execution_context(cudaStream_t lookup_stream, cudaStream_t modify_stream,
                                         thread_pool_ptr_t thread_pool,
                                         allocator_ptr_t allocator) override;

  void clear(context_ptr_t& ctx) override;

  int64_t size(context_ptr_t& ctx) const override;
  int64_t get_keys(context_ptr_t& ctx, int64_t capacity, void* keys) const override;

  void reset_lookup_counter(context_ptr_t& ctx) override;
  void get_lookup_counter(context_ptr_t& ctx, int64_t* counter) const override;
  bool lookup_counter_hits() const override;

  int32_t get_device_id() const override;
  int64_t get_max_row_size() const override;
  int64_t get_key_size() const override;
  int64_t get_invalid_key() const override;
  DataType_t get_value_type() const override;

  void erase(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys) override;

  void find(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, buffer_ptr<bitmask64_t> hit_mask,
            int64_t value_stride, buffer_ptr<void> values, buffer_ptr<int64_t> value_sizes) const override;

  void insert(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, int64_t value_stride,
              int64_t value_size, buffer_ptr<const void> values) override;

  void update(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, int64_t value_stride,
              int64_t value_size, buffer_ptr<const void> values) override;

  void update_accumulate(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, int64_t update_stride,
                         int64_t update_size, buffer_ptr<const void> updates, DataType_t update_dtype) override;

  const SphTableConfig& config() const { return config_; }

  /**
   * Ratio of allocated payload slots to live keys (>= 1.0). This is what the
   * automatic defrag trigger compares against config().defrag_threshold.
   */
  double get_fragmentation() const;

  /**
   * Compact the payload. Blocks until the table is defragmented and the new
   * layout is visible to every stream. Called automatically at the end of
   * insert() when fragmentation exceeds the configured threshold; exposed for
   * callers (and tests) that want to force it.
   */
  void defrag(context_ptr_t& ctx);

 private:
  // Runs defrag() iff auto-defrag is enabled and the table is over threshold.
  void maybe_defrag(context_ptr_t& ctx);
  // Body of defrag(); caller must already hold modify_mutex_ and table_mutex_ exclusively.
  void defrag_locked(context_ptr_t& ctx);
  // Order `stream` behind every live context's lookup stream. Caller must
  // already hold table_mutex_ exclusively.
  void drain_lookups(cudaStream_t stream);

  const table_id_t id_;
  const SphTableConfig config_;
  int64_t capacity_{0};
  std::unique_ptr<sph::ShardedPerfectHashMap> engine_;
  // The engine allows find to overlap an insert but never two modifies (they
  // share scratch pools and readback counters), so modifies are serialized by
  // modify_mutex_ while table_mutex_ keeps lookups off defrag/clear. Lock order
  // is always modify_mutex_ first, then table_mutex_.
  mutable std::mutex modify_mutex_;
  // Shared by find and modify ops, taken exclusively by defrag and clear.
  mutable std::shared_mutex table_mutex_;
  // Live contexts, so defrag can order itself behind their lookup streams.
  std::shared_ptr<ContextRegistry> contexts_;
};

/**
 * Factory configuration for the SPH plugin. No knobs today; kept for symmetry
 * with the other plugins and forward compatibility.
 */
struct SphTableFactoryConfig {
  void check() const {}
};

void from_json(const nlohmann::json& json, SphTableFactoryConfig& conf);
void to_json(nlohmann::json& json, const SphTableFactoryConfig& conf);

class SphTableFactory final : public TableFactory {
 public:
  using config_type = SphTableFactoryConfig;

  NVE_PREVENT_COPY_AND_MOVE_(SphTableFactory);

  SphTableFactory() = delete;

  explicit SphTableFactory(const SphTableFactoryConfig& config);

  ~SphTableFactory() override = default;

  table_ptr_t produce(table_id_t id, const nlohmann::json& json) override;

 private:
  const SphTableFactoryConfig config_;
};

}  // namespace plugin
}  // namespace nve
