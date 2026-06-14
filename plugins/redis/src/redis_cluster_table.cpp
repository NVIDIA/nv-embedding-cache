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

#include <execution_context.hpp>
#include <host_table_detail.hpp>
#include <redis_cluster_table.hpp>
#include <buffer_wrapper.hpp>
#include <redis_utils.hpp>
#include <cstring>
#include <string>
#include <string_view>

namespace nve {
namespace plugin {

void RedisClusterTableConfig::check() const {
  base_type::check();

  NVE_CHECK_(max_batch_size > 0 && max_batch_size % mask_size == 0);

  NVE_CHECK_(!num_partitions || (num_partitions >= 0 && has_single_bit(static_cast<uint64_t>(num_partitions))));
  NVE_CHECK_(!workgroups.empty());

  overflow_policy.check();
}

void from_json(const nlohmann::json& json, RedisClusterTableConfig& conf) {
  using base_type = RedisClusterTableConfig::base_type;
  from_json(json, static_cast<base_type&>(conf));

  NVE_READ_JSON_FIELD_(max_batch_size);

  NVE_READ_JSON_FIELD_(num_partitions);
  NVE_READ_JSON_FIELD_(partitioner);
  NVE_READ_JSON_FIELD_(workgroups);
  NVE_READ_JSON_FIELD_(hash_key);
  NVE_READ_JSON_FIELD_(string_namespace_id);

  NVE_READ_JSON_FIELD_(overflow_policy);
}

void to_json(nlohmann::json& json, const RedisClusterTableConfig& conf) {
  using base_type = RedisClusterTableConfig::base_type;
  to_json(json, static_cast<const base_type&>(conf));

  NVE_WRITE_JSON_FIELD_(max_batch_size);

  NVE_WRITE_JSON_FIELD_(num_partitions);
  NVE_WRITE_JSON_FIELD_(partitioner);
  NVE_WRITE_JSON_FIELD_(workgroups);
  NVE_WRITE_JSON_FIELD_(hash_key);
  NVE_WRITE_JSON_FIELD_(string_namespace_id);

  NVE_WRITE_JSON_FIELD_(overflow_policy);
}

/**
 * Avoids frequent memory allocations.
 */
class HKey final {
 public:
  HKey() = delete;

  inline HKey(const table_id_t table_id, const int64_t part_idx, const char suffix) {
    const int req_size{std::snprintf(c_str_.data(), c_str_.size(), "nv_nve_table{%ld/%ld}%c",
                                     table_id, part_idx, suffix)};
    NVE_CHECK_(req_size > 0 && req_size <= static_cast<int64_t>(c_str_.size()));
    size_ = req_size - 1;
  }

  inline const char* c_str() const noexcept { return c_str_.data(); }

  inline operator sw::redis::StringView() const noexcept { return {c_str_.data(), static_cast<uint64_t>(size_)}; }

 private:
  // Format:
  // nv_nve_table{ = 13 chars
  // table_id = up to 20 chars => -9'223'372'036'854'775'808
  // / = 1 char
  // part_idx = up to 20 chars => -9'223'372'036'854'775'808
  // suffix = 1 char
  // \0 = 1 char
  // 13 + 20 + 1 + 20 + 1 + 1 = 56;
  std::array<char, 56> c_str_;
  int64_t size_;
};

// True when the table stores each entry as a plain Redis string (`MSET`/`MGET`/`DEL`): a standalone
// server with no shared hash key set. The other modes (single shared hash, or multi-hash sharding)
// store entries as fields of Redis hashes.
inline static bool is_string_mode(const redis_conn_ptr_t& conn,
                                  const RedisClusterTableConfig& config) {
  return conn->is_single_node() && config.hash_key.empty();
}

// In single-node string mode a key may optionally be namespaced by prepending the raw bytes of
// `config.string_namespace_id` (an `int64_t`) ahead of the key bytes.
static constexpr int64_t kKeyPrefixSize{sizeof(int64_t)};

// Size in bytes of a single Redis key in string mode: the (optional) prefix followed by the key.
inline static int64_t string_key_size(int64_t string_namespace_id, int64_t key_size) {
  return (string_namespace_id >= 0 ? kKeyPrefixSize : 0) + key_size;
}

// Returns the Redis key view for `key` in string mode. When `string_namespace_id < 0` the raw key bytes are
// the Redis key, so the view points straight at `key` (zero-copy, like the cluster path). When a
// prefix is set, the prefixed key is materialized into slot `slot` of the scratch buffer `buf`
// (capacity must cover `(slot + 1) * entry_size` bytes) and a view onto that is returned.
// `entry_size` must equal `string_key_size(string_namespace_id, sizeof(KeyType))`.
template <typename KeyType>
inline static sw::redis::StringView write_string_key(char* const buf, const int64_t slot,
                                                     const int64_t entry_size,
                                                     const int64_t string_namespace_id, const KeyType& key) {
  if (string_namespace_id < 0) {
    return {reinterpret_cast<const char*>(&key), sizeof(KeyType)};
  }
  char* const dst{buf + slot * entry_size};
  std::memcpy(dst, &string_namespace_id, sizeof(int64_t));
  std::memcpy(dst + sizeof(int64_t), &key, sizeof(KeyType));
  return {dst, static_cast<uint64_t>(entry_size)};
}

// Builds the `SCAN MATCH` glob pattern selecting all keys carrying `string_namespace_id`: the prefix bytes
// (with glob meta-characters escaped) followed by '*'.
inline static std::string make_prefix_scan_pattern(int64_t string_namespace_id) {
  const char* const p{reinterpret_cast<const char*>(&string_namespace_id)};
  std::string pattern;
  pattern.reserve(static_cast<uint64_t>(kKeyPrefixSize) * 2 + 1);
  for (int64_t i{}; i != kKeyPrefixSize; ++i) {
    const char c{p[i]};
    if (c == '*' || c == '?' || c == '[' || c == ']' || c == '\\') pattern.push_back('\\');
    pattern.push_back(c);
  }
  pattern.push_back('*');
  return pattern;
}

// Word-aligned [lo, hi) sub-range for task `t` of `parts` over [0, n). Aligning boundaries to `Align`
// keys ensures no two tasks touch the same hit-mask word, so hit-mask updates stay non-atomic.
template <int64_t Align>
inline static void task_range(const int64_t n, const int64_t parts, const int64_t t, int64_t& lo,
                              int64_t& hi) {
  const int64_t chunk{next_aligned<Align>((n + parts - 1) / parts)};
  lo = std::min(t * chunk, n);
  hi = std::min(lo + chunk, n);
}

// Runs `task(lo, hi)` over the [0, n) index space: inline when there is a single partition, otherwise
// split into `num_parts` word-aligned sub-ranges executed on the thread pool. Used by standalone
// string mode to parallelize the client-side work of building/parsing Redis commands.
template <int64_t Align, typename TaskFn>
inline static void run_partitioned(context_ptr_t& ctx, const int64_t n, const int64_t num_parts,
                                   const std::vector<int64_t>& workgroups, TaskFn&& task) {
  const int64_t parts{num_parts > 1 ? num_parts : 1};
  if (parts <= 1) {
    task(int64_t{0}, n);
    return;
  }
  ctx->get_thread_pool()->execute_n(
      0, parts,
      [n, parts, &task](const int64_t t) {
        int64_t lo, hi;
        task_range<Align>(n, parts, t, lo, hi);
        task(lo, hi);
      },
      workgroups, 1);
}

template <typename MaskType, typename KeyType, typename MetaType, typename PartitionerType>
RedisClusterTable<MaskType, KeyType, MetaType, PartitionerType>::RedisClusterTable(
    table_id_t table_id, const config_type& config, redis_conn_ptr_t& conn)
    : base_type(table_id, config), conn_{conn} {}

template <typename MaskType, typename KeyType, typename MetaType, typename PartitionerType>
void RedisClusterTable<MaskType, KeyType, MetaType, PartitionerType>::clear(context_ptr_t& ctx) {
  const auto& __restrict config{config_};
  redis_conn_ptr_t& cluster{conn_};

  const int64_t num_parts{config.num_partitions};

  // String mode (any `num_partitions`): the keys are plain Redis strings regardless of the parallel
  // work split, so clearing is a single keyspace operation.
  if (is_string_mode(cluster, config)) {
    const int64_t string_namespace_id{config.string_namespace_id};
    if (string_namespace_id < 0) {
      NVE_LOG_WARNING_(
          "Redis table ", this->id,
          ": clearing in raw single-node mode issues FLUSHDB, which wipes the ENTIRE Redis server "
          "(every key, across all tables). Set `string_namespace_id` to namespace this table and clear only "
          "its keys.");
      cluster->flushdb();
      return;
    }

    // Prefixed string mode: SCAN for this table's keys and DEL them in batches.
    const std::string pattern{make_prefix_scan_pattern(string_namespace_id)};
    const int64_t max_batch_size{config.max_batch_size};
    sw::redis::Cursor cursor{0};
    std::vector<std::string> batch;
    do {
      batch.clear();
      cursor = cluster->scan(cursor, pattern, max_batch_size, std::back_inserter(batch));
      if (!batch.empty()) {
        cluster->del(batch.begin(), batch.end());
      }
    } while (cursor != 0);
    return;
  }

  if (num_parts == 0) {
    // Single-hash mode: drop the shared hash.
    cluster->del(sw::redis::StringView{config.hash_key});
    return;
  }

  const table_id_t table_id{this->id};
  const auto f{[table_id, &cluster](const int64_t task_idx) {
    const HKey v_key(table_id, task_idx, 'v');
    const HKey m_key(table_id, task_idx, 'm');
    const std::array<sw::redis::StringView, 2> hkeys{v_key, m_key};
    cluster->del(hkeys.begin(), hkeys.end());
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename MaskType, typename KeyType, typename MetaType, typename PartitionerType>
void RedisClusterTable<MaskType, KeyType, MetaType, PartitionerType>::erase(context_ptr_t& ctx,
                                                                            int64_t n,
                                                                            buffer_ptr<const void> keys_bw) {
  if (n <= 0) return;
  const auto& __restrict config{config_};
  redis_conn_ptr_t& cluster{conn_};

  const void* const keys_vptr{
      keys_bw ? keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/,
                                       ctx->get_modify_stream())
              : nullptr};
  const key_type* const __restrict keys{reinterpret_cast<const key_type*>(keys_vptr)};
  const int64_t max_batch_size{std::min(n, config.max_batch_size)};

  const int64_t num_parts{config.num_partitions};
  // Standalone string mode: DEL (optionally prefixed) per-key strings, split across `num_parts`
  // thread-pool tasks for client-side parallelism.
  if (is_string_mode(cluster, config)) {
    const int64_t string_namespace_id{config.string_namespace_id};
    const int64_t entry_size{string_key_size(string_namespace_id, sizeof(key_type))};
    const auto task{[&cluster, keys, string_namespace_id, entry_size, max_batch_size](const int64_t lo,
                                                                             const int64_t hi) {
      std::vector<sw::redis::StringView> k_views;
      k_views.reserve(static_cast<uint64_t>(max_batch_size));
      // Only needed to materialize prefixed keys; raw keys are referenced in place (zero-copy).
      std::vector<char> key_buf(
          static_cast<uint64_t>(string_namespace_id >= 0 ? max_batch_size * entry_size : 0));

      const auto process_batch{
          [&cluster, &k_views]() { cluster->del(k_views.begin(), k_views.end()); }};

      for (int64_t i{lo}; i != hi; ++i) {
        k_views.emplace_back(write_string_key(key_buf.data(),
                                              static_cast<int64_t>(k_views.size()), entry_size,
                                              string_namespace_id, keys[i]));
        if NVE_LIKELY_(static_cast<int64_t>(k_views.size()) < max_batch_size) continue;

        process_batch();
        k_views.clear();
      }
      if (!k_views.empty()) {
        process_batch();
      }
    }};
    run_partitioned<mask_type::num_bits>(ctx, n, num_parts, config.workgroups, task);
    return;
  }

  if (num_parts == 0) {
    // Single-hash mode: HDEL fields from the shared hash.
    const sw::redis::StringView v_key{config.hash_key};
    std::vector<sw::redis::StringView> k_views;
    k_views.reserve(static_cast<uint64_t>(max_batch_size));
    const auto process_batch{[&cluster, &k_views, &v_key]() {
      cluster->hdel(v_key, k_views.begin(), k_views.end());
    }};

    for (int64_t i{}; i != n; ++i) {
      k_views.emplace_back(reinterpret_cast<const char*>(&keys[i]), sizeof(key_type));
      if NVE_LIKELY_(static_cast<int64_t>(k_views.size()) < max_batch_size) continue;

      process_batch();
      k_views.clear();
    }
    if (!k_views.empty()) {
      process_batch();
    }

    return;
  }

  const int64_t num_parts_mask{num_parts - 1};
  const table_id_t table_id{this->id};
  const auto f{
      [n, keys, table_id, max_batch_size, &cluster, num_parts_mask](const int64_t task_idx) {
        const HKey v_key(table_id, task_idx, 'v');
        const HKey m_key(table_id, task_idx, 'm');

        // TODO: Prone to memory fragmentation. Use scratch buffer instead?
        std::vector<sw::redis::StringView> k_views;
        k_views.reserve(static_cast<uint64_t>(max_batch_size));

        const auto process_batch{[&cluster, &v_key, &m_key, &k_views]() {
          sw::redis::Pipeline pipe{cluster->pipeline(v_key)};
          pipe.hdel(v_key, k_views.begin(), k_views.end());
          pipe.hdel(m_key, k_views.begin(), k_views.end());
          pipe.exec();
        }};

        for (int64_t i{}; i != n; ++i) {
          const key_type& key{keys[i]};
          if (partitioner(key, num_parts_mask) != task_idx) continue;

          k_views.emplace_back(reinterpret_cast<const char*>(&key), sizeof(key_type));
          if NVE_LIKELY_(static_cast<int64_t>(k_views.size()) < max_batch_size) continue;

          process_batch();
          k_views.clear();
        }
        if (!k_views.empty()) {
          process_batch();
        }
      }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename MaskType, typename KeyType, typename MetaType, typename PartitionerType>
void RedisClusterTable<MaskType, KeyType, MetaType, PartitionerType>::find(
    context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys_bw,
    buffer_ptr<max_bitmask_repr_t> hit_mask_bw, const int64_t value_stride,
    buffer_ptr<void> values_bw, buffer_ptr<int64_t> value_sizes_bw) const {
  if (n <= 0) return;

  auto lookup_stream = ctx->get_lookup_stream();
  const void* const keys_vptr{
      keys_bw ? keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, lookup_stream)
              : nullptr};
  max_bitmask_repr_t* const hit_mask{
      hit_mask_bw ? hit_mask_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/,
                                               lookup_stream)
                  : nullptr};
  void* const values_vptr{
      values_bw ? values_bw->access_buffer(cudaMemoryTypeUnregistered, false /*copy_content*/,
                                           lookup_stream)
                : nullptr};
  int64_t* const value_sizes{
      value_sizes_bw ? value_sizes_bw->access_buffer(cudaMemoryTypeUnregistered,
                                                     false /*copy_content*/, lookup_stream)
                     : nullptr};
  const key_type* const __restrict keys{reinterpret_cast<const key_type*>(keys_vptr)};
  char* const __restrict hm{reinterpret_cast<char*>(hit_mask)};
  char* const __restrict values{reinterpret_cast<char*>(values_vptr)};

  if (values) {
    if (value_sizes) {
      n = find_<true, true>(ctx, n, keys, hm, value_stride, values, value_sizes);
    } else {
      n = find_<true, false>(ctx, n, keys, hm, value_stride, values, value_sizes);
    }
  } else {
    if (value_sizes) {
      n = find_<false, true>(ctx, n, keys, hm, value_stride, values, value_sizes);
    } else {
      n = find_<false, false>(ctx, n, keys, hm, value_stride, values, value_sizes);
    }
  }
  auto counter = this->lookup_counter_storage(ctx);
  NVE_CHECK_(counter != nullptr, "Invalid key counter");
  *counter += n;
}

template <typename MaskType, typename KeyType, typename MetaType, typename PartitionerType>
void RedisClusterTable<MaskType, KeyType, MetaType, PartitionerType>::insert(
    context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw,
    const int64_t value_stride, const int64_t value_size, buffer_ptr<const void> values_bw) {
  if (n <= 0) return;
  const auto& __restrict config{config_};
  redis_conn_ptr_t& cluster{conn_};

  auto modify_stream = ctx->get_modify_stream();
  const void* const keys_vptr{
      keys_bw ? keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream)
              : nullptr};
  const void* const values_vptr{
      values_bw ? values_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream)
                : nullptr};
  const key_type* __restrict keys{reinterpret_cast<const key_type*>(keys_vptr)};
  const char* const __restrict values{reinterpret_cast<const char*>(values_vptr)};

  const table_id_t table_id{this->id};

  const int64_t max_batch_size{std::min(n, config.max_batch_size)};
  NVE_CHECK_(value_size >= 0);
  const int64_t overflow_margin{config.overflow_policy.overflow_margin};
  const int64_t resolution_margin{static_cast<int64_t>(static_cast<double>(overflow_margin) *
                                                     config.overflow_policy.resolution_margin)};

  const auto resolve_overflow{[table_id, max_batch_size, resolution_margin, &cluster](
                                  const int64_t part_idx, const sw::redis::StringView& v_key,
                                  const sw::redis::StringView& m_key, int64_t part_size) {
    NVE_LOG_VERBOSE_("RedisCluster table ", table_id, " part ", part_idx,
                     " is overflowing (size = ", part_size, "). Attempting to resolve...");

    std::vector<sw::redis::StringView> k_views;
    k_views.reserve(static_cast<uint64_t>(max_batch_size));

    const auto delete_batch{[table_id, &cluster, part_idx, &v_key, &m_key, &k_views]() {
      NVE_LOG_VERBOSE_("RedisCluster table ", table_id, " part ", part_idx,
                       ": Attempting to evict ", k_views.size(),
                       " (mode = ", overflow_handler<meta_type>(), ").");

      sw::redis::Pipeline pipe{cluster->pipeline(v_key)};
      if constexpr (std::is_same_v<meta_type, no_meta_type>) {
        (void)m_key;
      } else if constexpr (std::is_same_v<meta_type, lru_meta_type>) {
        pipe.hdel(m_key, k_views.begin(), k_views.end());
      } else if constexpr (std::is_same_v<meta_type, lfu_meta_type>) {
        pipe.hdel(m_key, k_views.begin(), k_views.end());
      } else {
        static_assert(dependent_false_v<meta_type>, "Overflow handler not implemented.");
      }
      pipe.hdel(v_key, k_views.begin(), k_views.end());
      pipe.hlen(v_key);

      sw::redis::QueuedReplies replies{pipe.exec()};
      return replies.get<long long>(replies.size() - 1);
    }};

    do {
      if constexpr (std::is_same_v<meta_type, no_meta_type>) {
        using parser_t = ReplyParser<sw::redis::StringView>;

        // Fetch all keys in partition and shuffle them.
        std::vector<key_type> keys;
        keys.reserve(static_cast<uint64_t>(part_size));
        cluster->hkeys(v_key,
                       parser_t([&keys](int64_t, const sw::redis::StringView& k_view) {
                         keys.emplace_back(view_as_value<key_type>(k_view));
                       }));
        part_size = static_cast<int64_t>(keys.size());
        if (part_size <= resolution_margin) {
          NVE_LOG_VERBOSE_("RedisCluster table ", table_id, ", part ", part_idx,
                           " (size = ", part_size,
                           "): Overflow was already resolved by another process.");
          break;
        }
        std::shuffle(keys.begin(), keys.end(), std::default_random_engine{random_device()});

        for (auto it{keys.begin() + resolution_margin}; it < keys.end();
             ++it) {
          k_views.emplace_back(reinterpret_cast<const char*>(&*it), sizeof(key_type));
          if NVE_LIKELY_(static_cast<int64_t>(k_views.size()) < max_batch_size) continue;

          part_size = delete_batch();
          k_views.clear();
          if (part_size <= resolution_margin) break;
        }
        if (!k_views.empty()) {
          part_size = delete_batch();
        }
      } else if constexpr (std::is_same_v<meta_type, lru_meta_type>) {
        using view_pair_t = std::pair<sw::redis::StringView, sw::redis::StringView>;
        using parser_t = ReplyParser<view_pair_t>;

        // Fetch all keys, and sort by timestamp.
        std::vector<std::pair<key_type, lru_meta_type>> keys_metas;
        keys_metas.reserve(static_cast<uint64_t>(part_size));
        cluster->hgetall(
            m_key, parser_t([&keys_metas](int64_t, const view_pair_t& view) {
              keys_metas.emplace_back(*reinterpret_cast<const key_type*>(view.first.data()),
                                      *reinterpret_cast<const lru_meta_type*>(view.second.data()));
            }));
        part_size = static_cast<int64_t>(keys_metas.size());
        if (part_size <= resolution_margin) {
          NVE_LOG_VERBOSE_("RedisCluster table ", table_id, ", part ", part_idx,
                           " (size = ", part_size,
                           "): Overflow was already resolved by another process.");
          break;
        }
        std::sort(keys_metas.begin(), keys_metas.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        auto it{keys_metas.begin() + resolution_margin};
        for (; it < keys_metas.end(); ++it) {
          k_views.emplace_back(reinterpret_cast<const char*>(&it->first), sizeof(key_type));
          if NVE_LIKELY_(static_cast<int64_t>(k_views.size()) < max_batch_size) continue;

          part_size = delete_batch();
          k_views.clear();
          if (part_size <= resolution_margin) break;
        }
        if (!k_views.empty()) {
          part_size = delete_batch();
        }
      } else if constexpr (std::is_same_v<meta_type, lfu_meta_type>) {
        using view_pair_t = std::pair<sw::redis::StringView, sw::redis::StringView>;
        using parser_t = ReplyParser<view_pair_t>;

        // Fetch all keys, and sort by access count.
        std::vector<std::pair<key_type, lfu_meta_type>> keys_metas;
        keys_metas.reserve(static_cast<uint64_t>(part_size));
        cluster->hgetall(
            m_key, parser_t([&keys_metas](int64_t, const view_pair_t& view) {
              keys_metas.emplace_back(*reinterpret_cast<const key_type*>(view.first.data()),
                                      *reinterpret_cast<const lfu_meta_type*>(view.second.data()));
            }));
        part_size = static_cast<int64_t>(keys_metas.size());
        if (part_size <= resolution_margin) {
          NVE_LOG_VERBOSE_("RedisCluster table ", table_id, ", part ", part_idx,
                           " (size = ", part_size,
                           "): Overflow was already resolved by another process.");
          break;
        }
        std::sort(keys_metas.begin(), keys_metas.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        int64_t i{};
        for (const int64_t mid{std::min(part_size - resolution_margin, {})};
             i != mid; ++i) {
          k_views.emplace_back(reinterpret_cast<const char*>(&keys_metas[static_cast<uint64_t>(i)].first),
                               sizeof(key_type));
          if NVE_LIKELY_(static_cast<int64_t>(k_views.size()) < max_batch_size) continue;

          part_size = delete_batch();
          k_views.clear();
          if (part_size <= resolution_margin) break;
        }
        if (!k_views.empty()) {
          part_size = delete_batch();
        }

        for (; i < part_size; i += max_batch_size) {
          sw::redis::Pipeline pipe{cluster->pipeline(m_key)};

          const int64_t batch_size{std::min(part_size - i, max_batch_size)};
          for (int64_t j{}; j != batch_size; ++j) {
            const auto [key, meta]{keys_metas[static_cast<uint64_t>(i + j)]};
            pipe.hincrby(m_key, view_as_string(key), std::max(meta, {}) / -2);
          }

          NVE_LOG_VERBOSE_("RedisCluster table ", table_id, " part ", part_idx,
                           ": Updating meta data for ", batch_size, " entries");
          pipe.exec();
        }
      } else {
        static_assert(dependent_false_v<meta_type>, "Overflow handler not implemented.");
      }
    } while (false);

    NVE_LOG_VERBOSE_("RedisCluster table ", table_id, " part ", part_idx,
                     ": Overflow resolution complete! (size = ", part_size, ')');
    return part_size;
  }};

  const int64_t num_parts{config.num_partitions};

  // Standalone string mode: MSET (optionally prefixed) per-key strings, split across `num_parts`
  // thread-pool tasks. No metadata/overflow handling.
  if (is_string_mode(cluster, config)) {
    const int64_t string_namespace_id{config.string_namespace_id};
    const int64_t entry_size{string_key_size(string_namespace_id, sizeof(key_type))};
    const auto task{[&cluster, keys, values, value_stride, value_size, string_namespace_id, entry_size,
                     max_batch_size](const int64_t lo, const int64_t hi) {
      std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> kv_views;
      kv_views.reserve(static_cast<uint64_t>(max_batch_size));
      // Only needed to materialize prefixed keys; raw keys are referenced in place (zero-copy).
      std::vector<char> key_buf(
          static_cast<uint64_t>(string_namespace_id >= 0 ? max_batch_size * entry_size : 0));

      const auto process_batch{
          [&cluster, &kv_views]() { cluster->mset(kv_views.begin(), kv_views.end()); }};

      for (int64_t i{lo}; i != hi; ++i) {
        kv_views.emplace_back(
            std::piecewise_construct,
            std::forward_as_tuple(write_string_key(key_buf.data(),
                                                   static_cast<int64_t>(kv_views.size()), entry_size,
                                                   string_namespace_id, keys[i])),
            std::forward_as_tuple(values + i * value_stride, value_size));
        if NVE_LIKELY_(static_cast<int64_t>(kv_views.size()) < max_batch_size) continue;

        process_batch();
        kv_views.clear();
      }
      if (!kv_views.empty()) {
        process_batch();
      }
    }};
    run_partitioned<mask_type::num_bits>(ctx, n, num_parts, config.workgroups, task);
    return;
  }

  if (num_parts == 0) {
    // Single-hash mode: HSET fields into the shared hash, with overflow handling.
    const sw::redis::StringView v_key{config.hash_key};
    std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> kv_views;
    kv_views.reserve(static_cast<uint64_t>(max_batch_size));
    const auto process_batch{[overflow_margin, &cluster, &resolve_overflow, &kv_views, &v_key]() {
      int64_t part_size;
      {
        sw::redis::Pipeline pipe{cluster->pipeline(v_key)};
        pipe.hset(v_key, kv_views.begin(), kv_views.end());
        pipe.hlen(v_key);

        sw::redis::QueuedReplies replies{pipe.exec()};
        part_size = replies.get<long long>(replies.size() - 1);
      }
      if (part_size > overflow_margin) {
        // Handle overflows situations.
        resolve_overflow(0, v_key, v_key, part_size);
      }
    }};

    for (int64_t i{}; i != n; ++i) {
      kv_views.emplace_back(
          std::piecewise_construct,
          std::forward_as_tuple(reinterpret_cast<const char*>(&keys[i]), sizeof(key_type)),
          std::forward_as_tuple(values + i * value_stride, value_size));
      if NVE_LIKELY_(static_cast<int64_t>(kv_views.size()) < max_batch_size) continue;

      process_batch();
      kv_views.clear();
    }
    if (!kv_views.empty()) {
      process_batch();
    }

    return;
  }

  const int64_t num_parts_mask{num_parts - 1};
  const auto f{[n, keys, value_stride, value_size, values, table_id, max_batch_size,
                overflow_margin, &cluster, num_parts_mask,
                &resolve_overflow](const int64_t task_idx) {
    const HKey v_key(table_id, task_idx, 'v');
    const HKey m_key(table_id, task_idx, 'm');

    // TODO: Prone to memory fragmentation. Use scratch buffer instead?
    std::vector<std::pair<const sw::redis::StringView, sw::redis::StringView>> kv_views;
    kv_views.reserve(static_cast<uint64_t>(max_batch_size));

    const auto process_batch{
        [overflow_margin, &cluster, &resolve_overflow, task_idx, &v_key, &m_key, &kv_views]() {
          int64_t part_size;
          {
            sw::redis::Pipeline pipe{cluster->pipeline(v_key)};
            pipe.hset(v_key, kv_views.begin(), kv_views.end());

            lru_meta_type lru_value;
            if constexpr (std::is_same_v<meta_type, no_meta_type>) {
            } else if constexpr (std::is_same_v<meta_type, lru_meta_type>) {
              lru_value = lru_meta_value();
              for (const auto& kv_view : kv_views) {
                pipe.hsetnx(m_key, kv_view.first, view_as_string(lru_value));
              }
            } else if constexpr (std::is_same_v<meta_type, lfu_meta_type>) {
              for (const auto& kv_view : kv_views) {
                pipe.hincrby(m_key, kv_view.first, 1);
              }
            } else {
              static_assert(dependent_false_v<meta_type>, "Overflow handler not implemented.");
            }
            pipe.hlen(v_key);

            sw::redis::QueuedReplies replies{pipe.exec()};
            part_size = replies.get<long long>(replies.size() - 1);
          }
          if (part_size > overflow_margin) {
            // Handle overflows situations.
            resolve_overflow(task_idx, v_key, m_key, part_size);
          }
        }};

    for (int64_t i{}; i < n; ++i) {
      const key_type& key{keys[i]};
      if (partitioner(key, num_parts_mask) != task_idx) continue;
      kv_views.emplace_back(
          std::piecewise_construct,
          std::forward_as_tuple(reinterpret_cast<const char*>(&key), sizeof(key_type)),
          std::forward_as_tuple(values + i * value_stride, value_size));
      if NVE_LIKELY_(static_cast<int64_t>(kv_views.size()) < max_batch_size) continue;

      process_batch();
      kv_views.clear();
    }
    if (!kv_views.empty()) {
      process_batch();
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename MaskType, typename KeyType, typename MetaType, typename PartitionerType>
int64_t RedisClusterTable<MaskType, KeyType, MetaType, PartitionerType>::size(context_ptr_t& ctx,
                                                                              const bool) const {
  const auto& __restrict config{config_};
  const redis_conn_ptr_t& cluster{conn_};

  const int64_t num_parts{config.num_partitions};

  // String mode (any `num_partitions`): keys are plain Redis strings, so size is a keyspace query.
  if (is_string_mode(cluster, config)) {
    const int64_t string_namespace_id{config.string_namespace_id};
    if (string_namespace_id < 0) {
      // Raw mode owns the whole DB, so its size is the DB size.
      return static_cast<int64_t>(cluster->dbsize());
    }

    // Prefixed string mode: count this table's keys via SCAN.
    const std::string pattern{make_prefix_scan_pattern(string_namespace_id)};
    const int64_t max_batch_size{config.max_batch_size};
    int64_t total{0};
    sw::redis::Cursor cursor{0};
    std::vector<std::string> batch;
    do {
      batch.clear();
      cursor = cluster->scan(cursor, pattern, max_batch_size, std::back_inserter(batch));
      total += static_cast<int64_t>(batch.size());
    } while (cursor != 0);
    return total;
  }

  if (num_parts == 0) {
    // Single-hash mode: number of fields in the shared hash.
    return static_cast<int64_t>(cluster->hlen(sw::redis::StringView{config.hash_key}));
  }

  const table_id_t table_id{this->id};
  std::atomic_int64_t total_size{0};
  auto f{[table_id, &cluster, &total_size](const int64_t task_idx) {
    const HKey v_key(table_id, task_idx, 'v');
    const int64_t part_size{cluster->hlen(v_key)};
    total_size.fetch_add(part_size, std::memory_order_relaxed);
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
  return total_size.load(std::memory_order_relaxed);
}

template <typename MaskType, typename KeyType, typename MetaType, typename PartitionerType>
void RedisClusterTable<MaskType, KeyType, MetaType, PartitionerType>::update(
    context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw,
    const int64_t value_stride, const int64_t value_size, buffer_ptr<const void> values_bw) {
  if (n <= 0) return;
  const auto& __restrict config{config_};
  redis_conn_ptr_t& cluster{conn_};

  auto modify_stream = ctx->get_modify_stream();
  const void* const keys_vptr{
      keys_bw ? keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream)
              : nullptr};
  const void* const values_vptr{
      values_bw ? values_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream)
                : nullptr};
  const key_type* const __restrict keys{reinterpret_cast<const key_type*>(keys_vptr)};
  const char* const __restrict values{reinterpret_cast<const char*>(values_vptr)};

  // TODO: Prone to memory fragmantation. Also inefficent. What we would need is an op that does
  // the opposite to `HSETNX`.
  std::vector<max_bitmask_repr_t> hit_mask(static_cast<uint64_t>(max_bitmask_t::mask_size(n)), {});
  char* const __restrict hm{reinterpret_cast<char*>(hit_mask.data())};
  find_<false, false>(ctx, n, keys, hm, value_stride, nullptr, nullptr);

  const int64_t max_batch_size{std::min(n, config.max_batch_size)};
  NVE_CHECK_(value_size >= 0 && value_size <= config.max_value_size);

  const int64_t num_parts{config.num_partitions};

  // Standalone string mode: MSET the found keys' new values, split across `num_parts` thread-pool
  // tasks. Each task reads its own word-aligned slice of the hit mask (read-only) and writes plain
  // string keys.
  if (is_string_mode(cluster, config)) {
    const int64_t string_namespace_id{config.string_namespace_id};
    const int64_t entry_size{string_key_size(string_namespace_id, sizeof(key_type))};
    const auto task{[&cluster, keys, hm, values, value_stride, value_size, string_namespace_id, entry_size,
                     max_batch_size](const int64_t lo, const int64_t hi) {
      std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> kv_views;
      kv_views.reserve(static_cast<uint64_t>(max_batch_size));
      std::vector<char> key_buf(
          static_cast<uint64_t>(string_namespace_id >= 0 ? max_batch_size * entry_size : 0));

      const auto process_batch{
          [&cluster, &kv_views]() { cluster->mset(kv_views.begin(), kv_views.end()); }};

      for (int64_t i{lo}; i < hi; i += mask_type::num_bits) {
        for (mask_repr_type it{mask_type::load(hm, i)}; mask_type::has_next(it);
             it = mask_type::skip(it)) {
          const int64_t ij{i + mask_type::next(it)};
          kv_views.emplace_back(
              std::piecewise_construct,
              std::forward_as_tuple(write_string_key(key_buf.data(),
                                                     static_cast<int64_t>(kv_views.size()),
                                                     entry_size, string_namespace_id, keys[ij])),
              std::forward_as_tuple(values + ij * value_stride, value_size));
          if NVE_LIKELY_(static_cast<int64_t>(kv_views.size()) < max_batch_size) continue;

          process_batch();
          kv_views.clear();
        }
      }
      if (!kv_views.empty()) {
        process_batch();
      }
    }};
    run_partitioned<mask_type::num_bits>(ctx, n, num_parts, config.workgroups, task);
    return;
  }

  if (num_parts == 0) {
    // Single-hash mode: HSET the found keys into the shared hash.
    const sw::redis::StringView v_key{config.hash_key};
    std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> kv_views;
    kv_views.reserve(static_cast<uint64_t>(max_batch_size));
    const auto process_batch{[&cluster, &kv_views, &v_key]() {
      cluster->hset(v_key, kv_views.begin(), kv_views.end());
    }};

    for (int64_t i{}; i < n; i += mask_type::num_bits) {
      for (mask_repr_type it{mask_type::load(hm, i)}; mask_type::has_next(it);
           it = mask_type::skip(it)) {
        const int64_t ij{i + mask_type::next(it)};
        kv_views.emplace_back(
            std::piecewise_construct,
            std::forward_as_tuple(reinterpret_cast<const char*>(&keys[ij]), sizeof(key_type)),
            std::forward_as_tuple(values + ij * value_stride, value_size));
        if NVE_LIKELY_(static_cast<int64_t>(kv_views.size()) < max_batch_size) continue;

        process_batch();
        kv_views.clear();
      }
    }
    if (!kv_views.empty()) {
      process_batch();
    }

    return;
  }

  const int64_t num_parts_mask{num_parts - 1};
  const table_id_t table_id{this->id};
  const auto f{[n, keys, hm, value_stride, value_size, values, table_id, max_batch_size, &cluster,
                num_parts_mask](const int64_t task_idx) {
    const HKey v_key(table_id, task_idx, 'v');

    // TODO: Prone to memory fragmentation. Use scratch buffer instead?
    std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> kv_views;
    kv_views.reserve(static_cast<uint64_t>(max_batch_size));

    // Fill up batch and lodge queries as we go.
    for (int64_t i{}; i < n; i += mask_type::num_bits) {
      for (mask_repr_type it{mask_type::load(hm, i)}; mask_type::has_next(it);
           it = mask_type::skip(it)) {
        const int64_t ij{i + mask_type::next(it)};
        const key_type& key{keys[ij]};
        if (partitioner(key, num_parts_mask) != task_idx) continue;
        kv_views.emplace_back(
            std::piecewise_construct,
            std::forward_as_tuple(reinterpret_cast<const char*>(&key), sizeof(key_type)),
            std::forward_as_tuple(values + ij * value_stride, value_size));
        if NVE_LIKELY_(static_cast<int64_t>(kv_views.size()) < max_batch_size) continue;

        cluster->hset(v_key, kv_views.begin(), kv_views.end());
        kv_views.clear();
      }
    }
    if (!kv_views.empty()) {
      cluster->hset(v_key, kv_views.begin(), kv_views.end());
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename MaskType, typename KeyType, typename MetaType, typename PartitionerType>
void RedisClusterTable<MaskType, KeyType, MetaType, PartitionerType>::update_accumulate(
    context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw,
    const int64_t update_stride, const int64_t update_size, buffer_ptr<const void> updates_bw,
    const DataType_t update_dtype) {
  if (n <= 0) return;
  const auto& __restrict config{config_};
  redis_conn_ptr_t& cluster{conn_};

  auto modify_stream = ctx->get_modify_stream();
  const void* const keys_vptr{
      keys_bw ? keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream)
              : nullptr};
  const void* const updates_vptr{
      updates_bw ? updates_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream)
                 : nullptr};
  const key_type* const __restrict keys{reinterpret_cast<const key_type*>(keys_vptr)};
  const char* const __restrict updates{reinterpret_cast<const char*>(updates_vptr)};

  const int64_t max_value_size{config.max_value_size};
  NVE_CHECK_(update_size >= 0 && update_size <= max_value_size);
  const int64_t max_batch_size{std::min(n, config.max_batch_size)};
  const update_kernel_t update_kernel{pick_cpu_update_kernel(config.value_dtype, update_dtype)};

  // TODO: Prone to memory fragmentation. Also inefficent. What we would need is an op that does
  // the opposite to `HSETNX`.
  std::vector<max_bitmask_repr_t> hit_mask(static_cast<uint64_t>(max_bitmask_t::mask_size(n)), {});
  char* const __restrict hm{reinterpret_cast<char*>(hit_mask.data())};

  std::vector<char> values_vec(static_cast<uint64_t>(n * max_value_size));
  std::vector<int64_t> value_sizes_vec(static_cast<uint64_t>(n));
  find_<true, true>(ctx, n, keys, hm, max_value_size, values_vec.data(), value_sizes_vec.data());

  char* const __restrict values{values_vec.data()};
  const int64_t* const __restrict value_sizes{value_sizes_vec.data()};

  const int64_t num_parts{config.num_partitions};

  // Standalone string mode: accumulate into the found keys and MSET them back, split across
  // `num_parts` thread-pool tasks (each owns a disjoint, word-aligned slice of the indices).
  if (is_string_mode(cluster, config)) {
    const int64_t string_namespace_id{config.string_namespace_id};
    const int64_t entry_size{string_key_size(string_namespace_id, sizeof(key_type))};
    const auto task{[&cluster, &update_kernel, keys, hm, values, value_sizes, updates, update_stride,
                     update_size, max_value_size, string_namespace_id, entry_size,
                     max_batch_size](const int64_t lo, const int64_t hi) {
      std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> kv_views;
      kv_views.reserve(static_cast<uint64_t>(max_batch_size));
      std::vector<char> key_buf(
          static_cast<uint64_t>(string_namespace_id >= 0 ? max_batch_size * entry_size : 0));

      const auto process_batch{
          [&cluster, &kv_views]() { cluster->mset(kv_views.begin(), kv_views.end()); }};

      for (int64_t i{lo}; i < hi; i += mask_type::num_bits) {
        for (mask_repr_type it{mask_type::load(hm, i)}; mask_type::has_next(it);
             it = mask_type::skip(it)) {
          const int64_t ij{i + mask_type::next(it)};

          update_kernel(&values[ij * max_value_size], &updates[ij * update_stride], update_size);
          kv_views.emplace_back(
              std::piecewise_construct,
              std::forward_as_tuple(write_string_key(key_buf.data(),
                                                     static_cast<int64_t>(kv_views.size()),
                                                     entry_size, string_namespace_id, keys[ij])),
              std::forward_as_tuple(values + ij * max_value_size,
                                    std::max(value_sizes[ij], update_size)));
          if NVE_LIKELY_(static_cast<int64_t>(kv_views.size()) < max_batch_size) continue;

          process_batch();
          kv_views.clear();
        }
      }
      if (!kv_views.empty()) {
        process_batch();
      }
    }};
    run_partitioned<mask_type::num_bits>(ctx, n, num_parts, config.workgroups, task);
    return;
  }

  if (num_parts == 0) {
    // Single-hash mode: accumulate into the found keys and HSET them back into the shared hash.
    const sw::redis::StringView v_key{config.hash_key};
    std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> kv_views;
    kv_views.reserve(static_cast<uint64_t>(max_batch_size));
    const auto process_batch{[&cluster, &kv_views, &v_key]() {
      cluster->hset(v_key, kv_views.begin(), kv_views.end());
    }};

    for (int64_t i{}; i < n; i += mask_type::num_bits) {
      for (mask_repr_type it{mask_type::load(hm, i)}; mask_type::has_next(it);
           it = mask_type::skip(it)) {
        const int64_t ij{i + mask_type::next(it)};

        update_kernel(&values[ij * max_value_size], &updates[ij * update_stride], update_size);
        kv_views.emplace_back(
            std::piecewise_construct,
            std::forward_as_tuple(reinterpret_cast<const char*>(&keys[ij]), sizeof(key_type)),
            std::forward_as_tuple(values + ij * max_value_size,
                                  std::max(value_sizes[ij], update_size)));
        if NVE_LIKELY_(static_cast<int64_t>(kv_views.size()) < max_batch_size) continue;

        process_batch();
        kv_views.clear();
      }
    }
    if (!kv_views.empty()) {
      process_batch();
    }

    return;
  }

  const int64_t num_parts_mask{num_parts - 1};
  const table_id_t table_id{this->id};
  const auto f{[n, keys, update_stride, update_size, updates, table_id, max_value_size,
                max_batch_size, &update_kernel, &cluster, num_parts_mask, hm, values,
                value_sizes](const int64_t task_idx) {
    const HKey v_key(table_id, task_idx, 'v');

    // TODO: Prone to memory fragmentation. Use scratch buffer instead?
    std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> kv_views;
    kv_views.reserve(static_cast<uint64_t>(max_batch_size));

    // Fill up batch and lodge queries as we go.
    for (int64_t i{}; i < n; i += mask_type::num_bits) {
      for (mask_repr_type it{mask_type::load(hm, i)}; mask_type::has_next(it);
           it = mask_type::skip(it)) {
        const int64_t ij{i + mask_type::next(it)};
        const key_type& key{keys[ij]};
        if (partitioner(key, num_parts_mask) != task_idx) continue;

        update_kernel(&values[ij * max_value_size], &updates[ij * update_stride], update_size);
        kv_views.emplace_back(
            std::piecewise_construct,
            std::forward_as_tuple(reinterpret_cast<const char*>(&key), sizeof(key_type)),
            std::forward_as_tuple(values + ij * max_value_size,
                                  std::max(value_sizes[ij], update_size)));
        if NVE_LIKELY_(static_cast<int64_t>(kv_views.size()) < max_batch_size) continue;

        cluster->hset(v_key, kv_views.begin(), kv_views.end());
        kv_views.clear();
      }
    }
    if (!kv_views.empty()) {
      cluster->hset(v_key, kv_views.begin(), kv_views.end());
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename MaskType, typename KeyType, typename MetaType, typename PartitionerType>
template <bool WithValues, bool WithValueSizes>
int64_t RedisClusterTable<MaskType, KeyType, MetaType, PartitionerType>::find_(
    context_ptr_t& ctx, const int64_t n, const key_type* const __restrict keys,
    char* const __restrict hm, const int64_t value_stride, char* const __restrict values,
    int64_t* const __restrict value_sizes) const {
  const auto& __restrict config{config_};
  const redis_conn_ptr_t& cluster{conn_};

  using parser_t = ReplyParser<sw::redis::Optional<sw::redis::StringView>>;

  const int64_t max_batch_size{std::min(n, config.max_batch_size)};

  const int64_t num_parts{config.num_partitions};

  // Standalone string mode: MGET (optionally prefixed) per-key strings, split across `num_parts`
  // thread-pool tasks. Ranges are word-aligned so each task owns its hit-mask words and writes stay
  // non-atomic.
  if (is_string_mode(cluster, config)) {
    const int64_t string_namespace_id{config.string_namespace_id};
    const int64_t entry_size{string_key_size(string_namespace_id, sizeof(key_type))};
    const bool prefixed{string_namespace_id >= 0};
    std::atomic_int64_t total_num_hits{0};

    const auto task{[&cluster, &total_num_hits, keys, hm, value_stride, values, value_sizes,
                     string_namespace_id, entry_size, prefixed, max_batch_size](const int64_t lo,
                                                                       const int64_t hi) {
      // Only needed to materialize prefixed keys; raw keys are referenced in place (zero-copy).
      std::vector<char> key_buf(static_cast<uint64_t>(prefixed ? max_batch_size * entry_size : 0));

      std::vector<sw::redis::StringView> k_views;
      k_views.reserve(static_cast<uint64_t>(max_batch_size));
      // Prefixed keys live in `key_buf`, so their source index `ij` needs this parallel map;
      // un-prefixed keys are referenced in place and recovered by pointer arithmetic into `keys`.
      std::vector<int64_t> view_to_ij;
      if (prefixed) view_to_ij.reserve(static_cast<uint64_t>(max_batch_size));

      int64_t num_hits{};
      const auto callback{
          [keys, hm, value_stride, values, value_sizes, &k_views, &view_to_ij, prefixed, &num_hits](
              const int64_t view_idx, const sw::redis::Optional<sw::redis::StringView>& v_view_opt) {
            if (!v_view_opt) return;

            const sw::redis::StringView& __restrict k_view{
                k_views[static_cast<uint64_t>(view_idx)]};
            const int64_t ij{prefixed ? view_to_ij[static_cast<uint64_t>(view_idx)]
                                       : reinterpret_cast<const key_type*>(k_view.data()) - keys};

            const sw::redis::StringView& v_view(*v_view_opt);
            const int64_t value_size{static_cast<int64_t>(v_view.size())};
            NVE_CHECK_(value_size <= value_stride, "The value stored in Redis (=", value_size,
                       ") exceeds the avaiable value stride (=", value_stride, ").");
            if (value_sizes) {
              value_sizes[ij] = value_size;
            }
            if (values) {
              std::copy_n(v_view.data(), value_size, &values[ij * value_stride]);
            }

            ++num_hits;
            mask_repr_type mask{mask_type::load(hm, ij)};
            mask = mask_type::insert(mask, ij & mask_type::num_bits_mask);
            mask_type::store(hm, ij, mask);
          }};

      const auto process_batch{[&cluster, &k_views, &callback]() {
        cluster->mget(k_views.begin(), k_views.end(), parser_t(callback));
      }};

      for (int64_t i{lo}; i < hi; i += mask_type::num_bits) {
        auto it{mask_type::clip(mask_type::invert(mask_type::load(hm, i)), hi - i)};

        // Run query if the batch is about to overflow.
        if NVE_UNLIKELY_(static_cast<int64_t>(k_views.size()) + mask_type::count(it) >
                         max_batch_size) {
          process_batch();
          k_views.clear();
          view_to_ij.clear();
        }

        for (; mask_type::has_next(it); it = mask_type::skip(it)) {
          const int64_t ij{i + mask_type::next(it)};
          const sw::redis::StringView k_view{
              prefixed ? write_string_key(key_buf.data(), static_cast<int64_t>(k_views.size()),
                                          entry_size, string_namespace_id, keys[ij])
                       : sw::redis::StringView{reinterpret_cast<const char*>(&keys[ij]),
                                               sizeof(key_type)}};
          k_views.emplace_back(k_view);
          if (prefixed) view_to_ij.emplace_back(ij);
        }
      }
      if (!k_views.empty()) {
        process_batch();
      }

      total_num_hits.fetch_add(num_hits, std::memory_order_relaxed);
    }};

    run_partitioned<mask_type::num_bits>(ctx, n, num_parts, config.workgroups, task);
    return total_num_hits.load(std::memory_order_relaxed);
  }

  if (num_parts == 0) {
    // Single-hash mode: HMGET fields from the shared hash (hash_key). Single-threaded.
    const sw::redis::StringView v_key{config.hash_key};
    std::vector<sw::redis::StringView> k_views;
    k_views.reserve(static_cast<uint64_t>(max_batch_size));

    int64_t num_hits{};
    const auto callback{[keys, hm, value_stride, values, value_sizes, &k_views, &num_hits](
                            const int64_t view_idx,
                            const sw::redis::Optional<sw::redis::StringView>& v_view_opt) {
      if (!v_view_opt) return;

      const sw::redis::StringView& __restrict k_view{k_views[static_cast<uint64_t>(view_idx)]};
      const int64_t ij{reinterpret_cast<const key_type*>(k_view.data()) - keys};

      const sw::redis::StringView& v_view(*v_view_opt);
      const int64_t value_size{static_cast<int64_t>(v_view.size())};
      NVE_CHECK_(value_size <= value_stride, "The value stored in Redis (=", value_size,
                 ") exceeds the avaiable value stride (=", value_stride, ").");
      if (value_sizes) {
        value_sizes[ij] = value_size;
      }
      if (values) {
        std::copy_n(v_view.data(), value_size, &values[ij * value_stride]);
      }

      ++num_hits;
      mask_repr_type mask{mask_type::load(hm, ij)};
      mask = mask_type::insert(mask, ij & mask_type::num_bits_mask);
      mask_type::store(hm, ij, mask);
    }};

    const auto process_batch{[&cluster, &k_views, &callback, &v_key]() {
      cluster->hmget(v_key, k_views.begin(), k_views.end(), parser_t(callback));
    }};

    for (int64_t i{}; i < n; i += mask_type::num_bits) {
      auto it{mask_type::clip(mask_type::invert(mask_type::load(hm, i)), n - i)};

      // Run query if the batch is about to overflow.
      if NVE_UNLIKELY_(static_cast<int64_t>(k_views.size()) + mask_type::count(it) > max_batch_size) {
        process_batch();
        k_views.clear();
      }

      for (; mask_type::has_next(it); it = mask_type::skip(it)) {
        const key_type& key{keys[i + mask_type::next(it)]};
        k_views.emplace_back(reinterpret_cast<const char*>(&key), sizeof(key_type));
      }
    }
    if (!k_views.empty()) {
      process_batch();
    }

    return num_hits;
  }

  const table_id_t table_id{this->id};
  std::atomic_int64_t total_num_hits{0};
  const auto f{[n, keys, hm, value_stride, values, value_sizes, table_id, max_batch_size, &cluster,
                num_parts, &total_num_hits](const int64_t part_idx) {
    const HKey v_key(table_id, part_idx, 'v');
    const HKey m_key(table_id, part_idx, 'm');

    // TODO: Prone to memory fragmentation. Use scratch buffer instead?
    std::vector<sw::redis::StringView> k_views;
    k_views.reserve(static_cast<uint64_t>(max_batch_size));

    lru_meta_type lru_value;
    std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> meta_km_views;
    std::vector<sw::redis::StringView> meta_k_views;
    if constexpr (std::is_same_v<meta_type, no_meta_type>) {
    } else if constexpr (std::is_same_v<meta_type, lru_meta_type>) {
      meta_km_views.reserve(static_cast<uint64_t>(max_batch_size));
    } else if constexpr (std::is_same_v<meta_type, lfu_meta_type>) {
      meta_k_views.reserve(static_cast<uint64_t>(max_batch_size));
    } else {
      static_assert(dependent_false_v<meta_type>, "Overflow handler not implemented.");
    }

    int64_t num_hits{};
    const auto callback{[keys, hm, value_stride, values, value_sizes, &k_views, &lru_value,
                         &meta_km_views, &meta_k_views,
                         &num_hits](const int64_t view_idx,
                                    const sw::redis::Optional<sw::redis::StringView>& v_view_opt) {
      if (!v_view_opt) return;

      // Reconstruct ij from the key view.
      const sw::redis::StringView& __restrict k_view{k_views[static_cast<uint64_t>(view_idx)]};
      const int64_t ij{reinterpret_cast<const key_type*>(k_view.data()) - keys};

      const sw::redis::StringView& v_view(*v_view_opt);
      const int64_t value_size{static_cast<int64_t>(v_view.size())};
      NVE_CHECK_(value_size <= value_stride, "The value stored in Redis (=", value_size,
                 ") exceeds the avaiable value stride (=", value_stride, ").");
      if (value_sizes) {
        value_sizes[ij] = value_size;
      }
      if (values) {
        std::copy_n(v_view.data(), value_size, &values[ij * value_stride]);
      }

      if constexpr (std::is_same_v<meta_type, no_meta_type>) {
        (void)lru_value;
        (void)meta_km_views;
        (void)meta_k_views;
      } else if constexpr (std::is_same_v<meta_type, lru_meta_type>) {
        meta_km_views.emplace_back(k_view, view_as_string(lru_value));
        (void)meta_k_views;
      } else if constexpr (std::is_same_v<meta_type, lfu_meta_type>) {
        meta_k_views.emplace_back(k_view);
        (void)lru_value;
        (void)meta_km_views;
      } else {
        static_assert(dependent_false_v<meta_type>, "Overflow handler not implemented.");
      }

      ++num_hits;
      const mask_repr_type mask{
          mask_type::single(ij & mask_type::num_bits_mask)};
      mask_type::atomic_join(hm, ij, mask);
    }};

    const auto process_batch{[&cluster, &v_key, &m_key, &k_views, &lru_value, &meta_km_views,
                              &meta_k_views, &callback]() {
      cluster->hmget(v_key, k_views.begin(), k_views.end(), parser_t(callback));

      if constexpr (std::is_same_v<meta_type, no_meta_type>) {
        (void)m_key;
        (void)lru_value;
        (void)meta_km_views;
        (void)meta_k_views;
      } else if constexpr (std::is_same_v<meta_type, lru_meta_type>) {
        if (!meta_km_views.empty()) {
          lru_value = lru_meta_value();
          cluster->hset(m_key, meta_km_views.begin(), meta_km_views.end());
          meta_km_views.clear();
        }
        (void)meta_k_views;
      } else if constexpr (std::is_same_v<meta_type, lfu_meta_type>) {
        if (!meta_k_views.empty()) {
          sw::redis::Pipeline pipe{cluster->pipeline(m_key)};
          for (const sw::redis::StringView& k_view : meta_k_views) {
            pipe.hincrby(m_key, k_view, 1);
          }
          pipe.exec();
          meta_k_views.clear();
        }
        (void)lru_value;
        (void)meta_km_views;
      } else {
        static_assert(dependent_false_v<meta_type>, "Overflow handler not implemented.");
      }
    }};

    const int64_t n_aligned{next_aligned<mask_type::num_bits>(n)};
    const int64_t off{n_aligned / mask_type::num_bits * part_idx / num_parts * mask_type::num_bits};
    const int64_t num_parts_mask{num_parts - 1};

    for (int64_t i0{}; i0 != n_aligned; i0 += mask_type::num_bits) {
      const int64_t i{(i0 + off) % n_aligned};
      mask_repr_type it{mask_type::load(hm, i)};
      it = mask_type::clip(mask_type::invert(it), n - i);

      // Run query if the batch is about to overflow.
      if NVE_UNLIKELY_(static_cast<int64_t>(k_views.size()) + mask_type::count(it) > max_batch_size) {
        process_batch();
        k_views.clear();
      }

      for (; mask_type::has_next(it); it = mask_type::skip(it)) {
        const key_type& key{keys[i + mask_type::next(it)]};
        if (partitioner(key, num_parts_mask) != part_idx) continue;
        k_views.emplace_back(reinterpret_cast<const char*>(&key), sizeof(key_type));
      }
    }
    if (!k_views.empty()) {
      process_batch();
    }

    total_num_hits.fetch_add(num_hits, std::memory_order_relaxed);
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
  return total_num_hits;
}

void RedisClusterTableFactoryConfig::check() const {
  base_type::check();

  // TODO: Validate address using regex.
  NVE_CHECK_(!address.empty());
  NVE_CHECK_(!user_name.empty());

  NVE_CHECK_(connections_per_node > 0);

  if (use_tls) {
    NVE_CHECK_(!ca_certificate.empty());
    NVE_CHECK_(!client_certificate.empty());
    NVE_CHECK_(!client_key.empty());
    NVE_CHECK_(!server_name_identification.empty());
  }
}

void from_json(const nlohmann::json& json, RedisClusterTableFactoryConfig& conf) {
  using base_type = RedisClusterTableFactoryConfig::base_type;
  from_json(json, static_cast<base_type&>(conf));

  NVE_READ_JSON_FIELD_(address);
  NVE_READ_JSON_FIELD_(user_name);
  NVE_READ_JSON_FIELD_(password);
  NVE_READ_JSON_FIELD_(single_node);

  NVE_READ_JSON_FIELD_(keep_alive);
  NVE_READ_JSON_FIELD_(connections_per_node);

  NVE_READ_JSON_FIELD_(use_tls);
  NVE_READ_JSON_FIELD_(ca_certificate);
  NVE_READ_JSON_FIELD_(client_certificate);
  NVE_READ_JSON_FIELD_(client_key);
  NVE_READ_JSON_FIELD_(server_name_identification);
}

void to_json(nlohmann::json& json, const RedisClusterTableFactoryConfig& conf) {
  using base_type = RedisClusterTableFactoryConfig::base_type;
  to_json(json, static_cast<const base_type&>(conf));

  NVE_WRITE_JSON_FIELD_(address);
  NVE_WRITE_JSON_FIELD_(user_name);
  NVE_WRITE_JSON_FIELD_(password);
  NVE_WRITE_JSON_FIELD_(single_node);

  NVE_WRITE_JSON_FIELD_(keep_alive);
  NVE_WRITE_JSON_FIELD_(connections_per_node);

  NVE_WRITE_JSON_FIELD_(use_tls);
  NVE_WRITE_JSON_FIELD_(ca_certificate);
  NVE_WRITE_JSON_FIELD_(client_certificate);
  NVE_WRITE_JSON_FIELD_(client_key);
  NVE_WRITE_JSON_FIELD_(server_name_identification);
}

RedisClusterTableFactory::RedisClusterTableFactory(const config_type& config)
    : base_type(config), single_node_{config.single_node} {
  const std::string& addr{config.address};
  const char* const kind{single_node_ ? "standalone" : "cluster"};
  NVE_LOG_INFO_("Connecting to Redis ", kind, " '", addr, "'.");

  sw::redis::ConnectionOptions conn_opts;
  sw::redis::ConnectionPoolOptions pool_opts;

  // Basic connection setup.
  const uint64_t colon_pos{addr.find(':')};
  if (colon_pos == std::string::npos) {
    conn_opts.host = addr;
  } else {
    conn_opts.host = addr.substr(0, colon_pos);
    conn_opts.port = std::stoi(addr.substr(colon_pos + 1));
  }
  conn_opts.user = config.user_name;
  conn_opts.password = config.password;

  conn_opts.keep_alive = config.keep_alive;
  pool_opts.size = static_cast<uint64_t>(config.connections_per_node);

  // TLS/SSL related seutp.
  conn_opts.tls.enabled = config.use_tls;
  if (std::filesystem::is_directory(config.ca_certificate)) {
    conn_opts.tls.cacertdir = config.ca_certificate;
  } else {
    conn_opts.tls.cacert = config.ca_certificate;
  }
  conn_opts.tls.cert = config.client_certificate;
  conn_opts.tls.key = config.client_key;
  conn_opts.tls.sni = config.server_name_identification;

  // Connect to the server (standalone) or cluster, wrapping the connection so the table can issue
  // commands without caring which kind it is.
  if (single_node_) {
    redis_ptr_t standalone{new sw::redis::Redis(conn_opts, pool_opts),
                           [addr](sw::redis::Redis* const p) {
                             NVE_LOG_INFO_("Disconnecting from Redis standalone '", addr, "'.");
                             delete p;
                             NVE_LOG_INFO_("Disconnection from Redis standalone '", addr,
                                           "' concluded.");
                           }};
    conn_ = std::make_shared<RedisConn>(std::move(standalone));
  } else {
    redis_cluster_ptr_t cluster{new sw::redis::RedisCluster(conn_opts, pool_opts),
                                [addr](sw::redis::RedisCluster* const p) {
                                  NVE_LOG_INFO_("Disconnecting from Redis cluster '", addr, "'.");
                                  delete p;
                                  NVE_LOG_INFO_("Disconnection from Redis cluster '", addr,
                                                "' concluded.");
                                }};
    conn_ = std::make_shared<RedisConn>(std::move(cluster));
  }

  NVE_LOG_INFO_("Connection to Redis ", kind, " '", addr, "' established.");
}

template <typename MaskType, typename KeyType, typename MetaType>
inline static host_table_ptr_t make_redis_cluster_table_3(table_id_t id,
                                                          const RedisClusterTableConfig& config,
                                                          redis_conn_ptr_t& conn,
                                                          const bool string_mode) {
  // String mode splits work by index range and never calls the partitioner, and a single partition
  // routes everything to one task — both use AlwaysZeroPartitioner so no `ht_part_*` feature is
  // required.
  if (config.num_partitions == 1 || string_mode) {
    if (!string_mode && config.partitioner != Partitioner_t::AlwaysZero) {
      NVE_LOG_VERBOSE_("Selected ", config.partitioner, " partitioner was disabled because table has only 1 partition.");
    }
    return std::make_shared<RedisClusterTable<MaskType, KeyType, MetaType, AlwaysZeroPartitioner>>(id, config, conn);
  }

  switch (config.partitioner) {
#if defined(NVE_FEATURE_HT_PART_FNV1A)
    case Partitioner_t::FowlerNollVo:
      return std::make_shared<RedisClusterTable<MaskType, KeyType, MetaType, FowlerNollVoPartitioner>>(id, config, conn);
#endif
#if defined(NVE_FEATURE_HT_PART_MURMUR)
    case Partitioner_t::Murmur3:
      return std::make_shared<RedisClusterTable<MaskType, KeyType, MetaType, Murmur3Partitioner>>(id, config, conn);
#endif
#if defined(NVE_FEATURE_HT_PART_RRXMRRXMSX0)
    case Partitioner_t::Rrxmrrxmsx0:
      return std::make_shared<RedisClusterTable<MaskType, KeyType, MetaType, Rrxmrrxmsx0Partitioner>>(id, config, conn);
#endif
#if defined(NVE_FEATURE_HT_PART_STD_HASH)
    case Partitioner_t::StdHash:
      return std::make_shared<RedisClusterTable<MaskType, KeyType, MetaType, StdHashPartitioner>>(id, config, conn);
#endif
    default:
      NVE_THROW_("`config.partitioner` (", config.partitioner, ") is out of bounds!");
  }
}

template <typename MaskType, typename KeyType>
inline static host_table_ptr_t make_redis_cluster_table_2(const table_id_t id,
                                                          const RedisClusterTableConfig& config,
                                                          redis_conn_ptr_t& conn,
                                                          const bool string_mode) {
  switch (config.overflow_policy.handler) {
    case OverflowHandler_t::EvictRandom:
      return make_redis_cluster_table_3<MaskType, KeyType, no_meta_type>(id, config, conn, string_mode);
    case OverflowHandler_t::EvictLRU:
      return make_redis_cluster_table_3<MaskType, KeyType, lru_meta_type>(id, config, conn, string_mode);
    case OverflowHandler_t::EvictLFU:
      return make_redis_cluster_table_3<MaskType, KeyType, lfu_meta_type>(id, config, conn, string_mode);
  }
  NVE_THROW_("`config.overflow_policy.handler` (", config.overflow_policy.handler,
             ") is out of bounds!");
}

template <typename MaskType>
static host_table_ptr_t make_redis_cluster_table_1(table_id_t id,
                                                   const RedisClusterTableConfig& config,
                                                   redis_conn_ptr_t& conn,
                                                   const bool string_mode) {
  switch (config.key_size) {
#if defined(NVE_FEATURE_HT_KEY_8)
    case sizeof(int8_t):
      return make_redis_cluster_table_2<MaskType, int8_t>(id, config, conn, string_mode);
#endif
#if defined(NVE_FEATURE_HT_KEY_16)
    case sizeof(int16_t):
      return make_redis_cluster_table_2<MaskType, int16_t>(id, config, conn, string_mode);
#endif
#if defined(NVE_FEATURE_HT_KEY_32)
    case sizeof(int32_t):
      return make_redis_cluster_table_2<MaskType, int32_t>(id, config, conn, string_mode);
#endif
#if defined(NVE_FEATURE_HT_KEY_64)
    case sizeof(int64_t):
      return make_redis_cluster_table_2<MaskType, int64_t>(id, config, conn, string_mode);
#endif
  }
  NVE_THROW_("`config.key_size` (", config.key_size, ") is out of bounds!");
}

host_table_ptr_t RedisClusterTableFactory::produce(const table_id_t id,
                                                   const RedisClusterTableConfig& config) {
  // String mode: store each entry as a Redis string (`MSET`/`MGET`/`DEL`) on a standalone server.
  // `num_partitions` (when > 1) only splits the client-side work across threads — the keys are plain
  // strings, so no partitioner is needed and storage is unaffected. String mode requires a standalone
  // connection (`MSET`/`MGET` across arbitrary keys hit `CROSSSLOT` errors on a cluster) and offers no
  // metadata/eviction support (capacity must be bounded via the Redis server's `maxmemory` policy).
  const bool string_mode{is_string_mode(conn_, config)};
  if (string_mode) {
    NVE_CHECK_(config.overflow_policy.handler == OverflowHandler_t::EvictRandom,
               "String mode does not support LRU/LFU eviction; set the overflow handler to "
               "`evict_random` and bound capacity via the Redis server's `maxmemory` policy.");
  } else {
    // Hash modes shard keys across `num_partitions` Redis hashes by hash slot, which only makes sense
    // against a cluster; a standalone connection can host a single hash (`num_partitions` 0 or 1).
    NVE_CHECK_(single_node_ ? (config.num_partitions == 0 || config.num_partitions == 1) : true,
               "A standalone Redis connection in hash mode requires `num_partitions` of 0 or 1, but "
               "got ",
               config.num_partitions, '.');
  }

  switch (config.mask_size) {
#if defined(NVE_FEATURE_HT_MASK_8)
    case bitmask8_t::size:
      return make_redis_cluster_table_1<bitmask8_t>(id, config, conn_, string_mode);
#endif
#if defined(NVE_FEATURE_HT_MASK_16)
    case bitmask16_t::size:
      return make_redis_cluster_table_1<bitmask16_t>(id, config, conn_, string_mode);
#endif
#if defined(NVE_FEATURE_HT_MASK_32)
    case bitmask32_t::size:
      return make_redis_cluster_table_1<bitmask32_t>(id, config, conn_, string_mode);
#endif
#if defined(NVE_FEATURE_HT_MASK_64)
    case bitmask64_t::size:
      return make_redis_cluster_table_1<bitmask64_t>(id, config, conn_, string_mode);
#endif
  }
  NVE_THROW_("`config.mask_size` (", config.mask_size, ") is out of bounds!");
}

}  // namespace plugin
}  // namespace nve
