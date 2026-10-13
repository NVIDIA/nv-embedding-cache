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
#include <charconv>
#include <cstring>
#include <string>
#include <string_view>

namespace nve {
namespace plugin {

void RedisClusterTableConfig::check() const {
  base_type::check();

  NVE_CHECK_(max_batch_size > 0 && max_batch_size % bitmask64::num_bits == 0);

  NVE_CHECK_(!num_partitions || (num_partitions >= 0 && has_single_bit(to_uint(num_partitions))));
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
    NVE_CHECK_(req_size > 0 && req_size < static_cast<int64_t>(c_str_.size()));
    size_ = req_size;
  }

  constexpr const char* c_str() const noexcept { return c_str_.data(); }

  constexpr operator sw::redis::StringView() const noexcept { return {c_str_.data(), to_uint(size_)}; }

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
  return {dst, to_uint(entry_size)};
}

// Builds the `SCAN MATCH` glob pattern selecting all keys carrying `string_namespace_id`: the prefix bytes
// (with glob meta-characters escaped) followed by '*'.
inline static std::string make_prefix_scan_pattern(int64_t string_namespace_id) {
  const char* const p{reinterpret_cast<const char*>(&string_namespace_id)};
  std::string pattern;
  pattern.reserve(to_uint(kKeyPrefixSize) * 2 + 1);
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
constexpr static void task_range(const int64_t n, const int64_t parts, const int64_t t, int64_t& lo,
                              int64_t& hi) {
  const int64_t chunk{round_up(ceil_div(n, parts), Align)};
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

template <typename KeyType, typename MetaType, typename PartitionerType>
RedisClusterTable<KeyType, MetaType, PartitionerType>::RedisClusterTable(
    table_id_t table_id, const config_type& config, redis_conn_ptr_t& conn)
    : base_type(table_id, config), conn_{conn} {}

template <typename KeyType, typename MetaType, typename PartitionerType>
void RedisClusterTable<KeyType, MetaType, PartitionerType>::clear(context_ptr_t& ctx) {
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
    const auto hkeys{make_array(
      HKey{table_id, task_idx, 'v'},
      HKey{table_id, task_idx, 'm'}
    )};
    cluster->del(hkeys.begin(), hkeys.end());
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename KeyType, typename MetaType, typename PartitionerType>
void RedisClusterTable<KeyType, MetaType, PartitionerType>::erase(context_ptr_t& ctx,
                                                                            int64_t n,
                                                                            buffer_ptr<const void> keys_bw) {
  if (n <= 0) return;
  auto modify_stream{ctx->get_modify_stream()};
  const auto& __restrict config{config_};
  redis_conn_ptr_t& cluster{conn_};

  NVE_CHECK_ARG_(keys_bw != nullptr, "keys must not be null");
  const void* const keys_vptr{
      keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy_content*/, modify_stream)};
  const key_type* const __restrict keys{reinterpret_cast<const key_type*>(keys_vptr)};
  const int64_t max_batch_size{std::min(n, config.max_batch_size)};

  const int64_t num_parts{config.num_partitions};
  // Standalone string mode: DEL (optionally prefixed) per-key strings, split across `num_parts`
  // thread-pool tasks for client-side parallelism.
  if (is_string_mode(cluster, config)) {
    const int64_t string_namespace_id{config.string_namespace_id};
    const int64_t entry_size{string_key_size(string_namespace_id, sizeof(key_type))};

    const auto task{[&cluster, keys, string_namespace_id, entry_size, max_batch_size](const int64_t lo, const int64_t hi) {
      std::vector<sw::redis::StringView> k_views;
      k_views.reserve(to_uint(max_batch_size));
      // Only needed to materialize prefixed keys; raw keys are referenced in place (zero-copy).
      std::vector<char> key_buf(string_namespace_id >= 0 ? to_uint(max_batch_size * entry_size) : 0);

      for (int64_t ij{lo}; ij < hi; ++ij) {
        k_views.emplace_back(write_string_key(key_buf.data(),
                                              static_cast<int64_t>(k_views.size()), entry_size,
                                              string_namespace_id, keys[ij]));
        if NVE_LIKELY_(static_cast<int64_t>(k_views.size()) < max_batch_size) continue;

        cluster->del(k_views.begin(), k_views.end());
        k_views.clear();
      }
      if (!k_views.empty()) {
        cluster->del(k_views.begin(), k_views.end());
      }
    }};
    run_partitioned<bitmask64::num_bits>(ctx, n, num_parts, config.workgroups, task);
    return;
  }

  if (num_parts == 0) {
    // Single-hash mode: HDEL fields from the shared hash.
    const sw::redis::StringView v_key{config.hash_key};

    std::vector<sw::redis::StringView> k_views;
    k_views.reserve(to_uint(max_batch_size));

    for (int64_t ij{}; ij < n; ++ij) {
      k_views.emplace_back(reinterpret_cast<const char*>(&keys[ij]), sizeof(key_type));
      if NVE_LIKELY_(static_cast<int64_t>(k_views.size()) < max_batch_size) continue;

      cluster->hdel(v_key, k_views.begin(), k_views.end());
      k_views.clear();
    }
    if (!k_views.empty()) {
      cluster->hdel(v_key, k_views.begin(), k_views.end());
    }

    return;
  }

  const int64_t num_parts_mask{num_parts - 1};
  const table_id_t table_id{this->id};

  const auto f{
      [n, keys, table_id, max_batch_size, &cluster, num_parts_mask](const int64_t task_idx) {
        const HKey v_key{table_id, task_idx, 'v'};
        const HKey m_key{table_id, task_idx, 'm'};

        // TODO: Prone to memory fragmentation. Use scratch buffer instead?
        std::vector<sw::redis::StringView> k_views;
        k_views.reserve(to_uint(max_batch_size));

        const auto process_batch{[&cluster, &v_key, &m_key, &k_views]() {
          sw::redis::Pipeline pipe{cluster->pipeline(v_key)};
          pipe.hdel(v_key, k_views.begin(), k_views.end());
          pipe.hdel(m_key, k_views.begin(), k_views.end());
          pipe.exec();
        }};

        for (int64_t ij{}; ij < n; ++ij) {
          const key_type& __restrict key{keys[ij]};
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

template <typename KeyType, typename MetaType, typename PartitionerType>
void RedisClusterTable<KeyType, MetaType, PartitionerType>::find(
    context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys_bw,
    buffer_ptr<bitmask64_t> hit_mask_bw, const int64_t value_stride,
    buffer_ptr<void> values_bw, buffer_ptr<int64_t> value_sizes_bw) const {
  if (n <= 0) return;
  auto lookup_stream{ctx->get_lookup_stream()};

  NVE_CHECK_ARG_(keys_bw != nullptr, "keys must not be null");
  const key_type* keys{static_cast<const key_type*>(
      keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, lookup_stream))};
  bitmask64_t* hit_mask;
  if (hit_mask_bw) {
    hit_mask = hit_mask_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, lookup_stream);
  } else {
    const uint64_t size{to_uint(ceil_div(n, bitmask64::num_bits)) * sizeof(bitmask64_t)};
    auto p{ctx->get_buffer("hitmask", size, true /*host_alloc*/)};
    std::memset(p, 0, size);
    hit_mask = static_cast<bitmask64_t*>(p);
  }
  char* values{nullptr};
  if (values_bw) {
    auto p{values_bw->access_buffer(cudaMemoryTypeUnregistered, false, lookup_stream)};
    values = static_cast<char*>(p);
  }
  int64_t* value_sizes{nullptr};
  if (value_sizes_bw) {
    value_sizes = value_sizes_bw->access_buffer(cudaMemoryTypeUnregistered, false, lookup_stream);
  }

  if (values) {
    if (value_sizes) {
      n = find_<true, true>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
    } else {
      n = find_<true, false>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
    }
  } else {
    if (value_sizes) {
      n = find_<false, true>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
    } else {
      n = find_<false, false>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
    }
  }

  auto counter = this->lookup_counter_storage(ctx);
  NVE_CHECK_(counter != nullptr, "Invalid key counter");
  *counter += n;
}

template <typename KeyType, typename MetaType, typename PartitionerType>
void RedisClusterTable<KeyType, MetaType, PartitionerType>::insert(
    context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw,
    const int64_t value_stride, const int64_t value_size, buffer_ptr<const void> values_bw) {
  if (n <= 0) return;
  auto modify_stream{ctx->get_modify_stream()};
  const auto& __restrict config{config_};
  redis_conn_ptr_t& cluster{conn_};

  NVE_CHECK_ARG_(keys_bw != nullptr, "keys must not be null");
  const key_type* __restrict keys{static_cast<const key_type*>(
      keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream))};
  const char* __restrict values{nullptr};
  if (values_bw) {
    auto p{values_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream)};
    values = static_cast<const char*>(p);
  }

  const table_id_t table_id{this->id};

  const int64_t max_batch_size{std::min(n, config.max_batch_size)};
  NVE_CHECK_(value_size >= 0);
  const int64_t overflow_margin{config.overflow_policy.overflow_margin};
  const int64_t resolution_margin{config.overflow_policy.abs_resolution_margin()};

  const auto resolve_overflow{[table_id, max_batch_size, resolution_margin, &cluster](
                                  const int64_t part_idx, const sw::redis::StringView& v_key,
                                  const sw::redis::StringView& m_key, int64_t part_size) {
    NVE_LOG_VERBOSE_("RedisCluster table ", table_id, " part ", part_idx,
                     " is overflowing (size = ", part_size, "). Attempting to resolve...");

    std::vector<sw::redis::StringView> k_views;
    k_views.reserve(to_uint(max_batch_size));

    const auto delete_batch{[table_id, &cluster, part_idx, &v_key, &m_key, &k_views]() {
      NVE_LOG_VERBOSE_("RedisCluster table ", table_id, " part ", part_idx,
                       ": Attempting to evict ", k_views.size(),
                       " (mode = ", overflow_handler_v<meta_type>, ").");

      sw::redis::Pipeline pipe{cluster->pipeline(v_key)};
      if constexpr (!std::is_same_v<meta_type, no_meta_t>) {
        pipe.hdel(m_key, k_views.begin(), k_views.end());
      } else {
        (void)m_key;
      }
      pipe.hdel(v_key, k_views.begin(), k_views.end());
      pipe.exec();
    }};

    do {
      if constexpr (std::is_same_v<meta_type, no_meta_t>) {
        using parser_t = ReplyParser<sw::redis::StringView>;

        // Fetch all keys in partition and shuffle them.
        std::vector<key_type> keys;
        keys.reserve(to_uint(part_size));
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
        std::shuffle(keys.begin(), keys.end(), std::default_random_engine{random_seed()});

        for (auto it{keys.begin() + resolution_margin}; it < keys.end(); ++it) {
          k_views.emplace_back(reinterpret_cast<const char*>(&*it), sizeof(key_type));
          if NVE_LIKELY_(static_cast<int64_t>(k_views.size()) < max_batch_size) continue;

          delete_batch();
          k_views.clear();
        }
        if (!k_views.empty()) {
          delete_batch();
        }
      } else if constexpr (std::is_same_v<meta_type, lru_meta_t>) {
        using view_pair_t = std::pair<sw::redis::StringView, sw::redis::StringView>;
        using parser_t = ReplyParser<view_pair_t>;

        // Fetch all keys, and sort by timestamp.
        std::vector<std::pair<key_type, lru_meta_t>> keys_metas;
        keys_metas.reserve(to_uint(part_size));
        cluster->hgetall(
            m_key, parser_t([&keys_metas](int64_t, const view_pair_t& view) {
              keys_metas.emplace_back(*reinterpret_cast<const key_type*>(view.first.data()),
                                      *reinterpret_cast<const lru_meta_t*>(view.second.data()));
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

        for (auto it{keys_metas.begin() + resolution_margin}; it < keys_metas.end(); ++it) {
          k_views.emplace_back(reinterpret_cast<const char*>(&it->first), sizeof(key_type));
          if NVE_LIKELY_(static_cast<int64_t>(k_views.size()) < max_batch_size) continue;

          delete_batch();
          k_views.clear();
        }
        if (!k_views.empty()) {
          delete_batch();
        }
      } else if constexpr (std::is_same_v<meta_type, lfu_meta_t>) {
        using view_pair_t = std::pair<sw::redis::StringView, sw::redis::StringView>;
        using parser_t = ReplyParser<view_pair_t>;

        // Fetch all keys, and sort by access count.
        std::vector<std::pair<key_type, lfu_meta_t>> keys_metas;
        keys_metas.reserve(to_uint(part_size));
        cluster->hgetall(
            m_key, parser_t{[&keys_metas](int64_t, const view_pair_t& view) {
              auto key{*reinterpret_cast<const key_type*>(view.first.data())};
              int64_t meta_value{};
              std::from_chars(view.second.data(), view.second.data() + view.second.size(), meta_value);
              meta_value = std::min(meta_value, static_cast<int64_t>(max_lfu_meta));
              keys_metas.emplace_back(key, static_cast<meta_type>(meta_value));
            }});
        part_size = static_cast<int64_t>(keys_metas.size());
        if (part_size <= resolution_margin) {
          NVE_LOG_VERBOSE_("RedisCluster table ", table_id, ", part ", part_idx,
                           " (size = ", part_size,
                           "): Overflow was already resolved by another process.");
          break;
        }
        std::sort(keys_metas.begin(), keys_metas.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        const int64_t mid{std::max(part_size - resolution_margin, {})};
        for (int64_t i{}; i < mid; ++i) {
          k_views.emplace_back(reinterpret_cast<const char*>(&keys_metas[to_uint(i)].first),
                               sizeof(key_type));
          if NVE_LIKELY_(static_cast<int64_t>(k_views.size()) < max_batch_size) continue;

          delete_batch();
          k_views.clear();
        }
        if (!k_views.empty()) {
          delete_batch();
        }

        for (int64_t i{mid}; i < part_size; i += max_batch_size) {
          sw::redis::Pipeline pipe{cluster->pipeline(m_key)};

          const int64_t batch_size{std::min(part_size - i, max_batch_size)};
          for (int64_t j{}; j != batch_size; ++j) {
            const auto [key, meta]{keys_metas[to_uint(i + j)]};
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
                     ": Overflow resolution complete! (size = ", resolution_margin, ')');
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
      kv_views.reserve(to_uint(max_batch_size));

      // Only needed to materialize prefixed keys; raw keys are referenced in place (zero-copy).
      std::vector<char> key_buf(string_namespace_id >= 0 ? to_uint(max_batch_size * entry_size) : 0);

      for (int64_t i{lo}; i != hi; ++i) {
        kv_views.emplace_back(
            std::piecewise_construct,
            std::forward_as_tuple(write_string_key(key_buf.data(),
                                                   static_cast<int64_t>(kv_views.size()), entry_size,
                                                   string_namespace_id, keys[i])),
            std::forward_as_tuple(values + i * value_stride, value_size));
        if NVE_LIKELY_(static_cast<int64_t>(kv_views.size()) < max_batch_size) continue;

        cluster->mset(kv_views.begin(), kv_views.end());
        kv_views.clear();
      }
      if (!kv_views.empty()) {
        cluster->mset(kv_views.begin(), kv_views.end());
      }
    }};
    run_partitioned<bitmask64::num_bits>(ctx, n, num_parts, config.workgroups, task);
    return;
  }

  if (num_parts == 0) {
    // Single-hash mode: HSET fields into the shared hash, with overflow handling.
    const sw::redis::StringView v_key{config.hash_key};

    std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> kv_views;
    kv_views.reserve(to_uint(max_batch_size));

    const auto process_batch{[overflow_margin, &cluster, &resolve_overflow, &kv_views, &v_key]() {
      int64_t part_size;
      {
        sw::redis::Pipeline pipe{cluster->pipeline(v_key)};
        pipe.hset(v_key, kv_views.begin(), kv_views.end());
        pipe.hlen(v_key);

        sw::redis::QueuedReplies replies{pipe.exec()};
        part_size = replies.get<long long>(replies.size() - 1);
      }

      // Handle overflows situations.
      if (part_size > overflow_margin) {
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
    const HKey v_key{table_id, task_idx, 'v'};
    const HKey m_key{table_id, task_idx, 'm'};

    // TODO: Prone to memory fragmentation. Use scratch buffer instead?
    std::vector<std::pair<const sw::redis::StringView, sw::redis::StringView>> kv_views;
    kv_views.reserve(to_uint(max_batch_size));

    const auto process_batch{
        [overflow_margin, &cluster, &resolve_overflow, task_idx, &v_key, &m_key, &kv_views]() {
          int64_t part_size;
          {
            sw::redis::Pipeline pipe{cluster->pipeline(v_key)};
            pipe.hset(v_key, kv_views.begin(), kv_views.end());
            if constexpr (!std::is_same_v<meta_type, no_meta_t>) {
              const meta_type meta_value{default_meta_value<meta_type>()};
              for (const auto& kv_view : kv_views) {
                if constexpr (std::is_same_v<meta_type, lru_meta_t>) {
                  pipe.hsetnx(m_key, kv_view.first, view_as_string(meta_value));
                } else if constexpr (std::is_same_v<meta_type, lfu_meta_t>) {
                  pipe.hincrby(m_key, kv_view.first, meta_value);
                }
              }
            }
            pipe.hlen(v_key);

            sw::redis::QueuedReplies replies{pipe.exec()};
            part_size = replies.get<long long>(replies.size() - 1);
          }

          // Handle overflows situations.
          if (part_size > overflow_margin) {
            resolve_overflow(task_idx, v_key, m_key, part_size);
          }
        }};

    for (int64_t ij{}; ij < n; ++ij) {
      const key_type& key{keys[ij]};
      if (partitioner(key, num_parts_mask) != task_idx) continue;

      kv_views.emplace_back(
          std::piecewise_construct,
          std::forward_as_tuple(reinterpret_cast<const char*>(&key), sizeof(key_type)),
          std::forward_as_tuple(values + ij * value_stride, value_size));
      if NVE_LIKELY_(kv_views.size() < to_uint(max_batch_size)) continue;

      process_batch();
      kv_views.clear();
    }
    if (!kv_views.empty()) {
      process_batch();
    }
  }};
  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename KeyType, typename MetaType, typename PartitionerType>
int64_t RedisClusterTable<KeyType, MetaType, PartitionerType>::size(context_ptr_t& ctx, bool) const {
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
      // TODO: There is no need to retain the strings in `batch`.
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
    const HKey v_key{table_id, task_idx, 'v'};
    const int64_t part_size{cluster->hlen(v_key)};
    total_size.fetch_add(part_size, std::memory_order_relaxed);
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
  return total_size.load(std::memory_order_relaxed);
}

template <typename KeyType, typename MetaType, typename PartitionerType>
void RedisClusterTable<KeyType, MetaType, PartitionerType>::update(
    context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw,
    const int64_t value_stride, const int64_t value_size, buffer_ptr<const void> values_bw) {
  if (n <= 0) return;
  auto modify_stream{ctx->get_modify_stream()};
  const auto& __restrict config{config_};
  redis_conn_ptr_t& cluster{conn_};

  NVE_CHECK_ARG_(keys_bw != nullptr, "keys must not be null");
  const key_type* __restrict keys{static_cast<const key_type*>(
      keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream))};
  bitmask64_t* __restrict const hit_mask{[&](){
    const uint64_t size{to_uint(ceil_div(n, bitmask64::num_bits)) * sizeof(bitmask64_t)};
    auto p{ctx->get_buffer("hitmask", size, true /*host_alloc*/)};
    std::memset(p, 0, size);
    return static_cast<bitmask64_t*>(p);
  }()};
  const char* __restrict values{nullptr};
  if (values_bw) {
    auto p{values_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream)};
    values = static_cast<const char*>(p);
  }

  // TODO: Prone to memory fragmantation. Also inefficent. What we would need is an op that does
  // the opposite to `HSETNX`.
  if (!find_<false, false>(ctx, n, keys, hit_mask, 0, nullptr, nullptr)) return;

  const int64_t max_batch_size{std::min(n, config.max_batch_size)};
  NVE_CHECK_(value_size >= 0 && value_size <= config.max_value_size);

  const int64_t num_parts{config.num_partitions};

  // Standalone string mode: MSET the found keys' new values, split across `num_parts` thread-pool
  // tasks. Each task reads its own word-aligned slice of the hit mask (read-only) and writes plain
  // string keys.
  if (is_string_mode(cluster, config)) {
    const int64_t string_namespace_id{config.string_namespace_id};
    const int64_t entry_size{string_key_size(string_namespace_id, sizeof(key_type))};
    const auto task{[&cluster, keys, hit_mask, values, value_stride, value_size, string_namespace_id, entry_size,
                     max_batch_size](const int64_t lo, const int64_t hi) {
      std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> kv_views;
      kv_views.reserve(to_uint(max_batch_size));

      std::vector<char> key_buf(string_namespace_id >= 0 ? to_uint(max_batch_size * entry_size) : 0);

      for (int64_t i0{lo}; i0 < hi; i0 += bitmask64::num_bits) {
        for (auto it{hit_mask[i0 / bitmask64::num_bits]}; it; it = bitmask64::skip(it)) {
          const int64_t ij{i0 + bitmask64::next(it)};

          kv_views.emplace_back(
              std::piecewise_construct,
              std::forward_as_tuple(write_string_key(key_buf.data(),
                                                     static_cast<int64_t>(kv_views.size()),
                                                     entry_size, string_namespace_id, keys[ij])),
              std::forward_as_tuple(values + ij * value_stride, value_size));
          if NVE_LIKELY_(static_cast<int64_t>(kv_views.size()) < max_batch_size) continue;

          cluster->mset(kv_views.begin(), kv_views.end());
          kv_views.clear();
        }
      }
      if (!kv_views.empty()) {
        cluster->mset(kv_views.begin(), kv_views.end());
      }
    }};
    run_partitioned<bitmask64::num_bits>(ctx, n, num_parts, config.workgroups, task);
    return;
  }

  if (num_parts == 0) {
    // Single-hash mode: HSET the found keys into the shared hash.
    const sw::redis::StringView v_key{config.hash_key};

    std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> kv_views;
    kv_views.reserve(to_uint(max_batch_size));

    for (int64_t i0{}; i0 < n; i0 += bitmask64::num_bits) {
      for (auto it{hit_mask[i0 / bitmask64::num_bits]}; it; it = bitmask64::skip(it)) {
        const int64_t ij{i0 + bitmask64::next(it)};

        kv_views.emplace_back(
            std::piecewise_construct,
            std::forward_as_tuple(reinterpret_cast<const char*>(&keys[ij]), sizeof(key_type)),
            std::forward_as_tuple(values + ij * value_stride, value_size));
        if NVE_LIKELY_(static_cast<int64_t>(kv_views.size()) < max_batch_size) continue;

        cluster->hset(v_key, kv_views.begin(), kv_views.end());
        kv_views.clear();
      }
    }
    if (!kv_views.empty()) {
      cluster->hset(v_key, kv_views.begin(), kv_views.end());
    }

    return;
  }

  const int64_t num_parts_mask{num_parts - 1};
  const table_id_t table_id{this->id};
  const auto f{[n, keys, hit_mask, value_stride, value_size, values, table_id, max_batch_size, &cluster,
                num_parts_mask](const int64_t task_idx) {
    const HKey v_key{table_id, task_idx, 'v'};

    // TODO: Prone to memory fragmentation. Use scratch buffer instead?
    std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> kv_views;
    kv_views.reserve(to_uint(max_batch_size));

    // Fill up batch and lodge queries as we go.
    for (int64_t i0{}; i0 < n; i0 += bitmask64::num_bits) {
      for (auto it{hit_mask[i0 / bitmask64::num_bits]}; it; it = bitmask64::skip(it)) {
        const int64_t ij{i0 + bitmask64::next(it)};
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

template <typename KeyType, typename MetaType, typename PartitionerType>
void RedisClusterTable<KeyType, MetaType, PartitionerType>::update_accumulate(
    context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw,
    const int64_t update_stride, const int64_t update_size, buffer_ptr<const void> updates_bw,
    const DataType_t update_dtype) {
  if (n <= 0) return;
  auto modify_stream{ctx->get_modify_stream()};
  const auto& __restrict config{config_};
  redis_conn_ptr_t& cluster{conn_};

  NVE_CHECK_ARG_(keys_bw != nullptr, "keys must not be null");
  const key_type* __restrict keys{static_cast<const key_type*>(
      keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream))};
  bitmask64_t* __restrict const hit_mask{[&](){
    const uint64_t size{to_uint(ceil_div(n, bitmask64::num_bits)) * sizeof(bitmask64_t)};
    auto p{ctx->get_buffer("hitmask", size, true /*host_alloc*/)};
    std::memset(p, 0, size);
    return static_cast<bitmask64_t*>(p);
  }()};
  const char* __restrict updates{nullptr};
  if (updates_bw) {
    auto p{updates_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream)};
    updates = static_cast<const char*>(p);
  }

  const int64_t max_value_size{config.max_value_size};
  NVE_CHECK_(update_size >= 0 && update_size % dtype_size(update_dtype) == 0);
  NVE_CHECK_(update_size / dtype_size(update_dtype) <= max_value_size / config.value_dtype_size());
  const update_kernel_t update_kernel{pick_cpu_update_kernel(config.value_dtype, update_dtype)};
  const int64_t max_batch_size{std::min(n, config.max_batch_size)};
  const int64_t num_parts{config.num_partitions};

  // TODO: Prone to memory fragmentation. Maybe should avoid running a full `find_`?
  std::vector<char> values(to_uint(n * max_value_size));
  std::vector<int64_t> value_sizes(to_uint(n));
  if (!find_<true, true>(ctx, n, keys, hit_mask, max_value_size, values.data(), value_sizes.data())) return;

  // Standalone string mode: accumulate into the found keys and MSET them back, split across
  // `num_parts` thread-pool tasks (each owns a disjoint, word-aligned slice of the indices).
  if (is_string_mode(cluster, config)) {
    const int64_t string_namespace_id{config.string_namespace_id};
    const int64_t entry_size{string_key_size(string_namespace_id, sizeof(key_type))};

    const auto task{[&cluster, &update_kernel, keys, hit_mask, &values, &value_sizes, updates, update_stride,
                     update_size, max_value_size, string_namespace_id, entry_size,
                     max_batch_size](const int64_t lo, const int64_t hi) {
      std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> kv_views;
      kv_views.reserve(to_uint(max_batch_size));

      std::vector<char> key_buf(string_namespace_id >= 0 ? to_uint(max_batch_size * entry_size) : 0);

      for (int64_t i0{lo}; i0 < hi; i0 += bitmask64::num_bits) {
        for (auto it{hit_mask[to_uint(i0 / bitmask64::num_bits)]}; it; it = bitmask64::skip(it)) {
          const int64_t ij{i0 + bitmask64::next(it)};

          char* const value{&values[to_uint(ij * max_value_size)]};
          const int64_t value_size{value_sizes[to_uint(ij)]};
          update_kernel(value, value, value_size, &updates[ij * update_stride], update_size);

          kv_views.emplace_back(
              std::piecewise_construct,
              std::forward_as_tuple(write_string_key(key_buf.data(),
                                                     static_cast<int64_t>(kv_views.size()),
                                                     entry_size, string_namespace_id, keys[ij])),
              std::forward_as_tuple(value, value_size));
          if NVE_LIKELY_(kv_views.size() < to_uint(max_batch_size)) continue;

          cluster->mset(kv_views.begin(), kv_views.end());
          kv_views.clear();
        }
      }
      if (!kv_views.empty()) {
        cluster->mset(kv_views.begin(), kv_views.end());
      }
    }};
    run_partitioned<bitmask64::num_bits>(ctx, n, num_parts, config.workgroups, task);
    return;
  }

  if (num_parts == 0) {
    // Single-hash mode: accumulate into the found keys and HSET them back into the shared hash.
    const sw::redis::StringView v_key{config.hash_key};

    std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> kv_views;
    kv_views.reserve(to_uint(max_batch_size));

    for (int64_t i0{}; i0 < n; i0 += bitmask64::num_bits) {
      for (auto it{hit_mask[i0 / bitmask64::num_bits]}; it; it = bitmask64::skip(it)) {
        const int64_t ij{i0 + bitmask64::next(it)};

        char* const value{&values[to_uint(ij * max_value_size)]};
        const int64_t value_size{value_sizes[to_uint(ij)]};
        update_kernel(value, value, value_size, &updates[ij * update_stride], update_size);

        kv_views.emplace_back(
            std::piecewise_construct,
            std::forward_as_tuple(reinterpret_cast<const char*>(&keys[ij]), sizeof(key_type)),
            std::forward_as_tuple(value, value_size));
        if NVE_LIKELY_(kv_views.size() < to_uint(max_batch_size)) continue;

        cluster->hset(v_key, kv_views.begin(), kv_views.end());
        kv_views.clear();
      }
    }
    if (!kv_views.empty()) {
      cluster->hset(v_key, kv_views.begin(), kv_views.end());
    }

    return;
  }

  const int64_t num_parts_mask{num_parts - 1};
  const table_id_t table_id{this->id};

  const auto f{[n, keys, update_stride, update_size, updates, table_id, max_value_size,
                max_batch_size, &update_kernel, &cluster, num_parts_mask, hit_mask, &values,
                &value_sizes](const int64_t task_idx) {
    const HKey v_key{table_id, task_idx, 'v'};

    // TODO: Prone to memory fragmentation. Use scratch buffer instead?
    std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> kv_views;
    kv_views.reserve(to_uint(max_batch_size));

    // Fill up batch and lodge queries as we go.
    for (int64_t i0{}; i0 < n; i0 += bitmask64::num_bits) {
      for (auto it{hit_mask[i0 / bitmask64::num_bits]}; it; it = bitmask64::skip(it)) {
        const int64_t ij{i0 + bitmask64::next(it)};
        const key_type& key{keys[ij]};
        if (partitioner(key, num_parts_mask) != task_idx) continue;

        char* const value{&values[to_uint(ij * max_value_size)]};
        const int64_t value_size{value_sizes[to_uint(ij)]};
        update_kernel(value, value, value_size, &updates[ij * update_stride], update_size);

        kv_views.emplace_back(
            std::piecewise_construct,
            std::forward_as_tuple(reinterpret_cast<const char*>(&key), sizeof(key_type)),
            std::forward_as_tuple(value, value_size));
        if NVE_LIKELY_(kv_views.size() < to_uint(max_batch_size)) continue;

        cluster->hset(v_key, kv_views.begin(), kv_views.end());
        kv_views.clear();
      }
    }
    if (!kv_views.empty()) {
      cluster->hset(v_key, kv_views.begin(), kv_views.end());
    }
  }};

  // TODO: Use more multi-threading by splitting the loop into sub-parts.
  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename KeyType, typename MetaType, typename PartitionerType>
template <bool WithValues, bool WithValueSizes>
int64_t RedisClusterTable<KeyType, MetaType, PartitionerType>::find_(
    context_ptr_t& ctx, const int64_t n, const key_type* const __restrict keys,
    bitmask64_t* const __restrict hit_mask, const int64_t value_stride, char* const __restrict values,
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

    const auto task{[&cluster, &total_num_hits, keys, hit_mask, value_stride, values, value_sizes,
                     string_namespace_id, entry_size, prefixed, max_batch_size](const int64_t lo,
                                                                       const int64_t hi) {
      // Only needed to materialize prefixed keys; raw keys are referenced in place (zero-copy).
      std::vector<char> key_buf(prefixed ? to_uint(max_batch_size * entry_size) : 0);

      std::vector<sw::redis::StringView> k_views;
      k_views.reserve(to_uint(max_batch_size));

      // Prefixed keys live in `key_buf`, so their source index `ij` needs this parallel map;
      // un-prefixed keys are referenced in place and recovered by pointer arithmetic into `keys`.
      std::vector<int64_t> view_to_ij;
      if (prefixed) view_to_ij.reserve(to_uint(max_batch_size));

      int64_t num_hits{};
      const auto callback{
          [keys, hit_mask, value_stride, values, value_sizes, &k_views, &view_to_ij, prefixed, &num_hits](
              const int64_t view_idx, const sw::redis::Optional<sw::redis::StringView>& v_view_opt) {
            if (!v_view_opt) return;
            const sw::redis::StringView& v_view(*v_view_opt);

            const sw::redis::StringView& __restrict k_view{k_views[to_uint(view_idx)]};
            const int64_t ij{prefixed ? view_to_ij[to_uint(view_idx)]
                                       : reinterpret_cast<const key_type*>(k_view.data()) - keys};
            const int64_t value_size{static_cast<int64_t>(v_view.size())};
            if (value_sizes) {
              value_sizes[ij] = value_size;
            }
            if (values) {
              // Only guards the copy below; probe-only callers pass no buffer and no stride.
              NVE_CHECK_(value_size <= value_stride, "The value stored in Redis (=", value_size,
                         ") exceeds the avaiable value stride (=", value_stride, ").");
              std::memcpy(&values[ij * value_stride], v_view.data(), to_uint(value_size));
            }

            ++num_hits;
            const int64_t i{ij / bitmask64::num_bits};
            const int64_t j{ij % bitmask64::num_bits};
            hit_mask[i] |= bitmask64::single(j);
          }};

      for (int64_t i0{lo}; i0 < hi; i0 += bitmask64::num_bits) {
        auto it{bitmask64::clip(~hit_mask[i0 / bitmask64::num_bits], hi - i0)};

        // Run query if the batch is about to overflow.
        if NVE_UNLIKELY_(static_cast<int64_t>(k_views.size()) + bitmask64::count(it) >
                         max_batch_size) {
          cluster->mget(k_views.begin(), k_views.end(), parser_t(callback));
          k_views.clear();
          view_to_ij.clear();
        }

        for (; it; it = bitmask64::skip(it)) {
          const int64_t ij{i0 + bitmask64::next(it)};
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
        cluster->mget(k_views.begin(), k_views.end(), parser_t(callback));
      }

      total_num_hits.fetch_add(num_hits, std::memory_order_relaxed);
    }};

    run_partitioned<bitmask64::num_bits>(ctx, n, num_parts, config.workgroups, task);
    return total_num_hits.load(std::memory_order_relaxed);
  }

  // Single-hash mode: HMGET fields from the shared hash (hash_key). Single-threaded.
  if (num_parts == 0) {
    const sw::redis::StringView v_key{config.hash_key};

    std::vector<sw::redis::StringView> k_views;
    k_views.reserve(to_uint(max_batch_size));

    int64_t num_hits{};
    const auto callback{[keys, hit_mask, value_stride, values, value_sizes, &k_views, &num_hits](
                            const int64_t view_idx,
                            const sw::redis::Optional<sw::redis::StringView>& v_view_opt) {
      if (!v_view_opt) return;
      const sw::redis::StringView& v_view(*v_view_opt);

      const sw::redis::StringView& __restrict k_view{k_views[to_uint(view_idx)]};
      const int64_t ij{reinterpret_cast<const key_type*>(k_view.data()) - keys};

      const int64_t value_size{static_cast<int64_t>(v_view.size())};
      if (value_sizes) {
        value_sizes[ij] = value_size;
      }
      if (values) {
        // Only guards the copy below; probe-only callers pass no buffer and no stride.
        NVE_CHECK_(value_size <= value_stride, "The value stored in Redis (=", value_size,
                   ") exceeds the avaiable value stride (=", value_stride, ").");
        std::memcpy(&values[ij * value_stride], v_view.data(), to_uint(value_size));
      }

      ++num_hits;
      const int64_t i{ij / bitmask64::num_bits};
      const int64_t j{ij % bitmask64::num_bits};
      hit_mask[i] |= bitmask64::single(j);
    }};

    for (int64_t i0{}; i0 < n; i0 += bitmask64::num_bits) {
      auto it{bitmask64::clip(~hit_mask[i0 / bitmask64::num_bits], n - i0)};

      // Run query if the batch is about to overflow.
      if NVE_UNLIKELY_(static_cast<int64_t>(k_views.size()) + bitmask64::count(it) > max_batch_size) {
        cluster->hmget(v_key, k_views.begin(), k_views.end(), parser_t(callback));
        k_views.clear();
      }

      for (; it; it = bitmask64::skip(it)) {
        const key_type& key{keys[i0 + bitmask64::next(it)]};
        k_views.emplace_back(reinterpret_cast<const char*>(&key), sizeof(key_type));
      }
    }
    if (!k_views.empty()) {
      cluster->hmget(v_key, k_views.begin(), k_views.end(), parser_t(callback));
    }

    return num_hits;
  }

  const table_id_t table_id{this->id};
  std::atomic_int64_t total_num_hits{0};
  const auto f{[n, keys, hit_mask, value_stride, values, value_sizes, table_id, max_batch_size, &cluster,
                num_parts, &total_num_hits](const int64_t part_idx) {
    const HKey v_key{table_id, part_idx, 'v'};
    const HKey m_key{table_id, part_idx, 'm'};

    // TODO: Prone to memory fragmentation. Use scratch buffer instead?
    std::vector<sw::redis::StringView> k_views;
    k_views.reserve(to_uint(max_batch_size));

    std::vector<std::pair<sw::redis::StringView, sw::redis::StringView>> meta_km_views;
    if constexpr (std::is_same_v<meta_type, lru_meta_t>) {
      meta_km_views.reserve(to_uint(max_batch_size));
    }
    std::vector<sw::redis::StringView> meta_k_views;
    if constexpr (std::is_same_v<meta_type, lfu_meta_t>) {
      meta_k_views.reserve(to_uint(max_batch_size));
    }
    meta_type meta_value;

    int64_t num_hits{};
    const auto callback{[keys, hit_mask, value_stride, values, value_sizes, &k_views, &meta_value,
                         &meta_km_views, &meta_k_views,
                         &num_hits](const int64_t view_idx,
                                    const sw::redis::Optional<sw::redis::StringView>& v_view_opt) {
      if (!v_view_opt) return;
      const sw::redis::StringView& v_view(*v_view_opt);

      // Reconstruct ij from the key view.
      const sw::redis::StringView& __restrict k_view{k_views[to_uint(view_idx)]};
      const int64_t ij{reinterpret_cast<const key_type*>(k_view.data()) - keys};

      const int64_t value_size{static_cast<int64_t>(v_view.size())};
      if (value_sizes) {
        value_sizes[ij] = value_size;
      }
      if (values) {
        // Only guards the copy below; probe-only callers pass no buffer and no stride.
        NVE_CHECK_(value_size <= value_stride, "The value stored in Redis (=", value_size,
                   ") exceeds the avaiable value stride (=", value_stride, ").");
        std::memcpy(&values[ij * value_stride], v_view.data(), to_uint(value_size));
      }

      if constexpr (std::is_same_v<meta_type, lru_meta_t>) {
        meta_km_views.emplace_back(k_view, view_as_string(meta_value));
        (void)meta_k_views;
      } else if constexpr (std::is_same_v<meta_type, lfu_meta_t>) {
        (void)meta_km_views; (void)meta_value;
        meta_k_views.emplace_back(k_view);
      } else {
        (void)meta_km_views; (void)meta_value;
        (void)meta_k_views;
      }

      ++num_hits;
      const int64_t i{ij / bitmask64::num_bits};
      const int64_t j{ij % bitmask64::num_bits};
      bitmask64::atomic_merge(&hit_mask[i], bitmask64::single(j));
    }};

    const auto process_batch{[&cluster, &v_key, &m_key, &k_views, &meta_value, &meta_km_views,
                              &meta_k_views, &callback]() {
      cluster->hmget(v_key, k_views.begin(), k_views.end(), parser_t(callback));

      meta_value = default_meta_value<meta_type>();
      if constexpr (std::is_same_v<meta_type, lru_meta_t>) {
        if (!meta_km_views.empty()) {
          cluster->hset(m_key, meta_km_views.begin(), meta_km_views.end());
          meta_km_views.clear();
        }
        (void)meta_k_views;
      } else if constexpr (std::is_same_v<meta_type, lfu_meta_t>) {
        if (!meta_k_views.empty()) {
          sw::redis::Pipeline pipe{cluster->pipeline(m_key)};
          for (const sw::redis::StringView& k_view : meta_k_views) {
            pipe.hincrby(m_key, k_view, meta_value);
          }
          pipe.exec();
          meta_k_views.clear();
        }
        (void)meta_km_views;
      } else {
        (void)m_key;
        (void)meta_km_views;
        (void)meta_k_views;
        static_assert(std::is_same_v<meta_type, no_meta_t>, "Overflow handler not implemented.");
      }
    }};

    const int64_t n_aligned{round_up(n, bitmask64::num_bits)};
    const int64_t base{n_aligned / bitmask64::num_bits * part_idx / num_parts * bitmask64::num_bits};
    const int64_t num_parts_mask{num_parts - 1};

    for (int64_t off{}; off < n; off += bitmask64::num_bits) {
      const int64_t i0{(base + off) % n_aligned};

      auto it{bitmask64::atomic_load(&hit_mask[i0 / bitmask64::num_bits])};
      it = bitmask64::clip(~it, n - i0);

      // Run query if the batch is about to overflow.
      if NVE_UNLIKELY_(static_cast<int64_t>(k_views.size()) + bitmask64::count(it) > max_batch_size) {
        process_batch();
        k_views.clear();
      }

      for (; it; it = bitmask64::skip(it)) {
        const key_type& key{keys[i0 + bitmask64::next(it)]};
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
  return total_num_hits.load(std::memory_order_relaxed);
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
  pool_opts.size = to_uint(config.connections_per_node);

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

template <typename KeyType, typename MetaType>
static host_table_ptr_t make_redis_cluster_table_3(table_id_t id,
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
    return std::make_shared<RedisClusterTable<KeyType, MetaType, AlwaysZeroPartitioner>>(id, config, conn);
  }

  switch (config.partitioner) {
#if defined(NVE_FEATURE_HT_PART_FNV1A)
    case Partitioner_t::FowlerNollVo:
      return std::make_shared<RedisClusterTable<KeyType, MetaType, FowlerNollVoPartitioner>>(id, config, conn);
#endif
#if defined(NVE_FEATURE_HT_PART_MURMUR3)
    case Partitioner_t::Murmur3:
      return std::make_shared<RedisClusterTable<KeyType, MetaType, Murmur3Partitioner>>(id, config, conn);
#endif
#if defined(NVE_FEATURE_HT_PART_RRXMRRXMSX0)
    case Partitioner_t::Rrxmrrxmsx0:
      return std::make_shared<RedisClusterTable<KeyType, MetaType, Rrxmrrxmsx0Partitioner>>(id, config, conn);
#endif
#if defined(NVE_FEATURE_HT_PART_STD_HASH)
    case Partitioner_t::StdHash:
      return std::make_shared<RedisClusterTable<KeyType, MetaType, StdHashPartitioner>>(id, config, conn);
#endif
    default:
      NVE_THROW_("`config.partitioner` (", config.partitioner, ") is out of bounds!");
  }
}

template <typename KeyType>
static host_table_ptr_t make_redis_cluster_table_2(const table_id_t id,
                                                   const RedisClusterTableConfig& config,
                                                   redis_conn_ptr_t& conn,
                                                   const bool string_mode) {
  switch (config.overflow_policy.handler) {
    case OverflowHandler_t::EvictRandom:
      return make_redis_cluster_table_3<KeyType, no_meta_t>(id, config, conn, string_mode);
    case OverflowHandler_t::EvictLRU:
      return make_redis_cluster_table_3<KeyType, lru_meta_t>(id, config, conn, string_mode);
    case OverflowHandler_t::EvictLFU:
      return make_redis_cluster_table_3<KeyType, lfu_meta_t>(id, config, conn, string_mode);
  }
  NVE_THROW_("`config.overflow_policy.handler` (", config.overflow_policy.handler,
             ") is out of bounds!");
}

static host_table_ptr_t make_redis_cluster_table_1(table_id_t id,
                                                   const RedisClusterTableConfig& config,
                                                   redis_conn_ptr_t& conn,
                                                   const bool string_mode) {
  switch (config.key_size) {
#if defined(NVE_FEATURE_HT_KEY_8)
    case sizeof(int8_t):
      return make_redis_cluster_table_2<int8_t>(id, config, conn, string_mode);
#endif
#if defined(NVE_FEATURE_HT_KEY_16)
    case sizeof(int16_t):
      return make_redis_cluster_table_2<int16_t>(id, config, conn, string_mode);
#endif
#if defined(NVE_FEATURE_HT_KEY_32)
    case sizeof(int32_t):
      return make_redis_cluster_table_2<int32_t>(id, config, conn, string_mode);
#endif
#if defined(NVE_FEATURE_HT_KEY_64)
    case sizeof(int64_t):
      return make_redis_cluster_table_2<int64_t>(id, config, conn, string_mode);
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

  return make_redis_cluster_table_1(id, config, conn_, string_mode);
}

}  // namespace plugin
}  // namespace nve
