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

// Template member definitions of NvhmMapTable. Included only by the generated
// nvhm_map_table_inst_*.cpp translation units (see cmake/gen_nvhm_map_table_instantiations.py),
// which explicitly instantiate the class so the instantiation matrix compiles across many TUs
// instead of one. Keep the generator's combination space in sync with the factory dispatch chain
// in nvhm_map_table.cpp.

#pragma once

#include <cstring>
#include <random>
#include <host_table_detail.hpp>
#include <nvhm_map_table.hpp>
#include <buffer_wrapper.hpp>
#include <thread_pool.hpp>

namespace nve {
namespace plugin {

constexpr int64_t max_prefetch_size{8 * cpu_cache_line_size};

template <typename MapType, typename PartitionerType>
NvhmMapTable<MapType, PartitionerType>::NvhmMapTable(
    const table_id_t id, const NvhmMapTableConfig& config)
    : base_type(id, config), parts_(to_uint(config.num_partitions)) {
  NVE_CHECK_(config.key_size == sizeof(key_type));

  const int64_t max_value_size{config.max_value_size};
  conf_type conf;
  conf.set_capacity(config.initial_capacity);
  conf.set_blob(max_value_size, round_up(max_value_size, config.value_alignment));
  conf.auto_adjust();
  for (auto& part : parts_) {
    part.map = map_type(conf);
  }
}

template <typename MapType, typename PartitionerType>
void NvhmMapTable<MapType, PartitionerType>::clear(context_ptr_t& ctx) {
  const auto& __restrict config{config_};

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};

  const auto f{[&parts](const int64_t task_idx) {
    Partition& __restrict part{parts[to_uint(task_idx)]};
    std::lock_guard lock(part.read_write);

    part.map.clear();
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename MapType, typename PartitionerType>
void NvhmMapTable<MapType, PartitionerType>::erase(
    context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw) {
  if (n <= 0) return;
  auto modify_stream{ctx->get_modify_stream()};

  const key_type* keys{nullptr};
  if (keys_bw) {
    auto p{keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream)};
    keys = static_cast<const key_type*>(p);
  }

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};
  const int64_t num_parts_mask{num_parts - 1};

  const auto f{[n, keys, &parts, num_parts_mask](const int64_t task_idx) {
    Partition& __restrict part{parts[to_uint(task_idx)]};
    std::lock_guard lock(part.read_write);

    for (int64_t ij{}; ij < n; ++ij) {
      const key_type key{keys[ij]};
      if (partitioner(key, num_parts_mask) != task_idx) continue;

      part.map.erase(key);
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config_.workgroups, 1);
}

template <typename MapType, typename PartitionerType>
void NvhmMapTable<MapType, PartitionerType>::find(
    context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys_bw,
    buffer_ptr<bitmask64_t> hit_mask_bw, const int64_t value_stride,
    buffer_ptr<void> values_bw, buffer_ptr<int64_t> value_sizes_bw) const {
  if (n <= 0) return;
  auto lookup_stream{ctx->get_lookup_stream()};

  const key_type* keys{nullptr};
  if (keys_bw) {
    auto p{keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, lookup_stream)};
    keys = static_cast<const key_type*>(p);
  }
  bitmask64_t* hit_mask;
  if (hit_mask_bw) {
    hit_mask = hit_mask_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, lookup_stream);
  } else {
    const uint64_t size{to_uint(ceil_div(n, bitmask64::num_bits)) * sizeof(bitmask64_t)};
    auto p{ctx->get_buffer("hitmask", size, true /*host_alloc*/)};
    std::memset(p, 0, size);
    hit_mask = static_cast<bitmask64_t*>(p);
  }
  std::byte* values{nullptr};
  if (values_bw) {
    auto p{values_bw->access_buffer(cudaMemoryTypeUnregistered, false, lookup_stream)};
    values = static_cast<std::byte*>(p);
  }
  int64_t* value_sizes{nullptr};
  if (value_sizes_bw) {
    value_sizes = value_sizes_bw->access_buffer(cudaMemoryTypeUnregistered, false, lookup_stream);
  }

  if (config_.prefetch_values) {
    switch (config_.key_fetch_queue_length) {
      case 0:
        n = find_<0, true>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
        break;
      case 1:
        n = find_<1, true>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
        break;
      case 2:
        n = find_<2, true>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
        break;
      case 4:
        n = find_<4, true>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
        break;
      case 8:
        n = find_<8, true>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
        break;
      default:
        NVE_THROW_("`key_fetch_queue_length` (", config_.key_fetch_queue_length, ") is out of bounds!");
    }
  } else {
    switch (config_.key_fetch_queue_length) {
      case 0:
        n = find_<0, false>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
        break;
      case 1:
        n = find_<1, false>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
        break;
      case 2:
        n = find_<2, false>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
        break;
      case 4:
        n = find_<4, false>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
        break;
      case 8:
        n = find_<8, false>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
        break;
      default:
        NVE_THROW_("`key_fetch_queue_length` (", config_.key_fetch_queue_length, ") is out of bounds!");
    }
  }

  auto counter = this->lookup_counter_storage(ctx);
  NVE_CHECK_(counter != nullptr, "Invalid key counter");
  *counter += n;
}

template <typename MapType, typename PartitionerType>
void NvhmMapTable<MapType, PartitionerType>::insert(
    context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw,
    const int64_t value_stride, const int64_t value_size, buffer_ptr<const void> values_bw) {
  if (n <= 0) return;
  auto modify_stream{ctx->get_modify_stream()};

  const key_type* __restrict keys{nullptr};
  if (keys_bw) {
    auto p{keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream)};
    keys = static_cast<const key_type*>(p);
  }
  const std::byte* __restrict values{nullptr};
  if (values_bw) {
    auto p{values_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream)};
    values = static_cast<const std::byte*>(p);
  }

  // TODO: Allow dynamically sized vectors.
  NVE_CHECK_(value_size >= 0 && value_size <= config_.max_value_size);
  switch (config_.insert_fetch_queue_length) {
    case 0:
      insert_<0>(ctx, n, keys, value_stride, value_size, values);
      break;
    case 1:
      insert_<1>(ctx, n, keys, value_stride, value_size, values);
      break;
    case 2:
      insert_<2>(ctx, n, keys, value_stride, value_size, values);
      break;
    case 4:
      insert_<4>(ctx, n, keys, value_stride, value_size, values);
      break;
    case 8:
      insert_<8>(ctx, n, keys, value_stride, value_size, values);
      break;
    default:
      NVE_THROW_("`insert_fetch_queue_length` (", config_.insert_fetch_queue_length, ") is out of bounds!");
  }
}

template <typename MapType, typename PartitionerType>
template <int64_t PrefetchQueueLength>
void NvhmMapTable<MapType, PartitionerType>::insert_(
    context_ptr_t& ctx, const int64_t n, const key_type* __restrict keys,
    const int64_t value_stride, const int64_t value_size, const std::byte* __restrict values) {
  const int64_t overflow_margin{config_.overflow_policy.overflow_margin};
  const int64_t resolution_margin{config_.overflow_policy.abs_resolution_margin()};

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};
  const int64_t num_parts_mask{num_parts - 1};

  const table_id_t table_id{this->id};
  const auto f{[n, keys, value_stride, value_size, values, overflow_margin,
                resolution_margin, &parts, num_parts_mask, table_id](const int64_t task_idx) {
    Partition& __restrict part{parts[to_uint(task_idx)]};
    std::lock_guard lock(part.read_write);
    map_type& __restrict map{part.map};

    const meta_type meta_value{default_meta_value<meta_type>()};
    int64_t part_size{static_cast<int64_t>(map.size())};

    // Stores row `ij` under `key`; `hint` is the optional result of a preceding `write_prefetch`.
    const auto store{[&](const key_type& key, const int64_t ij, const auto&... hint) {
      const auto [pos, op]{map.insert(key, hint...)};
      map.set_blob_at(pos, &values[ij * value_stride], value_size);

      if (op == nvhm::insert_op_t::found) return;
      ++part_size;

      if constexpr (!std::is_same_v<meta_type, no_meta_t>) {
        map.value_at(pos) = meta_value;
      }

      // Handle overflows.
      if NVE_LIKELY_(part_size < overflow_margin) return;
      NVE_LOG_VERBOSE_("NV map table ", table_id, " part ", task_idx,
                       " is overflowing. Attempting to resolve...");

      if constexpr (std::is_same_v<meta_type, no_meta_t>) {
        // Fetch all keys and shuffle them randomly.
        std::vector<key_type> all_keys(map.keys());
        std::shuffle(all_keys.begin(), all_keys.end(), std::default_random_engine{random_seed()});

        // Reclaim slots above the resolution margin.
        for (auto it{all_keys.begin() + resolution_margin}; it < all_keys.end(); ++it) {
          map.erase(*it);
        }
      } else {
        // Fetch all keys and sort them by the DESCENDING by the associated meta-values.
        std::vector<std::pair<key_type, meta_type>> keys_metas(map.keys_and_values());
        std::sort(keys_metas.begin(), keys_metas.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        // Reclaim slots above the resolution margin.
        for (auto it{keys_metas.begin() + resolution_margin}; it < keys_metas.end(); ++it) {
          map.erase(it->first);
        }

        // Realign remaining frequency values to avoid entering a steady state.
        if constexpr (std::is_same_v<meta_type, lfu_meta_t>) {
          map.for_each_value([](meta_type& meta_value) { meta_value >>= 1; });
        }
      }

      // Update part_size after erasing.
      part_size = static_cast<int64_t>(map.size());
    }};

    if constexpr (PrefetchQueueLength == 0) {
      for (int64_t ij{}; ij < n; ++ij) {
        const key_type key{keys[ij]};
        if (partitioner(key, num_parts_mask) != task_idx) continue;
        store(key, ij);
      }
    } else {
      // Software pipeline: a key's bucket is prefetched when the key enters the ring and the key is
      // inserted PrefetchQueueLength keys later, so several cache misses are in flight at once.
      // `queued_ij` mirrors the ring to keep each key's row index; `head` tracks the oldest entry.
      nvhm::ring_prefetch_queue<key_type, prefetch_hint_type, PrefetchQueueLength> queue{};
      int64_t queued_ij[PrefetchQueueLength]{};
      int64_t head{};
      constexpr int64_t ring_mask{PrefetchQueueLength - 1};

      int64_t ij{};
      for (; ij < n && !queue.full(); ++ij) {
        const key_type key{keys[ij]};
        if (partitioner(key, num_parts_mask) != task_idx) continue;
        queued_ij[queue.size()] = ij;
        queue.prefill_write(map, key);
      }
      for (; ij < n; ++ij) {
        const key_type key{keys[ij]};
        if (partitioner(key, num_parts_mask) != task_idx) continue;
        const int64_t slot{head++ & ring_mask};
        const int64_t qij{queued_ij[slot]};
        queued_ij[slot] = ij;
        const auto [qkey, qhint]{queue.push_write(map, key)};
        store(qkey, qij, qhint);
      }
      while (!queue.empty()) {
        const int64_t slot{head++ & ring_mask};
        const auto [qkey, qhint]{queue.pop()};
        store(qkey, queued_ij[slot], qhint);
      }
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config_.workgroups, 1);
}

template <typename MapType, typename PartitionerType>
int64_t NvhmMapTable<MapType, PartitionerType>::size(context_ptr_t& ctx,
                                                                    const bool) const {
  const std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};

  std::atomic_int64_t total_n{0};
  const auto f{[&parts, &total_n](const int64_t task_idx) {
    int64_t n;
    {
      const Partition& __restrict part{parts[to_uint(task_idx)]};
      std::shared_lock lock(part.read_write);

      n = static_cast<int64_t>(part.map.size());
    }
    total_n.fetch_add(n, std::memory_order_relaxed);
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config_.workgroups, 1);
  return total_n.load(std::memory_order_relaxed);
}

template <typename MapType, typename PartitionerType>
void NvhmMapTable<MapType, PartitionerType>::update(
    context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw,
    const int64_t value_stride, const int64_t value_size, buffer_ptr<const void> values_bw) {
  if (n <= 0) return;
  auto modify_stream{ctx->get_modify_stream()};

  const key_type* __restrict keys{nullptr};
  if (keys_bw) {
    auto p{keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream)};
    keys = static_cast<const key_type*>(p);
  }
  const std::byte* __restrict values{nullptr};
  if (values_bw) {
    auto p{values_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream)};
    values = static_cast<const std::byte*>(p);
  }

  NVE_CHECK_(value_size >= 0 && value_size <= config_.max_value_size);

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};
  const int64_t num_parts_mask{num_parts - 1};

  const auto f{[n, keys, value_stride, value_size, values, &parts, num_parts_mask](const int64_t task_idx) {
    Partition& __restrict part{parts[to_uint(task_idx)]};
    std::lock_guard lock(part.read_write);
    map_type& __restrict map{part.map};

    for (int64_t ij{}; ij < n; ++ij) {
      const key_type key{keys[ij]};
      if (partitioner(key, num_parts_mask) != task_idx) continue;

      const write_pos_type pos{map.update(key)};
      if (pos == nvhm::npos) continue;

      map.set_blob_at(pos, &values[ij * value_stride], value_size);
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config_.workgroups, 1);
}

template <typename MapType, typename PartitionerType>
void NvhmMapTable<MapType, PartitionerType>::update_accumulate(
    context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw,
    const int64_t update_stride, const int64_t update_size, buffer_ptr<const void> updates_bw,
    const DataType_t update_dtype) {
  if (n <= 0) return;
  auto modify_stream{ctx->get_modify_stream()};

  const key_type* __restrict keys{nullptr};
  if (keys_bw) {
    auto p{keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream)};
    keys = static_cast<const key_type*>(p);
  }
  const std::byte* __restrict updates{nullptr};
  if (updates_bw) {
    auto p{updates_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream)};
    updates = static_cast<const std::byte*>(p);
  }

  const int64_t max_value_size{config_.max_value_size};
  NVE_CHECK_(update_size >= 0 && update_size % dtype_size(update_dtype) == 0);
  NVE_CHECK_(update_size / dtype_size(update_dtype) <= max_value_size / config_.value_dtype_size());
  const update_kernel_t update_kernel{pick_cpu_update_kernel(config_.value_dtype, update_dtype)};

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};
  const int64_t num_parts_mask{num_parts - 1};

  const auto f{[n, keys, update_stride, update_size, updates, max_value_size, &update_kernel, &parts, num_parts_mask](const int64_t task_idx) {
    Partition& __restrict part{parts[to_uint(task_idx)]};
    std::lock_guard lock(part.read_write);
    map_type& __restrict map{part.map};

    for (int64_t ij{}; ij < n; ++ij) {
      const key_type key{keys[ij]};
      if (partitioner(key, num_parts_mask) != task_idx) continue;

      const write_pos_type pos{map.update(key)};
      if (pos == nvhm::npos) continue;

      // TODO: Support variable length vector sizes.
      const int64_t value_size{max_value_size};
      std::byte* const blob{map.blob_at(pos)};
      update_kernel(blob, blob, value_size, &updates[ij * update_stride], update_size);
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config_.workgroups, 1);
}

template <typename MapType, typename PartitionerType>
template <int64_t PrefetchQueueLength, bool PrefetchValues>
int64_t NvhmMapTable<MapType, PartitionerType>::find_(
  context_ptr_t& ctx, const int64_t n, const key_type* const __restrict keys,
  bitmask64_t* const __restrict hit_mask, const int64_t value_stride, std::byte* const __restrict values,
  int64_t* const __restrict value_sizes) const {
  if (values) {
    if (value_sizes) {
      return find_<PrefetchQueueLength, PrefetchValues, true, true>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
    } else {
      return find_<PrefetchQueueLength, PrefetchValues, true, false>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
    }
  } else {
    if (value_sizes) {
      return find_<PrefetchQueueLength, PrefetchValues, false, true>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
    } else {
      return find_<PrefetchQueueLength, PrefetchValues, false, false>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
    }
  }
}

template <typename MapType, typename PartitionerType>
template <int64_t PrefetchQueueLength, bool PrefetchValues, bool WithValues, bool WithValueSizes>
int64_t NvhmMapTable<MapType, PartitionerType>::find_(
    context_ptr_t& ctx, const int64_t n, const key_type* const __restrict keys,
    bitmask64_t* const __restrict hit_mask, const int64_t value_stride, std::byte* const __restrict values,
    int64_t* const __restrict value_sizes) const {
  const int64_t num_masks{ceil_div(n, bitmask64::num_bits)};

  const int64_t max_value_size{config_.max_value_size};
  NVE_CHECK_(value_stride >= max_value_size);
  const int64_t max_task_size{config_.max_find_task_size};

  const std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};
  const int64_t num_parts_mask{num_parts - 1};

  const int64_t num_tasks_per_part{ceil_div(num_masks, max_task_size)};
  const int64_t num_tasks{num_parts * num_tasks_per_part};

  std::atomic_int64_t total_num_hits{0};
  const auto f{[n, keys, hit_mask, value_stride, values, value_sizes, num_masks,
                max_value_size, max_task_size, &parts, num_parts, num_parts_mask,
                num_tasks_per_part, &total_num_hits](int64_t task_idx) {
    int64_t num_hits{};

    const int64_t part_idx{task_idx / num_tasks_per_part};
    task_idx %= num_tasks_per_part;

    // Scatter parts accross input domain.
    int64_t i_base{max_task_size * task_idx};
    const int64_t task_size{std::min(num_masks - i_base, max_task_size)};
    i_base += num_masks * part_idx / num_parts;

    {
      const Partition& __restrict part{parts[to_uint(part_idx)]};
      std::shared_lock lock(part.read_write);
      const map_type& __restrict map{part.map};

      const std::byte* __restrict prev_src{};
      std::byte* __restrict prev_dst{};
      int64_t prev_value_size{max_value_size};
      const meta_type meta_update{default_meta_value<meta_type>()};

      const auto process_next{[&](const int64_t j, const int64_t ij, read_pos_type&& pos) {
        // TODO: Support variable length vector sizes.
        const int64_t value_size{max_value_size};

        if constexpr (WithValues) {
          const std::byte* const __restrict src{map.blob_at(pos)};
          std::byte* const __restrict dst{&values[ij * value_stride]};
          if constexpr (PrefetchValues) {
            l1_prefetch(src, dst, std::min(value_size, max_prefetch_size));
            if (prev_src) {
              std::memcpy(prev_dst, prev_src, to_uint(prev_value_size));
            }
            prev_src = src;
            prev_dst = dst;
            prev_value_size = value_size;
          } else {
            std::memcpy(dst, src, to_uint(value_size));
          }
        } else {
          (void)value_stride;
          (void)values;
        }

        // This may data-race with multiple threads. But losing some meta-updates is fine during find.
        if constexpr (!std::is_same_v<meta_type, no_meta_t>) {
          meta_type& __restrict meta_value{const_cast<meta_type&>(map.value_at(pos))};
          if constexpr (std::is_same_v<meta_type, lru_meta_t>) {
            meta_value = meta_update;
          } else if constexpr (std::is_same_v<meta_type, lfu_meta_t>) {
            if (meta_value < max_lfu_meta) meta_value += meta_update;
          }
        } else {
          (void)meta_update;
        }

        if constexpr (WithValueSizes) {
          value_sizes[ij] = value_size;
        } else {
          (void)value_sizes;
        }

        return bitmask64::single(j);
      }};

      for (int64_t i_off{}; i_off < task_size; ++i_off) {
        const int64_t i{(i_base + i_off) % num_masks};
        const int64_t i0{i * bitmask64::num_bits};

        bitmask64_t mask{bitmask64::atomic_load(&hit_mask[i])};
        num_hits -= bitmask64::count(mask);

        auto it{bitmask64::clip(~mask, n - i0)};
        if constexpr (PrefetchQueueLength) {
          nvhm::ring_prefetch_queue<key_type, prefetch_hint_type, PrefetchQueueLength> queue;
          bitmask64_t it_queue{};

          while (it) {
            const int64_t j{bitmask64::next(it)};
            const int64_t ij{i0 + j};
            it = bitmask64::skip(it);

            key_type key{keys[ij]};
            if NVE_LIKELY_(partitioner(key, num_parts_mask) != part_idx) continue;
            it_queue |= bitmask64::single(j);

            queue.prefill_read(map, std::move(key));
            if (queue.full()) break;
          }

          for (; it; it = bitmask64::skip(it)) {
            int64_t j{bitmask64::next(it)};
            int64_t ij{i0 + j};

            key_type key{keys[ij]};
            if NVE_LIKELY_(partitioner(key, num_parts_mask) != part_idx) continue;
            it_queue |= bitmask64::single(j);

            auto [qkey, qhint]{queue.push_read(map, std::move(key))};
            read_pos_type pos{map.find(std::move(qkey), std::move(qhint))};
            if (pos != nvhm::npos) {
              j = bitmask64::next(it_queue);
              ij = i0 + j;
              mask |= process_next(j, ij, std::move(pos));
            }
            it_queue = bitmask64::skip(it_queue);
          }

          for (; it_queue; it_queue = bitmask64::skip(it_queue)) {
            auto [qkey, qhint]{queue.pop()};
            read_pos_type pos{map.find(std::move(qkey), std::move(qhint))};
            if (pos != nvhm::npos) {
              const int64_t j{bitmask64::next(it_queue)};
              const int64_t ij{i0 + j};
              mask |= process_next(j, ij, std::move(pos));
            }
          }
          NVHM_ASSERT_(queue.empty());
        } else {
          for (; it; it = bitmask64::skip(it)) {
            const int64_t j{bitmask64::next(it)};
            const int64_t ij{i0 + j};

            key_type key{keys[ij]};
            if NVE_LIKELY_(partitioner(key, num_parts_mask) != part_idx) continue;

            read_pos_type pos{map.find(std::move(key))};
            if (pos != nvhm::npos) {
              mask |= process_next(j, ij, std::move(pos));
            }
          }
        }

        num_hits += bitmask64::count(mask);
        bitmask64::atomic_merge(&hit_mask[i], mask);
      }

      if constexpr (WithValues && PrefetchValues) {
        if (prev_src) {
          std::memcpy(prev_dst, prev_src, to_uint(prev_value_size));
        }
      }
    }

    total_num_hits.fetch_add(num_hits, std::memory_order_relaxed);
  }};

  ctx->get_thread_pool()->execute_n(0, num_tasks, f, config_.workgroups, num_tasks_per_part);
  return total_num_hits.load(std::memory_order_relaxed);
}

}  // namespace plugin
}  // namespace nve
