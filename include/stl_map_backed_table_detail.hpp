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

#include <random>
#include <host_table_detail.hpp>
#include <stl_map_backed_table.hpp>
#include <buffer_wrapper.hpp>
#include <thread_pool.hpp>

namespace nve {

constexpr int64_t max_prefetch_size{8 * cpu_cache_line_size};

template <typename ConfigType, typename KeyType, typename MetaType,
          typename PartitionerType>
STLContainerTable<ConfigType, KeyType, MetaType, PartitionerType>::STLContainerTable(
    const table_id_t id, const config_type& config)
    : base_type(id, config), parts_(to_uint(config.num_partitions)) {
  NVE_CHECK_(config.key_size == sizeof(key_type));
}

template <typename ConfigType, typename KeyType, typename MetaType,
          typename PartitionerType>
void STLContainerTable<ConfigType, KeyType, MetaType, PartitionerType>::clear(
    context_ptr_t& ctx) {

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};

  const auto f{[&parts](const int64_t task_idx) {
    Partition& __restrict part{parts[to_uint(task_idx)]};
    std::lock_guard lock(part.read_write);

    part.slot_map.clear();
    part.available_slots.clear();
    part.slot_buffers.clear();
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config_.workgroups, 1);
}

template <typename ConfigType, typename KeyType, typename MetaType,
          typename PartitionerType>
void STLContainerTable<ConfigType, KeyType, MetaType, PartitionerType>::erase(
    context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw) {
  if (n <= 0) return;
  auto modify_stream{ctx->get_modify_stream()};

  NVE_CHECK_ARG_(keys_bw != nullptr, "keys must not be null");
  const key_type* __restrict keys{static_cast<const key_type*>(
      keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream))};

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};
  const int64_t num_parts_mask{num_parts - 1};

  const auto f{[n, keys, &parts, num_parts_mask](const int64_t task_idx) {
    Partition& __restrict part{parts[static_cast<uint64_t>(task_idx)]};
    std::lock_guard lock(part.read_write);

    for (int64_t ij{}; ij < n; ++ij) {
      const key_type key{keys[ij]};
      if (partitioner(key, num_parts_mask) != task_idx) continue;

      const auto pos{part.slot_map.find(key)};
      if (pos == part.slot_map.end()) continue;

      // Protected by part.read_write above; Coverity merges lock evidence across template instantiations.
      // coverity[MISSING_LOCK]
      part.available_slots.emplace_back(pos->second);
      part.slot_map.erase(pos);
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config_.workgroups, 1);
}

template <typename ConfigType, typename KeyType, typename MetaType,
          typename PartitionerType>
void STLContainerTable<ConfigType, KeyType, MetaType, PartitionerType>::find(
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
    if (values) {
      if (value_sizes) {
        n = find_<true, true, true>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
      } else {
        n = find_<true, true, false>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
      }
    } else {
      if (value_sizes) {
        n = find_<true, false, true>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
      } else {
        n = find_<true, false, false>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
      }
    }
  } else {
    if (values) {
      if (value_sizes) {
        n = find_<false, true, true>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
      } else {
        n = find_<false, true, false>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
      }
    } else {
      if (value_sizes) {
        n = find_<false, false, true>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
      } else {
        n = find_<false, false, false>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
      }
    }
  }

  auto counter = this->lookup_counter_storage(ctx);
  NVE_CHECK_(counter != nullptr, "Invalid key counter");
  *counter += n;
}

template <typename ConfigType, typename KeyType, typename MetaType,
          typename PartitionerType>
void STLContainerTable<ConfigType, KeyType, MetaType, PartitionerType>::insert(
    context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw,
    const int64_t value_stride, const int64_t value_size, buffer_ptr<const void> values_bw) {
  if (n <= 0) return;
  auto modify_stream{ctx->get_modify_stream()};

  NVE_CHECK_ARG_(keys_bw != nullptr, "keys must not be null");
  const key_type* __restrict keys{static_cast<const key_type*>(
      keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream))};
  const std::byte* __restrict values{nullptr};
  if (values_bw) {
    auto p{values_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream)};
    values = static_cast<const std::byte*>(p);
  }

  // TODO: Allow dynamically sized vectors.
  NVE_CHECK_(value_size >= 0 && value_size <= config_.max_value_size);
  const int64_t meta_offset{config_.meta_offset()};
  const int64_t slot_stride{config_.slot_stride()};
  const int64_t slots_per_buffer{config_.allocation_rate / slot_stride};
  const int64_t overflow_margin{config_.overflow_policy.overflow_margin};
  const int64_t resolution_margin{config_.overflow_policy.abs_resolution_margin()};

  const auto get_meta_value{[meta_offset](std::byte* __restrict slot) -> meta_type& {
    return *reinterpret_cast<meta_type*>(&slot[meta_offset]);
  }};

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};
  const int64_t num_parts_mask{num_parts - 1};

  const table_id_t table_id{this->id};
  const auto f{[n, keys, value_stride, value_size, values, slot_stride, slots_per_buffer,
                overflow_margin, resolution_margin, &get_meta_value, &parts, num_parts_mask,
                table_id](const int64_t task_idx) {
    Partition& __restrict part{parts[to_uint(task_idx)]};
    std::lock_guard lock(part.read_write);

    const meta_type meta_value{default_meta_value<meta_type>()};

    for (int64_t ij{}; ij < n; ++ij) {
      const key_type key{keys[ij]};
      if (partitioner(key, num_parts_mask) != task_idx) continue;

      {
        const auto pos{part.slot_map.find(key)};
        if (pos != part.slot_map.end()) {
          std::memcpy(pos->second, &values[ij * value_stride], to_uint(value_size));
          continue;
        }
      }

      // Handle overflows.
      if NVE_UNLIKELY_(part.slot_map.size() >= to_uint(overflow_margin)) {
        NVE_LOG_VERBOSE_("STL table ", table_id, " part ", task_idx,
                         " is overflowing. Attempting to resolve...");

        // Fetch all key/slot pairs.
        std::vector<std::pair<key_type, std::byte*>> keys_slots(
          part.slot_map.begin(), part.slot_map.end());

        if constexpr (std::is_same_v<meta_type, no_meta_t>) {
          // Randomly shuffle slots.
          std::shuffle(keys_slots.begin(), keys_slots.end(),
                       std::default_random_engine{random_seed()});
          (void)get_meta_value;
        } else {
          // Sort DESCENDING by the associated meta-values.
          std::sort(keys_slots.begin(), keys_slots.end(),
                    [&](const auto& a, const auto& b) {
                      return get_meta_value(a.second) > get_meta_value(b.second);
                    });
        }

        // Evict keys and re-claim slots above the resolution margin.
        for (auto it{keys_slots.begin() + resolution_margin}; it < keys_slots.end(); ++it) {
          part.available_slots.emplace_back(it->second);
          part.slot_map.erase(it->first);
        }

        // Realign remaining frequency values to avoid entering a steady state.
        if constexpr (std::is_same_v<meta_type, lfu_meta_t>) {
          for (std::pair<const key_type, std::byte*>& pair : part.slot_map) {
            get_meta_value(pair.second) >>= 1;
          }
        }
      }

      // Allocate new slot memory if there are no slots available.
      if NVE_UNLIKELY_(part.available_slots.empty()) {
        int64_t off{slots_per_buffer * slot_stride};
        std::vector<std::byte>& __restrict slot_buffer{
          part.slot_buffers.emplace_back(to_uint(off))};

        part.available_slots.reserve(to_uint(slots_per_buffer));
        while (off) {
          off -= slot_stride;
          part.available_slots.emplace_back(&slot_buffer[to_uint(off)]);
        }
      }

      std::byte* const __restrict slot{part.available_slots.back()};
      part.available_slots.pop_back();

      std::memcpy(slot, &values[ij * value_stride], to_uint(value_size));

      if constexpr (!std::is_same_v<meta_type, no_meta_t>) {
        get_meta_value(slot) = meta_value;
      }

      part.slot_map.emplace(key, slot);
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config_.workgroups, 1);
}

template <typename ConfigType, typename KeyType, typename MetaType,
          typename PartitionerType>
int64_t STLContainerTable<ConfigType, KeyType, MetaType, PartitionerType>::size(
    context_ptr_t& ctx) const {

  const std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};

  std::atomic_int64_t total_n{0};
  const auto f{[&parts, &total_n](const int64_t task_idx) {
    int64_t n;
    {
      const Partition& __restrict part{parts[to_uint(task_idx)]};
      std::shared_lock lock(part.read_write);

      n = static_cast<int64_t>(part.slot_map.size());
    }
    total_n.fetch_add(n, std::memory_order_relaxed);
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config_.workgroups, 1);
  return total_n.load(std::memory_order_relaxed);
}

template <typename ConfigType, typename KeyType, typename MetaType,
          typename PartitionerType>
int64_t STLContainerTable<ConfigType, KeyType, MetaType, PartitionerType>::get_keys(
    context_ptr_t& ctx, const int64_t capacity, void* const keys_ptr) const {
  NVE_CHECK_(capacity >= 0, "capacity must not be negative");
  key_type* const __restrict keys{static_cast<key_type*>(keys_ptr)};

  const std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};

  // Each partition claims the next contiguous output range; ranges past `capacity` are dropped.
  std::atomic_int64_t claimed{0};
  const auto f{[keys, capacity, &parts, &claimed](const int64_t task_idx) {
    const Partition& __restrict part{parts[to_uint(task_idx)]};
    std::shared_lock lock(part.read_write);
    const map_type& __restrict slot_map{part.slot_map};

    const auto [begin, room]{claim_output_range(claimed, static_cast<int64_t>(slot_map.size()), capacity)};
    auto it{slot_map.begin()};
    for (int64_t i{}; i < room; ++i, ++it) {
      keys[begin + i] = it->first;
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config_.workgroups, 1);
  return std::min(claimed.load(std::memory_order_relaxed), capacity);
}

template <typename ConfigType, typename KeyType, typename MetaType,
          typename PartitionerType>
void STLContainerTable<ConfigType, KeyType, MetaType, PartitionerType>::update(
    context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw,
    const int64_t value_stride, const int64_t value_size, buffer_ptr<const void> values_bw) {
  if (n <= 0) return;
  auto modify_stream{ctx->get_modify_stream()};

  NVE_CHECK_ARG_(keys_bw != nullptr, "keys must not be null");
  const key_type* __restrict keys{static_cast<const key_type*>(
      keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream))};
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

    const auto slot_map_end{part.slot_map.end()};

    for (int64_t ij{}; ij < n; ++ij) {
      const key_type key{keys[ij]};
      if (partitioner(key, num_parts_mask) != task_idx) continue;

      const auto pos{part.slot_map.find(key)};
      if (pos == slot_map_end) continue;

      // Protected by part.read_write above; Coverity merges lock evidence across template instantiations.
      // coverity[MISSING_LOCK]
      std::memcpy(pos->second, &values[ij * value_stride], to_uint(value_size));
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config_.workgroups, 1);
}

template <typename ConfigType, typename KeyType, typename MetaType,
          typename PartitionerType>
void STLContainerTable<ConfigType, KeyType, MetaType, PartitionerType>::update_accumulate(
    context_ptr_t& ctx, const int64_t n, buffer_ptr<const void> keys_bw,
    const int64_t update_stride, const int64_t update_size, buffer_ptr<const void> updates_bw,
    const DataType_t update_dtype) {
  if (n <= 0) return;
  auto modify_stream{ctx->get_modify_stream()};

  NVE_CHECK_ARG_(keys_bw != nullptr, "keys must not be null");
  const key_type* __restrict keys{static_cast<const key_type*>(
      keys_bw->access_buffer(cudaMemoryTypeUnregistered, true /*copy*/, modify_stream))};
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

    const auto slot_map_end{part.slot_map.end()};

    for (int64_t ij{}; ij < n; ++ij) {
      const key_type key{keys[ij]};
      if (partitioner(key, num_parts_mask) != task_idx) continue;

      const auto pos{part.slot_map.find(key)};
      if (pos == slot_map_end) continue;

      // TODO: Support variable length vector sizes.
      const int64_t value_size{max_value_size};
      update_kernel(pos->second, pos->second, value_size, &updates[ij * update_stride], update_size);
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config_.workgroups, 1);
}

template <typename ConfigType, typename KeyType, typename MetaType,
          typename PartitionerType>
template <bool PrefetchValues, bool WithValues, bool WithValueSizes>
int64_t STLContainerTable<ConfigType, KeyType, MetaType, PartitionerType>::find_(
    context_ptr_t& ctx, const int64_t n, const key_type* const __restrict keys,
    bitmask64_t* const __restrict hit_mask, const int64_t value_stride, std::byte* const __restrict values,
    int64_t* const __restrict value_sizes) const {
  const int64_t num_masks{ceil_div(n, bitmask64::num_bits)};

  const int64_t max_value_size{config_.max_value_size};
  NVE_CHECK_(value_stride >= max_value_size);
  const int64_t meta_offset{config_.meta_offset()};
  const int64_t max_task_size{config_.max_find_task_size};

  const std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};
  const int64_t num_parts_mask{num_parts - 1};

  const int64_t num_tasks_per_part{ceil_div(num_masks, max_task_size)};
  const int64_t num_tasks{num_parts * num_tasks_per_part};

  std::atomic_int64_t total_num_hits{0};
  const auto f{[n, keys, hit_mask, value_stride, values, value_sizes, num_masks,
                max_value_size, meta_offset, max_task_size, &parts, num_parts,
                num_parts_mask, num_tasks_per_part, &total_num_hits](int64_t task_idx) {
    int64_t num_hits{};

    const int64_t part_idx{task_idx / num_tasks_per_part};
    task_idx %= num_tasks_per_part;

    // Scatter parts accross input domain.
    int64_t hm_base{max_task_size * task_idx};
    const int64_t task_size{std::min(num_masks - hm_base, max_task_size)};
    hm_base += num_masks * part_idx / num_parts;

    {
      const Partition& __restrict part{parts[to_uint(part_idx)]};
      std::shared_lock lock(part.read_write);
      const map_type& __restrict slot_map{part.slot_map};
      const auto slot_map_end{slot_map.end()};

      const std::byte* __restrict prev_src{};
      std::byte* __restrict prev_dst{};
      int64_t prev_value_size{max_value_size};
      const meta_type meta_update{default_meta_value<meta_type>()};

      const auto process_next{[&](const int64_t j, const int64_t ij, std::byte* __restrict src) {
        // TODO: Support variable length vector sizes.
        const int64_t value_size{max_value_size};

        if constexpr (WithValues) {
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

        // This will data-race with multiple threads. But losing some updates is fine during find.
        if constexpr (!std::is_same_v<meta_type, no_meta_t>) {
          meta_type& __restrict meta_value{*reinterpret_cast<meta_type*>(&src[meta_offset])};
          if constexpr (std::is_same_v<meta_type, lru_meta_t>) {
            meta_value = meta_update;
          } else if constexpr (std::is_same_v<meta_type, lfu_meta_t>) {
            if (meta_value < max_lfu_meta) meta_value += meta_update;
          }
        } else {
          (void)meta_offset;
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
        const int64_t i{(hm_base + i_off) % num_masks};
        const int64_t i0{i * bitmask64::num_bits};

        bitmask64_t mask{bitmask64::atomic_load(&hit_mask[i])};
        num_hits -= bitmask64::count(mask);

        for (auto it{bitmask64::clip(~mask, n - i0)}; it; it = bitmask64::skip(it)) {
          const int64_t j{bitmask64::next(it)};
          const int64_t ij{i0 + j};

          const key_type key{keys[ij]};
          if NVE_LIKELY_(partitioner(key, num_parts_mask) != part_idx) continue;

          const auto pos{slot_map.find(key)};
          if (pos != slot_map_end) {
            mask |= process_next(j, ij, pos->second);
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

}  // namespace nve
