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

#include <stl_map_backed_table.hpp>
#include <atomic>
#include <thread_pool.hpp>
#include <execution_context.hpp>

namespace nve {

template <typename ConfigType, typename MaskType, typename KeyType, typename MetaType,
          typename PartitionerType>
STLContainerTable<ConfigType, MaskType, KeyType, MetaType, PartitionerType>::STLContainerTable(
    const table_id_t id, const config_type& config)
    : base_type(id, config), parts_(static_cast<size_t>(config.num_partitions)) {
  NVE_CHECK_(config.key_size == sizeof(key_type));
}

template <typename ConfigType, typename MaskType, typename KeyType, typename MetaType,
          typename PartitionerType>
void STLContainerTable<ConfigType, MaskType, KeyType, MetaType, PartitionerType>::clear(
    context_ptr_t& ctx) {
  const auto& __restrict config{this->config_};

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};

  const auto f{[&parts](const int64_t task_idx) {
    Partition& __restrict part{parts[static_cast<uint64_t>(task_idx)]};
    std::lock_guard lock(part.read_write);

    part.slot_map.clear();
    part.available_slots.clear();
    part.slot_buffers.clear();
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename ConfigType, typename MaskType, typename KeyType, typename MetaType,
          typename PartitionerType>
void STLContainerTable<ConfigType, MaskType, KeyType, MetaType, PartitionerType>::erase(
    context_ptr_t& ctx, const int64_t n, const void* const keys_vptr) {
  if (n <= 0) return;
  const auto& __restrict config{this->config_};

  const key_type* const __restrict keys{reinterpret_cast<const key_type*>(keys_vptr)};

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};
  const int64_t num_parts_mask{num_parts - 1};

  const auto f{[n, keys, &parts, num_parts_mask](const int64_t task_idx) {
    Partition& __restrict part{parts[static_cast<uint64_t>(task_idx)]};
    std::lock_guard lock(part.read_write);

    for (int64_t i{}; i != n; ++i) {
      const key_type key{keys[i]};
      if (partitioner(key, num_parts_mask) != task_idx) continue;

      const auto it{part.slot_map.find(key)};
      if (it == part.slot_map.end()) continue;

      part.available_slots.emplace_back(it->second);
      part.slot_map.erase(it);
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename ConfigType, typename MaskType, typename KeyType, typename MetaType,
          typename PartitionerType>
void STLContainerTable<ConfigType, MaskType, KeyType, MetaType, PartitionerType>::find(
    context_ptr_t& ctx, int64_t n, const void* const keys_vptr,
    max_bitmask_repr_t* const hit_mask, const int64_t value_stride, void* const values_vptr,
    int64_t* const value_sizes) const {
  if (n <= 0) return;
  const auto& __restrict config{this->config_};

  const key_type* const keys{reinterpret_cast<const key_type*>(keys_vptr)};
  mask_repr_type* const hm{reinterpret_cast<mask_repr_type*>(hit_mask)};
  char* const values{reinterpret_cast<char*>(values_vptr)};

  if (config.prefetch_values) {
    if (values_vptr) {
      if (value_sizes) {
        n = find_<true, true, true>(ctx, n, keys, hm, value_stride, values, value_sizes);
      } else {
        n = find_<true, true, false>(ctx, n, keys, hm, value_stride, values, value_sizes);
      }
    } else {
      if (value_sizes) {
        n = find_<true, false, true>(ctx, n, keys, hm, value_stride, values, value_sizes);
      } else {
        n = find_<true, false, false>(ctx, n, keys, hm, value_stride, values, value_sizes);
      }
    }
  } else {
    if (values_vptr) {
      if (value_sizes) {
        n = find_<false, true, true>(ctx, n, keys, hm, value_stride, values, value_sizes);
      } else {
        n = find_<false, true, false>(ctx, n, keys, hm, value_stride, values, value_sizes);
      }
    } else {
      if (value_sizes) {
        n = find_<false, false, true>(ctx, n, keys, hm, value_stride, values, value_sizes);
      } else {
        n = find_<false, false, false>(ctx, n, keys, hm, value_stride, values, value_sizes);
      }
    }
  }
  auto counter = this->get_internal_counter(ctx);
  NVE_CHECK_(counter != nullptr, "Invalid key counter");
  *counter += n;
}

template <typename ConfigType, typename MaskType, typename KeyType, typename MetaType,
          typename PartitionerType>
void STLContainerTable<ConfigType, MaskType, KeyType, MetaType, PartitionerType>::insert(
    context_ptr_t& ctx, const int64_t n, const void* const keys_vptr, const int64_t value_stride,
    const int64_t value_size, const void* const values_vptr) {
  if (n <= 0) return;
  const auto& __restrict config{this->config_};

  const key_type* const __restrict keys{reinterpret_cast<const key_type*>(keys_vptr)};
  const char* const __restrict values{reinterpret_cast<const char*>(values_vptr)};

  // TODO: Allow dynamically sized vectors.
  const int64_t max_value_size{config.max_value_size};
  NVE_CHECK_(value_size >= 0 && value_size <= max_value_size);
  const uint64_t slot_stride{static_cast<uint64_t>(config.slot_stride())};
  const uint64_t slots_per_buffer{static_cast<uint64_t>(config.allocation_rate) / slot_stride};
  const uint64_t overflow_margin{static_cast<uint64_t>(config.overflow_policy.overflow_margin)};
  const int64_t resolution_margin{static_cast<int64_t>(static_cast<double>(overflow_margin) *
                                                       config.overflow_policy.resolution_margin)};

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};
  const int64_t num_parts_mask{num_parts - 1};

  const table_id_t table_id{this->id};
  const auto f{[n, keys, value_stride, value_size, values, table_id, max_value_size, slot_stride,
                slots_per_buffer, overflow_margin, resolution_margin, &parts,
                num_parts_mask](const int64_t task_idx) {
    Partition& __restrict part{parts[static_cast<uint64_t>(task_idx)]};
    std::lock_guard lock(part.read_write);

    lru_meta_type lru_value;
    if constexpr (std::is_same_v<meta_type, no_meta_type>) {
    } else if constexpr (std::is_same_v<meta_type, lru_meta_type>) {
      lru_value = lru_meta_value();
    } else if constexpr (std::is_same_v<meta_type, lfu_meta_type>) {
    } else {
      static_assert(dependent_false_v<meta_type>, "Overflow handler not implemented.");
    }

    for (int64_t i{}; i != n; ++i) {
      const key_type key{keys[i]};
      if (partitioner(key, num_parts_mask) != task_idx) continue;

      {
        const auto it{part.slot_map.find(key)};
        if (it != part.slot_map.end()) {
          std::copy_n(&values[i * value_stride], value_size, it->second);
          continue;
        }
      }

      // Handle overflows.
      if NVE_UNLIKELY_(part.slot_map.size() >= overflow_margin) {
        NVE_LOG_VERBOSE_("STL table ", table_id, " part ", task_idx,
                         " is overflowing. Attempting to resolve...");

        // Fetch all key/slot pairs.
        std::vector<std::pair<key_type, char*>> keys_slots(part.slot_map.begin(),
                                                           part.slot_map.end());

        // Select keys to evict.
        if constexpr (std::is_same_v<meta_type, no_meta_type>) {
          // Randomly shuffle slots.
          std::shuffle(keys_slots.begin(), keys_slots.end(),
                       std::default_random_engine{random_device()});

          (void)max_value_size;
        } else {
          // Order ASCENDING by LRU/LFU metadata tag.
          std::sort(keys_slots.begin(), keys_slots.end(),
                    [max_value_size](const auto& a, const auto& b) {
                      return *reinterpret_cast<meta_type*>(&a.second[max_value_size]) <
                             *reinterpret_cast<meta_type*>(&b.second[max_value_size]);
                    });
        }

        // Evict keys and recolaim slots until the resolution margin is reached.
        auto it{keys_slots.begin()};
        for (const auto mid{keys_slots.end() - resolution_margin}; it < mid; ++it) {
          part.available_slots.emplace_back(it->second);
          part.slot_map.erase(it->first);
        }

        if constexpr (std::is_same_v<meta_type, no_meta_type>) {
        } else if constexpr (std::is_same_v<meta_type, lru_meta_type>) {
        } else if constexpr (std::is_same_v<meta_type, lfu_meta_type>) {
          // Realign remaining frequency values to avoid entering a steady state.
          for (; it != keys_slots.end(); ++it) {
            *reinterpret_cast<meta_type*>(&it->second[max_value_size]) /= 2;
          }
        } else {
          static_assert(dependent_false_v<meta_type>, "Overflow handler not implemented.");
        }
      }

      // Allocate new slot memory if there are no slots available.
      if NVE_UNLIKELY_(part.available_slots.empty()) {
        std::vector<char>& __restrict slot_buffer{
            part.slot_buffers.emplace_back(slots_per_buffer * slot_stride)};

        part.available_slots.reserve(slots_per_buffer);
        for (size_t off{slot_buffer.size()}; off;) {
          off -= slot_stride;
          part.available_slots.emplace_back(&slot_buffer[off]);
        }
      }

      char* const __restrict slot{part.available_slots.back()};
      part.available_slots.pop_back();

      std::copy_n(&values[i * value_stride], value_size, slot);

      if constexpr (std::is_same_v<meta_type, no_meta_type>) {
      } else if constexpr (std::is_same_v<meta_type, lru_meta_type>) {
        *reinterpret_cast<meta_type*>(&slot[max_value_size]) = lru_value;
      } else if constexpr (std::is_same_v<meta_type, lfu_meta_type>) {
        *reinterpret_cast<meta_type*>(&slot[max_value_size]) = 1;
      } else {
        static_assert(dependent_false_v<meta_type>, "Overflow handler not implemented.");
      }

      part.slot_map.emplace(key, slot);
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename ConfigType, typename MaskType, typename KeyType, typename MetaType,
          typename PartitionerType>
int64_t STLContainerTable<ConfigType, MaskType, KeyType, MetaType, PartitionerType>::size(
    context_ptr_t& ctx, const bool) const {
  const auto& __restrict config{this->config_};

  const std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};

  std::atomic_int64_t total_n{0};
  const auto f{[&parts, &total_n](const int64_t task_idx) {
    int64_t n;
    {
      const Partition& __restrict part{parts[static_cast<uint64_t>(task_idx)]};
      std::shared_lock lock(part.read_write);

      n = static_cast<int64_t>(part.slot_map.size());
    }
    total_n.fetch_add(n, std::memory_order_relaxed);
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
  return total_n.load(std::memory_order_relaxed);
}

template <typename ConfigType, typename MaskType, typename KeyType, typename MetaType,
          typename PartitionerType>
void STLContainerTable<ConfigType, MaskType, KeyType, MetaType, PartitionerType>::update(
    context_ptr_t& ctx, const int64_t n, const void* const keys_vptr, const int64_t value_stride,
    const int64_t value_size, const void* const values_vptr) {
  if (n <= 0) return;
  const auto& __restrict config{this->config_};

  const key_type* const __restrict keys{reinterpret_cast<const key_type*>(keys_vptr)};
  const char* const __restrict values{reinterpret_cast<const char*>(values_vptr)};

  NVE_CHECK_(value_size >= 0 && value_size <= config.max_value_size);

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};
  const int64_t num_parts_mask{num_parts - 1};

  const auto f{[n, keys, value_stride, value_size, values, &parts, num_parts_mask](const int64_t task_idx) {
    Partition& __restrict part{parts[static_cast<size_t>(task_idx)]};
    std::lock_guard lock(part.read_write);

    const auto slot_map_end{part.slot_map.end()};

    for (int64_t i{}; i != n; ++i) {
      const key_type key{keys[i]};
      if (partitioner(key, num_parts_mask) != task_idx) continue;

      const auto it{part.slot_map.find(key)};
      if (it == slot_map_end) continue;

      std::copy_n(&values[i * value_stride], value_size, it->second);
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename ConfigType, typename MaskType, typename KeyType, typename MetaType,
          typename PartitionerType>
void STLContainerTable<ConfigType, MaskType, KeyType, MetaType, PartitionerType>::update_accumulate(
    context_ptr_t& ctx, const int64_t n, const void* const keys_vptr, const int64_t update_stride,
    const int64_t update_size, const void* const updates_vptr, const DataType_t update_dtype) {
  if (n <= 0) return;
  const auto& __restrict config{this->config_};

  const key_type* const __restrict keys{reinterpret_cast<const key_type*>(keys_vptr)};
  const char* const __restrict updates{reinterpret_cast<const char*>(updates_vptr)};

  NVE_CHECK_(update_size <= config.max_value_size);
  const update_kernel_t update_kernel{pick_cpu_update_kernel(config.value_dtype, update_dtype)};

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};
  const int64_t num_parts_mask{num_parts - 1};

  const auto f{[n, keys, update_stride, update_size, updates, &update_kernel, &parts, num_parts_mask](const int64_t task_idx) {
    Partition& __restrict part{parts[static_cast<uint64_t>(task_idx)]};
    std::lock_guard lock(part.read_write);

    const auto slot_map_end{part.slot_map.end()};

    for (int64_t i{}; i != n; ++i) {
      const key_type key{keys[i]};
      if (partitioner(key, num_parts_mask) != task_idx) continue;

      const auto it{part.slot_map.find(key)};
      if (it == slot_map_end) continue;

      update_kernel(it->second, &updates[i * update_stride], update_size);
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename ConfigType, typename MaskType, typename KeyType, typename MetaType,
          typename PartitionerType>
template <bool PrefetchValues, bool WithValues, bool WithValueSizes>
int64_t STLContainerTable<ConfigType, MaskType, KeyType, MetaType, PartitionerType>::find_(
    context_ptr_t& ctx, const int64_t n, const key_type* const __restrict keys,
    mask_repr_type* const __restrict hit_mask, const int64_t value_stride, char* const __restrict values,
    int64_t* const __restrict value_sizes) const {
  const auto& __restrict config{this->config_};

  const int64_t hm_size{mask_type::mask_size(n)};

  const int64_t max_value_size{config.max_value_size};
  NVE_CHECK_(value_stride >= max_value_size);
  const int64_t max_find_task_size{config.max_find_task_size};

  const std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};
  const int64_t num_parts_mask{num_parts - 1};

  const int64_t num_tasks_per_part{(hm_size + max_find_task_size - 1) / max_find_task_size};
  const int64_t num_tasks{num_parts * num_tasks_per_part};

  std::atomic_int64_t total_num_hits{0};
  const auto f{[n, keys, hit_mask, value_stride, values, value_sizes, hm_size,
                max_value_size, max_find_task_size, &parts, num_parts, num_parts_mask,
                num_tasks_per_part, &total_num_hits](int64_t task_idx) {
    int64_t num_hits{};

    const int64_t part_idx{task_idx / num_tasks_per_part};
    task_idx %= num_tasks_per_part;

    int64_t hm_off0{max_find_task_size * task_idx};
    const int64_t task_size{std::min(max_find_task_size, hm_size - hm_off0)};

    // Scatter parts accross input domain.
    hm_off0 += hm_size * part_idx / num_parts;
    {
      const Partition& __restrict part{parts[static_cast<uint64_t>(part_idx)]};
      std::shared_lock lock(part.read_write);
      const map_type& __restrict slot_map{part.slot_map};
      const auto slot_map_end{slot_map.end()};

      char* __restrict prev_src{};
      char* __restrict prev_dst;
      int64_t prev_value_size{max_value_size};
      const lru_meta_type lru_time{lru_meta_value()};

      for (int64_t hm_idx{}; hm_idx != task_size; ++hm_idx) {       
        const int64_t hm_off{(hm_off0 + hm_idx) % hm_size};
        mask_repr_type mask{mask_type::load(&hit_mask[hm_off])};
        num_hits -= mask_type::count(mask);
        const int64_t i{hm_off * mask_type::num_bits};

        for (auto it{mask_type::clip(mask_type::invert(mask), n - i)}; mask_type::has_next(it);
             it = mask_type::skip(it)) {
          const int64_t j{mask_type::next(it)};
          const int64_t ij{i + j};

          const key_type key{keys[ij]};
          if NVE_LIKELY_(partitioner(key, num_parts_mask) != part_idx) continue;

          const auto pos{slot_map.find(key)};
          if (pos == slot_map_end) continue;
          char* const __restrict src{pos->second};

          // TODO: Support variable length vector sizes.
          const int64_t value_size{max_value_size};
          if constexpr (WithValueSizes) {
            value_sizes[ij] = value_size;
          } else {
            (void)value_sizes;
          }
          if constexpr (WithValues) {
            char* const __restrict dst{&values[ij * value_stride]};
            if constexpr (PrefetchValues) {
              l1_prefetch(src, dst, std::min(value_size, 8 * cpu_cache_line_size));
              if (prev_src) {
#pragma GCC diagnostic push
#if !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
                std::copy_n(prev_src, static_cast<uint64_t>(prev_value_size), prev_dst);
#pragma GCC diagnostic pop
              }
              prev_src = src;
              prev_dst = dst;
              prev_value_size = value_size;
            } else {
              std::copy_n(src, value_size, dst);
            }
          } else {
            (void)value_stride;
            (void)values;
          }
          
          update_meta_data<meta_type>(&src[max_value_size], lru_time);
          mask = mask_type::set(mask, j);
        }

        num_hits += mask_type::count(mask);
        mask_type::atomic_join(&hit_mask[hm_off], mask);
      }

      if constexpr (WithValues && PrefetchValues) {
        if (prev_src) {
#pragma GCC diagnostic push
#if !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
          std::copy_n(prev_src, static_cast<uint64_t>(prev_value_size), prev_dst);
#pragma GCC diagnostic pop
        }
      }
    }
    total_num_hits.fetch_add(num_hits, std::memory_order_relaxed);
  }};

  ctx->get_thread_pool()->execute_n(0, num_tasks, f, config.workgroups, num_tasks_per_part);
  return total_num_hits.load(std::memory_order_relaxed);
}

}  // namespace nve
