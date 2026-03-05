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
#include <nvhm_map_table.hpp>
#include <thread_pool.hpp>

namespace nve {
namespace plugin {

// TODO: Should we enable optimistic prefetch?
static constexpr bool use_optimistic_prefetch{false};

void NvhmMapTableConfig::check() const {
  base_type::check();

  NVE_CHECK_(num_partitions >= 0 && has_single_bit(static_cast<uint64_t>(num_partitions)));
  NVE_CHECK_(!workgroups.empty());
  NVE_CHECK_(max_find_task_size > 0);

  NVE_CHECK_(key_fetch_queue_length == 0 || (key_fetch_queue_length <= 8 && has_single_bit(static_cast<uint64_t>(key_fetch_queue_length))));

  NVE_CHECK_(initial_capacity >= 0);
  NVE_CHECK_(value_alignment >= 0 && has_single_bit(static_cast<uint64_t>(value_alignment)));

  overflow_policy.check();
}

void from_json(const nlohmann::json& json, NvhmMapTableConfig& conf) {
  using base_type = NvhmMapTableConfig::base_type;
  from_json(json, static_cast<base_type&>(conf));

  NVE_READ_JSON_FIELD_(num_partitions);
  NVE_READ_JSON_FIELD_(partitioner);
  NVE_READ_JSON_FIELD_(workgroups);
  NVE_READ_JSON_FIELD_(max_find_task_size);

  NVE_READ_JSON_FIELD_(kernel_size);
  NVE_READ_JSON_FIELD_(initial_capacity);
  NVE_READ_JSON_FIELD_(value_alignment);

  NVE_READ_JSON_FIELD_(key_fetch_queue_length);
  NVE_READ_JSON_FIELD_(prefetch_values);

  NVE_READ_JSON_FIELD_(minimize_psl);
  NVE_READ_JSON_FIELD_(auto_shrink);

  NVE_READ_JSON_FIELD_(overflow_policy);
}

void to_json(nlohmann::json& json, const NvhmMapTableConfig& conf) {
  using base_type = NvhmMapTableConfig::base_type;
  to_json(json, static_cast<const base_type&>(conf));

  NVE_WRITE_JSON_FIELD_(num_partitions);
  NVE_WRITE_JSON_FIELD_(partitioner);
  NVE_WRITE_JSON_FIELD_(workgroups);
  NVE_WRITE_JSON_FIELD_(max_find_task_size);

  NVE_WRITE_JSON_FIELD_(kernel_size);
  NVE_WRITE_JSON_FIELD_(initial_capacity);
  NVE_WRITE_JSON_FIELD_(value_alignment);

  NVE_WRITE_JSON_FIELD_(key_fetch_queue_length);
  NVE_WRITE_JSON_FIELD_(prefetch_values);

  NVE_WRITE_JSON_FIELD_(minimize_psl);
  NVE_WRITE_JSON_FIELD_(auto_shrink);

  NVE_WRITE_JSON_FIELD_(overflow_policy);
}

template <typename MaskType, typename MapType, typename PartitionerType>
NvhmMapTable<MaskType, MapType, PartitionerType>::NvhmMapTable(
    const table_id_t id, const NvhmMapTableConfig& config)
    : base_type(id, config), parts_(static_cast<uint64_t>(config.num_partitions)) {
  NVE_CHECK_(config.key_size == sizeof(key_type));
  for (auto& part : parts_) {
    part.map = map_type(static_cast<uint64_t>(config.initial_capacity),
                        static_cast<uint64_t>(config.max_value_size),
                        static_cast<uint64_t>(config.value_alignment));
  }
}

template <typename MaskType, typename MapType, typename PartitionerType>
void NvhmMapTable<MaskType, MapType, PartitionerType>::clear(context_ptr_t& ctx) {
  const auto& __restrict config{config_};

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};

  const auto f{[&parts](const int64_t task_idx) {
    Partition& __restrict part{parts[static_cast<uint64_t>(task_idx)]};
    std::lock_guard lock(part.read_write);

    part.map.clear();
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename MaskType, typename MapType, typename PartitionerType>
void NvhmMapTable<MaskType, MapType, PartitionerType>::erase(context_ptr_t& ctx, const int64_t n,
                                                                  const void* const keys_vptr) {
  if (n <= 0) return;
  const auto& __restrict config{config_};

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

      part.map.erase(key);
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename MaskType, typename MapType, typename PartitionerType>
void NvhmMapTable<MaskType, MapType, PartitionerType>::find(
    context_ptr_t& ctx, int64_t n, const void* const keys_vptr,
    max_bitmask_repr_t* const hit_mask, const int64_t value_stride, void* const values_vptr,
    int64_t* const value_sizes) const {
  if (n <= 0) return;
  const auto& __restrict config{config_};

  const key_type* const keys{reinterpret_cast<const key_type*>(keys_vptr)};
  mask_repr_type* const hm{reinterpret_cast<mask_repr_type*>(hit_mask)};
  char* const values{reinterpret_cast<char*>(values_vptr)};

  if (config.prefetch_values) {
    switch (config.key_fetch_queue_length) {
      case 0:
        n = find_<0, true>(ctx, n, keys, hm, value_stride, values, value_sizes);
        break;
      case 1:
        n = find_<1, true>(ctx, n, keys, hm, value_stride, values, value_sizes);
        break;
      case 2:
        n = find_<2, true>(ctx, n, keys, hm, value_stride, values, value_sizes);
        break;
      case 4:
        n = find_<4, true>(ctx, n, keys, hm, value_stride, values, value_sizes);
        break;
      case 8:
        n = find_<8, true>(ctx, n, keys, hm, value_stride, values, value_sizes);
        break;
      default:
        NVE_THROW_("`config.key_fetch_queue_length` (", config.key_fetch_queue_length, ") is out of bounds!");
    }
  } else {
    switch (config.key_fetch_queue_length) {
      case 0:
        n = find_<0, false>(ctx, n, keys, hm, value_stride, values, value_sizes);
        break;
      case 1:
        n = find_<1, false>(ctx, n, keys, hm, value_stride, values, value_sizes);
        break;
      case 2:
        n = find_<2, false>(ctx, n, keys, hm, value_stride, values, value_sizes);
        break;
      case 4:
        n = find_<4, false>(ctx, n, keys, hm, value_stride, values, value_sizes);
        break;
      case 8:
        n = find_<8, false>(ctx, n, keys, hm, value_stride, values, value_sizes);
        break;
      default:
        NVE_THROW_("`config.key_fetch_queue_length` (", config.key_fetch_queue_length, ") is out of bounds!");
    }
  }
  auto counter = this->get_internal_counter(ctx);
  NVE_CHECK_(counter != nullptr, "Invalid key counter");
  *counter += n;
}

template <typename MaskType, typename MapType, typename PartitionerType>
void NvhmMapTable<MaskType, MapType, PartitionerType>::insert(
    context_ptr_t& ctx, const int64_t n, const void* const keys_vptr, const int64_t value_stride,
    const int64_t value_size, const void* const values_vptr) {
  if (n <= 0) return;
  const auto& __restrict config{config_};

  const key_type* const __restrict keys{reinterpret_cast<const key_type*>(keys_vptr)};
  const char* const __restrict values{reinterpret_cast<const char*>(values_vptr)};

  // TODO: Allow dynamically sized vectors.
  NVE_CHECK_(value_size >= 0 && value_size <= config.max_value_size);
  const int64_t overflow_margin{config.overflow_policy.overflow_margin};
  const int64_t resolution_margin{static_cast<int64_t>(static_cast<double>(overflow_margin) *
                                                       config.overflow_policy.resolution_margin)};

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};
  const int64_t num_parts_mask{num_parts - 1};

  const table_id_t table_id{this->id};
  const auto f{[n, keys, value_stride, value_size, values, table_id, overflow_margin,
                resolution_margin, &parts, num_parts_mask](const int64_t task_idx) {
    Partition& __restrict part{parts[static_cast<uint64_t>(task_idx)]};
    std::lock_guard lock(part.read_write);
    map_type& __restrict map{part.map};

    lru_meta_type lru_value;
    if constexpr (std::is_same_v<meta_type, no_meta_type>) {
    } else if constexpr (std::is_same_v<meta_type, lru_meta_type>) {
      lru_value = lru_meta_value();
    } else if constexpr (std::is_same_v<meta_type, lfu_meta_type>) {
    } else {
      static_assert(dependent_false_v<meta_type>, "Overflow handler not implemented.");
    }

    int64_t part_size{static_cast<int64_t>(map.size())};
    for (int64_t i{}; i != n; ++i) {
      const key_type key{keys[i]};
      if (partitioner(key, num_parts_mask) != task_idx) continue;

      int64_t new_part_size{part_size};
      const write_pos_type pos{map.upsert(key, new_part_size)};
      map.set_raw_values_at(pos, &values[i * value_stride], static_cast<uint64_t>(value_size));

      if (new_part_size == part_size) continue;
      part_size = new_part_size;

      if constexpr (std::is_same_v<meta_type, no_meta_type>) {
      } else if constexpr (std::is_same_v<meta_type, lru_meta_type>) {
        map.value_at(pos) = lru_value;
      } else if constexpr (std::is_same_v<meta_type, lfu_meta_type>) {
        map.value_at(pos) = 1;
      } else {
        static_assert(dependent_false_v<meta_type>, "Overflow handler not implemented.");
      }

      // Handle overflows.
      if NVE_LIKELY_(part_size < overflow_margin) continue;
      NVE_LOG_VERBOSE_("NV map table ", table_id, " part ", task_idx,
                       " is overflowing. Attempting to resolve...");

      if constexpr (std::is_same_v<meta_type, no_meta_type>) {
        // Fetch all keys and shuffle them randomly.
        std::vector<key_type> keys(map.size());
        auto it{keys.begin()};
        it = map.keys(it);
        NVE_CHECK_(it == keys.end());
        std::shuffle(keys.begin(), keys.end(), std::default_random_engine{random_device()});

        // Reclaim slots until the resolution margin is reached.
        const auto margin_it{keys.end() - resolution_margin};
        for (it = keys.begin(); it < margin_it; ++it) {
          map.erase(*it);
        }
      } else {
        // Fetch all keys and sort them by the ASCENDING by the associated meta-values.
        std::vector<std::pair<key_type, meta_type>> keys_metas(map.size());
        auto it{keys_metas.begin()};
        it = map.keys_and_values(it);
        NVE_CHECK_(it == keys_metas.end());
        std::sort(keys_metas.begin(), keys_metas.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        // Reclaim slots until the resolution margin is reached.
        const auto margin_it{keys_metas.end() - resolution_margin};
        for (it = keys_metas.begin(); it < margin_it; ++it) {
          map.erase(it->first);
        }

        if constexpr (std::is_same_v<meta_type, lru_meta_type>) {
        } else if constexpr (std::is_same_v<meta_type, lfu_meta_type>) {
          // Realign remaining frequency values to avoid entering a steady state.
          map.transform_values([](meta_type& meta) { meta >>= 1; });
        } else {
          static_assert(dependent_false_v<meta_type>, "Overflow handler not implemented.");
        }
      }

      // Update part_size after erasing
      part_size = static_cast<int64_t>(map.size());
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename MaskType, typename MapType, typename PartitionerType>
int64_t NvhmMapTable<MaskType, MapType, PartitionerType>::size(context_ptr_t& ctx,
                                                                    const bool) const {
  const std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};

  std::atomic_int64_t total_n{0};
  const auto f{[&parts, &total_n](const int64_t task_idx) {
    int64_t n;
    {
      const Partition& __restrict part{parts[static_cast<uint64_t>(task_idx)]};
      std::shared_lock lock(part.read_write);

      n = static_cast<int64_t>(part.map.size());
    }
    total_n.fetch_add(n, std::memory_order_relaxed);
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config_.workgroups, 1);
  return total_n.load(std::memory_order_relaxed);
}

template <typename MaskType, typename MapType, typename PartitionerType>
void NvhmMapTable<MaskType, MapType, PartitionerType>::update(
    context_ptr_t& ctx, const int64_t n, const void* const keys_vptr, const int64_t value_stride,
    const int64_t value_size, const void* const values_vptr) {
  if (n <= 0) return;
  const auto& __restrict config{config_};

  const key_type* const __restrict keys{reinterpret_cast<const key_type*>(keys_vptr)};
  const char* const __restrict values{reinterpret_cast<const char*>(values_vptr)};

  NVE_CHECK_(value_size >= 0 && value_size <= config.max_value_size);

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};
  const int64_t num_parts_mask{num_parts - 1};

  const auto f{[n, keys, value_stride, value_size, values, &parts, num_parts_mask](const int64_t task_idx) {
    Partition& __restrict part{parts[static_cast<uint64_t>(task_idx)]};
    std::lock_guard lock(part.read_write);
    map_type& __restrict map{part.map};

    for (int64_t i{}; i != n; ++i) {
      const key_type key{keys[i]};
      if (partitioner(key, num_parts_mask) != task_idx) continue;

      const write_pos_type pos{map.update(key)};
      if (pos == nvhm::npos) continue;

      map.set_raw_values_at(pos, &values[i * value_stride], static_cast<uint64_t>(value_size));
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename MaskType, typename MapType, typename PartitionerType>
void NvhmMapTable<MaskType, MapType, PartitionerType>::update_accumulate(
    context_ptr_t& ctx, const int64_t n, const void* const keys_vptr, const int64_t update_stride,
    const int64_t update_size, const void* const updates_vptr, const DataType_t update_dtype) {
  if (n <= 0) return;
  const auto& __restrict config{config_};

  const key_type* const __restrict keys{reinterpret_cast<const key_type*>(keys_vptr)};
  const char* const __restrict updates{reinterpret_cast<const char*>(updates_vptr)};

  NVE_CHECK_(update_size >= 0 && update_size <= config.max_value_size);
  const update_kernel_t update_kernel{pick_cpu_update_kernel(config.value_dtype, update_dtype)};

  std::vector<Partition>& __restrict parts{parts_};
  const int64_t num_parts{static_cast<int64_t>(parts.size())};
  const int64_t num_parts_mask{num_parts - 1};

  const auto f{[n, keys, update_stride, update_size, updates, &update_kernel, &parts, num_parts_mask](const int64_t task_idx) {
    Partition& __restrict part{parts[static_cast<uint64_t>(task_idx)]};
    std::lock_guard lock(part.read_write);
    map_type& __restrict map{part.map};

    for (int64_t i{}; i != n; ++i) {
      const key_type key{keys[i]};
      if (partitioner(key, num_parts_mask) != task_idx) continue;

      const write_pos_type pos{map.update(key)};
      if (pos == nvhm::npos) continue;

      update_kernel(map.raw_values_at(pos), &updates[i * update_stride], update_size);
    }
  }};

  ctx->get_thread_pool()->execute_n(0, num_parts, f, config.workgroups, 1);
}

template <typename MaskType, typename MapType, typename PartitionerType>
template <size_t KeyFetchQueueLength, bool PrefetchValues>
int64_t NvhmMapTable<MaskType, MapType, PartitionerType>::find_(
  context_ptr_t& ctx, const int64_t n, const key_type* const __restrict keys,
  mask_repr_type* const __restrict hit_mask, const int64_t value_stride, char* const __restrict values,
  int64_t* const __restrict value_sizes) const {
  if (values) {
    if (value_sizes) {
      return find_<KeyFetchQueueLength, PrefetchValues, true, true>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
    } else {
      return find_<KeyFetchQueueLength, PrefetchValues, true, false>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
    }
  } else {
    if (value_sizes) {
      return find_<KeyFetchQueueLength, PrefetchValues, false, true>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
    } else {
      return find_<KeyFetchQueueLength, PrefetchValues, false, false>(ctx, n, keys, hit_mask, value_stride, values, value_sizes);
    }
  }
}

template <typename MaskType, typename MapType, typename PartitionerType>
template <size_t KeyFetchQueueLength, bool PrefetchValues, bool WithValues, bool WithValueSizes>
int64_t NvhmMapTable<MaskType, MapType, PartitionerType>::find_(
    context_ptr_t& ctx, const int64_t n, const key_type* const __restrict keys,
    mask_repr_type* const __restrict hit_mask, const int64_t value_stride, char* const __restrict values,
    int64_t* const __restrict value_sizes) const {
  const auto& __restrict config{config_};

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
      const map_type& __restrict map{part.map};

      const char* __restrict prev_src{};
#pragma GCC diagnostic push
#if !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
      char* __restrict prev_dst;
#pragma GCC diagnostic pop
      int64_t prev_value_size{max_value_size};
      const lru_meta_type lru_time{lru_meta_value()};

      for (int64_t hm_idx{}; hm_idx != task_size; ++hm_idx) {
        const int64_t hm_off{(hm_off0 + hm_idx) % hm_size};
        mask_repr_type mask{mask_type::load(&hit_mask[hm_off])};
        const int64_t i{hm_off * mask_type::num_bits};
        num_hits -= mask_type::count(mask);

        auto it_head{mask_type::clip(mask_type::invert(mask), n - i)};
        if constexpr (KeyFetchQueueLength) {
          mask_repr_type it_tail{};
          const auto process_tail{[&](const int j, const read_pos_type& pos) {
            const int64_t ij{i + j};

            // TODO: Support variable length vector sizes.
            const int64_t value_size{max_value_size};
            if constexpr (WithValueSizes) {
              value_sizes[ij] = value_size;
            } else {
              (void)value_sizes;
            }
            if constexpr (WithValues) {
              const char* const __restrict src{map.raw_values_at(pos)};
              char* const __restrict dst{&values[ij * value_stride]};
              if constexpr (PrefetchValues) {
                l1_prefetch(src, dst, std::min(value_size, 8 * cpu_cache_line_size));
                if (prev_src) {
#pragma GCC diagnostic push
#if !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
                  nvhm::fast_copy(prev_dst, prev_src, static_cast<uint64_t>(prev_value_size));
#pragma GCC diagnostic pop
                }
                prev_src = src;
                prev_dst = dst;
                prev_value_size = value_size;
              } else {
                nvhm::fast_copy(dst, src, static_cast<uint64_t>(value_size));
              }
            } else {
              (void)value_stride;
              (void)values;
            }

            if constexpr (std::is_same_v<meta_type, no_meta_type>) {
            } else if constexpr (std::is_same_v<meta_type, lru_meta_type>) {
              const_cast<lru_meta_type&>(map.value_at(pos)) = lru_time;
            } else if constexpr (std::is_same_v<meta_type, lfu_meta_type>) {
              ++const_cast<lfu_meta_type&>(map.value_at(pos));
            } else {
              static_assert(dependent_false_v<meta_type>, "Overflow handler not implemented.");
            }
            mask = mask_type::insert(mask, j);
          }};
          
          nvhm::experimental::ring_prefetch_queue<key_type, prefetch_type, KeyFetchQueueLength> queue;
          while (mask_type::has_next(it_head)) {
            const int64_t j{mask_type::next(it_head)};
            it_head = mask_type::skip(it_head);
            const int64_t ij{i + j};
  
            const key_type key{keys[ij]};
            if NVE_LIKELY_(partitioner(key, num_parts_mask) != part_idx) continue;
            it_tail = mask_type::insert(it_tail, j);

            queue.prepare_lookup(map, key, use_optimistic_prefetch);
            if (queue.full()) break;
          }

          while (mask_type::has_next(it_head)) {
            const int64_t j{mask_type::next(it_head)};
            it_head = mask_type::skip(it_head);
            const int64_t ij{i + j};

            const key_type key{keys[ij]};
            if NVE_LIKELY_(partitioner(key, num_parts_mask) != part_idx) continue;
            it_tail = mask_type::insert(it_tail, j);

            const read_pos_type pos{queue.pop_and_lookup(map)};
            if (pos != nvhm::npos) {
              process_tail(mask_type::next(it_tail), pos);
            }
            it_tail = mask_type::skip(it_tail);

            queue.prepare_lookup(map, key, use_optimistic_prefetch);
          }

          while (mask_type::has_next(it_tail)) {
            const read_pos_type pos{queue.pop_and_lookup(map)};
            if (pos != nvhm::npos) {
              process_tail(mask_type::next(it_tail), pos);
            }
            it_tail = mask_type::skip(it_tail);
          }
        } else {
          for (; mask_type::has_next(it_head); it_head = mask_type::skip(it_head)) {
            const int64_t j{mask_type::next(it_head)};
            const int64_t ij{i + j};

            const key_type key{keys[ij]};
            if NVE_LIKELY_(partitioner(key, num_parts_mask) != part_idx) continue;

            const read_pos_type pos{map.lookup(key)};
            if (pos == nvhm::npos) continue;
 
            const int64_t value_size{max_value_size};  // TODO: Support variable length vector sizes.
            if constexpr (WithValueSizes) {
              value_sizes[ij] = value_size;
            } else {
              (void)value_sizes;
            }
            if constexpr (WithValues) {
              const char* const __restrict src{map.raw_values_at(pos)};
              char* const __restrict dst{&values[ij * value_stride]};
              if constexpr (PrefetchValues) {
                l1_prefetch(src, dst, std::min(value_size, 8 * cpu_cache_line_size));
                if (prev_src) {
#pragma GCC diagnostic push
#if !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
                  nvhm::fast_copy(prev_dst, prev_src, static_cast<uint64_t>(prev_value_size));
#pragma GCC diagnostic pop
                }
                prev_src = src;
                prev_dst = dst;
                prev_value_size = value_size;
              } else {
                nvhm::fast_copy(dst, src, static_cast<uint64_t>(value_size));
              }
            } else {
              (void)value_stride;
              (void)values;
            }

            if constexpr (std::is_same_v<meta_type, no_meta_type>) {
            } else if constexpr (std::is_same_v<meta_type, lru_meta_type>) {
              const_cast<lru_meta_type&>(map.value_at(pos)) = lru_time;
            } else if constexpr (std::is_same_v<meta_type, lfu_meta_type>) {
              ++const_cast<lfu_meta_type&>(map.value_at(pos));
            } else {
              static_assert(dependent_false_v<meta_type>, "Overflow handler not implemented.");
            }
            mask = mask_type::insert(mask, j);
          }
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
          nvhm::fast_copy(prev_dst, prev_src, static_cast<uint64_t>(prev_value_size));
#pragma GCC diagnostic pop
        }
      }
    }

    total_num_hits.fetch_add(num_hits, std::memory_order_relaxed);
  }};

  ctx->get_thread_pool()->execute_n(0, num_tasks, f, config.workgroups, num_tasks_per_part);
  return total_num_hits.load(std::memory_order_relaxed);
}

void NvhmMapTableFactoryConfig::check() const { base_type::check(); }

void from_json(const nlohmann::json& json, NvhmMapTableFactoryConfig& conf) {
  using base_type = NvhmMapTableFactoryConfig::base_type;
  from_json(json, static_cast<base_type&>(conf));
}

void to_json(nlohmann::json& json, const NvhmMapTableFactoryConfig& conf) {
  using base_type = NvhmMapTableFactoryConfig::base_type;
  to_json(json, static_cast<const base_type&>(conf));
}

NvhmMapTableFactory::NvhmMapTableFactory(const config_type& config) : base_type(config) {}

template <typename MaskType, typename MapType>
inline static host_table_ptr_t make_nvhm_map_table_6(const table_id_t id,
                                                  const NvhmMapTableConfig& config) {
  if (config.num_partitions == 1) {
    if (config.partitioner != Partitioner_t::AlwaysZero) {
      NVE_LOG_VERBOSE_("Selected ", config.partitioner, " partitioner was disabled because table has only 1 partition.");
    }
    return std::make_shared<NvhmMapTable<MaskType, MapType, AlwaysZeroPartitioner>>(id, config);
  }

  switch (config.partitioner) {
#if defined(NVE_FEATURE_HT_PART_FNV1A)
    case Partitioner_t::FowlerNollVo:
      return std::make_shared<NvhmMapTable<MaskType, MapType, FowlerNollVoPartitioner>>(id, config);
#endif
#if defined(NVE_FEATURE_HT_PART_MURMUR)
    case Partitioner_t::Murmur3:
      return std::make_shared<NvhmMapTable<MaskType, MapType, Murmur3Partitioner>>(id, config);
#endif
#if defined(NVE_FEATURE_HT_PART_RRXMRRXMSX0)
    case Partitioner_t::Rrxmrrxmsx0:
      return std::make_shared<NvhmMapTable<MaskType, MapType, Rrxmrrxmsx0Partitioner>>(id, config);
#endif
#if defined(NVE_FEATURE_HT_PART_STD_HASH)
    case Partitioner_t::StdHash:
      return std::make_shared<NvhmMapTable<MaskType, MapType, StdHashPartitioner>>(id, config);
#endif
    default:
      NVE_THROW_("`config.partitioner` (", config.partitioner, ") is out of bounds!");
  }
}

template <typename MaskType, typename KeyType, typename MetaType, typename KernelType,
          bool MinimizePSL>
inline static host_table_ptr_t make_nvhm_map_table_5(const table_id_t id,
                                                          const NvhmMapTableConfig& config) {
  if (config.auto_shrink) {
    using map_t = nvhm::map<KeyType, MetaType, char, KernelType,
                                     nvhm::default_seq_t, MinimizePSL, true>;
    return make_nvhm_map_table_6<MaskType, map_t>(id, config);
  } else {
    using map_t = nvhm::map<KeyType, MetaType, char, KernelType,
                                     nvhm::default_seq_t, MinimizePSL, false>;
    return make_nvhm_map_table_6<MaskType, map_t>(id, config);
  }
}

template <typename MaskType, typename KeyType, typename MetaType, typename KernelType>
inline static host_table_ptr_t make_nvhm_map_table_4(const table_id_t id,
                                                          const NvhmMapTableConfig& config) {
  if (config.minimize_psl) {
    return make_nvhm_map_table_5<MaskType, KeyType, MetaType, KernelType, true>(id, config);
  } else {
    return make_nvhm_map_table_5<MaskType, KeyType, MetaType, KernelType, false>(id, config);
  }
}

template <typename MaskType, typename KeyType, typename MetaType>
inline static host_table_ptr_t make_nvhm_map_table_3(const table_id_t id,
                                                          const NvhmMapTableConfig& config) {
  switch (config.kernel_size) {
#if defined(NVE_FEATURE_HT_KERNEL_8)
    case 1:
      return make_nvhm_map_table_4<MaskType, KeyType, MetaType,
                                        nvhm::default_kernel8_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KERNEL_16)
    case 2:
      return make_nvhm_map_table_4<MaskType, KeyType, MetaType,
                                        nvhm::default_kernel16_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KERNEL_32)
    case 4:
      return make_nvhm_map_table_4<MaskType, KeyType, MetaType,
                                        nvhm::default_kernel32_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KERNEL_64)
    case 8:
      return make_nvhm_map_table_4<MaskType, KeyType, MetaType,
                                        nvhm::default_kernel64_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KERNEL_128)
    case 16:
      return make_nvhm_map_table_4<MaskType, KeyType, MetaType,
                                        nvhm::default_kernel128_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KERNEL_256)
    case 32:
      return make_nvhm_map_table_4<MaskType, KeyType, MetaType,
                                        nvhm::default_kernel256_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KERNEL_512)
    case 64:
      return make_nvhm_map_table_4<MaskType, KeyType, MetaType,
                                        nvhm::default_kernel512_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KERNEL_1024)
    case 128:
      return make_nvhm_map_table_4<MaskType, KeyType, MetaType,
                                        nvhm::default_kernel1024_t>(id, config);
#endif
  }
  NVE_THROW_("`config.kernel_size` (", config.kernel_size, ") is out of bounds!");
}

template <typename MaskType, typename KeyType>
inline static host_table_ptr_t make_nvhm_map_table_2(const table_id_t id,
                                                          const NvhmMapTableConfig& config) {
  switch (config.overflow_policy.handler) {
    case OverflowHandler_t::EvictRandom:
      return make_nvhm_map_table_3<MaskType, KeyType, no_meta_type>(id, config);
    case OverflowHandler_t::EvictLRU:
      return make_nvhm_map_table_3<MaskType, KeyType, lru_meta_type>(id, config);
    case OverflowHandler_t::EvictLFU:
      return make_nvhm_map_table_3<MaskType, KeyType, lfu_meta_type>(id, config);
  }
  NVE_THROW_("`config.overflow_policy.handler` (", config.overflow_policy.handler,
             ") is out of bounds!");
}

template <typename MaskType>
inline static host_table_ptr_t make_nvhm_map_table_1(const table_id_t id,
                                                          const NvhmMapTableConfig& config) {
  switch (config.key_size) {
#if defined(NVE_FEATURE_HT_KEY_8)
    case sizeof(int8_t):
      return make_nvhm_map_table_2<MaskType, int8_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KEY_16)
    case sizeof(int16_t):
      return make_nvhm_map_table_2<MaskType, int16_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KEY_32)
    case sizeof(int32_t):
      return make_nvhm_map_table_2<MaskType, int32_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_KEY_64)
    case sizeof(int64_t):
      return make_nvhm_map_table_2<MaskType, int64_t>(id, config);
#endif
  }
  NVE_THROW_("`config.key_size` (", config.key_size, ") is out of bounds!");
}

host_table_ptr_t NvhmMapTableFactory::produce(const table_id_t id,
                                                   const NvhmMapTableConfig& config) {
  switch (config.mask_size) {
#if defined(NVE_FEATURE_HT_MASK_8)
    case bitmask8_t::size:
      return make_nvhm_map_table_1<bitmask8_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_MASK_16)
    case bitmask16_t::size:
      return make_nvhm_map_table_1<bitmask16_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_MASK_32)
    case bitmask32_t::size:
      return make_nvhm_map_table_1<bitmask32_t>(id, config);
#endif
#if defined(NVE_FEATURE_HT_MASK_64)
    case bitmask64_t::size:
      return make_nvhm_map_table_1<bitmask64_t>(id, config);
#endif
  }
  NVE_THROW_("`config.mask_size` (", config.mask_size, ") is out of bounds!");
}

}  // namespace plugin
}  // namespace nve
