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

#include <nve_types.hpp>
#include <execution_context.hpp>

namespace nve {

template<typename T>
class BufferWrapper;

class Table {
 public:
  NVE_PREVENT_COPY_AND_MOVE_(Table);

  Table();

  virtual ~Table();

  /**
   * Create an execution context to use with this table.
   * An execution context holds resources needed for a single parallel run and is reusable.
   * So multiple execution contexts can be used at the same time, but at any given time a specific 
   * context is only used once.
   * @param lookup_stream CUDA stream to use for lookup operations
   * @param modify_stream CUDA stream to use for modify ops (e.g. update, update_accumulate, clear etc.)
   * @param thread_pool ThreadPool to use for CPU work, nullptr implies the default global thread pool
   * @param allocator Allocator to use for large buffer allocations, nullptr implies the default global allocator.
   * @note All execution contexts created for a table must be destroyed before the table is destroyed.
   */
  virtual context_ptr_t create_execution_context(
    cudaStream_t lookup_stream,
    cudaStream_t modify_stream,
    thread_pool_ptr_t thread_pool,
    allocator_ptr_t allocator);

  /**
   * Empties the table. Erases all entries.
   *
   * @param ctx Execution context to use if additional resources are needed to run the command.
   */
  virtual void clear(context_ptr_t& ctx) = 0;

  /**
   * Erases the provided keys from the table.
   *
   * @param ctx Execution context to use if additional resources are needed to run the command.
   * @param n Number of keys in `keys`.
   * @param keys Pointer to an array of `n` keys.
   */
  virtual void erase(context_ptr_t& ctx, int64_t n, const void* keys) = 0;

  /**
   * Looks up the values for the provided keys if they exist in the table. Can be reduced to a
   * counting/key_exists checks by setting `values` to `nullptr`.
   *
   * @param ctx Execution context to use if additional resources are needed to run the command.
   * @param n The number of keys in `keys`.
   * @param keys Pointer to the `n` key values.
   * @param hit_mask A pointer to a mutable array of `n` bits (`(n +
   * max_bitmask_t::num_bits) / max_bitmask_t::num_bits` bytes)` bytes), last uint64_t must be zero
   * padded). If present the hit-mask will indicate which keys need to be retrieved. If it is a
   * `nullptr`, the layer assumes a zero-initialized hit-mask.
   * @param value_stride Spacing / stride between two values in `values`.
   * @param values Either `nullptr` or pointer to an array with room for at least `n * value_stride`
   * bytes of data. The order of values corresponds to the sequence of non-active bits in the
   * hit-mask. Hence, index 3 would coresspond ot the third bit that was `0` when calling the
   * function. provided the indices pointer will contain the coresponding key's index. Any key where
   * the hit-mask is still `0` will be subsequently written here.
   * @param value_sizes Either `nullptr` or pointer to an array of `n` 64 bit integers. If present,
   * `find` will record the actual size of each value (i.e, the respective number of bytes copied to
   * `values`).
   */
  virtual void find(context_ptr_t& ctx, int64_t n, const void* keys, max_bitmask_repr_t* hit_mask,
                    int64_t value_stride, void* values, int64_t* value_sizes) const = 0;

  /**
   * Insert key/value pairs into the table. If a key already exists, its value may be replaced
   * or ignored.
   *
   * Note: This is a best effort operation. If an overflow condition is triggered during the
   * insertion procedure, some of the freshly inserted keys might immediately be evicted in
   * accordance with the selected overflow resolution strategy.
   *
   * @param ctx Execution context to use if additional resources are needed to run the command.
   * @param n Number of keys in `keys`.
   * @param keys Pointer to an array of `n` keys.
   * @param value_stride Spacing / raster between two values in `values`.
   * @param value_size Size of each value in bytes.
   * @param values Pointer to an array containing at least `n * value_stride` bytes.
   */
  virtual void insert(context_ptr_t& ctx, int64_t n, const void* keys, int64_t value_stride,
                      int64_t value_size, const void* values) = 0;

  /**
   * Replaces the value if - and only if - the corresponding key already exists in the table.
   *
   * @param ctx Execution context to use if additional resources are needed to run the command.
   * @param n Number of keys in `keys`.
   * @param keys Pointer to an array of `n` keys.
   * @param value_stride Spacing / raster between two values in `values`.
   * @param value_size Size of each value in bytes.
   * @param values Pointer to an array containing at least `n * value_stride` bytes.
   */
  virtual void update(context_ptr_t& ctx, int64_t n, const void* keys, int64_t value_stride,
                      int64_t value_size, const void* values) = 0;

  /**
   * Accumulate the value if - and only if - the corresponding key already exists in the table.
   * i.e. the provided value should be added to the value already existing in the table for the
   * given key.
   *
   * @param ctx Execution context to use if additional resources are needed to run the command.
   * @param n Number of keys in `keys`.
   * @param keys Pointer to an array of `n` keys.
   * @param update_stride Spacing / raster between two values in `values`.
   * @param update_size Size of each value in bytes (should be a multiple of
   * `dtype_size(update_dtype)`).
   * @param updates Pointer to an array containing at least `n * value_stride` bytes.
   * @param update_dtype Data type of the provided values. This type may be different from the
   * table's storage format. Not all combinations of storage dtype and update dtype may be supported
   * by each table implementation.
   */
  virtual void update_accumulate(context_ptr_t& ctx, int64_t n, const void* keys,
                                 int64_t update_stride, int64_t update_size, const void* updates,
                                 DataType_t update_dtype) = 0;

  /**
   * Reset the lookup key counter for the given context
   * @param ctx Execution context for which lookup keys are collected.
   */
  virtual void reset_lookup_counter(context_ptr_t& ctx) = 0;

  /**
   * Get the lookup key counter for the given context
   * @param ctx Execution context for which lookup keys are collected.
   * @param counter Address of the counter to be filled.
   * @note Tables using a lookup stream require a sync call before reading counter.
   */
  virtual void get_lookup_counter(context_ptr_t& ctx, int64_t* counter) = 0;

  /**
   * Checks whether the key count refers to lookup hits or misses
   * @return True iff the table counts hits, otherwise the table counts misses.
   */
  virtual bool lookup_counter_hits() = 0;

  // Getters for table properties used by layers. /////////////////////////////////////////////////
  /**
   * @returns GPU device id that's used for execution or -1 for CPU.
   * @note A table cannot change it's device id after initialization.
   */
  virtual int32_t get_device_id() const = 0;
  /**
   * @returns Maximal supported row size in bytes.
   */
  virtual int64_t get_max_row_size() const = 0;

  // BufferWrapper version with default fallback to normal buffer version.
  // These functions will be used by the layers and will delegate buffer management to the wrappers.
  template<typename T>
  using buffer_ptr = std::shared_ptr<BufferWrapper<T>>;
  virtual void erase_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys);
  virtual void find_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, buffer_ptr<max_bitmask_repr_t> hit_mask,
                       int64_t value_stride, buffer_ptr<void> values, buffer_ptr<int64_t> value_sizes) const;
  virtual void insert_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, int64_t value_stride,
                         int64_t value_size, buffer_ptr<const void> values);
  virtual void update_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, int64_t value_stride,
                         int64_t value_size, buffer_ptr<const void> values);
  virtual void update_accumulate_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys,
                                    int64_t update_stride, int64_t update_size, buffer_ptr<const void> updates,
                                    DataType_t update_dtype);
};

}  // namespace nve
