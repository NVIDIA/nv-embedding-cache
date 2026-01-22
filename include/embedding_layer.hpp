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
#include <vector>
#include <nve_types.hpp>
#include <execution_context.hpp>

namespace nve {

class ContextRegistry;

/**
 * Base class for an embedding layer (pure virtual).
 * Derived layers support varied configurations of tables (caches) and handle their interactions.
 * A layer will handle allocation of internal buffers needed (stored in the execution context) and
 * data transfers between them.
 */
class EmbeddingLayerBase {
 public:
  NVE_PREVENT_COPY_AND_MOVE_(EmbeddingLayerBase);

  EmbeddingLayerBase() = default;
  virtual ~EmbeddingLayerBase() = default;

  /**
   * This struct defines the parameters needed to perform pooling during a lookup call.
   */
  struct PoolingParams {
    PoolingType_t
        pooling_type;  // Pooling type. Concatenate means no pooling (i.e. rest will be ignored).

    // SparseType     | key_indices
    // Fixed Hotness  | 1 value for the hotness (or batch size)
    // CSR            | 1 value per batch (bag) + 1 for the last offset
    // COO            | 2 values perf key (bag_id, id_in_bag) - assumed to be sorted row_wise
    SparseType_t sparse_type;
    const int64_t* key_indices;
    int64_t num_key_indices;

    void* default_values;  // Single vector (row size) to use when a key was missed by all caches,
                           // null implies pooling isn't allowed with misses
    const void* sparse_weights;  // Weights for weighted_sum pooling
    DataType_t weight_type;  // Datatype for the provided weights (doesn't have to be the same as
                             // Value type, not all combinations supported)
  };

  /**
   * Main lookup function. For each given key, return the associated data vector.
   * @param ctx An execution context to use.
   * @param num_keys Number of keys to query
   * @param keys Array of keys
   * @param output Output buffer for the resolved datavectors
   * @param output_stride Number of bytes between each datavector in output
   * @param hitmask Buffer to denote each key's successful resolution. Bit i will be 1 iff it was
   * successfully resolved (undefined for i >= num_keys). nullptr implies lookup doesn't populate
   * this buffer.
   * @param pool_params Parameters required for pooling. nullptr implies no pooling (datavector are
   * concatenated)
   * @param hitrates Array of hit rates for each internal table calculated as (#resolved keys /
   * num_keys) nullptr implies no hitrates will be reported. Collecting hitrates may cause synchronization
   * for GPU tables.
   */
  virtual void lookup(
      context_ptr_t& ctx,                // execution context
      const int64_t num_keys,           // number of queried keys per table
      const void* keys,                 // input keys
      void* output,                     // embedding vector output buffer per table
      const int64_t output_stride,      // row stride per output buffer
      max_bitmask_repr_t* hitmask,      // bitmask where the i'th bit is 1 iff it was resolved by the
                                        // lookup. null implies no hitmask result is required
      const PoolingParams* pool_params, // Pooling params, null implies no pooling (i.e. concat)
      float* hitrates                   // array of hitrates achieved for each table [device,host,remote]
                                        // hitrate[i] := float(hits_for_table_i) / num_keys
                                        // Must have at least one float per table in the layer
      ) = 0;

  /**
   * Insert new key-vector pairs by examining a representative dataset.
   * This function will analyze a given set of keys and decide which should be added to the given
   * table. Not all given keys are guaranteed to be added.
   * @warning Do not use insert() instead of update() - if a key used for insert is already in the
   * table specified, it's datavector may be ignored.
   * @param ctx An execution context to use.
   * @param num_keys Number of keys to consider
   * @param keys Array of keys
   * @param value_stride Number of bytes between each datavector in output
   * @param value_size Size of each datavector in values
   * @param values Array of datavectors
   * @param table_id Index of the table to perfrom insert on
   */
  virtual void insert(context_ptr_t& ctx,
                      const int64_t num_keys,      // number of keys
                      const void* keys,            // input keys
                      const int64_t value_stride,  // stride in the values buffer
                      const int64_t value_size,    // size of each value
                      const void* values,          // data vector array to insert
                      const int64_t table_id       // index fo table to insert to
                      ) = 0;

  /**
   * Update existing keys with new datavectors.
   * This function will search each table for the given keys and if a key exists its' datavector
   * will be updated (overwritten). Note that this function does not change the residency of tables
   * (which key is stored where).
   * @param ctx An execution context to use.
   * @param num_keys Number of keys to consider
   * @param keys Array of keys
   * @param value_stride Number of bytes between each datavector in output
   * @param value_size Size of each datavector in values
   * @param values Array of datavectors
   */
  virtual void update(context_ptr_t& ctx,
                      const int64_t num_keys,       // number of keys per table
                      const void* keys,             // input keys
                      const int64_t vector_stride,  // stride in the values buffer
                      const int64_t value_size,     // size of each value
                      const void* values            // data vector array to update
                      ) = 0;

  /**
   * Accumulate gradients into existing keys' datavectors.
   * This function will search all tables for the given keys and if a key exists its'
   * datavector will be increased with the given value (gradient). Note that this function does not
   * change the residency of tables (which key is stored where).
   * @param ctx An execution context to use.
   * @param num_keys Number of keys to consider
   * @param keys Array of keys
   * @param value_stride Number of bytes between each datavector in output
   * @param value_size Size of each datavector in values
   * @param values Array of datavectors (gradients)
   * @param value_type Datatype of the gradients given in vales (can be different from the datatype
   * used in the tables).
   */
  virtual void accumulate(
      context_ptr_t& ctx,
      const int64_t num_keys,       // number of keys per table
      const void* keys,             // input keys
      const int64_t vector_stride,  // stride in the values buffer
      const int64_t value_size,     // size of each value
      const void* values,           // data vector array to accumulate (gradients)
      DataType_t value_type  // data type for values (can be different from the vaules in the layer
                             // - e.g. int8 update for fp16 table)
      ) = 0;

  /**
   * Clear all tables contents.
   * @warning this method is not synchronized in any way and expected to be called only when no
   * other ops are in progress (lookup, insert, etc.)
   * @note this may or may not reduce memory capacity used.
   * @param ctx An execution context to use.
   */
  virtual void clear(context_ptr_t& ctx) = 0;

  /**
   * Erases the provided keys from all tables.
   * @note keys not resident in the table will be ignored.
   * @param ctx An execution context to use.
   * @param num_keys Number of keys to erase
   * @param keys Array of keys
   * @param table_id Index of the table to erase from
   */
  virtual void erase(context_ptr_t& ctx,
                     const int64_t num_keys,  // number of keys per table
                     const void* keys,        // input keys
                     const int64_t table_id   // index fo table to erase from
                     ) = 0;

  /**
   * Create an execution context to use with this layer.
   * An execution context holds resources needed for a single parallel run and is reusable.
   * So multiple execution contexts can be used at the same time, but at any given time a specific 
   * context is only used once.
   * @param lookup_stream CUDA stream to use for lookup operations
   * @param modify_stream CUDA stream to use for modify ops (e.g. update, update_accumulate, clear etc.)
   * @param thread_pool ThreadPool to use for CPU work, nullptr implies the default global thread pool
   * @param allocator Allocator to use for large buffer allocations, nullptr implies using the allocator the layer was initialized with.
   * @warning All execution context ptrs created for an embedding layer must be wait()'ed and released before the layer is destroyed.
   */
  virtual context_ptr_t create_execution_context(
    cudaStream_t lookup_stream,
    cudaStream_t modify_stream,
    thread_pool_ptr_t thread_pool,
    allocator_ptr_t allocator) = 0;
};

}  // namespace nve
