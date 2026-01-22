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

#include <cuda_runtime.h>
#include <common.hpp>
#include <cstdint>
#include <memory>
#include <mutex>
#include <default_allocator.hpp>
#include <sstream>
#include <table.hpp>
#include <ecache/embedding_cache_combined.h>
#include <nve_types.hpp>

namespace nve {

class ExecutionContext;
class ContextRegistry;

enum class KernelType : uint64_t {
    DynamicKernel = 0,
    LookupUVM = 1,
    SortGather = 2,
    PipelineGather = 3,
  };

struct GPUTableConfig {
  int device_id{0};                         // Device id of the GPU used
  size_t cache_size;                        // Total size of storage in bytes
  //TRTREC-78: relevant also here,need to correct documentation for managed host cache case!
  int64_t row_size_in_bytes;                // Size in bytes for each cache row. Must divide by 2 atm (for fp16
                                            // loads)
  void* uvm_table{nullptr};                 // Optional pointer to linear table in UVM.
                                            // This pointer will be used to resolve misses (no
                                            // reolution when empty/null) Can be in GPU or host memory.
  bool count_misses{true};                  // When true, create lookup contexts will collect miss count
                                            // Counter is needed for Insert Heuristics, disable only when not using it.
  int64_t max_modify_size{1 << 20};         // Maximal amount of modify entries allowed in a single op (insert/update/accumulate)
                                            // Using a small amount here can cause modify ops to be less efficient.
  DataType_t value_dtype{
      DataType_t::Unknown};                 // Storage data type of the table values. Only used for accumulate
  cudaStream_t private_stream{0};           // When non-zero will be used for all loookup/modify ops.
                                            // Execution contexts' stream will be synchronized with the private stream.
  bool disable_uvm_update{false};           // When true, update/update_accumulate will not update the uvm table.
                                            // Use this when multipe GPU Tables use the same uvm_table (only one should update it)
  bool uvm_cpu_accumulate{true};            // When true, update_accumulate will perform the accumulation for the uvm_table
                                            // on the cpu (instead of a GPU kernel)
  bool data_storage_on_host{false};         // When true, cache data is allocated on host (tags remain on device)
  bool modify_on_gpu{true};                 // When true, cache modifying operations (insert, update, update accumulate etc.)
                                            // are performed on GPU and do not manage cache metadata on host                            
  uint64_t kernel_mode_type{0};             // Force kernel mode type, 0 means use defaults
  uint64_t kernel_mode_value{0};            // Interpretation depends on kernel mode type
};
void from_json(const nlohmann::json& json, GPUTableConfig& conf);
void to_json(nlohmann::json& json, const GPUTableConfig& conf);

template <typename KeyType>
class GpuTable : public Table {
 public:
  using base_type = Table;
  using key_type = KeyType;
  using CacheType = nve::EmbedCacheSA<KeyType, KeyType>;
  using config_type = nve::GPUTableConfig;

  NVE_PREVENT_COPY_AND_MOVE_(GpuTable);

  /**
   * Create a Device database.
   *
   * @param config The database configuration used to initialize the embedding cache.
   * @param allocator Allocator for large memory allocations. nullptr implies using the nve::DefaultAllocator
   */
  GpuTable(const GPUTableConfig& config, allocator_ptr_t allocator = {});

  ~GpuTable() override;

  /**
   * Clear all entries in the database.
   *
   * @param ctx An execution context for this database.
   */
  void clear(context_ptr_t& ctx) override;

  /**
   * Erase a set of entries from the database.
   *
   * @param ctx An execution context for this database.
   * @param num_keys The number of entries to erase.
   * @param keys An array of entry keys to erase.
   * This array should reside in host memory when modify_on_gpu was set to true.
   */
  void erase(context_ptr_t& ctx, int64_t num_keys, const void* keys) override;

  /**
   * Find entries for a given set of keys.
   *
   * @param ctx An execution context for this database.
   * @param num_keys The number of given keys.
   * @param keys An array of keys to find in the database.
   * This array should reside in UVM (preferably in GPU memory).
   * @param hit_mask [Input/Output] An array to write the hit mask for the given keys. The hit mask is of
   * size roundup(n/64) int64 elements. Where the i'th bit equals 1 iff the i'th key was found in
   * the device database. This array should reside in UVM (preferably in GPU memory).
   * @param value_stride The number of bytes between every entry in the output buffer ("values")
   * @param values [Output] An array to write the retrieved entries to. Must be large enough to hold
   * num_keys entries (considering value_stride) This array should reside in UVM (preferably in GPU
   * memory).
   * @param value_sizes Must be nullptr (not supported)
   */
  void find(context_ptr_t& ctx, int64_t num_keys, const void* keys, max_bitmask_repr_t* hit_mask,
               int64_t value_stride, void* values, int64_t* value_sizes) const override;

  /**
   * Insert new key-value pairs to the databases.
   * This operation can evict other keys-value from the database.
   * This operation does not guarantee all key-value pairs given will be successfully inserted (up
   * to implementation limitations).
   *
   * @param ctx An execution context for this database.
   * @param num_keys The number of given keys.
   * @param keys An array of entry keys.
   * This array should reside in host memory.
   * @param value_stride The number of bytes between every entry in the buffer ("values")
   * @param values An array of values to read entries from.
   * This array should reside in GPU accessible memory.
   */
  void insert(context_ptr_t& ctx, int64_t num_keys, const void* h_keys, int64_t value_stride,
              int64_t value_size, const void* values) override;

  /**
   * Update (overwrite) values for a given set of keys iff they already exist in the database.
   * Values for keys not available in the database will be ignored.
   *
   * @param ctx An execution context for this database.
   * @param num_keys The number of given keys.
   * @param keys An array of entry keys.
   * This array should reside in host memory when modify_on_gpu was set to true.
   * @param update_stride The number of bytes between every entry in the buffer ("values")
   * @param update_size The size in bytes of an update vector
   * @param updates An array of values to read entries from.
   * This array should reside in GPU accessible memory.
   */
  void update(context_ptr_t& ctx, int64_t num_keys, const void* keys, int64_t update_stride,
              int64_t update_size, const void* updates) override;

  /**
   * Update (accumulate) values for a given set of keys iff they already exist in the database.
   * Values for keys not available in the database will be ignored.
   *
   * @param ctx An execution context for this database.
   * @param num_keys The number of given keys.
   * @param keys An array of entry keys.
   * This array should reside in GPU accessible memory when using private_stream or modify_on_gpu was set to false.
   * Otherwise it should reside in host memory.
   * @param update_stride The number of bytes between every entry in the buffer ("values")
   * @param update_size The size in bytes of an update vector
   * @param updates An array of values to read entries from.
   * This array should reside in GPU accessible memory.
   * @param update_dtype The data type of the updates (can be different from the table data type).
   */
  void update_accumulate(context_ptr_t& ctx, int64_t num_keys, const void* keys, int64_t update_stride,
                         int64_t update_size, const void* updates,
                         DataType_t update_dtype) override;

  /**
   * Find entries in the database and combine them.
   * This operation is only supported for databases that were initialized with a linear UVM table.

   * @param ctx An execution context for this database.
   * @param num_keys The number of given keys.
   * @param keys An array of keys to find in the database.
   * This array should reside in UVM (preferably in GPU memory).
   * @param hot_type The type of hotness used for grouping keys (e.g. Fixed, CSR, COO).
   * @param num_offsets The number of offset values in "offsets".
   * @param offsets The offsets used to group keys. Structure will change depending on hotness type.
   * If hotness type is Fixed, this array is ignored.
   * This array should reside in UVM (preferably in GPU memory).
   * @param fixed_hotness The hotness value (i.e. bag size) to be used with Fixed hotness.
   * If hotness isn't Fixed, this value is ignored.
   * @param pooling_type The type of combiner to use on entries in the same group.
   * Note that for the mode "Concat" it is more efficient to use the find() call instead.
   * @param weights An array of weights to use with weighted combiners.
   * This array should reside in UVM (preferably in GPU memory).
   * @param value_stride The number of bytes between every entry in the output buffer ("values")
   * @param values [Output] An array to write the retrieved entries to. Must be large enough to hold
   n entries (considering value_stride)
   * This array should reside in UVM (preferably in GPU memory).
  */

  template <typename OffsetType = KeyType,    // Type used for Offset during lookup of COO/CSR
            typename ValueType = float,       // Type used for the data vectors
            typename OutputType = ValueType,  // Type used for output data vectors (can differ from
                                              // ValueType only when combining multiple rows)
            typename WeightType =
                float>  // Type used for weights used by some combiner types (e.g. weighted sum)
  void find_and_combine(context_ptr_t& ctx, int64_t num_keys, const void* keys, SparseType_t sparse_type,
                        int64_t num_offsets, const OffsetType* offsets, int64_t fixed_hotness,
                        PoolingType_t pooling_type, const WeightType* weights, int64_t value_stride,
                        OutputType* values);

  const GPUTableConfig& config() const { return config_; }
  allocator_ptr_t get_allocator() { return allocator_; }

  /**
   * Reset the lookup key counter for the given context
   * @param ctx Execution context for which lookup keys are collected.
   */
  void reset_lookup_counter(context_ptr_t& ctx) override;
  /**
   * Get the lookup key counter for the given context
   * @param ctx Execution context for which lookup keys are collected.
   * @param counter Address of the counter to be filled.
   * GPU tables count lookup key misses, so values expected values will be <=0
   * @note GPU tables use the context's lookup stream and require a sync call before reading counter.
   * (e.g. "cudaStreamSynchronize(ctx->get_lookup_stream())")
   */
  void get_lookup_counter(context_ptr_t& ctx, int64_t* counter) override;
  /**
   * Checks whether the key count refers to lookup hits or misses
   * @return True iff the table counts hits, otherwise the table counts misses.
   */
  bool lookup_counter_hits() override;

  /**
   * Create an execution context for the table.
   * An execution context will hold resources needed for a single operation and is reusable.
   * The same context cannot be used by multiple threads at the same time.
   * @param lookup_stream A CUDA stream to use for lookup operations (find/find_and_combine)
   * @param modify_stream A CUDA stream to use for modify operations (update/acumulate/erase/...)
   * @param thread_pool A thread pool to use for CPU work
   * @param allocator An allocator to use for allocating temporary resources in CPU/GPU memory
   * 
   * @note In private stream mode, using a different stream as a lookup/modify stream will trigger additional CUDA synchronzations
   */
  virtual context_ptr_t create_execution_context(
    cudaStream_t lookup_stream,
    cudaStream_t modify_stream,
    thread_pool_ptr_t thread_pool,
    allocator_ptr_t allocator) override;

  /**
   * @returns GPU device id that's used for execution or -1 for CPU.
   */
  virtual int32_t get_device_id() const override;
  /**
   * @returns Maximal supported row size in bytes.
   */
  virtual int64_t get_max_row_size() const override;

  virtual void erase_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys) override;
  virtual void find_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, buffer_ptr<max_bitmask_repr_t> hit_mask,
                       int64_t value_stride, buffer_ptr<void> values, buffer_ptr<int64_t> value_sizes) const override;
  virtual void insert_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, int64_t value_stride,
                      int64_t value_size, buffer_ptr<const void> values) override;
  virtual void update_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys, int64_t value_stride,
                         int64_t value_size, buffer_ptr<const void> values) override;
  virtual void update_accumulate_bw(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys,
                                 int64_t update_stride, int64_t update_size, buffer_ptr<const void> updates,
                                 DataType_t update_dtype) override;

  template <typename OffsetType = KeyType,    // Type used for Offset during lookup of COO/CSR
            typename ValueType = float,       // Type used for the data vectors
            typename OutputType = ValueType,  // Type used for output data vectors (can differ from
                                              // ValueType only when combining multiple rows)
            typename WeightType =
                float>  // Type used for weights used by some combiner types (e.g. weighted sum)
  void find_and_combine_bw(context_ptr_t& ctx, int64_t num_keys, buffer_ptr<const void> keys, SparseType_t sparse_type,
                        int64_t num_offsets, buffer_ptr<const OffsetType> offsets, int64_t fixed_hotness,
                        PoolingType_t pooling_type, buffer_ptr<const WeightType> weights, int64_t value_stride,
                        buffer_ptr<void> values);

 private:
  const GPUTableConfig config_;

  allocator_ptr_t allocator_;
  std::shared_ptr<CacheType> cache_;

  std::shared_ptr<ContextRegistry> contexts_;
  std::shared_ptr<DefaultECEvent> create_sync_event();

  mutable std::mutex uvm_table_mutex_;
};

}  // namespace nve
