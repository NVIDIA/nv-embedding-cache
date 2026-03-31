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

#ifndef NVE_C_API_H
#define NVE_C_API_H

#include <stdint.h>
#include <stddef.h>

/* Visibility / export macro */
#if defined(_WIN32)
  #error "Win32 build isn't supported"
#else
  #define NVE_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Status codes
 * ============================================================================ */

typedef enum {
  NVE_SUCCESS = 0,
  NVE_ERROR_INVALID_ARGUMENT = 1,
  NVE_ERROR_CUDA = 2,
  NVE_ERROR_RUNTIME = 3,
  NVE_ERROR_NOT_IMPLEMENTED = 4,
  NVE_ERROR_OUT_OF_MEMORY = 5,
} nve_status_t;

/**
 * Retrieve the error message from the last failed C API call on this thread.
 * @param message Output pointer set to a thread-local null-terminated string.
 *                Valid until the next C API call on this thread.
 * @return NVE_SUCCESS
 */
NVE_C_API nve_status_t nve_get_last_error(const char** message);

/* ============================================================================
 * Enumerations
 * ============================================================================ */

typedef enum {
  NVE_DTYPE_UNKNOWN = 0,
  NVE_DTYPE_FLOAT32,
  NVE_DTYPE_BFLOAT16,
  NVE_DTYPE_FLOAT16,
  NVE_DTYPE_E4M3,
  NVE_DTYPE_E5M2,
  NVE_DTYPE_FLOAT64,
} nve_data_type_t;

typedef enum {
  NVE_SPARSE_FIXED = 0,
  NVE_SPARSE_CSR,
  NVE_SPARSE_COO,
} nve_sparse_type_t;

typedef enum {
  NVE_POOL_CONCATENATE = 0,
  NVE_POOL_SUM,
  NVE_POOL_MEAN,
  NVE_POOL_WEIGHTED_SUM,
  NVE_POOL_WEIGHTED_MEAN,
} nve_pooling_type_t;

typedef enum {
  NVE_PART_ALWAYS_ZERO = 0,
  NVE_PART_FOWLER_NOLL_VO,
  NVE_PART_MURMUR3,
  NVE_PART_RRXMRRXMSX0,
  NVE_PART_STD_HASH,
} nve_partitioner_t;

typedef enum {
  NVE_OVERFLOW_EVICT_RANDOM = 0,
  NVE_OVERFLOW_EVICT_LRU,
  NVE_OVERFLOW_EVICT_LFU,
} nve_overflow_handler_t;

typedef enum {
  NVE_KEY_INT32 = 0,
  NVE_KEY_INT64,
} nve_key_type_t;

typedef enum {
  NVE_KERNEL_DYNAMIC = 0,
  NVE_KERNEL_LOOKUP_UVM,
  NVE_KERNEL_SORT_GATHER,
  NVE_KERNEL_PIPELINE_GATHER,
} nve_kernel_type_t;

typedef enum {
  NVE_HEURISTIC_DEFAULT = 0,
  NVE_HEURISTIC_FSM,
  NVE_HEURISTIC_STATISTICAL,
} nve_heuristic_type_t;

/* ============================================================================
 * Opaque handle types
 * ============================================================================ */

typedef struct nve_context_s*       nve_context_t;
typedef struct nve_table_s*         nve_table_t;
typedef struct nve_layer_s*         nve_layer_t;
typedef struct nve_thread_pool_s*   nve_thread_pool_t;
typedef struct nve_allocator_s*     nve_allocator_t;
typedef struct nve_heuristic_s*     nve_heuristic_t;
typedef struct nve_host_factory_s*  nve_host_factory_t;

/* ============================================================================
 * Configuration structs
 * ============================================================================ */

typedef struct {
  int32_t  device_id;
  uint64_t cache_size;
  int64_t  row_size_in_bytes;
  void*    uvm_table;               /* Optional UVM backing table, NULL for none */
  int      count_misses;            /* bool: collect miss count for heuristics */
  int64_t  max_modify_size;
  nve_data_type_t value_dtype;
  void*    private_stream;          /* cudaStream_t, 0/NULL for default */
  int      disable_uvm_update;      /* bool */
  int      uvm_cpu_accumulate;      /* bool */
  int      data_storage_on_host;    /* bool */
  int      modify_on_gpu;           /* bool */
  uint64_t kernel_mode_type;
  uint64_t kernel_mode_value;
} nve_gpu_table_config_t;

typedef struct {
  const char* layer_name;
  int32_t     device_id;
  void*       embedding_table;       /* Pointer to linear table in GPU memory */
  int64_t     num_embeddings;
  int64_t     embedding_width_in_bytes;
  nve_data_type_t value_dtype;
} nve_gpu_embedding_layer_config_t;

typedef struct {
  const char*     layer_name;
  nve_heuristic_t insert_heuristic;  /* NULL uses default heuristic; use nve_heuristic_create_never() to disable */
  int64_t         min_insert_freq_gpu;
  int64_t         min_insert_size_gpu;
} nve_linear_uvm_layer_config_t;

typedef struct {
  const char*     layer_name;
  nve_heuristic_t insert_heuristic;  /* NULL uses default heuristic; use nve_heuristic_create_never() to disable */
  int64_t         min_insert_freq_gpu;
  int64_t         min_insert_freq_host;
  int64_t         min_insert_size_gpu;
  int64_t         min_insert_size_host;
} nve_hierarchical_layer_config_t;

typedef struct {
  int64_t              overflow_margin;
  nve_overflow_handler_t handler;
  double               resolution_margin;
} nve_overflow_policy_config_t;

typedef struct {
  int64_t         mask_size;
  int64_t         key_size;
  int64_t         max_value_size;
  nve_data_type_t value_dtype;
} nve_host_table_config_t;

/* ============================================================================
 * Config initializers (provide sane defaults matching C++ defaults)
 * ============================================================================ */

NVE_C_API nve_gpu_table_config_t             nve_gpu_table_config_default(void);
NVE_C_API nve_gpu_embedding_layer_config_t   nve_gpu_embedding_layer_config_default(void);
NVE_C_API nve_linear_uvm_layer_config_t      nve_linear_uvm_layer_config_default(void);
NVE_C_API nve_hierarchical_layer_config_t    nve_hierarchical_layer_config_default(void);
NVE_C_API nve_overflow_policy_config_t       nve_overflow_policy_config_default(void);
NVE_C_API nve_host_table_config_t            nve_host_table_config_default(void);

/* ============================================================================
 * Version
 * ============================================================================ */

NVE_C_API nve_status_t nve_version(int32_t* major, int32_t* minor, int32_t* patch);

/* ============================================================================
 * Plugin loading
 * ============================================================================ */

NVE_C_API nve_status_t nve_load_host_table_plugin(const char* plugin_name);

/* ============================================================================
 * Thread Pool
 * ============================================================================ */

/**
 * Create a thread pool from JSON configuration.
 * @param out Output handle.
 * @param json_config JSON string with thread pool configuration.
 */
NVE_C_API nve_status_t nve_thread_pool_create(nve_thread_pool_t* out, const char* json_config);
NVE_C_API nve_status_t nve_thread_pool_destroy(nve_thread_pool_t pool);
NVE_C_API nve_status_t nve_thread_pool_num_workers(nve_thread_pool_t pool, int64_t* out);
NVE_C_API nve_status_t nve_configure_default_thread_pool(const char* json_config);

/* ============================================================================
 * Insert Heuristics
 * ============================================================================ */

NVE_C_API nve_status_t nve_heuristic_create_default(
    nve_heuristic_t* out, const float* thresholds, int64_t num_thresholds);
NVE_C_API nve_status_t nve_heuristic_create_never(nve_heuristic_t* out);
NVE_C_API nve_status_t nve_heuristic_destroy(nve_heuristic_t heuristic);

/* ============================================================================
 * Execution Context
 * ============================================================================ */

NVE_C_API nve_status_t nve_context_destroy(nve_context_t ctx);
NVE_C_API nve_status_t nve_context_wait(nve_context_t ctx);
NVE_C_API nve_status_t nve_context_get_lookup_stream(nve_context_t ctx, void** out_stream);
NVE_C_API nve_status_t nve_context_get_modify_stream(nve_context_t ctx, void** out_stream);

/* ============================================================================
 * Tables (GPU and Host -- polymorphic via Table base class)
 * ============================================================================ */

/**
 * Create a GPU table.
 * @param out Output handle.
 * @param key_type Key type (NVE_KEY_INT32 or NVE_KEY_INT64).
 * @param config GPU table configuration.
 * @param allocator Optional allocator handle, NULL for default.
 */
NVE_C_API nve_status_t nve_gpu_table_create(
    nve_table_t* out, nve_key_type_t key_type,
    const nve_gpu_table_config_t* config, nve_allocator_t allocator);

NVE_C_API nve_status_t nve_table_destroy(nve_table_t table);

NVE_C_API nve_status_t nve_table_find(
    nve_table_t table, nve_context_t ctx, int64_t n, const void* keys,
    uint64_t* hit_mask, int64_t value_stride, void* values, int64_t* value_sizes);

NVE_C_API nve_status_t nve_table_insert(
    nve_table_t table, nve_context_t ctx, int64_t n, const void* keys,
    int64_t value_stride, int64_t value_size, const void* values);

NVE_C_API nve_status_t nve_table_update(
    nve_table_t table, nve_context_t ctx, int64_t n, const void* keys,
    int64_t value_stride, int64_t value_size, const void* values);

NVE_C_API nve_status_t nve_table_update_accumulate(
    nve_table_t table, nve_context_t ctx, int64_t n, const void* keys,
    int64_t update_stride, int64_t update_size, const void* updates,
    nve_data_type_t update_dtype);

NVE_C_API nve_status_t nve_table_clear(nve_table_t table, nve_context_t ctx);

NVE_C_API nve_status_t nve_table_erase(
    nve_table_t table, nve_context_t ctx, int64_t n, const void* keys);

NVE_C_API nve_status_t nve_table_create_execution_context(
    nve_table_t table, nve_context_t* out,
    void* lookup_stream, void* modify_stream,
    nve_thread_pool_t thread_pool, nve_allocator_t allocator);

NVE_C_API nve_status_t nve_table_get_device_id(nve_table_t table, int32_t* out);
NVE_C_API nve_status_t nve_table_get_max_row_size(nve_table_t table, int64_t* out);

NVE_C_API nve_status_t nve_table_reset_lookup_counter(nve_table_t table, nve_context_t ctx);
NVE_C_API nve_status_t nve_table_get_lookup_counter(nve_table_t table, nve_context_t ctx, int64_t* counter);

/* ============================================================================
 * Host Tables (via plugins / JSON config)
 * ============================================================================ */

NVE_C_API nve_status_t nve_create_host_table_factory(
    nve_host_factory_t* out, const char* json_config);
NVE_C_API nve_status_t nve_host_factory_destroy(nve_host_factory_t factory);

/**
 * Produce a host table from a factory.
 * @param factory Factory handle created by nve_create_host_table_factory.
 * @param table_id Numeric ID for the table.
 * @param json_config JSON string with table configuration (mask_size, key_size, etc.).
 * @param out Output table handle.
 */
NVE_C_API nve_status_t nve_host_factory_produce(
    nve_host_factory_t factory, int64_t table_id,
    const char* json_config, nve_table_t* out);

/**
 * Build a complete host database from JSON configuration.
 * Loads plugins, creates factories, and produces tables as specified.
 * @param json_config JSON string with host_database configuration.
 * @param out_tables Output array of table handles (caller must free with nve_free_host_database).
 * @param out_ids Output array of table IDs (parallel to out_tables).
 * @param out_count Number of tables created.
 */
NVE_C_API nve_status_t nve_build_host_database(
    const char* json_config,
    nve_table_t** out_tables, int64_t** out_ids, int64_t* out_count);
NVE_C_API nve_status_t nve_free_host_database(
    nve_table_t* tables, int64_t* ids, int64_t count);

NVE_C_API nve_status_t nve_host_table_size(
    nve_table_t table, nve_context_t ctx, int exact, int64_t* out);

/* ============================================================================
 * Embedding Layers
 * ============================================================================ */

/**
 * Create a GPU embedding layer (no caching, linear table in GPU memory).
 */
NVE_C_API nve_status_t nve_gpu_embedding_layer_create(
    nve_layer_t* out, nve_key_type_t key_type,
    const nve_gpu_embedding_layer_config_t* config, nve_allocator_t allocator);

/**
 * Create a Linear UVM embedding layer (GPU cache + UVM backup).
 * @param gpu_table A GPU table handle to use as the cache.
 */
NVE_C_API nve_status_t nve_linear_uvm_layer_create(
    nve_layer_t* out, nve_key_type_t key_type,
    const nve_linear_uvm_layer_config_t* config,
    nve_table_t gpu_table, nve_allocator_t allocator);

/**
 * Create a hierarchical embedding layer (GPU -> Host -> Remote).
 * @param tables Array of table handles, processed in order during lookup.
 * @param num_tables Number of tables.
 */
NVE_C_API nve_status_t nve_hierarchical_layer_create(
    nve_layer_t* out, nve_key_type_t key_type,
    const nve_hierarchical_layer_config_t* config,
    const nve_table_t* tables, int64_t num_tables, nve_allocator_t allocator);

NVE_C_API nve_status_t nve_layer_destroy(nve_layer_t layer);

/**
 * Lookup embeddings for the given keys.
 * @param hitmask Output bitmask (bit i = 1 iff key i was resolved). Can be NULL.
 * @param hitrates Output hit rates per internal table. Can be NULL.
 */
NVE_C_API nve_status_t nve_layer_lookup(
    nve_layer_t layer, nve_context_t ctx,
    int64_t num_keys, const void* keys,
    void* output, int64_t output_stride,
    uint64_t* hitmask, float* hitrates);

/**
 * Lookup with pooling/combining across bags.
 */
NVE_C_API nve_status_t nve_layer_lookup_pooled(
    nve_layer_t layer, nve_context_t ctx,
    int64_t num_keys, const void* keys,
    void* output, int64_t output_stride,
    uint64_t* hitmask,
    nve_pooling_type_t pooling_type, nve_sparse_type_t sparse_type,
    const int64_t* key_indices, int64_t num_key_indices,
    void* default_values, const void* sparse_weights,
    nve_data_type_t weight_type, float* hitrates);

NVE_C_API nve_status_t nve_layer_insert(
    nve_layer_t layer, nve_context_t ctx,
    int64_t num_keys, const void* keys,
    int64_t value_stride, int64_t value_size, const void* values,
    int64_t table_id);

NVE_C_API nve_status_t nve_layer_update(
    nve_layer_t layer, nve_context_t ctx,
    int64_t num_keys, const void* keys,
    int64_t value_stride, int64_t value_size, const void* values);

NVE_C_API nve_status_t nve_layer_accumulate(
    nve_layer_t layer, nve_context_t ctx,
    int64_t num_keys, const void* keys,
    int64_t value_stride, int64_t value_size, const void* values,
    nve_data_type_t value_type);

NVE_C_API nve_status_t nve_layer_clear(nve_layer_t layer, nve_context_t ctx);

NVE_C_API nve_status_t nve_layer_erase(
    nve_layer_t layer, nve_context_t ctx,
    int64_t num_keys, const void* keys, int64_t table_id);

NVE_C_API nve_status_t nve_layer_create_execution_context(
    nve_layer_t layer, nve_context_t* out,
    void* lookup_stream, void* modify_stream,
    nve_thread_pool_t thread_pool, nve_allocator_t allocator);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* NVE_C_API_H */
