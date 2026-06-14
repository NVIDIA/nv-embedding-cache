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
  /* Per-row symmetric-quantized int8 (int8 values + per-row scale, no offset), dequantized as
   * value*scale. The suffix is the scale precision: F32 = fp32 (4 bytes/row), F16 = fp16 (2
   * bytes/row). */
  NVE_DTYPE_QINT8_ROWWISE_F32,
  NVE_DTYPE_QINT8_ROWWISE_F16,
  /* Per-row affine-quantized uint8 (uint8 values + per-row scale & offset), dequantized as
   * value*scale + offset. The suffix is the scale/offset precision: F32 = fp32 (8 bytes/row),
   * F16 = fp16 (4 bytes/row). */
  NVE_DTYPE_QUINT8_ROWWISE_F32,
  NVE_DTYPE_QUINT8_ROWWISE_F16,
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
  int64_t  invalid_key;             /* Sentinel for invalid entries. Default -1; cast to the table's key type. */
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
  /* Default embedding row returned for keys missing from all tables.
   * NULL or default_embedding_size=0 disables the default (misses left undefined).
   * default_embedding_size, when non-zero, must equal the tables' row size. */
  const void*     default_embedding;
  int64_t         default_embedding_size;
} nve_hierarchical_layer_config_t;

typedef struct {
  const char* layer_name;
  /* Default embedding row returned for keys missing from the host table.
   * NULL or default_embedding_size=0 disables the default (misses left undefined).
   * default_embedding_size, when non-zero, must equal the host table row size. */
  const void* default_embedding;
  int64_t     default_embedding_size;
} nve_host_embedding_layer_config_t;

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
  int64_t         invalid_key;       /* Sentinel for invalid entries. Default -1; cast to the table's key type. */
} nve_host_table_config_t;

/* ============================================================================
 * Config initializers (provide sane defaults matching C++ defaults)
 * ============================================================================ */

/**
 * Return a GPU table config initialized to library defaults
 * (matches nve::GPUTableConfig{} from gpu_table.hpp). Callers should overwrite
 * required fields (e.g. cache_size, row_size_in_bytes) before use.
 */
NVE_C_API nve_gpu_table_config_t             nve_gpu_table_config_default(void);

/**
 * Return a GPU embedding-layer config initialized to library defaults
 * (matches nve::GPUEmbeddingLayerConfig{}). Caller must set layer_name,
 * embedding_table, num_embeddings, and embedding_width_in_bytes before use.
 */
NVE_C_API nve_gpu_embedding_layer_config_t   nve_gpu_embedding_layer_config_default(void);

/**
 * Return a Linear-UVM layer config initialized to library defaults
 * (matches nve::LinearUVMEmbeddingLayer::Config{}).
 */
NVE_C_API nve_linear_uvm_layer_config_t      nve_linear_uvm_layer_config_default(void);

/**
 * Return a hierarchical-layer config initialized to library defaults
 * (matches nve::HierarchicalEmbeddingLayer::Config{}).
 */
NVE_C_API nve_hierarchical_layer_config_t    nve_hierarchical_layer_config_default(void);

/**
 * Return a host embedding-layer config initialized to library defaults
 * (matches nve::HostEmbeddingLayer::Config{}).
 */
NVE_C_API nve_host_embedding_layer_config_t  nve_host_embedding_layer_config_default(void);

/**
 * Return an overflow-policy config initialized to library defaults
 * (matches nve::OverflowPolicyConfig{}: EvictRandom, overflow_margin=INT64_MAX
 * which disables overflow checks, resolution_margin=0.8).
 */
NVE_C_API nve_overflow_policy_config_t       nve_overflow_policy_config_default(void);

/**
 * Return a host-table config initialized to library defaults
 * (matches nve::HostTableConfig{}).
 */
NVE_C_API nve_host_table_config_t            nve_host_table_config_default(void);

/* ============================================================================
 * Version
 * ============================================================================ */

/**
 * Get the NVE library version (read from version.txt at build time, in
 * Year.Month.Patch form, e.g. 26.05.0).
 * @param major Output major version (year component).
 * @param minor Output minor version (month component).
 * @param patch Output patch version.
 */
NVE_C_API nve_status_t nve_version(int32_t* major, int32_t* minor, int32_t* patch);

/* ============================================================================
 * Plugin loading
 * ============================================================================ */

/**
 * Load a host-table plugin shared object.
 *
 * @param plugin_name Shared object name/path, for example
 *        "libnve-plugin-abseil.so" or "/tmp/my_plugin.so".
 */
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

/**
 * Destroy a thread pool. All work submitted to the pool must be drained before
 * calling this (the destructor joins worker threads).
 * @param pool Thread pool handle to destroy.
 */
NVE_C_API nve_status_t nve_thread_pool_destroy(nve_thread_pool_t pool);

/**
 * Return the total number of worker threads in the pool across all workgroups.
 * @param pool Thread pool handle to query.
 * @param out Output worker count.
 */
NVE_C_API nve_status_t nve_thread_pool_num_workers(nve_thread_pool_t pool, int64_t* out);

/**
 * Configure the process-wide default thread pool from a JSON description.
 * Any subsequent layer/table that's created without an explicit thread pool
 * will use this one. Must be called before the default pool is first used.
 * @param json_config JSON string describing the default pool configuration.
 */
NVE_C_API nve_status_t nve_configure_default_thread_pool(const char* json_config);

/* ============================================================================
 * Insert Heuristics
 * ============================================================================ */

/**
 * Create the default (threshold-based) insert heuristic.
 * For each lookup the heuristic decides whether to populate a table by
 * sampling a uniform random value and comparing against the per-table hitrate
 * threshold.
 * @param out Output handle.
 * @param thresholds Array of per-table thresholds in [0,1]. May be NULL when
 *        num_thresholds==0; in that case the default threshold (0.75) is used
 *        for every table.
 * @param num_thresholds Length of `thresholds`. Should equal the number of
 *        tables the heuristic will be applied to.
 */
NVE_C_API nve_status_t nve_heuristic_create_default(
    nve_heuristic_t* out, const float* thresholds, int64_t num_thresholds);

/**
 * Create a heuristic that never recommends auto-insert. Use this to disable
 * automatic cache population in a layer configuration.
 * @param out Output heuristic handle.
 */
NVE_C_API nve_status_t nve_heuristic_create_never(nve_heuristic_t* out);

/**
 * Destroy a heuristic handle. Safe to call after the heuristic has been
 * passed to a layer config — the layer takes its own reference.
 * @param heuristic Heuristic handle to destroy.
 */
NVE_C_API nve_status_t nve_heuristic_destroy(nve_heuristic_t heuristic);

/* ============================================================================
 * Execution Context
 * ============================================================================ */

/**
 * Destroy an execution context. The context must have been wait()'ed (no
 * outstanding work) and must be destroyed before its owning table/layer.
 * @param ctx Context handle to destroy.
 */
NVE_C_API nve_status_t nve_context_destroy(nve_context_t ctx);

/**
 * Block until all pending work submitted with this context has completed.
 * Synchronizes the context's lookup, modify, and auxiliary CUDA streams, and
 * waits for any CPU work offloaded to its thread pool.
 * @param ctx Context handle to wait on.
 */
NVE_C_API nve_status_t nve_context_wait(nve_context_t ctx);

/**
 * Get the CUDA stream the context uses for lookup operations.
 * @param ctx Context handle to query.
 * @param out_stream Output set to a cudaStream_t (cast as void*).
 */
NVE_C_API nve_status_t nve_context_get_lookup_stream(nve_context_t ctx, void** out_stream);

/**
 * Get the CUDA stream the context uses for modify operations (insert/update/
 * accumulate/clear/erase).
 * @param ctx Context handle to query.
 * @param out_stream Output set to a cudaStream_t (cast as void*).
 */
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

/**
 * Destroy a table handle (GPU or host). All execution contexts created from
 * this table must be destroyed first.
 * @param table Table handle to destroy.
 */
NVE_C_API nve_status_t nve_table_destroy(nve_table_t table);

/**
 * Look up values for `n` keys.
 * @param table Table handle to query.
 * @param ctx Execution context.
 * @param n Number of keys.
 * @param keys Array of `n` keys (key type matches the table's key type).
 * @param hit_mask Optional bitmask buffer. On input, bit i = 1 marks key i as
 *        "already resolved" and the table skips it. On output, bit i = 1 iff
 *        key i was resolved by this call. Size = ceil(n / 64) uint64_t words,
 *        with the last word zero-padded. Pass NULL to treat all keys as
 *        unresolved on entry.
 * @param value_stride Spacing in bytes between successive value slots in
 *        `values`.
 * @param values Output buffer for resolved values, at least n * value_stride
 *        bytes. Pass NULL to use the call as an exists/count check.
 * @param value_sizes Optional output array of n int64_t, filled with the
 *        actual number of bytes written per key. May be NULL.
 */
NVE_C_API nve_status_t nve_table_find(
    nve_table_t table, nve_context_t ctx, int64_t n, const void* keys,
    uint64_t* hit_mask, int64_t value_stride, void* values, int64_t* value_sizes);

/**
 * Insert key/value pairs. Best-effort: if the table overflows, freshly
 * inserted entries may be evicted per the overflow policy. Existing keys may
 * be overwritten or ignored depending on the table implementation.
 * @param table Table handle.
 * @param ctx Execution context.
 * @param n Number of key/value pairs.
 * @param keys Array of `n` keys.
 * @param value_stride Spacing in bytes between successive values in `values`.
 * @param value_size Size in bytes of each value.
 * @param values Array of at least `n * value_stride` bytes.
 */
NVE_C_API nve_status_t nve_table_insert(
    nve_table_t table, nve_context_t ctx, int64_t n, const void* keys,
    int64_t value_stride, int64_t value_size, const void* values);

/**
 * Overwrite values for keys that already exist in the table. Keys that are
 * not currently present are ignored (no insertion occurs).
 * @param table Table handle.
 * @param ctx Execution context.
 * @param n Number of key/value pairs.
 * @param keys Array of `n` keys.
 * @param value_stride Spacing in bytes between successive values in `values`.
 * @param value_size Size in bytes of each value.
 * @param values Array of at least `n * value_stride` bytes.
 */
NVE_C_API nve_status_t nve_table_update(
    nve_table_t table, nve_context_t ctx, int64_t n, const void* keys,
    int64_t value_stride, int64_t value_size, const void* values);

/**
 * Accumulate (add) `updates` into the values of keys that already exist in
 * the table. Missing keys are ignored. `update_size` must be a multiple of
 * dtype_size(update_dtype). Not all combinations of table storage dtype and
 * `update_dtype` are supported by every backend.
 * @param table Table handle.
 * @param ctx Execution context.
 * @param n Number of keys.
 * @param keys Array of `n` keys.
 * @param update_stride Spacing in bytes between successive updates.
 * @param update_size Size in bytes of each update vector.
 * @param updates Array of at least `n * update_stride` bytes.
 * @param update_dtype Data type of the update values.
 */
NVE_C_API nve_status_t nve_table_update_accumulate(
    nve_table_t table, nve_context_t ctx, int64_t n, const void* keys,
    int64_t update_stride, int64_t update_size, const void* updates,
    nve_data_type_t update_dtype);

/**
 * Remove all entries from the table.
 * @warning Not synchronized — caller must ensure no other operations are in
 *          flight against the table when calling this.
 * @param table Table handle.
 * @param ctx Execution context.
 */
NVE_C_API nve_status_t nve_table_clear(nve_table_t table, nve_context_t ctx);

/**
 * Erase the given keys. Keys not present in the table are silently ignored.
 * @param table Table handle.
 * @param ctx Execution context.
 * @param n Number of keys.
 * @param keys Array of `n` keys to erase.
 */
NVE_C_API nve_status_t nve_table_erase(
    nve_table_t table, nve_context_t ctx, int64_t n, const void* keys);

/**
 * Create an execution context bound to this table. Contexts hold per-operation
 * resources (temporary buffers, etc.) and are reusable but not thread-safe —
 * a single context may not be used concurrently from multiple threads.
 * @param table Table handle the context will be bound to.
 * @param out Output context handle.
 * @param lookup_stream CUDA stream for lookup ops (cudaStream_t cast to void*).
 * @param modify_stream CUDA stream for modify ops.
 * @param thread_pool Thread pool for CPU work, NULL for the global default.
 * @param allocator Allocator for large buffers, NULL for the default.
 * @note All contexts created from a table must be destroyed before the table.
 */
NVE_C_API nve_status_t nve_table_create_execution_context(
    nve_table_t table, nve_context_t* out,
    void* lookup_stream, void* modify_stream,
    nve_thread_pool_t thread_pool, nve_allocator_t allocator);

/**
 * Get the CUDA device id used by the table, or -1 for CPU/host tables.
 * @param table Table handle to query.
 * @param out Output device id.
 */
NVE_C_API nve_status_t nve_table_get_device_id(const nve_table_t table, int32_t* out);

/**
 * Get the maximum supported value (row) size in bytes for this table.
 * @param table Table handle to query.
 * @param out Output max row size in bytes.
 */
NVE_C_API nve_status_t nve_table_get_max_row_size(const nve_table_t table, int64_t* out);

/**
 * Get the size of a key in bytes for this table (e.g. 4 for int32, 8 for
 * int64).
 * @param table Table handle to query.
 * @param out Output key size in bytes.
 */
NVE_C_API nve_status_t nve_table_get_key_size(const nve_table_t table, int64_t* out);

/**
 * Reset the lookup key counter associated with the given context.
 * The counter feeds insert heuristics; see nve_table_get_lookup_counter for
 * the hit-vs-miss semantics.
 * @param table Table handle.
 * @param ctx Context whose counter should be reset.
 */
NVE_C_API nve_status_t nve_table_reset_lookup_counter(nve_table_t table, nve_context_t ctx);

/**
 * Read the lookup key counter for the given context. GPU tables count
 * lookup misses (counter <= 0); host tables count hits. Tables using a lookup
 * stream require synchronization on the context's lookup stream before
 * reading the counter.
 * @param table Table handle.
 * @param ctx Context whose counter should be read.
 * @param counter Output counter value.
 */
NVE_C_API nve_status_t nve_table_get_lookup_counter(const nve_table_t table, nve_context_t ctx, int64_t* counter);

/* ============================================================================
 * Host Tables (via plugins / JSON config)
 * ============================================================================ */

/**
 * Create a host-table factory from a JSON description. The JSON must select a
 * registered plugin implementation (load plugins first via
 * nve_load_host_table_plugin) and supply any implementation-specific options.
 * @param out Output factory handle.
 * @param json_config JSON string describing the factory configuration.
 */
NVE_C_API nve_status_t nve_create_host_table_factory(
    nve_host_factory_t* out, const char* json_config);

/**
 * Destroy a host-table factory handle. Tables produced from the factory are
 * independent and remain valid.
 * @param factory Factory handle to destroy.
 */
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
 * Loads plugins, creates factories, and produces tables as specified. JSON
 * "plugins" entries are shared object names/paths passed directly to dlopen().
 * @param json_config JSON string with host_database configuration.
 * @param out_tables Output array of table handles (caller must free with nve_free_host_database).
 * @param out_ids Output array of table IDs (parallel to out_tables).
 * @param out_count Number of tables created.
 */
NVE_C_API nve_status_t nve_build_host_database(
    const char* json_config,
    nve_table_t** out_tables, int64_t** out_ids, int64_t* out_count);

/**
 * Free the arrays returned by nve_build_host_database. Destroys each table
 * handle in `tables` and frees both arrays.
 * @param tables Table-handle array returned by nve_build_host_database.
 * @param ids Table-id array returned by nve_build_host_database.
 * @param count Number of entries (same value returned in `out_count`).
 */
NVE_C_API nve_status_t nve_free_host_database(
    nve_table_t* tables, int64_t* ids, int64_t count);

/**
 * Get the number of entries in a host table.
 * @param table Host table handle.
 * @param ctx Execution context.
 * @param exact Non-zero requests an accurate count; zero permits an
 *        approximate count for backends where the exact size is expensive
 *        to compute.
 * @param out Output entry count.
 */
NVE_C_API nve_status_t nve_host_table_size(
    const nve_table_t table, nve_context_t ctx, int exact, int64_t* out);

/* ============================================================================
 * Embedding Layers
 * ============================================================================ */

/**
 * Create a GPU embedding layer (no caching, linear table in GPU memory).
 * @param out Output layer handle.
 * @param key_type Key type (NVE_KEY_INT32 or NVE_KEY_INT64).
 * @param config Layer configuration.
 * @param allocator Optional allocator handle, NULL for default.
 */
NVE_C_API nve_status_t nve_gpu_embedding_layer_create(
    nve_layer_t* out, nve_key_type_t key_type,
    const nve_gpu_embedding_layer_config_t* config, nve_allocator_t allocator);

/**
 * Create a Linear UVM embedding layer (GPU cache + UVM backup).
 * @param out Output layer handle.
 * @param key_type Key type (NVE_KEY_INT32 or NVE_KEY_INT64).
 * @param config Layer configuration.
 * @param gpu_table A GPU table handle to use as the cache.
 * @param allocator Optional allocator handle, NULL for default.
 */
NVE_C_API nve_status_t nve_linear_uvm_layer_create(
    nve_layer_t* out, nve_key_type_t key_type,
    const nve_linear_uvm_layer_config_t* config,
    nve_table_t gpu_table, nve_allocator_t allocator);

/**
 * Create a hierarchical embedding layer (GPU -> Host -> Remote).
 * @param out Output layer handle.
 * @param key_type Key type (NVE_KEY_INT32 or NVE_KEY_INT64).
 * @param config Layer configuration.
 * @param tables Array of table handles, processed in order during lookup.
 * @param num_tables Number of tables.
 * @param allocator Optional allocator handle, NULL for default.
 */
NVE_C_API nve_status_t nve_hierarchical_layer_create(
    nve_layer_t* out, nve_key_type_t key_type,
    const nve_hierarchical_layer_config_t* config,
    const nve_table_t* tables, int64_t num_tables, nve_allocator_t allocator);

/**
 * Create a host-only embedding layer wrapping one host table.
 * @param out Output layer handle.
 * @param key_type Key type (NVE_KEY_INT32 or NVE_KEY_INT64).
 * @param config Layer configuration.
 * @param host_table Host table handle (must report CPU/host device).
 * @param allocator Optional allocator handle, NULL for default.
 */
NVE_C_API nve_status_t nve_host_embedding_layer_create(
    nve_layer_t* out, nve_key_type_t key_type,
    const nve_host_embedding_layer_config_t* config,
    nve_table_t host_table, nve_allocator_t allocator);

/**
 * Destroy an embedding layer. All execution contexts created from this layer
 * must be wait()'ed and destroyed first.
 * @param layer Layer handle to destroy.
 */
NVE_C_API nve_status_t nve_layer_destroy(nve_layer_t layer);

/**
 * Lookup embeddings for the given keys.
 * @param layer Layer handle to query.
 * @param ctx Execution context.
 * @param num_keys Number of keys.
 * @param keys Array of `num_keys` keys.
 * @param output Output buffer for resolved embedding vectors.
 * @param output_stride Spacing in bytes between successive output vectors.
 * @param hitmask Optional output bitmask (bit i = 1 iff key i was resolved).
 *        Pass NULL to skip.
 * @param hitrates Optional array sized to the number of internal tables;
 *        receives per-table hit rate. Pass NULL to skip.
 */
NVE_C_API nve_status_t nve_layer_lookup(
    nve_layer_t layer, nve_context_t ctx,
    int64_t num_keys, const void* keys,
    void* output, int64_t output_stride,
    uint64_t* hitmask, float* hitrates);

/**
 * Lookup with pooling/combining across bags.
 * @param layer Layer handle to query.
 * @param ctx Execution context.
 * @param num_keys Number of keys.
 * @param keys Array of `num_keys` keys.
 * @param output Output buffer for pooled vectors.
 * @param output_stride Spacing in bytes between successive output vectors.
 * @param hitmask Optional output bitmask, see nve_layer_lookup. May be NULL.
 * @param pooling_type Reduction applied per bag. NVE_POOL_CONCATENATE means
 *        no pooling — prefer nve_layer_lookup for that case.
 * @param sparse_type Layout of `key_indices` describing bag membership:
 *        FIXED: key_indices is a single element (the per-bag hotness).
 *        CSR: one offset per bag plus one trailing offset (num_bags + 1).
 *        COO: two elements per key (bag_id, id_in_bag), sorted row-wise.
 * @param key_indices Bag-membership data; layout depends on `sparse_type`. The
 *        element type must match the layer's key type (int32 or int64).
 * @param num_key_indices Number of elements in `key_indices`.
 * @param sparse_weights Optional per-key weights for weighted pooling. May
 *        be NULL when pooling_type is not weighted.
 * @param weight_type Data type of `sparse_weights`; need not match the
 *        layer's value dtype, but not all combinations are supported.
 * @param hitrates Optional array sized to the number of internal tables;
 *        receives per-table hit rate. May trigger synchronization.
 */
NVE_C_API nve_status_t nve_layer_lookup_pooled(
    nve_layer_t layer, nve_context_t ctx,
    int64_t num_keys, const void* keys,
    void* output, int64_t output_stride,
    uint64_t* hitmask,
    nve_pooling_type_t pooling_type, nve_sparse_type_t sparse_type,
    const void* key_indices, int64_t num_key_indices,
    const void* sparse_weights,
    nve_data_type_t weight_type, float* hitrates);

/**
 * Insert key/value pairs into a specific internal table of the layer.
 * Unlike update, insert does not guarantee replacement: if a key already
 * exists in the target table, its existing value may be retained.
 * @param layer Layer handle.
 * @param ctx Execution context.
 * @param num_keys Number of keys.
 * @param keys Array of `num_keys` keys.
 * @param value_stride Spacing in bytes between successive values.
 * @param value_size Size in bytes of each value.
 * @param values Array of at least `num_keys * value_stride` bytes.
 * @param table_id Index of the internal table to insert into (0 = top tier), negative index implies all.
 */
NVE_C_API nve_status_t nve_layer_insert(
    nve_layer_t layer, nve_context_t ctx,
    int64_t num_keys, const void* keys,
    int64_t value_stride, int64_t value_size, const void* values,
    int64_t table_id);

/**
 * Overwrite values for the given keys in every internal table where they
 * already reside. Does not change which tier holds a key.
 * @param layer Layer handle.
 * @param ctx Execution context.
 * @param num_keys Number of keys.
 * @param keys Array of `num_keys` keys.
 * @param value_stride Spacing in bytes between successive values.
 * @param value_size Size in bytes of each value.
 * @param values Array of at least `num_keys * value_stride` bytes.
 * @param table_id Index of the internal table to update, negative index implies all.
 */
NVE_C_API nve_status_t nve_layer_update(
    nve_layer_t layer, nve_context_t ctx,
    int64_t num_keys, const void* keys,
    int64_t value_stride, int64_t value_size, const void* values,
    int64_t table_id);

/**
 * Accumulate (add) values (e.g. gradients) into existing keys in every
 * internal table where they reside. Does not change tier residency.
 * @param layer Layer handle.
 * @param ctx Execution context.
 * @param num_keys Number of keys.
 * @param keys Array of `num_keys` keys.
 * @param value_stride Spacing in bytes between successive values.
 * @param value_size Size in bytes of each value.
 * @param values Array of at least `num_keys * value_stride` bytes.
 * @param value_type Data type of the supplied values; may differ from the
 *        layer's storage dtype (e.g. int8 update into fp16 table) subject to
 *        backend support.
 * @param table_id Index of the internal table to accumulate into, negative index implies all.
 */
NVE_C_API nve_status_t nve_layer_accumulate(
    nve_layer_t layer, nve_context_t ctx,
    int64_t num_keys, const void* keys,
    int64_t value_stride, int64_t value_size, const void* values,
    nve_data_type_t value_type, int64_t table_id);

/**
 * Clear all internal tables.
 * @warning Not synchronized — caller must ensure no other ops are in flight
 *          against this layer when calling.
 * @param layer Layer handle.
 * @param ctx Execution context.
 */
NVE_C_API nve_status_t nve_layer_clear(nve_layer_t layer, nve_context_t ctx);

/**
 * Erase the given keys from a specific internal table. Keys not present in
 * the target table are ignored.
 * @param layer Layer handle.
 * @param ctx Execution context.
 * @param num_keys Number of keys.
 * @param keys Array of `num_keys` keys to erase.
 * @param table_id Index of the internal table to erase from, negative index implies all.
 */
NVE_C_API nve_status_t nve_layer_erase(
    nve_layer_t layer, nve_context_t ctx,
    int64_t num_keys, const void* keys, int64_t table_id);

/**
 * Create an execution context bound to this layer. See
 * nve_table_create_execution_context for the threading and lifetime rules —
 * the same restrictions apply here.
 * @param layer Layer handle the context will be bound to.
 * @param out Output context handle.
 * @param lookup_stream CUDA stream for lookup ops (cudaStream_t cast to void*).
 * @param modify_stream CUDA stream for modify ops.
 * @param thread_pool Thread pool for CPU work, NULL for the global default.
 * @param allocator Allocator for large buffers, NULL for the default.
 */
NVE_C_API nve_status_t nve_layer_create_execution_context(
    nve_layer_t layer, nve_context_t* out,
    void* lookup_stream, void* modify_stream,
    nve_thread_pool_t thread_pool, nve_allocator_t allocator);

/**
 * Get the number of tables used by the layer.
 * @param layer Layer handle to query.
 * @param out Output number of tables.
 */
NVE_C_API nve_status_t nve_layer_get_num_tables(nve_layer_t layer, int64_t* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* NVE_C_API_H */
