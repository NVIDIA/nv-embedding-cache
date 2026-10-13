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

/* Frozen ABI 2.0 client declarations. Deliberately do not include nve_c_api.h:
 * this client must keep the initial layouts and calling conventions as NVE grows.
 * Compiled as C into c_api_tests and run by c_api_sized_config_test.cpp.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

typedef int32_t nve_status_t;
typedef int32_t nve_data_type_t;
enum { NVE_SUCCESS = 0, NVE_ERROR_INVALID_ARGUMENT = 1,
       NVE_DTYPE_UNKNOWN = 0, NVE_DTYPE_FLOAT32 = 1 };
typedef enum { NVE_KEY_INT32 = 0, NVE_KEY_INT64 } nve_key_type_t;
typedef struct nve_table_s* nve_table_t;
typedef struct nve_layer_s* nve_layer_t;
typedef struct nve_context_s* nve_context_t;
typedef struct nve_allocator_s* nve_allocator_t;
typedef struct nve_heuristic_s* nve_heuristic_t;
typedef struct nve_thread_pool_s* nve_thread_pool_t;
typedef struct nve_table_factory_s* nve_table_factory_t;

typedef struct {
  uint64_t header;
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
  int64_t  uvm_num_rows;            /* Rows in uvm_table, required when uvm_table is set */
} nve_gpu_table_config_t;

typedef struct {
  uint64_t header;
  void*    emb_table;              /* Caller-owned, host-accessible row-major table (preferably cudaMallocHost) */
  int64_t  num_rows;               /* Rows in emb_table; keys outside [0, num_rows) miss */
  int64_t  row_size_in_bytes;      /* Bytes per row */
  nve_data_type_t value_dtype;     /* Storage dtype; accumulate requires FLOAT32 */
  int64_t  max_threads;            /* Positive; upper bound on CPU threads per operation */
  int64_t  task_size_threshold;    /* Positive; CPU threads used <= ceil(num_keys / threshold) */
} nve_linear_host_table_config_t;

typedef struct {
  uint64_t header;
  const char* layer_name;
  int32_t     device_id;
  void*       embedding_table;       /* Pointer to linear table in GPU memory */
  int64_t     num_embeddings;
  int64_t     embedding_width_in_bytes; /* Stored bytes per row. Quantized rows must have an even width. */
  nve_data_type_t value_dtype; /* FLOAT16/FLOAT32 or rowwise QINT8/QUINT8 F16/F32 storage. */
  /* Index of a table row holding the default embedding, returned by lookups of keys outside
   * [0, num_embeddings). The caller owns the row's content.
   * Negative (the default) disables the check, user is responsible for all keys being valid. */
  int64_t     default_row_index;
} nve_gpu_embedding_layer_config_t;

typedef struct {
  uint64_t header;
  const char*     layer_name;
  nve_heuristic_t insert_heuristic;  /* NULL uses default heuristic; use nve_heuristic_create_never() to disable */
  int64_t         min_insert_freq_gpu;
  int64_t         min_insert_size_gpu;
  /* Index of a UVM table row holding the default embedding, returned by lookups of keys outside
   * [0, uvm_num_rows). The caller owns the row's content.
   * Negative (the default) disables the check, user is responsible for all keys being valid. */
  int64_t         default_row_index;
} nve_linear_uvm_layer_config_t;

typedef struct {
  uint64_t header;
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
  uint64_t header;
  const char* layer_name;
  /* Default embedding row returned for keys missing from the host table.
   * NULL or default_embedding_size=0 disables the default (misses left undefined).
   * default_embedding_size, when non-zero, must equal the host table row size. */
  const void* default_embedding;
  int64_t     default_embedding_size;
  int64_t     task_size_threshold;  /* Positive; CPU task count <= ceil(num_keys / threshold) */
  int         disable_threadpool;   /* bool: run lookups on the caller thread */
} nve_host_embedding_layer_config_t;

nve_status_t nve_gpu_table_config_init(nve_gpu_table_config_t* config, size_t config_size);
nve_status_t nve_linear_host_table_config_init(nve_linear_host_table_config_t* config, size_t config_size);
nve_status_t nve_gpu_embedding_layer_config_init(nve_gpu_embedding_layer_config_t* config, size_t config_size);
nve_status_t nve_linear_uvm_layer_config_init(nve_linear_uvm_layer_config_t* config, size_t config_size);
nve_status_t nve_hierarchical_layer_config_init(nve_hierarchical_layer_config_t* config, size_t config_size);
nve_status_t nve_host_embedding_layer_config_init(nve_host_embedding_layer_config_t* config, size_t config_size);
nve_status_t nve_gpu_table_create(
    nve_table_t* out, nve_key_type_t key_type,
    const nve_gpu_table_config_t* config, nve_allocator_t allocator);
nve_status_t nve_linear_host_table_create(
    nve_table_t* out, nve_key_type_t key_type,
    const nve_linear_host_table_config_t* config);
nve_status_t nve_gpu_embedding_layer_create(
    nve_layer_t* out, nve_key_type_t key_type,
    const nve_gpu_embedding_layer_config_t* config, nve_allocator_t allocator);
nve_status_t nve_linear_uvm_layer_create(
    nve_layer_t* out, nve_key_type_t key_type,
    const nve_linear_uvm_layer_config_t* config,
    nve_table_t gpu_table, nve_allocator_t allocator);
nve_status_t nve_hierarchical_layer_create(
    nve_layer_t* out, nve_key_type_t key_type,
    const nve_hierarchical_layer_config_t* config,
    const nve_table_t* tables, int64_t num_tables, nve_allocator_t allocator);
nve_status_t nve_host_embedding_layer_create(
    nve_layer_t* out, nve_key_type_t key_type,
    const nve_host_embedding_layer_config_t* config,
    nve_table_t host_table, nve_allocator_t allocator);
nve_status_t nve_create_table_factory(
    nve_table_factory_t* out, const char* plugin_path, const char* json_config);
nve_status_t nve_table_factory_produce(
    nve_table_factory_t factory, int64_t table_id,
    const char* json_config, nve_table_t* out);
nve_status_t nve_layer_create_execution_context(
    nve_layer_t layer, nve_context_t* out,
    void* lookup_stream, void* modify_stream,
    nve_thread_pool_t thread_pool, nve_allocator_t allocator);
nve_status_t nve_layer_insert(
    nve_layer_t layer, nve_context_t ctx,
    int64_t num_keys, const void* keys,
    int64_t value_stride, int64_t value_size, const void* values,
    int64_t table_id);
nve_status_t nve_layer_lookup(
    nve_layer_t layer, nve_context_t ctx,
    int64_t num_keys, const void* keys,
    void* output, int64_t output_stride,
    uint64_t* hitmask, float* hitrates);
nve_status_t nve_context_wait(nve_context_t ctx);
nve_status_t nve_context_destroy(nve_context_t ctx);
nve_status_t nve_layer_destroy(nve_layer_t layer);
nve_status_t nve_table_destroy(nve_table_t table);
nve_status_t nve_table_factory_destroy(nve_table_factory_t factory);
nve_status_t nve_abi_version(int32_t* major, int32_t* minor);
nve_status_t nve_get_last_error(const char** message);

/* Library-private header encoding, checked here only to pin the current behavior:
 * per-type tag in the upper 32 bits, size in the lower 32 bits. */
#define CONFIG_SIZE(config) ((config).header & 0xFFFFFFFFu)

#define CHECK(expression) \
  do { \
    if (!(expression)) { \
      const char* message = NULL; \
      nve_get_last_error(&message); \
      fprintf(stderr, "ABI 2.0 client failed at line %d: %s; %s\n", \
              __LINE__, #expression, message ? message : ""); \
      return 1; \
    } \
  } while (0)

int nve_frozen_abi_2_0_client(void) {
  int32_t major, minor;
  CHECK(nve_abi_version(&major, &minor) == NVE_SUCCESS);
  CHECK(major == 2 && minor >= 0);
  nve_gpu_table_config_t gpu;
  CHECK(nve_gpu_table_config_init(&gpu, sizeof(gpu)) == NVE_SUCCESS);
  CHECK(CONFIG_SIZE(gpu) == sizeof(gpu) && (gpu.header >> 32) != 0);
  CHECK(gpu.device_id == 0 && gpu.cache_size == 0 && gpu.row_size_in_bytes == 0);
  CHECK(gpu.uvm_table == NULL && gpu.uvm_num_rows == 0 && gpu.count_misses == 1);
  CHECK(gpu.max_modify_size == (1 << 20) && gpu.value_dtype == NVE_DTYPE_UNKNOWN);
  CHECK(gpu.private_stream == NULL && gpu.disable_uvm_update == 0);
  CHECK(gpu.uvm_cpu_accumulate == 1 && gpu.data_storage_on_host == 0 && gpu.modify_on_gpu == 1);
  CHECK(gpu.kernel_mode_type == 0 && gpu.kernel_mode_value == 0 && gpu.invalid_key == -1);

  nve_linear_host_table_config_t linear;
  CHECK(nve_linear_host_table_config_init(&linear, sizeof(linear)) == NVE_SUCCESS);
  CHECK(CONFIG_SIZE(linear) == sizeof(linear) && (linear.header >> 32) != 0);
  CHECK(linear.emb_table == NULL && linear.num_rows == 0 && linear.row_size_in_bytes == 8);
  CHECK(linear.value_dtype == NVE_DTYPE_UNKNOWN && linear.max_threads == INT64_MAX);
  CHECK(linear.task_size_threshold == 1024);

  nve_gpu_embedding_layer_config_t emb;
  CHECK(nve_gpu_embedding_layer_config_init(&emb, sizeof(emb)) == NVE_SUCCESS);
  CHECK(CONFIG_SIZE(emb) == sizeof(emb) && (emb.header >> 32) != 0);
  CHECK(strcmp(emb.layer_name, "") == 0 && emb.device_id == 0 && emb.embedding_table == NULL);
  CHECK(emb.num_embeddings == 0 && emb.embedding_width_in_bytes == 0);
  CHECK(emb.value_dtype == NVE_DTYPE_UNKNOWN && emb.default_row_index == -1);

  nve_linear_uvm_layer_config_t uvm;
  CHECK(nve_linear_uvm_layer_config_init(&uvm, sizeof(uvm)) == NVE_SUCCESS);
  CHECK(CONFIG_SIZE(uvm) == sizeof(uvm) && (uvm.header >> 32) != 0);
  CHECK(strcmp(uvm.layer_name, "") == 0 && uvm.insert_heuristic == NULL);
  CHECK(uvm.min_insert_freq_gpu == 0 && uvm.min_insert_size_gpu == (1 << 16));
  CHECK(uvm.default_row_index == -1);

  nve_hierarchical_layer_config_t hier;
  CHECK(nve_hierarchical_layer_config_init(&hier, sizeof(hier)) == NVE_SUCCESS);
  CHECK(CONFIG_SIZE(hier) == sizeof(hier) && (hier.header >> 32) != 0);
  CHECK(strcmp(hier.layer_name, "") == 0 && hier.insert_heuristic == NULL);
  CHECK(hier.min_insert_freq_gpu == 0 && hier.min_insert_freq_host == 0);
  CHECK(hier.min_insert_size_gpu == (1 << 16) && hier.min_insert_size_host == 0);
  CHECK(hier.default_embedding == NULL && hier.default_embedding_size == 0);

  nve_host_embedding_layer_config_t host;
  CHECK(nve_host_embedding_layer_config_init(&host, sizeof(host)) == NVE_SUCCESS);
  CHECK(CONFIG_SIZE(host) == sizeof(host) && (host.header >> 32) != 0);
  CHECK(strcmp(host.layer_name, "") == 0);
  CHECK(host.default_embedding == NULL && host.default_embedding_size == 0);
  CHECK(host.task_size_threshold == 1024 && host.disable_threadpool == 0);

  // Invalid keys/resources exercise creation config conversions without requiring a GPU.
  nve_table_t table = NULL;
  nve_layer_t layer = NULL;
  CHECK(nve_gpu_table_create(&table, (nve_key_type_t)99, &gpu, NULL) == NVE_ERROR_INVALID_ARGUMENT);
  CHECK(nve_gpu_embedding_layer_create(&layer, (nve_key_type_t)99, &emb, NULL) == NVE_ERROR_INVALID_ARGUMENT);
  CHECK(nve_linear_uvm_layer_create(&layer, NVE_KEY_INT64, &uvm, NULL, NULL) == NVE_ERROR_INVALID_ARGUMENT);

  nve_table_factory_t factory = NULL;
  CHECK(nve_create_table_factory(&factory, "libnve-plugin-stl-map.so", "{}") == NVE_SUCCESS);
  CHECK(nve_table_factory_produce(factory, 0,
      "{\"key_size\":8,\"max_value_size\":4,\"value_dtype\":\"float32\"}", &table) == NVE_SUCCESS);
  const float default_embedding = 7.5f;
  host.layer_name = "abi_host";
  host.default_embedding = &default_embedding;
  host.default_embedding_size = sizeof(default_embedding);
  host.task_size_threshold = 1;
  host.disable_threadpool = 1;
  CHECK(nve_host_embedding_layer_create(&layer, NVE_KEY_INT64, &host, table, NULL) == NVE_SUCCESS);
  nve_context_t ctx = NULL;
  CHECK(nve_layer_create_execution_context(layer, &ctx, NULL, NULL, NULL, NULL) == NVE_SUCCESS);
  const int64_t keys[] = {3, 7};
  const float value = 12.5f;
  CHECK(nve_layer_insert(layer, ctx, 1, keys, 4, 4, &value, 0) == NVE_SUCCESS);
  CHECK(nve_context_wait(ctx) == NVE_SUCCESS);
  float output[] = {0.f, 0.f};
  uint64_t hitmask = 0;
  CHECK(nve_layer_lookup(layer, ctx, 2, keys, output, 4, &hitmask, NULL) == NVE_SUCCESS);
  CHECK(nve_context_wait(ctx) == NVE_SUCCESS);
  CHECK(output[0] == value && output[1] == default_embedding && (hitmask & 3) == 1);
  CHECK(nve_context_destroy(ctx) == NVE_SUCCESS);
  CHECK(nve_layer_destroy(layer) == NVE_SUCCESS);

  hier.layer_name = "abi_hierarchical";
  hier.default_embedding = &default_embedding;
  hier.default_embedding_size = sizeof(default_embedding);
  CHECK(nve_hierarchical_layer_create(&layer, NVE_KEY_INT64, &hier, &table, 1, NULL) == NVE_SUCCESS);
  CHECK(nve_layer_destroy(layer) == NVE_SUCCESS);
  CHECK(nve_table_destroy(table) == NVE_SUCCESS);
  CHECK(nve_table_factory_destroy(factory) == NVE_SUCCESS);

  float rows[] = {1.f, 2.f, 3.f};
  linear.emb_table = rows;
  linear.num_rows = 3;
  linear.row_size_in_bytes = sizeof(float);
  linear.value_dtype = NVE_DTYPE_FLOAT32;
  linear.max_threads = 2;
  linear.task_size_threshold = 1;
  CHECK(nve_linear_host_table_create(&table, NVE_KEY_INT64, &linear) == NVE_SUCCESS);
  host.layer_name = "abi_linear_host";
  CHECK(nve_host_embedding_layer_create(&layer, NVE_KEY_INT64, &host, table, NULL) == NVE_SUCCESS);
  CHECK(nve_layer_create_execution_context(layer, &ctx, NULL, NULL, NULL, NULL) == NVE_SUCCESS);
  const int64_t linear_keys[] = {1, 7};  /* Key 7 is past num_rows and gets the default. */
  hitmask = 0;
  CHECK(nve_layer_lookup(layer, ctx, 2, linear_keys, output, 4, &hitmask, NULL) == NVE_SUCCESS);
  CHECK(nve_context_wait(ctx) == NVE_SUCCESS);
  CHECK(output[0] == rows[1] && output[1] == default_embedding && (hitmask & 3) == 1);
  CHECK(nve_context_destroy(ctx) == NVE_SUCCESS);
  CHECK(nve_layer_destroy(layer) == NVE_SUCCESS);
  CHECK(nve_table_destroy(table) == NVE_SUCCESS);
  return 0;
}
