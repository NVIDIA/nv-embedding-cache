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

/*
 * Sample external-mode NVE plugin: a host-memory chained hash map written in
 * pure C. It includes only the C headers and never links or calls a C++ NVE
 * symbol, demonstrating the unknown-toolchain contract.
 *
 * Concurrency contract: the map itself is not synchronized across calls.
 * Concurrent find calls on one table are safe (the parallel find keeps each
 * hit-mask word private to one task and counts hits atomically), but a modify
 * op (insert/update/update_accumulate/erase/clear) must not run concurrently
 * with any other call on the same table. A production plugin that needs
 * concurrent modification must add its own locking.
 */

#include <plugin/nve_external_plugin.h>

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Metadata symbols + idempotent NVE-services handshake. */
NVE_DEFINE_EXTERNAL_PLUGIN("Sample External Map", "NVIDIA Corporation")

/* ============================================================================
 * Error reporting (thread-local, per the ABI last_error contract)
 * ============================================================================ */

static _Thread_local char g_last_error[512];

static nve_status_t set_error(nve_status_t status, const char* message) {
  snprintf(g_last_error, sizeof(g_last_error), "%s", message);
  return status;
}

static const char* my_last_error(void* self) {
  (void)self;
  return g_last_error;
}

/* ============================================================================
 * Minimal JSON integer extraction (sample-grade; no external dependencies)
 * ============================================================================ */

static int64_t json_get_int(const char* json, const char* key, int64_t fallback) {
  if (json == NULL) return fallback;
  char pattern[64];
  snprintf(pattern, sizeof(pattern), "\"%s\"", key);
  const char* pos = strstr(json, pattern);
  if (pos == NULL) return fallback;
  pos = strchr(pos + strlen(pattern), ':');
  if (pos == NULL) return fallback;
  return strtoll(pos + 1, NULL, 10);
}

/* ============================================================================
 * Table: chained hash map with fixed-size value rows
 * ============================================================================ */

typedef struct entry_s {
  int64_t key;
  struct entry_s* next;
  /* row_size value bytes follow */
} entry_t;

typedef struct {
  int64_t row_size;
  int64_t num_buckets;
  int64_t invalid_key;
  entry_t** buckets;
  atomic_llong hit_counter; /* host-table convention: counts lookup hits */
} my_table_t;

typedef struct {
  int64_t num_buckets; /* factory-level default for produced tables */
} my_factory_t;

static uint64_t hash_key(int64_t key) {
  uint64_t h = (uint64_t)key;
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  return h;
}

static entry_t** bucket_of(my_table_t* t, int64_t key) {
  return &t->buckets[hash_key(key) % (uint64_t)t->num_buckets];
}

static entry_t* find_entry(my_table_t* t, int64_t key) {
  for (entry_t* e = *bucket_of(t, key); e != NULL; e = e->next) {
    if (e->key == key) return e;
  }
  return NULL;
}

static char* entry_value(entry_t* e) { return (char*)(e + 1); }

/* Inserts or overwrites. Returns NULL on allocation failure. */
static entry_t* upsert_entry(my_table_t* t, int64_t key) {
  entry_t* e = find_entry(t, key);
  if (e != NULL) return e;
  e = (entry_t*)calloc(1, sizeof(entry_t) + (size_t)t->row_size);
  if (e == NULL) return NULL;
  entry_t** bucket = bucket_of(t, key);
  e->key = key;
  e->next = *bucket;
  *bucket = e;
  return e;
}

/* ============================================================================
 * Table ops
 * ============================================================================ */

static void my_destroy(void* self) {
  my_table_t* t = (my_table_t*)self;
  for (int64_t b = 0; b < t->num_buckets; ++b) {
    entry_t* e = t->buckets[b];
    while (e != NULL) {
      entry_t* next = e->next;
      free(e);
      e = next;
    }
  }
  free(t->buckets);
  free(t);
}

static nve_buffer_location_t my_pref_loc(void* self) {
  (void)self;
  return NVE_BUFFER_LOCATION_HOST;
}

static int32_t my_device_id(void* self) {
  (void)self;
  return -1; /* host table; no CUDA device to select */
}

static int64_t my_max_row(void* self) { return ((my_table_t*)self)->row_size; }

static int64_t my_key_size(void* self) {
  (void)self;
  return (int64_t)sizeof(int64_t);
}

static int64_t my_invalid_key(void* self) { return ((my_table_t*)self)->invalid_key; }

static nve_data_type_t my_dtype(void* self) {
  (void)self;
  return NVE_DTYPE_FLOAT32;
}

static int64_t my_counter_hits(void* self) {
  (void)self;
  return 1;
}

static nve_status_t my_reset_counter(void* self, nve_ext_context_t ctx) {
  (void)ctx;
  atomic_store(&((my_table_t*)self)->hit_counter, 0);
  return NVE_SUCCESS;
}

static nve_status_t my_get_counter(void* self, nve_ext_context_t ctx, int64_t* counter) {
  (void)ctx;
  if (counter == NULL) return set_error(NVE_ERROR_INVALID_ARGUMENT, "counter is null");
  *counter = (int64_t)atomic_load(&((my_table_t*)self)->hit_counter);
  return NVE_SUCCESS;
}

/* One parallel chunk of a find call. Chunks are aligned to 64 keys so no two
 * tasks touch the same hit-mask word. */
typedef struct {
  my_table_t* table;
  int64_t n;
  int64_t chunk;
  const int64_t* keys;
  uint64_t* hit_mask;
  int64_t value_stride;
  char* values;
  int64_t* value_sizes;
} find_job_t;

static void find_range(find_job_t* job, int64_t begin, int64_t end) {
  int64_t hits = 0;
  for (int64_t i = begin; i < end; ++i) {
    if (job->hit_mask != NULL && (job->hit_mask[i / 64] >> (i % 64)) & 1u) {
      continue; /* already resolved by a faster tier */
    }
    entry_t* e = find_entry(job->table, job->keys[i]);
    if (e == NULL) continue;
    if (job->values != NULL) {
      memcpy(job->values + i * job->value_stride, entry_value(e),
             (size_t)job->table->row_size);
    }
    if (job->value_sizes != NULL) job->value_sizes[i] = job->table->row_size;
    if (job->hit_mask != NULL) job->hit_mask[i / 64] |= 1ull << (i % 64);
    ++hits;
  }
  atomic_fetch_add(&job->table->hit_counter, hits);
}

static nve_status_t find_task(void* user, int64_t index) {
  find_job_t* job = (find_job_t*)user;
  const int64_t begin = index * job->chunk;
  int64_t end = begin + job->chunk;
  if (end > job->n) end = job->n;
  if (begin < end) find_range(job, begin, end);
  return NVE_SUCCESS;
}

static nve_status_t my_find(void* self, nve_ext_context_t ctx, int64_t n, const void* keys,
                            uint64_t* hit_mask, int64_t value_stride, void* values,
                            int64_t* value_sizes) {
  my_table_t* t = (my_table_t*)self;
  if (n < 0 || keys == NULL) return set_error(NVE_ERROR_INVALID_ARGUMENT, "invalid find args");

  find_job_t job = {.table = t,
                    .n = n,
                    .chunk = 0,
                    .keys = (const int64_t*)keys,
                    .hit_mask = hit_mask,
                    .value_stride = value_stride,
                    .values = (char*)values,
                    .value_sizes = value_sizes};

  /* Parallelize through the host threadpool when the handshake happened;
   * fall back to a serial scan otherwise. */
  const nve_ext_nve_services_t* services = nve_plugin_nve_services();
  int64_t workers = 0;
  if (services != NULL && NVE_ABI_HAS_FIELD(services, nve_ext_nve_services_t, execute_n) &&
      NVE_ABI_HAS_FIELD(services, nve_ext_nve_services_t, num_workers) &&
      NVE_ABI_HAS_FIELD(services, nve_ext_nve_services_t, last_error) &&
      services->num_workers(ctx, &workers) == NVE_SUCCESS && workers > 1 && n > 64) {
    /* 64-key alignment keeps hit-mask words private to one task. */
    int64_t chunk = (n + workers - 1) / workers;
    chunk = (chunk + 63) / 64 * 64;
    const int64_t num_tasks = (n + chunk - 1) / chunk;
    job.chunk = chunk;
    const nve_status_t status = services->execute_n(ctx, num_tasks, find_task, &job);
    if (status != NVE_SUCCESS) {
      return set_error(status, services->last_error(ctx));
    }
    return NVE_SUCCESS;
  }

  find_range(&job, 0, n);
  return NVE_SUCCESS;
}

static nve_status_t my_insert(void* self, nve_ext_context_t ctx, int64_t n, const void* keys,
                              int64_t value_stride, int64_t value_size, const void* values) {
  (void)ctx;
  my_table_t* t = (my_table_t*)self;
  if (n < 0 || keys == NULL || values == NULL || value_size > t->row_size) {
    return set_error(NVE_ERROR_INVALID_ARGUMENT, "invalid insert args");
  }
  const int64_t* k = (const int64_t*)keys;
  const char* v = (const char*)values;
  for (int64_t i = 0; i < n; ++i) {
    entry_t* e = upsert_entry(t, k[i]);
    if (e == NULL) return set_error(NVE_ERROR_OUT_OF_MEMORY, "entry allocation failed");
    memcpy(entry_value(e), v + i * value_stride, (size_t)value_size);
  }
  return NVE_SUCCESS;
}

static nve_status_t my_update(void* self, nve_ext_context_t ctx, int64_t n, const void* keys,
                              int64_t value_stride, int64_t value_size, const void* values) {
  (void)ctx;
  my_table_t* t = (my_table_t*)self;
  if (n < 0 || keys == NULL || values == NULL || value_size > t->row_size) {
    return set_error(NVE_ERROR_INVALID_ARGUMENT, "invalid update args");
  }
  const int64_t* k = (const int64_t*)keys;
  const char* v = (const char*)values;
  for (int64_t i = 0; i < n; ++i) {
    entry_t* e = find_entry(t, k[i]); /* update only writes existing keys */
    if (e != NULL) memcpy(entry_value(e), v + i * value_stride, (size_t)value_size);
  }
  return NVE_SUCCESS;
}

static nve_status_t my_update_accum(void* self, nve_ext_context_t ctx, int64_t n,
                                    const void* keys, int64_t update_stride,
                                    int64_t update_size, const void* updates,
                                    nve_data_type_t update_dtype) {
  (void)ctx;
  my_table_t* t = (my_table_t*)self;
  if (update_dtype != NVE_DTYPE_FLOAT32) {
    return set_error(NVE_ERROR_NOT_IMPLEMENTED, "only float32 accumulation is supported");
  }
  if (n < 0 || keys == NULL || updates == NULL || update_size > t->row_size ||
      update_size % (int64_t)sizeof(float) != 0) {
    return set_error(NVE_ERROR_INVALID_ARGUMENT, "invalid update_accumulate args");
  }
  const int64_t* k = (const int64_t*)keys;
  const char* u = (const char*)updates;
  const int64_t num_floats = update_size / (int64_t)sizeof(float);
  for (int64_t i = 0; i < n; ++i) {
    entry_t* e = find_entry(t, k[i]); /* accumulate only into existing keys */
    if (e == NULL) continue;
    float* row = (float*)entry_value(e);
    const float* upd = (const float*)(u + i * update_stride);
    for (int64_t j = 0; j < num_floats; ++j) row[j] += upd[j];
  }
  return NVE_SUCCESS;
}

static nve_status_t my_erase(void* self, nve_ext_context_t ctx, int64_t n, const void* keys) {
  (void)ctx;
  my_table_t* t = (my_table_t*)self;
  if (n < 0 || keys == NULL) return set_error(NVE_ERROR_INVALID_ARGUMENT, "invalid erase args");
  const int64_t* k = (const int64_t*)keys;
  for (int64_t i = 0; i < n; ++i) {
    entry_t** link = bucket_of(t, k[i]);
    while (*link != NULL && (*link)->key != k[i]) link = &(*link)->next;
    if (*link != NULL) {
      entry_t* dead = *link;
      *link = dead->next;
      free(dead);
    }
  }
  return NVE_SUCCESS;
}

static nve_status_t my_clear(void* self, nve_ext_context_t ctx) {
  (void)ctx;
  my_table_t* t = (my_table_t*)self;
  for (int64_t b = 0; b < t->num_buckets; ++b) {
    entry_t* e = t->buckets[b];
    while (e != NULL) {
      entry_t* next = e->next;
      free(e);
      e = next;
    }
    t->buckets[b] = NULL;
  }
  return NVE_SUCCESS;
}

/* ABI 3.0 additions: key count and key enumeration. */
static nve_status_t my_size(void* self, nve_ext_context_t ctx, int64_t* out_size) {
  (void)ctx;
  my_table_t* t = (my_table_t*)self;
  if (out_size == NULL) return set_error(NVE_ERROR_INVALID_ARGUMENT, "out_size is null");
  int64_t n = 0;
  for (int64_t b = 0; b < t->num_buckets; ++b) {
    for (entry_t* e = t->buckets[b]; e != NULL; e = e->next) ++n;
  }
  *out_size = n;
  return NVE_SUCCESS;
}

static nve_status_t my_get_keys(void* self, nve_ext_context_t ctx, int64_t capacity, void* keys,
                                int64_t* out_num_keys) {
  (void)ctx;
  my_table_t* t = (my_table_t*)self;
  if (capacity < 0 || (capacity > 0 && keys == NULL) || out_num_keys == NULL) {
    return set_error(NVE_ERROR_INVALID_ARGUMENT, "invalid get_keys args");
  }
  int64_t* k = (int64_t*)keys;
  int64_t n = 0;
  for (int64_t b = 0; b < t->num_buckets && n < capacity; ++b) {
    for (entry_t* e = t->buckets[b]; e != NULL && n < capacity; e = e->next) k[n++] = e->key;
  }
  *out_num_keys = n;
  return NVE_SUCCESS;
}

static const nve_ext_table_ops_t kOps = {
    sizeof(nve_ext_table_ops_t),
    my_destroy,
    my_pref_loc,
    my_device_id,
    my_max_row,
    my_key_size,
    my_invalid_key,
    my_dtype,
    my_counter_hits,
    my_reset_counter,
    my_get_counter,
    my_find,
    my_insert,
    my_update,
    my_update_accum,
    my_erase,
    my_clear,
    my_last_error,
    my_size,
    my_get_keys,
};

/* ============================================================================
 * Factory
 * ============================================================================ */

static void my_factory_destroy(void* self) { free(self); }

static nve_status_t my_produce(void* self, int64_t table_id, const char* cfg,
                               nve_ext_table_t* out) {
  (void)table_id;
  my_factory_t* factory = (my_factory_t*)self;
  if (out == NULL) return set_error(NVE_ERROR_INVALID_ARGUMENT, "out_table is null");
  /* Output ownership rule: zero before work, transfer only on success. */
  memset(out, 0, sizeof(*out));

  const int64_t row_size = json_get_int(cfg, "max_value_size", 256);
  const int64_t num_buckets = json_get_int(cfg, "num_buckets", factory->num_buckets);
  if (row_size <= 0 || num_buckets <= 0) {
    return set_error(NVE_ERROR_INVALID_ARGUMENT, "invalid max_value_size/num_buckets");
  }

  my_table_t* t = (my_table_t*)calloc(1, sizeof(my_table_t));
  if (t == NULL) return set_error(NVE_ERROR_OUT_OF_MEMORY, "table allocation failed");
  t->row_size = row_size;
  t->num_buckets = num_buckets;
  t->invalid_key = json_get_int(cfg, "invalid_key", -1);
  t->buckets = (entry_t**)calloc((size_t)num_buckets, sizeof(entry_t*));
  if (t->buckets == NULL) {
    free(t); /* error path: the plugin releases every partial allocation */
    return set_error(NVE_ERROR_OUT_OF_MEMORY, "bucket allocation failed");
  }
  atomic_init(&t->hit_counter, 0);

  out->self = t;
  out->ops = &kOps;
  return NVE_SUCCESS; /* ownership transfers only here */
}

static const nve_ext_table_factory_ops_t kFactoryOps = {
    sizeof(nve_ext_table_factory_ops_t),
    my_factory_destroy,
    my_produce,
    my_last_error,
};

nve_status_t nve_plugin_create_external_factory(const char* json_config,
                                                nve_ext_table_factory_t* out,
                                                const char** out_error) {
  if (out == NULL || out_error == NULL) return NVE_ERROR_INVALID_ARGUMENT;
  memset(out, 0, sizeof(*out));
  *out_error = NULL;

  my_factory_t* factory = (my_factory_t*)calloc(1, sizeof(my_factory_t));
  if (factory == NULL) {
    *out_error = "factory allocation failed";
    return NVE_ERROR_OUT_OF_MEMORY;
  }
  factory->num_buckets = json_get_int(json_config, "num_buckets", 1024);
  if (factory->num_buckets <= 0) {
    free(factory);
    *out_error = "num_buckets must be positive";
    return NVE_ERROR_INVALID_ARGUMENT;
  }

  out->self = factory;
  out->ops = &kFactoryOps;
  return NVE_SUCCESS; /* ownership transfers only here */
}
