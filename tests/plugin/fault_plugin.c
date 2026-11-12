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
 * Test-only external plugin with JSON-driven failure injection. It stores no
 * data; every knob makes one ABI stage fail with a chosen status so tests can
 * assert status/message fidelity through the adapter and the public C API.
 *
 * Factory JSON:   {"factory_status": N}   creation fails with status N.
 * Produce JSON:   {"produce_status": N}   produce fails with status N.
 *                 {"operation_status": N} every fallible table op returns N.
 *                 {"task_status": N}      find's execute_n_wg tasks return N.
 *                 {"value_dtype": N}      raw dtype ID the table reports.
 *                 {"report_key_size": N}  key size the table reports.
 *                 {"omit_key_ops": 1}     leaves the mandatory size/get_keys ops NULL.
 */

#include <plugin/nve_external_plugin.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

NVE_DEFINE_EXTERNAL_PLUGIN("NVE Fault Injection Test Plugin", "NVIDIA Corporation")

static _Thread_local char g_last_error[256];

static nve_status_t set_error(nve_status_t status, const char* message) {
  snprintf(g_last_error, sizeof(g_last_error), "%s", message);
  return status;
}

static const char* fault_last_error(void* self) {
  (void)self;
  return g_last_error;
}

/* Minimal quoted-key JSON integer extraction (test-grade). */
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
 * Table
 * ============================================================================ */

typedef struct {
  nve_status_t operation_status; /* != 0: every fallible op returns this   */
  nve_status_t task_status;      /* != 0: execute_n_wg tasks return this   */
  nve_data_type_t value_dtype;
  int64_t key_size;
  int64_t row_size;
} fault_table_t;

static void fault_destroy(void* self) { free(self); }

static nve_buffer_location_t fault_pref_loc(void* self) {
  (void)self;
  return NVE_BUFFER_LOCATION_HOST;
}

static int32_t fault_device_id(void* self) {
  (void)self;
  return -1;
}

static int64_t fault_max_row(void* self) { return ((fault_table_t*)self)->row_size; }

static int64_t fault_key_size(void* self) { return ((fault_table_t*)self)->key_size; }

static int64_t fault_invalid_key(void* self) {
  (void)self;
  return -1;
}

static nve_data_type_t fault_dtype(void* self) { return ((fault_table_t*)self)->value_dtype; }

static int64_t fault_counter_hits(void* self) {
  (void)self;
  return 1;
}

static nve_status_t fault_injected(fault_table_t* t) {
  if (t->operation_status == NVE_SUCCESS) return NVE_SUCCESS;
  return set_error(t->operation_status, "injected operation failure");
}

static nve_status_t fault_reset_counter(void* self, nve_ext_context_t ctx) {
  (void)ctx;
  return fault_injected((fault_table_t*)self);
}

static nve_status_t fault_get_counter(void* self, nve_ext_context_t ctx, int64_t* counter) {
  (void)ctx;
  fault_table_t* t = (fault_table_t*)self;
  if (t->operation_status != NVE_SUCCESS) return fault_injected(t);
  if (counter != NULL) *counter = 0;
  return NVE_SUCCESS;
}

/* One execute_n_wg task: marks its 64-key hit-mask word (task-private) as all
 * found, then returns the injected task status. */
typedef struct {
  uint64_t* hit_mask;
  int64_t n;
  nve_status_t task_status;
} find_job_t;

static nve_status_t find_task(void* user, int64_t index) {
  find_job_t* job = (find_job_t*)user;
  if (job->hit_mask != NULL) {
    const int64_t begin = index * 64;
    const int64_t count = job->n - begin < 64 ? job->n - begin : 64;
    if (count > 0) {
      job->hit_mask[index] = count == 64 ? ~0ull : (1ull << count) - 1u;
    }
  }
  return job->task_status;
}

static nve_status_t fault_find(void* self, nve_ext_context_t ctx, int64_t n, const void* keys,
                               uint64_t* hit_mask, int64_t value_stride, void* values,
                               int64_t* value_sizes) {
  (void)keys;
  (void)value_stride;
  (void)values;
  (void)value_sizes;
  fault_table_t* t = (fault_table_t*)self;
  if (t->operation_status != NVE_SUCCESS) return fault_injected(t);

  find_job_t job = {.hit_mask = hit_mask, .n = n, .task_status = t->task_status};
  const int64_t num_tasks = (n + 63) / 64;
  const nve_ext_nve_services_t* services = nve_plugin_nve_services();
  if (num_tasks > 0 && services != NULL &&
      NVE_ABI_HAS_FIELD(services, nve_ext_nve_services_t, execute_n_wg) &&
      NVE_ABI_HAS_FIELD(services, nve_ext_nve_services_t, last_error)) {
    const int64_t workgroups[2] = {0, 1};
    const nve_status_t status =
        services->execute_n_wg(ctx, num_tasks, find_task, &job, workgroups, 2,
                               /*tasks_per_workgroup=*/1);
    if (status != NVE_SUCCESS) {
      return set_error(status, services->last_error(ctx));
    }
  }
  return NVE_SUCCESS;
}

static nve_status_t fault_insert(void* self, nve_ext_context_t ctx, int64_t n, const void* keys,
                                 int64_t value_stride, int64_t value_size, const void* values) {
  (void)ctx;
  (void)n;
  (void)keys;
  (void)value_stride;
  (void)value_size;
  (void)values;
  return fault_injected((fault_table_t*)self);
}

static nve_status_t fault_update(void* self, nve_ext_context_t ctx, int64_t n, const void* keys,
                                 int64_t value_stride, int64_t value_size, const void* values) {
  (void)ctx;
  (void)n;
  (void)keys;
  (void)value_stride;
  (void)value_size;
  (void)values;
  return fault_injected((fault_table_t*)self);
}

static nve_status_t fault_update_accum(void* self, nve_ext_context_t ctx, int64_t n,
                                       const void* keys, int64_t update_stride,
                                       int64_t update_size, const void* updates,
                                       nve_data_type_t update_dtype) {
  (void)ctx;
  (void)n;
  (void)keys;
  (void)update_stride;
  (void)update_size;
  (void)updates;
  (void)update_dtype;
  return fault_injected((fault_table_t*)self);
}

static nve_status_t fault_erase(void* self, nve_ext_context_t ctx, int64_t n, const void* keys) {
  (void)ctx;
  (void)n;
  (void)keys;
  return fault_injected((fault_table_t*)self);
}

static nve_status_t fault_clear(void* self, nve_ext_context_t ctx) {
  (void)ctx;
  return fault_injected((fault_table_t*)self);
}

/* Stores no data: size and get_keys report no keys. */
static nve_status_t fault_size(void* self, nve_ext_context_t ctx, int64_t* out_size) {
  (void)ctx;
  fault_table_t* t = (fault_table_t*)self;
  if (t->operation_status != NVE_SUCCESS) return fault_injected(t);
  *out_size = 0;
  return NVE_SUCCESS;
}

static nve_status_t fault_get_keys(void* self, nve_ext_context_t ctx, int64_t capacity, void* keys,
                                   int64_t* out_num_keys) {
  (void)ctx;
  (void)capacity;
  (void)keys;
  fault_table_t* t = (fault_table_t*)self;
  if (t->operation_status != NVE_SUCCESS) return fault_injected(t);
  *out_num_keys = 0;
  return NVE_SUCCESS;
}

static const nve_ext_table_ops_t kOps = {
    sizeof(nve_ext_table_ops_t),
    fault_destroy,
    fault_pref_loc,
    fault_device_id,
    fault_max_row,
    fault_key_size,
    fault_invalid_key,
    fault_dtype,
    fault_counter_hits,
    fault_reset_counter,
    fault_get_counter,
    fault_find,
    fault_insert,
    fault_update,
    fault_update_accum,
    fault_erase,
    fault_clear,
    fault_last_error,
    fault_size,
    fault_get_keys,
};

/* kOps without the mandatory size/get_keys ops; filled by produce for {"omit_key_ops": 1}. */
static nve_ext_table_ops_t g_ops_without_key_ops;

/* ============================================================================
 * Factory
 * ============================================================================ */

static void fault_factory_destroy(void* self) { free(self); }

static nve_status_t fault_produce(void* self, int64_t table_id, const char* cfg,
                                  nve_ext_table_t* out) {
  (void)self;
  (void)table_id;
  if (out == NULL) return set_error(NVE_ERROR_INVALID_ARGUMENT, "out_table is null");
  memset(out, 0, sizeof(*out));

  const nve_status_t produce_status =
      (nve_status_t)json_get_int(cfg, "produce_status", NVE_SUCCESS);
  if (produce_status != NVE_SUCCESS) {
    return set_error(produce_status, "injected produce failure");
  }

  fault_table_t* t = (fault_table_t*)calloc(1, sizeof(fault_table_t));
  if (t == NULL) return set_error(NVE_ERROR_OUT_OF_MEMORY, "table allocation failed");
  t->operation_status = (nve_status_t)json_get_int(cfg, "operation_status", NVE_SUCCESS);
  t->task_status = (nve_status_t)json_get_int(cfg, "task_status", NVE_SUCCESS);
  t->value_dtype = (nve_data_type_t)json_get_int(cfg, "value_dtype", NVE_DTYPE_FLOAT32);
  t->key_size = json_get_int(cfg, "report_key_size", 8);
  t->row_size = json_get_int(cfg, "report_row_size", 32);

  out->self = t;
  out->ops = &kOps;
  if (json_get_int(cfg, "omit_key_ops", 0) != 0) {
    g_ops_without_key_ops = kOps;
    g_ops_without_key_ops.size = NULL;
    g_ops_without_key_ops.get_keys = NULL;
    out->ops = &g_ops_without_key_ops;
  }
  return NVE_SUCCESS; /* ownership transfers only here */
}

static const nve_ext_table_factory_ops_t kFactoryOps = {
    sizeof(nve_ext_table_factory_ops_t),
    fault_factory_destroy,
    fault_produce,
    fault_last_error,
};

nve_status_t nve_plugin_create_external_factory(const char* json_config,
                                                nve_ext_table_factory_t* out,
                                                const char** out_error) {
  if (out == NULL || out_error == NULL) return NVE_ERROR_INVALID_ARGUMENT;
  memset(out, 0, sizeof(*out));
  *out_error = NULL;

  const nve_status_t factory_status =
      (nve_status_t)json_get_int(json_config, "factory_status", NVE_SUCCESS);
  if (factory_status != NVE_SUCCESS) {
    *out_error = "injected factory-creation failure";
    return factory_status;
  }

  void* factory = calloc(1, sizeof(int)); /* stateless; a non-null self */
  if (factory == NULL) {
    *out_error = "factory allocation failed";
    return NVE_ERROR_OUT_OF_MEMORY;
  }
  out->self = factory;
  out->ops = &kFactoryOps;
  return NVE_SUCCESS; /* ownership transfers only here */
}
