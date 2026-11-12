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

#ifndef NVE_EXTERNAL_PLUGIN_H
#define NVE_EXTERNAL_PLUGIN_H

/*
 * Interface for external-mode NVE plugins (pure C).
 *
 * This header contains the external table/factory ABI, NVE service callbacks,
 * and the NVE_DEFINE_EXTERNAL_PLUGIN authoring helper. The shared metadata
 * contract is provided by plugin/nve_plugin_common.h.
 *
 * No-unwind rule: an external plugin written in C++ must catch its own
 * exceptions and translate them to nve_status_t before returning. No language
 * exception may cross a function boundary declared in this interface.
 *
 * ABI layout baseline: 64-bit ELF, 8-bit bytes, 32-bit int32_t, 64-bit
 * pointers, default (non-packed) C struct layout. The static assertions at
 * the end of this header enforce the published layout at compile time.
 */

#include <stddef.h>

#include <plugin/nve_plugin_common.h>

#ifdef __cplusplus
#define NVE_ABI_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#include <stdalign.h> /* alignof in C11 */
#define NVE_ABI_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * External-plugin value types
 * ============================================================================ */

typedef int32_t nve_buffer_location_t;
enum {
  NVE_BUFFER_LOCATION_HOST = 0,   /* table operates on host pointers   */
  NVE_BUFFER_LOCATION_DEVICE = 1, /* table operates on device pointers */
};

/* ============================================================================
 * Backward-compatible struct growth
 * ============================================================================
 * Every growable ABI struct begins with `uint32_t struct_size`, set by the
 * producing side to sizeof(the_struct) at its build time. Fields appended by
 * a later ABI minor must only be read after this bounds check passes.
 */

#define NVE_ABI_HAS_FIELD(PTR, TYPE, FIELD)      \
  ((PTR) != NULL && (PTR)->struct_size >=        \
    offsetof(TYPE, FIELD) + sizeof(((TYPE*)0)->FIELD))

/* ============================================================================
 * External table (pure-C "object": opaque self + const vtable)
 * ============================================================================ */

/* Opaque host-owned per-operation context. Valid only for the dynamic extent
 * of the current table callback; a plugin must not retain it or use it
 * asynchronously. */
typedef struct nve_ext_context_s* nve_ext_context_t;

/*
 * Mirrors the pure-virtual set of nve::Table. create_execution_context is
 * intentionally NOT exposed: the host adapter inherits Table's default host
 * implementation, so contexts are host-managed and never cross the ABI.
 *
 * No-unwind rule: no function in this struct may allow a language exception
 * to cross the C boundary. Fallible ops return nve_status_t; last_error
 * returns the calling thread's message for its last failed operation.
 *
 * Capability getters (preferred_buffer_location .. value_dtype and
 * lookup_counter_hits) are infallible, non-throwing and synchronous; the host
 * reads them once during adoption. device_id is the bootstrap query: it (and
 * everything before the host caches a validated device ID) must select and
 * restore any required CUDA device itself. All later callbacks run with the
 * table's device already current.
 */
typedef struct nve_ext_table_ops_s {
  uint32_t struct_size; /* sizeof(nve_ext_table_ops_t) at plugin build time */

  /* lifecycle */
  void (*destroy)(void* self);

  /* capabilities — read once during host adoption */
  nve_buffer_location_t (*preferred_buffer_location)(void* self);
  int32_t (*device_id)(void* self);    /* -1 host, >=0 device */
  int64_t (*max_row_size)(void* self);
  int64_t (*key_size)(void* self);
  int64_t (*invalid_key)(void* self);
  nve_data_type_t (*value_dtype)(void* self);

  /* lookup counter */
  int64_t (*lookup_counter_hits)(void* self); /* nonzero: counter counts hits */
  nve_status_t (*reset_lookup_counter)(void* self, nve_ext_context_t ctx);
  nve_status_t (*get_lookup_counter)(void* self, nve_ext_context_t ctx, int64_t* counter);

  /* data ops — pointers are already resolved to the preferred location.
   * Nullability matches nve::Table::find: hit_mask may be NULL (treat all
   * keys as unresolved), values may be NULL (exists/count check only) and
   * value_sizes may be NULL. All other buffer arguments are non-NULL. */
  nve_status_t (*find)(void* self, nve_ext_context_t ctx, int64_t n,
                       const void* keys, uint64_t* hit_mask, int64_t value_stride,
                       void* values, int64_t* value_sizes);
  nve_status_t (*insert)(void* self, nve_ext_context_t ctx, int64_t n, const void* keys,
                         int64_t value_stride, int64_t value_size, const void* values);
  nve_status_t (*update)(void* self, nve_ext_context_t ctx, int64_t n, const void* keys,
                         int64_t value_stride, int64_t value_size, const void* values);
  nve_status_t (*update_accumulate)(void* self, nve_ext_context_t ctx, int64_t n,
                         const void* keys, int64_t update_stride, int64_t update_size,
                         const void* updates, nve_data_type_t update_dtype);
  nve_status_t (*erase)(void* self, nve_ext_context_t ctx, int64_t n, const void* keys);
  nve_status_t (*clear)(void* self, nve_ext_context_t ctx);

  /* calling thread's error text for its last op that returned != NVE_SUCCESS */
  const char* (*last_error)(void* self);

  nve_status_t (*size)(void* self, nve_ext_context_t ctx, int64_t* out_size);
  nve_status_t (*get_keys)(void* self, nve_ext_context_t ctx, int64_t capacity, void* keys,
                           int64_t* out_num_keys);
} nve_ext_table_ops_t;

typedef struct nve_ext_table_s { /* what produce() fills in */
  void* self;
  const nve_ext_table_ops_t* ops; /* stable storage owned by the plugin */
} nve_ext_table_t;

/* ============================================================================
 * External table factory
 * ============================================================================ */

typedef struct nve_ext_table_factory_ops_s {
  uint32_t struct_size; /* sizeof(nve_ext_table_factory_ops_t) */
  void (*destroy)(void* self);
  /* Zero out_table before work; transfer a nonzero handle only on success. */
  nve_status_t (*produce)(void* self, int64_t table_id, const char* json_config,
                          nve_ext_table_t* out_table);
  const char* (*last_error)(void* self);
} nve_ext_table_factory_ops_t;

typedef struct nve_ext_table_factory_s {
  void* self;
  const nve_ext_table_factory_ops_t* ops;
} nve_ext_table_factory_t;

/*
 * The external creation entry point (well-known symbol). Zero out_factory
 * before work; transfer a nonzero handle only on success.
 *
 * out_error is set to NULL on success. On failure it may point to
 * plugin-owned thread-local text valid until the next factory-creation call
 * on that thread; the host copies it immediately.
 */
NVE_C_API nve_status_t
nve_plugin_create_external_factory(const char* json_config,
                                   nve_ext_table_factory_t* out_factory,
                                   const char** out_error);

/* ============================================================================
 * NVE services (NVE -> plugin callbacks)
 * ============================================================================
 * The instance lives in the NVE binary and is stable for the process
 * lifetime. Every fallible callback catches all NVE exceptions, stores a
 * thread-local message retrievable via last_error, and returns a status.
 */
typedef struct nve_ext_nve_services_s {
  uint32_t struct_size; /* sizeof(nve_ext_nve_services_t) at NVE build time */

  /* threadpool services — execute_* calls block until all num_tasks complete.
     task is invoked with index in [0, num_tasks). A task returns a status and
     MUST NOT throw; execute_n returns the first failing task status. */
  nve_status_t (*num_workers)(nve_ext_context_t ctx, int64_t* out_workers);
  nve_status_t (*execute_n)(nve_ext_context_t ctx, int64_t num_tasks,
                            nve_status_t (*task)(void* user, int64_t index),
                            void* user);
  /* workgroup-aware flavor — distributes tasks across the given workgroup IDs
     in bursts of tasks_per_workgroup, mirroring the workgroup-collection
     overload of the host thread pool used by in-tree host tables. */
  nve_status_t (*execute_n_wg)(nve_ext_context_t ctx, int64_t num_tasks,
                            nve_status_t (*task)(void* user, int64_t index),
                            void* user, const int64_t* workgroups,
                            int64_t num_workgroups, int64_t tasks_per_workgroup);
  /* opaque cudaStream_t for device-capable tables */
  nve_status_t (*get_lookup_stream)(nve_ext_context_t ctx, void** out_stream);
  nve_status_t (*get_modify_stream)(nve_ext_context_t ctx, void** out_stream);
  /* context scratch allocation (host_alloc != 0 => host memory).
     NOT thread-safe: call it only from the table-op thread, never from inside
     an execute_n/execute_n_wg task. Growing a named buffer invalidates
     pointers previously returned for that name. */
  nve_status_t (*get_scratch)(nve_ext_context_t ctx, const char* name,
                              size_t size, int32_t host_alloc, void** out_ptr);
  /* message for the last NVE-service call that returned an error */
  const char* (*last_error)(nve_ext_context_t ctx);
} nve_ext_nve_services_t;

/* Optional symbol; NVE calls it after each external Plugin construction.
 * Registration must be thread-safe and idempotent for the same pointer. */
NVE_C_API void nve_plugin_set_nve_services(const nve_ext_nve_services_t* services);

#ifdef __cplusplus
} /* extern "C" */
#endif

/* ============================================================================
 * Published ABI layout (compile-time enforced)
 * ============================================================================ */

NVE_ABI_STATIC_ASSERT(sizeof(void*) == 8, "NVE plugin ABI requires 64-bit pointers");
NVE_ABI_STATIC_ASSERT(sizeof(nve_status_t) == 4, "nve_status_t must be 32-bit");
NVE_ABI_STATIC_ASSERT(sizeof(nve_data_type_t) == 4, "nve_data_type_t must be 32-bit");
NVE_ABI_STATIC_ASSERT(sizeof(nve_buffer_location_t) == 4,
                      "nve_buffer_location_t must be 32-bit");

NVE_ABI_STATIC_ASSERT(sizeof(nve_ext_table_ops_t) == 160, "unexpected table ops layout");
NVE_ABI_STATIC_ASSERT(alignof(nve_ext_table_ops_t) == 8, "unexpected table ops alignment");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_ops_t, struct_size) == 0, "table ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_ops_t, destroy) == 8, "table ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_ops_t, preferred_buffer_location) == 16,
                      "table ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_ops_t, device_id) == 24, "table ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_ops_t, max_row_size) == 32, "table ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_ops_t, key_size) == 40, "table ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_ops_t, invalid_key) == 48, "table ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_ops_t, value_dtype) == 56, "table ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_ops_t, lookup_counter_hits) == 64,
                      "table ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_ops_t, reset_lookup_counter) == 72,
                      "table ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_ops_t, get_lookup_counter) == 80,
                      "table ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_ops_t, find) == 88, "table ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_ops_t, insert) == 96, "table ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_ops_t, update) == 104, "table ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_ops_t, update_accumulate) == 112,
                      "table ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_ops_t, erase) == 120, "table ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_ops_t, clear) == 128, "table ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_ops_t, last_error) == 136, "table ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_ops_t, size) == 144, "table ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_ops_t, get_keys) == 152, "table ops layout");

NVE_ABI_STATIC_ASSERT(sizeof(nve_ext_table_t) == 16, "unexpected table handle layout");
NVE_ABI_STATIC_ASSERT(alignof(nve_ext_table_t) == 8, "unexpected table handle alignment");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_t, self) == 0, "table handle layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_t, ops) == 8, "table handle layout");

NVE_ABI_STATIC_ASSERT(sizeof(nve_ext_table_factory_ops_t) == 32,
                      "unexpected factory ops layout");
NVE_ABI_STATIC_ASSERT(alignof(nve_ext_table_factory_ops_t) == 8,
                      "unexpected factory ops alignment");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_factory_ops_t, struct_size) == 0,
                      "factory ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_factory_ops_t, destroy) == 8,
                      "factory ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_factory_ops_t, produce) == 16,
                      "factory ops layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_factory_ops_t, last_error) == 24,
                      "factory ops layout");

NVE_ABI_STATIC_ASSERT(sizeof(nve_ext_table_factory_t) == 16,
                      "unexpected factory handle layout");
NVE_ABI_STATIC_ASSERT(alignof(nve_ext_table_factory_t) == 8,
                      "unexpected factory handle alignment");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_factory_t, self) == 0, "factory handle layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_table_factory_t, ops) == 8, "factory handle layout");

NVE_ABI_STATIC_ASSERT(sizeof(nve_ext_nve_services_t) == 64,
                      "unexpected NVE services layout");
NVE_ABI_STATIC_ASSERT(alignof(nve_ext_nve_services_t) == 8,
                      "unexpected NVE services alignment");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_nve_services_t, struct_size) == 0,
                      "NVE services layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_nve_services_t, num_workers) == 8,
                      "NVE services layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_nve_services_t, execute_n) == 16,
                      "NVE services layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_nve_services_t, execute_n_wg) == 24,
                      "NVE services layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_nve_services_t, get_lookup_stream) == 32,
                      "NVE services layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_nve_services_t, get_modify_stream) == 40,
                      "NVE services layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_nve_services_t, get_scratch) == 48,
                      "NVE services layout");
NVE_ABI_STATIC_ASSERT(offsetof(nve_ext_nve_services_t, last_error) == 56,
                      "NVE services layout");

/* ============================================================================
 * External-plugin authoring helpers
 * ============================================================================ */

/* Atomic pointer storage that works in both C11 and C++ translation units. */
#ifdef __cplusplus
#include <atomic>
#define NVE_PLUGIN_ATOMIC_SERVICES_PTR_ std::atomic<const nve_ext_nve_services_t*>
#else
#include <stdatomic.h>
#define NVE_PLUGIN_ATOMIC_SERVICES_PTR_ const nve_ext_nve_services_t* _Atomic
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Returns the size-prefixed NVE-services table registered through the
 * handshake, or NULL before registration. The definition is emitted by
 * NVE_DEFINE_EXTERNAL_PLUGIN. Check appended services with NVE_ABI_HAS_FIELD.
 */
const nve_ext_nve_services_t* nve_plugin_nve_services(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

/**
 * Defines the metadata symbols for an external plugin plus the NVE-services
 * handshake and nve_plugin_nve_services() accessor.
 *
 * Expand this macro exactly once per plugin SO: it defines external-linkage
 * symbols, so a second expansion causes duplicate-symbol link errors. The
 * plugin must separately implement nve_plugin_create_external_factory().
 */
#define NVE_DEFINE_EXTERNAL_PLUGIN(NAME_STR, AUTHOR_STR)                                  \
  static NVE_PLUGIN_ATOMIC_SERVICES_PTR_ nve_plugin_nve_services_ptr_;                    \
  NVE_C_API uint32_t nve_plugin_abi_version_major(void) {                                 \
    return NVE_PLUGIN_ABI_VERSION_MAJOR; }                                                \
  NVE_C_API uint32_t nve_plugin_abi_version_minor(void) {                                 \
    return NVE_PLUGIN_ABI_VERSION_MINOR; }                                                \
  NVE_C_API const char* nve_plugin_name(void) { return NAME_STR; }                        \
  NVE_C_API const char* nve_plugin_author(void) { return AUTHOR_STR; }                    \
  NVE_C_API nve_plugin_mode_t nve_plugin_mode(void) { return NVE_PLUGIN_MODE_EXTERNAL; }  \
  NVE_C_API void nve_plugin_set_nve_services(const nve_ext_nve_services_t* services) {    \
    nve_plugin_nve_services_ptr_ = services; /* atomic store; repeat is a no-op */ }      \
  const nve_ext_nve_services_t* nve_plugin_nve_services(void) {                           \
    return nve_plugin_nve_services_ptr_; }

#endif /* NVE_EXTERNAL_PLUGIN_H */
