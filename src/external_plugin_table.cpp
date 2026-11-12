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

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <vector>

#include <buffer_wrapper.hpp>
#include <common.hpp>
#include <cuda_ops/cuda_common.h>
#include <cuda_support.hpp>
#include <execution_context.hpp>
#include <nlohmann/json.hpp>
#include <plugin/external_plugin_table.hpp>
#include <thread_pool_base.hpp>

/* Host-side definition of the opaque per-operation context. The plugin only
 * ever holds the pointer, so members may be added without an ABI change. */
struct nve_ext_context_s {
  nve::context_ptr_t* ctx;  // the live execution context for this op
  int32_t device_id;        // -1 host; selected in callbacks and worker tasks
};

namespace nve {

namespace {

/* ============================================================================
 * NVE services (NVE -> plugin callbacks)
 * ============================================================================ */

thread_local char g_nve_service_error[1024] = {0};

void set_nve_service_error(const char* message) noexcept {
  std::snprintf(g_nve_service_error, sizeof(g_nve_service_error), "%s",
                message ? message : "unknown NVE-service error");
}

/* Every fallible service callback runs under this fence: no exception may
 * cross the external function-pointer boundary. */
template <class Fn>
nve_status_t svc_guard(nve_ext_context_t c, Fn&& fn) noexcept {
  if (c == nullptr || c->ctx == nullptr || !*c->ctx) {
    set_nve_service_error("invalid external operation context");
    return NVE_ERROR_INVALID_ARGUMENT;
  }
  try {
    g_nve_service_error[0] = '\0';
    return fn();
  } catch (const ExternalPluginError& e) {
    set_nve_service_error(e.what());
    return e.status();
  } catch (const std::bad_alloc&) {
    set_nve_service_error("out of memory");
    return NVE_ERROR_OUT_OF_MEMORY;
  } catch (const nve::InvalidArgumentError& e) {
    set_nve_service_error(e.what());
    return NVE_ERROR_INVALID_ARGUMENT;
  } catch (const nve::RuntimeError<cudaError_t>& e) {
    set_nve_service_error(e.what());
    return NVE_ERROR_CUDA;
  } catch (const std::exception& e) {
    set_nve_service_error(e.what());
    return NVE_ERROR_RUNTIME;
  } catch (...) {
    set_nve_service_error("unknown NVE-service error");
    return NVE_ERROR_RUNTIME;
  }
}

nve_status_t svc_num_workers(nve_ext_context_t c, int64_t* out) noexcept {
  return svc_guard(c, [&] {
    NVE_CHECK_ARG_(out != nullptr, "out_workers is null");
    *out = (*c->ctx)->get_thread_pool()->num_workers();  // int64_t, no narrowing
    return NVE_SUCCESS;
  });
}

/* Folds per-task statuses to the first failure and installs a generic message
 * on the calling table-op thread after the synchronous barrier (worker TLS is
 * not visible here). Per-task detail stays in the plugin's own `user` state. */
template <class Run>
nve_status_t svc_execute_common(nve_ext_context_t c, int64_t num_tasks,
                               nve_status_t (*task)(void*, int64_t), void* user,
                               Run&& run) {
  NVE_CHECK_ARG_(task != nullptr && num_tasks >= 0, "invalid execute_n arguments");
  std::atomic<int32_t> first_error{NVE_SUCCESS};
  run([=, &first_error](int64_t i) noexcept {
    // Fully fenced: a throw here would unwind through the pool barrier while
    // queued task copies still reference the stack state captured above.
    nve_status_t s;
    try {
      ScopedDevice scope(c->device_id);  // CUDA state is thread-local; no-op for -1
      s = task(user, i);  // external task obeys the no-unwind rule
    } catch (const nve::RuntimeError<cudaError_t>&) {
      s = NVE_ERROR_CUDA;
    } catch (...) {
      s = NVE_ERROR_RUNTIME;
    }
    int32_t expected = NVE_SUCCESS;
    if (s != NVE_SUCCESS) first_error.compare_exchange_strong(expected, s);
  });
  const nve_status_t result =
      sanitize_external_status(static_cast<nve_status_t>(first_error.load()));
  if (result != NVE_SUCCESS) {
    set_nve_service_error("external execute_n task returned an error");
  }
  return result;
}

nve_status_t svc_execute_n(nve_ext_context_t c, int64_t num_tasks,
                          nve_status_t (*task)(void*, int64_t), void* user) noexcept {
  return svc_guard(c, [&] {
    return svc_execute_common(c, num_tasks, task, user,
                             [&](const ThreadPool::indexed_task_type& wrapped) {
                               (*c->ctx)->get_thread_pool()->execute_n(0, num_tasks, wrapped);
                             });
  });
}

nve_status_t svc_execute_n_wg(nve_ext_context_t c, int64_t num_tasks,
                             nve_status_t (*task)(void*, int64_t), void* user,
                             const int64_t* workgroups, int64_t num_workgroups,
                             int64_t tasks_per_workgroup) noexcept {
  return svc_guard(c, [&] {
    NVE_CHECK_ARG_(workgroups != nullptr && num_workgroups > 0 && tasks_per_workgroup > 0 &&
                       num_tasks >= 0,
                   "invalid execute_n_wg workgroup arguments");
    // The pool dereferences one workgroup entry per burst (wrapping), so only
    // the first ceil(num_tasks / tasks_per_workgroup) entries are ever used.
    // Copy and validate exactly those: a bogus huge num_workgroups fails as an
    // invalid argument instead of exhausting memory before validation, and a
    // negative ID would only trip a check inside the pool after earlier bursts
    // were already queued.
    const int64_t bursts{num_tasks / tasks_per_workgroup +
                         (num_tasks % tasks_per_workgroup != 0 ? 1 : 0)};
    const std::vector<int64_t> wgs(workgroups, workgroups + std::min(num_workgroups, bursts));
    NVE_CHECK_ARG_(std::all_of(wgs.begin(), wgs.end(), [](int64_t wg) { return wg >= 0; }),
                   "execute_n_wg workgroup IDs must be non-negative");
    return svc_execute_common(c, num_tasks, task, user,
                             [&](const ThreadPool::indexed_task_type& wrapped) {
                               (*c->ctx)->get_thread_pool()->execute_n(0, num_tasks, wrapped, wgs,
                                                                       tasks_per_workgroup);
                             });
  });
}

nve_status_t svc_get_lookup_stream(nve_ext_context_t c, void** out) noexcept {
  return svc_guard(c, [&] {
    NVE_CHECK_ARG_(out != nullptr, "out_stream is null");
    ScopedDevice scope(c->device_id);
    *out = (*c->ctx)->get_lookup_stream();
    return NVE_SUCCESS;
  });
}

nve_status_t svc_get_modify_stream(nve_ext_context_t c, void** out) noexcept {
  return svc_guard(c, [&] {
    NVE_CHECK_ARG_(out != nullptr, "out_stream is null");
    ScopedDevice scope(c->device_id);
    *out = (*c->ctx)->get_modify_stream();
    return NVE_SUCCESS;
  });
}

nve_status_t svc_get_scratch(nve_ext_context_t c, const char* name, size_t size,
                            int32_t host_alloc, void** out) noexcept {
  return svc_guard(c, [&] {
    NVE_CHECK_ARG_(name != nullptr && out != nullptr, "invalid scratch arguments");
    ScopedDevice scope(c->device_id);
    // Table-op thread only: ExecutionContext::get_buffer is unsynchronized.
    *out = (*c->ctx)->get_buffer(name, size, host_alloc != 0 /*host_alloc*/);
    return NVE_SUCCESS;
  });
}

const char* svc_last_error(nve_ext_context_t c) noexcept {
  (void)c;
  return g_nve_service_error;
}

const nve_ext_nve_services_t kNveServices = {
    sizeof(nve_ext_nve_services_t), svc_num_workers,       svc_execute_n,   svc_execute_n_wg,
    svc_get_lookup_stream,          svc_get_modify_stream, svc_get_scratch, svc_last_error};

/* ============================================================================
 * External handle adoption (non-throwing destroy guards)
 * ============================================================================ */

[[noreturn]] void throw_malformed(const char* what) {
  throw ExternalPluginError(NVE_ERROR_RUNTIME, what);
}

/** Owns a successful external table handle until the C++ wrapper takes over. */
class ExternalPluginTableGuard {
 public:
  explicit ExternalPluginTableGuard(nve_ext_table_t handle) : handle_(handle) {}
  ~ExternalPluginTableGuard() noexcept {
    if (released_) return;
    try {
      // Before the device ID is cached the plugin self-selects its device
      // (bootstrap rule); afterwards destruction runs under the table device.
      ScopedDevice scope(device_known_ ? device_id_ : -1);
      handle_.ops->destroy(handle_.self);
    } catch (const std::exception& e) {
      NVE_LOG_ERROR_("External table cleanup failed: ", e.what());
    } catch (...) {
      NVE_LOG_ERROR_("External table cleanup failed with an unknown exception");
    }
  }

  NVE_PREVENT_COPY_AND_MOVE_(ExternalPluginTableGuard);

  void cache_device(int32_t device_id) {
    device_id_ = device_id;
    device_known_ = true;
  }
  void release() { released_ = true; }
  const nve_ext_table_t& handle() const { return handle_; }
  const ExternalPluginTableCapabilities& capabilities() const { return caps_; }
  ExternalPluginTableCapabilities& capabilities() { return caps_; }

 private:
  nve_ext_table_t handle_{};
  ExternalPluginTableCapabilities caps_{};
  int32_t device_id_{-1};
  bool device_known_{false};
  bool released_{false};
};

/** Owns an external factory handle until ExternalPluginTableFactory takes over. */
class ExternalPluginFactoryGuard {
 public:
  explicit ExternalPluginFactoryGuard(nve_ext_table_factory_t handle) : handle_(handle) {}
  ~ExternalPluginFactoryGuard() noexcept {
    if (released_) return;
    try {
      handle_.ops->destroy(handle_.self);  // plugin self-scopes if device-sensitive
    } catch (const std::exception& e) {
      NVE_LOG_ERROR_("External factory cleanup failed: ", e.what());
    } catch (...) {
      NVE_LOG_ERROR_("External factory cleanup failed with an unknown exception");
    }
  }

  NVE_PREVENT_COPY_AND_MOVE_(ExternalPluginFactoryGuard);

  void release() { released_ = true; }
  const nve_ext_table_factory_t& handle() const { return handle_; }

 private:
  nve_ext_table_factory_t handle_{};
  bool released_{false};
};

/* Validates a produced table handle and captures its immutable capabilities.
 * The returned guard owns the handle until release(). */
void validate_external_table(ExternalPluginTableGuard& guard) {
  const nve_ext_table_ops_t* ops{guard.handle().ops};

  // Mandatory callbacks beyond destroy (which was validated before guard install).
  if (!NVE_ABI_HAS_FIELD(ops, nve_ext_table_ops_t, get_keys) ||
      ops->preferred_buffer_location == nullptr || ops->device_id == nullptr ||
      ops->max_row_size == nullptr || ops->key_size == nullptr ||
      ops->invalid_key == nullptr || ops->value_dtype == nullptr ||
      ops->lookup_counter_hits == nullptr || ops->reset_lookup_counter == nullptr ||
      ops->get_lookup_counter == nullptr || ops->find == nullptr || ops->insert == nullptr ||
      ops->update == nullptr || ops->update_accumulate == nullptr || ops->erase == nullptr ||
      ops->clear == nullptr || ops->last_error == nullptr || ops->size == nullptr ||
      ops->get_keys == nullptr) {
    throw_malformed("External table vtable is missing mandatory callbacks");
  }

  void* self{guard.handle().self};

  // Bootstrap query: the plugin self-selects its device inside device_id().
  const int32_t device_id{ops->device_id(self)};
  if (device_id < -1) {
    throw_malformed("External table reported an invalid device ID");
  }
  guard.cache_device(device_id);

  // All remaining capability queries run under the table's device.
  ScopedDevice scope(device_id);
  ExternalPluginTableCapabilities& caps{guard.capabilities()};
  caps.device_id = device_id;

  const nve_buffer_location_t location{ops->preferred_buffer_location(self)};
  switch (location) {
    case NVE_BUFFER_LOCATION_HOST:
      if (device_id >= 0) {
        throw_malformed("External table prefers host buffers but reports a CUDA device ID");
      }
      caps.mem_type = cudaMemoryTypeUnregistered;
      break;
    case NVE_BUFFER_LOCATION_DEVICE:
      if (device_id < 0) {
        throw_malformed("External table prefers device buffers but reports no CUDA device");
      }
      caps.mem_type = cudaMemoryTypeDevice;
      break;
    default:
      throw_malformed("External table reported an unknown buffer location");
  }

  caps.max_row_size = ops->max_row_size(self);
  caps.key_size = ops->key_size(self);
  caps.invalid_key = ops->invalid_key(self);
  const nve_data_type_t raw_dtype{ops->value_dtype(self)};
  caps.value_dtype = convert_external_dtype(raw_dtype);
  if (caps.value_dtype == DataType_t::Unknown && raw_dtype != NVE_DTYPE_UNKNOWN) {
    throw_malformed("External table reported an unknown value dtype");
  }
  caps.lookup_counter_hits = ops->lookup_counter_hits(self) != 0;

  if (caps.key_size <= 0 || caps.max_row_size <= 0) {
    throw_malformed("External table reported non-positive key or row size");
  }
}

}  // namespace

/* ============================================================================
 * ExternalPluginTable
 * ============================================================================ */

namespace {

/* Operation-scoped C context + shared error raising for table ops. */
void check_table_status(const nve_ext_table_t& tbl, nve_status_t status, const char* op) {
  if (status == NVE_SUCCESS) return;
  const char* message{tbl.ops->last_error(tbl.self)};
  check_external_status(status, message != nullptr && message[0] != '\0' ? message : op);
}

}  // namespace

ExternalPluginTable::ExternalPluginTable(plugin_library_ptr_t library, nve_ext_table_t handle,
                                         const ExternalPluginTableCapabilities& caps)
    : library_(std::move(library)), tbl_(handle), caps_(caps) {}

ExternalPluginTable::~ExternalPluginTable() noexcept {
  try {
    ScopedDevice scope(caps_.device_id);
    tbl_.ops->destroy(tbl_.self);  // external callback obeys the no-unwind rule
  } catch (const std::exception& e) {
    // Prefer a logged leak over a second destroy attempt on an unknown device.
    NVE_LOG_ERROR_("External table destroy failed: ", e.what());
  } catch (...) {
    NVE_LOG_ERROR_("External table destroy failed with an unknown exception");
  }
}

void ExternalPluginTable::clear(context_ptr_t& ctx) {
  ScopedDevice scope(caps_.device_id);
  nve_ext_context_s cctx{&ctx, caps_.device_id};
  check_table_status(tbl_, tbl_.ops->clear(tbl_.self, &cctx), "External table clear failed");
}

int64_t ExternalPluginTable::size(context_ptr_t& ctx) const {
  ScopedDevice scope(caps_.device_id);
  nve_ext_context_s cctx{&ctx, caps_.device_id};
  int64_t n{};
  check_table_status(tbl_, tbl_.ops->size(tbl_.self, &cctx, &n), "External table size failed");
  NVE_CHECK_(n >= 0, "External table reported a negative size");
  return n;
}

int64_t ExternalPluginTable::get_keys(context_ptr_t& ctx, const int64_t capacity, void* const keys) const {
  NVE_CHECK_(capacity >= 0, "capacity must not be negative");
  ScopedDevice scope(caps_.device_id);
  nve_ext_context_s cctx{&ctx, caps_.device_id};
  int64_t n{};
  check_table_status(tbl_, tbl_.ops->get_keys(tbl_.self, &cctx, capacity, keys, &n),
                     "External table get_keys failed");
  NVE_CHECK_(n >= 0 && n <= capacity, "External table reported an invalid key count");
  return n;
}

void ExternalPluginTable::reset_lookup_counter(context_ptr_t& ctx) {
  ScopedDevice scope(caps_.device_id);
  nve_ext_context_s cctx{&ctx, caps_.device_id};
  check_table_status(tbl_, tbl_.ops->reset_lookup_counter(tbl_.self, &cctx),
                     "External table reset_lookup_counter failed");
}

void ExternalPluginTable::get_lookup_counter(context_ptr_t& ctx, int64_t* counter) const {
  NVE_CHECK_ARG_(counter != nullptr, "counter must not be null");
  ScopedDevice scope(caps_.device_id);
  nve_ext_context_s cctx{&ctx, caps_.device_id};
  check_table_status(tbl_, tbl_.ops->get_lookup_counter(tbl_.self, &cctx, counter),
                     "External table get_lookup_counter failed");
}

void ExternalPluginTable::find(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys,
                               buffer_ptr<bitmask64_t> hit_mask, int64_t value_stride,
                               buffer_ptr<void> values,
                               buffer_ptr<int64_t> value_sizes) const {
  ScopedDevice scope(caps_.device_id);
  auto lookup_stream = ctx->get_lookup_stream();
  NVE_CHECK_ARG_(n <= 0 || keys != nullptr, "keys must not be null");
  const void* k = keys ? keys->access_buffer(caps_.mem_type, true, lookup_stream) : nullptr;
  uint64_t* h = hit_mask ? hit_mask->access_buffer(caps_.mem_type, true, lookup_stream) : nullptr;
  void* v = values ? values->access_buffer(caps_.mem_type, false, lookup_stream) : nullptr;
  int64_t* s = value_sizes ? value_sizes->access_buffer(caps_.mem_type, false, lookup_stream) : nullptr;
  nve_ext_context_s cctx{&ctx, caps_.device_id};  // opaque to the plugin; operation-scoped
  check_table_status(tbl_, tbl_.ops->find(tbl_.self, &cctx, n, k, h, value_stride, v, s),
                     "External table find failed");
}

void ExternalPluginTable::insert(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys,
                                 int64_t value_stride, int64_t value_size,
                                 buffer_ptr<const void> values) {
  ScopedDevice scope(caps_.device_id);
  auto modify_stream = ctx->get_modify_stream();
  NVE_CHECK_ARG_(n <= 0 || keys != nullptr, "keys must not be null");
  const void* k = keys ? keys->access_buffer(caps_.mem_type, true, modify_stream) : nullptr;
  const void* v = values ? values->access_buffer(caps_.mem_type, true, modify_stream) : nullptr;
  nve_ext_context_s cctx{&ctx, caps_.device_id};
  check_table_status(tbl_,
                     tbl_.ops->insert(tbl_.self, &cctx, n, k, value_stride, value_size, v),
                     "External table insert failed");
}

void ExternalPluginTable::update(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys,
                                 int64_t value_stride, int64_t value_size,
                                 buffer_ptr<const void> values) {
  ScopedDevice scope(caps_.device_id);
  auto modify_stream = ctx->get_modify_stream();
  NVE_CHECK_ARG_(n <= 0 || keys != nullptr, "keys must not be null");
  const void* k = keys ? keys->access_buffer(caps_.mem_type, true, modify_stream) : nullptr;
  const void* v = values ? values->access_buffer(caps_.mem_type, true, modify_stream) : nullptr;
  nve_ext_context_s cctx{&ctx, caps_.device_id};
  check_table_status(tbl_,
                     tbl_.ops->update(tbl_.self, &cctx, n, k, value_stride, value_size, v),
                     "External table update failed");
}

void ExternalPluginTable::update_accumulate(context_ptr_t& ctx, int64_t n,
                                            buffer_ptr<const void> keys, int64_t update_stride,
                                            int64_t update_size, buffer_ptr<const void> updates,
                                            DataType_t update_dtype) {
  ScopedDevice scope(caps_.device_id);
  auto modify_stream = ctx->get_modify_stream();
  NVE_CHECK_ARG_(n <= 0 || keys != nullptr, "keys must not be null");
  const void* k = keys ? keys->access_buffer(caps_.mem_type, true, modify_stream) : nullptr;
  const void* u = updates ? updates->access_buffer(caps_.mem_type, true, modify_stream) : nullptr;
  nve_ext_context_s cctx{&ctx, caps_.device_id};
  check_table_status(tbl_,
                     tbl_.ops->update_accumulate(tbl_.self, &cctx, n, k, update_stride,
                                                 update_size, u,
                                                 convert_external_dtype(update_dtype)),
                     "External table update_accumulate failed");
}

void ExternalPluginTable::erase(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys) {
  ScopedDevice scope(caps_.device_id);
  auto modify_stream = ctx->get_modify_stream();
  NVE_CHECK_ARG_(n <= 0 || keys != nullptr, "keys must not be null");
  const void* k = keys ? keys->access_buffer(caps_.mem_type, true, modify_stream) : nullptr;
  nve_ext_context_s cctx{&ctx, caps_.device_id};
  check_table_status(tbl_, tbl_.ops->erase(tbl_.self, &cctx, n, k),
                     "External table erase failed");
}

/* ============================================================================
 * ExternalPluginTableFactory
 * ============================================================================ */

ExternalPluginTableFactory::~ExternalPluginTableFactory() noexcept {
  try {
    factory_.ops->destroy(factory_.self);  // plugin self-scopes if device-sensitive
  } catch (const std::exception& e) {
    NVE_LOG_ERROR_("External factory destroy failed: ", e.what());
  } catch (...) {
    NVE_LOG_ERROR_("External factory destroy failed with an unknown exception");
  }
}

table_ptr_t ExternalPluginTableFactory::produce(table_id_t id, const nlohmann::json& json) {
  nve_ext_table_t t{};
  const std::string cfg = json.dump();
  const nve_status_t s = factory_.ops->produce(factory_.self, id, cfg.c_str(), &t);
  if (s != NVE_SUCCESS) {
    // Error contract: the output stays zero and the plugin cleaned up.
    check_external_status(s, factory_.ops->last_error(factory_.self));
  }

  // A success handle without a valid destroy prefix cannot be reclaimed safely.
  if (t.self == nullptr || !NVE_ABI_HAS_FIELD(t.ops, nve_ext_table_ops_t, destroy) ||
      t.ops->destroy == nullptr) {
    throw ExternalPluginError(NVE_ERROR_RUNTIME,
                              "External factory produced a malformed table handle");
  }

  ExternalPluginTableGuard guard(t);   // owns t while validation and allocation run
  validate_external_table(guard);      // caches device ID + immutable capabilities
  auto table = std::make_shared<ExternalPluginTable>(library_, guard.handle(),
                                                     guard.capabilities());
  guard.release();  // ExternalPluginTable owns t only after full construction
  return table;
}

table_factory_ptr_t adopt_external_factory(plugin_library_ptr_t library,
                                           nve_ext_table_factory_t handle) {
  // Minimal lifecycle prefix first: without a valid destroy we must reject
  // the handle without attempting an unsafe reclaim.
  if (handle.self == nullptr ||
      !NVE_ABI_HAS_FIELD(handle.ops, nve_ext_table_factory_ops_t, destroy) ||
      handle.ops->destroy == nullptr) {
    throw ExternalPluginError(NVE_ERROR_RUNTIME,
                              "External plugin returned a malformed factory handle");
  }

  ExternalPluginFactoryGuard guard(handle);
  if (!NVE_ABI_HAS_FIELD(handle.ops, nve_ext_table_factory_ops_t, last_error) ||
      handle.ops->produce == nullptr || handle.ops->last_error == nullptr) {
    throw ExternalPluginError(NVE_ERROR_RUNTIME,
                              "External factory vtable is missing mandatory ABI 1.0 callbacks");
  }
  auto factory = std::make_shared<ExternalPluginTableFactory>(std::move(library), guard.handle());
  guard.release();  // ExternalPluginTableFactory owns the handle from here on
  return factory;
}

const nve_ext_nve_services_t* external_nve_services() noexcept { return &kNveServices; }

}  // namespace nve
