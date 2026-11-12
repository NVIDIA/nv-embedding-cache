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

#include <stdexcept>
#include <string>

#include <plugin/nve_external_plugin.h>
#include <plugin/plugin.hpp>
#include <plugin/plugin_loader.hpp>

namespace nve {

/**
 * Exception thrown for every external-plugin failure. Carries the original
 * (sanitized) nve_status_t so the public C API can round-trip the exact
 * status category instead of collapsing everything to a generic error.
 */
class ExternalPluginError final : public std::runtime_error {
 public:
  ExternalPluginError(nve_status_t status, std::string message)
      : std::runtime_error(std::move(message)), status_(status) {}

  nve_status_t status() const noexcept { return status_; }

 private:
  nve_status_t status_;
};

/** Maps unknown numeric statuses from external code to NVE_ERROR_RUNTIME. */
inline nve_status_t sanitize_external_status(nve_status_t status) noexcept {
  switch (status) {
    case NVE_SUCCESS:
    case NVE_ERROR_INVALID_ARGUMENT:
    case NVE_ERROR_CUDA:
    case NVE_ERROR_RUNTIME:
    case NVE_ERROR_NOT_IMPLEMENTED:
    case NVE_ERROR_OUT_OF_MEMORY:
      return status;
    default:
      return NVE_ERROR_RUNTIME;
  }
}

/** Converts a non-success external status into an ExternalPluginError. */
inline void check_external_status(nve_status_t status, const char* message) {
  if (status == NVE_SUCCESS) return;
  status = sanitize_external_status(status);
  throw ExternalPluginError(status, message ? message : "External plugin error");
}

/* Dtype mapping shared with the public C API (defined in nve_c_api_util.cpp).
 * Both directions are exhaustive and map out-of-range values to Unknown
 * instead of throwing. */
DataType_t convert_external_dtype(nve_data_type_t dt) noexcept;
nve_data_type_t convert_external_dtype(DataType_t dt) noexcept;

/**
 * Immutable capabilities captured once when an external table handle is
 * adopted. The ExternalPluginTable constructor copies these and performs no
 * plugin callbacks, so a C++ allocation failure cannot strand the handle.
 */
struct ExternalPluginTableCapabilities {
  int32_t device_id{-1};
  cudaMemoryType mem_type{cudaMemoryTypeUnregistered};
  int64_t max_row_size{0};
  int64_t key_size{0};
  int64_t invalid_key{-1};
  DataType_t value_dtype{DataType_t::Unknown};
  bool lookup_counter_hits{false};
};

/**
 * Adapts an external plugin's C table vtable behind the ordinary nve::Table
 * interface. Derives Table directly (not HostTableLike) so device-resident
 * external tables are supported; execution contexts use Table's default
 * host-managed implementation and never cross the C ABI.
 */
class ExternalPluginTable final : public Table {
 public:
  /** Called only by the adoption path, after `caps` has been validated. */
  ExternalPluginTable(plugin_library_ptr_t library, nve_ext_table_t handle,
                      const ExternalPluginTableCapabilities& caps);
  ~ExternalPluginTable() noexcept override;

  int32_t get_device_id() const override { return caps_.device_id; }
  int64_t get_max_row_size() const override { return caps_.max_row_size; }
  int64_t get_key_size() const override { return caps_.key_size; }
  int64_t get_invalid_key() const override { return caps_.invalid_key; }
  DataType_t get_value_type() const override { return caps_.value_dtype; }
  bool lookup_counter_hits() const override { return caps_.lookup_counter_hits; }

  void clear(context_ptr_t& ctx) override;
  void reset_lookup_counter(context_ptr_t& ctx) override;
  void get_lookup_counter(context_ptr_t& ctx, int64_t* counter) const override;

  void erase(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys) override;
  void find(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys,
            buffer_ptr<bitmask64_t> hit_mask, int64_t value_stride,
            buffer_ptr<void> values, buffer_ptr<int64_t> value_sizes) const override;
  void insert(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys,
              int64_t value_stride, int64_t value_size, buffer_ptr<const void> values) override;
  void update(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys,
              int64_t value_stride, int64_t value_size, buffer_ptr<const void> values) override;
  void update_accumulate(context_ptr_t& ctx, int64_t n, buffer_ptr<const void> keys,
                         int64_t update_stride, int64_t update_size,
                         buffer_ptr<const void> updates, DataType_t update_dtype) override;
  int64_t size(context_ptr_t& ctx) const override;
  int64_t get_keys(context_ptr_t& ctx, int64_t capacity, void* keys) const override;

 private:
  plugin_library_ptr_t library_;  // destroyed last: outlives the plugin code below
  nve_ext_table_t tbl_{};
  ExternalPluginTableCapabilities caps_{};
};

/**
 * Adapts an external plugin's C factory vtable behind nve::TableFactory.
 * Produced tables are validated, capability-cached ExternalPluginTable
 * instances that independently retain the library lease.
 */
class ExternalPluginTableFactory final : public TableFactory {
 public:
  /** Called only after a validated factory guard is ready to transfer ownership. */
  ExternalPluginTableFactory(plugin_library_ptr_t library, nve_ext_table_factory_t handle)
      : library_(std::move(library)), factory_(handle) {}
  ~ExternalPluginTableFactory() noexcept override;

  table_ptr_t produce(table_id_t id, const nlohmann::json& json) override;

 private:
  plugin_library_ptr_t library_;
  nve_ext_table_factory_t factory_{};
};

/**
 * Validates a successful external factory handle (mandatory 1.0 prefix and
 * callbacks) under a non-throwing destroy guard, and transfers it into a
 * ExternalPluginTableFactory only after construction succeeds. Used by the
 * Plugin loader; throws ExternalPluginError on a malformed handle.
 */
table_factory_ptr_t adopt_external_factory(plugin_library_ptr_t library,
                                           nve_ext_table_factory_t handle);

/**
 * The process-lifetime NVE-services table handed to external plugins via
 * their optional `nve_plugin_set_nve_services` symbol.
 */
const nve_ext_nve_services_t* external_nve_services() noexcept;

}  // namespace nve
