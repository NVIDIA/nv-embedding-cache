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

#include "nve_c_api_config.hpp"
#include <buffer_wrapper.hpp>
#include <json_support.hpp>
#include <plugin/plugin_loader.hpp>

/* ============================================================================
 * Config conversion helpers
 * ============================================================================ */

static nve::GPUTableConfig convert_gpu_table_config(const nve_gpu_table_config_t* c) {
  nve::GPUTableConfig cfg;
  cfg.device_id = c->device_id;
  cfg.cache_size = c->cache_size;
  cfg.row_size_in_bytes = c->row_size_in_bytes;
  cfg.uvm_table = c->uvm_table;
  cfg.uvm_num_rows = c->uvm_num_rows;
  cfg.count_misses = c->count_misses != 0;
  cfg.max_modify_size = c->max_modify_size;
  cfg.value_dtype = convert_dtype(c->value_dtype);
  cfg.private_stream = static_cast<cudaStream_t>(c->private_stream);
  cfg.disable_uvm_update = c->disable_uvm_update != 0;
  cfg.uvm_cpu_accumulate = c->uvm_cpu_accumulate != 0;
  cfg.data_storage_on_host = c->data_storage_on_host != 0;
  cfg.modify_on_gpu = c->modify_on_gpu != 0;
  cfg.kernel_mode_type = c->kernel_mode_type;
  cfg.kernel_mode_value = c->kernel_mode_value;
  cfg.invalid_key = c->invalid_key;
  return cfg;
}

static nve::LinearHostTableConfig convert_linear_host_table_config(
    const nve_linear_host_table_config_t* c, int64_t key_size) {
  nve::LinearHostTableConfig cfg;
  cfg.key_size = key_size;
  cfg.max_value_size = c->row_size_in_bytes;
  cfg.value_dtype = convert_dtype(c->value_dtype);
  cfg.emb_table = c->emb_table;
  cfg.num_rows = c->num_rows;
  cfg.max_threads = c->max_threads;
  cfg.task_size_threshold = c->task_size_threshold;
  return cfg;
}

extern "C" {

/* ============================================================================
 * GPU Table creation
 * ============================================================================ */

nve_status_t nve_gpu_table_create(
    nve_table_t* out, nve_key_type_t key_type,
    const nve_gpu_table_config_t* config, nve_allocator_t allocator) {
  if (!out) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "out must not be NULL");
  }
  *out = NULL;
  NVE_C_TRY
    nve_gpu_table_config_t normalized;
    const auto status = nve_read_config(config, normalized);
    if (status != NVE_SUCCESS) {
      return status;
    }
    config = &normalized;

    auto cpp_config = convert_gpu_table_config(config);
    auto alloc = unwrap_allocator(allocator);
    nve::table_ptr_t table;
    switch (key_type) {
      case NVE_KEY_INT32:
        table = std::make_shared<nve::GpuTable<int32_t>>(cpp_config, alloc);
        break;
      case NVE_KEY_INT64:
        table = std::make_shared<nve::GpuTable<int64_t>>(cpp_config, alloc);
        break;
      default:
        return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "Unsupported key type");
    }
    *out = new nve_table_s{std::move(table), key_type};
    return NVE_SUCCESS;
  NVE_C_CATCH
}

/* ============================================================================
 * Linear Host Table creation
 * ============================================================================ */

nve_status_t nve_linear_host_table_create(
    nve_table_t* out, nve_key_type_t key_type,
    const nve_linear_host_table_config_t* config) {
  if (!out) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "out must not be NULL");
  }
  *out = NULL;
  NVE_C_TRY
    nve_linear_host_table_config_t normalized;
    const auto status = nve_read_config(config, normalized);
    if (status != NVE_SUCCESS) {
      return status;
    }

    nve::table_ptr_t table;
    switch (key_type) {
      case NVE_KEY_INT32:
        table = std::make_shared<nve::LinearHostTable<int32_t>>(
            convert_linear_host_table_config(&normalized, sizeof(int32_t)));
        break;
      case NVE_KEY_INT64:
        table = std::make_shared<nve::LinearHostTable<int64_t>>(
            convert_linear_host_table_config(&normalized, sizeof(int64_t)));
        break;
      default:
        return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "Unsupported key type");
    }
    *out = new nve_table_s{std::move(table), key_type};
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_table_destroy(nve_table_t table) {
  if (!table) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "table must not be NULL");
  }
  delete table;
  return NVE_SUCCESS;
}

/* ============================================================================
 * Table operations (polymorphic via virtual Table interface)
 * ============================================================================ */

nve_status_t nve_table_find(
    nve_table_t table, nve_context_t ctx, int64_t n, const void* keys,
    uint64_t* hit_mask, int64_t value_stride, void* values, int64_t* value_sizes) {
  if (!table || !table->ptr || !ctx || !ctx->ptr) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "table and ctx must not be NULL");
  }
  if (n > 0 && !keys) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "keys must not be NULL when n > 0");
  }
  NVE_C_TRY
    const auto key_size = table->ptr->get_key_size();
    const auto keys_buffer_size = static_cast<size_t>(n * key_size);
    const auto hitmask_elements = static_cast<size_t>(nve::ceil_div(n, nve::bitmask64::num_bits));
    const auto hitmask_buffer_size = static_cast<size_t>(hitmask_elements * sizeof(nve::bitmask64_t));
    const auto values_buffer_size = static_cast<size_t>(n * value_stride);
    const auto value_sizes_buffer_size = static_cast<size_t>(n) * sizeof(int64_t);

    auto keys_bw = keys
      ? std::make_shared<nve::BufferWrapper<const void>>(ctx->ptr, "keys", keys, keys_buffer_size)
      : nullptr;
    auto hit_mask_bw = hit_mask
      ? std::make_shared<nve::BufferWrapper<nve::bitmask64_t>>(ctx->ptr, "hit_mask", hit_mask, hitmask_buffer_size)
      : nullptr;
    auto values_bw = values
      ? std::make_shared<nve::BufferWrapper<void>>(ctx->ptr, "values", values, values_buffer_size)
      : nullptr;
    auto value_sizes_bw = value_sizes
      ? std::make_shared<nve::BufferWrapper<int64_t>>(ctx->ptr, "value_sizes", value_sizes, value_sizes_buffer_size)
      : nullptr;

    table->ptr->find(ctx->ptr, n, std::move(keys_bw), std::move(hit_mask_bw),
                        value_stride, std::move(values_bw), std::move(value_sizes_bw));
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_table_insert(
    nve_table_t table, nve_context_t ctx, int64_t n, const void* keys,
    int64_t value_stride, int64_t value_size, const void* values) {
  if (!table || !table->ptr || !ctx || !ctx->ptr) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "table and ctx must not be NULL");
  }
  if (n > 0 && !keys) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "keys must not be NULL when n > 0");
  }
  NVE_C_TRY
    const auto key_size = table->ptr->get_key_size();
    const auto keys_buffer_size = static_cast<size_t>(n * key_size);
    const auto values_buffer_size = static_cast<size_t>(n * value_stride);

    auto keys_bw = keys
      ? std::make_shared<nve::BufferWrapper<const void>>(ctx->ptr, "keys", keys, keys_buffer_size)
      : nullptr;
    auto values_bw = values
      ? std::make_shared<nve::BufferWrapper<const void>>(ctx->ptr, "values", values, values_buffer_size)
      : nullptr;

    table->ptr->insert(ctx->ptr, n, std::move(keys_bw), value_stride, value_size, std::move(values_bw));
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_table_update(
    nve_table_t table, nve_context_t ctx, int64_t n, const void* keys,
    int64_t value_stride, int64_t value_size, const void* values) {
  if (!table || !table->ptr || !ctx || !ctx->ptr) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "table and ctx must not be NULL");
  }
  if (n > 0 && !keys) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "keys must not be NULL when n > 0");
  }
  NVE_C_TRY
    const auto key_size = table->ptr->get_key_size();
    const auto keys_buffer_size = static_cast<size_t>(n * key_size);
    const auto values_buffer_size = static_cast<size_t>(n * value_stride);

    auto keys_bw = keys
      ? std::make_shared<nve::BufferWrapper<const void>>(ctx->ptr, "keys", keys, keys_buffer_size)
      : nullptr;
    auto values_bw = values
      ? std::make_shared<nve::BufferWrapper<const void>>(ctx->ptr, "values", values, values_buffer_size)
      : nullptr;

    table->ptr->update(ctx->ptr, n, std::move(keys_bw), value_stride, value_size, std::move(values_bw));
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_table_update_accumulate(
    nve_table_t table, nve_context_t ctx, int64_t n, const void* keys,
    int64_t update_stride, int64_t update_size, const void* updates,
    nve_data_type_t update_dtype) {
  if (!table || !table->ptr || !ctx || !ctx->ptr) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "table and ctx must not be NULL");
  }
  if (n > 0 && !keys) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "keys must not be NULL when n > 0");
  }
  NVE_C_TRY
    const auto key_size = table->ptr->get_key_size();
    const auto keys_buffer_size = static_cast<size_t>(n * key_size);
    const auto updates_buffer_size = static_cast<size_t>(n * update_stride);

    auto keys_bw = keys
      ? std::make_shared<nve::BufferWrapper<const void>>(ctx->ptr, "keys", keys, keys_buffer_size)
      : nullptr;
    auto updates_bw = updates
      ? std::make_shared<nve::BufferWrapper<const void>>(ctx->ptr, "updates", updates, updates_buffer_size)
      : nullptr;

    table->ptr->update_accumulate(ctx->ptr, n, std::move(keys_bw), update_stride, update_size,
                                     std::move(updates_bw), convert_dtype(update_dtype));
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_table_clear(nve_table_t table, nve_context_t ctx) {
  if (!table || !table->ptr || !ctx || !ctx->ptr) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "table and ctx must not be NULL");
  }
  NVE_C_TRY
    table->ptr->clear(ctx->ptr);
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_table_erase(
    nve_table_t table, nve_context_t ctx, int64_t n, const void* keys) {
  if (!table || !table->ptr || !ctx || !ctx->ptr) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "table and ctx must not be NULL");
  }
  if (n > 0 && !keys) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "keys must not be NULL when n > 0");
  }
  NVE_C_TRY
    const auto key_size = table->ptr->get_key_size();
    const auto keys_buffer_size = static_cast<size_t>(n * key_size);

    auto keys_bw = keys
      ? std::make_shared<nve::BufferWrapper<const void>>(ctx->ptr, "keys", keys, keys_buffer_size)
      : nullptr;

    table->ptr->erase(ctx->ptr, n, std::move(keys_bw));
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_table_create_execution_context(
    nve_table_t table, nve_context_t* out,
    void* lookup_stream, void* modify_stream,
    nve_thread_pool_t thread_pool, nve_allocator_t allocator) {
  if (!table || !table->ptr || !out) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "table and out must not be NULL");
  }
  NVE_C_TRY
    auto ctx = table->ptr->create_execution_context(
        static_cast<cudaStream_t>(lookup_stream),
        static_cast<cudaStream_t>(modify_stream),
        unwrap_thread_pool(thread_pool),
        unwrap_allocator(allocator));
    *out = new nve_context_s{std::move(ctx)};
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_table_get_device_id(const nve_table_t table, int32_t* out) {
  if (!table || !table->ptr || !out) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "table and out must not be NULL");
  }
  NVE_C_TRY
    *out = table->ptr->get_device_id();
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_table_get_max_row_size(const nve_table_t table, int64_t* out) {
  if (!table || !table->ptr || !out) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "table and out must not be NULL");
  }
  NVE_C_TRY
    *out = table->ptr->get_max_row_size();
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_table_get_key_size(const nve_table_t table, int64_t* out) {
  if (!table || !table->ptr || !out) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "table and out must not be NULL");
  }
  NVE_C_TRY
    *out = table->ptr->get_key_size();
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_table_reset_lookup_counter(nve_table_t table, nve_context_t ctx) {
  if (!table || !table->ptr || !ctx || !ctx->ptr) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "table and ctx must not be NULL");
  }
  NVE_C_TRY
    table->ptr->reset_lookup_counter(ctx->ptr);
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_table_get_lookup_counter(const nve_table_t table, nve_context_t ctx, int64_t* counter) {
  if (!table || !table->ptr || !ctx || !ctx->ptr || !counter) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "Invalid arguments");
  }
  NVE_C_TRY
    table->ptr->get_lookup_counter(ctx->ptr, counter);
    return NVE_SUCCESS;
  NVE_C_CATCH
}

/* ============================================================================
 * Tables (via plugins / JSON config)
 * ============================================================================ */

nve_status_t nve_create_table_factory(
    nve_table_factory_t* out, const char* plugin_path, const char* json_config) {
  if (out) {
    *out = NULL;  // never expose a partial handle, whatever fails below
  }
  if (!out || !plugin_path || !json_config) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT,
                         "out, plugin_path, and json_config must not be NULL");
  }
  if (plugin_path[0] == '\0') {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "plugin_path must not be empty");
  }
  NVE_C_TRY
    auto json = nlohmann::json::parse(json_config);
    if (!json.is_object()) {
      return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "json_config must be a JSON object");
    }
    // The local Plugin may die here: the factory retains the library lease.
    nve::Plugin plugin{std::filesystem::path{plugin_path}};
    auto factory = plugin.create_table_factory(json);
    *out = new nve_table_factory_s{std::move(factory)};
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_table_factory_destroy(nve_table_factory_t factory) {
  if (!factory) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "factory must not be NULL");
  }
  delete factory;
  return NVE_SUCCESS;
}

nve_status_t nve_table_factory_produce(
    nve_table_factory_t factory, int64_t table_id,
    const char* json_config, nve_table_t* out) {
  if (out) {
    *out = NULL;  // never expose a partial handle, whatever fails below
  }
  if (!factory || !factory->ptr || !json_config || !out) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "factory, json_config, and out must not be NULL");
  }
  NVE_C_TRY
    auto json = nlohmann::json::parse(json_config);
    if (!json.is_object()) {
      return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "json_config must be a JSON object");
    }
    auto table = factory->ptr->produce(table_id, json);
    if (!table) {
      return nve_set_error(NVE_ERROR_RUNTIME, "factory produced a null table");
    }
    // The produced table's validated key size is authoritative for the C
    // handle's key type; a classification failure destroys the table normally.
    const int64_t key_size = table->get_key_size();
    nve_key_type_t key_type;
    if (key_size == 8) {
      key_type = NVE_KEY_INT64;
    } else if (key_size == 4) {
      key_type = NVE_KEY_INT32;
    } else {
      return nve_set_error(NVE_ERROR_INVALID_ARGUMENT,
                           "produced table reports unsupported key size (must be 4 or 8)");
    }
    *out = new nve_table_s{std::move(table), key_type};
    return NVE_SUCCESS;
  NVE_C_CATCH
}

}  // extern "C"
