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

#include "nve_c_api_internal.hpp"
#include <json_support.hpp>

/* ============================================================================
 * Config conversion helpers
 * ============================================================================ */

static nve::GPUTableConfig convert_gpu_table_config(const nve_gpu_table_config_t* c) {
  nve::GPUTableConfig cfg;
  cfg.device_id = c->device_id;
  cfg.cache_size = c->cache_size;
  cfg.row_size_in_bytes = c->row_size_in_bytes;
  cfg.uvm_table = c->uvm_table;
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
  return cfg;
}

extern "C" {

/* ============================================================================
 * GPU Table creation
 * ============================================================================ */

nve_status_t nve_gpu_table_create(
    nve_table_t* out, nve_key_type_t key_type,
    const nve_gpu_table_config_t* config, nve_allocator_t allocator) {
  if (!out || !config) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "out and config must not be NULL");
  }
  NVE_C_TRY
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
  NVE_C_TRY
    table->ptr->find(ctx->ptr, n, keys, hit_mask, value_stride, values, value_sizes);
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_table_insert(
    nve_table_t table, nve_context_t ctx, int64_t n, const void* keys,
    int64_t value_stride, int64_t value_size, const void* values) {
  if (!table || !table->ptr || !ctx || !ctx->ptr) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "table and ctx must not be NULL");
  }
  NVE_C_TRY
    table->ptr->insert(ctx->ptr, n, keys, value_stride, value_size, values);
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_table_update(
    nve_table_t table, nve_context_t ctx, int64_t n, const void* keys,
    int64_t value_stride, int64_t value_size, const void* values) {
  if (!table || !table->ptr || !ctx || !ctx->ptr) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "table and ctx must not be NULL");
  }
  NVE_C_TRY
    table->ptr->update(ctx->ptr, n, keys, value_stride, value_size, values);
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
  NVE_C_TRY
    table->ptr->update_accumulate(ctx->ptr, n, keys, update_stride, update_size,
                                  updates, convert_dtype(update_dtype));
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
  NVE_C_TRY
    table->ptr->erase(ctx->ptr, n, keys);
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

nve_status_t nve_table_get_device_id(nve_table_t table, int32_t* out) {
  if (!table || !table->ptr || !out) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "table and out must not be NULL");
  }
  NVE_C_TRY
    *out = table->ptr->get_device_id();
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_table_get_max_row_size(nve_table_t table, int64_t* out) {
  if (!table || !table->ptr || !out) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "table and out must not be NULL");
  }
  NVE_C_TRY
    *out = table->ptr->get_max_row_size();
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

nve_status_t nve_table_get_lookup_counter(nve_table_t table, nve_context_t ctx, int64_t* counter) {
  if (!table || !table->ptr || !ctx || !ctx->ptr || !counter) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "Invalid arguments");
  }
  NVE_C_TRY
    table->ptr->get_lookup_counter(ctx->ptr, counter);
    return NVE_SUCCESS;
  NVE_C_CATCH
}

/* ============================================================================
 * Host Tables (via plugins / JSON config)
 * ============================================================================ */

nve_status_t nve_create_host_table_factory(
    nve_host_factory_t* out, const char* json_config) {
  if (!out || !json_config) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "out and json_config must not be NULL");
  }
  NVE_C_TRY
    auto json = nlohmann::json::parse(json_config);
    auto factory = nve::create_host_table_factory(json);
    *out = new nve_host_factory_s{std::move(factory)};
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_host_factory_destroy(nve_host_factory_t factory) {
  if (!factory) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "factory must not be NULL");
  }
  delete factory;
  return NVE_SUCCESS;
}

nve_status_t nve_host_factory_produce(
    nve_host_factory_t factory, int64_t table_id,
    const char* json_config, nve_table_t* out) {
  if (!factory || !factory->ptr || !json_config || !out) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "factory, json_config, and out must not be NULL");
  }
  NVE_C_TRY
    auto json = nlohmann::json::parse(json_config);
    auto key_size_it = json.find("key_size");
    if (key_size_it == json.end()) {
      return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "json_config must contain key_size");
    }
    int64_t key_size = key_size_it->get<int64_t>();
    nve_key_type_t key_type;
    if (key_size == 8) {
      key_type = NVE_KEY_INT64;
    } else if (key_size == 4) {
      key_type = NVE_KEY_INT32;
    } else {
      return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "key_size must be 4 or 8");
    }
    auto table = factory->ptr->produce(table_id, json);
    *out = new nve_table_s{std::move(table), key_type};
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_build_host_database(
    const char* json_config,
    nve_table_t** out_tables, int64_t** out_ids, int64_t* out_count) {
  if (!json_config || !out_tables || !out_ids || !out_count) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "All output pointers must not be NULL");
  }
  NVE_C_TRY
    auto json = nlohmann::json::parse(json_config);
    auto db = nve::build_host_database(json);

    const size_t count = db.size();
    auto* tables = new nve_table_t[count]();
    auto* ids = new int64_t[count]();

    int64_t i = 0;
    try {
      for (auto& [id, table_ptr] : db) {
        ids[i] = id;
        tables[i] = new nve_table_s{std::move(table_ptr), NVE_KEY_INT64};
        ++i;
      }
    } catch (...) {
      for (int64_t j = 0; j < i; ++j) {
        delete tables[j];
      }
      delete[] tables;
      delete[] ids;
      throw;
    }

    *out_tables = tables;
    *out_ids = ids;
    *out_count = static_cast<int64_t>(count);
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_free_host_database(
    nve_table_t* tables, int64_t* ids, int64_t count) {
  if (tables) {
    for (int64_t i = 0; i < count; ++i) {
      delete tables[i];
    }
    delete[] tables;
  }
  delete[] ids;
  return NVE_SUCCESS;
}

nve_status_t nve_host_table_size(
    nve_table_t table, nve_context_t ctx, int exact, int64_t* out) {
  if (!table || !table->ptr || !ctx || !ctx->ptr || !out) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "Invalid arguments");
  }
  NVE_C_TRY
    auto* host_table = dynamic_cast<nve::HostTableLike*>(table->ptr.get());
    if (!host_table) {
      return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "Table is not a host table");
    }
    *out = host_table->size(ctx->ptr, exact != 0);
    return NVE_SUCCESS;
  NVE_C_CATCH
}

}  // extern "C"
