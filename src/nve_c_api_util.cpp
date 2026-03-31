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
#include <host_table.hpp>
#include <climits>

extern "C" {

/* ============================================================================
 * Version
 * ============================================================================ */

nve_status_t nve_version(int32_t* major, int32_t* minor, int32_t* patch) {
  if (!major || !minor || !patch) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "Output pointers must not be NULL");
  }
  *major = NVE_VERSION_MAJOR;
  *minor = NVE_VERSION_MINOR;
  *patch = NVE_VERSION_PATCH;
  return NVE_SUCCESS;
}

/* ============================================================================
 * Plugin loading
 * ============================================================================ */

nve_status_t nve_load_host_table_plugin(const char* plugin_name) {
  if (!plugin_name) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "plugin_name must not be NULL");
  }
  NVE_C_TRY
    nve::load_host_table_plugin(plugin_name);
    return NVE_SUCCESS;
  NVE_C_CATCH
}

/* ============================================================================
 * Config defaults
 * ============================================================================ */

nve_gpu_table_config_t nve_gpu_table_config_default(void) {
  nve_gpu_table_config_t c;
  c.device_id = 0;
  c.cache_size = 0;
  c.row_size_in_bytes = 0;
  c.uvm_table = NULL;
  c.count_misses = 1;
  c.max_modify_size = 1 << 20;
  c.value_dtype = NVE_DTYPE_UNKNOWN;
  c.private_stream = NULL;
  c.disable_uvm_update = 0;
  c.uvm_cpu_accumulate = 1;
  c.data_storage_on_host = 0;
  c.modify_on_gpu = 1;
  c.kernel_mode_type = 0;
  c.kernel_mode_value = 0;
  return c;
}

nve_gpu_embedding_layer_config_t nve_gpu_embedding_layer_config_default(void) {
  nve_gpu_embedding_layer_config_t c;
  c.layer_name = "";
  c.device_id = 0;
  c.embedding_table = NULL;
  c.num_embeddings = 0;
  c.embedding_width_in_bytes = 0;
  c.value_dtype = NVE_DTYPE_UNKNOWN;
  return c;
}

nve_linear_uvm_layer_config_t nve_linear_uvm_layer_config_default(void) {
  nve_linear_uvm_layer_config_t c;
  c.layer_name = "";
  c.insert_heuristic = NULL;
  c.min_insert_freq_gpu = 0;
  c.min_insert_size_gpu = 1 << 16;
  return c;
}

nve_hierarchical_layer_config_t nve_hierarchical_layer_config_default(void) {
  nve_hierarchical_layer_config_t c;
  c.layer_name = "";
  c.insert_heuristic = NULL;
  c.min_insert_freq_gpu = 0;
  c.min_insert_freq_host = 0;
  c.min_insert_size_gpu = 1 << 16;
  c.min_insert_size_host = 0;
  return c;
}

nve_overflow_policy_config_t nve_overflow_policy_config_default(void) {
  nve_overflow_policy_config_t c;
  c.overflow_margin = INT64_MAX;
  c.handler = NVE_OVERFLOW_EVICT_RANDOM;
  c.resolution_margin = 0.8;
  return c;
}

nve_host_table_config_t nve_host_table_config_default(void) {
  nve_host_table_config_t c;
  c.mask_size = 64;
  c.key_size = 8;
  c.max_value_size = 8;
  c.value_dtype = NVE_DTYPE_UNKNOWN;
  return c;
}

}  // extern "C"
