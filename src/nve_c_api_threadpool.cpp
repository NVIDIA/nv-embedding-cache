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

extern "C" {

nve_status_t nve_thread_pool_create(nve_thread_pool_t* out, const char* json_config) {
  if (!out || !json_config) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "out and json_config must not be NULL");
  }
  NVE_C_TRY
    auto json = nlohmann::json::parse(json_config);
    auto pool = nve::create_thread_pool(json);
    *out = new nve_thread_pool_s{std::move(pool)};
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_thread_pool_destroy(nve_thread_pool_t pool) {
  if (!pool) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "pool must not be NULL");
  }
  delete pool;
  return NVE_SUCCESS;
}

nve_status_t nve_thread_pool_num_workers(nve_thread_pool_t pool, int64_t* out) {
  if (!pool || !pool->ptr || !out) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "Invalid arguments");
  }
  NVE_C_TRY
    *out = pool->ptr->num_workers();
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_configure_default_thread_pool(const char* json_config) {
  if (!json_config) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "json_config must not be NULL");
  }
  NVE_C_TRY
    auto json = nlohmann::json::parse(json_config);
    nve::configure_default_thread_pool(json);
    return NVE_SUCCESS;
  NVE_C_CATCH
}

}  // extern "C"
