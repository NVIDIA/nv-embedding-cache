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

extern "C" {

nve_status_t nve_context_destroy(nve_context_t ctx) {
  if (!ctx) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "ctx must not be NULL");
  }
  delete ctx;
  return NVE_SUCCESS;
}

nve_status_t nve_context_wait(nve_context_t ctx) {
  if (!ctx || !ctx->ptr) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "ctx must not be NULL");
  }
  NVE_C_TRY
    ctx->ptr->wait();
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_context_get_lookup_stream(nve_context_t ctx, void** out_stream) {
  if (!ctx || !ctx->ptr || !out_stream) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "Invalid arguments");
  }
  *out_stream = static_cast<void*>(ctx->ptr->get_lookup_stream());
  return NVE_SUCCESS;
}

nve_status_t nve_context_get_modify_stream(nve_context_t ctx, void** out_stream) {
  if (!ctx || !ctx->ptr || !out_stream) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "Invalid arguments");
  }
  *out_stream = static_cast<void*>(ctx->ptr->get_modify_stream());
  return NVE_SUCCESS;
}

}  // extern "C"
