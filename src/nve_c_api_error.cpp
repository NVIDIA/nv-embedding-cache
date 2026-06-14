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
#include <nve_types.hpp>

static thread_local std::string g_last_error;

nve_status_t nve_set_error(nve_status_t status, const char* message) {
  g_last_error = message ? message : "";
  return status;
}

nve_status_t nve_set_error(nve_status_t status, const std::string& message) {
  g_last_error = message;
  return status;
}

extern "C" {

nve_status_t nve_get_last_error(const char** message) {
  if (message) {
    *message = g_last_error.c_str();
  }
  return NVE_SUCCESS;
}

}  // extern "C"
