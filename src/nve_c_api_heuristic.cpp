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

nve_status_t nve_heuristic_create_default(
    nve_heuristic_t* out, const float* thresholds, int64_t num_thresholds) {
  if (!out || !thresholds || num_thresholds <= 0) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "Invalid arguments");
  }
  NVE_C_TRY
    std::vector<float> t(thresholds, thresholds + num_thresholds);
    auto h = std::make_shared<nve::DefaultInsertHeuristic>(t);
    *out = new nve_heuristic_s{std::move(h)};
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_heuristic_create_never(nve_heuristic_t* out) {
  if (!out) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "out must not be NULL");
  }
  NVE_C_TRY
    auto h = std::make_shared<nve::NeverInsertHeuristic>();
    *out = new nve_heuristic_s{std::move(h)};
    return NVE_SUCCESS;
  NVE_C_CATCH
}

nve_status_t nve_heuristic_destroy(nve_heuristic_t heuristic) {
  if (!heuristic) {
    return nve_set_error(NVE_ERROR_INVALID_ARGUMENT, "heuristic must not be NULL");
  }
  delete heuristic;
  return NVE_SUCCESS;
}

}  // extern "C"
