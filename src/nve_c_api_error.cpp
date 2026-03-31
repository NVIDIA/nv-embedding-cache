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

/* ============================================================================
 * Enum conversion implementations
 * ============================================================================ */

nve::DataType_t convert_dtype(nve_data_type_t dt) {
  switch (dt) {
    case NVE_DTYPE_UNKNOWN:  return nve::DataType_t::Unknown;
    case NVE_DTYPE_FLOAT32:  return nve::DataType_t::Float32;
    case NVE_DTYPE_BFLOAT16: return nve::DataType_t::BFloat;
    case NVE_DTYPE_FLOAT16:  return nve::DataType_t::Float16;
    case NVE_DTYPE_E4M3:     return nve::DataType_t::E4M3;
    case NVE_DTYPE_E5M2:     return nve::DataType_t::E5M2;
    case NVE_DTYPE_FLOAT64:  return nve::DataType_t::Float64;
  }
  return nve::DataType_t::Unknown;
}


nve::SparseType_t convert_sparse_type(nve_sparse_type_t st) {
  switch (st) {
    case NVE_SPARSE_FIXED: return nve::SparseType_t::Fixed;
    case NVE_SPARSE_CSR:   return nve::SparseType_t::CSR;
    case NVE_SPARSE_COO:   return nve::SparseType_t::COO;
  }
  return nve::SparseType_t::Fixed;
}

nve::PoolingType_t convert_pooling_type(nve_pooling_type_t pt) {
  switch (pt) {
    case NVE_POOL_CONCATENATE:  return nve::PoolingType_t::Concatenate;
    case NVE_POOL_SUM:          return nve::PoolingType_t::Sum;
    case NVE_POOL_MEAN:         return nve::PoolingType_t::Mean;
    case NVE_POOL_WEIGHTED_SUM: return nve::PoolingType_t::WeightedSum;
    case NVE_POOL_WEIGHTED_MEAN:return nve::PoolingType_t::WeightedMean;
  }
  return nve::PoolingType_t::Concatenate;
}

nve::Partitioner_t convert_partitioner(nve_partitioner_t p) {
  switch (p) {
    case NVE_PART_ALWAYS_ZERO:    return nve::Partitioner_t::AlwaysZero;
    case NVE_PART_FOWLER_NOLL_VO: return nve::Partitioner_t::FowlerNollVo;
    case NVE_PART_MURMUR3:        return nve::Partitioner_t::Murmur3;
    case NVE_PART_RRXMRRXMSX0:    return nve::Partitioner_t::Rrxmrrxmsx0;
    case NVE_PART_STD_HASH:       return nve::Partitioner_t::StdHash;
  }
  return nve::Partitioner_t::FowlerNollVo;
}

nve::OverflowHandler_t convert_overflow_handler(nve_overflow_handler_t oh) {
  switch (oh) {
    case NVE_OVERFLOW_EVICT_RANDOM: return nve::OverflowHandler_t::EvictRandom;
    case NVE_OVERFLOW_EVICT_LRU:    return nve::OverflowHandler_t::EvictLRU;
    case NVE_OVERFLOW_EVICT_LFU:    return nve::OverflowHandler_t::EvictLFU;
  }
  return nve::OverflowHandler_t::EvictRandom;
}
