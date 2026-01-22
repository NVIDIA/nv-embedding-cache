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

#pragma GCC diagnostic push
#if defined(__clang__)
#pragma GCC diagnostic ignored "-Wimplicit-int-conversion"
#endif
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#pragma GCC diagnostic pop

#include <bit_ops.hpp>
#include <json_support.hpp>

namespace nve {

enum class SparseType_t : uint64_t {
  Fixed,
  CSR,
  COO,

  // Potential additions:
  //   CSR_NoLast: same as CSR but without the last offset
  //   COO_Transposed: COO but instead of array of pairs {row0,col0,row1,col1,...}, arrays of rows
  //   then all cols {row0,row1,...,col0,col1,...}
};

enum class PoolingType_t : uint64_t {
  Concatenate,
  Sum,
  Mean,
  WeightedSum,
  WeightedMean,
};

// We rely on DataTypeID_t being 32bit when converting to DataType_t
enum class DataTypeID_t : uint32_t {
  Unknown,
  Float32,
  BFloat,
  Float16,
  E4M3,
  E5M2,
  Float64,
};

static constexpr const char* to_string(const DataTypeID_t dt_id) {
  switch (dt_id) {
    case DataTypeID_t::Unknown:
      return "unknown";
    case DataTypeID_t::Float32:
      return "float32";
    case DataTypeID_t::Float16:
      return "float16";
    case DataTypeID_t::BFloat:
      return "bfloat";
    case DataTypeID_t::E4M3:
      return "e4m3";
    case DataTypeID_t::E5M2:
      return "e5m2";
    case DataTypeID_t::Float64:
      return "float64";
  }
  NVE_THROW_("Unknown data type ID!");
}

static inline std::ostream& operator<<(std::ostream& o, const DataTypeID_t dt_id) {
  return o << to_string(dt_id);
}

template <typename T>
static constexpr uint64_t make_dtype(const DataTypeID_t id) noexcept {
  NVE_ASSERT_(sizeof(T) < (UINT64_C(1) << 16));
  static_assert(sizeof(DataTypeID_t) == sizeof(uint32_t));
  constexpr auto id_bits = sizeof(id) * 8;
  return static_cast<uint64_t>(id) | (sizeof(T) << id_bits);
}

enum class DataType_t : uint64_t {
  Unknown = make_dtype<char>(DataTypeID_t::Unknown),  // Invalid data type (default).
  Float32 =
      make_dtype<float>(DataTypeID_t::Float32),  // IEEE-754 32 bit single precision format (E8M23).
  Float16 =
      make_dtype<half>(DataTypeID_t::Float16),  // IEEE-754 16 bit half precision format (E5M10).
  BFloat = make_dtype<nv_bfloat16>(DataTypeID_t::BFloat),  // Brain floating point format (E8M7).
  E4M3 = make_dtype<__nv_fp8_e4m3>(
      DataTypeID_t::E4M3),  // https://arxiv.org/abs/2209.05433, typically used for activations.
  E5M2 = make_dtype<__nv_fp8_e5m2>(
      DataTypeID_t::E5M2),  // https://arxiv.org/abs/2209.05433, typically used for gradients.
  Float64 = make_dtype<double>(
      DataTypeID_t::Float64),  // IEEE-754 64 bit double precision format (E11M52).
};

static constexpr DataTypeID_t dtype_id(const DataType_t dtype) noexcept {
  return static_cast<DataTypeID_t>(dtype);
}

static constexpr int64_t dtype_size(const DataType_t dtype) noexcept {
  return static_cast<int64_t>(static_cast<uint64_t>(dtype) >> 32);
}

static constexpr const char* to_string(const DataType_t dtype) { return to_string(dtype_id(dtype)); }

static inline std::ostream& operator<<(std::ostream& o, const DataType_t dtype) {
  return o << to_string(dtype);
}

void to_json(nlohmann::json& json, const DataType_t e);

void from_json(const nlohmann::json& j, DataType_t& e);

template <typename T>
static constexpr DataType_t data_type() {
  if constexpr (std::is_same_v<T, char>) {
    return DataType_t::Unknown;
  } else if constexpr (std::is_same_v<T, float>) {
    return DataType_t::Float32;
  } else if constexpr (std::is_same_v<T, half>) {
    return DataType_t::Float16;
  } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
    return DataType_t::BFloat;
  } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
    return DataType_t::E4M3;
  } else if constexpr (std::is_same_v<T, __nv_fp8_e5m2>) {
    return DataType_t::E5M2;
  } else if constexpr (std::is_same_v<T, double>) {
    return DataType_t::Float64;
  } else {
    static_assert(dependent_false_v<T>);
  }
}

template <DataType_t DType>
using type_t = std::conditional_t<
    DType == DataType_t::Float32, float,
    std::conditional_t<
        DType == DataType_t::Float16, half,
        std::conditional_t<
            DType == DataType_t::BFloat, nv_bfloat16,
            std::conditional_t<DType == DataType_t::E4M3, __nv_fp8_e4m3,
                               std::conditional_t<DType == DataType_t::E5M2, __nv_fp8_e5m2,
                                                  std::conditional_t<DType == DataType_t::Float64,
                                                                     double, void>>>>>>;

class Allocator;
using allocator_ptr_t = std::shared_ptr<Allocator>;

class ExecutionContext;
using context_ptr_t = std::shared_ptr<ExecutionContext>;

class ThreadPool;
using thread_pool_ptr_t = std::shared_ptr<ThreadPool>;

class Table;
using table_ptr_t = std::shared_ptr<Table>;

}  // namespace nve
