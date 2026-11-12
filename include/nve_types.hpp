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

  // Potential additions:
  //   CSR_NoLast: same as CSR but without the last offset
  //   COO: two elements per key (bag_id, id_in_bag), sorted row-wise.
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

inline constexpr bool is_weighted_pooling(const PoolingType_t pooling_type) noexcept {
  return pooling_type == PoolingType_t::WeightedSum ||
         pooling_type == PoolingType_t::WeightedMean;
}

// We rely on DataTypeID_t being 32bit when converting to DataType_t
enum class DataTypeID_t : uint32_t {
  Unknown,
  Float32,
  BFloat,
  Float16,
  E4M3,
  E5M2,
  Float64,
  QInt8RowwiseF32,  // int8 values with per-row symmetric quantization (fp32 scale, no offset),
                    // dequantized as value*scale on lookup.
  QInt8RowwiseF16,  // int8 values with per-row symmetric quantization (fp16 scale, no offset).
  QUint8RowwiseF32, // uint8 values with per-row affine quantization (fp32 scale + offset),
                    // dequantized as value*scale + offset on lookup.
  QUint8RowwiseF16, // uint8 values with per-row affine quantization (fp16 scale + offset).
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
    case DataTypeID_t::QInt8RowwiseF32:
      return "qint8_rowwise_f32";
    case DataTypeID_t::QInt8RowwiseF16:
      return "qint8_rowwise_f16";
    case DataTypeID_t::QUint8RowwiseF32:
      return "quint8_rowwise_f32";
    case DataTypeID_t::QUint8RowwiseF16:
      return "quint8_rowwise_f16";
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
  // Per-row quantized int8/uint8. dtype_size() is the value element size (1 byte) for all;
  // the per-row scale (and offset for the affine uint8 variants) lives as trailing row metadata
  // accounted for in the row stride, not here. int8 variants are symmetric (scale only: 4 bytes
  // for F32, 2 bytes for F16); uint8 variants are affine (scale + offset: 8 bytes for F32, 4 bytes
  // for F16).
  QInt8RowwiseF32 = make_dtype<int8_t>(DataTypeID_t::QInt8RowwiseF32),
  QInt8RowwiseF16 = make_dtype<int8_t>(DataTypeID_t::QInt8RowwiseF16),
  QUint8RowwiseF32 = make_dtype<uint8_t>(DataTypeID_t::QUint8RowwiseF32),
  QUint8RowwiseF16 = make_dtype<uint8_t>(DataTypeID_t::QUint8RowwiseF16),
};

static constexpr DataTypeID_t dtype_id(const DataType_t dtype) noexcept {
  return static_cast<DataTypeID_t>(dtype);
}

static constexpr int64_t dtype_size(const DataType_t dtype) noexcept {
  return static_cast<int64_t>(static_cast<uint64_t>(dtype) >> 32);
}

// True for the per-row quantized storage types (int8 symmetric / uint8 affine). These store value
// bytes followed by trailing scale [+ offset] metadata and must be dequantized to float/half on
// lookup; dtype_size() reports the 1-byte value element for all of them.
inline constexpr bool is_quant_rowwise(const DataType_t dtype) noexcept {
  return dtype == DataType_t::QInt8RowwiseF32 || dtype == DataType_t::QInt8RowwiseF16 ||
         dtype == DataType_t::QUint8RowwiseF32 || dtype == DataType_t::QUint8RowwiseF16;
}

// Float type used for rowwise-quantization parameters and native dequantized output. The F32/F16
// suffix on a quantized dtype denotes this precision.
inline DataType_t quant_rowwise_output_dtype(const DataType_t dtype) {
  switch (dtype) {
    case DataType_t::QInt8RowwiseF32:
    case DataType_t::QUint8RowwiseF32: return DataType_t::Float32;
    case DataType_t::QInt8RowwiseF16:
    case DataType_t::QUint8RowwiseF16: return DataType_t::Float16;
    default:
      NVE_THROW_("Not a rowwise-quantized data type: ", static_cast<uint64_t>(dtype));
  }
  return DataType_t::Unknown;
}

// Number of trailing parameter values stored after each row's quantized values: symmetric QInt8
// stores only a scale; affine QUint8 stores a scale and offset.
inline int64_t quant_rowwise_meta_count(const DataType_t dtype) {
  switch (dtype) {
    case DataType_t::QInt8RowwiseF32:
    case DataType_t::QInt8RowwiseF16: return 1;
    case DataType_t::QUint8RowwiseF32:
    case DataType_t::QUint8RowwiseF16: return 2;
    default:
      NVE_THROW_("Not a rowwise-quantized data type: ", static_cast<uint64_t>(dtype));
  }
  return 0;
}

inline int64_t quant_rowwise_scale_bytes(const DataType_t dtype) {
  return dtype_size(quant_rowwise_output_dtype(dtype));
}

inline int64_t quant_rowwise_meta_bytes(const DataType_t dtype) {
  return quant_rowwise_meta_count(dtype) * quant_rowwise_scale_bytes(dtype);
}

// Stored bytes for one row of num_elements values, including rowwise-quant metadata.
// Throws instead of wrapping on negative or overflowing sizes.
inline int64_t row_size_in_bytes(const DataType_t dtype, const int64_t num_elements) {
  NVE_CHECK_(num_elements >= 0, "Row element count must be non-negative, got ", num_elements);
  const uint64_t value_bytes = checked_mul(static_cast<uint64_t>(num_elements),
                                           static_cast<uint64_t>(dtype_size(dtype)), "Row size");
  const int64_t meta_bytes = is_quant_rowwise(dtype) ? quant_rowwise_meta_bytes(dtype) : 0;
  NVE_CHECK_(value_bytes <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - meta_bytes),
             "Row size computation overflows result type");
  return static_cast<int64_t>(value_bytes) + meta_bytes;
}

// Inverse of row_size_in_bytes(): value elements in a stored row, excluding rowwise-quant metadata.
// Throws if the row holds no values or is not a whole number of elements.
inline int64_t row_num_elements(const DataType_t dtype, const int64_t row_bytes) {
  const int64_t meta_bytes = is_quant_rowwise(dtype) ? quant_rowwise_meta_bytes(dtype) : 0;
  NVE_CHECK_(row_bytes > meta_bytes, "Row size ", row_bytes, " holds no values beyond its ",
             meta_bytes, " metadata bytes");
  const int64_t value_bytes = row_bytes - meta_bytes;
  NVE_CHECK_(value_bytes % dtype_size(dtype) == 0, "Row size ", row_bytes,
             " is not a whole number of value elements");
  return value_bytes / dtype_size(dtype);
}

static constexpr const char* to_string(const DataType_t dtype) { return to_string(dtype_id(dtype)); }

static inline std::ostream& operator<<(std::ostream& o, const DataType_t dtype) {
  return o << to_string(dtype);
}

void to_json(nlohmann::json& json, const DataType_t e);

void from_json(const nlohmann::json& j, DataType_t& e);

static constexpr DataType_t data_type(const DataTypeID_t dtype_id) {
  switch (dtype_id) {
    case DataTypeID_t::Float32:
      return DataType_t::Float32;
    case DataTypeID_t::Float16:
      return DataType_t::Float16;
    case DataTypeID_t::BFloat:
      return DataType_t::BFloat;
    case DataTypeID_t::E4M3:
      return DataType_t::E4M3;
    case DataTypeID_t::E5M2:
      return DataType_t::E5M2;
    case DataTypeID_t::Float64:
      return DataType_t::Float64;
    case DataTypeID_t::QInt8RowwiseF32:
      return DataType_t::QInt8RowwiseF32;
    case DataTypeID_t::QInt8RowwiseF16:
      return DataType_t::QInt8RowwiseF16;
    case DataTypeID_t::QUint8RowwiseF32:
      return DataType_t::QUint8RowwiseF32;
    case DataTypeID_t::QUint8RowwiseF16:
      return DataType_t::QUint8RowwiseF16;
    default:
      NVE_THROW_("Unknown data type ID!");
  }
}

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
