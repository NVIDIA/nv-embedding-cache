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

#include "include/common.hpp"
#include "include/thread_pool.hpp"
#include "include/nve_types.hpp"
#include <cuda_fp16.h>
#include <algorithm>
#include <cstring>
#include <vector>

namespace nve {

// Per-row quantized int8/uint8 layout: `row_width` value bytes followed by per-row
// scale (and offset, for the affine uint8 variants) as trailing metadata. The
// metadata element type is float for the *F32 variants and __half for the *F16
// variants. Dequant:
//   QInt8Rowwise*  (symmetric, int8):  value * scale
//   QUint8Rowwise* (affine,    uint8): value * scale + offset
inline bool is_quant_rowwise(const DataType_t dtype) {
  return dtype == DataType_t::QInt8RowwiseF32 || dtype == DataType_t::QInt8RowwiseF16 ||
         dtype == DataType_t::QUint8RowwiseF32 || dtype == DataType_t::QUint8RowwiseF16;
}

// Number of trailing scalar metadata elements per row: 1 (scale) for the symmetric
// int8 variants, 2 (scale + offset) for the affine uint8 variants.
inline int64_t quant_rowwise_meta_count(const DataType_t dtype) {
  return (dtype == DataType_t::QUint8RowwiseF32 || dtype == DataType_t::QUint8RowwiseF16) ? 2 : 1;
}

// Size in bytes of each scale/offset metadata element (fp32 vs fp16 variant).
inline int64_t quant_rowwise_scale_bytes(const DataType_t dtype) {
  return (dtype == DataType_t::QInt8RowwiseF32 || dtype == DataType_t::QUint8RowwiseF32)
             ? static_cast<int64_t>(sizeof(float))
             : static_cast<int64_t>(sizeof(__half));
}

// Bytes occupied by the trailing per-row metadata (scale [+ offset]).
inline int64_t quant_rowwise_meta_bytes(const DataType_t dtype) {
  return quant_rowwise_meta_count(dtype) * quant_rowwise_scale_bytes(dtype);
}

// Host-side scalar conversions to/from the float accumulation type. cuda_fp16.h
// (pulled in via nve_types.hpp) provides host implementations of __half2float /
// __float2half, so these compile under the plain host compiler.
template <typename T>
inline float pooling_to_float(T v) {
  return static_cast<float>(v);
}
template <>
inline float pooling_to_float<__half>(__half v) {
  return __half2float(v);
}

template <typename T>
inline T pooling_from_float(float v) {
  return static_cast<T>(v);
}
template <>
inline __half pooling_from_float<__half>(float v) {
  return __float2half(v);
}

/**
 * CPU pooling op: reduce `num_keys` gathered rows into `num_bags` output rows.
 *
 * The input is the per-key gathered embeddings (one row per key, concatenated).
 * Bags are defined by the sparse layout:
 *   - Fixed: each bag is `fixed_hotness` consecutive keys (num_bags = num_keys/hotness).
 *   - CSR:   bag b spans keys [offsets[b], offsets[b+1]); offsets has num_bags+1 entries.
 * Accumulation is performed in float regardless of in/out type for numerical
 * stability (notably for fp16), then cast to the output type.
 *
 * Parallelized over bags via the context thread pool; each task owns a disjoint
 * contiguous range of output rows, so there is no cross-task contention.
 *
 * @tparam InT      input (table) element type (float / __half)
 * @tparam OutT     output element type (float / __half)
 * @tparam WeightT  per-key weight element type (float / __half), unused unless weighted
 */
template <typename InT, typename OutT, typename WeightT, typename OffsetT>
void cpu_kernel_pooling(thread_pool_ptr_t thread_pool,
                        const int64_t num_bags,
                        const int64_t row_width,        // elements per row
                        const int8_t* input,            // gathered rows (num_keys rows)
                        const int64_t input_stride,     // bytes between input rows
                        int8_t* output,                 // pooled rows (num_bags rows)
                        const int64_t output_stride,    // bytes between output rows
                        const SparseType_t sparse_type,
                        const int64_t fixed_hotness,    // Fixed only
                        const OffsetT* offsets,         // CSR only (num_bags+1 entries)
                        const PoolingType_t pooling_type,
                        const WeightT* weights,         // per-key weights or nullptr
                        const int64_t num_workers) {
  const bool weighted = (pooling_type == PoolingType_t::WeightedSum) ||
                        (pooling_type == PoolingType_t::WeightedMean);
  const bool mean = (pooling_type == PoolingType_t::Mean) ||
                    (pooling_type == PoolingType_t::WeightedMean);

  const int64_t workers = std::max<int64_t>(1, num_workers);
  const int64_t bags_per_task = std::max<int64_t>(1, (num_bags + workers - 1) / workers);
  const int64_t num_tasks = (num_bags + bags_per_task - 1) / bags_per_task;

  const auto pool_task = [=](const int64_t task_idx) {
    const int64_t bag_start = task_idx * bags_per_task;
    const int64_t bag_end = std::min<int64_t>(bag_start + bags_per_task, num_bags);

    // Per-task float accumulator reused across this task's bags. Iterating with the
    // element index as the inner loop (contiguous in both the input row and acc)
    // turns the accumulation into an element-wise map the compiler can auto-vectorize,
    // rather than a per-element scalar reduction.
    std::vector<float> acc(static_cast<size_t>(row_width));

    for (int64_t bag = bag_start; bag < bag_end; bag++) {
      // Resolve the [start_key, end_key) span for this bag.
      int64_t start_key;
      int64_t end_key;
      if (sparse_type == SparseType_t::Fixed) {
        start_key = bag * fixed_hotness;
        end_key = start_key + fixed_hotness;
      } else {  // CSR
        start_key = offsets[bag];
        end_key = offsets[bag + 1];
      }
      const int64_t count = end_key - start_key;

      auto* out_row = reinterpret_cast<OutT*>(output + bag * output_stride);
      if (count <= 0) {
        std::memset(out_row, 0, static_cast<size_t>(row_width) * sizeof(OutT));
        continue;
      }

      // Accumulate this bag's rows in float, summing the weights as we go.
      std::fill(acc.begin(), acc.end(), 0.f);
      float weight_sum = 0.f;
      for (int64_t k = start_key; k < end_key; k++) {
        const auto* in_row = reinterpret_cast<const InT*>(input + k * input_stride);
        const float w = weighted ? pooling_to_float<WeightT>(weights[k]) : 1.f;
        weight_sum += w;
        for (int64_t e = 0; e < row_width; e++) {
          float v = pooling_to_float<InT>(in_row[e]);
          if (weighted) {
            v *= w;
          }
          acc[static_cast<size_t>(e)] += v;
        }
      }

      // Averaging denominator: WeightedMean divides by the sum of weights, Mean by
      // the element count. A zero denominator (e.g. weights summing to zero) yields
      // a zero row rather than a division by zero.
      float inv = 1.f;
      if (mean) {
        const float denom = weighted ? weight_sum : static_cast<float>(count);
        inv = (denom != 0.f) ? 1.f / denom : 0.f;
      }
      for (int64_t e = 0; e < row_width; e++) {
        out_row[e] = pooling_from_float<OutT>(acc[static_cast<size_t>(e)] * inv);
      }
    }
  };

  thread_pool->execute_n(0, num_tasks, pool_task);
}

/**
 * CPU pooling op for per-row quantized int8/uint8 input. Mirrors cpu_kernel_pooling
 * but dequantizes each row's values to float before combining: value*scale for the
 * symmetric int8 variants (HasOffset == false), or value*scale + offset for the
 * affine uint8 variants (HasOffset == true). The scale/offset are read once per key
 * (key-outer loop with a per-task float accumulator) since they live in trailing row
 * metadata away from the values.
 *
 * @tparam BaseT     stored value type (int8_t / uint8_t)
 * @tparam QScaleT   scale/offset metadata element type (float / __half)
 * @tparam HasOffset whether the row carries an offset after the scale (affine uint8)
 * @tparam OutT      output element type (float / __half)
 * @tparam WeightT   per-key weight element type (float / __half), unused unless weighted
 */
template <typename BaseT, typename QScaleT, bool HasOffset, typename OutT, typename WeightT, typename OffsetT>
void cpu_kernel_pooling_quant(thread_pool_ptr_t thread_pool,
                              const int64_t num_bags,
                              const int64_t row_width,        // values per row
                              const int8_t* input,            // gathered quantized rows
                              const int64_t input_stride,     // bytes per row (values + metadata)
                              int8_t* output,                 // pooled rows (num_bags rows)
                              const int64_t output_stride,    // bytes between output rows
                              const SparseType_t sparse_type,
                              const int64_t fixed_hotness,    // Fixed only
                              const OffsetT* offsets,         // CSR only (num_bags+1 entries)
                              const PoolingType_t pooling_type,
                              const WeightT* weights,         // per-key weights or nullptr
                              const int64_t num_workers) {
  const bool weighted = (pooling_type == PoolingType_t::WeightedSum) ||
                        (pooling_type == PoolingType_t::WeightedMean);
  const bool mean = (pooling_type == PoolingType_t::Mean) ||
                    (pooling_type == PoolingType_t::WeightedMean);

  const int64_t workers = std::max<int64_t>(1, num_workers);
  const int64_t bags_per_task = std::max<int64_t>(1, (num_bags + workers - 1) / workers);
  const int64_t num_tasks = (num_bags + bags_per_task - 1) / bags_per_task;

  const auto pool_task = [=](const int64_t task_idx) {
    const int64_t bag_start = task_idx * bags_per_task;
    const int64_t bag_end = std::min<int64_t>(bag_start + bags_per_task, num_bags);

    // Per-task float accumulator reused across this task's bags.
    std::vector<float> acc(static_cast<size_t>(row_width));

    for (int64_t bag = bag_start; bag < bag_end; bag++) {
      int64_t start_key;
      int64_t end_key;
      if (sparse_type == SparseType_t::Fixed) {
        start_key = bag * fixed_hotness;
        end_key = start_key + fixed_hotness;
      } else {  // CSR
        start_key = offsets[bag];
        end_key = offsets[bag + 1];
      }
      const int64_t count = end_key - start_key;

      auto* out_row = reinterpret_cast<OutT*>(output + bag * output_stride);
      if (count <= 0) {
        std::memset(out_row, 0, static_cast<size_t>(row_width) * sizeof(OutT));
        continue;
      }

      std::fill(acc.begin(), acc.end(), 0.f);
      float weight_sum = 0.f;
      for (int64_t k = start_key; k < end_key; k++) {
        const int8_t* row_bytes = input + k * input_stride;
        const auto* row = reinterpret_cast<const BaseT*>(row_bytes);
        // Load scale/offset from a potentially-unaligned address (row_width bytes may
        // not be a multiple of alignof(QScaleT)). We memcpy into a plain integer type
        // rather than directly into QScaleT to avoid -Wclass-memaccess: __half is a
        // struct with a protected member, which GCC rejects as a memcpy destination.
        // The compiler reduces each 2-/4-byte memcpy + conversion to a single load.
        const auto load_scale = [&](const int8_t* p) -> float {
          if constexpr (std::is_same_v<QScaleT, float>) {
            float v; std::memcpy(&v, p, sizeof(float)); return v;
          } else {
            uint16_t raw; std::memcpy(&raw, p, sizeof(uint16_t));
            return __half2float(__half_raw{raw});
          }
        };
        const float scale = load_scale(row_bytes + row_width);
        const float offset = HasOffset ? load_scale(row_bytes + row_width + sizeof(QScaleT)) : 0.f;
        const float w = weighted ? pooling_to_float<WeightT>(weights[k]) : 1.f;
        weight_sum += w;
        for (int64_t e = 0; e < row_width; e++) {
          float v = static_cast<float>(row[e]) * scale + offset;
          if (weighted) {
            v *= w;
          }
          acc[static_cast<size_t>(e)] += v;
        }
      }
      // Averaging denominator: WeightedMean divides by the sum of weights, Mean by
      // the element count. A zero denominator yields a zero row rather than dividing
      // by zero.
      float inv = 1.f;
      if (mean) {
        const float denom = weighted ? weight_sum : static_cast<float>(count);
        inv = (denom != 0.f) ? 1.f / denom : 0.f;
      }
      for (int64_t e = 0; e < row_width; e++) {
        const float val = acc[static_cast<size_t>(e)] * inv;
        out_row[e] = pooling_from_float<OutT>(val);
      }
    }
  };

  thread_pool->execute_n(0, num_tasks, pool_task);
}

/**
 * CPU concatenate op: type-convert or dequantize `num_rows` gathered rows into `num_rows`
 * output rows without any reduction. Each input row maps to exactly one output row.
 *
 * For Float→Float: element-wise type conversion (compiler reduces same-type to memcpy).
 * For quantized input: dequantization is handled by the overloads below.
 *
 * @tparam InT   input element type (float / __half)
 * @tparam OutT  output element type (float / __half)
 */
template <typename InT, typename OutT>
void cpu_kernel_concatenate(thread_pool_ptr_t thread_pool,
                            const int64_t num_rows,
                            const int64_t row_width,
                            const int8_t* input,
                            const int64_t input_stride,
                            int8_t* output,
                            const int64_t output_stride,
                            const int64_t num_workers) {
  const int64_t workers = std::max<int64_t>(1, num_workers);
  const int64_t rows_per_task = std::max<int64_t>(1, (num_rows + workers - 1) / workers);
  const int64_t num_tasks = (num_rows + rows_per_task - 1) / rows_per_task;

  const auto task = [=](const int64_t task_idx) {
    const int64_t row_start = task_idx * rows_per_task;
    const int64_t row_end = std::min<int64_t>(row_start + rows_per_task, num_rows);
    for (int64_t r = row_start; r < row_end; r++) {
      const auto* in_row = reinterpret_cast<const InT*>(input + r * input_stride);
      auto* out_row = reinterpret_cast<OutT*>(output + r * output_stride);
      for (int64_t e = 0; e < row_width; e++) {
        out_row[e] = pooling_from_float<OutT>(pooling_to_float<InT>(in_row[e]));
      }
    }
  };

  thread_pool->execute_n(0, num_tasks, task);
}

/**
 * CPU concatenate op for per-row quantized int8/uint8 input. Dequantizes each row's
 * values (value*scale for symmetric int8, value*scale+offset for affine uint8) and
 * writes one output row per input row. No reduction.
 *
 * @tparam BaseT     stored value type (int8_t / uint8_t)
 * @tparam QScaleT   scale/offset metadata element type (float / __half)
 * @tparam HasOffset whether the row carries an offset after the scale (affine uint8)
 * @tparam OutT      output element type (float / __half)
 */
template <typename BaseT, typename QScaleT, bool HasOffset, typename OutT>
void cpu_kernel_concatenate_quant(thread_pool_ptr_t thread_pool,
                                  const int64_t num_rows,
                                  const int64_t row_width,
                                  const int8_t* input,
                                  const int64_t input_stride,
                                  int8_t* output,
                                  const int64_t output_stride,
                                  const int64_t num_workers) {
  const int64_t workers = std::max<int64_t>(1, num_workers);
  const int64_t rows_per_task = std::max<int64_t>(1, (num_rows + workers - 1) / workers);
  const int64_t num_tasks = (num_rows + rows_per_task - 1) / rows_per_task;

  const auto task = [=](const int64_t task_idx) {
    const int64_t row_start = task_idx * rows_per_task;
    const int64_t row_end = std::min<int64_t>(row_start + rows_per_task, num_rows);

    const auto load_scale = [&](const int8_t* p) -> float {
      if constexpr (std::is_same_v<QScaleT, float>) {
        float v; std::memcpy(&v, p, sizeof(float)); return v;
      } else {
        uint16_t raw; std::memcpy(&raw, p, sizeof(uint16_t));
        return __half2float(__half_raw{raw});
      }
    };

    for (int64_t r = row_start; r < row_end; r++) {
      const int8_t* row_bytes = input + r * input_stride;
      const auto* row = reinterpret_cast<const BaseT*>(row_bytes);
      const float scale = load_scale(row_bytes + row_width);
      const float offset = HasOffset ? load_scale(row_bytes + row_width + sizeof(QScaleT)) : 0.f;
      auto* out_row = reinterpret_cast<OutT*>(output + r * output_stride);
      for (int64_t e = 0; e < row_width; e++) {
        out_row[e] = pooling_from_float<OutT>(static_cast<float>(row[e]) * scale + offset);
      }
    }
  };

  thread_pool->execute_n(0, num_tasks, task);
}

// Dispatch over output dtype for float/half concatenate.
template <typename InT>
void cpu_kernel_concatenate_dispatch_out(thread_pool_ptr_t thread_pool, const int64_t num_rows,
                                         const int64_t row_width, const int8_t* input,
                                         const int64_t input_stride, int8_t* output,
                                         const int64_t output_stride, const DataType_t out_type,
                                         const int64_t num_workers) {
  switch (out_type) {
    case DataType_t::Float32:
      cpu_kernel_concatenate<InT, float>(std::move(thread_pool), num_rows, row_width, input,
                                         input_stride, output, output_stride, num_workers);
      break;
    case DataType_t::Float16:
      cpu_kernel_concatenate<InT, __half>(std::move(thread_pool), num_rows, row_width, input,
                                          input_stride, output, output_stride, num_workers);
      break;
    default:
      NVE_THROW_("Unsupported concatenate output type ", out_type);
  }
}

// Dispatch over output dtype for quantized-input concatenate.
template <typename BaseT, typename QScaleT, bool HasOffset>
void cpu_kernel_concatenate_quant_dispatch_out(thread_pool_ptr_t thread_pool, const int64_t num_rows,
                                               const int64_t row_width, const int8_t* input,
                                               const int64_t input_stride, int8_t* output,
                                               const int64_t output_stride, const DataType_t out_type,
                                               const int64_t num_workers) {
  switch (out_type) {
    case DataType_t::Float32:
      cpu_kernel_concatenate_quant<BaseT, QScaleT, HasOffset, float>(
          std::move(thread_pool), num_rows, row_width, input, input_stride, output, output_stride,
          num_workers);
      break;
    case DataType_t::Float16:
      cpu_kernel_concatenate_quant<BaseT, QScaleT, HasOffset, __half>(
          std::move(thread_pool), num_rows, row_width, input, input_stride, output, output_stride,
          num_workers);
      break;
    default:
      NVE_THROW_("Unsupported concatenate output type ", out_type);
  }
}

/**
 * Entry point for the concatenate (no-reduction) path. Converts or dequantizes
 * `num_rows` gathered rows into `num_rows` output rows. The output buffer must be
 * sized for at least `num_rows * output_stride` bytes.
 *
 * Callers that have same-type Float32/Float16 input and output (i.e. a plain copy)
 * should prefer a direct memcpy rather than calling this function.
 */
inline void cpu_kernel_concatenate_dispatch(thread_pool_ptr_t thread_pool, const int64_t num_rows,
                                            const int64_t row_width, const int8_t* input,
                                            const int64_t input_stride, int8_t* output,
                                            const int64_t output_stride, const DataType_t in_type,
                                            const DataType_t out_type, const int64_t num_workers) {
  if (!is_quant_rowwise(in_type) && in_type == out_type) {
    NVE_LOG_WARNING_("cpu_kernel_concatenate_dispatch called with identical input and output type (",
                     in_type, ") — this is a plain memcpy; prefer a direct memcpy instead.");
  }
  switch (in_type) {
    case DataType_t::Float32:
      cpu_kernel_concatenate_dispatch_out<float>(std::move(thread_pool), num_rows, row_width, input,
                                                  input_stride, output, output_stride, out_type,
                                                  num_workers);
      break;
    case DataType_t::Float16:
      cpu_kernel_concatenate_dispatch_out<__half>(std::move(thread_pool), num_rows, row_width, input,
                                                   input_stride, output, output_stride, out_type,
                                                   num_workers);
      break;
    case DataType_t::QInt8RowwiseF32:
      cpu_kernel_concatenate_quant_dispatch_out<int8_t, float, /*HasOffset=*/false>(
          std::move(thread_pool), num_rows, row_width, input, input_stride, output, output_stride,
          out_type, num_workers);
      break;
    case DataType_t::QInt8RowwiseF16:
      cpu_kernel_concatenate_quant_dispatch_out<int8_t, __half, /*HasOffset=*/false>(
          std::move(thread_pool), num_rows, row_width, input, input_stride, output, output_stride,
          out_type, num_workers);
      break;
    case DataType_t::QUint8RowwiseF32:
      cpu_kernel_concatenate_quant_dispatch_out<uint8_t, float, /*HasOffset=*/true>(
          std::move(thread_pool), num_rows, row_width, input, input_stride, output, output_stride,
          out_type, num_workers);
      break;
    case DataType_t::QUint8RowwiseF16:
      cpu_kernel_concatenate_quant_dispatch_out<uint8_t, __half, /*HasOffset=*/true>(
          std::move(thread_pool), num_rows, row_width, input, input_stride, output, output_stride,
          out_type, num_workers);
      break;
    default:
      NVE_THROW_("Unsupported concatenate input type ", in_type);
  }
}

// Dispatch over weight dtype for weighted pooling (fp32 / fp16 weights).
template <typename InT, typename OutT, typename OffsetT>
void cpu_kernel_pooling_dispatch_weight(thread_pool_ptr_t thread_pool, const int64_t num_bags,
                                        const int64_t row_width, const int8_t* input,
                                        const int64_t input_stride, int8_t* output,
                                        const int64_t output_stride, const SparseType_t sparse_type,
                                        const int64_t fixed_hotness, const OffsetT* offsets,
                                        const PoolingType_t pooling_type, const void* weights,
                                        const DataType_t weight_type, const int64_t num_workers) {
  const bool weighted = (pooling_type == PoolingType_t::WeightedSum) ||
                        (pooling_type == PoolingType_t::WeightedMean);
  if (!weighted) {
    // Weight type is irrelevant; pick float as a placeholder (weights unused).
    cpu_kernel_pooling<InT, OutT, float, OffsetT>(std::move(thread_pool), num_bags, row_width, input,
                                         input_stride, output, output_stride, sparse_type,
                                         fixed_hotness, offsets, pooling_type, nullptr, num_workers);
    return;
  }
  NVE_CHECK_(weights != nullptr, "Weights must be provided for weighted pooling");
  switch (weight_type) {
    case DataType_t::Float32:
      cpu_kernel_pooling<InT, OutT, float, OffsetT>(std::move(thread_pool), num_bags, row_width, input,
                                           input_stride, output, output_stride, sparse_type,
                                           fixed_hotness, offsets, pooling_type,
                                           reinterpret_cast<const float*>(weights), num_workers);
      break;
    case DataType_t::Float16:
      cpu_kernel_pooling<InT, OutT, __half, OffsetT>(std::move(thread_pool), num_bags, row_width, input,
                                            input_stride, output, output_stride, sparse_type,
                                            fixed_hotness, offsets, pooling_type,
                                            reinterpret_cast<const __half*>(weights), num_workers);
      break;
    default:
      NVE_THROW_("Unsupported pooling weight type ", weight_type);
  }
}

// Dispatch over output dtype (fp32 / fp16).
template <typename InT, typename OffsetT>
void cpu_kernel_pooling_dispatch_out(thread_pool_ptr_t thread_pool, const int64_t num_bags,
                                     const int64_t row_width, const int8_t* input,
                                     const int64_t input_stride, int8_t* output,
                                     const int64_t output_stride, const SparseType_t sparse_type,
                                     const int64_t fixed_hotness, const OffsetT* offsets,
                                     const PoolingType_t pooling_type, const void* weights,
                                     const DataType_t weight_type, const DataType_t out_type,
                                     const int64_t num_workers) {
  switch (out_type) {
    case DataType_t::Float32:
      cpu_kernel_pooling_dispatch_weight<InT, float, OffsetT>(
          std::move(thread_pool), num_bags, row_width, input, input_stride, output, output_stride,
          sparse_type, fixed_hotness, offsets, pooling_type, weights, weight_type, num_workers);
      break;
    case DataType_t::Float16:
      cpu_kernel_pooling_dispatch_weight<InT, __half, OffsetT>(
          std::move(thread_pool), num_bags, row_width, input, input_stride, output, output_stride,
          sparse_type, fixed_hotness, offsets, pooling_type, weights, weight_type, num_workers);
      break;
    default:
      NVE_THROW_("Unsupported pooling output type ", out_type);
  }
}

// Dispatch over weight dtype for quantized-input weighted pooling.
template <typename BaseT, typename QScaleT, bool HasOffset, typename OutT, typename OffsetT>
void cpu_kernel_pooling_quant_dispatch_weight(
    thread_pool_ptr_t thread_pool, const int64_t num_bags, const int64_t row_width,
    const int8_t* input, const int64_t input_stride, int8_t* output, const int64_t output_stride,
    const SparseType_t sparse_type, const int64_t fixed_hotness, const OffsetT* offsets,
    const PoolingType_t pooling_type, const void* weights, const DataType_t weight_type,
    const int64_t num_workers) {
  const bool weighted = (pooling_type == PoolingType_t::WeightedSum) ||
                        (pooling_type == PoolingType_t::WeightedMean);
  if (!weighted) {
    cpu_kernel_pooling_quant<BaseT, QScaleT, HasOffset, OutT, float, OffsetT>(
        std::move(thread_pool), num_bags, row_width, input, input_stride, output, output_stride,
        sparse_type, fixed_hotness, offsets, pooling_type, nullptr, num_workers);
    return;
  }
  NVE_CHECK_(weights != nullptr, "Weights must be provided for weighted pooling");
  switch (weight_type) {
    case DataType_t::Float32:
      cpu_kernel_pooling_quant<BaseT, QScaleT, HasOffset, OutT, float, OffsetT>(
          std::move(thread_pool), num_bags, row_width, input, input_stride, output, output_stride,
          sparse_type, fixed_hotness, offsets, pooling_type,
          reinterpret_cast<const float*>(weights), num_workers);
      break;
    case DataType_t::Float16:
      cpu_kernel_pooling_quant<BaseT, QScaleT, HasOffset, OutT, __half, OffsetT>(
          std::move(thread_pool), num_bags, row_width, input, input_stride, output, output_stride,
          sparse_type, fixed_hotness, offsets, pooling_type,
          reinterpret_cast<const __half*>(weights), num_workers);
      break;
    default:
      NVE_THROW_("Unsupported pooling weight type ", weight_type);
  }
}

// Dispatch over output dtype (fp32 / fp16) for quantized input.
template <typename BaseT, typename QScaleT, bool HasOffset, typename OffsetT>
void cpu_kernel_pooling_quant_dispatch_out(
    thread_pool_ptr_t thread_pool, const int64_t num_bags, const int64_t row_width,
    const int8_t* input, const int64_t input_stride, int8_t* output, const int64_t output_stride,
    const SparseType_t sparse_type, const int64_t fixed_hotness, const OffsetT* offsets,
    const PoolingType_t pooling_type, const void* weights, const DataType_t weight_type,
    const DataType_t out_type, const int64_t num_workers) {
  switch (out_type) {
    case DataType_t::Float32:
      cpu_kernel_pooling_quant_dispatch_weight<BaseT, QScaleT, HasOffset, float, OffsetT>(
          std::move(thread_pool), num_bags, row_width, input, input_stride, output, output_stride,
          sparse_type, fixed_hotness, offsets, pooling_type, weights, weight_type, num_workers);
      break;
    case DataType_t::Float16:
      cpu_kernel_pooling_quant_dispatch_weight<BaseT, QScaleT, HasOffset, __half, OffsetT>(
          std::move(thread_pool), num_bags, row_width, input, input_stride, output, output_stride,
          sparse_type, fixed_hotness, offsets, pooling_type, weights, weight_type, num_workers);
      break;
    default:
      NVE_THROW_("Unsupported pooling output type ", out_type);
  }
}

/**
 * Entry point: dispatch the pooling op over input/output/weight dtypes.
 * Input may be Float32 / Float16, or per-row quantized: symmetric int8
 * (QInt8RowwiseF32/F16, value*scale) or affine uint8 (QUint8RowwiseF32/F16,
 * value*scale + offset), dequantized to the output type during pooling. For
 * quantized input, `row_width` is the number of value bytes per row (excluding the
 * trailing scale[/offset] metadata) and `input_stride` is the full row stride in
 * bytes. Output is restricted to Float32 / Float16.
 *
 * For PoolingType_t::Concatenate the output has `total_keys` rows (one per input key,
 * no per-bag reduction). `total_keys` is derived from the sparse layout: for Fixed it
 * is `num_bags * fixed_hotness`; for CSR it is `offsets[num_bags]`. The output buffer
 * must be sized for at least `total_keys * output_stride` bytes.
 */
template <typename OffsetT>
void cpu_kernel_pooling_dispatch(thread_pool_ptr_t thread_pool, const int64_t num_bags,
                                 const int64_t row_width, const int8_t* input,
                                 const int64_t input_stride, int8_t* output,
                                 const int64_t output_stride, const SparseType_t sparse_type,
                                 const int64_t fixed_hotness, const OffsetT* offsets,
                                 const PoolingType_t pooling_type, const void* weights,
                                 const DataType_t weight_type, const DataType_t in_type,
                                 const DataType_t out_type, const int64_t num_workers) {
  static_assert(std::is_same_v<OffsetT, int32_t> || std::is_same_v<OffsetT, int64_t>,
                "cpu_kernel_pooling_dispatch: OffsetT must be int32_t or int64_t");
  if (pooling_type == PoolingType_t::Concatenate) {
    const int64_t total_rows = (sparse_type == SparseType_t::Fixed)
                                   ? num_bags * fixed_hotness
                                   : static_cast<int64_t>(offsets[num_bags]);
    cpu_kernel_concatenate_dispatch(std::move(thread_pool), total_rows, row_width, input,
                                    input_stride, output, output_stride, in_type, out_type,
                                    num_workers);
    return;
  }
  switch (in_type) {
    case DataType_t::Float32:
      cpu_kernel_pooling_dispatch_out<float, OffsetT>(
          std::move(thread_pool), num_bags, row_width, input, input_stride, output, output_stride,
          sparse_type, fixed_hotness, offsets, pooling_type, weights, weight_type, out_type,
          num_workers);
      break;
    case DataType_t::Float16:
      cpu_kernel_pooling_dispatch_out<__half, OffsetT>(
          std::move(thread_pool), num_bags, row_width, input, input_stride, output, output_stride,
          sparse_type, fixed_hotness, offsets, pooling_type, weights, weight_type, out_type,
          num_workers);
      break;
    case DataType_t::QInt8RowwiseF32:
      cpu_kernel_pooling_quant_dispatch_out<int8_t, float, /*HasOffset=*/false, OffsetT>(
          std::move(thread_pool), num_bags, row_width, input, input_stride, output, output_stride,
          sparse_type, fixed_hotness, offsets, pooling_type, weights, weight_type, out_type,
          num_workers);
      break;
    case DataType_t::QInt8RowwiseF16:
      cpu_kernel_pooling_quant_dispatch_out<int8_t, __half, /*HasOffset=*/false, OffsetT>(
          std::move(thread_pool), num_bags, row_width, input, input_stride, output, output_stride,
          sparse_type, fixed_hotness, offsets, pooling_type, weights, weight_type, out_type,
          num_workers);
      break;
    case DataType_t::QUint8RowwiseF32:
      cpu_kernel_pooling_quant_dispatch_out<uint8_t, float, /*HasOffset=*/true, OffsetT>(
          std::move(thread_pool), num_bags, row_width, input, input_stride, output, output_stride,
          sparse_type, fixed_hotness, offsets, pooling_type, weights, weight_type, out_type,
          num_workers);
      break;
    case DataType_t::QUint8RowwiseF16:
      cpu_kernel_pooling_quant_dispatch_out<uint8_t, __half, /*HasOffset=*/true, OffsetT>(
          std::move(thread_pool), num_bags, row_width, input, input_stride, output, output_stride,
          sparse_type, fixed_hotness, offsets, pooling_type, weights, weight_type, out_type,
          num_workers);
      break;
    default:
      NVE_THROW_("Unsupported pooling input type ", in_type);
  }
}

}  // namespace nve
