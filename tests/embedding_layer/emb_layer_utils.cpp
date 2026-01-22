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

#include "emb_layer_utils.hpp"
#include <cuda_fp16.h>

namespace nve {

void InitTableRows(
  int8_t* table,
  uint64_t row_size,
  uint64_t start_row,
  uint64_t end_row,
  DataType_t dtype,
  size_t seed) {
  const int64_t row_elements = static_cast<int64_t>(row_size) / dtype_size(dtype);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist_float(0.f, 1.f);
  for (uint64_t row = start_row ; row<end_row ; row++) {
    auto* output_row = table + (row * row_size);
    for (int64_t i=0 ; i<row_elements ; i++) {
      store_as_dtype(output_row, i, dtype, dist_float(gen));
    }
  }
}

void GenerateWeights(std::vector<int8_t>& weights, uint64_t num_weights, DataType_t dtype, size_t seed) {
  weights.resize(static_cast<size_t>(num_weights * static_cast<uint64_t>(dtype_size(dtype))));
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist_float(0.f, 1.f);
  const auto weights_buffer = weights.data();
  for (int64_t i=0; i<static_cast<int64_t>(num_weights) ; i++) {
      store_as_dtype(weights_buffer, i, dtype, dist_float(gen));
  }
}

float load_as_float(const void* ptr, int64_t idx, DataType_t dtype) {
  assert(ptr);
  switch (dtype) {
    case DataType_t::Float16:
      return __half2float((reinterpret_cast<const half*>(ptr))[idx]);
    case DataType_t::Float32:
      return (reinterpret_cast<const float*>(ptr))[idx];
    default:
      throw std::runtime_error("Not implemented");
  }
  return 0; // unreachable
}

void store_as_dtype(void* ptr, int64_t idx, DataType_t dtype, float val) {
  assert(ptr);
  switch (dtype) {
    case DataType_t::Float16:
      (reinterpret_cast<half*>(ptr))[idx] = __float2half(val);
      break;
    case DataType_t::Float32:
      (reinterpret_cast<float*>(ptr))[idx] = val;
      break;
    default:
      throw std::runtime_error("Not implemented");
  }
}

}  // namespace nve