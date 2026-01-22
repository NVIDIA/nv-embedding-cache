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
#include <atomic>
#include <memory>
#include <unordered_set>
#include <vector>

#include "cuda_ops/cuda_common.h"
#include "datagen.h"

template <typename IndexT>
void GenerateInput(uint64_t num_rows, uint64_t hotness, uint64_t batch_size, float alpha,
                   int device_id, std::vector<IndexT>& input) {
  ScopedDevice scope_device(device_id);
  assert(input.size() ==
         0);  // If input/input_unique aren't empty, need to cudaHostUnregister before clearing.
  static std::atomic<size_t> seed = 1337;
  auto inputGenerator = nve::getSampleGenerator<IndexT>(alpha, static_cast<IndexT>(num_rows), static_cast<size_t>(hotness), seed++);
  for (size_t b = 0; b < batch_size; b++) {
    auto sample = inputGenerator->getCategoryIndices();
    input.insert(input.end(), sample.begin(), sample.end());
  }
  NVE_CHECK_(
      cudaHostRegister(input.data(), input.size() * sizeof(IndexT), cudaHostRegisterDefault));
}
