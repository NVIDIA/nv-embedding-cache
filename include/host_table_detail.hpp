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

#include <host_table.hpp>
#include <algorithm>
#include <atomic>
#include <utility>

namespace nve {

using update_kernel_t = void (*)(void*, const void* const, int64_t, const void*, int64_t);

update_kernel_t pick_cpu_update_kernel(DataType_t value_dtype, DataType_t update_dtype);

// Reserves the next `n` slots of an output buffer that parallel tasks fill. Returns the first slot and how
// many of the `n` slots fall below `capacity` (<= 0 once the buffer is full).
inline std::pair<int64_t, int64_t> claim_output_range(std::atomic_int64_t& claimed, const int64_t n,
                                                      const int64_t capacity) {
  const int64_t begin{claimed.fetch_add(n, std::memory_order_relaxed)};
  return {begin, std::min(begin + n, capacity) - begin};
}

}  // namespace nve
