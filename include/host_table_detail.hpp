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

namespace nve {

template <typename T, typename U>
void cpu_update_kernel(char* const __restrict table_values,
                       const char* const __restrict update_values, int64_t n) noexcept {
  using tmp_type =
      std::conditional_t<(sizeof(T) > sizeof(float) || sizeof(U) > sizeof(float)), double, float>;

  T* __restrict tab{reinterpret_cast<T* __restrict>(table_values)};
  const U* __restrict upd{reinterpret_cast<const U* __restrict>(update_values)};

  NVE_ASSERT_(n % sizeof(U) == 0);
  n /= sizeof(U);

  for (int64_t i{}; i != n; ++i) {
    const tmp_type t{tab[i]};
    const tmp_type u{upd[i]};
    tab[i] = static_cast<T>(t + u);
  }
}

using update_kernel_t = void (*)(char* __restrict, const char* __restrict, int64_t n);

template <typename TableType>
inline update_kernel_t pick_cpu_update_kernel(const DataType_t update_dtype) {
  // TODO: Add specializations for x86 and ARM low precision instruction sets.
  switch (update_dtype) {
    case DataType_t::Float32:
      return cpu_update_kernel<TableType, type_t<DataType_t::Float32>>;
    case DataType_t::Float16:
      return cpu_update_kernel<TableType, type_t<DataType_t::Float16>>;
    case DataType_t::BFloat:
      return cpu_update_kernel<TableType, type_t<DataType_t::BFloat>>;
    case DataType_t::E4M3:
      return cpu_update_kernel<TableType, type_t<DataType_t::E4M3>>;
    case DataType_t::E5M2:
      return cpu_update_kernel<TableType, type_t<DataType_t::E5M2>>;
    case DataType_t::Float64:
      return cpu_update_kernel<TableType, type_t<DataType_t::Float64>>;
    default:
      NVE_THROW_("Combining data-types (table = ", data_type<TableType>(),
                 " and update =", update_dtype, ") is currently not supported!");
  }
}

update_kernel_t pick_cpu_update_kernel(const DataType_t table_dtype, const DataType_t update_dtype) {
  switch (table_dtype) {
    case DataType_t::Float32:
      return pick_cpu_update_kernel<type_t<DataType_t::Float32>>(update_dtype);
    case DataType_t::Float16:
      return pick_cpu_update_kernel<type_t<DataType_t::Float16>>(update_dtype);
    case DataType_t::BFloat:
      return pick_cpu_update_kernel<type_t<DataType_t::BFloat>>(update_dtype);
    case DataType_t::E4M3:
      return pick_cpu_update_kernel<type_t<DataType_t::E4M3>>(update_dtype);
    case DataType_t::E5M2:
      return pick_cpu_update_kernel<type_t<DataType_t::E5M2>>(update_dtype);
    case DataType_t::Float64:
      return pick_cpu_update_kernel<type_t<DataType_t::Float64>>(update_dtype);
    default:
      NVE_THROW_("Combining data-types (table = ", table_dtype, " and update =", update_dtype,
                 ") is currently not supported!");
  }
}

}  // namespace nve
